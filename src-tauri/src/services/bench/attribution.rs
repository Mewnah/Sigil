//! Per-stage attribution of a run (P3B-A): glue time (total minus stages) and harness time
//! (resample, SOLA, post), syncs per tick, GPU time per plan (when P3B-G's CUDA events report it),
//! onset ticks reported separately, the SOLA join histograms (`VcStreamHarness::last_join`), the T1
//! per-tick decomposition from the paced timeline, and the `load_twice` run (cold vs warm load of a
//! second voice in the same process). Reads `neural_audio::telemetry`.
//!
//! # Frozen by P3B-S0 (signatures; P3B-A implements)
//! - [`run_load_twice`]`(run, spec, cancel, progress) -> Result<RunOutcome, String>`: the
//!   `load_twice` run mode.
//!
//! # Added by P3B-A
//! - [`run_load_twice_with`]: the same with the spec's loaded engine, which streams in the
//!   background while the second voice loads (the hot-voice load of the voice changer).
//! - [`LoadBreakdown`]: the telemetry load spans of a window attributed to wall-clock time
//!   (`load_breakdown_ms`, which sums to the window exactly with its `other` bucket).
//! - [`insert_call_metrics`]: per-call stage, glue, harness, sync, GPU-time and onset metrics.
//! - [`JoinStats`]: the SOLA join histograms. [`T1Parts`]: the paced latency decomposition.
//! - [`algorithmic_latency_table`]: the harness's algorithmic latency at the 10 ms device block vs
//!   the 512-frame one for 20 / 40 / 60 / 80 ms engine blocks (the measure-first input of P3B-H's
//!   10 ms device block).
//!
//! # Metric names
//! Per call (offline, paced, contention, quality): `retrieval_ms_*`, `post_ms_*`, `glue_ms_*`
//! (the model call minus the wall time of its stages, i.e. their union, so stages that run in
//! parallel count once: upsampling, protect, noise windows, phase, speaker vector, the output
//! copy), `overlap_ms_*` (stage time that ran in parallel with another stage), `harness_ms_*` (a model tick's worker time minus its model calls: resampling,
//! SOLA, concealment, formant tilt, gains), `syncs_per_call_*`, `gpu_<content|pitch|enc|flowdec|
//! knn>_ms_*`, `onset_calls`, `call_ms_onset_*` / `call_ms_steady_*` and `f0_ms_onset_*` /
//! `f0_ms_steady_*` (an onset call analyses clearly more F0 frames than the run's steady calls,
//! see [`onset_threshold`]: the whole window after a silence skip), plus `content_ms_*` / `f0_ms_*` / `enc_ms_*` / `flowdec_ms_*` from the telemetry
//! spans when the build has no per-component breakdown (factory builds). Joins: `join_offset_ms_*`
//! (|offset - centre|), `join_step_ms_*` (|consecutive offset change|), `join_ncc_*`,
//! `joins_silent`. Paced T1: `t1_<capture|block_wait|queue|compute|engine_wait|model|hold|output>_ms_*`, the
//! decomposition of the median latency `t1_at_p50_<part>_ms` / `t1_at_p50_sum_ms`, and
//! `t1_residual_p50_ms` (that sum minus `latency_ms_p50`). Load: `load_<stage>_ms`,
//! `load_other_ms`, `load_attributed_ratio`, `generator_builds`, `run_generator_builds`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::services::neural_audio::telemetry::{self, LoadEvent, LoadStage, Stage, SyncKind};
use crate::services::neural_audio::VcStreamHarness;
use crate::services::rt_thread::WorkerPriority;

use super::engine::{build_engine, probe_algorithmic_latency, CallRecord, ModelTimes};
use super::gpu_state::VramSampler;
use super::metrics::{insert_summary, process_vram_mb, Metrics, Summary};
use super::report::{RunOutcome, TickRecord};
use super::spec::{BenchSpec, LoadTwiceRun, StreamSpec};
use super::vc_rt::VcTick;
use super::Prepared;

/// The least margin (F0 frames) above a run's steady count before a call is an onset call.
const ONSET_MARGIN_FRAMES: usize = 4;

/// The F0 frame count above which a call is an onset call (the whole window after a silence skip).
///
/// A steady call's count follows the block (`block / 10 ms` frames) plus the extractor's margin and
/// lead, so no constant fits every run: a fixed 31 tagged every call of a 120 ms-block run
/// (12 + 8 + 12 = 32 frames) as an onset, and would break again with an extractor of a longer
/// lead. The threshold is derived from the run's own converted calls instead: the steady count is
/// their most common count (the smallest of equally common ones: onsets are rarer than steady
/// calls), and an onset call analyses more than that plus a margin of 4 frames or 25 %, whichever
/// is more. `usize::MAX` when no call reported a frame count.
pub fn onset_threshold(frames: &[usize]) -> usize {
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for &f in frames.iter().filter(|&&f| f > 0) {
        *counts.entry(f).or_insert(0) += 1;
    }
    // Ascending counts, so on a tie the smaller (steady) one stays.
    let steady = counts
        .iter()
        .fold(None::<(usize, usize)>, |best, (&f, &n)| match best {
            Some((_, best_n)) if best_n >= n => best,
            _ => Some((f, n)),
        })
        .map(|(f, _)| f);
    steady.map_or(usize::MAX, |s| s + ONSET_MARGIN_FRAMES.max(s / 4))
}

/// The `load_twice` run mode without the spec's loaded engine: refused (the run needs the first
/// voice loaded; `run_spec` calls [`run_load_twice_with`]).
pub(super) fn run_load_twice(
    run: &LoadTwiceRun,
    spec: &BenchSpec,
    cancel: &AtomicBool,
    progress: &(dyn Fn(f64, &str) + Sync),
) -> Result<RunOutcome, String> {
    run_load_twice_with(run, spec, None, cancel, progress)
}

// ---------------------------------------------------------------------------------------------
// Load breakdown
// ---------------------------------------------------------------------------------------------

/// One span of a load window (report rows).
#[derive(Clone, Debug, Serialize)]
pub struct SpanRow {
    pub stage: &'static str,
    pub thread: u64,
    /// From the window start.
    pub start_ms: f64,
    pub ms: f64,
    /// Without nested spans on the same thread.
    pub self_ms: f64,
    /// This process's VRAM change over the span (MiB; with the benchmark's VRAM probe on).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_delta_mb: Option<f64>,
    /// The change without the nested spans' changes: what this stage itself allocated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_self_mb: Option<f64>,
    /// The session component the stage loads (`LoadStage::component`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub component: Option<&'static str>,
    /// A span of another thread overlapped this one: its VRAM change (a process-wide reading)
    /// may include that thread's allocations.
    pub concurrent: bool,
}

/// The load spans of a window, attributed to its wall-clock time.
///
/// Each thread's spans nest (a session plan's first runs inside its session init, the tick stages
/// of `warm_once` inside it); at every instant the innermost span of each thread is the one
/// working, and when spans of several threads overlap (a parallel cold start) the instant is
/// split equally among them. So the attributed times plus `other_ms` (no span open: the
/// benchmark's own resolution, the factory's glue) are exactly the window.
#[derive(Clone, Debug, Default, Serialize)]
pub struct LoadBreakdown {
    pub window_ms: f64,
    /// Stage key -> wall-attributed ms (stages with time only).
    pub stages_ms: BTreeMap<&'static str, f64>,
    pub other_ms: f64,
    /// Stage key -> summed exclusive time over threads (above the wall time when threads overlap).
    pub self_ms: BTreeMap<&'static str, f64>,
    /// Generator builds (`enc` session plans) in the window.
    pub generator_builds: u32,
    pub replans: u32,
    pub threads: usize,
    pub spans: Vec<SpanRow>,
}

impl LoadBreakdown {
    /// The breakdown of `events` over `[start_ns, end_ns]` (events are clipped to it).
    pub fn of(events: &[LoadEvent], start_ns: u64, end_ns: u64) -> LoadBreakdown {
        let end_ns = end_ns.max(start_ns);
        let clip = |e: &LoadEvent| -> Option<(u64, u64)> {
            let s = e.start_ns.max(start_ns);
            let t = e.end_ns.min(end_ns);
            (t > s).then_some((s, t))
        };
        let ms = |ns: u64| ns as f64 / 1e6;
        let mut b = LoadBreakdown {
            window_ms: ms(end_ns - start_ns),
            ..LoadBreakdown::default()
        };
        // Innermost segments per thread: (start, end, stage).
        let mut segments: Vec<(u64, u64, LoadStage)> = Vec::new();
        let mut threads: Vec<u64> = events.iter().filter(|e| clip(e).is_some()).map(|e| e.thread).collect();
        threads.sort_unstable();
        threads.dedup();
        b.threads = threads.len();
        for &th in &threads {
            let mut evs: Vec<(u64, u64, &LoadEvent)> = events
                .iter()
                .filter(|e| e.thread == th)
                .filter_map(|e| clip(e).map(|(s, t)| (s, t, e)))
                .collect();
            // Outer spans first: earlier start, then longer.
            evs.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
            for (i, &(s, t, e)) in evs.iter().enumerate() {
                // Children: spans of this thread strictly inside [s, t] that are not inside a
                // deeper child (direct children cover the time this span did not do itself).
                let mut covered: Vec<(u64, u64)> = Vec::new();
                let mut children_vram = Some(0.0f64);
                for &(cs, ct, ce) in evs.iter().skip(i + 1) {
                    if cs >= t {
                        break;
                    }
                    if cs >= s && ct <= t && !covered.iter().any(|&(a, z)| cs >= a && ct <= z) {
                        covered.push((cs, ct));
                        children_vram = match (children_vram, vram_delta(ce)) {
                            (Some(sum), Some(d)) => Some(sum + d),
                            _ => None,
                        };
                    }
                }
                let concurrent = events
                    .iter()
                    .filter(|o| o.thread != th)
                    .filter_map(&clip)
                    .any(|(os, ot)| os < t && ot > s);
                covered.sort_unstable();
                let mut cursor = s;
                let mut self_ns = 0u64;
                for (cs, ct) in covered.iter().copied() {
                    if cs > cursor {
                        segments.push((cursor, cs, e.stage));
                        self_ns += cs - cursor;
                    }
                    cursor = cursor.max(ct);
                }
                if t > cursor {
                    segments.push((cursor, t, e.stage));
                    self_ns += t - cursor;
                }
                *b.self_ms.entry(e.stage.key()).or_insert(0.0) += ms(self_ns);
                b.spans.push(SpanRow {
                    stage: e.stage.key(),
                    thread: th,
                    start_ms: ms(s - start_ns),
                    ms: ms(t - s),
                    self_ms: ms(self_ns),
                    vram_delta_mb: vram_delta(e),
                    vram_self_mb: vram_delta(e).zip(children_vram).map(|(d, c)| d - c),
                    component: e.stage.component(),
                    concurrent,
                });
                match e.stage {
                    LoadStage::EncSessionInit => b.generator_builds += 1,
                    LoadStage::Replan(_) => b.replans += 1,
                    _ => {}
                }
            }
        }
        b.spans.sort_by(|a, c| a.start_ms.total_cmp(&c.start_ms));
        // Sweep: split each elementary interval among the innermost segments active in it.
        let mut cuts: Vec<u64> = segments.iter().flat_map(|&(s, t, _)| [s, t]).collect();
        cuts.push(start_ns);
        cuts.push(end_ns);
        cuts.sort_unstable();
        cuts.dedup();
        let mut covered_ns = 0u64;
        for w in cuts.windows(2) {
            let (a, z) = (w[0], w[1]);
            let active: Vec<LoadStage> = segments.iter().filter(|&&(s, t, _)| s <= a && t >= z).map(|&(_, _, st)| st).collect();
            if active.is_empty() {
                continue;
            }
            covered_ns += z - a;
            let share = ms(z - a) / active.len() as f64;
            for st in active {
                *b.stages_ms.entry(st.key()).or_insert(0.0) += share;
            }
        }
        b.other_ms = ms((end_ns - start_ns).saturating_sub(covered_ns));
        b
    }

    /// The VRAM each component's load spans allocated themselves (MiB, the sum of their
    /// `vram_self_mb`; nested spans count once), and whether any of them overlapped another
    /// thread's span (then a reading may include that thread's allocations). Empty without the
    /// VRAM probe.
    pub fn vram_by_component(&self) -> BTreeMap<&'static str, (f64, bool)> {
        let mut out: BTreeMap<&'static str, (f64, bool)> = BTreeMap::new();
        for r in &self.spans {
            if let (Some(c), Some(mb)) = (r.component, r.vram_self_mb) {
                let e = out.entry(c).or_insert((0.0, false));
                e.0 += mb;
                e.1 |= r.concurrent;
            }
        }
        out
    }

    /// The attributed share of the window (1 = every millisecond inside a span).
    pub fn attributed_ratio(&self) -> Option<f64> {
        (self.window_ms > 0.0).then(|| (self.window_ms - self.other_ms) / self.window_ms)
    }

    /// `<prefix>_<stage>_ms` metrics, `<prefix>_other_ms`, the attributed ratio and the build
    /// counts.
    pub fn insert_metrics(&self, prefix: &str, m: &mut Metrics) {
        for (k, v) in &self.stages_ms {
            m.insert(format!("{prefix}_{k}_ms"), *v);
        }
        m.insert(format!("{prefix}_other_ms"), self.other_ms);
        if let Some(r) = self.attributed_ratio() {
            m.insert(format!("{prefix}_attributed_ratio"), r);
        }
    }
}

/// A load event's VRAM change (MiB), when the probe measured it.
fn vram_delta(e: &LoadEvent) -> Option<f64> {
    match (e.vram_before_mb, e.vram_after_mb) {
        (Some(a), Some(z)) => Some(z as f64 - a as f64),
        _ => None,
    }
}

/// Stage keys of [`LoadStage::key`] (for the metric catalogue).
pub fn load_stage_keys() -> impl Iterator<Item = &'static str> {
    LoadStage::ALL.iter().map(|s| s.key())
}

// ---------------------------------------------------------------------------------------------
// Per-call attribution
// ---------------------------------------------------------------------------------------------

/// GPU-time components with a metric (`gpu_<name>_ms_*`); others go to `details` only.
pub const GPU_COMPONENTS: [&str; 5] = ["content", "pitch", "enc", "flowdec", "knn"];

/// What [`insert_call_metrics`] found (for `details.attribution`).
#[derive(Clone, Debug, Default, Serialize)]
pub struct CallAttribution {
    /// `tick_stats` (a direct build's per-component breakdown) or `telemetry` (the spans).
    pub stage_source: Option<&'static str>,
    pub calls: usize,
    pub converted_calls: usize,
    pub onset_calls: usize,
    /// A call analysing more F0 frames than this is an onset call (`onset_threshold` of the run's
    /// converted calls); `None` when no call reported a frame count.
    pub onset_above_f0_frames: Option<usize>,
    /// Mean host waits on the GPU per converted call, by kind.
    pub syncs_per_call_by_kind: BTreeMap<&'static str, f64>,
    /// Mean GPU time per converted call, by component (ms).
    pub gpu_ms_per_call: BTreeMap<String, f64>,
    /// Calls whose stages summed above the call (timer overlap; their glue counts as 0).
    pub negative_glue_calls: usize,
    /// Mean share of the model call per stage (and glue), converted calls.
    pub share_of_model: BTreeMap<&'static str, f64>,
}

/// One converted call's stage times (ms), by [`Stage::index`].
fn stages_of(c: &CallRecord) -> Option<[f64; 6]> {
    let tel = c.tel.as_ref().filter(|t| t.ran(Stage::Content) || t.ran(Stage::Enc) || t.ran(Stage::Flowdec));
    let mut s = [0.0f64; 6];
    match (&c.stats, tel) {
        (Some(st), _) if !st.skipped => {
            s[Stage::Content.index()] = st.content_ms;
            s[Stage::F0.index()] = st.f0_ms;
            s[Stage::Enc.index()] = st.enc_ms;
            s[Stage::Flowdec.index()] = st.flowdec_ms;
            if let Some(t) = c.tel.as_ref() {
                s[Stage::Retrieval.index()] = t.stage(Stage::Retrieval);
                s[Stage::Post.index()] = t.stage(Stage::Post);
            }
            Some(s)
        }
        (Some(_), _) => None,
        (None, Some(t)) => Some(t.stage_ms),
        (None, None) => None,
    }
}

fn f0_frames_of(c: &CallRecord) -> Option<usize> {
    c.stats.map(|s| s.f0_frames).or(c.stage.map(|s| s.f0_frames))
}

/// The wall time a converted call spent in its stages (ms): the union of the stages' intervals
/// from the telemetry spans, so stages that ran in parallel (F0 beside the content encoder) count
/// once; the plain sum of `stages` when the call has no span intervals.
fn staged_wall_ms(c: &CallRecord, stages: &[f64; 6]) -> f64 {
    let sum: f64 = stages.iter().sum();
    match c.tel.as_ref().filter(|t| t.stage_end_ns.iter().any(|&e| e > 0)) {
        // The union never exceeds the sum (a direct build's breakdown and the spans time the same
        // stages, up to timer placement).
        Some(t) => t.stages_union_ms().min(sum),
        None => sum,
    }
}

/// Insert the per-call metrics of a run (see the module docs). `timeline` is the run's ticks and
/// `calls` the model calls they made, in order. `gpu_engine`: the engine runs on a GPU EP; on the
/// CPU rung an eager session run (`SyncKind::EagerRun`, recorded by the content encoder, plain
/// RMVPE and kNN whatever their EP) waits on no GPU, so it does not count as a sync there (it stays
/// in `syncs_per_call_by_kind`). A GPU engine whose content encoder fell back to the CPU still
/// counts that encoder's eager runs (the recorder does not know their EP).
pub fn insert_call_metrics(m: &mut Metrics, timeline: &[TickRecord], calls: &[CallRecord], gpu_engine: bool) -> CallAttribution {
    let mut out = CallAttribution {
        calls: calls.len(),
        ..CallAttribution::default()
    };
    // The run's steady F0 frame count decides which calls are onsets (see `onset_threshold`).
    let f0_frames: Vec<usize> = calls.iter().filter(|c| !c.error && stages_of(c).is_some()).filter_map(f0_frames_of).collect();
    let onset_above = onset_threshold(&f0_frames);
    out.onset_above_f0_frames = (!f0_frames.is_empty()).then_some(onset_above);
    let mut stage_vals: [Vec<f64>; 6] = Default::default();
    let mut ran: [bool; 6] = [false; 6];
    let (mut glue, mut harness, mut syncs, mut overlap) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut onset_tick, mut steady_tick, mut f0_onset, mut f0_steady) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut sync_kinds = [0u64; 5];
    let mut gpu: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut share_sum: BTreeMap<&'static str, f64> = BTreeMap::new();
    let mut from_stats = false;
    let mut next = 0usize;
    for t in timeline.iter().filter(|t| t.calls > 0) {
        let n = t.calls as usize;
        let Some(group) = calls.get(next..next + n) else {
            break;
        };
        next += n;
        let model_ms: f64 = group.iter().map(|c| c.model_ms).sum();
        harness.push((t.compute_ms - model_ms).max(0.0));
        let mut onset = false;
        let mut has_f0_info = false;
        for c in group {
            // A failed call's stages stop part-way: it is not a converted call.
            let Some(s) = stages_of(c).filter(|_| !c.error) else {
                continue;
            };
            has_f0_info |= f0_frames_of(c).is_some();
            from_stats |= c.stats.is_some();
            out.converted_calls += 1;
            for i in 0..6 {
                let did = match &c.tel {
                    Some(tel) => tel.stage_spans[i] > 0 || (c.stats.is_some() && matches!(i, 0 | 2 | 3 | 4) && s[i] > 0.0),
                    None => s[i] > 0.0,
                };
                if did {
                    ran[i] = true;
                    stage_vals[i].push(s[i]);
                }
            }
            let total: f64 = s.iter().sum();
            let staged = staged_wall_ms(c, &s);
            let g = c.model_ms - staged;
            if g < 0.0 {
                out.negative_glue_calls += 1;
            }
            glue.push(g.max(0.0));
            overlap.push((total - staged).max(0.0));
            if c.model_ms > 0.0 {
                for st in Stage::ALL {
                    *share_sum.entry(st.as_str()).or_insert(0.0) += s[st.index()] / c.model_ms;
                }
                *share_sum.entry("glue").or_insert(0.0) += g.max(0.0) / c.model_ms;
                *share_sum.entry("overlap").or_insert(0.0) += (total - staged).max(0.0) / c.model_ms;
            }
            if let Some(tel) = &c.tel {
                let eager = if gpu_engine { 0 } else { tel.syncs[SyncKind::EagerRun.index()] };
                syncs.push((tel.syncs_total() - eager) as f64);
                for k in SyncKind::ALL {
                    sync_kinds[k.index()] += tel.syncs[k.index()];
                }
                for gt in tel.gpu.iter().flatten() {
                    let name = gt.component.strip_prefix("vc.").unwrap_or(gt.component).to_string();
                    gpu.entry(name).or_default().push(gt.ms);
                }
            }
            match f0_frames_of(c) {
                Some(f) if f > onset_above => {
                    onset = true;
                    out.onset_calls += 1;
                    f0_onset.push(s[Stage::F0.index()]);
                }
                Some(_) => f0_steady.push(s[Stage::F0.index()]),
                None => {}
            }
        }
        // Ticks that converted with a known F0 frame count (a silence-skipped tick is neither).
        if has_f0_info {
            if onset {
                onset_tick.push(t.compute_ms);
            } else {
                steady_tick.push(t.compute_ms);
            }
        }
    }
    if out.converted_calls == 0 {
        insert_summary(m, "harness_ms", &harness);
        return out;
    }
    out.stage_source = Some(if from_stats { "tick_stats" } else { "telemetry" });
    for st in Stage::ALL {
        let key = format!("{}_ms", st.as_str());
        // A direct build's breakdown is already in (`insert_components`); the telemetry fills in
        // what it lacks.
        if ran[st.index()] && !m.contains_key(&format!("{key}_p50")) {
            insert_summary(m, &key, &stage_vals[st.index()]);
        }
    }
    insert_summary(m, "glue_ms", &glue);
    insert_summary(m, "harness_ms", &harness);
    // Stage time that ran in parallel with another stage (the sum of the stages minus their union).
    if overlap.iter().any(|&o| o > 0.0) {
        insert_summary(m, "overlap_ms", &overlap);
    }
    if !syncs.is_empty() {
        insert_summary(m, "syncs_per_call", &syncs);
        for k in SyncKind::ALL {
            out.syncs_per_call_by_kind.insert(k.as_str(), sync_kinds[k.index()] as f64 / syncs.len() as f64);
        }
    }
    for (name, v) in &gpu {
        out.gpu_ms_per_call.insert(name.clone(), v.iter().sum::<f64>() / out.converted_calls as f64);
        if GPU_COMPONENTS.contains(&name.as_str()) {
            insert_summary(m, &format!("gpu_{name}_ms"), v);
        }
    }
    if !onset_tick.is_empty() || !steady_tick.is_empty() {
        m.insert("onset_calls".into(), out.onset_calls as f64);
        insert_summary(m, "call_ms_onset", &onset_tick);
        insert_summary(m, "call_ms_steady", &steady_tick);
        insert_summary(m, "f0_ms_onset", &f0_onset);
        insert_summary(m, "f0_ms_steady", &f0_steady);
    }
    for (k, v) in share_sum {
        out.share_of_model.insert(k, v / out.converted_calls as f64);
    }
    out
}

/// The share of each stage in the p95 of the model call (`details.p95_shares`): the stage times
/// summed over the calls at or above the call's p95, as a share of those calls' time; `glue` is the
/// time outside every stage and `overlap` the stage time that ran in parallel with another stage,
/// so the stages + `glue` - `overlap` = 1. Which stage dominates the slow calls (the input that
/// orders the engine workstreams).
pub fn p95_shares(calls: &[CallRecord]) -> BTreeMap<&'static str, f64> {
    let converted: Vec<(&CallRecord, [f64; 6])> = calls.iter().filter_map(|c| stages_of(c).map(|s| (c, s))).collect();
    let mut out = BTreeMap::new();
    if converted.len() < 20 {
        return out;
    }
    let mut ms: Vec<f64> = converted.iter().map(|(c, _)| c.model_ms).collect();
    ms.sort_by(f64::total_cmp);
    let p95 = super::metrics::percentile_sorted(&ms, 95.0);
    let slow: Vec<&(&CallRecord, [f64; 6])> = converted.iter().filter(|(c, _)| c.model_ms >= p95).collect();
    let total: f64 = slow.iter().map(|(c, _)| c.model_ms).sum();
    if total <= 0.0 {
        return out;
    }
    for st in Stage::ALL {
        out.insert(st.as_str(), slow.iter().map(|(_, s)| s[st.index()]).sum::<f64>() / total);
    }
    let sum: f64 = slow.iter().map(|(_, s)| s.iter().sum::<f64>()).sum();
    let wall: f64 = slow.iter().map(|(c, s)| staged_wall_ms(c, s)).sum();
    out.insert("glue", ((total - wall) / total).max(0.0));
    out.insert("overlap", ((sum - wall) / total).max(0.0));
    out
}

// ---------------------------------------------------------------------------------------------
// SOLA joins
// ---------------------------------------------------------------------------------------------

/// The SOLA joins of a run (`VcStreamHarness::last_join` after every tick that converted).
#[derive(Clone, Debug, Default)]
pub struct JoinStats {
    /// `(offset - centre in model samples, NCC)`, `None` for a silent tail.
    joins: Vec<Option<(isize, f32)>>,
}

/// A histogram as `[lower bound, count]` rows.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Histogram {
    pub bin: f64,
    pub rows: Vec<(f64, u64)>,
}

impl Histogram {
    /// Bins of width `bin` (values on a bin edge go up).
    pub fn of(values: &[f64], bin: f64) -> Histogram {
        let mut counts: BTreeMap<i64, u64> = BTreeMap::new();
        for &v in values.iter().filter(|v| v.is_finite()) {
            *counts.entry((v / bin).floor() as i64).or_insert(0) += 1;
        }
        Histogram {
            bin,
            rows: counts.into_iter().map(|(k, n)| (k as f64 * bin, n)).collect(),
        }
    }
}

impl JoinStats {
    pub fn with_capacity(n: usize) -> JoinStats {
        JoinStats { joins: Vec::with_capacity(n) }
    }

    /// Record the harness's last join after a tick that ran the model.
    pub fn record(&mut self, join: Option<(isize, f32)>) {
        self.joins.push(join);
    }

    /// Insert `join_*` metrics and return the histograms (`model_rate`: the model's output rate,
    /// where SOLA works).
    pub fn insert(&self, m: &mut Metrics, model_rate: u32) -> serde_json::Value {
        let to_ms = |s: f64| s * 1000.0 / f64::from(model_rate.max(1));
        let live: Vec<(isize, f32)> = self.joins.iter().flatten().copied().collect();
        let silent = self.joins.len() - live.len();
        if self.joins.is_empty() {
            return serde_json::Value::Null;
        }
        m.insert("joins_silent".into(), silent as f64);
        if live.is_empty() {
            return serde_json::json!({"joins": self.joins.len(), "silent": silent});
        }
        let offsets: Vec<f64> = live.iter().map(|&(o, _)| to_ms(o as f64)).collect();
        let abs: Vec<f64> = offsets.iter().map(|o| o.abs()).collect();
        let ncc: Vec<f64> = live.iter().map(|&(_, c)| f64::from(c)).collect();
        // Consecutive joins only (a silent tail breaks the chain).
        let steps: Vec<f64> = self
            .joins
            .windows(2)
            .filter_map(|w| match (w[0], w[1]) {
                (Some((a, _)), Some((b, _))) => Some(to_ms((b - a) as f64).abs()),
                _ => None,
            })
            .collect();
        insert_summary(m, "join_offset_ms", &abs);
        insert_summary(m, "join_step_ms", &steps);
        insert_summary(m, "join_ncc", &ncc);
        let sample_ms = to_ms(1.0);
        serde_json::json!({
            "joins": self.joins.len(),
            "silent": silent,
            "model_rate": model_rate,
            "offset_ms": Histogram::of(&offsets, (sample_ms * 4.0).max(0.05)),
            "step_ms": Histogram::of(&steps, (sample_ms * 4.0).max(0.05)),
            "ncc": Histogram::of(&ncc, 0.05),
            "offset_ms_signed": Summary::of(&offsets),
        })
    }
}

// ---------------------------------------------------------------------------------------------
// T1 decomposition
// ---------------------------------------------------------------------------------------------

/// One played sample's latency, split along its path (ms; the parts sum to its latency exactly):
/// - `capture`: the capture device period (sampled until its callback delivered it);
/// - `block_wait`: until the rest of its device block was captured;
/// - `queue`: until the worker took the block (busy with earlier blocks, a stall);
/// - `compute`: the device block's tick (the model call when the model runs on every tick);
/// - `engine_wait`: a fixed-block engine: from that tick until the tick that ran the model call
///   of the sample's engine block started (the rest of the engine block and its lookahead
///   arriving; 0 when the device block's own tick ran it);
/// - `model`: that later tick (the model call);
/// - `hold`: until the tick that emitted the sample's converted copy finished (the SOLA tail, the
///   resamplers, the fixed-block hold-back);
/// - `output`: until the playback callback took it (the output cushion / prefill gate and the
///   callback period).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct T1Parts {
    pub capture: f64,
    pub block_wait: f64,
    pub queue: f64,
    pub compute: f64,
    pub engine_wait: f64,
    pub model: f64,
    pub hold: f64,
    pub output: f64,
}

impl T1Parts {
    pub const NAMES: [&'static str; 8] = ["capture", "block_wait", "queue", "compute", "engine_wait", "model", "hold", "output"];

    pub fn values(&self) -> [f64; 8] {
        [self.capture, self.block_wait, self.queue, self.compute, self.engine_wait, self.model, self.hold, self.output]
    }

    pub fn total(&self) -> f64 {
        self.values().iter().sum()
    }
}

/// Share of the samples around the median latency that the at-p50 decomposition averages (the
/// central 10 %: ranks p45..p55, at least one sample).
const T1_P50_BAND: f64 = 0.10;

/// The decomposition of the median latency: the mean parts of the samples whose total latency
/// ranks in the central [`T1_P50_BAND`] (their parts add up to the band's mean latency, which sits
/// at the p50 within the band's spread), and that band's latency range.
pub fn t1_at_p50(parts: &[T1Parts]) -> Option<(T1Parts, f64, f64)> {
    if parts.is_empty() {
        return None;
    }
    let mut sorted: Vec<&T1Parts> = parts.iter().collect();
    sorted.sort_by(|a, b| a.total().total_cmp(&b.total()));
    let n = sorted.len();
    let width = ((n as f64 * T1_P50_BAND).round() as usize).max(1).min(n);
    // Centre the band on the median rank.
    let lo = ((n - width) as f64 / 2.0).round() as usize;
    let band = &sorted[lo..lo + width];
    let k = band.len() as f64;
    let mean = |f: fn(&T1Parts) -> f64| band.iter().map(|p| f(p)).sum::<f64>() / k;
    let at = T1Parts {
        capture: mean(|p| p.capture),
        block_wait: mean(|p| p.block_wait),
        queue: mean(|p| p.queue),
        compute: mean(|p| p.compute),
        engine_wait: mean(|p| p.engine_wait),
        model: mean(|p| p.model),
        hold: mean(|p| p.hold),
        output: mean(|p| p.output),
    };
    Some((at, band[0].total(), band[band.len() - 1].total()))
}

/// Insert `t1_<part>_ms_*` (each part's distribution), `t1_at_p50_<part>_ms` (the decomposition of
/// the median latency, [`t1_at_p50`]) and `t1_residual_p50_ms` (that decomposition's sum minus the
/// run's latency p50: the acceptance check that the parts explain the measured p50), and return the
/// details. The parts' means add up to the mean latency exactly; medians of parts do not add
/// (`details.t1.sum_of_p50_ms` shows them).
pub fn insert_t1(m: &mut Metrics, parts: &[T1Parts], latency_p50: Option<f64>) -> serde_json::Value {
    if parts.is_empty() {
        return serde_json::Value::Null;
    }
    let mut p50_sum = 0.0;
    let mut mean_sum = 0.0;
    let mut rows = serde_json::Map::new();
    for (i, name) in T1Parts::NAMES.iter().enumerate() {
        let v: Vec<f64> = parts.iter().map(|p| p.values()[i]).collect();
        if let Some(s) = insert_summary(m, &format!("t1_{name}_ms"), &v) {
            p50_sum += s.p50;
            mean_sum += s.mean;
            rows.insert((*name).to_string(), serde_json::to_value(s).unwrap_or_default());
        }
    }
    let total: Vec<f64> = parts.iter().map(T1Parts::total).collect();
    let total_mean = total.iter().sum::<f64>() / total.len() as f64;
    let at_p50 = t1_at_p50(parts);
    let mut at_rows = serde_json::Map::new();
    if let Some((at, _, _)) = &at_p50 {
        for (name, v) in T1Parts::NAMES.iter().zip(at.values()) {
            m.insert(format!("t1_at_p50_{name}_ms"), v);
            at_rows.insert((*name).to_string(), v.into());
        }
        m.insert("t1_at_p50_sum_ms".into(), at.total());
        if let Some(l) = latency_p50 {
            m.insert("t1_residual_p50_ms".into(), at.total() - l);
        }
    }
    serde_json::json!({
        "parts": rows,
        "at_p50": at_rows,
        "at_p50_sum_ms": at_p50.as_ref().map(|(a, _, _)| a.total()),
        "at_p50_band_ms": at_p50.as_ref().map(|(_, lo, hi)| [*lo, *hi]),
        "sum_of_p50_ms": p50_sum,
        "sum_of_means_ms": mean_sum,
        "latency_mean_ms": total_mean,
        "latency_p50_ms": latency_p50,
        "samples": parts.len(),
    })
}

// ---------------------------------------------------------------------------------------------
// Algorithmic latency at 10 ms vs 512-frame device blocks
// ---------------------------------------------------------------------------------------------

/// One row of [`algorithmic_latency_table`].
#[derive(Clone, Debug, Serialize)]
pub struct AlgRow {
    pub engine_block_ms: f64,
    pub device_frames: u32,
    pub device_block_ms: f64,
    pub latency_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// The harness's algorithmic latency (identity model, the stream's rates, extra context and
/// crossfade) with a fixed engine block of `blocks_ms` each, on the 10 ms device block
/// (`rate / 100` frames, 480 at 48 kHz) and on the 512-frame one: the 10 ms block divides every
/// engine block, so the harness holds nothing back (P3B-H's W7 input).
pub fn algorithmic_latency_table(stream: &StreamSpec, out_sr: u32, min_context: usize, blocks_ms: &[f64]) -> Vec<AlgRow> {
    let ten_ms = stream.input_rate / 100;
    let mut rows = Vec::new();
    for &block_ms in blocks_ms {
        for frames in [ten_ms, 512] {
            let mut s = stream.clone();
            s.block_frames = frames;
            let fixed = (block_ms * 16.0).round() as usize;
            let r = probe_algorithmic_latency(&s, out_sr, min_context, Some(fixed), 0.0);
            rows.push(AlgRow {
                engine_block_ms: block_ms,
                device_frames: frames,
                device_block_ms: s.block_ms(),
                latency_ms: r.as_ref().ok().map(|p| p.ms),
                error: r.err(),
            });
        }
    }
    rows
}

// ---------------------------------------------------------------------------------------------
// load_twice
// ---------------------------------------------------------------------------------------------

/// What the background voice measured while the second one loaded: `(end of the call, ms)` of
/// every model call.
#[derive(Default)]
struct Background {
    calls: Vec<(Instant, f64)>,
    budget_ms: f64,
    error: Option<String>,
    /// Model calls of the background voice that failed.
    model_errors: u64,
}

/// The background voice streams this long before the second load starts (its baseline calls).
const BACKGROUND_BASELINE: Duration = Duration::from_millis(400);

/// Raises a stop flag when dropped, whatever ends the scope that owns it. A scoped thread that
/// loops until a flag is set must not depend on the code after the work that may panic to set it:
/// `thread::scope` waits for every thread, so a flag that is never raised hangs the benchmark
/// instead of letting the panic reach `catch_panic`.
struct StopOnDrop<'a>(&'a AtomicBool);

impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Stream the first voice at real-time pace until `stop` (its per-call worker time).
fn stream_background(spec: &BenchSpec, p: &Prepared, ready: &AtomicBool, stop: &AtomicBool) -> Background {
    let s = &spec.stream;
    // The worker's thread policy, as the voice changer's worker runs.
    crate::services::rt_thread::set_worker_priority(WorkerPriority::from_str_lossy(&s.worker_priority), p.engine.info.gpu());
    let mut bg = Background {
        budget_ms: s.block_ms().max(p.engine.fixed_block().map_or(0.0, |b| b as f64 / 16.0)),
        ..Background::default()
    };
    let times = Arc::new(Mutex::new(ModelTimes::default()));
    let harness = match VcStreamHarness::new(
        p.engine.handle(times.clone()),
        s.input_rate,
        s.output_rate,
        usize::from(s.input_channels),
        usize::from(s.output_channels),
        s.extra_samples_16k(),
        s.crossfade_samples(p.engine.info.output_rate),
        s.block_frames as usize,
    ) {
        Ok(h) => h,
        Err(e) => {
            bg.error = Some(e);
            ready.store(true, Ordering::SeqCst);
            return bg;
        }
    };
    let mut tick = VcTick::new(harness, s);
    let ch = usize::from(s.input_channels);
    let block = s.block_frames as usize;
    let per_pass = p.input.len() / ch / block.max(1);
    let period = Duration::from_secs_f64(s.block_ms() / 1000.0);
    let mut work = Vec::with_capacity(block * ch);
    ready.store(true, Ordering::SeqCst);
    let start = Instant::now();
    let mut i = 0usize;
    while !stop.load(Ordering::Relaxed) && per_pass > 0 {
        let b = i % per_pass;
        work.clear();
        work.extend_from_slice(&p.input[b * block * ch..(b + 1) * block * ch]);
        let before = times.lock().map(|t| t.convert_ms.len()).unwrap_or(0);
        let t0 = Instant::now();
        let _ = tick.process(&mut work);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if times.lock().map(|t| t.convert_ms.len() > before).unwrap_or(false) {
            bg.calls.push((Instant::now(), ms));
        }
        i += 1;
        bg.model_errors = times.lock().map(|t| t.errors).unwrap_or(0);
        let due = start + period.mul_f64(i as f64);
        let now = Instant::now();
        if due > now {
            std::thread::sleep(due - now);
        }
    }
    bg
}

/// The `load_twice` run: while the spec's voice (loaded by the spec, the cold load) streams in the
/// background at real-time pace, load the same voice a second time in the same process (the warm
/// load: process-wide GPU setup, shared encoders and pitch sessions already exist), with the VRAM
/// load-peak sampler and the load spans; then release it. Reports `second_load_ms`,
/// `vram_second_voice_mb` (what the second voice adds: the per-voice delta of `hot_slots = 2`),
/// `vram_second_load_peak_mb`, `second_generator_builds`, `second_load_<stage>_ms`, and the
/// background voice's `background_call_ms_*` / `background_over_budget_ratio` during the load.
pub(super) fn run_load_twice_with(
    run: &LoadTwiceRun,
    spec: &BenchSpec,
    prepared: Option<&Prepared>,
    cancel: &AtomicBool,
    progress: &(dyn Fn(f64, &str) + Sync),
) -> Result<RunOutcome, String> {
    let p = prepared.ok_or_else(|| format!("run '{}': load_twice needs the spec's engine loaded first", run.name))?;
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }
    let priority = WorkerPriority::from_str_lossy(&spec.stream.worker_priority);
    let ready = AtomicBool::new(false);
    let stop = AtomicBool::new(false);
    let sample_ms = if spec.vram.sample_ms > 0.0 { spec.vram.sample_ms } else { 20.0 };
    let (second, bg) = std::thread::scope(|scope| {
        // Raised on every exit of this closure, a panic in `build_engine` included.
        let _stop_background = StopOnDrop(&stop);
        let bg = std::thread::Builder::new()
            .name("sigil-bench-background-voice".into())
            .stack_size(super::BENCH_STACK_BYTES)
            .spawn_scoped(scope, || stream_background(spec, p, &ready, &stop));
        // The background voice's harness warms up first (its own warm-up is not the load's), then
        // streams a baseline.
        let t_wait = Instant::now();
        while !ready.load(Ordering::SeqCst) && t_wait.elapsed() < Duration::from_secs(120) {
            std::thread::sleep(Duration::from_millis(5));
        }
        std::thread::sleep(BACKGROUND_BASELINE);
        progress(0.2, "loading the second voice");
        let vram0 = process_vram_mb();
        let mark = telemetry::load_mark();
        let sampler = VramSampler::start(Duration::from_secs_f64(sample_ms / 1000.0));
        let w0 = telemetry::now_ns();
        let t0 = Instant::now();
        let built = build_engine(&spec.engine, &spec.stream, priority, spec.seed);
        let t1 = Instant::now();
        let load_ms = t1.duration_since(t0).as_secs_f64() * 1000.0;
        let w1 = telemetry::now_ns();
        let (peak, samples) = sampler.stop();
        let vram1 = process_vram_mb();
        stop.store(true, Ordering::SeqCst);
        let bg = bg.ok().and_then(|h| h.join().ok());
        let breakdown = LoadBreakdown::of(&telemetry::load_events_since(mark), w0, w1);
        // Release the second voice and measure what it gave back.
        let held = built.map(|e| {
            let info = e.info.clone();
            drop(e);
            info
        });
        let vram2 = process_vram_mb();
        ((held, load_ms, vram0, vram1, vram2, peak, samples, breakdown, (t0, t1)), bg)
    });
    let (info, load_ms, vram0, vram1, vram2, peak, samples, breakdown, (t0, t1)) = second;
    let info = info?;
    progress(0.9, "scoring");
    let mut m = Metrics::new();
    // The cold load (the spec's engine) next to the warm one.
    m.insert("engine_load_ms".into(), p.engine.info.load_ms);
    m.insert("second_load_ms".into(), load_ms);
    m.insert("second_generator_builds".into(), f64::from(breakdown.generator_builds));
    breakdown.insert_metrics("second_load", &mut m);
    let delta = |a: Option<u64>, b: Option<u64>| match (a, b) {
        (Some(a), Some(b)) => Some(b as f64 - a as f64),
        _ => None,
    };
    if let Some(d) = delta(vram0, vram1) {
        m.insert("vram_second_voice_mb".into(), d.max(0.0));
    }
    if let (Some(a), Some(pk)) = (vram0, peak) {
        m.insert("vram_second_load_peak_mb".into(), (pk as f64 - a as f64).max(0.0));
    }
    // The background load's transient above where the second voice settles (T3: <= that voice's
    // steady level + 50 MB).
    if let (Some(pk), Some(v1)) = (peak, vram1) {
        m.insert("second_load_overshoot_mb".into(), (pk as f64 - v1 as f64).max(0.0));
    }
    let mut notes = Vec::new();
    let bg = bg.unwrap_or_default();
    if let Some(e) = &bg.error {
        notes.push(format!("the background voice did not stream: {e}"));
    }
    // The background voice's failed model calls (a spec's `model_errors` threshold applies here).
    m.insert("model_errors".into(), bg.model_errors as f64);
    // The background calls that ended during the second load, and its baseline before it.
    let during: Vec<f64> = bg.calls.iter().filter(|(at, _)| *at >= t0 && *at <= t1).map(|&(_, ms)| ms).collect();
    let baseline: Vec<f64> = bg.calls.iter().filter(|(at, _)| *at < t0).map(|&(_, ms)| ms).collect();
    if !during.is_empty() {
        insert_summary(&mut m, "background_call_ms", &during);
        let over = during.iter().filter(|&&c| c > bg.budget_ms).count();
        m.insert("background_over_budget_ratio".into(), over as f64 / during.len() as f64);
    } else {
        notes.push("the second load ended before the background voice converted a block".to_string());
    }
    let details = serde_json::json!({
        "second_voice": {
            "load_ms": load_ms,
            "execution_provider_active": info.execution_provider_active,
            "build": info.build,
        },
        "vram_mb": {"before": vram0, "after_load": vram1, "peak": peak, "after_release": vram2, "samples": samples, "sample_ms": sample_ms},
        "load_breakdown": breakdown,
        "background": {
            "calls": bg.calls.len(),
            "calls_during_load": during.len(),
            "budget_ms": bg.budget_ms,
            "baseline_call_ms": Summary::of(&baseline),
        },
        "notes": notes,
    });
    Ok(RunOutcome {
        metrics: m,
        details,
        ..RunOutcome::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::telemetry::{ReplanReason, TickTelemetry};
    use crate::services::neural_audio::vc::engine::TickStats;

    fn ev(stage: LoadStage, thread: u64, s_ms: u64, e_ms: u64) -> LoadEvent {
        LoadEvent {
            stage,
            thread,
            start_ns: s_ms * 1_000_000,
            end_ns: e_ms * 1_000_000,
            vram_before_mb: None,
            vram_after_mb: None,
        }
    }

    /// Nested spans give their parent only the time they do not cover; parallel spans split the
    /// instant; the attributed stages plus `other` are exactly the window.
    #[test]
    fn load_breakdowns_sum_to_the_window() {
        let events = vec![
            ev(LoadStage::FirstRunsCapture, 1, 30, 50),
            ev(LoadStage::EncSessionInit, 1, 20, 60),
            ev(LoadStage::ContentSessionInit, 2, 40, 80),
            ev(LoadStage::WarmOnce, 1, 90, 100),
            ev(LoadStage::Replan(ReplanReason::Lookahead), 1, 100, 110),
            // Outside the window: clipped away.
            ev(LoadStage::WeightsRead, 1, 150, 160),
        ];
        let b = LoadBreakdown::of(&events, 0, 120_000_000);
        assert_eq!(b.window_ms, 120.0);
        let sum: f64 = b.stages_ms.values().sum::<f64>() + b.other_ms;
        assert!((sum - 120.0).abs() < 1e-9, "{b:?}");
        // enc 20..30 alone, 50..60 shared with content (2 threads) -> 10 + 5 = 15.
        assert!((b.stages_ms["enc_session_init"] - 15.0).abs() < 1e-9, "{:?}", b.stages_ms);
        // first runs 30..40 alone, 40..50 shared -> 15.
        assert!((b.stages_ms["first_runs_capture"] - 15.0).abs() < 1e-9);
        // content 40..60 shared (10), 60..80 alone (20).
        assert!((b.stages_ms["content_session_init"] - 30.0).abs() < 1e-9);
        assert_eq!(b.stages_ms["warm_once"], 10.0);
        assert_eq!(b.stages_ms["replan_lookahead"], 10.0);
        // 0..20, 80..90 and 110..120 have no span.
        assert!((b.other_ms - 40.0).abs() < 1e-9, "{}", b.other_ms);
        assert_eq!(b.self_ms["enc_session_init"], 20.0, "its own 20 ms (the 20 ms capture is nested)");
        assert_eq!((b.generator_builds, b.replans, b.threads), (1, 1, 2));
        assert_eq!(b.spans.len(), 5);
        assert!((b.attributed_ratio().unwrap() - 80.0 / 120.0).abs() < 1e-12);
        let mut m = Metrics::new();
        b.insert_metrics("load", &mut m);
        assert_eq!(m["load_warm_once_ms"], 10.0);
        assert_eq!(m["load_other_ms"], 40.0);
        assert!(load_stage_keys().any(|k| k == "replan_configure_stream"));
    }

    /// VRAM by component counts a nested span's allocation once (the outer span's own change is
    /// its change minus the nested one's) and flags readings that overlapped another thread.
    #[test]
    fn span_vram_counts_nested_allocations_once() {
        let with_vram = |stage, thread, s, e, before: u64, after: u64| LoadEvent {
            vram_before_mb: Some(before),
            vram_after_mb: Some(after),
            ..ev(stage, thread, s, e)
        };
        let events = vec![
            // The plain pitch session (+100) around its plan (+300), then content (+200) on
            // another thread while the plan builds.
            with_vram(LoadStage::PitchPlan, 1, 20, 40, 600, 900),
            with_vram(LoadStage::PitchPlain, 1, 10, 50, 500, 1000),
            with_vram(LoadStage::ContentSessionInit, 2, 60, 70, 1000, 1200),
            with_vram(LoadStage::FlowdecSessionInit, 2, 30, 35, 800, 810),
        ];
        let b = LoadBreakdown::of(&events, 0, 100_000_000);
        let v = b.vram_by_component();
        assert_eq!(v["vc.pitch"], (500.0, true), "300 (plan) + 200 (plain's own), overlapped by flowdec: {v:?}");
        assert_eq!(v["vc.content"], (200.0, false));
        assert_eq!(v["vc.flowdec"], (10.0, true));
        let plain = b.spans.iter().find(|r| r.stage == "pitch_plain").unwrap();
        assert_eq!((plain.vram_delta_mb, plain.vram_self_mb), (Some(500.0), Some(200.0)));
        assert!(LoadBreakdown::of(&[ev(LoadStage::WarmOnce, 1, 0, 5)], 0, 10_000_000).vram_by_component().is_empty());
    }

    #[test]
    fn an_empty_window_is_all_other() {
        let b = LoadBreakdown::of(&[], 5, 5_000_005);
        assert_eq!((b.window_ms, b.other_ms), (5.0, 5.0));
        assert!(b.stages_ms.is_empty());
    }

    fn tel(stages: [f64; 6], syncs: u64) -> TickTelemetry {
        let mut t = TickTelemetry::default();
        for (i, &ms) in stages.iter().enumerate() {
            t.stage_ms[i] = ms;
            t.stage_spans[i] = u32::from(ms > 0.0);
        }
        t.syncs[SyncKind::RunEnd.index()] = syncs;
        t
    }

    fn tick(compute_ms: f64, calls: u32) -> TickRecord {
        TickRecord {
            compute_ms,
            calls,
            ..TickRecord::default()
        }
    }

    /// Glue is the model call minus its stages, harness the tick minus its calls; onset calls (a
    /// whole-window F0) are reported apart; silence-skipped calls do not count as converted.
    #[test]
    fn calls_split_into_stages_glue_harness_and_onsets() {
        let timeline = vec![tick(0.5, 0), tick(10.0, 1), tick(12.0, 1), tick(1.0, 1), tick(9.0, 1)];
        let stats = |f0_frames: usize, skipped: bool| TickStats {
            content_ms: 3.0,
            f0_ms: 2.0,
            enc_ms: 1.0,
            flowdec_ms: 2.5,
            total_ms: 9.0,
            f0_frames,
            skipped,
        };
        let calls = vec![
            CallRecord {
                model_ms: 9.0,
                stats: Some(stats(24, false)),
                tel: Some(tel([3.0, 0.0, 2.0, 1.0, 2.5, 0.3], 4)),
                ..CallRecord::default()
            },
            CallRecord {
                model_ms: 11.0,
                stats: Some(stats(64, false)),
                tel: Some(tel([3.0, 0.0, 2.0, 1.0, 2.5, 0.3], 4)),
                ..CallRecord::default()
            },
            CallRecord {
                model_ms: 0.2,
                stats: Some(stats(0, true)),
                tel: Some(TickTelemetry::default()),
                ..CallRecord::default()
            },
            // A factory build: stages from the spans only.
            CallRecord {
                model_ms: 8.0,
                tel: Some(tel([2.0, 0.5, 1.5, 1.0, 2.0, 0.2], 2)),
                ..CallRecord::default()
            },
        ];
        let mut m = Metrics::new();
        let a = insert_call_metrics(&mut m, &timeline, &calls, true);
        assert_eq!((a.calls, a.converted_calls, a.onset_calls), (4, 3, 1));
        assert_eq!(a.stage_source, Some("tick_stats"));
        // Glue: 9 - 8.8, 11 - 8.8, 8 - 7.2.
        assert!((m["glue_ms_max"] - 2.2).abs() < 1e-9, "{m:?}");
        assert!((m["glue_ms_min"] - 0.2).abs() < 1e-9);
        // Harness: 10 - 9, 12 - 11, 1 - 0.2, 9 - 8.
        assert!((m["harness_ms_max"] - 1.0).abs() < 1e-9 && (m["harness_ms_min"] - 0.8).abs() < 1e-9);
        assert_eq!(m["onset_calls"], 1.0);
        assert_eq!(m["call_ms_onset_p50"], 12.0);
        assert!(m.contains_key("call_ms_steady_p50") && m.contains_key("f0_ms_onset_p50"));
        assert!(m.contains_key("post_ms_p50") && m.contains_key("retrieval_ms_p50"));
        assert_eq!(m["syncs_per_call_max"], 4.0);
        assert!((a.syncs_per_call_by_kind["run_end"] - 10.0 / 3.0).abs() < 1e-9);
        assert!(!m.contains_key("gpu_enc_ms_p50"), "no GPU timing recorded");
        assert!(a.share_of_model.contains_key("glue"));
    }

    /// The onset threshold follows the run's own steady F0 frame count: 24 frames at 40 ms blocks,
    /// 32 at 120 ms (a fixed 31 tagged every call of the CPU rows as an onset), 40 with a longer-lead
    /// extractor; onsets (the whole window) are rarer than steady calls, and a tie keeps the smaller.
    #[test]
    fn the_onset_threshold_follows_the_runs_steady_frame_count() {
        // 40 ms blocks, RMVPE: steady 24, one onset of 64 frames.
        let t40 = onset_threshold(&[64, 24, 24, 24, 25, 24]);
        assert_eq!(t40, 30);
        assert!((25..64).contains(&t40), "24 and 25 are steady, 64 is the onset");
        // 120 ms blocks (12 + 8 + 12 = 32): steady, not onset; the 72-frame window is an onset.
        let t120 = onset_threshold(&[32, 32, 72, 32, 32]);
        assert_eq!(t120, 40);
        assert!((32..72).contains(&t120), "the old fixed 31 tagged every one of these");
        // A longer-lead extractor (steady 40): 40 is steady, 44 is jitter, 88 is an onset.
        let t_lead = onset_threshold(&[40, 40, 44, 40, 88]);
        assert_eq!(t_lead, 50);
        // Equally common counts: the smaller is the steady one (onsets are the rarer kind).
        assert_eq!(onset_threshold(&[24, 64]), 30);
        // Zero-frame calls (skipped, no F0) and an empty run do not count.
        assert_eq!(onset_threshold(&[0, 0, 24, 24]), 30);
        assert_eq!(onset_threshold(&[]), usize::MAX);
        assert_eq!(onset_threshold(&[0]), usize::MAX);
    }

    /// A run of 120 ms-block calls (32 steady frames) reports its steady calls as steady, and its
    /// whole-window call as the one onset.
    #[test]
    fn long_blocks_do_not_make_every_call_an_onset() {
        let timeline: Vec<TickRecord> = (0..6).map(|_| tick(10.0, 1)).collect();
        let stats = |f0_frames: usize| TickStats {
            content_ms: 3.0,
            f0_ms: 2.0,
            enc_ms: 1.0,
            flowdec_ms: 2.5,
            total_ms: 9.0,
            f0_frames,
            skipped: false,
        };
        let frames = [32usize, 32, 72, 32, 32, 33];
        let calls: Vec<CallRecord> = frames
            .iter()
            .map(|&f| CallRecord {
                model_ms: 9.0,
                stats: Some(stats(f)),
                tel: Some(tel([3.0, 0.0, 2.0, 1.0, 2.5, 0.3], 4)),
                ..CallRecord::default()
            })
            .collect();
        let mut m = Metrics::new();
        let a = insert_call_metrics(&mut m, &timeline, &calls, true);
        assert_eq!(a.onset_above_f0_frames, Some(40));
        assert_eq!((a.converted_calls, a.onset_calls), (6, 1));
        assert_eq!(m["onset_calls"], 1.0);
        assert!(m.contains_key("call_ms_steady_p50") && m.contains_key("f0_ms_steady_p50"), "{m:?}");
    }

    /// A scoped thread that loops until a flag is set is stopped when the scope's closure panics:
    /// the guard raises the flag during the unwind, so `thread::scope` returns (and re-raises the
    /// panic) instead of waiting for a thread that never ends.
    #[test]
    fn a_panic_in_the_second_load_stops_the_background_voice() {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let stop = AtomicBool::new(false);
            let ran = AtomicBool::new(false);
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                std::thread::scope::<_, ()>(|scope| {
                    let _stop = StopOnDrop(&stop);
                    scope.spawn(|| {
                        while !stop.load(Ordering::SeqCst) {
                            ran.store(true, Ordering::SeqCst);
                            std::thread::sleep(Duration::from_millis(2));
                        }
                    });
                    std::thread::sleep(Duration::from_millis(20));
                    panic!("build_engine panicked");
                })
            }));
            let _ = tx.send((outcome.is_err(), ran.load(Ordering::SeqCst), stop.load(Ordering::SeqCst)));
        });
        let (panicked, ran, stopped) = rx
            .recv_timeout(Duration::from_secs(20))
            .expect("the scope must return: the background thread has to be stopped by the guard");
        assert!(panicked && ran && stopped);
    }

    /// Stages that overlap (F0 beside the content encoder) count once: glue is the call minus the
    /// union of the stages, and the overlap is reported apart.
    #[test]
    fn overlapping_stages_count_once_in_glue() {
        // Content 0..4 ms and F0 1..4 ms in parallel, enc 4..5, flowdec 5..7: union 7 ms.
        let mut t = tel([4.0, 0.0, 3.0, 1.0, 2.0, 0.0], 0);
        let ms = 1_000_000u64;
        for (i, (a, b)) in [(0, 4), (0, 0), (1, 4), (4, 5), (5, 7), (0, 0)].into_iter().enumerate() {
            t.stage_start_ns[i] = 100 * ms + a * ms;
            t.stage_end_ns[i] = if b > 0 { 100 * ms + b * ms } else { 0 };
        }
        assert!((t.stages_union_ms() - 7.0).abs() < 1e-9);
        let calls: Vec<CallRecord> = (0..20)
            .map(|_| CallRecord {
                model_ms: 8.0,
                tel: Some(t),
                ..CallRecord::default()
            })
            .collect();
        let timeline: Vec<TickRecord> = (0..20).map(|_| tick(8.5, 1)).collect();
        let mut m = Metrics::new();
        let a = insert_call_metrics(&mut m, &timeline, &calls, true);
        assert_eq!(a.negative_glue_calls, 0);
        assert!((m["glue_ms_p50"] - 1.0).abs() < 1e-9, "8 - 7: {m:?}");
        assert!((m["overlap_ms_p50"] - 3.0).abs() < 1e-9, "10 - 7");
        let sh = p95_shares(&calls);
        let staged: f64 = Stage::ALL.iter().map(|st| sh[st.as_str()]).sum();
        assert!((staged + sh["glue"] - sh["overlap"] - 1.0).abs() < 1e-9, "{sh:?}");
        assert!((sh["glue"] - 0.125).abs() < 1e-9 && (sh["overlap"] - 0.375).abs() < 1e-9);
    }

    /// Eager session runs are host waits on the GPU only on a GPU engine.
    #[test]
    fn eager_runs_count_as_syncs_on_gpu_engines_only() {
        let mut t = tel([2.0, 0.0, 1.0, 1.0, 2.0, 0.0], 1);
        t.syncs[SyncKind::EagerRun.index()] = 2;
        let calls = vec![CallRecord {
            model_ms: 7.0,
            tel: Some(t),
            ..CallRecord::default()
        }];
        let (mut gpu, mut cpu) = (Metrics::new(), Metrics::new());
        insert_call_metrics(&mut gpu, &[tick(8.0, 1)], &calls, true);
        let a = insert_call_metrics(&mut cpu, &[tick(8.0, 1)], &calls, false);
        assert_eq!((gpu["syncs_per_call_p50"], cpu["syncs_per_call_p50"]), (3.0, 1.0));
        assert_eq!(a.syncs_per_call_by_kind["eager_run"], 2.0, "still reported by kind");
    }

    #[test]
    fn gpu_time_per_component_becomes_metrics() {
        let mut t = tel([2.0, 0.0, 1.0, 1.0, 2.0, 0.0], 1);
        t.gpu[0] = Some(telemetry::GpuTime { component: "vc.enc", ms: 0.4, runs: 1 });
        t.gpu[1] = Some(telemetry::GpuTime { component: "custom.thing", ms: 0.1, runs: 1 });
        let calls = vec![CallRecord {
            model_ms: 7.0,
            tel: Some(t),
            ..CallRecord::default()
        }];
        let mut m = Metrics::new();
        let a = insert_call_metrics(&mut m, &[tick(8.0, 1)], &calls, true);
        assert_eq!(m["gpu_enc_ms_p50"], 0.4);
        assert!(!m.keys().any(|k| k.starts_with("gpu_custom")));
        assert_eq!(a.gpu_ms_per_call["custom.thing"], 0.1);
        assert_eq!(a.stage_source, Some("telemetry"));
    }

    #[test]
    fn joins_report_offsets_steps_and_correlation() {
        let mut j = JoinStats::with_capacity(8);
        for x in [Some((2isize, 0.9f32)), Some((-2, 0.8)), None, Some((4, 0.95)), Some((4, 0.97))] {
            j.record(x);
        }
        let mut m = Metrics::new();
        let h = j.insert(&mut m, 40_000);
        assert_eq!(m["joins_silent"], 1.0);
        // |offsets| 0.05, 0.05, 0.1, 0.1 ms.
        assert!((m["join_offset_ms_max"] - 0.1).abs() < 1e-12);
        // Steps: 2 -> -2 (0.1 ms) and 4 -> 4 (0); the silent tail breaks the chain.
        assert_eq!(m["join_step_ms_p50"], 0.05);
        assert!((m["join_ncc_min"] - 0.8).abs() < 1e-6);
        assert_eq!(h["joins"], 5);
        assert!(h["ncc"]["rows"].as_array().unwrap().len() >= 2);
        let empty = JoinStats::default().insert(&mut Metrics::new(), 40_000);
        assert!(empty.is_null());
    }

    #[test]
    fn t1_parts_sum_to_the_latency() {
        let parts = vec![
            T1Parts { capture: 5.0, block_wait: 20.0, queue: 1.0, compute: 9.0, hold: 15.0, output: 40.0, ..T1Parts::default() },
            T1Parts { capture: 5.0, block_wait: 10.0, queue: 0.0, compute: 10.0, hold: 15.0, output: 42.0, ..T1Parts::default() },
        ];
        let mut m = Metrics::new();
        let d = insert_t1(&mut m, &parts, Some(86.0));
        assert_eq!(parts[0].total(), 90.0);
        assert!((d["sum_of_means_ms"].as_f64().unwrap() - d["latency_mean_ms"].as_f64().unwrap()).abs() < 1e-9);
        assert!(m.contains_key("t1_hold_ms_p50") && m.contains_key("t1_engine_wait_ms_p50") && m.contains_key("t1_residual_p50_ms"));
        assert!(insert_t1(&mut Metrics::new(), &[], None).is_null());
    }

    /// The at-p50 decomposition explains the median latency even when the distribution is skewed
    /// (where the means do not: a few slow samples pull the mean away from the p50).
    #[test]
    fn the_at_p50_decomposition_sums_to_the_median() {
        // 99 samples: output alternates 40 / 50 ms (the playback phase), plus 9 slow ticks.
        let mut parts: Vec<T1Parts> = (0..90)
            .map(|i| T1Parts { capture: 5.0, block_wait: 20.0, queue: 0.5, compute: 9.0, hold: 15.0, output: if i % 2 == 0 { 40.0 } else { 50.0 }, ..T1Parts::default() })
            .collect();
        parts.extend((0..9).map(|_| T1Parts { capture: 5.0, block_wait: 20.0, queue: 100.0, compute: 140.0, hold: 15.0, output: 40.0, ..T1Parts::default() }));
        let mut totals: Vec<f64> = parts.iter().map(T1Parts::total).collect();
        totals.sort_by(f64::total_cmp);
        let p50 = super::super::metrics::percentile_sorted(&totals, 50.0);
        let (at, lo, hi) = t1_at_p50(&parts).unwrap();
        assert!(lo <= p50 && p50 <= hi, "{lo} {p50} {hi}");
        let mut m = Metrics::new();
        let d = insert_t1(&mut m, &parts, Some(p50));
        assert!(m["t1_residual_p50_ms"].abs() <= 1.0 + 1e-9, "{}", m["t1_residual_p50_ms"]);
        assert!((m["t1_at_p50_sum_ms"] - at.total()).abs() < 1e-9 && m.contains_key("t1_at_p50_queue_ms"));
        // The means are pulled up by the slow ticks: their sum misses the p50 by several ms.
        assert!(d["sum_of_means_ms"].as_f64().unwrap() - p50 > 3.0, "{d}");
        // One sample: the band is that sample.
        let (one, a, b) = t1_at_p50(&parts[..1]).unwrap();
        assert_eq!((one, a, b), (parts[0], parts[0].total(), parts[0].total()));
        assert!(t1_at_p50(&[]).is_none());
    }

    /// The 10 ms device block removes the fixed-block hold-back the 512-frame block needs (a 512-
    /// frame block never divides a 40 ms engine block at 48 kHz).
    #[test]
    fn ten_ms_device_blocks_have_less_algorithmic_latency() {
        let stream = StreamSpec::default();
        let rows = algorithmic_latency_table(&stream, 40_000, 720, &[40.0]);
        assert_eq!(rows.len(), 2);
        let (ten, legacy) = (rows[0].latency_ms.expect("480"), rows[1].latency_ms.expect("512"));
        assert_eq!((rows[0].device_frames, rows[1].device_frames), (480, 512));
        assert!(ten < legacy, "{ten} vs {legacy}");
    }

    #[test]
    fn histograms_bin_values() {
        let h = Histogram::of(&[0.0, 0.04, 0.05, 0.12, f64::NAN, -0.01], 0.05);
        assert_eq!(h.rows, vec![(-0.05, 1), (0.0, 2), (0.05, 1), (0.1, 1)]);
    }
}
