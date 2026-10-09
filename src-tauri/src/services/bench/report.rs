//! Benchmark reports (`sigil-bench-report/1`): the metric catalogue, threshold evaluation, and the
//! JSON / Markdown / CSV writers.
//!
//! Every run reports a flat `metrics` map with stable names (the perf gate and thresholds read
//! them) plus mode-specific `details`. Distribution metrics are `<prefix>_<suffix>` with the
//! suffixes of [`SUMMARY_SUFFIXES`] (`block_ms_p95`, `latency_ms_p99`, ...). All times are in
//! milliseconds, ratios in `0..1`; lower is better for every metric except those of
//! [`HIGHER_IS_BETTER`] (`throughput_x`, `e2e_detected`, the load duties, speaker similarity,
//! DNSMOS and the compare margins).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::metrics::{Metrics, SUMMARY_SUFFIXES};
use super::spec::{ClockKind, Limit, RunMode, RunSpec, Thresholds};

/// Schema id of [`BenchReport`].
pub const REPORT_SCHEMA: &str = "sigil-bench-report/1";

/// Distribution prefixes of offline and paced runs. `block_ms`: every device block; `call_ms`: the
/// device blocks that ran the model (a fixed-block engine converts once every few device blocks,
/// so this is its per-block compute), and `rtf` / `over_budget_ratio` are judged on those;
/// `model_ms`: per model call.
const BLOCK_PREFIXES: &[&str] = &["block_ms", "call_ms", "model_ms", "rtf"];
/// Scalars of offline and paced runs.
const BLOCK_SCALARS: &[&str] = &[
    "blocks",
    "over_budget_ratio",
    "model_errors",
    "algorithmic_latency_ms",
    "engine_load_ms",
    "harness_init_ms",
    "cpu_cores",
    "worker_cpu_cores",
    "vram_engine_mb",
];
const OFFLINE_SCALARS: &[&str] = &["throughput_x"];
const PACED_PREFIXES: &[&str] = &["latency_ms", "block_age_ms"];
const PACED_SCALARS: &[&str] = &[
    "underruns",
    "underrun_ms",
    "dropped_ms",
    "drop_events",
    "overflow_ms",
    "backlog_max_ms",
    "backlog_over_budget_ms",
    "stall_recovery_blocks",
    "stall_latency_recovery_ms",
    "output_trimmed_ms",
    "gpu_load_duty",
    "cpu_load_duty",
];
const E2E_PREFIXES: &[&str] = &["e2e_ms"];
const E2E_SCALARS: &[&str] = &["e2e_detected", "e2e_missed", "e2e_timebase_jitter_ms"];
/// SVE per-component compute (`build: direct` template voices; P3B-A: any template voice through
/// the telemetry spans) and the per-call attribution of `attribution.rs` (P3B-A).
const COMPONENT_PREFIXES: &[&str] = &[
    "content_ms",
    "f0_ms",
    "enc_ms",
    "flowdec_ms",
    // ---- P3B-A ----
    "retrieval_ms",
    "post_ms",
    "glue_ms",
    "overlap_ms",
    "harness_ms",
    "syncs_per_call",
    "gpu_content_ms",
    "gpu_pitch_ms",
    "gpu_enc_ms",
    "gpu_flowdec_ms",
    "gpu_knn_ms",
    "call_ms_onset",
    "call_ms_steady",
    "f0_ms_onset",
    "f0_ms_steady",
    "join_offset_ms",
    "join_step_ms",
    "join_ncc",
];
/// Engine-level and per-run scalars of every engine run (P3B-A): VRAM, CUDA, load and run state.
const ATTRIBUTION_SCALARS: &[&str] = &[
    "onset_calls",
    "joins_silent",
    "core_s_per_audio_s",
    "vram_load_peak_mb",
    "vram_load_overshoot_mb",
    "vram_cuda_context_mb",
    "vram_cuda_device_delta_mb",
    "cuda_stack_limit_kb",
    "cuda_context_ms",
    "generator_builds",
    "run_generator_builds",
    "f0_calls",
    "f0_long_calls",
    "run_load_spans_ms",
    "load_other_ms",
    "load_attributed_ratio",
    "state_sm_clock_mhz",
    "state_mem_clock_mhz",
    "state_speed_index_ms",
    "state_host_probe_ms",
    "state_cpu_effective_mhz",
    "state_decoder_util_max_pct",
];
/// The paced T1 decomposition (P3B-A).
const T1_PREFIXES: &[&str] = &[
    "t1_capture_ms",
    "t1_block_wait_ms",
    "t1_queue_ms",
    "t1_compute_ms",
    "t1_engine_wait_ms",
    "t1_model_ms",
    "t1_hold_ms",
    "t1_output_ms",
];

/// A T1 decomposition metric: a part's distribution, the decomposition of the median latency
/// (`t1_at_p50_<part>_ms`, `t1_at_p50_sum_ms`) or its residual against the latency p50.
fn t1_metric(name: &str) -> bool {
    if prefixed(T1_PREFIXES, name) || name == "t1_residual_p50_ms" || name == "t1_at_p50_sum_ms" {
        return true;
    }
    name.strip_prefix("t1_at_p50_")
        .and_then(|r| r.strip_suffix("_ms"))
        .is_some_and(|part| super::attribution::T1Parts::NAMES.contains(&part))
}
/// The `load_twice` run (P3B-A).
const LOAD_TWICE_PREFIXES: &[&str] = &["background_call_ms"];
const LOAD_TWICE_SCALARS: &[&str] = &[
    "second_load_ms",
    "second_generator_builds",
    "vram_second_voice_mb",
    "vram_second_load_peak_mb",
    "second_load_overshoot_mb",
    "second_load_other_ms",
    "second_load_attributed_ratio",
    "background_over_budget_ratio",
];

/// `<prefix>_<load stage>_ms` (the load breakdown of `attribution::LoadBreakdown`).
fn load_stage_metric(prefix: &str, name: &str) -> bool {
    name.strip_prefix(prefix)
        .and_then(|r| r.strip_prefix('_'))
        .and_then(|r| r.strip_suffix("_ms"))
        .is_some_and(|stage| super::attribution::load_stage_keys().any(|k| k == stage))
}
/// Quality (fidelity against the offline ceiling; plan T5).
pub const QUALITY_SCALARS: &[&str] = &[
    "mcd_db",
    "f0_rmse_cents",
    "gross_pitch_error",
    "vuv_error",
    "speaker_cosine",
    "speaker_cosine_reference",
    "dnsmos_ovrl",
    "dnsmos_sig",
    "dnsmos_bak",
    "ceiling_dnsmos_ovrl",
    "dnsmos_ovrl_delta",
    "cer",
    "cer_ceiling",
    "cer_input",
    "cer_delta_pts",
    "clicks",
    "clicks_per_10min",
    "artifact_events",
    "seam_flux_ratio",
    "ceiling_seam_flux_ratio",
    "f0_frames",
    "offset_ms",
    "quality_seconds",
    "ceiling_compute_ms",
    "ceiling_rtf",
];
/// Compare (competitor head-to-head; plan T6).
pub const COMPARE_SCALARS: &[&str] = &[
    "sigil_dnsmos_ovrl",
    "sigil_dnsmos_sig",
    "sigil_dnsmos_bak",
    "sigil_gross_pitch_error",
    "sigil_f0_rmse_cents",
    "sigil_vuv_error",
    "sigil_speaker_cosine",
    "sigil_latency_ms",
    "sigil_artifact_events_per_min",
    "best_competitor_dnsmos_ovrl",
    "dnsmos_ovrl_margin",
    "min_competitor_gross_pitch_error",
    "gross_pitch_error_margin",
    "best_competitor_latency_ms",
    "latency_ratio",
    "candidates",
    "candidates_scored",
    // Compare v2 (P3B-C; `compare.rs` corpus runs report these in place of the v1 names above).
    "corpus_clips",
    "corpus_clips_dnsmos",
    "points_sigil",
    "points_competitor",
    "points_ablation",
    "points_refused",
    "points_flagged",
    "points_excluded",
    "points_incomplete",
    "clips_unscored",
    "renditions_refused",
    "t6_pass",
    "t6_undecided",
    "t6_rows_pass",
    "t6_rows_fail",
    "t6_rows_not_shown",
    "t6_rows_undecided",
    "t6_rows_refused",
    "t6_rows_excluded",
    "dnsmos_paired_sd",
    "clips_needed_dnsmos_50",
    "clips_needed_dnsmos_80",
];
/// Compare v2 ablation pairings: `ablation_<name>_<metric>_diff_mean|_sd|_lo|_hi` and `_clips`.
fn compare_ablation_metric(name: &str) -> bool {
    name.strip_prefix("ablation_").is_some_and(|rest| {
        ["_diff_mean", "_diff_sd", "_diff_lo", "_diff_hi", "_clips"]
            .iter()
            .any(|suffix| rest.len() > suffix.len() && rest.ends_with(suffix))
    })
}
/// Autotune (the chosen configuration; plan "Autotuner").
pub const AUTOTUNE_SCALARS: &[&str] = &[
    "autotune_block_ms",
    "autotune_extra_context_ms",
    "autotune_rtf_p99",
    "autotune_compute_ms_p95",
    "autotune_latency_ms_p50",
    "autotune_latency_ms_p99",
    "autotune_fidelity",
    "autotune_gpu",
    "autotune_candidates",
    "autotune_seconds",
];
/// Live record (the running voice changer's timeline; plan T1 / T2 / T7 H1 evidence).
const LIVE_PREFIXES: &[&str] = &["latency_ms", "block_ms", "model_ms", "backlog_ms", "queued_out_ms"];
pub const LIVE_SCALARS: &[&str] = &[
    "ticks",
    "live_seconds",
    "underrun_ticks",
    "budget_drop_ticks",
    "gap_ticks",
    "error_ticks",
    "silent_ratio",
    "governor_steps",
    "governor_level_max",
    "dropped_ms",
    "backlog_max_ms",
    "over_budget_ratio",
    "cpu_cores",
    "vram_mb_peak",
    "timeline_dropped",
    "underruns",
];

/// Metrics where a larger value is better (the perf gate and the scorecard invert them).
pub const HIGHER_IS_BETTER: &[&str] = &[
    "throughput_x",
    "e2e_detected",
    "gpu_load_duty",
    "cpu_load_duty",
    "speaker_cosine",
    "speaker_cosine_reference",
    "dnsmos_ovrl",
    "dnsmos_sig",
    "dnsmos_bak",
    "ceiling_dnsmos_ovrl",
    "sigil_dnsmos_ovrl",
    "sigil_dnsmos_sig",
    "sigil_dnsmos_bak",
    "sigil_speaker_cosine",
    "best_competitor_dnsmos_ovrl",
    "dnsmos_ovrl_margin",
    "gross_pitch_error_margin",
    "autotune_fidelity",
    "t6_pass",
];

fn prefixed(prefixes: &[&str], name: &str) -> bool {
    prefixes.iter().any(|p| {
        name.strip_prefix(p)
            .and_then(|rest| rest.strip_prefix('_'))
            .is_some_and(|suffix| SUMMARY_SUFFIXES.contains(&suffix))
    })
}

/// Whether runs of `mode` produce metric `name` (when measurable on the PC).
pub fn known_metric(mode: RunMode, name: &str) -> bool {
    let engine = BLOCK_SCALARS.contains(&name) || ATTRIBUTION_SCALARS.contains(&name) || load_stage_metric("load", name);
    let block = prefixed(BLOCK_PREFIXES, name) || prefixed(COMPONENT_PREFIXES, name) || engine;
    let paced = block || prefixed(PACED_PREFIXES, name) || PACED_SCALARS.contains(&name) || t1_metric(name);
    match mode {
        RunMode::Offline => block || OFFLINE_SCALARS.contains(&name),
        RunMode::Paced | RunMode::Contention => paced,
        RunMode::E2eLoopback => prefixed(E2E_PREFIXES, name) || E2E_SCALARS.contains(&name),
        RunMode::Quality => block || QUALITY_SCALARS.contains(&name) || super::quality::QUALITY_V2_SCALARS.contains(&name),
        RunMode::Compare => COMPARE_SCALARS.contains(&name) || compare_ablation_metric(name),
        RunMode::Autotune => AUTOTUNE_SCALARS.contains(&name),
        RunMode::LiveRecord => prefixed(LIVE_PREFIXES, name) || LIVE_SCALARS.contains(&name),
        // A rescore scores like a quality run (v1 and v2 metrics) and counts its clips and pairs.
        RunMode::Rescore => {
            QUALITY_SCALARS.contains(&name)
                || super::quality::QUALITY_V2_SCALARS.contains(&name)
                || matches!(name, "clips" | "pairs")
        }
        // P3B-A: the second voice's load, the background voice, and the spec engine's scalars.
        RunMode::LoadTwice => {
            engine || LOAD_TWICE_SCALARS.contains(&name) || prefixed(LOAD_TWICE_PREFIXES, name) || load_stage_metric("second_load", name)
        }
    }
}

/// Why `run`'s configuration never produces metric `name` although its mode can: the stall
/// metrics need `stalls`, the load duties their load, the CPU figures real compute on the clock
/// (not `simulated`), CER an STT engine, the reference speaker cosine a reference clip, and the
/// compare margins / latencies their inputs. `None` when the run produces it.
pub fn unproduced_reason(run: &RunSpec, name: &str) -> Option<&'static str> {
    if prefixed(COMPONENT_PREFIXES, name) || ATTRIBUTION_SCALARS.contains(&name) || load_stage_metric("load", name) {
        // Only a template voice reports its components, a CUDA engine its context, a GPU its
        // clocks, ...; the spec cannot tell.
        return None;
    }
    let paced_reason = |stalls: bool, gpu: bool, cpu: bool, clock: ClockKind| -> Option<&'static str> {
        match name {
            "stall_recovery_blocks" | "stall_latency_recovery_ms" if !stalls => Some("the run has no stalls"),
            "gpu_load_duty" if !gpu => Some("the run has no load.gpu"),
            "cpu_load_duty" if !cpu => Some("the run has no load.cpu"),
            "cpu_cores" | "worker_cpu_cores" if clock == ClockKind::Simulated => {
                Some("the simulated clock charges no real compute, so the run reports no CPU figures")
            }
            _ => None,
        }
    };
    match run {
        RunSpec::Paced(p) => paced_reason(!p.stalls.is_empty(), p.load.gpu.is_some(), p.load.cpu.is_some(), p.clock),
        RunSpec::Contention(c) => {
            let (load, stalls) = super::contention::profile_parts(c);
            paced_reason(!stalls.is_empty(), load.gpu.is_some(), load.cpu.is_some(), c.clock)
        }
        RunSpec::Quality(q) => match name {
            "cer" | "cer_ceiling" | "cer_input" | "cer_delta_pts" if q.stt.is_none() => Some("the run has no stt"),
            "speaker_cosine_reference" if q.scoring.speaker_reference.is_none() => Some("the run has no scoring.speaker_reference"),
            "speaker_cosine" if q.scoring.speaker_model.trim().eq_ignore_ascii_case("off") => Some("scoring.speaker_model is off"),
            "dnsmos_ovrl" | "dnsmos_sig" | "dnsmos_bak" | "ceiling_dnsmos_ovrl" | "dnsmos_ovrl_delta"
                if q.scoring.dnsmos_model.trim().eq_ignore_ascii_case("off") =>
            {
                Some("scoring.dnsmos_model is off")
            }
            _ => None,
        },
        RunSpec::Compare(c) => {
            // A corpus run reports the v2 metrics (`details.v2`); the v1 single-clip names are not
            // produced, and the v2 names need no inline candidates.
            let v1_name = name.starts_with("sigil_")
                || name.starts_with("best_competitor_")
                || name.starts_with("min_competitor_")
                || name.ends_with("_margin")
                || matches!(name, "latency_ratio" | "candidates" | "candidates_scored");
            if c.corpus.is_some() {
                return v1_name.then_some("a corpus run reports the v2 metrics in details.v2, not the single-clip names");
            }
            if !v1_name {
                return Some("the v2 metrics need compare.corpus");
            }
            let sigil_latency = c.sigil.include && (c.sigil.latency_ms.is_some() || !c.sigil.latency_from.trim().is_empty());
            let competitor_latency = c.candidates.iter().any(|x| x.latency_ms.is_some());
            match name {
                n if n.starts_with("sigil_") && !c.sigil.include => Some("sigil.include is false"),
                "sigil_latency_ms" if !sigil_latency => Some("the run has no sigil.latency_ms or sigil.latency_from"),
                "sigil_speaker_cosine" if c.scoring.speaker_reference.is_none() => Some("the run has no scoring.speaker_reference"),
                "best_competitor_dnsmos_ovrl" | "min_competitor_gross_pitch_error" if c.candidates.is_empty() => Some("the run has no candidates"),
                "dnsmos_ovrl_margin" | "gross_pitch_error_margin" if c.candidates.is_empty() || !c.sigil.include => {
                    Some("a margin needs sigil.include and candidates")
                }
                "best_competitor_latency_ms" if !competitor_latency => Some("no candidate has a latency_ms"),
                "latency_ratio" if !(sigil_latency && competitor_latency) => Some("the ratio needs Sigil's and a candidate's latency"),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Whether `run` produces metric `name` (when measurable on the PC): its mode reports it and its
/// configuration asks for it. A run that should produce a metric and did not measured nothing
/// (e.g. no audio ever played), which thresholds treat as a failure.
pub fn run_produces(run: &RunSpec, name: &str) -> bool {
    known_metric(run.mode(), name) && unproduced_reason(run, name).is_none()
}

/// One block of a paced or offline run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TickRecord {
    pub tick: u64,
    /// Block start on the run's clock.
    pub t_ms: f64,
    /// Age of the block's newest captured frame at processing start (paced).
    pub age_ms: f64,
    /// Worker time for the block (conversion, concealment, post), injected stall excluded.
    pub compute_ms: f64,
    /// Model `convert` time within it.
    pub model_ms: f64,
    /// Model `convert` calls within it (0 while a fixed-block engine accumulates its block).
    pub calls: u32,
    pub stall_ms: f64,
    /// Capture waiting after this wake's budget enforcement (paced).
    pub waiting_ms: f64,
    /// Input dropped at this wake (budget + ring overflow; paced).
    pub dropped_ms: f64,
    /// Output queued for playback after the push (paced).
    pub queued_out_ms: f64,
    pub error: bool,
}

const TIMELINE_HEADER: &str = "tick,t_ms,age_ms,compute_ms,model_ms,calls,stall_ms,waiting_ms,dropped_ms,queued_out_ms,error";

/// Timeline as CSV.
pub fn timeline_csv(ticks: &[TickRecord]) -> String {
    let mut s = String::with_capacity(ticks.len() * 64 + 128);
    s.push_str(TIMELINE_HEADER);
    s.push('\n');
    for t in ticks {
        let _ = writeln!(
            s,
            "{},{:.3},{:.3},{:.4},{:.4},{},{:.1},{:.3},{:.3},{:.3},{}",
            t.tick,
            t.t_ms,
            t.age_ms,
            t.compute_ms,
            t.model_ms,
            t.calls,
            t.stall_ms,
            t.waiting_ms,
            t.dropped_ms,
            t.queued_out_ms,
            u8::from(t.error)
        );
    }
    s
}

/// A driver's result, before thresholds.
#[derive(Debug, Default)]
pub struct RunOutcome {
    pub metrics: Metrics,
    pub details: serde_json::Value,
    pub timeline: Vec<TickRecord>,
    /// `(interleaved, rate, channels)`.
    pub audio: Option<(Vec<f32>, u32, usize)>,
    /// More audio files: `(suffix, interleaved, rate, channels)`, written as
    /// `<run>.<suffix>.wav` (a quality run's ceiling).
    pub extra_audio: Vec<(String, Vec<f32>, u32, usize)>,
}

/// Status of one run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Passed,
    /// Completed, but a threshold failed.
    Failed,
    /// Could not complete (engine, device or driver error).
    Error,
    Cancelled,
}

/// One threshold check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThresholdResult {
    pub metric: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    /// `None`: the run did not measure the metric.
    pub value: Option<f64>,
    pub passed: bool,
    /// `spec` (spec-wide) or `run`.
    pub scope: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunReport {
    pub name: String,
    pub mode: RunMode,
    pub status: RunStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub duration_ms: f64,
    pub metrics: Metrics,
    pub thresholds: Vec<ThresholdResult>,
    pub details: serde_json::Value,
    /// Extra files written for this run (relative to the report folder).
    pub files: Vec<String>,
    /// Metrics of this run where a larger value is better (P3-F; the perf gate reads it).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub higher_is_better: Vec<String>,
}

/// Whether `name` is scored by a pitch analyzer (F0 RMSE, gross / fine pitch error, V/UV, the
/// fed-F0 metrics): such a comparison across reports needs the same analyzer (P3B-A; the analyzer
/// stamp is P3B-B's `analyzer_version`). Compute times (`f0_ms_*`) are not F0 metrics.
#[allow(dead_code)] // The perf gate (PowerShell) mirrors it; tests prove the rule.
pub fn is_f0_metric(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    ["f0_rmse", "pitch_error", "vuv", "cents", "gpe", "fpe", "fed_f0", "onset_lag"]
        .iter()
        .any(|k| n.contains(k))
}

/// Whether `key` holds a pitch analyzer's version stamp: `analyzer_version`, or a prefixed one
/// such as P3B-B's `details.scorers.f0_analyzer_version`.
#[allow(dead_code)] // As above.
pub fn is_analyzer_stamp(key: &str) -> bool {
    key == "analyzer_version" || key.ends_with("_analyzer_version")
}

/// Every analyzer version stamp in a report (any depth: report, engine, run details; see
/// [`is_analyzer_stamp`]).
#[allow(dead_code)] // As above.
pub fn analyzer_versions(report: &serde_json::Value) -> std::collections::BTreeSet<String> {
    fn walk(v: &serde_json::Value, out: &mut std::collections::BTreeSet<String>) {
        match v {
            serde_json::Value::Object(o) => {
                for (k, x) in o {
                    if is_analyzer_stamp(k) {
                        match x {
                            serde_json::Value::String(s) => {
                                out.insert(s.clone());
                            }
                            serde_json::Value::Null => {}
                            other => {
                                out.insert(other.to_string());
                            }
                        }
                    } else {
                        walk(x, out);
                    }
                }
            }
            serde_json::Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    let mut out = std::collections::BTreeSet::new();
    walk(report, &mut out);
    out
}

/// Why the F0 metrics of report sets `a` (before) and `b` (after) cannot be compared: their
/// analyzer stamps differ (a report without a stamp, from a build before the stamp existed, counts
/// as its own analyzer). `None` when they may be.
#[allow(dead_code)] // As above.
pub fn f0_comparison_refusal(a: &[&serde_json::Value], b: &[&serde_json::Value]) -> Option<String> {
    let stamps = |set: &[&serde_json::Value]| -> std::collections::BTreeSet<String> {
        set.iter()
            .map(|r| {
                let v = analyzer_versions(r);
                if v.is_empty() {
                    "<none>".to_string()
                } else {
                    v.into_iter().collect::<Vec<_>>().join("+")
                }
            })
            .collect()
    };
    let (sa, sb) = (stamps(a), stamps(b));
    (sa != sb || sa.len() > 1).then(|| {
        format!(
            "F0 metrics come from different analyzers (before: {}; after: {}): rescore both with one analyzer",
            sa.into_iter().collect::<Vec<_>>().join(", "),
            sb.into_iter().collect::<Vec<_>>().join(", ")
        )
    })
}

/// Distributions where a larger value is better (P3B-A: the SOLA join correlation).
const HIGHER_IS_BETTER_PREFIXES: &[&str] = &["join_ncc"];

/// Whether a larger value of `name` is better.
pub fn higher_is_better(name: &str) -> bool {
    HIGHER_IS_BETTER.contains(&name) || super::quality::QUALITY_V2_HIGHER_IS_BETTER.contains(&name) || prefixed(HIGHER_IS_BETTER_PREFIXES, name)
}

/// The metrics of `metrics` where a larger value is better (from [`HIGHER_IS_BETTER`]).
pub fn higher_is_better_of(metrics: &Metrics) -> Vec<String> {
    metrics.keys().filter(|k| higher_is_better(k)).cloned().collect()
}

/// Where and on what the benchmark ran.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Environment {
    pub sigil_version: String,
    pub build: String,
    pub os: String,
    pub arch: String,
    pub cpu: Option<String>,
    pub logical_cores: usize,
    pub onnx_runtime: String,
    pub gpus: Vec<serde_json::Value>,
    /// The voice changer was running during the benchmark (its numbers are then contended).
    pub voice_changer_running: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub schema: String,
    pub name: String,
    pub description: String,
    pub passed: bool,
    /// One line per failed run / threshold.
    pub failures: Vec<String>,
    pub started_unix_ms: u64,
    pub duration_ms: f64,
    /// sha256 of the canonical spec JSON (same spec = same hash, for comparing reports).
    pub spec_sha256: String,
    pub spec: serde_json::Value,
    pub environment: Environment,
    /// The engine every offline / paced run used (`null` for loopback-only specs). P3B-A: its
    /// `attribution` object holds the load breakdown and VRAM ledger, the CUDA context and arena,
    /// the host state and QoS, and the algorithmic latency table.
    pub engine: Option<serde_json::Value>,
    pub runs: Vec<RunReport>,
}

/// Check a run's `metrics` against its own thresholds (`run` scope) and the spec-wide ones (`spec`
/// scope). A spec-wide threshold applies to every run that produces the metric
/// ([`run_produces`]) and is skipped for the others. An applicable threshold whose metric the run
/// did not report fails as "not measured" in either scope: a run that measured nothing (no audio
/// ever played, no burst detected) never passes a gate by omission.
pub fn evaluate(metrics: &Metrics, spec_wide: &Thresholds, run: &RunSpec) -> Vec<ThresholdResult> {
    let check = |metric: &str, limit: &Limit, scope: &str| -> ThresholdResult {
        let value = metrics.get(metric).copied();
        let passed = value.is_some_and(|v| limit.min.is_none_or(|lo| v >= lo) && limit.max.is_none_or(|hi| v <= hi));
        ThresholdResult {
            metric: metric.to_string(),
            min: limit.min,
            max: limit.max,
            value,
            passed,
            scope: scope.to_string(),
        }
    };
    let own = run.thresholds();
    let mut out: Vec<ThresholdResult> = own.iter().map(|(m, l)| check(m, l, "run")).collect();
    for (m, l) in spec_wide {
        // A run threshold on the same metric overrides the spec-wide one.
        if !own.contains_key(m) && run_produces(run, m) {
            out.push(check(m, l, "spec"));
        }
    }
    out
}

fn limit_text(t: &ThresholdResult) -> String {
    match (t.min, t.max) {
        (Some(lo), Some(hi)) => format!("{lo} ..= {hi}"),
        (Some(lo), None) => format!(">= {lo}"),
        (None, Some(hi)) => format!("<= {hi}"),
        (None, None) => String::new(),
    }
}

fn fmt_value(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e12 {
        format!("{v:.0}")
    } else if v.abs() >= 100.0 {
        format!("{v:.1}")
    } else if v.abs() >= 1.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.4}")
    }
}

/// Failure lines of a run (for the report summary and the CLI).
pub fn run_failures(run: &RunReport) -> Vec<String> {
    let mut f = Vec::new();
    match run.status {
        RunStatus::Error => f.push(format!("{}: error: {}", run.name, run.error.as_deref().unwrap_or("unknown"))),
        RunStatus::Cancelled => f.push(format!("{}: cancelled", run.name)),
        _ => {}
    }
    for t in run.thresholds.iter().filter(|t| !t.passed) {
        let got = t.value.map_or_else(|| "not measured".to_string(), fmt_value);
        f.push(format!("{}: {} = {got} (limit {})", run.name, t.metric, limit_text(t)));
    }
    f
}

/// Headline metrics per mode, in display order.
fn headline(mode: RunMode) -> &'static [(&'static str, &'static str)] {
    match mode {
        RunMode::Quality => &[
            ("speaker_cosine", "speaker cos"),
            ("mcd_db", "MCD dB"),
            ("f0_rmse_cents", "F0 RMSE c"),
            ("gross_pitch_error", "GPE"),
            ("vuv_error", "V/UV err"),
            ("clicks", "clicks"),
            ("dnsmos_ovrl", "DNSMOS"),
            ("cer_delta_pts", "CER +pts"),
            ("block_ms_p95", "block p95 ms"),
        ],
        RunMode::Compare => &[
            ("sigil_dnsmos_ovrl", "Sigil DNSMOS"),
            ("dnsmos_ovrl_margin", "DNSMOS margin"),
            ("sigil_gross_pitch_error", "Sigil GPE"),
            ("gross_pitch_error_margin", "GPE margin"),
            ("latency_ratio", "latency ratio"),
            ("candidates_scored", "candidates"),
            ("t6_pass", "T6 pass"),
            ("corpus_clips", "corpus clips"),
            ("points_refused", "points refused"),
            ("dnsmos_paired_sd", "DNSMOS paired SD"),
        ],
        RunMode::Contention => &[
            ("latency_ms_p50", "latency p50 ms"),
            ("latency_ms_p99", "latency p99 ms"),
            ("block_ms_p95", "block p95 ms"),
            ("block_ms_p99", "block p99 ms"),
            ("over_budget_ratio", "over budget"),
            ("underruns", "underruns"),
            ("stall_recovery_blocks", "stall recovery blocks"),
            ("gpu_load_duty", "GPU load"),
            ("cpu_cores", "CPU cores"),
        ],
        RunMode::Autotune => &[
            ("autotune_block_ms", "block ms"),
            ("autotune_rtf_p99", "RTF p99"),
            ("autotune_latency_ms_p50", "latency p50 ms"),
            ("autotune_fidelity", "fidelity"),
            ("autotune_candidates", "candidates"),
        ],
        RunMode::LiveRecord => &[
            ("latency_ms_p50", "latency p50 ms"),
            ("latency_ms_p99", "latency p99 ms"),
            ("block_ms_p95", "block p95 ms"),
            ("block_ms_p99", "block p99 ms"),
            ("underrun_ticks", "underrun ticks"),
            ("dropped_ms", "dropped ms"),
            ("cpu_cores", "CPU cores"),
            ("ticks", "ticks"),
        ],
        RunMode::Rescore => &[
            ("speaker_cosine", "speaker cos"),
            ("mcd_db", "MCD dB"),
            ("f0_rmse_cents", "F0 RMSE c"),
            ("vuv_error", "V/UV err"),
            ("clicks", "clicks"),
        ],
        RunMode::LoadTwice => &[
            ("engine_load_ms", "cold load ms"),
            ("second_load_ms", "warm load ms"),
            ("vram_engine_mb", "VRAM MB"),
            ("vram_second_voice_mb", "2nd voice MB"),
            ("vram_second_load_peak_mb", "2nd load peak MB"),
            ("background_call_ms_p99", "background p99 ms"),
        ],
        RunMode::Offline => &[
            ("block_ms_p50", "block p50 ms"),
            ("block_ms_p95", "block p95 ms"),
            ("block_ms_p99", "block p99 ms"),
            ("rtf_p99", "RTF p99"),
            ("throughput_x", "throughput x"),
            ("algorithmic_latency_ms", "algorithmic ms"),
            ("cpu_cores", "CPU cores"),
        ],
        RunMode::Paced => &[
            ("latency_ms_p50", "latency p50 ms"),
            ("latency_ms_p99", "latency p99 ms"),
            ("block_ms_p95", "block p95 ms"),
            ("block_ms_p99", "block p99 ms"),
            ("rtf_p99", "RTF p99"),
            ("underruns", "underruns"),
            ("dropped_ms", "dropped ms"),
            ("backlog_max_ms", "backlog max ms"),
            ("stall_recovery_blocks", "stall recovery blocks"),
            ("cpu_cores", "CPU cores"),
        ],
        RunMode::E2eLoopback => &[
            ("e2e_ms_p50", "e2e p50 ms"),
            ("e2e_ms_p95", "e2e p95 ms"),
            ("e2e_ms_p99", "e2e p99 ms"),
            ("e2e_ms_stddev", "e2e stddev ms"),
            ("e2e_detected", "detected"),
            ("e2e_missed", "missed"),
        ],
    }
}

/// One-line summary of a run (CLI output).
pub fn run_line(run: &RunReport) -> String {
    let status = match run.status {
        RunStatus::Passed => "PASS",
        RunStatus::Failed => "FAIL",
        RunStatus::Error => "ERROR",
        RunStatus::Cancelled => "CANCELLED",
    };
    let mut line = format!("  {:<5} {:<14} {:<12}", status, run.name, run.mode.as_str());
    for (key, label) in headline(run.mode) {
        if let Some(v) = run.metrics.get(*key) {
            let _ = write!(line, "  {label} {}", fmt_value(*v));
        }
    }
    if let Some(e) = &run.error {
        let _ = write!(line, "  ({e})");
    }
    line
}

/// Markdown rendering of a report.
pub fn to_markdown(r: &BenchReport) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# Benchmark: {}\n", r.name);
    if !r.description.is_empty() {
        let _ = writeln!(s, "{}\n", r.description);
    }
    let _ = writeln!(
        s,
        "**Result: {}** · {} run(s) · {:.1} s · spec `{}`\n",
        if r.passed { "PASSED" } else { "FAILED" },
        r.runs.len(),
        r.duration_ms / 1000.0,
        &r.spec_sha256[..r.spec_sha256.len().min(12)]
    );
    let e = &r.environment;
    let _ = writeln!(s, "## Environment\n");
    let _ = writeln!(s, "| | |\n|---|---|");
    let _ = writeln!(s, "| Sigil | {} ({} build) |", e.sigil_version, e.build);
    let _ = writeln!(s, "| OS | {} {} |", e.os, e.arch);
    let _ = writeln!(
        s,
        "| CPU | {} ({} logical cores) |",
        e.cpu.as_deref().unwrap_or("unknown"),
        e.logical_cores
    );
    let _ = writeln!(s, "| ONNX Runtime | {} |", e.onnx_runtime);
    for g in &e.gpus {
        let name = g.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        let driver = g.get("driver_version").and_then(|v| v.as_str()).unwrap_or("?");
        let vram = g.get("dedicated_vram_mb").and_then(|v| v.as_u64()).unwrap_or(0);
        let _ = writeln!(s, "| GPU | {name} ({vram} MB, driver {driver}) |");
    }
    if e.voice_changer_running {
        let _ = writeln!(s, "| Note | the voice changer was running during the benchmark |");
    }
    if let Some(engine) = &r.engine {
        let _ = writeln!(s, "\n## Engine\n");
        let get = |k: &str| engine.get(k).map(|v| v.to_string().trim_matches('"').to_string());
        let _ = writeln!(s, "| | |\n|---|---|");
        for key in [
            "description",
            "output_rate",
            "execution_provider",
            "execution_provider_active",
            "execution_provider_reason",
            "pitch_extractor",
            "load_ms",
        ] {
            if let Some(v) = get(key).filter(|v| v != "null") {
                let _ = writeln!(s, "| {key} | {v} |");
            }
        }
        for w in engine.get("warnings").and_then(|w| w.as_array()).into_iter().flatten() {
            if let Some(w) = w.as_str().filter(|w| !w.is_empty()) {
                let _ = writeln!(s, "| warning | {} |", w.replace('|', "/"));
            }
        }
        if let Some(a) = engine.get("attribution") {
            attribution_markdown(&mut s, a);
        }
    }
    let _ = writeln!(s, "\n## Runs\n");
    let _ = writeln!(s, "| Run | Mode | Status | Headline |\n|---|---|---|---|");
    for run in &r.runs {
        let head: Vec<String> = headline(run.mode)
            .iter()
            .filter_map(|(k, label)| run.metrics.get(*k).map(|v| format!("{label} {}", fmt_value(*v))))
            .collect();
        let status = match run.status {
            RunStatus::Passed => "PASS",
            RunStatus::Failed => "FAIL",
            RunStatus::Error => "ERROR",
            RunStatus::Cancelled => "CANCELLED",
        };
        let _ = writeln!(s, "| {} | {} | {} | {} |", run.name, run.mode.as_str(), status, head.join(" · "));
    }
    if !r.failures.is_empty() {
        let _ = writeln!(s, "\n## Failures\n");
        for f in &r.failures {
            let _ = writeln!(s, "- {f}");
        }
    }
    for run in &r.runs {
        let _ = writeln!(s, "\n## {} ({})\n", run.name, run.mode.as_str());
        if let Some(e) = &run.error {
            let _ = writeln!(s, "Error: {e}\n");
        }
        if !run.thresholds.is_empty() {
            let _ = writeln!(s, "| Threshold | Limit | Value | Result |\n|---|---|---|---|");
            for t in &run.thresholds {
                let _ = writeln!(
                    s,
                    "| {} | {} | {} | {} |",
                    t.metric,
                    limit_text(t),
                    t.value.map_or_else(|| "not measured".into(), fmt_value),
                    if t.passed { "pass" } else { "FAIL" }
                );
            }
            s.push('\n');
        }
        let _ = writeln!(s, "| Metric | Value |\n|---|---|");
        for (k, v) in &run.metrics {
            let _ = writeln!(s, "| {k} | {} |", fmt_value(*v));
        }
        if let Some(rows) = run.details.get("rows").and_then(|r| r.as_array()) {
            compare_rows_markdown(&mut s, rows);
        }
        if let Some(cands) = run.details.get("candidates").and_then(|r| r.as_array()) {
            autotune_markdown(&mut s, &run.details, cands);
        }
        run_attribution_markdown(&mut s, &run.details);
        if let Some(notes) = run.details.get("notes").and_then(|r| r.as_array()).filter(|n| !n.is_empty()) {
            let _ = writeln!(s, "\nNotes:");
            for n in notes.iter().filter_map(|n| n.as_str()) {
                let _ = writeln!(s, "- {n}");
            }
        }
        if !run.files.is_empty() {
            let _ = writeln!(s, "\nFiles: {}", run.files.join(", "));
        }
    }
    s
}

fn cell(v: Option<&serde_json::Value>) -> String {
    match v {
        Some(serde_json::Value::Number(n)) => n.as_f64().map(fmt_value).unwrap_or_default(),
        Some(serde_json::Value::String(t)) => t.clone(),
        Some(serde_json::Value::Bool(b)) => b.to_string(),
        _ => "-".to_string(),
    }
}

/// The per-candidate table of a compare run (`details.rows`).
fn compare_rows_markdown(s: &mut String, rows: &[serde_json::Value]) {
    let _ = writeln!(
        s,
        "\n| Candidate | DNSMOS OVRL | SIG | BAK | GPE | F0 RMSE c | V/UV err | Speaker cos | Latency ms | Artifact events / min | Offset ms |\n|---|---|---|---|---|---|---|---|---|---|---|"
    );
    for r in rows {
        let q = r.get("quality");
        let g = |k: &str| cell(q.and_then(|q| q.get(k)));
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            cell(r.get("name")),
            g("dnsmos_ovrl"),
            g("dnsmos_sig"),
            g("dnsmos_bak"),
            g("gross_pitch_error"),
            g("f0_rmse_cents"),
            g("vuv_error"),
            g("speaker_cosine"),
            cell(r.get("latency_ms")),
            cell(r.get("artifact_events_per_min")),
            g("offset_ms"),
        );
    }
}

fn num(v: Option<&serde_json::Value>) -> Option<f64> {
    v.and_then(serde_json::Value::as_f64)
}

/// The engine's `attribution` (P3B-A): load breakdown, VRAM, CUDA and the algorithmic latency
/// table.
fn attribution_markdown(s: &mut String, a: &serde_json::Value) {
    if let Some(load) = a.get("load") {
        let _ = writeln!(s, "\n### Load breakdown\n");
        let window = num(load.get("window_ms")).unwrap_or(0.0);
        let _ = writeln!(
            s,
            "Window {} ms, {} generator build(s), {} re-plan(s), {} thread(s); wall-clock time by stage (parallel spans split the instant):\n",
            fmt_value(window),
            cell(load.get("generator_builds")),
            cell(load.get("replans")),
            cell(load.get("threads"))
        );
        let _ = writeln!(s, "| Stage | ms | share |\n|---|---|---|");
        let mut rows: Vec<(String, f64)> = load
            .get("stages_ms")
            .and_then(|m| m.as_object())
            .map(|m| m.iter().filter_map(|(k, v)| v.as_f64().map(|v| (k.clone(), v))).collect())
            .unwrap_or_default();
        rows.sort_by(|x, y| y.1.total_cmp(&x.1));
        if let Some(o) = num(load.get("other_ms")) {
            rows.push(("other (no span)".into(), o));
        }
        for (k, v) in rows {
            let share = if window > 0.0 { format!("{:.1} %", 100.0 * v / window) } else { "-".into() };
            let _ = writeln!(s, "| {k} | {} | {share} |", fmt_value(v));
        }
    }
    if let Some(v) = a.get("vram") {
        let _ = writeln!(s, "\n### VRAM\n");
        let _ = writeln!(s, "| | MB |\n|---|---|");
        for key in ["engine_mb", "load_peak_mb", "cuda_context_mb", "cuda_device_delta_mb", "cuda_stack_limit_kb"] {
            if let Some(x) = num(v.get(key)) {
                let _ = writeln!(s, "| {key} | {} |", fmt_value(x));
            }
        }
        for (k, x) in v.get("ledger").and_then(|l| l.as_object()).into_iter().flatten() {
            if let Some(x) = x.as_f64() {
                let _ = writeln!(s, "| ledger {k} | {} |", fmt_value(x));
            }
        }
        for (k, x) in v.get("measured").and_then(|l| l.as_object()).into_iter().flatten() {
            if let Some(mb) = x.get("mb").and_then(|m| m.as_f64()) {
                let overlap = if x.get("concurrent").and_then(|c| c.as_bool()).unwrap_or(false) { " (overlapped another thread's load)" } else { "" };
                let _ = writeln!(s, "| measured {k}{overlap} | {} |", fmt_value(mb));
            }
        }
        for (k, x) in v.get("marks").and_then(|l| l.as_object()).into_iter().flatten() {
            if let Some(x) = x.as_f64() {
                let _ = writeln!(s, "| mark {k} | {} |", fmt_value(x));
            }
        }
    }
    if let Some(rows) = a.get("algorithmic_latency_table").and_then(|r| r.as_array()).filter(|r| !r.is_empty()) {
        let _ = writeln!(s, "\n### Algorithmic latency by device block\n");
        let _ = writeln!(s, "| Engine block ms | Device frames | Device block ms | Latency ms |\n|---|---|---|---|");
        for r in rows {
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} |",
                cell(r.get("engine_block_ms")),
                cell(r.get("device_frames")),
                cell(r.get("device_block_ms")),
                cell(r.get("latency_ms"))
            );
        }
    }
}

/// A run's state, T1 decomposition and p95 shares (P3B-A).
fn run_attribution_markdown(s: &mut String, d: &serde_json::Value) {
    if let Some(st) = d.get("state") {
        let sum = st.get("summary");
        let g = |k: &str| sum.and_then(|x| x.get(k));
        let cpu = st.get("cpu").and_then(|c| c.get("effective_mhz")).and_then(|v| v.as_f64());
        let _ = writeln!(
            s,
            "\nState: SM {} MHz (min {}), P-states {}, reasons {}, GPU decoder max {} %, speed index {} ms, host probe {} ms, CPU {} MHz",
            cell(g("sm_clock_median_mhz")),
            cell(g("sm_clock_min_mhz")),
            g("pstates").map_or_else(|| "-".to_string(), |v| v.to_string()),
            g("reasons").map_or_else(|| "-".to_string(), |v| v.to_string()),
            cell(g("decoder_util_max_pct")),
            cell(st.get("start").and_then(|x| x.get("speed_index_ms"))),
            cell(st.get("start").and_then(|x| x.get("host_probe_ms"))),
            cpu.map_or_else(|| "-".to_string(), |c| format!("{c:.0}"))
        );
        for w in st.get("warnings").and_then(|w| w.as_array()).into_iter().flatten().filter_map(|w| w.as_str()) {
            let _ = writeln!(s, "- state: {w}");
        }
    }
    if let Some(parts) = d.get("t1").and_then(|t| t.get("parts")).and_then(|p| p.as_object()) {
        let t1 = d.get("t1");
        let at = t1.and_then(|t| t.get("at_p50"));
        let _ = writeln!(s, "\n| T1 part | at the p50 ms | mean ms | p50 ms | p99 ms |\n|---|---|---|---|---|");
        for name in super::attribution::T1Parts::NAMES {
            if let Some(p) = parts.get(name) {
                let _ = writeln!(
                    s,
                    "| {name} | {} | {} | {} | {} |",
                    cell(at.and_then(|a| a.get(name))),
                    cell(p.get("mean")),
                    cell(p.get("p50")),
                    cell(p.get("p99"))
                );
            }
        }
        let _ = writeln!(
            s,
            "| sum | {} | {} | {} | latency p50 {} |",
            cell(t1.and_then(|t| t.get("at_p50_sum_ms"))),
            cell(t1.and_then(|t| t.get("sum_of_means_ms"))),
            cell(t1.and_then(|t| t.get("sum_of_p50_ms"))),
            cell(t1.and_then(|t| t.get("latency_p50_ms")))
        );
    }
    if let Some(sh) = d.get("p95_shares").and_then(|x| x.as_object()).filter(|m| !m.is_empty()) {
        let parts: Vec<String> = sh
            .iter()
            .filter_map(|(k, v)| v.as_f64().map(|v| format!("{k} {:.0} %", v * 100.0)))
            .collect();
        let _ = writeln!(s, "\nShare of the slowest 5 % of model calls: {}", parts.join(", "));
    }
}

/// The candidates an autotune run measured and the configuration it chose.
fn autotune_markdown(s: &mut String, details: &serde_json::Value, cands: &[serde_json::Value]) {
    if let Some(params) = details.get("result").and_then(|r| r.get("params")) {
        let _ = writeln!(s, "\nChosen parameters: `{params}`");
    }
    let _ = writeln!(
        s,
        "\n| Stage | EP | F0 | Block ms | Context ms | Precision | Levers | RTF p99 | Compute p95 ms | Latency p50 ms | Fidelity | OK | Note |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    );
    for c in cands {
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            cell(c.get("stage")),
            cell(c.get("ep")),
            cell(c.get("pitch")),
            cell(c.get("block_ms")),
            cell(c.get("extra_context_ms")),
            cell(c.get("precision")),
            cell(c.get("levers")),
            cell(c.get("rtf_p99")),
            cell(c.get("compute_ms_p95")),
            cell(c.get("latency_ms_p50")),
            cell(c.get("fidelity")),
            cell(c.get("ok")),
            cell(c.get("note")),
        );
    }
}

/// Write `report.json` and `report.md` into `dir`; returns their paths.
pub fn write_report(dir: &Path, report: &BenchReport) -> Result<(PathBuf, PathBuf), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let json = serde_json::to_string_pretty(report).map_err(|e| format!("report: {e}"))?;
    let json_path = dir.join("report.json");
    let md_path = dir.join("report.md");
    write_atomic(&json_path, json.as_bytes())?;
    write_atomic(&md_path, to_markdown(report).as_bytes())?;
    Ok((json_path, md_path))
}

/// Write through a temp file and rename, so a crash never leaves half a report.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_names_are_scoped_by_mode() {
        assert!(known_metric(RunMode::Offline, "block_ms_p95"));
        assert!(known_metric(RunMode::Offline, "throughput_x"));
        assert!(!known_metric(RunMode::Offline, "latency_ms_p99"));
        assert!(known_metric(RunMode::Paced, "latency_ms_p99"));
        assert!(known_metric(RunMode::Paced, "rtf_p99"));
        assert!(known_metric(RunMode::Paced, "stall_recovery_blocks"));
        assert!(!known_metric(RunMode::Paced, "block_ms_p97"));
        assert!(!known_metric(RunMode::Paced, "block_ms"));
        assert!(known_metric(RunMode::E2eLoopback, "e2e_ms_p50"));
        assert!(!known_metric(RunMode::E2eLoopback, "block_ms_p50"));
    }

    /// P3B-A metric names: the per-call attribution, load, VRAM, state, T1 and `load_twice` ones.
    #[test]
    fn attribution_metrics_are_known() {
        for m in ["glue_ms_p95", "harness_ms_p50", "syncs_per_call_p50", "gpu_enc_ms_p99", "call_ms_onset_p95", "join_ncc_p50", "onset_calls"] {
            assert!(known_metric(RunMode::Offline, m) && known_metric(RunMode::Contention, m), "{m}");
        }
        for m in ["load_warm_once_ms", "load_replan_configure_stream_ms", "load_other_ms", "vram_load_peak_mb", "vram_cuda_context_mb", "state_sm_clock_mhz"] {
            assert!(known_metric(RunMode::Paced, m) && known_metric(RunMode::Quality, m), "{m}");
        }
        assert!(!known_metric(RunMode::Offline, "load_bogus_ms"));
        assert!(known_metric(RunMode::Paced, "t1_hold_ms_p50") && known_metric(RunMode::Paced, "t1_residual_p50_ms"));
        assert!(known_metric(RunMode::Paced, "t1_at_p50_engine_wait_ms") && known_metric(RunMode::Contention, "t1_at_p50_sum_ms"));
        assert!(!known_metric(RunMode::Paced, "t1_at_p50_bogus_ms") && !known_metric(RunMode::Offline, "t1_at_p50_hold_ms"));
        assert!(!known_metric(RunMode::Offline, "t1_hold_ms_p50"), "T1 needs a device clock");
        for m in ["second_load_ms", "vram_second_voice_mb", "second_load_warm_once_ms", "background_call_ms_p99", "engine_load_ms", "vram_engine_mb"] {
            assert!(known_metric(RunMode::LoadTwice, m), "{m}");
        }
        assert!(!known_metric(RunMode::LoadTwice, "latency_ms_p50"));
        assert!(higher_is_better("join_ncc_p95") && !higher_is_better("join_offset_ms_p95") && higher_is_better("throughput_x"));
        // Thresholds on them validate (components are reported only when measurable).
        let lt = run(serde_json::json!({"mode": "load_twice", "name": "lt", "thresholds": {"second_load_ms": {"max": 3000}}}));
        assert!(run_produces(&lt, "second_load_ms"));
        let off = run(serde_json::json!({"mode": "offline", "name": "o"}));
        assert!(run_produces(&off, "gpu_flowdec_ms_p95") && run_produces(&off, "vram_load_peak_mb"));
    }

    /// The gate's analyzer rule: F0 metrics compare only between reports scored by the same
    /// analyzer (a report from before the stamp counts as its own analyzer).
    #[test]
    fn f0_comparisons_across_analyzers_are_refused() {
        for m in ["f0_rmse_cents", "gross_pitch_error", "vuv_error", "fine_rmse_cents", "fed_gpe", "sigil_gross_pitch_error"] {
            assert!(is_f0_metric(m), "{m}");
        }
        for m in ["f0_ms_p95", "call_ms_p95", "speaker_cosine", "mcd_db", "gpu_pitch_ms_p50"] {
            assert!(!is_f0_metric(m), "{m}");
        }
        let v1 = serde_json::json!({"runs": [{"details": {"scoring": {}}}]});
        let v2a = serde_json::json!({"runs": [{"details": {"scoring": {"analyzer_version": "rmvpe-7f3a"}}}]});
        let v2b = serde_json::json!({"engine": {"analyzer_version": "rmvpe-91c0"}});
        assert_eq!(analyzer_versions(&v2a).into_iter().collect::<Vec<_>>(), vec!["rmvpe-7f3a".to_string()]);
        // P3B-B's stamp (`details.scorers.f0_analyzer_version`) is the same stamp.
        let v2c = serde_json::json!({"runs": [{"details": {"scorers": {"f0_analyzer_version": "rmvpe-7f3a"}}}]});
        assert!(is_analyzer_stamp("f0_analyzer_version") && !is_analyzer_stamp("analyzer"));
        assert!(f0_comparison_refusal(&[&v2a], &[&v2c]).is_none(), "one analyzer under both keys");
        assert!(f0_comparison_refusal(&[&v1], &[&v2c]).is_some());
        assert!(f0_comparison_refusal(&[&v1, &v1], &[&v1]).is_none(), "both unstamped: one analyzer");
        assert!(f0_comparison_refusal(&[&v2a], &[&v2a, &v2a]).is_none());
        let why = f0_comparison_refusal(&[&v1], &[&v2a]).expect("refused");
        assert!(why.contains("<none>") && why.contains("rmvpe-7f3a"), "{why}");
        assert!(f0_comparison_refusal(&[&v2a], &[&v2b]).is_some());
        assert!(f0_comparison_refusal(&[&v2a, &v2b], &[&v2a, &v2b]).is_some(), "mixed stamps within a side");
    }

    /// P3B-I0: thresholds can name scorer v2's and compare v2's metrics; a corpus compare produces the
    /// v2 names and not the single-clip ones; v2 metrics where more is better are inverted by the gate.
    #[test]
    fn v2_metric_names_are_known() {
        for m in ["env_mcd_over_floor_db", "fed_input_fpe_cents", "clicks_binomial", "speaker_set_sim_delta", "env_mcd_db_spread"] {
            assert!(known_metric(RunMode::Quality, m) && known_metric(RunMode::Rescore, m), "{m}");
        }
        assert!(known_metric(RunMode::Rescore, "clips") && known_metric(RunMode::Rescore, "pairs"));
        assert!(!known_metric(RunMode::Quality, "clips"));
        for m in ["t6_pass", "corpus_clips", "points_refused", "dnsmos_paired_sd", "ablation_anchor_dnsmos_diff_mean", "ablation_anchor_gpe_clips"] {
            assert!(known_metric(RunMode::Compare, m), "{m}");
        }
        assert!(!known_metric(RunMode::Compare, "ablation_dnsmos") && !known_metric(RunMode::Compare, "ablation__clips"));
        for m in ["f0_calls", "f0_long_calls"] {
            assert!(known_metric(RunMode::Offline, m) && known_metric(RunMode::Paced, m), "{m}");
        }
        assert!(higher_is_better("t6_pass") && higher_is_better("speaker_set_sim_delta") && !higher_is_better("env_mcd_over_floor_db"));
        let corpus = run(serde_json::json!({"mode": "compare", "name": "c", "corpus": "corpus.json", "sigil": {"include": false}}));
        assert!(run_produces(&corpus, "t6_pass") && !run_produces(&corpus, "dnsmos_ovrl_margin") && !run_produces(&corpus, "sigil_dnsmos_ovrl"));
        let single = run(serde_json::json!({"mode": "compare", "name": "c", "sigil": {"include": true}}));
        assert!(!run_produces(&single, "t6_pass") && run_produces(&single, "sigil_dnsmos_ovrl"));
    }

    fn metrics(pairs: &[(&str, f64)]) -> Metrics {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn run(v: serde_json::Value) -> RunSpec {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn runs_produce_metrics_by_mode_and_configuration() {
        let plain = run(serde_json::json!({"mode": "paced", "name": "p", "clock": "simulated"}));
        assert!(run_produces(&plain, "latency_ms_p99") && run_produces(&plain, "underruns"));
        assert!(!run_produces(&plain, "stall_recovery_blocks") && !run_produces(&plain, "stall_latency_recovery_ms"));
        assert!(!run_produces(&plain, "gpu_load_duty") && !run_produces(&plain, "cpu_load_duty"));
        assert!(!run_produces(&plain, "cpu_cores") && !run_produces(&plain, "worker_cpu_cores"));
        assert!(!run_produces(&plain, "throughput_x"), "offline only");
        let full = run(serde_json::json!({"mode": "paced", "name": "p", "clock": "wall",
            "stalls": [{"at_s": 2, "ms": 100}],
            "load": {"cpu": {"threads": 1}, "gpu": {"duty": 0.5, "execution_provider": "cpu"}}}));
        for m in ["stall_recovery_blocks", "stall_latency_recovery_ms", "gpu_load_duty", "cpu_load_duty", "cpu_cores"] {
            assert!(run_produces(&full, m), "{m}");
        }
        let offline = run(serde_json::json!({"mode": "offline", "name": "o"}));
        assert!(run_produces(&offline, "cpu_cores") && run_produces(&offline, "throughput_x"));
        let e2e = run(serde_json::json!({"mode": "e2e_loopback", "name": "l"}));
        assert!(run_produces(&e2e, "e2e_ms_p95") && !run_produces(&e2e, "block_ms_p95"));
    }

    #[test]
    fn thresholds_pass_fail_and_scope() {
        // A paced run that measured no latency at all (no audio ever played).
        let m = metrics(&[("block_ms_p95", 4.0), ("underruns", 0.0), ("throughput_x", 0.5)]);
        let lim = |min: Option<f64>, max: Option<f64>| Limit { min, max };
        let spec_wide: Thresholds = [
            ("block_ms_p95".to_string(), lim(None, Some(5.0))),
            // The mode produces it but the run did not report it: fails as not measured.
            ("latency_ms_p99".to_string(), lim(None, Some(80.0))),
            ("underruns".to_string(), lim(None, Some(0.0))),
            // Not produced by this run's mode or configuration: skipped.
            ("stall_recovery_blocks".to_string(), lim(None, Some(2.0))),
            ("throughput_x".to_string(), lim(Some(1.0), None)),
            ("cpu_cores".to_string(), lim(None, Some(1.0))),
        ]
        .into();
        let paced = run(serde_json::json!({"mode": "paced", "name": "p", "clock": "simulated", "thresholds": {
            "block_ms_p95": {"max": 3.0},    // overrides the spec-wide 5.0
            "backlog_max_ms": {"max": 100.0} // absent at run scope: fails
        }}));
        let r = evaluate(&m, &spec_wide, &paced);
        assert_eq!(r.len(), 4, "{r:?}");
        let get = |k: &str| r.iter().find(|t| t.metric == k).unwrap();
        assert!(!get("block_ms_p95").passed && get("block_ms_p95").scope == "run");
        assert!(!get("backlog_max_ms").passed && get("backlog_max_ms").value.is_none());
        let lat = get("latency_ms_p99");
        assert!(!lat.passed && lat.value.is_none() && lat.scope == "spec", "{lat:?}");
        assert!(get("underruns").passed && get("underruns").scope == "spec");

        // The same spec-wide set on an offline run: only the offline metrics apply.
        let offline = run(serde_json::json!({"mode": "offline", "name": "o"}));
        let r = evaluate(&m, &spec_wide, &offline);
        let names: Vec<&str> = r.iter().map(|t| t.metric.as_str()).collect();
        assert_eq!(names, vec!["block_ms_p95", "cpu_cores", "throughput_x"], "{r:?}");
        assert!(r.iter().all(|t| t.scope == "spec"));
        assert!(!r[1].passed && r[1].value.is_none(), "cpu_cores unmeasured on this PC fails");
        assert!(!r[2].passed, "throughput 0.5 < 1");

        let exact = run(serde_json::json!({"mode": "offline", "name": "o", "thresholds": {"block_ms_p95": {"min": 4.0, "max": 4.0}}}));
        let both = evaluate(&m, &Thresholds::new(), &exact);
        assert!(both[0].passed, "limits are inclusive");
    }

    fn sample_report() -> BenchReport {
        let run = RunReport {
            name: "paced".into(),
            mode: RunMode::Paced,
            status: RunStatus::Failed,
            error: None,
            duration_ms: 1234.0,
            metrics: metrics(&[("latency_ms_p50", 61.25), ("latency_ms_p99", 70.5), ("underruns", 2.0)]),
            thresholds: vec![ThresholdResult {
                metric: "underruns".into(),
                min: None,
                max: Some(0.0),
                value: Some(2.0),
                passed: false,
                scope: "run".into(),
            }],
            details: serde_json::json!({}),
            files: vec!["paced.timeline.csv".into()],
            higher_is_better: Vec::new(),
        };
        let mut r = BenchReport {
            schema: REPORT_SCHEMA.into(),
            name: "t".into(),
            description: "desc".into(),
            passed: false,
            failures: Vec::new(),
            started_unix_ms: 1,
            duration_ms: 2_000.0,
            spec_sha256: "ab".repeat(32),
            spec: serde_json::json!({"name": "t"}),
            environment: Environment {
                sigil_version: "1.0.0".into(),
                build: "debug".into(),
                os: "windows".into(),
                arch: "x86_64".into(),
                cpu: Some("Test CPU".into()),
                logical_cores: 8,
                onnx_runtime: "1.28.0".into(),
                gpus: vec![serde_json::json!({"name": "GPU0", "driver_version": "1.2", "dedicated_vram_mb": 8192})],
                voice_changer_running: false,
            },
            engine: Some(serde_json::json!({"description": "stub", "output_rate": 40000,
                                            "warnings": ["stream.block_frames = 1920 | not a device block"]})),
            runs: vec![run],
        };
        r.failures = r.runs.iter().flat_map(run_failures).collect();
        r
    }

    #[test]
    fn markdown_and_summary_lines_show_results() {
        let r = sample_report();
        assert_eq!(r.failures, vec!["paced: underruns = 2 (limit <= 0)".to_string()]);
        let md = to_markdown(&r);
        for needle in [
            "# Benchmark: t",
            "**Result: FAILED**",
            "| paced | paced | FAIL |",
            "latency p50 ms 61.25",
            "| underruns | <= 0 | 2 | FAIL |",
            "| GPU | GPU0 (8192 MB, driver 1.2) |",
            "Files: paced.timeline.csv",
            "| warning | stream.block_frames = 1920 / not a device block |",
        ] {
            assert!(md.contains(needle), "missing {needle:?} in:\n{md}");
        }
        let line = run_line(&r.runs[0]);
        assert!(line.contains("FAIL") && line.contains("latency p99 ms 70.50"), "{line}");
    }

    #[test]
    fn reports_round_trip_and_write_atomically() {
        let r = sample_report();
        let dir = std::env::temp_dir().join(format!("sigil-bench-report-{}", std::process::id()));
        let (json, md) = write_report(&dir, &r).unwrap();
        let back: BenchReport = serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
        assert_eq!(back.runs[0].metrics, r.runs[0].metrics);
        assert_eq!(back.runs[0].status, RunStatus::Failed);
        assert!(md.is_file());
        assert!(!dir.join("report.tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn timeline_csv_has_one_row_per_tick() {
        let rows = vec![
            TickRecord {
                tick: 0,
                compute_ms: 1.5,
                ..TickRecord::default()
            },
            TickRecord {
                tick: 1,
                error: true,
                ..TickRecord::default()
            },
        ];
        let csv = timeline_csv(&rows);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], TIMELINE_HEADER);
        assert!(lines[1].starts_with("0,0.000,0.000,1.5000,0.0000,0,"));
        assert!(lines[2].ends_with(",1"));
    }
}
