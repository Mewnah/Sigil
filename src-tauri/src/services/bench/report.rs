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
/// SVE per-component compute (`build: direct` template voices).
const COMPONENT_PREFIXES: &[&str] = &["content_ms", "f0_ms", "enc_ms", "flowdec_ms"];
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
];
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
    let block = prefixed(BLOCK_PREFIXES, name) || prefixed(COMPONENT_PREFIXES, name) || BLOCK_SCALARS.contains(&name);
    let paced = block || prefixed(PACED_PREFIXES, name) || PACED_SCALARS.contains(&name);
    match mode {
        RunMode::Offline => block || OFFLINE_SCALARS.contains(&name),
        RunMode::Paced | RunMode::Contention => paced,
        RunMode::E2eLoopback => prefixed(E2E_PREFIXES, name) || E2E_SCALARS.contains(&name),
        RunMode::Quality => block || QUALITY_SCALARS.contains(&name),
        RunMode::Compare => COMPARE_SCALARS.contains(&name),
        RunMode::Autotune => AUTOTUNE_SCALARS.contains(&name),
        RunMode::LiveRecord => prefixed(LIVE_PREFIXES, name) || LIVE_SCALARS.contains(&name),
    }
}

/// Why `run`'s configuration never produces metric `name` although its mode can: the stall
/// metrics need `stalls`, the load duties their load, the CPU figures real compute on the clock
/// (not `simulated`), CER an STT engine, the reference speaker cosine a reference clip, and the
/// compare margins / latencies their inputs. `None` when the run produces it.
pub fn unproduced_reason(run: &RunSpec, name: &str) -> Option<&'static str> {
    if name.starts_with("content_ms") || name.starts_with("f0_ms") || name.starts_with("enc_ms") || name.starts_with("flowdec_ms") {
        // Only a template voice built directly reports its components; the spec cannot tell.
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

/// The metrics of `metrics` where a larger value is better (from [`HIGHER_IS_BETTER`]).
pub fn higher_is_better_of(metrics: &Metrics) -> Vec<String> {
    metrics
        .keys()
        .filter(|k| HIGHER_IS_BETTER.contains(&k.as_str()))
        .cloned()
        .collect()
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
    /// The engine every offline / paced run used (`null` for loopback-only specs).
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
