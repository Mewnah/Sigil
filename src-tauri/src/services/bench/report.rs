//! Benchmark reports (`sigil-bench-report/1`): the metric catalogue, threshold evaluation, and the
//! JSON / Markdown / CSV writers.
//!
//! Every run reports a flat `metrics` map with stable names (the perf gate and thresholds read
//! them) plus mode-specific `details`. Distribution metrics are `<prefix>_<suffix>` with the
//! suffixes of [`SUMMARY_SUFFIXES`] (`block_ms_p95`, `latency_ms_p99`, ...). All times are in
//! milliseconds, ratios in `0..1`; lower is better for every metric except `throughput_x`,
//! `e2e_detected` and the load duties.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::metrics::{Metrics, SUMMARY_SUFFIXES};
use super::spec::{ClockKind, Limit, RunMode, RunSpec, Thresholds};

/// Schema id of [`BenchReport`].
pub const REPORT_SCHEMA: &str = "sigil-bench-report/1";

/// Distribution prefixes of offline and paced runs.
const BLOCK_PREFIXES: &[&str] = &["block_ms", "model_ms", "rtf"];
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

fn prefixed(prefixes: &[&str], name: &str) -> bool {
    prefixes.iter().any(|p| {
        name.strip_prefix(p)
            .and_then(|rest| rest.strip_prefix('_'))
            .is_some_and(|suffix| SUMMARY_SUFFIXES.contains(&suffix))
    })
}

/// Whether runs of `mode` produce metric `name` (when measurable on the PC).
pub fn known_metric(mode: RunMode, name: &str) -> bool {
    match mode {
        RunMode::Offline => prefixed(BLOCK_PREFIXES, name) || BLOCK_SCALARS.contains(&name) || OFFLINE_SCALARS.contains(&name),
        RunMode::Paced => {
            prefixed(BLOCK_PREFIXES, name)
                || prefixed(PACED_PREFIXES, name)
                || BLOCK_SCALARS.contains(&name)
                || PACED_SCALARS.contains(&name)
        }
        RunMode::E2eLoopback => prefixed(E2E_PREFIXES, name) || E2E_SCALARS.contains(&name),
    }
}

/// Why `run`'s configuration never produces metric `name` although its mode can: the stall
/// metrics need `stalls`, the load duties their load, and the CPU figures real compute on the
/// clock (not `simulated`). `None` when the run produces it.
pub fn unproduced_reason(run: &RunSpec, name: &str) -> Option<&'static str> {
    let RunSpec::Paced(p) = run else { return None };
    match name {
        "stall_recovery_blocks" | "stall_latency_recovery_ms" if p.stalls.is_empty() => Some("the run has no stalls"),
        "gpu_load_duty" if p.load.gpu.is_none() => Some("the run has no load.gpu"),
        "cpu_load_duty" if p.load.cpu.is_none() => Some("the run has no load.cpu"),
        "cpu_cores" | "worker_cpu_cores" if p.clock == ClockKind::Simulated => {
            Some("the simulated clock charges no real compute, so the run reports no CPU figures")
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
    pub stall_ms: f64,
    /// Capture waiting after this wake's budget enforcement (paced).
    pub waiting_ms: f64,
    /// Input dropped at this wake (budget + ring overflow; paced).
    pub dropped_ms: f64,
    /// Output queued for playback after the push (paced).
    pub queued_out_ms: f64,
    pub error: bool,
}

const TIMELINE_HEADER: &str = "tick,t_ms,age_ms,compute_ms,model_ms,stall_ms,waiting_ms,dropped_ms,queued_out_ms,error";

/// Timeline as CSV.
pub fn timeline_csv(ticks: &[TickRecord]) -> String {
    let mut s = String::with_capacity(ticks.len() * 64 + 128);
    s.push_str(TIMELINE_HEADER);
    s.push('\n');
    for t in ticks {
        let _ = writeln!(
            s,
            "{},{:.3},{:.3},{:.4},{:.4},{:.1},{:.3},{:.3},{:.3},{}",
            t.tick,
            t.t_ms,
            t.age_ms,
            t.compute_ms,
            t.model_ms,
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
        if !run.files.is_empty() {
            let _ = writeln!(s, "\nFiles: {}", run.files.join(", "));
        }
    }
    s
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
            engine: Some(serde_json::json!({"description": "stub", "output_rate": 40000})),
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
        assert!(lines[1].starts_with("0,0.000,0.000,1.5000"));
        assert!(lines[2].ends_with(",1"));
    }
}
