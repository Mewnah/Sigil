//! `Sigil.exe --headless bench <spec.json> [--out <dir>]`.
//!
//! Loads the spec (relative paths resolve against its folder), runs it, writes `report.json`,
//! `report.md` and the per-run files into the output folder (default
//! `bench-results/<spec file stem>-<unix ms>/` under the working directory), prints progress to
//! stderr and the summary to stdout. Any failed run or threshold returns `Err`, so the process
//! exits with code 1 (the reports are written either way).

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;
use std::time::Instant;

use super::report::{run_line, write_report, BenchReport};
use super::spec::BenchSpec;
use super::{run_spec, unix_ms_now, BenchContext, Progress};

/// Default output folder for a spec file.
fn default_out_dir(spec_path: &Path) -> PathBuf {
    let stem = spec_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "bench".into());
    PathBuf::from("bench-results").join(format!("{stem}-{}", unix_ms_now()))
}

/// Summary text of a finished report.
pub(super) fn summary(report: &BenchReport, json: Option<&Path>) -> String {
    let mut s = format!(
        "Sigil bench '{}': {} ({} run(s), {:.1} s)\n",
        report.name,
        if report.passed { "PASSED" } else { "FAILED" },
        report.runs.len(),
        report.duration_ms / 1000.0
    );
    if let Some(engine) = &report.engine {
        if let Some(d) = engine.get("description").and_then(|v| v.as_str()) {
            s.push_str(&format!("  engine: {d}\n"));
        }
    }
    for run in &report.runs {
        s.push_str(&run_line(run));
        s.push('\n');
    }
    for f in &report.failures {
        s.push_str(&format!("  failure: {f}\n"));
    }
    if let Some(p) = json {
        s.push_str(&format!("  report: {} (+ report.md)", p.display()));
    }
    s.trim_end().to_string()
}

/// Body of `Sigil.exe --headless bench <spec.json> [--out <dir>]` (frozen P1-S0b contract).
pub fn headless_bench(spec: &Path, out: Option<&Path>) -> Result<String, String> {
    // On a thread with the benchmark stack, not Windows' 1 MB main thread.
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("sigil-bench-cli".into())
            .stack_size(super::BENCH_STACK_BYTES)
            .spawn_scoped(scope, || headless_bench_on_this_thread(spec, out))
            .map_err(|e| format!("cannot start the benchmark thread: {e}"))?
            .join()
            .map_err(|_| "the benchmark failed unexpectedly".to_string())?
    })
}

fn headless_bench_on_this_thread(spec: &Path, out: Option<&Path>) -> Result<String, String> {
    let parsed = BenchSpec::load(spec)?;
    let out_dir = out.map(Path::to_path_buf).unwrap_or_else(|| default_out_dir(spec));
    let cancel = AtomicBool::new(false);
    let last = Mutex::new((String::new(), Instant::now()));
    let progress = |p: &Progress| {
        // One line when a run starts, then at most one every 2 s.
        let mut l = last.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let new_run = l.0 != p.run;
        if new_run || l.1.elapsed().as_secs() >= 2 {
            eprintln!(
                "bench [{}/{}] {}: {:>3.0}% {}",
                p.run_index + 1,
                p.run_count,
                p.run,
                p.fraction * 100.0,
                p.message
            );
            *l = (p.run.clone(), Instant::now());
        }
    };
    let report = run_spec(&parsed, Some(&out_dir), &BenchContext { cancel: &cancel, progress: &progress });
    let (json, _md) = write_report(&out_dir, &report)?;
    let text = summary(&report, Some(&json));
    if report.passed {
        Ok(text)
    } else {
        Err(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sigil-bench-cli-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn headless_bench_writes_reports_and_maps_failures_to_err() {
        let dir = temp("ok");
        let spec = dir.join("mini.json");
        std::fs::write(
            &spec,
            serde_json::json!({
                "name": "mini",
                "engine": {"kind": "stub"},
                "input": {"kind": "synthetic", "seconds": 1.5},
                "runs": [{"mode": "paced", "name": "p", "seconds": 1.5, "clock": "simulated",
                          "thresholds": {"backlog_over_budget_ms": {"max": 0}, "model_errors": {"max": 0}}}]
            })
            .to_string(),
        )
        .unwrap();
        let out = dir.join("out");
        let text = headless_bench(&spec, Some(&out)).expect("passes");
        assert!(text.contains("PASSED") && text.contains("PASS "), "{text}");
        assert!(out.join("report.json").is_file() && out.join("report.md").is_file());
        assert!(out.join("p.timeline.csv").is_file());

        // The same spec with an impossible threshold: reports still written, Err returned.
        std::fs::write(
            &spec,
            serde_json::json!({
                "name": "mini",
                "engine": {"kind": "stub"},
                "input": {"kind": "synthetic", "seconds": 1.5},
                "runs": [{"mode": "paced", "name": "p", "seconds": 1.5, "clock": "simulated",
                          "thresholds": {"latency_ms_p50": {"max": 0.001}}}]
            })
            .to_string(),
        )
        .unwrap();
        let out2 = dir.join("out2");
        let err = headless_bench(&spec, Some(&out2)).unwrap_err();
        assert!(err.contains("FAILED") && err.contains("failure: p: latency_ms_p50"), "{err}");
        assert!(out2.join("report.json").is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bad_specs_are_errors_before_anything_runs() {
        let dir = temp("bad");
        let spec = dir.join("bad.json");
        std::fs::write(&spec, r#"{"name": "x", "engine": {"kind": "stub"}, "runs": [], "bogus": 1}"#).unwrap();
        let err = headless_bench(&spec, Some(&dir.join("out"))).unwrap_err();
        assert!(err.contains("bogus"), "{err}");
        assert!(!dir.join("out").exists());
        assert!(headless_bench(&dir.join("missing.json"), None).unwrap_err().contains("cannot read"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_out_dir_uses_the_spec_stem() {
        let d = default_out_dir(Path::new("scripts/bench/specs/stub-smoke.json"));
        assert!(d.starts_with("bench-results"));
        assert!(d.file_name().unwrap().to_string_lossy().starts_with("stub-smoke-"));
    }
}
