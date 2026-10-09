//! The out-of-process GPU load (P3B-A): `Sigil.exe --headless gpu-load <spec.json>` runs the
//! benchmark's conv load on DirectML / D3D12 at a set duty in its own process (a game's process and
//! queue, which in-process CUDA stream priority cannot reach), inside a KILL_ON_JOB_CLOSE Job Object
//! started by the benchmark (`load.gpu.out_of_process`), and reports its achieved duty as a
//! frame-time proxy.
//!
//! # Frozen by P3B-S0 (signature; P3B-A implements)
//! - [`run`]`(spec: &Path) -> Result<String, String>`: the body of the headless arm (`Ok` = the
//!   summary printed to stdout, exit 0; `Err` = exit 1).
//!
//! # Protocol (P3B-A)
//! The spec is a [`GpuLoadChildSpec`] written by the parent (`load_sim::OopLoad`): the load, a
//! stop file, a ready file and a report file. The child opens the load's session, runs it once,
//! writes the ready file (the EP it runs on), then holds its duty until the stop file appears, its
//! `max_seconds` pass or its parent process is gone, and writes its [`GpuLoadReport`] (achieved
//! duty, runs per second, run and frame times) to the report file. The parent starts it suspended
//! in its own Job Object with `KILL_ON_JOB_CLOSE` (`process_host::job`, then resumed), so it never
//! outlives the benchmark; a child that does not stop within a grace period is terminated through
//! its job.

use std::path::Path;
use std::time::{Duration, Instant};

use super::load_sim::{run_gpu_load, GpuLoadChildSpec, GpuLoadReport, LoadCpu};

/// How often the stop file and the parent are checked (the duty loop asks after every run and at
/// least once per period; checks are spaced at least this far apart).
const CHECK_EVERY: Duration = Duration::from_millis(50);

/// Run the GPU load described by the spec file at `spec` until its duration ends or the parent
/// process goes (see the module docs).
pub fn run(spec: &Path) -> Result<String, String> {
    let text = std::fs::read_to_string(spec).map_err(|e| format!("cannot read {}: {e}", spec.display()))?;
    let child: GpuLoadChildSpec = serde_json::from_str(&text).map_err(|e| format!("{}: invalid gpu-load spec: {e}", spec.display()))?;
    let report = run_child(&child)?;
    let json = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    super::report::write_atomic(&child.report_file, json.as_bytes())?;
    let summary = format!(
        "gpu-load: {} on {} ({}), duty {:.2} of {:.2}, {} runs ({:.1}/s), run p50 {:.2} ms, frame p99 {:.2} ms, stopped: {}",
        child.kind_name(),
        report.execution_provider_active,
        report.execution_provider,
        report.duty,
        report.duty_target,
        report.runs,
        report.runs_per_second,
        report.run_ms.map_or(0.0, |s| s.p50),
        report.frame_ms.map_or(0.0, |s| s.p99),
        report.stop_reason.as_deref().unwrap_or("?"),
    );
    match &report.error {
        Some(e) => Err(format!("{summary}; error: {e}")),
        None => Ok(summary),
    }
}

impl GpuLoadChildSpec {
    fn kind_name(&self) -> &'static str {
        match self.kind {
            super::spec::GpuLoadKind::Conv => "conv",
            super::spec::GpuLoadKind::Matmul => "matmul",
        }
    }

    fn validate(&self) -> Result<(), String> {
        let g = &self.load;
        let range = |what: &str, v: f64, lo: f64, hi: f64| {
            if v.is_finite() && (lo..=hi).contains(&v) {
                Ok(())
            } else {
                Err(format!("gpu-load: {what} = {v} is outside {lo}..={hi}"))
            }
        };
        range("load.duty", g.duty, 0.01, 1.0)?;
        range("load.period_ms", g.period_ms, 1.0, 1_000.0)?;
        range("load.size", g.size as f64, 16.0, 8_192.0)?;
        range("load.depth", g.depth as f64, 1.0, 64.0)?;
        range("max_seconds", self.max_seconds, 0.1, 6.0 * 3600.0)?;
        Ok(())
    }
}

/// The child's body (separate from the file I/O of [`run`] for tests).
pub(super) fn run_child(child: &GpuLoadChildSpec) -> Result<GpuLoadReport, String> {
    child.validate()?;
    let started = Instant::now();
    let parent = child.parent_pid.and_then(ParentWatch::open);
    let last_check = std::cell::Cell::new(Instant::now() - CHECK_EVERY);
    let reason = std::cell::Cell::new(None::<&'static str>);
    let should_stop = || {
        if reason.get().is_some() {
            return true;
        }
        if started.elapsed().as_secs_f64() >= child.max_seconds {
            reason.set(Some("duration"));
            return true;
        }
        if last_check.get().elapsed() < CHECK_EVERY {
            return false;
        }
        last_check.set(Instant::now());
        if child.stop_file.exists() {
            reason.set(Some("stop"));
        } else if parent.as_ref().is_some_and(|p| !p.alive()) {
            reason.set(Some("parent_gone"));
        }
        reason.get().is_some()
    };
    let mut ready_error = None;
    let mut on_ready = |r: Result<&str, &str>| match r {
        Ok(active) => {
            if let Err(e) = super::report::write_atomic(&child.ready_file, active.as_bytes()) {
                ready_error = Some(e);
            }
        }
        Err(e) => ready_error = Some(e.to_string()),
    };
    let mut report = run_gpu_load(&child.load, child.kind, &should_stop, &mut on_ready, LoadCpu::default());
    report.out_of_process = true;
    report.process_id = Some(std::process::id());
    if report.stop_reason.is_none() {
        report.stop_reason = reason.get().map(str::to_string);
    }
    if report.error.is_none() {
        report.error = ready_error;
    }
    Ok(report)
}

/// The parent process, watched for exit.
struct ParentWatch(imp::Handle);

impl ParentWatch {
    fn open(pid: u32) -> Option<ParentWatch> {
        imp::open(pid).map(ParentWatch)
    }

    fn alive(&self) -> bool {
        imp::alive(&self.0)
    }
}

#[cfg(windows)]
mod imp {
    use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_TIMEOUT};
    use windows::Win32::System::Threading::{OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE};

    pub struct Handle(HANDLE);

    // SAFETY: a process handle is a kernel object reference usable from any thread.
    unsafe impl Send for Handle {}

    impl Drop for Handle {
        fn drop(&mut self) {
            // SAFETY: opened by `open`, closed once.
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }

    pub fn open(pid: u32) -> Option<Handle> {
        // SAFETY: plain OpenProcess for SYNCHRONIZE only.
        unsafe { OpenProcess(PROCESS_SYNCHRONIZE, false, pid) }.ok().map(Handle)
    }

    pub fn alive(h: &Handle) -> bool {
        // SAFETY: a valid process handle with SYNCHRONIZE access.
        unsafe { WaitForSingleObject(h.0, 0) == WAIT_TIMEOUT }
    }
}

#[cfg(not(windows))]
mod imp {
    pub struct Handle;

    pub fn open(_pid: u32) -> Option<Handle> {
        None
    }

    pub fn alive(_h: &Handle) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::bench::load_sim::OopLoad;
    use crate::services::bench::spec::{GpuLoadKind, GpuLoadSpec};

    fn load(ep: &str) -> GpuLoadSpec {
        GpuLoadSpec {
            duty: 0.5,
            period_ms: 10.0,
            execution_provider: ep.into(),
            device_index: 0,
            size: 16,
            depth: 2,
            out_of_process: true,
        }
    }

    fn temp(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("sigil-gpu-load-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn child_spec(dir: &Path, max_seconds: f64) -> GpuLoadChildSpec {
        GpuLoadChildSpec {
            load: load("cpu"),
            kind: GpuLoadKind::Conv,
            max_seconds,
            stop_file: dir.join("stop"),
            ready_file: dir.join("ready"),
            report_file: dir.join("report.json"),
            parent_pid: Some(std::process::id()),
        }
    }

    /// The child body on the CPU EP: ready file, duty, then a stop file ends it.
    #[test]
    fn the_child_runs_until_its_stop_file() {
        let dir = temp("stop");
        let spec = child_spec(&dir, 30.0);
        let stop = spec.stop_file.clone();
        let ready = spec.ready_file.clone();
        let t0 = Instant::now();
        let report = std::thread::scope(|s| {
            let h = s.spawn(|| run_child(&spec).unwrap());
            while !ready.is_file() && t0.elapsed() < Duration::from_secs(30) {
                std::thread::sleep(Duration::from_millis(10));
            }
            std::thread::sleep(Duration::from_millis(300));
            std::fs::write(&stop, b"stop").unwrap();
            h.join().unwrap()
        });
        assert_eq!(std::fs::read_to_string(&ready).unwrap(), "cpu");
        assert!(report.error.is_none(), "{:?}", report.error);
        assert_eq!(report.stop_reason.as_deref(), Some("stop"));
        assert!(report.out_of_process && report.process_id == Some(std::process::id()));
        assert!(report.runs > 3, "{}", report.runs);
        assert!(report.duty > 0.2 && report.duty < 0.8, "{}", report.duty);
        assert!(report.runs_per_second > 0.0 && report.frame_ms.is_some());
        assert!(t0.elapsed() < Duration::from_secs(20));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The headless body: a spec file in, the report file and a summary out; its duration ends it.
    #[test]
    fn the_headless_body_reports_to_its_file() {
        let dir = temp("file");
        let spec = child_spec(&dir, 0.4);
        let path = dir.join("spec.json");
        std::fs::write(&path, serde_json::to_string(&spec).unwrap()).unwrap();
        let summary = run(&path).unwrap();
        assert!(summary.contains("gpu-load: conv on cpu") && summary.contains("stopped: duration"), "{summary}");
        let back: GpuLoadReport = serde_json::from_str(&std::fs::read_to_string(&spec.report_file).unwrap()).unwrap();
        assert_eq!(back.stop_reason.as_deref(), Some("duration"));
        assert!(run(&dir.join("missing.json")).unwrap_err().contains("cannot read"));
        std::fs::write(&path, r#"{"load": {"duty": 0.5}, "bogus": 1}"#).unwrap();
        assert!(run(&path).unwrap_err().contains("invalid gpu-load spec"));
        let mut bad = child_spec(&dir, 0.4);
        bad.load.duty = 2.0;
        std::fs::write(&path, serde_json::to_string(&bad).unwrap()).unwrap();
        assert!(run(&path).unwrap_err().contains("load.duty"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The parent's side of the protocol against a stand-in child (PowerShell plays the child:
    /// ready file, then wait for the stop file, then the report): start, stop, report, cleanup; a
    /// child that never becomes ready is an error and does not linger.
    #[cfg(windows)]
    #[test]
    fn the_parent_starts_stops_and_reads_the_child() {
        // `-Command` joins the remaining arguments into the command: the spec path the parent
        // appends becomes the script block's argument.
        let script = concat!(
            "& { param($p) $s = Get-Content -Raw $p | ConvertFrom-Json; ",
            "Set-Content -NoNewline -Path $s.ready_file -Value 'dml'; ",
            "while (-not (Test-Path $s.stop_file)) { Start-Sleep -Milliseconds 20 }; ",
            "$r = @{kind='conv'; execution_provider='dml'; execution_provider_active='dml'; reason=$null; size=16; depth=2; duty_target=0.5; duty=0.49; runs=12; run_ms=$null; gflops=0.0; cpu_ms=0.0; error=$null; stop_reason='stop'; runs_per_second=60.0; process_id=$PID}; ",
            "Set-Content -Path $s.report_file -Value ($r | ConvertTo-Json) }"
        );
        let ps = std::path::PathBuf::from(std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into()))
            .join(r"System32\WindowsPowerShell\v1.0\powershell.exe");
        if !ps.is_file() {
            eprintln!("no Windows PowerShell; skipped");
            return;
        }
        let args = ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script];
        let l = OopLoad::start_with(&ps, &args, &load("dml"), GpuLoadKind::Conv, Duration::from_secs(60)).expect("ready");
        std::thread::sleep(Duration::from_millis(100));
        let r = l.finish();
        assert!(r.error.is_none(), "{:?}", r.error);
        assert!(r.out_of_process && r.runs == 12 && (r.duty - 0.49).abs() < 1e-9, "{r:?}");
        assert_eq!(r.stop_reason.as_deref(), Some("stop"));
        // Never ready: refused within the timeout, the child is stopped by its job.
        let never = ["-NoProfile", "-NonInteractive", "-Command", "& { Start-Sleep -Seconds 30 }"];
        let t0 = Instant::now();
        let e = OopLoad::start_with(&ps, &never, &load("dml"), GpuLoadKind::Conv, Duration::from_secs(2)).err().expect("not ready");
        assert!(e.contains("not ready"), "{e}");
        assert!(t0.elapsed() < Duration::from_secs(45), "{:?}", t0.elapsed());
        // A child that dies at once: an error with its exit, not a hang.
        let dies = ["-NoProfile", "-NonInteractive", "-Command", "& { [Environment]::Exit(3) }"];
        let e = OopLoad::start_with(&ps, &dies, &load("dml"), GpuLoadKind::Conv, Duration::from_secs(30)).err().expect("exited");
        assert!(e.contains("exit"), "{e}");
    }
}
