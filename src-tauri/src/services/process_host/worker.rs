//! One-shot isolated workers: run a program to completion inside its own Job Object with a memory
//! cap and a wall-clock limit, capture bounded stdout/stderr, and report why it ended. Built for the
//! untrusted-file parsers (`Sigil.exe --headless vc-import-worker <job.json>`, plan decision
//! "Parsing untrusted .pth/.zip": 2–4 GB, 60–120 s), usable for any batch child.
//!
//! Blocking; call it from `tokio::task::spawn_blocking` in async code.

use super::job::{self, JobLimits};
use super::spawn::{self, Launch};
use super::HelperPriority;
use std::io::{Read, Write};
use std::os::windows::io::AsRawHandle;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// How long to wait for the output readers after the process tree is gone.
const READER_WAIT: Duration = Duration::from_secs(2);

/// What to run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkerSpec {
    /// Short id for errors and thread names (same rules as helper names).
    pub name: String,
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    /// Inherited variables removed first (`NAME` or `PREFIX*`).
    pub env_remove: Vec<String>,
    pub cwd: Option<PathBuf>,
    /// Job-wide committed-memory cap.
    pub memory_limit_mb: Option<u64>,
    /// Wall-clock limit; the whole process tree is terminated when it passes.
    pub timeout_ms: u64,
    /// Bytes kept per stream; the rest is read and discarded (the child never blocks on a full pipe).
    pub max_output_bytes: usize,
    /// Written to stdin, then stdin is closed.
    pub stdin: Option<Vec<u8>>,
    pub priority: HelperPriority,
}

impl Default for WorkerSpec {
    fn default() -> Self {
        Self {
            name: "worker".into(),
            program: PathBuf::new(),
            args: Vec::new(),
            env: Vec::new(),
            env_remove: Vec::new(),
            cwd: None,
            memory_limit_mb: None,
            timeout_ms: 120_000,
            max_output_bytes: 1 << 20,
            stdin: None,
            priority: HelperPriority::Normal,
        }
    }
}

/// How the worker ended.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorkerOutput {
    /// `None` when it was terminated for the time limit or the code could not be read.
    pub exit_code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub timed_out: bool,
    /// An allocation failed because of `memory_limit_mb`.
    pub memory_limit_hit: bool,
    pub peak_memory_mb: u64,
    pub elapsed_ms: u64,
}

#[allow(dead_code)] // API for P3-B; tested here.
impl WorkerOutput {
    /// Exit code 0 within the time limit.
    pub fn success(&self) -> bool {
        self.exit_code == Some(0) && !self.timed_out
    }

    /// One-line reason for logs and errors, e.g. `timed out after 60.0 s`.
    pub fn describe(&self) -> String {
        let mut s = if self.timed_out {
            format!("timed out after {:.1} s", self.elapsed_ms as f64 / 1000.0)
        } else {
            match self.exit_code {
                Some(c) => job::describe_exit_code(c),
                None => "exit code unknown".into(),
            }
        };
        if self.memory_limit_hit {
            s.push_str("; it ran into its memory limit");
        }
        s
    }
}

fn read_bounded<R: Read + Send + 'static>(mut reader: R, max: usize, tx: mpsc::Sender<(Vec<u8>, bool)>) -> std::io::Result<()> {
    std::thread::Builder::new()
        .name("worker-output".into())
        .spawn(move || {
            let mut kept = Vec::new();
            let mut truncated = false;
            let mut buf = [0u8; 8192];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        let room = max.saturating_sub(kept.len());
                        if n > room {
                            truncated = true;
                        }
                        kept.extend_from_slice(&buf[..n.min(room)]);
                    }
                }
            }
            let _ = tx.send((kept, truncated));
        })?;
    Ok(())
}

/// Run `spec` to completion (or its time limit) in a fresh job. `Err` only when it could not be
/// started; a crash, a non-zero exit, a timeout or a memory-limit hit are reported in the output.
#[allow(dead_code)] // API for P3-B (`vc-import-worker` child); tested here.
pub fn run_worker(spec: &WorkerSpec) -> Result<WorkerOutput, String> {
    spawn::validate_name(&spec.name)?;
    if spec.timeout_ms == 0 {
        return Err(format!("{}: the time limit must be greater than 0", spec.name));
    }
    if spec.program.components().count() > 1 && !spec.program.is_file() {
        return Err(format!("{}: program not found: {}", spec.name, spec.program.display()));
    }
    if let Some(mb) = spec.memory_limit_mb {
        if mb < job::MIN_MEMORY_LIMIT_MB {
            return Err(format!(
                "{}: memory limit {mb} MB is below the minimum of {} MB",
                spec.name,
                job::MIN_MEMORY_LIMIT_MB
            ));
        }
    }
    let started = Instant::now();
    let launched = spawn::launch(&Launch {
        name: &spec.name,
        program: &spec.program,
        args: spec.args.clone(),
        env: spec.env.clone(),
        env_remove: &spec.env_remove,
        cwd: spec.cwd.clone(),
        limits: JobLimits {
            memory_limit_mb: spec.memory_limit_mb,
            priority: spec.priority,
        },
        high_qos: false,
    })?;
    let spawn::Launched { mut child, job, .. } = launched;
    let kill = |child: &mut std::process::Child, msg: String| -> String {
        job.terminate(1);
        let _ = child.wait();
        format!("{}: {msg}", spec.name)
    };

    let (out_tx, out_rx) = mpsc::channel();
    let (err_tx, err_rx) = mpsc::channel();
    if let Some(out) = child.stdout.take() {
        if let Err(e) = read_bounded(out, spec.max_output_bytes, out_tx) {
            return Err(kill(&mut child, format!("cannot start the stdout reader: {e}")));
        }
    }
    if let Some(err) = child.stderr.take() {
        if let Err(e) = read_bounded(err, spec.max_output_bytes, err_tx) {
            return Err(kill(&mut child, format!("cannot start the stderr reader: {e}")));
        }
    }
    if let Some(mut stdin) = child.stdin.take() {
        let data = spec.stdin.clone().unwrap_or_default();
        // A separate thread, so a child that does not read its input cannot deadlock the caller.
        let spawned = std::thread::Builder::new()
            .name("worker-input".into())
            .spawn(move || {
                let _ = stdin.write_all(&data);
                drop(stdin);
            });
        if let Err(e) = spawned {
            return Err(kill(&mut child, format!("cannot start the stdin writer: {e}")));
        }
    }

    let exited = job::wait_process(child.as_raw_handle(), Duration::from_millis(spec.timeout_ms));
    // Kill the rest of the tree either way (and the worker itself on timeout).
    job.terminate(1);
    let status = child.wait().ok();
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let (stdout, stdout_truncated) = out_rx.recv_timeout(READER_WAIT).unwrap_or_default();
    let (stderr, stderr_truncated) = err_rx.recv_timeout(READER_WAIT).unwrap_or_default();
    let events = job.take_events();
    let stats = job.stats();
    Ok(WorkerOutput {
        exit_code: if exited { status.and_then(|s| s.code()) } else { None },
        stdout,
        stderr,
        stdout_truncated,
        stderr_truncated,
        timed_out: !exited,
        memory_limit_hit: events.memory_limit_hit,
        peak_memory_mb: stats.peak_job_memory_bytes / (1024 * 1024),
        elapsed_ms,
    })
}
