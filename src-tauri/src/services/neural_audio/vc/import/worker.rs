//! Parent side of the isolated importer: runs `Sigil.exe --headless vc-import-worker <job.json>`
//! in its own Job Object (`process_host`: `CREATE_NO_WINDOW`, kill-on-close, a committed-memory
//! cap, below-normal priority so a running game and voice changer keep the CPU), with a
//! wall-clock limit, bounded output and cancellation, and turns its exit into the import record or
//! an [`ImportError`]:
//! - exit 0: the record JSON the worker printed (`import::ImportRecord`);
//! - exit 1: the worker's `vc-import-worker: <code>: <message>` line, code preserved;
//! - the memory cap was hit: `archive_rejected` (a hostile or absurdly large file);
//! - the parent's hard limit (the worker's own limit plus [`PARENT_GRACE`]): `time_limit`;
//! - cancelled: `job`; a crash: `io`.
//!
//! After any failure the slot folder is recovered (`import::recover_slot_folder`), so a worker
//! killed in the middle of its slot swap never leaves the previous voice missing.
//!
//! # Frozen by P3-S0 (signatures)
//! [`available`], [`WorkerLimits`] (+ `Default`), [`run_isolated`]`(exe, job, limits) ->
//! Result<serde_json::Value, ImportError>` (the worker's stdout record, `import::ImportRecord`'s
//! JSON), [`write_job_file`].

use std::io::Read;
use std::os::windows::io::AsRawHandle;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::services::process_host::job::{self, JobLimits};
use crate::services::process_host::spawn::{self, Launch};
use crate::services::process_host::HelperPriority;

use super::{write_atomic, ErrorCode, ImportError, IMPORT_RECORD_FORMAT};

/// Whether imports run in the isolated worker in this build.
pub fn available() -> bool {
    cfg!(windows)
}

/// Limits of one worker run.
#[derive(Clone, Copy, Debug)]
pub struct WorkerLimits {
    pub memory_limit_mb: u64,
    pub time_limit_s: u64,
}

impl Default for WorkerLimits {
    fn default() -> Self {
        Self { memory_limit_mb: 3072, time_limit_s: 120 }
    }
}

/// Time the parent allows beyond the worker's own limit (its watchdog reports `time_limit`
/// cleanly and lets a slot swap finish) before it kills the process tree.
pub const PARENT_GRACE: Duration = Duration::from_secs(20);
/// Bytes of stdout / stderr kept (the record is a few KB).
const MAX_OUTPUT: usize = 1 << 20;
/// How long to wait for the output readers after the process tree is gone.
const READER_WAIT: Duration = Duration::from_secs(2);
/// Cancellation and time-limit polling interval.
const POLL: Duration = Duration::from_millis(50);
/// Prefix of the worker's error line (`cli::report`).
const ERROR_PREFIX: &str = "vc-import-worker: ";

/// The job file for the worker (`import::ImportJob` schema v1).
#[derive(Clone, Debug, Serialize)]
pub struct WorkerJob {
    pub version: u32,
    pub source: PathBuf,
    pub out_dir: PathBuf,
    pub templates_dir: Option<PathBuf>,
    pub time_limit_s: Option<u64>,
    pub smoke: bool,
    pub smoke_fp16: bool,
    /// Display name for the manifest (`None`: the worker's default).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// A retrieval index file to adopt (`None`: the voice archive's own).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<PathBuf>,
    /// P5-S0: a voice-family import (`import::ImportJob::family`; `vc::families::run_import_job`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    /// P5-S0: the family's options for that import.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family_options: Option<serde_json::Value>,
}

impl WorkerJob {
    /// A job with the defaults: smoke decode on, fp16 smoke off.
    pub fn new(source: PathBuf, out_dir: PathBuf, templates_dir: Option<PathBuf>) -> WorkerJob {
        WorkerJob {
            version: 1,
            source,
            out_dir,
            templates_dir,
            time_limit_s: None,
            smoke: true,
            smoke_fp16: false,
            name: None,
            index: None,
            family: None,
            family_options: None,
        }
    }
}

fn nonce() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!("{:x}-{:x}-{:x}", std::process::id(), super::unix_ms(), COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Write `job` into `dir` as a hidden file (never listed as a slot) and return its path.
pub fn write_job_file(dir: &Path, job: &WorkerJob) -> Result<PathBuf, ImportError> {
    std::fs::create_dir_all(dir).map_err(|e| ImportError::new(ErrorCode::Io, format!("{}: {e}", dir.display())))?;
    let path = dir.join(format!(".sigil-import-job-{}.json", nonce()));
    let bytes = serde_json::to_vec_pretty(job).map_err(|e| ImportError::new(ErrorCode::Job, e.to_string()))?;
    write_atomic(&path, &bytes)?;
    Ok(path)
}

/// How to start the worker: the program, its arguments and extra environment, where `{job}`
/// stands for the job file's path. The app runs itself headless ([`WorkerCommand::app`]); tests
/// re-run their own binary.
#[derive(Clone, Debug)]
pub struct WorkerCommand {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
}

impl WorkerCommand {
    /// `exe --headless vc-import-worker <job>`.
    pub fn app(exe: &Path) -> WorkerCommand {
        WorkerCommand {
            program: exe.to_path_buf(),
            args: vec!["--headless".into(), "vc-import-worker".into(), "{job}".into()],
            env: Vec::new(),
        }
    }
}

/// How the worker process ended.
#[derive(Debug, Default)]
struct ChildOutput {
    exit_code: Option<i32>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    timed_out: bool,
    cancelled: bool,
    memory_limit_hit: bool,
    /// Peak committed memory of the worker's Job Object (bytes; 0 when unknown).
    peak_memory_bytes: u64,
    elapsed: Duration,
}

fn read_bounded<R: Read + Send + 'static>(mut reader: R, tx: mpsc::Sender<Vec<u8>>) -> std::io::Result<()> {
    std::thread::Builder::new().name("vc-import-output".into()).spawn(move || {
        let mut kept = Vec::new();
        let mut buf = [0u8; 8192];
        loop {
            match reader.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let room = MAX_OUTPUT.saturating_sub(kept.len());
                    kept.extend_from_slice(&buf[..n.min(room)]);
                }
            }
        }
        let _ = tx.send(kept);
    })?;
    Ok(())
}

/// Run the worker to completion, its hard limit, or cancellation, in a fresh Job Object.
fn run_child(cmd: &WorkerCommand, job_path: &Path, memory_limit_mb: u64, hard_limit: Duration, cancel: &AtomicBool) -> Result<ChildOutput, ImportError> {
    let job_s = job_path.to_string_lossy();
    let subst = |s: &String| s.replace("{job}", &job_s);
    let started = Instant::now();
    let launched = spawn::launch(&Launch {
        name: "vc-import-worker",
        program: &cmd.program,
        args: cmd.args.iter().map(subst).collect(),
        env: cmd.env.iter().map(|(k, v)| (k.clone(), subst(v))).collect(),
        env_remove: &[],
        cwd: None,
        limits: JobLimits { memory_limit_mb: Some(memory_limit_mb.max(job::MIN_MEMORY_LIMIT_MB)), priority: HelperPriority::BelowNormal },
        high_qos: false,
    })
    .map_err(|e| ImportError::new(ErrorCode::Io, format!("cannot start the import worker: {e}")))?;
    let spawn::Launched { mut child, job, .. } = launched;
    drop(child.stdin.take());
    let (out_tx, out_rx) = mpsc::channel();
    let (err_tx, err_rx) = mpsc::channel();
    let readers = child
        .stdout
        .take()
        .map_or(Ok(()), |o| read_bounded(o, out_tx))
        .and_then(|_| child.stderr.take().map_or(Ok(()), |e| read_bounded(e, err_tx)));
    if let Err(e) = readers {
        job.terminate(1);
        let _ = child.wait();
        return Err(ImportError::new(ErrorCode::Io, format!("cannot read the import worker's output: {e}")));
    }
    let deadline = started + hard_limit;
    let mut out = ChildOutput::default();
    let exited = loop {
        if job::wait_process(child.as_raw_handle(), POLL) {
            break true;
        }
        if cancel.load(Ordering::Acquire) {
            out.cancelled = true;
            break false;
        }
        if Instant::now() >= deadline {
            out.timed_out = true;
            break false;
        }
    };
    out.peak_memory_bytes = job.stats().peak_job_memory_bytes;
    // Kill the rest of the tree either way (and the worker itself when it did not exit).
    job.terminate(1);
    let status = child.wait().ok();
    out.elapsed = started.elapsed();
    out.exit_code = if exited { status.and_then(|s| s.code()) } else { None };
    out.stdout = out_rx.recv_timeout(READER_WAIT).unwrap_or_default();
    out.stderr = err_rx.recv_timeout(READER_WAIT).unwrap_or_default();
    out.memory_limit_hit = job.take_events().memory_limit_hit;
    Ok(out)
}

/// The import record in the worker's stdout: the first JSON object that starts a line.
fn parse_record(stdout: &[u8]) -> Option<serde_json::Value> {
    let text = String::from_utf8_lossy(stdout);
    let start = if text.starts_with('{') { 0 } else { text.find("\n{")? + 1 };
    let value = serde_json::Deserializer::from_str(&text[start..]).into_iter::<serde_json::Value>().next()?.ok()?;
    (value.get("format").and_then(|f| f.as_str()) == Some(IMPORT_RECORD_FORMAT)).then_some(value)
}

/// The last lines of `stderr`, for an error message.
fn tail(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let shown = lines[lines.len().saturating_sub(3)..].join(" | ");
    shown.chars().take(600).collect()
}

/// Turn the worker's exit into the record or an error (see the module docs).
fn interpret(out: &ChildOutput, limits: WorkerLimits) -> Result<serde_json::Value, ImportError> {
    if out.cancelled {
        return Err(ImportError::new(ErrorCode::Job, "the import was cancelled"));
    }
    if out.memory_limit_hit {
        return Err(ImportError::new(
            ErrorCode::ArchiveRejected,
            format!("the import needed more than {} MB of memory (a malformed or unusually large file)", limits.memory_limit_mb),
        ));
    }
    if out.timed_out {
        return Err(ImportError::new(
            ErrorCode::TimeLimit,
            format!("the import took longer than {} s and was stopped", out.elapsed.as_secs()),
        ));
    }
    match out.exit_code {
        Some(0) => parse_record(&out.stdout)
            .ok_or_else(|| ImportError::new(ErrorCode::Io, "the import worker finished without an import record")),
        Some(1) => {
            let text = String::from_utf8_lossy(&out.stderr);
            let reported = text.lines().rev().find_map(|l| l.trim().strip_prefix(ERROR_PREFIX).and_then(ImportError::parse));
            Err(reported.unwrap_or_else(|| ImportError::new(ErrorCode::Io, format!("the import worker failed: {}", tail(&out.stderr)))))
        }
        Some(code) => Err(ImportError::new(
            ErrorCode::Io,
            format!("the import worker crashed ({}){}", job::describe_exit_code(code), {
                let t = tail(&out.stderr);
                if t.is_empty() {
                    String::new()
                } else {
                    format!(": {t}")
                }
            }),
        )),
        None => Err(ImportError::new(ErrorCode::Io, "the import worker ended without an exit code")),
    }
}

/// Run the import in `exe --headless vc-import-worker <job>` under `limits`.
#[allow(dead_code)] // P3-S0 frozen API; the IPC layer uses the cancellable `run_isolated_with`.
pub fn run_isolated(exe: &Path, job: &WorkerJob, limits: WorkerLimits) -> Result<serde_json::Value, ImportError> {
    run_isolated_with(&WorkerCommand::app(exe), job, limits, &AtomicBool::new(false), None)
}

/// [`run_isolated`] with a cancellation flag (set it to stop the worker) and an explicit worker
/// command. `hard_limit` overrides the parent's kill deadline (default: the worker's limit plus
/// [`PARENT_GRACE`]). The job file is written next to the slot folder and removed afterwards. The
/// record returned gains `worker: {peak_memory_mb, wall_ms}` (the Job Object's peak committed
/// memory and the process's wall time; additive, for the T8 corpus report and the log).
pub fn run_isolated_with(
    cmd: &WorkerCommand,
    job: &WorkerJob,
    limits: WorkerLimits,
    cancel: &AtomicBool,
    hard_limit: Option<Duration>,
) -> Result<serde_json::Value, ImportError> {
    if !available() {
        return Err(ImportError::new(ErrorCode::Job, "the isolated import worker is not available on this platform"));
    }
    let time_limit_s = limits.time_limit_s.clamp(5, 600);
    let job = WorkerJob { time_limit_s: Some(time_limit_s), ..job.clone() };
    let dir = job.out_dir.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new(".")).to_path_buf();
    let job_path = write_job_file(&dir, &job)?;
    let hard_limit = hard_limit.unwrap_or(Duration::from_secs(time_limit_s) + PARENT_GRACE);
    let result = run_child(cmd, &job_path, limits.memory_limit_mb, hard_limit, cancel).and_then(|out| {
        let outcome = interpret(&out, limits);
        let peak_mb = out.peak_memory_bytes as f64 / (1024.0 * 1024.0);
        tracing::info!(
            "vc import worker: {} in {} ms, peak {peak_mb:.0} MB (exit {:?})",
            if outcome.is_ok() { "ok" } else { "failed" },
            out.elapsed.as_millis(),
            out.exit_code
        );
        outcome.map(|mut record| {
            if let Some(obj) = record.as_object_mut() {
                obj.insert(
                    "worker".into(),
                    serde_json::json!({ "peak_memory_mb": (peak_mb * 10.0).round() / 10.0, "wall_ms": out.elapsed.as_millis() as u64 }),
                );
            }
            record
        })
    });
    let _ = std::fs::remove_file(&job_path);
    if result.is_err() {
        if let Err(e) = super::recover_slot_folder(&job.out_dir) {
            tracing::warn!("vc import: could not restore {} after a failed import: {e}", job.out_dir.display());
        }
    }
    result
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    const ROLE_ENV: &str = "SIGIL_VC_FAKE_WORKER";
    const JOB_ENV: &str = "SIGIL_VC_FAKE_WORKER_JOB";
    const ENTRY: &str = "services::neural_audio::vc::import::worker::tests::fake_worker_entry";

    /// The test binary re-run as a worker in `role` (see [`fake_worker_entry`]; `real` runs the
    /// actual worker body, as `Sigil.exe --headless vc-import-worker` does).
    pub(crate) fn fake(role: &str) -> WorkerCommand {
        WorkerCommand {
            program: std::env::current_exe().expect("test binary"),
            args: ["--exact", ENTRY, "--ignored", "--nocapture", "--test-threads=1", "-q"].iter().map(|s| s.to_string()).collect(),
            env: vec![(ROLE_ENV.into(), role.into()), (JOB_ENV.into(), "{job}".into())],
        }
    }

    /// Child side: `real` runs the actual worker body on the job, printing like `cli::report`.
    #[test]
    #[ignore = "child process of the isolated worker tests (SIGIL_VC_FAKE_WORKER)"]
    fn fake_worker_entry() {
        let (Ok(role), Ok(job)) = (std::env::var(ROLE_ENV), std::env::var(JOB_ENV)) else {
            return;
        };
        std::thread::spawn(|| {
            std::thread::sleep(Duration::from_secs(120));
            std::process::exit(99);
        });
        match role.as_str() {
            "real" => match super::super::run_import_worker(Path::new(&job)) {
                Ok(text) => {
                    println!("{text}");
                    std::process::exit(0)
                }
                Err(e) => {
                    eprintln!("{ERROR_PREFIX}{e}");
                    std::process::exit(1)
                }
            },
            "sleep" => {
                std::thread::sleep(Duration::from_secs(60));
                std::process::exit(0)
            }
            "hog" => {
                let mut hold = Vec::new();
                loop {
                    hold.push(vec![1u8; 32 << 20]);
                }
            }
            "noise" => {
                println!("not a record");
                std::process::exit(0)
            }
            "crash" => std::process::exit(-1073741819), // 0xC0000005, as an access violation reports
            _ => std::process::exit(2),
        }
    }

    fn tiny() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("test-fixtures").join("vc").join("tiny")
    }

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sigil-vc-worker-{tag}-{}-{}", std::process::id(), nonce()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn job_for(source: PathBuf, root: &Path) -> WorkerJob {
        WorkerJob::new(source, root.join("models").join("voice"), Some(tiny()))
    }

    #[test]
    fn the_real_worker_imports_in_its_own_job() {
        let root = temp("real");
        let job = WorkerJob { name: Some("Tiny".into()), ..job_for(tiny().join("tiny.pth"), &root) };
        let record = run_isolated_with(&fake("real"), &job, WorkerLimits::default(), &AtomicBool::new(false), None).unwrap();
        assert_eq!(record["template"], "v2-16k-f0-hifigan");
        assert_eq!(record["name"], "Tiny");
        assert!(record["worker"]["peak_memory_mb"].as_f64().unwrap() > 1.0, "{}", record["worker"]);
        assert!(record["worker"]["wall_ms"].as_u64().is_some());
        let slot = root.join("models").join("voice");
        assert!(slot.join(super::super::WEIGHTS_FILE).is_file() && slot.join(super::super::MANIFEST_FILE).is_file());
        let leftovers: Vec<_> = std::fs::read_dir(root.join("models")).unwrap().flatten().map(|e| e.file_name()).collect();
        assert_eq!(leftovers.len(), 1, "the job file is removed: {leftovers:?}");

        // A refused file keeps its code through the process boundary.
        let bad = root.join("bad.pth");
        std::fs::write(&bad, [0x80u8, 0x02, 0x8a]).unwrap();
        let err = run_isolated_with(&fake("real"), &job_for(bad, &root), WorkerLimits::default(), &AtomicBool::new(false), None).unwrap_err();
        assert_eq!(err.code, ErrorCode::LegacyFormat, "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cancellation_time_limit_memory_cap_and_crashes() {
        let root = temp("limits");
        let job = job_for(tiny().join("tiny.pth"), &root);
        let limits = WorkerLimits::default();

        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        let flag = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(400));
            flag.store(true, Ordering::Release);
        });
        let started = Instant::now();
        let err = run_isolated_with(&fake("sleep"), &job, limits, &cancel, None).unwrap_err();
        assert_eq!(err.code, ErrorCode::Job, "{err}");
        assert!(err.message.contains("cancelled"));
        assert!(started.elapsed() < Duration::from_secs(10));

        let err = run_isolated_with(&fake("sleep"), &job, limits, &AtomicBool::new(false), Some(Duration::from_millis(800))).unwrap_err();
        assert_eq!(err.code, ErrorCode::TimeLimit, "{err}");

        let small = WorkerLimits { memory_limit_mb: 256, ..limits };
        let err = run_isolated_with(&fake("hog"), &job, small, &AtomicBool::new(false), None).unwrap_err();
        assert_eq!(err.code, ErrorCode::ArchiveRejected, "{err}");
        assert!(err.message.contains("256 MB"), "{err}");

        let err = run_isolated_with(&fake("crash"), &job, limits, &AtomicBool::new(false), None).unwrap_err();
        assert_eq!(err.code, ErrorCode::Io, "{err}");
        assert!(err.message.contains("crashed"), "{err}");

        let err = run_isolated_with(&fake("noise"), &job, limits, &AtomicBool::new(false), None).unwrap_err();
        assert_eq!(err.code, ErrorCode::Io, "{err}");
        assert!(!root.join("models").join("voice").exists(), "nothing was imported");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Import corpus files in isolated worker processes; every result must meet the corpus
    /// expectation after crossing the process boundary.
    fn corpus_through_workers(pick: impl Fn(usize, &str) -> bool) -> usize {
        use super::super::corpus_tests::{corpus, Outcome};
        let (templates, files) = corpus();
        let root = temp("corpus");
        let mut failures = Vec::new();
        let mut ran = 0;
        for (i, (path, f)) in files.iter().enumerate().filter(|(i, (_, f))| pick(*i, &f.file)) {
            let job = WorkerJob { smoke: false, ..WorkerJob::new(path.clone(), root.join("models").join(format!("v{i}")), Some(templates.clone())) };
            let outcome = match run_isolated_with(&fake("real"), &job, WorkerLimits::default(), &AtomicBool::new(false), None) {
                Ok(_) => Outcome::Ok,
                Err(e) => Outcome::Err(e.code, e.message),
            };
            ran += 1;
            if !outcome.meets(&f.expect) {
                failures.push(format!("{}: expected {}, got {outcome:?}", f.file, f.expect));
            }
        }
        let _ = std::fs::remove_dir_all(&root);
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        ran
    }

    /// One file of each kind through the isolated path (the whole corpus: the ignored test below).
    #[test]
    fn corpus_samples_cross_the_process_boundary_with_their_codes() {
        let wanted = [
            "zip-bomb.zip", "zip-name-00.zip", "global-os-system.pth", "opcode-build.pth", "bomb-codecs-copies.pth",
            "torch-torchscript.pth", "tensor-view-outside.pth", "semantic-shape-mismatch.pth", "meta-embedder-spin-v1.pth",
            "valid-voice-archive.zip", "mutated-000.pth",
        ];
        assert_eq!(corpus_through_workers(|_, f| wanted.contains(&f)), wanted.len());
    }

    #[test]
    #[ignore = "200 worker processes (about a minute): the whole fuzz corpus through the isolated path"]
    fn the_whole_corpus_crosses_the_process_boundary() {
        assert_eq!(corpus_through_workers(|_, _| true), 200);
    }

    #[test]
    fn worker_output_is_interpreted_strictly() {
        let limits = WorkerLimits::default();
        let out = |code: Option<i32>, stdout: &str, stderr: &str| ChildOutput {
            exit_code: code,
            stdout: stdout.as_bytes().to_vec(),
            stderr: stderr.as_bytes().to_vec(),
            ..ChildOutput::default()
        };
        let rec = r#"{"format":"sigil-rvc-import","template":"t"}"#;
        assert_eq!(interpret(&out(Some(0), rec, ""), limits).unwrap()["template"], "t");
        // libtest / console noise before the record, trailing text after it.
        assert!(interpret(&out(Some(0), &format!("\nrunning 1 test\n{rec}\nbye"), ""), limits).is_ok());
        assert_eq!(interpret(&out(Some(0), r#"{"format":"other"}"#, ""), limits).unwrap_err().code, ErrorCode::Io);
        let e = interpret(&out(Some(1), "", "warn: x\nvc-import-worker: shape_mismatch: a is [1] not [2]\n"), limits).unwrap_err();
        assert_eq!((e.code, e.message.as_str()), (ErrorCode::ShapeMismatch, "a is [1] not [2]"));
        let e = interpret(&out(Some(1), "", "vc-import-worker: nonsense: x"), limits).unwrap_err();
        assert_eq!(e.code, ErrorCode::Io);
        assert_eq!(interpret(&out(Some(2), "", "usage"), limits).unwrap_err().code, ErrorCode::Io);
        assert_eq!(interpret(&out(None, "", ""), limits).unwrap_err().code, ErrorCode::Io);
        assert_eq!(ImportError::parse("time_limit: slow").unwrap().code, ErrorCode::TimeLimit);
        assert!(ImportError::parse("C: drive").is_none());
    }

    #[test]
    fn job_files_round_trip_into_the_worker_schema() {
        let root = temp("jobfile");
        let job = WorkerJob { name: Some("N".into()), index: Some(root.join("x.index")), ..WorkerJob::new(root.join("a.pth"), root.join("s"), None) };
        let path = write_job_file(&root, &job).unwrap();
        assert!(path.file_name().unwrap().to_string_lossy().starts_with('.'), "hidden");
        let parsed: super::super::ImportJob = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!((parsed.name.as_deref(), parsed.smoke, parsed.smoke_fp16), (Some("N"), true, false));
        assert_eq!(parsed.index.as_deref(), Some(root.join("x.index").as_path()));
        let bare: super::super::ImportJob = serde_json::from_value(serde_json::to_value(WorkerJob::new(root.join("a"), root.join("b"), None)).unwrap()).unwrap();
        assert!(bare.name.is_none() && bare.index.is_none());
        let _ = std::fs::remove_dir_all(&root);
    }
}
