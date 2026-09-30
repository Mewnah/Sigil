//! Worker-process isolation for speech engines (P2-D).
//!
//! transcribe.cpp's GPU backend ends the whole process on CUDA errors it cannot return (a failed
//! memory-pool growth on a full card, a device lost to a driver reset during a game: ADR 0003b,
//! "Known in-process abort paths"). With `isolation: "worker"` (or `SIGIL_STT_ISOLATION=worker`)
//! the engine runs in a `Sigil.exe --headless stt-worker` child instead, so such an abort ends the
//! worker, never Sigil.exe; the parent wraps the child as an [`AsrEngine`] ([`WorkerEngine`]).
//!
//! # Process
//! The child is started with `process_host`'s launcher (`process_host::instance::Instance`): a
//! kill-on-close Job Object with a committed-memory cap, `CREATE_NO_WINDOW`, no port, stdin/stdout
//! as a binary protocol ([`StdioMode::Protocol`]) and stderr piped into the helper log (target
//! `sigil::helper`, so the child's `tracing` output and a ggml abort message reach Sigil's log).
//! Killing Sigil kills the worker.
//!
//! # Wire protocol (private to this module; both ends are the same `Sigil.exe`)
//! Frames: `b"SGW1"` · header length (u32 LE) · payload length (u32 LE) · header (JSON) · payload
//! (little-endian f32 samples, 16 kHz mono). Bytes before a frame's magic are skipped (and logged),
//! so stray output of native code on stdout cannot desynchronize the stream. Requests
//! ([`Request`]): `init {spec}` (first, unless the spec file was given), `transcribe`,
//! `begin_stream`, `push_stream`, `end_stream`, `abort_stream` (each with an `id`), `cancel`
//! (handled at once by the child's reader thread through the engine's canceller) and `shutdown`
//! (or EOF). Replies ([`Reply`]): `ready {info}`, `status` (engine load / fallback reports, any
//! time), and per request `result` / `hypothesis` / `done` / `error`, each with the engine's
//! current [`EngineInfo`] (a model can move to the CPU inside the worker). The spec may carry
//! secrets: it is never logged.
//!
//! # Crashes, hangs and the CPU fallback
//! - A worker that ends during a call fails that call with [`AsrErrorKind::Runtime`] (the utterance
//!   is lost; a two-pass pipeline promotes its CPU final); the next call starts a new worker.
//! - After [`MAX_GPU_CRASHES`] crashes of a GPU worker, or a crash while it loads, the worker
//!   restarts from the CPU spec (`WorkerSetup::cpu_spec`), releases the GPU claims and reports
//!   `engine_fallback`; without a CPU spec (`cpuFallback: false`) it is not restarted again. After
//!   [`MAX_CRASHES`] crashes or failed restarts (a restarted worker whose engine refuses to load
//!   counts too) in total it is not restarted again.
//! - A call without an answer within its deadline (a hung GPU) kills the worker and counts as a
//!   crash.
//!
//! # VRAM
//! The parent holds the claim for the whole child (`resources::claim_external_vram`, taken with
//! the governor's admission in `engines::transcribe`) and resizes it to the child's measured
//! process VRAM ([`EngineInfo::vram_mb`]). A child on the CPU releases it, or keeps what it reports
//! it still holds (the CUDA context of a GPU runtime it initialized before it moved to the CPU;
//! the parent's own info then says 0). A worker that ended holds nothing, so its claims go with it,
//! and every restart from the GPU spec runs the governor again ([`WorkerSetup::readmit`]): a new
//! admission and claim, the CPU for now when the governor says no (`engine_fallback`), or an error
//! without the CPU fallback. The GPU is never used on a stale admission.
//!
//! # Frozen signature (P2-S0)
//! - [`headless_stt_worker`]`(spec: Option<&Path>) -> Result<String, String>`: body of the
//!   `stt-worker` CLI arm (`Ok` → stdout + exit 0, `Err` → stderr + exit 1).

use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::engine_api::{
    AsrEngine, AsrError, AsrErrorKind, AsrResult, Canceller, DecodeOptions, EngineContext, EngineInfo, EngineSpec, EngineStatus, Hypothesis,
    StatusLevel, StatusSink, SAMPLE_RATE,
};
use super::events::codes;
use crate::services::process_host::instance::Instance;
use crate::services::process_host::logs::LogBook;
use crate::services::process_host::{HelperSpec, StdioMode};
use crate::services::speech_rt::transcribe::governor::GpuClaims;

/// `inProcess` (default) or `worker`: where GPU speech engines run when their spec does not say.
pub const ISOLATION_ENV: &str = "SIGIL_STT_ISOLATION";
/// Development / tests only (debug builds): the executable that runs `--headless stt-worker`
/// instead of the current one (a test binary cannot host the worker).
pub const WORKER_EXE_ENV: &str = "SIGIL_STT_WORKER_EXE";
/// GPU worker crashes after which it restarts on the CPU spec.
pub const MAX_GPU_CRASHES: u32 = 2;
/// Crashes after which the worker is not restarted again.
pub const MAX_CRASHES: u32 = 5;

const LOG: &str = "sigil::speech::worker_proc";
const MAGIC: [u8; 4] = *b"SGW1";
/// Largest header (JSON) of a frame.
const MAX_HEADER_BYTES: usize = 1 << 20;
/// Largest payload of a frame (about 17 minutes of 16 kHz f32 audio).
const MAX_PAYLOAD_BYTES: usize = 64 << 20;
/// Stray bytes kept for the log per frame.
const MAX_JUNK_LOGGED: usize = 512;
/// Model load (and GPU warm-up) in a fresh worker.
const INIT_TIMEOUT: Duration = Duration::from_secs(300);
/// Base deadline of a call, plus [`CALL_TIMEOUT_PER_AUDIO_SECOND`] per second of audio in it.
const CALL_TIMEOUT: Duration = Duration::from_secs(60);
const CALL_TIMEOUT_PER_AUDIO_SECOND: Duration = Duration::from_secs(6);
/// A stopping worker gets this long to unload before it is killed.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(3);

static IN_WORKER: AtomicBool = AtomicBool::new(false);

/// Whether this process is a `--headless stt-worker` child (engines then never isolate again).
pub fn is_worker_process() -> bool {
    IN_WORKER.load(Ordering::SeqCst)
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

// ---------------------------------------------------------------------------------------------
// Wire protocol
// ---------------------------------------------------------------------------------------------

/// Parent → worker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub(crate) enum Request {
    Init { spec: EngineSpec },
    /// Payload: the utterance.
    Transcribe { id: u64, options: DecodeOptions },
    BeginStream { id: u64, options: DecodeOptions },
    /// Payload: the chunk.
    PushStream { id: u64 },
    EndStream { id: u64 },
    AbortStream { id: u64 },
    Cancel,
    Shutdown,
}

impl Request {
    fn id(&self) -> Option<u64> {
        match self {
            Request::Transcribe { id, .. }
            | Request::BeginStream { id, .. }
            | Request::PushStream { id }
            | Request::EndStream { id }
            | Request::AbortStream { id } => Some(*id),
            Request::Init { .. } | Request::Cancel | Request::Shutdown => None,
        }
    }
}

/// Worker → parent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub(crate) enum Reply {
    Ready { info: EngineInfo, cancellable: bool },
    Status { status: EngineStatus },
    Result { id: u64, result: AsrResult, info: EngineInfo },
    Hypothesis { id: u64, hypothesis: Option<Hypothesis>, info: EngineInfo },
    Done { id: u64, info: EngineInfo },
    Error { id: Option<u64>, error: AsrError, info: Option<EngineInfo> },
}

impl Reply {
    fn id(&self) -> Option<u64> {
        match self {
            Reply::Result { id, .. } | Reply::Hypothesis { id, .. } | Reply::Done { id, .. } => Some(*id),
            Reply::Error { id, .. } => *id,
            Reply::Ready { .. } | Reply::Status { .. } => None,
        }
    }

    fn info(&self) -> Option<&EngineInfo> {
        match self {
            Reply::Ready { info, .. } | Reply::Result { info, .. } | Reply::Hypothesis { info, .. } | Reply::Done { info, .. } => Some(info),
            Reply::Error { info, .. } => info.as_ref(),
            Reply::Status { .. } => None,
        }
    }
}

/// One decoded frame.
#[derive(Debug, PartialEq)]
pub(crate) struct Frame {
    pub header: Vec<u8>,
    pub payload: Vec<u8>,
}

impl Frame {
    fn samples(&self) -> Vec<f32> {
        self.payload
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }
}

/// Encode one frame (a single buffer, written with one `write_all`, then flushed).
pub(crate) fn encode_frame(header: &impl Serialize, payload: &[f32]) -> io::Result<Vec<u8>> {
    let json = serde_json::to_vec(header).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let bytes = payload.len() * 4;
    if json.len() > MAX_HEADER_BYTES || bytes > MAX_PAYLOAD_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("frame too large ({} header bytes, {bytes} payload bytes)", json.len()),
        ));
    }
    let mut buf = Vec::with_capacity(12 + json.len() + bytes);
    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&(json.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(bytes as u32).to_le_bytes());
    buf.extend_from_slice(&json);
    for s in payload {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    Ok(buf)
}

pub(crate) fn write_frame<W: Write + ?Sized>(w: &mut W, header: &impl Serialize, payload: &[f32]) -> io::Result<()> {
    let buf = encode_frame(header, payload)?;
    w.write_all(&buf)?;
    w.flush()
}

fn read_byte<R: Read + ?Sized>(r: &mut R) -> io::Result<Option<u8>> {
    let mut b = [0u8; 1];
    loop {
        match r.read(&mut b) {
            Ok(0) => return Ok(None),
            Ok(_) => return Ok(Some(b[0])),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
}

fn keep_junk(junk: &mut Vec<u8>, bytes: &[u8]) {
    let room = MAX_JUNK_LOGGED.saturating_sub(junk.len());
    junk.extend_from_slice(&bytes[..bytes.len().min(room)]);
}

/// The next frame; `Ok(None)` at a clean end of the stream. Bytes before a frame's magic are
/// skipped and (the first [`MAX_JUNK_LOGGED`] of them) left in `junk`. Wrap `r` in a `BufReader`:
/// the magic is searched byte by byte.
pub(crate) fn read_frame<R: Read + ?Sized>(r: &mut R, junk: &mut Vec<u8>) -> io::Result<Option<Frame>> {
    // The magic has no repeated prefix, so a mismatch restarts the search at the current byte.
    let mut matched = 0usize;
    while matched < MAGIC.len() {
        let Some(b) = read_byte(r)? else {
            keep_junk(junk, &MAGIC[..matched]);
            return Ok(None);
        };
        if b == MAGIC[matched] {
            matched += 1;
        } else {
            keep_junk(junk, &MAGIC[..matched]);
            if b == MAGIC[0] {
                matched = 1;
            } else {
                keep_junk(junk, &[b]);
                matched = 0;
            }
        }
    }
    let mut lens = [0u8; 8];
    r.read_exact(&mut lens)?;
    let header_len = u32::from_le_bytes([lens[0], lens[1], lens[2], lens[3]]) as usize;
    let payload_len = u32::from_le_bytes([lens[4], lens[5], lens[6], lens[7]]) as usize;
    if header_len == 0 || header_len > MAX_HEADER_BYTES || payload_len > MAX_PAYLOAD_BYTES || !payload_len.is_multiple_of(4) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad frame lengths ({header_len} header bytes, {payload_len} payload bytes)"),
        ));
    }
    let mut header = vec![0u8; header_len];
    r.read_exact(&mut header)?;
    let mut payload = vec![0u8; payload_len];
    r.read_exact(&mut payload)?;
    Ok(Some(Frame { header, payload }))
}

/// Stray output for the log (lossy text, control characters shown as spaces).
fn junk_text(junk: &[u8]) -> String {
    String::from_utf8_lossy(junk)
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect::<String>()
        .trim()
        .to_string()
}

// ---------------------------------------------------------------------------------------------
// Worker side
// ---------------------------------------------------------------------------------------------

/// Creates the hosted engine (production: `engines::create`).
pub(crate) type EngineFactory = Arc<dyn Fn(&EngineSpec, &EngineContext) -> Result<Box<dyn AsrEngine>, AsrError> + Send + Sync>;

enum ChildMsg {
    Request(Request, Vec<f32>),
    Quit(Option<String>),
}

/// Serve one engine over `input` / `output` until `shutdown`, EOF or a broken pipe. `init`: the
/// spec when it was given up front (the spec-file form); otherwise the first request must be
/// `init`. Returns an error only when the engine could not be created.
pub(crate) fn serve<R, W>(input: R, output: W, init: Option<EngineSpec>, factory: EngineFactory) -> Result<(), String>
where
    R: Read + Send + 'static,
    W: Write + Send + 'static,
{
    let out: Arc<Mutex<W>> = Arc::new(Mutex::new(output));
    let send = {
        let out = Arc::clone(&out);
        move |reply: &Reply, payload: &[f32]| -> bool { write_frame(&mut *lock(&out), reply, payload).is_ok() }
    };
    let canceller: Arc<Mutex<Option<Canceller>>> = Arc::new(Mutex::new(None));
    let (tx, rx) = mpsc::channel::<ChildMsg>();

    let reader_canceller = Arc::clone(&canceller);
    std::thread::Builder::new()
        .name("sigil-stt-worker-in".into())
        .spawn(move || {
            let mut input = BufReader::new(input);
            loop {
                let mut junk = Vec::new();
                let frame = read_frame(&mut input, &mut junk);
                if !junk.is_empty() {
                    tracing::warn!(target: LOG, "skipped {} unexpected byte(s) on stdin", junk.len());
                }
                let frame = match frame {
                    Ok(Some(f)) => f,
                    Ok(None) => {
                        let _ = tx.send(ChildMsg::Quit(None));
                        return;
                    }
                    Err(e) => {
                        let _ = tx.send(ChildMsg::Quit(Some(format!("bad request stream: {e}"))));
                        return;
                    }
                };
                let request: Request = match serde_json::from_slice(&frame.header) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx.send(ChildMsg::Quit(Some(format!("bad request: {e}"))));
                        return;
                    }
                };
                match request {
                    Request::Cancel => {
                        let c = lock(&reader_canceller).clone();
                        if let Some(c) = c {
                            c.cancel();
                        }
                    }
                    Request::Shutdown => {
                        let _ = tx.send(ChildMsg::Quit(None));
                        return;
                    }
                    other => {
                        let samples = frame.samples();
                        if tx.send(ChildMsg::Request(other, samples)).is_err() {
                            return;
                        }
                    }
                }
            }
        })
        .map_err(|e| format!("cannot start the request reader: {e}"))?;

    let spec = match init {
        Some(spec) => spec,
        None => loop {
            match rx.recv() {
                Ok(ChildMsg::Request(Request::Init { spec }, _)) => break spec,
                Ok(ChildMsg::Request(other, _)) => {
                    let error = AsrError::invalid_input("the worker has no engine yet (init first)");
                    if !send(&Reply::Error { id: other.id(), error, info: None }, &[]) {
                        return Ok(());
                    }
                }
                Ok(ChildMsg::Quit(reason)) => {
                    if let Some(r) = reason {
                        tracing::warn!(target: LOG, "{r}");
                    }
                    return Ok(());
                }
                Err(_) => return Ok(()),
            }
        },
    };

    let status_out = Arc::clone(&out);
    let ctx = EngineContext {
        status: StatusSink::new(move |status| {
            let _ = write_frame(&mut *lock(&status_out), &Reply::Status { status }, &[]);
        }),
    };
    let mut engine = match factory(&spec, &ctx) {
        Ok(e) => e,
        Err(error) => {
            let message = error.to_string();
            send(&Reply::Error { id: None, error, info: None }, &[]);
            return Err(message);
        }
    };
    let engine_canceller = engine.canceller();
    let cancellable = engine_canceller.is_some();
    *lock(&canceller) = engine_canceller;
    if !send(
        &Reply::Ready {
            info: engine.info().clone(),
            cancellable,
        },
        &[],
    ) {
        return Ok(());
    }

    loop {
        let (request, audio) = match rx.recv() {
            Ok(ChildMsg::Request(r, a)) => (r, a),
            Ok(ChildMsg::Quit(reason)) => {
                if let Some(r) = reason {
                    tracing::warn!(target: LOG, "{r}");
                }
                break;
            }
            Err(_) => break,
        };
        let error = |id: Option<u64>, error: AsrError, engine: &dyn AsrEngine| Reply::Error {
            id,
            error,
            info: Some(engine.info().clone()),
        };
        let reply = match request {
            Request::Transcribe { id, options } => match engine.transcribe(&audio, &options) {
                Ok(result) => Reply::Result {
                    id,
                    result,
                    info: engine.info().clone(),
                },
                Err(e) => error(Some(id), e, &*engine),
            },
            Request::BeginStream { id, options } => match engine.begin_stream(&options) {
                Ok(()) => Reply::Done {
                    id,
                    info: engine.info().clone(),
                },
                Err(e) => error(Some(id), e, &*engine),
            },
            Request::PushStream { id } => match engine.push_stream(&audio) {
                Ok(hypothesis) => Reply::Hypothesis {
                    id,
                    hypothesis,
                    info: engine.info().clone(),
                },
                Err(e) => error(Some(id), e, &*engine),
            },
            Request::EndStream { id } => match engine.end_stream() {
                Ok(result) => Reply::Result {
                    id,
                    result,
                    info: engine.info().clone(),
                },
                Err(e) => error(Some(id), e, &*engine),
            },
            Request::AbortStream { id } => {
                engine.abort_stream();
                Reply::Done {
                    id,
                    info: engine.info().clone(),
                }
            }
            Request::Init { .. } => error(None, AsrError::invalid_input("the worker already hosts an engine"), &*engine),
            // Handled by the reader thread.
            Request::Cancel | Request::Shutdown => continue,
        };
        if !send(&reply, &[]) {
            break;
        }
    }
    *lock(&canceller) = None;
    drop(engine);
    Ok(())
}

/// Log to stderr (the parent pipes it into Sigil's log); stdout carries the protocol.
fn init_worker_logging() {
    use tracing_subscriber::filter::Targets;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::Layer;
    const DEFAULT: &str = "sigil=info,warn";
    let spec = std::env::var(crate::services::diagnostics::logging::LOG_ENV)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT.to_string());
    let targets: Targets = spec
        .parse()
        .unwrap_or_else(|_| DEFAULT.parse().expect("the default filter parses"));
    let layer = tracing_subscriber::fmt::layer()
        .with_writer(io::stderr)
        .with_ansi(false)
        .with_target(true)
        .with_filter(targets);
    let _ = tracing_subscriber::registry().with(layer).try_init();
}

/// Body of `Sigil.exe --headless stt-worker [<spec.json>]`: host one speech engine over
/// stdin / stdout (see the module docs). `spec`: an [`EngineSpec`] JSON file to load at once
/// instead of waiting for `init`.
pub fn headless_stt_worker(spec: Option<&Path>) -> Result<String, String> {
    IN_WORKER.store(true, Ordering::SeqCst);
    init_worker_logging();
    let init = match spec {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            Some(serde_json::from_str::<EngineSpec>(&text).map_err(|e| format!("{} is not an engine spec: {e}", path.display()))?)
        }
        None => None,
    };
    tracing::info!(target: LOG, "speech worker {} started", std::process::id());
    serve(io::stdin(), io::stdout(), init, Arc::new(super::engines::create))?;
    tracing::info!(target: LOG, "speech worker {} stopped", std::process::id());
    // Nothing follows on stdout: the CLI prints this (empty) text after the protocol ended.
    Ok(String::new())
}

// ---------------------------------------------------------------------------------------------
// Parent side
// ---------------------------------------------------------------------------------------------

/// A started worker: its pipes and a handle on the process.
pub(crate) struct Launched {
    pub writer: Box<dyn Write + Send>,
    pub reader: Box<dyn Read + Send>,
    pub process: Box<dyn WorkerProcess>,
}

/// The worker process as the parent sees it.
pub(crate) trait WorkerProcess: Send {
    /// Terminate it (and everything it started). Idempotent.
    fn kill(&self);
    /// Why it ended, waiting up to `wait` for the exit to be known.
    fn exit_description(&self, wait: Duration) -> String;
}

/// Starts workers (production: `Sigil.exe --headless stt-worker` under the process host).
pub(crate) trait Launcher: Send + Sync {
    fn launch(&self, name: &str, memory_limit_mb: Option<u64>) -> Result<Launched, String>;
}

/// The executable that hosts workers.
fn worker_program() -> Result<PathBuf, String> {
    if cfg!(debug_assertions) {
        if let Some(p) = std::env::var_os(WORKER_EXE_ENV).filter(|v| !v.is_empty()) {
            return Ok(PathBuf::from(p));
        }
    }
    std::env::current_exe().map_err(|e| format!("cannot locate Sigil.exe: {e}"))
}

/// `Sigil.exe --headless stt-worker` in a process-host job.
struct ProcessLauncher;

struct HostedProcess {
    instance: Instance,
    book: Arc<LogBook>,
}

impl WorkerProcess for HostedProcess {
    fn kill(&self) {
        self.instance.kill();
    }

    fn exit_description(&self, wait: Duration) -> String {
        let deadline = Instant::now() + wait;
        while !self.instance.has_exited() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        let rx = self.instance.exit_rx();
        let exit = rx.borrow().clone();
        match exit {
            Some(info) => format!("{info}{}", self.book.tail_for_error()),
            None => format!("process {} stopped answering", self.instance.pid),
        }
    }
}

impl Launcher for ProcessLauncher {
    fn launch(&self, name: &str, memory_limit_mb: Option<u64>) -> Result<Launched, String> {
        let program = worker_program()?;
        let mut env = vec![(ISOLATION_ENV.to_string(), "inProcess".to_string())];
        if let Some(dir) = crate::services::diagnostics::app_data_dir() {
            env.push((
                crate::services::diagnostics::APP_DATA_ENV.to_string(),
                dir.to_string_lossy().into_owned(),
            ));
        }
        let spec = HelperSpec {
            name: name.to_string(),
            program,
            args: vec![crate::services::cli::HEADLESS_FLAG.to_string(), "stt-worker".to_string()],
            env,
            memory_limit_mb,
            high_qos: true,
            stdio: StdioMode::Protocol,
            ..HelperSpec::default()
        };
        let book = Arc::new(LogBook::new(name, None));
        let instance = Instance::start(&spec, None, None, &book)?;
        let stdio = instance
            .take_stdio()
            .ok_or_else(|| format!("{name}: the worker's pipes are not available"))?;
        Ok(Launched {
            writer: Box::new(stdio.stdin),
            reader: Box::new(stdio.stdout),
            process: Box::new(HostedProcess { instance, book }),
        })
    }
}

/// A fresh governor decision for a restart of a GPU worker ([`WorkerSetup::readmit`]).
pub enum Readmission {
    /// Admitted again: start `spec` (carrying the new admission) holding `claims`.
    Gpu { spec: EngineSpec, claims: GpuClaims },
    /// The governor keeps the model off the GPU for now: start `spec` (a CPU admission with its
    /// reason) and report `reason` as `engine_fallback`. The next restart asks again.
    Cpu { spec: EngineSpec, reason: String },
    /// Off the GPU and the CPU fallback is off: the call fails with this error and no worker starts
    /// (the next call asks again).
    Refused(AsrError),
}

/// Runs the governor for a restart from the GPU spec.
pub type Readmit = Box<dyn FnMut() -> Readmission + Send>;

/// What the parent needs to run an engine in a worker.
pub struct WorkerSetup {
    /// The engine to host (its options say `isolation: inProcess`, plus the forwarded admission).
    pub spec: EngineSpec,
    /// `Some` when `spec` runs on the GPU: the governor again for every restart of it (the first
    /// start uses the admission `claims` came with). `None`: `spec` runs on the CPU.
    pub readmit: Option<Readmit>,
    /// The spec to restart with after repeated GPU crashes (`None` for a GPU spec: the CPU
    /// fallback is off, so the worker is not restarted after them).
    pub cpu_spec: Option<EngineSpec>,
    pub ctx: EngineContext,
    /// The external VRAM claim of the whole child for the first start (empty on the CPU).
    pub claims: GpuClaims,
    /// Job-wide committed-memory cap.
    pub memory_limit_mb: Option<u64>,
    /// Model label for logs and the worker's name (no secrets).
    pub label: String,
}

type SharedWriter = Arc<Mutex<Box<dyn Write + Send>>>;

enum Incoming {
    Reply(Box<Reply>),
    Closed(String),
}

/// One running worker.
struct Link {
    writer: SharedWriter,
    rx: mpsc::Receiver<Incoming>,
    process: Box<dyn WorkerProcess>,
}

impl Link {
    fn send(&self, request: &Request, payload: &[f32]) -> io::Result<()> {
        write_frame(&mut **lock(&self.writer), request, payload)
    }
}

/// Cancellation shared with [`WorkerEngine::canceller`].
#[derive(Default)]
struct CancelShared {
    gen: AtomicU64,
    /// The current worker's stdin (replaced on restart).
    writer: Mutex<Option<SharedWriter>>,
}

/// Why a call failed.
enum CallFailure {
    /// The engine answered with an error.
    Engine(AsrError),
    /// The worker ended or stopped answering (counted as a crash; the next call restarts it).
    Crashed(String),
}

static WORKER_SEQ: AtomicU64 = AtomicU64::new(0);

/// An [`AsrEngine`] hosted in a `--headless stt-worker` child (see the module docs).
pub struct WorkerEngine {
    link: Option<Link>,
    launcher: Arc<dyn Launcher>,
    spec: EngineSpec,
    /// The governor for restarts of the GPU spec (`None`: a CPU setup).
    readmit: Option<Readmit>,
    cpu_spec: Option<EngineSpec>,
    /// Left the GPU spec for good (repeated GPU crashes), or a CPU setup.
    on_cpu_spec: bool,
    /// The setup's admission (and `claims`) are still unused: the first start takes them; every
    /// later start from the GPU spec runs the governor again.
    fresh_admission: bool,
    /// The running (or last started) worker hosts the GPU spec: its crashes are GPU crashes.
    current_gpu: bool,
    ctx: EngineContext,
    memory_limit_mb: Option<u64>,
    label: String,
    info: EngineInfo,
    /// Why the worker left the GPU spec (kept across restarts).
    fallback_reason: Option<String>,
    /// Crashes plus failed restarts.
    crashes: u32,
    gpu_crashes: u32,
    /// Why the last start failed (for the final error once restarts stop).
    last_failure: Option<String>,
    next_id: u64,
    cancel: Arc<CancelShared>,
    /// Samples pushed into the open stream (deadline of `end_stream`); `None` = no stream.
    stream_samples: Option<usize>,
    /// Test hook: deadline of calls (production: [`call_timeout`]).
    timeout_override: Option<Duration>,
    // Declared last: the claims outlive the worker (the process is gone before VRAM is released).
    claims: GpuClaims,
}

impl std::fmt::Debug for WorkerEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerEngine")
            .field("label", &self.label)
            .field("device", &self.info.device)
            .field("crashes", &self.crashes)
            .finish_non_exhaustive()
    }
}

/// Deadline of a call carrying `samples` of audio.
fn call_timeout(samples: usize) -> Duration {
    let seconds = samples as f64 / f64::from(SAMPLE_RATE);
    CALL_TIMEOUT + CALL_TIMEOUT_PER_AUDIO_SECOND.mul_f64(seconds)
}

/// A worker name for the process host (`stt-worker-<label>-<n>`, allowed characters only).
fn worker_name(label: &str) -> String {
    let mut slug: String = label
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_') { c } else { '-' })
        .collect();
    slug.truncate(40);
    let slug = slug.trim_matches(|c: char| !c.is_ascii_alphanumeric());
    let n = WORKER_SEQ.fetch_add(1, Ordering::SeqCst) + 1;
    if slug.is_empty() {
        format!("stt-worker-{n}")
    } else {
        format!("stt-worker-{slug}-{n}")
    }
}

impl WorkerEngine {
    /// Start the worker and load the engine in it (blocking; see the module docs for crashes).
    pub fn spawn(setup: WorkerSetup) -> Result<WorkerEngine, AsrError> {
        WorkerEngine::spawn_with(setup, Arc::new(ProcessLauncher))
    }

    pub(crate) fn spawn_with(setup: WorkerSetup, launcher: Arc<dyn Launcher>) -> Result<WorkerEngine, AsrError> {
        let info = EngineInfo {
            kind: setup.spec.engine,
            model: setup.label.clone(),
            runtime: "worker process".into(),
            device: "cpu".into(),
            caps: Default::default(),
            vram_mb: 0,
            fallback_reason: None,
        };
        let gpu = setup.readmit.is_some();
        let mut engine = WorkerEngine {
            link: None,
            launcher,
            on_cpu_spec: !gpu,
            fresh_admission: gpu,
            current_gpu: false,
            spec: setup.spec,
            readmit: setup.readmit,
            // A CPU setup restarts its own spec.
            cpu_spec: if gpu { setup.cpu_spec } else { None },
            ctx: setup.ctx,
            memory_limit_mb: setup.memory_limit_mb,
            label: setup.label,
            info,
            fallback_reason: None,
            crashes: 0,
            gpu_crashes: 0,
            last_failure: None,
            next_id: 0,
            cancel: Arc::new(CancelShared::default()),
            stream_samples: None,
            timeout_override: None,
            claims: setup.claims,
        };
        engine.ensure_link()?;
        Ok(engine)
    }

    fn report(&self, code: &str, detail: String, level: StatusLevel) {
        self.ctx.status.report(EngineStatus {
            code: code.to_string(),
            detail,
            progress: None,
            level,
            model_id: None,
        });
    }

    /// Leave the GPU spec for the CPU spec (repeated crashes), releasing the GPU claims.
    fn switch_to_cpu(&mut self, reason: String) {
        self.on_cpu_spec = true;
        self.claims.release();
        tracing::warn!(target: LOG, "{}: {reason}", self.label);
        self.fallback_reason = Some(reason.clone());
        self.report(codes::ENGINE_FALLBACK, reason, StatusLevel::Warning);
    }

    /// Record a worker crash (the link is gone, and with the process the VRAM it held).
    fn crashed(&mut self, why: &str) {
        self.link = None;
        *lock(&self.cancel.writer) = None;
        self.stream_samples = None;
        self.claims.release();
        self.crashes += 1;
        if self.current_gpu {
            self.gpu_crashes += 1;
        }
        self.last_failure = Some(why.to_string());
        tracing::warn!(
            target: LOG,
            "{}: the speech worker ended ({why}); {} crash(es) so far",
            self.label,
            self.crashes
        );
    }

    /// The model's info as the parent reports it: the worker's, marked as isolated, plus the
    /// parent's own fallback reason.
    fn adopt_info(&mut self, info: &EngineInfo) {
        let mut info = info.clone();
        if !info.runtime.ends_with("(worker process)") {
            info.runtime = format!("{} (worker process)", info.runtime);
        }
        if info.fallback_reason.is_none() {
            info.fallback_reason = self.fallback_reason.clone();
        }
        let off_gpu = info.device == "cpu" || info.device == "remote";
        if off_gpu && info.vram_mb == 0 {
            self.claims.release();
        } else if info.vram_mb > 0 && !self.claims.is_empty() && info.vram_mb != self.claims.model_mb() {
            // The child measured its whole process (model + runtime context; on the CPU the context
            // a GPU runtime left behind): that is the claim.
            self.claims.resize_model(info.vram_mb);
        }
        if off_gpu {
            // The engine's contract: 0 on the CPU (what the process still holds is in the ledger).
            info.vram_mb = 0;
        }
        self.info = info;
    }

    /// Start a worker when none runs (restarting after a crash; on the CPU spec after repeated GPU
    /// crashes).
    fn ensure_link(&mut self) -> Result<(), AsrError> {
        loop {
            if self.link.is_some() {
                return Ok(());
            }
            if self.crashes >= MAX_CRASHES {
                return Err(AsrError::runtime(format!(
                    "{}: the speech worker process crashed or failed to start {} times; it is not restarted again{}",
                    self.label,
                    self.crashes,
                    self.last_failure
                        .as_deref()
                        .map(|w| format!(" (last: {w})"))
                        .unwrap_or_default()
                )));
            }
            if !self.on_cpu_spec && self.gpu_crashes >= MAX_GPU_CRASHES {
                if self.cpu_spec.is_none() {
                    return Err(AsrError::runtime(format!(
                        "{}: its GPU speech worker process crashed {} times; it is not restarted again (the CPU fallback is off)",
                        self.label, self.gpu_crashes
                    )));
                }
                let reason = format!(
                    "{} runs on the CPU: its GPU worker process crashed {} times",
                    self.label, self.gpu_crashes
                );
                self.switch_to_cpu(reason);
            }
            let (spec, gpu) = self.next_spec()?;
            self.current_gpu = gpu;
            match self.start_link(spec) {
                Ok(()) => return Ok(()),
                Err(CallFailure::Engine(e)) => {
                    // Not a crash, but a restart that fails the same way on every call must not
                    // start a process per utterance forever: it counts toward MAX_CRASHES.
                    self.claims.release();
                    self.crashes += 1;
                    self.last_failure = Some(e.message.clone());
                    return Err(e);
                }
                Err(CallFailure::Crashed(why)) => {
                    self.crashed(&why);
                    if gpu && self.cpu_spec.is_some() {
                        // A crash while loading repeats on every load: go to the CPU at once.
                        let reason = format!("{} runs on the CPU: its GPU worker process crashed while loading ({why})", self.label);
                        self.switch_to_cpu(reason);
                    } else {
                        return Err(AsrError::runtime(format!("{}: the speech worker process failed to start ({why})", self.label)));
                    }
                }
            }
        }
    }

    /// The spec of the next start and whether it runs on the GPU: the CPU spec once the worker
    /// left the GPU; the setup's admission for the first GPU start; the governor again for every
    /// later one (see the module docs).
    fn next_spec(&mut self) -> Result<(EngineSpec, bool), AsrError> {
        if self.on_cpu_spec {
            return Ok((self.cpu_spec.clone().unwrap_or_else(|| self.spec.clone()), false));
        }
        if std::mem::take(&mut self.fresh_admission) {
            return Ok((self.spec.clone(), true));
        }
        let Some(readmit) = self.readmit.as_mut() else {
            return Ok((self.spec.clone(), true));
        };
        // Nothing runs now: the new admission must not count an old claim of this model.
        self.claims.release();
        match readmit() {
            Readmission::Gpu { spec, claims } => {
                self.claims = claims;
                Ok((spec, true))
            }
            Readmission::Cpu { spec, reason } => {
                tracing::warn!(target: LOG, "{}: {reason}", self.label);
                self.report(codes::ENGINE_FALLBACK, reason, StatusLevel::Warning);
                Ok((spec, false))
            }
            Readmission::Refused(e) => Err(e),
        }
    }

    /// Launch a worker for `spec`, send `init` and wait for `ready`.
    fn start_link(&mut self, spec: EngineSpec) -> Result<(), CallFailure> {
        let name = worker_name(&self.label);
        let launched = self
            .launcher
            .launch(&name, self.memory_limit_mb)
            .map_err(|e| CallFailure::Engine(AsrError::runtime(format!("cannot start the speech worker: {e}"))))?;
        let Launched { writer, reader, process } = launched;
        let writer: SharedWriter = Arc::new(Mutex::new(writer));
        let (tx, rx) = mpsc::channel::<Incoming>();
        let status = self.ctx.status.clone();
        let reader_name = format!("{name}-out");
        let spawned = std::thread::Builder::new()
            .name(reader_name)
            .spawn(move || pump_replies(reader, tx, status));
        if let Err(e) = spawned {
            process.kill();
            return Err(CallFailure::Engine(AsrError::runtime(format!("cannot start the worker reader: {e}"))));
        }
        let link = Link { writer, rx, process };
        if let Err(e) = link.send(&Request::Init { spec }, &[]) {
            let why = format!("{e}; {}", link.process.exit_description(Duration::from_secs(1)));
            link.process.kill();
            return Err(CallFailure::Crashed(why));
        }
        let timeout = self.timeout_override.unwrap_or(INIT_TIMEOUT);
        match wait_reply(&link, None, timeout) {
            Ok(Reply::Ready { info, .. }) => {
                *lock(&self.cancel.writer) = Some(Arc::clone(&link.writer));
                self.link = Some(link);
                self.adopt_info(&info);
                tracing::info!(
                    target: LOG,
                    "{} ready in a worker process on {} ({}{})",
                    self.label,
                    self.info.device,
                    self.info.runtime,
                    if self.info.vram_mb > 0 { format!(", ~{} MB VRAM", self.info.vram_mb) } else { String::new() }
                );
                Ok(())
            }
            Ok(Reply::Error { error, .. }) => {
                // The engine could not load (not installed, bad model): not a crash.
                shut_down(link, Duration::from_millis(500));
                Err(CallFailure::Engine(error))
            }
            Ok(other) => {
                shut_down(link, Duration::from_millis(500));
                Err(CallFailure::Engine(AsrError::runtime(format!(
                    "the speech worker answered init with {other:?}"
                ))))
            }
            Err(failure) => Err(failure),
        }
    }

    /// Send one request and wait for its reply (starting a worker first when none runs).
    fn call(&mut self, request: Request, payload: &[f32]) -> Result<Reply, AsrError> {
        self.ensure_link()?;
        let id = request.id();
        let timeout = self
            .timeout_override
            .unwrap_or_else(|| call_timeout(payload.len().max(self.stream_samples.unwrap_or(0))));
        let result = {
            let link = self.link.as_ref().expect("ensure_link");
            match link.send(&request, payload) {
                Ok(()) => wait_reply(link, id, timeout),
                Err(e) => Err(CallFailure::Crashed(format!(
                    "{e}; {}",
                    link.process.exit_description(Duration::from_secs(1))
                ))),
            }
        };
        match result {
            Ok(reply) => {
                if let Some(info) = reply.info().cloned() {
                    self.adopt_info(&info);
                }
                match reply {
                    Reply::Error { error, .. } => Err(error),
                    other => Ok(other),
                }
            }
            Err(CallFailure::Engine(e)) => Err(e),
            Err(CallFailure::Crashed(why)) => {
                self.crashed(&why);
                Err(AsrError::runtime(format!(
                    "{}: the speech worker process ended ({why}); it restarts for the next utterance",
                    self.label
                )))
            }
        }
    }

    fn next_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    fn cancelled_since(&self, gen: u64) -> bool {
        self.cancel.gen.load(Ordering::SeqCst) != gen
    }

    /// Test hook: shorter deadlines.
    #[cfg(test)]
    pub(crate) fn set_timeout(&mut self, timeout: Duration) {
        self.timeout_override = Some(timeout);
    }

    /// The crash counters (tests, diagnostics).
    #[cfg(test)]
    pub(crate) fn crash_counts(&self) -> (u32, u32) {
        (self.crashes, self.gpu_crashes)
    }

    /// Whether the claims are still held (tests).
    #[cfg(test)]
    pub(crate) fn holds_claims(&self) -> bool {
        !self.claims.is_empty()
    }
}

/// Read replies until the stream ends; status reports go straight to the engine's status sink.
fn pump_replies(reader: Box<dyn Read + Send>, tx: mpsc::Sender<Incoming>, status: StatusSink) {
    let mut reader = BufReader::new(reader);
    loop {
        let mut junk = Vec::new();
        let frame = read_frame(&mut reader, &mut junk);
        if !junk.is_empty() {
            let text = junk_text(&junk);
            if !text.is_empty() {
                tracing::warn!(target: LOG, "the speech worker wrote unexpected output: {text}");
            }
        }
        let frame = match frame {
            Ok(Some(f)) => f,
            Ok(None) => {
                let _ = tx.send(Incoming::Closed("its output ended".into()));
                return;
            }
            Err(e) => {
                let _ = tx.send(Incoming::Closed(format!("its output broke off ({e})")));
                return;
            }
        };
        match serde_json::from_slice::<Reply>(&frame.header) {
            Ok(Reply::Status { status: s }) => status.report(s),
            Ok(reply) => {
                if tx.send(Incoming::Reply(Box::new(reply))).is_err() {
                    return;
                }
            }
            Err(e) => {
                let _ = tx.send(Incoming::Closed(format!("it sent an unreadable reply ({e})")));
                return;
            }
        }
    }
}

/// Wait for the reply to `id` (`None`: `ready` or an id-less error). Replies to earlier calls that
/// timed out are skipped. A worker that ends or stays silent past `timeout` is a crash (a silent
/// one is killed).
fn wait_reply(link: &Link, id: Option<u64>, timeout: Duration) -> Result<Reply, CallFailure> {
    let deadline = Instant::now() + timeout;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match link.rx.recv_timeout(left) {
            Ok(Incoming::Reply(reply)) => {
                let matches = match id {
                    Some(want) => reply.id() == Some(want),
                    None => matches!(*reply, Reply::Ready { .. } | Reply::Error { id: None, .. }),
                };
                if matches {
                    return Ok(*reply);
                }
            }
            Ok(Incoming::Closed(why)) => {
                return Err(CallFailure::Crashed(format!(
                    "{why}; {}",
                    link.process.exit_description(Duration::from_millis(1500))
                )))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(CallFailure::Crashed(link.process.exit_description(Duration::from_millis(1500))))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                link.process.kill();
                return Err(CallFailure::Crashed(format!(
                    "no answer within {} s; the worker was stopped",
                    timeout.as_secs_f32().round()
                )));
            }
        }
    }
}

/// Ask the worker to quit, give it `grace` to unload, then kill it.
fn shut_down(link: Link, grace: Duration) {
    let _ = link.send(&Request::Shutdown, &[]);
    let deadline = Instant::now() + grace;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match link.rx.recv_timeout(left) {
            Ok(Incoming::Reply(_)) => continue,
            Ok(Incoming::Closed(_)) | Err(_) => break,
        }
    }
    link.process.kill();
}

impl Drop for WorkerEngine {
    fn drop(&mut self) {
        *lock(&self.cancel.writer) = None;
        if let Some(link) = self.link.take() {
            shut_down(link, SHUTDOWN_GRACE);
        }
    }
}

impl AsrEngine for WorkerEngine {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn transcribe(&mut self, audio: &[f32], options: &DecodeOptions) -> Result<AsrResult, AsrError> {
        let gen = self.cancel.gen.load(Ordering::SeqCst);
        let id = self.next_id();
        let reply = self.call(
            Request::Transcribe {
                id,
                options: options.clone(),
            },
            audio,
        )?;
        if self.cancelled_since(gen) {
            return Err(AsrError::cancelled());
        }
        match reply {
            Reply::Result { result, .. } => Ok(result),
            other => Err(AsrError::runtime(format!("unexpected worker reply {other:?}"))),
        }
    }

    fn begin_stream(&mut self, options: &DecodeOptions) -> Result<(), AsrError> {
        if !self.info.caps.streaming {
            return Err(AsrError::unsupported(format!("{} does not stream", self.label)));
        }
        let id = self.next_id();
        self.call(
            Request::BeginStream {
                id,
                options: options.clone(),
            },
            &[],
        )?;
        self.stream_samples = Some(0);
        Ok(())
    }

    fn push_stream(&mut self, audio: &[f32]) -> Result<Option<Hypothesis>, AsrError> {
        if self.stream_samples.is_none() || self.link.is_none() {
            return Err(AsrError::invalid_input("no stream is open (begin_stream first)"));
        }
        let gen = self.cancel.gen.load(Ordering::SeqCst);
        let id = self.next_id();
        let reply = self.call(Request::PushStream { id }, audio)?;
        if let Some(n) = self.stream_samples.as_mut() {
            *n = n.saturating_add(audio.len());
        }
        if self.cancelled_since(gen) {
            return Err(AsrError::cancelled());
        }
        match reply {
            Reply::Hypothesis { hypothesis, .. } => Ok(hypothesis),
            other => Err(AsrError::runtime(format!("unexpected worker reply {other:?}"))),
        }
    }

    fn end_stream(&mut self) -> Result<AsrResult, AsrError> {
        if self.stream_samples.is_none() || self.link.is_none() {
            return Err(AsrError::invalid_input("no stream is open"));
        }
        let gen = self.cancel.gen.load(Ordering::SeqCst);
        let id = self.next_id();
        let reply = self.call(Request::EndStream { id }, &[]);
        self.stream_samples = None;
        let reply = reply?;
        if self.cancelled_since(gen) {
            return Err(AsrError::cancelled());
        }
        match reply {
            Reply::Result { result, .. } => Ok(result),
            other => Err(AsrError::runtime(format!("unexpected worker reply {other:?}"))),
        }
    }

    fn abort_stream(&mut self) {
        if self.stream_samples.take().is_none() || self.link.is_none() {
            return;
        }
        let id = self.next_id();
        if let Err(e) = self.call(Request::AbortStream { id }, &[]) {
            tracing::debug!(target: LOG, "{}: abort_stream: {e}", self.label);
        }
    }

    fn canceller(&self) -> Option<Canceller> {
        let shared = Arc::clone(&self.cancel);
        Some(Canceller::new(move || {
            shared.gen.fetch_add(1, Ordering::SeqCst);
            let writer = lock(&shared.writer).clone();
            if let Some(w) = writer {
                let _ = write_frame(&mut **lock(&w), &Request::Cancel, &[]);
            }
        }))
    }
}

/// Whether `kind` is an error of the worker plumbing rather than the engine (for callers that
/// retry).
#[allow(dead_code)] // Diagnostics helper for P2-I0 / P7 (worker crash statistics).
pub(crate) fn is_worker_failure(e: &AsrError) -> bool {
    e.kind == AsrErrorKind::Runtime && e.message.contains("speech worker")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::speech::engine_api::{EngineCaps, EngineKind};
    use std::sync::atomic::AtomicUsize;

    // -----------------------------------------------------------------------------------------
    // In-memory pipes and a fake engine
    // -----------------------------------------------------------------------------------------

    struct PipeWriter(mpsc::Sender<Vec<u8>>);

    impl Write for PipeWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0
                .send(buf.to_vec())
                .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "closed"))?;
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    struct PipeReader {
        rx: mpsc::Receiver<Vec<u8>>,
        buf: Vec<u8>,
        pos: usize,
    }

    impl Read for PipeReader {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            if self.pos == self.buf.len() {
                match self.rx.recv() {
                    Ok(b) => {
                        self.buf = b;
                        self.pos = 0;
                    }
                    Err(_) => return Ok(0),
                }
            }
            let n = out.len().min(self.buf.len() - self.pos);
            out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    fn pipe() -> (PipeWriter, PipeReader) {
        let (tx, rx) = mpsc::channel();
        (
            PipeWriter(tx),
            PipeReader {
                rx,
                buf: Vec::new(),
                pos: 0,
            },
        )
    }

    /// Shared observations of the fake workers.
    #[derive(Default)]
    struct World {
        launches: AtomicUsize,
        specs: Mutex<Vec<String>>,
        dropped: AtomicUsize,
        killed: AtomicUsize,
        readmissions: AtomicUsize,
    }

    /// Fake engine: model names select behaviour (`gpu` / `cpu`, `crash-load`, `not-installed`).
    /// Audio of 13 samples panics (a crash), 17 samples blocks until cancelled or 5 s pass, 19
    /// samples sleeps 2 s (a hang for short deadlines), 23 samples moves the model to the CPU (the
    /// wrapper's VRAM-headroom guard), 29 samples too but the process keeps a 400 MB GPU context.
    struct Fake {
        info: EngineInfo,
        cancelled: Arc<AtomicBool>,
        stream: Option<usize>,
        world: Arc<World>,
    }

    impl Drop for Fake {
        fn drop(&mut self) {
            self.world.dropped.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl AsrEngine for Fake {
        fn info(&self) -> &EngineInfo {
            &self.info
        }
        fn transcribe(&mut self, audio: &[f32], options: &DecodeOptions) -> Result<AsrResult, AsrError> {
            match audio.len() {
                13 => panic!("simulated ggml abort"),
                17 => {
                    let t0 = Instant::now();
                    while !self.cancelled.swap(false, Ordering::SeqCst) {
                        if t0.elapsed() > Duration::from_secs(5) {
                            return Ok(AsrResult::default());
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    return Err(AsrError::cancelled());
                }
                19 => std::thread::sleep(Duration::from_secs(2)),
                23 | 29 => {
                    self.info.device = "cpu".into();
                    self.info.vram_mb = if audio.len() == 29 { 400 } else { 0 };
                    self.info.fallback_reason = Some("moved to the CPU: VRAM headroom".into());
                }
                _ => {}
            }
            Ok(AsrResult {
                text: format!("{} heard {} samples", self.info.device, audio.len()),
                lang: options.language.clone(),
                ..AsrResult::default()
            })
        }
        fn begin_stream(&mut self, _options: &DecodeOptions) -> Result<(), AsrError> {
            self.stream = Some(0);
            Ok(())
        }
        fn push_stream(&mut self, audio: &[f32]) -> Result<Option<Hypothesis>, AsrError> {
            let n = self.stream.as_mut().ok_or_else(|| AsrError::invalid_input("no stream"))?;
            *n += audio.len();
            Ok(Some(Hypothesis {
                text: format!("partial {n}"),
                lang: None,
            }))
        }
        fn end_stream(&mut self) -> Result<AsrResult, AsrError> {
            let n = self.stream.take().ok_or_else(|| AsrError::invalid_input("no stream"))?;
            Ok(AsrResult {
                text: format!("final {n}"),
                ..AsrResult::default()
            })
        }
        fn abort_stream(&mut self) {
            self.stream = None;
        }
        fn canceller(&self) -> Option<Canceller> {
            Some(Canceller::from_flag(Arc::clone(&self.cancelled)))
        }
    }

    fn fake_factory(world: Arc<World>) -> EngineFactory {
        Arc::new(move |spec: &EngineSpec, ctx: &EngineContext| {
            lock(&world.specs).push(spec.model.clone());
            if spec.model.contains("not-installed") {
                return Err(AsrError::not_installed("stt/fake: the model is not installed"));
            }
            if spec.model.contains("crash-load") && spec.model.contains("gpu") {
                panic!("simulated abort while loading");
            }
            let gpu = spec.model.contains("gpu");
            ctx.status.report(EngineStatus {
                code: codes::LOADING_MODEL.into(),
                detail: format!("Loading {}", spec.model),
                ..EngineStatus::default()
            });
            Ok(Box::new(Fake {
                info: EngineInfo {
                    kind: EngineKind::Transcribe,
                    model: spec.model.clone(),
                    runtime: "fake 1.0".into(),
                    device: if gpu { "cuda".into() } else { "cpu".into() },
                    caps: EngineCaps {
                        streaming: true,
                        utterance: true,
                        ..EngineCaps::default()
                    },
                    vram_mb: if gpu { 3470 } else { 0 },
                    fallback_reason: None,
                },
                cancelled: Arc::new(AtomicBool::new(false)),
                stream: None,
                world: Arc::clone(&world),
            }) as Box<dyn AsrEngine>)
        })
    }

    struct ThreadProcess {
        world: Arc<World>,
        done: Arc<AtomicBool>,
    }

    impl WorkerProcess for ThreadProcess {
        fn kill(&self) {
            self.world.killed.fetch_add(1, Ordering::SeqCst);
        }
        fn exit_description(&self, wait: Duration) -> String {
            let deadline = Instant::now() + wait;
            while !self.done.load(Ordering::SeqCst) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            if self.done.load(Ordering::SeqCst) {
                "the test worker ended".into()
            } else {
                "the test worker is still running".into()
            }
        }
    }

    /// Runs [`serve`] on a thread over in-memory pipes (a panic there is a "crash").
    struct ThreadLauncher {
        world: Arc<World>,
    }

    impl Launcher for ThreadLauncher {
        fn launch(&self, name: &str, _memory_limit_mb: Option<u64>) -> Result<Launched, String> {
            assert!(crate::services::process_host::spawn::validate_name(name).is_ok(), "{name}");
            self.world.launches.fetch_add(1, Ordering::SeqCst);
            let (req_w, req_r) = pipe();
            let (rep_w, rep_r) = pipe();
            let done = Arc::new(AtomicBool::new(false));
            let factory = fake_factory(Arc::clone(&self.world));
            let thread_done = Arc::clone(&done);
            std::thread::spawn(move || {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| serve(req_r, rep_w, None, factory)));
                thread_done.store(true, Ordering::SeqCst);
                drop(r);
            });
            Ok(Launched {
                writer: Box::new(req_w),
                reader: Box::new(rep_r),
                process: Box::new(ThreadProcess {
                    world: Arc::clone(&self.world),
                    done,
                }),
            })
        }
    }

    fn spec(model: &str) -> EngineSpec {
        EngineSpec::new(EngineKind::Transcribe, model)
    }

    fn status_log() -> (EngineContext, Arc<Mutex<Vec<EngineStatus>>>) {
        let seen: Arc<Mutex<Vec<EngineStatus>>> = Arc::default();
        let sink = Arc::clone(&seen);
        (
            EngineContext {
                status: StatusSink::new(move |s| lock(&sink).push(s)),
            },
            seen,
        )
    }

    /// A setup; a `gpu` model gets a governor that admits it again on every restart (no claims).
    fn setup(model: &str, cpu: Option<&str>, ctx: EngineContext, claims: GpuClaims) -> WorkerSetup {
        let readmit = model.contains("gpu").then(|| {
            let model = model.to_string();
            Box::new(move || Readmission::Gpu {
                spec: spec(&model),
                claims: GpuClaims::none(),
            }) as Readmit
        });
        WorkerSetup {
            spec: spec(model),
            readmit,
            cpu_spec: cpu.map(spec),
            ctx,
            claims,
            memory_limit_mb: Some(4096),
            label: model.to_string(),
        }
    }

    /// A governor whose `n`-th decision (from 1) is `script(n)`, counted in `world.readmissions`.
    fn scripted(world: &Arc<World>, mut script: impl FnMut(usize) -> Readmission + Send + 'static) -> Option<Readmit> {
        let world = Arc::clone(world);
        Some(Box::new(move || {
            let n = world.readmissions.fetch_add(1, Ordering::SeqCst) + 1;
            script(n)
        }))
    }

    /// The ledger's claims labelled `label` (MB).
    fn ledger(label: &str) -> Vec<u64> {
        crate::services::resources::snapshot()
            .claims
            .into_iter()
            .filter(|c| c.label == label)
            .map(|c| c.mb)
            .collect()
    }

    // -----------------------------------------------------------------------------------------
    // Frames
    // -----------------------------------------------------------------------------------------

    #[test]
    fn frames_round_trip_and_resync_after_stray_output() {
        let request = Request::Transcribe {
            id: 7,
            options: DecodeOptions {
                language: Some("ja".into()),
                ..DecodeOptions::default()
            },
        };
        let audio = [0.5f32, -0.25, 1.0];
        let mut bytes = b"ggml_cuda_init: found 1 CUDA device\nSGW".to_vec();
        bytes.extend(encode_frame(&request, &audio).unwrap());
        bytes.extend(b"\r\n");
        bytes.extend(encode_frame(&Request::Shutdown, &[]).unwrap());
        bytes.extend(b"SG");
        let mut r = BufReader::new(&bytes[..]);
        let mut junk = Vec::new();
        let f = read_frame(&mut r, &mut junk).unwrap().unwrap();
        assert_eq!(junk_text(&junk), "ggml_cuda_init: found 1 CUDA device SGW");
        assert_eq!(serde_json::from_slice::<Request>(&f.header).unwrap(), request);
        assert_eq!(f.samples(), audio);
        junk.clear();
        let f = read_frame(&mut r, &mut junk).unwrap().unwrap();
        assert_eq!(serde_json::from_slice::<Request>(&f.header).unwrap(), Request::Shutdown);
        assert!(f.payload.is_empty());
        assert_eq!(junk, b"\r\n");
        junk.clear();
        assert!(read_frame(&mut r, &mut junk).unwrap().is_none(), "clean end");
        assert_eq!(junk, b"SG", "a partial magic at the end is junk");

        // Wire names are the module's contract with older / newer workers of the same build.
        let v: serde_json::Value = serde_json::from_slice(&encode_frame(&request, &[]).unwrap()[12..]).unwrap();
        assert_eq!(v["op"], "transcribe");
        assert_eq!(v["options"]["language"], "ja");
    }

    #[test]
    fn broken_frames_are_errors_not_hangs() {
        // Truncated payload.
        let mut bytes = encode_frame(&Request::PushStream { id: 1 }, &[1.0, 2.0]).unwrap();
        bytes.truncate(bytes.len() - 3);
        let err = read_frame(&mut BufReader::new(&bytes[..]), &mut Vec::new()).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
        // Lengths out of range (a corrupt or hostile stream never allocates gigabytes).
        let mut bad = MAGIC.to_vec();
        bad.extend((u32::MAX).to_le_bytes());
        bad.extend(0u32.to_le_bytes());
        assert_eq!(read_frame(&mut &bad[..], &mut Vec::new()).unwrap_err().kind(), io::ErrorKind::InvalidData);
        let mut odd = MAGIC.to_vec();
        odd.extend(2u32.to_le_bytes());
        odd.extend(3u32.to_le_bytes());
        odd.extend(b"{}xyz");
        assert_eq!(read_frame(&mut &odd[..], &mut Vec::new()).unwrap_err().kind(), io::ErrorKind::InvalidData);
        // Oversized payloads are refused when encoding.
        let huge = vec![0f32; MAX_PAYLOAD_BYTES / 4 + 1];
        assert_eq!(encode_frame(&Request::PushStream { id: 1 }, &huge).unwrap_err().kind(), io::ErrorKind::InvalidInput);
        // Junk is capped.
        let noise = vec![b'x'; 10_000];
        let mut junk = Vec::new();
        assert!(read_frame(&mut BufReader::new(&noise[..]), &mut junk).unwrap().is_none());
        assert_eq!(junk.len(), MAX_JUNK_LOGGED);
    }

    #[test]
    fn worker_names_are_valid_process_host_names() {
        for label in ["stt/qwen3-asr-1.7b-q8_0", "Model.GGUF", "", "日本語", "-x-"] {
            let name = worker_name(label);
            assert!(crate::services::process_host::spawn::validate_name(&name).is_ok(), "{label} -> {name}");
            assert!(name.starts_with("stt-worker-"));
        }
        assert!(worker_name("stt/qwen3-asr-1.7b-q8_0").starts_with("stt-worker-stt-qwen3-asr-1.7b-q8_0-"));
    }

    #[test]
    fn call_deadlines_grow_with_the_audio() {
        assert_eq!(call_timeout(0), CALL_TIMEOUT);
        assert_eq!(call_timeout(SAMPLE_RATE as usize * 20), CALL_TIMEOUT + CALL_TIMEOUT_PER_AUDIO_SECOND * 20);
    }

    // -----------------------------------------------------------------------------------------
    // Worker engine over the protocol
    // -----------------------------------------------------------------------------------------

    #[test]
    fn a_worker_engine_decodes_streams_and_forwards_status() {
        let world = Arc::new(World::default());
        let (ctx, seen) = status_log();
        let launcher = Arc::new(ThreadLauncher { world: Arc::clone(&world) });
        let mut e = WorkerEngine::spawn_with(setup("stt/fake-gpu", Some("stt/fake-cpu"), ctx, GpuClaims::none()), launcher).unwrap();
        assert_eq!(world.launches.load(Ordering::SeqCst), 1);
        assert_eq!(e.info().device, "cuda");
        assert_eq!(e.info().runtime, "fake 1.0 (worker process)");
        assert!(e.info().caps.streaming);
        assert!(lock(&seen).iter().any(|s| s.code == codes::LOADING_MODEL && s.detail.contains("stt/fake-gpu")));

        let options = DecodeOptions {
            language: Some("ja".into()),
            ..DecodeOptions::default()
        };
        let r = e.transcribe(&[0.0; 1600], &options).unwrap();
        assert_eq!((r.text.as_str(), r.lang.as_deref()), ("cuda heard 1600 samples", Some("ja")));

        e.begin_stream(&DecodeOptions::default()).unwrap();
        assert_eq!(e.push_stream(&[0.0; 160]).unwrap().unwrap().text, "partial 160");
        assert_eq!(e.push_stream(&[0.0; 40]).unwrap().unwrap().text, "partial 200");
        assert_eq!(e.end_stream().unwrap().text, "final 200");
        assert_eq!(e.end_stream().unwrap_err().kind, AsrErrorKind::InvalidInput);
        e.begin_stream(&DecodeOptions::default()).unwrap();
        e.abort_stream();
        assert_eq!(e.push_stream(&[0.0; 4]).unwrap_err().kind, AsrErrorKind::InvalidInput);

        // Dropping the engine shuts the worker down cleanly (the hosted engine is dropped).
        drop(e);
        let t0 = Instant::now();
        while world.dropped.load(Ordering::SeqCst) == 0 && t0.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(world.dropped.load(Ordering::SeqCst), 1);
        assert_eq!(world.launches.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn load_errors_pass_through_without_restarts() {
        let world = Arc::new(World::default());
        let launcher = Arc::new(ThreadLauncher { world: Arc::clone(&world) });
        let err = WorkerEngine::spawn_with(setup("stt/not-installed", None, EngineContext::headless(), GpuClaims::none()), launcher).unwrap_err();
        assert_eq!(err.kind, AsrErrorKind::NotInstalled);
        assert!(err.message.starts_with("stt/fake"), "{err}");
        assert_eq!(world.launches.load(Ordering::SeqCst), 1, "not a crash: no restart");
    }

    #[test]
    fn crashes_restart_the_worker_then_move_it_to_the_cpu() {
        let world = Arc::new(World::default());
        let (ctx, seen) = status_log();
        let launcher = Arc::new(ThreadLauncher { world: Arc::clone(&world) });
        const LABEL: &str = "test: worker_proc crash claims";
        let claims = GpuClaims::test_external(LABEL, 3900);
        let mut s = setup("stt/fake-gpu", Some("stt/fake-cpu"), ctx, claims);
        s.readmit = scripted(&world, |_| Readmission::Gpu {
            spec: spec("stt/fake-gpu"),
            claims: GpuClaims::test_external(LABEL, 3900),
        });
        let mut e = WorkerEngine::spawn_with(s, launcher).unwrap();
        assert!(e.holds_claims());
        assert_eq!(ledger(LABEL), vec![3470], "resized to the child's measured process VRAM");
        let opts = DecodeOptions::default();

        // Crash 1: the utterance fails and the claim goes with the process; the next utterance
        // restarts the GPU worker on a fresh admission.
        let err = e.transcribe(&[0.0; 13], &opts).unwrap_err();
        assert_eq!(err.kind, AsrErrorKind::Runtime);
        assert!(err.message.contains("speech worker process ended"), "{err}");
        assert!(is_worker_failure(&err));
        assert_eq!(e.crash_counts(), (1, 1));
        assert!(!e.holds_claims() && ledger(LABEL).is_empty());
        assert_eq!(e.transcribe(&[0.0; 8], &opts).unwrap().text, "cuda heard 8 samples");
        assert_eq!(world.launches.load(Ordering::SeqCst), 2);
        assert_eq!(world.readmissions.load(Ordering::SeqCst), 1);
        assert!(e.holds_claims());
        assert_eq!(ledger(LABEL), vec![3470]);

        // Crash 2: the restart uses the CPU spec (no admission), releases the GPU claims and says
        // why.
        assert!(e.transcribe(&[0.0; 13], &opts).is_err());
        assert_eq!(e.transcribe(&[0.0; 8], &opts).unwrap().text, "cpu heard 8 samples");
        assert_eq!(world.launches.load(Ordering::SeqCst), 3);
        assert_eq!(world.readmissions.load(Ordering::SeqCst), 1);
        assert!(ledger(LABEL).is_empty());
        assert_eq!(lock(&world.specs).last().map(String::as_str), Some("stt/fake-cpu"));
        assert!(!e.holds_claims());
        assert_eq!(e.info().device, "cpu");
        assert_eq!(e.info().vram_mb, 0);
        let reason = e.info().fallback_reason.clone().unwrap();
        assert!(reason.contains("crashed 2 times"), "{reason}");
        assert!(lock(&seen)
            .iter()
            .any(|s| s.code == codes::ENGINE_FALLBACK && s.level == StatusLevel::Warning));

        // Crashes on the CPU spec restart it until the budget is gone.
        for _ in 0..(MAX_CRASHES - 2) {
            assert!(e.transcribe(&[0.0; 13], &opts).is_err());
        }
        let err = e.transcribe(&[0.0; 8], &opts).unwrap_err();
        assert!(err.message.contains("not restarted again"), "{err}");
        // The third worker (CPU) served one call and crashed; each later crash cost one launch.
        assert_eq!(world.launches.load(Ordering::SeqCst), 3 + (MAX_CRASHES as usize - 3));
    }

    #[test]
    fn a_crash_while_loading_on_the_gpu_goes_to_the_cpu_at_once() {
        let world = Arc::new(World::default());
        let (ctx, seen) = status_log();
        let launcher = Arc::new(ThreadLauncher { world: Arc::clone(&world) });
        let e = WorkerEngine::spawn_with(setup("stt/crash-load-gpu", Some("stt/fake-cpu"), ctx, GpuClaims::none()), launcher).unwrap();
        assert_eq!(world.launches.load(Ordering::SeqCst), 2);
        assert_eq!(e.info().device, "cpu");
        assert!(e
            .info()
            .fallback_reason
            .as_deref()
            .is_some_and(|r| r.contains("crashed while loading")));
        assert!(lock(&seen).iter().any(|s| s.code == codes::ENGINE_FALLBACK));
        // Without a CPU spec the load crash is an error.
        let launcher = Arc::new(ThreadLauncher { world: Arc::clone(&world) });
        let err = WorkerEngine::spawn_with(setup("stt/crash-load-gpu", None, EngineContext::headless(), GpuClaims::none()), launcher).unwrap_err();
        assert_eq!(err.kind, AsrErrorKind::Runtime);
        assert!(err.message.contains("failed to start"), "{err}");
    }

    /// Review P2-D: a GPU worker whose model moved to the CPU (so its claim was released) and
    /// then crashed is never restarted on the GPU without a fresh admission and claim.
    #[test]
    fn a_restart_runs_the_governor_again_after_the_claim_was_released() {
        const LABEL: &str = "test: worker_proc readmit after cpu";
        let world = Arc::new(World::default());
        let (ctx, _) = status_log();
        let launcher = Arc::new(ThreadLauncher { world: Arc::clone(&world) });
        let mut s = setup("stt/fake-gpu", Some("stt/fake-cpu"), ctx, GpuClaims::test_external(LABEL, 3500));
        s.readmit = scripted(&world, |_| Readmission::Gpu {
            spec: spec("stt/fake-gpu"),
            claims: GpuClaims::test_external(LABEL, 3500),
        });
        let mut e = WorkerEngine::spawn_with(s, launcher).unwrap();
        let opts = DecodeOptions::default();
        assert_eq!(ledger(LABEL), vec![3470]);

        // The headroom guard moves the model to the CPU inside the worker: the claim goes.
        e.transcribe(&[0.0; 23], &opts).unwrap();
        assert_eq!((e.info().device.as_str(), e.info().vram_mb), ("cpu", 0));
        assert!(!e.holds_claims() && ledger(LABEL).is_empty());

        // The worker dies; the restart from the GPU spec asks the governor first.
        assert!(e.transcribe(&[0.0; 13], &opts).is_err());
        assert_eq!(e.crash_counts(), (1, 1));
        assert_eq!(e.transcribe(&[0.0; 8], &opts).unwrap().text, "cuda heard 8 samples");
        assert_eq!(world.readmissions.load(Ordering::SeqCst), 1);
        assert!(e.holds_claims());
        assert_eq!(ledger(LABEL), vec![3470], "the GPU worker is on the ledger again");
    }

    /// A restart the governor keeps off the GPU runs the CPU for now (and says so); without the
    /// CPU fallback it fails without starting a process; a later restart may get the GPU back.
    #[test]
    fn a_governor_that_says_no_keeps_the_restart_off_the_gpu() {
        let world = Arc::new(World::default());
        let (ctx, seen) = status_log();
        let launcher = Arc::new(ThreadLauncher { world: Arc::clone(&world) });
        let mut s = setup("stt/fake-gpu", Some("stt/fake-cpu"), ctx, GpuClaims::none());
        s.readmit = scripted(&world, |n| match n {
            1 => Readmission::Cpu {
                spec: spec("stt/fake-cpu-for-now"),
                reason: "fake runs on the CPU: the voice changer is using the GPU".into(),
            },
            2 => Readmission::Refused(AsrError::new(AsrErrorKind::OutOfMemory, "fake: the GPU is full (the CPU fallback is off)")),
            _ => Readmission::Gpu {
                spec: spec("stt/fake-gpu"),
                claims: GpuClaims::none(),
            },
        });
        let mut e = WorkerEngine::spawn_with(s, launcher).unwrap();
        let opts = DecodeOptions::default();

        assert!(e.transcribe(&[0.0; 13], &opts).is_err());
        assert_eq!(e.transcribe(&[0.0; 8], &opts).unwrap().text, "cpu heard 8 samples");
        assert_eq!(lock(&world.specs).last().map(String::as_str), Some("stt/fake-cpu-for-now"));
        assert!(lock(&seen)
            .iter()
            .any(|s| s.code == codes::ENGINE_FALLBACK && s.detail.contains("voice changer")));
        // A crash of that CPU worker is not a GPU crash.
        assert!(e.transcribe(&[0.0; 13], &opts).is_err());
        assert_eq!(e.crash_counts(), (2, 1));

        // Refused: the call fails at once, no process starts.
        let launches = world.launches.load(Ordering::SeqCst);
        let err = e.transcribe(&[0.0; 8], &opts).unwrap_err();
        assert_eq!(err.kind, AsrErrorKind::OutOfMemory, "{err}");
        assert_eq!(world.launches.load(Ordering::SeqCst), launches);
        assert_eq!(e.crash_counts(), (2, 1), "a refusal is not a crash");

        // The card has room again.
        assert_eq!(e.transcribe(&[0.0; 8], &opts).unwrap().text, "cuda heard 8 samples");
        assert_eq!(world.readmissions.load(Ordering::SeqCst), 3);
    }

    /// Review P2-D: with `cpuFallback: false` there is no CPU spec, so repeated GPU crashes end
    /// the worker instead of starting a process per utterance that refuses to load.
    #[test]
    fn without_the_cpu_fallback_repeated_gpu_crashes_end_the_worker() {
        let world = Arc::new(World::default());
        let launcher = Arc::new(ThreadLauncher { world: Arc::clone(&world) });
        let mut e = WorkerEngine::spawn_with(setup("stt/fake-gpu", None, EngineContext::headless(), GpuClaims::none()), launcher).unwrap();
        let opts = DecodeOptions::default();
        for _ in 0..MAX_GPU_CRASHES {
            assert!(e.transcribe(&[0.0; 13], &opts).is_err());
        }
        assert_eq!(e.crash_counts(), (MAX_GPU_CRASHES, MAX_GPU_CRASHES));
        let launches = world.launches.load(Ordering::SeqCst);
        for _ in 0..5 {
            let err = e.transcribe(&[0.0; 8], &opts).unwrap_err();
            assert!(err.message.contains("CPU fallback is off"), "{err}");
        }
        assert_eq!(world.launches.load(Ordering::SeqCst), launches, "no more workers");
    }

    /// A restarted worker whose engine refuses to load is not restarted on every call forever.
    #[test]
    fn a_restart_that_fails_to_load_counts_toward_the_crash_budget() {
        let world = Arc::new(World::default());
        let launcher = Arc::new(ThreadLauncher { world: Arc::clone(&world) });
        let mut s = setup("stt/fake-gpu", None, EngineContext::headless(), GpuClaims::none());
        s.readmit = scripted(&world, |_| Readmission::Gpu {
            spec: spec("stt/not-installed-gpu"),
            claims: GpuClaims::none(),
        });
        let mut e = WorkerEngine::spawn_with(s, launcher).unwrap();
        let opts = DecodeOptions::default();
        assert!(e.transcribe(&[0.0; 13], &opts).is_err());
        let mut last = None;
        for _ in 0..10 {
            last = Some(e.transcribe(&[0.0; 8], &opts).unwrap_err());
        }
        // One launch per failed restart until the budget is gone (the first start plus
        // MAX_CRASHES - 1 restarts), then none.
        assert_eq!(world.launches.load(Ordering::SeqCst), MAX_CRASHES as usize);
        let last = last.unwrap();
        assert!(last.message.contains("not restarted again") && last.message.contains("not installed"), "{last}");
    }

    /// A child on the CPU that still holds a GPU context (its GPU runtime was initialized) keeps a
    /// claim of that size; the engine's own info says 0 as on any CPU.
    #[test]
    fn a_worker_on_the_cpu_keeps_claiming_the_context_it_still_holds() {
        const LABEL: &str = "test: worker_proc leftover context";
        let world = Arc::new(World::default());
        let launcher = Arc::new(ThreadLauncher { world: Arc::clone(&world) });
        let s = setup("stt/fake-gpu", Some("stt/fake-cpu"), EngineContext::headless(), GpuClaims::test_external(LABEL, 3500));
        let mut e = WorkerEngine::spawn_with(s, launcher).unwrap();
        e.transcribe(&[0.0; 29], &DecodeOptions::default()).unwrap();
        assert_eq!((e.info().device.as_str(), e.info().vram_mb), ("cpu", 0));
        assert!(e.holds_claims());
        assert_eq!(ledger(LABEL), vec![400]);
        assert!(e.info().fallback_reason.is_some());
        drop(e);
        assert!(ledger(LABEL).is_empty());
    }

    #[test]
    fn cancel_reaches_the_running_call_in_the_worker() {
        let world = Arc::new(World::default());
        let launcher = Arc::new(ThreadLauncher { world: Arc::clone(&world) });
        let mut e = WorkerEngine::spawn_with(setup("stt/fake-gpu", None, EngineContext::headless(), GpuClaims::none()), launcher).unwrap();
        let canceller = e.canceller().unwrap();
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            canceller.cancel();
        });
        let t0 = Instant::now();
        let err = e.transcribe(&[0.0; 17], &DecodeOptions::default()).unwrap_err();
        t.join().unwrap();
        assert!(err.is_cancelled(), "{err}");
        assert!(t0.elapsed() < Duration::from_secs(3), "the cancel interrupted the call");
        // The next call is not affected.
        assert_eq!(e.transcribe(&[0.0; 5], &DecodeOptions::default()).unwrap().text, "cuda heard 5 samples");
    }

    #[test]
    fn a_silent_worker_is_killed_and_restarted() {
        let world = Arc::new(World::default());
        let launcher = Arc::new(ThreadLauncher { world: Arc::clone(&world) });
        let mut e = WorkerEngine::spawn_with(setup("stt/fake-cpu", None, EngineContext::headless(), GpuClaims::none()), launcher).unwrap();
        e.set_timeout(Duration::from_millis(300));
        let err = e.transcribe(&[0.0; 19], &DecodeOptions::default()).unwrap_err();
        assert!(err.message.contains("no answer within"), "{err}");
        assert_eq!(world.killed.load(Ordering::SeqCst), 1);
        assert_eq!(e.crash_counts(), (1, 0));
        e.set_timeout(Duration::from_secs(10));
        assert_eq!(e.transcribe(&[0.0; 3], &DecodeOptions::default()).unwrap().text, "cpu heard 3 samples");
        assert_eq!(world.launches.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn requests_before_init_are_refused_and_eof_ends_the_worker() {
        let (req_w, req_r) = pipe();
        let (rep_w, rep_r) = pipe();
        let world = Arc::new(World::default());
        let factory = fake_factory(Arc::clone(&world));
        let server = std::thread::spawn(move || serve(req_r, rep_w, None, factory));
        let mut w = req_w;
        write_frame(&mut w, &Request::EndStream { id: 4 }, &[]).unwrap();
        let mut r = BufReader::new(rep_r);
        let f = read_frame(&mut r, &mut Vec::new()).unwrap().unwrap();
        match serde_json::from_slice::<Reply>(&f.header).unwrap() {
            Reply::Error { id, error, .. } => {
                assert_eq!(id, Some(4));
                assert_eq!(error.kind, AsrErrorKind::InvalidInput);
            }
            other => panic!("{other:?}"),
        }
        drop(w);
        assert!(server.join().unwrap().is_ok());
        assert!(lock(&world.specs).is_empty());
    }

    // -----------------------------------------------------------------------------------------
    // A real worker process (Sigil.exe)
    // -----------------------------------------------------------------------------------------

    /// `cargo test ... worker_proc -- --ignored` with `SIGIL_STT_WORKER_EXE` = a debug `sigil.exe`
    /// built from this tree (default `target/debug/sigil.exe` next to the test binary's folder):
    /// the real process host starts `--headless stt-worker`, the protocol runs over its pipes, and
    /// a load error (a model file that does not exist) comes back as the engine's error.
    #[test]
    #[ignore = "needs a debug sigil.exe built from this tree (SIGIL_STT_WORKER_EXE)"]
    fn a_real_worker_process_answers_over_its_pipes() {
        if std::env::var_os(WORKER_EXE_ENV).is_none() {
            let exe = std::env::current_exe().unwrap();
            let sigil = exe.parent().and_then(Path::parent).map(|d| d.join("sigil.exe"));
            match sigil.filter(|p| p.is_file()) {
                Some(p) => std::env::set_var(WORKER_EXE_ENV, p),
                None => panic!("build sigil.exe first (cargo build) or set {WORKER_EXE_ENV}"),
            }
        }
        let missing = std::env::temp_dir().join("sigil-worker-test-missing.gguf");
        let err = WorkerEngine::spawn(WorkerSetup {
            spec: EngineSpec::new(EngineKind::Transcribe, missing.to_string_lossy()),
            readmit: None,
            cpu_spec: None,
            ctx: EngineContext::headless(),
            claims: GpuClaims::none(),
            memory_limit_mb: Some(2048),
            label: "missing-model".into(),
        })
        .unwrap_err();
        assert_eq!(err.kind, AsrErrorKind::InvalidInput, "{err}");
        assert!(err.message.contains("does not exist"), "{err}");
    }
}
