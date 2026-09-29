//! One running helper process: launched in its job, stdout/stderr pumped into the helper's log
//! book (in [`StdioMode::Protocol`] stdout is kept for the client instead), and a waiter thread that
//! reaps it, kills whatever is left in its job (the main process defines the helper's lifetime) and
//! publishes the exit.

use super::health::{self, ExitInfo};
use super::job::{self, Job, JobLimits};
use super::logs::{self, LogBook, LogStream, PumpGate};
use super::spawn::{self, Launch};
use super::{HelperSpec, StdioMode};
use std::io::Write;
use std::os::windows::io::{AsHandle, AsRawHandle, OwnedHandle};
use std::process::{ChildStdin, ChildStdout};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, TryLockError};
use std::time::{Duration, Instant};
use tokio::sync::watch;

/// Exit code given to processes Sigil terminates.
pub const KILL_EXIT_CODE: u32 = 1;
/// How long the waiter lets the pumps drain the pipes after the process exited.
const DRAIN_TIMEOUT: Duration = Duration::from_millis(500);
/// While a graceful stop waits, how often it retries closing a stdin pipe that a client write holds.
const STDIN_CLOSE_RETRY: Duration = Duration::from_millis(10);

/// The helper's stdin, shared by the host (which closes it to stop the helper) and the
/// [`HelperStdin`] writers of a protocol client.
#[derive(Debug)]
struct StdinShared {
    /// `None` once closed (EOF for the helper).
    pipe: Mutex<Option<ChildStdin>>,
    /// Set by a stop: writers refuse new writes and the first one to get the lock closes the pipe,
    /// so EOF reaches the helper right after an in-flight write even when a client writes in a loop.
    closing: AtomicBool,
}

impl StdinShared {
    fn new(pipe: Option<ChildStdin>) -> Self {
        Self {
            pipe: Mutex::new(pipe),
            closing: AtomicBool::new(false),
        }
    }

    /// Mark the pipe as closing and close it unless a writer holds it right now (that writer, or a
    /// later call, closes it then). Never blocks. True when the pipe is closed.
    fn close(&self) -> bool {
        self.closing.store(true, Ordering::SeqCst);
        let mut guard = match self.pipe.try_lock() {
            Ok(g) => g,
            Err(TryLockError::Poisoned(p)) => p.into_inner(),
            Err(TryLockError::WouldBlock) => return false,
        };
        drop(guard.take());
        true
    }
}

pub struct Instance {
    pub pid: u32,
    pub port: Option<u16>,
    /// When the host declared it ready (health check passed, exposure audit clean). The restart
    /// budget resets only after the helper has *run ready* for a while, never because of a long load
    /// (the launch time is deliberately not kept: it must not be used for that decision).
    ready_at: OnceLock<Instant>,
    job: Arc<Job>,
    process: OwnedHandle,
    stdin: Arc<StdinShared>,
    /// Protocol mode only, until a client takes it.
    stdout: Mutex<Option<ChildStdout>>,
    exit_rx: watch::Receiver<Option<ExitInfo>>,
    killed: Arc<AtomicBool>,
}

impl std::fmt::Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            .field("pid", &self.pid)
            .field("port", &self.port)
            .finish_non_exhaustive()
    }
}

impl Instance {
    /// Launch `spec` with the chosen port and key. Blocking (CreateProcess, thread spawns).
    pub fn start(spec: &HelperSpec, port: Option<u16>, api_key: Option<&str>, book: &Arc<LogBook>) -> Result<Instance, String> {
        let args = spec
            .args
            .iter()
            .map(|a| spawn::substitute(a, port, api_key))
            .collect();
        let env = spec
            .env
            .iter()
            .map(|(k, v)| (k.clone(), spawn::substitute(v, port, api_key)))
            .collect();
        let launched = spawn::launch(&Launch {
            name: &spec.name,
            program: &spec.program,
            args,
            env,
            env_remove: &spec.env_remove,
            cwd: spec.cwd.clone(),
            limits: JobLimits {
                memory_limit_mb: spec.memory_limit_mb,
                priority: spec.priority,
            },
            high_qos: spec.high_qos,
        })?;
        let spawn::Launched { mut child, job, pid } = launched;
        let job = Arc::new(job);
        let fail = |job: &Job, child: &mut std::process::Child, msg: String| -> String {
            job.terminate(KILL_EXIT_CODE);
            let _ = child.wait();
            format!("{}: {msg}", spec.name)
        };

        let process = match child.as_handle().try_clone_to_owned() {
            Ok(h) => h,
            Err(e) => return Err(fail(&job, &mut child, format!("cannot duplicate the process handle: {e}"))),
        };
        let gate = Arc::new(PumpGate::default());
        let mut protocol_stdout = None;
        if let Some(out) = child.stdout.take() {
            if spec.stdio == StdioMode::Protocol {
                protocol_stdout = Some(out);
            } else if let Err(e) = logs::spawn_pump(book.clone(), LogStream::Stdout, out, gate.clone()) {
                return Err(fail(&job, &mut child, format!("cannot start the stdout reader: {e}")));
            }
        }
        if let Some(err) = child.stderr.take() {
            if let Err(e) = logs::spawn_pump(book.clone(), LogStream::Stderr, err, gate.clone()) {
                return Err(fail(&job, &mut child, format!("cannot start the stderr reader: {e}")));
            }
        }
        let stdin = child.stdin.take();

        let (exit_tx, exit_rx) = watch::channel(None);
        let killed = Arc::new(AtomicBool::new(false));
        let waiter_job = job.clone();
        let waiter_killed = killed.clone();
        let waiter_book = book.clone();
        let memory_limit_mb = spec.memory_limit_mb;
        let (waiter_child, spawned) = {
            // The child moves into the waiter; keep a way to kill it if the thread cannot start.
            let child_cell = Arc::new(Mutex::new(Some(child)));
            let cell = child_cell.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("helper-{}-wait", spec.name))
                .spawn(move || {
                    let child = cell.lock().unwrap_or_else(PoisonError::into_inner).take();
                    let code = child.and_then(|mut c| c.wait().ok()).and_then(|s| s.code());
                    // Children of the helper die with it.
                    waiter_job.terminate(KILL_EXIT_CODE);
                    gate.wait_idle(DRAIN_TIMEOUT);
                    let events = waiter_job.take_events();
                    let stats = waiter_job.stats();
                    let info = ExitInfo {
                        code,
                        killed_by_host: waiter_killed.load(Ordering::SeqCst),
                        memory_limit_hit: events.memory_limit_hit,
                        memory_limit_mb,
                        peak_memory_mb: stats.peak_job_memory_bytes / (1024 * 1024),
                    };
                    waiter_book.note(&format!("process {pid} ended: {info}"));
                    let _ = exit_tx.send(Some(info));
                });
            (child_cell, spawned)
        };
        if let Err(e) = spawned {
            let mut guard = waiter_child.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(mut c) = guard.take() {
                return Err(fail(&job, &mut c, format!("cannot start the exit watcher: {e}")));
            }
            return Err(format!("{}: cannot start the exit watcher: {e}", spec.name));
        }

        Ok(Instance {
            pid,
            port,
            ready_at: OnceLock::new(),
            job,
            process,
            stdin: Arc::new(StdinShared::new(stdin)),
            stdout: Mutex::new(protocol_stdout),
            exit_rx,
            killed,
        })
    }

    /// Processes currently in the helper's job (the helper and anything it started).
    pub fn process_ids(&self) -> Vec<u32> {
        let mut ids = self.job.process_ids();
        if ids.is_empty() && !self.has_exited() {
            ids.push(self.pid);
        }
        ids
    }

    pub fn exit_rx(&self) -> watch::Receiver<Option<ExitInfo>> {
        self.exit_rx.clone()
    }

    pub fn has_exited(&self) -> bool {
        self.exit_rx.borrow().is_some()
    }

    /// Record that the host declared this instance ready (the first call wins).
    pub fn mark_ready(&self) {
        let _ = self.ready_at.set(Instant::now());
    }

    /// How long the instance has been ready (`None` if it never became ready).
    pub fn ready_for(&self) -> Option<Duration> {
        self.ready_at.get().map(Instant::elapsed)
    }

    /// Close the helper's stdin: helpers that read stdin treat EOF as "quit" (graceful stop). New
    /// client writes are refused from now on. Never blocks: when a client write holds the pipe, the
    /// pipe is closed right after that write (by the next writer, or by [`Instance::stop`]'s
    /// retries); a write stuck on a full pipe is ended by the stop's kill. True when it is closed.
    pub fn close_stdin(&self) -> bool {
        self.stdin.close()
    }

    /// Hand the protocol pipes to a client (once per instance; `None` in log mode or when taken).
    pub fn take_stdio(&self) -> Option<HelperStdio> {
        let stdout = self
            .stdout
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()?;
        Some(HelperStdio {
            pid: self.pid,
            stdin: HelperStdin { shared: self.stdin.clone() },
            stdout,
        })
    }

    /// Terminate the helper and everything it started. Idempotent.
    pub fn kill(&self) {
        self.killed.store(true, Ordering::SeqCst);
        self.job.terminate(KILL_EXIT_CODE);
    }

    /// Wait for the exit to be published (the waiter has reaped the process and drained its
    /// output), at most `timeout`. True when it exited.
    pub async fn wait_exit(&self, timeout: Duration) -> bool {
        let mut rx = self.exit_rx();
        tokio::time::timeout(timeout, health::exited(&mut rx))
            .await
            .is_ok()
    }

    /// Blocking wait on the process handle (no async runtime needed; used at shutdown).
    pub fn wait_exit_blocking(&self, timeout: Duration) -> bool {
        job::wait_process(self.process.as_raw_handle(), timeout)
    }

    /// Graceful stop: close stdin and wait up to `grace`, then kill; wait for the exit.
    pub async fn stop(&self, grace: Duration) -> bool {
        let mut closed = self.close_stdin();
        if !grace.is_zero() {
            let deadline = Instant::now() + grace;
            // A client write in progress holds the pipe; new writes are refused, so it frees up
            // right after that write. Keep closing it while waiting for the exit.
            while !closed {
                if self.wait_exit(STDIN_CLOSE_RETRY).await {
                    return true;
                }
                if Instant::now() >= deadline {
                    break;
                }
                closed = self.close_stdin();
            }
            if self
                .wait_exit(deadline.saturating_duration_since(Instant::now()))
                .await
            {
                return true;
            }
        }
        self.kill();
        self.wait_exit(Duration::from_secs(5)).await
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // Last reference gone: nothing may outlive it (the job would kill on close anyway once the
        // waiter thread drops its reference; do it now so the process never lingers).
        if !self.has_exited() {
            self.kill();
        }
    }
}

/// The stdio pipes of a [`StdioMode::Protocol`] helper instance (e.g. JSON lines).
///
/// `stdout` reaches EOF when the process ends (crash, restart, stop): take the pipes of the new
/// instance with [`super::take_stdio`] after a restart. Writes to `stdin` fail with `BrokenPipe`
/// once Sigil closed it for a stop.
#[derive(Debug)]
#[allow(dead_code)] // Read by P4-E's Bergamot client; exercised by the protocol test.
pub struct HelperStdio {
    pub pid: u32,
    pub stdin: HelperStdin,
    pub stdout: ChildStdout,
}

/// Writer for a protocol helper's stdin. The host keeps the pipe, so a stop can still close it
/// (EOF = quit) even while a client holds this writer; once a stop began, writes fail with
/// `BrokenPipe`.
#[derive(Debug, Clone)]
pub struct HelperStdin {
    shared: Arc<StdinShared>,
}

impl HelperStdin {
    fn closed() -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::BrokenPipe, "the helper's stdin is closed")
    }

    /// Run `f` on the pipe under its lock (one lock per call, so concurrent writers never
    /// interleave inside a message). Once a stop began, close the pipe instead.
    fn with_pipe<T>(&self, f: impl FnOnce(&mut ChildStdin) -> std::io::Result<T>) -> std::io::Result<T> {
        let mut guard = self
            .shared
            .pipe
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if self.shared.closing.load(Ordering::SeqCst) {
            drop(guard.take());
            return Err(Self::closed());
        }
        match guard.as_mut() {
            Some(pipe) => f(pipe),
            None => Err(Self::closed()),
        }
    }
}

impl Write for HelperStdin {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.with_pipe(|pipe| pipe.write(buf))
    }

    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.with_pipe(|pipe| pipe.write_all(buf))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.with_pipe(|pipe| pipe.flush())
    }
}
