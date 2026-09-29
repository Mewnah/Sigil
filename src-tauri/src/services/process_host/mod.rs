//! `process-host` plugin: helper processes (llama-server, TTS servers, research sidecars) that run
//! in a kill-on-close Job Object with `CREATE_NO_WINDOW`, an optional memory cap, a free
//! `127.0.0.1` port, a random API key, a `/health` poll, logs piped to `tracing`, and a restart
//! policy with backoff. Helpers never bind anything but `127.0.0.1`.
//!
//! Owned by P1-E; scaffolded by P1-S0b (plugin registered in main.rs, capability
//! `capabilities/process-host.json`, commands in `plugin-commands/process-host.txt`, and
//! `RunEvent::Exit` → [`shutdown_all`] in main.rs). Decision record: `docs/adr/0005-process-host-llama.md`.
//!
//! # Frozen contract (P1-S0b; change only through INTEGRATE; structs may gain fields)
//! - `async` [`spawn`]`(HelperSpec) -> Result<HelperHandle, String>`: start (or replace) the helper
//!   named `spec.name`, wait for its health check, return where to reach it. `{port}` and
//!   `{api_key}` in `args` / `env` values are substituted with the chosen port and key.
//! - `async` [`stop`]`(name)`, [`get`]`(name) -> Option<HelperInfo>`, [`list`]`() -> Vec<HelperInfo>`.
//! - [`shutdown_all`]`()`: synchronous, idempotent, called on `RunEvent::Exit` (the Job Object also
//!   kills every child if Sigil is killed).
//! - IPC `plugin:process-host|list_helpers -> HelperInfo[]`, `plugin:process-host|stop_helper({ name })`.
//! - Event [`STATE_EVENT`] `"process_host:state"` (payload [`HelperInfo`]) on every state change.
//!
//! # Added by P1-E (additive)
//! - Spec fields [`HelperSpec::env_remove`], [`HelperSpec::priority`], [`HelperSpec::high_qos`],
//!   [`HelperSpec::stop_grace_ms`], [`HelperSpec::stdio`] and [`HealthCheck::send_api_key`] (all
//!   `#[serde(default)]`; build specs with `..Default::default()`).
//! - [`take_stdio`]`(name) -> Option<HelperStdio>`: stdin writer + stdout of a
//!   [`StdioMode::Protocol`] helper (JSON-lines helpers such as Bergamot, P4-E).
//! - [`handle`]`(name) -> Option<HelperHandle>`: the current port / key of a running helper (they
//!   can change after a restart: the port when the old one was taken, never the key).
//! - [`logs`]`(name) -> Vec<HelperLogLine>` and IPC `get_helper_logs({ name })`: the last
//!   [`logs::RING_CAPACITY`] lines of output, API key redacted.
//! - [`run_worker`]`(&WorkerSpec) -> Result<WorkerOutput, String>`: one-shot isolated worker (the
//!   `--headless vc-import-worker` child) in its own job with a memory cap and a wall-clock limit.
//! - [`recipes::llama_server`]: the llama-server spec (127.0.0.1, auto port, key via `LLAMA_API_KEY`).
//!
//! # Behaviour
//! - **Launch** ([`spawn`](mod@spawn)): `CREATE_NO_WINDOW | CREATE_SUSPENDED`, piped stdio, joined to
//!   a fresh Job Object ([`job`]) *before* it runs, then resumed; nothing it starts can escape the
//!   job. The job has `KILL_ON_JOB_CLOSE` (Sigil killed from Task Manager → every helper dies) and
//!   `DIE_ON_UNHANDLED_EXCEPTION` (no Windows Error Reporting dialog), plus the optional job-wide
//!   committed-memory cap and priority class. When a helper's main process exits, the rest of its
//!   job is terminated too. Helpers start with `SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX |
//!   SEM_NOOPENFILEERRORBOX` (inherited error mode, set only around `CreateProcess`): a missing DLL
//!   ends the helper with 0xC0000135 instead of a modal loader box, and such loader failures are
//!   never retried.
//! - **Ports** ([`ports`]): `auto` picks a free `127.0.0.1` port. Before the first HTTP probe the
//!   OS listener table must show the port's listener in the helper's job; a foreign owner, or an
//!   exit before the helper ever answered with the port now held by someone else, is a collision
//!   and is retried on a new port (at most [`lifecycle::MAX_PORT_COLLISIONS`] times, not counted as
//!   a failure; nothing but TCP connects reaches the foreign program, never the key). Once the
//!   helper is ready, the exposure audit reads the OS listener table: any TCP listener of a process
//!   in the helper's job on a non-loopback address (any port) stops the helper with an error.
//! - **Key**: 256-bit random hex from the OS CSPRNG, redacted from every log line (also when a long
//!   line is cut into pieces). Pass it through
//!   `env` (e.g. `LLAMA_API_KEY={api_key}`) rather than `args`: command lines are visible to every
//!   local process.
//! - **Health** ([`health`]): TCP connect probes until the port listens, then
//!   `GET http://127.0.0.1:{port}{path}` every `interval_ms` until 2xx; an exit or a timeout fails
//!   the attempt (the process is killed); no proxy, no redirects.
//! - **Restart**: `on_failure` restarts after `backoff_ms · 2^(n-1)` (capped at 30 s) up to
//!   `max_restarts` consecutive failures; a helper that ran *ready* (after its health check) for
//!   [`lifecycle::STABLE_AFTER`] resets the budget (load time does not count). The API key stays
//!   the same across restarts; an auto port is reused when still free.
//! - **Stop**: stdin is closed (EOF = quit for stdio helpers; new client writes are refused and a
//!   write in flight is followed by EOF), `stop_grace_ms` later the job is terminated. The last
//!   request wins: a `stop` also cancels a `spawn` of the same name that is still queued.
//!   `shutdown_all` kills everything at once and waits at most 3 s.
//! - **Logs** ([`logs`]): stdout/stderr lines → `tracing` target `sigil::helper` (INFO, rate
//!   limited) and a ring buffer; lifecycle problems are WARN (forwarded to the UI log).

pub mod health;
pub mod instance;
pub mod job;
pub mod lifecycle;
pub mod logs;
pub mod ports;
pub mod recipes;
pub mod spawn;
#[cfg(test)]
mod tests;
pub mod worker;

#[allow(unused_imports)] // API for P4-E (Bergamot helper, JSON lines over stdio).
pub use instance::{HelperStdin, HelperStdio};
pub use logs::HelperLogLine;
#[allow(unused_imports)] // API for the isolated import worker (P3-B `vc-import-worker`).
pub use worker::{run_worker, WorkerOutput, WorkerSpec};

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::OnceLock;
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Emitter, Runtime,
};

/// Event emitted when a helper changes state (payload [`HelperInfo`]).
pub const STATE_EVENT: &str = "process_host:state";

/// Which port the helper listens on (always on 127.0.0.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode", content = "port")]
pub enum PortRequest {
    /// The helper opens no port (stdio protocol).
    #[default]
    None,
    /// Pick a free 127.0.0.1 port (retried on collision) and substitute it for `{port}`.
    Auto,
    /// A fixed port.
    Fixed(u16),
}

/// HTTP health check polled after start (`http://127.0.0.1:{port}{path}` must answer 2xx).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthCheck {
    pub path: String,
    /// Give up (kill, then apply the restart policy) after this long.
    pub timeout_ms: u64,
    pub interval_ms: u64,
    /// Send `Authorization: Bearer <api key>` with the health request (for servers that protect
    /// every route; llama-server's `/health` is public, so the default is false).
    #[serde(default)]
    pub send_api_key: bool,
}

impl Default for HealthCheck {
    fn default() -> Self {
        Self {
            path: "/health".into(),
            timeout_ms: 60_000,
            interval_ms: 250,
            send_api_key: false,
        }
    }
}

/// What to do when the helper exits unexpectedly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum RestartPolicy {
    #[default]
    Never,
    /// Restart up to `max_restarts` times, waiting `backoff_ms` doubled per attempt (capped at 30 s).
    OnFailure { max_restarts: u32, backoff_ms: u64 },
}

/// What happens to the helper's stdout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StdioMode {
    /// stdout and stderr go to the log (HTTP servers such as llama-server).
    #[default]
    Log,
    /// stdout carries a protocol (e.g. JSON lines, the Bergamot helper): the client takes it and a
    /// stdin writer with [`take_stdio`] and must keep reading stdout (a helper blocks once the pipe
    /// buffer is full); stderr still goes to the log.
    Protocol,
}

/// Scheduling priority forced on every process of the helper's job.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HelperPriority {
    #[default]
    Normal,
    /// Yields the CPU to VRChat, OBS and the voice changer (CPU-heavy helpers such as llama-server on
    /// the CPU backend).
    BelowNormal,
    /// Only runs when the CPU is otherwise idle (background batch work).
    Idle,
}

/// How to start a helper.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HelperSpec {
    /// Unique key (`llama-server`, `bergamot`, `qwentts`, ...); spawning an existing name replaces it.
    pub name: String,
    pub program: PathBuf,
    /// Arguments; `{port}` and `{api_key}` are substituted.
    pub args: Vec<String>,
    /// Extra environment; values support the same placeholders.
    pub env: Vec<(String, String)>,
    /// Working directory; `None` = the program's folder when `program` is a path.
    pub cwd: Option<PathBuf>,
    pub port: PortRequest,
    /// Generate a random API key for `{api_key}` (returned in [`HelperHandle::api_key`]).
    pub api_key: bool,
    /// Poll this after start; `None` = ready as soon as the process runs.
    pub health: Option<HealthCheck>,
    /// Job Object memory cap.
    pub memory_limit_mb: Option<u64>,
    pub restart: RestartPolicy,
    /// Inherited environment variables removed before `env` is applied: exact names, or a prefix
    /// ending in `*` (e.g. `LLAMA_ARG_*`, so a user's global llama.cpp settings cannot override
    /// Sigil's). Case-insensitive, like Windows environment names.
    #[serde(default)]
    pub env_remove: Vec<String>,
    /// Priority class of the helper's job.
    #[serde(default)]
    pub priority: HelperPriority,
    /// Opt the helper out of Windows 11 EcoQoS (efficiency cores while Sigil is in the background),
    /// like Sigil itself. For latency-sensitive helpers (translation, TTS).
    #[serde(default)]
    pub high_qos: bool,
    /// On stop, close stdin and wait this long for a clean exit before terminating (0 = terminate
    /// at once; HTTP servers without persistent state need no grace).
    #[serde(default)]
    pub stop_grace_ms: u64,
    /// Log mode (default) or a stdio protocol.
    #[serde(default)]
    pub stdio: StdioMode,
}

/// Where to reach a running helper.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HelperHandle {
    pub name: String,
    pub pid: u32,
    pub port: Option<u16>,
    /// `http://127.0.0.1:{port}` when a port is open.
    pub base_url: Option<String>,
    /// Bearer key when `api_key` was requested. Never logged or sent to the UI.
    #[serde(skip_serializing)]
    pub api_key: Option<String>,
}

/// Lifecycle state of a helper.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HelperState {
    Starting,
    Running,
    Restarting,
    Stopped,
    Failed,
}

/// Row of `list_helpers` and payload of [`STATE_EVENT`] (never contains the API key).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HelperInfo {
    pub name: String,
    pub state: HelperState,
    pub pid: Option<u32>,
    pub port: Option<u16>,
    pub restarts: u32,
    /// Last error or exit reason.
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------------------------
// Event emission (installed in the plugin setup; a no-op in tests and headless runs)
// ---------------------------------------------------------------------------------------------

type EmitFn = Box<dyn Fn(&HelperInfo) + Send + Sync>;
static EMITTER: OnceLock<EmitFn> = OnceLock::new();

pub(crate) fn emit_state(info: &HelperInfo) {
    if let Some(emit) = EMITTER.get() {
        emit(info);
    }
}

// ---------------------------------------------------------------------------------------------
// Rust API
// ---------------------------------------------------------------------------------------------

/// Start (or replace) a helper and wait until it is healthy.
#[allow(dead_code)] // Frozen API; first in-tree consumer is P4-B (llm_server), then P4-E / P6-E.
pub async fn spawn(spec: HelperSpec) -> Result<HelperHandle, String> {
    lifecycle::spawn(spec).await
}

/// Stop a helper (graceful, then kill); Ok when it was not running.
pub async fn stop(name: &str) -> Result<(), String> {
    lifecycle::stop(name).await
}

#[allow(dead_code)] // Frozen API (P4-B status command).
pub fn get(name: &str) -> Option<HelperInfo> {
    lifecycle::get(name)
}

pub fn list() -> Vec<HelperInfo> {
    lifecycle::list()
}

/// Where to reach `name` right now (`None` unless it is running).
#[allow(dead_code)] // API for P4-B (the port can change after a restart).
pub fn handle(name: &str) -> Option<HelperHandle> {
    lifecycle::handle(name)
}

/// The stdio pipes of the running [`StdioMode::Protocol`] helper `name` (once per process).
#[allow(dead_code)] // API for P4-E (Bergamot helper, JSON lines over stdio).
pub fn take_stdio(name: &str) -> Option<HelperStdio> {
    lifecycle::take_stdio(name)
}

/// Recent output of `name` (API key redacted), oldest first.
pub fn logs(name: &str) -> Vec<HelperLogLine> {
    lifecycle::logs(name)
}

/// Kill every helper (synchronous, idempotent). Called on `RunEvent::Exit`.
pub fn shutdown_all() {
    lifecycle::shutdown_all()
}

// ---------------------------------------------------------------------------------------------
// IPC
// ---------------------------------------------------------------------------------------------

#[command]
async fn list_helpers() -> Result<Vec<HelperInfo>, String> {
    Ok(list())
}

#[command]
async fn stop_helper(name: String) -> Result<(), String> {
    stop(&name).await
}

#[command]
async fn get_helper_logs(name: String) -> Result<Vec<HelperLogLine>, String> {
    spawn::validate_name(&name)?;
    Ok(logs(&name))
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("process-host")
        .setup(|app, _api| {
            let handle = app.clone();
            let _ = EMITTER.set(Box::new(move |info| {
                let _ = handle.emit(STATE_EVENT, info);
            }));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![list_helpers, stop_helper, get_helper_logs])
        .build()
}
