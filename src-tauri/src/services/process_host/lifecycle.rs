//! Helper registry and lifecycle: start (with port-collision retry and the restart policy), the
//! supervisor that restarts crashed helpers with backoff, stop (graceful, then kill), replace, and
//! the synchronous `shutdown_all` for `RunEvent::Exit`.
//!
//! Concurrency model:
//! - One `Helper` entry per name in the registry. Starting a name that exists replaces the entry.
//! - Lifecycle operations on one name (spawn, stop) are serialized by a per-name async gate. Both
//!   first *cancel* the current entry without waiting for the gate, so an in-flight start (which may
//!   be waiting minutes for a health check) aborts promptly instead of blocking a stop.
//! - The last request wins: every spawn or stop takes a per-name ticket when it is called. A
//!   request that is no longer the latest once it holds the gate does nothing (a spawn queued
//!   behind a replace never starts after a later stop), and a spawn installs its entry only while
//!   its ticket is still the latest (checked under the registry lock).
//! - An entry's cancel flag is checked under its `inner` lock whenever an instance is installed, so
//!   a process started concurrently with a stop is killed immediately, never orphaned.
//! - State changes of a cancelled entry are ignored, except the final `stopped`.

use super::health::{self, ExitInfo, Readiness};
use super::instance::Instance;
use super::job;
use super::logs::LogBook;
use super::{emit_state, ports, HelperHandle, HelperInfo, HelperSpec, HelperState, PortRequest, RestartPolicy};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, Instant};
use tokio::sync::watch;

/// A helper that ran *ready* (after its health check) at least this long before crashing resets its
/// restart budget. Load time does not count: a model that loads for minutes and then crashes still
/// uses up the budget.
pub const STABLE_AFTER: Duration = Duration::from_secs(60);
/// Longest restart backoff.
pub const MAX_BACKOFF: Duration = Duration::from_secs(30);
/// Retries with a different port when the auto-picked port was taken before the helper bound it.
pub const MAX_PORT_COLLISIONS: u32 = 3;
/// Without a health check, a helper that exits within this window failed to start.
const EARLY_EXIT_WINDOW: Duration = Duration::from_millis(200);
/// How long `shutdown_all` waits for all helpers to die.
const SHUTDOWN_WAIT: Duration = Duration::from_secs(3);
const TARGET: &str = super::logs::LOG_TARGET;

static SHUTTING_DOWN: AtomicBool = AtomicBool::new(false);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

#[derive(Default)]
struct Registry {
    helpers: HashMap<String, Arc<Helper>>,
    /// Per name, the ticket of the latest spawn or stop request.
    tickets: HashMap<String, u64>,
}

impl Registry {
    fn is_latest(&self, name: &str, ticket: u64) -> bool {
        self.tickets.get(name).copied() == Some(ticket)
    }
}

fn registry() -> &'static Mutex<Registry> {
    static R: OnceLock<Mutex<Registry>> = OnceLock::new();
    R.get_or_init(Default::default)
}

/// Register a spawn (`create`) or stop request for `name`: take the next ticket and cancel the
/// current helper at once (outside the registry lock). `None` for a stop of a name that was never
/// spawned (nothing to do, and no per-name state is created for arbitrary names).
fn new_request(name: &str, create: bool) -> Option<u64> {
    let (ticket, current) = {
        let mut r = lock(registry());
        if !create && !r.tickets.contains_key(name) {
            return None;
        }
        let t = r.tickets.entry(name.to_string()).or_insert(0);
        *t += 1;
        (*t, r.helpers.get(name).cloned())
    };
    if let Some(helper) = current {
        helper.cancel();
    }
    Some(ticket)
}

fn gate(name: &str) -> Arc<tokio::sync::Mutex<()>> {
    static G: OnceLock<Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> = OnceLock::new();
    lock(G.get_or_init(Default::default))
        .entry(name.to_string())
        .or_default()
        .clone()
}

fn registry_get(name: &str) -> Option<Arc<Helper>> {
    lock(registry()).helpers.get(name).cloned()
}

/// How long a helper must run ready before a crash resets its restart budget.
fn stable_after(name: &str) -> Duration {
    #[cfg(test)]
    if let Some(d) = test_hooks::stable_after(name) {
        return d;
    }
    let _ = name;
    STABLE_AFTER
}

/// Consecutive failures after an unexpected exit of an instance that had been ready for
/// `ready_for`: a helper that ran ready for at least `stable` starts a fresh budget, one that
/// never became ready or crashed soon after adds to it.
pub fn failures_after_exit(failures: u32, ready_for: Option<Duration>, stable: Duration) -> u32 {
    if ready_for.is_some_and(|d| d >= stable) {
        1
    } else {
        failures.saturating_add(1)
    }
}

/// Backoff before restart number `failures` (1-based): `backoff_ms · 2^(failures-1)`, capped at
/// [`MAX_BACKOFF`]. `None` when the policy allows no further restart.
pub fn restart_delay(policy: &RestartPolicy, failures: u32) -> Option<Duration> {
    match *policy {
        RestartPolicy::Never => None,
        RestartPolicy::OnFailure { max_restarts, backoff_ms } => {
            if failures == 0 || failures > max_restarts {
                return None;
            }
            let factor = 1u64 << (failures - 1).min(20);
            Some(Duration::from_millis(backoff_ms.saturating_mul(factor)).min(MAX_BACKOFF))
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Helper entry
// ---------------------------------------------------------------------------------------------

struct Inner {
    info: HelperInfo,
    instance: Option<Arc<Instance>>,
    /// Port of the last healthy instance (an auto-port helper restarts on it when it is free).
    last_port: Option<u16>,
}

pub struct Helper {
    pub name: String,
    spec: HelperSpec,
    api_key: Option<String>,
    cancel_tx: watch::Sender<bool>,
    pub book: Arc<LogBook>,
    inner: Mutex<Inner>,
}

impl Helper {
    fn new(spec: HelperSpec, api_key: Option<String>) -> Helper {
        let (cancel_tx, _) = watch::channel(false);
        Helper {
            name: spec.name.clone(),
            book: Arc::new(LogBook::new(&spec.name, api_key.clone())),
            inner: Mutex::new(Inner {
                info: HelperInfo {
                    name: spec.name.clone(),
                    state: HelperState::Starting,
                    pid: None,
                    port: None,
                    restarts: 0,
                    error: None,
                },
                instance: None,
                last_port: None,
            }),
            spec,
            api_key,
            cancel_tx,
        }
    }

    fn is_cancelled(&self) -> bool {
        *self.cancel_tx.borrow()
    }

    fn cancel_rx(&self) -> watch::Receiver<bool> {
        self.cancel_tx.subscribe()
    }

    /// Stop starting or restarting; the running process (if any) gets EOF on stdin.
    fn cancel(&self) {
        self.cancel_tx.send_replace(true);
        if let Some(inst) = lock(&self.inner).instance.clone() {
            inst.close_stdin();
        }
    }

    fn info(&self) -> HelperInfo {
        lock(&self.inner).info.clone()
    }

    /// Change the public state (ignored once cancelled) and emit [`super::STATE_EVENT`].
    fn update(&self, f: impl FnOnce(&mut HelperInfo)) {
        let info = {
            let mut g = lock(&self.inner);
            if self.is_cancelled() {
                return;
            }
            f(&mut g.info);
            g.info.clone()
        };
        emit_state(&info);
    }

    /// Install a freshly started instance; false (and the caller must kill it) once cancelled.
    fn install(&self, inst: &Arc<Instance>) -> bool {
        let mut g = lock(&self.inner);
        if self.is_cancelled() {
            return false;
        }
        g.instance = Some(inst.clone());
        true
    }

    /// Forget `inst` if it is still the current instance.
    fn clear_instance(&self, inst: &Arc<Instance>) {
        let mut g = lock(&self.inner);
        if g.instance
            .as_ref()
            .is_some_and(|cur| Arc::ptr_eq(cur, inst))
        {
            g.instance = None;
        }
    }

    fn take_instance(&self) -> Option<Arc<Instance>> {
        lock(&self.inner).instance.take()
    }

    #[cfg(test)]
    fn current_instance(&self) -> Option<Arc<Instance>> {
        lock(&self.inner).instance.clone()
    }

    fn last_port(&self) -> Option<u16> {
        lock(&self.inner).last_port
    }

    fn forget_port(&self) {
        lock(&self.inner).last_port = None;
    }

    fn set_running(&self, inst: &Arc<Instance>, started: Instant) {
        {
            let mut g = lock(&self.inner);
            if !self.is_cancelled() {
                g.last_port = inst.port;
            }
        }
        self.update(|i| {
            i.state = HelperState::Running;
            i.pid = Some(inst.pid);
            i.port = inst.port;
            i.error = None;
        });
        let port = inst
            .port
            .map(|p| format!(", 127.0.0.1:{p}"))
            .unwrap_or_default();
        let msg = format!("running: pid {}{port}, ready in {} ms", inst.pid, started.elapsed().as_millis());
        tracing::info!(target: TARGET, "[{}] {msg}", self.name);
        self.book.note(&msg);
    }

    /// Final state after a stop (applies even when cancelled). `emit` is false at app exit.
    fn mark_stopped(&self, emit: bool) {
        let info = {
            let mut g = lock(&self.inner);
            g.info.state = HelperState::Stopped;
            g.info.pid = None;
            g.info.port = None;
            g.info.error = None;
            g.info.clone()
        };
        if emit {
            emit_state(&info);
        }
    }

    pub fn handle(&self) -> Option<HelperHandle> {
        let g = lock(&self.inner);
        let inst = g.instance.as_ref()?;
        if g.info.state != HelperState::Running || inst.has_exited() {
            return None;
        }
        Some(HelperHandle {
            name: self.name.clone(),
            pid: inst.pid,
            port: inst.port,
            base_url: inst.port.map(|p| format!("http://127.0.0.1:{p}")),
            api_key: self.api_key.clone(),
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Starting
// ---------------------------------------------------------------------------------------------

/// Why one start attempt failed.
enum Attempt {
    /// Retry per the restart policy (crash during startup, health timeout).
    Failed(String),
    /// The auto-picked port was taken by another program before the helper bound it.
    Collision(u16),
    /// Deterministic problem (cannot launch, port in use, policy violation): no retry.
    Fatal(String),
    Cancelled,
}

const CANCELLED: &str = "cancelled";

fn choose_port(helper: &Helper) -> Result<Option<u16>, Attempt> {
    match helper.spec.port {
        PortRequest::None => Ok(None),
        PortRequest::Fixed(p) => {
            if ports::loopback_port_free(p) {
                Ok(Some(p))
            } else {
                Err(Attempt::Fatal(format!(
                    "{}: port {p} on 127.0.0.1 is already in use by another program",
                    helper.name
                )))
            }
        }
        PortRequest::Auto => {
            #[cfg(test)]
            if let Some(p) = ports::forced::next(&helper.name) {
                return Ok(Some(p));
            }
            if let Some(p) = helper.last_port().filter(|p| ports::loopback_port_free(*p)) {
                return Ok(Some(p));
            }
            ports::pick_free_port().map(Some).map_err(Attempt::Fatal)
        }
    }
}

async fn kill_and_reap(helper: &Helper, inst: &Arc<Instance>) {
    helper.clear_instance(inst);
    inst.kill();
    if !inst.wait_exit(Duration::from_secs(5)).await {
        tracing::warn!(target: TARGET, "[{}] pid {} did not exit 5 s after being terminated", helper.name, inst.pid);
    }
}

async fn attempt(helper: &Arc<Helper>, cancel: &mut watch::Receiver<bool>) -> Result<Arc<Instance>, Attempt> {
    let name = helper.name.as_str();
    let port = choose_port(helper)?;
    let inst = {
        let h = helper.clone();
        tokio::task::spawn_blocking(move || Instance::start(&h.spec, port, h.api_key.as_deref(), &h.book))
            .await
            .map_err(|e| Attempt::Fatal(format!("{name}: start task failed: {e}")))?
            .map_err(Attempt::Fatal)?
    };
    let inst = Arc::new(inst);
    if !helper.install(&inst) {
        kill_and_reap(helper, &inst).await;
        return Err(Attempt::Cancelled);
    }
    helper.update(|i| {
        i.pid = Some(inst.pid);
        i.port = port;
    });
    helper.book.note(&format!(
        "started pid {}{}",
        inst.pid,
        port.map(|p| format!(" on 127.0.0.1:{p}"))
            .unwrap_or_default()
    ));

    match (&helper.spec.health, port) {
        (Some(check), Some(p)) => {
            let bearer = if check.send_api_key { helper.api_key.as_deref() } else { None };
            let mut exit_rx = inst.exit_rx();
            let helper_pids = || inst.process_ids();
            let (outcome, answered) = health::wait_ready(p, check, bearer, &helper_pids, &mut exit_rx, cancel).await;
            match outcome {
                Readiness::Ready => {}
                Readiness::Cancelled => {
                    kill_and_reap(helper, &inst).await;
                    return Err(Attempt::Cancelled);
                }
                Readiness::Exited(info) => {
                    helper.clear_instance(&inst);
                    return Err(startup_exit(helper, Some(p), answered, &info));
                }
                Readiness::Foreign { pid } => {
                    // Another program answers on the port (it took it before the helper bound it);
                    // nothing was sent to it but TCP connects. The helper cannot serve there.
                    kill_and_reap(helper, &inst).await;
                    return Err(match helper.spec.port {
                        PortRequest::Auto => Attempt::Collision(p),
                        _ => Attempt::Fatal(format!(
                            "{name}: port {p} on 127.0.0.1 is held by another program (pid {pid}), not by the helper"
                        )),
                    });
                }
                Readiness::TimedOut { last } => {
                    kill_and_reap(helper, &inst).await;
                    return Err(Attempt::Failed(format!(
                        "{name} did not become healthy within {:.1} s (last {} check: {last}){}",
                        check.timeout_ms as f64 / 1000.0,
                        check.path,
                        helper.book.tail_for_error()
                    )));
                }
            }
        }
        _ => {
            // No health check: ready once it survives a short window.
            let mut exit_rx = inst.exit_rx();
            let mut cancel_rx = cancel.clone();
            tokio::select! {
                biased;
                _ = health::cancelled(&mut cancel_rx) => {
                    kill_and_reap(helper, &inst).await;
                    return Err(Attempt::Cancelled);
                }
                info = health::exited(&mut exit_rx) => {
                    helper.clear_instance(&inst);
                    return Err(startup_exit(helper, port, false, &info));
                }
                _ = tokio::time::sleep(EARLY_EXIT_WINDOW) => {}
            }
        }
    }

    let exposed = exposure_audit(&inst, port).await;
    if !exposed.is_empty() {
        kill_and_reap(helper, &inst).await;
        let list: Vec<String> = exposed.iter().map(ToString::to_string).collect();
        return Err(Attempt::Fatal(format!(
            "{name} listens on {} (not only 127.0.0.1); helpers must bind 127.0.0.1 only, so it was stopped",
            list.join(", ")
        )));
    }
    if helper.is_cancelled() {
        kill_and_reap(helper, &inst).await;
        return Err(Attempt::Cancelled);
    }
    inst.mark_ready();
    Ok(inst)
}

/// Non-loopback TCP listeners of the helper's processes (every port, not only the announced one),
/// read from the OS listener table. If the table cannot be read, falls back to connecting to the
/// announced port on this PC's other addresses.
async fn exposure_audit(inst: &Instance, port: Option<u16>) -> Vec<std::net::SocketAddr> {
    match ports::exposed_listeners(&inst.process_ids()) {
        Ok(found) => found,
        Err(e) => {
            tracing::debug!(target: TARGET, "listener table unavailable ({e}); probing the port instead");
            match port {
                Some(p) => ports::exposure(p).await,
                None => Vec::new(),
            }
        }
    }
}

/// Classify an exit during startup: the Windows loader refused the program (missing or mismatched
/// DLL, wrong architecture: deterministic, never retried), a port collision (auto port taken by
/// someone else, and the helper never answered HTTP), or an ordinary failure.
fn startup_exit(helper: &Helper, port: Option<u16>, answered: bool, info: &ExitInfo) -> Attempt {
    if !info.killed_by_host && info.code.is_some_and(job::is_loader_failure) {
        return Attempt::Fatal(format!("{} could not start ({info}){}", helper.name, helper.book.tail_for_error()));
    }
    if let Some(p) = port {
        if helper.spec.port == PortRequest::Auto && !answered && !ports::loopback_port_free(p) {
            return Attempt::Collision(p);
        }
    }
    Attempt::Failed(format!("{} exited while starting ({info}){}", helper.name, helper.book.tail_for_error()))
}

/// Start an instance and wait until it is ready, restarting per the policy. `failures` counts the
/// consecutive failures so far (0 for a fresh start); `last_error` is the latest one. Returns the
/// instance and the updated failure count.
async fn start_until_ready(helper: &Arc<Helper>, mut failures: u32, mut last_error: Option<String>) -> Result<(Arc<Instance>, u32), String> {
    let mut cancel = helper.cancel_rx();
    let mut collisions = 0u32;
    let mut pending_restart = failures > 0;
    loop {
        if helper.is_cancelled() {
            return Err(CANCELLED.into());
        }
        if pending_restart {
            pending_restart = false;
            let err = last_error.clone().unwrap_or_else(|| "failed".into());
            let Some(delay) = restart_delay(&helper.spec.restart, failures) else {
                return Err(match helper.spec.restart {
                    RestartPolicy::OnFailure { max_restarts, .. } if max_restarts > 0 => {
                        format!("{err} (gave up after {max_restarts} restarts)")
                    }
                    _ => err,
                });
            };
            helper.update(|i| {
                i.state = HelperState::Restarting;
                i.error = Some(err.clone());
                i.restarts = i.restarts.saturating_add(1);
                i.pid = None;
                i.port = None;
            });
            tracing::warn!(target: TARGET, "[{}] restarting in {} ms: {err}", helper.name, delay.as_millis());
            helper
                .book
                .note(&format!("restarting in {} ms", delay.as_millis()));
            tokio::select! {
                biased;
                _ = health::cancelled(&mut cancel) => return Err(CANCELLED.into()),
                _ = tokio::time::sleep(delay) => {}
            }
        }
        match attempt(helper, &mut cancel).await {
            Ok(inst) => return Ok((inst, failures)),
            Err(Attempt::Cancelled) => return Err(CANCELLED.into()),
            Err(Attempt::Fatal(e)) => return Err(e),
            Err(Attempt::Collision(p)) => {
                collisions += 1;
                helper.forget_port();
                if collisions > MAX_PORT_COLLISIONS {
                    return Err(format!(
                        "{}: the chosen port was taken by another program {collisions} times in a row (last {p})",
                        helper.name
                    ));
                }
                tracing::info!(target: TARGET, "[{}] port {p} was taken before the helper bound it; retrying on another port", helper.name);
                helper
                    .book
                    .note(&format!("port {p} was taken by another program; retrying"));
            }
            Err(Attempt::Failed(e)) => {
                failures += 1;
                last_error = Some(e);
                pending_restart = true;
            }
        }
    }
}

/// Watch a running helper; restart it per the policy when it exits unexpectedly.
async fn supervise(helper: Arc<Helper>, mut inst: Arc<Instance>, mut failures: u32) {
    let mut cancel = helper.cancel_rx();
    loop {
        let mut exit_rx = inst.exit_rx();
        let info = tokio::select! {
            biased;
            _ = health::cancelled(&mut cancel) => return,
            info = health::exited(&mut exit_rx) => info,
        };
        if helper.is_cancelled() {
            return;
        }
        helper.clear_instance(&inst);
        failures = failures_after_exit(failures, inst.ready_for(), stable_after(&helper.name));
        let msg = format!("{} exited unexpectedly ({info}){}", helper.name, helper.book.tail_for_error());
        tracing::warn!(target: TARGET, "[{}] {msg}", helper.name);
        let started = Instant::now();
        match start_until_ready(&helper, failures, Some(msg)).await {
            Ok((next, f)) => {
                failures = f;
                helper.set_running(&next, started);
                inst = next;
            }
            Err(e) => {
                if !helper.is_cancelled() {
                    tracing::warn!(target: TARGET, "[{}] failed: {e}", helper.name);
                    helper.book.note(&format!("failed: {e}"));
                    helper.update(|i| {
                        i.state = HelperState::Failed;
                        i.error = Some(e.clone());
                        i.pid = None;
                        i.port = None;
                    });
                }
                return;
            }
        }
    }
}

/// Stop `helper`'s process (graceful, then kill) and mark it stopped.
async fn retire(helper: &Arc<Helper>, reason: &str) {
    if let Some(inst) = helper.take_instance() {
        let grace = Duration::from_millis(helper.spec.stop_grace_ms);
        if !inst.stop(grace).await {
            tracing::warn!(target: TARGET, "[{}] pid {} did not exit after being terminated", helper.name, inst.pid);
        }
        helper.book.note(&format!("stopped ({reason})"));
        tracing::info!(target: TARGET, "[{}] stopped ({reason})", helper.name);
    }
    helper.mark_stopped(true);
}

// ---------------------------------------------------------------------------------------------
// Public operations (wrapped by mod.rs)
// ---------------------------------------------------------------------------------------------

/// Start (or replace) a helper. The work runs on a detached task, so a caller that stops waiting
/// (a dropped future, a timeout around this call) never leaves a half-started helper behind: the
/// start still completes (running and supervised, or failed).
pub async fn spawn(spec: HelperSpec) -> Result<HelperHandle, String> {
    if SHUTTING_DOWN.load(Ordering::SeqCst) {
        return Err("Sigil is shutting down; helpers cannot start".into());
    }
    super::spawn::validate(&spec)?;
    // The request counts from here: a stop or spawn of this name issued later wins, even while this
    // one still waits behind an earlier operation.
    let ticket = new_request(&spec.name, true).unwrap_or_default();
    tauri::async_runtime::spawn(spawn_detached(spec, ticket))
        .await
        .map_err(|e| format!("helper start task failed: {e}"))?
}

fn superseded(name: &str) -> String {
    format!("{name}: start cancelled (stopped or replaced)")
}

async fn spawn_detached(spec: HelperSpec, ticket: u64) -> Result<HelperHandle, String> {
    let name = spec.name.clone();
    let gate = gate(&name);
    let _guard = gate.lock().await;
    if !lock(registry()).is_latest(&name, ticket) {
        return Err(superseded(&name)); // the later request handles the current entry
    }
    if let Some(old) = registry_get(&name) {
        old.cancel();
        retire(&old, "replaced").await;
    }
    if SHUTTING_DOWN.load(Ordering::SeqCst) {
        return Err("Sigil is shutting down; helpers cannot start".into());
    }
    let api_key = if spec.api_key {
        Some(super::spawn::random_key().map_err(|e| format!("{name}: cannot create an API key: {e}"))?)
    } else {
        None
    };
    let helper = Arc::new(Helper::new(spec, api_key));
    {
        let mut r = lock(registry());
        // A stop issued while the old helper was being retired wins over this start.
        if !r.is_latest(&name, ticket) {
            return Err(superseded(&name));
        }
        r.helpers.insert(name.clone(), helper.clone());
    }
    emit_state(&helper.info());
    helper
        .book
        .note(&format!("starting {}", helper.spec.program.display()));
    tracing::info!(target: TARGET, "[{name}] starting {}", helper.spec.program.display());

    let started = Instant::now();
    match start_until_ready(&helper, 0, None).await {
        Ok((inst, failures)) => {
            helper.set_running(&inst, started);
            let handle = helper.handle();
            let h = helper.clone();
            tauri::async_runtime::spawn(supervise(h, inst, failures));
            handle.ok_or_else(|| format!("{name}: stopped while starting"))
        }
        Err(_) if helper.is_cancelled() => Err(superseded(&name)),
        Err(e) => {
            tracing::warn!(target: TARGET, "[{name}] failed to start: {e}");
            helper.book.note(&format!("failed to start: {e}"));
            helper.update(|i| {
                i.state = HelperState::Failed;
                i.error = Some(e.clone());
                i.pid = None;
                i.port = None;
            });
            Err(e)
        }
    }
}

/// Stop a helper (detached like [`spawn`], so the stop completes even if the caller goes away).
/// Also cancels a spawn of this name that was requested earlier and is still queued.
pub async fn stop(name: &str) -> Result<(), String> {
    let Some(ticket) = new_request(name, false) else {
        return Ok(()); // never spawned
    };
    let name = name.to_string();
    tauri::async_runtime::spawn(async move { stop_detached(&name, ticket).await })
        .await
        .map_err(|e| format!("helper stop task failed: {e}"))
}

async fn stop_detached(name: &str, ticket: u64) {
    let gate = gate(name);
    let _guard = gate.lock().await;
    let current = {
        let r = lock(registry());
        // A later spawn or stop supersedes this stop (the last request wins).
        if !r.is_latest(name, ticket) {
            return;
        }
        r.helpers.get(name).cloned()
    };
    // Every entry was installed before this request's ticket, so it is already cancelled.
    if let Some(helper) = current {
        helper.cancel();
        retire(&helper, "stopped").await;
    }
}

pub fn get(name: &str) -> Option<HelperInfo> {
    registry_get(name).map(|h| h.info())
}

pub fn list() -> Vec<HelperInfo> {
    let mut out: Vec<HelperInfo> = lock(registry())
        .helpers
        .values()
        .map(|h| h.info())
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

pub fn handle(name: &str) -> Option<HelperHandle> {
    registry_get(name)?.handle()
}

pub fn take_stdio(name: &str) -> Option<super::HelperStdio> {
    let helper = registry_get(name)?;
    let g = lock(&helper.inner);
    if g.info.state != HelperState::Running {
        return None;
    }
    g.instance.as_ref()?.take_stdio()
}

pub fn logs(name: &str) -> Vec<super::logs::HelperLogLine> {
    registry_get(name)
        .map(|h| h.book.snapshot())
        .unwrap_or_default()
}

/// Kill every helper and wait (bounded) for them to die. Synchronous and idempotent; later spawns
/// are refused.
pub fn shutdown_all() {
    SHUTTING_DOWN.store(true, Ordering::SeqCst);
    let helpers: Vec<Arc<Helper>> = lock(registry()).helpers.values().cloned().collect();
    let mut running = Vec::new();
    for h in &helpers {
        h.cancel();
        if let Some(inst) = h.take_instance() {
            inst.kill();
            running.push((h.clone(), inst));
        }
    }
    let deadline = Instant::now() + SHUTDOWN_WAIT;
    for (h, inst) in &running {
        let left = deadline.saturating_duration_since(Instant::now());
        if !inst.wait_exit_blocking(left) {
            tracing::warn!(target: TARGET, "[{}] pid {} still running at shutdown (the Job Object kills it when Sigil exits)", h.name, inst.pid);
        }
    }
    for h in &helpers {
        h.mark_stopped(false);
    }
    if !running.is_empty() {
        tracing::info!(target: TARGET, "stopped {} helper(s) at shutdown", running.len());
    }
}

/// The currently installed instance of `name` (tests).
#[cfg(test)]
pub fn current_pid(name: &str) -> Option<u32> {
    registry_get(name)?.current_instance().map(|i| i.pid)
}

/// Test hook: a shorter [`STABLE_AFTER`] for one helper name (restart-budget tests).
#[cfg(test)]
pub mod test_hooks {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    use std::time::Duration;

    fn table() -> &'static Mutex<HashMap<String, Duration>> {
        static T: OnceLock<Mutex<HashMap<String, Duration>>> = OnceLock::new();
        T.get_or_init(Default::default)
    }

    pub fn set_stable_after(name: &str, d: Duration) {
        table().lock().unwrap().insert(name.to_string(), d);
    }

    pub(super) fn stable_after(name: &str) -> Option<Duration> {
        table().lock().unwrap().get(name).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restart_budget_resets_only_after_running_ready() {
        let stable = Duration::from_secs(60);
        // Never ready (crashed while loading): counts, however long the load took.
        assert_eq!(failures_after_exit(1, None, stable), 2);
        // Crashed soon after becoming ready: counts.
        assert_eq!(failures_after_exit(2, Some(Duration::from_secs(5)), stable), 3);
        // Ran ready for the stable period: a fresh budget with this failure as the first.
        assert_eq!(failures_after_exit(3, Some(Duration::from_secs(60)), stable), 1);
        assert_eq!(failures_after_exit(u32::MAX, None, stable), u32::MAX);
    }

    #[test]
    fn restart_delay_doubles_and_caps() {
        let p = RestartPolicy::OnFailure {
            max_restarts: 10,
            backoff_ms: 500,
        };
        assert_eq!(restart_delay(&p, 0), None);
        assert_eq!(restart_delay(&p, 1), Some(Duration::from_millis(500)));
        assert_eq!(restart_delay(&p, 2), Some(Duration::from_millis(1000)));
        assert_eq!(restart_delay(&p, 4), Some(Duration::from_millis(4000)));
        assert_eq!(restart_delay(&p, 8), Some(MAX_BACKOFF));
        assert_eq!(restart_delay(&p, 11), None);
        assert_eq!(restart_delay(&RestartPolicy::Never, 1), None);
        let huge = RestartPolicy::OnFailure {
            max_restarts: u32::MAX,
            backoff_ms: u64::MAX,
        };
        assert_eq!(restart_delay(&huge, 1000), Some(MAX_BACKOFF));
    }
}
