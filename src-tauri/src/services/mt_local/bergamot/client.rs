//! The Bergamot engine: one helper process (`process_host`, `StdioMode::Protocol`), one [`Conn`] to it at a time,
//! and the request logic on top (routing, on-demand model loading with an LRU cap, pivots, cancellation, retries,
//! status). Where models and the helper come from is abstracted by [`Environment`] (the model store in the app, fakes
//! in tests).
//!
//! Threads: every connection has a writer thread (stdin) and a reader thread (stdout; it must keep reading or the
//! helper blocks on a full pipe). Callers are async and never block on pipe I/O.
//!
//! Generations: every stop bumps the engine's generation. A connection belongs to the generation it was opened
//! under and requests only use connections of their own generation, so a stop ends exactly the helpers that ran
//! before it: a start right after it (a pipeline restart) keeps its own helper.
//!
//! Memory: the helper's job has a hard memory cap, so the helper never holds more directions than [`loaded_limit`]
//! allows, not even while loading. A load first unloads the least recently used directions no line is using, and
//! waits for the lines still translating with them when it has to.
//!
//! Cancellation: starting the helper and loading models run on their own tasks. A cancelled line returns at once
//! and gives its slot back; the start or load it began completes for the lines after it.

use super::super::{thread_count, MtEngineState, MtEngineStatus, MtError, MtErrorKind, MtKind, MtPair, MtRequest, MtResponse, ENGINE_BERGAMOT};
use super::catalog::{self, Direction, Directions, ModelFiles, Route};
use super::protocol::{self, HelloInfo, HelperError, Line, Priority, Translated, MAX_LINE_BYTES, PROTOCOL};
use super::LOG_TARGET;
use crate::services::process_host::{self, HelperSpec, HelperState, HelperStdio};
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::io::{BufReader, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::task::Poll;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{oneshot, watch, Notify};

/// Longest input line (characters). Caption lines are far shorter; this bounds the helper's work per request.
pub const MAX_TEXT_CHARS: usize = 5000;
/// Memory the loaded directions may use inside the helper (its job is capped at 1.5 GB by the recipe).
pub const MODEL_BUDGET_MB: usize = 1200;
/// Marian's per-worker workspace (`workspace: 128` in the helper's model config): every loaded direction keeps one
/// replica, and so one workspace, per worker thread.
pub const WORKSPACE_MB: usize = 128;
/// A typical Mozilla direction (model, shortlist and vocabularies: 36-70 MB).
pub const DIRECTION_MB: usize = 64;
/// A model whose load ended the helper this many times is refused until it is reinstalled or the engine stops.
pub const MAX_LOAD_CRASHES: u32 = 2;

/// Memory one loaded direction takes in the helper with `threads` workers (the model plus a workspace per worker).
pub fn direction_mb(threads: u32) -> usize {
    DIRECTION_MB + WORKSPACE_MB * threads.max(1) as usize
}

/// How many directions fit in [`MODEL_BUDGET_MB`] with `threads` workers (never fewer than the two legs of a pivot).
/// This is also the peak: loads make room before they start.
pub fn loaded_limit(configured: usize, threads: u32) -> usize {
    configured.min((MODEL_BUDGET_MB / direction_mb(threads)).max(2))
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

// ---------------------------------------------------------------------------------------------
// Environment and configuration
// ---------------------------------------------------------------------------------------------

/// Where the engine gets its directions, installed models and helper from.
pub trait Environment: Send + Sync + 'static {
    /// The catalog's Bergamot directions.
    fn directions(&self) -> Arc<Directions>;
    /// Install folder of a catalog id, when installed.
    fn installed_path(&self, id: &str) -> Option<PathBuf>;
    /// The helper to start (runtime pack checked); `not_installed` names the runtime id when it is missing.
    /// May block briefly (hashes the executable once per session).
    fn helper_spec(&self, name: &str, threads: u32) -> Result<HelperSpec, MtError>;
}

/// Limits and timeouts.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// `process_host` helper name.
    pub helper_name: String,
    /// Directions kept loaded in the helper (least recently used ones are unloaded to make room beyond this, or
    /// beyond what [`loaded_limit`] allows for the worker count).
    pub max_loaded: usize,
    pub hello_timeout: Duration,
    /// Longest wait for one model load, and for room to load it while lines still use the other models.
    pub load_timeout: Duration,
    pub final_timeout: Duration,
    pub interim_timeout: Duration,
    /// Interims beyond this many in flight fail with `busy` (the next partial supersedes them anyway).
    pub max_interims_in_flight: usize,
    pub max_in_flight: usize,
    /// After the helper failed to start, requests fail fast (`busy`) for this long.
    pub failure_cooldown: Duration,
    /// How long to wait for a helper `process_host` is restarting before starting a fresh one.
    pub restart_wait: Duration,
}

impl EngineConfig {
    pub fn production() -> EngineConfig {
        EngineConfig {
            helper_name: crate::services::process_host::recipes::BERGAMOT_NAME.into(),
            max_loaded: 6,
            hello_timeout: Duration::from_secs(20),
            load_timeout: Duration::from_secs(90),
            final_timeout: Duration::from_secs(30),
            interim_timeout: Duration::from_secs(10),
            max_interims_in_flight: 3,
            max_in_flight: 64,
            failure_cooldown: Duration::from_secs(30),
            restart_wait: Duration::from_secs(8),
        }
    }
}

/// Status callback (installed by the Tauri commands; absent in tests and headless runs).
pub type StatusSink = Box<dyn Fn(&MtEngineStatus) + Send + Sync>;

// ---------------------------------------------------------------------------------------------
// Cancellation
// ---------------------------------------------------------------------------------------------

/// Cancellation flag of one in-flight request.
#[derive(Debug)]
pub struct CancelFlag {
    tx: watch::Sender<bool>,
    /// Engine generation the request began in (a stop cancels the requests of earlier generations only).
    generation: u64,
}

impl CancelFlag {
    fn new(generation: u64) -> Self {
        Self {
            tx: watch::channel(false).0,
            generation,
        }
    }

    pub fn cancel(&self) {
        self.tx.send_replace(true);
    }

    pub fn is_cancelled(&self) -> bool {
        *self.tx.borrow()
    }

    /// Resolves once cancelled.
    pub async fn cancelled(&self) {
        let mut rx = self.tx.subscribe();
        let _ = rx.wait_for(|v| *v).await;
    }
}

// ---------------------------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum CallError {
    /// The helper process ended or its pipes closed while the request was out.
    Disconnected(String),
    /// The request never reached the helper (the connection was closed, or writing to it failed).
    NotSent(String),
    Timeout,
    Cancelled,
    Helper(HelperError),
}

type Waiter = oneshot::Sender<Result<Value, CallError>>;

/// A direction loaded in the helper.
#[derive(Debug)]
struct Loaded {
    last_used: Instant,
    /// Lines translating with it right now ([`Lease`]s): it is never unloaded while any is.
    users: usize,
}

/// One running helper process.
struct Conn {
    pid: u32,
    threads: u32,
    /// Engine generation the helper was started (or attached) under.
    generation: u64,
    hello: Mutex<HelloInfo>,
    writer: mpsc::Sender<(Option<u64>, String)>,
    pending: Mutex<HashMap<u64, Waiter>>,
    next_id: AtomicU64,
    closed: AtomicBool,
    /// Sigil is ending this helper (a stop, a restart after it hung, a replacement): no new request uses it, and
    /// whatever fails on it from now on is not a model's fault.
    retired: AtomicBool,
    close_reason: Mutex<String>,
    /// Loaded directions by key.
    models: Mutex<HashMap<String, Loaded>>,
    /// Signalled when a line stops using its models and when the connection closes (for a load waiting for room).
    released: Notify,
    load_lock: tokio::sync::Mutex<()>,
}

impl Conn {
    /// Take over the helper's pipes: a writer thread for stdin, a reader thread for stdout. `on_close` runs on the
    /// reader thread when stdout ends.
    fn open(
        io: HelperStdio,
        threads: u32,
        generation: u64,
        on_close: impl FnOnce(&Arc<Conn>, String) + Send + 'static,
    ) -> Result<Arc<Conn>, MtError> {
        let HelperStdio { pid, stdin, stdout } = io;
        let (tx, rx) = mpsc::channel::<(Option<u64>, String)>();
        let conn = Arc::new(Conn {
            pid,
            threads,
            generation,
            hello: Mutex::new(HelloInfo::default()),
            writer: tx,
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            closed: AtomicBool::new(false),
            retired: AtomicBool::new(false),
            close_reason: Mutex::new(String::new()),
            models: Mutex::new(HashMap::new()),
            released: Notify::new(),
            load_lock: tokio::sync::Mutex::new(()),
        });

        let weak = Arc::downgrade(&conn);
        let mut stdin = stdin;
        std::thread::Builder::new()
            .name("bergamot-stdin".into())
            .spawn(move || {
                for (id, line) in rx {
                    let result = stdin
                        .write_all(line.as_bytes())
                        .and_then(|()| stdin.write_all(b"\n"))
                        .and_then(|()| stdin.flush());
                    if let Err(e) = result {
                        let Some(conn) = weak.upgrade() else { break };
                        if let Some(id) = id {
                            conn.fail(id, CallError::NotSent(format!("cannot write to the helper: {e}")));
                        }
                    }
                }
            })
            .map_err(|e| MtError::runtime(format!("cannot start the Bergamot writer thread: {e}")))?;

        let weak = Arc::downgrade(&conn);
        std::thread::Builder::new()
            .name("bergamot-stdout".into())
            .spawn(move || {
                let mut reader = BufReader::with_capacity(64 * 1024, stdout);
                let reason = loop {
                    match protocol::read_line_capped(&mut reader, MAX_LINE_BYTES) {
                        Ok(Line::Text(text)) => match protocol::parse_reply(&text) {
                            Some(reply) => match weak.upgrade() {
                                Some(conn) => conn.dispatch(reply.id, reply.outcome),
                                None => break "the connection was dropped".to_string(),
                            },
                            None if text.trim().is_empty() => {}
                            None => tracing::debug!(target: LOG_TARGET, "ignoring a non-protocol line from the helper ({} bytes)", text.len()),
                        },
                        Ok(Line::TooLong) => tracing::warn!(target: LOG_TARGET, "the helper sent a line over {MAX_LINE_BYTES} bytes; ignored"),
                        Ok(Line::Eof) => break "the helper process ended".to_string(),
                        Err(e) => break format!("cannot read from the helper: {e}"),
                    }
                };
                if let Some(conn) = weak.upgrade() {
                    conn.close(&reason);
                    on_close(&conn, reason);
                }
            })
            .map_err(|e| MtError::runtime(format!("cannot start the Bergamot reader thread: {e}")))?;
        Ok(conn)
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    fn is_retired(&self) -> bool {
        self.retired.load(Ordering::SeqCst)
    }

    /// Mark the helper as ended by Sigil (before stopping it).
    fn retire(&self) {
        self.retired.store(true, Ordering::SeqCst);
    }

    /// Open and not being ended by Sigil: new requests may use it.
    fn usable(&self) -> bool {
        !self.is_closed() && !self.is_retired()
    }

    fn close_reason(&self) -> String {
        let r = lock(&self.close_reason).clone();
        if r.is_empty() {
            "the helper connection closed".into()
        } else {
            r
        }
    }

    /// Mark closed and fail every pending call.
    fn close(&self, reason: &str) {
        {
            let mut r = lock(&self.close_reason);
            if r.is_empty() {
                *r = reason.to_string();
            }
        }
        self.closed.store(true, Ordering::SeqCst);
        let drained: Vec<Waiter> = lock(&self.pending).drain().map(|(_, w)| w).collect();
        for w in drained {
            let _ = w.send(Err(CallError::Disconnected(reason.to_string())));
        }
        self.released.notify_waiters();
    }

    fn dispatch(&self, id: Option<u64>, outcome: Result<Value, HelperError>) {
        let Some(id) = id else {
            if let Err(e) = outcome {
                tracing::warn!(target: LOG_TARGET, "the helper rejected a message: {} ({})", e.message, e.kind);
            }
            return;
        };
        let waiter = lock(&self.pending).remove(&id);
        if let Some(w) = waiter {
            let _ = w.send(outcome.map_err(CallError::Helper));
        }
    }

    fn fail(&self, id: u64, error: CallError) {
        let waiter = lock(&self.pending).remove(&id);
        if let Some(w) = waiter {
            let _ = w.send(Err(error));
        }
    }

    /// Send a request whose reply is awaited through the returned [`Pending`].
    fn send(self: &Arc<Self>, body: Value) -> Result<Pending, CallError> {
        if self.is_closed() {
            return Err(CallError::NotSent(self.close_reason()));
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        lock(&self.pending).insert(id, tx);
        let line = protocol::encode(id, body);
        if self.writer.send((Some(id), line)).is_err() || self.is_closed() {
            lock(&self.pending).remove(&id);
            return Err(CallError::NotSent(self.close_reason()));
        }
        Ok(Pending {
            conn: self.clone(),
            id,
            rx: Some(rx),
            settled: false,
        })
    }

    /// Send a request whose reply is ignored (`unload` to make room, `cancel`).
    fn send_untracked(&self, body: Value) {
        if self.is_closed() {
            return;
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let _ = self.writer.send((None, protocol::encode(id, body)));
    }

    fn is_loaded(&self, key: &str) -> bool {
        lock(&self.models).contains_key(key)
    }

    fn mark_loaded(&self, key: &str) {
        let now = Instant::now();
        lock(&self.models)
            .entry(key.to_string())
            .and_modify(|m| m.last_used = now)
            .or_insert(Loaded { last_used: now, users: 0 });
    }

    fn forget(&self, keys: &[String]) {
        let mut models = lock(&self.models);
        for k in keys {
            models.remove(k);
        }
    }

    /// Hold `keys` for one line when all of them are loaded (none is unloaded until the lease is dropped).
    fn lease(self: &Arc<Self>, keys: &[String]) -> Option<Lease> {
        let mut models = lock(&self.models);
        if !keys.iter().all(|k| models.contains_key(k)) {
            return None;
        }
        for k in keys {
            if let Some(m) = models.get_mut(k) {
                m.users += 1;
            }
        }
        Some(Lease {
            conn: self.clone(),
            keys: keys.to_vec(),
        })
    }

    fn release(&self, keys: &[String]) {
        {
            let mut models = lock(&self.models);
            let now = Instant::now();
            for k in keys {
                if let Some(m) = models.get_mut(k) {
                    m.users = m.users.saturating_sub(1);
                    m.last_used = now;
                }
            }
        }
        self.released.notify_waiters();
    }

    /// The directions to unload so that `adding` more fit in `limit`: least recently used first, never one of
    /// `keep`, never one a line is using. They are removed from the loaded set when there is enough room; `None`
    /// (nothing removed) while lines still use too many of the others.
    fn take_victims(&self, limit: usize, keep: &[String], adding: usize) -> Option<Vec<String>> {
        let mut models = lock(&self.models);
        let excess = (models.len() + adding).saturating_sub(limit);
        if excess == 0 {
            return Some(Vec::new());
        }
        let mut idle: Vec<(&String, Instant)> = models
            .iter()
            .filter(|(k, m)| m.users == 0 && !keep.contains(k))
            .map(|(k, m)| (k, m.last_used))
            .collect();
        if idle.len() < excess {
            return None;
        }
        idle.sort_by_key(|(_, t)| *t);
        let victims: Vec<String> = idle
            .iter()
            .take(excess)
            .map(|(k, _)| (*k).clone())
            .collect();
        for v in &victims {
            models.remove(v);
        }
        Some(victims)
    }

    fn loaded_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = lock(&self.models).keys().cloned().collect();
        keys.sort();
        keys
    }
}

/// One line's hold on the directions it translates with (released on drop).
struct Lease {
    conn: Arc<Conn>,
    keys: Vec<String>,
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.conn.release(&self.keys);
    }
}

/// A sent request. Dropping it before the reply arrived (timeout, cancel, a dropped future) forgets the request and
/// asks the helper to drop it if it is still queued.
struct Pending {
    conn: Arc<Conn>,
    id: u64,
    rx: Option<oneshot::Receiver<Result<Value, CallError>>>,
    settled: bool,
}

impl Pending {
    async fn wait(mut self, timeout: Duration, cancel: Option<&CancelFlag>) -> Result<Value, CallError> {
        let Some(rx) = self.rx.take() else {
            return Err(CallError::Disconnected("no reply channel".into()));
        };
        let reply = async {
            match tokio::time::timeout(timeout, rx).await {
                Ok(Ok(outcome)) => outcome,
                Ok(Err(_)) => Err(CallError::Disconnected(self.conn.close_reason())),
                Err(_) => Err(CallError::Timeout),
            }
        };
        let result = match cancel {
            Some(flag) => {
                tokio::select! {
                    biased;
                    _ = flag.cancelled() => Err(CallError::Cancelled),
                    r = reply => r,
                }
            }
            None => reply.await,
        };
        if !matches!(result, Err(CallError::Timeout | CallError::Cancelled)) {
            self.settled = true;
        }
        result
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        let was_pending = lock(&self.conn.pending).remove(&self.id).is_some();
        if was_pending {
            self.conn.send_untracked(protocol::cancel(self.id));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------------------------

/// In-flight accounting (released on drop).
struct Slot<'a> {
    engine: &'a Engine,
    interim: bool,
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.engine.in_flight.fetch_sub(1, Ordering::SeqCst);
        if self.interim {
            self.engine
                .interims_in_flight
                .fetch_sub(1, Ordering::SeqCst);
        }
    }
}

/// Registration of a request's cancel flag (removed on drop).
struct Registration<'a> {
    engine: &'a Engine,
    key: String,
    flag: Arc<CancelFlag>,
}

impl Drop for Registration<'_> {
    fn drop(&mut self) {
        let mut map = lock(&self.engine.requests);
        if let Some(list) = map.get_mut(&self.key) {
            list.retain(|f| !Arc::ptr_eq(f, &self.flag));
            if list.is_empty() {
                map.remove(&self.key);
            }
        }
    }
}

/// Identity of an installed model file: a reinstall (another file, size or modification time) starts its load-crash
/// count over.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ModelStamp {
    path: PathBuf,
    len: u64,
    modified: Option<SystemTime>,
}

impl ModelStamp {
    fn of(files: &ModelFiles) -> ModelStamp {
        let meta = std::fs::metadata(&files.model).ok();
        ModelStamp {
            path: files.model.clone(),
            len: meta.as_ref().map_or(0, |m| m.len()),
            modified: meta.and_then(|m| m.modified().ok()),
        }
    }
}

pub struct Engine {
    env: Arc<dyn Environment>,
    cfg: EngineConfig,
    me: Weak<Engine>,
    conn: Mutex<Option<Arc<Conn>>>,
    /// Serializes connecting and stopping.
    lifecycle: tokio::sync::Mutex<()>,
    /// Bumped by every stop: requests that began before it are cancelled and never restart the helper, and the
    /// helpers started before it are ended by it.
    generation: AtomicU64,
    /// Worker threads for the next start.
    threads: AtomicU32,
    /// The helper `process_host` runs for us: the generation and worker count it was started with (requests of
    /// that generation attach to it again after the process host restarted it).
    helper_run: Mutex<Option<(u64, u32)>>,
    status: Mutex<MtEngineStatus>,
    emitter: OnceLock<StatusSink>,
    requests: Mutex<HashMap<String, Vec<Arc<CancelFlag>>>>,
    anon_requests: AtomicU64,
    in_flight: AtomicUsize,
    interims_in_flight: AtomicUsize,
    failed_until: Mutex<Option<(Instant, String)>>,
    /// Directions whose load ended the helper on its own: the model file it happened with, and how often.
    load_crashes: Mutex<HashMap<String, (ModelStamp, u32)>>,
}

/// What a failed load or connect means for the request.
enum Step {
    Retry(String),
    Fail(MtError),
}

impl Engine {
    pub fn new(env: Arc<dyn Environment>, cfg: EngineConfig) -> Arc<Engine> {
        Arc::new_cyclic(|me| Engine {
            env,
            cfg,
            me: me.clone(),
            conn: Mutex::new(None),
            lifecycle: tokio::sync::Mutex::new(()),
            generation: AtomicU64::new(0),
            threads: AtomicU32::new(thread_count(None)),
            helper_run: Mutex::new(None),
            status: Mutex::new(MtEngineStatus {
                engine: ENGINE_BERGAMOT.into(),
                state: MtEngineState::Idle,
                device: Some("cpu".into()),
                ..Default::default()
            }),
            emitter: OnceLock::new(),
            requests: Mutex::new(HashMap::new()),
            anon_requests: AtomicU64::new(0),
            in_flight: AtomicUsize::new(0),
            interims_in_flight: AtomicUsize::new(0),
            failed_until: Mutex::new(None),
            load_crashes: Mutex::new(HashMap::new()),
        })
    }

    pub fn set_emitter(&self, sink: StatusSink) {
        let _ = self.emitter.set(sink);
    }

    #[cfg(test)]
    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

    pub fn status(&self) -> MtEngineStatus {
        lock(&self.status).clone()
    }

    /// `bergamot_pairs`.
    pub fn pairs(&self) -> Vec<MtPair> {
        let env = &self.env;
        env.directions()
            .pairs(|id| env.installed_path(id).is_some())
    }

    fn update_status(&self, f: impl FnOnce(&mut MtEngineStatus)) {
        let snapshot = {
            let mut s = lock(&self.status);
            let before = s.clone();
            f(&mut s);
            if *s == before {
                return;
            }
            s.clone()
        };
        if let Some(emit) = self.emitter.get() {
            emit(&snapshot);
        }
    }

    fn set_state(&self, state: MtEngineState, detail: Option<String>) {
        self.update_status(|s| {
            s.state = state;
            s.detail = detail;
        });
    }

    fn ready_status(&self, conn: &Conn) {
        let keys = conn.loaded_keys();
        let described: Vec<String> = keys.iter().map(|k| key_label(k)).collect();
        let hello = lock(&conn.hello).clone();
        self.update_status(|s| {
            s.state = MtEngineState::Ready;
            s.model = (!described.is_empty()).then(|| described.join(", "));
            s.threads = Some(conn.threads);
            s.device = Some("cpu".into());
            s.detail = (!hello.helper.is_empty()).then(|| hello.helper.clone());
        });
    }

    fn current(&self) -> Option<Arc<Conn>> {
        lock(&self.conn).clone()
    }

    /// The current connection when requests of `generation` can use it as it is.
    fn ready_conn(&self, generation: u64) -> Option<Arc<Conn>> {
        self.current()
            .filter(|c| c.usable() && c.generation == generation)
    }

    /// Called on the reader thread when a helper's stdout ends (by connection, not pid: Windows reuses pids).
    fn on_conn_closed(&self, conn: &Arc<Conn>, reason: String) {
        let pid = conn.pid;
        let was_current = {
            let mut cur = lock(&self.conn);
            if cur.as_ref().is_some_and(|c| Arc::ptr_eq(c, conn)) {
                *cur = None;
                true
            } else {
                false
            }
        };
        if was_current {
            tracing::warn!(target: LOG_TARGET, "the Bergamot helper (pid {pid}) stopped: {reason}");
            self.update_status(|s| {
                s.state = MtEngineState::Error;
                s.model = None;
                s.detail = Some(format!("The Bergamot helper stopped ({reason}); it starts again with the next line."));
            });
        }
    }

    fn check_cooldown(&self) -> Result<(), MtError> {
        let mut guard = lock(&self.failed_until);
        if let Some((until, reason)) = guard.as_ref() {
            let now = Instant::now();
            if now < *until {
                let secs = until.duration_since(now).as_secs().max(1);
                return Err(MtError::new(
                    MtErrorKind::Busy,
                    format!("the Bergamot helper failed to start ({reason}); trying again in {secs} s"),
                ));
            }
            *guard = None;
        }
        Ok(())
    }

    /// The live connection of `generation`, starting (or re-attaching to) the helper when needed. `threads`: the
    /// worker count to start with; `restart_on_change` restarts a running helper that uses a different count
    /// (`bergamot_start`).
    async fn connection(&self, threads: Option<u32>, restart_on_change: bool, generation: u64) -> Result<Arc<Conn>, MtError> {
        let wanted = threads.map(|t| thread_count(Some(t)));
        let acceptable = |c: &Conn| c.usable() && c.generation == generation && (!restart_on_change || wanted.is_none_or(|w| w == c.threads));
        if let Some(c) = self.current().filter(|c| acceptable(c)) {
            return Ok(c);
        }
        let _guard = self.lifecycle.lock().await;
        if self.generation.load(Ordering::SeqCst) != generation {
            return Err(MtError::cancelled());
        }
        let stale = self.current();
        if let Some(c) = stale.as_ref().filter(|c| acceptable(c)) {
            return Ok(c.clone());
        }
        self.check_cooldown()?;
        let threads = wanted.unwrap_or_else(|| self.threads.load(Ordering::SeqCst));
        self.threads.store(threads, Ordering::SeqCst);
        // A running helper that does not fit (another thread count) is replaced: Sigil ends it, so what fails on it
        // from now on is not a model's fault.
        let replace = stale.as_ref().is_some_and(|c| !c.is_closed());
        if let Some(c) = &stale {
            c.retire();
        }
        *lock(&self.conn) = None;
        self.set_state(MtEngineState::Loading, Some("Starting the Bergamot helper…".into()));
        match self.open(threads, replace, generation).await {
            Ok(conn) => {
                if self.generation.load(Ordering::SeqCst) != generation {
                    // Stopped while starting: end the helper this start brought up (no start of the new generation
                    // runs while this one holds the lifecycle lock).
                    conn.retire();
                    conn.close("stopped");
                    let _ = process_host::stop(&self.cfg.helper_name).await;
                    *lock(&self.helper_run) = None;
                    return Err(MtError::cancelled());
                }
                *lock(&self.conn) = Some(conn.clone());
                *lock(&self.failed_until) = None;
                self.ready_status(&conn);
                Ok(conn)
            }
            Err(_) if self.generation.load(Ordering::SeqCst) != generation => {
                // The start may have brought a helper up before it failed: end it too.
                let _ = process_host::stop(&self.cfg.helper_name).await;
                *lock(&self.helper_run) = None;
                Err(MtError::cancelled())
            }
            Err(e) => {
                self.update_status(|s| {
                    s.state = MtEngineState::Error;
                    s.model = None;
                    s.detail = Some(e.message.clone());
                });
                Err(e)
            }
        }
    }

    /// Attach to the running helper (after `process_host` restarted it) or start one, then say hello.
    async fn open(&self, threads: u32, replace: bool, generation: u64) -> Result<Arc<Conn>, MtError> {
        let name = self.cfg.helper_name.clone();
        let env = self.env.clone();
        let spec_name = name.clone();
        let spec = tokio::task::spawn_blocking(move || env.helper_spec(&spec_name, threads))
            .await
            .map_err(|e| MtError::runtime(format!("the Bergamot start task failed: {e}")))??;

        let stopped = || self.generation.load(Ordering::SeqCst) != generation;
        let ours = *lock(&self.helper_run) == Some((generation, threads));
        if !replace && ours {
            let deadline = Instant::now() + self.cfg.restart_wait;
            loop {
                if stopped() {
                    return Err(MtError::cancelled());
                }
                match process_host::get(&name).map(|i| i.state) {
                    Some(HelperState::Running) => {
                        if let Some(io) = process_host::take_stdio(&name) {
                            return self.handshake(io, threads, generation).await;
                        }
                    }
                    Some(HelperState::Starting | HelperState::Restarting) => {}
                    _ => break,
                }
                if Instant::now() >= deadline {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }

        if stopped() {
            return Err(MtError::cancelled());
        }
        if let Err(e) = process_host::spawn(spec).await {
            if stopped() {
                // A stop cancelled the start: not a failure.
                return Err(MtError::cancelled());
            }
            let reason = e.trim().to_string();
            *lock(&self.failed_until) = Some((Instant::now() + self.cfg.failure_cooldown, first_line(&reason)));
            return Err(MtError::runtime(format!("the Bergamot helper could not start: {reason}")));
        }
        *lock(&self.helper_run) = Some((generation, threads));
        let io = process_host::take_stdio(&name).ok_or_else(|| MtError::runtime("the Bergamot helper stopped right after starting"))?;
        self.handshake(io, threads, generation).await
    }

    async fn handshake(&self, io: HelperStdio, threads: u32, generation: u64) -> Result<Arc<Conn>, MtError> {
        let me = self.me.clone();
        let conn = Conn::open(io, threads, generation, move |conn, reason| {
            if let Some(engine) = me.upgrade() {
                engine.on_conn_closed(conn, reason);
            }
        })?;
        let reply = match conn.send(protocol::hello()) {
            Ok(p) => p.wait(self.cfg.hello_timeout, None).await,
            Err(e) => Err(e),
        };
        let failure = match reply {
            Ok(v) => {
                let hello: HelloInfo = serde_json::from_value(v).unwrap_or_default();
                if hello.protocol == PROTOCOL {
                    tracing::info!(target: LOG_TARGET, "Bergamot helper ready: {} ({}), {} thread(s), pid {}", hello.helper, hello.engine, hello.threads, conn.pid);
                    *lock(&conn.hello) = hello;
                    return Ok(conn);
                }
                format!(
                    "the installed Bergamot helper speaks protocol {}, this Sigil needs protocol {PROTOCOL}: rebuild it with scripts/runtime-packs/bergamot/build.ps1 and import it again",
                    hello.protocol
                )
            }
            Err(CallError::Timeout) => format!("the Bergamot helper did not answer within {} s", self.cfg.hello_timeout.as_secs()),
            Err(CallError::Disconnected(r) | CallError::NotSent(r)) => format!("the Bergamot helper stopped while starting ({r})"),
            Err(CallError::Helper(e)) => format!("the Bergamot helper refused to start: {}", e.message),
            Err(CallError::Cancelled) => return Err(MtError::cancelled()),
        };
        conn.retire();
        conn.close("handshake failed");
        let _ = process_host::stop(&self.cfg.helper_name).await;
        *lock(&self.helper_run) = None;
        Err(MtError::runtime(failure))
    }

    /// Stop a helper that stopped answering (the next request starts a fresh one).
    async fn restart_stuck(&self, conn: &Arc<Conn>, what: &str) {
        tracing::warn!(target: LOG_TARGET, "the Bergamot helper (pid {}) did not answer {what}; restarting it", conn.pid);
        // Loads still waiting on it fail from here on; that is not their model's fault.
        conn.retire();
        let _guard = self.lifecycle.lock().await;
        let is_current = self.current().is_some_and(|c| Arc::ptr_eq(&c, conn));
        if is_current {
            *lock(&self.conn) = None;
            let _ = process_host::stop(&self.cfg.helper_name).await;
            *lock(&self.helper_run) = None;
            self.update_status(|s| {
                s.state = MtEngineState::Error;
                s.model = None;
                s.detail = Some(format!("The Bergamot helper stopped answering {what} and was restarted."));
            });
        }
        conn.close("stopped: not answering");
    }

    /// `bergamot_start`: start (or keep) the helper with `threads` workers.
    pub async fn start(&self, threads: Option<u32>) -> Result<MtEngineStatus, MtError> {
        let generation = self.generation.load(Ordering::SeqCst);
        let wanted = Some(thread_count(threads));
        self.connection(wanted, true, generation).await?;
        Ok(self.status())
    }

    /// `bergamot_stop`: cancel everything in flight and stop the helper. Only the helpers started before this stop
    /// are ended: a start right after it (a pipeline restart) keeps its helper.
    pub async fn stop(&self) {
        let name = self.cfg.helper_name.clone();
        // The helper this stop ends: whatever fails on it from now on is not a model's fault.
        let old = self.current();
        if let Some(c) = &old {
            c.retire();
        }
        // Register the stop with the process host before the generation changes. The host lets the last request
        // win, so every start of the new generation comes after this stop, and this stop also aborts a start of the
        // old generation that is still waiting for its helper (a stop never waits out a slow start).
        let mut stopping = std::pin::pin!(process_host::stop(&name));
        let stopped = std::future::poll_fn(|cx| Poll::Ready(stopping.as_mut().poll(cx).is_ready())).await;
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        self.cancel_older(generation);
        lock(&self.load_crashes).clear();
        *lock(&self.failed_until) = None;
        {
            let mut cur = lock(&self.conn);
            if cur.as_ref().is_some_and(|c| c.generation < generation) {
                *cur = None;
            }
        }
        if let Some(c) = &old {
            c.close("stopped");
        }
        if !stopped {
            let _ = stopping.await;
        }
        // A start of the old generation still in progress ends its own helper; wait for it. One that made its
        // helper current after the stop was registered is ended here.
        let _guard = self.lifecycle.lock().await;
        if self.generation.load(Ordering::SeqCst) != generation {
            return; // a later stop finishes the job
        }
        if let Some(c) = self.current() {
            if c.generation >= generation {
                return; // a newer start owns the helper and the status
            }
            *lock(&self.conn) = None;
            c.retire();
            c.close("stopped");
            let _ = process_host::stop(&name).await;
        }
        self.update_status(|s| {
            s.state = MtEngineState::Idle;
            s.model = None;
            s.detail = None;
        });
    }

    /// `bergamot_cancel`: one request id, or every request in flight.
    pub fn cancel(&self, request_id: Option<&str>) {
        let flags: Vec<Arc<CancelFlag>> = {
            let map = lock(&self.requests);
            match request_id {
                Some(id) => map.get(id).cloned().unwrap_or_default(),
                None => map.values().flatten().cloned().collect(),
            }
        };
        for f in flags {
            f.cancel();
        }
    }

    /// Cancel the requests that began before `generation` (a stop).
    fn cancel_older(&self, generation: u64) {
        let flags: Vec<Arc<CancelFlag>> = lock(&self.requests)
            .values()
            .flatten()
            .filter(|f| f.generation < generation)
            .cloned()
            .collect();
        for f in flags {
            f.cancel();
        }
    }

    fn register(&self, request_id: Option<&str>, generation: u64) -> Registration<'_> {
        let key = match request_id.map(str::trim).filter(|s| !s.is_empty()) {
            Some(id) => id.to_string(),
            None => format!("\u{0}anon-{}", self.anon_requests.fetch_add(1, Ordering::SeqCst)),
        };
        let flag = Arc::new(CancelFlag::new(generation));
        lock(&self.requests)
            .entry(key.clone())
            .or_default()
            .push(flag.clone());
        Registration { engine: self, key, flag }
    }

    fn admit(&self, interim: bool) -> Result<Slot<'_>, MtError> {
        let total = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        let interims = if interim {
            self.interims_in_flight.fetch_add(1, Ordering::SeqCst) + 1
        } else {
            0
        };
        // The guard owns both counters from here on: every early return releases them.
        let slot = Slot { engine: self, interim };
        if total > self.cfg.max_in_flight {
            return Err(MtError::new(MtErrorKind::Busy, "too many Bergamot requests in flight"));
        }
        if interim && interims > self.cfg.max_interims_in_flight {
            return Err(MtError::new(MtErrorKind::Busy, "Bergamot is still translating earlier partial lines"));
        }
        Ok(slot)
    }

    /// `bergamot_translate`.
    pub async fn translate(&self, request: MtRequest) -> Result<MtResponse, MtError> {
        let started = Instant::now();
        let generation = self.generation.load(Ordering::SeqCst);
        let text = request.text.trim();
        if text.is_empty() {
            return Err(MtError::new(MtErrorKind::InvalidInput, "nothing to translate"));
        }
        if text.chars().count() > MAX_TEXT_CHARS {
            return Err(MtError::new(
                MtErrorKind::InvalidInput,
                format!("the line is longer than {MAX_TEXT_CHARS} characters"),
            ));
        }
        let directions = self.env.directions();
        let (legs, pivot) = match directions.route(&request.source, &request.target)? {
            Route::Same => {
                return Ok(MtResponse {
                    text: text.to_string(),
                    engine: ENGINE_BERGAMOT.into(),
                    model: String::new(),
                    elapsed_ms: 0,
                    pivot: false,
                })
            }
            Route::Models { legs, pivot } => (legs, pivot),
        };
        let files = self.resolve(&legs)?;
        let Some(engine) = self.me.upgrade() else {
            return Err(MtError::cancelled());
        };
        let interim = request.kind == MtKind::Interim;
        let _slot = self.admit(interim)?;
        let registration = self.register(request.request_id.as_deref(), generation);
        let flag = registration.flag.clone();
        let keys: Vec<String> = legs.iter().map(|d| d.key.clone()).collect();
        let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        let priority = if interim { Priority::Interim } else { Priority::Final };
        let timeout = if interim { self.cfg.interim_timeout } else { self.cfg.final_timeout };
        // Finals get one more attempt after the helper crashed or restarted; interims are superseded anyway.
        let attempts = if interim { 1 } else { 2 };

        let mut attempt = 0;
        loop {
            attempt += 1;
            if flag.is_cancelled() || self.generation.load(Ordering::SeqCst) != generation {
                return Err(MtError::cancelled());
            }
            // Starting the helper runs on its own task: a cancelled line returns at once, and the start it began
            // completes for the lines after it (a start dropped half-way would leave a helper nobody can attach to).
            let conn = match self.ready_conn(generation) {
                Some(c) => c,
                None => {
                    let (e, threads) = (engine.clone(), request.threads);
                    let connecting = tokio::spawn(async move { e.connection(threads, false, generation).await });
                    let joined = tokio::select! {
                        biased;
                        _ = flag.cancelled() => return Err(MtError::cancelled()),
                        r = connecting => r,
                    };
                    joined.map_err(|e| MtError::runtime(format!("the Bergamot start task failed: {e}")))??
                }
            };
            // Hold the line's models (they are not unloaded while it translates), loading them first when needed.
            // The load runs on its own task too: a cancel does not abort a load the helper already started.
            let lease = match conn.lease(&keys) {
                Some(lease) => lease,
                None => {
                    let loading = tokio::spawn(
                        engine
                            .clone()
                            .load_models(conn.clone(), legs.clone(), files.clone(), generation, flag.clone()),
                    );
                    let joined = tokio::select! {
                        biased;
                        _ = flag.cancelled() => return Err(MtError::cancelled()),
                        r = loading => r,
                    };
                    match joined.map_err(|e| MtError::runtime(format!("the Bergamot load task failed: {e}")))? {
                        Ok(lease) => lease,
                        Err(Step::Retry(reason)) if attempt < attempts => {
                            tracing::info!(target: LOG_TARGET, "retrying a line after the helper stopped: {reason}");
                            continue;
                        }
                        Err(Step::Retry(reason)) => {
                            return Err(MtError::runtime(format!("the Bergamot helper stopped while loading a model ({reason})")))
                        }
                        Err(Step::Fail(e)) => return Err(e),
                    }
                }
            };
            if flag.is_cancelled() {
                return Err(MtError::cancelled());
            }
            let result = match conn.send(protocol::translate(&key_refs, text, priority)) {
                Ok(p) => p.wait(timeout, Some(&flag)).await,
                Err(e) => Err(e),
            };
            drop(lease);
            match result {
                Ok(v) => {
                    let t: Translated =
                        serde_json::from_value(v).map_err(|e| MtError::runtime(format!("the Bergamot helper sent an unreadable reply: {e}")))?;
                    return Ok(MtResponse {
                        text: t.text.trim().to_string(),
                        engine: ENGINE_BERGAMOT.into(),
                        model: catalog::describe(&legs),
                        elapsed_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
                        pivot,
                    });
                }
                Err(CallError::Cancelled) => return Err(MtError::cancelled()),
                Err(CallError::Helper(e)) if e.is("not_loaded") && attempt < 2 => {
                    // Lost between the check and the request: load again.
                    conn.forget(&keys);
                    continue;
                }
                Err(CallError::Helper(e)) => return Err(e.to_mt()),
                Err(CallError::Disconnected(reason) | CallError::NotSent(reason)) if attempt < attempts => {
                    tracing::info!(target: LOG_TARGET, "retrying a final line after the helper stopped: {reason}");
                    continue;
                }
                Err(CallError::Disconnected(reason) | CallError::NotSent(reason)) => {
                    return Err(MtError::runtime(format!("the Bergamot helper stopped while translating ({reason})")))
                }
                Err(CallError::Timeout) => {
                    self.restart_stuck(&conn, "a translation").await;
                    return Err(MtError::runtime(format!(
                        "Bergamot did not answer within {} s; the helper was restarted",
                        timeout.as_secs()
                    )));
                }
            }
        }
    }

    /// Install folders and model files of the legs; `not_installed` names every missing id.
    fn resolve(&self, legs: &[Direction]) -> Result<Vec<ModelFiles>, MtError> {
        let mut missing = Vec::new();
        let mut files = Vec::new();
        for leg in legs {
            match self.env.installed_path(&leg.id) {
                Some(dir) => match catalog::model_files(leg, &dir) {
                    Ok(f) => files.push(f),
                    Err(e) if e.kind == MtErrorKind::NotInstalled => missing.push(leg.id.clone()),
                    Err(e) => return Err(e),
                },
                None => missing.push(leg.id.clone()),
            }
        }
        if missing.is_empty() {
            Ok(files)
        } else {
            Err(MtError::not_installed(missing))
        }
    }

    fn load_crash_count(&self, key: &str, stamp: &ModelStamp) -> u32 {
        lock(&self.load_crashes)
            .get(key)
            .filter(|(s, _)| s == stamp)
            .map_or(0, |(_, n)| *n)
    }

    fn note_load_crash(&self, key: &str, stamp: &ModelStamp) {
        let mut crashes = lock(&self.load_crashes);
        let entry = crashes
            .entry(key.to_string())
            .or_insert_with(|| (stamp.clone(), 0));
        if entry.0 != *stamp {
            *entry = (stamp.clone(), 0);
        }
        entry.1 += 1;
    }

    /// Load the directions of `legs` that `conn` is missing and hold all of them for one line. Runs on its own task
    /// (see [`Engine::translate`]): a load the helper already started completes even when the line was cancelled,
    /// since the lines after it need the model too; no further load starts for a cancelled line.
    async fn load_models(
        self: Arc<Self>,
        conn: Arc<Conn>,
        legs: Vec<Direction>,
        files: Vec<ModelFiles>,
        generation: u64,
        flag: Arc<CancelFlag>,
    ) -> Result<Lease, Step> {
        let keys: Vec<String> = legs.iter().map(|d| d.key.clone()).collect();
        let _guard = conn.load_lock.lock().await;
        if let Some(lease) = conn.lease(&keys) {
            return Ok(lease);
        }
        if flag.is_cancelled() {
            return Err(Step::Fail(MtError::cancelled()));
        }
        let missing: Vec<(&Direction, &ModelFiles, ModelStamp)> = legs
            .iter()
            .zip(&files)
            .filter(|(leg, _)| !conn.is_loaded(&leg.key))
            .map(|(leg, f)| (leg, f, ModelStamp::of(f)))
            .collect();
        for (leg, _, stamp) in &missing {
            if self.load_crash_count(&leg.key, stamp) >= MAX_LOAD_CRASHES {
                return Err(Step::Fail(MtError::runtime(format!(
                    "the {} model stopped the Bergamot helper twice while loading; remove and download it again in Models & Runtimes",
                    key_label(&leg.key)
                ))));
            }
        }
        // Make room first: the helper's job has a hard memory cap, so it never holds more than the limit, not even
        // while loading (a pivot's two legs always fit).
        let limit = loaded_limit(self.cfg.max_loaded, conn.threads).max(legs.len());
        self.make_room(&conn, limit, &keys, missing.len(), &flag)
            .await?;
        for (i, (leg, f, stamp)) in missing.iter().enumerate() {
            if i > 0 && flag.is_cancelled() {
                return Err(Step::Fail(MtError::cancelled()));
            }
            self.update_status(|s| {
                s.state = MtEngineState::Loading;
                s.detail = Some(format!("Loading the {} model…", key_label(&leg.key)));
            });
            let pending = match conn.send(protocol::load(&leg.key, f)) {
                Ok(p) => p,
                // It never reached the helper: not the model's fault.
                Err(CallError::Disconnected(reason) | CallError::NotSent(reason)) => return Err(Step::Retry(reason)),
                Err(e) => return Err(Step::Retry(format!("{e:?}"))),
            };
            match pending.wait(self.cfg.load_timeout, None).await {
                Ok(_) => {
                    conn.mark_loaded(&leg.key);
                    lock(&self.load_crashes).remove(&leg.key);
                }
                Err(CallError::Disconnected(reason)) => {
                    // Only a helper that ended on its own counts against the model: not one Sigil stopped,
                    // restarted after a hang or replaced.
                    if !conn.is_retired() && self.generation.load(Ordering::SeqCst) == generation {
                        self.note_load_crash(&leg.key, stamp);
                    }
                    return Err(Step::Retry(reason));
                }
                Err(CallError::NotSent(reason)) => return Err(Step::Retry(reason)),
                Err(CallError::Helper(e)) => {
                    self.ready_status(&conn);
                    return Err(Step::Fail(MtError::runtime(format!(
                        "cannot load the {} model: {}",
                        key_label(&leg.key),
                        e.message.trim()
                    ))));
                }
                Err(CallError::Timeout) => {
                    self.restart_stuck(&conn, "a model load").await;
                    return Err(Step::Fail(MtError::runtime(format!(
                        "loading the {} model took over {} s; the helper was restarted",
                        key_label(&leg.key),
                        self.cfg.load_timeout.as_secs()
                    ))));
                }
                Err(CallError::Cancelled) => return Err(Step::Fail(MtError::cancelled())),
            }
        }
        self.ready_status(&conn);
        conn.lease(&keys)
            .ok_or_else(|| Step::Retry("the loaded models were dropped".into()))
    }

    /// Unload least recently used directions until `adding` more fit in `limit`, waiting for the lines that still
    /// use them (at most the load timeout). The unloads go out before the load, and the helper handles them in order.
    async fn make_room(&self, conn: &Arc<Conn>, limit: usize, keep: &[String], adding: usize, flag: &CancelFlag) -> Result<(), Step> {
        let deadline = Instant::now() + self.cfg.load_timeout;
        loop {
            // Created before looking, so a line that finishes in between still ends this wait.
            let released = conn.released.notified();
            if !conn.usable() {
                return Err(Step::Retry(conn.close_reason()));
            }
            if let Some(victims) = conn.take_victims(limit, keep, adding) {
                for v in victims {
                    tracing::debug!(target: LOG_TARGET, "unloading the least recently used model {v} to make room");
                    conn.send_untracked(protocol::unload(&v));
                }
                return Ok(());
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(Step::Fail(MtError::new(
                    MtErrorKind::Busy,
                    "Bergamot is still translating with the other loaded models; try again",
                )));
            }
            tokio::select! {
                _ = released => {}
                _ = flag.cancelled() => return Err(Step::Fail(MtError::cancelled())),
                _ = tokio::time::sleep(left) => {}
            }
        }
    }

    /// The helper's own counters; `None` when it is not running.
    #[cfg(test)]
    pub async fn helper_stats(&self) -> Option<protocol::Stats> {
        let conn = self.current()?;
        let v = conn
            .send(protocol::stats())
            .ok()?
            .wait(Duration::from_secs(5), None)
            .await
            .ok()?;
        serde_json::from_value(v).ok()
    }
}

/// `en_ja` → `en-ja`, `zh-hant_en` → `zh-Hant-en` (status texts).
fn key_label(key: &str) -> String {
    match key.split_once('_') {
        Some((s, t)) => format!("{}-{}", catalog::canonical_case(s), catalog::canonical_case(t)),
        None => key.to_string(),
    }
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or(s).chars().take(200).collect()
}
