//! Process-wide diagnostics log: a size-rotated file sink (10 MiB × 5 files) for every `tracing`
//! event and `log` record, plus forwarding of warnings and errors to the UI system log.
//!
//! - **File**: `<app_log_dir>/sigil.log`, rotated to `sigil.1.log` … `sigil.4.log` when the next
//!   line would push it past [`MAX_FILE_BYTES`]. Writes are synchronous (one `write_all` per event,
//!   no user-space buffer) so the lines before a crash are on disk; the log is low-volume and audio
//!   callbacks never log (plan rule), so a short mutex hold is fine.
//! - **Filter**: `sigil=info,ort=info,warn` by default; override with `SIGIL_LOG` (tracing
//!   `Targets` syntax, e.g. `SIGIL_LOG=debug` or `SIGIL_LOG=sigil=debug,ort=info,warn`).
//! - **UI forwarding**: WARN/ERROR events from Sigil, and ERROR from anything (ONNX Runtime
//!   included; its routine warnings stay in the file, see `forwardable`), are kept in a ring
//!   buffer ([`recent_entries`], so the webview can backfill what happened before it attached)
//!   and sent to the forwarder installed with [`set_forwarder`] (the plugin emits them as
//!   [`LOG_EVENT`]). Forwarding is rate limited and never blocks the logging thread.
//! - **Panics** are logged at ERROR (with thread and location) before the default hook runs.
//! - Debug builds also print to stderr (the `pnpm tauri dev` terminal).

use serde::Serialize;
use std::collections::VecDeque;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, Instant};
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::filter::Targets;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;

/// Tauri event carrying one forwarded [`LogEntry`].
pub const LOG_EVENT: &str = "diagnostics:log";
/// Base name of the log files (`sigil.log`, `sigil.1.log`, …).
pub const LOG_FILE_STEM: &str = "sigil";
/// Rotate before a file grows past this size.
pub const MAX_FILE_BYTES: u64 = 10 * 1024 * 1024;
/// Files kept in total (the current one plus `MAX_FILES - 1` rotated ones).
pub const MAX_FILES: usize = 5;
/// Environment variable overriding the level filter (tracing `Targets` syntax).
pub const LOG_ENV: &str = "SIGIL_LOG";
/// Target used for lines the webview asks to write (never forwarded back to the UI).
pub const FRONTEND_TARGET: &str = "sigil::frontend";

const DEFAULT_FILTER: &str = "sigil=info,ort=info,warn";
/// Warnings/errors kept for the webview's backfill.
const RECENT_CAPACITY: usize = 200;
/// Forwarder queue; entries beyond it are dropped (they are still in the file and ring buffer).
const FORWARD_QUEUE: usize = 256;
/// At most this many entries per [`FORWARD_WINDOW`] reach the UI; the rest are summarized.
const FORWARD_BURST: u32 = 12;
const FORWARD_WINDOW: Duration = Duration::from_secs(2);
/// When a rotation fails (file locked by another program), retry after this many more bytes.
const ROTATE_RETRY_BYTES: u64 = 1024 * 1024;
/// Longest message forwarded to the UI (the file keeps the full line).
const FORWARD_MAX_CHARS: usize = 600;

/// One warning or error forwarded to the UI (payload of [`LOG_EVENT`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct LogEntry {
    /// Monotonic per process; lets the webview de-duplicate backfill and live events.
    pub seq: u64,
    pub at_unix_ms: u64,
    /// `"warn"` or `"error"`.
    pub level: &'static str,
    /// tracing target (module path), e.g. `sigil::services::voice_changer::worker`.
    pub target: String,
    pub message: String,
}

/// Where the log lives (returned by `get_log_info`).
#[derive(Clone, Debug, Serialize)]
pub struct LogInfo {
    pub dir: String,
    pub file: String,
    pub max_file_bytes: u64,
    pub max_files: usize,
    pub filter: String,
}

struct LogState {
    dir: PathBuf,
    file: PathBuf,
    filter: String,
}

static STATE: OnceLock<LogState> = OnceLock::new();
static FORWARD_TX: OnceLock<SyncSender<LogEntry>> = OnceLock::new();
type Forwarder = Box<dyn Fn(&LogEntry) + Send + Sync>;
static FORWARDER: OnceLock<Forwarder> = OnceLock::new();

fn recent() -> &'static Mutex<VecDeque<LogEntry>> {
    static RECENT: OnceLock<Mutex<VecDeque<LogEntry>>> = OnceLock::new();
    RECENT.get_or_init(|| Mutex::new(VecDeque::with_capacity(RECENT_CAPACITY)))
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn unix_ms_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------------------------
// Size-rotated file
// ---------------------------------------------------------------------------------------------

/// Append-only file that rotates `stem.log` → `stem.1.log` → … → `stem.{max_files-1}.log`
/// (oldest dropped) once the next write would exceed `max_bytes`.
pub struct RotatingFile {
    dir: PathBuf,
    stem: String,
    max_bytes: u64,
    max_files: usize,
    file: Option<File>,
    size: u64,
    rotate_at: u64,
}

impl RotatingFile {
    pub fn open(dir: &Path, stem: &str, max_bytes: u64, max_files: usize) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        let mut me = Self {
            dir: dir.to_path_buf(),
            stem: stem.to_string(),
            max_bytes: max_bytes.max(1),
            max_files: max_files.max(1),
            file: None,
            size: 0,
            rotate_at: max_bytes.max(1),
        };
        me.reopen_append()?;
        Ok(me)
    }

    /// `stem.log` for index 0, `stem.<index>.log` otherwise.
    pub fn path_for(&self, index: usize) -> PathBuf {
        if index == 0 {
            self.dir.join(format!("{}.log", self.stem))
        } else {
            self.dir.join(format!("{}.{index}.log", self.stem))
        }
    }

    fn reopen_append(&mut self) -> io::Result<()> {
        let file = OpenOptions::new().create(true).append(true).open(self.path_for(0))?;
        self.size = file.metadata().map(|m| m.len()).unwrap_or(0);
        self.file = Some(file);
        Ok(())
    }

    fn rotate(&mut self) -> io::Result<()> {
        // Close the current file first: Windows cannot rename a file this process holds open
        // without FILE_SHARE_DELETE.
        self.file = None;
        let shifted = (|| -> io::Result<()> {
            if self.max_files > 1 {
                for i in (1..self.max_files - 1).rev() {
                    match fs::rename(self.path_for(i), self.path_for(i + 1)) {
                        Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
                        _ => {}
                    }
                }
                fs::rename(self.path_for(0), self.path_for(1))
            } else {
                fs::remove_file(self.path_for(0))
            }
        })();
        match shifted {
            Ok(()) => {
                let file = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(self.path_for(0))?;
                self.file = Some(file);
                self.size = 0;
                self.rotate_at = self.max_bytes;
                Ok(())
            }
            Err(e) => {
                // Another program holds a rotated file: keep appending to the current one and
                // retry later; as a last resort (twice the cap) truncate so the log stays bounded.
                self.reopen_append()?;
                if self.size >= self.max_bytes.saturating_mul(2) {
                    if let Some(f) = self.file.as_mut() {
                        f.set_len(0)?;
                    }
                    self.size = 0;
                    self.rotate_at = self.max_bytes;
                } else {
                    self.rotate_at = self.size.saturating_add(ROTATE_RETRY_BYTES);
                }
                Err(e)
            }
        }
    }
}

impl Write for RotatingFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.size > 0 && self.size.saturating_add(buf.len() as u64) > self.rotate_at {
            // A failed rotation already reopened the current file; the line still gets written.
            let _ = self.rotate();
        }
        if self.file.is_none() {
            self.reopen_append()?;
        }
        let file = self.file.as_mut().ok_or_else(|| io::Error::other("log file unavailable"))?;
        file.write_all(buf)?;
        self.size = self.size.saturating_add(buf.len() as u64);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        match self.file.as_mut() {
            Some(f) => f.flush(),
            None => Ok(()),
        }
    }
}

/// `MakeWriter` over a shared [`RotatingFile`] that survives a poisoned lock (the stock
/// `Mutex<W>` impl panics on poison, which would take logging down after one panic).
#[derive(Clone)]
struct SharedFile(Arc<Mutex<RotatingFile>>);

struct SharedFileGuard<'a>(MutexGuard<'a, RotatingFile>);

impl Write for SharedFileGuard<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        // One formatted event = one call: never split a line across a rotation.
        self.0.write(buf).map(|_| ())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

impl<'a> MakeWriter<'a> for SharedFile {
    type Writer = SharedFileGuard<'a>;
    fn make_writer(&'a self) -> Self::Writer {
        SharedFileGuard(lock(&self.0))
    }
}

// ---------------------------------------------------------------------------------------------
// UI forwarding layer
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct MessageVisitor {
    message: String,
    fields: String,
    /// Real target of a `log` crate record (tracing-log events report the target "log").
    log_target: Option<String>,
}

impl Visit for MessageVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "message" => self.message.push_str(value),
            "log.target" => self.log_target = Some(value.to_string()),
            name if name.starts_with("log.") => {}
            name => self.push_field(name, value),
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        let rendered = format!("{value:?}");
        if field.name() == "message" {
            self.message.push_str(&rendered);
        } else if !field.name().starts_with("log.") {
            // `log.target` / `log.module_path` / ... are tracing-log bookkeeping, not content.
            self.push_field(field.name(), &rendered);
        }
    }
}

impl MessageVisitor {
    fn push_field(&mut self, name: &str, value: &str) {
        if !self.fields.is_empty() {
            self.fields.push(' ');
        }
        self.fields.push_str(name);
        self.fields.push('=');
        self.fields.push_str(value);
    }

    fn finish(&self) -> String {
        match (self.message.is_empty(), self.fields.is_empty()) {
            (_, true) => self.message.clone(),
            (true, false) => self.fields.clone(),
            (false, false) => format!("{} ({})", self.message, self.fields),
        }
    }
}

/// Whether a WARN/ERROR event from `target` is shown in the UI system log.
///
/// Sigil's own warnings are forwarded; everything else only at ERROR. That includes ONNX Runtime
/// (`ort::logging`): its warnings are routine on every GPU session ("Some nodes were not assigned
/// to the preferred execution providers", "Removing initializer … not used by any node") and read
/// like a GPU failure next to "running on CUDA". They stay in the file; the EP actually in use is
/// reported by Sigil's own `sigil::ort` lines.
pub(crate) fn forwardable(level: Level, target: &str) -> bool {
    if level > Level::WARN || target.starts_with(FRONTEND_TARGET) {
        return false;
    }
    level == Level::ERROR || target == "sigil" || target.starts_with("sigil::")
}

pub(crate) fn truncate_chars(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s.to_string(),
    }
}

struct ForwardLayer;

impl<S: Subscriber> Layer<S> for ForwardLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let meta = event.metadata();
        let level = *meta.level();
        if level > Level::WARN {
            return;
        }
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        // tracing-log records carry their real target in a field; `meta.target()` is then "log".
        let target = visitor.log_target.take().unwrap_or_else(|| meta.target().to_string());
        if !forwardable(level, &target) {
            return;
        }
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let message = truncate_chars(visitor.finish().trim(), FORWARD_MAX_CHARS);
        // Number, store and enqueue under one lock so the live stream is in `seq` order (the
        // webview drops a seq it has already passed as a duplicate of the backfill).
        let mut ring = lock(recent());
        let entry = LogEntry {
            seq: SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            at_unix_ms: unix_ms_now(),
            level: if level == Level::ERROR { "error" } else { "warn" },
            target,
            message,
        };
        if ring.len() == RECENT_CAPACITY {
            ring.pop_front();
        }
        ring.push_back(entry.clone());
        if let Some(tx) = FORWARD_TX.get() {
            // Never block the logging thread: a full queue drops the UI copy only (the entry is
            // still in the file and the ring buffer).
            let _ = tx.try_send(entry);
        }
    }
}

/// Rate limiter for UI forwarding: at most `burst` entries per `window`; the rest are counted and
/// reported as one summary line when the window rolls over.
pub(crate) struct ForwardLimiter {
    burst: u32,
    window: Duration,
    window_start: Option<Instant>,
    sent: u32,
    suppressed: u32,
}

impl ForwardLimiter {
    pub(crate) fn new(burst: u32, window: Duration) -> Self {
        Self {
            burst,
            window,
            window_start: None,
            sent: 0,
            suppressed: 0,
        }
    }

    /// Returns (suppressed-count summary to emit first, whether `now`'s entry may be sent).
    pub(crate) fn admit(&mut self, now: Instant) -> (Option<u32>, bool) {
        let mut summary = None;
        let expired = self.window_start.is_none_or(|s| now.duration_since(s) >= self.window);
        if expired {
            if self.suppressed > 0 {
                summary = Some(self.suppressed);
            }
            self.window_start = Some(now);
            self.sent = 0;
            self.suppressed = 0;
        }
        if self.sent < self.burst {
            self.sent += 1;
            (summary, true)
        } else {
            self.suppressed += 1;
            (summary, false)
        }
    }
}

fn run_forwarder(rx: Receiver<LogEntry>) {
    let mut limiter = ForwardLimiter::new(FORWARD_BURST, FORWARD_WINDOW);
    while let Ok(entry) = rx.recv() {
        let (summary, send) = limiter.admit(Instant::now());
        let Some(forward) = FORWARDER.get() else {
            continue; // The ring buffer still has it; the webview backfills on attach.
        };
        if let Some(n) = summary {
            forward(&LogEntry {
                seq: 0,
                at_unix_ms: unix_ms_now(),
                level: "warn",
                target: "sigil::diagnostics".to_string(),
                message: format!("{n} more warnings/errors were not shown here; see the log file."),
            });
        }
        if send {
            forward(&entry);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------------------

fn filter_spec() -> String {
    std::env::var(LOG_ENV)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_FILTER.to_string())
}

fn parse_filter(spec: &str) -> (Targets, String) {
    match spec.parse::<Targets>() {
        Ok(t) => (t, spec.to_string()),
        Err(_) => (
            DEFAULT_FILTER.parse::<Targets>().expect("default filter parses"),
            DEFAULT_FILTER.to_string(),
        ),
    }
}

/// Install the global subscriber writing to `<dir>/sigil.log`. Idempotent: later calls return the
/// first call's result. Never panics; on failure the app runs without a file log.
pub fn init(dir: &Path) -> Result<LogInfo, String> {
    if let Some(info) = info() {
        return Ok(info);
    }
    let rotating = RotatingFile::open(dir, LOG_FILE_STEM, MAX_FILE_BYTES, MAX_FILES)
        .map_err(|e| format!("cannot open the log file in {}: {e}", dir.display()))?;
    let file_path = rotating.path_for(0);
    let writer = SharedFile(Arc::new(Mutex::new(rotating)));
    let (targets, filter) = parse_filter(&filter_spec());

    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(writer)
        .with_ansi(false)
        .with_target(true)
        .with_thread_names(true)
        .with_filter(targets.clone());
    let stderr_layer = cfg!(debug_assertions).then(|| {
        tracing_subscriber::fmt::layer()
            .with_writer(io::stderr)
            .with_target(true)
            .with_filter(targets.clone())
    });
    let forward_layer = ForwardLayer.with_filter(targets);

    let (tx, rx) = sync_channel::<LogEntry>(FORWARD_QUEUE);
    if FORWARD_TX.set(tx).is_ok() {
        let _ = std::thread::Builder::new()
            .name("sigil-log-forward".into())
            .spawn(move || run_forwarder(rx));
    }

    tracing_subscriber::registry()
        .with(file_layer)
        .with(stderr_layer)
        .with(forward_layer)
        .try_init()
        .map_err(|e| format!("a global log subscriber is already installed: {e}"))?;

    let _ = STATE.set(LogState {
        dir: dir.to_path_buf(),
        file: file_path,
        filter,
    });
    install_panic_hook();
    tracing::info!(
        target: "sigil::diagnostics",
        "===== Sigil {} started (pid {}, {}) =====",
        env!("CARGO_PKG_VERSION"),
        std::process::id(),
        if cfg!(debug_assertions) { "debug build" } else { "release build" }
    );
    info().ok_or_else(|| "log state missing after init".to_string())
}

/// The active log location, once [`init`] succeeded.
pub fn info() -> Option<LogInfo> {
    STATE.get().map(|s| LogInfo {
        dir: s.dir.display().to_string(),
        file: s.file.display().to_string(),
        max_file_bytes: MAX_FILE_BYTES,
        max_files: MAX_FILES,
        filter: s.filter.clone(),
    })
}

/// Install the UI forwarder (the plugin emits [`LOG_EVENT`]). First call wins.
pub fn set_forwarder(f: impl Fn(&LogEntry) + Send + Sync + 'static) {
    let _ = FORWARDER.set(Box::new(f));
}

/// Warnings/errors forwarded so far (oldest first), for the webview's backfill.
pub fn recent_entries() -> Vec<LogEntry> {
    lock(recent()).iter().cloned().collect()
}

fn install_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let payload = info
                .payload()
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| info.payload().downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            let location = info
                .location()
                .map(|l| format!("{}:{}", l.file(), l.line()))
                .unwrap_or_else(|| "<unknown>".to_string());
            let thread = std::thread::current();
            tracing::error!(
                target: "sigil::panic",
                "thread '{}' panicked at {location}: {payload}",
                thread.name().unwrap_or("<unnamed>")
            );
            previous(info);
        }));
    });
}

/// Write a line the webview asked for (system-log warnings/errors) into the file log.
pub fn log_frontend(level: &str, tag: &str, message: &str) {
    let tag = truncate_chars(tag.trim(), 64);
    let message = truncate_chars(message.trim(), 2000);
    match level {
        "error" => tracing::error!(target: "sigil::frontend", "[{tag}] {message}"),
        "warning" | "warn" => tracing::warn!(target: "sigil::frontend", "[{tag}] {message}"),
        _ => tracing::info!(target: "sigil::frontend", "[{tag}] {message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sigil-log-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn rotating_file_keeps_at_most_max_files_and_never_splits_a_line() {
        let dir = temp_dir("rotate");
        let mut f = RotatingFile::open(&dir, "t", 100, 3).unwrap();
        let line = [b'x'; 39];
        for i in 0..20u8 {
            let mut l = line.to_vec();
            l[0] = b'a' + i;
            l.push(b'\n');
            f.write_all(&l).unwrap();
        }
        drop(f);
        let mut names: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        names.sort();
        assert_eq!(names, vec!["t.1.log", "t.2.log", "t.log"]);
        for n in &names {
            let bytes = fs::read(dir.join(n)).unwrap();
            assert!(bytes.len() as u64 <= 100, "{n} is {} bytes", bytes.len());
            assert_eq!(bytes.len() % 40, 0, "{n} holds whole lines only");
        }
        // The newest line is in the current file.
        let current = fs::read_to_string(dir.join("t.log")).unwrap();
        assert!(current.contains(&format!("{}", (b'a' + 19) as char)));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rotating_file_appends_across_reopen() {
        let dir = temp_dir("reopen");
        {
            let mut f = RotatingFile::open(&dir, "s", 1000, 5).unwrap();
            f.write_all(b"first\n").unwrap();
        }
        let mut f = RotatingFile::open(&dir, "s", 1000, 5).unwrap();
        f.write_all(b"second\n").unwrap();
        drop(f);
        assert_eq!(fs::read_to_string(dir.join("s.log")).unwrap(), "first\nsecond\n");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn forwarding_rules() {
        assert!(forwardable(Level::WARN, "sigil::services::voice_changer"));
        assert!(forwardable(Level::WARN, "sigil::ort"), "Sigil's own EP outcome lines reach the UI");
        assert!(forwardable(Level::ERROR, "ort::session"));
        assert!(
            !forwardable(Level::WARN, "ort::logging"),
            "routine ONNX Runtime warnings (node placement, unused initializers) stay in the file"
        );
        assert!(!forwardable(Level::WARN, "sigil_other_crate"), "only the sigil target tree");
        assert!(forwardable(Level::ERROR, "wry::webview"));
        assert!(!forwardable(Level::WARN, "wry::webview"), "third-party warnings stay in the file");
        assert!(!forwardable(Level::INFO, "sigil::x"));
        assert!(!forwardable(Level::ERROR, FRONTEND_TARGET), "no echo of webview lines");
    }

    #[test]
    fn limiter_summarizes_bursts() {
        let mut l = ForwardLimiter::new(2, Duration::from_secs(1));
        let t0 = Instant::now();
        assert_eq!(l.admit(t0), (None, true));
        assert_eq!(l.admit(t0), (None, true));
        assert_eq!(l.admit(t0), (None, false));
        assert_eq!(l.admit(t0), (None, false));
        assert_eq!(l.admit(t0 + Duration::from_millis(1500)), (Some(2), true));
    }

    #[test]
    fn message_visitor_and_truncation() {
        let mut v = MessageVisitor::default();
        v.message.push_str("EP fell back");
        v.push_field("component", "vc");
        assert_eq!(v.finish(), "EP fell back (component=vc)");
        assert_eq!(truncate_chars("あいうえお", 3), "あいう…");
        assert_eq!(truncate_chars("abc", 3), "abc");
    }

    #[test]
    fn filter_spec_falls_back_on_garbage() {
        let (_, spec) = parse_filter("sigil=debug,warn");
        assert_eq!(spec, "sigil=debug,warn");
        let (_, spec) = parse_filter("sigil=notalevel");
        assert_eq!(spec, DEFAULT_FILTER);
    }
}
