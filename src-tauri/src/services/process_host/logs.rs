//! Helper output: stdout and stderr are read line by line on two threads per process, cleaned
//! (ANSI escapes, CR progress lines, very long lines, invalid UTF-8), stripped of the helper's API
//! key, kept in a per-helper ring buffer (IPC `get_helper_logs`, error messages) and written to the
//! Sigil file log through `tracing` (target `sigil::helper`, INFO, rate limited so a chatty helper
//! cannot flood the log). Lifecycle notes from Sigil itself use the `host` stream.

use crate::services::diagnostics::logging::unix_ms_now;
use serde::Serialize;
use std::collections::VecDeque;
use std::io::Read;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// tracing target of helper output and lifecycle lines.
pub const LOG_TARGET: &str = "sigil::helper";
/// Lines kept per helper (across restarts).
pub const RING_CAPACITY: usize = 400;
/// Longer lines are split; a helper printing without newlines cannot grow a line without bound.
pub const MAX_LINE_BYTES: usize = 2048;
/// Rate limit of lines written to the file log: burst, then lines per second.
const LOG_BURST: f64 = 200.0;
const LOG_PER_SEC: f64 = 50.0;
/// Replacement for the API key wherever a helper prints it.
pub const REDACTED: &str = "[redacted]";

/// Which stream a line came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LogStream {
    Stdout,
    Stderr,
    /// Written by Sigil (start, exit, restart, stop).
    Host,
}

impl LogStream {
    pub fn as_str(self) -> &'static str {
        match self {
            LogStream::Stdout => "stdout",
            LogStream::Stderr => "stderr",
            LogStream::Host => "host",
        }
    }
}

/// One line of helper output (IPC `get_helper_logs`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct HelperLogLine {
    /// Monotonic per helper.
    pub seq: u64,
    pub at_unix_ms: u64,
    pub stream: LogStream,
    pub text: String,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

// ---------------------------------------------------------------------------------------------
// Line splitting and cleaning
// ---------------------------------------------------------------------------------------------

/// Replace every occurrence of `needle` in `buf` with `with` (one pass, left to right).
fn replace_bytes(buf: &mut Vec<u8>, needle: &[u8], with: &[u8]) {
    if needle.is_empty() || !buf.windows(needle.len()).any(|w| w == needle) {
        return;
    }
    let mut out = Vec::with_capacity(buf.len());
    let mut i = 0;
    while i < buf.len() {
        if buf[i..].starts_with(needle) {
            out.extend_from_slice(with);
            i += needle.len();
        } else {
            out.push(buf[i]);
            i += 1;
        }
    }
    *buf = out;
}

/// Splits a byte stream into lines at `\n` or `\r` (progress bars), and every [`MAX_LINE_BYTES`].
///
/// With a secret ([`LineSplitter::redacting`]), the secret is replaced by [`REDACTED`] in the bytes
/// of every piece before it is emitted, and when an over-long line is cut the last
/// `secret.len() - 1` bytes stay buffered for the next piece: a secret is never split across two
/// pieces, where neither half would match (the log book redacts each piece again as text).
#[derive(Debug, Default)]
pub struct LineSplitter {
    buf: Vec<u8>,
    secret: Option<Vec<u8>>,
}

impl LineSplitter {
    pub fn redacting(secret: Option<&str>) -> Self {
        Self {
            buf: Vec::new(),
            secret: secret
                .filter(|s| !s.is_empty())
                .map(|s| s.as_bytes().to_vec()),
        }
    }

    pub fn feed(&mut self, bytes: &[u8], mut emit: impl FnMut(&[u8])) {
        for &b in bytes {
            if b == b'\n' || b == b'\r' {
                self.flush(&mut emit);
            } else {
                self.buf.push(b);
                if self.buf.len() >= MAX_LINE_BYTES {
                    self.cut(&mut emit);
                }
            }
        }
    }

    pub fn finish(&mut self, mut emit: impl FnMut(&[u8])) {
        self.flush(&mut emit);
    }

    fn redact_buf(&mut self) {
        if let Some(secret) = &self.secret {
            replace_bytes(&mut self.buf, secret, REDACTED.as_bytes());
        }
    }

    /// Emit the whole buffer (end of a line).
    fn flush<F: FnMut(&[u8])>(&mut self, emit: &mut F) {
        if self.buf.is_empty() {
            return;
        }
        self.redact_buf();
        emit(&self.buf);
        self.buf.clear();
    }

    /// Emit the head of an over-long line, keeping a tail that may be the start of the secret.
    fn cut<F: FnMut(&[u8])>(&mut self, emit: &mut F) {
        self.redact_buf();
        if self.buf.len() < MAX_LINE_BYTES {
            return; // the replacement shortened it; keep collecting
        }
        let keep = self
            .secret
            .as_ref()
            .map_or(0, |s| s.len() - 1)
            .min(MAX_LINE_BYTES / 2);
        let at = self.buf.len() - keep;
        emit(&self.buf[..at]);
        self.buf.drain(..at);
    }
}

/// Remove ANSI escape sequences (CSI `ESC [ ... final`, OSC `ESC ] ... BEL|ESC \`, and two-byte
/// escapes) and other control characters except tab.
pub fn clean_line(raw: &[u8]) -> String {
    let text = String::from_utf8_lossy(raw);
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            match chars.peek() {
                Some('[') => {
                    chars.next();
                    for d in chars.by_ref() {
                        if ('@'..='~').contains(&d) {
                            break;
                        }
                    }
                }
                Some(']') => {
                    chars.next();
                    while let Some(d) = chars.next() {
                        if d == '\u{7}' {
                            break;
                        }
                        if d == '\u{1b}' {
                            if chars.peek() == Some(&'\\') {
                                chars.next();
                            }
                            break;
                        }
                    }
                }
                Some(_) => {
                    chars.next();
                }
                None => {}
            }
            continue;
        }
        if c.is_control() && c != '\t' {
            continue;
        }
        out.push(c);
    }
    out.trim_end().to_string()
}

// ---------------------------------------------------------------------------------------------
// Rate limiter
// ---------------------------------------------------------------------------------------------

/// Token bucket for lines written to the file log.
#[derive(Debug)]
pub struct RateLimiter {
    burst: f64,
    per_sec: f64,
    tokens: f64,
    last: Instant,
    suppressed: u64,
}

impl RateLimiter {
    pub fn new(burst: f64, per_sec: f64) -> Self {
        Self {
            burst,
            per_sec,
            tokens: burst,
            last: Instant::now(),
            suppressed: 0,
        }
    }

    /// `Some(n)` when the line may be logged, `n` being the lines suppressed just before it.
    pub fn admit(&mut self, now: Instant) -> Option<u64> {
        let dt = now.saturating_duration_since(self.last).as_secs_f64();
        self.last = now;
        self.tokens = (self.tokens + dt * self.per_sec).min(self.burst);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            Some(std::mem::take(&mut self.suppressed))
        } else {
            self.suppressed += 1;
            None
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Log book
// ---------------------------------------------------------------------------------------------

/// Output of one helper (all its restarts).
#[derive(Debug)]
pub struct LogBook {
    name: String,
    redact: Option<String>,
    lines: Mutex<VecDeque<HelperLogLine>>,
    limiter: Mutex<RateLimiter>,
    seq: AtomicU64,
}

impl LogBook {
    pub fn new(name: &str, redact: Option<String>) -> Self {
        Self {
            name: name.to_string(),
            redact: redact.filter(|k| !k.is_empty()),
            lines: Mutex::new(VecDeque::with_capacity(64)),
            limiter: Mutex::new(RateLimiter::new(LOG_BURST, LOG_PER_SEC)),
            seq: AtomicU64::new(1),
        }
    }

    /// The API key this book redacts (for [`LineSplitter::redacting`]).
    pub fn redaction_key(&self) -> Option<&str> {
        self.redact.as_deref()
    }

    /// Replace the API key in `text`.
    pub fn redact(&self, text: &str) -> String {
        match &self.redact {
            Some(key) if text.contains(key.as_str()) => text.replace(key.as_str(), REDACTED),
            _ => text.to_string(),
        }
    }

    /// Record one line of helper output (or a host note) in the ring buffer and the file log.
    pub fn push(&self, stream: LogStream, text: &str) {
        let text = self.redact(text);
        if text.is_empty() {
            return;
        }
        let line = HelperLogLine {
            seq: self.seq.fetch_add(1, Ordering::Relaxed),
            at_unix_ms: unix_ms_now(),
            stream,
            text,
        };
        let admitted = if stream == LogStream::Host {
            Some(0) // lifecycle notes are rare and always logged
        } else {
            lock(&self.limiter).admit(Instant::now())
        };
        if let Some(suppressed) = admitted {
            if suppressed > 0 {
                tracing::info!(target: LOG_TARGET, "[{}] ({suppressed} lines not written to the log: rate limit)", self.name);
            }
            tracing::info!(target: LOG_TARGET, "[{}:{}] {}", self.name, stream.as_str(), line.text);
        }
        let mut lines = lock(&self.lines);
        if lines.len() >= RING_CAPACITY {
            lines.pop_front();
        }
        lines.push_back(line);
    }

    /// A lifecycle note from Sigil.
    pub fn note(&self, text: &str) {
        self.push(LogStream::Host, text);
    }

    pub fn snapshot(&self) -> Vec<HelperLogLine> {
        lock(&self.lines).iter().cloned().collect()
    }

    /// The last `n` lines the helper printed (not host notes), oldest first.
    pub fn tail(&self, n: usize) -> Vec<String> {
        let lines = lock(&self.lines);
        let mut out: Vec<String> = lines
            .iter()
            .rev()
            .filter(|l| l.stream != LogStream::Host)
            .take(n)
            .map(|l| l.text.clone())
            .collect();
        out.reverse();
        out
    }

    /// `": <last lines>"` for error messages (empty when the helper printed nothing).
    pub fn tail_for_error(&self) -> String {
        let tail = self.tail(4);
        if tail.is_empty() {
            return String::new();
        }
        let joined = tail.join(" | ");
        format!(". Last output: {}", crate::services::diagnostics::truncate_chars(&joined, 600))
    }
}

// ---------------------------------------------------------------------------------------------
// Pumps
// ---------------------------------------------------------------------------------------------

/// Counts running pump threads so the exit handler can wait for the last lines.
#[derive(Debug, Default)]
pub struct PumpGate {
    active: Mutex<usize>,
    idle: Condvar,
}

impl PumpGate {
    fn enter(&self) {
        *lock(&self.active) += 1;
    }

    fn leave(&self) {
        let mut n = lock(&self.active);
        *n = n.saturating_sub(1);
        if *n == 0 {
            self.idle.notify_all();
        }
    }

    /// Wait until every pump has drained its pipe, at most `timeout`.
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let n = lock(&self.active);
        let (n, _) = self
            .idle
            .wait_timeout_while(n, timeout, |n| *n > 0)
            .unwrap_or_else(PoisonError::into_inner);
        *n == 0
    }
}

/// Read `reader` until EOF on a named thread, pushing each line into `book`.
pub fn spawn_pump<R: Read + Send + 'static>(book: Arc<LogBook>, stream: LogStream, mut reader: R, gate: Arc<PumpGate>) -> std::io::Result<()> {
    gate.enter();
    let thread_name = format!("helper-{}-{}", book.name, stream.as_str());
    let g = gate.clone();
    let spawned = std::thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            let mut splitter = LineSplitter::redacting(book.redaction_key());
            let mut buf = [0u8; 4096];
            let mut emit = |raw: &[u8]| book.push(stream, &clean_line(raw));
            loop {
                match reader.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => splitter.feed(&buf[..n], &mut emit),
                }
            }
            splitter.finish(&mut emit);
            g.leave();
        });
    if let Err(e) = spawned {
        gate.leave();
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(chunks: &[&[u8]]) -> Vec<String> {
        let mut s = LineSplitter::default();
        let mut out = Vec::new();
        for c in chunks {
            s.feed(c, |l| out.push(String::from_utf8_lossy(l).into_owned()));
        }
        s.finish(|l| out.push(String::from_utf8_lossy(l).into_owned()));
        out
    }

    #[test]
    fn splitter_handles_crlf_cr_partial_and_long_lines() {
        assert_eq!(split(&[b"a\r\nb\n", b"c"]), ["a", "b", "c"]);
        assert_eq!(split(&[b"load 10%\rload 20%\rdone\n"]), ["load 10%", "load 20%", "done"]);
        assert_eq!(split(&[b"par", b"tial", b" line\n\n\n"]), ["partial line"]);
        let long = vec![b'x'; MAX_LINE_BYTES * 2 + 5];
        let out = split(&[&long]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), MAX_LINE_BYTES);
        assert_eq!(out[2].len(), 5);
    }

    /// A key that straddles the 2 KB cut of a long line is still redacted: no emitted piece holds
    /// any 8-character run of it, whatever the offset and however the bytes arrive.
    #[test]
    fn a_key_straddling_the_long_line_cut_is_redacted() {
        let key = "0123456789abcdef".repeat(4).replacen("0123", "f9e8", 2);
        assert_eq!(key.len(), 64);
        let fragments: Vec<&str> = (0..=key.len() - 8).map(|i| &key[i..i + 8]).collect();
        for start in (MAX_LINE_BYTES - 80)..(MAX_LINE_BYTES + 8) {
            let mut line = vec![b'x'; start];
            line.extend_from_slice(key.as_bytes());
            line.extend_from_slice(b" tail\n");
            for chunk in [1usize, 7, 4096] {
                let mut s = LineSplitter::redacting(Some(&key));
                let mut pieces = Vec::new();
                for c in line.chunks(chunk) {
                    s.feed(c, |p| pieces.push(String::from_utf8_lossy(p).into_owned()));
                }
                s.finish(|p| pieces.push(String::from_utf8_lossy(p).into_owned()));
                for p in &pieces {
                    assert!(p.len() <= MAX_LINE_BYTES, "piece of {} bytes", p.len());
                    for f in &fragments {
                        assert!(!p.contains(f), "start {start}, chunk {chunk}: {f} leaked");
                    }
                }
                let joined = pieces.concat();
                assert!(joined.contains(REDACTED) && joined.ends_with("tail"), "start {start}");
                assert_eq!(joined.len(), start + REDACTED.len() + 5, "start {start}: only the key is replaced");
            }
        }
        // The pump splits with the book's key.
        let book = Arc::new(LogBook::new("cut", Some(key.clone())));
        let gate = Arc::new(PumpGate::default());
        let mut data = vec![b'y'; MAX_LINE_BYTES - 30];
        data.extend_from_slice(key.as_bytes());
        data.push(b'\n');
        spawn_pump(book.clone(), LogStream::Stderr, std::io::Cursor::new(data), gate.clone()).unwrap();
        assert!(gate.wait_idle(Duration::from_secs(5)));
        let snap = book.snapshot();
        assert!(snap
            .iter()
            .all(|l| fragments.iter().all(|f| !l.text.contains(f))));
        assert!(snap.iter().any(|l| l.text.contains(REDACTED)));
    }

    #[test]
    fn clean_line_strips_escapes_and_controls() {
        assert_eq!(clean_line(b"\x1b[1;32minfo\x1b[0m: ok"), "info: ok");
        assert_eq!(clean_line(b"\x1b]0;title\x07text"), "text");
        assert_eq!(clean_line(b"a\x00b\tc  "), "ab\tc");
        assert_eq!(clean_line(&[0x66, 0xff, 0x6f]), "f\u{fffd}o");
    }

    #[test]
    fn rate_limiter_admits_a_burst_then_reports_suppressed_lines() {
        let mut r = RateLimiter::new(3.0, 1.0);
        let t0 = r.last;
        assert_eq!(r.admit(t0), Some(0));
        assert_eq!(r.admit(t0), Some(0));
        assert_eq!(r.admit(t0), Some(0));
        assert_eq!(r.admit(t0), None);
        assert_eq!(r.admit(t0), None);
        assert_eq!(r.admit(t0 + Duration::from_millis(1100)), Some(2));
    }

    #[test]
    fn log_book_redacts_the_key_and_keeps_a_bounded_tail() {
        let book = LogBook::new("t", Some("s3cr3t".into()));
        book.push(LogStream::Stderr, "key=s3cr3t and again s3cr3t");
        book.note("started");
        for i in 0..RING_CAPACITY + 10 {
            book.push(LogStream::Stdout, &format!("line {i}"));
        }
        let snap = book.snapshot();
        assert_eq!(snap.len(), RING_CAPACITY);
        assert!(snap.iter().all(|l| !l.text.contains("s3cr3t")));
        assert_eq!(
            book.tail(2),
            [format!("line {}", RING_CAPACITY + 8), format!("line {}", RING_CAPACITY + 9)]
        );
        let fresh = LogBook::new("u", Some("k".into()));
        fresh.push(LogStream::Stderr, "boom k");
        fresh.note("host line");
        assert_eq!(fresh.tail(5), [format!("boom {REDACTED}")]);
        assert!(fresh.tail_for_error().contains("boom"));
        assert_eq!(LogBook::new("v", None).tail_for_error(), "");
    }

    #[test]
    fn pump_reads_until_eof_and_gate_goes_idle() {
        let book = Arc::new(LogBook::new("pump", None));
        let gate = Arc::new(PumpGate::default());
        let data: &'static [u8] = b"one\ntwo\r\nthree";
        spawn_pump(book.clone(), LogStream::Stdout, data, gate.clone()).unwrap();
        assert!(gate.wait_idle(Duration::from_secs(5)));
        let texts: Vec<String> = book.snapshot().into_iter().map(|l| l.text).collect();
        assert_eq!(texts, ["one", "two", "three"]);
    }
}
