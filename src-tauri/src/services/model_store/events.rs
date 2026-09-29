//! Event delivery: `models:progress` (at most 4 Hz per download) and `models:state` (every state
//! change). The store talks to an [`EventSink`] so the same code runs in the app (Tauri events), in
//! `Sigil.exe --headless fetch` (lines on stderr) and in tests (recorded).

use super::{EntryState, EntryStatus, ProgressEvent, PROGRESS_EVENT, STATE_EVENT};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Runtime};

/// Minimum spacing of progress events of one download (4 Hz).
pub(crate) const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);

pub(crate) trait EventSink: Send + Sync {
    fn progress(&self, event: &ProgressEvent);
    fn state(&self, status: &EntryStatus);
}

/// Drops everything (before the plugin is set up).
pub(crate) struct NullSink;

impl EventSink for NullSink {
    fn progress(&self, _: &ProgressEvent) {}
    fn state(&self, _: &EntryStatus) {}
}

/// Tauri events to every webview.
pub(crate) struct TauriSink<R: Runtime>(pub AppHandle<R>);

impl<R: Runtime> EventSink for TauriSink<R> {
    fn progress(&self, event: &ProgressEvent) {
        let _ = self.0.emit(PROGRESS_EVENT, event);
    }
    fn state(&self, status: &EntryStatus) {
        let _ = self.0.emit(STATE_EVENT, status);
    }
}

/// Human-readable byte count (`1.5 GB`).
pub(crate) fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1000.0 && u < UNITS.len() - 1 {
        v /= 1000.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

/// `Sigil.exe --headless fetch`: one progress line per second per file and every state change, on
/// stderr (stdout carries only the result).
pub(crate) struct StderrSink {
    last: Mutex<Option<Instant>>,
}

impl StderrSink {
    pub fn new() -> Self {
        Self { last: Mutex::new(None) }
    }
}

impl EventSink for StderrSink {
    fn progress(&self, e: &ProgressEvent) {
        let mut last = self.last.lock().unwrap_or_else(|p| p.into_inner());
        let done = e.bytes_total.is_some_and(|t| e.bytes_done >= t);
        if !done && last.is_some_and(|t| t.elapsed() < Duration::from_secs(1)) {
            return;
        }
        *last = Some(Instant::now());
        let total = match e.bytes_total {
            Some(t) if t > 0 => format!(
                " / {} ({:.0}%)",
                human_bytes(t),
                (e.bytes_done as f64 / t as f64 * 100.0).min(100.0)
            ),
            _ => String::new(),
        };
        let mirror = serde_json::to_value(e.mirror)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default();
        eprintln!(
            "[fetch] {} {}: {}{} at {}/s via {mirror} (attempt {})",
            e.id,
            e.file,
            human_bytes(e.bytes_done),
            total,
            human_bytes(e.bytes_per_sec.max(0.0) as u64),
            e.attempt
        );
    }

    fn state(&self, s: &EntryStatus) {
        let state = serde_json::to_value(s.state)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default();
        match (&s.error, s.state) {
            (Some(err), _) => eprintln!("[fetch] {}: {state}: {err}", s.id),
            (None, EntryState::Installed) => eprintln!(
                "[fetch] {}: installed{} ({})",
                s.id,
                if s.adopted { " (adopted existing files)" } else { "" },
                human_bytes(s.bytes_on_disk)
            ),
            _ => eprintln!("[fetch] {}: {state}", s.id),
        }
    }
}

/// Records everything (tests).
#[cfg(test)]
#[derive(Default)]
pub(crate) struct RecordingSink {
    pub progress: Mutex<Vec<ProgressEvent>>,
    pub states: Mutex<Vec<EntryStatus>>,
}

#[cfg(test)]
impl EventSink for RecordingSink {
    fn progress(&self, e: &ProgressEvent) {
        self.progress.lock().unwrap().push(e.clone());
    }
    fn state(&self, s: &EntryStatus) {
        self.states.lock().unwrap().push(s.clone());
    }
}

/// Throttles progress of one download and smooths its speed.
pub(crate) struct ProgressMeter {
    interval: Duration,
    last_emit: Option<Instant>,
    started_at: Instant,
    start_bytes: u64,
    sample_at: Instant,
    sample_bytes: u64,
    speed: f64,
}

impl ProgressMeter {
    pub fn with_interval(start_bytes: u64, interval: Duration) -> Self {
        let now = Instant::now();
        Self {
            interval,
            last_emit: None,
            started_at: now,
            start_bytes,
            sample_at: now,
            sample_bytes: start_bytes,
            speed: 0.0,
        }
    }

    /// `Some(bytes_per_sec)` when an event is due (the first call always is), else `None`.
    pub fn tick(&mut self, bytes_done: u64, force: bool) -> Option<f64> {
        let now = Instant::now();
        let dt = now.duration_since(self.sample_at).as_secs_f64();
        if dt >= 0.2 {
            let inst = bytes_done.saturating_sub(self.sample_bytes) as f64 / dt;
            // Exponential smoothing (~1 s time constant at 4 Hz).
            self.speed = if self.speed == 0.0 { inst } else { self.speed * 0.7 + inst * 0.3 };
            self.sample_at = now;
            self.sample_bytes = bytes_done;
        }
        let due = force || self.last_emit.is_none_or(|t| now.duration_since(t) >= self.interval);
        if due {
            self.last_emit = Some(now);
            // Before the first 200 ms sample (small files), report the average so far.
            let elapsed = now.duration_since(self.started_at).as_secs_f64();
            let speed = if self.speed == 0.0 && elapsed > 0.001 {
                bytes_done.saturating_sub(self.start_bytes) as f64 / elapsed
            } else {
                self.speed
            };
            Some(speed)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meter_emits_first_then_at_most_every_interval() {
        let mut m = ProgressMeter::with_interval(0, Duration::from_millis(250));
        assert!(m.tick(10, false).is_some());
        assert!(m.tick(20, false).is_none());
        assert!(m.tick(30, true).is_some(), "forced events always pass");
        std::thread::sleep(Duration::from_millis(260));
        let speed = m.tick(1_000_030, false).expect("due after the interval");
        assert!(speed > 0.0);
        // A transfer shorter than one sample still reports a speed.
        let mut quick = ProgressMeter::with_interval(0, Duration::from_millis(250));
        std::thread::sleep(Duration::from_millis(5));
        assert!(quick.tick(3_196, true).unwrap() > 0.0);
    }

    #[test]
    fn bytes_are_human_readable() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(41_205_931), "41.2 MB");
        assert_eq!(human_bytes(3_095_033_483), "3.1 GB");
    }
}
