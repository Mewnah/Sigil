//! Shared by the cloud TTS plugins (`edge-tts` and `gemini-tts`, P6-H): how a failed request is
//! classified and tagged for the TS fallback chain, the per-plugin cancel registry, the sink
//! hand-off and cancellable waits.
//!
//! # Failure tag (wire contract with `src/core/services/tts/pipeline.ts` `parseTtsFailure`)
//! A failed `*_speak` command rejects with `"[cloud-tts:<kind>[:partial][:retry=<ms>]] <message>"`:
//! - `<kind>`: `network` (no connection), `rate_limited` (HTTP 429 / quota), `unavailable` (5xx,
//!   the service closed the connection), `stalled` (no audio in time), `auth` (key or token
//!   refused), `rejected` (the request was refused: unknown voice or model), `no_audio` (the
//!   service answered but produced no audio for this text), `other`.
//! - `partial`: audio of the sentence was already pushed into the sink utterance.
//! - `retry=<ms>`: the service said when to retry (`Retry-After`, Google `RetryInfo`).
//!
//! The TS chain strips the tag before showing the message. A cancelled request (stop / skip, or
//! the plugin's cancel command) rejects with the plain [`CANCELLED`], like every sink engine.
//! Failures on this PC's side (the sink refused or timed out a push, the utterance is gone, a
//! request for the utterance is already running) are not the service's: they reject **untagged**
//! with the sink's own text ([`SpeakError::Local`]), like the local engines, so the TS chain
//! rethrows them for the pipeline to report and never blames the service, opens its breaker or
//! retries the sentence with the local voice into the same failing utterance.

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use crate::services::openai_speech::{CancelProbe, PcmSink};
use crate::services::tts_sink::{self, SinkError, UtteranceWriter};

/// Error text of a cancelled request (the sink's `SinkError::Cancelled` text; TS `SINK_CANCELLED`).
pub(crate) const CANCELLED: &str = "cancelled";
/// Prefix of the failure tag.
pub(crate) const TAG_PREFIX: &str = "[cloud-tts:";
/// How often a cancellable wait checks for cancellation.
pub(crate) const CANCEL_POLL: Duration = Duration::from_millis(25);
/// Longest retry hint passed on (a longer one is still capped by the TS breaker).
const MAX_RETRY_HINT: Duration = Duration::from_secs(3600);

// ---------------------------------------------------------------------------------------------
// Failures
// ---------------------------------------------------------------------------------------------

/// Why a cloud request failed (see the module docs). Wire: snake_case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FailureKind {
    Network,
    RateLimited,
    Unavailable,
    Stalled,
    Auth,
    Rejected,
    NoAudio,
    Other,
}

impl FailureKind {
    pub(crate) fn wire(self) -> &'static str {
        match self {
            FailureKind::Network => "network",
            FailureKind::RateLimited => "rate_limited",
            FailureKind::Unavailable => "unavailable",
            FailureKind::Stalled => "stalled",
            FailureKind::Auth => "auth",
            FailureKind::Rejected => "rejected",
            FailureKind::NoAudio => "no_audio",
            FailureKind::Other => "other",
        }
    }

    /// The kind of a non-success HTTP status.
    pub(crate) fn for_status(status: u16) -> Self {
        match status {
            429 => FailureKind::RateLimited,
            401 | 403 | 407 => FailureKind::Auth,
            408 | 500..=599 => FailureKind::Unavailable,
            400..=499 => FailureKind::Rejected,
            _ => FailureKind::Other,
        }
    }
}

/// A failed request: what kind, a readable message, whether audio was already pushed, and the
/// service's retry hint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Failure {
    pub kind: FailureKind,
    pub message: String,
    pub partial: bool,
    pub retry_after: Option<Duration>,
}

impl Failure {
    pub(crate) fn new(kind: FailureKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into(), partial: false, retry_after: None }
    }

    pub(crate) fn with_partial(mut self, partial: bool) -> Self {
        self.partial = partial;
        self
    }

    pub(crate) fn with_retry_after(mut self, retry_after: Option<Duration>) -> Self {
        self.retry_after = retry_after.map(|d| d.min(MAX_RETRY_HINT));
        self
    }

    /// The IPC error text (see the module docs).
    pub(crate) fn to_wire(&self) -> String {
        let mut tag = format!("{TAG_PREFIX}{}", self.kind.wire());
        if self.partial {
            tag.push_str(":partial");
        }
        if let Some(d) = self.retry_after {
            tag.push_str(&format!(":retry={}", d.as_millis()));
        }
        tag.push_str("] ");
        tag.push_str(self.message.trim());
        tag
    }
}

/// How one speak request ended without success.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SpeakError {
    /// Stop / skip / the plugin's cancel command: not a failure to report.
    Cancelled,
    /// The service failed (tagged on the wire).
    Failed(Failure),
    /// This PC's side failed (the sink, the request registry): sent untagged, with this text.
    Local(String),
}

impl SpeakError {
    pub(crate) fn fail(kind: FailureKind, message: impl Into<String>) -> Self {
        SpeakError::Failed(Failure::new(kind, message))
    }

    /// A failed sink push or hand-off: cancelled when the sink cancelled the utterance, else a
    /// local failure carrying the sink's text.
    pub(crate) fn from_sink(e: SinkError) -> Self {
        match e {
            SinkError::Cancelled => SpeakError::Cancelled,
            other => SpeakError::Local(other.to_string()),
        }
    }

    /// The command's error text.
    pub(crate) fn into_wire(self) -> String {
        match self {
            SpeakError::Cancelled => CANCELLED.to_string(),
            SpeakError::Failed(f) => f.to_wire(),
            SpeakError::Local(message) => message,
        }
    }

    /// Mark a failure as partial when audio was already pushed.
    pub(crate) fn with_partial(self, partial: bool) -> Self {
        match self {
            SpeakError::Failed(f) => {
                let partial = partial || f.partial;
                SpeakError::Failed(f.with_partial(partial))
            }
            other => other,
        }
    }
}

impl From<Failure> for SpeakError {
    fn from(f: Failure) -> Self {
        SpeakError::Failed(f)
    }
}

/// A `Retry-After` header value: delay seconds, or an HTTP date relative to `now_unix`.
pub(crate) fn parse_retry_after(value: &str, now_unix: i64) -> Option<Duration> {
    let v = value.trim();
    if let Ok(secs) = v.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    let at = parse_http_date(v)?;
    Some(Duration::from_secs(u64::try_from(at - now_unix).ok()?))
}

// ---------------------------------------------------------------------------------------------
// Dates (no chrono in the dependency tree)
// ---------------------------------------------------------------------------------------------

/// Days since 1970-01-01 of a proleptic Gregorian date (H. Hinnant's `days_from_civil`).
pub(crate) fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (i64::from(m) + 9) % 12;
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Year, month and day of a day count since 1970-01-01 (`civil_from_days`).
pub(crate) fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

pub(crate) const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
pub(crate) const WEEKDAYS: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];

/// Unix seconds of an RFC 7231 IMF-fixdate (`Wed, 01 Oct 2026 12:34:56 GMT`), the form servers
/// send in `Date` and `Retry-After`. Other legacy forms are not needed here.
pub(crate) fn parse_http_date(value: &str) -> Option<i64> {
    let rest = value.trim().split_once(',').map_or(value.trim(), |(_, r)| r.trim());
    let mut parts = rest.split_whitespace();
    let day: u32 = parts.next()?.parse().ok()?;
    let month = parts.next()?;
    let month = MONTHS.iter().position(|m| m.eq_ignore_ascii_case(month))? as u32 + 1;
    let year: i64 = parts.next()?.parse().ok()?;
    let mut hms = parts.next()?.split(':');
    let h: i64 = hms.next()?.parse().ok()?;
    let mi: i64 = hms.next()?.parse().ok()?;
    let s: i64 = hms.next()?.parse().ok()?;
    if !(1..=31).contains(&day) || h > 23 || mi > 59 || s > 60 {
        return None;
    }
    Some(days_from_civil(year, month, day) * 86_400 + h * 3600 + mi * 60 + s)
}

/// Seconds since the Unix epoch now.
pub(crate) fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

// ---------------------------------------------------------------------------------------------
// Cancellation registry
// ---------------------------------------------------------------------------------------------

/// Running requests of one plugin by sink utterance id.
#[derive(Default)]
pub(crate) struct Registry {
    map: Mutex<HashMap<String, Arc<AtomicBool>>>,
}

/// A registered request; unregisters on drop.
pub(crate) struct Registration<'a> {
    registry: &'a Registry,
    id: String,
    pub flag: Arc<AtomicBool>,
}

impl Drop for Registration<'_> {
    fn drop(&mut self) {
        let mut map = self.registry.map.lock().unwrap_or_else(PoisonError::into_inner);
        if map.get(&self.id).is_some_and(|f| Arc::ptr_eq(f, &self.flag)) {
            map.remove(&self.id);
        }
    }
}

impl Registry {
    /// Register a request for utterance `id` (one request per utterance at a time).
    pub(crate) fn register(&self, id: &str) -> Result<Registration<'_>, String> {
        let mut map = self.map.lock().unwrap_or_else(PoisonError::into_inner);
        if map.contains_key(id) {
            return Err(format!("TTS utterance {id:?} is already being spoken."));
        }
        let flag = Arc::new(AtomicBool::new(false));
        map.insert(id.to_string(), Arc::clone(&flag));
        Ok(Registration { registry: self, id: id.to_string(), flag })
    }

    /// Cancel the request of utterance `id`, or (`None`) every running one.
    pub(crate) fn cancel(&self, id: Option<&str>) {
        let map = self.map.lock().unwrap_or_else(PoisonError::into_inner);
        match id {
            Some(id) => {
                if let Some(flag) = map.get(id) {
                    flag.store(true, Ordering::Release);
                }
            }
            None => map.values().for_each(|f| f.store(true, Ordering::Release)),
        }
    }

    #[cfg(test)]
    pub(crate) fn running(&self) -> usize {
        self.map.lock().unwrap_or_else(PoisonError::into_inner).len()
    }
}

/// The `*_cancel` commands' argument: `None` = every request, a blank id = nothing.
pub(crate) fn cancel_command(registry: &Registry, utterance_id: Option<&str>) {
    match utterance_id.map(str::trim) {
        None => registry.cancel(None),
        Some("") => {}
        Some(id) => registry.cancel(Some(id)),
    }
}

// ---------------------------------------------------------------------------------------------
// Sink hand-off
// ---------------------------------------------------------------------------------------------

/// A sink utterance as a [`PcmSink`].
pub(crate) struct SinkPush(pub UtteranceWriter);

impl PcmSink for SinkPush {
    fn push(&mut self, samples: &[f32], sample_rate: u32, channels: u16) -> Result<(), SinkError> {
        self.0.push(samples, sample_rate, channels)
    }

    fn cancel_probe(&self) -> CancelProbe {
        // A second writer of the same utterance shares its cancel flag.
        match tts_sink::writer(self.0.id()) {
            Ok(w) => Arc::new(move || w.is_cancelled()),
            Err(_) => Arc::new(|| true),
        }
    }
}

/// The sink writer of begun utterance `id` (a missing or unavailable sink is a local failure).
pub(crate) fn sink_for(id: &str) -> Result<Box<dyn PcmSink>, SpeakError> {
    match tts_sink::writer(id) {
        Ok(w) => Ok(Box::new(SinkPush(w))),
        Err(e) => Err(SpeakError::from_sink(e)),
    }
}

/// Both cancel sources of a request: its registry flag and its sink utterance.
#[derive(Clone)]
pub(crate) struct CancelWatch {
    flag: Arc<AtomicBool>,
    probe: CancelProbe,
}

impl CancelWatch {
    pub(crate) fn new(flag: Arc<AtomicBool>, probe: CancelProbe) -> Self {
        Self { flag, probe }
    }

    pub(crate) fn is_set(&self) -> bool {
        self.flag.load(Ordering::Acquire) || (self.probe)()
    }
}

/// Why a watched wait ended early.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Interrupted {
    Cancelled,
    TimedOut,
}

/// Await `fut`, giving up at `deadline` or once `cancel` is set (checked every [`CANCEL_POLL`]).
pub(crate) async fn watched<F: Future>(fut: F, deadline: Option<Instant>, cancel: &CancelWatch) -> Result<F::Output, Interrupted> {
    tokio::pin!(fut);
    loop {
        if cancel.is_set() {
            return Err(Interrupted::Cancelled);
        }
        let now = Instant::now();
        let tick = match deadline {
            Some(d) if d <= now => return Err(Interrupted::TimedOut),
            Some(d) => (d - now).min(CANCEL_POLL),
            None => CANCEL_POLL,
        };
        tokio::select! {
            biased;
            out = &mut fut => return Ok(out),
            _ = tokio::time::sleep(tick) => {}
        }
    }
}

/// Decoded interleaved samples for the hand-off thread.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Block {
    pub samples: Vec<f32>,
    pub rate: u32,
    pub channels: u16,
}

/// What a push thread did.
#[derive(Debug, Default)]
pub(crate) struct PushOutcome {
    pub frames: u64,
    pub rate: u32,
    /// Time from the request start to the first / last push.
    pub first: Option<Duration>,
    pub last: Option<Duration>,
    pub error: Option<SinkError>,
}

/// Longest single push (a cancel between pushes stops the rest).
const PUSH_MS: usize = 500;

/// Push `samples` into `sink` in pieces of at most [`PUSH_MS`], recording timing in `out`.
/// Returns false when the push failed (`out.error`) or `stop` returned true between pieces.
pub(crate) fn push_pieces(
    sink: &mut dyn PcmSink,
    block: &Block,
    started: Instant,
    out: &mut PushOutcome,
    stop: &dyn Fn() -> bool,
) -> bool {
    let ch = usize::from(block.channels.max(1));
    let whole = block.samples.len() / ch * ch;
    let step = (block.rate as usize * PUSH_MS / 1000).max(1) * ch;
    for piece in block.samples[..whole].chunks(step) {
        if stop() {
            return false;
        }
        if let Err(e) = sink.push(piece, block.rate, block.channels) {
            out.error = Some(e);
            return false;
        }
        out.frames += (piece.len() / ch) as u64;
        let now = started.elapsed();
        out.first.get_or_insert(now);
        out.last = Some(now);
        if out.rate == 0 {
            out.rate = block.rate;
        }
    }
    true
}

/// A blocking thread that pushes blocks into a sink (whose back-pressure blocks), fed through a
/// bounded channel, so network waits never wait on playback.
pub(crate) struct HandOff {
    tx: Option<mpsc::Sender<Block>>,
    task: tokio::task::JoinHandle<PushOutcome>,
    abort: Arc<AtomicBool>,
    /// Blocks handed over (pushed or queued).
    pub handed: u64,
}

/// Blocks buffered between the network and the hand-off thread.
const CHANNEL_BLOCKS: usize = 64;

impl HandOff {
    pub(crate) fn start(mut sink: Box<dyn PcmSink>, started: Instant) -> Self {
        let (tx, mut rx) = mpsc::channel::<Block>(CHANNEL_BLOCKS);
        let abort = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&abort);
        let task = tokio::task::spawn_blocking(move || {
            let mut out = PushOutcome::default();
            let stop_fn = || stop.load(Ordering::Acquire);
            while let Some(block) = rx.blocking_recv() {
                if !push_pieces(sink.as_mut(), &block, started, &mut out, &stop_fn) {
                    break;
                }
            }
            out
        });
        Self { tx: Some(tx), task, abort, handed: 0 }
    }

    /// Queue `block`. `Ok(false)` when the hand-off thread stopped (a push failed).
    pub(crate) async fn send(&mut self, block: Block, cancel: &CancelWatch) -> Result<bool, Interrupted> {
        if block.samples.is_empty() {
            return Ok(true);
        }
        let Some(tx) = &self.tx else {
            return Ok(false);
        };
        match watched(tx.send(block), None, cancel).await? {
            Ok(()) => {
                self.handed += 1;
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }

    /// Close the channel and wait for the last push.
    pub(crate) async fn finish(mut self, cancel: &CancelWatch) -> Result<PushOutcome, Interrupted> {
        self.tx = None;
        let abort = Arc::clone(&self.abort);
        let result = watched(&mut self.task, None, cancel).await;
        match result {
            Ok(Ok(outcome)) => Ok(outcome),
            Ok(Err(_)) => Ok(PushOutcome { error: Some(SinkError::Unavailable("the audio hand-off failed".into())), ..PushOutcome::default() }),
            Err(e) => {
                abort.store(true, Ordering::Release);
                Err(e)
            }
        }
    }
}

impl Drop for HandOff {
    fn drop(&mut self) {
        // A request that ends early stops its hand-off thread at the next piece.
        self.abort.store(true, Ordering::Release);
    }
}

/// Milliseconds of `d`, saturating.
pub(crate) fn ms(d: Duration) -> u32 {
    d.as_millis().min(u128::from(u32::MAX)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_tags_round_trip_the_contract() {
        let f = Failure::new(FailureKind::RateLimited, "  busy  ").with_retry_after(Some(Duration::from_millis(1500)));
        assert_eq!(f.to_wire(), "[cloud-tts:rate_limited:retry=1500] busy");
        let p = Failure::new(FailureKind::Stalled, "slow").with_partial(true);
        assert_eq!(p.to_wire(), "[cloud-tts:stalled:partial] slow");
        assert_eq!(SpeakError::Cancelled.into_wire(), CANCELLED);
        let e = SpeakError::fail(FailureKind::Network, "down").with_partial(true);
        assert_eq!(e.into_wire(), "[cloud-tts:network:partial] down");
        // The sink's own failures are not the service's: untagged, with the sink's text.
        let sink = SpeakError::from_sink(SinkError::Unavailable("timed out waiting for queued speech to play".into()));
        assert_eq!(sink.clone().with_partial(true).into_wire(), "TTS sink unavailable: timed out waiting for queued speech to play");
        assert!(!sink.into_wire().starts_with(TAG_PREFIX));
        assert_eq!(SpeakError::from_sink(SinkError::NotFound("u1".into())).into_wire(), "no TTS utterance \"u1\"");
        assert_eq!(SpeakError::from_sink(SinkError::Cancelled), SpeakError::Cancelled);
        let capped = Failure::new(FailureKind::RateLimited, "x").with_retry_after(Some(Duration::from_secs(99_999)));
        assert_eq!(capped.retry_after, Some(MAX_RETRY_HINT));
    }

    #[test]
    fn statuses_map_to_kinds() {
        assert_eq!(FailureKind::for_status(429), FailureKind::RateLimited);
        assert_eq!(FailureKind::for_status(503), FailureKind::Unavailable);
        assert_eq!(FailureKind::for_status(500), FailureKind::Unavailable);
        assert_eq!(FailureKind::for_status(408), FailureKind::Unavailable);
        assert_eq!(FailureKind::for_status(401), FailureKind::Auth);
        assert_eq!(FailureKind::for_status(403), FailureKind::Auth);
        assert_eq!(FailureKind::for_status(404), FailureKind::Rejected);
        assert_eq!(FailureKind::for_status(400), FailureKind::Rejected);
        assert_eq!(FailureKind::for_status(302), FailureKind::Other);
    }

    #[test]
    fn civil_dates_round_trip() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(days_from_civil(2000, 3, 1), 11_017);
        for z in [-1000_i64, 0, 59, 60, 10_957, 19_999, 20_727, 60_000] {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z, "{z}");
        }
        // 2026-10-01 is a Thursday: 1970-01-01 was one too.
        let z = days_from_civil(2026, 10, 1);
        assert_eq!(WEEKDAYS[z.rem_euclid(7) as usize], "Thu");
    }

    #[test]
    fn http_dates_and_retry_after_parse() {
        let t = parse_http_date("Wed, 30 Sep 2026 23:59:58 GMT").unwrap();
        assert_eq!(t, days_from_civil(2026, 9, 30) * 86_400 + 86_398);
        assert_eq!(parse_http_date("not a date"), None);
        assert_eq!(parse_http_date("Wed, 32 Sep 2026 00:00:00 GMT"), None);
        assert_eq!(parse_retry_after("12", 0), Some(Duration::from_secs(12)));
        assert_eq!(parse_retry_after("Wed, 30 Sep 2026 23:59:58 GMT", t - 5), Some(Duration::from_secs(5)));
        // A date in the past says nothing useful.
        assert_eq!(parse_retry_after("Wed, 30 Sep 2026 23:59:58 GMT", t + 5), None);
    }

    #[test]
    fn registry_cancels_by_id_or_all_and_unregisters() {
        let r = Registry::default();
        let a = r.register("a").unwrap();
        let b = r.register("b").unwrap();
        assert!(r.register("a").is_err());
        cancel_command(&r, Some("  "));
        assert!(!a.flag.load(Ordering::Acquire));
        cancel_command(&r, Some("a"));
        assert!(a.flag.load(Ordering::Acquire));
        assert!(!b.flag.load(Ordering::Acquire));
        cancel_command(&r, None);
        assert!(b.flag.load(Ordering::Acquire));
        drop(a);
        assert_eq!(r.running(), 1);
        drop(b);
        assert_eq!(r.running(), 0);
        assert!(r.register("a").is_ok());
    }
}
