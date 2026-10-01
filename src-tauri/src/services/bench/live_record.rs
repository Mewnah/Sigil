//! Live-record mode (P3-F): records the running voice changer's per-tick timeline
//! (`voice_changer::timeline`) for up to 10 minutes during a real VRChat + OBS session and writes
//! a report friends can export (the authoritative H1 evidence, plan T1 / T2 / T7).
//!
//! A recorder thread subscribes to the timeline (bounded, drop-oldest: it never slows the worker)
//! and drains it every 100 ms; it also samples the voice changer's own counters through a
//! [`LiveProbe`] (processing time, backlog, drops, underruns), process CPU time and VRAM. It stops
//! at `max_seconds` (default 600 s), when the voice changer stops for more than 2 s, or on [`stop`].
//!
//! The report folder holds `report.json` / `report.md` (the benchmark report format, one run of
//! mode `live_record`), `live.timeline.csv` (one row per tick) and `live.samples.csv` (the 100 ms
//! counter samples). It holds no audio and no file paths: the voice is named by its folder name only,
//! so the folder can be shared as it is.
//!
//! Metrics (see `report.rs`): `latency_ms_*` (capture of a block's newest frame to the estimated
//! playout of its first converted sample), `block_ms_*` (worker time per block), `model_ms_*`,
//! `backlog_ms_*`, `queued_out_ms_*`, `ticks`, `live_seconds`, the flag counts (`underrun_ticks`,
//! `budget_drop_ticks`, `gap_ticks`, `error_ticks`), `silent_ratio`, `governor_steps`,
//! `governor_level_max`, `dropped_ms`, `backlog_max_ms`, `over_budget_ratio` (worker time over the
//! block's duration), `cpu_cores`, `vram_mb_peak`, `timeline_dropped` and `underruns` (the
//! counter). Without timeline ticks (a build whose worker does not publish them) the block times
//! come from the counter samples, with a note.
//!
//! # Frozen by P3-S0 (signatures; P3-F implements)
//! [`available`], [`LiveRecordRequest`], [`LiveRecordSummary`], [`start`], [`stop`],
//! [`is_recording`]. Added: [`start_with_probe`], [`LiveProbe`], [`LiveSample`], [`analyze`].

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::metrics::{cores, insert_summary, process_cpu_time, Metrics, VramPeak};
use super::report::{write_atomic, write_report, BenchReport, RunReport, RunStatus, REPORT_SCHEMA};
use super::spec::RunMode;
use crate::services::voice_changer::timeline::{self, tick_flags, VcTick};

/// Whether live recording runs in this build.
pub fn available() -> bool {
    true
}

/// Longest recording (and the default).
pub const MAX_SECONDS: f64 = 600.0;
/// Timeline ring of the recorder (about 40 s of 10 ms ticks between drains).
const SUBSCRIBE_CAPACITY: usize = 4_096;
/// Ticks kept at most (about an hour of 10 ms blocks).
const MAX_TICKS: usize = 400_000;
const POLL: Duration = Duration::from_millis(100);
/// The voice changer counts as stopped after this long not running.
const STOPPED_GRACE: Duration = Duration::from_secs(2);

/// `voice_changer_live_record_start` request (snake_case).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LiveRecordRequest {
    /// Stop on its own after this long (default 600 s).
    #[serde(default)]
    pub max_seconds: Option<f64>,
    /// Free text stored in the report (game, world, GPU load).
    #[serde(default)]
    pub note: Option<String>,
}

/// What a recording produced (snake_case).
#[derive(Clone, Debug, Default, Serialize)]
pub struct LiveRecordSummary {
    pub seconds: f64,
    pub ticks: u64,
    pub report_dir: String,
    /// The report's metrics (same names as `bench::report`).
    pub metrics: BTreeMap<String, f64>,
    // ---- P3-F ----
    pub notes: Vec<String>,
}

/// One sample of the voice changer's own counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct LiveSample {
    pub running: bool,
    /// Latest block's processing time and its recent p99 (µs).
    pub process_us: u32,
    pub process_p99_us: u32,
    /// Capture frames waiting.
    pub backlog_frames: u32,
    /// Since the voice changer started.
    pub dropped_frames: u64,
    pub underruns: u64,
    pub capture_rate: u32,
    pub output_rate: u32,
}

/// The running voice changer, as the recorder sees it.
pub trait LiveProbe: Send + 'static {
    fn sample(&self) -> LiveSample;
    /// Its configuration for the report (no file paths).
    fn describe(&self) -> serde_json::Value;
}

struct Recording {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<Result<LiveRecordSummary, String>>>,
}

static RECORDING: Mutex<Option<Recording>> = Mutex::new(None);

fn slot() -> std::sync::MutexGuard<'static, Option<Recording>> {
    RECORDING.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Start recording into `out_dir` (an existing recording is an error).
#[allow(dead_code)] // P3-S0 frozen API; the IPC command starts with the voice changer's probe.
pub fn start(request: &LiveRecordRequest, out_dir: &Path) -> Result<(), String> {
    start_with_probe(request, out_dir, None)
}

/// [`start`] with the voice changer's counters.
pub fn start_with_probe(request: &LiveRecordRequest, out_dir: &Path, probe: Option<Box<dyn LiveProbe>>) -> Result<(), String> {
    let seconds = request.max_seconds.unwrap_or(MAX_SECONDS);
    if !(seconds.is_finite() && (1.0..=MAX_SECONDS).contains(&seconds)) {
        return Err(format!("max_seconds = {seconds} is outside 1..={MAX_SECONDS}"));
    }
    let mut guard = slot();
    if let Some(r) = guard.as_ref() {
        if r.handle.as_ref().is_some_and(|h| !h.is_finished()) {
            return Err("a live recording is already running".into());
        }
    }
    std::fs::create_dir_all(out_dir).map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;
    let stop = Arc::new(AtomicBool::new(false));
    let (stop2, out, req) = (stop.clone(), out_dir.to_path_buf(), request.clone());
    let handle = std::thread::Builder::new()
        .name("sigil-live-record".into())
        .spawn(move || record(&req, seconds, &out, probe, &stop2))
        .map_err(|e| format!("cannot start the recorder: {e}"))?;
    *guard = Some(Recording {
        stop,
        handle: Some(handle),
    });
    Ok(())
}

/// Stop and write the report (or collect the report of a recording that ended on its own).
pub fn stop() -> Result<LiveRecordSummary, String> {
    let rec = slot().take().ok_or("no live recording")?;
    rec.stop.store(true, Ordering::SeqCst);
    let handle = rec.handle.ok_or("no live recording")?;
    // Wake the recorder from its poll wait: it stops at once instead of after its period.
    handle.thread().unpark();
    handle.join().map_err(|_| "the live recorder failed unexpectedly".to_string())?
}

pub fn is_recording() -> bool {
    slot()
        .as_ref()
        .and_then(|r| r.handle.as_ref())
        .is_some_and(|h| !h.is_finished())
}

/// The recorder thread.
fn record(req: &LiveRecordRequest, max_seconds: f64, out: &Path, probe: Option<Box<dyn LiveProbe>>, stop: &AtomicBool) -> Result<LiveRecordSummary, String> {
    let started = Instant::now();
    let started_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    super::metrics::prepare_cpu_meter();
    let mut sub = timeline::subscribe(SUBSCRIBE_CAPACITY);
    let mut ticks: Vec<VcTick> = Vec::with_capacity(((max_seconds * 110.0) as usize).min(MAX_TICKS));
    let mut samples: Vec<(f64, LiveSample)> = Vec::with_capacity((max_seconds * 10.0) as usize + 16);
    let first = probe.as_ref().map(|p| p.sample());
    let cpu0 = process_cpu_time();
    let mut vram = VramPeak::default();
    let mut vram_at = Instant::now() - Duration::from_secs(1);
    let mut notes = Vec::new();
    let mut stopped_since: Option<Instant> = None;
    let mut truncated = false;
    let mut stop_reason = "stopped";
    loop {
        // `stop` unparks the thread; a spurious wake only samples early.
        std::thread::park_timeout(POLL);
        sub.drain(&mut ticks);
        if ticks.len() > MAX_TICKS {
            ticks.truncate(MAX_TICKS);
            truncated = true;
        }
        let t = started.elapsed().as_secs_f64();
        if let Some(p) = &probe {
            let s = p.sample();
            samples.push((t, s));
            if s.running {
                stopped_since = None;
            } else {
                let since = *stopped_since.get_or_insert_with(Instant::now);
                if since.elapsed() >= STOPPED_GRACE {
                    stop_reason = "the voice changer stopped";
                    break;
                }
            }
        }
        if vram_at.elapsed() >= Duration::from_secs(1) {
            vram_at = Instant::now();
            vram.sample();
        }
        if stop.load(Ordering::SeqCst) {
            break;
        }
        if t >= max_seconds {
            stop_reason = "max_seconds reached";
            break;
        }
    }
    sub.drain(&mut ticks);
    let wall = started.elapsed();
    let cpu = match (cpu0, process_cpu_time()) {
        (Some(a), Some(b)) => Some(cores(b.saturating_sub(a), wall)),
        _ => None,
    };
    if truncated {
        notes.push(format!("the timeline was truncated at {MAX_TICKS} ticks"));
    }
    let rate = samples
        .iter()
        .rev()
        .map(|s| s.1.capture_rate)
        .find(|&r| r > 0)
        .unwrap_or(48_000);
    let mut metrics = analyze(&ticks, rate);
    if ticks.is_empty() {
        notes.push("no timeline ticks were published (the voice changer was idle, or this build's worker does not publish its timeline); block times come from the 100 ms counter samples".into());
        let block: Vec<f64> = samples
            .iter()
            .filter(|s| s.1.running && s.1.process_us > 0)
            .map(|s| f64::from(s.1.process_us) / 1000.0)
            .collect();
        insert_summary(&mut metrics, "block_ms", &block);
    }
    let rate_ms = |frames: f64| frames * 1000.0 / f64::from(rate.max(1));
    if let (Some(a), Some(b)) = (first, samples.last().map(|s| s.1)) {
        metrics.insert("underruns".into(), b.underruns.saturating_sub(a.underruns) as f64);
        if ticks.is_empty() {
            metrics.insert("dropped_ms".into(), rate_ms(b.dropped_frames.saturating_sub(a.dropped_frames) as f64));
        }
        if !metrics.contains_key("backlog_max_ms") {
            let max = samples.iter().map(|s| s.1.backlog_frames).max().unwrap_or(0);
            metrics.insert("backlog_max_ms".into(), rate_ms(f64::from(max)));
        }
    }
    metrics.insert("live_seconds".into(), wall.as_secs_f64());
    metrics.insert("timeline_dropped".into(), sub.dropped() as f64);
    if let Some(c) = cpu {
        metrics.insert("cpu_cores".into(), c);
    }
    if let Some(v) = vram.peak_mb {
        metrics.insert("vram_mb_peak".into(), v as f64);
    }
    // Files.
    let mut files = Vec::new();
    if !ticks.is_empty() {
        write_atomic(&out.join("live.timeline.csv"), timeline_csv(&ticks).as_bytes())?;
        files.push("live.timeline.csv".to_string());
    }
    if !samples.is_empty() {
        write_atomic(&out.join("live.samples.csv"), samples_csv(&samples).as_bytes())?;
        files.push("live.samples.csv".to_string());
    }
    let request_json = serde_json::to_value(req).unwrap_or_default();
    let details = serde_json::json!({
        "stop_reason": stop_reason,
        "device_rate": rate,
        "voice_changer": probe.as_ref().map(|p| p.describe()),
        "notes": notes,
    });
    let run = RunReport {
        name: "live".into(),
        mode: RunMode::LiveRecord,
        status: RunStatus::Passed,
        error: None,
        duration_ms: wall.as_secs_f64() * 1000.0,
        metrics: metrics.clone(),
        thresholds: Vec::new(),
        details,
        files,
        higher_is_better: super::report::higher_is_better_of(&metrics),
    };
    let report = BenchReport {
        schema: REPORT_SCHEMA.into(),
        name: "live-record".into(),
        description: req.note.clone().unwrap_or_default(),
        passed: true,
        failures: Vec::new(),
        started_unix_ms,
        duration_ms: wall.as_secs_f64() * 1000.0,
        spec_sha256: hex::encode(Sha256::digest(request_json.to_string().as_bytes())),
        spec: request_json,
        environment: super::environment(),
        engine: probe.as_ref().map(|p| p.describe()),
        runs: vec![run],
    };
    write_report(out, &report)?;
    Ok(LiveRecordSummary {
        seconds: wall.as_secs_f64(),
        ticks: ticks.len() as u64,
        report_dir: out.display().to_string(),
        metrics,
        notes,
    })
}

/// Metrics of a recorded timeline (`rate`: the capture device rate, for block durations).
pub fn analyze(ticks: &[VcTick], rate: u32) -> Metrics {
    let mut m = Metrics::new();
    m.insert("ticks".into(), ticks.len() as f64);
    if ticks.is_empty() {
        return m;
    }
    let ns_ms = |ns: i64| ns as f64 / 1e6;
    let latency: Vec<f64> = ticks
        .iter()
        .filter(|t| t.capture_end_ns > 0 && t.est_playout_ns >= t.capture_end_ns)
        .map(|t| ns_ms(t.est_playout_ns - t.capture_end_ns))
        .collect();
    insert_summary(&mut m, "latency_ms", &latency);
    let block: Vec<f64> = ticks.iter().filter(|t| t.end_ns >= t.start_ns && t.start_ns > 0).map(|t| ns_ms(t.end_ns - t.start_ns)).collect();
    insert_summary(&mut m, "block_ms", &block);
    let model: Vec<f64> = ticks.iter().map(|t| f64::from(t.model_ms)).collect();
    insert_summary(&mut m, "model_ms", &model);
    let backlog: Vec<f64> = ticks.iter().map(|t| f64::from(t.backlog_ms)).collect();
    insert_summary(&mut m, "backlog_ms", &backlog);
    let queued: Vec<f64> = ticks.iter().map(|t| f64::from(t.queued_out_ms)).collect();
    insert_summary(&mut m, "queued_out_ms", &queued);
    let flag = |f: u8| ticks.iter().filter(|t| t.flags & f != 0).count() as f64;
    m.insert("underrun_ticks".into(), flag(tick_flags::UNDERRUN));
    m.insert("budget_drop_ticks".into(), flag(tick_flags::BUDGET_DROP));
    m.insert("gap_ticks".into(), flag(tick_flags::GAP));
    m.insert("error_ticks".into(), flag(tick_flags::ERROR));
    m.insert("governor_steps".into(), flag(tick_flags::GOVERNOR_STEP));
    m.insert("silent_ratio".into(), flag(tick_flags::SILENT) / ticks.len() as f64);
    m.insert(
        "governor_level_max".into(),
        f64::from(ticks.iter().map(|t| t.governor_level).max().unwrap_or(0)),
    );
    m.insert("dropped_ms".into(), ticks.iter().map(|t| f64::from(t.dropped_ms)).sum());
    m.insert(
        "backlog_max_ms".into(),
        ticks.iter().map(|t| f64::from(t.backlog_ms)).fold(0.0, f64::max),
    );
    let over = ticks
        .iter()
        .filter(|t| {
            let dur = f64::from(t.block_frames) * 1000.0 / f64::from(rate.max(1));
            t.end_ns >= t.start_ns && t.start_ns > 0 && dur > 0.0 && ns_ms(t.end_ns - t.start_ns) > dur
        })
        .count();
    m.insert("over_budget_ratio".into(), over as f64 / ticks.len() as f64);
    m
}

const TIMELINE_HEADER: &str =
    "seq,capture_end_ns,start_ns,end_ns,enqueue_ns,est_playout_ns,block_frames,backlog_ms,dropped_ms,queued_out_ms,model_ms,governor_level,flags";

fn timeline_csv(ticks: &[VcTick]) -> String {
    let mut s = String::with_capacity(ticks.len() * 96 + 128);
    s.push_str(TIMELINE_HEADER);
    s.push('\n');
    for t in ticks {
        let _ = writeln!(
            s,
            "{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.4},{},{}",
            t.seq,
            t.capture_end_ns,
            t.start_ns,
            t.end_ns,
            t.enqueue_ns,
            t.est_playout_ns,
            t.block_frames,
            t.backlog_ms,
            t.dropped_ms,
            t.queued_out_ms,
            t.model_ms,
            t.governor_level,
            t.flags
        );
    }
    s
}

fn samples_csv(samples: &[(f64, LiveSample)]) -> String {
    let mut s = String::from("t_s,running,process_us,process_p99_us,backlog_frames,dropped_frames,underruns,capture_rate,output_rate\n");
    for (t, x) in samples {
        let _ = writeln!(
            s,
            "{t:.3},{},{},{},{},{},{},{},{}",
            u8::from(x.running),
            x.process_us,
            x.process_p99_us,
            x.backlog_frames,
            x.dropped_frames,
            x.underruns,
            x.capture_rate,
            x.output_rate
        );
    }
    s
}

/// Default report folder of a recording: `app_data/bench/results/live-<unix ms>/`.
pub fn default_out_dir(app_data: &Path) -> PathBuf {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    app_data.join("bench").join("results").join(format!("live-{ms}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tick(seq: u64, t0_ms: f64, latency_ms: f64, work_ms: f64, flags: u8) -> VcTick {
        let ns = |ms: f64| (ms * 1e6) as i64;
        VcTick {
            seq,
            capture_end_ns: ns(t0_ms),
            start_ns: ns(t0_ms + 0.5),
            end_ns: ns(t0_ms + 0.5 + work_ms),
            enqueue_ns: ns(t0_ms + 0.6 + work_ms),
            est_playout_ns: ns(t0_ms + latency_ms),
            block_frames: 512,
            backlog_ms: 5.0,
            dropped_ms: if flags & tick_flags::BUDGET_DROP != 0 { 150.0 } else { 0.0 },
            queued_out_ms: 30.0,
            model_ms: work_ms as f32,
            governor_level: u8::from(flags & tick_flags::GOVERNOR_STEP != 0),
            flags,
        }
    }

    /// The analysis on a synthetic timeline: latency from capture to playout, work time, the flag
    /// counts and the over-budget ratio against the 10.7 ms block.
    #[test]
    fn timelines_are_analyzed() {
        let mut ticks: Vec<VcTick> = (0..100).map(|i| tick(i, i as f64 * 10.667, 70.0, 3.0, 0)).collect();
        ticks[10] = tick(10, 106.67, 90.0, 15.0, tick_flags::UNDERRUN | tick_flags::BUDGET_DROP);
        ticks[20].flags = tick_flags::SILENT;
        ticks[30] = tick(30, 320.0, 70.0, 3.0, tick_flags::GOVERNOR_STEP);
        let m = analyze(&ticks, 48_000);
        assert_eq!(m["ticks"], 100.0);
        assert!((m["latency_ms_p50"] - 70.0).abs() < 1e-6, "{m:?}");
        assert!((m["latency_ms_max"] - 90.0).abs() < 1e-6);
        assert!((m["block_ms_p50"] - 3.0).abs() < 1e-6);
        assert_eq!(m["underrun_ticks"], 1.0);
        assert_eq!(m["budget_drop_ticks"], 1.0);
        assert_eq!(m["dropped_ms"], 150.0);
        assert!((m["silent_ratio"] - 0.01).abs() < 1e-9);
        assert_eq!((m["governor_steps"], m["governor_level_max"]), (1.0, 1.0));
        assert!((m["over_budget_ratio"] - 0.01).abs() < 1e-9, "the 15 ms block of a 10.7 ms budget");
        assert_eq!(analyze(&[], 48_000)["ticks"], 0.0);
        let csv = timeline_csv(&ticks[..2]);
        assert_eq!(csv.lines().count(), 3);
        assert!(csv.starts_with(TIMELINE_HEADER));
        // Every live metric name is in the report catalogue.
        for k in m.keys() {
            assert!(super::super::report::known_metric(RunMode::LiveRecord, k), "{k}");
        }
    }

    struct FakeProbe {
        running: Arc<AtomicBool>,
    }

    impl LiveProbe for FakeProbe {
        fn sample(&self) -> LiveSample {
            LiveSample {
                running: self.running.load(Ordering::SeqCst),
                process_us: 2_500,
                process_p99_us: 4_000,
                backlog_frames: 480,
                dropped_frames: 0,
                underruns: 3,
                capture_rate: 48_000,
                output_rate: 48_000,
            }
        }

        fn describe(&self) -> serde_json::Value {
            serde_json::json!({"mode": "rvc"})
        }
    }

    /// Start, stop, and the report folder; a second start while recording is refused; the
    /// recorder ends by itself when the voice changer stops.
    #[test]
    fn recordings_start_stop_and_write_reports() {
        let dir = std::env::temp_dir().join(format!("sigil-live-record-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let running = Arc::new(AtomicBool::new(true));
        assert!(start(&LiveRecordRequest { max_seconds: Some(0.5), note: None }, &dir).is_err(), "at least 1 s");
        start_with_probe(
            &LiveRecordRequest { max_seconds: Some(30.0), note: Some("test".into()) },
            &dir,
            Some(Box::new(FakeProbe { running: running.clone() })),
        )
        .unwrap();
        assert!(is_recording());
        assert!(start(&LiveRecordRequest::default(), &dir).is_err(), "one at a time");
        std::thread::sleep(Duration::from_millis(450));
        // The voice changer stops: the recorder ends within the grace period.
        running.store(false, Ordering::SeqCst);
        let t0 = Instant::now();
        while is_recording() && t0.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(!is_recording());
        let s = stop().unwrap();
        assert!(s.seconds > 2.0 && s.seconds < 10.0, "{s:?}");
        assert!(s.metrics["block_ms_p50"] > 0.0, "counter samples stand in for the timeline: {s:?}");
        assert!(dir.join("report.json").is_file() && dir.join("report.md").is_file() && dir.join("live.samples.csv").is_file());
        let back: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("report.json")).unwrap()).unwrap();
        assert_eq!(back["runs"][0]["mode"], "live_record");
        assert_eq!(back["runs"][0]["details"]["stop_reason"], "the voice changer stopped");
        assert!(stop().is_err(), "nothing left to stop");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
