//! The `pipeline` STT driver: the real speech pipeline (Silero VAD segmentation, the router, two-pass
//! finals, the hallucination filter) fed from corpus clips instead of a microphone.
//!
//! One pipeline session runs per language group of items, through the speech plugin's own session
//! code (`speech::pipeline::start_session` with a private registry, so it never touches the app's
//! `mic` / `loopback` keys). A [`ClipFeed`] stands in for the capture hub: the driver pushes
//! `lead silence + clip + tail silence` of an item, then waits until the pipeline has settled that
//! item (every segment that opened is closed, every closed segment has its final: the second-pass
//! final on a two-pass route, or was dropped) before it feeds the next one, so nothing is lost to the
//! pipeline's backlog cap or latency budget. A [`Tracker`] collects the pipeline's events; the finals
//! are assigned to items by the audio time of the stream (a segment belongs to the item whose window
//! holds its midpoint) and joined into the item's hypothesis.
//!
//! The source hands the pipeline 20 ms blocks (the streaming front end keeps only a couple of seconds
//! of history), and a route whose only engine is a streaming one is fed at real time whatever the spec's
//! pace says ([`effective_pace`]): the pipeline drops the live text of a stream more than 3 s behind.
//!
//! Item timings: `final_latency_ms` is the pipeline's own VAD-close-to-final latency of the item's
//! last final; `first_partial_ms` runs from the item's first fed sample to its first partial (the lead
//! silence is fed at once and takes no wall time, so at real-time pace this is also the time from the
//! clip's start: the lead is not subtracted); `compute_ms` (asap pace only) is the wall time from the
//! item's first fed sample to its last final.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::report::Segment;
use super::spec::{EngineDef, ExecutionProvider, Pace, Paths};
use super::stt::{engine_spec, DriverInfo, SttDriver, SttOutput, SttRequest, SAMPLES_PER_MS};
use crate::services::speech::engine_api::{EngineSpec, PipelineConfig, RoutePlan, RouteRequest, StatusLevel, Tier};
use crate::services::speech::events::{FinalPass, SpeechFinal, SpeechPartial, SpeechState, SpeechStatus, SpeechTurnEnd, SpeechUtterance};
use crate::services::speech::pipeline::{start_session, stop_session, AudioSource, EventSink, PipelineStarted, Registry, SessionDeps, SourceRead};
use crate::services::speech_core::VadConfig;

/// Pipeline key of the bench's private sessions.
const PIPELINE_ID: &str = "speech-bench";
/// Silence added (in steps) while a segment is still open after the tail, up to this much.
const EXTRA_SILENCE_STEP_MS: u32 = 500;
const EXTRA_SILENCE_MAX_MS: u32 = 10_000;

// ---------------------------------------------------------------------------------------------
// The audio the pipeline "captures"
// ---------------------------------------------------------------------------------------------

/// Audio waiting to be read by the pipeline's capture thread.
#[derive(Default)]
pub struct ClipFeed {
    state: Mutex<FeedState>,
    cond: Condvar,
}

#[derive(Default)]
struct FeedState {
    queue: VecDeque<Vec<f32>>,
    /// Samples pushed in total.
    pushed: u64,
    /// Samples handed to the capture thread.
    delivered: u64,
    /// Samples the capture thread has processed: what it had been handed when it asked again.
    acked: u64,
    /// The pipeline started; before that the source delivers nothing.
    open: bool,
    closed: bool,
}

impl ClipFeed {
    fn state(&self) -> std::sync::MutexGuard<'_, FeedState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Let the capture thread read (the plan is known now).
    pub fn open(&self) {
        self.state().open = true;
        self.cond.notify_all();
    }

    pub fn push(&self, samples: Vec<f32>) {
        if samples.is_empty() {
            return;
        }
        let mut s = self.state();
        s.pushed += samples.len() as u64;
        s.queue.push_back(samples);
        drop(s);
        self.cond.notify_all();
    }

    /// Samples pushed so far.
    pub fn pushed(&self) -> u64 {
        self.state().pushed
    }

    /// The capture thread has processed everything pushed.
    pub fn consumed(&self) -> bool {
        let s = self.state();
        s.queue.is_empty() && s.acked == s.pushed
    }

    pub fn close(&self) {
        self.state().closed = true;
        self.cond.notify_all();
    }
}

/// The pipeline's [`AudioSource`] over a [`ClipFeed`].
pub struct ClipSource(pub Arc<ClipFeed>);

/// Samples handed to the capture thread per read: 20 ms, like the capture hub's blocks. The pipeline's
/// streaming front end keeps only about two seconds of history for the audio a segment confirms late,
/// so a read must never carry more than a block or two of audio.
const READ_SAMPLES: usize = 20 * SAMPLES_PER_MS;

impl AudioSource for ClipSource {
    fn read(&mut self, out: &mut Vec<f32>) -> SourceRead {
        let mut s = self.0.state();
        // Asking again means the previous read has been processed.
        s.acked = s.delivered;
        if s.open {
            let mut want = READ_SAMPLES;
            while want > 0 {
                let Some(mut block) = s.queue.pop_front() else { break };
                if block.len() > want {
                    let rest = block.split_off(want);
                    s.queue.push_front(rest);
                }
                want -= block.len();
                s.delivered += block.len() as u64;
                out.extend_from_slice(&block);
            }
        }
        SourceRead::default()
    }

    fn wait(&self, timeout: Duration) {
        let s = self.0.state();
        if s.open && !s.queue.is_empty() || s.closed {
            return;
        }
        let _ = self.0.cond.wait_timeout(s, timeout.min(Duration::from_millis(20)));
    }

    fn muted(&self) -> bool {
        false
    }

    fn ended(&self) -> Option<String> {
        None
    }

    fn is_loopback(&self) -> bool {
        false
    }

    fn key(&self) -> String {
        "speech-bench".into()
    }

    fn dropped_frames(&self) -> u64 {
        0
    }

    fn close(self: Box<Self>) {
        self.0.close();
    }
}

// ---------------------------------------------------------------------------------------------
// The pipeline's events
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct FinalRec {
    revision: u32,
    pass: FinalPass,
    text: String,
    lang: Option<String>,
    engine: String,
    model: String,
    start_ms: u64,
    end_ms: u64,
    latency_ms: u32,
    tags: Vec<String>,
    at: Instant,
}

#[derive(Clone, Debug, Default)]
struct SegRec {
    started: bool,
    closed: bool,
    dropped: Option<String>,
    start_ms: u64,
    end_ms: u64,
    partials: u32,
    first_partial_at: Option<Instant>,
    finals: Vec<FinalRec>,
}

#[derive(Default)]
struct TrackerState {
    segs: BTreeMap<u64, SegRec>,
    two_pass: bool,
    error: Option<String>,
    warnings: Vec<String>,
}

impl TrackerState {
    fn complete(&self, rec: &SegRec) -> bool {
        rec.dropped.is_some()
            || if self.two_pass {
                rec.finals.iter().any(|f| f.pass == FinalPass::Second)
            } else {
                !rec.finals.is_empty()
            }
    }

    /// A segment opened and has not closed.
    fn open(&self) -> usize {
        self.segs.values().filter(|r| r.started && !r.closed && r.dropped.is_none()).count()
    }

    /// Closed segments still waiting for their final.
    fn unresolved(&self) -> usize {
        self.segs.values().filter(|r| r.closed && !self.complete(r)).count()
    }
}

/// Collects the pipeline's events (the session's [`EventSink`]). `SIGIL_BENCH_TRACE=1` prints every
/// event to stderr.
#[derive(Default)]
pub struct Tracker {
    state: Mutex<TrackerState>,
    cond: Condvar,
    trace: bool,
}

impl Tracker {
    fn state(&self) -> std::sync::MutexGuard<'_, TrackerState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn new() -> Tracker {
        Tracker {
            trace: std::env::var_os("SIGIL_BENCH_TRACE").is_some_and(|v| !v.is_empty() && v != "0"),
            ..Tracker::default()
        }
    }

    fn trace(&self, what: impl FnOnce() -> String) {
        if self.trace {
            eprintln!("[speech-bench trace] {}", what());
        }
    }

    fn set_plan(&self, plan: &RoutePlan) {
        self.state().two_pass = plan.streaming.is_some() && plan.final_engine.is_some();
    }

    fn update<R>(&self, f: impl FnOnce(&mut TrackerState) -> R) -> R {
        let r = f(&mut self.state());
        self.cond.notify_all();
        r
    }
}

impl EventSink for Tracker {
    fn partial(&self, e: &SpeechPartial) {
        self.trace(|| format!("partial seg {} rev {}: {:?}", e.segment_id, e.revision, e.text));
        self.update(|s| {
            let rec = s.segs.entry(e.segment_id).or_default();
            rec.partials += 1;
            if !e.text.trim().is_empty() {
                rec.first_partial_at.get_or_insert_with(Instant::now);
            }
        });
    }

    fn final_transcript(&self, e: &SpeechFinal) {
        self.trace(|| format!("final seg {} rev {} {:?} {}-{} ms latency {} ms: {:?}", e.segment_id, e.revision, e.pass, e.start_ms, e.end_ms, e.latency_ms, e.text));
        self.update(|s| {
            let rec = s.segs.entry(e.segment_id).or_default();
            rec.start_ms = e.start_ms;
            rec.end_ms = e.end_ms;
            rec.finals.push(FinalRec {
                revision: e.revision,
                pass: e.pass,
                text: e.text.clone(),
                lang: e.lang.clone(),
                engine: e.engine.clone(),
                model: e.model.clone(),
                start_ms: e.start_ms,
                end_ms: e.end_ms,
                latency_ms: e.latency_ms,
                tags: e.tags.clone(),
                at: Instant::now(),
            });
        });
    }

    fn status(&self, e: &SpeechStatus) {
        self.trace(|| format!("status {:?} {} {}", e.state, e.code, e.detail));
        self.update(|s| {
            if e.state == SpeechState::Error {
                s.error.get_or_insert_with(|| format!("{}: {}", e.code, e.detail));
            } else if e.level != StatusLevel::Info && !e.detail.is_empty() && s.warnings.len() < 20 {
                let w = format!("{}: {}", e.code, e.detail);
                if !s.warnings.contains(&w) {
                    s.warnings.push(w);
                }
            }
        });
    }

    fn utterance(&self, e: &SpeechUtterance, audio_ms: u64) {
        self.trace(|| format!("utterance {e:?} ({audio_ms} ms)"));
        self.update(|s| match e {
            SpeechUtterance::SpeechStart { segment_id, start_ms, .. } => {
                let rec = s.segs.entry(*segment_id).or_default();
                rec.started = true;
                rec.start_ms = *start_ms;
            }
            SpeechUtterance::Utterance {
                segment_id, start_ms, end_ms, ..
            } => {
                let rec = s.segs.entry(*segment_id).or_default();
                rec.started = true;
                rec.closed = true;
                rec.start_ms = *start_ms;
                rec.end_ms = *end_ms;
            }
            SpeechUtterance::Dropped { segment_id, reason, .. } => {
                let rec = s.segs.entry(*segment_id).or_default();
                rec.dropped = Some(reason.clone());
            }
        });
    }

    fn turn_end(&self, _e: &SpeechTurnEnd) {}
}

// ---------------------------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------------------------

/// Builds the session's dependencies; production: the app's, with the clip feed as its source.
pub type DepsFactory = Box<dyn Fn(Arc<Tracker>, Arc<ClipFeed>) -> SessionDeps + Send>;

/// The app's session dependencies over a clip feed.
pub fn production_deps(tracker: Arc<Tracker>, feed: Arc<ClipFeed>) -> SessionDeps {
    let mut deps = SessionDeps::production(tracker);
    deps.source = Box::new(move |_source: &str| Ok(Box::new(ClipSource(feed)) as Box<dyn AudioSource>));
    // Segment times are audio times of the fed stream.
    deps.capture_clock = false;
    deps
}

/// The pace the clips are fed at. The pipeline keeps up with audio that arrives at once only when a
/// final engine decodes whole segments; a route whose only engine is a streaming one drops the live
/// text of a stream that falls more than 3 s behind, so such a route is always fed at real time.
pub fn effective_pace(plan: &RoutePlan, requested: Pace) -> Pace {
    if plan.streaming.is_some() && plan.final_engine.is_none() {
        Pace::Realtime
    } else {
        requested
    }
}

/// One item's place in the fed stream.
#[derive(Clone, Debug)]
struct Span {
    /// Where the item's window starts and ends (lead silence, clip, tail and extra silence), in ms.
    window_start_ms: f64,
    window_end_ms: f64,
    clip_start_ms: f64,
    started_at: Instant,
}

pub struct PipelineDriver {
    config: PipelineConfig,
    deps: DepsFactory,
    info: DriverInfo,
    notes: Vec<String>,
    loaded_info: bool,
    /// Pipeline sessions started so far: one per group of same-language items (model loading is
    /// inside the engine's timed window, so the report says how many there were).
    sessions: u32,
}

/// The pipeline configuration of a spec engine: its router request (tier from the execution
/// provider unless `options.route.tier` says), `options.vad`, and, when the run's feed is the
/// pipeline for a plain engine, that engine as the explicit streaming or final engine.
pub fn pipeline_config(def: &EngineDef, paths: &Paths, explicit: bool) -> Result<PipelineConfig, String> {
    let mut config = PipelineConfig {
        pipeline_id: PIPELINE_ID.into(),
        latency_budget_ms: 120_000,
        max_backlog: 10,
        ..PipelineConfig::default()
    };
    if explicit {
        let spec: EngineSpec = engine_spec(def, paths)?;
        if spec.engine == crate::services::speech::engine_api::EngineKind::SherpaOnline {
            config.streaming = Some(spec);
        } else {
            config.final_engine = Some(spec);
        }
        return finish_config(def, config);
    }
    let mut route: RouteRequest = match def.options.get("route") {
        Some(r) => serde_json::from_value(r.clone()).map_err(|e| format!("options.route: {e}"))?,
        None => RouteRequest::default(),
    };
    let tier_given = def.options.get("route").and_then(|r| r.get("tier")).is_some();
    if !tier_given {
        route.tier = match def.execution_provider {
            ExecutionProvider::Cpu => Tier::Cpu,
            ExecutionProvider::Cuda | ExecutionProvider::Dml | ExecutionProvider::Vulkan => Tier::Gpu,
            ExecutionProvider::Auto => Tier::Auto,
        };
    }
    config.route = route;
    finish_config(def, config)
}

fn finish_config(def: &EngineDef, mut config: PipelineConfig) -> Result<PipelineConfig, String> {
    if let Some(v) = def.options.get("vad") {
        config.vad = serde_json::from_value::<VadConfig>(v.clone()).map_err(|e| format!("options.vad: {e}"))?;
    }
    if let Some(b) = def.options.get("hallucinationFilter").and_then(Value::as_bool) {
        config.hallucination_filter = b;
    }
    Ok(config)
}

impl PipelineDriver {
    pub fn new(config: PipelineConfig, deps: DepsFactory) -> PipelineDriver {
        PipelineDriver {
            config,
            deps,
            info: DriverInfo {
                device: "auto".into(),
                runtime: Some("speech pipeline".into()),
                streaming: false,
                ..DriverInfo::default()
            },
            notes: Vec::new(),
            loaded_info: false,
            sessions: 0,
        }
    }

    /// The production driver for `def`.
    pub fn production(def: &EngineDef, paths: &Paths, explicit: bool) -> Result<PipelineDriver, String> {
        Ok(PipelineDriver::new(pipeline_config(def, paths, explicit)?, Box::new(production_deps)))
    }

    fn record_plan(&mut self, registry: &Registry, started: &PipelineStarted) {
        if self.loaded_info {
            return;
        }
        self.loaded_info = true;
        let status = registry.statuses().into_iter().find(|s| s.pipeline_id == started.pipeline_id);
        let engines: Vec<_> = status.iter().flat_map(|s| [s.streaming.as_ref(), s.final_engine.as_ref()]).flatten().collect();
        let device_of = |d: &str| if d == "directml" { "dml".to_string() } else { d.to_string() };
        // The tier that decides the finals: the final engine's device, else the streaming engine's.
        if let Some(e) = status.as_ref().and_then(|s| s.final_engine.as_ref().or(s.streaming.as_ref())) {
            self.info.device = device_of(&e.device);
            self.info.ep_reason = e.fallback_reason.clone();
        }
        self.info.vram_mb = Some(engines.iter().map(|e| e.vram_mb).sum());
        self.info.streaming = started.plan.streaming.is_some();
        let describe = |e: &crate::services::speech::engine_api::EngineInfo| json!({ "engine": e.kind.as_str(), "model": e.model, "device": device_of(&e.device), "runtime": e.runtime, "vram_mb": e.vram_mb });
        self.info.extra = Some(json!({
            "language": started.plan.language,
            "tier": format!("{:?}", started.plan.tier).to_lowercase(),
            "streaming": status.as_ref().and_then(|s| s.streaming.as_ref()).map(describe),
            "final": status.as_ref().and_then(|s| s.final_engine.as_ref()).map(describe),
            "notes": started.plan.notes,
            "missing": started.plan.missing,
        }));
        self.info.runtime = Some(format!(
            "speech pipeline ({})",
            engines.iter().map(|e| format!("{} {}", e.kind.as_str(), e.model)).collect::<Vec<_>>().join(" + ")
        ));
    }

    /// Count a started session in the engine row's facts (`extra.sessions`).
    fn count_session(&mut self) {
        self.sessions += 1;
        if let Some(Value::Object(extra)) = self.info.extra.as_mut() {
            extra.insert("sessions".into(), json!(self.sessions));
        }
    }

    fn run_group(&mut self, reqs: &[SttRequest<'_>]) -> Vec<Result<SttOutput, String>> {
        let fail_all = |msg: String| reqs.iter().map(|_| Err(msg.clone())).collect::<Vec<_>>();
        let tracker = Arc::new(Tracker::new());
        let feed = Arc::new(ClipFeed::default());
        let registry = Registry::new();
        let language = reqs.first().and_then(|r| r.language.clone());
        let mut config = self.config.clone();
        config.language = language.clone().unwrap_or_else(|| "auto".into());
        config.route.language = config.language.clone();
        let deps = (self.deps)(tracker.clone(), feed.clone());
        let started = match tauri::async_runtime::block_on(start_session(&registry, config, deps, false)) {
            Ok(s) => s,
            Err(e) => return fail_all(e),
        };
        tracker.set_plan(&started.plan);
        self.record_plan(&registry, &started);
        self.count_session();
        if started.plan.streaming.is_none() && started.plan.final_engine.is_none() {
            tauri::async_runtime::block_on(stop_session(&registry, PIPELINE_ID));
            return fail_all(format!(
                "the router found no engine ({}){}",
                if started.plan.notes.is_empty() { "no notes".to_string() } else { started.plan.notes.join(", ") },
                if started.plan.missing.is_empty() { String::new() } else { format!("; not installed: {}", started.plan.missing.join(", ")) }
            ));
        }
        let paced = effective_pace(&started.plan, reqs.first().map_or(Pace::Asap, |r| r.feed.pace));
        if paced != reqs.first().map_or(paced, |r| r.feed.pace) {
            let note = "asap pace was fed at real time: the route's only engine is a streaming one, and the pipeline skips the live text of a streaming engine that falls more than 3 s behind";
            if !self.notes.iter().any(|n| n == note) {
                self.notes.push(note.to_string());
            }
        }
        feed.open();

        let mut spans: Vec<Span> = Vec::with_capacity(reqs.len());
        let mut timed_out: Vec<bool> = vec![false; reqs.len()];
        for (i, req) in reqs.iter().enumerate() {
            let feed_cfg = req.feed;
            let t_item = Instant::now();
            let window_start_ms = feed.pushed() as f64 / SAMPLES_PER_MS as f64;
            let lead = vec![0.0; feed_cfg.lead_silence_ms as usize * SAMPLES_PER_MS];
            feed.push(lead);
            let clip_start_ms = feed.pushed() as f64 / SAMPLES_PER_MS as f64;
            if paced == Pace::Realtime {
                let chunk = (feed_cfg.chunk_ms as usize * SAMPLES_PER_MS).max(1);
                let total = req.audio.len() + feed_cfg.tail_silence_ms as usize * SAMPLES_PER_MS;
                let mut sent = 0usize;
                while sent < req.audio.len() {
                    let end = (sent + chunk).min(req.audio.len());
                    let due = Duration::from_micros(((end as u64) * 1_000_000) / 16_000);
                    if let Some(wait) = due.checked_sub(t_item.elapsed()) {
                        std::thread::sleep(wait);
                    }
                    feed.push(req.audio[sent..end].to_vec());
                    sent = end;
                }
                let due = Duration::from_micros(((total as u64) * 1_000_000) / 16_000);
                if let Some(wait) = due.checked_sub(t_item.elapsed()) {
                    std::thread::sleep(wait);
                }
            } else {
                feed.push(req.audio.to_vec());
            }
            feed.push(vec![0.0; feed_cfg.tail_silence_ms as usize * SAMPLES_PER_MS]);
            let deadline = Instant::now() + req.timeout;
            let mut extra_ms = 0u32;
            loop {
                let (open, unresolved, error) = {
                    let s = tracker.state();
                    (s.open(), s.unresolved(), s.error.clone())
                };
                if error.is_some() {
                    break;
                }
                if feed.consumed() {
                    if open > 0 && extra_ms < EXTRA_SILENCE_MAX_MS {
                        // A segment is still open: more silence lets the VAD close it.
                        feed.push(vec![0.0; EXTRA_SILENCE_STEP_MS as usize * SAMPLES_PER_MS]);
                        extra_ms += EXTRA_SILENCE_STEP_MS;
                        continue;
                    }
                    if open == 0 && unresolved == 0 {
                        break;
                    }
                }
                if Instant::now() >= deadline {
                    timed_out[i] = true;
                    break;
                }
                let s = tracker.state();
                let _ = tracker.cond.wait_timeout(s, Duration::from_millis(25));
            }
            if tracker.state().error.is_some() {
                spans.push(Span {
                    window_start_ms,
                    window_end_ms: feed.pushed() as f64 / SAMPLES_PER_MS as f64,
                    clip_start_ms,
                    started_at: t_item,
                });
                break;
            }
            spans.push(Span {
                window_start_ms,
                window_end_ms: feed.pushed() as f64 / SAMPLES_PER_MS as f64,
                clip_start_ms,
                started_at: t_item,
            });
        }
        tauri::async_runtime::block_on(stop_session(&registry, PIPELINE_ID));

        let state = tracker.state();
        for w in &state.warnings {
            if !self.notes.contains(w) {
                self.notes.push(w.clone());
            }
        }
        let mut out = Vec::with_capacity(reqs.len());
        for (i, req) in reqs.iter().enumerate() {
            let Some(span) = spans.get(i) else {
                out.push(Err(state.error.clone().unwrap_or_else(|| "the pipeline stopped before this item".into())));
                continue;
            };
            out.push(assemble(&state, span, req, timed_out[i], paced == Pace::Realtime, state.error.as_deref()));
        }
        out
    }
}

/// The item's output from the segments inside its window.
fn assemble(state: &TrackerState, span: &Span, req: &SttRequest<'_>, timed_out: bool, realtime: bool, pipeline_error: Option<&str>) -> Result<SttOutput, String> {
    let mut ordered: Vec<(&u64, &SegRec)> = state
        .segs
        .iter()
        .filter(|(_, r)| {
            let mid = (r.start_ms as f64 + r.end_ms as f64) / 2.0;
            mid >= span.window_start_ms && mid < span.window_end_ms
        })
        .collect();
    ordered.sort_by_key(|(_, r)| r.start_ms);
    let mut segments = Vec::new();
    let mut tags: Vec<String> = Vec::new();
    let mut language = None;
    let mut partials = 0u64;
    let mut first_partial: Option<Instant> = None;
    let mut last_event: Option<Instant> = None;
    let mut last_latency: Option<f64> = None;
    let mut incomplete = false;
    for (_, rec) in &ordered {
        partials += u64::from(rec.partials);
        if let Some(t) = rec.first_partial_at {
            first_partial = Some(first_partial.map_or(t, |p| p.min(t)));
        }
        if !state.complete(rec) && rec.closed {
            incomplete = true;
        }
        // The latest revision is the user-visible final (a second pass replaces the first).
        let Some(fin) = rec.finals.iter().max_by_key(|f| f.revision) else {
            continue;
        };
        last_event = Some(last_event.map_or(fin.at, |l| l.max(fin.at)));
        last_latency = Some(f64::from(fin.latency_ms));
        language = fin.lang.clone().or(language);
        for t in &fin.tags {
            if !tags.contains(t) {
                tags.push(t.clone());
            }
        }
        if fin.text.trim().is_empty() {
            continue;
        }
        segments.push(Segment {
            start_ms: (fin.start_ms as f64 - span.clip_start_ms).max(0.0),
            end_ms: (fin.end_ms as f64 - span.clip_start_ms).max(0.0),
            text: fin.text.trim().to_string(),
            lang: fin.lang.clone(),
            engine: Some(format!("{} {}", fin.engine, fin.model).trim().to_string()),
            latency_ms: Some(f64::from(fin.latency_ms)),
        });
    }
    if segments.is_empty() && (timed_out || incomplete) {
        return Err(if timed_out { super::stt::timeout_message(req.timeout) } else { "the pipeline did not finish this item".to_string() });
    }
    if segments.is_empty() {
        // No text: the VAD heard nothing, or the engine or the pipeline lost the utterance.
        let lost = ordered
            .iter()
            .find_map(|(_, r)| r.dropped.as_deref().filter(|reason| crate::services::speech::utterance::reasons::is_loss(reason)));
        if let Some(reason) = lost {
            return Err(format!("the pipeline dropped the utterance ({reason})"));
        }
        if let Some(e) = pipeline_error {
            return Err(e.to_string());
        }
    }
    let no_gap = req.language.as_deref().is_some_and(|l| matches!(l, "ja" | "zh" | "yue"));
    let text = segments.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(if no_gap { "" } else { " " });
    Ok(SttOutput {
        text,
        language,
        tags,
        segments,
        partials: Some(partials),
        // From the item's first fed sample. The lead silence is pushed at once (it takes no wall
        // time) and the clip is paced from the same instant, so this is also the time from the
        // clip's start; subtracting the lead here would understate the latency by exactly that much.
        first_partial_ms: first_partial.map(|t| t.saturating_duration_since(span.started_at).as_secs_f64() * 1000.0),
        final_latency_ms: last_latency,
        compute_ms: if realtime { None } else { last_event.map(|t| t.saturating_duration_since(span.started_at).as_secs_f64() * 1000.0) },
    })
}

impl SttDriver for PipelineDriver {
    fn info(&self) -> &DriverInfo {
        &self.info
    }

    fn transcribe(&mut self, req: &SttRequest<'_>) -> Result<SttOutput, String> {
        self.run_group(std::slice::from_ref(req)).into_iter().next().unwrap_or_else(|| Err("no result".into()))
    }

    fn grouped(&self) -> bool {
        true
    }

    fn transcribe_group(&mut self, reqs: &[SttRequest<'_>]) -> Vec<Result<SttOutput, String>> {
        if reqs.is_empty() {
            return Vec::new();
        }
        self.run_group(reqs)
    }

    fn notes(&self) -> Vec<String> {
        self.notes.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::super::spec::{Feed, FeedKind};
    use super::*;
    use crate::services::speech::engine_api::{
        AsrEngine, AsrError, AsrResult, DecodeOptions, EngineCaps, EngineContext, EngineInfo, EngineKind,
    };
    use crate::services::speech::pipeline::EngineFactory;
    use crate::services::speech_core::segmenter::{tests::LevelClassifier, SegmenterEngine};
    use crate::services::speech_core::SpeechSegmenter;

    /// An engine that names the length of what it decoded.
    struct Fake {
        info: EngineInfo,
        fail: bool,
    }

    impl AsrEngine for Fake {
        fn info(&self) -> &EngineInfo {
            &self.info
        }
        fn transcribe(&mut self, audio: &[f32], options: &DecodeOptions) -> Result<AsrResult, AsrError> {
            if self.fail {
                return Err(AsrError::runtime("boom"));
            }
            Ok(AsrResult {
                text: format!("said {}ms", audio.len() / SAMPLES_PER_MS / 100 * 100),
                lang: options.language.clone(),
                ..AsrResult::default()
            })
        }
    }

    fn fake_engine(fail: bool) -> EngineFactory {
        Arc::new(move |spec: &EngineSpec, _ctx: &EngineContext| {
            Ok(Box::new(Fake {
                info: EngineInfo {
                    kind: spec.engine,
                    model: spec.model.clone(),
                    runtime: "fake".into(),
                    device: "cpu".into(),
                    caps: EngineCaps {
                        utterance: true,
                        ..EngineCaps::default()
                    },
                    vram_mb: 0,
                    fallback_reason: None,
                },
                fail,
            }) as Box<dyn AsrEngine>)
        })
    }

    fn test_deps(engines: EngineFactory) -> DepsFactory {
        Box::new(move |tracker, feed| {
            let mut deps = production_deps(tracker, feed);
            deps.engines = engines.clone();
            deps.segmenter = Box::new(|vad: &VadConfig| Ok(Box::new(SegmenterEngine::new(LevelClassifier, vad)) as Box<dyn SpeechSegmenter>));
            deps.turn = Box::new(|| Err("Smart Turn is off in this test".into()));
            deps.second_pass = Box::new(|_, _, _, _| Err(AsrError::unsupported("two-pass is off in this test")));
            deps.route = Box::new(|r| RoutePlan {
                language: r.language.clone(),
                tier: Tier::Cpu,
                final_engine: Some(EngineSpec::new(EngineKind::Whisper, "fake-model")),
                ..RoutePlan::default()
            });
            deps.requirements = Arc::new(|_| Vec::new());
            deps.batch_policy = Arc::new(|_| None);
            deps
        })
    }

    /// The test segmenter reads sample values as speech probabilities: 0.9 is speech.
    fn clip(speech_ms: usize) -> Vec<f32> {
        vec![0.9; speech_ms * SAMPLES_PER_MS]
    }

    fn feed_cfg(pace: Pace) -> Feed {
        Feed {
            kind: FeedKind::Pipeline,
            chunk_ms: 100,
            pace,
            lead_silence_ms: 300,
            tail_silence_ms: 800,
        }
    }

    fn req<'a>(item: &'a str, audio: &'a [f32], feed: &'a Feed) -> SttRequest<'a> {
        SttRequest {
            item,
            audio,
            language: Some("en".into()),
            feed,
            timeout: Duration::from_secs(20),
        }
    }

    #[test]
    fn items_of_a_group_flow_through_the_pipeline_one_at_a_time() {
        let mut driver = PipelineDriver::new(PipelineConfig::default(), test_deps(fake_engine(false)));
        let (a, b, c) = (clip(1200), clip(2000), clip(1500));
        let f = feed_cfg(Pace::Asap);
        let reqs = [req("en/1/clean", &a, &f), req("en/2/clean", &b, &f), req("en/3/clean", &c, &f)];
        let outs = driver.transcribe_group(&reqs);
        assert_eq!(outs.len(), 3);
        for (i, out) in outs.iter().enumerate() {
            let out = out.as_ref().unwrap_or_else(|e| panic!("item {i}: {e}"));
            assert!(out.text.starts_with("said "), "{:?}", out.text);
            assert_eq!(out.segments.len(), 1, "one utterance per item: {:?}", out.segments);
            assert!(out.final_latency_ms.is_some() && out.compute_ms.is_some());
            assert_eq!(out.language.as_deref(), Some("en"));
        }
        // Longer clips decode longer audio (pre-roll and end padding included).
        let ms = |o: &Result<SttOutput, String>| o.as_ref().unwrap().text.trim_start_matches("said ").trim_end_matches("ms").parse::<u32>().unwrap();
        assert!(ms(&outs[1]) > ms(&outs[0]) && ms(&outs[0]) >= 1200);
        assert!(driver.info().runtime.as_deref().unwrap().contains("fake-model"));
        assert_eq!(driver.info().device, "cpu");
        assert!(driver.grouped());
        // The group was one session; the engine row says so, and counts the next one.
        assert_eq!(driver.info().extra.as_ref().unwrap()["sessions"], 1);
        let again = driver.transcribe_group(&[req("en/4/clean", &a, &f)]);
        assert!(again[0].is_ok());
        assert_eq!(driver.info().extra.as_ref().unwrap()["sessions"], 2, "model loading is inside the engine window, so the report counts the sessions");
    }

    /// A tracker state with one closed segment whose first partial came `first_partial_ms` after the item started.
    fn state_with_partial(started_at: Instant, first_partial_ms: u64) -> TrackerState {
        let mut state = TrackerState::default();
        state.segs.insert(
            1,
            SegRec {
                started: true,
                closed: true,
                dropped: None,
                start_ms: 500,
                end_ms: 2500,
                partials: 3,
                first_partial_at: Some(started_at + Duration::from_millis(first_partial_ms)),
                finals: vec![FinalRec {
                    revision: 1,
                    pass: FinalPass::First,
                    text: "hello there".into(),
                    lang: Some("en".into()),
                    engine: "whisper".into(),
                    model: "fake-model".into(),
                    start_ms: 500,
                    end_ms: 2500,
                    latency_ms: 300,
                    tags: Vec::new(),
                    at: started_at + Duration::from_millis(2900),
                }],
            },
        );
        state
    }

    #[test]
    fn first_partial_runs_from_the_clip_start_whatever_the_lead_silence() {
        // The lead silence is pushed at once and the clip is paced from the same instant, so the wall
        // time from the item's start is the time from the clip's start: nothing is subtracted.
        let started_at = Instant::now();
        let span = Span {
            window_start_ms: 0.0,
            window_end_ms: 4000.0,
            clip_start_ms: 500.0,
            started_at,
        };
        let audio = clip(2000);
        let mut f = feed_cfg(Pace::Realtime);
        f.lead_silence_ms = 500;
        let r = req("en/1/clean", &audio, &f);
        for realtime in [true, false] {
            let out = assemble(&state_with_partial(started_at, 800), &span, &r, false, realtime, None).unwrap();
            let got = out.first_partial_ms.unwrap();
            assert!((got - 800.0).abs() < 1.0, "realtime {realtime}: an 800 ms partial reads 800 ms, not 300: {got}");
            // A partial sooner than the lead is long is no longer clamped to zero.
            let out = assemble(&state_with_partial(started_at, 300), &span, &r, false, realtime, None).unwrap();
            let got = out.first_partial_ms.unwrap();
            assert!((got - 300.0).abs() < 1.0, "realtime {realtime}: {got}");
        }
        // Real-time runs report no compute time; asap runs the time to the last final.
        let rt = assemble(&state_with_partial(started_at, 800), &span, &r, false, true, None).unwrap();
        let asap = assemble(&state_with_partial(started_at, 800), &span, &r, false, false, None).unwrap();
        assert!(rt.compute_ms.is_none());
        assert!((asap.compute_ms.unwrap() - 2900.0).abs() < 1.0);
    }

    #[test]
    fn a_failing_engine_costs_its_items_a_clear_error_and_not_the_run() {
        let mut driver = PipelineDriver::new(PipelineConfig::default(), test_deps(fake_engine(true)));
        let a = clip(1000);
        let f = feed_cfg(Pace::Asap);
        let out = driver.transcribe(&req("en/1/clean", &a, &f));
        // The pipeline drops the utterance (engine_error); the item has no text and says so.
        assert!(out.is_err(), "{out:?}");
    }

    #[test]
    fn an_engine_that_cannot_be_created_fails_the_group_with_its_message() {
        let factory: EngineFactory = Arc::new(|_spec, _ctx| Err(AsrError::not_installed("stt/fake-model")));
        let mut driver = PipelineDriver::new(PipelineConfig::default(), test_deps(factory));
        let a = clip(1000);
        let f = feed_cfg(Pace::Asap);
        let outs = driver.transcribe_group(&[req("a", &a, &f), req("b", &a, &f)]);
        assert_eq!(outs.len(), 2);
        for o in outs {
            assert!(o.unwrap_err().contains("stt/fake-model"));
        }
    }

    #[test]
    fn a_router_without_engines_names_what_is_missing() {
        let mut driver = PipelineDriver::new(
            PipelineConfig::default(),
            Box::new(|tracker, feed| {
                let mut deps = production_deps(tracker, feed);
                deps.segmenter = Box::new(|vad: &VadConfig| Ok(Box::new(SegmenterEngine::new(LevelClassifier, vad)) as Box<dyn SpeechSegmenter>));
                deps.route = Box::new(|_| RoutePlan {
                    notes: vec!["gpu_denied".into()],
                    missing: vec!["stt/x".into()],
                    ..RoutePlan::default()
                });
                deps
            }),
        );
        let a = clip(1000);
        let f = feed_cfg(Pace::Asap);
        let err = driver.transcribe(&req("a", &a, &f)).unwrap_err();
        assert!(err.contains("no speech engine") || err.contains("no engine"), "{err}");
    }

    #[test]
    fn pipeline_configs_come_from_the_engine_definition() {
        let paths = Paths::with_base(std::env::temp_dir());
        let def = |v: Value| -> EngineDef {
            let mut base = json!({"id": "p", "task": "stt", "backend": "pipeline"});
            base.as_object_mut().unwrap().extend(v.as_object().unwrap().clone());
            serde_json::from_value(base).unwrap()
        };
        let c = pipeline_config(&def(json!({"execution_provider": "cpu", "options": {"route": {"language": "", "twoPass": true}}})), &paths, false).unwrap();
        assert_eq!((c.route.tier, c.route.two_pass), (Tier::Cpu, true));
        let c = pipeline_config(&def(json!({"execution_provider": "cpu", "options": {"route": {"tier": "gpu", "twoPass": false}}})), &paths, false).unwrap();
        assert_eq!((c.route.tier, c.route.two_pass), (Tier::Gpu, false), "an explicit tier wins");
        let c = pipeline_config(&def(json!({"execution_provider": "auto"})), &paths, false).unwrap();
        assert_eq!(c.route.tier, Tier::Auto);
        assert_eq!((c.max_backlog, c.latency_budget_ms), (10, 120_000));
        let c = pipeline_config(&def(json!({"options": {"vad": {"threshold": 0.7}}})), &paths, false).unwrap();
        assert_eq!(c.vad.threshold, 0.7);
        // An explicit engine: a streaming one fills `streaming`, any other the finals.
        let w = pipeline_config(&def(json!({"backend": "whisper", "model": "whisper/small-q5_1"})), &paths, true).unwrap();
        assert!(w.final_engine.is_some() && w.streaming.is_none());
        let s = pipeline_config(&def(json!({"backend": "sherpa-online", "model": "stt/nemotron-3.5-streaming-560ms-int8"})), &paths, true).unwrap();
        assert!(s.streaming.is_some() && s.final_engine.is_none());
        assert!(pipeline_config(&def(json!({"options": {"route": {"tier": "warp"}}})), &paths, false).is_err());
    }

    #[test]
    fn a_streaming_only_route_is_always_fed_at_real_time() {
        let engine = || Some(EngineSpec::new(EngineKind::SherpaOnline, "stt/x"));
        let streaming_only = RoutePlan { streaming: engine(), ..RoutePlan::default() };
        let two_pass = RoutePlan { streaming: engine(), final_engine: Some(EngineSpec::new(EngineKind::Transcribe, "stt/y")), ..RoutePlan::default() };
        let finals_only = RoutePlan { final_engine: Some(EngineSpec::new(EngineKind::Whisper, "w")), ..RoutePlan::default() };
        assert_eq!(effective_pace(&streaming_only, Pace::Asap), Pace::Realtime);
        assert_eq!(effective_pace(&two_pass, Pace::Asap), Pace::Asap);
        assert_eq!(effective_pace(&finals_only, Pace::Asap), Pace::Asap);
        assert_eq!(effective_pace(&finals_only, Pace::Realtime), Pace::Realtime);
    }

    #[test]
    fn the_source_hands_out_20_ms_blocks_and_knows_when_the_capture_has_processed_them() {
        let feed = Arc::new(ClipFeed::default());
        let mut src = ClipSource(feed.clone());
        feed.push(vec![0.5; 1000]);
        let mut buf = Vec::new();
        src.read(&mut buf);
        assert!(buf.is_empty(), "nothing is delivered before the pipeline has started");
        feed.open();
        let mut sizes = Vec::new();
        for _ in 0..4 {
            buf.clear();
            src.read(&mut buf);
            sizes.push(buf.len());
        }
        assert_eq!(sizes, [320, 320, 320, 40]);
        assert_eq!(feed.pushed(), 1000);
        assert!(!feed.consumed(), "the last block is still being processed");
        buf.clear();
        src.read(&mut buf);
        assert!(buf.is_empty() && feed.consumed(), "asking again acknowledges the previous block");
        feed.push(vec![0.1; 10]);
        assert!(!feed.consumed());
    }
}
