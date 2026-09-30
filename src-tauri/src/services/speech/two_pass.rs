//! Two-pass final re-decode queue with a budget (P2-C; research §1.1 "two-pass captioning").
//!
//! A streaming route shows partials from the streaming engine and ends each segment with that
//! engine's (first-pass) final. On a two-pass route the stream worker (P2-A) then submits the
//! closed segment here: the route's `final` engine re-decodes the **whole** closed utterance on
//! this queue's own worker thread, and the result replaces the first final under the same
//! `segmentId` with the next revision (`FinalPass::Second`). Nothing here shortens, splits or
//! re-times a segment: the job's audio is decoded exactly as the VAD closed it (Smart Turn's
//! commit hints never touch it).
//!
//! # Budget
//! A replacement is only worth showing while the caption is fresh, so a job with a first-pass
//! final is skipped (`budget`: the first final stands) when it cannot finish within
//! `budget_ms + budget_per_audio_second_ms × audio seconds` (capped at `max_budget_ms`) of the
//! segment's close: checked before the decode starts and again when it returns (an engine that
//! can be cancelled is not interrupted for this; its late result is discarded). The length-scaled
//! part lets CPU finals of long utterances through (Qwen3-ASR-0.6B at RTF ~0.25 needs ~2 s for 8 s)
//! while a GPU final normally lands within a few hundred milliseconds. A job without an emitted
//! first-pass final (the stream was skipped, failed or heard nothing, or the post-filter removed
//! its text; see [`FinalJob::first_pass`]) is the segment's only chance of a final, so it gets the
//! pipeline's utterance budget (`final_only_budget_ms`, 20 s).
//!
//! # Queue
//! Jobs are decoded in submission (segment) order. At most `max_queue` wait; beyond that the
//! oldest waiting replacement job is skipped (`queue_full`), or the new job is refused when it is
//! a replacement and only final-only jobs wait (an utterance's only final outranks an
//! improvement). [`TwoPass::submit`] never blocks.
//!
//! # Failures
//! The first failure of the primary (GPU) engine promotes the CPU `fallback` for the rest of the
//! session: the primary is dropped on the worker thread (freeing its VRAM), the same job is
//! re-decoded by the fallback, and every later outcome says `promoted_fallback`. Without a
//! fallback a failure skips the job (`engine_error`), and an engine that is gone for good
//! (`not_installed`, `unsupported`, `out_of_memory`, or five failures in a row) stops decoding:
//! later jobs are skipped at once. Bad input (`invalid_input`) costs only that job.
//!
//! # Frozen signatures (P2-S0)
//! - [`TwoPass::spawn`]`(primary, fallback, config, on_outcome)`: owns the engines and one worker
//!   thread. `on_outcome` runs on that thread; the pipeline (P2-A) re-checks its epoch inside it
//!   before emitting `speech:final`.
//! - [`TwoPass::submit`]`(FinalJob) -> bool`: `false` when the job was refused (queue full / shut
//!   down); never blocks.
//! - [`TwoPass::cancel_pending`]: drop queued jobs and cancel the running decode.
//! - [`TwoPass::shutdown`]: cancel, join the worker; no `on_outcome` call after it returns.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle, ThreadId};
use std::time::{Duration, Instant};

use super::engine_api::{AsrEngine, AsrError, AsrErrorKind, AsrResult, Canceller, DecodeOptions, SAMPLE_RATE};

/// Log target (shared with the pipeline).
const LOG: &str = "sigil::speech";
/// Failures in a row (without a fallback) after which the queue stops decoding.
const MAX_CONSECUTIVE_FAILURES: u32 = 5;

/// `TwoPassOutcome::Skipped` reasons (wire strings the pipeline maps to `dropped` reasons).
pub mod skip {
    /// The job could not finish within its budget; the first-pass final stands.
    pub const BUDGET: &str = "budget";
    /// Evicted by newer jobs (or refused).
    pub const QUEUE_FULL: &str = "queue_full";
    /// The re-decode heard nothing (or the audio was empty).
    pub const EMPTY: &str = "empty";
    /// `cancel_pending` / `shutdown`.
    pub const CANCELLED: &str = "cancelled";
    /// The final engine failed (and no fallback could take over).
    pub const ENGINE_ERROR: &str = "engine_error";
}

/// Queue settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwoPassConfig {
    /// Base budget of a replacement job, from its segment's close (ms).
    pub budget_ms: u32,
    /// Jobs that may wait; the oldest is skipped beyond this.
    pub max_queue: usize,
    /// Extra budget per second of utterance audio (ms), so slower (CPU) finals of long
    /// utterances still land.
    pub budget_per_audio_second_ms: u32,
    /// Cap of a replacement job's budget (ms).
    pub max_budget_ms: u32,
    /// Budget of a job without a first-pass final: its only final (ms; the pipeline's default
    /// `latencyBudgetMs`).
    pub final_only_budget_ms: u32,
}

impl Default for TwoPassConfig {
    fn default() -> Self {
        Self {
            budget_ms: 1_500,
            max_queue: 4,
            budget_per_audio_second_ms: 500,
            max_budget_ms: 8_000,
            final_only_budget_ms: 20_000,
        }
    }
}

impl TwoPassConfig {
    /// How long after its segment closed a job of `samples` (16 kHz) may finish.
    pub fn budget_for(&self, samples: usize, replaces_first_pass: bool) -> Duration {
        let audio_ms = samples as u64 * 1_000 / u64::from(SAMPLE_RATE);
        let scaled = u64::from(self.budget_ms)
            .saturating_add(audio_ms.saturating_mul(u64::from(self.budget_per_audio_second_ms)) / 1_000)
            .min(u64::from(self.max_budget_ms.max(self.budget_ms)));
        let ms = if replaces_first_pass {
            scaled
        } else {
            scaled.max(u64::from(self.final_only_budget_ms))
        };
        Duration::from_millis(ms)
    }
}

/// One segment to re-decode.
#[derive(Clone, Debug)]
pub struct FinalJob {
    pub segment_id: u64,
    /// Revision the replacement final must carry (the first final's revision + 1).
    pub revision: u32,
    /// 16 kHz mono audio of the closed segment.
    pub audio: Arc<[f32]>,
    pub options: DecodeOptions,
    /// The first-pass final that was **emitted** for the segment (it stands when the re-decode is
    /// skipped or fails). `None` (or blank text) when no first final went out: the stream was
    /// skipped, failed or heard nothing, or the session's post-filter removed its text. Such a job
    /// is the segment's only chance of a final: it gets `final_only_budget_ms` and is never
    /// evicted for a replacement, so the stream worker must not pass a result it did not emit.
    pub first_pass: Option<AsrResult>,
    /// When the segment closed (VAD end); the budget counts from here.
    pub closed_at: Instant,
}

impl FinalJob {
    /// An emitted first-pass final with text exists, so this job only improves it (see
    /// [`FinalJob::first_pass`]).
    fn replaces_first_pass(&self) -> bool {
        self.first_pass
            .as_ref()
            .is_some_and(|r| !r.text.trim().is_empty())
    }
}

/// What happened to a job.
#[derive(Clone, Debug)]
pub enum TwoPassOutcome {
    /// Emit this as the segment's final with `revision`.
    Final {
        segment_id: u64,
        /// The job's revision (informational: the pipeline numbers revisions in its segment book).
        #[allow(dead_code)]
        revision: u32,
        result: AsrResult,
        /// `EngineKind` wire name and model of the engine that produced it.
        engine: String,
        model: String,
        /// The CPU fallback produced it (the GPU engine failed).
        promoted_fallback: bool,
    },
    /// No replacement: the first-pass final stands. `reason`: `budget`, `queue_full`, `empty`,
    /// `cancelled`, `engine_error` ([`skip`]).
    Skipped { segment_id: u64, reason: String },
}

fn skipped(segment_id: u64, reason: &str) -> TwoPassOutcome {
    TwoPassOutcome::Skipped {
        segment_id,
        reason: reason.to_string(),
    }
}

/// A waiting job.
struct Queued {
    job: FinalJob,
    deadline: Instant,
    replaces: bool,
}

#[derive(Default)]
struct State {
    queue: VecDeque<Queued>,
    /// Outcomes decided outside the worker (evictions, cancellations), delivered by the worker.
    notices: Vec<TwoPassOutcome>,
    shutdown: bool,
    /// A decode is running.
    running: bool,
    /// `cancel_pending` / `shutdown` hit the running decode.
    cancel_running: bool,
    /// Interrupts the active engine.
    canceller: Option<Canceller>,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    config: TwoPassConfig,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Drop the waiting jobs (reported as `cancelled`) and interrupt the running decode.
    fn cancel_all(&self, shutdown: bool) {
        let canceller = {
            let mut st = self.lock();
            st.shutdown |= shutdown;
            let ids: Vec<u64> = st.queue.drain(..).map(|q| q.job.segment_id).collect();
            st.notices
                .extend(ids.into_iter().map(|id| skipped(id, skip::CANCELLED)));
            if st.running {
                st.cancel_running = true;
                st.canceller.clone()
            } else {
                None
            }
        };
        self.wake.notify_all();
        if let Some(c) = canceller {
            c.cancel();
        }
    }
}

/// The re-decode queue.
pub struct TwoPass {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
    worker_id: ThreadId,
}

impl TwoPass {
    /// Start the worker thread that owns `primary` and `fallback` (they are dropped on it).
    pub fn spawn(
        primary: Box<dyn AsrEngine>,
        fallback: Option<Box<dyn AsrEngine>>,
        config: TwoPassConfig,
        on_outcome: Box<dyn FnMut(TwoPassOutcome) + Send>,
    ) -> Result<Self, AsrError> {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                canceller: primary.canceller(),
                ..State::default()
            }),
            wake: Condvar::new(),
            config,
        });
        let mut worker = Worker {
            shared: Arc::clone(&shared),
            engine: primary,
            fallback,
            promoted: false,
            failures: 0,
            disabled: false,
            on_outcome,
        };
        let handle = thread::Builder::new()
            .name("sigil-speech-two-pass".into())
            .spawn(move || worker.run())
            .map_err(|e| AsrError::runtime(format!("cannot start the two-pass worker: {e}")))?;
        let worker_id = handle.thread().id();
        Ok(Self {
            shared,
            worker: Some(handle),
            worker_id,
        })
    }

    /// Queue a re-decode; `false` when refused (shut down, or a replacement while only final-only
    /// jobs wait in a full queue). Never blocks.
    pub fn submit(&self, job: FinalJob) -> bool {
        let config = &self.shared.config;
        let replaces = job.replaces_first_pass();
        let deadline = job.closed_at + config.budget_for(job.audio.len(), replaces);
        {
            let mut st = self.shared.lock();
            if st.shutdown {
                return false;
            }
            if st.queue.len() >= config.max_queue.max(1) {
                match st.queue.iter().position(|q| q.replaces) {
                    Some(i) => {
                        let old = st.queue.remove(i).expect("index from position");
                        st.notices
                            .push(skipped(old.job.segment_id, skip::QUEUE_FULL));
                    }
                    None if replaces => return false,
                    None => {
                        let old = st.queue.pop_front().expect("a full queue");
                        st.notices
                            .push(skipped(old.job.segment_id, skip::QUEUE_FULL));
                    }
                }
            }
            st.queue.push_back(Queued { job, deadline, replaces });
        }
        self.shared.wake.notify_all();
        true
    }

    /// Drop queued jobs and cancel the running decode (each reported as `cancelled`).
    #[allow(dead_code)] // Frozen API (P2-S0); the pipeline stops the whole queue with `shutdown`.
    pub fn cancel_pending(&self) {
        self.shared.cancel_all(false);
    }

    /// Cancel everything and join the worker (its engines are dropped there). No `on_outcome` call
    /// happens after this returns (called on the worker thread itself it cannot join, and only
    /// stops it).
    pub fn shutdown(mut self) {
        self.shared.cancel_all(true);
        if let Some(handle) = self.worker.take() {
            if thread::current().id() == self.worker_id {
                return;
            }
            if handle.join().is_err() {
                tracing::error!(target: LOG, "the two-pass worker panicked");
            }
        }
    }
}

impl Drop for TwoPass {
    /// Stops the worker without waiting for a running decode (use [`TwoPass::shutdown`] to wait).
    fn drop(&mut self) {
        if self.worker.is_some() {
            self.shared.cancel_all(true);
        }
    }
}

/// The worker thread's state.
struct Worker {
    shared: Arc<Shared>,
    engine: Box<dyn AsrEngine>,
    fallback: Option<Box<dyn AsrEngine>>,
    promoted: bool,
    failures: u32,
    /// The engine is gone for good: skip every job.
    disabled: bool,
    on_outcome: Box<dyn FnMut(TwoPassOutcome) + Send>,
}

enum Next {
    Notices(Vec<TwoPassOutcome>),
    Job(Box<Queued>),
    Exit,
}

impl Worker {
    fn run(&mut self) {
        loop {
            let next = {
                let mut st = self.shared.lock();
                loop {
                    if !st.notices.is_empty() {
                        break Next::Notices(std::mem::take(&mut st.notices));
                    }
                    if st.shutdown {
                        break Next::Exit;
                    }
                    if let Some(q) = st.queue.pop_front() {
                        st.running = true;
                        st.cancel_running = false;
                        break Next::Job(Box::new(q));
                    }
                    st = self
                        .shared
                        .wake
                        .wait(st)
                        .unwrap_or_else(PoisonError::into_inner);
                }
            };
            match next {
                Next::Notices(list) => list.into_iter().for_each(|o| (self.on_outcome)(o)),
                Next::Exit => break,
                Next::Job(q) => {
                    let outcome = self.process(*q);
                    let cancelled = {
                        let mut st = self.shared.lock();
                        st.running = false;
                        std::mem::take(&mut st.cancel_running)
                    };
                    // A cancelled decode's result is never delivered as a final.
                    let outcome = match outcome {
                        TwoPassOutcome::Final { segment_id, .. } if cancelled => skipped(segment_id, skip::CANCELLED),
                        o => o,
                    };
                    (self.on_outcome)(outcome);
                }
            }
        }
        tracing::debug!(target: LOG, "two-pass worker stopped");
    }

    fn cancel_requested(&self) -> bool {
        self.shared.lock().cancel_running
    }

    fn process(&mut self, q: Queued) -> TwoPassOutcome {
        let Queued { job, deadline, replaces } = q;
        let id = job.segment_id;
        if job.audio.is_empty() {
            return skipped(id, skip::EMPTY);
        }
        if self.disabled {
            return skipped(id, skip::ENGINE_ERROR);
        }
        loop {
            if self.cancel_requested() {
                return skipped(id, skip::CANCELLED);
            }
            if Instant::now() >= deadline {
                tracing::debug!(target: LOG, "two-pass: segment {id} skipped (budget spent while waiting)");
                return skipped(id, skip::BUDGET);
            }
            let result = self.engine.transcribe(&job.audio, &job.options);
            if self.cancel_requested() {
                return skipped(id, skip::CANCELLED);
            }
            match result {
                Ok(result) => {
                    self.failures = 0;
                    if Instant::now() > deadline {
                        tracing::debug!(
                            target: LOG,
                            "two-pass: segment {id} decoded after its budget ({} ms decode); the first final stands",
                            result.decode_ms
                        );
                        return skipped(id, skip::BUDGET);
                    }
                    if result.text.trim().is_empty() {
                        return skipped(id, skip::EMPTY);
                    }
                    let info = self.engine.info();
                    return TwoPassOutcome::Final {
                        segment_id: id,
                        revision: job.revision,
                        result,
                        engine: info.kind.as_str().to_string(),
                        model: info.model.clone(),
                        promoted_fallback: self.promoted,
                    };
                }
                Err(e) if e.is_cancelled() => {
                    let reason = if Instant::now() >= deadline { skip::BUDGET } else { skip::CANCELLED };
                    return skipped(id, reason);
                }
                Err(e) if e.kind == AsrErrorKind::InvalidInput => {
                    tracing::warn!(target: LOG, "two-pass: segment {id} could not be re-decoded: {e}");
                    return skipped(id, skip::ENGINE_ERROR);
                }
                Err(e) => {
                    if self.promote(&e) {
                        continue; // the same job, on the fallback
                    }
                    self.failed(&e, replaces);
                    return skipped(id, skip::ENGINE_ERROR);
                }
            }
        }
    }

    /// Replace the failed primary by the fallback for the rest of the session.
    fn promote(&mut self, e: &AsrError) -> bool {
        if self.promoted {
            return false;
        }
        let Some(fallback) = self.fallback.take() else { return false };
        let failed = std::mem::replace(&mut self.engine, fallback);
        {
            let (f, n) = (failed.info(), self.engine.info());
            tracing::warn!(
                target: LOG,
                "two-pass: the {} final engine ({} on {}) failed ({e}); {} {} on {} takes over for this session",
                f.kind,
                f.model,
                f.device,
                n.kind,
                n.model,
                n.device
            );
        }
        self.shared.lock().canceller = self.engine.canceller();
        drop(failed); // frees its (V)RAM now, on this thread
        self.promoted = true;
        self.failures = 0;
        true
    }

    /// A failure without a fallback.
    fn failed(&mut self, e: &AsrError, replaces: bool) {
        self.failures += 1;
        let gone = matches!(e.kind, AsrErrorKind::NotInstalled | AsrErrorKind::Unsupported | AsrErrorKind::OutOfMemory);
        if gone || self.failures >= MAX_CONSECUTIVE_FAILURES {
            self.disabled = true;
            tracing::warn!(
                target: LOG,
                "two-pass: the final engine failed ({e}); re-decoding stops for this session{}",
                if replaces { " (first-pass finals stand)" } else { "" }
            );
        } else {
            tracing::warn!(target: LOG, "two-pass: the final engine failed: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::speech::engine_api::{EngineCaps, EngineInfo, EngineKind};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;

    type Script = Box<dyn FnMut(&[f32], &DecodeOptions, &AtomicBool) -> Result<AsrResult, AsrError> + Send>;

    /// A scripted engine: records the audio lengths it decoded, reports its drop and the thread.
    struct Fake {
        info: EngineInfo,
        script: Script,
        cancel: Arc<AtomicBool>,
        calls: Arc<Mutex<Vec<usize>>>,
        dropped_on: Arc<Mutex<Option<ThreadId>>>,
    }

    impl Drop for Fake {
        fn drop(&mut self) {
            *self.dropped_on.lock().unwrap() = Some(thread::current().id());
        }
    }

    impl AsrEngine for Fake {
        fn info(&self) -> &EngineInfo {
            &self.info
        }
        fn transcribe(&mut self, audio: &[f32], options: &DecodeOptions) -> Result<AsrResult, AsrError> {
            self.cancel.store(false, Ordering::SeqCst);
            self.calls.lock().unwrap().push(audio.len());
            (self.script)(audio, options, &self.cancel)
        }
        fn canceller(&self) -> Option<Canceller> {
            Some(Canceller::from_flag(self.cancel.clone()))
        }
    }

    struct Handles {
        calls: Arc<Mutex<Vec<usize>>>,
        dropped_on: Arc<Mutex<Option<ThreadId>>>,
    }

    fn fake(kind: EngineKind, model: &str, device: &str, script: Script) -> (Box<dyn AsrEngine>, Handles) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let dropped_on = Arc::new(Mutex::new(None));
        let engine = Fake {
            info: EngineInfo {
                kind,
                model: model.into(),
                runtime: "test".into(),
                device: device.into(),
                caps: EngineCaps {
                    utterance: true,
                    ..EngineCaps::default()
                },
                vram_mb: 0,
                fallback_reason: None,
            },
            script,
            cancel: Arc::new(AtomicBool::new(false)),
            calls: Arc::clone(&calls),
            dropped_on: Arc::clone(&dropped_on),
        };
        (Box::new(engine), Handles { calls, dropped_on })
    }

    fn text(t: &str) -> AsrResult {
        AsrResult {
            text: t.into(),
            lang: Some("ja".into()),
            ..AsrResult::default()
        }
    }

    /// Echoes "<model>:<samples>".
    fn echo(model: &'static str) -> Script {
        Box::new(move |audio, _, _| Ok(text(&format!("{model}:{}", audio.len()))))
    }

    fn job(segment_id: u64, samples: usize, first: Option<&str>, closed_at: Instant) -> FinalJob {
        FinalJob {
            segment_id,
            revision: 7,
            audio: Arc::from(vec![0.1f32; samples]),
            options: DecodeOptions {
                language: Some("ja".into()),
                ..DecodeOptions::default()
            },
            first_pass: first.map(text),
            closed_at,
        }
    }

    fn collector() -> (Box<dyn FnMut(TwoPassOutcome) + Send>, mpsc::Receiver<TwoPassOutcome>) {
        let (tx, rx) = mpsc::channel();
        (Box::new(move |o| drop(tx.send(o))), rx)
    }

    fn recv(rx: &mpsc::Receiver<TwoPassOutcome>) -> TwoPassOutcome {
        rx.recv_timeout(Duration::from_secs(10))
            .expect("an outcome")
    }

    fn skip_reason(o: &TwoPassOutcome) -> Option<(u64, &str)> {
        match o {
            TwoPassOutcome::Skipped { segment_id, reason } => Some((*segment_id, reason.as_str())),
            _ => None,
        }
    }

    /// A gate the scripted engine waits on (so tests control when a decode finishes).
    fn gate() -> (Arc<(Mutex<bool>, Condvar)>, impl Fn()) {
        let g = Arc::new((Mutex::new(false), Condvar::new()));
        let open = {
            let g = Arc::clone(&g);
            move || {
                *g.0.lock().unwrap() = true;
                g.1.notify_all();
            }
        };
        (g, open)
    }

    fn wait_gate(g: &(Mutex<bool>, Condvar), cancel: &AtomicBool) -> bool {
        let mut open = g.0.lock().unwrap();
        while !*open {
            if cancel.load(Ordering::SeqCst) {
                return false;
            }
            open = g.1.wait_timeout(open, Duration::from_millis(5)).unwrap().0;
        }
        true
    }

    #[test]
    fn budget_scales_with_the_utterance_and_final_only_jobs_get_the_long_one() {
        let c = TwoPassConfig::default();
        assert_eq!(c.budget_for(0, true), Duration::from_millis(1_500));
        assert_eq!(c.budget_for(8 * 16_000, true), Duration::from_millis(5_500));
        assert_eq!(c.budget_for(30 * 16_000, true), Duration::from_millis(8_000));
        assert_eq!(c.budget_for(8 * 16_000, false), Duration::from_millis(20_000));
        let odd = TwoPassConfig {
            budget_ms: 3_000,
            max_budget_ms: 1_000,
            ..TwoPassConfig::default()
        };
        assert_eq!(odd.budget_for(16_000, true), Duration::from_millis(3_000), "the cap never cuts the base");
    }

    /// The final replaces the first pass under the same segment with the job's revision, and the
    /// engine decodes the whole closed utterance (a turn end never shortens a caption final).
    #[test]
    fn a_re_decode_replaces_the_first_final_over_the_whole_utterance() {
        let (engine, h) = fake(EngineKind::Transcribe, "stt/qwen3-asr-1.7b-q8_0", "cuda", echo("gpu"));
        let (on, rx) = collector();
        let tp = TwoPass::spawn(engine, None, TwoPassConfig::default(), on).unwrap();
        let now = Instant::now();
        assert!(tp.submit(job(3, 8 * 16_000, Some("first"), now)));
        match recv(&rx) {
            TwoPassOutcome::Final {
                segment_id,
                revision,
                result,
                engine,
                model,
                promoted_fallback,
            } => {
                assert_eq!((segment_id, revision), (3, 7));
                assert_eq!(result.text, format!("gpu:{}", 8 * 16_000));
                assert_eq!((engine.as_str(), model.as_str()), ("transcribe", "stt/qwen3-asr-1.7b-q8_0"));
                assert!(!promoted_fallback);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(*h.calls.lock().unwrap(), vec![8 * 16_000]);
        let worker = tp.worker_id;
        tp.shutdown();
        assert_eq!(*h.dropped_on.lock().unwrap(), Some(worker), "the engine is dropped on the worker thread");
    }

    #[test]
    fn decode_options_reach_the_engine() {
        let script: Script = Box::new(|_, options, _| {
            Ok(text(&format!(
                "{}|{}",
                options.language.as_deref().unwrap_or("-"),
                options.hotwords.join(",")
            )))
        });
        let (engine, _h) = fake(EngineKind::SherpaOffline, "cpu", "cpu", script);
        let (on, rx) = collector();
        let tp = TwoPass::spawn(engine, None, TwoPassConfig::default(), on).unwrap();
        let mut j = job(9, 1_000, None, Instant::now());
        j.options.hotwords = vec!["Sigil".into(), "VRChat".into()];
        assert!(tp.submit(j));
        match recv(&rx) {
            TwoPassOutcome::Final { result, .. } => assert_eq!(result.text, "ja|Sigil,VRChat"),
            other => panic!("{other:?}"),
        }
        tp.shutdown();
    }

    #[test]
    fn jobs_past_their_budget_are_skipped_but_final_only_jobs_still_decode() {
        let (engine, h) = fake(EngineKind::Transcribe, "gpu", "cuda", echo("gpu"));
        let (on, rx) = collector();
        let tp = TwoPass::spawn(engine, None, TwoPassConfig::default(), on).unwrap();
        let long_ago = Instant::now() - Duration::from_secs(10);
        assert!(tp.submit(job(1, 16_000, Some("first"), long_ago)));
        assert_eq!(skip_reason(&recv(&rx)), Some((1, skip::BUDGET)));
        assert!(h.calls.lock().unwrap().is_empty(), "a stale job is never decoded");
        // No first-pass text: this is the segment's only final, so the long budget applies.
        assert!(tp.submit(job(2, 16_000, Some("  "), long_ago)));
        assert!(matches!(recv(&rx), TwoPassOutcome::Final { segment_id: 2, .. }));
        // Nor when no first final was emitted (the post-filter removed it: the stream worker
        // passes `None`).
        assert!(tp.submit(job(3, 16_000, None, long_ago)));
        assert!(matches!(recv(&rx), TwoPassOutcome::Final { segment_id: 3, .. }));
        tp.shutdown();
    }

    #[test]
    fn a_result_that_arrives_after_the_budget_is_discarded() {
        let config = TwoPassConfig {
            budget_ms: 50,
            budget_per_audio_second_ms: 0,
            ..TwoPassConfig::default()
        };
        let slow: Script = Box::new(|_, _, _| {
            thread::sleep(Duration::from_millis(150));
            Ok(text("late"))
        });
        let (engine, _h) = fake(EngineKind::SherpaOffline, "cpu", "cpu", slow);
        let (on, rx) = collector();
        let tp = TwoPass::spawn(engine, None, config, on).unwrap();
        assert!(tp.submit(job(4, 16_000, Some("first"), Instant::now())));
        assert_eq!(skip_reason(&recv(&rx)), Some((4, skip::BUDGET)));
        tp.shutdown();
    }

    #[test]
    fn a_failing_gpu_engine_promotes_the_cpu_fallback_for_the_session() {
        let failing: Script = Box::new(|_, _, _| Err(AsrError::runtime("CUDA error: device lost")));
        let (gpu, gpu_h) = fake(EngineKind::Transcribe, "stt/qwen3-asr-1.7b-q8_0", "cuda", failing);
        let (cpu, cpu_h) = fake(EngineKind::SherpaOffline, "stt/qwen3-asr-0.6b-int8", "cpu", echo("cpu"));
        let (on, rx) = collector();
        let tp = TwoPass::spawn(gpu, Some(cpu), TwoPassConfig::default(), on).unwrap();
        let worker = tp.worker_id;
        let now = Instant::now();
        assert!(tp.submit(job(1, 16_000, Some("first"), now)));
        assert!(tp.submit(job(2, 32_000, Some("first"), now)));
        for (want_id, want_len) in [(1u64, 16_000usize), (2, 32_000)] {
            match recv(&rx) {
                TwoPassOutcome::Final {
                    segment_id,
                    result,
                    engine,
                    model,
                    promoted_fallback,
                    ..
                } => {
                    assert_eq!(segment_id, want_id);
                    assert_eq!(result.text, format!("cpu:{want_len}"));
                    assert_eq!((engine.as_str(), model.as_str()), ("sherpa-offline", "stt/qwen3-asr-0.6b-int8"));
                    assert!(promoted_fallback);
                }
                other => panic!("{other:?}"),
            }
        }
        assert_eq!(gpu_h.calls.lock().unwrap().len(), 1, "the failed GPU engine is not tried again");
        assert_eq!(*gpu_h.dropped_on.lock().unwrap(), Some(worker), "and was dropped on the worker at once");
        assert_eq!(*cpu_h.calls.lock().unwrap(), vec![16_000, 32_000]);
        tp.shutdown();
        assert_eq!(*cpu_h.dropped_on.lock().unwrap(), Some(worker));
    }

    #[test]
    fn without_a_fallback_failures_skip_and_a_lost_engine_stops_decoding() {
        let calls = Arc::new(AtomicUsize::new(0));
        let script: Script = {
            let calls = Arc::clone(&calls);
            Box::new(move |_, _, _| {
                let n = calls.fetch_add(1, Ordering::SeqCst);
                match n {
                    0 => Err(AsrError::runtime("transient")),
                    1 => Err(AsrError::invalid_input("audio too short")),
                    2 => Ok(text("ok")),
                    _ => Err(AsrError::new(AsrErrorKind::OutOfMemory, "VRAM")),
                }
            })
        };
        let (engine, h) = fake(EngineKind::Transcribe, "gpu", "cuda", script);
        let (on, rx) = collector();
        // Room for all five jobs: on a loaded machine the worker may not take job 1 before job 5 is
        // submitted, and the default `max_queue` (4) would then skip it as `queue_full` (covered by
        // its own test below); this test is about engine errors only.
        let config = TwoPassConfig {
            max_queue: 8,
            ..TwoPassConfig::default()
        };
        let tp = TwoPass::spawn(engine, None, config, on).unwrap();
        let now = Instant::now();
        for id in 1..=5 {
            assert!(tp.submit(job(id, 16_000, Some("first"), now)));
        }
        assert_eq!(skip_reason(&recv(&rx)), Some((1, skip::ENGINE_ERROR)));
        assert_eq!(skip_reason(&recv(&rx)), Some((2, skip::ENGINE_ERROR)));
        assert!(matches!(recv(&rx), TwoPassOutcome::Final { segment_id: 3, .. }));
        assert_eq!(skip_reason(&recv(&rx)), Some((4, skip::ENGINE_ERROR)));
        assert_eq!(skip_reason(&recv(&rx)), Some((5, skip::ENGINE_ERROR)));
        assert_eq!(h.calls.lock().unwrap().len(), 4, "out of memory without a fallback stops re-decoding");
        tp.shutdown();
    }

    #[test]
    fn empty_results_and_empty_audio_are_skipped() {
        let (engine, _h) = fake(EngineKind::SherpaOffline, "cpu", "cpu", Box::new(|_, _, _| Ok(text("   "))));
        let (on, rx) = collector();
        let tp = TwoPass::spawn(engine, None, TwoPassConfig::default(), on).unwrap();
        assert!(tp.submit(job(1, 16_000, Some("first"), Instant::now())));
        assert_eq!(skip_reason(&recv(&rx)), Some((1, skip::EMPTY)));
        assert!(tp.submit(job(2, 0, None, Instant::now())));
        assert_eq!(skip_reason(&recv(&rx)), Some((2, skip::EMPTY)));
        tp.shutdown();
    }

    #[test]
    fn a_full_queue_skips_the_oldest_replacement_and_keeps_only_finals() {
        let (g, open) = gate();
        let script: Script = {
            let g = Arc::clone(&g);
            Box::new(move |audio, _, cancel| {
                if !wait_gate(&g, cancel) {
                    return Err(AsrError::cancelled());
                }
                Ok(text(&format!("{}", audio.len())))
            })
        };
        let (engine, _h) = fake(EngineKind::Transcribe, "gpu", "cuda", script);
        let (on, rx) = collector();
        let config = TwoPassConfig {
            max_queue: 2,
            ..TwoPassConfig::default()
        };
        let tp = TwoPass::spawn(engine, None, config, on).unwrap();
        let now = Instant::now();
        assert!(tp.submit(job(1, 100, Some("a"), now))); // running (blocked on the gate)
                                                         // Wait until job 1 left the queue.
        let started = Instant::now();
        while !tp.shared.lock().queue.is_empty() || !tp.shared.lock().running {
            assert!(started.elapsed() < Duration::from_secs(5));
            thread::sleep(Duration::from_millis(2));
        }
        assert!(tp.submit(job(2, 200, None, now))); // final-only
        assert!(tp.submit(job(3, 300, Some("c"), now))); // replacement
        assert!(tp.submit(job(4, 400, None, now))); // full: the replacement (3) goes
        assert!(!tp.submit(job(5, 500, Some("e"), now)), "a replacement never evicts an only final");
        assert!(tp.submit(job(6, 600, None, now))); // only finals wait: the oldest (2) goes
        open();
        let mut finals = Vec::new();
        let mut skips = Vec::new();
        for _ in 0..5 {
            match recv(&rx) {
                TwoPassOutcome::Final { segment_id, .. } => finals.push(segment_id),
                TwoPassOutcome::Skipped { segment_id, reason } => skips.push((segment_id, reason)),
            }
        }
        skips.sort();
        assert_eq!(skips, vec![(2, skip::QUEUE_FULL.to_string()), (3, skip::QUEUE_FULL.to_string())]);
        assert_eq!(finals, vec![1, 4, 6], "decoded in segment order");
        tp.shutdown();
    }

    #[test]
    fn cancel_pending_cancels_the_running_decode_and_the_queue() {
        let (g, _open) = gate();
        let script: Script = {
            let g = Arc::clone(&g);
            Box::new(move |_, _, cancel| {
                if !wait_gate(&g, cancel) {
                    return Err(AsrError::cancelled());
                }
                Ok(text("never"))
            })
        };
        let (engine, _h) = fake(EngineKind::Transcribe, "gpu", "cuda", script);
        let (on, rx) = collector();
        let tp = TwoPass::spawn(engine, None, TwoPassConfig::default(), on).unwrap();
        let now = Instant::now();
        assert!(tp.submit(job(1, 100, Some("a"), now)));
        let started = Instant::now();
        while !tp.shared.lock().running {
            assert!(started.elapsed() < Duration::from_secs(5));
            thread::sleep(Duration::from_millis(2));
        }
        assert!(tp.submit(job(2, 100, None, now)));
        tp.cancel_pending();
        let mut got: Vec<(u64, String)> = (0..2)
            .map(|_| match recv(&rx) {
                TwoPassOutcome::Skipped { segment_id, reason } => (segment_id, reason),
                other => panic!("{other:?}"),
            })
            .collect();
        got.sort();
        assert_eq!(got, vec![(1, skip::CANCELLED.to_string()), (2, skip::CANCELLED.to_string())]);
        // The queue keeps working after a cancel.
        assert!(tp.submit(job(3, 100, Some("a"), Instant::now())));
        tp.shutdown();
    }

    #[test]
    fn shutdown_joins_and_nothing_is_delivered_afterwards() {
        let delivered = Arc::new(AtomicUsize::new(0));
        let on: Box<dyn FnMut(TwoPassOutcome) + Send> = {
            let delivered = Arc::clone(&delivered);
            Box::new(move |_| {
                delivered.fetch_add(1, Ordering::SeqCst);
            })
        };
        let slow: Script = Box::new(|_, _, cancel| {
            for _ in 0..200 {
                if cancel.load(Ordering::SeqCst) {
                    return Err(AsrError::cancelled());
                }
                thread::sleep(Duration::from_millis(5));
            }
            Ok(text("done"))
        });
        let (engine, h) = fake(EngineKind::Whisper, "whisper/base-q5_1", "cpu", slow);
        let tp = TwoPass::spawn(engine, None, TwoPassConfig::default(), on).unwrap();
        for id in 1..=3 {
            assert!(tp.submit(job(id, 100, Some("x"), Instant::now())));
        }
        thread::sleep(Duration::from_millis(20));
        let started = Instant::now();
        tp.shutdown();
        assert!(started.elapsed() < Duration::from_secs(1), "the running decode was cancelled");
        let after = delivered.load(Ordering::SeqCst);
        assert!(h.dropped_on.lock().unwrap().is_some(), "engines are dropped before shutdown returns");
        thread::sleep(Duration::from_millis(50));
        assert_eq!(delivered.load(Ordering::SeqCst), after);
    }

    #[test]
    fn submit_after_shutdown_is_refused_and_drop_stops_the_worker() {
        let (engine, h) = fake(EngineKind::Whisper, "w", "cpu", echo("w"));
        let (on, _rx) = collector();
        let tp = TwoPass::spawn(engine, None, TwoPassConfig::default(), on).unwrap();
        tp.shared.cancel_all(true);
        assert!(!tp.submit(job(1, 100, None, Instant::now())));
        drop(tp);
        let started = Instant::now();
        while h.dropped_on.lock().unwrap().is_none() {
            assert!(started.elapsed() < Duration::from_secs(5), "the worker exits after drop");
            thread::sleep(Duration::from_millis(2));
        }
    }
}
