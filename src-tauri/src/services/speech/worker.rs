//! Engine workers of the speech pipeline (P2-A). Internal to the pipeline: no frozen API.
//!
//! Inference never runs on async runtime threads or audio threads: every engine lives on its own
//! worker thread and is dropped there.
//!
//! - **Utterance worker** (finals-only routes): decodes closed utterances strictly in arrival
//!   order, one at a time. The queue in front of it drops its oldest entry past `maxBacklog`
//!   (`dropped` / `backlog`), and an utterance that waited longer than the latency budget is
//!   dropped instead of decoded late (`latency_budget`). When the engine falls behind, everything
//!   already waiting (up to 28 s, 200 ms of silence in between) is decoded as one batch: the first
//!   segment's final carries the text, the others end with `dropped` / `merged`. An engine with a
//!   [`BatchPolicy`] (Groq's billing minimum) may hold a short batch up to 3.5 s (8 s while the
//!   next utterance is being spoken) so later utterances can join it. The route's language-ID
//!   engine runs first in auto mode; a GPU engine that fails at runtime is replaced by the route's
//!   CPU fallback for the rest of the session.
//! - **Stream worker** (streaming routes): owns the streaming engine, turns hypotheses into
//!   `speech:partial` (at most every 80 ms, only on change), ends each segment's stream with the
//!   first-pass final and, on two-pass routes, submits the segment to the two-pass queue
//!   (`two_pass::TwoPass`, P2-C) after that final, so the replacement always has the higher
//!   revision.
//!
//! Errors: `not_installed`, `invalid_input`, `unsupported` and `out_of_memory` (without a
//! fallback) end the pipeline (`speech:status` `error`); other failures cost only the utterance
//! (`dropped` / `engine_error` or `rate_limited`, plus a warning), until five in a row end it.
//! Exceptions: a stream the engine refuses to open for a segment (`unsupported` / `invalid_input`
//! from `begin_stream`, e.g. a language the model lacks) costs only that segment's stream, and an
//! utterance an engine gives up because waiting would pass its latency budget ([`scope`]) ends
//! with `dropped` / `rate_limited` without counting as a failure.
//!
//! Languages: every engine gets a language only when it lists it (or lists none): the utterance
//! engine, the streaming engine (see `pipeline::StreamFeed`) and the two-pass re-decode alike.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use super::engine_api::{AsrEngine, AsrError, AsrErrorKind, AsrResult, DecodeOptions, EngineContext, EngineSpec};
use super::events::{codes, FinalPass};
use super::lang;
use super::pipeline::{
    catalog_id_in, run_session_thread, DoneGuard, EngineFactory, FinalMeta, OutcomeFn, SecondPassFactory, SessionCtx, LOG,
};
use super::two_pass::{FinalJob, TwoPassConfig, TwoPassOutcome};
use super::utterance::{merge_audio, ms_to_samples, reasons, samples_to_ms, MERGE_GAP_MS, MERGE_MAX_MS};
use crate::services::speech_core::utterance::{BoundedQueue, Pop};

/// Transient failures in a row that end the pipeline.
const MAX_CONSECUTIVE_FAILURES: u32 = 5;
/// Longest hold of a short batch (billing minimum), from its oldest utterance's close.
const HOLD: Duration = Duration::from_millis(3_500);
/// Longest hold while the next utterance is being spoken.
const HOLD_WHILE_SPEAKING: Duration = Duration::from_millis(8_000);
/// Minimum spacing of `speech:partial` events of one segment.
const PARTIAL_INTERVAL: Duration = Duration::from_millis(80);

/// Request pacing an utterance engine asks the worker for (the HTTP engine's Groq limiter).
pub(crate) trait BatchPolicy: Send + Sync {
    /// Seconds of audio a request should carry now before it is sent (0 = send at once).
    fn min_batch_seconds(&self) -> f32;
}

/// The two-pass re-decode queue as the stream worker uses it (production: `two_pass::TwoPass`).
pub(crate) trait SecondPass {
    /// Queue a re-decode; `false` when refused (queue full / shut down). Never blocks.
    fn submit(&self, job: FinalJob) -> bool;
    /// Cancel, join; no outcome is delivered after it returns.
    fn shutdown(self: Box<Self>);
}

/// A closed utterance waiting for the utterance engine.
pub(crate) struct UtteranceJob {
    pub segment_id: u64,
    pub audio: Arc<[f32]>,
    /// VAD close (the latency budget counts from here).
    pub closed_at: Instant,
}

/// A closed segment handed to the stream worker.
pub(crate) struct ClosedSegment {
    pub audio: Arc<[f32]>,
    pub closed_at: Instant,
}

/// Capture thread → stream worker, in capture order.
pub(crate) enum StreamMsg {
    /// Open the segment's stream (pre-roll audio included).
    Begin {
        segment_id: u64,
        options: DecodeOptions,
        audio: Vec<f32>,
    },
    Push {
        segment_id: u64,
        audio: Vec<f32>,
    },
    /// The segment closed: end its stream (first-pass final), then two-pass.
    End { segment_id: u64, closed: ClosedSegment },
    /// The segment closed without a stream (engine behind, language unknown): two-pass only.
    FinalOnly { segment_id: u64, closed: ClosedSegment },
    /// Drop the segment's stream (muted, engine behind).
    Abort { segment_id: u64 },
}

impl StreamMsg {
    /// Samples of audio the message carries (for the lag accounting).
    pub(crate) fn samples(&self) -> usize {
        match self {
            StreamMsg::Begin { audio, .. } | StreamMsg::Push { audio, .. } => audio.len(),
            _ => 0,
        }
    }
}

/// The CPU engine that replaces a failing (GPU) utterance engine, created when first needed.
pub(crate) struct Fallback {
    pub spec: EngineSpec,
    pub factory: EngineFactory,
}

/// Engines and factory of the two-pass queue (started on the stream worker thread).
pub(crate) struct SecondPassSetup {
    pub factory: SecondPassFactory,
    pub final_engine: Box<dyn AsrEngine>,
    pub fallback: Option<Box<dyn AsrEngine>>,
    pub config: TwoPassConfig,
}

/// Engine errors that no retry or other utterance can fix.
fn is_fatal(kind: AsrErrorKind) -> bool {
    matches!(
        kind,
        AsrErrorKind::NotInstalled | AsrErrorKind::InvalidInput | AsrErrorKind::Unsupported | AsrErrorKind::OutOfMemory
    )
}

fn error_code(kind: AsrErrorKind) -> &'static str {
    match kind {
        AsrErrorKind::NotInstalled => codes::NOT_INSTALLED,
        AsrErrorKind::RateLimited => codes::RATE_LIMITED,
        _ => codes::ENGINE_ERROR,
    }
}

/// The catalog id to name in a failure status: the one a `not_installed` message names, else the
/// engine's model when it is a catalog id.
fn failure_model_id(e: &AsrError, engine: &dyn AsrEngine) -> Option<String> {
    let from_message = (e.kind == AsrErrorKind::NotInstalled)
        .then(|| catalog_id_in(&e.message))
        .flatten();
    from_message.or_else(|| Some(engine.info().model.clone()).filter(|m| m.contains('/')))
}

fn audio_ms(audio: &[f32]) -> u64 {
    samples_to_ms(audio.len() as u64)
}

/// `code` when `languages` (an engine's `caps.languages`) is empty or lists it, else `None` (the
/// engine then detects on its own, or uses its model's default).
pub(crate) fn supported_language(languages: &[String], code: Option<String>) -> Option<String> {
    code.filter(|c| languages.is_empty() || languages.iter().any(|l| l == c))
}

/// The language an engine should decode with: `code` unless the engine lists its languages and
/// `code` is not one of them (then it detects on its own).
fn language_for(engine: &dyn AsrEngine, code: Option<String>) -> Option<String> {
    supported_language(&engine.info().caps.languages, code)
}

/// Context of the decode running on this thread. The utterance worker sets it around its engine
/// calls (the frozen `AsrEngine` / `DecodeOptions` have no field for it); engines read it in the
/// synchronous part of `transcribe`, on the calling thread:
/// - [`abandoned`](scope::abandoned): the session that asked for the decode ended. An engine
///   checks it right after taking its cancel snapshot, so a stop that landed between the worker's
///   epoch check and that snapshot is honoured (every later stop moves the engine's cancel
///   generation, since the pipeline ends the epoch before it cancels the engines);
/// - [`deadline`](scope::deadline): the utterance's latency deadline (its close plus the
///   latency budget). An engine that would wait past it before it can work (the HTTP engine's rate
///   limits) calls [`give_up`](scope::give_up) and fails; the worker then drops the utterance as
///   `rate_limited` without counting a failure, instead of emitting a final long after the words.
///
/// Threads without a scope (the stream worker, the two-pass queue, tests) see no deadline and a
/// live session.
pub(crate) mod scope {
    use std::cell::RefCell;
    use std::sync::Arc;
    use std::time::Instant;

    /// Whether the session that asked for the decode is still current.
    pub(crate) type Live = Arc<dyn Fn() -> bool + Send + Sync>;

    struct Scope {
        deadline: Option<Instant>,
        live: Option<Live>,
        gave_up: bool,
    }

    thread_local! {
        static CURRENT: RefCell<Option<Scope>> = const { RefCell::new(None) };
    }

    /// Restores the enclosing scope, also on unwind.
    struct Restore(Option<Scope>);

    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            CURRENT.with(|c| *c.borrow_mut() = previous);
        }
    }

    /// Run `f` in a decode scope; returns its result and whether an engine gave up for the
    /// deadline.
    pub(crate) fn run<T>(deadline: Option<Instant>, live: Option<Live>, f: impl FnOnce() -> T) -> (T, bool) {
        let previous = CURRENT.with(|c| {
            c.borrow_mut().replace(Scope {
                deadline,
                live,
                gave_up: false,
            })
        });
        let restore = Restore(previous);
        let out = f();
        let gave_up = CURRENT.with(|c| c.borrow().as_ref().is_some_and(|s| s.gave_up));
        drop(restore);
        (out, gave_up)
    }

    /// The latency deadline of the current decode.
    pub(crate) fn deadline() -> Option<Instant> {
        CURRENT.with(|c| c.borrow().as_ref().and_then(|s| s.deadline))
    }

    /// The session that asked for the current decode ended: return `cancelled` without working.
    pub(crate) fn abandoned() -> bool {
        let live = CURRENT.with(|c| c.borrow().as_ref().and_then(|s| s.live.clone()));
        live.is_some_and(|live| !live())
    }

    /// The engine gives the current utterance up because waiting would pass its deadline.
    pub(crate) fn give_up() {
        CURRENT.with(|c| {
            if let Some(s) = c.borrow_mut().as_mut() {
                s.gave_up = true;
            }
        });
    }
}

/// Run the language-ID engine on `audio`; the detected language (normalized), remembered for
/// streaming engines.
fn detect_language(ctx: &SessionCtx, lid: &mut Option<Box<dyn AsrEngine>>, warned: &mut bool, audio: &[f32]) -> Option<String> {
    let engine = lid.as_mut()?;
    match engine.transcribe(audio, &DecodeOptions::default()) {
        Ok(r) => {
            let code = r.lang.as_deref().and_then(lang::normalize);
            ctx.note_language(code.as_deref());
            code
        }
        Err(e) if e.is_cancelled() => None,
        Err(e) => {
            if !*warned {
                *warned = true;
                ctx.warn(codes::ENGINE_ERROR, format!("spoken language identification failed: {e}"));
            }
            None
        }
    }
}

fn spawn_named(name: String, f: impl FnOnce() + Send + 'static) -> Result<(), String> {
    thread::Builder::new()
        .name(name)
        .spawn(f)
        .map(|_| ())
        .map_err(|e| format!("cannot start a speech worker thread: {e}"))
}

// ---------------------------------------------------------------------------------------------
// Utterance worker
// ---------------------------------------------------------------------------------------------

/// Everything the utterance worker owns.
pub(crate) struct UtteranceWorkerSetup {
    pub ctx: Arc<SessionCtx>,
    pub queue: Arc<BoundedQueue<UtteranceJob>>,
    pub engine: Box<dyn AsrEngine>,
    pub fallback: Option<Fallback>,
    pub language_id: Option<Box<dyn AsrEngine>>,
    pub policy: Option<Arc<dyn BatchPolicy>>,
    /// The capture thread has a segment open.
    pub speaking: Arc<AtomicBool>,
}

/// Start the utterance worker; `done` is dropped after its engines.
pub(crate) fn spawn_utterance_worker(setup: UtteranceWorkerSetup, done: DoneGuard) -> Result<(), String> {
    let name = format!("sigil-speech-{}-final", setup.ctx.pipeline_id);
    spawn_named(name, move || {
        let _done = done;
        let ctx = Arc::clone(&setup.ctx);
        // A panic ends this thread alone: it becomes the session's engine error.
        run_session_thread(&ctx, codes::ENGINE_ERROR, "speech engine", move || UtteranceWorker::new(setup).run());
    })
}

struct UtteranceWorker {
    ctx: Arc<SessionCtx>,
    queue: Arc<BoundedQueue<UtteranceJob>>,
    engine: Box<dyn AsrEngine>,
    fallback: Option<Fallback>,
    language_id: Option<Box<dyn AsrEngine>>,
    lid_warned: bool,
    policy: Option<Arc<dyn BatchPolicy>>,
    speaking: Arc<AtomicBool>,
    /// A job popped while holding that did not fit the batch (decoded next).
    carry: Option<UtteranceJob>,
    failures: u32,
    gap: usize,
    max_batch: usize,
}

impl UtteranceWorker {
    fn new(s: UtteranceWorkerSetup) -> Self {
        Self {
            ctx: s.ctx,
            queue: s.queue,
            engine: s.engine,
            fallback: s.fallback,
            language_id: s.language_id,
            lid_warned: false,
            policy: s.policy,
            speaking: s.speaking,
            carry: None,
            failures: 0,
            gap: ms_to_samples(MERGE_GAP_MS),
            max_batch: ms_to_samples(MERGE_MAX_MS),
        }
    }

    fn run(mut self) {
        while let Some(batch) = self.next_batch() {
            self.ctx.set_backlog(self.queue.len());
            if !self.ctx.is_current() || !self.decode(batch) {
                break;
            }
        }
        tracing::debug!(target: LOG, "[{}] utterance worker of session {} exited", self.ctx.pipeline_id, self.ctx.epoch);
    }

    /// Drop `job` when it waited longer than the latency budget.
    fn expire(&self, job: &UtteranceJob) -> bool {
        let waited = job.closed_at.elapsed();
        if waited <= self.ctx.settings.latency_budget {
            return false;
        }
        self.ctx.dropped(job.segment_id, reasons::LATENCY_BUDGET, audio_ms(&job.audio));
        self.ctx.warn(
            codes::BACKLOG_DROPPED,
            format!(
                "speech recognition is slower than real time; an utterance that waited {:.1} s was dropped",
                waited.as_secs_f32()
            ),
        );
        true
    }

    /// The next batch to decode (in order), `None` once the queue is closed.
    fn next_batch(&mut self) -> Option<Vec<UtteranceJob>> {
        let first = loop {
            let job = match self.carry.take() {
                Some(job) => job,
                None => self.queue.pop()?,
            };
            if !self.expire(&job) {
                break job;
            }
        };
        let budget = self.ctx.settings.latency_budget;
        let mut total = first.audio.len();
        let mut batch = vec![first];
        loop {
            // Everything already waiting that fits (merge on backlog).
            while let Some(job) = self
                .queue
                .pop_front_if(|j| j.closed_at.elapsed() > budget || total + self.gap + j.audio.len() <= self.max_batch)
            {
                if self.expire(&job) {
                    continue;
                }
                total += self.gap + job.audio.len();
                batch.push(job);
            }
            // Hold a short batch for the billing minimum (only when nothing else is waiting).
            let min = self
                .policy
                .as_ref()
                .map_or(0.0, |p| p.min_batch_seconds())
                .max(0.0);
            let min_samples = ms_to_samples((min * 1000.0).min(MERGE_MAX_MS as f32) as u32);
            if min_samples == 0 || total >= min_samples || !self.queue.is_empty() {
                break;
            }
            let hold = if self.speaking.load(Ordering::SeqCst) {
                HOLD_WHILE_SPEAKING
            } else {
                HOLD
            };
            let deadline = batch[0].closed_at + hold;
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            match self.queue.pop_timeout(deadline - now) {
                Pop::Item(job) => {
                    if self.expire(&job) {
                        continue;
                    }
                    if total + self.gap + job.audio.len() <= self.max_batch {
                        total += self.gap + job.audio.len();
                        batch.push(job);
                    } else {
                        self.carry = Some(job);
                        break;
                    }
                }
                Pop::Closed => return None,
                Pop::TimedOut => break,
            }
        }
        Some(batch)
    }

    /// Decode one batch; `false` ends the worker.
    fn decode(&mut self, batch: Vec<UtteranceJob>) -> bool {
        let parts: Vec<Arc<[f32]>> = batch.iter().map(|j| Arc::clone(&j.audio)).collect();
        let audio = merge_audio(&parts, self.gap);
        let language = match &self.ctx.settings.language {
            Some(l) => Some(l.clone()),
            None => detect_language(&self.ctx, &mut self.language_id, &mut self.lid_warned, &audio),
        };
        let started = Instant::now();
        // The engine may give the batch up rather than wait past its oldest utterance's budget.
        let deadline = batch[0].closed_at + self.ctx.settings.latency_budget;
        let live: scope::Live = {
            let ctx = Arc::clone(&self.ctx);
            Arc::new(move || ctx.is_current())
        };
        let (result, gave_up) = scope::run(Some(deadline), Some(live), || loop {
            let options = self.ctx.decode_options(language_for(&*self.engine, language.clone()));
            match self.engine.transcribe(&audio, &options) {
                Err(e) if self.can_promote(&e) => {
                    if !self.promote(&e) {
                        return None;
                    }
                }
                r => break Some(r),
            }
        });
        let Some(result) = result else { return false };
        let first = batch[0].segment_id;
        match result {
            Ok(r) => {
                self.failures = 0;
                tracing::debug!(
                    target: LOG,
                    "[{}] decoded {} utterance(s), {} ms of audio, in {} ms",
                    self.ctx.pipeline_id,
                    batch.len(),
                    audio_ms(&audio),
                    started.elapsed().as_millis()
                );
                self.ctx.note_language(r.lang.as_deref());
                match self.ctx.clean_text(&r) {
                    Some(text) => {
                        let info = self.engine.info();
                        let meta = FinalMeta {
                            lang: r.lang.as_deref().and_then(lang::normalize).or(language),
                            tags: r.tags.clone(),
                            pass: FinalPass::First,
                            engine: info.kind.as_str().to_string(),
                            model: info.model.clone(),
                            merged_until: (batch.len() > 1).then(|| batch[batch.len() - 1].segment_id),
                        };
                        self.ctx.final_transcript(first, text, meta);
                        for job in &batch[1..] {
                            self.ctx.dropped(job.segment_id, reasons::MERGED, audio_ms(&job.audio));
                        }
                    }
                    None => {
                        for job in &batch {
                            self.ctx.dropped(job.segment_id, reasons::NO_SPEECH, audio_ms(&job.audio));
                        }
                    }
                }
                true
            }
            Err(e) if e.is_cancelled() && !self.ctx.is_current() => false,
            Err(e) if gave_up => {
                self.gave_up(&batch, &e);
                true
            }
            Err(e) => self.failed(&batch, e),
        }
    }

    /// The engine gave the batch up because waiting (a rate limit) would pass the latency budget:
    /// the utterances are lost, but the engine did not fail (no failure is counted).
    fn gave_up(&self, batch: &[UtteranceJob], e: &AsrError) {
        for job in batch {
            self.ctx.dropped(job.segment_id, reasons::RATE_LIMITED, audio_ms(&job.audio));
        }
        self.ctx.warn(
            codes::RATE_LIMITED,
            format!("an utterance was dropped: it would have waited past the latency budget ({e})"),
        );
    }

    fn failed(&mut self, batch: &[UtteranceJob], e: AsrError) -> bool {
        let reason = if e.kind == AsrErrorKind::RateLimited {
            reasons::RATE_LIMITED
        } else {
            reasons::ENGINE_ERROR
        };
        for job in batch {
            self.ctx.dropped(job.segment_id, reason, audio_ms(&job.audio));
        }
        let model_id = failure_model_id(&e, &*self.engine);
        if is_fatal(e.kind) {
            self.ctx.fatal(error_code(e.kind), e.to_string(), model_id);
            return false;
        }
        self.failures += 1;
        if self.failures >= MAX_CONSECUTIVE_FAILURES {
            self.ctx.fatal(
                error_code(e.kind),
                format!("{} utterances failed in a row; last error: {e}", self.failures),
                model_id,
            );
            return false;
        }
        self.ctx.warn(error_code(e.kind), e.to_string());
        true
    }

    fn can_promote(&self, e: &AsrError) -> bool {
        self.fallback.is_some()
            && matches!(e.kind, AsrErrorKind::Runtime | AsrErrorKind::OutOfMemory | AsrErrorKind::Busy)
    }

    /// Replace the failing engine by the fallback (once); `false` ends the pipeline.
    fn promote(&mut self, e: &AsrError) -> bool {
        let Some(fb) = self.fallback.take() else { return false };
        let failed = self.engine.info().clone();
        let engine_ctx = EngineContext {
            status: self.ctx.engine_status_sink(fb.spec.engine),
        };
        match (fb.factory)(&fb.spec, &engine_ctx) {
            Ok(engine) => {
                let info = engine.info();
                self.ctx.warn(
                    codes::ENGINE_FALLBACK,
                    format!(
                        "the {} engine ({} on {}) failed ({e}); using {} {} on {} instead",
                        failed.kind, failed.model, failed.device, info.kind, info.model, info.device
                    ),
                );
                self.engine = engine;
                true
            }
            Err(fe) => {
                self.ctx.fatal(
                    error_code(fe.kind),
                    format!("the {} engine failed ({e}) and its fallback could not load ({fe})", failed.kind),
                    Some(fb.spec.model.clone()).filter(|m| m.contains('/')),
                );
                false
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Stream worker
// ---------------------------------------------------------------------------------------------

/// Everything the stream worker owns.
pub(crate) struct StreamWorkerSetup {
    pub ctx: Arc<SessionCtx>,
    pub rx: mpsc::Receiver<StreamMsg>,
    /// Samples sent by the capture thread and not processed yet.
    pub queued: Arc<AtomicUsize>,
    pub engine: Box<dyn AsrEngine>,
    pub second: Option<SecondPassSetup>,
    pub language_id: Option<Box<dyn AsrEngine>>,
    /// The capture side withholds streams until a language is known (a two-pass final covers the
    /// skipped ones). Cleared here when the two-pass queue cannot start: nothing would detect a
    /// language or decode those utterances then.
    pub wait_for_language: Arc<AtomicBool>,
}

/// Start the stream worker; `done` is dropped after its engines (and the two-pass queue).
pub(crate) fn spawn_stream_worker(setup: StreamWorkerSetup, done: DoneGuard) -> Result<(), String> {
    let name = format!("sigil-speech-{}-stream", setup.ctx.pipeline_id);
    spawn_named(name, move || {
        let _done = done;
        let ctx = Arc::clone(&setup.ctx);
        // A panic ends this thread alone: it becomes the session's engine error.
        run_session_thread(&ctx, codes::ENGINE_ERROR, "streaming speech engine", move || run_stream_worker(setup));
    })
}

/// The two-pass queue's callback: emits the replacement final under the session's epoch.
pub(crate) fn outcome_handler(ctx: Arc<SessionCtx>) -> OutcomeFn {
    let mut announced = false;
    Box::new(move |outcome| match outcome {
        TwoPassOutcome::Final {
            segment_id,
            result,
            engine,
            model,
            promoted_fallback,
            ..
        } => {
            if promoted_fallback && !announced {
                announced = true;
                ctx.warn(
                    codes::ENGINE_FALLBACK,
                    format!("the final (two-pass) engine failed; finals now come from {engine} {model}"),
                );
            }
            ctx.note_language(result.lang.as_deref());
            match ctx.clean_text(&result) {
                Some(text) => {
                    let meta = FinalMeta {
                        lang: result.lang.as_deref().and_then(lang::normalize),
                        tags: result.tags.clone(),
                        pass: FinalPass::Second,
                        engine,
                        model,
                        merged_until: None,
                    };
                    ctx.final_transcript(segment_id, text, meta);
                }
                None => {
                    if ctx.is_pending(segment_id) {
                        ctx.dropped(segment_id, reasons::NO_SPEECH, 0);
                    }
                }
            }
        }
        TwoPassOutcome::WorkerDied { detail } => {
            // Final-only segments would be dropped for good: end the session so it restarts.
            ctx.fatal(
                codes::ENGINE_ERROR,
                format!("the second-pass speech engine stopped unexpectedly ({detail})"),
                None,
            );
        }
        TwoPassOutcome::Skipped { segment_id, reason } => {
            let reason = match reason.as_str() {
                "cancelled" => return,
                "budget" => reasons::LATENCY_BUDGET,
                "queue_full" => reasons::QUEUE_FULL,
                "empty" => reasons::NO_SPEECH,
                _ => reasons::ENGINE_ERROR,
            };
            if ctx.is_pending(segment_id) {
                ctx.dropped(segment_id, reason, 0);
            }
        }
    })
}

/// The segment whose stream is open.
struct Current {
    segment_id: u64,
    /// The stream is open (false after an engine error).
    ok: bool,
    last_text: String,
    last_emit: Option<Instant>,
    /// A hypothesis held back by [`PARTIAL_INTERVAL`].
    pending: Option<(String, Option<String>)>,
}

struct StreamWorker {
    ctx: Arc<SessionCtx>,
    engine: Box<dyn AsrEngine>,
    second: Option<Box<dyn SecondPass>>,
    /// Languages of the two-pass final engine (its re-decodes get only those).
    final_languages: Vec<String>,
    language_id: Option<Box<dyn AsrEngine>>,
    lid_warned: bool,
    current: Option<Current>,
    failures: u32,
}

fn run_stream_worker(setup: StreamWorkerSetup) {
    let StreamWorkerSetup {
        ctx,
        rx,
        queued,
        engine,
        second,
        language_id,
        wait_for_language,
    } = setup;
    let final_languages = second
        .as_ref()
        .map(|s| s.final_engine.info().caps.languages.clone())
        .unwrap_or_default();
    let second = second.and_then(|s| {
        let final_info = s.final_engine.info().clone();
        match (s.factory)(s.final_engine, s.fallback, s.config, outcome_handler(Arc::clone(&ctx))) {
            Ok(p) => Some(p),
            Err(e) => {
                // Streams must not wait for a language only the second pass would detect.
                wait_for_language.store(false, Ordering::SeqCst);
                ctx.warn(
                    codes::ENGINE_FALLBACK,
                    format!(
                        "two-pass decoding with {} {} is unavailable ({e}); finals come from the streaming engine",
                        final_info.kind, final_info.model
                    ),
                );
                None
            }
        }
    });
    let mut w = StreamWorker {
        ctx: Arc::clone(&ctx),
        engine,
        second,
        final_languages,
        language_id,
        lid_warned: false,
        current: None,
        failures: 0,
    };
    while let Ok(msg) = rx.recv() {
        let samples = msg.samples();
        let go_on = ctx.is_current() && w.handle(msg);
        queued.fetch_sub(samples, Ordering::SeqCst);
        if !go_on || !ctx.is_current() {
            break;
        }
    }
    w.engine.abort_stream();
    if let Some(p) = w.second.take() {
        p.shutdown();
    }
    tracing::debug!(target: LOG, "[{}] stream worker of session {} exited", ctx.pipeline_id, ctx.epoch);
}

impl StreamWorker {
    /// `false` ends the worker.
    fn handle(&mut self, msg: StreamMsg) -> bool {
        match msg {
            StreamMsg::Begin {
                segment_id,
                options,
                audio,
            } => {
                if self.current.take().is_some() {
                    self.engine.abort_stream();
                }
                self.current = Some(Current {
                    segment_id,
                    ok: false,
                    last_text: String::new(),
                    last_emit: None,
                    pending: None,
                });
                match self.engine.begin_stream(&options) {
                    Ok(()) => {
                        if let Some(c) = self.current.as_mut() {
                            c.ok = true;
                        }
                        self.push(segment_id, &audio)
                    }
                    Err(e) => self.begin_failed(e),
                }
            }
            StreamMsg::Push { segment_id, audio } => self.push(segment_id, &audio),
            StreamMsg::Abort { segment_id } => {
                if self.current.as_ref().is_some_and(|c| c.segment_id == segment_id) {
                    self.current = None;
                    self.engine.abort_stream();
                }
                true
            }
            StreamMsg::End { segment_id, closed } => self.end(segment_id, closed),
            StreamMsg::FinalOnly { segment_id, closed } => {
                let language = self.final_language(&closed.audio);
                self.second_pass(segment_id, closed, None, language, reasons::STREAM_SKIPPED)
            }
        }
    }

    fn push(&mut self, segment_id: u64, audio: &[f32]) -> bool {
        let Some(c) = self.current.as_ref() else { return true };
        if c.segment_id != segment_id || !c.ok || audio.is_empty() {
            return true;
        }
        match self.engine.push_stream(audio) {
            Ok(hyp) => {
                self.failures = 0;
                let hyp = hyp.map(|h| (h.text, h.lang));
                self.partial(hyp);
                true
            }
            Err(e) => self.stream_error(e),
        }
    }

    /// Emit a changed hypothesis (throttled; the newest held back one goes out later).
    fn partial(&mut self, hyp: Option<(String, Option<String>)>) {
        let Some(c) = self.current.as_mut() else { return };
        if let Some((text, lang)) = hyp {
            let text = text.trim().to_string();
            if text.is_empty() || text == c.last_text {
                c.pending = None;
            } else {
                c.pending = Some((text, lang));
            }
        }
        if c.last_emit.is_some_and(|t| t.elapsed() < PARTIAL_INTERVAL) {
            return;
        }
        let Some((text, lang)) = c.pending.take() else { return };
        let lang = lang.as_deref().and_then(lang::normalize);
        if self.ctx.partial(c.segment_id, &text, lang) {
            c.last_text = text;
            c.last_emit = Some(Instant::now());
        }
    }

    /// End the segment's stream: first-pass final, then two-pass.
    fn end(&mut self, segment_id: u64, closed: ClosedSegment) -> bool {
        let current = self.current.take().filter(|c| c.segment_id == segment_id);
        let language = self.final_language(&closed.audio);
        match current {
            Some(c) if c.ok => match self.engine.end_stream() {
                Ok(r) => {
                    self.failures = 0;
                    self.ctx.note_language(r.lang.as_deref());
                    if let Some(text) = self.ctx.clean_text(&r) {
                        let info = self.engine.info();
                        let meta = FinalMeta {
                            lang: r.lang.as_deref().and_then(lang::normalize).or_else(|| language.clone()),
                            tags: r.tags.clone(),
                            pass: FinalPass::First,
                            engine: info.kind.as_str().to_string(),
                            model: info.model.clone(),
                            merged_until: None,
                        };
                        self.ctx.final_transcript(segment_id, text, meta);
                    }
                    self.second_pass(segment_id, closed, Some(r), language, reasons::NO_SPEECH)
                }
                Err(e) => {
                    if !self.stream_error(e) {
                        return false;
                    }
                    self.second_pass(segment_id, closed, None, language, reasons::ENGINE_ERROR)
                }
            },
            Some(_) => self.second_pass(segment_id, closed, None, language, reasons::ENGINE_ERROR),
            None => self.second_pass(segment_id, closed, None, language, reasons::STREAM_SKIPPED),
        }
    }

    /// The explicit language, else the language-ID result for this segment (auto mode).
    fn final_language(&mut self, audio: &[f32]) -> Option<String> {
        match &self.ctx.settings.language {
            Some(l) => Some(l.clone()),
            None => detect_language(&self.ctx, &mut self.language_id, &mut self.lid_warned, audio),
        }
    }

    /// Queue the two-pass re-decode; without one, a segment that got no first final ends with
    /// `dropped` (`reason`).
    fn second_pass(
        &mut self,
        segment_id: u64,
        closed: ClosedSegment,
        first_pass: Option<AsrResult>,
        language: Option<String>,
        reason: &str,
    ) -> bool {
        let has_final = !self.ctx.is_pending(segment_id);
        let ms = audio_ms(&closed.audio);
        match &self.second {
            Some(p) => {
                let job = FinalJob {
                    segment_id,
                    revision: self.ctx.peek_revision(segment_id).unwrap_or(0),
                    audio: closed.audio,
                    options: self
                        .ctx
                        .decode_options(supported_language(&self.final_languages, language)),
                    // A first-pass result the post-filter removed was never emitted: the job
                    // then counts as having no first final (final-only budget, never evicted).
                    first_pass: first_pass.filter(|_| has_final),
                    closed_at: closed.closed_at,
                };
                if !p.submit(job) && !has_final {
                    self.ctx.dropped(segment_id, reasons::QUEUE_FULL, ms);
                }
            }
            None => {
                if !has_final {
                    self.ctx.dropped(segment_id, reason, ms);
                }
            }
        }
        true
    }

    /// `begin_stream` failed. A refusal of this segment's options (`unsupported` / `invalid_input`:
    /// a language the model lacks or needs, say) costs only the segment's stream: a two-pass final
    /// still covers it, and without one it counts as a transient failure (five in a row end the
    /// pipeline, since every utterance would be lost). Other errors are [`Self::stream_error`]s.
    fn begin_failed(&mut self, e: AsrError) -> bool {
        if !matches!(e.kind, AsrErrorKind::Unsupported | AsrErrorKind::InvalidInput) {
            return self.stream_error(e);
        }
        if let Some(c) = self.current.as_mut() {
            c.ok = false;
        }
        self.engine.abort_stream();
        if self.second.is_some() {
            self.ctx.warn(
                error_code(e.kind),
                format!("the streaming engine skipped an utterance (its final still follows): {e}"),
            );
            return true;
        }
        self.transient_failure(e)
    }

    /// A streaming call failed: the segment loses its stream; `false` ends the worker.
    fn stream_error(&mut self, e: AsrError) -> bool {
        if let Some(c) = self.current.as_mut() {
            c.ok = false;
        }
        self.engine.abort_stream();
        if e.is_cancelled() {
            return self.ctx.is_current();
        }
        if is_fatal(e.kind) {
            let model_id = failure_model_id(&e, &*self.engine);
            self.ctx.fatal(error_code(e.kind), e.to_string(), model_id);
            return false;
        }
        self.transient_failure(e)
    }

    /// Count a failure that cost one segment; `false` (after five in a row) ends the worker.
    fn transient_failure(&mut self, e: AsrError) -> bool {
        let model_id = failure_model_id(&e, &*self.engine);
        self.failures += 1;
        if self.failures >= MAX_CONSECUTIVE_FAILURES {
            self.ctx.fatal(
                error_code(e.kind),
                format!("the streaming engine failed {} times in a row; last error: {e}", self.failures),
                model_id,
            );
            return false;
        }
        self.ctx.warn(error_code(e.kind), format!("the streaming engine failed: {e}"));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::speech::engine_api::{EngineCaps, EngineInfo, EngineKind};

    struct Caps(EngineInfo);

    impl AsrEngine for Caps {
        fn info(&self) -> &EngineInfo {
            &self.0
        }
        fn transcribe(&mut self, _audio: &[f32], _options: &DecodeOptions) -> Result<AsrResult, AsrError> {
            Ok(AsrResult::default())
        }
    }

    fn engine(languages: &[&str]) -> Caps {
        Caps(EngineInfo {
            kind: EngineKind::Whisper,
            model: "m".into(),
            runtime: "test".into(),
            device: "cpu".into(),
            caps: EngineCaps {
                utterance: true,
                languages: languages.iter().map(|s| s.to_string()).collect(),
                ..EngineCaps::default()
            },
            vram_mb: 0,
            fallback_reason: None,
        })
    }

    #[test]
    fn languages_are_passed_only_to_engines_that_know_them() {
        assert_eq!(language_for(&engine(&[]), Some("ja".into())).as_deref(), Some("ja"));
        assert_eq!(language_for(&engine(&["en"]), Some("ja".into())), None);
        assert_eq!(language_for(&engine(&["en", "ja"]), Some("ja".into())).as_deref(), Some("ja"));
        assert_eq!(language_for(&engine(&["en"]), None), None);
        let two = ["en".to_string(), "de".to_string()];
        assert_eq!(supported_language(&two, Some("de".into())).as_deref(), Some("de"));
        assert_eq!(supported_language(&two, Some("ja".into())), None);
        assert_eq!(supported_language(&[], Some("ja".into())).as_deref(), Some("ja"));
    }

    #[test]
    fn a_decode_scope_carries_the_deadline_liveness_and_give_up() {
        assert_eq!(scope::deadline(), None);
        assert!(!scope::abandoned());
        let deadline = Instant::now() + Duration::from_secs(2);
        let live: scope::Live = Arc::new(|| true);
        let (seen, gave_up) = scope::run(Some(deadline), Some(live), || {
            assert!(!scope::abandoned());
            assert_eq!(scope::deadline(), Some(deadline));
            // A nested scope (a fallback engine created inside a decode) is its own.
            let ended: scope::Live = Arc::new(|| false);
            let ((), inner) = scope::run(None, Some(ended), || {
                assert!(scope::abandoned() && scope::deadline().is_none());
                scope::give_up();
            });
            assert!(inner);
            scope::deadline()
        });
        assert_eq!((seen, gave_up), (Some(deadline), false), "the inner give-up stays inner");
        let ((), gave_up) = scope::run(None, None, scope::give_up);
        assert!(gave_up);
        assert_eq!(scope::deadline(), None, "restored after the scope");
    }

    #[test]
    fn error_kinds_split_into_fatal_and_transient() {
        for k in [AsrErrorKind::NotInstalled, AsrErrorKind::InvalidInput, AsrErrorKind::Unsupported, AsrErrorKind::OutOfMemory] {
            assert!(is_fatal(k), "{k:?}");
        }
        for k in [AsrErrorKind::Network, AsrErrorKind::RateLimited, AsrErrorKind::Runtime, AsrErrorKind::Busy, AsrErrorKind::Cancelled] {
            assert!(!is_fatal(k), "{k:?}");
        }
        assert_eq!(error_code(AsrErrorKind::NotInstalled), codes::NOT_INSTALLED);
        assert_eq!(error_code(AsrErrorKind::RateLimited), codes::RATE_LIMITED);
        assert_eq!(error_code(AsrErrorKind::Network), codes::ENGINE_ERROR);
    }

    #[test]
    fn stream_messages_account_their_audio() {
        let closed = || ClosedSegment {
            audio: Arc::from(vec![0.0f32; 5]),
            closed_at: Instant::now(),
        };
        assert_eq!(
            StreamMsg::Begin {
                segment_id: 1,
                options: DecodeOptions::default(),
                audio: vec![0.0; 7]
            }
            .samples(),
            7
        );
        assert_eq!(StreamMsg::Push { segment_id: 1, audio: vec![0.0; 3] }.samples(), 3);
        assert_eq!(StreamMsg::End { segment_id: 1, closed: closed() }.samples(), 0);
        assert_eq!(StreamMsg::Abort { segment_id: 1 }.samples(), 0);
    }
}
