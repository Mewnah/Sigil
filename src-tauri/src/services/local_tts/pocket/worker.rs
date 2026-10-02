//! The worker both P6-I voice-cloning engines run on ([`super`] Pocket TTS and
//! `local_tts::chatterbox_ort`): one thread per engine (`sigil-tts-<name>`) that owns the loaded
//! model and runs every request in order, never on the main thread or an async worker. It follows
//! `local_tts::engine` (the Kokoro / Piper / Kitten worker) and reuses its sink port
//! ([`SinkPort`], [`AppSink`]), sentence chunking and constants, with typed cloning requests and
//! without that engine's unconditional warm-up phrase (a cloning model's warm-up is the reference
//! voice it prepares, which the backend decides).
//!
//! - **Requests.** `speak` with text pushes the sentence into the begun `tts_sink` utterance; `speak`
//!   with an empty text only loads the model and prepares the request's voice ([`CloneBackend::
//!   warm_up`]): the TS backends send one at Start, so a missing download, a missing voice or a
//!   failing graphics driver is reported then, and the first sentence does not pay for the load.
//! - **One model at a time.** A request whose [`CloneBackend::key`] differs loads the new model and
//!   frees the previous one first. A loaded model is dropped by `unload` (unless a request already
//!   waits behind the unload: a stop followed at once by a start keeps the model), or after the idle
//!   timeout when nothing holds it.
//! - **Held while started.** A successful empty-text request (the TS backends' Start) holds the
//!   model: the idle timeout does not free it, however long the gaps between sentences, because a
//!   reload costs a cold start (seconds, for Chatterbox much more on the CPU) before the next
//!   sentence while the service shows Connected. `unload` (Stop) releases the hold. A model loaded
//!   only by sentences (a panel preview) is not held.
//! - **Order.** First in, first out, except that among `speak` requests waiting together the one
//!   whose sink utterance plays first goes first (the sink's begin order).
//! - **PCM contract.** Audio goes straight into the sink utterance in pushes of at most one second,
//!   so a cancel is noticed promptly; the utterance is never ended here (the TS pipeline ends or
//!   cancels it).
//! - **Cancel.** `cancel` or a sink cancel (stop / skip) stops a running request at the backend's
//!   next check (between decode steps for Chatterbox, between decoder chunks for Pocket); a waiting
//!   request is answered `"cancelled"` at once.
//! - **Latency.** When less than [`CRITICAL_AHEAD_MS`] of audio is queued ahead of an utterance, its
//!   first clause is synthesized on its own (`local_tts::text::chunks`).
//! - **Panics.** A panic in model code fails only its own request: the model is dropped and the
//!   worker keeps serving. Should the thread still end, waiting requests are answered and the next
//!   request starts a new worker.

use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tokio::sync::oneshot;

use crate::services::local_tts::commands::{CloningTtsRequest, SpeakReport};
use crate::services::local_tts::engine::{SinkPort, CANCELLED, CRITICAL_AHEAD_MS, WORKER_STOPPED};
use crate::services::local_tts::text;
use crate::services::tts_sink::SinkError;

/// Longest push into the sink (seconds of audio), so cancels are noticed between pushes.
const PUSH_SECONDS: f32 = 1.0;

/// A voice-cloning engine as the worker drives it. Every method runs on the worker thread.
pub(crate) trait CloneBackend: Send + Sync + 'static {
    /// The loaded model (owned by the worker thread).
    type Model: Send + 'static;

    /// Short engine name (`pocket`, `chatterbox`): logs and the thread name.
    fn name(&self) -> &'static str;

    /// What must match for a loaded model to serve `request` (model, threads, execution
    /// provider...). Cheap; `Err` refuses the request (unknown model, bad option).
    fn key(&self, request: &CloningTtsRequest) -> Result<String, String>;

    /// Load the model for `request` (blocking). `cancel` may be checked between expensive steps.
    fn load(&self, request: &CloningTtsRequest, cancel: &AtomicBool) -> Result<Self::Model, String>;

    /// Prepare `request`'s voice and warm the model up: after every load, and for an empty-text
    /// request (Start). Errors fail that request (a missing library voice is reported at Start).
    fn warm_up(&self, model: &mut Self::Model, request: &CloningTtsRequest, cancel: &AtomicBool) -> Result<(), String>;

    /// Synthesize `text` (one chunk of `request.text`) with `request`'s voice and options. `out`
    /// receives mono audio in order with its sample rate and returns `false` to stop. `Ok(false)` =
    /// stopped early (cancelled).
    fn synthesize(
        &self,
        model: &mut Self::Model,
        request: &CloningTtsRequest,
        text: &str,
        cancel: &AtomicBool,
        out: &mut dyn FnMut(&[f32], u32) -> bool,
    ) -> Result<bool, String>;
}

type Reply<T> = oneshot::Sender<Result<T, String>>;

enum JobKind {
    Speak {
        request: CloningTtsRequest,
        utterance_id: String,
        reply: Reply<SpeakReport>,
    },
    Unload {
        reply: Reply<()>,
    },
}

struct Job {
    seq: u64,
    cancel: Arc<AtomicBool>,
    received: Instant,
    kind: JobKind,
}

impl Job {
    /// The sink utterance of a speak job with text (warm-ups have none).
    fn utterance(&self) -> Option<&str> {
        match &self.kind {
            JobKind::Speak { utterance_id, request, .. } if !request.text.trim().is_empty() => Some(utterance_id),
            _ => None,
        }
    }

    fn is_speak(&self) -> bool {
        matches!(self.kind, JobKind::Speak { .. })
    }

    fn refuse(self, error: &str) {
        let e = error.to_string();
        match self.kind {
            JobKind::Speak { reply, .. } => {
                let _ = reply.send(Err(e));
            }
            JobKind::Unload { reply } => {
                let _ = reply.send(Err(e));
            }
        }
    }
}

struct Running {
    seq: u64,
    utterance: Option<String>,
    cancel: Arc<AtomicBool>,
    cancellable: bool,
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    running: Option<Running>,
    next_seq: u64,
    worker: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    wake: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// A loaded model and the key it was loaded for.
struct Loaded<M> {
    key: String,
    model: M,
}

/// The worker of one cloning engine (see the module docs). Cheap to clone (`Arc` inside).
pub(crate) struct CloneWorker<B: CloneBackend> {
    shared: Arc<Shared>,
    backend: Arc<B>,
    sink: Arc<dyn SinkPort>,
    idle_unload: Duration,
}

impl<B: CloneBackend> Clone for CloneWorker<B> {
    fn clone(&self) -> Self {
        Self {
            shared: self.shared.clone(),
            backend: self.backend.clone(),
            sink: self.sink.clone(),
            idle_unload: self.idle_unload,
        }
    }
}

impl<B: CloneBackend> CloneWorker<B> {
    pub fn new(backend: Arc<B>, sink: Arc<dyn SinkPort>, idle_unload: Duration) -> Self {
        Self {
            shared: Arc::new(Shared {
                queue: Mutex::new(Queue::default()),
                wake: Condvar::new(),
            }),
            backend,
            sink,
            idle_unload,
        }
    }

    /// Speak `request` into the begun sink utterance `utterance_id`; an empty `request.text` only
    /// loads the model and prepares the voice (no utterance needed).
    pub fn speak(&self, request: CloningTtsRequest, utterance_id: String) -> oneshot::Receiver<Result<SpeakReport, String>> {
        let (tx, rx) = oneshot::channel();
        self.submit(JobKind::Speak {
            request,
            utterance_id,
            reply: tx,
        });
        rx
    }

    /// Free the model after the requests queued before this call, unless a request already waits
    /// behind it (a restart keeps the model).
    pub fn unload(&self) -> oneshot::Receiver<Result<(), String>> {
        let (tx, rx) = oneshot::channel();
        self.submit(JobKind::Unload { reply: tx });
        rx
    }

    /// Cancel the request of one utterance, or (`None`) every waiting and running request.
    pub fn cancel(&self, utterance_id: Option<&str>) {
        let refused: Vec<Job> = {
            let mut q = self.shared.lock();
            let matches = |job: &Job| match utterance_id {
                Some(id) => job.utterance() == Some(id),
                None => job.is_speak(),
            };
            let (drop, keep): (Vec<Job>, Vec<Job>) = q.jobs.drain(..).partition(|j| matches(j));
            q.jobs = keep.into();
            if let Some(r) = &q.running {
                let hit = match utterance_id {
                    Some(id) => r.utterance.as_deref() == Some(id),
                    None => r.cancellable,
                };
                if hit {
                    r.cancel.store(true, Ordering::Release);
                }
            }
            drop
        };
        for job in refused {
            job.refuse(CANCELLED);
        }
    }

    /// Requests waiting (not running).
    #[cfg(test)]
    pub fn waiting(&self) -> usize {
        self.shared.lock().jobs.len()
    }

    fn submit(&self, kind: JobKind) {
        let spawn = {
            let mut q = self.shared.lock();
            let seq = q.next_seq;
            q.next_seq += 1;
            q.jobs.push_back(Job {
                seq,
                cancel: Arc::new(AtomicBool::new(false)),
                received: Instant::now(),
                kind,
            });
            let spawn = !q.worker;
            q.worker = true;
            spawn
        };
        if spawn {
            let worker = self.clone();
            let started = std::thread::Builder::new()
                .name(format!("sigil-tts-{}", self.backend.name()))
                .spawn(move || worker.run_loop());
            if let Err(e) = started {
                tracing::error!(target: "sigil::tts", "{} TTS: cannot start the worker thread: {e}", self.backend.name());
                let jobs: Vec<Job> = {
                    let mut q = self.shared.lock();
                    q.worker = false;
                    q.jobs.drain(..).collect()
                };
                for job in jobs {
                    job.refuse("the speech worker could not start");
                }
                return;
            }
        }
        self.shared.wake.notify_one();
    }

    /// Next job: first in, first out, except that consecutive speak jobs at the front go in sink
    /// play order (warm-ups and unknown utterances keep their arrival order behind known ones).
    fn pick(&self, q: &mut Queue) -> Option<Job> {
        let front = q.jobs.front()?;
        if !front.is_speak() {
            return q.jobs.pop_front();
        }
        let run = q.jobs.iter().take_while(|j| j.is_speak()).count();
        if run == 1 || q.jobs.iter().take(run).all(|j| j.utterance().is_none()) {
            return q.jobs.pop_front();
        }
        let order = self.sink.order();
        let rank = |j: &Job| match j.utterance() {
            // A warm-up first: it is what the next sentence waits for.
            None => 0,
            Some(u) => order
                .iter()
                .position(|o| o == u)
                .map_or(usize::MAX, |p| p + 1),
        };
        let best = (0..run).min_by_key(|&i| (rank(&q.jobs[i]), q.jobs[i].seq))?;
        q.jobs.remove(best)
    }

    fn run_loop(self) {
        // Declared first so it drops last: if the thread ever unwinds, the queue is answered.
        let _exit = WorkerExit {
            shared: self.shared.clone(),
            name: self.backend.name(),
        };
        let mut loaded: Option<Loaded<B::Model>> = None;
        // A Start holds the model (see the module docs).
        let mut held = false;
        let mut idle_since = Instant::now();
        loop {
            let job = {
                let mut q = self.shared.lock();
                loop {
                    if let Some(job) = self.pick(&mut q) {
                        q.running = Some(Running {
                            seq: job.seq,
                            utterance: job.utterance().map(str::to_string),
                            cancel: job.cancel.clone(),
                            cancellable: job.is_speak(),
                        });
                        break job;
                    }
                    let waited = idle_since.elapsed();
                    if loaded.is_some() && !held && waited >= self.idle_unload {
                        drop(q);
                        if let Some(l) = loaded.take() {
                            tracing::info!(
                                target: "sigil::tts",
                                "{} TTS: unloading {} after {} s idle",
                                self.backend.name(),
                                l.key,
                                waited.as_secs()
                            );
                            discard(l, self.backend.name());
                        }
                        q = self.shared.lock();
                        continue;
                    }
                    let timeout = if loaded.is_some() && !held {
                        self.idle_unload
                            .saturating_sub(waited)
                            .max(Duration::from_millis(10))
                    } else {
                        Duration::from_secs(3600)
                    };
                    q = self
                        .shared
                        .wake
                        .wait_timeout(q, timeout)
                        .map(|(g, _)| g)
                        .unwrap_or_else(|e| e.into_inner().0);
                }
            };
            let seq = job.seq;
            self.run(job, &mut loaded, &mut held);
            idle_since = Instant::now();
            let mut q = self.shared.lock();
            if q.running.as_ref().is_some_and(|r| r.seq == seq) {
                q.running = None;
            }
        }
    }

    fn run(&self, job: Job, loaded: &mut Option<Loaded<B::Model>>, held: &mut bool) {
        let Job { cancel, received, kind, .. } = job;
        match kind {
            JobKind::Unload { reply } => {
                // Stop releases the hold; a restart's Start (waiting behind) holds it again.
                *held = false;
                // A request already waiting behind this unload is a restart (stop, then start): keep
                // the model; that request frees it itself when it needs another one.
                let restart = self.shared.lock().jobs.iter().any(Job::is_speak);
                if restart {
                    tracing::debug!(target: "sigil::tts", "{} TTS: unload skipped, a request is waiting", self.backend.name());
                } else if let Some(l) = loaded.take() {
                    tracing::info!(target: "sigil::tts", "{} TTS: unloaded {}", self.backend.name(), l.key);
                    discard(l, self.backend.name());
                }
                let _ = reply.send(Ok(()));
            }
            JobKind::Speak {
                request,
                utterance_id,
                reply,
            } => {
                let start = request.text.trim().is_empty();
                let result = self.guarded(loaded, |loaded| {
                    if start {
                        self.warm_up_job(&request, &cancel, received, loaded)
                    } else {
                        self.speak_job(&request, &utterance_id, &cancel, received, loaded)
                    }
                });
                if start && result.is_ok() {
                    *held = true;
                }
                let _ = reply.send(result);
            }
        }
    }

    /// Run model code, turning a panic into this request's error: the model may be half updated,
    /// so it is dropped (the next request loads it again).
    fn guarded<T>(
        &self,
        loaded: &mut Option<Loaded<B::Model>>,
        f: impl FnOnce(&mut Option<Loaded<B::Model>>) -> Result<T, String>,
    ) -> Result<T, String> {
        match catch_unwind(AssertUnwindSafe(|| f(loaded))) {
            Ok(result) => result,
            Err(panic) => {
                let what = panic_message(panic.as_ref());
                tracing::error!(target: "sigil::tts", "{} TTS: the engine panicked ({what}); the model is unloaded", self.backend.name());
                if let Some(l) = loaded.take() {
                    discard(l, self.backend.name());
                }
                Err(format!("the speech engine failed ({what})"))
            }
        }
    }

    /// The loaded model for `request`, loading it (and preparing its voice) when the key differs.
    /// Returns whether it was loaded now.
    fn ensure_loaded<'l>(
        &self,
        loaded: &'l mut Option<Loaded<B::Model>>,
        request: &CloningTtsRequest,
        cancel: &AtomicBool,
    ) -> Result<(&'l mut B::Model, bool), String> {
        if cancel.load(Ordering::Acquire) {
            return Err(CANCELLED.into());
        }
        let key = self.backend.key(request)?;
        let fresh = loaded.as_ref().map(|l| &l.key) != Some(&key);
        if fresh {
            if let Some(old) = loaded.take() {
                tracing::info!(target: "sigil::tts", "{} TTS: switching from {} to {key}", self.backend.name(), old.key);
                discard(old, self.backend.name());
            }
            let started = Instant::now();
            let model = self.backend.load(request, cancel)?;
            tracing::info!(
                target: "sigil::tts",
                "{} TTS: loaded {key} in {} ms",
                self.backend.name(),
                started.elapsed().as_millis()
            );
            // Kept even when this request is cancelled or its voice cannot be prepared: the model
            // itself is fine, and the next request (another voice, a restart) reuses it.
            let l = loaded.insert(Loaded { key, model });
            if cancel.load(Ordering::Acquire) {
                return Err(CANCELLED.into());
            }
            self.backend.warm_up(&mut l.model, request, cancel)?;
        }
        let l = loaded.as_mut().ok_or("the voice model is not loaded")?;
        Ok((&mut l.model, fresh))
    }

    fn warm_up_job(
        &self,
        request: &CloningTtsRequest,
        cancel: &AtomicBool,
        received: Instant,
        loaded: &mut Option<Loaded<B::Model>>,
    ) -> Result<SpeakReport, String> {
        let (model, fresh) = self.ensure_loaded(loaded, request, cancel)?;
        if !fresh {
            // Already loaded (for example a restart that kept the model): prepare this voice.
            self.backend.warm_up(model, request, cancel)?;
        }
        let total = ms(received.elapsed());
        Ok(SpeakReport {
            sample_rate: 0,
            frames: 0,
            first_audio_ms: total,
            total_ms: total,
        })
    }

    fn speak_job(
        &self,
        request: &CloningTtsRequest,
        utterance_id: &str,
        cancel: &AtomicBool,
        received: Instant,
        loaded: &mut Option<Loaded<B::Model>>,
    ) -> Result<SpeakReport, String> {
        if utterance_id.trim().is_empty() {
            return Err("utteranceId must not be empty".into());
        }
        if cancel.load(Ordering::Acquire) {
            return Err(CANCELLED.into());
        }
        let mut writer = self.sink.writer(utterance_id).map_err(sink_error)?;
        if writer.is_cancelled() {
            return Err(CANCELLED.into());
        }
        let (model, _) = self.ensure_loaded(loaded, request, cancel)?;
        let first_clause = self.sink.queued_ms() < CRITICAL_AHEAD_MS;
        let chunks = text::chunks(&request.text, first_clause);
        let mut frames: u64 = 0;
        let mut rate_seen: u32 = 0;
        let mut first_audio: Option<Duration> = None;
        let mut last_audio: Option<Duration> = None;
        let mut sink_failure: Option<SinkError> = None;
        for chunk in &chunks {
            if cancel.load(Ordering::Acquire) || writer.is_cancelled() {
                return Err(CANCELLED.into());
            }
            let mut on_audio = |samples: &[f32], rate: u32| -> bool {
                if rate == 0 {
                    return true;
                }
                let max_push = ((rate as f32 * PUSH_SECONDS) as usize).max(1);
                for piece in samples.chunks(max_push) {
                    if cancel.load(Ordering::Acquire) {
                        return false;
                    }
                    if let Err(e) = writer.push(piece, rate) {
                        sink_failure = Some(e);
                        return false;
                    }
                    frames += piece.len() as u64;
                    rate_seen = rate;
                    let now = received.elapsed();
                    first_audio.get_or_insert(now);
                    last_audio = Some(now);
                }
                true
            };
            let completed = self
                .backend
                .synthesize(model, request, chunk, cancel, &mut on_audio)?;
            match sink_failure.take() {
                Some(SinkError::Cancelled) => return Err(CANCELLED.into()),
                Some(e) => return Err(e.to_string()),
                None => {}
            }
            if !completed || cancel.load(Ordering::Acquire) {
                return Err(CANCELLED.into());
            }
        }
        let total = last_audio.unwrap_or_else(|| received.elapsed());
        Ok(SpeakReport {
            sample_rate: rate_seen,
            frames,
            first_audio_ms: ms(first_audio.unwrap_or(total)),
            total_ms: ms(total),
        })
    }
}

/// Drop a model; a panic while freeing it is logged instead of ending the worker.
fn discard<M>(model: Loaded<M>, name: &str) {
    if let Err(panic) = catch_unwind(AssertUnwindSafe(move || drop(model))) {
        tracing::error!(target: "sigil::tts", "{name} TTS: freeing the model panicked ({})", panic_message(panic.as_ref()));
    }
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Held by the worker thread: when it ends (only by unwinding), waiting requests are answered and
/// the next request starts a new worker.
struct WorkerExit {
    shared: Arc<Shared>,
    name: &'static str,
}

impl Drop for WorkerExit {
    fn drop(&mut self) {
        let waiting: Vec<Job> = {
            let mut q = self.shared.lock();
            q.worker = false;
            q.running = None;
            q.jobs.drain(..).collect()
        };
        tracing::error!(
            target: "sigil::tts",
            "{} TTS: the worker thread ended; {} waiting request(s) answered",
            self.name,
            waiting.len()
        );
        for job in waiting {
            job.refuse(WORKER_STOPPED);
        }
    }
}

fn ms(d: Duration) -> u32 {
    d.as_millis().min(u128::from(u32::MAX)) as u32
}

fn sink_error(e: SinkError) -> String {
    match e {
        SinkError::Cancelled => CANCELLED.into(),
        other => other.to_string(),
    }
}

/// The worker's answer, or the error of a dropped reply.
pub(crate) async fn answer<T>(rx: oneshot::Receiver<Result<T, String>>) -> Result<T, String> {
    rx.await.map_err(|_| WORKER_STOPPED.to_string())?
}

/// A number option of `request.options` (`null`, missing, non-numeric or non-finite → `None`).
pub(crate) fn option_f32(options: &serde_json::Value, key: &str) -> Option<f32> {
    let v = options.get(key)?;
    let n = match v {
        serde_json::Value::Number(n) => n.as_f64()?,
        serde_json::Value::String(s) => s.trim().parse::<f64>().ok()?,
        _ => return None,
    };
    let n = n as f32;
    n.is_finite().then_some(n)
}

/// A string option of `request.options` (trimmed; empty → `None`).
pub(crate) fn option_str<'a>(options: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    options
        .get(key)?
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// An integer option of `request.options` (numbers are truncated toward zero; strings parsed).
pub(crate) fn option_i64(options: &serde_json::Value, key: &str) -> Option<i64> {
    let v = options.get(key)?;
    match v {
        serde_json::Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().filter(|f| f.is_finite()).map(|f| f as i64)),
        serde_json::Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}
