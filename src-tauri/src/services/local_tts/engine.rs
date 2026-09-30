//! The local TTS worker: one thread (`sigil-local-tts`) that owns the loaded model and runs every
//! load, speak and synthesize request in order, never on the main thread or an async worker.
//!
//! - **One model at a time.** A request for another model, voice language (Kokoro), or thread
//!   count loads it first and frees the previous one. A loaded model is dropped after
//!   [`IDLE_UNLOAD`] without requests, or by `unload_local_tts` (unless a load already waits
//!   behind the unload: a stop followed at once by a start keeps the model).
//! - **Order.** Jobs run first in, first out, except that among `speak` jobs waiting together the
//!   one whose sink utterance plays first goes first (the sink's begin order), so a reordered IPC
//!   arrival never delays the sentence the listener is waiting for. sherpa-onnx also serializes
//!   generation process-wide (espeak-ng), so one worker loses nothing.
//! - **PCM contract.** `speak` pushes each generated chunk straight into the begun `tts_sink`
//!   utterance (at most one second per push, so a cancel is noticed promptly) and never ends it;
//!   the TS pipeline ends or cancels utterances.
//! - **Cancel.** `cancel_local_tts` or a sink cancel (stop / skip) stops a job at the next chunk
//!   boundary (sherpa cannot stop inside a sentence); a job still waiting is answered at once with
//!   `"cancelled"`.
//! - **Latency.** When no audio is queued ahead of an utterance (the first sentence of a burst), its
//!   first clause is synthesized on its own (`text::chunks`), so audio starts after one clause.
//! - **Panics.** A panic in model code (loaders, text frontends, inference) fails only its own job:
//!   the job answers with an error, the possibly half-updated model is dropped, and the worker keeps
//!   serving. Should the thread still end, the jobs waiting are answered and the next request
//!   starts a new worker.
//!
//! The engine is generic over the model loader and the sink ([`Loader`], [`SinkPort`]) so the
//! queue, cancellation and caching are tested without sherpa-onnx or an audio device.

use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tokio::sync::oneshot;

use super::commands::{LocalTtsModelInfo, LocalTtsRequest, LocalTtsVoice};
use super::text;
use crate::services::speech_rt::sherpa::TtsConfig;
use crate::services::tts_sink::{SinkError, SpeakReport};

/// A loaded model is freed after this long without requests.
pub const IDLE_UNLOAD: Duration = Duration::from_secs(10 * 60);
/// Below this much audio queued ahead of an utterance, its first clause is synthesized alone.
pub const CRITICAL_AHEAD_MS: u32 = 1500;
/// Longest push into the sink (seconds of audio), so cancels are noticed between pushes.
const PUSH_SECONDS: f32 = 1.0;
/// Error text of a cancelled request (same as the sink's).
pub const CANCELLED: &str = "cancelled";
/// Error text of jobs left waiting when the worker thread ended.
pub const WORKER_STOPPED: &str = "the local speech worker stopped";
/// Speech-rate range accepted from requests.
pub const SPEED_RANGE: (f32, f32) = (0.5, 2.0);

// -------------------------------------------------------------------------------------------------
// Loader / sink abstractions
// -------------------------------------------------------------------------------------------------

/// What must match for a loaded model to serve a request.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LoadKey {
    /// Catalog id, folder path or `hub:<repo>`.
    pub model: String,
    /// Model-level options (Kokoro: espeak voice, lexicon, Chinese rules).
    pub variant: String,
    pub threads: u32,
}

impl LoadKey {
    pub fn new(model: String, variant: String, threads: u32) -> Self {
        Self { model, variant, threads }
    }
}

/// The voice a request speaks with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VoiceChoice {
    /// Speaker id (sherpa).
    pub sid: u32,
    /// Voice name (the vendored Kitten; `""` = the model's default).
    pub name: String,
}

/// How to load a resolved model.
#[derive(Clone, Debug, PartialEq)]
pub enum OpenPlan {
    /// sherpa-onnx `OfflineTts` with this config; `validate_first` loads it in a child process
    /// first (user folders); `kitten_fallback` runs the vendored Kitten on the same package when the
    /// sherpa runtime cannot load.
    Sherpa {
        config: Box<TtsConfig>,
        validate_first: bool,
        kitten_fallback: Option<super::models::KittenFallback>,
    },
    /// A `speech_package_v1` folder (vendored Kitten).
    KittenPackage(std::path::PathBuf),
    /// A KittenML Hugging Face repo (vendored Kitten; files through the model store).
    KittenHub(String),
    /// Test loaders.
    #[cfg(test)]
    Test(String),
}

/// A request resolved against the installed models (cheap; see [`Loader::resolve`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Resolved {
    pub key: LoadKey,
    pub info: LocalTtsModelInfo,
    pub voice: VoiceChoice,
    /// Multiplies the requested speed (Piper's own length scale).
    pub speed_factor: f32,
    /// Language of the voice (warm-up text).
    pub language: Option<String>,
    pub plan: OpenPlan,
}

/// A loaded model.
pub trait Synth {
    fn sample_rate(&self) -> u32;

    /// Voices known only after loading (folders, Hub repos); `None` keeps the resolved list.
    fn voices(&self) -> Option<Vec<LocalTtsVoice>> {
        None
    }

    /// The voice used for an empty voice name, when only the loaded model knows it.
    fn default_voice(&self) -> Option<String> {
        None
    }

    /// Synthesize `text` (blocking). `on_chunk` receives audio as it is generated and returns
    /// `false` to stop; `cancel` stops at the next chunk boundary. `Ok(false)` = stopped early.
    fn generate(
        &mut self,
        text: &str,
        voice: &VoiceChoice,
        speed: f32,
        cancel: &AtomicBool,
        on_chunk: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<bool, String>;
}

/// Resolves requests and loads models.
pub trait Loader: Send + Sync {
    fn resolve(&self, model: &str, voice: &str, language: Option<&str>, threads: Option<u32>) -> Result<Resolved, String>;
    fn open(&self, resolved: &Resolved) -> Result<Box<dyn Synth>, String>;
}

/// Pushes into one sink utterance.
pub trait PcmWriter {
    fn push(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), SinkError>;
    fn is_cancelled(&self) -> bool;
}

/// The TTS sink as the engine uses it.
pub trait SinkPort: Send + Sync {
    fn writer(&self, utterance_id: &str) -> Result<Box<dyn PcmWriter>, SinkError>;
    /// Utterance ids in play order (playing first).
    fn order(&self) -> Vec<String>;
    /// Audio queued and not yet played (ms).
    fn queued_ms(&self) -> u32;
}

/// The app's sink (`tts_sink`).
pub struct AppSink;

struct AppWriter(crate::services::tts_sink::UtteranceWriter);

impl PcmWriter for AppWriter {
    fn push(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), SinkError> {
        self.0.push(samples, sample_rate, 1)
    }

    fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }
}

impl SinkPort for AppSink {
    fn writer(&self, utterance_id: &str) -> Result<Box<dyn PcmWriter>, SinkError> {
        Ok(Box::new(AppWriter(crate::services::tts_sink::writer(utterance_id)?)))
    }

    fn order(&self) -> Vec<String> {
        let s = crate::services::tts_sink::status();
        s.playing.into_iter().chain(s.queued).collect()
    }

    fn queued_ms(&self) -> u32 {
        crate::services::tts_sink::status().queued_ms
    }
}

// -------------------------------------------------------------------------------------------------
// Jobs
// -------------------------------------------------------------------------------------------------

/// Synthesized audio of `synthesize` (mono).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Synthesized {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

type Reply<T> = oneshot::Sender<Result<T, String>>;

enum JobKind {
    Load {
        model: String,
        voice: String,
        language: Option<String>,
        threads: Option<u32>,
        reply: Reply<LocalTtsModelInfo>,
    },
    Speak {
        request: LocalTtsRequest,
        utterance_id: String,
        reply: Reply<SpeakReport>,
    },
    Synthesize {
        request: LocalTtsRequest,
        reply: Reply<Synthesized>,
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
    fn utterance(&self) -> Option<&str> {
        match &self.kind {
            JobKind::Speak { utterance_id, .. } => Some(utterance_id),
            _ => None,
        }
    }

    fn is_speak(&self) -> bool {
        matches!(self.kind, JobKind::Speak { .. })
    }

    /// Answer with an error without running (cancelled or shut down).
    fn refuse(self, error: &str) {
        let e = error.to_string();
        match self.kind {
            JobKind::Load { reply, .. } => {
                let _ = reply.send(Err(e));
            }
            JobKind::Speak { reply, .. } => {
                let _ = reply.send(Err(e));
            }
            JobKind::Synthesize { reply, .. } => {
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

// -------------------------------------------------------------------------------------------------
// Engine
// -------------------------------------------------------------------------------------------------

/// The local TTS engine (see the module docs). Cheap to share (`Arc` inside).
#[derive(Clone)]
pub struct Engine {
    shared: Arc<Shared>,
    loader: Arc<dyn Loader>,
    sink: Arc<dyn SinkPort>,
    idle_unload: Duration,
}

impl Engine {
    pub fn new(loader: Arc<dyn Loader>, sink: Arc<dyn SinkPort>, idle_unload: Duration) -> Self {
        Self {
            shared: Arc::new(Shared {
                queue: Mutex::new(Queue::default()),
                wake: Condvar::new(),
            }),
            loader,
            sink,
            idle_unload,
        }
    }

    /// Load (or keep) a model and warm it up; resolves with its info.
    pub fn load(
        &self,
        model: String,
        voice: String,
        language: Option<String>,
        threads: Option<u32>,
    ) -> oneshot::Receiver<Result<LocalTtsModelInfo, String>> {
        let (tx, rx) = oneshot::channel();
        self.submit(JobKind::Load {
            model,
            voice,
            language,
            threads,
            reply: tx,
        });
        rx
    }

    /// Speak one request into the begun sink utterance `utterance_id`.
    pub fn speak(&self, request: LocalTtsRequest, utterance_id: String) -> oneshot::Receiver<Result<SpeakReport, String>> {
        let (tx, rx) = oneshot::channel();
        self.submit(JobKind::Speak {
            request,
            utterance_id,
            reply: tx,
        });
        rx
    }

    /// Synthesize one request into memory (previews and tests).
    pub fn synthesize(&self, request: LocalTtsRequest) -> oneshot::Receiver<Result<Synthesized, String>> {
        let (tx, rx) = oneshot::channel();
        self.submit(JobKind::Synthesize { request, reply: tx });
        rx
    }

    /// Free the loaded model (after the jobs queued before this call), unless a load is already
    /// waiting behind it (a restart keeps the model).
    pub fn unload(&self) -> oneshot::Receiver<Result<(), String>> {
        let (tx, rx) = oneshot::channel();
        self.submit(JobKind::Unload { reply: tx });
        rx
    }

    /// Cancel the speak job of one utterance, or (`None`) every waiting and running load, speak and
    /// synthesize job. Waiting jobs are answered `"cancelled"` at once; a running one stops at its
    /// next chunk boundary.
    pub fn cancel(&self, utterance_id: Option<&str>) {
        let refused: Vec<Job> = {
            let mut q = self.shared.lock();
            let matches = |job: &Job| match utterance_id {
                Some(id) => job.utterance() == Some(id),
                None => !matches!(job.kind, JobKind::Unload { .. }),
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

    /// Jobs waiting (not running).
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
            let engine = self.clone();
            let started = std::thread::Builder::new()
                .name("sigil-local-tts".into())
                .spawn(move || engine.worker());
            if let Err(e) = started {
                tracing::error!(target: "sigil::tts", "local TTS: cannot start the worker thread: {e}");
                let jobs: Vec<Job> = {
                    let mut q = self.shared.lock();
                    q.worker = false;
                    q.jobs.drain(..).collect()
                };
                for job in jobs {
                    job.refuse("the local speech worker could not start");
                }
                return;
            }
        }
        self.shared.wake.notify_one();
    }

    /// Next job: first in, first out, except that consecutive speak jobs at the front go in sink
    /// play order.
    fn pick(&self, q: &mut Queue) -> Option<Job> {
        let front = q.jobs.front()?;
        if !front.is_speak() {
            return q.jobs.pop_front();
        }
        let run = q.jobs.iter().take_while(|j| j.is_speak()).count();
        if run == 1 {
            return q.jobs.pop_front();
        }
        let order = self.sink.order();
        let rank = |j: &Job| {
            j.utterance()
                .and_then(|u| order.iter().position(|o| o == u))
                .unwrap_or(usize::MAX)
        };
        let best = (0..run).min_by_key(|&i| (rank(&q.jobs[i]), q.jobs[i].seq))?;
        q.jobs.remove(best)
    }

    fn worker(self) {
        // Declared first so it drops last: if the thread ever unwinds (a panic outside the guarded
        // model code), the queue is released and answered before the thread is gone.
        let _exit = WorkerExit(self.shared.clone());
        let mut loaded: Option<Loaded> = None;
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
                            cancellable: !matches!(job.kind, JobKind::Unload { .. }),
                        });
                        break job;
                    }
                    let waited = idle_since.elapsed();
                    if loaded.is_some() && waited >= self.idle_unload {
                        // Free the model outside the queue lock.
                        drop(q);
                        if let Some(l) = loaded.take() {
                            tracing::info!(target: "sigil::tts", "local TTS: unloading {} after {} s idle", l.key.model, waited.as_secs());
                            discard(l);
                        }
                        q = self.shared.lock();
                        continue;
                    }
                    let timeout = if loaded.is_some() {
                        self.idle_unload.saturating_sub(waited).max(Duration::from_millis(10))
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
            self.run(job, &mut loaded);
            idle_since = Instant::now();
            let mut q = self.shared.lock();
            if q.running.as_ref().is_some_and(|r| r.seq == seq) {
                q.running = None;
            }
        }
    }

    fn run(&self, job: Job, loaded: &mut Option<Loaded>) {
        let Job {
            cancel, received, kind, ..
        } = job;
        match kind {
            JobKind::Load {
                model,
                voice,
                language,
                threads,
                reply,
            } => {
                let result = self.guarded(loaded, |loaded| {
                    self.loader
                        .resolve(&model, &voice, language.as_deref(), threads)
                        .and_then(|r| self.ensure_loaded(loaded, r, &cancel).map(|(l, _)| l.info.clone()))
                });
                let _ = reply.send(result);
            }
            JobKind::Unload { reply } => {
                // A load already waiting behind this unload is a restart (the TS backend stops, then a
                // new one starts): keep the model instead of freeing and loading it again; the load
                // frees it itself when it needs another model.
                let restart = self
                    .shared
                    .lock()
                    .jobs
                    .iter()
                    .any(|j| matches!(j.kind, JobKind::Load { .. }));
                if restart {
                    tracing::debug!(target: "sigil::tts", "local TTS: unload skipped, a load is waiting");
                } else if let Some(l) = loaded.take() {
                    tracing::info!(target: "sigil::tts", "local TTS: unloaded {}", l.key.model);
                    discard(l);
                }
                let _ = reply.send(Ok(()));
            }
            JobKind::Speak {
                request,
                utterance_id,
                reply,
            } => {
                let result = self.guarded(loaded, |loaded| self.speak_job(&request, &utterance_id, &cancel, received, loaded));
                let _ = reply.send(result);
            }
            JobKind::Synthesize { request, reply } => {
                let result = self.guarded(loaded, |loaded| self.synthesize_job(&request, &cancel, loaded));
                let _ = reply.send(result);
            }
        }
    }

    /// Run a job's model code (resolve, load, synthesize), turning a panic into this job's error: the
    /// model it was using may be half updated, so it is dropped (the next request loads it again),
    /// and the worker keeps serving the other jobs.
    fn guarded<T>(
        &self,
        loaded: &mut Option<Loaded>,
        f: impl FnOnce(&mut Option<Loaded>) -> Result<T, String>,
    ) -> Result<T, String> {
        match catch_unwind(AssertUnwindSafe(|| f(loaded))) {
            Ok(result) => result,
            Err(panic) => {
                let what = panic_message(panic.as_ref());
                tracing::error!(target: "sigil::tts", "local TTS: the engine panicked ({what}); the model is unloaded");
                if let Some(l) = loaded.take() {
                    discard(l);
                }
                Err(format!("the local speech engine failed ({what})"))
            }
        }
    }

    /// The loaded model for `resolved`, loading it first when the key differs.
    fn ensure_loaded<'l>(
        &self,
        loaded: &'l mut Option<Loaded>,
        resolved: Resolved,
        cancel: &AtomicBool,
    ) -> Result<(&'l mut Loaded, VoiceChoice), String> {
        if cancel.load(Ordering::Acquire) {
            return Err(CANCELLED.into());
        }
        let voice = resolved.voice.clone();
        let speed_factor = resolved.speed_factor;
        if loaded.as_ref().map(|l| &l.key) != Some(&resolved.key) {
            // Free the previous model before loading the next (two in memory is pointless).
            if let Some(old) = loaded.take() {
                tracing::info!(target: "sigil::tts", "local TTS: switching from {} to {}", old.key.model, resolved.key.model);
            }
            let started = Instant::now();
            let mut synth = self.loader.open(&resolved)?;
            let mut info = resolved.info.clone();
            if let Some(v) = synth.voices() {
                info.voices = v;
            }
            if let Some(d) = synth.default_voice() {
                info.default_voice = d;
            }
            info.sample_rate = synth.sample_rate();
            warm_up(synth.as_mut(), &voice, resolved.language.as_deref());
            tracing::info!(
                target: "sigil::tts",
                "local TTS: loaded {} ({}) in {} ms",
                resolved.key.model,
                resolved.key.variant,
                started.elapsed().as_millis()
            );
            *loaded = Some(Loaded {
                key: resolved.key,
                synth,
                info,
                speed_factor,
            });
        } else if let Some(l) = loaded.as_mut() {
            l.speed_factor = speed_factor;
        }
        let l = loaded.as_mut().ok_or("the local voice model is not loaded")?;
        Ok((l, voice))
    }

    fn speak_job(
        &self,
        request: &LocalTtsRequest,
        utterance_id: &str,
        cancel: &AtomicBool,
        received: Instant,
        loaded: &mut Option<Loaded>,
    ) -> Result<SpeakReport, String> {
        if cancel.load(Ordering::Acquire) {
            return Err(CANCELLED.into());
        }
        let mut writer = self.sink.writer(utterance_id).map_err(sink_error)?;
        if writer.is_cancelled() {
            return Err(CANCELLED.into());
        }
        let resolved = self
            .loader
            .resolve(&request.model, &request.voice, request.language.as_deref(), request.threads)?;
        let (l, voice) = self.ensure_loaded(loaded, resolved, cancel)?;
        let rate = l.synth.sample_rate();
        let speed = sanitize_speed(request.speed) * l.speed_factor;
        let first_clause = self.sink.queued_ms() < CRITICAL_AHEAD_MS;
        let chunks = text::chunks(&request.text, first_clause);
        let max_push = ((rate as f32 * PUSH_SECONDS) as usize).max(1);
        let mut frames: u64 = 0;
        let mut first_audio: Option<Duration> = None;
        let mut last_audio: Option<Duration> = None;
        let mut sink_failure: Option<SinkError> = None;
        for chunk in &chunks {
            if cancel.load(Ordering::Acquire) || writer.is_cancelled() {
                return Err(CANCELLED.into());
            }
            let mut on_chunk = |samples: &[f32]| -> bool {
                for piece in samples.chunks(max_push) {
                    if cancel.load(Ordering::Acquire) {
                        return false;
                    }
                    if let Err(e) = writer.push(piece, rate) {
                        sink_failure = Some(e);
                        return false;
                    }
                    frames += piece.len() as u64;
                    let now = received.elapsed();
                    first_audio.get_or_insert(now);
                    last_audio = Some(now);
                }
                true
            };
            let completed = l.synth.generate(chunk, &voice, speed, cancel, &mut on_chunk)?;
            match sink_failure.take() {
                Some(SinkError::Cancelled) => return Err(CANCELLED.into()),
                Some(e) => return Err(e.to_string()),
                None => {}
            }
            if !completed {
                return Err(CANCELLED.into());
            }
        }
        let total = last_audio.unwrap_or_else(|| received.elapsed());
        Ok(SpeakReport {
            sample_rate: rate,
            frames,
            first_audio_ms: ms(first_audio.unwrap_or(total)),
            total_ms: ms(total),
        })
    }

    fn synthesize_job(&self, request: &LocalTtsRequest, cancel: &AtomicBool, loaded: &mut Option<Loaded>) -> Result<Synthesized, String> {
        let resolved = self
            .loader
            .resolve(&request.model, &request.voice, request.language.as_deref(), request.threads)?;
        let (l, voice) = self.ensure_loaded(loaded, resolved, cancel)?;
        let rate = l.synth.sample_rate();
        let speed = sanitize_speed(request.speed) * l.speed_factor;
        let mut samples = Vec::new();
        for chunk in text::chunks(&request.text, false) {
            if cancel.load(Ordering::Acquire) {
                return Err(CANCELLED.into());
            }
            let mut on_chunk = |s: &[f32]| {
                samples.extend_from_slice(s);
                true
            };
            if !l.synth.generate(&chunk, &voice, speed, cancel, &mut on_chunk)? {
                return Err(CANCELLED.into());
            }
        }
        Ok(Synthesized { samples, sample_rate: rate })
    }
}

struct Loaded {
    key: LoadKey,
    synth: Box<dyn Synth>,
    info: LocalTtsModelInfo,
    speed_factor: f32,
}

/// Drop a model; a panic while freeing it is logged instead of ending the worker.
fn discard(model: Loaded) {
    if let Err(panic) = catch_unwind(AssertUnwindSafe(move || drop(model))) {
        tracing::error!(target: "sigil::tts", "local TTS: freeing the model panicked ({})", panic_message(panic.as_ref()));
    }
}

/// The text of a panic payload.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Held by the worker thread: when the thread ends (it only ends by unwinding), the next request
/// starts a new worker and the jobs still waiting are answered with [`WORKER_STOPPED`] instead of
/// never.
struct WorkerExit(Arc<Shared>);

impl Drop for WorkerExit {
    fn drop(&mut self) {
        let waiting: Vec<Job> = {
            let mut q = self.0.lock();
            q.worker = false;
            q.running = None;
            q.jobs.drain(..).collect()
        };
        tracing::error!(
            target: "sigil::tts",
            "local TTS: the worker thread ended; {} waiting request(s) answered, the next one starts a new worker",
            waiting.len()
        );
        for job in waiting {
            job.refuse(WORKER_STOPPED);
        }
    }
}

/// A finite rate in [`SPEED_RANGE`]; anything else is 1.
pub fn sanitize_speed(speed: f32) -> f32 {
    if speed.is_finite() && speed > 0.0 {
        speed.clamp(SPEED_RANGE.0, SPEED_RANGE.1)
    } else {
        1.0
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

/// A short phrase in the voice's language, synthesized once after loading so the first real
/// sentence does not pay ONNX Runtime's first-run setup.
pub fn warm_up_text(language: Option<&str>) -> &'static str {
    match language.map(super::voices::base).as_deref() {
        Some("ja") => "こんにちは。",
        Some("zh") => "你好。",
        Some("es") => "Hola.",
        Some("fr") => "Bonjour.",
        Some("hi") => "नमस्ते।",
        Some("it") => "Ciao.",
        Some("pt") => "Olá.",
        Some("ru") | Some("uk") => "Привет.",
        Some("ar") => "مرحبا.",
        Some("de") => "Hallo.",
        _ => "Hello.",
    }
}

fn warm_up(synth: &mut dyn Synth, voice: &VoiceChoice, language: Option<&str>) {
    let started = Instant::now();
    let never = AtomicBool::new(false);
    match synth.generate(warm_up_text(language), voice, 1.0, &never, &mut |_| true) {
        Ok(_) => tracing::debug!(target: "sigil::tts", "local TTS: warm-up took {} ms", started.elapsed().as_millis()),
        // Not fatal: a warm-up phrase the voice cannot say must not fail the load.
        Err(e) => tracing::debug!(target: "sigil::tts", "local TTS: warm-up failed: {e}"),
    }
}
