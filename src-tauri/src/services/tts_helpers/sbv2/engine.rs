//! The `sbv2` worker: one thread (`sigil-sbv2-tts`) that owns the loaded engine (Japanese
//! frontend, BERT, one voice) and runs every start, speak and unload in order, never on the main
//! thread or an async worker.
//!
//! - **Order.** Jobs run first in, first out, except that among speak jobs waiting together the
//!   one whose sink utterance plays first goes first (the sink's begin order), as in `local_tts`.
//! - **PCM contract.** A speak pushes the sentence's audio into the begun `tts_sink` utterance (at
//!   most half a second per push, so a cancel is noticed promptly) and never ends it; the TS
//!   pipeline ends or cancels utterances.
//! - **Cancel.** `sbv2_cancel` or a sink cancel (stop / skip) stops a job between its stages and
//!   pushes; a job still waiting is answered `"cancelled"` at once.
//! - **Panics.** A panic in model code fails only its own job; the engine state is dropped (it
//!   may be half updated) and reloads with the next start or speak.
//!
//! Generic over the engine loader and the sink ([`Backend`], [`SinkPort`]) so queueing,
//! cancellation and the push contract are tested without models or an audio device.

use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use tokio::sync::oneshot;

use crate::services::tts_sink::{SinkError, SpeakReport};

/// Error text of a cancelled request (the sink's).
pub const CANCELLED: &str = "cancelled";
/// Error text of jobs left waiting when the worker thread ended.
pub const WORKER_STOPPED: &str = "the Style-Bert-VITS2 worker stopped";
/// Longest push into the sink, in seconds of audio.
const PUSH_SECONDS: f32 = 0.5;

/// What a started engine serves (from `tts_helper_start`'s config).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadSpec {
    /// Imported model id.
    pub model: String,
    /// Requested execution provider (`cpu`, `dml`, `cuda`).
    pub ep: String,
    pub threads: usize,
}

/// One sentence to speak (resolved against the loaded model by the backend).
#[derive(Clone, Debug, PartialEq)]
pub struct SpeakJob {
    /// Model id; `""` = the loaded one.
    pub model: String,
    pub text: String,
    pub speaker: String,
    pub style: String,
    pub style_weight: f32,
    pub sdp_ratio: f32,
    pub noise: f32,
    pub noise_w: f32,
    pub speed: f32,
}

/// What a started engine reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Loaded {
    pub model: String,
    pub sample_rate: u32,
    /// Execution provider the voice landed on.
    pub ep: String,
}

/// Checks a running job's cancellation.
pub trait Cancel {
    fn cancelled(&self) -> bool;
}

/// Loads and runs the engine (the real one in `super`, a stand-in in tests).
pub trait Backend: Send {
    /// Load (or keep) everything `spec` needs and warm up.
    fn load(&mut self, spec: &LoadSpec) -> Result<Loaded, String>;
    /// Synthesize one sentence (mono). `Ok(None)` when cancelled between stages.
    fn synthesize(&mut self, job: &SpeakJob, cancel: &dyn Cancel) -> Result<Option<(Vec<f32>, u32)>, String>;
    /// Drop the voice of `model` if it is loaded (before the model is deleted).
    fn drop_model(&mut self, model: &str);
    /// Drop everything (a stop: nothing loads again until the next `load`).
    fn unload(&mut self);
    /// Drop the loaded state after a panic in model code, which may have left it half updated;
    /// a started engine reloads with its next sentence. Default: [`Backend::unload`].
    fn reset(&mut self) {
        self.unload();
    }
}

/// Pushes into one sink utterance.
pub trait PcmWriter {
    fn push(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), SinkError>;
    fn is_cancelled(&self) -> bool;
}

/// The TTS sink as the worker uses it.
pub trait SinkPort: Send + Sync {
    fn writer(&self, utterance_id: &str) -> Result<Box<dyn PcmWriter>, SinkError>;
    /// Utterance ids in play order (playing first).
    fn order(&self) -> Vec<String>;
}

/// The app's sink.
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
}

type Reply<T> = oneshot::Sender<Result<T, String>>;

enum Kind {
    Load { spec: LoadSpec, reply: Reply<Loaded> },
    Speak { job: SpeakJob, utterance_id: String, reply: Reply<SpeakReport> },
    DropModel { model: String, reply: Reply<()> },
    Unload { reply: Reply<()> },
}

struct Job {
    cancel: Arc<AtomicBool>,
    kind: Kind,
}

impl Job {
    fn utterance(&self) -> Option<&str> {
        match &self.kind {
            Kind::Speak { utterance_id, .. } => Some(utterance_id),
            _ => None,
        }
    }

    fn refuse(self, error: &str) {
        let e = error.to_string();
        match self.kind {
            Kind::Load { reply, .. } => {
                let _ = reply.send(Err(e));
            }
            Kind::Speak { reply, .. } => {
                let _ = reply.send(Err(e));
            }
            Kind::DropModel { reply, .. } | Kind::Unload { reply } => {
                let _ = reply.send(Err(e));
            }
        }
    }
}

struct Running {
    utterance: Option<String>,
    cancel: Arc<AtomicBool>,
    /// Speak and load jobs can be cancelled; unload and drop cannot.
    cancellable: bool,
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    running: Option<Running>,
    worker: bool,
}

struct Shared {
    queue: Mutex<Queue>,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The worker (see the module docs). Cheap to share.
#[derive(Clone)]
pub struct Worker {
    shared: Arc<Shared>,
    backend: Arc<Mutex<Box<dyn Backend>>>,
    sink: Arc<dyn SinkPort>,
}

struct JobCancel<'a> {
    flag: &'a AtomicBool,
    writer: Option<&'a dyn PcmWriter>,
}

impl Cancel for JobCancel<'_> {
    fn cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire) || self.writer.is_some_and(|w| w.is_cancelled())
    }
}

impl Worker {
    pub fn new(backend: Box<dyn Backend>, sink: Arc<dyn SinkPort>) -> Self {
        Self {
            shared: Arc::new(Shared { queue: Mutex::new(Queue::default()) }),
            backend: Arc::new(Mutex::new(backend)),
            sink,
        }
    }

    pub fn load(&self, spec: LoadSpec) -> oneshot::Receiver<Result<Loaded, String>> {
        let (tx, rx) = oneshot::channel();
        self.submit(Kind::Load { spec, reply: tx });
        rx
    }

    pub fn speak(&self, job: SpeakJob, utterance_id: String) -> oneshot::Receiver<Result<SpeakReport, String>> {
        let (tx, rx) = oneshot::channel();
        self.submit(Kind::Speak { job, utterance_id, reply: tx });
        rx
    }

    pub fn drop_model(&self, model: String) -> oneshot::Receiver<Result<(), String>> {
        let (tx, rx) = oneshot::channel();
        self.submit(Kind::DropModel { model, reply: tx });
        rx
    }

    pub fn unload(&self) -> oneshot::Receiver<Result<(), String>> {
        let (tx, rx) = oneshot::channel();
        self.submit(Kind::Unload { reply: tx });
        rx
    }

    /// Cancel the speak job of one utterance, or (`None`) every waiting and running load and speak
    /// job. Waiting jobs are answered `"cancelled"` at once; a running one stops at its next check.
    pub fn cancel(&self, utterance_id: Option<&str>) {
        let refused: Vec<Job> = {
            let mut q = self.shared.lock();
            let hit = |job: &Job| match utterance_id {
                Some(id) => job.utterance() == Some(id),
                None => matches!(job.kind, Kind::Speak { .. } | Kind::Load { .. }),
            };
            let (drop, keep): (Vec<Job>, Vec<Job>) = q.jobs.drain(..).partition(|j| hit(j));
            q.jobs = keep.into();
            if let Some(r) = &q.running {
                let stop = match utterance_id {
                    Some(id) => r.utterance.as_deref() == Some(id),
                    None => r.cancellable,
                };
                if stop {
                    r.cancel.store(true, Ordering::Release);
                }
            }
            drop
        };
        for job in refused {
            job.refuse(CANCELLED);
        }
    }

    #[cfg(test)]
    pub fn waiting(&self) -> usize {
        self.shared.lock().jobs.len()
    }

    fn submit(&self, kind: Kind) {
        let spawn = {
            let mut q = self.shared.lock();
            q.jobs.push_back(Job { cancel: Arc::new(AtomicBool::new(false)), kind });
            let spawn = !q.worker;
            q.worker = true;
            spawn
        };
        if spawn {
            let worker = self.clone();
            if let Err(e) = std::thread::Builder::new().name("sigil-sbv2-tts".into()).spawn(move || worker.run()) {
                tracing::error!(target: "sigil::tts::sbv2", "cannot start the Style-Bert-VITS2 worker: {e}");
                let jobs: Vec<Job> = {
                    let mut q = self.shared.lock();
                    q.worker = false;
                    q.jobs.drain(..).collect()
                };
                for job in jobs {
                    job.refuse(WORKER_STOPPED);
                }
            }
        }
    }

    /// The next job: the first one, unless speak jobs wait, then the one whose utterance plays
    /// first (an utterance the sink does not know yet goes after the known ones).
    fn take_next(&self, q: &mut Queue) -> Option<Job> {
        let first_speak = q.jobs.iter().position(|j| j.utterance().is_some())?;
        // Non-speak jobs before the first speak job keep their place.
        if first_speak > 0 {
            return q.jobs.pop_front();
        }
        let speaks = q.jobs.iter().take_while(|j| j.utterance().is_some()).count();
        if speaks == 1 {
            return q.jobs.pop_front();
        }
        let order = self.sink.order();
        let rank = |j: &Job| j.utterance().and_then(|u| order.iter().position(|o| o == u)).unwrap_or(usize::MAX);
        let best = (0..speaks).min_by_key(|&i| (rank(&q.jobs[i]), i)).unwrap_or(0);
        q.jobs.remove(best)
    }

    fn run(self) {
        loop {
            let job = {
                let mut q = self.shared.lock();
                // Idle: the thread ends (the engine state lives in `backend`, not on this thread);
                // the next job starts a new one. Checked and cleared under the queue lock, so a job
                // submitted meanwhile either is seen here or spawns the next thread.
                let next = if q.jobs.iter().any(|j| j.utterance().is_some()) {
                    self.take_next(&mut q)
                } else {
                    q.jobs.pop_front()
                };
                let Some(job) = next else {
                    q.worker = false;
                    q.running = None;
                    return;
                };
                q.running = Some(Running {
                    utterance: job.utterance().map(str::to_string),
                    cancel: Arc::clone(&job.cancel),
                    cancellable: matches!(job.kind, Kind::Speak { .. } | Kind::Load { .. }),
                });
                job
            };
            self.execute(job);
            self.shared.lock().running = None;
        }
    }

    fn execute(&self, job: Job) {
        let cancel = Arc::clone(&job.cancel);
        let mut backend = self.backend.lock().unwrap_or_else(|e| e.into_inner());
        match job.kind {
            Kind::Load { spec, reply } => {
                if cancel.load(Ordering::Acquire) {
                    let _ = reply.send(Err(CANCELLED.into()));
                    return;
                }
                let out = guarded(&mut backend, |b| b.load(&spec));
                let _ = reply.send(out);
            }
            Kind::Speak { job, utterance_id, reply } => {
                let out = self.run_speak(&mut backend, &job, &utterance_id, &cancel);
                let _ = reply.send(out);
            }
            Kind::DropModel { model, reply } => {
                let _ = guarded(&mut backend, |b| {
                    b.drop_model(&model);
                    Ok(())
                });
                let _ = reply.send(Ok(()));
            }
            Kind::Unload { reply } => {
                backend.unload();
                let _ = reply.send(Ok(()));
            }
        }
    }

    fn run_speak(&self, backend: &mut Box<dyn Backend>, job: &SpeakJob, utterance_id: &str, flag: &AtomicBool) -> Result<SpeakReport, String> {
        let started = Instant::now();
        let mut writer = self.sink.writer(utterance_id).map_err(sink_error)?;
        if flag.load(Ordering::Acquire) || writer.is_cancelled() {
            return Err(CANCELLED.into());
        }
        let synthesized = {
            let cancel = JobCancel { flag, writer: Some(writer.as_ref()) };
            guarded(backend, |b| b.synthesize(job, &cancel))?
        };
        let Some((audio, rate)) = synthesized else {
            return Err(CANCELLED.into());
        };
        let first = started.elapsed();
        let step = ((rate as f32 * PUSH_SECONDS) as usize).max(1);
        let mut frames = 0u64;
        for piece in audio.chunks(step) {
            if flag.load(Ordering::Acquire) || writer.is_cancelled() {
                return Err(CANCELLED.into());
            }
            writer.push(piece, rate).map_err(sink_error)?;
            frames += piece.len() as u64;
        }
        if frames == 0 {
            return Ok(SpeakReport { sample_rate: rate, ..SpeakReport::default() });
        }
        Ok(SpeakReport {
            sample_rate: rate,
            frames,
            first_audio_ms: first.as_millis().min(u128::from(u32::MAX)) as u32,
            total_ms: started.elapsed().as_millis().min(u128::from(u32::MAX)) as u32,
        })
    }
}

/// A sink error as the command's error text (`"cancelled"` for a cancelled utterance).
fn sink_error(e: SinkError) -> String {
    match e {
        SinkError::Cancelled => CANCELLED.into(),
        other => other.to_string(),
    }
}

/// Run `f` on the backend; a panic fails only this call and drops the engine state.
fn guarded<T>(backend: &mut Box<dyn Backend>, f: impl FnOnce(&mut Box<dyn Backend>) -> Result<T, String>) -> Result<T, String> {
    match catch_unwind(AssertUnwindSafe(|| f(backend))) {
        Ok(out) => out,
        Err(panic) => {
            let why = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            tracing::error!(target: "sigil::tts::sbv2", "Style-Bert-VITS2 engine panicked: {why}");
            let _ = catch_unwind(AssertUnwindSafe(|| backend.reset()));
            Err(format!("the Style-Bert-VITS2 engine failed ({why})"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[derive(Default)]
    struct Log(Mutex<Vec<String>>);

    struct FakeBackend {
        log: Arc<Log>,
        loaded: Option<String>,
        /// Hold every synthesis until released (to queue jobs behind it).
        gate: Option<Arc<(Mutex<bool>, std::sync::Condvar)>>,
        panic_on: Option<String>,
    }

    impl Backend for FakeBackend {
        fn load(&mut self, spec: &LoadSpec) -> Result<Loaded, String> {
            self.log.0.lock().unwrap().push(format!("load {}", spec.model));
            if spec.model == "missing" {
                return Err("not imported".into());
            }
            self.loaded = Some(spec.model.clone());
            Ok(Loaded { model: spec.model.clone(), sample_rate: 1000, ep: "cpu".into() })
        }

        fn synthesize(&mut self, job: &SpeakJob, cancel: &dyn Cancel) -> Result<Option<(Vec<f32>, u32)>, String> {
            if let Some(g) = &self.gate {
                let (m, c) = &**g;
                let mut open = m.lock().unwrap();
                while !*open {
                    open = c.wait(open).unwrap();
                }
            }
            if self.panic_on.as_deref() == Some(job.text.as_str()) {
                panic!("boom");
            }
            self.log.0.lock().unwrap().push(format!("say {}", job.text));
            if cancel.cancelled() {
                return Ok(None);
            }
            Ok(Some((vec![0.25; 1200], 1000)))
        }

        fn drop_model(&mut self, model: &str) {
            if self.loaded.as_deref() == Some(model) {
                self.loaded = None;
                self.log.0.lock().unwrap().push(format!("drop {model}"));
            }
        }

        fn unload(&mut self) {
            self.loaded = None;
            self.log.0.lock().unwrap().push("unload".into());
        }
    }

    #[derive(Default)]
    struct FakeSink {
        pushes: Mutex<Vec<(String, usize, u32)>>,
        order: Mutex<Vec<String>>,
        cancelled: Mutex<Vec<String>>,
    }

    struct FakeWriter {
        sink: Arc<FakeSink>,
        id: String,
    }

    impl PcmWriter for FakeWriter {
        fn push(&mut self, samples: &[f32], rate: u32) -> Result<(), SinkError> {
            if self.is_cancelled() {
                return Err(SinkError::Cancelled);
            }
            self.sink.pushes.lock().unwrap().push((self.id.clone(), samples.len(), rate));
            Ok(())
        }
        fn is_cancelled(&self) -> bool {
            self.sink.cancelled.lock().unwrap().contains(&self.id)
        }
    }

    struct SinkHandle(Arc<FakeSink>);

    impl SinkPort for SinkHandle {
        fn writer(&self, id: &str) -> Result<Box<dyn PcmWriter>, SinkError> {
            Ok(Box::new(FakeWriter { sink: Arc::clone(&self.0), id: id.to_string() }))
        }
        fn order(&self) -> Vec<String> {
            self.0.order.lock().unwrap().clone()
        }
    }

    fn job(text: &str) -> SpeakJob {
        SpeakJob {
            model: String::new(),
            text: text.into(),
            speaker: String::new(),
            style: String::new(),
            style_weight: 1.0,
            sdp_ratio: 0.2,
            noise: 0.6,
            noise_w: 0.8,
            speed: 1.0,
        }
    }

    fn setup(gate: Option<Arc<(Mutex<bool>, std::sync::Condvar)>>) -> (Worker, Arc<Log>, Arc<FakeSink>) {
        let log = Arc::new(Log::default());
        let sink = Arc::new(FakeSink::default());
        let backend = FakeBackend { log: Arc::clone(&log), loaded: None, gate, panic_on: Some("panic".into()) };
        (Worker::new(Box::new(backend), Arc::new(SinkHandle(Arc::clone(&sink)))), log, sink)
    }

    fn block<T>(rx: oneshot::Receiver<T>) -> T {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(rx).expect("answered")
    }

    fn spec(model: &str) -> LoadSpec {
        LoadSpec { model: model.into(), ep: "cpu".into(), threads: 2 }
    }

    #[test]
    fn speak_pushes_in_half_second_pieces_and_reports() {
        let (w, log, sink) = setup(None);
        assert_eq!(block(w.load(spec("a"))).unwrap().model, "a");
        let report = block(w.speak(job("hi"), "u1".into())).unwrap();
        assert_eq!((report.frames, report.sample_rate), (1200, 1000));
        // 1200 samples at 1 kHz: 500 + 500 + 200.
        let pushes = sink.pushes.lock().unwrap().clone();
        assert_eq!(pushes.iter().map(|p| p.1).collect::<Vec<_>>(), [500, 500, 200]);
        assert!(pushes.iter().all(|p| p.0 == "u1" && p.2 == 1000));
        assert!(block(w.load(spec("missing"))).is_err());
        assert_eq!(log.0.lock().unwrap()[..3], ["load a".to_string(), "say hi".into(), "load missing".into()]);
    }

    #[test]
    fn waiting_jobs_follow_the_sink_order_and_cancel_at_once() {
        let gate = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
        let (w, log, sink) = setup(Some(Arc::clone(&gate)));
        *sink.order.lock().unwrap() = vec!["first".into(), "second".into(), "third".into()];
        let r0 = w.speak(job("blocker"), "first".into());
        // Let the blocker start, then queue the rest out of order.
        let t0 = std::time::Instant::now();
        while w.waiting() > 0 && t0.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(5));
        }
        let r3 = w.speak(job("three"), "third".into());
        let r2 = w.speak(job("two"), "second".into());
        let rx = w.speak(job("cancel me"), "x".into());
        assert_eq!(w.waiting(), 3);
        w.cancel(Some("x"));
        assert_eq!(block(rx).unwrap_err(), CANCELLED);
        {
            let (m, c) = &*gate;
            *m.lock().unwrap() = true;
            c.notify_all();
        }
        block(r0).unwrap();
        block(r2).unwrap();
        block(r3).unwrap();
        let said: Vec<String> = log.0.lock().unwrap().iter().filter(|l| l.starts_with("say")).cloned().collect();
        assert_eq!(said, ["say blocker", "say two", "say three"]);
    }

    #[test]
    fn a_cancelled_utterance_stops_and_says_cancelled() {
        let (w, _log, sink) = setup(None);
        sink.cancelled.lock().unwrap().push("gone".into());
        assert_eq!(block(w.speak(job("hi"), "gone".into())).unwrap_err(), CANCELLED);
        assert!(sink.pushes.lock().unwrap().is_empty());
    }

    #[test]
    fn a_panic_fails_one_job_and_unloads() {
        let (w, log, _sink) = setup(None);
        block(w.load(spec("a"))).unwrap();
        let e = block(w.speak(job("panic"), "u".into())).unwrap_err();
        assert!(e.contains("boom"), "{e}");
        assert!(log.0.lock().unwrap().contains(&"unload".to_string()));
        // The worker keeps serving.
        block(w.speak(job("after"), "u2".into())).unwrap();
    }

    #[test]
    fn drop_model_only_drops_that_model() {
        let (w, log, _sink) = setup(None);
        block(w.load(spec("a"))).unwrap();
        block(w.drop_model("b".into())).unwrap();
        block(w.drop_model("a".into())).unwrap();
        block(w.unload()).unwrap();
        let l = log.0.lock().unwrap().clone();
        assert_eq!(l, ["load a", "drop a", "unload"]);
    }
}
