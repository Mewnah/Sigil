//! The sink's engine: the ordered utterance queue, producer back-pressure and the scheduler that
//! feeds output rings and the voice changer.
//!
//! # Model
//! - `queue`: begun utterances not yet fully written, in begin order. Only the front ("head") is
//!   written; it starts once its prebuffer is filled ([`SinkConfig::prebuffer`], or it was ended, or
//!   [`SinkConfig::prebuffer_max_wait`] passed since its first audio) and, for the device route,
//!   once audio of an earlier utterance still playing on *another* device has finished (strict
//!   order across device changes).
//! - `draining`: fully written utterances still playing out. The next head starts writing right
//!   behind the previous one, so consecutive sentences join without a gap.
//! - Device route: the head's audio is converted (stateful resampler, channels, gain) into a FIFO
//!   and written into the output ring up to [`SinkConfig::ahead`] ahead of the playback position,
//!   tagged with the utterance's slot; an utterance is done when playback passed its last frame.
//!   Cancelling sets the slot's bit, so the callback drops its frames still in the ring at once.
//! - Voice-changer route: converted to the capture format and fed as tagged blocks into the
//!   voice changer's aux queue (its 60 s cap is the back-pressure; the voice changer ducks or
//!   mutes the mic meanwhile); done when none of the utterance's audio is left in that queue.
//!   `voiceChanger` falls back to the device while the voice changer is not running (the old
//!   `routeThroughVoiceChanger` behaviour); `both` then plays on the device only. If the voice
//!   changer stops in the middle of an utterance, the rest of it is dropped from that route.
//! - Back-pressure: a push is admitted while the head holds less than [`SinkConfig::head_cap`] of
//!   unplayed source audio, or (any other utterance) while all utterances together hold less than
//!   [`SinkConfig::total_cap`]; otherwise it waits (bounded by [`SinkConfig::push_timeout`],
//!   cancellation ends the wait). The head always makes progress, so producers of later
//!   utterances can never starve it. A push is admitted whole (a clip is never cut).
//! - A head that neither ends nor receives audio for [`SinkConfig::stall_timeout`] after its audio
//!   started fails with an error (a producer that died mid-sentence), so one broken producer cannot
//!   block speech forever. Before its first audio the bound is [`SinkConfig::first_audio_timeout`]:
//!   the pipeline begins an utterance before synthesis starts, and slow synthesis (a self-hosted
//!   server on a weak CPU may take minutes per sentence) plays late instead of being dropped.
//! - Events are emitted before blocked producers are woken, so a producer whose push fails with
//!   `Cancelled` because the sink dropped its utterance sees that utterance's terminal event first
//!   whenever the transport keeps order.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use super::convert::StreamConverter;
use super::output::{is_default_device, OpenOutput, OutputBackend, PlayoutShared, Tagged, SLOTS};
use super::vc_route::VcRoute;
use super::{SinkError, SinkRoute, SinkStatus, SinkTarget, UtteranceEvent, UtteranceSpec, UtteranceState};

/// Where the sink reports utterance and status changes (Tauri events in the app).
pub(crate) trait SinkEvents: Send + Sync {
    fn utterance(&self, event: &UtteranceEvent);
    fn status(&self, status: &SinkStatus);
}

/// Timing and capacity rules (see the module docs).
#[derive(Clone, Debug)]
pub(crate) struct SinkConfig {
    pub prebuffer: Duration,
    pub prebuffer_max_wait: Duration,
    /// Device audio kept written ahead of the playback position.
    pub ahead: Duration,
    /// Scheduler period while audio flows.
    pub tick: Duration,
    /// A head whose audio started fails after this long without a push or its end.
    pub stall_timeout: Duration,
    /// A head that has not received any audio yet fails after this long (its sentence may still
    /// be synthesizing: at least the longest synthesis budget of any backend).
    pub first_audio_timeout: Duration,
    pub push_timeout: Duration,
    pub head_cap: Duration,
    pub total_cap: Duration,
    /// Longest single push.
    pub max_push: Duration,
    /// An unused output stream closes after this long.
    pub idle_close: Duration,
    /// Minimum spacing of status events (at most ~4 Hz).
    pub status_interval: Duration,
    /// Size of the blocks fed to the voice changer.
    pub vc_block: Duration,
    /// Source audio converted per scheduler step.
    pub convert_slice: Duration,
    pub max_utterances: usize,
}

impl Default for SinkConfig {
    fn default() -> Self {
        Self {
            prebuffer: Duration::from_millis(200),
            prebuffer_max_wait: Duration::from_millis(1_500),
            ahead: Duration::from_millis(200),
            tick: Duration::from_millis(10),
            stall_timeout: STALL_TIMEOUT,
            first_audio_timeout: FIRST_AUDIO_TIMEOUT,
            push_timeout: Duration::from_secs(120),
            head_cap: Duration::from_secs(30),
            total_cap: Duration::from_secs(120),
            max_push: Duration::from_secs(600),
            idle_close: Duration::from_secs(300),
            status_interval: Duration::from_millis(250),
            vc_block: Duration::from_millis(100),
            convert_slice: Duration::from_millis(50),
            max_utterances: 512,
        }
    }
}

/// Default [`SinkConfig::stall_timeout`]: a playing sentence whose audio stopped arriving for this
/// long is dropped (`openai_speech` keeps its own stall bound below it).
pub(crate) const STALL_TIMEOUT: Duration = Duration::from_secs(120);
/// Default [`SinkConfig::first_audio_timeout`]: the longest per-request synthesis budget
/// (`http_client::LOCAL_TTS_TIMEOUT_MAX`, self-hosted servers on a slow CPU; VOICEVOX allows
/// 120 s + 120 s) plus a minute of margin for the request's own start-up.
pub(crate) const FIRST_AUDIO_TIMEOUT: Duration =
    Duration::from_secs(crate::services::http_client::LOCAL_TTS_TIMEOUT_MAX.as_secs() + 60);
/// Idle wake-up of the scheduler (idle-output checks).
const IDLE_WAIT: Duration = Duration::from_secs(1);
/// Wake-up with nothing queued and no output open.
const IDLE_SLEEP: Duration = Duration::from_secs(30);
/// How long a failed device open is remembered (queued utterances fail at once instead of each
/// waiting for its own open timeout).
const OPEN_ERROR_TTL: Duration = Duration::from_secs(1);
/// An idle default-device output re-checks the system default before the next utterance.
const DEFAULT_RECHECK: Duration = Duration::from_secs(2);
/// Ids of recently cancelled utterances kept, so late pushes / ends see `Cancelled`.
const RECENT_CANCELLED: usize = 256;
/// Sample rates and channel counts accepted from producers.
pub(crate) const MIN_RATE: u32 = 4_000;
pub(crate) const MAX_RATE: u32 = 384_000;
pub(crate) const MAX_CHANNELS: u16 = 16;

fn frames_us(frames: usize, rate: u32) -> u64 {
    (frames as u64).saturating_mul(1_000_000) / u64::from(rate.max(1))
}

fn frames_of(d: Duration, rate: u32) -> u64 {
    (d.as_micros() as u64).saturating_mul(u64::from(rate)) / 1_000_000
}

fn ms_of(frames: u64, rate: u32) -> u32 {
    (frames.saturating_mul(1_000) / u64::from(rate.max(1))).min(u64::from(u32::MAX)) as u32
}

/// Output key of a target device (`""` = the system default).
fn device_key(device: &str) -> String {
    if is_default_device(device) {
        String::new()
    } else {
        device.trim().to_string()
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

struct Chunk {
    samples: Vec<f32>,
    rate: u32,
    ch: usize,
}

struct DevicePlay {
    output: u64,
    slot: u8,
    /// Ring position of the utterance's first frame.
    start: u64,
    rate: u32,
    ch: usize,
    conv: Option<StreamConverter>,
    fifo: Vec<f32>,
    pos: usize,
}

impl DevicePlay {
    fn fifo_frames(&self) -> usize {
        (self.fifo.len() - self.pos) / self.ch
    }
}

struct VcPlay {
    format: (u32, usize),
    conv: Option<StreamConverter>,
    building: Vec<f32>,
    blocks: VecDeque<Vec<f32>>,
    /// Samples accepted by the voice changer.
    pushed: u64,
    /// False once the voice changer went away during this utterance.
    open: bool,
}

struct Play {
    device: Option<DevicePlay>,
    vc: Option<VcPlay>,
    /// Converters flushed (the utterance ended and all source audio was consumed).
    finished: bool,
}

struct Utt {
    id: String,
    tag: u64,
    target: SinkTarget,
    cancelled: Arc<AtomicBool>,
    chunks: VecDeque<Chunk>,
    head_offset: usize,
    pending_us: u64,
    ended: bool,
    first_push_at: Option<Instant>,
    last_activity: Instant,
    /// When it became the head (the stall timeout counts from here at the earliest).
    head_since: Option<Instant>,
    play: Option<Play>,
    announced: bool,
}

struct DrainDevice {
    output: u64,
    slot: u8,
    start: u64,
    end: u64,
    rate: u32,
}

struct DrainVc {
    format: (u32, usize),
    pushed: u64,
}

struct Draining {
    id: String,
    tag: u64,
    device: Option<DrainDevice>,
    vc: Option<DrainVc>,
    announced: bool,
}

/// A cancelled utterance's slot, freed once playback passed its last frame in the ring.
struct Zombie {
    output: u64,
    slot: u8,
    end: u64,
}

struct Output {
    id: u64,
    key: String,
    name: String,
    rate: u32,
    ch: usize,
    producer: rtrb::Producer<Tagged>,
    shared: Arc<PlayoutShared>,
    written: u64,
    slots_used: u64,
    last_active: Instant,
    default_checked: Instant,
    _handle: Box<dyn Send>,
}

impl Output {
    fn read(&self) -> u64 {
        self.shared.read_frames.load(Ordering::Acquire)
    }

    fn alloc_slot(&mut self) -> Option<u8> {
        let free = !self.slots_used;
        if free == 0 {
            return None;
        }
        let slot = free.trailing_zeros();
        debug_assert!(slot < SLOTS);
        self.slots_used |= 1 << slot;
        // A fresh tenant: never inherit a cancel bit.
        self.shared.cancelled.fetch_and(!(1u64 << slot), Ordering::AcqRel);
        Some(slot as u8)
    }

    fn free_slot(&mut self, slot: u8) {
        self.shared.cancelled.fetch_and(!(1u64 << slot), Ordering::AcqRel);
        self.slots_used &= !(1u64 << slot);
    }

    /// Write whole frames tagged with `slot`; returns the frames written.
    fn write(&mut self, slot: u8, samples: &[f32]) -> usize {
        let free = self.producer.slots() / self.ch;
        let n = (samples.len() / self.ch).min(free);
        if n == 0 {
            return 0;
        }
        match self.producer.write_chunk_uninit(n * self.ch) {
            Ok(chunk) => {
                let filled = chunk.fill_from_iter(samples[..n * self.ch].iter().map(|&v| Tagged { v, slot })) / self.ch;
                self.written += filled as u64;
                filled
            }
            Err(_) => 0,
        }
    }
}

enum Emit {
    Utterance(UtteranceEvent),
    Status(SinkStatus),
}

enum Step {
    Wait(Duration),
    Open(String),
    CheckDefault(u64),
}

enum Start {
    Started,
    Wait(Duration),
    Open(String),
    CheckDefault(u64),
    Failed(String),
}

#[derive(Default)]
struct State {
    queue: VecDeque<Utt>,
    draining: VecDeque<Draining>,
    zombies: Vec<Zombie>,
    outputs: Vec<Output>,
    recent_cancelled: VecDeque<String>,
    open_errors: Vec<(String, String, Instant)>,
    pending_total_us: u64,
    next_tag: u64,
    next_output: u64,
    /// Set by every producer-side change; the scheduler re-runs before waiting.
    dirty: bool,
    last_status: Option<SinkStatus>,
    last_status_at: Option<Instant>,
}

impl State {
    fn find(&self, id: &str) -> Option<usize> {
        self.queue.iter().position(|u| u.id == id)
    }

    fn was_cancelled(&self, id: &str) -> bool {
        self.recent_cancelled.iter().any(|c| c == id)
    }

    fn remember_cancelled(&mut self, id: String) {
        if self.recent_cancelled.len() >= RECENT_CANCELLED {
            self.recent_cancelled.pop_front();
        }
        self.recent_cancelled.push_back(id);
    }

    fn output_mut(&mut self, id: u64) -> Option<&mut Output> {
        self.outputs.iter_mut().find(|o| o.id == id)
    }

    fn output(&self, id: u64) -> Option<&Output> {
        self.outputs.iter().find(|o| o.id == id)
    }

    /// Stop playing `slot` of `output` now (its frames in the ring are dropped by the callback);
    /// the slot is freed once playback passed `end`.
    fn retire_slot(&mut self, output: u64, slot: u8, end: Option<u64>) {
        if let Some(o) = self.output_mut(output) {
            o.shared.cancelled.fetch_or(1u64 << slot, Ordering::AcqRel);
            let end = end.unwrap_or(o.written);
            self.zombies.push(Zombie { output, slot, end });
        }
    }

    fn remove_utterance_audio(&mut self, utt: &Utt) {
        self.pending_total_us = self.pending_total_us.saturating_sub(utt.pending_us);
    }
}

/// The sink.
pub(crate) struct SinkCore {
    state: Mutex<State>,
    /// Wakes the scheduler.
    wake: Condvar,
    /// Wakes producers waiting for room (and ends their wait on cancellation).
    space: Condvar,
    backend: Box<dyn OutputBackend>,
    vc: Box<dyn VcRoute>,
    events: Box<dyn SinkEvents>,
    config: SinkConfig,
    thread_started: AtomicBool,
}

impl SinkCore {
    pub(crate) fn new(config: SinkConfig, backend: Box<dyn OutputBackend>, vc: Box<dyn VcRoute>, events: Box<dyn SinkEvents>) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
            space: Condvar::new(),
            backend,
            vc,
            events,
            config,
            thread_started: AtomicBool::new(false),
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    fn touch(&self, st: &mut State) {
        st.dirty = true;
        self.wake.notify_one();
    }

    /// Start the scheduler thread (once).
    pub(crate) fn ensure_thread(self: &Arc<Self>) -> Result<(), SinkError> {
        if self.thread_started.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let core = Arc::clone(self);
        std::thread::Builder::new()
            .name("sigil-tts-sink".into())
            .spawn(move || core.run())
            .map(|_| ())
            .map_err(|e| {
                self.thread_started.store(false, Ordering::Release);
                SinkError::Unavailable(format!("could not start the TTS sink thread: {e}"))
            })
    }

    fn run(&self) {
        let mut wait = Duration::ZERO;
        loop {
            {
                let mut st = self.lock();
                if !st.dirty && !wait.is_zero() {
                    st = self.wake.wait_timeout(st, wait).unwrap_or_else(PoisonError::into_inner).0;
                }
                st.dirty = false;
            }
            wait = self.step(Instant::now());
        }
    }

    // -----------------------------------------------------------------------------------------
    // Producer API
    // -----------------------------------------------------------------------------------------

    pub(crate) fn begin(&self, spec: UtteranceSpec) -> Result<(), SinkError> {
        let id = spec.id.trim().to_string();
        if id.is_empty() || id.chars().count() > 128 {
            return Err(SinkError::InvalidFormat("the utterance id must be 1-128 characters".into()));
        }
        let mut target = spec.target;
        target.volume = if target.volume.is_finite() { target.volume.clamp(0.0, 2.0) } else { 1.0 };
        target.device = target.device.trim().to_string();
        let mut emit = Vec::new();
        {
            let mut st = self.lock();
            if st.find(&id).is_some() || st.draining.iter().any(|d| d.id == id) {
                return Err(SinkError::InvalidFormat(format!("utterance {id:?} was already begun")));
            }
            if st.queue.len() + st.draining.len() >= self.config.max_utterances {
                return Err(SinkError::Unavailable("too many utterances are queued".into()));
            }
            st.recent_cancelled.retain(|c| *c != id);
            let tag = {
                st.next_tag += 1;
                st.next_tag
            };
            let now = Instant::now();
            st.queue.push_back(Utt {
                id: id.clone(),
                tag,
                target,
                cancelled: Arc::new(AtomicBool::new(false)),
                chunks: VecDeque::new(),
                head_offset: 0,
                pending_us: 0,
                ended: false,
                first_push_at: None,
                last_activity: now,
                head_since: None,
                play: None,
                announced: false,
            });
            emit.push(Emit::Utterance(UtteranceEvent {
                id,
                state: UtteranceState::Queued,
                error: None,
                played_ms: 0,
            }));
            self.touch(&mut st);
        }
        self.emit_all(&mut emit);
        Ok(())
    }

    /// The cancel flag of a begun utterance (for [`super::UtteranceWriter`]).
    pub(crate) fn cancel_flag(&self, id: &str) -> Result<Arc<AtomicBool>, SinkError> {
        let st = self.lock();
        match st.find(id) {
            Some(i) => Ok(Arc::clone(&st.queue[i].cancelled)),
            None if st.was_cancelled(id) => Err(SinkError::Cancelled),
            None => Err(SinkError::NotFound(id.to_string())),
        }
    }

    /// Queue interleaved PCM for `id`, blocking (bounded) for back-pressure.
    pub(crate) fn push(&self, id: &str, samples: &[f32], rate: u32, channels: u16) -> Result<(), SinkError> {
        self.push_inner(id, samples, None, rate, channels)
    }

    /// [`Self::push`] for samples the caller hands over (IPC bodies): queued without another copy.
    pub(crate) fn push_owned(&self, id: &str, mut samples: Vec<f32>, rate: u32, channels: u16) -> Result<(), SinkError> {
        samples.iter_mut().filter(|s| !s.is_finite()).for_each(|s| *s = 0.0);
        let len = samples.len();
        self.push_inner(id, &[], Some((samples, len)), rate, channels)
    }

    fn push_inner(
        &self,
        id: &str,
        borrowed: &[f32],
        given: Option<(Vec<f32>, usize)>,
        rate: u32,
        channels: u16,
    ) -> Result<(), SinkError> {
        let (mut owned, len) = match given {
            Some((v, len)) => (Some(v), len),
            None => (None, borrowed.len()),
        };
        let samples = borrowed;
        if !(MIN_RATE..=MAX_RATE).contains(&rate) {
            return Err(SinkError::InvalidFormat(format!("sample rate {rate} Hz is not supported ({MIN_RATE}-{MAX_RATE} Hz)")));
        }
        if channels == 0 || channels > MAX_CHANNELS {
            return Err(SinkError::InvalidFormat(format!("{channels} channels are not supported (1-{MAX_CHANNELS})")));
        }
        let ch = usize::from(channels);
        if !len.is_multiple_of(ch) {
            return Err(SinkError::InvalidFormat(format!("{len} samples are not whole {ch}-channel frames")));
        }
        let frames = len / ch;
        let us = frames_us(frames, rate);
        if us > self.config.max_push.as_micros() as u64 {
            return Err(SinkError::InvalidFormat(format!(
                "one push may hold at most {} s of audio",
                self.config.max_push.as_secs()
            )));
        }
        let head_cap = self.config.head_cap.as_micros() as u64;
        let total_cap = self.config.total_cap.as_micros() as u64;
        let deadline = Instant::now() + self.config.push_timeout;
        let mut st = self.lock();
        loop {
            let Some(i) = st.find(id) else {
                return Err(if st.was_cancelled(id) {
                    SinkError::Cancelled
                } else {
                    SinkError::NotFound(id.to_string())
                });
            };
            let now = Instant::now();
            let total = st.pending_total_us;
            let utt = &mut st.queue[i];
            if utt.cancelled.load(Ordering::Acquire) {
                return Err(SinkError::Cancelled);
            }
            if utt.ended {
                return Err(SinkError::NotFound(format!("{id} (already ended)")));
            }
            if frames == 0 {
                return Ok(());
            }
            let admitted = if i == 0 { utt.pending_us < head_cap } else { total < total_cap };
            if admitted {
                // Copied outside any hot path: non-finite samples become silence.
                let data = owned
                    .take()
                    .unwrap_or_else(|| samples.iter().map(|&s| if s.is_finite() { s } else { 0.0 }).collect());
                utt.chunks.push_back(Chunk { samples: data, rate, ch });
                utt.pending_us += us;
                utt.first_push_at.get_or_insert(now);
                utt.last_activity = now;
                st.pending_total_us += us;
                self.touch(&mut st);
                return Ok(());
            }
            // A producer blocked here is alive: keep the stall timeout away.
            utt.last_activity = now;
            if now >= deadline {
                return Err(SinkError::Unavailable("timed out waiting for queued speech to play".into()));
            }
            if owned.is_none() {
                // Copy once while waiting (the caller's buffer stays borrowed anyway).
                drop(st);
                owned = Some(samples.iter().map(|&s| if s.is_finite() { s } else { 0.0 }).collect());
                st = self.lock();
                continue;
            }
            let slice = (deadline - now).min(Duration::from_millis(50));
            st = self.space.wait_timeout(st, slice).unwrap_or_else(PoisonError::into_inner).0;
        }
    }

    pub(crate) fn end(&self, id: &str) -> Result<(), SinkError> {
        let mut st = self.lock();
        match st.find(id) {
            Some(i) => {
                let utt = &mut st.queue[i];
                utt.ended = true;
                utt.last_activity = Instant::now();
                self.touch(&mut st);
                Ok(())
            }
            None if st.was_cancelled(id) => Err(SinkError::Cancelled),
            None if st.draining.iter().any(|d| d.id == id) => Ok(()),
            None => Err(SinkError::NotFound(id.to_string())),
        }
    }

    /// Cancel one utterance (queued, writing or playing out), or every one.
    pub(crate) fn cancel(&self, id: Option<&str>) {
        let mut emit = Vec::new();
        {
            let mut st = self.lock();
            self.cancel_locked(&mut st, id, &mut emit);
            self.touch(&mut st);
        }
        // Events first: a producer woken with `Cancelled` finds the terminal event already sent.
        self.emit_all(&mut emit);
        self.space.notify_all();
    }

    fn cancel_locked(&self, st: &mut State, id: Option<&str>, emit: &mut Vec<Emit>) {
        let matches = |uid: &str| id.is_none_or(|want| want == uid);
        let mut i = 0;
        while i < st.draining.len() {
            if !matches(&st.draining[i].id) {
                i += 1;
                continue;
            }
            let d = st.draining.remove(i).expect("index");
            let played = self.drain_played_ms(st, &d);
            if let Some(dev) = &d.device {
                st.retire_slot(dev.output, dev.slot, Some(dev.end));
            }
            if d.vc.is_some() {
                self.vc.cancel(d.tag);
            }
            emit.push(Emit::Utterance(UtteranceEvent {
                id: d.id.clone(),
                state: UtteranceState::Cancelled,
                error: None,
                played_ms: played,
            }));
            st.remember_cancelled(d.id);
        }
        let mut i = 0;
        while i < st.queue.len() {
            if !matches(&st.queue[i].id) {
                i += 1;
                continue;
            }
            let utt = st.queue.remove(i).expect("index");
            self.discard_utt(st, utt, UtteranceState::Cancelled, None, emit);
        }
    }

    /// Remove a begun utterance's audio everywhere and report `state`.
    fn discard_utt(&self, st: &mut State, utt: Utt, state: UtteranceState, error: Option<String>, emit: &mut Vec<Emit>) {
        utt.cancelled.store(true, Ordering::Release);
        st.remove_utterance_audio(&utt);
        let played = self.utt_played_ms(st, &utt);
        if let Some(play) = &utt.play {
            if let Some(d) = &play.device {
                st.retire_slot(d.output, d.slot, None);
            }
            if play.vc.as_ref().is_some_and(|v| v.pushed > 0) {
                self.vc.cancel(utt.tag);
            }
        }
        if state == UtteranceState::Error {
            tracing::warn!(
                target: "sigil::tts_sink",
                "utterance {} failed: {}",
                utt.id,
                error.as_deref().unwrap_or("unknown error")
            );
        }
        emit.push(Emit::Utterance(UtteranceEvent {
            id: utt.id.clone(),
            state,
            error,
            played_ms: played,
        }));
        st.remember_cancelled(utt.id);
    }

    /// Cancel everything and drop buffered audio (outputs stay open).
    pub(crate) fn flush(&self) {
        self.cancel(None);
    }

    pub(crate) fn status(&self) -> SinkStatus {
        let st = self.lock();
        self.status_of(&st)
    }

    // -----------------------------------------------------------------------------------------
    // Scheduler
    // -----------------------------------------------------------------------------------------

    /// One scheduling pass at `now` (opens outputs as needed, emits events). Returns how long the
    /// scheduler may sleep before the next pass.
    pub(crate) fn step(&self, now: Instant) -> Duration {
        let mut emit = Vec::new();
        let mut st = self.lock();
        let mut wait = loop {
            match self.tick(&mut st, now, &mut emit) {
                Step::Wait(w) => break w,
                Step::Open(key) => {
                    drop(st);
                    self.emit_all(&mut emit);
                    let opened = self.backend.open(&key);
                    st = self.lock();
                    match opened {
                        Ok(o) => self.install_output(&mut st, key, o, now),
                        Err(e) => {
                            tracing::warn!(target: "sigil::tts_sink", "opening output {key:?} failed: {e}");
                            st.open_errors.push((key, e, now));
                        }
                    }
                }
                Step::CheckDefault(id) => {
                    let current = st.output(id).map(|o| o.name.clone());
                    drop(st);
                    self.emit_all(&mut emit);
                    let default = self.backend.default_name();
                    st = self.lock();
                    let idle = self.output_is_idle(&st, id);
                    if let Some(o) = st.output_mut(id) {
                        o.default_checked = now;
                    }
                    if idle && default.is_some() && default != current {
                        tracing::info!(target: "sigil::tts_sink", "default output changed to {default:?}; reopening");
                        st.outputs.retain(|o| o.id != id);
                    }
                }
            }
        };
        let status = self.status_of(&st);
        if st.last_status.as_ref() != Some(&status) {
            let due = st.last_status_at.map_or(now, |t| t + self.config.status_interval);
            if now >= due {
                st.last_status = Some(status.clone());
                st.last_status_at = Some(now);
                emit.push(Emit::Status(status));
            } else {
                wait = wait.min(due - now);
            }
        }
        drop(st);
        // Events first: a producer woken with `Cancelled` (its utterance was dropped for a stall
        // or an output failure) finds the error event already sent.
        self.emit_all(&mut emit);
        self.space.notify_all();
        wait
    }

    fn install_output(&self, st: &mut State, key: String, o: OpenOutput, now: Instant) {
        st.next_output += 1;
        let id = st.next_output;
        st.outputs.push(Output {
            id,
            key,
            name: o.name,
            rate: o.rate,
            ch: o.channels.max(1),
            producer: o.producer,
            shared: o.shared,
            written: 0,
            slots_used: 0,
            last_active: now,
            default_checked: now,
            _handle: o.handle,
        });
    }

    fn output_is_idle(&self, st: &State, id: u64) -> bool {
        let used_by_queue = st
            .queue
            .iter()
            .any(|u| u.play.as_ref().and_then(|p| p.device.as_ref()).is_some_and(|d| d.output == id));
        let used = used_by_queue
            || st.draining.iter().any(|d| d.device.as_ref().is_some_and(|dd| dd.output == id))
            || st.zombies.iter().any(|z| z.output == id);
        !used && st.output(id).is_some_and(|o| o.read() >= o.written)
    }

    fn tick(&self, st: &mut State, now: Instant, emit: &mut Vec<Emit>) -> Step {
        let cfg = &self.config;
        st.open_errors.retain(|(_, _, at)| now.duration_since(*at) < OPEN_ERROR_TTL);
        self.check_failed_outputs(st, emit);
        self.release_zombies(st);
        let mut wait = IDLE_WAIT;
        self.update_draining(st, emit, &mut wait);

        while let Some(front) = st.queue.front_mut() {
            let since = *front.head_since.get_or_insert(now);
            // No audio yet: its sentence is still synthesizing (a slow engine plays late, it is not
            // dropped). After the first push, a long silence means the producer died.
            let limit = if front.first_push_at.is_none() {
                cfg.first_audio_timeout
            } else {
                cfg.stall_timeout
            };
            if !front.ended && now.saturating_duration_since(front.last_activity.max(since)) >= limit {
                let utt = st.queue.pop_front().expect("front");
                self.discard_utt(st, utt, UtteranceState::Error, Some("timed out waiting for audio".into()), emit);
                continue;
            }
            if front.play.is_none() && front.ended && front.first_push_at.is_none() {
                // Ended without any audio: nothing to route or play.
                let utt = st.queue.pop_front().expect("front");
                emit.push(Emit::Utterance(UtteranceEvent {
                    id: utt.id,
                    state: UtteranceState::Done,
                    error: None,
                    played_ms: 0,
                }));
                continue;
            }
            if front.play.is_none() {
                match self.try_start(st, now) {
                    Start::Started => {}
                    Start::Wait(w) => {
                        wait = wait.min(w);
                        break;
                    }
                    Start::Open(key) => return Step::Open(key),
                    Start::CheckDefault(id) => return Step::CheckDefault(id),
                    Start::Failed(e) => {
                        let utt = st.queue.pop_front().expect("front");
                        self.discard_utt(st, utt, UtteranceState::Error, Some(SinkError::Device(e).to_string()), emit);
                        continue;
                    }
                }
            }
            match self.pump_front(st, now) {
                Ok(true) => {
                    self.announce_front(st, emit);
                    let utt = st.queue.pop_front().expect("front");
                    self.to_draining(st, utt);
                    self.update_draining(st, emit, &mut wait);
                    continue;
                }
                Ok(false) => {
                    self.announce_front(st, emit);
                    wait = wait.min(cfg.tick);
                    break;
                }
                Err(e) => {
                    let utt = st.queue.pop_front().expect("front");
                    self.discard_utt(st, utt, UtteranceState::Error, Some(e), emit);
                    continue;
                }
            }
        }

        if !st.draining.is_empty() || !st.zombies.is_empty() {
            wait = wait.min(cfg.tick.max(Duration::from_millis(1)) * 2);
        }
        if st.queue.is_empty() && st.draining.is_empty() && st.zombies.is_empty() && st.outputs.is_empty() {
            // Nothing to play or close: sleep until a producer wakes the scheduler.
            wait = wait.max(IDLE_SLEEP);
        }
        // Close outputs nothing used for a while.
        let stale: Vec<u64> = st
            .outputs
            .iter()
            .filter(|o| now.saturating_duration_since(o.last_active) >= cfg.idle_close)
            .map(|o| o.id)
            .collect();
        let idle: Vec<u64> = stale.into_iter().filter(|&id| self.output_is_idle(st, id)).collect();
        if !idle.is_empty() {
            st.outputs.retain(|o| !idle.contains(&o.id));
        }
        Step::Wait(wait)
    }

    fn try_start(&self, st: &mut State, now: Instant) -> Start {
        let cfg = &self.config;
        let front = st.queue.front().expect("front");
        let ready = front.ended
            || front.pending_us >= cfg.prebuffer.as_micros() as u64
            || front.first_push_at.is_some_and(|t| now >= t + cfg.prebuffer_max_wait);
        if !ready {
            let w = front
                .first_push_at
                .map_or(IDLE_WAIT, |t| (t + cfg.prebuffer_max_wait).saturating_duration_since(now));
            return Start::Wait(w.max(Duration::from_millis(1)));
        }
        let route = front.target.route;
        let vc_format = match route {
            SinkRoute::VoiceChanger | SinkRoute::Both => self.vc.format(),
            SinkRoute::Device => None,
        };
        let want_device = route != SinkRoute::VoiceChanger || vc_format.is_none();
        let key = device_key(&front.target.device);
        // Strict order across route / device changes: audio of earlier utterances still playing on
        // other outputs (another device, or the voice changer vs the device) finishes first.
        let new_device = want_device.then_some(key.as_str());
        let new_vc = vc_format.is_some();
        let busy_elsewhere = st.draining.iter().any(|d| {
            let device = d.device.as_ref().and_then(|dd| st.output(dd.output)).map(|o| o.key.as_str());
            device != new_device || d.vc.is_some() != new_vc
        });
        if busy_elsewhere {
            return Start::Wait(cfg.tick);
        }
        let mut device = None;
        if want_device {
            let Some(idx) = st.outputs.iter().position(|o| o.key == key) else {
                if let Some((_, e, _)) = st.open_errors.iter().find(|(k, _, _)| *k == key) {
                    return Start::Failed(e.clone());
                }
                return Start::Open(key);
            };
            let id = st.outputs[idx].id;
            if key.is_empty()
                && now.saturating_duration_since(st.outputs[idx].default_checked) >= DEFAULT_RECHECK
                && now.saturating_duration_since(st.outputs[idx].last_active) >= DEFAULT_RECHECK
                && self.output_is_idle(st, id)
            {
                return Start::CheckDefault(id);
            }
            let o = &mut st.outputs[idx];
            let Some(slot) = o.alloc_slot() else {
                return Start::Wait(cfg.tick);
            };
            o.last_active = now;
            device = Some(DevicePlay {
                output: o.id,
                slot,
                start: o.written,
                rate: o.rate,
                ch: o.ch,
                conv: None,
                fifo: Vec::new(),
                pos: 0,
            });
        }
        let vc = vc_format.map(|format| VcPlay {
            format,
            conv: None,
            building: Vec::new(),
            blocks: VecDeque::new(),
            pushed: 0,
            open: true,
        });
        if route != SinkRoute::Device && vc.is_none() {
            tracing::debug!(target: "sigil::tts_sink", "voice changer not running: utterance plays on the device");
        }
        let front = st.queue.front_mut().expect("front");
        front.play = Some(Play {
            device,
            vc,
            finished: false,
        });
        Start::Started
    }

    /// Convert up to one slice of the head's source audio into its routes. False when there is
    /// no source audio left.
    fn consume(utt: &mut Utt, pending_total_us: &mut u64, slice: Duration, block: Duration) -> Result<bool, String> {
        let Utt {
            chunks,
            head_offset,
            pending_us,
            play,
            target,
            ..
        } = utt;
        let Some(chunk) = chunks.front() else { return Ok(false) };
        let play = play.as_mut().expect("started");
        let (rate, ch) = (chunk.rate, chunk.ch);
        let avail = (chunk.samples.len() - *head_offset) / ch;
        let n = avail.min((frames_of(slice, rate) as usize).max(1));
        let data = &chunk.samples[*head_offset..*head_offset + n * ch];
        let gain = target.volume;
        if let Some(d) = play.device.as_mut() {
            if d.pos > 0 && d.pos * 2 >= d.fifo.len() {
                d.fifo.drain(..d.pos);
                d.pos = 0;
            }
            feed(&mut d.conv, (d.rate, d.ch), gain, data, rate, ch, &mut d.fifo)?;
        }
        if let Some(v) = play.vc.as_mut().filter(|v| v.open) {
            feed(&mut v.conv, v.format, gain, data, rate, ch, &mut v.building)?;
            let block_samples = frames_of(block, v.format.0).max(1) as usize * v.format.1;
            while v.building.len() >= block_samples {
                let rest = v.building.split_off(block_samples);
                v.blocks.push_back(std::mem::replace(&mut v.building, rest));
            }
        }
        let before = avail;
        *head_offset += n * ch;
        let after = before - n;
        let consumed_us = frames_us(before, rate) - frames_us(after, rate);
        *pending_us = pending_us.saturating_sub(consumed_us);
        *pending_total_us = pending_total_us.saturating_sub(consumed_us);
        if *head_offset >= chunks.front().map_or(0, |c| c.samples.len()) {
            chunks.pop_front();
            *head_offset = 0;
        }
        Ok(true)
    }

    /// Flush every route's converter (the head ended and its source is used up).
    fn finish_play(play: &mut Play) -> Result<(), String> {
        if play.finished {
            return Ok(());
        }
        play.finished = true;
        if let Some(d) = play.device.as_mut() {
            if let Some(mut c) = d.conv.take() {
                c.finish(&mut d.fifo)?;
            }
        }
        if let Some(v) = play.vc.as_mut().filter(|v| v.open) {
            if let Some(mut c) = v.conv.take() {
                c.finish(&mut v.building)?;
            }
            if !v.building.is_empty() {
                let b = std::mem::take(&mut v.building);
                v.blocks.push_back(b);
            }
        }
        Ok(())
    }

    /// Write the head's audio into its routes. `Ok(true)` once it is fully written.
    fn pump_front(&self, st: &mut State, now: Instant) -> Result<bool, String> {
        let cfg = &self.config;
        let State {
            queue,
            outputs,
            pending_total_us,
            ..
        } = st;
        let utt = queue.front_mut().expect("front");
        let tag = utt.tag;

        // Device route: keep the ring `ahead` in front of playback.
        if let Some(out_id) = utt.play.as_ref().and_then(|p| p.device.as_ref()).map(|d| d.output) {
            let out = outputs.iter_mut().find(|o| o.id == out_id).ok_or("the output device closed")?;
            let target_frames = frames_of(cfg.ahead, out.rate).max(1);
            loop {
                let ahead = out.written.saturating_sub(out.read());
                if ahead >= target_frames {
                    break;
                }
                let fifo_frames = utt.play.as_ref().and_then(|p| p.device.as_ref()).map_or(0, DevicePlay::fifo_frames);
                if fifo_frames == 0 {
                    if Self::consume(utt, pending_total_us, cfg.convert_slice, cfg.vc_block)? {
                        continue;
                    }
                    let play = utt.play.as_mut().expect("started");
                    if utt.ended && !play.finished {
                        Self::finish_play(play)?;
                        continue;
                    }
                    break;
                }
                let d = utt.play.as_mut().and_then(|p| p.device.as_mut()).expect("device");
                let want = ((target_frames - ahead) as usize).min(fifo_frames);
                let n = out.write(d.slot, &d.fifo[d.pos..d.pos + want * d.ch]);
                d.pos += n * d.ch;
                out.last_active = now;
                if n == 0 {
                    break; // ring full
                }
            }
        }

        // Voice-changer route.
        let has_device = utt.play.as_ref().is_some_and(|p| p.device.is_some());
        let vc_open = utt.play.as_ref().and_then(|p| p.vc.as_ref()).is_some_and(|v| v.open);
        if vc_open && !has_device {
            // Nothing paces the conversion but the voice changer: stay a few blocks ahead.
            loop {
                let blocks = utt.play.as_ref().and_then(|p| p.vc.as_ref()).map_or(0, |v| v.blocks.len());
                if blocks >= 3 {
                    break;
                }
                if Self::consume(utt, pending_total_us, cfg.convert_slice, cfg.vc_block)? {
                    continue;
                }
                let play = utt.play.as_mut().expect("started");
                if utt.ended && !play.finished {
                    Self::finish_play(play)?;
                }
                break;
            }
        }
        if vc_open {
            let v = utt.play.as_mut().and_then(|p| p.vc.as_mut()).expect("vc");
            while let Some(b) = v.blocks.pop_front() {
                let len = b.len() as u64;
                match self.vc.try_push(tag, b, v.format) {
                    Ok(()) => v.pushed += len,
                    Err(Some(b)) => {
                        v.blocks.push_front(b);
                        break;
                    }
                    Err(None) => {
                        match self.vc.format() {
                            Some(f) if f != v.format => {
                                // Restarted in another format: convert the rest for it.
                                v.format = f;
                                v.conv = None;
                                v.building.clear();
                                v.blocks.clear();
                            }
                            _ => {
                                v.open = false;
                                v.blocks.clear();
                                v.building.clear();
                                v.conv = None;
                            }
                        }
                        break;
                    }
                }
            }
        }

        // No route left (the voice changer stopped mid-utterance): drop the rest of its audio.
        let any_route = utt
            .play
            .as_ref()
            .is_some_and(|p| p.device.is_some() || p.vc.as_ref().is_some_and(|v| v.open));
        if !any_route {
            while let Some(c) = utt.chunks.pop_front() {
                let frames = c.samples.len().saturating_sub(utt.head_offset) / c.ch;
                let us = frames_us(frames, c.rate);
                utt.pending_us = utt.pending_us.saturating_sub(us);
                *pending_total_us = pending_total_us.saturating_sub(us);
                utt.head_offset = 0;
            }
            if let Some(p) = utt.play.as_mut() {
                p.finished = true;
            }
        } else if utt.ended && utt.chunks.is_empty() {
            let play = utt.play.as_mut().expect("started");
            if !play.finished {
                Self::finish_play(play)?;
            }
        }

        let play = utt.play.as_ref().expect("started");
        let device_done = play.device.as_ref().is_none_or(|d| d.fifo_frames() == 0);
        let vc_done = play
            .vc
            .as_ref()
            .is_none_or(|v| !v.open || (v.blocks.is_empty() && v.building.is_empty()));
        Ok(utt.ended && utt.chunks.is_empty() && play.finished && device_done && vc_done)
    }

    fn to_draining(&self, st: &mut State, utt: Utt) {
        st.remove_utterance_audio(&utt);
        let play = utt.play.expect("started");
        let device = play.device.and_then(|d| {
            st.output(d.output).map(|o| DrainDevice {
                output: d.output,
                slot: d.slot,
                start: d.start,
                end: o.written,
                rate: o.rate,
            })
        });
        let vc = play.vc.filter(|v| v.pushed > 0).map(|v| DrainVc {
            format: v.format,
            pushed: v.pushed,
        });
        st.draining.push_back(Draining {
            id: utt.id,
            tag: utt.tag,
            device,
            vc,
            announced: utt.announced,
        });
    }

    fn announce_front(&self, st: &mut State, emit: &mut Vec<Emit>) {
        let Some(front) = st.queue.front() else { return };
        if front.announced {
            return;
        }
        let Some(play) = &front.play else { return };
        let audible = play
            .device
            .as_ref()
            .is_some_and(|d| st.output(d.output).is_some_and(|o| o.read() > d.start))
            || play
                .vc
                .as_ref()
                .is_some_and(|v| v.pushed > 0 && (self.vc.pending(front.tag) as u64) < v.pushed);
        if audible {
            let id = front.id.clone();
            let played = self.utt_played_ms(st, front);
            st.queue.front_mut().expect("front").announced = true;
            emit.push(Emit::Utterance(UtteranceEvent {
                id,
                state: UtteranceState::Playing,
                error: None,
                played_ms: played,
            }));
        }
    }

    fn update_draining(&self, st: &mut State, emit: &mut Vec<Emit>, wait: &mut Duration) {
        let mut i = 0;
        while i < st.draining.len() {
            let d = &st.draining[i];
            let read = d.device.as_ref().map(|dd| st.output(dd.output).map_or(u64::MAX, Output::read));
            let device_done = match (&d.device, read) {
                (Some(dd), Some(r)) => r >= dd.end,
                _ => true,
            };
            let vc_pending = if d.vc.is_some() { self.vc.pending(d.tag) } else { 0 };
            let device_audible = match (&d.device, read) {
                (Some(dd), Some(r)) => r > dd.start,
                _ => false,
            };
            let audible = device_audible || d.vc.as_ref().is_some_and(|v| (vc_pending as u64) < v.pushed);
            if !d.announced && (audible || (device_done && vc_pending == 0)) {
                let played = self.drain_played_ms(st, d);
                emit.push(Emit::Utterance(UtteranceEvent {
                    id: d.id.clone(),
                    state: UtteranceState::Playing,
                    error: None,
                    played_ms: played,
                }));
                st.draining[i].announced = true;
            }
            let d = &st.draining[i];
            if device_done && vc_pending == 0 {
                let played = self.drain_played_ms(st, d);
                let d = st.draining.remove(i).expect("index");
                if let Some(dd) = &d.device {
                    if let Some(o) = st.output_mut(dd.output) {
                        o.free_slot(dd.slot);
                    }
                }
                emit.push(Emit::Utterance(UtteranceEvent {
                    id: d.id,
                    state: UtteranceState::Done,
                    error: None,
                    played_ms: played,
                }));
                continue;
            }
            *wait = (*wait).min(if d.device.is_some() { self.config.tick } else { self.config.tick * 2 });
            i += 1;
        }
    }

    fn release_zombies(&self, st: &mut State) {
        let mut i = 0;
        while i < st.zombies.len() {
            let z = &st.zombies[i];
            let done = st.output(z.output).is_none_or(|o| o.read() >= z.end);
            if done {
                let z = st.zombies.swap_remove(i);
                if let Some(o) = st.output_mut(z.output) {
                    o.free_slot(z.slot);
                }
                continue;
            }
            i += 1;
        }
    }

    fn check_failed_outputs(&self, st: &mut State, emit: &mut Vec<Emit>) {
        let failed: Vec<(u64, String)> = st
            .outputs
            .iter()
            .filter(|o| o.shared.failed.load(Ordering::Acquire))
            .map(|o| (o.id, format!("{} ({})", o.shared.error_message(), o.name)))
            .collect();
        for (id, message) in failed {
            tracing::warn!(target: "sigil::tts_sink", "output failed: {message}");
            let error = SinkError::Device(message).to_string();
            // Draining utterances on it: done on the voice changer if they also went there, else failed.
            let mut i = 0;
            while i < st.draining.len() {
                if st.draining[i].device.as_ref().is_some_and(|d| d.output == id) {
                    if st.draining[i].vc.is_some() {
                        st.draining[i].device = None;
                        i += 1;
                    } else {
                        let d = st.draining.remove(i).expect("index");
                        emit.push(Emit::Utterance(UtteranceEvent {
                            id: d.id.clone(),
                            state: UtteranceState::Error,
                            error: Some(error.clone()),
                            played_ms: 0,
                        }));
                        st.remember_cancelled(d.id);
                    }
                    continue;
                }
                i += 1;
            }
            let head_on_it = st
                .queue
                .front()
                .and_then(|u| u.play.as_ref())
                .and_then(|p| p.device.as_ref())
                .is_some_and(|d| d.output == id);
            if head_on_it {
                let also_vc = st
                    .queue
                    .front()
                    .and_then(|u| u.play.as_ref())
                    .and_then(|p| p.vc.as_ref())
                    .is_some_and(|v| v.open);
                if also_vc {
                    if let Some(p) = st.queue.front_mut().and_then(|u| u.play.as_mut()) {
                        p.device = None;
                    }
                } else {
                    let mut utt = st.queue.pop_front().expect("front");
                    if let Some(p) = utt.play.as_mut() {
                        p.device = None; // the output is gone; nothing to retire
                    }
                    self.discard_utt(st, utt, UtteranceState::Error, Some(error.clone()), emit);
                }
            }
            st.zombies.retain(|z| z.output != id);
            st.outputs.retain(|o| o.id != id);
        }
    }

    fn utt_played_ms(&self, st: &State, utt: &Utt) -> u32 {
        let Some(play) = &utt.play else { return 0 };
        if let Some(d) = &play.device {
            if let Some(o) = st.output(d.output) {
                return ms_of(o.read().min(o.written).saturating_sub(d.start), d.rate);
            }
        }
        if let Some(v) = &play.vc {
            let pending = self.vc.pending(utt.tag) as u64;
            let (rate, ch) = v.format;
            return ms_of(v.pushed.saturating_sub(pending) / ch.max(1) as u64, rate);
        }
        0
    }

    fn drain_played_ms(&self, st: &State, d: &Draining) -> u32 {
        if let Some(dd) = &d.device {
            let read = st.output(dd.output).map_or(dd.end, Output::read);
            return ms_of(read.min(dd.end).saturating_sub(dd.start), dd.rate);
        }
        if let Some(v) = &d.vc {
            let pending = self.vc.pending(d.tag) as u64;
            return ms_of(v.pushed.saturating_sub(pending) / v.format.1.max(1) as u64, v.format.0);
        }
        0
    }

    fn status_of(&self, st: &State) -> SinkStatus {
        let mut ids: Vec<(&str, bool)> = st.draining.iter().map(|d| (d.id.as_str(), d.announced)).collect();
        ids.extend(st.queue.iter().map(|u| (u.id.as_str(), u.announced)));
        let playing = ids.iter().find(|(_, a)| *a).map(|(id, _)| id.to_string());
        let queued = ids
            .iter()
            .filter(|(id, _)| playing.as_deref() != Some(*id))
            .map(|(id, _)| id.to_string())
            .collect();
        let mut queued_ms = st.pending_total_us / 1_000;
        for o in &st.outputs {
            queued_ms += u64::from(ms_of(o.written.saturating_sub(o.read()), o.rate));
        }
        if let Some(d) = st.queue.front().and_then(|u| u.play.as_ref()).and_then(|p| p.device.as_ref()) {
            queued_ms += u64::from(ms_of(d.fifo_frames() as u64, d.rate));
        }
        SinkStatus {
            playing,
            queued,
            queued_ms: queued_ms.min(u64::from(u32::MAX)) as u32,
            devices: st.outputs.iter().map(|o| o.name.clone()).collect(),
        }
    }

    fn emit_all(&self, emit: &mut Vec<Emit>) {
        for e in emit.drain(..) {
            match e {
                Emit::Utterance(ev) => self.events.utterance(&ev),
                Emit::Status(st) => self.events.status(&st),
            }
        }
    }
}

/// Convert `data` (source format) through `conv` (created for the first chunk, and again when the
/// source format changes mid-utterance) into `out`.
fn feed(
    conv: &mut Option<StreamConverter>,
    out_format: (u32, usize),
    gain: f32,
    data: &[f32],
    rate: u32,
    ch: usize,
    out: &mut Vec<f32>,
) -> Result<(), String> {
    if conv.as_ref().is_some_and(|c| !c.accepts(rate, ch)) {
        if let Some(mut old) = conv.take() {
            old.finish(out)?;
        }
    }
    if conv.is_none() {
        *conv = Some(StreamConverter::new(rate, ch, out_format.0, out_format.1, gain)?);
    }
    conv.as_mut().expect("converter").push(data, out)
}
