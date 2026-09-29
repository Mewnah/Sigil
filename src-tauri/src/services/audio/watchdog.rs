//! Capture hub supervisor: owns the hub's topology on one thread and keeps streams alive.
//!
//! Every attach, detach, stream failure and device change is a [`Request`] handled in order on the
//! supervisor thread, so no lock guards the topology and device opens never race each other.
//!
//! - **Sources** (what consumers asked for, keyed by [`SourceSpec::key`]) carry the epoch reported
//!   in events. It is assigned when the first consumer attaches and never changes while the source
//!   has consumers: not on a restart, not on a move to a new default device.
//! - **Streams** (one per physical device, [`ResolvedDevice::physical_key`]) own the backend stream
//!   and its fan-out. A source maps to one stream; several sources may share it.
//! - **Failure:** the backend's error callback (or a stall: an input stream silent for
//!   [`HubConfig::stall_timeout`]) marks the stream failed, emits `audio:stream_error` once per
//!   outage for each source on it, and retries with exponential backoff
//!   ([`HubConfig::retry_min`] doubling up to [`HubConfig::retry_max`]; failures in a row without
//!   any audio in between keep backing off). A reopen installs fresh rings for every consumer;
//!   `audio:stream_recovered` (same epoch) follows once audio actually flows again (at once for
//!   loopback, which is silent while nothing plays). A source whose last consumer detaches during
//!   an outage gets `audio:stream_closed` instead, so no outage is left open.
//! - **Stall clock:** measured from when the (re)open *completed* (opens may take seconds), with
//!   [`HubConfig::first_data_timeout`] of grace until the first callback, and fed by the local
//!   time of each callback (never the driver's capture stamp).
//! - **Device changes:** failed streams retry at once (fresh backoff), and default-following
//!   sources move to the current default device.
//! - **Unsupervised** ([`HubConfig::supervise`] off, the legacy capture): no stall detection, no
//!   restart, no default-device follow. A failed stream reports `audio:stream_error` once and ends
//!   its consumers ([`super::consumer::CaptureConsumer::ended`]); only a new consumer attaching to
//!   it reopens the device.

use std::collections::{BTreeMap, HashMap};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use super::consumer::ConsumerShared;
use super::hub::{
    now_qpc_ns, ring_pair, CaptureBackend, CaptureStream, ConsumerOptions, ErrorSink, EventSlot, FanOutCmd, HubConfig,
    HubStatus, MuteGates, MuteGroup, ResolvedDevice, RingReader, Slot, SourceStatus, StreamEvent, StreamFormat,
    StreamHealth, StreamSink, StreamStatus, FANOUT_CMD_CAPACITY, FANOUT_RETIRED_CAPACITY, FANOUT_SLOT_CAPACITY,
    MAX_CONSUMERS_PER_STREAM,
};
use super::loopback::SourceSpec;

const LOG: &str = "sigil::audio::hub";
/// Supervisor wait with no stream to watch (requests still wake it at once).
const IDLE_WAIT: Duration = Duration::from_secs(3_600);

/// Requests to the supervisor thread.
pub(crate) enum Request {
    Attach {
        id: u64,
        spec: SourceSpec,
        opts: ConsumerOptions,
        shared: Arc<ConsumerShared>,
        reply: mpsc::Sender<Result<AttachReply, String>>,
    },
    Detach {
        id: u64,
        released: Option<mpsc::Sender<()>>,
    },
    StreamFailed {
        stream: String,
        generation: u64,
        message: String,
    },
    DevicesChanged,
    Status {
        reply: mpsc::Sender<HubStatus>,
    },
    Shutdown {
        done: mpsc::Sender<()>,
    },
}

/// What an attaching consumer learns.
pub(crate) struct AttachReply {
    pub source: String,
    pub epoch: u64,
}

/// Retry delay after `attempts` failed reopen attempts.
pub(crate) fn backoff(config: &HubConfig, attempts: u32) -> Duration {
    let factor = 1u32 << attempts.min(16);
    config.retry_min.saturating_mul(factor).min(config.retry_max)
}

/// Whether a running input stream is stalled. `started_ns` is when its open completed and
/// `last_callback_ns` the local time of its latest callback with audio (QPC ns). A stream silent
/// for `timeout` since its last audio is stalled; one that has delivered nothing yet gets
/// `first_data_timeout` (when longer) from `started_ns`.
pub(crate) fn is_stalled(
    loopback: bool,
    started_ns: i64,
    last_callback_ns: Option<i64>,
    now_ns: i64,
    timeout: Duration,
    first_data_timeout: Duration,
) -> bool {
    if loopback || timeout.is_zero() {
        // WASAPI loopback delivers nothing while nothing plays: silence is not a stall.
        return false;
    }
    let (since, limit) = match last_callback_ns {
        // A callback may land while the open is still completing (before `started_ns`).
        Some(t) => (t.max(started_ns), timeout),
        None => (started_ns, first_data_timeout.max(timeout)),
    };
    now_ns.saturating_sub(since) > i64::try_from(limit.as_nanos()).unwrap_or(i64::MAX)
}

struct ConsumerReg {
    source: String,
    stream: String,
    opts: ConsumerOptions,
    shared: Arc<ConsumerShared>,
}

struct Source {
    /// Display key (events, status).
    key: String,
    spec: SourceSpec,
    epoch: u64,
    stream: String,
    /// Suffix of stream keys when streams are not shared (`#<consumer id>`).
    unshared_suffix: Option<String>,
    consumers: Vec<u64>,
    /// In an outage (stream_error sent, stream_recovered pending).
    error: Option<String>,
}

struct Live {
    stream: Box<dyn CaptureStream>,
    cmds: rtrb::Producer<FanOutCmd>,
    retired: rtrb::Consumer<Slot>,
    health: Arc<StreamHealth>,
    /// When the backend's open returned (QPC ns): the stall clock starts here.
    started_ns: i64,
}

enum StreamState {
    Idle,
    Running(Box<Live>),
    Failed {
        message: String,
        attempts: u32,
        /// Next reopen attempt; `None`: none scheduled (unsupervised hub).
        retry_at: Option<Instant>,
    },
}

struct Stream {
    name: String,
    loopback: bool,
    state: StreamState,
    consumers: Vec<u64>,
    generation: u64,
    failures: u64,
    /// Failures in a row with no audio in between (drives the backoff of chronic stalls).
    strikes: u32,
    format: Option<StreamFormat>,
}

impl Stream {
    fn new(name: String, loopback: bool) -> Self {
        Self {
            name,
            loopback,
            state: StreamState::Idle,
            consumers: Vec::new(),
            generation: 0,
            failures: 0,
            strikes: 0,
            format: None,
        }
    }

    fn is_running(&self) -> bool {
        matches!(self.state, StreamState::Running(_))
    }
}

pub(crate) struct Supervisor {
    backend: Arc<dyn CaptureBackend>,
    config: HubConfig,
    gates: Arc<MuteGates>,
    events: Arc<EventSlot>,
    tx: mpsc::Sender<Request>,
    next_epoch: u64,
    sources: BTreeMap<String, Source>,
    streams: BTreeMap<String, Stream>,
    consumers: HashMap<u64, ConsumerReg>,
}

impl Supervisor {
    pub(crate) fn new(
        backend: Arc<dyn CaptureBackend>,
        config: HubConfig,
        gates: Arc<MuteGates>,
        events: Arc<EventSlot>,
        tx: mpsc::Sender<Request>,
    ) -> Self {
        Self {
            backend,
            config,
            gates,
            events,
            tx,
            next_epoch: 0,
            sources: BTreeMap::new(),
            streams: BTreeMap::new(),
            consumers: HashMap::new(),
        }
    }

    pub(crate) fn run(mut self, rx: mpsc::Receiver<Request>) {
        loop {
            let request = match rx.recv_timeout(self.next_wake()) {
                Ok(Request::Shutdown { done }) => {
                    self.shutdown();
                    let _ = done.send(());
                    return;
                }
                Ok(request) => Some(request),
                Err(mpsc::RecvTimeoutError::Timeout) => None,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    self.shutdown();
                    return;
                }
            };
            if let Some(request) = request {
                if catch_unwind(AssertUnwindSafe(|| self.handle(request))).is_err() {
                    tracing::error!(target: LOG, "capture hub request panicked; continuing");
                }
            }
            if catch_unwind(AssertUnwindSafe(|| self.tick())).is_err() {
                tracing::error!(target: LOG, "capture hub housekeeping panicked; continuing");
            }
        }
    }

    /// Until the next housekeeping tick or the earliest scheduled retry; idle (no stream) the
    /// supervisor sleeps until a request arrives.
    fn next_wake(&self) -> Duration {
        if self.streams.is_empty() {
            return IDLE_WAIT;
        }
        let now = Instant::now();
        self.streams
            .values()
            .filter_map(|s| match s.state {
                StreamState::Failed {
                    retry_at: Some(retry_at),
                    ..
                } => Some(retry_at.saturating_duration_since(now)),
                _ => None,
            })
            .fold(self.config.tick, Duration::min)
            .max(Duration::from_millis(1))
    }

    fn handle(&mut self, request: Request) {
        match request {
            Request::Attach {
                id,
                spec,
                opts,
                shared,
                reply,
            } => {
                let result = self.attach(id, spec, opts, shared);
                if reply.send(result).is_err() {
                    // The caller gave up (timeout): do not leave an orphan consumer behind.
                    self.detach(id, None);
                }
            }
            Request::Detach { id, released } => self.detach(id, released),
            Request::StreamFailed {
                stream,
                generation,
                message,
            } => {
                let current = self
                    .streams
                    .get(&stream)
                    .is_some_and(|s| s.generation == generation && s.is_running());
                if current {
                    self.enter_failed(&stream, message);
                }
            }
            Request::DevicesChanged => self.devices_changed(),
            Request::Status { reply } => {
                let _ = reply.send(self.status());
            }
            Request::Shutdown { .. } => unreachable!("handled in run"),
        }
    }

    // -----------------------------------------------------------------------------------------
    // Attach / detach
    // -----------------------------------------------------------------------------------------

    fn attach(
        &mut self,
        id: u64,
        spec: SourceSpec,
        opts: ConsumerOptions,
        shared: Arc<ConsumerShared>,
    ) -> Result<AttachReply, String> {
        let unshared_suffix = (!self.config.share_streams).then(|| format!("#{id}"));
        let source_map_key = format!("{}{}", spec.key(), unshared_suffix.as_deref().unwrap_or(""));
        let mut resolved = None;
        if !self.sources.contains_key(&source_map_key) {
            let device = self.backend.resolve(&spec)?;
            let stream_key = format!("{}{}", device.physical_key(), unshared_suffix.as_deref().unwrap_or(""));
            self.streams
                .entry(stream_key.clone())
                .or_insert_with(|| Stream::new(device.name.clone(), device.loopback));
            self.next_epoch += 1;
            self.sources.insert(
                source_map_key.clone(),
                Source {
                    key: spec.key(),
                    spec,
                    epoch: self.next_epoch,
                    stream: stream_key,
                    unshared_suffix,
                    consumers: Vec::new(),
                    error: None,
                },
            );
            resolved = Some(device);
        }
        let source = &self.sources[&source_map_key];
        let (stream_key, epoch, source_name) = (source.stream.clone(), source.epoch, source.key.clone());
        if self.streams[&stream_key].consumers.len() >= MAX_CONSUMERS_PER_STREAM {
            self.drop_source_if_unused(&source_map_key, &stream_key);
            return Err(format!(
                "Too many audio consumers on this device (limit {MAX_CONSUMERS_PER_STREAM})"
            ));
        }

        self.consumers.insert(
            id,
            ConsumerReg {
                source: source_map_key.clone(),
                stream: stream_key.clone(),
                opts,
                shared,
            },
        );
        self.sources
            .get_mut(&source_map_key)
            .expect("source inserted above")
            .consumers
            .push(id);
        let stream = self.streams.get_mut(&stream_key).expect("stream inserted above");
        stream.consumers.push(id);

        match &mut stream.state {
            StreamState::Running(_) => self.add_to_running(&stream_key, id),
            StreamState::Idle => {
                if let Err(e) = self.open_stream(&stream_key, resolved) {
                    self.detach(id, None);
                    return Err(e);
                }
            }
            StreamState::Failed { retry_at, attempts, .. } => {
                // Joining an outage: retry at once instead of waiting out the backoff (also the
                // only reopen an unsupervised hub makes: a new consumer is a fresh start).
                *retry_at = Some(Instant::now());
                *attempts = 0;
                if let Some(reg) = self.consumers.get(&id) {
                    reg.shared.set_recovering(true);
                }
            }
        }
        tracing::info!(
            target: LOG,
            "consumer {id} ({}) attached to source {source_name} (epoch {epoch}) on stream {stream_key}",
            self.consumers[&id].opts.label
        );
        Ok(AttachReply {
            source: source_name,
            epoch,
        })
    }

    fn drop_source_if_unused(&mut self, source_key: &str, stream_key: &str) {
        if self.sources.get(source_key).is_some_and(|s| s.consumers.is_empty()) {
            self.remove_source(source_key);
        }
        if self.streams.get(stream_key).is_some_and(|s| s.consumers.is_empty()) {
            self.remove_stream(stream_key, None);
        }
    }

    /// Forget a source; one in an outage gets `stream_closed` (no `stream_recovered` will come).
    fn remove_source(&mut self, source_key: &str) {
        let Some(source) = self.sources.remove(source_key) else { return };
        if source.error.is_some() {
            tracing::info!(target: LOG, "source {} (epoch {}) closed during an outage", source.key, source.epoch);
            if let Some(sink) = self.events.get() {
                sink.stream_closed(&StreamEvent {
                    source: source.key,
                    epoch: source.epoch,
                    message: None,
                });
            }
        }
    }

    fn detach(&mut self, id: u64, released: Option<mpsc::Sender<()>>) {
        let Some(reg) = self.consumers.remove(&id) else {
            if let Some(r) = released {
                let _ = r.send(());
            }
            return;
        };
        let emptied = self.sources.get_mut(&reg.source).is_some_and(|source| {
            source.consumers.retain(|c| *c != id);
            source.consumers.is_empty()
        });
        if emptied {
            self.remove_source(&reg.source);
        }
        self.unlink(&reg.stream, id, released);
        tracing::info!(target: LOG, "consumer {id} ({}) detached", reg.opts.label);
    }

    /// Remove consumer `id` from stream `key`; the stream stops with its last consumer.
    fn unlink(&mut self, key: &str, id: u64, released: Option<mpsc::Sender<()>>) {
        let Some(stream) = self.streams.get_mut(key) else {
            if let Some(r) = released {
                let _ = r.send(());
            }
            return;
        };
        stream.consumers.retain(|c| *c != id);
        if stream.consumers.is_empty() {
            self.remove_stream(key, released);
            return;
        }
        if let StreamState::Running(live) = &mut stream.state {
            // If the command ring is full the slot still goes: the consumer's ring end is dropped
            // with it, and the callback retires abandoned slots on its own.
            let _ = live.cmds.push(FanOutCmd::Detach(id));
        }
        if let Some(r) = released {
            // Other consumers keep the device open; this consumer is done with it.
            let _ = r.send(());
        }
    }

    fn remove_stream(&mut self, key: &str, released: Option<mpsc::Sender<()>>) {
        let Some(stream) = self.streams.remove(key) else {
            if let Some(r) = released {
                let _ = r.send(());
            }
            return;
        };
        match stream.state {
            StreamState::Running(live) => {
                tracing::info!(target: LOG, "stream {key} closed (no consumers)");
                live.stream.stop(released);
            }
            _ => {
                if let Some(r) = released {
                    let _ = r.send(());
                }
            }
        }
    }

    // -----------------------------------------------------------------------------------------
    // Opening, fan-out slots
    // -----------------------------------------------------------------------------------------

    fn make_slot(&self, id: u64, format: StreamFormat) -> Option<(Slot, RingReader)> {
        let reg = self.consumers.get(&id)?;
        let (writer, reader) = ring_pair(format, reg.opts.max_backlog_ms);
        let slot = Slot::new(
            id,
            writer,
            Arc::clone(&reg.shared.waker),
            &reg.opts,
            format,
            Arc::clone(&reg.shared.producer_dropped),
        );
        Some((slot, reader))
    }

    /// Open (or reopen) stream `key` with fresh rings for all of its consumers. On error the
    /// stream's state is left as it was.
    fn open_stream(&mut self, key: &str, resolved: Option<ResolvedDevice>) -> Result<(), String> {
        let (name, loopback, ids) = {
            let s = self.streams.get(key).ok_or("unknown stream")?;
            (s.name.clone(), s.loopback, s.consumers.clone())
        };
        let device = match resolved {
            Some(device) => device,
            None => self.backend.resolve(&SourceSpec::physical(&name, loopback))?,
        };
        let format = device.format;
        if format.sample_rate == 0 || format.channels == 0 {
            return Err(format!("Unsupported capture format: {} Hz, {} channels", format.sample_rate, format.channels));
        }
        let (cmd_tx, cmd_rx) = rtrb::RingBuffer::new(FANOUT_CMD_CAPACITY);
        let (retired_tx, retired_rx) = rtrb::RingBuffer::new(FANOUT_RETIRED_CAPACITY);
        let mut slots = Vec::with_capacity(FANOUT_SLOT_CAPACITY);
        let mut readers = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some((slot, reader)) = self.make_slot(id, format) {
                slots.push(slot);
                readers.push((id, reader));
            }
        }
        let health = Arc::new(StreamHealth::new());
        let sink = StreamSink::new(slots, cmd_rx, retired_tx, Arc::clone(&self.gates), Arc::clone(&health), format.channels);
        let stream = self.streams.get_mut(key).ok_or("unknown stream")?;
        stream.generation += 1;
        let errors = ErrorSink::new(key.to_string(), stream.generation, self.tx.clone());
        let handle = self.backend.open(device, sink, errors)?;
        // After the open: it may take seconds (up to the backend's timeout), none of which may
        // count against the stream's first audio.
        let started_ns = now_qpc_ns();
        stream.state = StreamState::Running(Box::new(Live {
            stream: handle,
            cmds: cmd_tx,
            retired: retired_rx,
            health,
            started_ns,
        }));
        stream.format = Some(format);
        tracing::info!(
            target: LOG,
            "stream {key} running: {} Hz, {} ch, {} consumer(s), generation {}",
            format.sample_rate,
            format.channels,
            readers.len(),
            stream.generation
        );
        for (id, reader) in readers {
            if let Some(reg) = self.consumers.get(&id) {
                reg.shared.deliver_ring(reader);
            }
        }
        Ok(())
    }

    /// Add consumer `id` to running stream `key` (a new ring through the command ring). When the
    /// command ring is full (a silent loopback stream drains none) the stream is rebuilt instead.
    fn add_to_running(&mut self, key: &str, id: u64) {
        let Some(format) = self.streams.get(key).and_then(|s| s.format) else { return };
        let Some((slot, reader)) = self.make_slot(id, format) else { return };
        let pushed = match self.streams.get_mut(key).map(|s| &mut s.state) {
            Some(StreamState::Running(live)) => live.cmds.push(FanOutCmd::Attach(slot)).is_ok(),
            _ => false,
        };
        if pushed {
            if let Some(reg) = self.consumers.get(&id) {
                reg.shared.deliver_ring(reader);
            }
            return;
        }
        tracing::info!(target: LOG, "stream {key}: command ring full, rebuilding the stream");
        self.stop_live(key);
        if let Err(e) = self.open_stream(key, None) {
            self.enter_failed(key, e);
        }
    }

    /// Stop a running stream (no events) and leave it idle.
    fn stop_live(&mut self, key: &str) {
        if let Some(stream) = self.streams.get_mut(key) {
            if let StreamState::Running(live) = std::mem::replace(&mut stream.state, StreamState::Idle) {
                live.stream.stop(None);
            }
        }
    }

    // -----------------------------------------------------------------------------------------
    // Failure, retry, recovery
    // -----------------------------------------------------------------------------------------

    /// Stream `key` failed: stop it, schedule a retry (unsupervised: end its consumers instead), and
    /// tell every source on it (once per outage).
    fn enter_failed(&mut self, key: &str, message: String) {
        let Some(stream) = self.streams.get_mut(key) else { return };
        let delivered = match std::mem::replace(&mut stream.state, StreamState::Idle) {
            StreamState::Running(live) => {
                let delivered = live.health.frames() > 0;
                live.stream.stop(None);
                delivered
            }
            _ => false,
        };
        stream.failures += 1;
        // A stream that failed again without delivering any audio keeps backing off.
        stream.strikes = if delivered { 1 } else { stream.strikes.saturating_add(1) };
        let attempts = stream.strikes - 1;
        let supervise = self.config.supervise;
        stream.state = StreamState::Failed {
            message: message.clone(),
            attempts,
            retry_at: supervise.then(|| Instant::now() + backoff(&self.config, attempts)),
        };
        if supervise {
            tracing::warn!(target: LOG, "stream {key} failed: {message}; reconnecting");
        } else {
            tracing::warn!(target: LOG, "stream {key} failed: {message}; unsupervised capture, not reconnecting");
        }
        let consumer_ids = stream.consumers.clone();
        for id in &consumer_ids {
            if let Some(reg) = self.consumers.get(id) {
                if supervise {
                    reg.shared.set_recovering(true);
                } else {
                    // Legacy capture: the device error ends the consumer's capture.
                    reg.shared.end(&message);
                }
            }
        }
        let sink = self.events.get();
        for source in self.sources.values_mut().filter(|s| s.stream == key) {
            if source.error.is_none() {
                source.error = Some(message.clone());
                let event = StreamEvent {
                    source: source.key.clone(),
                    epoch: source.epoch,
                    message: Some(message.clone()),
                };
                if let Some(sink) = &sink {
                    sink.stream_error(&event);
                }
            }
        }
    }

    fn retry(&mut self, key: &str) {
        match self.open_stream(key, None) {
            Ok(()) => {
                // `stream_recovered` follows from `tick` once audio flows.
                tracing::info!(target: LOG, "stream {key} reopened");
            }
            Err(e) => {
                let config = self.config.clone();
                if let Some(Stream {
                    state: StreamState::Failed {
                        message,
                        attempts,
                        retry_at,
                    },
                    ..
                }) = self.streams.get_mut(key)
                {
                    *attempts = attempts.saturating_add(1);
                    *retry_at = config.supervise.then(|| Instant::now() + backoff(&config, *attempts));
                    if *message != e {
                        tracing::debug!(target: LOG, "stream {key} reopen failed: {e}");
                        message.clone_from(&e);
                    }
                }
                if !config.supervise {
                    // Unsupervised: this was the one reopen a new consumer asked for; nobody waits.
                    let ids = self.streams.get(key).map(|s| s.consumers.clone()).unwrap_or_default();
                    for id in ids {
                        if let Some(reg) = self.consumers.get(&id) {
                            reg.shared.end(&e);
                        }
                    }
                }
            }
        }
    }

    /// Emit `stream_recovered` for every source on stream `key` that was in an outage (called by
    /// `tick` for running streams that deliver audio).
    fn announce_recovered(&mut self, key: &str) {
        let sink = self.events.get();
        for source in self.sources.values_mut().filter(|s| s.stream == key) {
            if source.error.take().is_some() {
                let event = StreamEvent {
                    source: source.key.clone(),
                    epoch: source.epoch,
                    message: None,
                };
                if let Some(sink) = &sink {
                    sink.stream_recovered(&event);
                }
            }
        }
    }

    fn devices_changed(&mut self) {
        if !self.config.supervise {
            // Legacy capture: streams neither reconnect nor follow the default device.
            return;
        }
        let now = Instant::now();
        for stream in self.streams.values_mut() {
            if let StreamState::Failed { attempts, retry_at, .. } = &mut stream.state {
                *attempts = 0;
                *retry_at = Some(now);
            }
        }
        let following: Vec<String> = self
            .sources
            .iter()
            .filter(|(_, s)| s.spec.follows_default())
            .map(|(k, _)| k.clone())
            .collect();
        for source_key in following {
            let Some(source) = self.sources.get(&source_key) else { continue };
            let device = match self.backend.resolve(&source.spec) {
                Ok(device) => device,
                // No default device right now: the source keeps waiting on its current stream.
                Err(_) => continue,
            };
            let target = format!("{}{}", device.physical_key(), source.unshared_suffix.as_deref().unwrap_or(""));
            if target != source.stream {
                self.move_source(&source_key, device, target);
            }
        }
    }

    /// Move every consumer of a (default-following) source to stream `target`.
    fn move_source(&mut self, source_key: &str, device: ResolvedDevice, target: String) {
        let Some(source) = self.sources.get_mut(source_key) else { return };
        let old = std::mem::replace(&mut source.stream, target.clone());
        let ids = source.consumers.clone();
        let source_name = source.key.clone();
        tracing::info!(target: LOG, "source {source_name} follows the default device: {old} -> {target}");
        for id in &ids {
            self.unlink(&old, *id, None);
        }
        let stream = self
            .streams
            .entry(target.clone())
            .or_insert_with(|| Stream::new(device.name.clone(), device.loopback));
        let fresh = matches!(stream.state, StreamState::Idle);
        for id in &ids {
            if let Some(reg) = self.consumers.get_mut(id) {
                reg.stream = target.clone();
            }
            stream.consumers.push(*id);
        }
        if fresh {
            if let Err(e) = self.open_stream(&target, Some(device)) {
                self.enter_failed(&target, e);
            }
        } else if self.streams.get(&target).is_some_and(Stream::is_running) {
            for id in &ids {
                self.add_to_running(&target, *id);
            }
        } else {
            for id in &ids {
                if let Some(reg) = self.consumers.get(id) {
                    reg.shared.set_recovering(true);
                }
            }
        }
        // An outage of the moved source ends (`tick`) once the new stream delivers audio.
    }

    // -----------------------------------------------------------------------------------------
    // Housekeeping
    // -----------------------------------------------------------------------------------------

    fn tick(&mut self) {
        let now = Instant::now();
        let now_ns = now_qpc_ns();
        let mut stalled = Vec::new();
        let mut due = Vec::new();
        let mut flowing = Vec::new();
        for (key, stream) in self.streams.iter_mut() {
            match &mut stream.state {
                StreamState::Running(live) => {
                    // Free slots the callback handed back.
                    while let Ok(slot) = live.retired.pop() {
                        drop(slot);
                    }
                    if live.health.frames() > 0 {
                        stream.strikes = 0;
                    }
                    // Loopback is silent while nothing plays: open counts as working.
                    if stream.loopback || live.health.frames() > 0 {
                        flowing.push(key.clone());
                    }
                    if self.config.supervise
                        && is_stalled(
                            stream.loopback,
                            live.started_ns,
                            live.health.last_callback_ns(),
                            now_ns,
                            self.config.stall_timeout,
                            self.config.first_data_timeout,
                        )
                    {
                        stalled.push(key.clone());
                    }
                }
                StreamState::Failed {
                    retry_at: Some(retry_at),
                    ..
                } if *retry_at <= now => due.push(key.clone()),
                _ => {}
            }
        }
        for key in stalled {
            let secs = self.config.stall_timeout.as_secs_f32();
            self.enter_failed(&key, format!("The device delivered no audio for {secs:.1} s (stalled)"));
        }
        if self.sources.values().any(|s| s.error.is_some()) {
            for key in flowing {
                // Skip a stream that just stalled above (it is failed again).
                if self.streams.get(&key).is_some_and(Stream::is_running) {
                    self.announce_recovered(&key);
                }
            }
        }
        for key in due {
            self.retry(&key);
        }
    }

    fn status(&self) -> HubStatus {
        let muted = MuteGroup::ALL.into_iter().filter(|g| self.gates.is_muted(*g)).collect();
        let sources = self
            .sources
            .values()
            .map(|s| SourceStatus {
                source: s.key.clone(),
                epoch: s.epoch,
                stream: s.stream.clone(),
                consumers: s
                    .consumers
                    .iter()
                    .filter_map(|id| self.consumers.get(id).map(|r| r.opts.label.to_string()))
                    .collect(),
                recovering: !self.streams.get(&s.stream).is_some_and(Stream::is_running),
                error: s.error.clone(),
            })
            .collect();
        let streams = self
            .streams
            .iter()
            .map(|(key, s)| {
                let (state, error, frames) = match &s.state {
                    StreamState::Idle => ("idle", None, 0),
                    StreamState::Running(live) => ("running", None, live.health.frames()),
                    StreamState::Failed { message, .. } => ("failed", Some(message.clone()), 0),
                };
                StreamStatus {
                    key: key.clone(),
                    device: s.name.clone(),
                    loopback: s.loopback,
                    state,
                    format: s.format,
                    consumers: s.consumers.len(),
                    generation: s.generation,
                    failures: s.failures,
                    frames,
                    error,
                }
            })
            .collect();
        HubStatus {
            sharing: self.config.share_streams,
            muted,
            sources,
            streams,
        }
    }

    fn shutdown(&mut self) {
        let keys: Vec<String> = self.streams.keys().cloned().collect();
        for key in keys {
            self.stop_live(&key);
        }
        for reg in self.consumers.values() {
            reg.shared.set_recovering(true);
        }
        self.streams.clear();
        self.sources.clear();
        self.consumers.clear();
        tracing::debug!(target: LOG, "capture hub stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_doubles_up_to_the_cap() {
        let c = HubConfig::default();
        assert_eq!(backoff(&c, 0), Duration::from_millis(100));
        assert_eq!(backoff(&c, 1), Duration::from_millis(200));
        assert_eq!(backoff(&c, 3), Duration::from_millis(800));
        assert_eq!(backoff(&c, 10), Duration::from_secs(3));
        assert_eq!(backoff(&c, u32::MAX), Duration::from_secs(3));
    }

    #[test]
    fn stall_rules() {
        let t = Duration::from_millis(500);
        let none = Duration::ZERO;
        let ms = 1_000_000i64;
        // Silent since start for longer than the timeout.
        assert!(is_stalled(false, 0, None, 600 * ms, t, none));
        assert!(!is_stalled(false, 0, None, 400 * ms, t, none));
        // Recent data.
        assert!(!is_stalled(false, 0, Some(300 * ms), 700 * ms, t, none));
        assert!(is_stalled(false, 0, Some(100 * ms), 700 * ms, t, none));
        // A callback during the open (before it completed) does not count against the stream.
        assert!(!is_stalled(false, 500 * ms, Some(0), 900 * ms, t, none));
        // Loopback is never stalled; a zero timeout disables the check.
        assert!(!is_stalled(true, 0, None, 60_000 * ms, t, none));
        assert!(!is_stalled(false, 0, None, 60_000 * ms, Duration::ZERO, none));
    }

    #[test]
    fn a_stream_gets_a_grace_period_for_its_first_audio() {
        let t = Duration::from_millis(500);
        let first = Duration::from_millis(2_000);
        let ms = 1_000_000i64;
        // Nothing yet: the longer first-data grace applies (from the completed open).
        assert!(!is_stalled(false, 0, None, 1_500 * ms, t, first));
        assert!(is_stalled(false, 0, None, 2_100 * ms, t, first));
        // Once audio flowed, the steady-state timeout applies.
        assert!(is_stalled(false, 0, Some(100 * ms), 700 * ms, t, first));
        // A grace shorter than the timeout never shortens it.
        assert!(!is_stalled(false, 0, None, 400 * ms, t, Duration::from_millis(100)));
    }
}
