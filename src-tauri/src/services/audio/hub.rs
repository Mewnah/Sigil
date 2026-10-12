//! Capture hub: one owned capture stream per physical source, fanned out to any number of consumers.
//!
//! A *source* is what a consumer asks for ([`SourceSpec`]: the default input, a named input, or an
//! output device opened for loopback). Sources resolve to *physical streams*, keyed by the actual
//! device, so "system default" and the same device picked by name share one cpal stream: at most
//! one stream per physical source.
//!
//! Data path (real-time rules: no allocation, locking, logging or IPC in the audio callback):
//! ```text
//! cpal callback (MMCSS "Pro Audio", HighQoS)
//!   -> StreamSink::push: per consumer, copy the block into its own rtrb ring + a block header
//!      (frames, QPC capture time, discontinuity flags), then unpark the consumer's thread
//!   -> CaptureConsumer::read (consumer thread): drop its own oldest blocks past `max_backlog_ms`,
//!      convert to 16 kHz mono (rubato, band-limited) or pass device-rate audio through.
//! ```
//! Consumers are added and removed while the stream runs through a command ring the callback
//! drains; removed slots go back through a second ring so ring memory is never freed on the audio
//! thread.
//!
//! Supervision ([`super::watchdog`], one thread): a stream error or a stall (an input delivering
//! nothing for [`HubConfig::stall_timeout`]) restarts the stream under the **same source epoch**
//! with exponential backoff and emits `audio:stream_error` once per outage, then
//! `audio:stream_recovered` (or `audio:stream_closed` when the source loses its last consumer
//! first). A device change ([`CaptureHub::devices_changed`], fed by the device
//! watcher) retries failed streams at once and moves default-following sources to the new default
//! device. Consumers see a restart as a discontinuity ([`ReadInfo::gap_after`]).
//!
//! Mute gate: [`CaptureHub::set_muted`] per [`MuteGroup`]; a gated consumer receives nothing while
//! muted (privacy: the audio never leaves the hub) and a discontinuity at both edges.
//!
//! Internal flags (soak tests and rollback; see [`HubConfig::from_env`]):
//! - `SIGIL_CAPTURE_SHARING=0`: the legacy capture. Every consumer gets its own stream and nothing
//!   supervises it (no stall detection, no restart, no default-device follow); a stream error ends
//!   its consumers' capture ([`CaptureConsumer::ended`]), as before the hub.
//! - `SIGIL_CAPTURE_SUPERVISE=0|1`: supervision on its own (default: on, off with the flag above),
//!   e.g. `SIGIL_CAPTURE_SHARING=0` + `SIGIL_CAPTURE_SUPERVISE=1` is the topology-only A/B.

use std::any::Any;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock, PoisonError, RwLock};
use std::thread::{self, JoinHandle, Thread};
use std::time::Duration;

use serde::{Deserialize, Serialize};

pub use super::consumer::CaptureConsumer;
use super::consumer::ConsumerShared;
pub use super::loopback::SourceSpec;
use super::watchdog::{AttachReply, Request, Supervisor};
use crate::services::rt_thread;

/// Tauri event: a capture stream failed; the hub is reconnecting it ([`StreamEvent`]).
pub const STREAM_ERROR_EVENT: &str = "audio:stream_error";
/// Tauri event: a failed capture stream delivers audio again, same epoch ([`StreamEvent`]).
pub const STREAM_RECOVERED_EVENT: &str = "audio:stream_recovered";
/// Tauri event: a source in an outage lost its last consumer, so no `stream_recovered` follows for
/// that epoch ([`StreamEvent`], `message` null). Ends the outage in the UI.
pub const STREAM_CLOSED_EVENT: &str = "audio:stream_closed";
/// Consumers per physical stream.
pub const MAX_CONSUMERS_PER_STREAM: usize = 16;
/// Slot storage reserved in the callback (twice the limit: detached slots may linger one callback).
pub(crate) const FANOUT_SLOT_CAPACITY: usize = MAX_CONSUMERS_PER_STREAM * 2;
/// Pending add/remove commands per stream (a silent loopback stream drains none until audio plays;
/// when full the supervisor rebuilds the stream instead).
pub(crate) const FANOUT_CMD_CAPACITY: usize = 64;
/// Slots handed back by the callback for the supervisor to free.
pub(crate) const FANOUT_RETIRED_CAPACITY: usize = 64;
/// Block headers per consumer ring (about 20 s of 10 ms device periods).
const BLOCK_RING_CAPACITY: usize = 2048;
/// How long a consumer waits for its stream to release the device on close.
pub const RELEASE_TIMEOUT: Duration = Duration::from_millis(1500);

const HUB_STOPPED: &str = "The audio capture hub is not running";

// ---------------------------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------------------------

/// QueryPerformanceCounter time in nanoseconds (the WASAPI capture timestamps' clock). Allocation-
/// and lock-free after [`CaptureHub::new`] (or any first call) cached the counter frequency.
pub fn now_qpc_ns() -> i64 {
    clock::now_ns()
}

#[cfg(windows)]
mod clock {
    use std::sync::OnceLock;

    // kernel32 is already imported by every Windows binary (std uses these two functions for
    // `Instant`), so this adds no new load-time dependency. The `windows` crate would need the
    // Win32_System_Performance feature for them.
    #[link(name = "kernel32")]
    extern "system" {
        fn QueryPerformanceCounter(count: *mut i64) -> i32;
        fn QueryPerformanceFrequency(frequency: *mut i64) -> i32;
    }

    static FREQUENCY: OnceLock<i64> = OnceLock::new();

    fn frequency() -> i64 {
        *FREQUENCY.get_or_init(|| {
            let mut f = 0i64;
            // SAFETY: valid out pointer.
            let ok = unsafe { QueryPerformanceFrequency(&mut f) } != 0;
            if ok && f > 0 {
                f
            } else {
                10_000_000
            }
        })
    }

    pub fn init() {
        let _ = frequency();
    }

    pub fn now_ns() -> i64 {
        let mut count = 0i64;
        // SAFETY: valid out pointer; QPC cannot fail on Windows XP and later.
        unsafe { QueryPerformanceCounter(&mut count) };
        (i128::from(count) * 1_000_000_000 / i128::from(frequency())) as i64
    }
}

#[cfg(not(windows))]
mod clock {
    use std::sync::OnceLock;
    use std::time::Instant;

    static START: OnceLock<Instant> = OnceLock::new();

    pub fn init() {
        let _ = START.get_or_init(Instant::now);
    }

    pub fn now_ns() -> i64 {
        START.get_or_init(Instant::now).elapsed().as_nanos() as i64
    }
}

// ---------------------------------------------------------------------------------------------
// Formats, options, mute groups
// ---------------------------------------------------------------------------------------------

/// Interleaved f32 stream format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct StreamFormat {
    pub sample_rate: u32,
    pub channels: u16,
}

impl StreamFormat {
    /// What [`ConsumerFormat::Raw16kMono`] consumers receive.
    pub const STT: StreamFormat = StreamFormat {
        sample_rate: 16_000,
        channels: 1,
    };

    /// Duration of `frames` in nanoseconds.
    pub fn frames_to_ns(&self, frames: u64) -> i64 {
        (i128::from(frames) * 1_000_000_000 / i128::from(self.sample_rate.max(1))) as i64
    }

    /// Interleaved samples in `ms` milliseconds.
    pub fn samples_for_ms(&self, ms: u32) -> usize {
        (u64::from(self.sample_rate) * u64::from(ms) / 1000) as usize * usize::from(self.channels.max(1))
    }
}

/// What a consumer receives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsumerFormat {
    /// 16 kHz mono (channels averaged, band-limited rubato resampling): STT, VAD, recorders.
    Raw16kMono,
    /// Interleaved audio exactly as the device delivers it (rate and channels may change when the
    /// stream restarts on another device; [`ReadInfo::format`] says which).
    DeviceRate,
}

/// Groups of consumers muted together by [`CaptureHub::set_muted`] (serde: snake_case).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MuteGroup {
    /// Speech-to-text from the user's microphone (STT hard mute, later VRChat MuteSelf).
    Stt,
    /// The voice changer's microphone feed.
    VoiceChanger,
    /// Other players' audio (loopback captions, P6).
    Others,
}

impl MuteGroup {
    pub const ALL: [MuteGroup; 3] = [MuteGroup::Stt, MuteGroup::VoiceChanger, MuteGroup::Others];

    fn index(self) -> usize {
        match self {
            MuteGroup::Stt => 0,
            MuteGroup::VoiceChanger => 1,
            MuteGroup::Others => 2,
        }
    }
}

/// Lock-free mute flags read by the audio callbacks.
#[derive(Default)]
pub(crate) struct MuteGates {
    flags: [AtomicBool; 3],
}

impl MuteGates {
    #[inline]
    pub(crate) fn is_muted(&self, group: MuteGroup) -> bool {
        self.flags[group.index()].load(Ordering::Relaxed)
    }

    /// Returns whether the state changed.
    pub(crate) fn set(&self, group: MuteGroup, muted: bool) -> bool {
        self.flags[group.index()].swap(muted, Ordering::Relaxed) != muted
    }
}

/// How a consumer is served.
#[derive(Clone, Debug)]
pub struct ConsumerOptions {
    pub format: ConsumerFormat,
    /// Audio the consumer may fall behind before it drops its own oldest blocks (20..=10 000 ms).
    pub max_backlog_ms: u32,
    /// Wake the consumer's thread after this much new audio; 0 wakes it on every device block.
    pub wake_ms: u32,
    /// Gate this consumer with a mute group (`None`: never muted by the hub).
    pub mute_group: Option<MuteGroup>,
    /// Short name for logs and status.
    pub label: &'static str,
}

impl ConsumerOptions {
    /// 16 kHz mono for speech engines: 2 s backlog, woken every 20 ms.
    pub fn speech(label: &'static str) -> Self {
        Self {
            format: ConsumerFormat::Raw16kMono,
            max_backlog_ms: 2_000,
            wake_ms: 20,
            mute_group: None,
            label,
        }
    }

    /// Device-rate audio for low-latency consumers (voice changer): 200 ms backlog, woken per block.
    #[allow(dead_code)] // Capture-hub API for the P2 speech engines and the P3 voice changer (used by tests).
    pub fn device_rate(label: &'static str) -> Self {
        Self {
            format: ConsumerFormat::DeviceRate,
            max_backlog_ms: 200,
            wake_ms: 0,
            mute_group: None,
            label,
        }
    }

    pub fn with_mute_group(mut self, group: Option<MuteGroup>) -> Self {
        self.mute_group = group;
        self
    }

    #[allow(dead_code)] // Capture-hub API for the P2 speech engines and the P3 voice changer (used by tests).
    pub fn with_max_backlog_ms(mut self, ms: u32) -> Self {
        self.max_backlog_ms = ms;
        self
    }

    pub(crate) fn sanitized(mut self) -> Self {
        self.max_backlog_ms = self.max_backlog_ms.clamp(20, 10_000);
        self.wake_ms = self.wake_ms.min(500);
        self
    }
}

// ---------------------------------------------------------------------------------------------
// Rings and the callback-side fan-out
// ---------------------------------------------------------------------------------------------

/// Header of one device block in a consumer ring (written after the block's samples).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockMeta {
    pub frames: u32,
    pub flags: u8,
    /// QPC capture time of the block's first frame (ns).
    pub qpc_ns: i64,
}

/// The producer dropped audio right before this block (the consumer's ring was full).
pub(crate) const FLAG_OVERRUN: u8 = 1;
/// The consumer was gated (muted) right before this block.
pub(crate) const FLAG_UNMUTED: u8 = 2;
/// Zero-length marker: the consumer's mute gate closed here.
pub(crate) const FLAG_MUTE_MARK: u8 = 4;
/// Any of the discontinuity flags.
pub(crate) const FLAG_BOUNDARY: u8 = FLAG_OVERRUN | FLAG_UNMUTED | FLAG_MUTE_MARK;

pub(crate) struct RingWriter {
    samples: rtrb::Producer<f32>,
    blocks: rtrb::Producer<BlockMeta>,
}

/// The consumer end of one stream generation's ring.
pub(crate) struct RingReader {
    pub(crate) samples: rtrb::Consumer<f32>,
    pub(crate) blocks: rtrb::Consumer<BlockMeta>,
    pub(crate) format: StreamFormat,
}

/// A consumer ring sized for `max_backlog_ms` plus headroom (so a consumer that reads at all never
/// makes the producer drop the newest audio; it drops its own oldest instead).
pub(crate) fn ring_pair(format: StreamFormat, max_backlog_ms: u32) -> (RingWriter, RingReader) {
    let headroom = (max_backlog_ms / 2).max(250);
    let samples = format.samples_for_ms(max_backlog_ms + headroom).max(8192);
    let (sp, sc) = rtrb::RingBuffer::new(samples);
    let (bp, bc) = rtrb::RingBuffer::new(BLOCK_RING_CAPACITY);
    (
        RingWriter { samples: sp, blocks: bp },
        RingReader {
            samples: sc,
            blocks: bc,
            format,
        },
    )
}

/// The consumer's thread, registered on its first wait (lock-free read in the callback).
#[derive(Default)]
pub(crate) struct WakerCell(OnceLock<Thread>);

impl WakerCell {
    #[inline]
    pub(crate) fn wake(&self) {
        if let Some(thread) = self.0.get() {
            thread.unpark();
        }
    }

    /// Register the calling thread (first caller wins). Returns whether it is the registered one.
    pub(crate) fn register_current(&self) -> bool {
        let me = thread::current();
        self.0.get_or_init(|| me.clone()).id() == me.id()
    }
}

/// One consumer inside a running stream's callback.
pub(crate) struct Slot {
    pub(crate) id: u64,
    ring: RingWriter,
    waker: Arc<WakerCell>,
    gate: Option<MuteGroup>,
    wake_frames: u32,
    since_wake: u32,
    pending_flags: u8,
    gated: bool,
    dropped: Arc<AtomicU64>,
}

impl Slot {
    pub(crate) fn new(
        id: u64,
        ring: RingWriter,
        waker: Arc<WakerCell>,
        opts: &ConsumerOptions,
        format: StreamFormat,
        dropped: Arc<AtomicU64>,
    ) -> Self {
        let wake_frames = (u64::from(format.sample_rate) * u64::from(opts.wake_ms) / 1000) as u32;
        Self {
            id,
            ring,
            waker,
            gate: opts.mute_group,
            wake_frames,
            since_wake: 0,
            pending_flags: 0,
            gated: false,
            dropped,
        }
    }

    #[inline]
    fn is_abandoned(&self) -> bool {
        self.ring.blocks.is_abandoned()
    }

    #[inline]
    fn push_block(&mut self, data: &[f32], frames: u32, qpc_ns: i64) {
        if self.ring.samples.slots() < data.len() || self.ring.blocks.slots() == 0 {
            // The consumer stopped reading for longer than its ring: drop this block (counted) and
            // flag the discontinuity on the next one that fits.
            self.dropped.fetch_add(u64::from(frames), Ordering::Relaxed);
            self.pending_flags |= FLAG_OVERRUN;
            return;
        }
        // Samples first, then the header: a visible header implies its samples are visible.
        let _ = self.ring.samples.push_entire_slice(data);
        let _ = self.ring.blocks.push(BlockMeta {
            frames,
            flags: self.pending_flags,
            qpc_ns,
        });
        self.pending_flags = 0;
        self.since_wake = self.since_wake.saturating_add(frames);
        if self.since_wake >= self.wake_frames {
            self.since_wake = 0;
            self.waker.wake();
        }
    }

    #[inline]
    fn push_marker(&mut self, flags: u8, qpc_ns: i64) {
        if self
            .ring
            .blocks
            .push(BlockMeta {
                frames: 0,
                flags: flags | self.pending_flags,
                qpc_ns,
            })
            .is_ok()
        {
            self.pending_flags = 0;
        } else {
            self.pending_flags |= FLAG_OVERRUN;
        }
        self.since_wake = 0;
        self.waker.wake();
    }
}

/// Add / remove requests from the supervisor to a running callback.
pub(crate) enum FanOutCmd {
    Attach(Slot),
    Detach(u64),
}

/// Health counters written by the callback, read by the supervisor.
pub(crate) struct StreamHealth {
    /// Local QPC time (ns) of the latest callback that delivered frames. Liveness uses this, never
    /// the driver's capture stamp: a driver that reports a zero or stale capture position (or
    /// `AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR`) would otherwise look silent while audio flows.
    last_callback_ns: AtomicI64,
    frames: AtomicU64,
}

impl StreamHealth {
    pub(crate) fn new() -> Self {
        Self {
            last_callback_ns: AtomicI64::new(i64::MIN),
            frames: AtomicU64::new(0),
        }
    }

    /// Local QPC time of the latest callback with audio, `None` before the first one.
    pub(crate) fn last_callback_ns(&self) -> Option<i64> {
        match self.last_callback_ns.load(Ordering::Relaxed) {
            i64::MIN => None,
            ns => Some(ns),
        }
    }

    pub(crate) fn frames(&self) -> u64 {
        self.frames.load(Ordering::Relaxed)
    }
}

/// The callback side of one physical stream: fans every device block out to the consumers.
/// Backends call [`StreamSink::push`] from the audio callback.
pub struct StreamSink {
    slots: Vec<Slot>,
    cmds: rtrb::Consumer<FanOutCmd>,
    retired: rtrb::Producer<Slot>,
    gates: Arc<MuteGates>,
    health: Arc<StreamHealth>,
    channels: usize,
}

impl StreamSink {
    pub(crate) fn new(
        slots: Vec<Slot>,
        cmds: rtrb::Consumer<FanOutCmd>,
        retired: rtrb::Producer<Slot>,
        gates: Arc<MuteGates>,
        health: Arc<StreamHealth>,
        channels: u16,
    ) -> Self {
        debug_assert!(slots.capacity() >= FANOUT_SLOT_CAPACITY);
        Self {
            slots,
            cmds,
            retired,
            gates,
            health,
            channels: usize::from(channels.max(1)),
        }
    }

    /// One device buffer: interleaved f32 in the stream format, captured at `capture_qpc_ns` (the
    /// driver's stamp; it only labels the blocks, liveness uses the local time, see [`StreamHealth`]).
    /// Real-time safe: no allocation (slot storage is reserved), no lock, no logging, no IPC.
    pub fn push(&mut self, data: &[f32], capture_qpc_ns: i64) {
        self.apply_commands();
        let frames = data.len() / self.channels;
        if frames == 0 {
            return;
        }
        let data = &data[..frames * self.channels];
        let frames = u32::try_from(frames).unwrap_or(u32::MAX);
        self.health.last_callback_ns.store(now_qpc_ns(), Ordering::Relaxed);
        self.health.frames.fetch_add(u64::from(frames), Ordering::Relaxed);
        let mut i = 0;
        while i < self.slots.len() {
            if self.slots[i].is_abandoned() {
                // The consumer is gone (its ring end was dropped).
                let slot = self.slots.swap_remove(i);
                self.retire(slot);
                continue;
            }
            let slot = &mut self.slots[i];
            let muted = slot.gate.is_some_and(|g| self.gates.is_muted(g));
            if muted {
                if !slot.gated {
                    slot.gated = true;
                    slot.push_marker(FLAG_MUTE_MARK, capture_qpc_ns);
                }
            } else {
                if slot.gated {
                    slot.gated = false;
                    slot.pending_flags |= FLAG_UNMUTED;
                }
                slot.push_block(data, frames, capture_qpc_ns);
            }
            i += 1;
        }
    }

    fn apply_commands(&mut self) {
        while let Ok(cmd) = self.cmds.pop() {
            match cmd {
                FanOutCmd::Attach(slot) => {
                    if self.slots.len() < self.slots.capacity() {
                        self.slots.push(slot);
                    } else {
                        self.retire(slot);
                    }
                }
                FanOutCmd::Detach(id) => {
                    if let Some(i) = self.slots.iter().position(|s| s.id == id) {
                        let slot = self.slots.swap_remove(i);
                        self.retire(slot);
                    }
                }
            }
        }
    }

    /// Hand a slot back to the supervisor, which frees it (never free ring memory here).
    fn retire(&mut self, slot: Slot) {
        if let Err(rtrb::PushError::Full(slot)) = self.retired.push(slot) {
            // Only reachable if the supervisor stopped draining (it drains every tick): freeing on
            // this thread is the last resort.
            drop(slot);
        }
    }
}

/// Reports the death of a stream to the supervisor (from the backend's error callback).
pub struct ErrorSink {
    stream: String,
    generation: u64,
    tx: mpsc::Sender<Request>,
    fired: bool,
}

impl ErrorSink {
    pub(crate) fn new(stream: String, generation: u64, tx: mpsc::Sender<Request>) -> Self {
        Self {
            stream,
            generation,
            tx,
            fired: false,
        }
    }

    /// The stream failed (device unplugged, driver reset, ...). Not for the per-buffer path.
    pub fn report(&mut self, message: String) {
        if std::mem::replace(&mut self.fired, true) {
            return;
        }
        let _ = self.tx.send(Request::StreamFailed {
            stream: self.stream.clone(),
            generation: self.generation,
            message,
        });
    }
}

// ---------------------------------------------------------------------------------------------
// Backends and events
// ---------------------------------------------------------------------------------------------

/// A resolved capture device.
pub struct ResolvedDevice {
    /// Device name (trimmed; the id the pickers show).
    pub name: String,
    pub loopback: bool,
    pub format: StreamFormat,
    /// Backend-private handle carried from `resolve` to `open` (e.g. the cpal device and config).
    pub handle: Option<Box<dyn Any + Send>>,
}

impl ResolvedDevice {
    /// Key of the physical stream (input and loopback of one endpoint are different streams).
    pub fn physical_key(&self) -> String {
        if self.loopback {
            format!("{}{}", super::loopback::LOOPBACK_PREFIX, self.name)
        } else {
            format!("in:{}", self.name)
        }
    }
}

/// Where capture streams come from (cpal in the app, a scripted fake in tests).
pub trait CaptureBackend: Send + Sync + 'static {
    /// Resolve a source to a device and its capture format (device enumeration, tens of ms).
    fn resolve(&self, spec: &SourceSpec) -> Result<ResolvedDevice, String>;
    /// Start capturing `device` into `sink`. The backend calls `sink.push` from its audio callback
    /// and `errors.report` once if the stream dies.
    fn open(&self, device: ResolvedDevice, sink: StreamSink, errors: ErrorSink) -> Result<Box<dyn CaptureStream>, String>;
    /// The names of the input devices listed now (`None`: this backend cannot list them). The
    /// supervisor snapshots the family of a microphone when it is lost and, while it reopens it,
    /// accepts a renumbered endpoint only if it was not listed at that moment (see
    /// `watchdog::Supervisor::resolve_reopen`). Without a listing it reopens by `resolve` alone.
    fn input_device_names(&self) -> Option<Vec<String>> {
        None
    }
}

/// A running backend stream.
pub trait CaptureStream: Send {
    /// Stop capturing; returns at once. `released` gets a message once the device is released.
    fn stop(self: Box<Self>, released: Option<mpsc::Sender<()>>);
}

/// Payload of `audio:stream_error` / `audio:stream_recovered` (TS `RustAudioStreamEvent`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct StreamEvent {
    /// Source key: `"default"`, a device name, `"loopback:default"` or `"loopback:<output>"`.
    pub source: String,
    /// Source epoch; unchanged across restarts and default-device moves.
    pub epoch: u64,
    /// Error text (`stream_error`) or `null`.
    pub message: Option<String>,
}

/// Receives the hub's stream events (the Tauri emitter in the app, a recorder in tests).
pub trait HubEventSink: Send + Sync {
    fn stream_error(&self, event: &StreamEvent);
    fn stream_recovered(&self, event: &StreamEvent);
    /// A source in an outage was removed (its last consumer detached) before it recovered.
    fn stream_closed(&self, event: &StreamEvent);
}

#[derive(Default)]
pub(crate) struct EventSlot(RwLock<Option<Arc<dyn HubEventSink>>>);

impl EventSlot {
    pub(crate) fn get(&self) -> Option<Arc<dyn HubEventSink>> {
        self.0.read().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn set(&self, sink: Arc<dyn HubEventSink>) {
        *self.0.write().unwrap_or_else(PoisonError::into_inner) = Some(sink);
    }
}

// ---------------------------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------------------------

/// `plugin:audio|get_capture_status`: what the hub is doing (diagnostics, tests).
#[derive(Clone, Debug, Serialize)]
pub struct HubStatus {
    pub sharing: bool,
    pub muted: Vec<MuteGroup>,
    pub sources: Vec<SourceStatus>,
    pub streams: Vec<StreamStatus>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceStatus {
    pub source: String,
    pub epoch: u64,
    /// Physical stream key serving this source.
    pub stream: String,
    /// Consumer labels.
    pub consumers: Vec<String>,
    /// Waiting for its device (failed stream, reconnecting).
    pub recovering: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct StreamStatus {
    pub key: String,
    pub device: String,
    pub loopback: bool,
    /// `idle`, `running` or `failed`.
    pub state: &'static str,
    pub format: Option<StreamFormat>,
    pub consumers: usize,
    /// Streams opened for this device so far (1 + restarts and rebuilds).
    pub generation: u64,
    /// Failures (errors and stalls).
    pub failures: u64,
    /// Frames captured by the current generation.
    pub frames: u64,
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------------------------
// The hub
// ---------------------------------------------------------------------------------------------

/// Supervisor timing and layout.
#[derive(Clone, Debug)]
pub struct HubConfig {
    /// Supervisor housekeeping period (stall checks, freeing retired slots).
    pub tick: Duration,
    /// An input stream that delivers nothing for this long is restarted (loopback never is).
    pub stall_timeout: Duration,
    /// Grace for the first audio after a stream (re)opened, when longer than `stall_timeout`: some
    /// endpoints (Bluetooth hands-free, a USB device after a profile switch) take seconds to
    /// deliver their first packet once started.
    pub first_data_timeout: Duration,
    /// First retry delay after a failure (doubles per failed attempt).
    pub retry_min: Duration,
    /// Longest retry delay.
    pub retry_max: Duration,
    /// Share one stream per physical device (off: one stream per consumer, the legacy layout).
    pub share_streams: bool,
    /// Supervise streams: stall detection, restart after errors (same epoch), default-device
    /// follow. Off (legacy / rollback): a failed stream stays down and ends its consumers' capture.
    pub supervise: bool,
    /// How long `attach` waits for the supervisor (device opens take up to a few seconds).
    pub attach_timeout: Duration,
}

impl Default for HubConfig {
    fn default() -> Self {
        Self {
            tick: Duration::from_millis(200),
            stall_timeout: Duration::from_millis(2_500),
            first_data_timeout: Duration::from_secs(5),
            retry_min: Duration::from_millis(100),
            retry_max: Duration::from_secs(3),
            share_streams: true,
            supervise: true,
            attach_timeout: Duration::from_secs(15),
        }
    }
}

/// An on/off flag: `Some(false)` for `0`/`false`/`off`/`no`, `Some(true)` for anything else, `None`
/// when unset or blank.
fn flag_value(value: Option<&str>) -> Option<bool> {
    let v = value?.trim().to_ascii_lowercase();
    if v.is_empty() {
        return None;
    }
    Some(!matches!(v.as_str(), "0" | "false" | "off" | "no"))
}

impl HubConfig {
    /// Defaults plus the internal flags (module docs): `SIGIL_CAPTURE_SHARING=0` is the legacy,
    /// unsupervised per-consumer capture (the rollback); `SIGIL_CAPTURE_SUPERVISE` overrides
    /// supervision on its own.
    pub fn from_env() -> Self {
        Self::from_flags(
            std::env::var("SIGIL_CAPTURE_SHARING").ok().as_deref(),
            std::env::var("SIGIL_CAPTURE_SUPERVISE").ok().as_deref(),
        )
    }

    fn from_flags(sharing: Option<&str>, supervise: Option<&str>) -> Self {
        let share_streams = flag_value(sharing).unwrap_or(true);
        Self {
            share_streams,
            supervise: flag_value(supervise).unwrap_or(share_streams),
            ..Self::default()
        }
    }
}

pub(crate) struct HubInner {
    pub(crate) tx: mpsc::Sender<Request>,
    pub(crate) gates: Arc<MuteGates>,
    events: Arc<EventSlot>,
    ids: AtomicU64,
    config: HubConfig,
    supervisor: Mutex<Option<JoinHandle<()>>>,
}

impl HubInner {
    /// Detach a consumer; `released` gets a message once its stream released the device (or at
    /// once when other consumers keep the device open).
    pub(crate) fn detach(&self, id: u64, released: Option<mpsc::Sender<()>>) {
        let _ = self.tx.send(Request::Detach { id, released });
    }
}

/// The capture hub. The app uses [`global`]; tests build their own over a fake backend.
pub struct CaptureHub {
    inner: Arc<HubInner>,
}

impl CaptureHub {
    pub fn new(backend: Arc<dyn CaptureBackend>, config: HubConfig) -> Result<Self, String> {
        clock::init();
        rt_thread::preload();
        let (tx, rx) = mpsc::channel();
        let gates = Arc::new(MuteGates::default());
        let events = Arc::new(EventSlot::default());
        let supervisor = Supervisor::new(backend, config.clone(), Arc::clone(&gates), Arc::clone(&events), tx.clone());
        let handle = thread::Builder::new()
            .name("sigil-capture-hub".into())
            .spawn(move || supervisor.run(rx))
            .map_err(|e| format!("Failed to start the audio capture hub: {e}"))?;
        Ok(Self {
            inner: Arc::new(HubInner {
                tx,
                gates,
                events,
                ids: AtomicU64::new(0),
                config,
                supervisor: Mutex::new(Some(handle)),
            }),
        })
    }

    /// Where `audio:stream_error` / `audio:stream_recovered` go.
    pub fn set_event_sink(&self, sink: Arc<dyn HubEventSink>) {
        self.inner.events.set(sink);
    }

    /// Start consuming `spec`. Opens the device when this is its first consumer (blocking: call
    /// it off the async runtime). Fails fast when the device does not exist or cannot be opened;
    /// once attached, device loss is recovered automatically.
    pub fn attach(&self, spec: SourceSpec, opts: ConsumerOptions) -> Result<CaptureConsumer, String> {
        let opts = opts.sanitized();
        let id = self.inner.ids.fetch_add(1, Ordering::SeqCst) + 1;
        let shared = Arc::new(ConsumerShared::new());
        let (reply_tx, reply_rx) = mpsc::channel();
        self.inner
            .tx
            .send(Request::Attach {
                id,
                spec,
                opts: opts.clone(),
                shared: Arc::clone(&shared),
                reply: reply_tx,
            })
            .map_err(|_| HUB_STOPPED.to_string())?;
        let reply: AttachReply = match reply_rx.recv_timeout(self.inner.config.attach_timeout) {
            Ok(result) => result?,
            Err(_) => {
                // Late success would leave an orphan registration: remove it.
                self.inner.detach(id, None);
                return Err("The audio device did not open in time".to_string());
            }
        };
        Ok(CaptureConsumer::new(Arc::clone(&self.inner), id, shared, opts, reply))
    }

    /// Mute or unmute a consumer group (takes effect on the next device block).
    pub fn set_muted(&self, group: MuteGroup, muted: bool) {
        if self.inner.gates.set(group, muted) {
            tracing::info!(target: "sigil::audio::hub", "capture mute {group:?} = {muted}");
        }
    }

    #[allow(dead_code)] // Capture-hub API for the P2 speech engines and the P3 voice changer (used by tests).
    pub fn is_muted(&self, group: MuteGroup) -> bool {
        self.inner.gates.is_muted(group)
    }

    /// The device list or a default device changed: retry failed streams now and move
    /// default-following sources to the new default device.
    pub fn devices_changed(&self) {
        let _ = self.inner.tx.send(Request::DevicesChanged);
    }

    /// Snapshot of sources and streams.
    pub fn status(&self) -> Result<HubStatus, String> {
        let (tx, rx) = mpsc::channel();
        self.inner
            .tx
            .send(Request::Status { reply: tx })
            .map_err(|_| HUB_STOPPED.to_string())?;
        rx.recv_timeout(self.inner.config.attach_timeout)
            .map_err(|_| "The audio capture hub did not answer".to_string())
    }

    /// Stop every stream and the supervisor (tests; the app's global hub lives until exit).
    pub fn shutdown(&self) {
        let (tx, rx) = mpsc::channel();
        if self.inner.tx.send(Request::Shutdown { done: tx }).is_ok() {
            let _ = rx.recv_timeout(Duration::from_secs(5));
        }
        let handle = self
            .inner
            .supervisor
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(handle) = handle {
            let _ = handle.join();
        }
    }
}

impl Drop for CaptureHub {
    fn drop(&mut self) {
        self.shutdown();
    }
}

static GLOBAL: OnceLock<Result<CaptureHub, String>> = OnceLock::new();

/// The app's capture hub (cpal / WASAPI), created on first use.
pub fn global() -> Result<&'static CaptureHub, String> {
    GLOBAL
        .get_or_init(|| CaptureHub::new(Arc::new(super::stream::CpalBackend), HubConfig::from_env()))
        .as_ref()
        .map_err(Clone::clone)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qpc_clock_is_monotonic_and_in_nanoseconds() {
        let a = now_qpc_ns();
        std::thread::sleep(Duration::from_millis(20));
        let b = now_qpc_ns();
        assert!(b - a >= 15_000_000, "{a} -> {b}");
        assert!(b - a < 2_000_000_000, "{a} -> {b}");
    }

    #[test]
    fn formats_convert_durations() {
        let f = StreamFormat {
            sample_rate: 48_000,
            channels: 2,
        };
        assert_eq!(f.samples_for_ms(10), 960);
        assert_eq!(f.frames_to_ns(480), 10_000_000);
        assert_eq!(StreamFormat::STT.samples_for_ms(1000), 16_000);
    }

    #[test]
    fn options_are_clamped_and_mute_groups_serialize() {
        let o = ConsumerOptions::speech("x").with_max_backlog_ms(1).sanitized();
        assert_eq!(o.max_backlog_ms, 20);
        let o = ConsumerOptions::device_rate("x").with_max_backlog_ms(999_999).sanitized();
        assert_eq!(o.max_backlog_ms, 10_000);
        assert_eq!(serde_json::to_string(&MuteGroup::VoiceChanger).unwrap(), "\"voice_changer\"");
        let g: MuteGroup = serde_json::from_str("\"stt\"").unwrap();
        assert_eq!(g, MuteGroup::Stt);
    }

    #[test]
    fn internal_flags_select_the_legacy_capture() {
        let d = HubConfig::from_flags(None, None);
        assert!(d.share_streams && d.supervise);
        let legacy = HubConfig::from_flags(Some(" 0 "), None);
        assert!(!legacy.share_streams && !legacy.supervise, "the rollback bypasses supervision too");
        let topology = HubConfig::from_flags(Some("off"), Some("1"));
        assert!(!topology.share_streams && topology.supervise);
        let unsupervised = HubConfig::from_flags(Some("1"), Some("false"));
        assert!(unsupervised.share_streams && !unsupervised.supervise);
        let blank = HubConfig::from_flags(Some(""), Some(" "));
        assert!(blank.share_streams && blank.supervise);
    }

    #[test]
    fn stream_event_wire_format_matches_the_frozen_ts_type() {
        let e = StreamEvent {
            source: "loopback:Speakers".into(),
            epoch: 4,
            message: None,
        };
        assert_eq!(
            serde_json::to_string(&e).unwrap(),
            r#"{"source":"loopback:Speakers","epoch":4,"message":null}"#
        );
    }
}
