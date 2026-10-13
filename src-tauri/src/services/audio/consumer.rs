//! Consumer end of the capture hub: reads its own ring on its own thread.
//!
//! - **Drop-oldest on lag:** a consumer more than `max_backlog_ms` behind discards its own oldest
//!   whole blocks (other consumers of the stream are unaffected), and reports the discontinuity.
//! - **Formats:** [`ConsumerFormat::Raw16kMono`] converts on this thread with the speech front
//!   end's streaming rubato resampler (group delay compensated; exact length over a continuous
//!   run); [`ConsumerFormat::DeviceRate`] passes the device's interleaved audio through.
//! - **Discontinuities** (stream restart or move to another device, mute edges, producer overrun,
//!   drop-oldest): a read never returns audio from both sides of one. It returns the audio before
//!   it (plus the 16 kHz resampler's flushed tail) with [`ReadInfo::gap_after`] set, so the caller
//!   can close its state (VAD utterance, crossfade) before the next read continues.
//! - **Waiting:** [`CaptureConsumer::wait`] parks the calling thread; the audio callback unparks it
//!   every `wake_ms` of new audio (no sleep-polling). The first thread that waits is the one woken.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, PoisonError};
use std::thread;
use std::time::Duration;

use super::hub::{
    ConsumerFormat, ConsumerOptions, HubInner, RingReader, StreamFormat, WakerCell, FLAG_BOUNDARY, RELEASE_TIMEOUT,
};
use super::watchdog::{AttachReply, Request};
use crate::services::speech_core::resample::StreamResampler;

/// How long [`CaptureConsumer::device_name`] waits for the supervisor (which may be busy opening a device).
const DEVICE_NAME_TIMEOUT: Duration = Duration::from_millis(500);

/// State shared between a consumer and the supervisor.
pub(crate) struct ConsumerShared {
    /// The ring of a newly (re)started stream, picked up once the current ring is drained.
    mailbox: Mutex<Option<RingReader>>,
    has_mail: AtomicBool,
    pub(crate) waker: Arc<WakerCell>,
    /// The consumer's stream failed and the hub is reconnecting it.
    recovering: AtomicBool,
    /// Unsupervised hub only: the stream died and is not reopened (why, once set).
    ended: Mutex<Option<String>>,
    has_ended: AtomicBool,
    /// Frames the producer dropped because this consumer's ring was full.
    pub(crate) producer_dropped: Arc<AtomicU64>,
}

impl ConsumerShared {
    pub(crate) fn new() -> Self {
        Self {
            mailbox: Mutex::new(None),
            has_mail: AtomicBool::new(false),
            waker: Arc::new(WakerCell::default()),
            recovering: AtomicBool::new(false),
            ended: Mutex::new(None),
            has_ended: AtomicBool::new(false),
            producer_dropped: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Supervisor (unsupervised hub): the stream failed for good; the consumer should end.
    pub(crate) fn end(&self, message: &str) {
        let mut ended = self.ended.lock().unwrap_or_else(PoisonError::into_inner);
        if ended.is_none() {
            *ended = Some(message.to_string());
        }
        drop(ended);
        self.has_ended.store(true, Ordering::Release);
        self.waker.wake();
    }

    /// Supervisor: hand over the ring of a (re)started stream and wake the consumer.
    pub(crate) fn deliver_ring(&self, ring: RingReader) {
        *self.mailbox.lock().unwrap_or_else(PoisonError::into_inner) = Some(ring);
        self.has_mail.store(true, Ordering::Release);
        self.recovering.store(false, Ordering::Release);
        self.waker.wake();
    }

    pub(crate) fn set_recovering(&self, recovering: bool) {
        self.recovering.store(recovering, Ordering::Release);
        self.waker.wake();
    }

    fn take_mail(&self) -> Option<RingReader> {
        if !self.has_mail.swap(false, Ordering::AcqRel) {
            return None;
        }
        self.mailbox.lock().unwrap_or_else(PoisonError::into_inner).take()
    }
}

/// Result of one [`CaptureConsumer::read`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReadInfo {
    /// Format of the samples this read appended (`None` when it appended nothing).
    pub format: Option<StreamFormat>,
    /// Device frames consumed.
    pub frames_in: u64,
    /// Samples appended to the output.
    pub samples_out: usize,
    /// The audio is discontinuous right after the samples of this read: close open state now.
    pub gap_after: bool,
    /// The discontinuity is a stream restart or a move to another device.
    pub restarted: bool,
    /// Frames this consumer dropped (its oldest) because it fell more than `max_backlog_ms` behind.
    pub dropped_frames: u64,
    /// QPC capture time (ns) just past the newest frame consumed.
    pub end_qpc_ns: Option<i64>,
}

/// A consumer of one capture source. Dropping it detaches it (the device closes with its last
/// consumer).
pub struct CaptureConsumer {
    hub: Arc<HubInner>,
    id: u64,
    shared: Arc<ConsumerShared>,
    opts: ConsumerOptions,
    source: String,
    epoch: u64,
    ring: Option<RingReader>,
    resampler: Option<StreamResampler>,
    resampler_format: Option<StreamFormat>,
    /// Audio went out since the last reported discontinuity.
    delivered_since_boundary: bool,
    dropped_frames: u64,
    detached: bool,
}

impl CaptureConsumer {
    pub(crate) fn new(
        hub: Arc<HubInner>,
        id: u64,
        shared: Arc<ConsumerShared>,
        opts: ConsumerOptions,
        reply: AttachReply,
    ) -> Self {
        Self {
            hub,
            id,
            shared,
            opts,
            source: reply.source,
            epoch: reply.epoch,
            ring: None,
            resampler: None,
            resampler_format: None,
            delivered_since_boundary: false,
            dropped_frames: 0,
            detached: false,
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    /// Source key (`"default"`, a device name, `"loopback:<output>"`...).
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Source epoch (stable across restarts and default-device moves).
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Name of the physical device this consumer is served from now: for a source that follows the
    /// Windows default it is the device behind `"default"` at this moment, which can change. Asks the
    /// supervisor (never call it from an audio callback or in a loop); waits at most
    /// [`DEVICE_NAME_TIMEOUT`] and gives `None` on a timeout, a stopped hub or an unknown source.
    pub fn device_name(&self) -> Option<String> {
        let (reply, answer) = mpsc::channel();
        self.hub.tx.send(Request::Status { reply }).ok()?;
        let status = answer.recv_timeout(DEVICE_NAME_TIMEOUT).ok()?;
        let source = status
            .sources
            .iter()
            .find(|s| s.source == self.source && s.epoch == self.epoch)?;
        status
            .streams
            .iter()
            .find(|s| s.key == source.stream)
            .map(|s| s.device.clone())
    }

    /// Output loopback, including per-process capture (`process:<exe>`, P6-A): such sources get the
    /// speech front end's loopback idle fill.
    pub fn is_loopback(&self) -> bool {
        super::loopback::SourceSpec::parse(Some(&self.source)).is_loopback()
    }

    /// The stream failed and the hub is reconnecting it (nothing arrives meanwhile).
    #[allow(dead_code)] // Capture-hub API for the P2 speech engines and the P3 voice changer (used by tests).
    pub fn is_recovering(&self) -> bool {
        self.shared.recovering.load(Ordering::Acquire)
    }

    /// The capture is over: the stream failed and will not be reopened (the error text). Only an
    /// unsupervised hub (`SIGIL_CAPTURE_SHARING=0`, the legacy capture) ends consumers; a
    /// supervised one reconnects instead. Check it when a read returns nothing, after the audio
    /// before the failure was read.
    pub fn ended(&self) -> Option<String> {
        if !self.shared.has_ended.load(Ordering::Acquire) {
            return None;
        }
        self.shared.ended.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Frames dropped by this consumer's drop-oldest so far.
    pub fn dropped_frames(&self) -> u64 {
        self.dropped_frames
    }

    /// Frames the producer could not store because this consumer's ring was full.
    pub fn producer_dropped_frames(&self) -> u64 {
        self.shared.producer_dropped.load(Ordering::Relaxed)
    }

    fn output_format(&self, ring_format: StreamFormat) -> StreamFormat {
        match self.opts.format {
            ConsumerFormat::Raw16kMono => StreamFormat::STT,
            ConsumerFormat::DeviceRate => ring_format,
        }
    }

    /// Whether a read would make progress (data, a ring switch, or a lost ring).
    fn has_pending(&self) -> bool {
        self.shared.has_mail.load(Ordering::Acquire)
            || self
                .ring
                .as_ref()
                .is_some_and(|r| !r.blocks.is_empty() || r.blocks.is_abandoned())
    }

    /// Park the calling thread until new audio (or a stream change) arrives, at most `timeout`.
    /// Always call it from the thread that reads.
    pub fn wait(&self, timeout: Duration) {
        self.shared.waker.register_current();
        if self.has_pending() {
            return;
        }
        thread::park_timeout(timeout);
    }

    /// Append everything available (up to the next discontinuity) to `out`; never blocks.
    pub fn read(&mut self, out: &mut Vec<f32>) -> ReadInfo {
        let start = out.len();
        let mut info = ReadInfo::default();
        'outer: loop {
            // 1. Stream restarted / moved / lost: switch rings once the current one is drained.
            let drained = self.ring.as_ref().is_none_or(|r| r.blocks.is_empty());
            if drained {
                if let Some(next) = self.shared.take_mail() {
                    let stop = self.boundary(out, &mut info);
                    self.ring = Some(next);
                    if stop {
                        info.restarted = true;
                        break;
                    }
                    continue;
                }
                // Abandoned first, then empty: a block pushed before the producer went away is
                // visible once the abandonment is.
                if self
                    .ring
                    .as_ref()
                    .is_some_and(|r| r.blocks.is_abandoned() && r.blocks.is_empty())
                {
                    self.ring = None;
                    if self.boundary(out, &mut info) {
                        info.restarted = true;
                        break;
                    }
                }
            }
            let Some(ring) = self.ring.as_mut() else { break };

            // 2. Drop-oldest past the backlog budget.
            let dropped = drop_oldest(ring, self.opts.max_backlog_ms);
            if dropped > 0 {
                self.dropped_frames += dropped;
                info.dropped_frames += dropped;
                if self.boundary(out, &mut info) {
                    break;
                }
            }

            // 3. Deliver blocks up to the next discontinuity.
            loop {
                let Some(ring) = self.ring.as_mut() else { break 'outer };
                let Ok(&meta) = ring.blocks.peek() else {
                    // Drained: a pending switch is handled at the top.
                    if self.shared.has_mail.load(Ordering::Acquire)
                        || ring.blocks.is_abandoned()
                    {
                        continue 'outer;
                    }
                    break 'outer;
                };
                if meta.flags & FLAG_BOUNDARY != 0 && self.boundary(out, &mut info) {
                    // Left in the ring: the next read delivers it (the boundary is then reported).
                    break 'outer;
                }
                let ring = self.ring.as_mut().expect("ring checked above");
                let _ = ring.blocks.pop();
                if meta.frames == 0 {
                    continue;
                }
                let format = ring.format;
                let n = meta.frames as usize * usize::from(format.channels.max(1));
                let Ok(chunk) = ring.samples.read_chunk(n) else {
                    // Header without its samples: the ring is inconsistent; drop it (recovers on
                    // the next stream generation).
                    tracing::error!(target: "sigil::audio::hub", "consumer {} ring out of sync; resetting", self.id);
                    self.ring = None;
                    break 'outer;
                };
                let (a, b) = chunk.as_slices();
                match self.opts.format {
                    ConsumerFormat::DeviceRate => {
                        out.extend_from_slice(a);
                        out.extend_from_slice(b);
                    }
                    ConsumerFormat::Raw16kMono => {
                        if self.resampler_format != Some(format) {
                            self.resampler = StreamResampler::new(format.sample_rate, format.channels).ok();
                            self.resampler_format = Some(format);
                        }
                        if let Some(r) = self.resampler.as_mut() {
                            let _ = r.push_interleaved(a, out);
                            let _ = r.push_interleaved(b, out);
                        }
                    }
                }
                chunk.commit_all();
                self.delivered_since_boundary = true;
                info.frames_in += u64::from(meta.frames);
                info.end_qpc_ns = Some(meta.qpc_ns + format.frames_to_ns(u64::from(meta.frames)));
                info.format = Some(self.output_format(format));
            }
        }
        info.samples_out = out.len() - start;
        if info.samples_out > 0 && info.format.is_none() {
            info.format = Some(StreamFormat::STT);
        }
        info
    }

    /// A discontinuity: when audio went out since the last one, flush the resampler tail into
    /// `out`, report `gap_after` and return `true` (the read stops here).
    fn boundary(&mut self, out: &mut Vec<f32>, info: &mut ReadInfo) -> bool {
        if !self.delivered_since_boundary {
            if let Some(r) = self.resampler.as_mut() {
                r.reset();
            }
            return false;
        }
        if let Some(r) = self.resampler.as_mut() {
            let _ = r.flush(out);
        }
        self.delivered_since_boundary = false;
        info.gap_after = true;
        true
    }

    /// End of use: flush the 16 kHz resampler's delayed tail (so a continuous run yields exactly
    /// `round(frames * 16000 / rate)` samples). No-op for device-rate consumers.
    #[allow(dead_code)] // Capture-hub API for the P2 speech engines and the P3 voice changer (used by tests).
    pub fn flush(&mut self, out: &mut Vec<f32>) {
        if let Some(r) = self.resampler.as_mut() {
            let _ = r.flush(out);
        }
        self.delivered_since_boundary = false;
    }

    /// Detach and wait (up to [`RELEASE_TIMEOUT`]) until the device is released when this was its
    /// last consumer. Blocking; `false` on timeout.
    pub fn close_and_wait(mut self) -> bool {
        self.detached = true;
        let (tx, rx) = mpsc::channel();
        self.hub.detach(self.id, Some(tx));
        rx.recv_timeout(RELEASE_TIMEOUT).is_ok()
    }
}

impl Drop for CaptureConsumer {
    fn drop(&mut self) {
        if !self.detached {
            self.hub.detach(self.id, None);
        }
    }
}

/// Discard whole blocks from the front of `ring` until it holds at most `max_backlog_ms`.
/// Returns the frames discarded.
fn drop_oldest(ring: &mut RingReader, max_backlog_ms: u32) -> u64 {
    let max_samples = ring.format.samples_for_ms(max_backlog_ms);
    let channels = usize::from(ring.format.channels.max(1));
    let mut dropped = 0u64;
    while ring.samples.slots() > max_samples {
        let Ok(meta) = ring.blocks.pop() else { break };
        let n = meta.frames as usize * channels;
        if let Ok(chunk) = ring.samples.read_chunk(n) {
            chunk.commit_all();
        }
        dropped += u64::from(meta.frames);
    }
    dropped
}
