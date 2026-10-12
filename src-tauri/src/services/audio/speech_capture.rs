//! VAD-segmented capture on the capture hub: hub consumer (16 kHz mono, raw pre-voice-changer
//! signal) → front-end worker thread (Silero segmenter) → caller's event handler.
//!
//! - The worker loads the Silero model before the device is attached, so a model problem fails the
//!   start instead of leaving a silent capture.
//! - Device loss is not fatal: the hub reconnects the stream (same epoch) and follows the default
//!   device. The worker closes any open utterance at every discontinuity (restart, device move,
//!   mute edge, overrun), so an utterance never spans two devices or a mute.
//! - Loopback sources deliver nothing while nothing plays: after 300 ms without audio the worker
//!   feeds up to 1.5 s of silence so an utterance cut off by silence still closes.
//! - [`SpeechCaptureEvent::Error`] is reserved for failures that end the capture (speech detection,
//!   a panic of the worker, or a device error under the unsupervised legacy capture,
//!   `SIGIL_CAPTURE_SHARING=0`); after it, and after [`SpeechCapture::stop`]/`signal_stop`, the
//!   worker delivers no further events.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

use tokio::sync::oneshot;

use super::hub::{CaptureConsumer, ConsumerOptions, SourceSpec};
use super::{attach_async, stt_mute_group};
use crate::services::speech_core::utterance::SpeechFrontEnd;
use crate::services::speech_core::{panic_text, SegmenterEvent, VadConfig, SAMPLE_RATE};

/// Worker wake-up period when no audio arrives (stop polling safety net; the hub wakes it every
/// ~20 ms of audio).
const IDLE_WAIT: Duration = Duration::from_millis(100);
/// Loopback: silence after which idle time is fed to the VAD as zeros.
const LOOPBACK_IDLE_AFTER: Duration = Duration::from_millis(300);
/// Loopback: at most this much synthetic silence per idle period.
const LOOPBACK_IDLE_FILL: Duration = Duration::from_millis(1_500);
/// Bound on waiting for the worker to release its hub consumer.
const STOP_TIMEOUT: Duration = Duration::from_secs(3);

/// What the front-end worker reports.
#[derive(Debug)]
pub enum SpeechCaptureEvent {
    /// Speech start / closed utterance (16 kHz sample indices since the capture started).
    Vad(SegmenterEvent),
    /// The capture failed (speech detection error, a panic of the worker; device error on the
    /// unsupervised legacy capture); no further events follow.
    Error(String),
}

/// A running VAD-segmented capture. Dropping it stops the capture (without waiting).
pub struct SpeechCapture {
    stop: Arc<AtomicBool>,
    worker: Thread,
    done: Option<oneshot::Receiver<()>>,
}

impl SpeechCapture {
    /// Stop delivering events and release the hub consumer; returns at once.
    pub fn signal_stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.worker.unpark();
    }

    /// Stop and wait until the consumer is released (the device closes with its last consumer).
    /// Bounded; `false` on timeout.
    pub async fn stop(mut self) -> bool {
        self.signal_stop();
        match self.done.take() {
            Some(done) => tokio::time::timeout(STOP_TIMEOUT, done).await.is_ok(),
            None => true,
        }
    }
}

impl Drop for SpeechCapture {
    fn drop(&mut self) {
        self.signal_stop();
    }
}

/// Capture `device_name` (any source id: `None`/`"default"`, an input name, `"loopback:..."`) through
/// the capture hub and segment it with Silero. `on_event` runs on the worker thread, in order.
pub async fn start_speech_capture<F>(
    device_name: Option<String>,
    vad: VadConfig,
    log_tag: &'static str,
    on_event: F,
) -> Result<SpeechCapture, String>
where
    F: FnMut(SpeechCaptureEvent) + Send + 'static,
{
    let spec = SourceSpec::parse(device_name.as_deref());
    let stop = Arc::new(AtomicBool::new(false));
    let (consumer_tx, consumer_rx) = mpsc::channel::<CaptureConsumer>();
    let (ready_tx, ready_rx) = oneshot::channel::<Result<(), String>>();
    let (done_tx, done_rx) = oneshot::channel::<()>();

    // Worker first: it loads Silero and reports readiness before the device is attached.
    let worker = {
        let stop = Arc::clone(&stop);
        thread::Builder::new()
            .name(format!("sigil-vad-{}", log_tag.to_ascii_lowercase()))
            .spawn(move || {
                let front_end = match SpeechFrontEnd::new(SAMPLE_RATE, 1, &vad) {
                    Ok(fe) => fe,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                if ready_tx.send(Ok(())).is_err() {
                    return;
                }
                // The consumer arrives once the device is attached (the sender drops on failure).
                let Ok(consumer) = consumer_rx.recv() else { return };
                run_worker(front_end, consumer, &stop, log_tag, on_event);
                let _ = done_tx.send(());
            })
            .map_err(|e| format!("[{log_tag}] Failed to start the speech worker: {e}"))?
            .thread()
            .clone()
    };
    let stop_worker = || {
        stop.store(true, Ordering::Release);
        worker.unpark();
    };
    match ready_rx.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err(format!("[{log_tag}] The speech worker exited during start")),
    }

    let opts = ConsumerOptions::speech(log_tag).with_mute_group(Some(stt_mute_group(&spec)));
    let consumer = match attach_async(spec, opts).await {
        Ok(consumer) => consumer,
        Err(e) => {
            stop_worker();
            return Err(e);
        }
    };
    tracing::info!(
        target: "sigil::stt",
        "[{log_tag}] capture started on source {} (epoch {}, consumer {})",
        consumer.source(),
        consumer.epoch(),
        consumer.id()
    );
    if consumer_tx.send(consumer).is_err() {
        return Err(format!("[{log_tag}] The speech worker exited during start"));
    }
    Ok(SpeechCapture {
        stop,
        worker,
        done: Some(done_rx),
    })
}

/// What the speech worker feeds device audio to ([`SpeechFrontEnd`]; a test scripts its own).
trait WorkerFrontEnd {
    fn push_interleaved(&mut self, interleaved: &[f32], out: &mut Vec<SegmenterEvent>) -> Result<(), String>;
    fn flush(&mut self, out: &mut Vec<SegmenterEvent>) -> Result<(), String>;
}

impl WorkerFrontEnd for SpeechFrontEnd {
    fn push_interleaved(&mut self, interleaved: &[f32], out: &mut Vec<SegmenterEvent>) -> Result<(), String> {
        SpeechFrontEnd::push_interleaved(self, interleaved, out)
    }

    fn flush(&mut self, out: &mut Vec<SegmenterEvent>) -> Result<(), String> {
        SpeechFrontEnd::flush(self, out)
    }
}

/// The speech worker after its device is attached: runs the front end until stopped and releases the
/// consumer. A panic ends this thread and nothing else watches it, so the capture would look alive
/// and hear nothing: it is reported like any other failure that ends the capture (the consumer is
/// released by the unwind), unless a stop was asked for first.
fn run_worker<FE, F>(front_end: FE, consumer: CaptureConsumer, stop: &AtomicBool, log_tag: &'static str, mut on_event: F)
where
    FE: WorkerFrontEnd,
    F: FnMut(SpeechCaptureEvent),
{
    match catch_unwind(AssertUnwindSafe(|| run_front_end(front_end, consumer, stop, log_tag, &mut on_event))) {
        Ok(consumer) => {
            consumer.close_and_wait();
        }
        Err(payload) => {
            let text = panic_text(payload.as_ref());
            tracing::error!(target: "sigil::stt", "[{log_tag}] the speech worker panicked: {text}");
            if !stop.load(Ordering::Acquire) {
                on_event(SpeechCaptureEvent::Error(format!("The speech worker stopped unexpectedly ({text})")));
            }
        }
    }
}

fn run_front_end<FE, F>(
    mut front_end: FE,
    mut consumer: CaptureConsumer,
    stop: &AtomicBool,
    log_tag: &'static str,
    on_event: &mut F,
) -> CaptureConsumer
where
    FE: WorkerFrontEnd,
    F: FnMut(SpeechCaptureEvent),
{
    let loopback = consumer.is_loopback();
    let mut buf: Vec<f32> = Vec::with_capacity(16_384);
    let mut events = Vec::new();
    let mut last_audio = Instant::now();
    let mut idle_filled = Duration::ZERO;
    let mut reported_drops = 0u64;
    let mut last_drop_log: Option<Instant> = None;
    // Delivers `events`; `false` once stopped.
    let deliver = |events: &mut Vec<SegmenterEvent>, on_event: &mut F| -> bool {
        for ev in events.drain(..) {
            if stop.load(Ordering::Acquire) {
                return false;
            }
            on_event(SpeechCaptureEvent::Vad(ev));
        }
        true
    };
    loop {
        if stop.load(Ordering::Acquire) {
            return consumer;
        }
        buf.clear();
        let info = consumer.read(&mut buf);
        if !buf.is_empty() {
            last_audio = Instant::now();
            idle_filled = Duration::ZERO;
            if let Err(e) = front_end.push_interleaved(&buf, &mut events) {
                tracing::error!(target: "sigil::stt", "[{log_tag}] speech detection failed: {e}");
                on_event(SpeechCaptureEvent::Error(format!("Speech detection failed: {e}")));
                return consumer;
            }
            if !deliver(&mut events, &mut *on_event) {
                return consumer;
            }
        }
        if info.gap_after {
            // Restart, device move, mute or overrun: close the open utterance here.
            if front_end.flush(&mut events).is_ok() && !deliver(&mut events, &mut *on_event) {
                return consumer;
            }
            if info.restarted {
                tracing::info!(target: "sigil::stt", "[{log_tag}] capture stream restarted; open utterance closed");
            }
        }
        let dropped = consumer.dropped_frames() + consumer.producer_dropped_frames();
        if dropped > reported_drops && last_drop_log.is_none_or(|t| t.elapsed() >= Duration::from_secs(10)) {
            tracing::warn!(
                target: "sigil::stt",
                "[{log_tag}] speech worker fell behind: {} device frames dropped",
                dropped - reported_drops
            );
            reported_drops = dropped;
            last_drop_log = Some(Instant::now());
        }
        if !buf.is_empty() || info.gap_after {
            continue;
        }
        if let Some(message) = consumer.ended() {
            // Unsupervised (legacy) capture only: the device error ends the capture, as before the hub.
            tracing::warn!(target: "sigil::stt", "[{log_tag}] input device error: {message}");
            on_event(SpeechCaptureEvent::Error(format!("Audio device error: {message}")));
            return consumer;
        }
        if loopback {
            let idle = last_audio.elapsed();
            if idle >= LOOPBACK_IDLE_AFTER && idle_filled < LOOPBACK_IDLE_FILL {
                let fill = idle.min(LOOPBACK_IDLE_FILL) - idle_filled;
                let samples = (fill.as_secs_f64() * f64::from(SAMPLE_RATE)) as usize;
                if samples > 0 {
                    buf.clear();
                    buf.resize(samples, 0.0);
                    if front_end.push_interleaved(&buf, &mut events).is_ok() && !deliver(&mut events, &mut *on_event) {
                        return consumer;
                    }
                }
                idle_filled += fill;
            }
        }
        consumer.wait(IDLE_WAIT);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::voice_library::recorder::tests::OneMic;
    use std::sync::Mutex;

    /// A front end that panics on the first audio it is given, after `before` ran.
    struct PanickingFrontEnd {
        before: Box<dyn Fn() + Send>,
    }

    impl WorkerFrontEnd for PanickingFrontEnd {
        fn push_interleaved(&mut self, _interleaved: &[f32], _out: &mut Vec<SegmenterEvent>) -> Result<(), String> {
            (self.before)();
            panic!("scripted front end panic");
        }

        fn flush(&mut self, _out: &mut Vec<SegmenterEvent>) -> Result<(), String> {
            Ok(())
        }
    }

    /// Runs a worker with `front_end` on a thread over a fake microphone, feeds it audio until it
    /// ends, checks that the hub then closed the microphone, and returns the errors it reported.
    fn run_to_the_end(front_end: PanickingFrontEnd, stop: Arc<AtomicBool>) -> Vec<String> {
        let mic = OneMic::default();
        let hub = mic.hub();
        let consumer = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::speech("test")).unwrap();
        let errors = Arc::new(Mutex::new(Vec::new()));
        let worker = {
            let errors = Arc::clone(&errors);
            thread::spawn(move || {
                run_worker(front_end, consumer, &stop, "test", move |ev| {
                    if let SpeechCaptureEvent::Error(message) = ev {
                        errors.lock().unwrap().push(message);
                    }
                });
            })
        };
        let deadline = Instant::now() + Duration::from_secs(10);
        while !worker.is_finished() {
            assert!(Instant::now() < deadline, "the worker never ended");
            mic.push(&vec![0.1; 960]);
            thread::sleep(Duration::from_millis(5));
        }
        worker.join().unwrap();
        // The consumer went with the worker: the hub closed the device.
        let released_by = Instant::now() + Duration::from_secs(10);
        while mic.push(&[0.0; 960]) {
            assert!(Instant::now() < released_by, "the microphone stayed open");
            thread::sleep(Duration::from_millis(5));
        }
        let reported = std::mem::take(&mut *errors.lock().unwrap());
        reported
    }

    /// A panic of the speech worker is a failure that ends the capture, reported once, with the
    /// device released: without the guard the capture stays "listening" and hears nothing.
    #[test]
    fn a_panic_in_the_speech_worker_is_reported_as_an_error_and_releases_the_device() {
        let stop = Arc::new(AtomicBool::new(false));
        let errors = run_to_the_end(PanickingFrontEnd { before: Box::new(|| {}) }, stop);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("stopped unexpectedly") && errors[0].contains("scripted front end panic"), "{}", errors[0]);
    }

    /// A panic while a stop is already asked for is the stop's business: nothing is reported.
    #[test]
    fn a_panic_after_a_stop_request_reports_nothing() {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let front_end = PanickingFrontEnd {
            before: Box::new(move || flag.store(true, Ordering::Release)),
        };
        let errors = run_to_the_end(front_end, stop);
        assert!(errors.is_empty(), "{errors:?}");
    }
}
