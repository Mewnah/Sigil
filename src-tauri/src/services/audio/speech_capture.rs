//! VAD-segmented microphone capture: cpal callback → `rtrb` ring → front-end worker thread
//! (downmix, rubato to 16 kHz, Silero segmenter) → caller's event handler.
//!
//! - The audio callback only copies samples into the ring and, once ~20 ms are buffered, unparks the
//!   worker (no allocation, locks, logging or IPC). If the worker ever falls 2 s behind, the newest
//!   samples are dropped and counted (logged by the worker), never queued without bound.
//! - The worker loads the Silero model before the device is opened, so a model problem fails the
//!   start instead of leaving a silent capture.
//! - A device error (unplugged, driver reset) closes any open utterance and reports
//!   [`SpeechCaptureEvent::Error`]; the worker then exits.
//! - After [`SpeechCapture::stop`]/`signal_stop` the worker delivers no further events.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

use cpal::traits::DeviceTrait;
use tokio::sync::oneshot;

use super::device_resolve::resolve_input_device;
use super::stream::{open_input_stream, InputStream};
use crate::services::speech_core::utterance::SpeechFrontEnd;
use crate::services::speech_core::{SegmenterEvent, VadConfig};

/// Ring capacity in seconds of interleaved device audio.
const RING_SECONDS: usize = 2;
/// Worker wake-up period when no data arrives (stop / device-error polling safety net).
const IDLE_PARK: Duration = Duration::from_millis(100);
/// Samples popped from the ring per front-end call.
const POP_CHUNK: usize = 8192;

/// What the front-end worker reports.
#[derive(Debug)]
pub enum SpeechCaptureEvent {
    /// Speech start / closed utterance (16 kHz sample indices since the capture started).
    Vad(SegmenterEvent),
    /// The capture failed (device lost, model error); no further events follow.
    Error(String),
}

/// A running VAD-segmented capture. Dropping it stops the capture (without waiting).
pub struct SpeechCapture {
    /// `None` only after `stop` consumed it.
    stream: Option<InputStream>,
    stop: Arc<AtomicBool>,
    worker: Thread,
}

impl SpeechCapture {
    /// Stop delivering events and release the device; returns at once.
    pub fn signal_stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.worker.unpark();
        if let Some(stream) = self.stream.as_mut() {
            stream.signal_stop();
        }
    }

    /// Stop and wait until the device is released (bounded; `false` on timeout).
    pub async fn stop(mut self) -> bool {
        self.stop.store(true, Ordering::Release);
        self.worker.unpark();
        match self.stream.take() {
            Some(stream) => stream.stop().await,
            None => true,
        }
    }
}

impl Drop for SpeechCapture {
    fn drop(&mut self) {
        self.signal_stop();
    }
}

/// Shared between the cpal error callback and the worker.
#[derive(Default)]
struct DeviceFailure {
    flag: AtomicBool,
    message: Mutex<Option<String>>,
}

/// Open the input device (`None`/`"default"`: system default) and start VAD segmentation.
/// `on_event` runs on the worker thread, in stream order.
pub async fn start_speech_capture<F>(
    device_name: Option<String>,
    vad: VadConfig,
    log_tag: &'static str,
    on_event: F,
) -> Result<SpeechCapture, String>
where
    F: FnMut(SpeechCaptureEvent) + Send + 'static,
{
    let host = cpal::default_host();
    let device = resolve_input_device(&host, device_name.as_deref())?;
    let config = device
        .default_input_config()
        .map_err(|e| format!("Input device format: {e}"))?;
    let rate = config.sample_rate().0;
    let channels = config.channels();
    let capacity = (rate as usize * usize::from(channels) * RING_SECONDS).max(POP_CHUNK * 2);
    let (mut producer, consumer) = rtrb::RingBuffer::<f32>::new(capacity);
    let stop = Arc::new(AtomicBool::new(false));
    let failure = Arc::new(DeviceFailure::default());
    let overrun_samples = Arc::new(AtomicU64::new(0));

    // Worker first: it loads Silero and reports readiness before the device is opened.
    let (ready_tx, ready_rx) = oneshot::channel::<Result<(), String>>();
    let worker = {
        let stop = Arc::clone(&stop);
        let failure = Arc::clone(&failure);
        let overrun_samples = Arc::clone(&overrun_samples);
        thread::Builder::new()
            .name(format!("sigil-vad-{}", log_tag.to_ascii_lowercase()))
            .spawn(move || {
                let front_end = match SpeechFrontEnd::new(rate, channels, &vad) {
                    Ok(fe) => fe,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                if ready_tx.send(Ok(())).is_err() {
                    return;
                }
                run_front_end(front_end, consumer, &stop, &failure, &overrun_samples, log_tag, on_event);
            })
            .map_err(|e| format!("[{log_tag}] Failed to start the speech worker: {e}"))?
            .thread()
            .clone()
    };
    ready_rx
        .await
        .map_err(|_| format!("[{log_tag}] The speech worker exited during start"))??;

    // ~20 ms of interleaved samples between wake-ups.
    let wake_after = (rate as usize * usize::from(channels) / 50).max(1);
    let frame = usize::from(channels.max(1));
    let data_worker = worker.clone();
    let error_worker = worker.clone();
    let error_failure = Arc::clone(&failure);
    let stream = open_input_stream(
        device,
        config,
        move |data: &[f32]| {
            // Whole frames only, so an overflow can never shift the channel interleaving.
            let fits = if data.len() <= producer.slots() {
                data.len()
            } else {
                producer.slots() / frame * frame
            };
            if fits > 0 {
                let _ = producer.push_entire_slice(&data[..fits]);
            }
            if fits < data.len() {
                overrun_samples.fetch_add((data.len() - fits) as u64, Ordering::Relaxed);
            }
            if capacity - producer.slots() >= wake_after {
                data_worker.unpark();
            }
        },
        move |err| {
            // Error path only (not the per-buffer callback): a lock here is fine.
            *error_failure.message.lock().unwrap_or_else(PoisonError::into_inner) = Some(err.to_string());
            error_failure.flag.store(true, Ordering::Release);
            error_worker.unpark();
        },
        log_tag,
    )
    .await;
    match stream {
        Ok(stream) => {
            tracing::info!(target: "sigil::stt", "[{log_tag}] capture started: {rate} Hz, {channels} ch");
            Ok(SpeechCapture {
                stream: Some(stream),
                stop,
                worker,
            })
        }
        Err(e) => {
            stop.store(true, Ordering::Release);
            worker.unpark();
            Err(e)
        }
    }
}

fn run_front_end<F>(
    mut front_end: SpeechFrontEnd,
    mut consumer: rtrb::Consumer<f32>,
    stop: &AtomicBool,
    failure: &DeviceFailure,
    overrun_samples: &AtomicU64,
    log_tag: &'static str,
    mut on_event: F,
) where
    F: FnMut(SpeechCaptureEvent),
{
    let mut buf = vec![0.0f32; POP_CHUNK];
    let mut events = Vec::new();
    let mut reported_overrun = 0u64;
    let mut last_overrun_log = Instant::now() - Duration::from_secs(60);
    loop {
        if stop.load(Ordering::Acquire) {
            return;
        }
        let n = consumer.pop_partial_slice(&mut buf).0.len();
        if n > 0 {
            if let Err(e) = front_end.push_interleaved(&buf[..n], &mut events) {
                tracing::error!(target: "sigil::stt", "[{log_tag}] speech detection failed: {e}");
                on_event(SpeechCaptureEvent::Error(format!("Speech detection failed: {e}")));
                return;
            }
            for ev in events.drain(..) {
                if stop.load(Ordering::Acquire) {
                    return;
                }
                on_event(SpeechCaptureEvent::Vad(ev));
            }
            let overrun = overrun_samples.load(Ordering::Relaxed);
            if overrun > reported_overrun && last_overrun_log.elapsed() >= Duration::from_secs(10) {
                tracing::warn!(
                    target: "sigil::stt",
                    "[{log_tag}] capture overrun: {} samples dropped (speech worker starved)",
                    overrun - reported_overrun
                );
                reported_overrun = overrun;
                last_overrun_log = Instant::now();
            }
            continue;
        }
        if failure.flag.load(Ordering::Acquire) {
            let message = failure
                .message
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take()
                .unwrap_or_else(|| "unknown error".to_string());
            tracing::warn!(target: "sigil::stt", "[{log_tag}] input device error: {message}");
            // Keep what was said before the device went away.
            if front_end.flush(&mut events).is_ok() {
                for ev in events.drain(..) {
                    if stop.load(Ordering::Acquire) {
                        return;
                    }
                    on_event(SpeechCaptureEvent::Vad(ev));
                }
            }
            if !stop.load(Ordering::Acquire) {
                on_event(SpeechCaptureEvent::Error(format!("Audio input device error: {message}")));
            }
            return;
        }
        if consumer.is_abandoned() {
            // The stream (and its producer) is gone without an error: nothing more will arrive.
            return;
        }
        thread::park_timeout(IDLE_PARK);
    }
}
