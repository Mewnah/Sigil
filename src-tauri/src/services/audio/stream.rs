//! The capture hub's cpal (WASAPI) backend: resolves sources to devices and runs each physical
//! stream on its own owner thread (cpal streams are not `Send`), with a stop signal and a
//! "device released" notification.
//!
//! The data callback runs on cpal's audio thread. It registers MMCSS "Pro Audio" and opts out of
//! EcoQoS once (`rt_thread::prepare_audio_callback_thread_once`), converts cpal's capture timestamp
//! to a QPC capture time, and hands the block to the hub's [`StreamSink`]: no allocation, locking,
//! logging or IPC. Any sample format cpal defines is converted to f32 without allocating
//! (`pcm_convert`). Output devices are captured through WASAPI loopback at their mix format.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, StreamTrait};

use super::hub::{now_qpc_ns, CaptureBackend, CaptureStream, ErrorSink, ResolvedDevice, StreamFormat, StreamSink};
use super::loopback::{capture_config, resolve_cpal_device, SourceSpec};
use crate::services::pcm_convert::build_input_stream_f32_timed;
use crate::services::rt_thread;

/// How long opening a device may take before the open is abandoned.
const OPEN_TIMEOUT: Duration = Duration::from_secs(5);

/// cpal device plus the config chosen at resolve time (carried to `open`).
struct CpalHandle {
    device: cpal::Device,
    config: cpal::SupportedStreamConfig,
}

/// The app's capture backend (cpal default host: WASAPI on Windows).
pub struct CpalBackend;

impl CaptureBackend for CpalBackend {
    fn resolve(&self, spec: &SourceSpec) -> Result<ResolvedDevice, String> {
        // Per-process loopback (P6-A) is not a cpal device.
        if let SourceSpec::Process(exe) = spec {
            return super::loopback_process::resolve(exe);
        }
        let host = cpal::default_host();
        let (device, loopback) = resolve_cpal_device(&host, spec)?;
        let name = device.name().map_err(|e| format!("Audio device name: {e}"))?;
        let config = capture_config(&device, loopback)?;
        let format = StreamFormat {
            sample_rate: config.sample_rate().0,
            channels: config.channels(),
        };
        Ok(ResolvedDevice {
            name: name.trim().to_string(),
            loopback,
            format,
            handle: Some(Box::new(CpalHandle { device, config })),
        })
    }

    fn open(&self, device: ResolvedDevice, sink: StreamSink, errors: ErrorSink) -> Result<Box<dyn CaptureStream>, String> {
        if super::loopback_process::is_process_device(&device) {
            return super::loopback_process::open(device, sink, errors);
        }
        let handle = match device.handle.and_then(|h| h.downcast::<CpalHandle>().ok()) {
            Some(handle) => *handle,
            None => {
                let host = cpal::default_host();
                let spec = SourceSpec::physical(&device.name, device.loopback);
                let (dev, loopback) = resolve_cpal_device(&host, &spec)?;
                let config = capture_config(&dev, loopback)?;
                CpalHandle { device: dev, config }
            }
        };
        // The sink was sized for the resolved format; a device that changed its mix format since
        // must be re-resolved rather than mis-interleaved.
        if handle.config.channels() != device.format.channels || handle.config.sample_rate().0 != device.format.sample_rate {
            return Err(format!("The format of '{}' changed while opening it", device.name));
        }
        let capture = spawn_capture_thread(handle, sink, errors, device.loopback)?;
        Ok(Box::new(capture))
    }
}

/// A running cpal stream on its owner thread.
struct CpalCapture {
    stop: Option<mpsc::Sender<Option<mpsc::Sender<()>>>>,
}

impl CaptureStream for CpalCapture {
    fn stop(mut self: Box<Self>, released: Option<mpsc::Sender<()>>) {
        if let Some(tx) = self.stop.take() {
            if let Err(mpsc::SendError(Some(released))) = tx.send(released) {
                // The owner thread already exited: the device is released.
                let _ = released.send(());
            }
        } else if let Some(released) = released {
            let _ = released.send(());
        }
    }
}

impl Drop for CpalCapture {
    fn drop(&mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(None);
        }
    }
}

/// Longest believable age of captured data when its callback runs. cpal passes WASAPI's
/// `GetBuffer` QPC position through unchecked (and ignores `AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR`):
/// a driver that reports 0 or a stale position would stamp blocks with an age near the uptime.
const MAX_CAPTURE_AGE_NS: i64 = 1_000_000_000;

/// Age of the data at the callback (ns), `0` when the driver's timestamp is missing, in the future
/// or implausibly old (the block is then stamped with the callback time).
#[inline]
fn capture_age_ns(age: Option<Duration>) -> i64 {
    match age.and_then(|d| i64::try_from(d.as_nanos()).ok()) {
        Some(ns) if ns <= MAX_CAPTURE_AGE_NS => ns,
        _ => 0,
    }
}

/// Capture time of a callback's first frame on the QPC clock: cpal reports the capture instant and
/// the callback instant on the device clock (QPC-based on WASAPI); their difference is the age of
/// the data when the callback started. Only labels the blocks (latency telemetry); stream liveness
/// uses the local callback time.
#[inline]
fn capture_qpc_ns(info: &cpal::InputCallbackInfo) -> i64 {
    let ts = info.timestamp();
    now_qpc_ns() - capture_age_ns(ts.callback.duration_since(&ts.capture))
}

fn spawn_capture_thread(
    handle: CpalHandle,
    mut sink: StreamSink,
    mut errors: ErrorSink,
    loopback: bool,
) -> Result<CpalCapture, String> {
    // Resolve avrt.dll here, off the audio thread, so the callback's first MMCSS call never loads a DLL.
    rt_thread::preload();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    let (stop_tx, stop_rx) = mpsc::channel::<Option<mpsc::Sender<()>>>();
    let name = if loopback { "sigil-capture-loopback" } else { "sigil-capture-input" };
    thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            let CpalHandle { device, config } = handle;
            let on_data = move |data: &[f32], info: &cpal::InputCallbackInfo| {
                // Thread-local check after the first call on this thread.
                rt_thread::prepare_audio_callback_thread_once();
                sink.push(data, capture_qpc_ns(info));
            };
            // Error path only (not per buffer): sending to the supervisor may allocate.
            let on_error = move |e: cpal::StreamError| errors.report(e.to_string());
            let started = build_input_stream_f32_timed(&device, &config, on_data, on_error)
                .and_then(|stream| stream.play().map(|()| stream).map_err(|e| format!("Failed to start the audio device: {e}")));
            match started {
                Ok(stream) => {
                    let released = if ready_tx.send(Ok(())).is_ok() {
                        // Until a stop is signalled or the handle is dropped.
                        stop_rx.recv().ok().flatten()
                    } else {
                        None
                    };
                    drop(stream);
                    if let Some(released) = released {
                        let _ = released.send(());
                    }
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            }
        })
        .map_err(|e| format!("Failed to start the capture thread: {e}"))?;
    match ready_rx.recv_timeout(OPEN_TIMEOUT) {
        Ok(Ok(())) => Ok(CpalCapture { stop: Some(stop_tx) }),
        Ok(Err(e)) => Err(e),
        // Dropping `stop_tx` makes a late-starting stream close itself at once.
        Err(_) => Err("Timed out opening the audio device".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implausible_driver_timestamps_fall_back_to_the_callback_time() {
        assert_eq!(capture_age_ns(Some(Duration::from_millis(12))), 12_000_000);
        assert_eq!(capture_age_ns(Some(Duration::from_secs(1))), 1_000_000_000);
        // Capture position 0 / stale QPC: an age of about the uptime.
        assert_eq!(capture_age_ns(Some(Duration::from_secs(86_400))), 0);
        assert_eq!(capture_age_ns(Some(Duration::MAX)), 0);
        // Capture after the callback (clock mismatch).
        assert_eq!(capture_age_ns(None), 0);
    }
}
