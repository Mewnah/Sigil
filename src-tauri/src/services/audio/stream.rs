//! Input streams on a dedicated thread (cpal streams are not `Send`), with a stop signal and a
//! "device released" notification.
//!
//! Real-time rules for `on_data` (it runs on the driver's callback thread): no allocation, no
//! locking, no logging, no IPC. Callers hand samples to a worker through `rtrb` and wake it with
//! `Thread::unpark`. The callback thread registers MMCSS "Pro Audio" once (see `rt_thread`).

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use cpal::StreamError;
use cpal::traits::StreamTrait;
use tokio::sync::oneshot;

use crate::services::pcm_convert::build_input_stream_f32;
use crate::services::rt_thread;

/// How long a stop waits for the capture thread to release the device.
pub const RELEASE_TIMEOUT: Duration = Duration::from_millis(1500);

/// A running input stream. Dropping it stops the stream (without waiting for the device release).
pub struct InputStream {
    stop: Option<mpsc::Sender<()>>,
    released: Option<oneshot::Receiver<()>>,
}

impl InputStream {
    /// Ask the stream thread to drop the stream. Idempotent; returns at once.
    pub fn signal_stop(&mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
    }

    /// Stop and wait (up to [`RELEASE_TIMEOUT`]) until the device is released. `false` on timeout.
    pub async fn stop(mut self) -> bool {
        self.signal_stop();
        match self.released.take() {
            Some(rx) => tokio::time::timeout(RELEASE_TIMEOUT, rx).await.is_ok(),
            None => true,
        }
    }
}

impl Drop for InputStream {
    fn drop(&mut self) {
        self.signal_stop();
    }
}

/// Open `device` with `config` on a dedicated thread and start it. `on_data` receives interleaved
/// f32 slices in the device format; `on_error` runs on cpal's thread when the stream fails (device
/// unplugged, driver reset). Resolves once the stream is playing, or with the build/play error.
pub async fn open_input_stream<D, E>(
    device: cpal::Device,
    config: cpal::SupportedStreamConfig,
    mut on_data: D,
    on_error: E,
    log_tag: &'static str,
) -> Result<InputStream, String>
where
    D: FnMut(&[f32]) + Send + 'static,
    E: FnMut(StreamError) + Send + 'static,
{
    // Resolve avrt.dll here, off the audio thread, so the callback's first MMCSS call never loads a DLL.
    rt_thread::preload();
    let (ready_tx, ready_rx) = oneshot::channel::<Result<(), String>>();
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let (released_tx, released_rx) = oneshot::channel::<()>();
    thread::Builder::new()
        .name(format!("sigil-capture-{}", log_tag.to_ascii_lowercase()))
        .spawn(move || {
            let callback = move |data: &[f32]| {
                // After the first call on this thread this is a thread-local check.
                rt_thread::register_mmcss_pro_audio_once();
                on_data(data);
            };
            let started = build_input_stream_f32(&device, &config, callback, on_error)
                .and_then(|stream| stream.play().map(|()| stream).map_err(|e| e.to_string()));
            match started {
                Ok(stream) => {
                    if ready_tx.send(Ok(())).is_ok() {
                        // Until a stop is signalled or the handle is dropped.
                        let _ = stop_rx.recv();
                    }
                    drop(stream);
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            }
            let _ = released_tx.send(());
        })
        .map_err(|e| format!("[{log_tag}] Failed to start the capture thread: {e}"))?;
    ready_rx
        .await
        .map_err(|_| format!("[{log_tag}] The capture thread exited before the stream started"))??;
    Ok(InputStream {
        stop: Some(stop_tx),
        released: Some(released_rx),
    })
}
