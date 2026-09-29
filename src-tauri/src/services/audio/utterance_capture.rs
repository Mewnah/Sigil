//! VAD-closed utterance capture for HTTP / cloud STT engines (frozen IPC contract, see
//! `speech_core` module docs):
//! - `plugin:audio|start_utterance_capture { deviceName?, vad? }` → capture id (replaces any running
//!   utterance capture; independent of `start_audio_capture` and of Whisper's own capture).
//! - `plugin:audio|stop_utterance_capture` → after it resolves, no event of that capture is emitted.
//! - `plugin:audio|take_utterance { captureId, seq }` → `ArrayBuffer` of little-endian f32 mono
//!   16 kHz PCM; each utterance can be taken once.
//! - Event `audio:utterance` ([`UtteranceEvent`], metadata only; no PCM over JSON).
//!
//! Epoch discipline: every start and stop bumps the epoch under the stop lock (`Shared::inner`);
//! the worker emits only while holding that lock with its epoch still current.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use tauri::{command, AppHandle, Emitter, Runtime, State};

use super::speech_capture::{start_speech_capture, SpeechCapture, SpeechCaptureEvent};
use crate::services::speech_core::utterance::{f32_to_le_bytes, samples_to_ms, UtteranceStore};
use crate::services::speech_core::{SegmenterEvent, UtteranceEvent, VadConfig, SAMPLE_RATE, UTTERANCE_EVENT};

/// Managed state of the utterance capture.
pub struct UtteranceCaptureState {
    shared: Arc<Shared>,
}

impl UtteranceCaptureState {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Shared {
                inner: Mutex::new(Inner {
                    active: None,
                    store: UtteranceStore::default(),
                }),
                epoch: AtomicU64::new(0),
                next_id: AtomicU64::new(0),
            }),
        }
    }
}

impl Default for UtteranceCaptureState {
    fn default() -> Self {
        Self::new()
    }
}

struct Shared {
    /// The stop lock: guards the running capture and the untaken utterances.
    inner: Mutex<Inner>,
    /// Bumped (under `inner`) by every start, stop and capture failure.
    epoch: AtomicU64,
    next_id: AtomicU64,
}

struct Inner {
    active: Option<SpeechCapture>,
    store: UtteranceStore,
}

fn lock(shared: &Shared) -> MutexGuard<'_, Inner> {
    shared.inner.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Runs on the capture's worker thread.
fn handle_event<R: Runtime>(app: &AppHandle<R>, shared: &Shared, epoch: u64, capture_id: u64, event: SpeechCaptureEvent) {
    let mut inner = lock(shared);
    if shared.epoch.load(Ordering::SeqCst) != epoch {
        return;
    }
    match event {
        SpeechCaptureEvent::Vad(SegmenterEvent::SpeechStart { seq, start_sample }) => {
            let _ = app.emit(
                UTTERANCE_EVENT,
                UtteranceEvent::SpeechStart {
                    capture_id,
                    seq,
                    start_ms: samples_to_ms(start_sample),
                },
            );
        }
        SpeechCaptureEvent::Vad(SegmenterEvent::Segment(segment)) => {
            let samples = segment.samples.len();
            let start_ms = samples_to_ms(segment.start_sample);
            let end_ms = samples_to_ms(segment.start_sample + samples as u64);
            let dropped = inner.store.insert(capture_id, segment.seq, segment.samples);
            for (dropped_capture, dropped_seq) in dropped {
                tracing::warn!(target: "sigil::stt", "utterance {dropped_seq} dropped: not taken in time (backlog full)");
                let _ = app.emit(
                    UTTERANCE_EVENT,
                    UtteranceEvent::Dropped {
                        capture_id: dropped_capture,
                        seq: dropped_seq,
                        reason: "backlog".to_string(),
                    },
                );
            }
            tracing::debug!(target: "sigil::stt", "utterance {} ready: {start_ms}-{end_ms} ms", segment.seq);
            let _ = app.emit(
                UTTERANCE_EVENT,
                UtteranceEvent::Utterance {
                    capture_id,
                    seq: segment.seq,
                    start_ms,
                    end_ms,
                    sample_rate: SAMPLE_RATE,
                    samples: u32::try_from(samples).unwrap_or(u32::MAX),
                    forced: segment.forced,
                },
            );
        }
        SpeechCaptureEvent::Error(message) => {
            let _ = app.emit(UTTERANCE_EVENT, UtteranceEvent::Error { capture_id, message });
            // The capture is over: retire it (untaken utterances stay takeable until the next
            // start/stop). Dropping only signals the threads; nothing is joined here.
            shared.epoch.fetch_add(1, Ordering::SeqCst);
            drop(inner.active.take());
        }
    }
}

#[command]
pub async fn start_utterance_capture<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, UtteranceCaptureState>,
    device_name: Option<String>,
    vad: Option<VadConfig>,
) -> Result<u64, String> {
    let shared = Arc::clone(&state.shared);
    let (epoch, capture_id, previous) = {
        let mut inner = lock(&shared);
        let epoch = shared.epoch.fetch_add(1, Ordering::SeqCst) + 1;
        inner.store.clear();
        (epoch, shared.next_id.fetch_add(1, Ordering::SeqCst) + 1, inner.active.take())
    };
    if let Some(previous) = previous {
        // Release the previous device before opening again.
        previous.stop().await;
    }
    let handler = {
        let shared = Arc::clone(&shared);
        move |event| handle_event(&app, &shared, epoch, capture_id, event)
    };
    let capture = start_speech_capture(device_name, vad.unwrap_or_default(), "Utterance", handler).await?;
    let cancelled = {
        let mut inner = lock(&shared);
        if shared.epoch.load(Ordering::SeqCst) == epoch {
            inner.active = Some(capture);
            None
        } else {
            Some(capture)
        }
    };
    match cancelled {
        // A stop (or a newer start) landed while the device was opening.
        Some(capture) => {
            capture.stop().await;
            Err("cancelled".to_string())
        }
        None => Ok(capture_id),
    }
}

#[command]
pub async fn stop_utterance_capture(state: State<'_, UtteranceCaptureState>) -> Result<(), String> {
    let previous = {
        let mut inner = lock(&state.shared);
        state.shared.epoch.fetch_add(1, Ordering::SeqCst);
        inner.store.clear();
        inner.active.take()
    };
    if let Some(previous) = previous {
        previous.stop().await;
    }
    Ok(())
}

#[command]
pub async fn take_utterance(
    state: State<'_, UtteranceCaptureState>,
    capture_id: u64,
    seq: u64,
) -> Result<tauri::ipc::Response, String> {
    let samples = lock(&state.shared).store.take(capture_id, seq).ok_or_else(|| {
        format!("Utterance {seq} of capture {capture_id} is not available (already taken, dropped, or the capture stopped)")
    })?;
    Ok(tauri::ipc::Response::new(f32_to_le_bytes(&samples)))
}
