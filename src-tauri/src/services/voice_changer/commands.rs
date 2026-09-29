//! Voice-changer IPC commands: params, enable, start/stop and model slots.

use super::params::{resolve_rvc_paths, VoiceChangerMode, VoiceChangerParams};
use super::state::{lock, VoiceChangerState, WorkerDoneGuard, WorkerHandle};
use super::worker::{emit_vc_error, panic_message, run_voice_changer_loop, RunGeneration, WorkerError};
use serde::Serialize;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::thread;
use std::time::Duration;
use tauri::{command, AppHandle, Emitter, Manager, Runtime, State};

/// Event carrying the effective params when Rust changed what the UI sent (entering a voice
/// applies that voice's `slot.json`); the UI should adopt them.
pub(super) const PARAMS_EVENT: &str = "voice_changer:params";
/// How long a new worker waits for its predecessor to release devices and sessions.
const PREDECESSOR_WAIT: Duration = Duration::from_secs(3);

#[command]
pub(super) fn get_voice_changer_params(state: State<'_, VoiceChangerState>) -> Result<VoiceChangerParams, String> {
    Ok(state.snapshot_params())
}

/// Store the params (clamped / normalized), persist them, and return the effective params. When
/// they differ from what was sent (a voice's own settings applied), `voice_changer:params` is also
/// emitted with them.
#[command]
pub(super) fn set_voice_changer_params<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, VoiceChangerState>,
    params: VoiceChangerParams,
) -> Result<VoiceChangerParams, String> {
    let (effective, adjusted) = state.update_params(params);
    if adjusted {
        let _ = app.emit(PARAMS_EVENT, &effective);
    }
    Ok(effective)
}

#[command]
pub(super) fn set_voice_changer_enabled(state: State<'_, VoiceChangerState>, enabled: bool) -> Result<(), String> {
    *lock(&state.enabled) = enabled;
    Ok(())
}

/// Start the worker. `input_device`, when given, replaces the stored input device (`""` = system
/// default) and is persisted; when absent the stored one is used.
#[command]
pub(super) async fn start_voice_changer<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, VoiceChangerState>,
    input_device: Option<String>,
) -> Result<(), String> {
    if state.is_running.load(Ordering::Acquire) {
        return Err("Voice changer already running".to_string());
    }

    let mode = lock(&state.mode).clone();
    if mode == VoiceChangerMode::Onnx {
        let path = lock(&state.onnx_model_path).clone();
        let p = path.trim();
        if p.is_empty() {
            return Err("Choose an ONNX model file for ONNX voice mode.".into());
        }
        if !Path::new(p).is_file() {
            return Err(format!("ONNX model file not found: {}", p));
        }
    }
    if mode == VoiceChangerMode::Sidecar {
        let url = lock(&state.sidecar_ws_url).clone();
        if url.trim().is_empty() {
            return Err("Enter a WebSocket URL for external voice conversion (ws:// or wss://).".into());
        }
    }
    if mode == VoiceChangerMode::Rvc && !state.rvc_consent_acknowledged.load(Ordering::Relaxed) {
        // Defense in depth: the TS service already refuses (and shows the notice) before this.
        return Err("Acknowledge the RVC voice-permission notice first.".into());
    }
    if mode == VoiceChangerMode::Rvc {
        // Same rule as the worker and the inspector: resolve the effective RVC EP (preset default
        // only when unset); on the CPU EP a GPU-only F0 runs as fcpe, so preflight checks the fcpe
        // file. The stored choice is kept: the worker falls back and emits `rvc_warning`, and a
        // later switch to CUDA/DML uses the chosen extractor again.
        let mut resolved = lock(&state.rvc).clone();
        let app_data = app.path().app_data_dir().map_err(|e| e.to_string())?;
        resolve_rvc_paths(&mut resolved, &app_data)?;
        // Keep the resolved voice folder (slot.json follows it), but not the base-model paths:
        // those stay as the user set them (usually empty = found automatically), so the saved
        // settings never pin an absolute path that was only a default. The worker does not rely
        // on either: it resolves again from its own snapshot.
        lock(&state.rvc).slot_dir = resolved.slot_dir;
    }
    if let Some(d) = input_device {
        *lock(&state.input_device) = d.trim().to_string();
    }
    state.save_settings();
    // A start is an explicit retry: GPU canaries that failed earlier (VRAM held by a game, a DLL
    // installed since) run again; passes stay cached. Every model session is built after this.
    crate::services::diagnostics::canary::forget_failures();
    let input_device_name = {
        let d = lock(&state.input_device).clone();
        (!d.is_empty()).then_some(d)
    };

    if let Some(sender) = lock(&state.stop_sender).take() {
        let _ = sender.send(());
    }

    let (tx, rx) = channel();
    *lock(&state.stop_sender) = Some(tx);
    // Claim the generation before publishing `is_running`, so a still-live predecessor can
    // never pass its exit-guard compare after we set the flag.
    let my_generation = state.run_generation.fetch_add(1, Ordering::SeqCst) + 1;
    let generation = RunGeneration {
        counter: state.run_generation.clone(),
        mine: my_generation,
    };
    state.is_running.store(true, Ordering::Release);
    // Resource governor: the model-size policy leaves room for a running voice changer. Set before
    // the worker exists, so a worker that stops itself at once always clears it afterwards.
    crate::services::resources::set_vc_running(true);
    *lock(&state.enabled) = true;
    let previous_worker = lock(&state.worker).take();
    if let Some(prev) = &previous_worker {
        // A parked predecessor sees the stop message at once.
        prev.join.thread().unpark();
    }

    let output_device_name = lock(&state.output_device).clone();
    let monitor_output_device = lock(&state.monitor_output_device).clone();
    let app_for_thread = app.clone();
    let run_state = state.inner().clone();
    let (done_tx, done_rx) = channel::<()>();

    let spawned = thread::Builder::new().name("sigil-vc-worker".into()).spawn(move || {
        let _done = WorkerDoneGuard(done_tx);
        // Wait (bounded) for the previous worker to release its devices/sessions before opening
        // ours. A worker blocked in a sidecar read or a model load must not pin every future
        // start, so give up after the deadline and leave it detached instead of joining forever.
        if let Some(prev) = previous_worker {
            match prev.done.recv_timeout(PREDECESSOR_WAIT) {
                Ok(()) | Err(RecvTimeoutError::Disconnected) => {
                    let _ = prev.join.join();
                }
                Err(RecvTimeoutError::Timeout) => {
                    tracing::warn!("voice changer: previous worker still busy after 3 s; continuing without it");
                }
            }
        }

        // Superseded while waiting: do not open devices or load models.
        if !generation.is_current() {
            return;
        }
        // Stopped while waiting: honour it here. The queued message is only read at the loop head,
        // which is after the devices are opened, the rates republished and (RVC) the models loaded
        // — i.e. seconds of live microphone after the user pressed Stop.
        if rx.try_recv().is_ok() {
            if generation.is_current() {
                *lock(&run_state.enabled) = false;
                run_state.is_running.store(false, Ordering::Release);
                crate::services::resources::set_vc_running(false);
                let _ = app_for_thread.emit("voice_changer:stopped", ());
            }
            return;
        }
        // `join()` orders the predecessor's final stores before this point; re-assert the flags to
        // close the remaining check-then-store window in its exit guard (which clears both).
        *lock(&run_state.enabled) = true;
        run_state.is_running.store(true, Ordering::Release);

        // A panic anywhere in the loop (model, DSP, device glue) must still reach the epilogue
        // below; otherwise `is_running` stays true with no worker, the rates stay published and
        // the UI never sees `stopped`. The cpal streams and sessions are dropped by the unwind.
        let app_for_loop = app_for_thread.clone();
        let generation_for_loop = generation.clone();
        let loop_state = run_state.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            run_voice_changer_loop(
                app_for_loop,
                generation_for_loop,
                rx,
                loop_state,
                input_device_name,
                output_device_name,
                monitor_output_device,
            )
        }))
        .map_err(|payload| {
            WorkerError::new(
                "worker_panic",
                format!("worker panicked: {}; voice changer stopped", panic_message(payload.as_ref())),
            )
        });
        let result = result.and_then(|r| r);

        if let Err(e) = &result {
            tracing::error!("voice changer: {e}");
        }

        // Epilogue for Ok, Err and a caught panic alike (an early `return Err` after the rates
        // were published must not leave them non-zero while stopped). A superseded worker must
        // not clear the flags, zero the rates/metrics its successor published (zeroed rates make
        // every TTS/aux push fail), show its error or report "stopped".
        if generation.is_current() {
            // Every self-stop reaches the user once, under its own code (a missing device, a
            // model or sidecar that did not load, a crash), before `stopped`.
            if let Err(e) = &result {
                emit_vc_error(&app_for_thread, e.code, e.message.clone());
            }
            run_state.reset_run_outputs();
            // A worker that stopped itself (load error, device error, panic) leaves `enabled`
            // false too, as `stop_voice_changer` does, so `get_voice_changer_params` (e.g. after a
            // webview reload) does not report a stopped voice changer as switched on.
            *lock(&run_state.enabled) = false;
            run_state.is_running.store(false, Ordering::Release);
            crate::services::resources::set_vc_running(false);
            let _ = app_for_thread.emit("voice_changer:stopped", ());
        }
    });
    let join = match spawned {
        Ok(j) => j,
        Err(e) => {
            if state.run_generation.load(Ordering::SeqCst) == my_generation {
                state.is_running.store(false, Ordering::Release);
                *lock(&state.enabled) = false;
                crate::services::resources::set_vc_running(false);
            }
            return Err(format!("could not start the voice-changer thread: {e}"));
        }
    };
    *lock(&state.worker) = Some(WorkerHandle { join, done: done_rx });

    let _ = app.emit("voice_changer:started", ());
    Ok(())
}

#[command]
pub(super) fn stop_voice_changer<R: Runtime>(_app: AppHandle<R>, state: State<'_, VoiceChangerState>) -> Result<(), String> {
    *lock(&state.enabled) = false;
    state.is_running.store(false, Ordering::Release);
    crate::services::resources::set_vc_running(false);
    if let Some(sender) = lock(&state.stop_sender).take() {
        let _ = sender.send(());
    }
    // Wake a parked worker so it stops now rather than at its next capture callback.
    if let Some(w) = lock(&state.worker).as_ref() {
        w.join.thread().unpark();
    }
    state.reset_run_outputs();
    Ok(())
}

#[command]
pub(super) fn is_voice_changer_running(state: State<'_, VoiceChangerState>) -> bool {
    state.is_running.load(Ordering::Acquire)
}

#[derive(Serialize, Clone, Debug)]
pub struct VoiceChangerModelSlot {
    pub id: String,
    pub path: String,
    pub has_manifest: bool,
}

#[command]
pub(super) fn list_voice_changer_model_slots<R: Runtime>(
    app: AppHandle<R>,
) -> Result<Vec<VoiceChangerModelSlot>, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("voice-changer")
        .join("models");
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let id = entry.file_name().to_string_lossy().to_string();
        let has_manifest = path.join("manifest.json").is_file();
        out.push(VoiceChangerModelSlot {
            id,
            path: path.to_string_lossy().to_string(),
            has_manifest,
        });
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}
