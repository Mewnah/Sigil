//! Voice-changer benchmark IPC (P3-F): the per-PC autotuner and the in-app live-record benchmark
//! ("Benchmark this PC" uses the `bench` plugin's `run_benchmark`). Payloads are snake_case,
//! arguments camelCase.
//!
//! # Frozen by P3-S0 (command names, arguments and payloads; P3-F implements)
//! Commands (also in `plugin-commands/voice-changer.txt` and `ipc.ts` `VC_COMMANDS`):
//! - `voice_changer_autotune { request: AutotuneRequest } -> AutotuneResult` (progress on
//!   `bench::autotune::AUTOTUNE_PROGRESS_EVENT`; the result is stored per GPU and driver; the UI
//!   applies its `params` with `set_voice_changer_params`)
//! - `voice_changer_cancel_autotune -> ()`
//! - `voice_changer_get_autotune -> AutotuneResult | null` (the stored result for this PC)
//! - `voice_changer_live_record_start { request: LiveRecordRequest } -> ()`
//! - `voice_changer_live_record_stop -> LiveRecordSummary`
//!
//! # Behaviour (P3-F)
//! - The autotuner measures what this PC can do, so nothing may compete with it: it refuses to
//!   start while the voice changer runs, claims the measurement slot it shares with the in-app
//!   benchmark in one atomic step (a benchmark and a tune never both start), and a watcher stops the
//!   tune with an error, storing nothing, when the voice changer starts meanwhile.
//! - It fills the voice and base-model paths, device block and preset from the voice changer's
//!   params when the request leaves them out, and `conditions` (everything else the voice changer
//!   runs with: crossfade, cushion, budget, worker priority, retrieval, voice controls, F0 range,
//!   denoiser, lookahead, levers, the last device format) so every candidate is measured as the
//!   user runs it. It runs on its own thread, emits progress (at most 4 per second, plus the
//!   `done` / `failed` stage), writes a benchmark report folder (`app_data/bench/results/
//!   autotune-<ms>/`, `report_dir` of the result) and stores the result for this GPU and driver.
//! - `voice_changer_get_autotune` fills the result's `stale` list: what changed since the tune
//!   (`autotune::stale_reasons`), so a tune for other settings is never shown as current.
//! - Live recording needs the voice changer running; the recorder samples its counters besides the
//!   timeline and writes `app_data/bench/results/live-<ms>/`. Stopping joins the recorder and
//!   writes its report off the main thread.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use tauri::{command, AppHandle, Emitter, Manager, Runtime, State};

use super::devices::{resolve_input, resolve_output, DeviceRole};
use super::state::{lock, VoiceChangerState};
use crate::services::bench::autotune::{
    self, AutotuneProgress, AutotuneRequest, AutotuneResult, TuneConditions, AUTOTUNE_PROGRESS_EVENT,
};
use crate::services::bench::live_record::{self, LiveProbe, LiveRecordRequest, LiveRecordSummary, LiveSample};
use crate::services::bench::{BenchContext, Progress};

static AUTOTUNE_CANCEL: AtomicBool = AtomicBool::new(false);

/// How often the tune checks whether the voice changer started.
const VC_WATCH_PERIOD: Duration = Duration::from_millis(100);

fn app_data<R: Runtime>(app: &AppHandle<R>) -> Result<std::path::PathBuf, String> {
    crate::services::diagnostics::app_data_dir()
        .or_else(|| app.path().app_data_dir().ok())
        .ok_or_else(|| "no app data folder".to_string())
}

fn some_path(s: &str) -> Option<String> {
    let t = s.trim();
    (!t.is_empty()).then(|| t.to_string())
}

/// The voice changer's settings as a tune measures them, with the device format of the running
/// worker (0 = not running: [`fill_device_format`] finds it).
fn current_conditions(state: &VoiceChangerState) -> TuneConditions {
    let p = state.snapshot_params();
    let channels = u16::try_from(state.output_channels.load(Ordering::Relaxed)).unwrap_or(0);
    TuneConditions::from_params(
        &p,
        state.capture_sample_rate.load(Ordering::Relaxed),
        state.output_sample_rate.load(Ordering::Relaxed),
        channels,
    )
}

/// Fill unknown device rates with what the voice changer's saved devices open with now (their
/// default formats, as its start resolves them); a device that is not found stays unknown (the
/// voice changer's 48 kHz default). Enumerates devices: never on the main thread.
fn fill_device_format(c: &mut TuneConditions, state: &VoiceChangerState) {
    if c.device_rate > 0 && c.output_rate > 0 {
        return;
    }
    let host = cpal::default_host();
    if c.device_rate == 0 {
        let saved = lock(&state.input_device).clone();
        c.device_rate = resolve_input(&host, &saved).map_or(0, |t| t.rate);
    }
    if c.output_rate == 0 {
        let saved = lock(&state.output_device).clone();
        if let Ok((_, _, config)) = resolve_output(&host, DeviceRole::Output, &saved) {
            c.output_rate = config.sample_rate().0;
            c.output_channels = config.channels();
        }
    }
}

/// The request with the voice changer's own voice, models, device block, preset and conditions
/// filled in.
fn fill_request(mut req: AutotuneRequest, state: &VoiceChangerState) -> Result<AutotuneRequest, String> {
    let p = state.snapshot_params();
    if req.slot_dir.as_deref().map(str::trim).is_none_or(str::is_empty) {
        req.slot_dir = Some(some_path(&p.rvc_slot_dir).ok_or("no voice selected: pick a voice before tuning")?);
    }
    req.hubert = req.hubert.or_else(|| some_path(&p.rvc_hubert_path));
    req.fcpe = req.fcpe.or_else(|| some_path(&p.rvc_fcpe_path));
    req.rmvpe = req.rmvpe.or_else(|| some_path(&p.rvc_rmvpe_path));
    req.chunk_frames = req.chunk_frames.or(Some(p.chunk_frames));
    req.preset = req.preset.or_else(|| some_path(&p.rvc_preset));
    if req.conditions.is_none() {
        req.conditions = Some(current_conditions(state));
    }
    Ok(req)
}

/// Ends the watcher when the tune returns or unwinds (a panicking tune must not leave the scope
/// waiting for a watcher that never stops).
struct StopWatcher<'a> {
    done: &'a AtomicBool,
    watcher: std::thread::Thread,
}

impl Drop for StopWatcher<'_> {
    fn drop(&mut self) {
        self.done.store(true, Ordering::SeqCst);
        self.watcher.unpark();
    }
}

/// Run `tune` while watching `vc_running`: when it turns on, `cancel` is set and the outcome is an
/// error (nothing measured afterwards may be trusted, nor stored).
fn run_watched<T>(vc_running: &AtomicBool, cancel: &AtomicBool, tune: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    let done = AtomicBool::new(false);
    let vc_started = AtomicBool::new(false);
    let result = std::thread::scope(|scope| {
        let watcher = scope.spawn(|| {
            while !done.load(Ordering::SeqCst) {
                if vc_running.load(Ordering::SeqCst) {
                    vc_started.store(true, Ordering::SeqCst);
                    cancel.store(true, Ordering::SeqCst);
                    break;
                }
                std::thread::park_timeout(VC_WATCH_PERIOD);
            }
        });
        let _stop = StopWatcher {
            done: &done,
            watcher: watcher.thread().clone(),
        };
        tune()
    });
    if vc_started.load(Ordering::SeqCst) {
        return Err("the voice changer started during tuning: the measurements competed with it, so nothing was stored; tune again with the voice changer stopped".into());
    }
    result
}

#[command]
pub(super) async fn voice_changer_autotune<R: Runtime>(app: AppHandle<R>, request: AutotuneRequest) -> Result<AutotuneResult, String> {
    let state = app.state::<VoiceChangerState>();
    if state.is_running.load(Ordering::SeqCst) {
        return Err("stop the voice changer before tuning: it would compete for the GPU and CPU the tuner measures".into());
    }
    // One atomic claim of the slot shared with the in-app benchmark.
    let guard = autotune::begin()?;
    AUTOTUNE_CANCEL.store(false, Ordering::SeqCst);
    let request = fill_request(request, &state)?;
    let data = app_data(&app)?;
    let vc = state.inner().clone();
    let emitter = app.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();
    std::thread::Builder::new()
        .name("sigil-autotune".into())
        .stack_size(crate::services::bench::BENCH_STACK_BYTES)
        .spawn(move || {
            let last = Mutex::new(Instant::now() - Duration::from_secs(1));
            let progress = |p: &Progress| {
                let mut t = last.lock().unwrap_or_else(PoisonError::into_inner);
                if p.fraction <= 0.0 || t.elapsed() >= Duration::from_millis(250) {
                    *t = Instant::now();
                    let _ = emitter.emit(
                        AUTOTUNE_PROGRESS_EVENT,
                        AutotuneProgress {
                            stage: if p.message.starts_with("verifying") { "verifying" } else { "sweeping" }.into(),
                            fraction: p.fraction,
                            candidate: p.message.split_once(": ").map(|(_, c)| c.to_string()),
                            message: p.message.clone(),
                        },
                    );
                }
            };
            let mut request = request;
            if let Some(c) = request.conditions.as_mut() {
                fill_device_format(c, &vc);
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_watched(&vc.is_running, &AUTOTUNE_CANCEL, || {
                    autotune::run_autotune(&request, &BenchContext { cancel: &AUTOTUNE_CANCEL, progress: &progress })
                })
            }))
            .unwrap_or_else(|_| Err("the autotuner failed unexpectedly".into()));
            let result = result.and_then(|mut r| {
                let dir = data.join("bench").join("results").join(format!("autotune-{}", r.created_unix_ms));
                match autotune::write_result_report(&r, &dir) {
                    Ok(()) => r.report_dir = Some(dir.display().to_string()),
                    Err(e) => r.notes.push(format!("report: {e}")),
                }
                autotune::store_result(&data, &r)?;
                Ok(r)
            });
            let (stage, message) = match &result {
                Ok(_) => ("done", "done".to_string()),
                Err(e) => ("failed", e.clone()),
            };
            let _ = emitter.emit(
                AUTOTUNE_PROGRESS_EVENT,
                AutotuneProgress {
                    stage: stage.into(),
                    fraction: 1.0,
                    candidate: None,
                    message,
                },
            );
            drop(guard);
            let _ = tx.send(result);
        })
        .map_err(|e| format!("cannot start the autotuner: {e}"))?;
    rx.await.map_err(|_| "the autotuner stopped unexpectedly".to_string())?
}

#[command]
pub(super) fn voice_changer_cancel_autotune() -> Result<(), String> {
    AUTOTUNE_CANCEL.store(true, Ordering::SeqCst);
    Ok(())
}

/// The stored tune for this PC with its `stale` list (adapter and device enumeration: on a blocking
/// worker, never on the main thread).
#[command]
pub(super) async fn voice_changer_get_autotune<R: Runtime>(app: AppHandle<R>) -> Result<Option<AutotuneResult>, String> {
    let data = app_data(&app)?;
    let state = app.state::<VoiceChangerState>().inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let mut r = autotune::stored_result(&data)?;
        let slot = state.snapshot_params().rvc_slot_dir;
        let mut now = current_conditions(&state);
        fill_device_format(&mut now, &state);
        r.stale = autotune::stale_reasons(&r, &now, &slot);
        Some(r)
    })
    .await
    .map_err(|e| format!("cannot read the stored tune: {e}"))
}

/// The running voice changer's counters and configuration (no file paths, no device names).
struct StateProbe(VoiceChangerState);

impl LiveProbe for StateProbe {
    fn sample(&self) -> LiveSample {
        let s = &self.0;
        LiveSample {
            running: s.is_running.load(Ordering::Relaxed),
            process_us: s.metrics_process_us.load(Ordering::Relaxed),
            process_p99_us: s.metrics_process_p99_us.load(Ordering::Relaxed),
            backlog_frames: s.metrics_backlog_frames.load(Ordering::Relaxed),
            dropped_frames: s.metrics_dropped_frames.load(Ordering::Relaxed),
            underruns: s.metrics_underruns.load(Ordering::Relaxed),
            capture_rate: s.capture_sample_rate.load(Ordering::Relaxed),
            output_rate: s.output_sample_rate.load(Ordering::Relaxed),
        }
    }

    fn describe(&self) -> serde_json::Value {
        let s = &self.0;
        let p = s.snapshot_params();
        let voice = std::path::Path::new(p.rvc_slot_dir.trim())
            .file_name()
            .map(|n| n.to_string_lossy().into_owned());
        let ep = lock(&s.run_ep).clone();
        serde_json::json!({
            "mode": p.mode,
            "voice": voice,
            "rvc_preset": p.rvc_preset,
            "rvc_execution_provider": p.rvc_execution_provider,
            "execution_provider_active": ep.as_ref().map(|e| e.active.clone()),
            "execution_provider_reason": ep.and_then(|e| e.reason),
            "pitch_extractor": p.pitch_extractor,
            "chunk_frames": p.chunk_frames,
            "block_ms": p.block_ms,
            "extra_context_ms": p.extra_context_ms,
            "crossfade_ms": p.crossfade_ms,
            "latency_preset": p.latency_preset,
            "output_prefill_ms": p.output_prefill_ms,
            "output_max_queue_ms": p.output_max_queue_ms,
            "latency_budget_ms": p.latency_budget_ms,
            "worker_priority": p.worker_priority,
            "precision": p.precision,
            "denoiser": p.denoiser,
            "levers": {
                "partial_decode": p.partial_decode, "anchored_noise": p.anchored_noise, "nsf_phase0": p.nsf_phase0,
                "f0_cache": p.f0_cache, "gpu_knn": p.gpu_knn, "io_binding": p.io_binding, "cuda_graphs": p.cuda_graphs,
                "gpu_priority": p.gpu_priority, "governor": p.governor,
            },
            "capture_rate": s.capture_sample_rate.load(Ordering::Relaxed),
            "capture_channels": s.capture_channels.load(Ordering::Relaxed),
            "output_rate": s.output_sample_rate.load(Ordering::Relaxed),
            "output_channels": s.output_channels.load(Ordering::Relaxed),
        })
    }
}

#[command]
pub(super) fn voice_changer_live_record_start<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, VoiceChangerState>,
    request: LiveRecordRequest,
) -> Result<(), String> {
    if !state.is_running.load(Ordering::SeqCst) {
        return Err("start the voice changer first: the recording measures it while you use it".into());
    }
    if live_record::is_recording() {
        return Err("a live recording is already running".into());
    }
    let dir = live_record::default_out_dir(&app_data(&app)?);
    live_record::start_with_probe(&request, &dir, Some(Box::new(StateProbe(state.inner().clone()))))
}

/// Joins the recorder and writes its report (CSV files, the analysis) on a blocking worker, never on
/// the main thread (a sync command would run there and freeze the window meanwhile).
#[command]
pub(super) async fn voice_changer_live_record_stop<R: Runtime>(
    _app: AppHandle<R>,
    _state: State<'_, VoiceChangerState>,
) -> Result<LiveRecordSummary, String> {
    tauri::async_runtime::spawn_blocking(live_record::stop)
        .await
        .map_err(|e| format!("the live recorder failed unexpectedly: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The request keeps what the UI sent and fills the rest from the voice changer's params.
    #[test]
    fn requests_are_filled_from_the_params() {
        let state = VoiceChangerState::new();
        let err = fill_request(AutotuneRequest::default(), &state).unwrap_err();
        assert!(err.contains("no voice selected"), "{err}");
        {
            let mut rvc = lock(&state.rvc);
            rvc.slot_dir = "C:/voices/a".into();
            rvc.hubert_path = "C:/base/hubert.onnx".into();
        }
        let r = fill_request(
            AutotuneRequest {
                fcpe: Some("C:/mine/fcpe.onnx".into()),
                ..AutotuneRequest::default()
            },
            &state,
        )
        .unwrap();
        assert_eq!(r.slot_dir.as_deref(), Some("C:/voices/a"));
        assert_eq!(r.hubert.as_deref(), Some("C:/base/hubert.onnx"));
        assert_eq!(r.fcpe.as_deref(), Some("C:/mine/fcpe.onnx"), "the request wins");
        assert!(r.rmvpe.is_none());
        assert!(r.chunk_frames.is_some() && r.preset.is_some());
        assert!(r.conditions.is_some(), "the voice changer's settings ride along");
        let d = StateProbe(state).describe();
        assert_eq!(d["voice"], "a", "the folder name only");
        assert!(!d.to_string().contains("C:/"), "no paths: {d}");
    }

    /// The tune measures with the user's crossfade, retrieval, priority, F0 range, denoiser and
    /// levers, not the defaults; a request that brings its own conditions keeps them.
    #[test]
    fn the_tuning_conditions_are_the_voice_changers() {
        use crate::services::neural_audio::vc::engine::options::{DenoiserKind, F0Range, Lever, LeverMode};
        use crate::services::rt_thread::WorkerPriority;
        let state = VoiceChangerState::new();
        {
            let mut rvc = lock(&state.rvc);
            rvc.slot_dir = "C:/voices/a".into();
            rvc.crossfade_ms = 10.0;
            rvc.index_rate = 0.75;
            rvc.f0_range = F0Range::Singing;
            rvc.denoiser = DenoiserKind::Gtcrn;
            rvc.lookahead_ms = 20.0;
            rvc.levers.set(Lever::GpuKnn, LeverMode::On);
            rvc.levers.set(Lever::IoBinding, LeverMode::Off);
        }
        {
            let mut h = lock(&state.hot_params);
            h.worker_priority = WorkerPriority::ProtectGame;
            h.latency_budget_ms = 500.0;
        }
        let r = fill_request(AutotuneRequest::default(), &state).unwrap();
        let c = r.conditions.as_ref().unwrap();
        assert_eq!(c.crossfade_ms, 10.0);
        assert_eq!(c.index_rate, 0.75);
        assert_eq!(c.worker_priority, "protect_game");
        assert_eq!(c.latency_budget_ms, 500.0);
        assert_eq!((c.f0_range.as_str(), c.denoiser.as_str()), ("singing", "gtcrn"));
        assert_eq!(c.lookahead_ms, 20.0);
        assert_eq!((c.levers["gpu_knn"].as_str(), c.levers["io_binding"].as_str()), ("on", "off"));
        assert_eq!((c.device_rate, c.output_rate), (0, 0), "not running: found when the tune starts");
        assert!(r.validate().is_ok(), "{:?}", r.validate());
        let own = TuneConditions {
            crossfade_ms: 30.0,
            ..TuneConditions::default()
        };
        let r = fill_request(
            AutotuneRequest {
                conditions: Some(own.clone()),
                ..AutotuneRequest::default()
            },
            &state,
        )
        .unwrap();
        assert_eq!(r.conditions, Some(own), "the request wins");
    }

    /// A voice changer that starts during the tune cancels it with an error; otherwise the tune's
    /// own outcome passes through.
    #[test]
    fn a_voice_changer_start_aborts_the_tune() {
        let running = AtomicBool::new(false);
        let cancel = AtomicBool::new(false);
        assert_eq!(run_watched(&running, &cancel, || Ok::<_, String>(7)), Ok(7));
        assert!(!cancel.load(Ordering::SeqCst));
        let r = run_watched(&running, &cancel, || {
            running.store(true, Ordering::SeqCst);
            // The tune notices the cancel flag the watcher sets, as `run_paced` does.
            let t0 = Instant::now();
            while !cancel.load(Ordering::SeqCst) {
                assert!(t0.elapsed() < Duration::from_secs(10), "the watcher never cancelled");
                std::thread::sleep(Duration::from_millis(5));
            }
            Err::<i32, String>("cancelled".into())
        });
        let e = r.unwrap_err();
        assert!(e.contains("the voice changer started during tuning"), "{e}");
        // A tune that panics still ends the watcher (the panic reaches the caller's catch_unwind).
        let idle = AtomicBool::new(false);
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_watched(&idle, &cancel, || -> Result<(), String> { panic!("boom") })
        }));
        assert!(caught.is_err());
    }
}
