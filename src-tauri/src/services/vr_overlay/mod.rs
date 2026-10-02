//! `vr-overlay` plugin: captions in a SteamVR overlay (P6-B).
//!
//! - `openvr_api.dll` is a runtime pack (`runtime/openvr`, `model-catalog/runtime-openvr.json`,
//!   Valve's OpenVR SDK v2.15.6), loaded lazily with `libloading` through hand-written minimal FFI
//!   ([`openvr`]). Nothing is load-time linked (an import library would stop Sigil starting on PCs
//!   without SteamVR). Developers can point `SIGIL_OPENVR_DIR` at any folder holding an
//!   `openvr_api.dll` (Steam ships one in its install folder) instead of installing the pack.
//! - Sigil initializes OpenVR as a *Background* application and only while SteamVR's
//!   `vrserver.exe` runs, so it never launches SteamVR: while SteamVR is not running the overlay
//!   waits (state `waiting`) and retries; it disconnects on `VREvent_Quit` ([`worker`]).
//! - Text is shaped with cosmic-text (system fonts, CJK per language, RTL) and rasterized with
//!   tiny-skia into an RGBA texture of fixed size ([`render`]), uploaded with `SetOverlayRaw` at
//!   most 10 Hz and only when it changed (`vr_overlay_set_text` is coalesced).
//! - Settings: `services.vrc.data.overlay` (`src/core/services/vrc/schema.ts`), UI
//!   `src/core/ui/inspector/vrc/OverlaySection.tsx`, TS driver `src/core/services/overlay/**`.
//!
//! # Contract (frozen by P6-S0; structs may gain fields; change only through INTEGRATE)
//! - IPC (`plugin-commands/vr-overlay.txt`; wire camelCase; TS mirror `RustVrOverlay*` in
//!   `src/types.ts`):
//!   - `vr_overlay_start { config: VrOverlayConfig } -> VrOverlayStatus`: start (or reconfigure)
//!     the overlay; returns at once (`waiting` while SteamVR is not running, `unavailable` without
//!     the runtime pack).
//!   - `vr_overlay_set_config { config } -> VrOverlayStatus`, `vr_overlay_stop -> ()`,
//!     `vr_overlay_status -> VrOverlayStatus`.
//!   - `vr_overlay_set_text { caption: VrOverlayCaption } -> ()`: the lines to show now (latest
//!     wins; rendered at most 10 Hz).
//!   - P6-B addition: `vr_overlay_recenter -> ()`: place a world-attached overlay in front of the
//!     user again (no-op otherwise).
//! - Event [`STATE_EVENT`] `"vr_overlay:state"` (payload [`VrOverlayStatus`]).
//! - P6-B addition: [`VrOverlayStatus::reason`] says why the overlay waits or failed.
//! - Rust: [`shutdown`]`()`: synchronous and idempotent; main.rs calls it on `RunEvent::Exit`.

mod color;
mod openvr;
mod placement;
mod render;
mod settings;
mod types;
mod worker;

pub use types::*;

use std::path::PathBuf;
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Emitter, Runtime,
};

use worker::{Command, Rasterizer, Timing, Worker};

/// Plugin name (`plugin:vr-overlay|...`, capability `vr-overlay:default`).
pub const PLUGIN_NAME: &str = "vr-overlay";
/// Event emitted on every state change (payload [`VrOverlayStatus`]).
pub const STATE_EVENT: &str = "vr_overlay:state";
/// Catalog id of the OpenVR runtime pack (`openvr_api.dll`).
pub const RUNTIME_ID: &str = "runtime/openvr";
/// Development override: a folder with `openvr_api.dll` used instead of the runtime pack.
pub const DEV_DIR_ENV: &str = "SIGIL_OPENVR_DIR";
/// Most lines accepted by `vr_overlay_set_text` (the newest are kept).
const MAX_LINES_IN: usize = 32;
/// Longest text accepted per line (characters; the newest are kept).
const MAX_TEXT_IN: usize = 4000;

// ---------------------------------------------------------------------------------------------
// Controller: one worker at a time, epoch-checked status.
// ---------------------------------------------------------------------------------------------

/// The running worker (if any).
static WORKER: Mutex<Option<Worker>> = Mutex::new(None);

/// Latest published status and the current worker generation (the "epoch"): a worker publishes
/// only while its generation is current, and stop / start bump it under this lock.
struct Published {
    generation: u64,
    status: VrOverlayStatus,
}

static PUBLISHED: Mutex<Published> = Mutex::new(Published {
    generation: 0,
    status: VrOverlayStatus {
        state: VrOverlayState::Off,
        runtime_installed: false,
        error: None,
        reason: None,
    },
});

/// The folder holding `openvr_api.dll`: `SIGIL_OPENVR_DIR` (development) or the runtime pack.
fn locate_runtime() -> Option<PathBuf> {
    let has_dll = |d: &PathBuf| d.join(openvr::DLL_NAME).is_file();
    std::env::var_os(DEV_DIR_ENV)
        .map(PathBuf::from)
        .filter(has_dll)
        .or_else(|| crate::services::model_store::installed_path(RUNTIME_ID).filter(has_dll))
}

fn off_status() -> VrOverlayStatus {
    VrOverlayStatus {
        state: VrOverlayState::Off,
        runtime_installed: locate_runtime().is_some(),
        error: None,
        reason: None,
    }
}

/// Start a new generation and record `status` as its first status. Returns the generation.
fn begin_generation(status: VrOverlayStatus) -> u64 {
    let mut p = PUBLISHED.lock().unwrap_or_else(PoisonError::into_inner);
    p.generation += 1;
    p.status = status;
    p.generation
}

fn current_status() -> VrOverlayStatus {
    PUBLISHED
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .status
        .clone()
}

/// Record `status` and `emit` it if `generation` is still current (true when it was). Emitted
/// under the lock, so a stop cannot slip between the check and the emit.
fn publish_if_current(generation: u64, status: VrOverlayStatus, emit: impl FnOnce(&VrOverlayStatus)) -> bool {
    let mut p = PUBLISHED.lock().unwrap_or_else(PoisonError::into_inner);
    if p.generation != generation {
        return false;
    }
    p.status = status;
    emit(&p.status);
    true
}

/// The status sink of generation `generation`: records and emits while it is current.
fn status_sink<R: Runtime>(app: AppHandle<R>, generation: u64) -> worker::StatusSink {
    std::sync::Arc::new(move |status: VrOverlayStatus| {
        publish_if_current(generation, status, |s| {
            let _ = app.emit(STATE_EVENT, s.clone());
        });
    })
}

/// The process-wide text renderer (fonts load on the worker thread, once).
struct SharedRenderer;

impl Rasterizer for SharedRenderer {
    fn render(&mut self, caption: &VrOverlayCaption, look: &settings::Look) -> render::Texture {
        render::with_renderer(|r| r.render(caption, look))
    }
}

/// Keep at most [`MAX_LINES_IN`] lines of at most [`MAX_TEXT_IN`] characters (the newest).
fn bound_caption(mut caption: VrOverlayCaption) -> VrOverlayCaption {
    let extra = caption.lines.len().saturating_sub(MAX_LINES_IN);
    caption.lines.drain(..extra);
    for line in &mut caption.lines {
        let count = line.text.chars().count();
        if count > MAX_TEXT_IN {
            line.text = line.text.chars().skip(count - MAX_TEXT_IN).collect();
        }
        if line.source.len() > 64 {
            line.source.clear();
        }
        if line.lang.as_ref().is_some_and(|l| l.len() > 35) {
            line.lang = None;
        }
    }
    caption
}

fn take_worker() -> Option<Worker> {
    WORKER.lock().unwrap_or_else(PoisonError::into_inner).take()
}

/// Stop the worker (if any), waiting up to `timeout` for it to close the session.
fn stop_worker(timeout: Duration) {
    // A new generation first: the old worker's late statuses are dropped.
    begin_generation(off_status());
    if let Some(worker) = take_worker() {
        worker.stop_and_join(timeout);
    }
}

/// Stop the overlay and release OpenVR (synchronous, idempotent; `RunEvent::Exit`).
pub fn shutdown() {
    stop_worker(Duration::from_secs(2));
}

#[command]
async fn vr_overlay_start<R: Runtime>(app: AppHandle<R>, config: VrOverlayConfig) -> Result<VrOverlayStatus, String> {
    {
        let guard = WORKER.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(worker) = guard.as_ref().filter(|w| w.is_alive()) {
            worker.send(Command::Config(Box::new(config)));
            return Ok(current_status());
        }
    }
    // Not running (or the thread died): start a fresh worker.
    if let Some(dead) = take_worker() {
        tauri::async_runtime::spawn_blocking(move || dead.stop_and_join(Duration::from_secs(3)))
            .await
            .map_err(|e| e.to_string())?;
    }
    let installed = locate_runtime().is_some();
    let first = VrOverlayStatus {
        state: if installed {
            VrOverlayState::Waiting
        } else {
            VrOverlayState::Unavailable
        },
        runtime_installed: installed,
        error: None,
        reason: (!installed).then_some(VrOverlayReason::RuntimeMissing),
    };
    let mut guard = WORKER.lock().unwrap_or_else(PoisonError::into_inner);
    if guard.as_ref().is_some_and(Worker::is_alive) {
        // A concurrent start won the race: reconfigure that one.
        if let Some(worker) = guard.as_ref() {
            worker.send(Command::Config(Box::new(config)));
        }
        return Ok(current_status());
    }
    let generation = begin_generation(first.clone());
    let _ = app.emit(STATE_EVENT, first.clone());
    let backend = openvr::OpenVrBackend::new(Box::new(locate_runtime));
    let sink = status_sink(app, generation);
    let worker = Worker::spawn(Box::new(backend), Box::new(SharedRenderer), &config, Timing::PRODUCTION, sink)
        .map_err(|e| format!("cannot start the SteamVR overlay: {e}"))?;
    *guard = Some(worker);
    drop(guard);
    Ok(current_status())
}

#[command]
async fn vr_overlay_set_config(config: VrOverlayConfig) -> Result<VrOverlayStatus, String> {
    let guard = WORKER.lock().unwrap_or_else(PoisonError::into_inner);
    match guard.as_ref() {
        Some(worker) => {
            worker.send(Command::Config(Box::new(config)));
            Ok(current_status())
        }
        None => Ok(off_status()),
    }
}

#[command]
async fn vr_overlay_stop<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(|| stop_worker(Duration::from_secs(3)))
        .await
        .map_err(|e| e.to_string())?;
    let _ = app.emit(STATE_EVENT, current_status());
    Ok(())
}

#[command]
async fn vr_overlay_status() -> Result<VrOverlayStatus, String> {
    let running = WORKER
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .is_some();
    Ok(if running { current_status() } else { off_status() })
}

#[command]
async fn vr_overlay_set_text(caption: VrOverlayCaption) -> Result<(), String> {
    let guard = WORKER.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(worker) = guard.as_ref() {
        worker.send(Command::Text(bound_caption(caption)));
    }
    Ok(())
}

#[command]
async fn vr_overlay_recenter() -> Result<(), String> {
    let guard = WORKER.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(worker) = guard.as_ref() {
        worker.send(Command::Recenter);
    }
    Ok(())
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new(PLUGIN_NAME)
        .invoke_handler(tauri::generate_handler![
            vr_overlay_start,
            vr_overlay_set_config,
            vr_overlay_stop,
            vr_overlay_status,
            vr_overlay_set_text,
            vr_overlay_recenter,
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_format_is_camel_case() {
        let status = VrOverlayStatus {
            state: VrOverlayState::Waiting,
            runtime_installed: true,
            error: None,
            reason: Some(VrOverlayReason::SteamvrNotRunning),
        };
        let json = serde_json::to_value(&status).unwrap();
        assert_eq!(
            json,
            serde_json::json!({ "state": "waiting", "runtimeInstalled": true, "error": null, "reason": "steamvrNotRunning" })
        );
        let config: VrOverlayConfig = serde_json::from_value(serde_json::json!({
            "attach": "leftHand", "widthM": 0.4, "fontSizePx": 40, "textColor": "#fff"
        }))
        .unwrap();
        assert_eq!(config.attach, VrOverlayAttach::LeftHand);
        assert_eq!(config.width_m, 0.4);
        assert_eq!(config.font_size_px, 40);
        // Missing fields take their defaults.
        assert_eq!(config.distance_m, 1.2);
        let caption: VrOverlayCaption = serde_json::from_value(serde_json::json!({
            "lines": [{ "text": "hi", "kind": "interim", "source": "others.stt", "lang": "ja" }, { "text": "x" }]
        }))
        .unwrap();
        assert_eq!(caption.lines[0].kind, VrOverlayLineKind::Interim);
        assert_eq!(caption.lines[1].kind, VrOverlayLineKind::Final);
        assert_eq!(caption.lines[1].lang, None);
    }

    #[test]
    fn captions_are_bounded() {
        let lines = (0..40)
            .map(|i| VrOverlayLine {
                text: format!("{i}{}", "x".repeat(if i == 39 { 5000 } else { 1 })),
                ..VrOverlayLine::default()
            })
            .collect();
        let bounded = bound_caption(VrOverlayCaption { lines });
        assert_eq!(bounded.lines.len(), MAX_LINES_IN);
        assert!(bounded.lines[0].text.starts_with('8'));
        assert_eq!(bounded.lines.last().unwrap().text.chars().count(), MAX_TEXT_IN);
    }

    #[test]
    fn status_of_a_stale_generation_is_dropped() {
        let running = VrOverlayStatus {
            state: VrOverlayState::Running,
            ..VrOverlayStatus::default()
        };
        let gen_a = begin_generation(VrOverlayStatus {
            state: VrOverlayState::Waiting,
            ..VrOverlayStatus::default()
        });
        // A stop / restart bumps the generation: the old worker's late status is neither recorded
        // nor emitted.
        let gen_b = begin_generation(VrOverlayStatus {
            state: VrOverlayState::Off,
            ..VrOverlayStatus::default()
        });
        let mut emitted = Vec::new();
        assert!(!publish_if_current(gen_a, running.clone(), |s| emitted.push(s.clone())));
        assert!(emitted.is_empty());
        assert_eq!(current_status().state, VrOverlayState::Off);
        assert!(publish_if_current(gen_b, running.clone(), |s| emitted.push(s.clone())));
        assert_eq!(emitted, vec![running]);
        assert_eq!(current_status().state, VrOverlayState::Running);
    }
}
