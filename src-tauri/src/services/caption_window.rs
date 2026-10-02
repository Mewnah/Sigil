//! `caption-window` plugin: the always-on-top, click-through caption window (P6-B).
//!
//! The window has the label [`CAPTIONS_WINDOW_LABEL`] and loads the app at the early route
//! [`CAPTIONS_ROUTE`] (`src/overlay/**`, mounted by `src/index.tsx` before any `ApiServer` /
//! `ApiClient` starts: the caption window never runs `ApiServer.init`, never opens the PeerJS
//! server and never writes the settings file). Its own IPC rights are `capabilities/captions.json`
//! (listening to events and dragging itself, nothing else); this plugin's commands are granted to
//! the main window only (`capabilities/caption-window.json`).
//!
//! - It is undecorated, transparent (only the caption box is drawn), skips the taskbar and never
//!   takes the keyboard focus, so it can sit over a game. Click-through
//!   (`set_ignore_cursor_events`) is on by default; with it off the page shows a frame to drag, and
//!   the window edges resize it. Users can also pin it in XSOverlay or OVR Toolkit, which capture
//!   windows by title ([`WINDOW_TITLE`]).
//! - Captions reach it as Tauri events the main window emits to this label (`caption_window:frame`,
//!   `src/overlay/frame.ts`), re-sent periodically while it is open, so a page that loads late or
//!   reloads catches up within a second.
//! - main.rs closes every secondary window when the main window is destroyed, so Sigil never keeps
//!   running with only the caption window open. Position and size persist through
//!   `tauri-plugin-window-state` (per label; this plugin restores only those two, never the
//!   visibility or the focus); the first open places it at the bottom centre of the primary
//!   monitor's work area. Opening it never takes the keyboard focus from a game: if anything
//!   focuses it, the focus goes back to the window that had it.
//! - Settings: the root key `captionWindow` (`src/core/services/overlay/schema.ts`).
//!
//! # Contract (frozen by P6-S0; structs may gain fields; change only through INTEGRATE)
//! - IPC (`plugin-commands/caption-window.txt`; wire camelCase; TS mirror `RustCaptionWindow*` in
//!   `src/types.ts`):
//!   - `caption_window_open { options: CaptionWindowOptions } -> CaptionWindowStatus`: create the
//!     window (or apply `options` to the open one) and show it. A window that cannot be created is
//!     reported in `error` (the command still succeeds).
//!   - `caption_window_set_options { options } -> CaptionWindowStatus`: always-on-top and
//!     click-through of the open window (stored for the next open when it is closed).
//!   - `caption_window_close -> ()` (Ok when it is not open; resolves once the window is gone, so an
//!     open right after it creates a new window), `caption_window_status -> CaptionWindowStatus`.
//!   - P6-B addition: `caption_window_reset_position -> CaptionWindowStatus`: move the open window
//!     back to its default place and size (a window lost off-screen, or behind the taskbar).
//! - Event [`STATE_EVENT`] `"caption_window:state"` (payload [`CaptionWindowStatus`]) whenever the
//!   window opens, closes (also when the user closes it) or changes options.
//! - P6-B addition: [`CaptionWindowStatus::closed_externally`] is true on the event of a window
//!   that closed without `caption_window_close` (Alt+F4, the OS), so the service can turn the
//!   setting off instead of reopening it at the next start.

use std::path::PathBuf;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Emitter, LogicalPosition, LogicalSize, Manager, Runtime, WebviewUrl, WebviewWindow, WebviewWindowBuilder, WindowEvent,
};
use tauri_plugin_window_state::{StateFlags, WindowExt as _};

/// Plugin name (`plugin:caption-window|...`, capability `caption-window:default`).
pub const PLUGIN_NAME: &str = "caption-window";
/// Window label of the caption window (also the `windows` entry of `capabilities/captions.json`).
pub const CAPTIONS_WINDOW_LABEL: &str = "captions";
/// App route the caption window loads (handled by `src/index.tsx` before the app boots).
pub const CAPTIONS_ROUTE: &str = "captions";
/// Event emitted on every state change (payload [`CaptionWindowStatus`]).
pub const STATE_EVENT: &str = "caption_window:state";
/// Window title (what XSOverlay / OVR Toolkit / OBS window capture list).
pub const WINDOW_TITLE: &str = "Sigil captions";

/// Default size (logical pixels) and distance from the bottom of the work area.
const DEFAULT_SIZE: (f64, f64) = (760.0, 200.0);
const MIN_SIZE: (f64, f64) = (160.0, 48.0);
const BOTTOM_MARGIN: f64 = 72.0;
/// How long `caption_window_close` waits for the window to be gone.
const CLOSE_WAIT: Duration = Duration::from_secs(2);

/// Window behaviour (TS `RustCaptionWindowOptions`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CaptionWindowOptions {
    /// Keep the window above other windows (games in borderless mode included).
    pub always_on_top: bool,
    /// Let mouse clicks pass through the window (`set_ignore_cursor_events`); off to move or resize it.
    pub click_through: bool,
}

impl CaptionWindowOptions {
    const DEFAULT: Self = Self {
        always_on_top: true,
        click_through: true,
    };
}

impl Default for CaptionWindowOptions {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Result of every command and payload of [`STATE_EVENT`] (TS `RustCaptionWindowStatus`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptionWindowStatus {
    pub open: bool,
    pub options: CaptionWindowOptions,
    /// Why the last open failed, if it did.
    pub error: Option<String>,
    /// P6-B addition: only on the close event of a window closed without `caption_window_close`
    /// (Alt+F4, the OS).
    #[serde(default)]
    pub closed_externally: bool,
}

/// Options for the window (applied to the open one, kept for the next open), the last error, and
/// whether a close in progress was asked for by `caption_window_close`.
struct Shared {
    options: CaptionWindowOptions,
    error: Option<String>,
    close_requested: bool,
}

static SHARED: Mutex<Shared> = Mutex::new(Shared {
    options: CaptionWindowOptions::DEFAULT,
    error: None,
    close_requested: false,
});

/// Serializes open / close / options so a quick off-on never finds the closing window (an async
/// lock: `caption_window_close` holds it until the window is gone).
static OPEN_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn shared() -> std::sync::MutexGuard<'static, Shared> {
    SHARED.lock().unwrap_or_else(PoisonError::into_inner)
}

fn status_with(open: bool) -> CaptionWindowStatus {
    let s = shared();
    CaptionWindowStatus {
        open,
        options: s.options.clone(),
        error: s.error.clone(),
        closed_externally: false,
    }
}

fn window<R: Runtime>(app: &AppHandle<R>) -> Option<WebviewWindow<R>> {
    app.get_webview_window(CAPTIONS_WINDOW_LABEL)
}

fn status<R: Runtime>(app: &AppHandle<R>) -> CaptionWindowStatus {
    status_with(window(app).is_some())
}

fn publish<R: Runtime>(app: &AppHandle<R>, status: &CaptionWindowStatus) {
    let _ = app.emit(STATE_EVENT, status.clone());
}

fn apply_options<R: Runtime>(window: &WebviewWindow<R>, options: &CaptionWindowOptions) -> Result<(), String> {
    window
        .set_always_on_top(options.always_on_top)
        .map_err(|e| format!("always on top: {e}"))?;
    window
        .set_ignore_cursor_events(options.click_through)
        .map_err(|e| format!("click-through: {e}"))
}

/// Default top-left corner (logical): bottom centre of the primary monitor's work area.
fn default_position<R: Runtime>(app: &AppHandle<R>) -> Option<LogicalPosition<f64>> {
    let monitor = app.primary_monitor().ok().flatten()?;
    let scale = monitor.scale_factor();
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    let area = monitor.work_area();
    let (x, y) = (f64::from(area.position.x) / scale, f64::from(area.position.y) / scale);
    let (w, h) = (f64::from(area.size.width) / scale, f64::from(area.size.height) / scale);
    Some(LogicalPosition::new(
        x + ((w - DEFAULT_SIZE.0) / 2.0).max(0.0),
        y + (h - DEFAULT_SIZE.1 - BOTTOM_MARGIN).max(0.0),
    ))
}

/// The foreground window (raw `HWND`, 0 for none).
#[cfg(windows)]
fn foreground_window() -> isize {
    // SAFETY: no arguments; returns a handle or null.
    unsafe { windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow() }.0 as isize
}

/// After the window-state plugin's own restore (queued on the main thread before this): restore
/// only the saved place and size, and if the new window took the foreground (the plugin focuses
/// every window it restores) hand it back to the window that had it, a game included.
fn settle<R: Runtime>(window: &WebviewWindow<R>, previous_foreground: isize) {
    if let Err(e) = window.restore_state(StateFlags::POSITION | StateFlags::SIZE) {
        tracing::debug!("caption window: no saved place restored: {e}");
    }
    #[cfg(windows)]
    {
        use windows::Win32::Foundation::HWND;
        use windows::Win32::UI::WindowsAndMessaging::SetForegroundWindow;
        let Ok(own) = window.hwnd() else { return };
        let own = own.0 as isize;
        if previous_foreground != 0 && previous_foreground != own && foreground_window() == own {
            // SAFETY: a handle read from GetForegroundWindow; Windows refuses a stale one.
            let _ = unsafe { SetForegroundWindow(HWND(previous_foreground as *mut std::ffi::c_void)) };
        }
    }
    #[cfg(not(windows))]
    let _ = previous_foreground;
}

/// Wait until the caption window is gone (true) or [`CLOSE_WAIT`] passed (false).
async fn wait_closed<R: Runtime>(app: &AppHandle<R>) -> bool {
    let deadline = Instant::now() + CLOSE_WAIT;
    while window(app).is_some() {
        if Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(15)).await;
    }
    true
}

fn build_window<R: Runtime>(app: &AppHandle<R>, options: &CaptionWindowOptions) -> Result<WebviewWindow<R>, String> {
    #[cfg(windows)]
    let previous_foreground = foreground_window();
    #[cfg(not(windows))]
    let previous_foreground = 0isize;
    let mut builder = WebviewWindowBuilder::new(app, CAPTIONS_WINDOW_LABEL, WebviewUrl::App(PathBuf::from(CAPTIONS_ROUTE)))
        .title(WINDOW_TITLE)
        .inner_size(DEFAULT_SIZE.0, DEFAULT_SIZE.1)
        .min_inner_size(MIN_SIZE.0, MIN_SIZE.1)
        .decorations(false)
        .transparent(true)
        .shadow(false)
        .resizable(true)
        .maximizable(false)
        .minimizable(false)
        .skip_taskbar(true)
        .always_on_top(options.always_on_top)
        // Never steals the keyboard focus from a game; dragging and resizing still work.
        .focused(false)
        .focusable(false)
        .zoom_hotkeys_enabled(false)
        // Same origin as the main window (tauri.conf.json `useHttpsScheme`).
        .use_https_scheme(true)
        .visible(true);
    // `settle` restores a saved place over this.
    if let Some(pos) = default_position(app) {
        builder = builder.position(pos.x, pos.y);
    }
    let window = builder
        .build()
        .map_err(|e| format!("cannot create the caption window: {e}"))?;
    let handle = app.clone();
    window.on_window_event(move |event| {
        if matches!(event, WindowEvent::Destroyed) {
            let requested = std::mem::take(&mut shared().close_requested);
            tracing::info!(requested, "caption window closed");
            publish(
                &handle,
                &CaptionWindowStatus {
                    closed_externally: !requested,
                    ..status_with(false)
                },
            );
        }
    });
    // Queued behind the window-state plugin's restore (both run on the main thread, in order).
    let settled = window.clone();
    if let Err(e) = window.run_on_main_thread(move || settle(&settled, previous_foreground)) {
        tracing::debug!("caption window: cannot settle the new window: {e}");
    }
    Ok(window)
}

#[command]
async fn caption_window_open<R: Runtime>(app: AppHandle<R>, options: CaptionWindowOptions) -> Result<CaptionWindowStatus, String> {
    let _serial = OPEN_LOCK.lock().await;
    shared().options = options.clone();
    let result = match window(&app) {
        Some(w) => apply_options(&w, &options).and_then(|()| {
            // Only a hidden window is shown (showing can activate it).
            if w.is_visible().unwrap_or(true) {
                Ok(())
            } else {
                w.show().map_err(|e| e.to_string())
            }
        }),
        None => build_window(&app, &options).and_then(|w| {
            tracing::info!("caption window opened");
            apply_options(&w, &options)
        }),
    };
    if let Err(e) = &result {
        tracing::warn!("caption window: {e}");
    }
    shared().error = result.err();
    let s = status(&app);
    publish(&app, &s);
    Ok(s)
}

#[command]
async fn caption_window_set_options<R: Runtime>(app: AppHandle<R>, options: CaptionWindowOptions) -> Result<CaptionWindowStatus, String> {
    let _serial = OPEN_LOCK.lock().await;
    shared().options = options.clone();
    if let Some(w) = window(&app) {
        let result = apply_options(&w, &options);
        shared().error = result.err();
    }
    let s = status(&app);
    publish(&app, &s);
    Ok(s)
}

#[command]
async fn caption_window_close<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    let _serial = OPEN_LOCK.lock().await;
    shared().error = None;
    let Some(w) = window(&app) else { return Ok(()) };
    shared().close_requested = true;
    // `close` (not `destroy`) so window-state saves the place and size.
    if let Err(e) = w.close() {
        shared().close_requested = false;
        return Err(e.to_string());
    }
    if !wait_closed(&app).await {
        tracing::warn!("caption window: still open {CLOSE_WAIT:?} after close; destroying it");
        w.destroy().map_err(|e| e.to_string())?;
        if !wait_closed(&app).await {
            return Err("the caption window did not close".into());
        }
    }
    Ok(())
}

#[command]
async fn caption_window_status<R: Runtime>(app: AppHandle<R>) -> Result<CaptionWindowStatus, String> {
    Ok(status(&app))
}

#[command]
async fn caption_window_reset_position<R: Runtime>(app: AppHandle<R>) -> Result<CaptionWindowStatus, String> {
    let _serial = OPEN_LOCK.lock().await;
    if let Some(w) = window(&app) {
        w.set_size(LogicalSize::new(DEFAULT_SIZE.0, DEFAULT_SIZE.1))
            .map_err(|e| e.to_string())?;
        if let Some(pos) = default_position(&app) {
            w.set_position(pos).map_err(|e| e.to_string())?;
        }
    }
    Ok(status(&app))
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new(PLUGIN_NAME)
        .invoke_handler(tauri::generate_handler![
            caption_window_open,
            caption_window_set_options,
            caption_window_close,
            caption_window_status,
            caption_window_reset_position,
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_format_is_camel_case() {
        let status = CaptionWindowStatus {
            open: true,
            options: CaptionWindowOptions {
                always_on_top: false,
                click_through: true,
            },
            error: Some("x".into()),
            closed_externally: false,
        };
        assert_eq!(
            serde_json::to_value(&status).unwrap(),
            serde_json::json!({
                "open": true,
                "options": { "alwaysOnTop": false, "clickThrough": true },
                "error": "x",
                "closedExternally": false
            })
        );
        // An older payload without the P6-B field still reads.
        let old: CaptionWindowStatus = serde_json::from_value(serde_json::json!({ "open": false, "options": {}, "error": null })).unwrap();
        assert!(!old.closed_externally);
        // Missing fields default to on (the safe overlay behaviour).
        let options: CaptionWindowOptions = serde_json::from_value(serde_json::json!({ "alwaysOnTop": false })).unwrap();
        assert_eq!(
            options,
            CaptionWindowOptions {
                always_on_top: false,
                click_through: true
            }
        );
        assert_eq!(CaptionWindowOptions::default(), CaptionWindowOptions::DEFAULT);
    }

    #[test]
    fn route_and_label_match_the_frontend() {
        // src/index.tsx routes `/captions` and capabilities/captions.json names this label.
        let capability: serde_json::Value = serde_json::from_str(include_str!("../../capabilities/captions.json")).unwrap();
        assert_eq!(capability["windows"], serde_json::json!([CAPTIONS_WINDOW_LABEL]));
        let index = include_str!("../../../src/index.tsx");
        assert!(
            index.contains(&format!("\"/{CAPTIONS_ROUTE}\"")),
            "src/index.tsx must route /{CAPTIONS_ROUTE}"
        );
    }
}
