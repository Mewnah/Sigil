//! Size of the main window: the first launch, and the fit of a restored window to its screen.
//!
//! The window in `tauri.conf.json` is a fixed 800 x 600 (logical px). On a 4K monitor at 100% Windows
//! scale that is a small box and the whole UI is small with it, so the very first launch (no saved state
//! for `main`) opens the window sized from the monitor it lands on instead:
//!
//! 1. `s` = the Auto "Interface size" of that monitor ([`auto_ui_scale`], the rule in
//!    `src/core/ui/uiScale.fixture.json`, which the TypeScript implementation passes case by case too).
//!    The page scales its rem sizes by the same `s`.
//! 2. The window is (960 x 720) x `s` logical px, at most 90 % of the monitor's work area, never below
//!    800 x 600 unless the work area is smaller, centred in the work area ([`first_window_placement`]).
//!
//! Only the host app's `main` window is touched. A saved state (the window-state plugin's file has an
//! entry for `main` that is not its all-default one) means the plugin restores the user's own size and
//! the first-launch step ([`init`]) does nothing; so do `SIGIL_START_MINIMIZED=1` runs
//! (`enabled == false`, both steps) and every other window (the captions window, which restores its
//! own place).
//!
//! # Fit after a restore
//!
//! A restored window is the user's, but two things can leave it wrong for the screen it comes back on,
//! so a second step ([`init_after_restore`]) looks at the window once the plugin has restored it:
//!
//! - **Too small for the interface size.** With Interface size Auto on, a 4K monitor at 100% runs the
//!   interface at 1.5, so a window saved at 800 x 600 (any profile that existed before Auto) holds only
//!   533 x 400 of the layout the page was designed for. The window is grown to 800 x 600 x `s` logical
//!   px, so that it holds at least that 800 x 600 again ([`fit_restored_window`]). A window that is
//!   already large enough, every window at s = 1 (nothing changed there since before Auto: a window the
//!   user made small stays small), and every profile that picked S, M, L or XL (Auto off) are left
//!   alone: the size the user picked is not Auto's to second-guess ([`interface_size_is_auto`] reads
//!   the choice from the persisted settings, as the front end will decide it at this launch).
//! - **Too big for the monitor.** The plugin saves and restores physical pixels and checks only that one
//!   corner is on some monitor, so a 1440 x 1080 window saved on a 4K monitor comes back taller than the
//!   work area of a 1080p laptop it is undocked to (or of the same monitor at a higher Windows scale).
//!   The window is cut to the work area of the monitor it is on, in either interface mode.
//!
//! The page applies the same 800 x 600 x `s` floor at run time to the size in effect, a manual one above 1
//! included (`windowGrowth` in `src/core/ui/uiScale.ts`; it also follows the window to another monitor).
//! For Auto the two agree on the numbers, so whichever runs first leaves nothing for the other. This step
//! runs first, before the page exists, and is the only one that cuts a window that is too big.
//!
//! A window that is maximized, minimized or full screen is the OS's to size and is skipped. Only the size
//! is judged: a window of fitting size that lies partly off the screen is left where the plugin put it.
//! When the size changes, the window keeps its centre and slides back inside the work area. The window is
//! already visible then (the plugin shows it at the end of its restore), so the resize is seen once, on a
//! webview that has not painted a page yet.
//!
//! # Timing
//!
//! Tauri creates the config windows in `setup`, before the `.setup()` closure runs, and the
//! window-state plugin restores (and, for a window with no saved state, shows and focuses) a window in
//! its `on_window_ready` hook. [`init`] has an `on_window_ready` hook as well and is registered BEFORE
//! the window-state plugin (hooks run in registration order, a `Vec` in tauri's `PluginStore`), so on a
//! first launch the window is moved and sized right before the plugin records and shows it: this is the
//! same place and the same two calls (`set_position`, `set_size`) the plugin makes to restore a saved
//! window on every later launch, in the same event-loop turn, with the webview created but without a
//! page painted yet (the navigation has only started; nothing pumps messages in between). Focus and
//! visibility are not touched. The order also makes a wrong "first launch" guess harmless: if the plugin
//! does have a saved state, its restore runs after this hook and wins. [`init_after_restore`] is
//! registered AFTER the plugin, so the restored size is what it sees. `main.rs` holds both orders and a
//! test (`main_registers_…`) fails when one moves.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use tauri::plugin::{Builder, TauriPlugin};
use tauri::{Manager, Monitor, PhysicalPosition, PhysicalSize, Runtime, Window};
use tauri_plugin_window_state::AppHandleExt as _;

/// Plugin names. They have no commands, so they need no capability.
pub const PLUGIN_NAME: &str = "first-window-size";
pub const FIT_PLUGIN_NAME: &str = "restored-window-fit";
/// The host window (`tauri.conf.json` `app.windows[0].label`, the default `main`).
pub const MAIN_WINDOW_LABEL: &str = "main";

/// Logical size of the first window at interface size 1 (`s` multiplies it).
const BASE_SIZE: (f64, f64) = (960.0, 720.0);
/// Logical size the window never shrinks below (the size of `tauri.conf.json`), unless the work area is smaller.
const MIN_SIZE: (f64, f64) = (800.0, 600.0);
/// Share of the work area (logical) the window may take at most.
const WORK_AREA_FRACTION: f64 = 0.9;
/// The Auto interface size rule: 1440 logical px on the short side and 2560 on the long side are 1.0.
const REFERENCE_SHORT_SIDE: f64 = 1440.0;
const REFERENCE_LONG_SIDE: f64 = 2560.0;
/// Auto interface size bounds and step.
const AUTO_SCALE_MIN: f64 = 1.0;
const AUTO_SCALE_MAX: f64 = 2.0;
const AUTO_SCALE_STEPS_PER_UNIT: f64 = 20.0;

/// Interface size "Auto" for a monitor: logical = physical / Windows scale factor;
/// `s = min(shortSide / 1440, longSide / 2560)`, clamped to [1, 2], rounded to the nearest 0.05.
/// A zero or invalid size or scale factor gives 1. Shared rule: `src/core/ui/uiScale.fixture.json`.
pub fn auto_ui_scale(width: u32, height: u32, scale_factor: f64) -> f64 {
    if width == 0 || height == 0 || !scale_factor.is_finite() || scale_factor <= 0.0 {
        return 1.0;
    }
    let (w, h) = (f64::from(width) / scale_factor, f64::from(height) / scale_factor);
    let (short, long) = if w < h { (w, h) } else { (h, w) };
    let raw = (short / REFERENCE_SHORT_SIDE)
        .min(long / REFERENCE_LONG_SIDE)
        .clamp(AUTO_SCALE_MIN, AUTO_SCALE_MAX);
    // Whole 0.05 steps: `k / 20` is the double nearest to the decimal, like the JSON literals.
    (raw * AUTO_SCALE_STEPS_PER_UNIT).round() / AUTO_SCALE_STEPS_PER_UNIT
}

/// What the placement needs to know about the monitor the window opens on (physical px).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonitorFacts {
    /// The monitor's resolution (the Auto interface size is derived from this).
    pub size: (u32, u32),
    /// Top-left corner of the work area (the monitor minus the taskbar), in virtual-screen coordinates.
    pub work_position: (i32, i32),
    pub work_size: (u32, u32),
    /// Windows display scale of this monitor (1.0 = 100%).
    pub scale_factor: f64,
}

impl From<&Monitor> for MonitorFacts {
    fn from(monitor: &Monitor) -> Self {
        let work = monitor.work_area();
        MonitorFacts {
            size: (monitor.size().width, monitor.size().height),
            work_position: (work.position.x, work.position.y),
            work_size: (work.size.width, work.size.height),
            scale_factor: monitor.scale_factor(),
        }
    }
}

/// Position and size of the window in physical px.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowPlacement {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// The first-launch window for a monitor: `BASE_SIZE x s` logical, at most 90 % of the work area, at
/// least 800 x 600 unless the work area is smaller, centred in the work area. `None` when the work
/// area or the scale factor is unusable (the configured size stays).
pub fn first_window_placement(monitor: &MonitorFacts) -> Option<WindowPlacement> {
    let scale = monitor.scale_factor;
    let (work_w, work_h) = monitor.work_size;
    if !scale.is_finite() || scale <= 0.0 || work_w == 0 || work_h == 0 {
        return None;
    }
    let s = auto_ui_scale(monitor.size.0, monitor.size.1, scale);
    let fit = |want: f64, work_physical: u32, floor: f64| {
        let work = f64::from(work_physical) / scale;
        want.min(work * WORK_AREA_FRACTION).max(floor).min(work)
    };
    let logical_w = fit(BASE_SIZE.0 * s, work_w, MIN_SIZE.0);
    let logical_h = fit(BASE_SIZE.1 * s, work_h, MIN_SIZE.1);
    // Rounding up can overshoot the work area by a pixel: the clamp keeps the window inside it.
    let width = ((logical_w * scale).round() as u32).clamp(1, work_w);
    let height = ((logical_h * scale).round() as u32).clamp(1, work_h);
    let margin = |free: u32| i32::try_from(free / 2).unwrap_or(0);
    Some(WindowPlacement {
        x: monitor
            .work_position
            .0
            .saturating_add(margin(work_w - width)),
        y: monitor
            .work_position
            .1
            .saturating_add(margin(work_h - height)),
        width,
        height,
    })
}

/// The version of `tauri-plugin-window-state` that `SavedWindowState` and `is_plugin_default` mirror.
/// `window_state_mirror_matches_the_locked_plugin_version` fails when `Cargo.lock` moves off it.
#[cfg(test)]
const MIRRORED_WINDOW_STATE_VERSION: &str = "2.4.1";

/// One entry of the window-state plugin's file (`.window-state.json` in the app config dir: a JSON
/// object of window label -> state). The fields are the plugin's own (`WindowState` in
/// tauri-plugin-window-state 2.4.1, see `MIRRORED_WINDOW_STATE_VERSION`) and all are required, because
/// the plugin fails to load the whole file when one entry does not fit; this mirror fails the same way.
/// After a plugin upgrade, compare its `WindowState` fields and `Default` impl with this and
/// `is_plugin_default` before raising the version.
#[derive(Debug, Deserialize)]
struct SavedWindowState {
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    prev_x: i32,
    prev_y: i32,
    maximized: bool,
    visible: bool,
    decorated: bool,
    fullscreen: bool,
}

impl SavedWindowState {
    /// The plugin's `WindowState::default()`: what it stores for a window it tracks but has never
    /// saved, and what its restore treats as "no saved state".
    fn is_plugin_default(&self) -> bool {
        (self.width, self.height, self.x, self.y, self.prev_x, self.prev_y) == (0, 0, 0, 0, 0, 0)
            && !self.maximized
            && self.visible
            && self.decorated
            && !self.fullscreen
    }
}

/// Does the window-state file (its text, `None` for a missing or unreadable file) hold a real saved
/// state for `label`? Anything the plugin could not load counts as none, as does its default entry.
pub fn has_saved_state(file_text: Option<&str>, label: &str) -> bool {
    let Some(text) = file_text else { return false };
    let Ok(states) = serde_json::from_str::<HashMap<String, SavedWindowState>>(text) else {
        return false;
    };
    states
        .get(label)
        .is_some_and(|state| !state.is_plugin_default())
}

/// The window-state file's text; a missing, unreadable or non-UTF-8 file is `None`.
fn read_state_file(dir: &Path, file_name: &str) -> Option<String> {
    std::fs::read_to_string(dir.join(file_name)).ok()
}

/// The decision for one new window: its placement when it is the main window of a first launch on a
/// usable monitor, otherwise `None` (leave it to the configured size or the plugin's restore).
/// `monitor` is only called when it is needed.
pub fn plan(label: &str, enabled: bool, state_file: Option<&str>, monitor: impl FnOnce() -> Option<MonitorFacts>) -> Option<WindowPlacement> {
    if !enabled || label != MAIN_WINDOW_LABEL || has_saved_state(state_file, label) {
        return None;
    }
    first_window_placement(&monitor()?)
}

/// The window as the fit step finds it once the plugin has restored it (physical px).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowFacts {
    pub rect: WindowPlacement,
    pub maximized: bool,
    pub minimized: bool,
    pub fullscreen: bool,
}

/// The size a restored window should have on `monitor`, or `None` when it is fine as it is (or the
/// monitor cannot be used). `auto_scale` is the Auto interface size in effect (`Some(s)`), or `None`
/// when the user picked a size of their own; a value of 1 or less means nothing to grow for.
///
/// - Grown, with Auto above 1: per axis to `800 x 600 logical x s`, so the window holds at least the
///   800 x 600 of layout it is designed for. A floor, not a factor: growing a window by `s` on every
///   launch would never stop, and a second look at a grown window changes nothing.
/// - Cut, in any mode: to the work area of the monitor.
/// - When the size changes, the window keeps its centre and slides back inside the work area.
pub fn fit_restored_window(rect: WindowPlacement, monitor: &MonitorFacts, auto_scale: Option<f64>) -> Option<WindowPlacement> {
    let scale = monitor.scale_factor;
    let (work_w, work_h) = monitor.work_size;
    if !scale.is_finite() || scale <= 0.0 || work_w == 0 || work_h == 0 || rect.width == 0 || rect.height == 0 {
        return None;
    }
    // Only an Auto size above 1 grows a window: at 1 nothing changed since before Auto, and a window the
    // user made small stays small.
    let grow_by = auto_scale.filter(|s| s.is_finite() && *s > 1.0);
    let fit = |current: u32, floor_logical: f64, work: u32| {
        // `as u32` saturates, so an absurd scale cannot wrap.
        let required = grow_by.map_or(0, |s| (floor_logical * s * scale).round() as u32);
        current.max(required).min(work)
    };
    let width = fit(rect.width, MIN_SIZE.0, work_w);
    let height = fit(rect.height, MIN_SIZE.1, work_h);
    if (width, height) == (rect.width, rect.height) {
        return None;
    }
    let place = |origin: i32, extent: u32, new_extent: u32, work_origin: i32, work_extent: u32| {
        let centre = i64::from(origin) + i64::from(extent / 2);
        let start = centre - i64::from(new_extent / 2);
        let lowest = i64::from(work_origin);
        let highest = lowest + i64::from(work_extent - new_extent);
        i32::try_from(start.clamp(lowest, highest)).unwrap_or(work_origin)
    };
    Some(WindowPlacement {
        x: place(rect.x, rect.width, width, monitor.work_position.0, work_w),
        y: place(rect.y, rect.height, height, monitor.work_position.1, work_h),
        width,
        height,
    })
}

/// The decision of the fit step for one window: the new place and size, or `None` to leave it. Only
/// the main window of a run that is not `SIGIL_START_MINIMIZED`, and not while it is maximized,
/// minimized or full screen (the OS sizes those). `auto` says whether Interface size is Auto
/// ([`interface_size_is_auto`]).
pub fn plan_fit(label: &str, enabled: bool, window: &WindowFacts, monitor: &MonitorFacts, auto: bool) -> Option<WindowPlacement> {
    if !enabled || label != MAIN_WINDOW_LABEL || window.maximized || window.minimized || window.fullscreen {
        return None;
    }
    let auto_scale = auto.then(|| auto_ui_scale(monitor.size.0, monitor.size.1, monitor.scale_factor));
    fit_restored_window(window.rect, monitor, auto_scale)
}

// -------------------------------------------------------------------------------------------------
// Interface size choice
// -------------------------------------------------------------------------------------------------

/// The host's persisted settings, relative to the app data folder (`SETTINGS_PATH` in
/// `src/core/services/state/index.ts`), and the atomic writer's backup.
const SETTINGS_DIR: &str = "user";
const SETTINGS_FILE: &str = "settings";
const SETTINGS_BACKUP_FILE: &str = "settings.bak";
/// What Windows PowerShell 5.1 and Notepad put in front of a text file; the front end skips it too.
const UTF8_BOM: &[u8] = &[0xEF, 0xBB, 0xBF];
/// "Is 1" for the manual size (a hand-edited 1.0 or a slider value): `UiScaleAutoState` in the front end.
const SAME_SCALE_EPSILON: f64 = 0.001;

fn settings_object(bytes: &[u8]) -> Option<serde_json::Value> {
    let value: serde_json::Value = serde_json::from_slice(bytes.strip_prefix(UTF8_BOM).unwrap_or(bytes)).ok()?;
    value.is_object().then_some(value)
}

/// The persisted settings (a JSON object) as the front end will load them: `<app_data>/user/settings`,
/// else its backup when that file is damaged. `None` for a new install (no settings file: the front end
/// ignores a backup then) and for settings it would reset to the defaults.
/// (`web::origin::read_persisted_settings` is the same read, private to the web plugin.)
pub fn read_persisted_settings(app_data_dir: &Path) -> Option<serde_json::Value> {
    let dir = app_data_dir.join(SETTINGS_DIR);
    match std::fs::read(dir.join(SETTINGS_FILE)) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Ok(bytes) => {
            if let Some(settings) = settings_object(&bytes) {
                return Some(settings);
            }
        }
        Err(_) => {}
    }
    settings_object(&std::fs::read(dir.join(SETTINGS_BACKUP_FILE)).ok()?)
}

/// Is Interface size Auto at this launch? The front end decides it once, at the first launch that knows
/// about Auto (`applyUiScaleAutoDecision`, `src/core/services/state/uiScaleAuto.ts`), after this code has
/// run, so the same decision is made here: before `uiScaleAutoDecided`, a profile that picked S, L or XL
/// (`uiScale` is not 1) keeps its size and any other uses Auto; afterwards `uiScaleAuto` is the choice.
/// Missing or invalid keys have the schema defaults (`zSafe`): `uiScale` 1, `uiScaleAuto` true,
/// `uiScaleAutoDecided` false. No settings (`None`) is a new install: Auto.
pub fn interface_size_is_auto(settings: Option<&serde_json::Value>) -> bool {
    let Some(settings) = settings.and_then(serde_json::Value::as_object) else {
        return true;
    };
    let flag = |key: &str, default: bool| {
        settings
            .get(key)
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(default)
    };
    if flag("uiScaleAutoDecided", false) {
        return flag("uiScaleAuto", true);
    }
    let manual = settings
        .get("uiScale")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(1.0);
    (manual - 1.0).abs() <= SAME_SCALE_EPSILON
}

// -------------------------------------------------------------------------------------------------
// Runtime glue
// -------------------------------------------------------------------------------------------------

/// The monitor the new window sits on (where Windows opened it), else the primary one.
fn window_monitor<R: Runtime>(window: &Window<R>) -> Option<MonitorFacts> {
    let monitor = window
        .current_monitor()
        .ok()
        .flatten()
        .or_else(|| window.primary_monitor().ok().flatten())?;
    Some(MonitorFacts::from(&monitor))
}

/// Moves and sizes the window. Position first, then size: the order of the plugin's own restore.
/// False (after a warning) when either call failed.
fn apply_placement<R: Runtime>(window: &Window<R>, placement: WindowPlacement, what: &str) -> bool {
    if let Err(e) = window.set_position(PhysicalPosition::new(placement.x, placement.y)) {
        tracing::warn!("{what}: could not move the window: {e}");
        return false;
    }
    if let Err(e) = window.set_size(PhysicalSize::new(placement.width, placement.height)) {
        tracing::warn!("{what}: could not resize the window: {e}");
        return false;
    }
    true
}

fn place_new_window<R: Runtime>(window: &Window<R>, enabled: bool) {
    if !enabled || window.label() != MAIN_WINDOW_LABEL {
        return;
    }
    let app = window.app_handle();
    let state_file = app
        .path()
        .app_config_dir()
        .ok()
        .and_then(|dir| read_state_file(&dir, &app.filename()));
    let mut facts = None;
    let Some(placement) = plan(window.label(), enabled, state_file.as_deref(), || {
        facts = window_monitor(window);
        facts
    }) else {
        return;
    };
    if !apply_placement(window, placement, "first window size") {
        return;
    }
    if let Some(monitor) = facts {
        tracing::info!(
            "first window size: {}x{} at ({}, {}) on a {}x{} monitor at {:.0}% (interface size Auto {})",
            placement.width,
            placement.height,
            placement.x,
            placement.y,
            monitor.size.0,
            monitor.size.1,
            monitor.scale_factor * 100.0,
            auto_ui_scale(monitor.size.0, monitor.size.1, monitor.scale_factor),
        );
    }
}

/// The window as it is right now; `None` when the OS will not say (the window is then left alone).
fn window_facts<R: Runtime>(window: &Window<R>) -> Option<WindowFacts> {
    let size = window.inner_size().ok()?;
    let position = window.outer_position().ok()?;
    Some(WindowFacts {
        rect: WindowPlacement {
            x: position.x,
            y: position.y,
            width: size.width,
            height: size.height,
        },
        maximized: window.is_maximized().ok()?,
        minimized: window.is_minimized().ok()?,
        fullscreen: window.is_fullscreen().ok()?,
    })
}

/// The fit step on a window the plugin has just restored (see the module docs).
fn fit_restored<R: Runtime>(window: &Window<R>, enabled: bool) {
    if !enabled || window.label() != MAIN_WINDOW_LABEL {
        return;
    }
    let (Some(facts), Some(monitor)) = (window_facts(window), window_monitor(window)) else {
        return;
    };
    let settings = crate::services::diagnostics::app_data_dir().and_then(|dir| read_persisted_settings(&dir));
    let auto = interface_size_is_auto(settings.as_ref());
    let Some(fitted) = plan_fit(window.label(), enabled, &facts, &monitor, auto) else {
        return;
    };
    if !apply_placement(window, fitted, "restored window fit") {
        return;
    }
    tracing::info!(
        "restored window fit: {}x{} at ({}, {}) -> {}x{} at ({}, {}) on a {}x{} monitor at {:.0}% (interface size {})",
        facts.rect.width,
        facts.rect.height,
        facts.rect.x,
        facts.rect.y,
        fitted.width,
        fitted.height,
        fitted.x,
        fitted.y,
        monitor.size.0,
        monitor.size.1,
        monitor.scale_factor * 100.0,
        if auto {
            format!("Auto {}", auto_ui_scale(monitor.size.0, monitor.size.1, monitor.scale_factor))
        } else {
            "chosen by the user".to_string()
        },
    );
}

/// Register this BEFORE the window-state plugin (see the module docs): the first-launch size. `enabled`
/// is false for automated runs (`SIGIL_START_MINIMIZED=1`).
pub fn init<R: Runtime>(enabled: bool) -> TauriPlugin<R> {
    Builder::new(PLUGIN_NAME)
        .on_window_ready(move |window| place_new_window(&window, enabled))
        .build()
}

/// Register this AFTER the window-state plugin (see the module docs): the fit of a restored window.
/// `enabled` is false for automated runs (`SIGIL_START_MINIMIZED=1`).
pub fn init_after_restore<R: Runtime>(enabled: bool) -> TauriPlugin<R> {
    Builder::new(FIT_PLUGIN_NAME)
        .on_window_ready(move |window| fit_restored(&window, enabled))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Auto rule's table, shared with the TypeScript side (`src/core/ui/uiScale.fixture.json`, owned by
    /// the Interface size change). The file must be committed with or before this module, or `cargo test` and
    /// `cargo clippy --all-targets` stop compiling.
    const UI_SCALE_FIXTURE: &str = include_str!("../../src/core/ui/uiScale.fixture.json");
    /// The sources the wiring tests read: `main.rs` registers the plugins in order, `schema.ts` names the settings.
    const MAIN_RS: &str = include_str!("main.rs");
    const SCHEMA_TS: &str = include_str!("../../src/core/schema.ts");
    /// Cargo.lock, for the window-state version the mirror was written against.
    const CARGO_LOCK: &str = include_str!("../Cargo.lock");

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct FixtureCase {
        name: String,
        width: u32,
        height: u32,
        scale_factor: f64,
        expected: f64,
    }

    #[derive(Deserialize)]
    struct Fixture {
        cases: Vec<FixtureCase>,
    }

    /// The Auto interface size is the rule in the shared fixture: the TypeScript side runs the same cases.
    #[test]
    fn auto_ui_scale_passes_every_fixture_case() {
        let fixture: Fixture = serde_json::from_str(UI_SCALE_FIXTURE).expect("uiScale.fixture.json parses");
        assert!(fixture.cases.len() >= 15, "{} cases", fixture.cases.len());
        for case in &fixture.cases {
            let got = auto_ui_scale(case.width, case.height, case.scale_factor);
            assert!(
                (got - case.expected).abs() < 1e-9,
                "{}: {}x{} at {} gave {got}, the fixture wants {}",
                case.name,
                case.width,
                case.height,
                case.scale_factor,
                case.expected
            );
        }
    }

    #[test]
    fn auto_ui_scale_is_whole_twentieths_within_one_to_two() {
        for (w, h, f) in [
            (2560, 1600, 1.0),
            (3000, 2000, 1.0),
            (3440, 1440, 1.25),
            (4096, 2304, 1.0),
            (6016, 3384, 1.0),
        ] {
            let s = auto_ui_scale(w, h, f);
            assert!((1.0..=2.0).contains(&s), "{w}x{h}@{f}: {s}");
            assert!((s * 20.0 - (s * 20.0).round()).abs() < 1e-9, "{w}x{h}@{f}: {s} is not a multiple of 0.05");
        }
        assert_eq!(auto_ui_scale(3840, 2160, f64::NAN), 1.0);
        assert_eq!(auto_ui_scale(3840, 2160, f64::INFINITY), 1.0);
        assert_eq!(auto_ui_scale(3840, 2160, -1.0), 1.0);
        assert_eq!(auto_ui_scale(0, 2160, 1.0), 1.0);
    }

    /// A monitor whose taskbar (bottom) takes `taskbar` physical px.
    fn monitor(width: u32, height: u32, scale_factor: f64, taskbar: u32) -> MonitorFacts {
        MonitorFacts {
            size: (width, height),
            work_position: (0, 0),
            work_size: (width, height - taskbar),
            scale_factor,
        }
    }

    fn placed(x: i32, y: i32, width: u32, height: u32) -> Option<WindowPlacement> {
        Some(WindowPlacement { x, y, width, height })
    }

    /// Case table of the first-launch placement (physical px out). Work areas have a 48 px taskbar at 100%.
    #[test]
    fn first_window_placement_cases() {
        let shifted = |origin: (i32, i32), base: MonitorFacts| MonitorFacts {
            work_position: origin,
            ..base
        };
        let cases: Vec<(&str, MonitorFacts, Option<WindowPlacement>)> = vec![
            // s = 1: 960 x 720.
            ("1080p at 100%", monitor(1920, 1080, 1.0, 48), placed(480, 156, 960, 720)),
            // s = 1.5: 1440 x 1080 logical, 1:1 physical at 100%.
            ("4K at 100%", monitor(3840, 2160, 1.0, 48), placed(1200, 516, 1440, 1080)),
            // s = 1.2: 1152 x 864 logical = 1440 x 1080 physical.
            ("4K at 125%", monitor(3840, 2160, 1.25, 60), placed(1200, 510, 1440, 1080)),
            // Windows already scales it: s = 1, 960 x 720 logical = 1440 x 1080 physical.
            ("4K at 150%", monitor(3840, 2160, 1.5, 72), placed(1200, 504, 1440, 1080)),
            // The height is capped to 90 % of the 720 px work area (648), above the 600 floor.
            ("1366x768 at 100%", monitor(1366, 768, 1.0, 48), placed(203, 36, 960, 648)),
            ("1280x720 at 100%", monitor(1280, 720, 1.0, 48), placed(160, 33, 960, 605)),
            // s = 2 (5K): 1920 x 1440.
            ("5K at 100%", monitor(5120, 2880, 1.0, 48), placed(1600, 696, 1920, 1440)),
            // Ultrawide 1440p: the short side is 1440, so s = 1.
            ("ultrawide 3440x1440 at 100%", monitor(3440, 1440, 1.0, 48), placed(1240, 336, 960, 720)),
            // 3840 x 1600: s = 1.1 (the short side sets it), 1056 x 792.
            ("ultrawide 3840x1600 at 100%", monitor(3840, 1600, 1.0, 48), placed(1392, 380, 1056, 792)),
            // 90 % and 800 x 600 apply to LOGICAL px: the caps are 737 x 518 here, the floor lifts the
            // width to 800 and the work area (576 logical high) holds the height at 576.
            (
                "1024x768 at 125% (floor, work area smaller)",
                monitor(1024, 768, 1.25, 48),
                placed(12, 0, 1000, 720),
            ),
            // A work area smaller than 800 x 600: the window is the work area.
            ("800x600 monitor", monitor(800, 600, 1.0, 40), placed(0, 0, 800, 560)),
            // Secondary monitor left of the primary (negative origin) and a taskbar at the top.
            (
                "monitor at a negative origin",
                shifted((-1920, 0), monitor(1920, 1080, 1.0, 48)),
                placed(-1440, 156, 960, 720),
            ),
            (
                "taskbar at the top",
                shifted((0, 48), monitor(1920, 1080, 1.0, 48)),
                placed(480, 204, 960, 720),
            ),
            // The monitor's resolution is unknown but its work area is fine: s = 1.
            (
                "unknown resolution",
                MonitorFacts {
                    size: (0, 0),
                    ..monitor(1920, 1080, 1.0, 48)
                },
                placed(480, 156, 960, 720),
            ),
            // Unusable work area or scale factor: nothing is changed.
            (
                "empty work area",
                MonitorFacts {
                    work_size: (0, 0),
                    ..monitor(1920, 1080, 1.0, 0)
                },
                None,
            ),
            ("scale factor 0", monitor(1920, 1080, 0.0, 48), None),
            ("scale factor NaN", monitor(1920, 1080, f64::NAN, 48), None),
        ];
        for (name, facts, want) in cases {
            assert_eq!(first_window_placement(&facts), want, "{name}");
        }
    }

    /// The window always stays inside the work area and is centred in it, for any monitor in the table.
    #[test]
    fn placement_stays_inside_the_work_area() {
        for (w, h, f) in [
            (1366, 768, 1.0),
            (1920, 1080, 1.0),
            (1920, 1080, 1.25),
            (2560, 1440, 1.0),
            (3440, 1440, 1.0),
            (3840, 2160, 1.0),
            (3840, 2160, 1.75),
            (5120, 2880, 1.0),
            (7680, 4320, 1.0),
            (1024, 768, 1.0),
            (1024, 768, 2.0),
        ] {
            let facts = monitor(w, h, f, (48.0 * f) as u32);
            let p = first_window_placement(&facts).unwrap_or_else(|| panic!("{w}x{h}@{f}"));
            let work_right = facts.work_position.0 + facts.work_size.0 as i32;
            let work_bottom = facts.work_position.1 + facts.work_size.1 as i32;
            assert!(p.x >= facts.work_position.0 && p.y >= facts.work_position.1, "{w}x{h}@{f}: {p:?}");
            assert!(
                p.x + p.width as i32 <= work_right && p.y + p.height as i32 <= work_bottom,
                "{w}x{h}@{f}: {p:?}"
            );
            // Centred horizontally to within a pixel.
            let left = p.x - facts.work_position.0;
            let right = work_right - (p.x + p.width as i32);
            assert!((left - right).abs() <= 1, "{w}x{h}@{f}: {p:?}");
        }
    }

    const DEFAULT_ENTRY: &str =
        r#"{"width":0,"height":0,"x":0,"y":0,"prev_x":0,"prev_y":0,"maximized":false,"visible":true,"decorated":true,"fullscreen":false}"#;
    const SAVED_ENTRY: &str = r#"{"width":1440,"height":1080,"x":1200,"y":516,"prev_x":1200,"prev_y":516,"maximized":false,"visible":true,"decorated":false,"fullscreen":false}"#;

    #[test]
    fn saved_state_detection_follows_the_plugin() {
        let file = |entries: &str| format!("{{{entries}}}");
        let main = |entry: &str| file(&format!(r#""main":{entry}"#));
        // Real state for main (pretty-printed like the plugin writes it).
        assert!(has_saved_state(Some(&main(SAVED_ENTRY)), "main"));
        let pretty = serde_json::to_string_pretty(&serde_json::from_str::<serde_json::Value>(&main(SAVED_ENTRY)).unwrap()).unwrap();
        assert!(has_saved_state(Some(&pretty), "main"));
        // A maximized window, or one that was hidden, is a saved state too.
        assert!(has_saved_state(
            Some(&main(&SAVED_ENTRY.replace(r#""maximized":false"#, r#""maximized":true"#))),
            "main"
        ));
        assert!(has_saved_state(
            Some(&main(&SAVED_ENTRY.replace(r#""visible":true"#, r#""visible":false"#))),
            "main"
        ));
        // Missing or unreadable file, empty or invalid JSON: none.
        assert!(!has_saved_state(None, "main"));
        for text in ["", "   ", "not json", "{", "[]", "null", "{}", r#"{"main":null}"#, r#"{"main":{}}"#] {
            assert!(!has_saved_state(Some(text), "main"), "{text:?}");
        }
        // The entry the plugin stores for a window it tracked but never saved is "no state".
        assert!(!has_saved_state(Some(&main(DEFAULT_ENTRY)), "main"));
        // Another window's state is not main's.
        assert!(!has_saved_state(Some(&file(&format!(r#""captions":{SAVED_ENTRY}"#))), "main"));
        assert!(has_saved_state(
            Some(&file(&format!(r#""captions":{SAVED_ENTRY},"main":{SAVED_ENTRY}"#))),
            "main"
        ));
        // One entry the plugin cannot read makes it drop the whole file: so does this.
        let missing_field = SAVED_ENTRY.replace(r#""fullscreen":false"#, r#""other":1"#);
        assert!(!has_saved_state(Some(&main(&missing_field)), "main"));
        assert!(!has_saved_state(
            Some(&file(&format!(r#""main":{SAVED_ENTRY},"captions":{{"width":"wide"}}"#))),
            "main"
        ));
        let negative_width = SAVED_ENTRY.replace(r#""width":1440"#, r#""width":-1"#);
        assert!(!has_saved_state(Some(&main(&negative_width)), "main"));
    }

    #[test]
    fn state_file_name_is_the_plugins_default() {
        // main.rs builds the plugin with its default file name; the module asks the plugin at run time.
        assert_eq!(tauri_plugin_window_state::DEFAULT_FILENAME, ".window-state.json");
    }

    #[test]
    fn state_file_reads_as_none_when_missing_or_unreadable() {
        let dir = std::env::temp_dir().join(format!("sigil-window-size-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Missing.
        assert_eq!(read_state_file(&dir, ".window-state.json"), None);
        assert_eq!(read_state_file(&dir.join("no-such-dir"), ".window-state.json"), None);
        // A directory where the file should be.
        std::fs::create_dir_all(dir.join(".window-state.json")).unwrap();
        assert_eq!(read_state_file(&dir, ".window-state.json"), None);
        // Not UTF-8.
        std::fs::write(dir.join("binary.json"), [0xff, 0xfe, 0x00]).unwrap();
        assert_eq!(read_state_file(&dir, "binary.json"), None);
        // A real file.
        std::fs::write(dir.join("ok.json"), format!(r#"{{"main":{SAVED_ENTRY}}}"#)).unwrap();
        let text = read_state_file(&dir, "ok.json").expect("readable file");
        assert!(has_saved_state(Some(&text), "main"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn big_monitor() -> Option<MonitorFacts> {
        Some(monitor(3840, 2160, 1.0, 48))
    }

    #[test]
    fn plan_places_only_the_main_window_of_a_first_launch() {
        let want = placed(1200, 516, 1440, 1080);
        // First launch: no file, an unreadable one, or the plugin's default entry.
        assert_eq!(plan("main", true, None, big_monitor), want);
        assert_eq!(plan("main", true, Some("garbage"), big_monitor), want);
        assert_eq!(plan("main", true, Some(&format!(r#"{{"main":{DEFAULT_ENTRY}}}"#)), big_monitor), want);
        // A file that only knows the captions window.
        assert_eq!(plan("main", true, Some(&format!(r#"{{"captions":{SAVED_ENTRY}}}"#)), big_monitor), want);
        // A saved state for main: the plugin restores the user's size, this module does nothing.
        assert_eq!(plan("main", true, Some(&format!(r#"{{"main":{SAVED_ENTRY}}}"#)), big_monitor), None);
        // SIGIL_START_MINIMIZED runs and other windows are never resized.
        assert_eq!(plan("main", false, None, big_monitor), None);
        assert_eq!(plan("captions", true, None, big_monitor), None);
        // No monitor information: nothing to do.
        assert_eq!(plan("main", true, None, || None), None);
    }

    #[test]
    fn plan_asks_for_the_monitor_only_when_it_places_the_window() {
        let ask = |label: &str, enabled: bool, file: Option<&str>| {
            let mut asked = false;
            plan(label, enabled, file, || {
                asked = true;
                big_monitor()
            });
            asked
        };
        assert!(ask("main", true, None));
        assert!(!ask("main", false, None));
        assert!(!ask("captions", true, None));
        assert!(!ask("main", true, Some(&format!(r#"{{"main":{SAVED_ENTRY}}}"#))));
    }

    // ---------------------------------------------------------------------------------------------
    // Fit after a restore
    // ---------------------------------------------------------------------------------------------

    fn rect(x: i32, y: i32, width: u32, height: u32) -> WindowPlacement {
        WindowPlacement { x, y, width, height }
    }

    /// Case table of the fit step (physical px in and out; `None` = leave the window). The 4K monitors
    /// have a 48 px taskbar at 100%, 60 px at 125%.
    #[test]
    fn fit_restored_window_cases() {
        let at_origin = |origin: (i32, i32), base: MonitorFacts| MonitorFacts {
            work_position: origin,
            ..base
        };
        let m4k = monitor(3840, 2160, 1.0, 48);
        let laptop = monitor(1920, 1080, 1.0, 48);
        // Name, monitor, the restored window, the Auto size (None = a size the user picked), the result.
        type FitCase = (&'static str, MonitorFacts, WindowPlacement, Option<f64>, Option<WindowPlacement>);
        let cases: Vec<FitCase> = vec![
            // The upgrade case: the saved 800 x 600 of a profile from before Auto, on 4K at 100% (s = 1.5):
            // grown to 1200 x 900, centre kept, then slid inside the work area.
            (
                "4K at 100%, saved 800x600",
                m4k,
                rect(100, 100, 800, 600),
                Some(1.5),
                Some(rect(0, 0, 1200, 900)),
            ),
            (
                "4K at 100%, saved 800x600, mid-screen",
                m4k,
                rect(1000, 800, 800, 600),
                Some(1.5),
                Some(rect(800, 650, 1200, 900)),
            ),
            // 4K at 125% (s = 1.2): 1000 x 750 physical is the same 800 x 600 logical.
            (
                "4K at 125%, saved 1000x750",
                monitor(3840, 2160, 1.25, 60),
                rect(1000, 500, 1000, 750),
                Some(1.2),
                Some(rect(900, 425, 1200, 900)),
            ),
            // A window that is large enough is the user's.
            ("4K at 100%, already 1440x1080", m4k, rect(1200, 516, 1440, 1080), Some(1.5), None),
            ("4K at 100%, exactly the floor", m4k, rect(0, 0, 1200, 900), Some(1.5), None),
            // One axis short: only that axis grows, the other keeps its size (and its centre).
            (
                "4K at 100%, wide and short",
                m4k,
                rect(500, 500, 1600, 500),
                Some(1.5),
                Some(rect(500, 300, 1600, 900)),
            ),
            // A floor, not a factor: a small window goes to the floor, not to its own size times s.
            (
                "4K at 100%, tiny window",
                m4k,
                rect(1000, 1000, 500, 300),
                Some(1.5),
                Some(rect(650, 700, 1200, 900)),
            ),
            // s = 1 changes nothing for a small window; neither does a manual size or a nonsense s.
            ("s = 1, tiny window", laptop, rect(100, 100, 500, 300), Some(1.0), None),
            ("manual size on 4K", m4k, rect(100, 100, 800, 600), None, None),
            ("s below 1", m4k, rect(100, 100, 800, 600), Some(0.5), None),
            ("s NaN", m4k, rect(100, 100, 800, 600), Some(f64::NAN), None),
            // The growth stops at the work area.
            (
                "growth capped by the work area",
                monitor(1024, 768, 1.0, 48),
                rect(100, 20, 800, 600),
                Some(2.0),
                Some(rect(0, 0, 1024, 720)),
            ),
            // Too big for the monitor, in either mode: the 1440 x 1080 of a 4K first launch on a 1080p laptop
            // (work area 1920 x 1032) loses height, and is slid inside the work area.
            (
                "4K size on a 1080p laptop",
                laptop,
                rect(1200, 516, 1440, 1080),
                Some(1.0),
                Some(rect(480, 0, 1440, 1032)),
            ),
            (
                "4K size on a 1080p laptop, manual",
                laptop,
                rect(1200, 516, 1440, 1080),
                None,
                Some(rect(480, 0, 1440, 1032)),
            ),
            (
                "bigger than the work area both ways",
                laptop,
                rect(-500, -300, 3000, 2000),
                None,
                Some(rect(0, 0, 1920, 1032)),
            ),
            ("exactly the work area", laptop, rect(0, 0, 1920, 1032), Some(1.0), None),
            // Work area with an origin: left of the primary monitor, and a taskbar at the top.
            (
                "monitor at a negative origin",
                at_origin((-3840, 0), m4k),
                rect(-3800, 20, 800, 600),
                Some(1.5),
                Some(rect(-3840, 0, 1200, 900)),
            ),
            (
                "taskbar at the top",
                at_origin((0, 48), laptop),
                rect(0, 0, 1440, 1080),
                None,
                Some(rect(0, 48, 1440, 1032)),
            ),
            // Nothing usable to judge by: nothing is changed.
            ("zero width", m4k, rect(0, 0, 0, 600), Some(1.5), None),
            ("zero height", m4k, rect(0, 0, 800, 0), Some(1.5), None),
            ("scale factor 0", monitor(3840, 2160, 0.0, 48), rect(0, 0, 800, 600), Some(1.5), None),
            (
                "scale factor NaN",
                monitor(3840, 2160, f64::NAN, 48),
                rect(0, 0, 800, 600),
                Some(1.5),
                None,
            ),
            (
                "empty work area",
                MonitorFacts { work_size: (0, 0), ..m4k },
                rect(0, 0, 800, 600),
                Some(1.5),
                None,
            ),
        ];
        for (name, facts, window, auto_scale, want) in cases {
            assert_eq!(fit_restored_window(window, &facts, auto_scale), want, "{name}");
        }
    }

    /// Whatever the fit returns lies inside the work area, and fitting again changes nothing (growing by
    /// a factor on every launch would not stop).
    #[test]
    fn fit_restored_window_is_inside_the_work_area_and_settles() {
        let monitors = [
            (1366, 768, 1.0),
            (1920, 1080, 1.0),
            (1920, 1080, 1.25),
            (2560, 1440, 1.0),
            (3840, 2160, 1.0),
            (3840, 2160, 1.25),
            (3840, 2160, 1.75),
            (5120, 2880, 1.0),
            (7680, 4320, 1.0),
            (1024, 768, 1.0),
        ];
        let windows = [
            rect(0, 0, 800, 600),
            rect(300, 200, 500, 300),
            rect(2000, 1500, 1000, 750),
            rect(-400, -200, 1440, 1080),
            rect(100, 100, 4000, 3000),
            rect(900, 900, 1200, 400),
        ];
        for (w, h, f) in monitors {
            let facts = monitor(w, h, f, (48.0 * f) as u32);
            let auto = auto_ui_scale(w, h, f);
            let work_right = facts.work_position.0 + facts.work_size.0 as i32;
            let work_bottom = facts.work_position.1 + facts.work_size.1 as i32;
            for window in windows {
                for auto_scale in [None, Some(auto)] {
                    let label = format!("{w}x{h}@{f} {window:?} {auto_scale:?}");
                    let Some(fitted) = fit_restored_window(window, &facts, auto_scale) else {
                        continue;
                    };
                    assert!(fitted.width > 0 && fitted.height > 0, "{label}: {fitted:?}");
                    assert!(
                        fitted.x >= facts.work_position.0 && fitted.y >= facts.work_position.1,
                        "{label}: {fitted:?}"
                    );
                    assert!(
                        fitted.x + fitted.width as i32 <= work_right && fitted.y + fitted.height as i32 <= work_bottom,
                        "{label}: {fitted:?}"
                    );
                    assert_eq!(fit_restored_window(fitted, &facts, auto_scale), None, "{label}: {fitted:?} moves again");
                }
            }
        }
    }

    /// The first-launch window is already what the fit wants: with the same Auto size it leaves it alone,
    /// so the two steps never fight on a new install.
    #[test]
    fn the_first_launch_placement_needs_no_fit() {
        for (w, h, f) in [
            (1280, 720, 1.0),
            (1366, 768, 1.0),
            (1920, 1080, 1.0),
            (1920, 1080, 1.25),
            (2560, 1440, 1.0),
            (3440, 1440, 1.0),
            (3840, 1600, 1.0),
            (3840, 2160, 1.0),
            (3840, 2160, 1.25),
            (3840, 2160, 1.5),
            (5120, 2880, 1.0),
            (7680, 4320, 1.0),
            (1024, 768, 1.25),
            (800, 600, 1.0),
        ] {
            let facts = monitor(w, h, f, (48.0 * f) as u32);
            let placed = first_window_placement(&facts).unwrap_or_else(|| panic!("{w}x{h}@{f}"));
            for auto_scale in [None, Some(auto_ui_scale(w, h, f))] {
                assert_eq!(
                    fit_restored_window(placed, &facts, auto_scale),
                    None,
                    "{w}x{h}@{f} {auto_scale:?}: {placed:?}"
                );
            }
        }
    }

    fn window_facts_of(rect: WindowPlacement) -> WindowFacts {
        WindowFacts {
            rect,
            maximized: false,
            minimized: false,
            fullscreen: false,
        }
    }

    #[test]
    fn plan_fit_judges_only_the_main_window_of_a_normal_run() {
        let m4k = monitor(3840, 2160, 1.0, 48);
        let small = window_facts_of(rect(100, 100, 800, 600));
        let want = Some(rect(0, 0, 1200, 900));
        assert_eq!(plan_fit("main", true, &small, &m4k, true), want);
        // Auto off: the user's own size is not grown (but see the cut below).
        assert_eq!(plan_fit("main", true, &small, &m4k, false), None);
        let too_big = window_facts_of(rect(1200, 516, 1440, 1080));
        assert_eq!(
            plan_fit("main", true, &too_big, &monitor(1920, 1080, 1.0, 48), false),
            Some(rect(480, 0, 1440, 1032))
        );
        // SIGIL_START_MINIMIZED runs and other windows are never resized.
        assert_eq!(plan_fit("main", false, &small, &m4k, true), None);
        assert_eq!(plan_fit("captions", true, &small, &m4k, true), None);
        // The OS sizes a maximized, minimized or full-screen window.
        for state in [
            WindowFacts { maximized: true, ..small },
            WindowFacts { minimized: true, ..small },
            WindowFacts { fullscreen: true, ..small },
        ] {
            assert_eq!(plan_fit("main", true, &state, &m4k, true), None, "{state:?}");
        }
    }

    #[test]
    fn plan_fit_uses_the_auto_size_of_the_monitor_the_window_is_on() {
        let small = window_facts_of(rect(100, 100, 800, 600));
        // 1080p at 100%: Auto is 1, nothing to grow for.
        assert_eq!(plan_fit("main", true, &small, &monitor(1920, 1080, 1.0, 48), true), None);
        // 4K at 150% is 1 too; 4K at 125% is 1.2 and 4K at 100% is 1.5.
        assert_eq!(plan_fit("main", true, &small, &monitor(3840, 2160, 1.5, 72), true), None);
        let at_125 = plan_fit("main", true, &small, &monitor(3840, 2160, 1.25, 60), true).expect("4K at 125% grows it");
        assert_eq!((at_125.width, at_125.height), (1200, 900));
        let at_100 = plan_fit("main", true, &small, &monitor(3840, 2160, 1.0, 48), true).expect("4K at 100% grows it");
        assert_eq!((at_100.width, at_100.height), (1200, 900));
        // 5K at 100% is 2: 1600 x 1200.
        let at_5k = plan_fit("main", true, &small, &monitor(5120, 2880, 1.0, 48), true).expect("5K grows it");
        assert_eq!((at_5k.width, at_5k.height), (1600, 1200));
    }

    // ---------------------------------------------------------------------------------------------
    // Interface size choice
    // ---------------------------------------------------------------------------------------------

    fn is_auto(settings: &str) -> bool {
        let value: serde_json::Value = serde_json::from_str(settings).expect("test settings are JSON");
        interface_size_is_auto(Some(&value))
    }

    /// The Auto decision of `applyUiScaleAutoDecision` and the schema defaults, key by key.
    #[test]
    fn interface_size_is_auto_follows_the_front_end_decision() {
        // Not decided yet (a profile from before Auto, or a new install): a size other than M is kept.
        assert!(is_auto("{}"));
        assert!(is_auto(r#"{"uiScale":1}"#));
        assert!(is_auto(r#"{"uiScale":1.0005}"#));
        assert!(!is_auto(r#"{"uiScale":0.8}"#));
        assert!(!is_auto(r#"{"uiScale":1.2}"#));
        assert!(!is_auto(r#"{"uiScale":1.4}"#));
        assert!(!is_auto(r#"{"uiScale":1.002}"#));
        // `uiScaleAuto` means nothing before the decision (the front end overwrites it).
        assert!(is_auto(r#"{"uiScale":1,"uiScaleAuto":false}"#));
        assert!(!is_auto(r#"{"uiScale":1.2,"uiScaleAuto":true}"#));
        // Decided: `uiScaleAuto` is the choice, and the manual size is only what Auto returns to.
        assert!(!is_auto(r#"{"uiScaleAutoDecided":true,"uiScaleAuto":false,"uiScale":1}"#));
        assert!(!is_auto(r#"{"uiScaleAutoDecided":true,"uiScaleAuto":false,"uiScale":1.2}"#));
        assert!(is_auto(r#"{"uiScaleAutoDecided":true,"uiScaleAuto":true,"uiScale":1.4}"#));
        // Schema defaults for a missing or invalid value (`zSafe`): Auto on, not decided, size 1.
        assert!(is_auto(r#"{"uiScaleAutoDecided":true}"#));
        assert!(is_auto(r#"{"uiScaleAutoDecided":true,"uiScaleAuto":"no"}"#));
        assert!(is_auto(r#"{"uiScaleAutoDecided":"yes","uiScale":1}"#));
        assert!(!is_auto(r#"{"uiScaleAutoDecided":"yes","uiScale":1.2}"#));
        assert!(is_auto(r#"{"uiScale":"big"}"#));
        assert!(is_auto(r#"{"uiScale":null}"#));
        // No settings, or something that is not an object, are defaults too: Auto.
        assert!(interface_size_is_auto(None));
        assert!(interface_size_is_auto(Some(&serde_json::json!([1, 2]))));
        assert!(interface_size_is_auto(Some(&serde_json::json!(null))));
    }

    /// The keys the front end persists are the ones this module reads: fails when the schema renames one.
    #[test]
    fn the_settings_keys_exist_in_the_front_end_schema() {
        for key in ["uiScale", "uiScaleAuto", "uiScaleAutoDecided"] {
            assert!(
                SCHEMA_TS
                    .lines()
                    .any(|line| line.trim_start().starts_with(&format!("{key}:"))),
                "src/core/schema.ts has no `{key}:`; update interface_size_is_auto and read_persisted_settings"
            );
        }
    }

    #[test]
    fn persisted_settings_are_read_like_the_front_end_does() {
        let dir = std::env::temp_dir().join(format!("sigil-window-size-settings-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // No app data folder content, no settings file: a new install.
        assert_eq!(read_persisted_settings(&dir), None);
        let user = dir.join("user");
        std::fs::create_dir_all(&user).unwrap();
        assert_eq!(read_persisted_settings(&dir), None);
        // A backup beside a missing settings file does not make it an existing profile.
        std::fs::write(user.join("settings.bak"), r#"{"uiScale":1.2}"#).unwrap();
        assert_eq!(read_persisted_settings(&dir), None);
        // The settings file, with or without a byte order mark.
        std::fs::write(user.join("settings"), r#"{"uiScale":1.4}"#).unwrap();
        assert_eq!(read_persisted_settings(&dir), Some(serde_json::json!({"uiScale": 1.4})));
        let mut with_bom = UTF8_BOM.to_vec();
        with_bom.extend_from_slice(br#"{"uiScale":0.8}"#);
        std::fs::write(user.join("settings"), with_bom).unwrap();
        assert_eq!(read_persisted_settings(&dir), Some(serde_json::json!({"uiScale": 0.8})));
        // A damaged settings file falls back to the backup; with no usable backup it is the defaults.
        for damaged in ["", "{", "[1]", "null"] {
            std::fs::write(user.join("settings"), damaged).unwrap();
            assert_eq!(read_persisted_settings(&dir), Some(serde_json::json!({"uiScale": 1.2})), "{damaged:?}");
        }
        std::fs::write(user.join("settings.bak"), "garbage").unwrap();
        assert_eq!(read_persisted_settings(&dir), None);
        std::fs::remove_file(user.join("settings.bak")).unwrap();
        assert_eq!(read_persisted_settings(&dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---------------------------------------------------------------------------------------------
    // Wiring
    // ---------------------------------------------------------------------------------------------

    /// `main.rs` is the only place that says in what order the plugins run and which runs are skipped; a
    /// test on the pure functions cannot see it move. Offsets are compared, so line endings do not matter.
    #[test]
    fn main_registers_the_plugins_in_order_and_skips_minimized_runs() {
        let once = |needle: &str| {
            assert_eq!(MAIN_RS.matches(needle).count(), 1, "main.rs must contain `{needle}` exactly once");
            MAIN_RS.find(needle).unwrap()
        };
        let first_launch = once(".plugin(window_size::init(!start_minimized))");
        let restore = once(".plugin(window_state.build())");
        let fit = once(".plugin(window_size::init_after_restore(!start_minimized))");
        assert!(
            first_launch < restore,
            "the first-launch size must be registered BEFORE the window-state plugin, or its restore no longer wins over a wrong guess"
        );
        assert!(
            restore < fit,
            "the fit must be registered AFTER the window-state plugin, or it sees the window before the restore"
        );
        // SIGIL_START_MINIMIZED: the plugin must not restore (and show and focus) the main window, and
        // only then.
        let skip = once(r#"skip_initial_state("main")"#);
        let guard = MAIN_RS[..skip]
            .rfind("if start_minimized {")
            .expect("the skip sits under `if start_minimized {`");
        assert!(
            !MAIN_RS[guard..skip].contains('}'),
            "skip_initial_state(\"main\") must stay directly inside `if start_minimized {{ ... }}`"
        );
        assert!(guard < restore, "the skip must be set up before the plugin is built");
    }

    /// `SavedWindowState` copies the plugin's `WindowState` and `is_plugin_default` its `Default`. A newer
    /// plugin may add or rename a field, so the lock file has to say the version they were written against.
    #[test]
    fn window_state_mirror_matches_the_locked_plugin_version() {
        let mut lines = CARGO_LOCK.lines().map(str::trim);
        assert!(
            lines.any(|line| line == r#"name = "tauri-plugin-window-state""#),
            "Cargo.lock has no tauri-plugin-window-state entry"
        );
        let version = lines.next().expect("the entry has a version line");
        assert_eq!(
            version,
            format!(r#"version = "{MIRRORED_WINDOW_STATE_VERSION}""#),
            "tauri-plugin-window-state moved: compare its WindowState fields and Default impl with SavedWindowState and \
             SavedWindowState::is_plugin_default in window_size.rs, then raise MIRRORED_WINDOW_STATE_VERSION"
        );
    }
}
