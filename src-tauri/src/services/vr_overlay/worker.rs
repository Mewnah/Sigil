//! The overlay worker: one thread that owns the OpenVR session and the rasterizer.
//!
//! - **Waiting**: while the runtime pack is missing the state is `unavailable`; while SteamVR's
//!   server is not running it is `waiting` and nothing is initialized (SteamVR is never launched).
//!   Retries every [`Timing::retry`].
//! - **Running**: events are polled every [`Timing::poll`]; `VREvent_Quit` (or SteamVR's server
//!   disappearing) closes the session at once and goes back to waiting. Text is rendered at most
//!   once per [`Timing::frame`] (10 Hz) and only when it changed: `vr_overlay_set_text` bursts are
//!   coalesced, the latest wins. An empty caption hides the overlay.
//! - Commands arrive over a channel; [`Worker::stop_and_join`] ends the thread, closing the session
//!   (`DestroyOverlay` + `VR_ShutdownInternal`) on the way out.
//!
//! The OpenVR side and the rasterizer are traits so the state machine is tested without SteamVR or
//! fonts.

use std::sync::mpsc::{self, RecvTimeoutError, TryRecvError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::placement::{self, Mat34};
use super::render::Texture;
use super::settings::{Look, Normalized, MAX_TEXTURE_PIXELS, MIN_TEXTURE_PIXELS};
use super::{VrOverlayAttach, VrOverlayCaption, VrOverlayConfig, VrOverlayReason, VrOverlayState, VrOverlayStatus};

/// Why a connect attempt failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ConnectError {
    /// SteamVR is not running (or is starting / shutting down).
    NotRunning,
    /// SteamVR is not installed (or its install is broken).
    NotInstalled,
    /// SteamVR lacks the interface versions Sigil uses (update SteamVR).
    TooOld(String),
    Failed(String),
}

/// What one poll saw.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PollOutcome {
    /// SteamVR is quitting (or restarting): disconnect now.
    pub quit: bool,
    /// A tracked device appeared, left or changed role (hand attach looks for its controller again).
    pub devices_changed: bool,
}

/// The OpenVR runtime as the worker sees it.
pub(crate) trait VrBackend: Send {
    /// The runtime pack (`openvr_api.dll`) is installed.
    fn runtime_installed(&mut self) -> bool;
    /// Whether SteamVR's server runs; `None` when that cannot be told (then a connect is tried).
    fn server_running(&mut self) -> Option<bool>;
    /// Whether SteamVR is installed at all (for the waiting reason only).
    fn steamvr_installed(&mut self) -> Option<bool>;
    /// Initialize as a Background application and create the overlay.
    fn connect(&mut self) -> Result<Box<dyn VrSession>, ConnectError>;
}

/// A connected session with the overlay created. Dropping or closing it destroys the overlay and
/// shuts OpenVR down.
pub(crate) trait VrSession {
    fn poll(&mut self) -> PollOutcome;
    fn set_width(&mut self, metres: f32) -> Result<(), String>;
    fn set_alpha(&mut self, alpha: f32) -> Result<(), String>;
    fn set_texture(&mut self, texture: &Texture) -> Result<(), String>;
    fn set_visible(&mut self, visible: bool) -> Result<(), String>;
    fn place_relative(&mut self, device: u32, transform: &Mat34) -> Result<(), String>;
    fn place_absolute(&mut self, transform: &Mat34) -> Result<(), String>;
    /// The headset's pose in the standing universe, when tracked.
    fn hmd_pose(&mut self) -> Option<Mat34>;
    /// Tracked device index of the left or right controller, when connected.
    fn controller(&mut self, left: bool) -> Option<u32>;
    fn close(self: Box<Self>);
}

/// Text to texture.
pub(crate) trait Rasterizer: Send {
    fn render(&mut self, caption: &VrOverlayCaption, look: &Look) -> Texture;
}

/// Receives every status change (the plugin publishes it while the worker is current).
pub(crate) type StatusSink = Arc<dyn Fn(VrOverlayStatus) + Send + Sync>;

/// Worker timing.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Timing {
    /// Between connect attempts while waiting.
    pub retry: Duration,
    /// Between attempts while SteamVR is not installed or is too old.
    pub slow_retry: Duration,
    /// Event polling while connected.
    pub poll: Duration,
    /// Shortest time between two texture uploads (10 Hz).
    pub frame: Duration,
    /// Pause after `VREvent_Quit` before trying again (SteamVR is shutting down).
    pub quit_cooldown: Duration,
    /// Check that SteamVR's server still runs while connected.
    pub server_check: Duration,
    /// Look for a missing controller again (hand attach).
    pub device_check: Duration,
}

impl Timing {
    pub(crate) const PRODUCTION: Timing = Timing {
        retry: Duration::from_secs(3),
        slow_retry: Duration::from_secs(15),
        poll: Duration::from_millis(100),
        frame: Duration::from_millis(1000 / super::MAX_UPDATE_HZ as u64),
        quit_cooldown: Duration::from_secs(8),
        server_check: Duration::from_secs(4),
        device_check: Duration::from_secs(1),
    };
}

/// Messages to the worker.
pub(crate) enum Command {
    Config(Box<VrOverlayConfig>),
    Text(VrOverlayCaption),
    /// Place a world-attached overlay in front of the user again.
    Recenter,
    Stop,
}

/// Consecutive OpenVR call failures after which the session is dropped and reconnected.
const MAX_FAILURES: u32 = 10;

/// Handle of a running worker thread.
pub(crate) struct Worker {
    tx: mpsc::Sender<Command>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    pub(crate) fn spawn(
        backend: Box<dyn VrBackend>,
        rasterizer: Box<dyn Rasterizer>,
        config: &VrOverlayConfig,
        timing: Timing,
        sink: StatusSink,
    ) -> std::io::Result<Self> {
        let (tx, rx) = mpsc::channel();
        let config = Normalized::from_config(config);
        let thread = std::thread::Builder::new()
            .name("sigil-vr-overlay".into())
            .spawn(move || {
                let panic_sink = sink.clone();
                let mut state = Loop {
                    backend,
                    rasterizer,
                    timing,
                    sink,
                    config,
                    caption: VrOverlayCaption::default(),
                    live: None,
                    next_attempt: Instant::now(),
                    status: None,
                    pixel_cap: MAX_TEXTURE_PIXELS,
                };
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| state.run(&rx)));
                // Dropping the state closes a session `run` did not (the overlay goes, OpenVR shuts down).
                drop(state);
                if outcome.is_err() {
                    tracing::error!("vr_overlay: the worker panicked; the overlay is stopped");
                    panic_sink(VrOverlayStatus {
                        state: VrOverlayState::Error,
                        runtime_installed: true,
                        error: Some("The SteamVR overlay stopped unexpectedly. Turn it off and on again.".into()),
                        reason: None,
                    });
                }
            })?;
        Ok(Self { tx, thread: Some(thread) })
    }

    /// Queue a command (false when the thread has ended).
    pub(crate) fn send(&self, command: Command) -> bool {
        self.tx.send(command).is_ok()
    }

    pub(crate) fn is_alive(&self) -> bool {
        self.thread.as_ref().is_some_and(|t| !t.is_finished())
    }

    /// Stop the thread and wait up to `timeout` for it (true when it ended). A thread that does not
    /// end in time is left to finish on its own.
    pub(crate) fn stop_and_join(mut self, timeout: Duration) -> bool {
        let _ = self.tx.send(Command::Stop);
        let Some(thread) = self.thread.take() else { return true };
        let deadline = Instant::now() + timeout;
        while !thread.is_finished() {
            if Instant::now() >= deadline {
                tracing::warn!("vr_overlay: the worker did not stop within {timeout:?}");
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        if thread.join().is_err() {
            tracing::error!("vr_overlay: the worker panicked");
        }
        true
    }
}

/// State of a connected session.
struct Live {
    session: Box<dyn VrSession>,
    config_dirty: bool,
    placement_dirty: bool,
    text_dirty: bool,
    visible: bool,
    last_frame: Option<Instant>,
    next_server_check: Instant,
    /// Hand attach without its controller: shown at the head until the controller appears.
    waiting_for_device: bool,
    next_device_check: Instant,
    failures: u32,
    /// Consecutive refused texture uploads.
    upload_failures: u32,
    last_error: Option<String>,
}

struct Loop {
    backend: Box<dyn VrBackend>,
    rasterizer: Box<dyn Rasterizer>,
    timing: Timing,
    sink: StatusSink,
    config: Normalized,
    caption: VrOverlayCaption,
    live: Option<Live>,
    next_attempt: Instant,
    status: Option<VrOverlayStatus>,
    /// Largest texture area uploaded (pixels): lowered when SteamVR keeps refusing uploads, since
    /// OpenVR bounds `SetOverlayRaw`'s data without naming the bound. Kept for the worker's life.
    pixel_cap: u32,
}

impl Loop {
    fn run(&mut self, rx: &mpsc::Receiver<Command>) {
        'outer: loop {
            let timeout = self.timeout(Instant::now());
            match rx.recv_timeout(timeout) {
                Ok(command) => {
                    if !self.apply(command) {
                        break;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
            // Coalesce whatever else is queued (text bursts: the latest wins).
            loop {
                match rx.try_recv() {
                    Ok(command) => {
                        if !self.apply(command) {
                            break 'outer;
                        }
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => break 'outer,
                }
            }
            self.tick(Instant::now());
        }
        self.disconnect();
        self.publish(VrOverlayState::Off, None, None);
    }

    fn timeout(&self, now: Instant) -> Duration {
        match &self.live {
            None => self.next_attempt.saturating_duration_since(now),
            Some(live) => {
                let mut t = self.timing.poll;
                if live.text_dirty {
                    let due = live.last_frame.map_or(now, |f| f + self.timing.frame);
                    t = t.min(due.saturating_duration_since(now));
                }
                t
            }
        }
    }

    /// Apply a command; false to stop.
    fn apply(&mut self, command: Command) -> bool {
        match command {
            Command::Stop => return false,
            Command::Config(config) => {
                let next = Normalized::from_config(&config);
                if next != self.config {
                    let placement_changed =
                        next.attach != self.config.attach || next.distance_m != self.config.distance_m || next.offset_y_m != self.config.offset_y_m;
                    let look_changed = next.look != self.config.look;
                    self.config = next;
                    if let Some(live) = &mut self.live {
                        live.config_dirty = true;
                        live.placement_dirty |= placement_changed;
                        live.text_dirty |= look_changed;
                    }
                }
            }
            Command::Text(caption) => {
                if caption != self.caption {
                    self.caption = caption;
                    if let Some(live) = &mut self.live {
                        live.text_dirty = true;
                    }
                }
            }
            Command::Recenter => {
                if let Some(live) = &mut self.live {
                    live.placement_dirty = true;
                }
            }
        }
        true
    }

    fn publish(&mut self, state: VrOverlayState, reason: Option<VrOverlayReason>, error: Option<String>) {
        let status = VrOverlayStatus {
            state,
            runtime_installed: state != VrOverlayState::Unavailable,
            error,
            reason,
        };
        if self.status.as_ref() != Some(&status) {
            if self.status.as_ref().map(|s| s.state) != Some(state) {
                tracing::info!(?state, ?reason, "vr_overlay: state");
            }
            self.status = Some(status.clone());
            (self.sink)(status);
        }
    }

    fn disconnect(&mut self) {
        if let Some(live) = self.live.take() {
            live.session.close();
        }
    }

    fn tick(&mut self, now: Instant) {
        if self.live.is_none() {
            if now >= self.next_attempt {
                self.try_connect(now);
            }
            return;
        }
        self.tick_live(now);
    }

    fn try_connect(&mut self, now: Instant) {
        self.next_attempt = now + self.timing.retry;
        if !self.backend.runtime_installed() {
            self.publish(VrOverlayState::Unavailable, Some(VrOverlayReason::RuntimeMissing), None);
            return;
        }
        if self.backend.server_running() == Some(false) {
            let reason = if self.backend.steamvr_installed() == Some(false) {
                self.next_attempt = now + self.timing.slow_retry;
                VrOverlayReason::SteamvrNotInstalled
            } else {
                VrOverlayReason::SteamvrNotRunning
            };
            self.publish(VrOverlayState::Waiting, Some(reason), None);
            return;
        }
        match self.backend.connect() {
            Ok(session) => {
                self.live = Some(Live {
                    session,
                    config_dirty: true,
                    placement_dirty: true,
                    text_dirty: true,
                    visible: false,
                    last_frame: None,
                    next_server_check: now + self.timing.server_check,
                    waiting_for_device: false,
                    next_device_check: now,
                    failures: 0,
                    upload_failures: 0,
                    last_error: None,
                });
                self.publish(VrOverlayState::Running, None, None);
                self.tick_live(now);
            }
            Err(ConnectError::NotRunning) => self.publish(VrOverlayState::Waiting, Some(VrOverlayReason::SteamvrNotRunning), None),
            Err(ConnectError::NotInstalled) => {
                self.next_attempt = now + self.timing.slow_retry;
                self.publish(VrOverlayState::Waiting, Some(VrOverlayReason::SteamvrNotInstalled), None);
            }
            Err(ConnectError::TooOld(why)) => {
                self.next_attempt = now + self.timing.slow_retry;
                self.publish(VrOverlayState::Error, Some(VrOverlayReason::SteamvrTooOld), Some(why));
            }
            Err(ConnectError::Failed(why)) => {
                tracing::warn!("vr_overlay: connecting to SteamVR failed: {why}");
                self.publish(VrOverlayState::Waiting, Some(VrOverlayReason::ConnectFailed), Some(why));
            }
        }
    }

    fn tick_live(&mut self, now: Instant) {
        let Some(live) = self.live.as_mut() else { return };
        let polled = live.session.poll();
        if polled.quit {
            self.disconnect();
            self.next_attempt = now + self.timing.quit_cooldown;
            self.publish(VrOverlayState::Waiting, Some(VrOverlayReason::SteamvrQuit), None);
            return;
        }
        if now >= live.next_server_check {
            live.next_server_check = now + self.timing.server_check;
            if self.backend.server_running() == Some(false) {
                self.disconnect();
                self.next_attempt = now + self.timing.retry;
                self.publish(VrOverlayState::Waiting, Some(VrOverlayReason::SteamvrNotRunning), None);
                return;
            }
        }
        // A hand overlay looks for its controller again when devices change (a controller turned off,
        // or the hands swapped roles); one shown at the head meanwhile keeps looking.
        let on_hand = matches!(self.config.attach, VrOverlayAttach::LeftHand | VrOverlayAttach::RightHand);
        if (polled.devices_changed && (on_hand || live.waiting_for_device)) || (live.waiting_for_device && now >= live.next_device_check) {
            live.placement_dirty = true;
        }

        // Only ticks that called OpenVR count for the failure tally (a quiet tick proves nothing).
        let mut attempted = false;
        let mut errors: Vec<String> = Vec::new();
        let config = &self.config;
        if live.config_dirty {
            live.config_dirty = false;
            attempted = true;
            errors.extend(live.session.set_width(config.width_m).err());
            errors.extend(live.session.set_alpha(config.opacity).err());
        }
        if live.placement_dirty {
            live.placement_dirty = false;
            attempted = true;
            errors.extend(place(live, config, now, self.timing.device_check).err());
        }
        if live.text_dirty && live.last_frame.is_none_or(|f| now >= f + self.timing.frame) {
            live.text_dirty = false;
            attempted = true;
            live.last_frame = Some(now);
            let look = Look {
                max_pixels: config.look.max_pixels.min(self.pixel_cap),
                ..config.look.clone()
            };
            let texture = self.rasterizer.render(&self.caption, &look);
            let s = &texture.stats;
            tracing::trace!(
                rows = s.rows_shown,
                rows_total = s.rows_total,
                glyphs = s.glyphs,
                scale = s.scale,
                "vr_overlay: frame {}x{}",
                texture.width,
                texture.height
            );
            if s.missing_glyphs > 0 {
                tracing::debug!(missing = s.missing_glyphs, "vr_overlay: no installed font has some caption characters");
            }
            if texture.is_blank() {
                if live.visible {
                    match live.session.set_visible(false) {
                        Ok(()) => live.visible = false,
                        Err(e) => errors.push(e),
                    }
                }
            } else {
                let uploaded = live.session.set_texture(&texture);
                live.upload_failures = if uploaded.is_ok() { 0 } else { live.upload_failures + 1 };
                match uploaded {
                    Ok(()) if !live.visible => match live.session.set_visible(true) {
                        Ok(()) => live.visible = true,
                        Err(e) => errors.push(e),
                    },
                    Ok(()) => {}
                    Err(e) => {
                        // Try again with the next frame; after two refusals in a row, smaller.
                        live.text_dirty = true;
                        let area = texture.width.saturating_mul(texture.height);
                        let lower = (area / 2).max(MIN_TEXTURE_PIXELS);
                        if live.upload_failures >= 2 && lower < area && lower < self.pixel_cap {
                            self.pixel_cap = lower;
                            live.upload_failures = 0;
                            tracing::warn!(pixels = lower, "vr_overlay: SteamVR refused the texture; rendering it smaller");
                        }
                        errors.push(e);
                    }
                }
            }
        }

        if !attempted {
            return;
        }
        if errors.is_empty() {
            if live.failures > 0 || live.last_error.is_some() {
                live.failures = 0;
                live.last_error = None;
                self.publish(VrOverlayState::Running, None, None);
            }
            return;
        }
        live.failures += 1;
        let message = errors.join("; ");
        tracing::warn!(failures = live.failures, "vr_overlay: {message}");
        live.last_error = Some(message.clone());
        if live.failures >= MAX_FAILURES {
            self.disconnect();
            self.next_attempt = now + self.timing.retry;
            self.publish(VrOverlayState::Waiting, Some(VrOverlayReason::ConnectFailed), Some(message));
        } else {
            self.publish(VrOverlayState::Running, Some(VrOverlayReason::OverlayError), Some(message));
        }
    }
}

/// Place the overlay for the configured attach mode. A hand without its controller (or a world
/// placement without a tracked headset) is shown at the head until it can be placed.
fn place(live: &mut Live, config: &Normalized, now: Instant, device_check: Duration) -> Result<(), String> {
    let head = placement::head(config.distance_m, config.offset_y_m);
    live.waiting_for_device = false;
    let fallback = |live: &mut Live| {
        live.waiting_for_device = true;
        live.next_device_check = now + device_check;
        live.session
            .place_relative(super::openvr::HMD_DEVICE, &head)
    };
    match config.attach {
        VrOverlayAttach::Head => live
            .session
            .place_relative(super::openvr::HMD_DEVICE, &head),
        VrOverlayAttach::LeftHand | VrOverlayAttach::RightHand => {
            match live
                .session
                .controller(config.attach == VrOverlayAttach::LeftHand)
            {
                Some(device) => live.session.place_relative(device, &placement::hand()),
                None => fallback(live),
            }
        }
        VrOverlayAttach::World => match live.session.hmd_pose() {
            Some(pose) => live
                .session
                .place_absolute(&placement::world(&pose, config.distance_m, config.offset_y_m)),
            None => fallback(live),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::vr_overlay::render::RenderStats;
    use crate::services::vr_overlay::{VrOverlayLine, VrOverlayLineKind};
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::Mutex;

    /// What the fake SteamVR saw and how it behaves.
    #[derive(Default)]
    struct World {
        runtime_installed: AtomicBool,
        server_running: AtomicBool,
        /// Init attempts (successful or not).
        attempts: AtomicU32,
        connects: AtomicU32,
        closes: AtomicU32,
        uploads: AtomicU32,
        quit_next_poll: AtomicBool,
        devices_changed_next_poll: AtomicBool,
        controller_present: AtomicBool,
        calls: Mutex<Vec<String>>,
        fail_uploads: AtomicBool,
        /// Refuse textures larger than this many pixels (0: no limit).
        max_upload_pixels: AtomicU32,
    }

    impl World {
        fn new() -> Arc<Self> {
            let w = Arc::new(World::default());
            w.runtime_installed.store(true, Ordering::SeqCst);
            w
        }

        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }

        fn log(&self, s: impl Into<String>) {
            self.calls.lock().unwrap().push(s.into());
        }
    }

    struct FakeBackend(Arc<World>);
    struct FakeSession(Arc<World>);

    impl VrBackend for FakeBackend {
        fn runtime_installed(&mut self) -> bool {
            self.0.runtime_installed.load(Ordering::SeqCst)
        }
        fn server_running(&mut self) -> Option<bool> {
            Some(self.0.server_running.load(Ordering::SeqCst))
        }
        fn steamvr_installed(&mut self) -> Option<bool> {
            Some(true)
        }
        fn connect(&mut self) -> Result<Box<dyn VrSession>, ConnectError> {
            self.0.attempts.fetch_add(1, Ordering::SeqCst);
            // A Background init: refused while the server is not running (never launches it).
            if !self.0.server_running.load(Ordering::SeqCst) {
                return Err(ConnectError::NotRunning);
            }
            self.0.connects.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(FakeSession(self.0.clone())))
        }
    }

    impl VrSession for FakeSession {
        fn poll(&mut self) -> PollOutcome {
            PollOutcome {
                quit: self.0.quit_next_poll.swap(false, Ordering::SeqCst),
                devices_changed: self
                    .0
                    .devices_changed_next_poll
                    .swap(false, Ordering::SeqCst),
            }
        }
        fn set_width(&mut self, m: f32) -> Result<(), String> {
            self.0.log(format!("width {m}"));
            Ok(())
        }
        fn set_alpha(&mut self, a: f32) -> Result<(), String> {
            self.0.log(format!("alpha {a}"));
            Ok(())
        }
        fn set_texture(&mut self, t: &Texture) -> Result<(), String> {
            let max = self.0.max_upload_pixels.load(Ordering::SeqCst);
            if self.0.fail_uploads.load(Ordering::SeqCst) || (max > 0 && t.width * t.height > max) {
                return Err("upload refused".into());
            }
            self.0.uploads.fetch_add(1, Ordering::SeqCst);
            self.0.log(format!("texture {}x{}", t.width, t.height));
            Ok(())
        }
        fn set_visible(&mut self, v: bool) -> Result<(), String> {
            self.0.log(if v { "show" } else { "hide" });
            Ok(())
        }
        fn place_relative(&mut self, device: u32, _: &Mat34) -> Result<(), String> {
            self.0.log(format!("relative {device}"));
            Ok(())
        }
        fn place_absolute(&mut self, _: &Mat34) -> Result<(), String> {
            self.0.log("absolute");
            Ok(())
        }
        fn hmd_pose(&mut self) -> Option<Mat34> {
            Some(placement::IDENTITY)
        }
        fn controller(&mut self, left: bool) -> Option<u32> {
            self.0
                .controller_present
                .load(Ordering::SeqCst)
                .then_some(if left { 3 } else { 4 })
        }
        fn close(self: Box<Self>) {
            // Counted in Drop, like the real session (closing and dropping are the same).
        }
    }

    impl Drop for FakeSession {
        fn drop(&mut self) {
            self.0.closes.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Blank for empty captions; a 1024 x 167 texture (the default look's size) shrunk to the look's
    /// pixel cap otherwise; logs what it rendered. Panics on the text "panic" (a renderer bug).
    struct FakeRaster(Arc<World>);
    impl Rasterizer for FakeRaster {
        fn render(&mut self, caption: &VrOverlayCaption, look: &Look) -> Texture {
            assert!(!caption.lines.iter().any(|l| l.text == "panic"), "renderer bug");
            let shown = caption
                .lines
                .iter()
                .filter(|l| !l.text.trim().is_empty())
                .count();
            let text: Vec<&str> = caption.lines.iter().map(|l| l.text.as_str()).collect();
            self.0.log(format!("render {}", text.join("|")));
            Texture {
                width: 1024,
                height: (look.max_pixels / 1024).clamp(1, 167),
                rgba: Vec::new(),
                stats: RenderStats {
                    rows_shown: shown,
                    ..RenderStats::default()
                },
            }
        }
    }

    const FAST: Timing = Timing {
        retry: Duration::from_millis(20),
        slow_retry: Duration::from_millis(40),
        poll: Duration::from_millis(5),
        frame: Duration::from_millis(100),
        quit_cooldown: Duration::from_millis(30),
        server_check: Duration::from_millis(15),
        device_check: Duration::from_millis(10),
    };

    struct Harness {
        world: Arc<World>,
        worker: Option<Worker>,
        statuses: Arc<Mutex<Vec<VrOverlayStatus>>>,
    }

    impl Harness {
        fn start(world: Arc<World>, config: VrOverlayConfig) -> Self {
            let statuses = Arc::new(Mutex::new(Vec::new()));
            let sink_statuses = statuses.clone();
            let sink: StatusSink = Arc::new(move |s| sink_statuses.lock().unwrap().push(s));
            let worker = Worker::spawn(
                Box::new(FakeBackend(world.clone())),
                Box::new(FakeRaster(world.clone())),
                &config,
                FAST,
                sink,
            )
            .unwrap();
            Self {
                world,
                worker: Some(worker),
                statuses,
            }
        }

        fn last(&self) -> Option<VrOverlayStatus> {
            self.statuses.lock().unwrap().last().cloned()
        }

        fn wait_for(&self, what: &str, f: impl Fn(&Self) -> bool) {
            let deadline = Instant::now() + Duration::from_secs(5);
            while !f(self) {
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for {what}; statuses {:?}",
                    self.statuses.lock().unwrap()
                );
                std::thread::sleep(Duration::from_millis(2));
            }
        }

        fn state(&self) -> Option<VrOverlayState> {
            self.last().map(|s| s.state)
        }

        fn send(&self, c: Command) {
            assert!(self.worker.as_ref().unwrap().send(c));
        }

        fn stop(&mut self) {
            assert!(self
                .worker
                .take()
                .unwrap()
                .stop_and_join(Duration::from_secs(5)));
        }
    }

    fn text(s: &str) -> VrOverlayCaption {
        VrOverlayCaption {
            lines: vec![VrOverlayLine {
                text: s.into(),
                kind: VrOverlayLineKind::Final,
                source: "text.stt".into(),
                lang: None,
            }],
        }
    }

    #[test]
    fn unavailable_without_the_runtime_pack() {
        let world = World::new();
        world.runtime_installed.store(false, Ordering::SeqCst);
        let mut h = Harness::start(world.clone(), VrOverlayConfig::default());
        h.wait_for("unavailable", |h| h.state() == Some(VrOverlayState::Unavailable));
        let last = h.last().unwrap();
        assert!(!last.runtime_installed);
        assert_eq!(last.reason, Some(VrOverlayReason::RuntimeMissing));
        // Installing the pack later is picked up without a restart.
        world.runtime_installed.store(true, Ordering::SeqCst);
        h.wait_for("waiting", |h| h.state() == Some(VrOverlayState::Waiting));
        h.stop();
        assert_eq!(world.connects.load(Ordering::SeqCst), 0);
        assert_eq!(h.state(), Some(VrOverlayState::Off));
    }

    #[test]
    fn never_initializes_while_steamvr_is_not_running() {
        let world = World::new();
        let mut h = Harness::start(world.clone(), VrOverlayConfig::default());
        h.wait_for("waiting", |h| h.state() == Some(VrOverlayState::Waiting));
        assert_eq!(h.last().unwrap().reason, Some(VrOverlayReason::SteamvrNotRunning));
        // Several retry periods pass without a single init attempt.
        std::thread::sleep(FAST.retry * 5);
        assert_eq!(world.attempts.load(Ordering::SeqCst), 0);
        // SteamVR starts: the overlay connects on its own.
        world.server_running.store(true, Ordering::SeqCst);
        h.wait_for("running", |h| h.state() == Some(VrOverlayState::Running));
        assert_eq!(world.connects.load(Ordering::SeqCst), 1);
        h.stop();
        assert_eq!(world.closes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn quit_event_disconnects_and_waits() {
        let world = World::new();
        world.server_running.store(true, Ordering::SeqCst);
        let mut h = Harness::start(world.clone(), VrOverlayConfig::default());
        h.wait_for("running", |h| h.state() == Some(VrOverlayState::Running));
        world.server_running.store(false, Ordering::SeqCst);
        world.quit_next_poll.store(true, Ordering::SeqCst);
        h.wait_for("quit", |h| {
            h.last()
                .is_some_and(|s| s.reason == Some(VrOverlayReason::SteamvrQuit))
        });
        assert_eq!(h.state(), Some(VrOverlayState::Waiting));
        assert_eq!(world.closes.load(Ordering::SeqCst), 1);
        // SteamVR comes back: reconnects after the cooldown.
        world.server_running.store(true, Ordering::SeqCst);
        h.wait_for("reconnected", |h| h.world.connects.load(Ordering::SeqCst) == 2);
        h.stop();
        assert_eq!(world.closes.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn server_gone_without_quit_disconnects() {
        let world = World::new();
        world.server_running.store(true, Ordering::SeqCst);
        let mut h = Harness::start(world.clone(), VrOverlayConfig::default());
        h.wait_for("running", |h| h.state() == Some(VrOverlayState::Running));
        world.server_running.store(false, Ordering::SeqCst);
        h.wait_for("waiting", |h| h.state() == Some(VrOverlayState::Waiting));
        assert_eq!(world.closes.load(Ordering::SeqCst), 1);
        h.stop();
    }

    #[test]
    fn text_bursts_are_coalesced_to_at_most_10_hz() {
        let world = World::new();
        world.server_running.store(true, Ordering::SeqCst);
        let mut h = Harness::start(world.clone(), VrOverlayConfig::default());
        h.wait_for("running", |h| h.state() == Some(VrOverlayState::Running));
        let started = Instant::now();
        for i in 0..200 {
            h.send(Command::Text(text(&format!("line {i}"))));
            std::thread::sleep(Duration::from_millis(1));
        }
        // Let the last frame land.
        std::thread::sleep(FAST.frame * 2);
        let elapsed = started.elapsed();
        let uploads = world.uploads.load(Ordering::SeqCst);
        let budget = (elapsed.as_millis() / FAST.frame.as_millis()) as u32 + 2;
        assert!(uploads >= 1 && uploads <= budget, "{uploads} uploads in {elapsed:?}");
        // The latest text is the one on screen.
        let renders: Vec<String> = world
            .calls()
            .into_iter()
            .filter(|c| c.starts_with("render "))
            .collect();
        assert_eq!(renders.last().map(String::as_str), Some("render line 199"));
        assert!(world.calls().iter().any(|c| c == "show"));
        h.stop();
    }

    #[test]
    fn empty_caption_hides_and_text_shows() {
        let world = World::new();
        world.server_running.store(true, Ordering::SeqCst);
        let mut h = Harness::start(world.clone(), VrOverlayConfig::default());
        h.wait_for("running", |h| h.state() == Some(VrOverlayState::Running));
        h.send(Command::Text(text("hello")));
        h.wait_for("show", |h| h.world.calls().iter().any(|c| c == "show"));
        h.send(Command::Text(VrOverlayCaption::default()));
        h.wait_for("hide", |h| h.world.calls().iter().any(|c| c == "hide"));
        h.stop();
    }

    #[test]
    fn hand_attach_falls_back_to_the_head_until_the_controller_appears() {
        let world = World::new();
        world.server_running.store(true, Ordering::SeqCst);
        let config = VrOverlayConfig {
            attach: VrOverlayAttach::RightHand,
            ..VrOverlayConfig::default()
        };
        let mut h = Harness::start(world.clone(), config);
        h.wait_for("head fallback", |h| h.world.calls().iter().any(|c| c == "relative 0"));
        world.controller_present.store(true, Ordering::SeqCst);
        h.wait_for("hand", |h| h.world.calls().iter().any(|c| c == "relative 4"));
        h.stop();
    }

    #[test]
    fn hand_attach_follows_device_changes() {
        let world = World::new();
        world.server_running.store(true, Ordering::SeqCst);
        world.controller_present.store(true, Ordering::SeqCst);
        let config = VrOverlayConfig {
            attach: VrOverlayAttach::LeftHand,
            ..VrOverlayConfig::default()
        };
        let mut h = Harness::start(world.clone(), config);
        h.wait_for("hand", |h| h.world.calls().iter().any(|c| c == "relative 3"));
        // The controller turns off: the overlay moves to the head instead of staying on a lost device.
        world.controller_present.store(false, Ordering::SeqCst);
        world
            .devices_changed_next_poll
            .store(true, Ordering::SeqCst);
        h.wait_for("head fallback", |h| h.world.calls().iter().any(|c| c == "relative 0"));
        // It comes back to the hand on its own.
        world.controller_present.store(true, Ordering::SeqCst);
        h.wait_for("hand again", |h| {
            h.world
                .calls()
                .iter()
                .filter(|c| *c == "relative 3")
                .count()
                >= 2
        });
        h.stop();
    }

    #[test]
    fn world_attach_places_once_and_on_recenter() {
        let world = World::new();
        world.server_running.store(true, Ordering::SeqCst);
        let config = VrOverlayConfig {
            attach: VrOverlayAttach::World,
            ..VrOverlayConfig::default()
        };
        let mut h = Harness::start(world.clone(), config.clone());
        h.wait_for("placed", |h| h.world.calls().iter().any(|c| c == "absolute"));
        // A look-only change does not move it.
        h.send(Command::Config(Box::new(VrOverlayConfig {
            font_size_px: 48,
            ..config.clone()
        })));
        std::thread::sleep(FAST.poll * 6);
        let placed = |w: &World| w.calls().iter().filter(|c| *c == "absolute").count();
        assert_eq!(placed(&world), 1);
        h.send(Command::Recenter);
        h.wait_for("re-placed", |h| placed(&h.world) == 2);
        h.stop();
    }

    #[test]
    fn config_changes_reach_the_session() {
        let world = World::new();
        world.server_running.store(true, Ordering::SeqCst);
        let mut h = Harness::start(world.clone(), VrOverlayConfig::default());
        h.wait_for("width", |h| h.world.calls().iter().any(|c| c == "width 0.6"));
        h.send(Command::Config(Box::new(VrOverlayConfig {
            width_m: 1.0,
            opacity: 0.5,
            ..VrOverlayConfig::default()
        })));
        h.wait_for("new width", |h| h.world.calls().iter().any(|c| c == "width 1"));
        assert!(world.calls().iter().any(|c| c == "alpha 0.5"));
        h.stop();
    }

    #[test]
    fn repeated_upload_failures_reconnect() {
        let world = World::new();
        world.server_running.store(true, Ordering::SeqCst);
        world.fail_uploads.store(true, Ordering::SeqCst);
        let mut h = Harness::start(world.clone(), VrOverlayConfig::default());
        h.wait_for("running", |h| h.state() == Some(VrOverlayState::Running));
        h.send(Command::Text(text("x")));
        h.wait_for("error reported", |h| {
            h.last()
                .is_some_and(|s| s.reason == Some(VrOverlayReason::OverlayError))
        });
        h.wait_for("reconnect", |h| h.world.connects.load(Ordering::SeqCst) >= 2);
        world.fail_uploads.store(false, Ordering::SeqCst);
        h.wait_for("recovered", |h| h.world.uploads.load(Ordering::SeqCst) >= 1);
        h.stop();
    }

    #[test]
    fn refused_uploads_render_smaller() {
        let world = World::new();
        world.server_running.store(true, Ordering::SeqCst);
        // This "SteamVR" takes at most 140K pixels; the default look is 1024 x 167 (171K).
        world.max_upload_pixels.store(1024 * 140, Ordering::SeqCst);
        let mut h = Harness::start(world.clone(), VrOverlayConfig::default());
        h.wait_for("running", |h| h.state() == Some(VrOverlayState::Running));
        h.send(Command::Text(text("x")));
        h.wait_for("uploaded smaller", |h| h.world.uploads.load(Ordering::SeqCst) >= 1);
        assert!(world.calls().iter().any(|c| c == "texture 1024x128"), "{:?}", world.calls());
        h.wait_for("recovered", |h| {
            h.last()
                .is_some_and(|s| s.state == VrOverlayState::Running && s.error.is_none())
        });
        assert_eq!(world.connects.load(Ordering::SeqCst), 1, "no reconnect needed");
        h.stop();
    }

    #[test]
    fn a_worker_panic_closes_the_session_and_reports_an_error() {
        let world = World::new();
        world.server_running.store(true, Ordering::SeqCst);
        let mut h = Harness::start(world.clone(), VrOverlayConfig::default());
        h.wait_for("running", |h| h.state() == Some(VrOverlayState::Running));
        h.send(Command::Text(text("panic")));
        h.wait_for("error", |h| h.state() == Some(VrOverlayState::Error));
        assert_eq!(world.closes.load(Ordering::SeqCst), 1, "the session was closed");
        let worker = h.worker.take().unwrap();
        // The thread has ended: the plugin starts a fresh worker on the next start.
        let deadline = Instant::now() + Duration::from_secs(5);
        while worker.is_alive() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(!worker.send(Command::Recenter));
        assert!(worker.stop_and_join(Duration::from_secs(1)));
    }
}
