//! Recovery from a WebView2 renderer failure of the host page (W3).
//!
//! Every caption passes through the host page: the Rust speech pipeline hands finals to the page's
//! JavaScript, which publishes them to OBS, the chatbox and the rest (`plugin:web|pubsub_broadcast`
//! is only ever called from there). When the page's renderer process exits (out of memory, a cleaner
//! killing `msedgewebview2.exe`) or stops responding, the window shows a blank or sad page and
//! nothing publishes until someone restarts Sigil: wry registers no `ProcessFailed` handler, so the
//! failure went unnoticed.
//!
//! [`watch`] adds one to the main window. [`RecoveryPolicy`] decides, with time passed in so it is
//! tested without a webview:
//!
//! - **The renderer exited** (`RENDER_PROCESS_EXITED`): the page is gone, so it is reloaded after a
//!   back-off ([`BACKOFF`]: 1 s, 5 s, 15 s) on a timer, never inline, so a renderer that dies while
//!   loading is not hammered. [`MAX_RELOADS`] reloads are allowed within [`RELOAD_WINDOW`]; when
//!   that budget is spent the failure is logged once (error level, once per episode) and one more
//!   reload is scheduled for the moment the oldest one leaves the window, so a burst of failures (a
//!   GPU driver reset, memory pressure) does not leave a blank window for good. A failure that comes
//!   while a reload is already scheduled adds nothing.
//! - **The renderer stopped responding** (`RENDER_PROCESS_UNRESPONSIVE`): WebView2 raises this again
//!   every few seconds for as long as the renderer is only busy or hung (a PC loaded by VRChat and
//!   OBS, a wake from sleep), so it is not a crash. The first event of an episode is logged and
//!   nothing else happens; only an episode that lasts [`HANG_RELOAD_AFTER`] reloads the page, once,
//!   from its own small budget ([`MAX_HANG_RELOADS`] per [`HANG_RELOAD_WINDOW`]) that the crash budget
//!   does not share.
//! - Before a reload (and when one is scheduled), the `mic` speech session of the failed page is
//!   stopped from Rust: its events went to a page that is gone and no new page can adopt it, so
//!   the microphone and the engine would stay busy under a dead or blank window. The next page
//!   stops whatever is left again (`Service_STT.init`) and, with "Start automatically" on, starts
//!   captions again. OBS `/client` mirrors reconnect by themselves.
//!
//! Not covered: a failure of the WebView2 browser process itself (every webview of the environment is
//! gone; only recreating the window recovers it), and GPU or utility process exits (WebView2
//! restarts those on its own). Both are logged.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use webview2_com::Microsoft::Web::WebView2::Win32::{
    ICoreWebView2, ICoreWebView2ProcessFailedEventArgs, ICoreWebView2ProcessFailedEventArgs2, COREWEBVIEW2_PROCESS_FAILED_KIND,
    COREWEBVIEW2_PROCESS_FAILED_REASON,
};
use webview2_com::ProcessFailedEventHandler;
use windows::core::Interface;

/// Reloads after a renderer exit allowed within [`RELOAD_WINDOW`].
const MAX_RELOADS: usize = 3;
/// The span [`MAX_RELOADS`] is counted over.
const RELOAD_WINDOW: Duration = Duration::from_secs(120);
/// Wait before the n-th reload of the window (n = reloads already made within [`RELOAD_WINDOW`]).
const BACKOFF: [Duration; MAX_RELOADS] = [Duration::from_secs(1), Duration::from_secs(5), Duration::from_secs(15)];
/// Added to the wait for the oldest reload to leave the window, so it has left when the timer fires.
const DEFER_MARGIN: Duration = Duration::from_millis(250);

/// Events of one unresponsive episode are at most this far apart (WebView2 raises them every few seconds).
const HANG_EPISODE_GAP: Duration = Duration::from_secs(20);
/// An unresponsive episode this long reloads the page (once).
const HANG_RELOAD_AFTER: Duration = Duration::from_secs(30);
/// Reloads for unresponsive episodes allowed within [`HANG_RELOAD_WINDOW`]; separate from [`MAX_RELOADS`].
const MAX_HANG_RELOADS: usize = 2;
/// The span [`MAX_HANG_RELOADS`] is counted over.
const HANG_RELOAD_WINDOW: Duration = Duration::from_secs(600);

/// `COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_EXITED`.
const KIND_RENDER_PROCESS_EXITED: i32 = 1;
/// `COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_UNRESPONSIVE`.
const KIND_RENDER_PROCESS_UNRESPONSIVE: i32 = 2;

/// Which process failed, as far as recovery cares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Failure {
    /// The page's renderer process exited.
    RenderProcessExited,
    /// The page's renderer process stopped responding (raised again every few seconds while it lasts).
    RenderProcessUnresponsive,
    /// Anything else (browser, GPU, utility, frame renderer ...).
    Other,
}

impl Failure {
    /// From a `COREWEBVIEW2_PROCESS_FAILED_KIND` value.
    pub(crate) fn from_kind(kind: i32) -> Self {
        match kind {
            KIND_RENDER_PROCESS_EXITED => Failure::RenderProcessExited,
            KIND_RENDER_PROCESS_UNRESPONSIVE => Failure::RenderProcessUnresponsive,
            _ => Failure::Other,
        }
    }
}

/// What to do about a failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Action {
    /// Reload the page after `delay` (a back-off step), on a timer.
    Reload { delay: Duration },
    /// The reload budget is spent: one more reload is scheduled for `delay` from now, when the oldest
    /// reload has left the window. `first`: the first deferral of this episode (logged at error level).
    Defer { delay: Duration, first: bool },
    /// A reload or deferred retry is already scheduled: this failure adds nothing.
    Pending,
    /// The first event of an unresponsive episode: log it, wait.
    NoteHung,
    /// The renderer has not responded for [`HANG_RELOAD_AFTER`]: reload the page now.
    ReloadHung,
    /// An unresponsive episode this long, but the hang budget is spent: logged once, left alone.
    HungLeftAlone,
    /// Nothing to do (a process a reload does not fix, or one more event of an episode already handled).
    Ignore,
}

impl Action {
    /// The page is about to be replaced, so the `mic` speech session its JavaScript owned is stopped.
    pub(crate) fn releases_microphone(self) -> bool {
        matches!(self, Action::Reload { .. } | Action::Defer { .. } | Action::ReloadHung)
    }
}

/// What a fired reload timer does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fire {
    /// Reload the page now (counted in the budget).
    Reload,
    /// The budget is still spent (the timer ran early): ask again after this wait.
    Wait(Duration),
}

/// One unresponsive episode.
#[derive(Clone, Copy, Debug)]
struct Hang {
    since: Instant,
    last: Instant,
    /// The page was reloaded (or left alone) for this episode already.
    handled: bool,
}

/// Decides what a WebView2 process failure does: back-off, a reload budget that retries later
/// instead of giving up, and a separate path for a renderer that is only unresponsive.
#[derive(Debug, Default)]
pub(crate) struct RecoveryPolicy {
    /// When the page was reloaded after a renderer exit, within [`RELOAD_WINDOW`].
    reloads: VecDeque<Instant>,
    /// A reload (or deferred retry) is scheduled on a timer.
    pending: bool,
    /// The deferral was logged in this episode (cleared when the page held for a whole window).
    announced: bool,
    hang: Option<Hang>,
    /// When the page was reloaded for an unresponsive episode, within [`HANG_RELOAD_WINDOW`].
    hang_reloads: VecDeque<Instant>,
}

impl RecoveryPolicy {
    pub(crate) fn decide(&mut self, failure: Failure, now: Instant) -> Action {
        match failure {
            Failure::Other => Action::Ignore,
            Failure::RenderProcessExited => {
                // A renderer that exited is no longer hung.
                self.hang = None;
                self.exited(now)
            }
            Failure::RenderProcessUnresponsive => self.unresponsive(now),
        }
    }

    /// The timer of a scheduled reload fired.
    pub(crate) fn fire(&mut self, now: Instant) -> Fire {
        self.prune(now);
        if self.reloads.len() < MAX_RELOADS {
            self.pending = false;
            self.reloads.push_back(now);
            // The new page has a new renderer: an unresponsive episode of the old one is over.
            self.hang = None;
            return Fire::Reload;
        }
        Fire::Wait(self.until_a_slot_frees(now))
    }

    /// The timer could not be started: allow the next failure to schedule again.
    pub(crate) fn cancel(&mut self) {
        self.pending = false;
    }

    fn exited(&mut self, now: Instant) -> Action {
        if self.pending {
            return Action::Pending;
        }
        self.prune(now);
        self.pending = true;
        if let Some(delay) = BACKOFF.get(self.reloads.len()) {
            return Action::Reload { delay: *delay };
        }
        let first = !std::mem::replace(&mut self.announced, true);
        Action::Defer {
            delay: self.until_a_slot_frees(now),
            first,
        }
    }

    fn unresponsive(&mut self, now: Instant) -> Action {
        if self.pending {
            // The renderer exited and a reload is on its way: late events of the dead renderer mean nothing.
            return Action::Ignore;
        }
        let continues = self
            .hang
            .is_some_and(|h| now.saturating_duration_since(h.last) <= HANG_EPISODE_GAP);
        if !continues {
            self.hang = Some(Hang {
                since: now,
                last: now,
                handled: false,
            });
            return Action::NoteHung;
        }
        let Some(hang) = self.hang.as_mut() else {
            return Action::Ignore;
        };
        hang.last = now;
        if hang.handled || now.saturating_duration_since(hang.since) < HANG_RELOAD_AFTER {
            return Action::Ignore;
        }
        hang.handled = true;
        while self
            .hang_reloads
            .front()
            .is_some_and(|t| now.saturating_duration_since(*t) > HANG_RELOAD_WINDOW)
        {
            self.hang_reloads.pop_front();
        }
        if self.hang_reloads.len() >= MAX_HANG_RELOADS {
            return Action::HungLeftAlone;
        }
        self.hang_reloads.push_back(now);
        Action::ReloadHung
    }

    /// Forget the reloads that left the window; with none left the page held for a whole window,
    /// so the next failure starts a new episode.
    fn prune(&mut self, now: Instant) {
        while self
            .reloads
            .front()
            .is_some_and(|t| now.saturating_duration_since(*t) > RELOAD_WINDOW)
        {
            self.reloads.pop_front();
        }
        if self.reloads.is_empty() {
            self.announced = false;
        }
    }

    /// How long until the oldest reload has left the window and the budget has room again.
    fn until_a_slot_frees(&self, now: Instant) -> Duration {
        let oldest = self.reloads.front().copied().unwrap_or(now);
        (oldest + RELOAD_WINDOW).saturating_duration_since(now) + DEFER_MARGIN
    }
}

fn locked(policy: &Mutex<RecoveryPolicy>) -> MutexGuard<'_, RecoveryPolicy> {
    policy.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Reload the main window's page when its WebView2 renderer fails. Best effort: a problem is logged
/// and the window works as before.
pub(crate) fn watch(window: &tauri::WebviewWindow) {
    let policy = Arc::new(Mutex::new(RecoveryPolicy::default()));
    let handle = window.clone();
    let queued = window.with_webview(move |platform| {
        // SAFETY: the controller and its WebView2 belong to this window and live on this (main) thread;
        // the handler is a COM object the webview keeps alive for as long as it registered it.
        unsafe {
            let webview = match platform.controller().CoreWebView2() {
                Ok(webview) => webview,
                Err(e) => {
                    tracing::warn!("webview recovery is off: no WebView2 ({e})");
                    return;
                }
            };
            let handler =
                ProcessFailedEventHandler::create(Box::new(move |sender, args| on_process_failed(&policy, &handle, sender, args)));
            let mut token = 0i64;
            if let Err(e) = webview.add_ProcessFailed(&handler, &mut token) {
                tracing::warn!("webview recovery is off: could not watch for process failures ({e})");
            }
        }
    });
    if let Err(e) = queued {
        tracing::warn!("webview recovery is off: could not reach the webview ({e})");
    }
}

fn on_process_failed(
    policy: &Arc<Mutex<RecoveryPolicy>>,
    window: &tauri::WebviewWindow,
    sender: Option<ICoreWebView2>,
    args: Option<ICoreWebView2ProcessFailedEventArgs>,
) -> windows::core::Result<()> {
    let Some(args) = args else { return Ok(()) };
    // SAFETY: `args` is the live event argument of this callback.
    let (kind, detail) = unsafe {
        let mut kind = COREWEBVIEW2_PROCESS_FAILED_KIND::default();
        args.ProcessFailedKind(&mut kind)?;
        (kind.0, failure_detail(&args))
    };
    let failure = Failure::from_kind(kind);
    let action = locked(policy).decide(failure, Instant::now());
    if action.releases_microphone() {
        stop_microphone_session();
    }
    match action {
        Action::Reload { delay } => {
            tracing::warn!(
                "webview process failed (kind {kind}{detail}): reloading the page in {} ms",
                delay.as_millis()
            );
            schedule_reload(policy, window, delay);
        }
        Action::Defer { delay, first } => {
            if first {
                tracing::error!(
                    "webview process failed again (kind {kind}{detail}): {MAX_RELOADS} reloads within {} s did not hold; trying once more in {} s",
                    RELOAD_WINDOW.as_secs(),
                    delay.as_secs()
                );
            } else {
                tracing::warn!(
                    "webview process failed again (kind {kind}{detail}): trying again in {} s",
                    delay.as_secs()
                );
            }
            schedule_reload(policy, window, delay);
        }
        Action::Pending => tracing::debug!("webview process failed (kind {kind}{detail}): a reload is scheduled already"),
        Action::NoteHung => tracing::warn!(
            "webview renderer is not responding (kind {kind}{detail}): waiting, the page is reloaded if this lasts {} s",
            HANG_RELOAD_AFTER.as_secs()
        ),
        Action::ReloadHung => {
            tracing::warn!(
                "webview renderer did not respond for {} s (kind {kind}{detail}): reloading the page",
                HANG_RELOAD_AFTER.as_secs()
            );
            if let Some(sender) = sender {
                // SAFETY: `sender` is the webview that raised the event, used on its own thread.
                unsafe { sender.Reload()? };
            }
        }
        Action::HungLeftAlone => tracing::error!(
            "webview renderer is still not responding (kind {kind}{detail}): {MAX_HANG_RELOADS} reloads within {} s did not help, leaving the page as it is",
            HANG_RELOAD_WINDOW.as_secs()
        ),
        Action::Ignore if failure == Failure::Other => {
            tracing::warn!("webview process failed (kind {kind}{detail}): no reload for this kind");
        }
        Action::Ignore => {}
    }
    Ok(())
}

/// Reload the page after `delay` on a timer thread (the `ProcessFailed` handler returns at once).
/// The thread asks the policy when it fires, so a budget that is still spent waits longer; the
/// reload itself runs on the main thread, where the webview lives.
fn schedule_reload(policy: &Arc<Mutex<RecoveryPolicy>>, window: &tauri::WebviewWindow, delay: Duration) {
    let timer_policy = Arc::clone(policy);
    let timer_window = window.clone();
    let spawned = std::thread::Builder::new().name("webview-reload".into()).spawn(move || {
        let mut wait = delay;
        loop {
            std::thread::sleep(wait);
            match locked(&timer_policy).fire(Instant::now()) {
                Fire::Reload => break,
                Fire::Wait(next) => wait = next,
            }
        }
        tracing::warn!("webview recovery: reloading the page");
        reload(&timer_window);
    });
    if let Err(e) = spawned {
        tracing::error!("webview recovery: could not schedule the reload ({e})");
        locked(policy).cancel();
    }
}

/// Reload the window's page on the main thread.
fn reload(window: &tauri::WebviewWindow) {
    let queued = window.with_webview(|platform| {
        // SAFETY: runs on the main thread, where this window's WebView2 lives.
        let reloaded = unsafe { platform.controller().CoreWebView2().and_then(|webview| webview.Reload()) };
        if let Err(e) = reloaded {
            tracing::warn!("webview recovery: the reload failed ({e})");
        }
    });
    if let Err(e) = queued {
        tracing::warn!("webview recovery: could not reach the webview to reload it ({e})");
    }
}

/// Stop the `mic` speech session of the page that failed (idempotent: nothing running is not an error).
fn stop_microphone_session() {
    tauri::async_runtime::spawn(async {
        if crate::services::speech::pipeline::stop_pipeline_key(crate::services::speech::engine_api::PIPELINE_MIC).await {
            tracing::info!("webview recovery: stopped the microphone session of the page that failed");
        }
    });
}

/// ", reason N, exit code N" when the runtime reports them (WebView2 runtime 1.0.1264+), else "".
///
/// # Safety
/// `args` must be a live event argument.
unsafe fn failure_detail(args: &ICoreWebView2ProcessFailedEventArgs) -> String {
    let Ok(args2) = args.cast::<ICoreWebView2ProcessFailedEventArgs2>() else {
        return String::new();
    };
    let mut reason = COREWEBVIEW2_PROCESS_FAILED_REASON::default();
    let mut exit_code = 0i32;
    if args2.Reason(&mut reason).is_err() || args2.ExitCode(&mut exit_code).is_err() {
        return String::new();
    }
    format!(", reason {}, exit code {exit_code}", reason.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use webview2_com::Microsoft::Web::WebView2::Win32::{
        COREWEBVIEW2_PROCESS_FAILED_KIND_BROWSER_PROCESS_EXITED, COREWEBVIEW2_PROCESS_FAILED_KIND_FRAME_RENDER_PROCESS_EXITED,
        COREWEBVIEW2_PROCESS_FAILED_KIND_GPU_PROCESS_EXITED, COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_EXITED,
        COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_UNRESPONSIVE, COREWEBVIEW2_PROCESS_FAILED_KIND_UTILITY_PROCESS_EXITED,
    };

    const EXITED: Failure = Failure::RenderProcessExited;
    const HUNG: Failure = Failure::RenderProcessUnresponsive;

    fn secs(s: u64) -> Duration {
        Duration::from_secs(s)
    }

    /// A page that crashes right after each reload: the failure arrives, the timer fires, repeat.
    /// Returns the actions of `failures` failures, the n-th at the moment of the previous reload.
    fn crash_loop(policy: &mut RecoveryPolicy, start: Instant, failures: usize) -> (Vec<Action>, Instant) {
        let mut now = start;
        let mut actions = Vec::new();
        for _ in 0..failures {
            let action = policy.decide(EXITED, now);
            actions.push(action);
            let wait = match action {
                Action::Reload { delay } | Action::Defer { delay, .. } => delay,
                other => panic!("unexpected {other:?}"),
            };
            now += wait;
            assert_eq!(policy.fire(now), Fire::Reload, "the timer reloads at {wait:?}");
        }
        (actions, now)
    }

    #[test]
    fn the_kinds_match_the_webview2_values() {
        assert_eq!(
            Failure::from_kind(COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_EXITED.0),
            Failure::RenderProcessExited
        );
        assert_eq!(
            Failure::from_kind(COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_UNRESPONSIVE.0),
            Failure::RenderProcessUnresponsive
        );
        for other in [
            COREWEBVIEW2_PROCESS_FAILED_KIND_BROWSER_PROCESS_EXITED,
            COREWEBVIEW2_PROCESS_FAILED_KIND_GPU_PROCESS_EXITED,
            COREWEBVIEW2_PROCESS_FAILED_KIND_UTILITY_PROCESS_EXITED,
            COREWEBVIEW2_PROCESS_FAILED_KIND_FRAME_RENDER_PROCESS_EXITED,
        ] {
            assert_eq!(Failure::from_kind(other.0), Failure::Other, "kind {}", other.0);
        }
        assert_eq!(Failure::from_kind(-1), Failure::Other);
    }

    #[test]
    fn a_renderer_that_exited_is_reloaded_after_a_back_off_and_other_processes_never() {
        let mut policy = RecoveryPolicy::default();
        let t = Instant::now();
        // A GPU or browser process exit never reloads, and does not use up the reload budget.
        for _ in 0..10 {
            assert_eq!(policy.decide(Failure::Other, t), Action::Ignore);
        }
        // The reload waits (on a timer, not inline) and counts only when it happens.
        assert_eq!(policy.decide(EXITED, t), Action::Reload { delay: secs(1) });
        assert_eq!(policy.fire(t + secs(1)), Fire::Reload);
    }

    #[test]
    fn each_reload_within_the_window_waits_longer() {
        let mut policy = RecoveryPolicy::default();
        let (actions, _) = crash_loop(&mut policy, Instant::now(), MAX_RELOADS);
        assert_eq!(
            actions,
            [
                Action::Reload { delay: secs(1) },
                Action::Reload { delay: secs(5) },
                Action::Reload { delay: secs(15) },
            ]
        );
    }

    #[test]
    fn the_back_off_starts_over_once_the_page_held_for_a_whole_window() {
        let mut policy = RecoveryPolicy::default();
        let t = Instant::now();
        let (_, last) = crash_loop(&mut policy, t, 2);
        let later = last + RELOAD_WINDOW + secs(1);
        assert_eq!(policy.decide(EXITED, later), Action::Reload { delay: secs(1) });
    }

    #[test]
    fn a_failure_while_a_reload_is_scheduled_adds_nothing() {
        let mut policy = RecoveryPolicy::default();
        let t = Instant::now();
        assert_eq!(policy.decide(EXITED, t), Action::Reload { delay: secs(1) });
        for i in 0..5 {
            assert_eq!(policy.decide(EXITED, t + Duration::from_millis(100 * i)), Action::Pending);
        }
        assert_eq!(policy.fire(t + secs(1)), Fire::Reload);
        // One reload was made, not six: the next failure is the second back-off step.
        assert_eq!(policy.decide(EXITED, t + secs(2)), Action::Reload { delay: secs(5) });
    }

    #[test]
    fn a_spent_budget_defers_one_retry_to_when_the_oldest_reload_leaves_the_window() {
        let mut policy = RecoveryPolicy::default();
        let t = Instant::now();
        // Reloads at t+1, t+6 and t+21 s.
        let (_, last) = crash_loop(&mut policy, t, MAX_RELOADS);
        assert_eq!(last, t + secs(21));
        // The page fails once more: no inline give-up, a retry is scheduled for when the t+1 reload leaves the window.
        let action = policy.decide(EXITED, last);
        assert_eq!(
            action,
            Action::Defer {
                delay: RELOAD_WINDOW - secs(20) + DEFER_MARGIN,
                first: true
            }
        );
        // Nothing else is scheduled meanwhile, and the timer reloads once the slot is free.
        assert_eq!(policy.decide(EXITED, last + secs(10)), Action::Pending);
        let Action::Defer { delay, .. } = action else { unreachable!() };
        assert_eq!(policy.fire(last + delay), Fire::Reload);
    }

    #[test]
    fn a_timer_that_runs_early_asks_again_instead_of_reloading_over_the_budget() {
        let mut policy = RecoveryPolicy::default();
        let t = Instant::now();
        let (_, last) = crash_loop(&mut policy, t, MAX_RELOADS);
        let Action::Defer { delay, .. } = policy.decide(EXITED, last) else {
            panic!("the budget is spent")
        };
        let early = last + delay - secs(30);
        assert_eq!(policy.fire(early), Fire::Wait(secs(30)));
        assert_eq!(policy.fire(early + secs(30)), Fire::Reload);
    }

    #[test]
    fn the_give_up_is_announced_once_per_episode_while_retries_go_on() {
        let mut policy = RecoveryPolicy::default();
        let t = Instant::now();
        let (_, mut now) = crash_loop(&mut policy, t, MAX_RELOADS);
        let mut firsts = Vec::new();
        // A page that never holds: ten deferred retries in a row.
        for _ in 0..10 {
            let action = policy.decide(EXITED, now);
            let Action::Defer { delay, first } = action else {
                panic!("expected a deferral, got {action:?}")
            };
            firsts.push(first);
            now += delay;
            assert_eq!(policy.fire(now), Fire::Reload, "the retry happens: it never gives up for good");
        }
        assert_eq!(firsts.iter().filter(|first| **first).count(), 1, "{firsts:?}");
        assert!(firsts[0]);

        // The page finally holds for a whole window: a later burst is a new episode, announced again.
        let (_, after) = crash_loop(&mut policy, now + RELOAD_WINDOW + secs(5), MAX_RELOADS);
        assert!(matches!(policy.decide(EXITED, after), Action::Defer { first: true, .. }));
    }

    #[test]
    fn a_busy_renderer_is_logged_once_and_never_reloaded_nor_charged_to_the_crash_budget() {
        let mut policy = RecoveryPolicy::default();
        let t = Instant::now();
        // The PC is loaded: WebView2 raises the event every 5 s for 25 s.
        assert_eq!(policy.decide(HUNG, t), Action::NoteHung);
        for i in 1..=5 {
            assert_eq!(policy.decide(HUNG, t + secs(5 * i)), Action::Ignore, "event {i}");
        }
        // The renderer answers again; a real crash is still given its whole budget.
        let (actions, _) = crash_loop(&mut policy, t + secs(40), MAX_RELOADS);
        assert_eq!(actions.len(), MAX_RELOADS);
        assert!(actions.iter().all(|a| matches!(a, Action::Reload { .. })), "{actions:?}");
    }

    #[test]
    fn a_renderer_that_stays_unresponsive_is_reloaded_once_after_thirty_seconds() {
        let mut policy = RecoveryPolicy::default();
        let t = Instant::now();
        assert_eq!(policy.decide(HUNG, t), Action::NoteHung);
        for i in 1..6 {
            assert_eq!(policy.decide(HUNG, t + secs(5 * i)), Action::Ignore, "{} s", 5 * i);
        }
        assert_eq!(policy.decide(HUNG, t + HANG_RELOAD_AFTER), Action::ReloadHung);
        // The same episode goes on: no second reload, no second log line.
        for i in 7..30 {
            assert_eq!(policy.decide(HUNG, t + secs(5 * i)), Action::Ignore, "{} s", 5 * i);
        }
        // The hang reload did not touch the crash budget: three more reloads after a renderer exit.
        let (actions, _) = crash_loop(&mut policy, t + secs(200), MAX_RELOADS);
        assert!(actions.iter().all(|a| matches!(a, Action::Reload { .. })), "{actions:?}");
    }

    #[test]
    fn events_far_apart_are_separate_episodes_that_never_add_up() {
        let mut policy = RecoveryPolicy::default();
        let t = Instant::now();
        // One event a minute: each one is its own short episode, none reaches thirty seconds.
        for i in 0..6 {
            assert_eq!(policy.decide(HUNG, t + secs(60 * i)), Action::NoteHung, "event {i}");
        }
    }

    #[test]
    fn repeated_hangs_get_their_own_small_budget_and_are_then_left_alone_once() {
        let mut policy = RecoveryPolicy::default();
        let mut now = Instant::now();
        let episode = |policy: &mut RecoveryPolicy, now: &mut Instant| {
            assert_eq!(policy.decide(HUNG, *now), Action::NoteHung);
            let mut seen = Vec::new();
            for _ in 0..8 {
                *now += secs(5);
                seen.push(policy.decide(HUNG, *now));
            }
            *now += secs(60); // the renderer answers: the episode ends
            seen.into_iter().find(|a| *a != Action::Ignore)
        };
        for _ in 0..MAX_HANG_RELOADS {
            assert_eq!(episode(&mut policy, &mut now), Some(Action::ReloadHung));
        }
        assert_eq!(episode(&mut policy, &mut now), Some(Action::HungLeftAlone));
        // The window moves on: a hang long after is reloaded again.
        now += HANG_RELOAD_WINDOW;
        assert_eq!(episode(&mut policy, &mut now), Some(Action::ReloadHung));
    }

    #[test]
    fn a_hang_does_not_reload_while_a_crash_reload_is_scheduled_and_ends_when_the_renderer_dies() {
        let mut policy = RecoveryPolicy::default();
        let t = Instant::now();
        assert_eq!(policy.decide(HUNG, t), Action::NoteHung);
        // The renderer dies; a reload is scheduled for t+31 s.
        assert_eq!(policy.decide(EXITED, t + secs(29)), Action::Reload { delay: secs(1) });
        // Events of the dying renderer that still arrive change nothing.
        assert_eq!(policy.decide(HUNG, t + secs(30)), Action::Ignore);
        assert_eq!(policy.decide(HUNG, t + secs(31)), Action::Ignore);
        assert_eq!(policy.fire(t + secs(30)), Fire::Reload);
        // The page that was loaded is a new episode.
        assert_eq!(policy.decide(HUNG, t + secs(32)), Action::NoteHung);
    }

    #[test]
    fn a_timer_that_could_not_start_does_not_block_the_next_failure() {
        let mut policy = RecoveryPolicy::default();
        let t = Instant::now();
        assert_eq!(policy.decide(EXITED, t), Action::Reload { delay: secs(1) });
        assert_eq!(policy.decide(EXITED, t + secs(1)), Action::Pending);
        policy.cancel();
        assert_eq!(policy.decide(EXITED, t + secs(2)), Action::Reload { delay: secs(1) });
    }

    #[test]
    fn the_microphone_session_is_released_whenever_the_page_is_about_to_be_replaced() {
        let delay = secs(1);
        for action in [
            Action::Reload { delay },
            Action::Defer { delay, first: true },
            Action::Defer { delay, first: false },
            Action::ReloadHung,
        ] {
            assert!(action.releases_microphone(), "{action:?}");
        }
        for action in [
            Action::Pending,
            Action::NoteHung,
            Action::HungLeftAlone,
            Action::Ignore,
        ] {
            assert!(!action.releases_microphone(), "{action:?}");
        }
        // A renderer exit that exhausted the budget still releases it (no reload is near, the page stays dead).
        let mut policy = RecoveryPolicy::default();
        let (_, last) = crash_loop(&mut policy, Instant::now(), MAX_RELOADS);
        assert!(policy.decide(EXITED, last).releases_microphone());
    }
}
