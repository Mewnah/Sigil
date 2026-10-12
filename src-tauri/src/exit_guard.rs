//! Exit guard (P7-D): a watchdog that ends the process when closing the main window does not.
//!
//! Closing Sigil's window has to end the process. It owns the web server port (OBS caption pages and LAN
//! clients are served from it), the audio devices and the helper processes, and a copy that lingers without a
//! window can only be found in Task Manager and holds the port: the next start waits for it (`port_precheck`)
//! before it gives up.
//!
//! What was seen, and the cause (measured on 2026-10-10 with the packaged smoke test,
//! `scripts/release/packaged-smoke-all.ps1`): the process outlived its closed window in about one automated launch
//! in seven. The event loop (tao) owns an invisible window of class "Tao Thread Event Target" (untitled, 16 x 16, a
//! tool window) that Windows counts as a visible, owner-less top-level window. Whoever posts `WM_CLOSE` to it, for
//! instance `Process.CloseMainWindow()` once the real window is minimized (a minimized window sits at the bottom of
//! the z-order, so the helper window is enumerated first), destroys the loop's own message window through
//! `DefWindowProc`. The real window still closes afterwards and the app starts to exit, but the event loop never
//! finishes, so the process stays alive. The smoke test now closes the window titled "Sigil" only, and a probe that
//! posted `WM_CLOSE` to the helper window first reproduced the stall in its one run (the process ended only through this
//! guard, 20 s after the close of the real window). Tools that post `WM_CLOSE` to every top-level window of a process
//! (`taskkill` without `/F`, installers, the Restart Manager) can do the same to a real install; Sigil cannot change
//! tao's window procedure, so the guard is the remedy.
//!
//! How it works: the first close request of the main window (or request to exit the app) starts one watchdog
//! thread that sleeps for [`GRACE`]. A healthy shutdown ends the process in well under a second, so the thread
//! never wakes. If it does wake, the process is still running 20 s after it was asked to stop: the guard logs one
//! WARN line starting with `exit guard:` (with the state of the main window, including whether Windows considers
//! its UI thread hung), then ends the process with `TerminateProcess`. Unlike `std::process::exit`, that does not
//! need the loader lock or any lock the stalled thread may hold. The helper processes die with the Job Object
//! (`KILL_ON_JOB_CLOSE`), everything Sigil persists is written atomically when it changes (nothing is saved at
//! exit), and the window state is saved on the close request, before the guard starts.
//!
//! Sigil never vetoes a close request (no `onCloseRequested` listener, no `prevent_exit`). Code that ever does
//! must call [`disarm`] when it cancels the shutdown, or the guard ends the app 20 s later.

use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::HWND;
use windows::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
use windows::Win32::UI::WindowsAndMessaging::{IsHungAppWindow, IsIconic, IsWindow, IsWindowVisible};

/// How long the process may take to end after it was asked to stop. A healthy shutdown takes well under a second
/// and the slowest bounded step (helper processes) is 3 s, so this leaves a wide margin for a slow disk.
pub const GRACE: Duration = Duration::from_secs(20);

/// Marker of the guard's log line; the packaged smoke test looks for it.
pub const LOG_MARKER: &str = "exit guard:";

#[derive(Default)]
struct State {
    /// Bumped by every `arm` and `disarm`; a watchdog only fires if its generation is still current.
    generation: AtomicU64,
    armed: AtomicBool,
}

/// One watchdog that can be armed once per shutdown and disarmed again.
#[derive(Default)]
pub struct ExitGuard {
    state: Arc<State>,
}

impl ExitGuard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start the watchdog unless one is already running. After `grace` without a [`disarm`](Self::disarm),
    /// `expire` runs on the watchdog thread with the time that passed. Returns true when this call started it.
    pub fn arm<F>(&self, grace: Duration, expire: F) -> bool
    where
        F: FnOnce(Duration) + Send + 'static,
    {
        if self.state.armed.swap(true, Ordering::SeqCst) {
            return false;
        }
        let generation = self.state.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let state = self.state.clone();
        let started = Instant::now();
        let spawned = std::thread::Builder::new()
            .name("sigil-exit-guard".into())
            .spawn(move || {
                std::thread::sleep(grace);
                if state.generation.load(Ordering::SeqCst) == generation {
                    expire(started.elapsed());
                }
            });
        if spawned.is_err() {
            // No thread, no guard: allow another attempt.
            self.state.armed.store(false, Ordering::SeqCst);
            return false;
        }
        true
    }

    /// Cancel the running watchdog (the shutdown was called off). A later `arm` starts a fresh one.
    pub fn disarm(&self) {
        self.state.generation.fetch_add(1, Ordering::SeqCst);
        self.state.armed.store(false, Ordering::SeqCst);
    }

    #[cfg(test)]
    pub fn is_armed(&self) -> bool {
        self.state.armed.load(Ordering::SeqCst)
    }
}

/// The window handle of the main window (0 = not known yet), kept for the diagnostics line.
static MAIN_WINDOW: AtomicIsize = AtomicIsize::new(0);

fn global() -> &'static ExitGuard {
    static GUARD: OnceLock<ExitGuard> = OnceLock::new();
    GUARD.get_or_init(ExitGuard::new)
}

/// Remember the main window's handle (setup).
pub fn set_main_window(hwnd: isize) {
    MAIN_WINDOW.store(hwnd, Ordering::Relaxed);
}

/// The app was asked to stop (main window close request, `ExitRequested`): make sure the process ends. Idempotent.
pub fn arm() {
    if global().arm(GRACE, expire_process) {
        tracing::debug!(target: "sigil::exit_guard", "armed: the process is ended {} s after the first close request if it is still running", GRACE.as_secs());
    }
}

/// A shutdown that was started has been called off (Sigil does not do this today).
#[allow(dead_code)]
pub fn disarm() {
    global().disarm();
}

/// The state of a window as one line for the log: whether it still exists, is shown, is minimized, and whether
/// Windows thinks its thread stopped pumping messages (`IsHungAppWindow`: no message handled for about 5 s).
pub fn describe_window(hwnd: isize) -> String {
    if hwnd == 0 {
        return "main window unknown".to_string();
    }
    let h = HWND(hwnd as *mut _);
    // SAFETY: plain user32 queries on a handle value; a stale handle just reports "gone".
    unsafe {
        if !IsWindow(Some(h)).as_bool() {
            return "main window already destroyed".to_string();
        }
        format!(
            "main window still exists: visible={}, minimized={}, UI thread unresponsive={}",
            IsWindowVisible(h).as_bool(),
            IsIconic(h).as_bool(),
            IsHungAppWindow(h).as_bool()
        )
    }
}

/// End this process at once without running exit handlers, DLL detach routines or any lock.
pub fn terminate_process(code: u32) -> ! {
    // SAFETY: terminates the calling process; the pseudo handle of the current process is always valid.
    unsafe {
        let _ = TerminateProcess(GetCurrentProcess(), code);
    }
    // TerminateProcess does not return for the current process; this is only the type-level `!`.
    std::process::abort()
}

/// The watchdog fired: log what the main window looks like, then end the process.
fn expire_process(elapsed: Duration) {
    let facts = describe_window(MAIN_WINDOW.load(Ordering::Relaxed));
    let line = format!(
        "{LOG_MARKER} Sigil was still running {} s after it was asked to stop ({facts}); ending the process",
        elapsed.as_secs()
    );
    // The log goes through a lock the stalled thread could hold, so it is written from a helper thread and
    // waited for at most 2 s.
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let logged = std::thread::Builder::new()
        .name("sigil-exit-guard-log".into())
        .spawn(move || {
            tracing::warn!(target: "sigil::exit_guard", "{line}");
            let _ = done_tx.send(());
        });
    if logged.is_ok() {
        let _ = done_rx.recv_timeout(Duration::from_secs(2));
    }
    terminate_process(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    const CHILD_ENV: &str = "SIGIL_EXIT_GUARD_CHILD";

    fn channel_guard(guard: &ExitGuard, grace: Duration) -> (bool, mpsc::Receiver<Duration>) {
        let (tx, rx) = mpsc::channel();
        let armed = guard.arm(grace, move |elapsed| {
            let _ = tx.send(elapsed);
        });
        (armed, rx)
    }

    #[test]
    fn fires_once_after_the_grace_period() {
        let guard = ExitGuard::new();
        assert!(!guard.is_armed());
        let (armed, rx) = channel_guard(&guard, Duration::from_millis(60));
        assert!(armed && guard.is_armed());
        let elapsed = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the guard fires");
        assert!(elapsed >= Duration::from_millis(60), "fired early: {elapsed:?}");
        // One watchdog per shutdown: nothing else fires.
        assert!(rx.recv_timeout(Duration::from_millis(200)).is_err());
    }

    #[test]
    fn a_second_arm_while_armed_does_nothing() {
        let guard = ExitGuard::new();
        let (first, rx) = channel_guard(&guard, Duration::from_millis(100));
        let (second, rx2) = channel_guard(&guard, Duration::from_millis(10));
        assert!(first);
        assert!(!second, "a close request that follows the first one must not start another watchdog");
        assert!(rx.recv_timeout(Duration::from_secs(5)).is_ok());
        assert!(rx2.recv_timeout(Duration::from_millis(200)).is_err(), "the second closure never runs");
    }

    #[test]
    fn disarm_cancels_and_a_later_arm_starts_afresh() {
        let guard = ExitGuard::new();
        let (_, cancelled) = channel_guard(&guard, Duration::from_millis(80));
        guard.disarm();
        assert!(!guard.is_armed());
        let (rearmed, fired) = channel_guard(&guard, Duration::from_millis(200));
        assert!(rearmed, "the shutdown was called off, a new one arms again");
        // The cancelled watchdog woke up in between and found its generation replaced.
        assert!(fired.recv_timeout(Duration::from_secs(5)).is_ok());
        assert!(cancelled.recv_timeout(Duration::from_millis(100)).is_err());
    }

    #[test]
    fn a_disarmed_guard_never_fires() {
        let guard = ExitGuard::new();
        let (_, rx) = channel_guard(&guard, Duration::from_millis(60));
        guard.disarm();
        assert!(rx.recv_timeout(Duration::from_millis(400)).is_err());
    }

    #[test]
    fn describes_a_missing_and_a_stale_window_without_crashing() {
        assert_eq!(describe_window(0), "main window unknown");
        // A handle value that is no window of this process.
        assert_eq!(describe_window(0x7fff_fff0), "main window already destroyed");
    }

    #[test]
    fn the_log_line_carries_the_marker_the_smoke_test_greps_for() {
        assert!(LOG_MARKER.ends_with(':') && LOG_MARKER.starts_with("exit guard"));
    }

    /// The "hung" process of `a_hung_process_is_ended_by_terminate_process`: arms the guard, then never returns.
    /// Does nothing unless the parent test started it.
    #[test]
    fn child_that_hangs() {
        if std::env::var_os(CHILD_ENV).is_none() {
            return;
        }
        let guard = ExitGuard::new();
        guard.arm(Duration::from_millis(300), |_| terminate_process(42));
        std::thread::sleep(Duration::from_secs(120));
    }

    /// The real thing, in a child process: a process that would run for two minutes ends with the exit code of
    /// `TerminateProcess` shortly after the grace period.
    #[test]
    fn a_hung_process_is_ended_by_terminate_process() {
        let exe = std::env::current_exe().expect("test executable");
        let started = Instant::now();
        let mut child = Command::new(exe)
            .args(["--exact", "exit_guard::tests::child_that_hangs", "--nocapture", "--test-threads=1"])
            .env(CHILD_ENV, "1")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("start the child test process");
        let status = loop {
            if let Some(status) = child.try_wait().expect("poll the child") {
                break status;
            }
            if started.elapsed() > Duration::from_secs(60) {
                let _ = child.kill();
                panic!("the guard did not end the hung process within 60 s");
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        assert_eq!(status.code(), Some(42), "ended by the guard's TerminateProcess, not by finishing");
        assert!(started.elapsed() < Duration::from_secs(60));
    }
}
