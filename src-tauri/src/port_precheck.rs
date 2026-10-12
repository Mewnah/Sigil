//! Start-up port pre-check: runs in `main()` before Tauri exists, so there is no window yet and
//! nothing here may show anything (or take the foreground) until the check has decided that it must.
//!
//! A held port used to end the start at once with a bare "Port N is not available!" box, which hit
//! three ordinary cases: starting Sigil again while it runs, starting it again within seconds of
//! closing it (the old copy still holds the port; the exit guard ends a stalled one after 20 s, see
//! `exit_guard`), and a port that is reserved or taken by another program. Now the check polls and
//! looks at who holds the port:
//!
//! - Another copy of Sigil with a window: that window is brought to the front and this start ends
//!   without a dialog. The window has to be seen twice in a row, so the window of a copy that was
//!   just closed (it is gone within a poll) is never mistaken for a running one. A window that does
//!   not answer (Windows' hung-window check, or a raise that does not return within
//!   [`Limits::raise`]) is never waited on: a frozen Sigil would freeze this process too.
//! - Another copy of Sigil without a window (closing, or stalled until the guard ends it), or a
//!   holder that cannot be identified: wait up to [`exit_guard::GRACE`] plus a margin for the port.
//! - Another program, or a port Windows reserved (Hyper-V, WSL, Docker: access denied): neither
//!   frees up by itself, so only a short wait for a transient holder. A Sigil whose window does not
//!   answer gets a little longer ([`Limits::frozen`]): it may just be busy.
//!
//! The holder comes from the OS table of listening sockets (a listener on any address counts: a
//! relay bound to one LAN address also blocks the wildcard bind) and the process image name (the
//! same lookup the helper-process audit uses). Only after the wait does a box explain what holds the
//! port and what to do. With `SIGIL_START_MINIMIZED` (automated runs on a PC someone is using) no
//! other window is ever raised, `main` shows no box, and a running copy with a window is not waited
//! for either (it never goes away by itself).
//!
//! Starts on one port take turns (a named lock, [`Limits::turn`]): without it, every extra click
//! during a silent wait would pass the check at the same moment the port frees and start a full
//! Sigil each. A start keeps its turn until the web server listens ([`release_start_turn`]), so a
//! second start never sees a free port that is about to be taken; it then finds the first copy's
//! window and raises it. The decisions and the Windows lookups are separate so the decisions are
//! tested without a second process.

use std::ffi::c_void;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::net::{IpAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use windows::core::{w, BOOL, PCWSTR, PWSTR};
use windows::Win32::Foundation::{CloseHandle, GetLastError, ERROR_ALREADY_EXISTS, HANDLE, HWND, LPARAM};
use windows::Win32::System::Threading::{OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetWindow, GetWindowTextW, GetWindowThreadProcessId, IsHungAppWindow, IsIconic, IsWindowVisible, MessageBoxW, SetForegroundWindow,
    ShowWindow, GW_OWNER, MB_ICONWARNING, MB_OK, SW_RESTORE,
};

use crate::exit_guard;
use crate::services::process_host::ports;

// `CreateMutexW` is gated on the `Win32_Security` feature in windows 0.61, which is not enabled; it is
// a plain kernel32 export (the same approach as `process_host/job.rs`).
#[link(name = "kernel32")]
extern "system" {
    fn CreateMutexW(attributes: *const c_void, initial_owner: i32, name: *const u16) -> *mut c_void;
}

/// Title of Sigil's main window (`tauri.conf.json`). The window of another program that happens
/// to be called "Sigil" (a folder in Explorer) is never matched: the process has to be the holder.
const MAIN_WINDOW_TITLE: &str = "Sigil";

/// The folder under `%LOCALAPPDATA%` that Tauri's `app_log_dir` resolves to (the bundle identifier
/// in `tauri.conf.json`; a test keeps the two together).
const LOG_FOLDER: &str = "com.sigil.app";

/// How long the check waits and how often it looks.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Pause between two attempts.
    pub poll: Duration,
    /// For a holder that may still be exiting (another Sigil without a window) or that cannot be
    /// identified: longer than the exit guard's grace.
    pub full: Duration,
    /// For a holder that will not go away by itself (another program, a reserved port, a running
    /// Sigil that an automated run may not raise).
    pub short: Duration,
    /// For a Sigil whose window does not answer. Windows calls a window hung after 5 s without a
    /// message, so a copy that is only busy (start-up, a heavy moment) answers again within a few
    /// more seconds and is raised then; one that stays silent gets the box.
    pub frozen: Duration,
    /// How many polls in a row the window of another Sigil must be seen before giving way to it.
    pub window_polls: u32,
    /// How long raising another Sigil's window may take. A window that has not answered by then is
    /// frozen: the call is a cross-thread message to the window's own thread.
    pub raise: Duration,
    /// How long a start waits for its turn (the start lock) before going on without it: longer than
    /// the wait of the start in front plus [`Limits::hold`].
    pub turn: Duration,
    /// How long a start that passed the check keeps its turn at most; it gives it up earlier as
    /// soon as the web server listens.
    pub hold: Duration,
}

impl Limits {
    pub const APP: Limits = Limits {
        poll: Duration::from_millis(250),
        full: exit_guard::GRACE.saturating_add(Duration::from_secs(5)),
        short: Duration::from_secs(3),
        frozen: Duration::from_secs(10),
        window_polls: 2,
        raise: Duration::from_secs(2),
        turn: Duration::from_secs(60),
        hold: Duration::from_secs(30),
    };
}

/// Who holds the port.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Holder {
    /// Another copy of Sigil. `window` is its main window; a copy that is closing or stalled has none.
    Sigil { pid: u32, window: Option<isize> },
    /// Another copy of Sigil whose window does not answer (a frozen UI thread).
    Frozen { pid: u32 },
    /// Another program (`exe` is unknown when Windows does not let this process look).
    Other { pid: u32, exe: Option<String> },
    /// Not found in the socket table: a range Windows reserved, or a table that could not be read.
    Unknown,
}

#[derive(Debug)]
pub enum Outcome {
    /// The port can be bound; the check released it again.
    Free,
    /// A running Sigil was brought to the front; this start should end quietly.
    Raised,
    /// Still not available after the wait.
    Busy { error: io::Error, holder: Holder },
}

#[derive(Debug, PartialEq, Eq)]
enum Step {
    Wait,
    Raise(isize),
    GiveUp,
}

/// How long to keep waiting for this holder.
fn budget(limits: &Limits, allow_raise: bool, kind: io::ErrorKind, holder: &Holder) -> Duration {
    match holder {
        // A running copy with a window does not go away by itself. Only an automated run (which
        // may not raise it) gets here with one; the short wait still covers a window that is about to close.
        Holder::Sigil { window: Some(_), .. } if !allow_raise => limits.short,
        Holder::Sigil { .. } => limits.full,
        Holder::Frozen { .. } => limits.frozen,
        Holder::Other { .. } => limits.short,
        Holder::Unknown if kind == io::ErrorKind::PermissionDenied => limits.short,
        Holder::Unknown => limits.full,
    }
}

/// What to do after a failed bind. `window_polls` counts the polls in a row that saw the same
/// window of the same Sigil process.
fn next_step(limits: &Limits, allow_raise: bool, kind: io::ErrorKind, holder: &Holder, waited: Duration, window_polls: u32) -> Step {
    if let Holder::Sigil { window: Some(hwnd), .. } = holder {
        if allow_raise && window_polls >= limits.window_polls {
            return Step::Raise(*hwnd);
        }
    }
    if waited >= budget(limits, allow_raise, kind, holder) {
        Step::GiveUp
    } else {
        Step::Wait
    }
}

/// Try `bind` until it works, the holder is a running Sigil to give way to, or the wait is over.
/// `inspect` names the holder after each failed attempt; `raise` brings a window to the front and
/// says whether it came back in time (`false`: the window is frozen, which ends the wait as `Busy`).
pub fn wait_for_port(
    limits: &Limits,
    allow_raise: bool,
    mut bind: impl FnMut() -> io::Result<()>,
    mut inspect: impl FnMut() -> Holder,
    mut raise: impl FnMut(isize) -> bool,
) -> Outcome {
    let started = Instant::now();
    let mut sighted: Option<(u32, isize)> = None;
    let mut window_polls = 0u32;
    loop {
        let error = match bind() {
            Ok(()) => return Outcome::Free,
            Err(error) => error,
        };
        let holder = inspect();
        match &holder {
            Holder::Sigil { pid, window: Some(hwnd) } if sighted == Some((*pid, *hwnd)) => window_polls += 1,
            Holder::Sigil { pid, window: Some(hwnd) } => {
                sighted = Some((*pid, *hwnd));
                window_polls = 1;
            }
            _ => {
                sighted = None;
                window_polls = 0;
            }
        }
        match next_step(limits, allow_raise, error.kind(), &holder, started.elapsed(), window_polls) {
            Step::Raise(hwnd) => {
                if raise(hwnd) {
                    return Outcome::Raised;
                }
                let pid = if let Holder::Sigil { pid, .. } = holder { pid } else { 0 };
                return Outcome::Busy {
                    error,
                    holder: Holder::Frozen { pid },
                };
            }
            Step::GiveUp => return Outcome::Busy { error, holder },
            Step::Wait => std::thread::sleep(limits.poll),
        }
    }
}

/// Whether a take of the start lock got it.
#[derive(Debug, PartialEq, Eq)]
enum Turn {
    /// This process created the lock; the value is its handle (closing it gives the turn up).
    Taken(isize),
    /// Another start has it.
    Held,
    /// The lock cannot be used here (access denied: another user or an elevated copy made it).
    Unavailable,
}

/// Wait for this start's turn: `take` until it works or [`Limits::turn`] is over. `None` goes on
/// without a lock (never worse than no lock at all): the lock could not be used, or the start in
/// front of this one took longer than any start should.
fn take_turn(limits: &Limits, mut take: impl FnMut() -> Turn) -> Option<isize> {
    let started = Instant::now();
    loop {
        match take() {
            Turn::Taken(handle) => return Some(handle),
            Turn::Unavailable => return None,
            Turn::Held => {}
        }
        if started.elapsed() >= limits.turn {
            return None;
        }
        std::thread::sleep(limits.poll);
    }
}

/// The lock of the starts on `port`: one per Windows session (`Local\`), none between ports.
fn lock_name(port: u16) -> String {
    format!("Local\\Sigil-start-{port}")
}

/// Create the named lock object. The lock is "this process holds a handle to a freshly created
/// object": a start that finds it existing closes its handle again at once, so the object ends with
/// its holder, even when that process is killed (no abandoned lock to clean up).
fn create_lock(name: &str) -> Turn {
    let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    // SAFETY: `wide` is NUL-terminated and outlives the call; default security, not owned.
    let handle = unsafe { CreateMutexW(std::ptr::null(), 0, wide.as_ptr()) };
    if handle.is_null() {
        return Turn::Unavailable;
    }
    // SAFETY: read straight after the call above, on the same thread.
    if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
        close_handle(handle as isize);
        Turn::Held
    } else {
        Turn::Taken(handle as isize)
    }
}

fn close_handle(handle: isize) {
    // SAFETY: a handle this module created, closed once.
    unsafe {
        let _ = CloseHandle(HANDLE(handle as *mut _));
    }
}

/// The start lock while this process keeps it.
struct Held(AtomicIsize);

impl Held {
    const fn new() -> Self {
        Self(AtomicIsize::new(0))
    }

    fn keep(&self, handle: isize) {
        self.release();
        self.0.store(handle, Ordering::SeqCst);
    }

    /// Give the turn up; a no-op when it is not held (or already given up).
    fn release(&self) {
        let handle = self.0.swap(0, Ordering::SeqCst);
        if handle != 0 {
            close_handle(handle);
        }
    }
}

static START_TURN: Held = Held::new();

/// The web server listens: the next start may take its turn (it will find this copy's window).
pub fn release_start_turn() {
    START_TURN.release();
}

/// Keep the turn of a start that passed the check until the web server listens or `hold` is over.
fn keep_turn(handle: isize, hold: Duration) {
    START_TURN.keep(handle);
    let timer = std::thread::Builder::new()
        .name("sigil-start-turn".into())
        .spawn(move || {
            std::thread::sleep(hold);
            release_start_turn();
        });
    if timer.is_err() {
        release_start_turn();
    }
}

/// The pre-check for `ip:port`. `allow_raise` is false for automated runs, which must never pull
/// another window to the front.
pub fn run(ip: IpAddr, port: u16, allow_raise: bool) -> Outcome {
    let limits = Limits::APP;
    let started = Instant::now();
    let turn = take_turn(&limits, || create_lock(&lock_name(port)));
    if turn.is_none() && started.elapsed() >= limits.turn {
        log_line(&format!(
            "start-up: another start held the port {port} turn for {} s; going on without waiting longer",
            limits.turn.as_secs()
        ));
    }
    let outcome = wait_for_port(
        &limits,
        allow_raise,
        || TcpListener::bind((ip, port)).map(drop),
        || inspect(port),
        |hwnd| bring_to_front_within(limits.raise, hwnd),
    );
    let waited = started.elapsed();
    match &outcome {
        Outcome::Free => {
            if waited >= Duration::from_secs(1) {
                log_line(&format!(
                    "start-up: port {port} was busy and became free after {:.1} s",
                    waited.as_secs_f32()
                ));
            }
        }
        Outcome::Raised => log_line(&format!(
            "start-up: port {port} is held by a running Sigil; its window was brought to the front and this start ends"
        )),
        Outcome::Busy { error, holder } => log_line(&format!(
            "start-up: port {port} is not available after {:.1} s ({error}); held by {holder:?}",
            waited.as_secs_f32()
        )),
    }
    match (&outcome, turn) {
        (Outcome::Free, Some(handle)) => keep_turn(handle, limits.hold),
        // Given up before the box is shown (it blocks until someone clicks it), and when this start ends.
        (_, Some(handle)) => close_handle(handle),
        (_, None) => {}
    }
    outcome
}

/// The text of the box for a port that stayed busy.
pub fn describe(port: u16, error: &io::Error, holder: &Holder) -> String {
    let other_port = if port < u16::MAX { port + 1 } else { port - 1 };
    let code = error
        .raw_os_error()
        .map(|c| format!("\n\n(Windows error {c})"))
        .unwrap_or_default();
    let elsewhere = format!("To use another port, start Sigil with --port {other_port}; the links you gave OBS then change.");
    match holder {
        Holder::Sigil { pid, window: Some(_) } => {
            format!("Sigil is already running (process {pid}), so a second copy cannot use port {port}. Switch to the running Sigil from the taskbar.{code}")
        }
        Holder::Sigil { pid, window: None } => format!(
            "Port {port} is not available!\n\nAnother copy of Sigil (process {pid}) is still running without a window. It ends by itself within about {} seconds after it was closed; if it does not, end Sigil.exe in Task Manager. Then start Sigil again.{code}",
            exit_guard::GRACE.as_secs()
        ),
        Holder::Frozen { pid } => format!(
            "Sigil is already running (process {pid}), but its window is not responding, so a second copy cannot use port {port}. Give it a moment; if it stays frozen, end Sigil.exe in Task Manager. Then start Sigil again.{code}"
        ),
        Holder::Other { pid, exe } => {
            let who = exe.as_deref().map_or_else(|| format!("process {pid}"), |exe| format!("{exe}, process {pid}"));
            format!("Port {port} is not available!\n\nAnother program is using it ({who}). Close that program and start Sigil again.\n\n{elsewhere}{code}")
        }
        Holder::Unknown if error.kind() == io::ErrorKind::PermissionDenied => format!(
            "Port {port} is not available!\n\nWindows does not let Sigil use it. Windows reserves port ranges for Hyper-V, WSL and Docker (the command netsh int ipv4 show excludedportrange protocol=tcp lists them), and a security program can block a port too.\n\n{elsewhere}{code}"
        ),
        Holder::Unknown => format!("Port {port} is not available!\n\nAnother program is using it. Close that program and start Sigil again.\n\n{elsewhere}{code}"),
    }
}

/// The box. English: native pre-webview errors stay English (P3: optional Tauri locale bridge if
/// product needs localized OS dialogs).
pub fn show_message(text: &str) {
    let wide: Vec<u16> = text.encode_utf16().chain(std::iter::once(0)).collect();
    // SAFETY: `wide` is NUL-terminated and outlives the call; the caption is a static literal.
    unsafe {
        MessageBoxW(None, PCWSTR(wide.as_ptr()), w!("Sigil"), MB_OK | MB_ICONWARNING);
    }
}

// ---------------------------------------------------------------------------------------------
// Log
// ---------------------------------------------------------------------------------------------

/// `%LOCALAPPDATA%\com.sigil.app\logs\sigil.log`, where the diagnostics plugin logs once Tauri runs.
fn log_file_path() -> Option<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA").filter(|v| !v.is_empty())?;
    Some(
        PathBuf::from(base)
            .join(LOG_FOLDER)
            .join("logs")
            .join("sigil.log"),
    )
}

/// A line shaped like the diagnostics log's own (`<time>  WARN <thread> <target>: <message>`), on one line.
fn format_log_line(stamp: &str, message: &str) -> String {
    let one_line = message.split_whitespace().collect::<Vec<_>>().join(" ");
    format!("{stamp}  WARN main sigil::startup: {one_line}\n")
}

fn append_log_line(path: &Path, line: &str) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(line.as_bytes())
}

/// One line for the support log. Runs before the logger exists (the diagnostics plugin starts it
/// inside Tauri), and a release build has no console, so it appends to the same file the logger
/// writes later; stderr serves console runs. Never fails.
pub fn log_line(message: &str) {
    use tracing_subscriber::fmt::{format::Writer, time::FormatTime, time::SystemTime};
    let _ = writeln!(io::stderr(), "{message}");
    let Some(path) = log_file_path() else {
        return;
    };
    let mut stamp = String::new();
    let _ = SystemTime.format_time(&mut Writer::new(&mut stamp));
    let _ = append_log_line(&path, &format_log_line(&stamp, message));
}

// ---------------------------------------------------------------------------------------------
// Windows lookups
// ---------------------------------------------------------------------------------------------

/// Who listens on `port` right now.
fn inspect(port: u16) -> Holder {
    match ports::tcp_listeners() {
        Ok(listeners) => holder_of(
            &listeners,
            port,
            own_exe_name().as_deref(),
            process_exe_name,
            main_window_of,
            window_is_hung,
        ),
        // The table is not readable.
        Err(_) => Holder::Unknown,
    }
}

/// The holder of `port` among `listeners`: the listener that receives loopback connections, else
/// any listener on that port. One bound to a single LAN address (a Docker or WSL relay, a dev
/// server) leaves a loopback connection unanswered but still makes the wildcard bind fail.
fn holder_of(
    listeners: &[ports::Listener],
    port: u16,
    own_exe: Option<&str>,
    exe_of: impl Fn(u32) -> Option<String>,
    window_of: impl Fn(u32) -> Option<isize>,
    hung: impl Fn(isize) -> bool,
) -> Holder {
    let pid = ports::loopback_receiver(listeners, port).or_else(|| {
        listeners
            .iter()
            .find(|l| l.addr.port() == port)
            .map(|l| l.pid)
    });
    match pid {
        Some(pid) => classify(pid, own_exe, exe_of, window_of, hung),
        // Nothing listens there (a range Windows reserved).
        None => Holder::Unknown,
    }
}

/// `pid` is another copy of Sigil when its image has the same file name as this one's (an
/// installed `Sigil.exe` and a dev build count as the same app: both want the same port). Its
/// window only counts as one to raise while it answers.
fn classify(
    pid: u32,
    own_exe: Option<&str>,
    exe_of: impl Fn(u32) -> Option<String>,
    window_of: impl Fn(u32) -> Option<isize>,
    hung: impl Fn(isize) -> bool,
) -> Holder {
    let exe = exe_of(pid);
    match (&exe, own_exe) {
        (Some(exe), Some(own)) if exe.eq_ignore_ascii_case(own) => match window_of(pid) {
            Some(window) if hung(window) => Holder::Frozen { pid },
            window => Holder::Sigil { pid, window },
        },
        _ => Holder::Other { pid, exe },
    }
}

fn own_exe_name() -> Option<String> {
    std::env::current_exe()
        .ok()?
        .file_name()?
        .to_str()
        .map(str::to_string)
}

/// The image file name of process `pid` (`Sigil.exe`), where this process may query it.
fn process_exe_name(pid: u32) -> Option<String> {
    // SAFETY: a limited-query OpenProcess; the handle is closed below.
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) }.ok()?;
    let mut buf = [0u16; 520];
    let mut len = buf.len() as u32;
    // SAFETY: a valid process handle and a buffer of `len` UTF-16 units.
    let ok = unsafe { QueryFullProcessImageNameW(handle, PROCESS_NAME_WIN32, PWSTR(buf.as_mut_ptr()), &mut len) }.is_ok();
    // SAFETY: opened above, closed once.
    unsafe {
        let _ = CloseHandle(handle);
    }
    if !ok {
        return None;
    }
    let path = String::from_utf16_lossy(&buf[..(len as usize).min(buf.len())]);
    path.rsplit(['\\', '/'])
        .next()
        .map(str::to_string)
        .filter(|name| !name.is_empty())
}

struct WindowSearch {
    pid: u32,
    title: Vec<u16>,
    found: Option<isize>,
}

unsafe extern "system" fn match_main_window(hwnd: HWND, lparam: LPARAM) -> BOOL {
    // SAFETY: `lparam` is the `&mut WindowSearch` that `main_window_titled` passes for the duration of EnumWindows.
    let search = unsafe { &mut *(lparam.0 as *mut WindowSearch) };
    let mut pid = 0u32;
    // SAFETY: plain user32 queries on a window handle EnumWindows handed us. GetWindowText reads the
    // caption of another process's window directly; it does not message (and so never waits on) it.
    unsafe {
        GetWindowThreadProcessId(hwnd, Some(&mut pid as *mut u32));
        if pid != search.pid || !IsWindowVisible(hwnd).as_bool() || GetWindow(hwnd, GW_OWNER).is_ok_and(|owner| !owner.is_invalid()) {
            return BOOL(1);
        }
        // One unit more than the wanted title, so a longer title is not cut down to a match.
        let mut title = vec![0u16; search.title.len() + 2];
        let len = GetWindowTextW(hwnd, &mut title);
        if len > 0 && title[..len as usize] == search.title[..] {
            search.found = Some(hwnd.0 as isize);
            return BOOL(0);
        }
    }
    BOOL(1)
}

/// The visible, unowned top-level window titled "Sigil" of process `pid`. The event loop's hidden
/// helper window and the caption window have other titles.
fn main_window_of(pid: u32) -> Option<isize> {
    main_window_titled(pid, MAIN_WINDOW_TITLE)
}

fn main_window_titled(pid: u32, title: &str) -> Option<isize> {
    let mut search = WindowSearch {
        pid,
        title: title.encode_utf16().collect(),
        found: None,
    };
    // SAFETY: the callback only touches `search` through the pointer, during this call. EnumWindows
    // reports an error when the callback stops it early, which is how a hit ends the walk.
    let _ = unsafe { EnumWindows(Some(match_main_window), LPARAM(&mut search as *mut WindowSearch as isize)) };
    search.found
}

/// Whether Windows thinks the window's thread stopped pumping messages (no message handled for
/// about 5 s): a call that messages that thread would not return either.
fn window_is_hung(hwnd: isize) -> bool {
    // SAFETY: a plain user32 query on a handle value; a window that is gone is not "hung".
    unsafe { IsHungAppWindow(HWND(hwnd as *mut _)).as_bool() }
}

/// Restore the window if it is minimized and ask for the foreground. Windows lets a process that
/// the user just started do that; if it refuses, the taskbar button flashes instead. Both calls go
/// to the window's own thread, so this can block for as long as that thread does.
fn bring_to_front(hwnd: isize) {
    let hwnd = HWND(hwnd as *mut _);
    // SAFETY: plain user32 calls on a handle value; a window that is gone makes them fail harmlessly.
    unsafe {
        if IsIconic(hwnd).as_bool() {
            let _ = ShowWindow(hwnd, SW_RESTORE);
        }
        let _ = SetForegroundWindow(hwnd);
    }
}

/// Run `work` on a helper thread and wait at most `limit` for it. `false` when it had not finished
/// (or no thread could be made): the thread is left behind, which is harmless because this process
/// ends right after the check.
fn finishes_within(limit: Duration, work: impl FnOnce() + Send + 'static) -> bool {
    let (done, finished) = mpsc::channel::<()>();
    let spawned = std::thread::Builder::new()
        .name("sigil-raise".into())
        .spawn(move || {
            work();
            let _ = done.send(());
        });
    spawned.is_ok() && finished.recv_timeout(limit).is_ok()
}

/// [`bring_to_front`] that cannot hold this process up: a frozen Sigil window (frozen after the last
/// look at it) would block the calls forever.
fn bring_to_front_within(limit: Duration, hwnd: isize) -> bool {
    finishes_within(limit, move || bring_to_front(hwnd))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use windows::Win32::UI::WindowsAndMessaging::{CreateWindowExW, DestroyWindow, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_POPUP, WS_VISIBLE};

    fn fast() -> Limits {
        Limits {
            poll: Duration::from_millis(5),
            full: Duration::from_millis(400),
            short: Duration::from_millis(60),
            frozen: Duration::from_millis(100),
            window_polls: 2,
            raise: Duration::from_millis(50),
            turn: Duration::from_millis(200),
            hold: Duration::from_millis(50),
        }
    }

    /// Budgets far longer than any test takes, for the cases that must not time out however slow the machine is.
    fn patient() -> Limits {
        Limits {
            full: Duration::from_secs(30),
            short: Duration::from_secs(30),
            frozen: Duration::from_secs(30),
            ..fast()
        }
    }

    fn in_use() -> io::Error {
        io::Error::from(io::ErrorKind::AddrInUse)
    }

    fn sigil(window: Option<isize>) -> Holder {
        Holder::Sigil { pid: 4242, window }
    }

    fn other() -> Holder {
        Holder::Other {
            pid: 77,
            exe: Some("node.exe".into()),
        }
    }

    /// A bind that is busy for the first `busy_calls` attempts.
    fn busy_for(busy_calls: u32, kind: io::ErrorKind) -> impl FnMut() -> io::Result<()> {
        let calls = Cell::new(0u32);
        move || {
            calls.set(calls.get() + 1);
            if calls.get() <= busy_calls {
                Err(io::Error::from(kind))
            } else {
                Ok(())
            }
        }
    }

    /// Runs `wait_for_port` against a bind that never frees up, counting the binds. The wait ends at
    /// the budget, so `elapsed >= budget` always holds and `binds <= budget / poll + 1` as well, however
    /// slow the machine: no assertion here depends on how fast the test ran.
    fn wait_in_vain(limits: &Limits, allow_raise: bool, kind: io::ErrorKind, holder: Holder) -> (Outcome, Duration, u32) {
        let binds = Cell::new(0u32);
        let started = Instant::now();
        let outcome = wait_for_port(
            limits,
            allow_raise,
            || {
                binds.set(binds.get() + 1);
                Err(io::Error::from(kind))
            },
            || holder.clone(),
            |_| panic!("no window to raise"),
        );
        (outcome, started.elapsed(), binds.get())
    }

    /// More binds than a wait of `budget` can make (with slack for a sleep that returns early); the
    /// wait for a full budget makes several times as many.
    fn most_binds(budget: Duration, limits: &Limits) -> u32 {
        (budget.as_millis() / limits.poll.as_millis()) as u32 * 2 + 2
    }

    #[test]
    fn a_port_that_frees_up_is_taken_without_any_dialog() {
        let holder = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = holder.local_addr().unwrap();
        let release = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(80));
            drop(holder);
        });
        let limits = Limits {
            full: Duration::from_secs(5),
            ..fast()
        };
        let started = Instant::now();
        let outcome = wait_for_port(
            &limits,
            true,
            || TcpListener::bind(addr).map(drop),
            || Holder::Unknown,
            |_| panic!("no window to raise"),
        );
        release.join().unwrap();
        assert!(matches!(outcome, Outcome::Free), "{outcome:?}");
        assert!(
            started.elapsed() >= Duration::from_millis(70),
            "it waited for the holder: {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn a_closing_sigil_is_waited_for_the_full_time() {
        // No window: the copy is exiting (or stalled until the exit guard ends it).
        let (outcome, waited, _) = wait_in_vain(&fast(), true, io::ErrorKind::AddrInUse, sigil(None));
        assert!(waited >= Duration::from_millis(400), "{waited:?}");
        let Outcome::Busy { error, holder } = outcome else {
            panic!("{outcome:?}")
        };
        assert_eq!(error.kind(), io::ErrorKind::AddrInUse);
        assert_eq!(holder, sigil(None));

        // ... and when it ends in time, the start goes on.
        let outcome = wait_for_port(
            &patient(),
            true,
            busy_for(5, io::ErrorKind::AddrInUse),
            || sigil(None),
            |_| panic!("no window to raise"),
        );
        assert!(matches!(outcome, Outcome::Free), "{outcome:?}");
    }

    #[test]
    fn an_unidentified_holder_is_waited_for_the_full_time() {
        let (outcome, waited, _) = wait_in_vain(&fast(), true, io::ErrorKind::AddrInUse, Holder::Unknown);
        assert!(waited >= Duration::from_millis(400), "{waited:?}");
        assert!(matches!(outcome, Outcome::Busy { holder: Holder::Unknown, .. }), "{outcome:?}");
    }

    #[test]
    fn another_program_or_a_reserved_port_gets_only_the_short_wait() {
        let l = fast();
        let (outcome, waited, binds) = wait_in_vain(&l, true, io::ErrorKind::AddrInUse, other());
        assert!(waited >= Duration::from_millis(60), "{waited:?}");
        assert!(binds <= most_binds(l.short, &l), "{binds} binds in the short wait");
        assert!(
            matches!(
                outcome,
                Outcome::Busy {
                    holder: Holder::Other { pid: 77, .. },
                    ..
                }
            ),
            "{outcome:?}"
        );

        let (outcome, waited, binds) = wait_in_vain(&l, true, io::ErrorKind::PermissionDenied, Holder::Unknown);
        assert!(waited >= Duration::from_millis(60), "{waited:?}");
        assert!(binds <= most_binds(l.short, &l), "{binds} binds in the short wait");
        let Outcome::Busy { error, .. } = outcome else { panic!("{outcome:?}") };
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn a_running_sigil_is_brought_forward_after_two_sightings() {
        let raised = Cell::new(None);
        let binds = Cell::new(0u32);
        let outcome = wait_for_port(
            &fast(),
            true,
            || {
                binds.set(binds.get() + 1);
                Err(in_use())
            },
            || sigil(Some(42)),
            |hwnd| {
                raised.set(Some(hwnd));
                true
            },
        );
        assert!(matches!(outcome, Outcome::Raised), "{outcome:?}");
        assert_eq!(raised.get(), Some(42));
        assert_eq!(binds.get(), 2, "the window was seen twice, no more waiting than that");
    }

    #[test]
    fn a_window_that_does_not_come_back_from_the_raise_is_a_frozen_sigil() {
        let outcome = wait_for_port(&fast(), true, || Err(in_use()), || sigil(Some(42)), |_| false);
        let Outcome::Busy { error, holder } = outcome else {
            panic!("{outcome:?}")
        };
        assert_eq!(error.kind(), io::ErrorKind::AddrInUse);
        assert_eq!(holder, Holder::Frozen { pid: 4242 });
    }

    #[test]
    fn a_window_that_does_not_answer_is_never_raised_and_only_waited_for_briefly() {
        // Frozen from the first look: no raise (the closure panics), the frozen wait, then Busy.
        let l = fast();
        let (outcome, waited, binds) = wait_in_vain(&l, true, io::ErrorKind::AddrInUse, Holder::Frozen { pid: 4242 });
        assert!(waited >= l.frozen, "{waited:?}");
        assert!(binds <= most_binds(l.frozen, &l), "{binds} binds in the frozen wait");
        assert!(
            matches!(
                outcome,
                Outcome::Busy {
                    holder: Holder::Frozen { pid: 4242 },
                    ..
                }
            ),
            "{outcome:?}"
        );

        // It comes back during the wait: then it is raised like any running Sigil.
        let polls = Cell::new(0u32);
        let raised = Cell::new(false);
        let outcome = wait_for_port(
            &patient(),
            true,
            || Err(in_use()),
            || {
                polls.set(polls.get() + 1);
                if polls.get() <= 2 {
                    Holder::Frozen { pid: 4242 }
                } else {
                    sigil(Some(42))
                }
            },
            |_| {
                raised.set(true);
                true
            },
        );
        assert!(matches!(outcome, Outcome::Raised) && raised.get(), "{outcome:?}");
    }

    #[test]
    fn the_window_of_a_copy_that_was_just_closed_is_not_taken_for_a_running_one() {
        // First poll: the old window is still there. Second: gone. Then the port is free.
        let polls = Cell::new(0u32);
        let outcome = wait_for_port(
            &patient(),
            true,
            busy_for(2, io::ErrorKind::AddrInUse),
            || {
                polls.set(polls.get() + 1);
                if polls.get() == 1 {
                    sigil(Some(42))
                } else {
                    sigil(None)
                }
            },
            |_| panic!("a window seen once must not be raised"),
        );
        assert!(matches!(outcome, Outcome::Free), "{outcome:?}");

        // A different window of the process in between starts the count again.
        let polls = Cell::new(0u32);
        let outcome = wait_for_port(
            &patient(),
            true,
            busy_for(3, io::ErrorKind::AddrInUse),
            || {
                polls.set(polls.get() + 1);
                sigil(Some(if polls.get().is_multiple_of(2) { 41 } else { 40 }))
            },
            |_| panic!("two different windows are not one sighting twice"),
        );
        assert!(matches!(outcome, Outcome::Free), "{outcome:?}");
    }

    #[test]
    fn automated_runs_never_raise_another_window_and_do_not_wait_long_for_a_running_one() {
        let l = fast();
        let (outcome, waited, binds) = wait_in_vain(&l, false, io::ErrorKind::AddrInUse, sigil(Some(42)));
        assert!(
            matches!(
                outcome,
                Outcome::Busy {
                    holder: Holder::Sigil { window: Some(42), .. },
                    ..
                }
            ),
            "{outcome:?}"
        );
        // The short wait, not the full one (a count that cannot flake: see `wait_in_vain`).
        assert!(waited >= Duration::from_millis(60), "{waited:?}");
        assert!(binds <= most_binds(l.short, &l), "{binds} binds: waited like for a closing copy");

        // A copy without a window is still the one that is on its way out: the full wait, also for automated runs.
        let (_, waited, _) = wait_in_vain(&l, false, io::ErrorKind::AddrInUse, sigil(None));
        assert!(waited >= Duration::from_millis(400), "{waited:?}");
    }

    #[test]
    fn next_step_decides_from_the_holder_and_the_time_waited() {
        let l = fast();
        let kind = io::ErrorKind::AddrInUse;
        let ms = Duration::from_millis;
        // A window seen often enough: raise at once, whatever the time.
        assert_eq!(next_step(&l, true, kind, &sigil(Some(9)), ms(0), 2), Step::Raise(9));
        assert_eq!(next_step(&l, true, kind, &sigil(Some(9)), ms(0), 1), Step::Wait);
        assert_eq!(next_step(&l, false, kind, &sigil(Some(9)), ms(0), 5), Step::Wait);
        // Times out per holder.
        assert_eq!(next_step(&l, true, kind, &sigil(None), ms(399), 0), Step::Wait);
        assert_eq!(next_step(&l, true, kind, &sigil(None), ms(400), 0), Step::GiveUp);
        assert_eq!(next_step(&l, false, kind, &sigil(None), ms(399), 0), Step::Wait);
        assert_eq!(next_step(&l, true, kind, &other(), ms(59), 0), Step::Wait);
        assert_eq!(next_step(&l, true, kind, &other(), ms(60), 0), Step::GiveUp);
        assert_eq!(next_step(&l, true, kind, &Holder::Frozen { pid: 1 }, ms(99), 0), Step::Wait);
        assert_eq!(next_step(&l, true, kind, &Holder::Frozen { pid: 1 }, ms(100), 0), Step::GiveUp);
        assert_eq!(next_step(&l, true, kind, &Holder::Unknown, ms(60), 0), Step::Wait);
        assert_eq!(
            next_step(&l, true, io::ErrorKind::PermissionDenied, &Holder::Unknown, ms(60), 0),
            Step::GiveUp
        );
        // A visible running Sigil that an automated run may not raise: the short wait.
        assert_eq!(next_step(&l, false, kind, &sigil(Some(9)), ms(59), 1), Step::Wait);
        assert_eq!(next_step(&l, false, kind, &sigil(Some(9)), ms(60), 1), Step::GiveUp);
        assert_eq!(
            next_step(&l, true, kind, &sigil(Some(9)), ms(60), 1),
            Step::Wait,
            "a start that may raise it still waits for a second sighting"
        );
        // The production wait outlasts the exit guard; a turn outlasts the start in front.
        assert!(Limits::APP.full > exit_guard::GRACE);
        assert!(Limits::APP.short < Limits::APP.full);
        assert!(Limits::APP.turn > Limits::APP.full + Limits::APP.hold);
        assert!(Limits::APP.raise < Limits::APP.short);
        assert!(
            Limits::APP.frozen > Duration::from_secs(5),
            "longer than the 5 s after which Windows calls a window hung"
        );
        assert!(Limits::APP.frozen < Limits::APP.full);
    }

    #[test]
    fn the_box_says_what_holds_the_port_and_what_to_do() {
        let code = io::Error::from_raw_os_error(10048);
        let text = describe(3030, &code, &other());
        assert!(text.contains("Port 3030 is not available!"), "{text}");
        assert!(text.contains("node.exe, process 77"), "{text}");
        assert!(text.contains("--port 3031"), "{text}");
        assert!(text.contains("Windows error 10048"), "{text}");

        let text = describe(3030, &code, &sigil(None));
        assert!(text.contains("still running without a window"), "{text}");
        assert!(text.contains("20 seconds"), "{text}");

        let text = describe(3030, &code, &sigil(Some(1)));
        assert!(text.starts_with("Sigil is already running"), "{text}");

        let text = describe(3030, &code, &Holder::Frozen { pid: 4242 });
        assert!(
            text.contains("process 4242") && text.contains("not responding") && text.contains("Task Manager"),
            "{text}"
        );

        let denied = io::Error::from_raw_os_error(10013);
        assert_eq!(denied.kind(), io::ErrorKind::PermissionDenied);
        let text = describe(3030, &denied, &Holder::Unknown);
        assert!(text.contains("netsh int ipv4 show excludedportrange"), "{text}");
        assert!(text.contains("Windows error 10013"), "{text}");

        let text = describe(3030, &code, &Holder::Unknown);
        assert!(text.contains("Another program is using it"), "{text}");
        // The last port still names another one.
        assert!(describe(u16::MAX, &code, &Holder::Unknown).contains("--port 65534"));
    }

    #[test]
    fn only_a_process_with_this_programs_image_name_counts_as_sigil() {
        let window = |pid: u32| (pid == 4242).then_some(42isize);
        let answers = |_: isize| false;
        let exe = |name: &'static str| move |_pid: u32| Some(name.to_string());
        assert_eq!(classify(4242, Some("Sigil.exe"), exe("sigil.exe"), window, answers), sigil(Some(42)));
        assert_eq!(
            classify(9, Some("Sigil.exe"), exe("Sigil.exe"), window, answers),
            Holder::Sigil { pid: 9, window: None }
        );
        assert_eq!(
            classify(4242, Some("Sigil.exe"), exe("node.exe"), window, answers),
            Holder::Other {
                pid: 4242,
                exe: Some("node.exe".into())
            }
        );
        // A process this one may not look at is another program, never a Sigil to raise.
        assert_eq!(
            classify(4242, Some("Sigil.exe"), |_| None, window, answers),
            Holder::Other { pid: 4242, exe: None }
        );
        assert_eq!(
            classify(4242, None, exe("Sigil.exe"), window, answers),
            Holder::Other {
                pid: 4242,
                exe: Some("Sigil.exe".into())
            }
        );
    }

    #[test]
    fn a_sigil_window_that_does_not_answer_is_frozen_not_raised() {
        let window = |_: u32| Some(42isize);
        let exe = |_: u32| Some("Sigil.exe".to_string());
        assert_eq!(
            classify(4242, Some("Sigil.exe"), exe, window, |hwnd| hwnd == 42),
            Holder::Frozen { pid: 4242 }
        );
        assert_eq!(classify(4242, Some("Sigil.exe"), exe, window, |_| false), sigil(Some(42)));
        // No window at all is not "frozen": that copy is on its way out.
        assert_eq!(classify(4242, Some("Sigil.exe"), exe, |_| None, |_| true), sigil(None));
    }

    fn listener(ip: &str, port: u16, pid: u32) -> ports::Listener {
        ports::Listener {
            addr: std::net::SocketAddr::new(ip.parse().unwrap(), port),
            pid,
        }
    }

    #[test]
    fn a_listener_on_any_address_of_the_port_is_the_holder() {
        let exe = |pid: u32| Some(if pid == 9 { "Sigil.exe" } else { "docker-proxy.exe" }.to_string());
        let window = |pid: u32| (pid == 9).then_some(42isize);
        let holder = |listeners: &[ports::Listener]| holder_of(listeners, 3030, Some("Sigil.exe"), exe, window, |_| false);

        // A relay bound to one LAN address: the wildcard bind fails, a loopback connection finds nobody.
        let table = [listener("127.0.0.1", 8080, 1), listener("192.168.1.5", 3030, 77)];
        assert_eq!(
            holder(&table),
            Holder::Other {
                pid: 77,
                exe: Some("docker-proxy.exe".into())
            }
        );

        // The listener that answers loopback connections wins over one on a LAN address.
        let table = [listener("192.168.1.5", 3030, 77), listener("0.0.0.0", 3030, 9)];
        assert_eq!(holder(&table), Holder::Sigil { pid: 9, window: Some(42) });

        // Another port, or none: nothing known.
        assert_eq!(holder(&[listener("192.168.1.5", 3031, 77)]), Holder::Unknown);
        assert_eq!(holder(&[]), Holder::Unknown);
    }

    #[test]
    fn the_lookups_name_a_real_holder_of_a_loopback_port() {
        // This test process stands in for the other copy: same image name, no window.
        let holder = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = holder.local_addr().unwrap().port();
        let own = own_exe_name().expect("own exe name");
        assert_eq!(
            process_exe_name(std::process::id())
                .as_deref()
                .map(str::to_ascii_lowercase),
            Some(own.to_ascii_lowercase())
        );
        assert_eq!(main_window_of(std::process::id()), None);
        assert_eq!(
            inspect(port),
            Holder::Sigil {
                pid: std::process::id(),
                window: None
            }
        );
        drop(holder);
        assert_eq!(inspect(port), Holder::Unknown, "nothing listens there any more");
    }

    // --- a real top-level window ---

    /// A window of this test process that nobody sees: a popup far off screen that never takes
    /// the activation (and has no taskbar button). Destroyed on the thread that made it.
    struct TestWindow(HWND);

    impl TestWindow {
        fn new(title: &str, owner: Option<&TestWindow>, visible: bool) -> TestWindow {
            let wide: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
            let style = if visible { WS_POPUP | WS_VISIBLE } else { WS_POPUP };
            // SAFETY: a system "STATIC" window with a NUL-terminated title; no window procedure of ours is involved.
            let hwnd = unsafe {
                CreateWindowExW(
                    WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
                    w!("STATIC"),
                    PCWSTR(wide.as_ptr()),
                    style,
                    -32000,
                    -32000,
                    1,
                    1,
                    owner.map(|o| o.0),
                    None,
                    None,
                    None,
                )
            }
            .expect("create a test window");
            TestWindow(hwnd)
        }

        fn id(&self) -> isize {
            self.0 .0 as isize
        }
    }

    impl Drop for TestWindow {
        fn drop(&mut self) {
            // SAFETY: a window this thread created, destroyed once.
            unsafe {
                let _ = DestroyWindow(self.0);
            }
        }
    }

    fn unique_title(what: &str) -> String {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        format!("Sigil test {what} {} {}", std::process::id(), N.fetch_add(1, Ordering::SeqCst))
    }

    #[test]
    fn the_main_window_is_the_visible_unowned_window_with_the_exact_title() {
        let pid = std::process::id();
        let title = unique_title("main");
        let main = TestWindow::new(&title, None, true);
        assert_eq!(main_window_titled(pid, &title), Some(main.id()));
        assert!(!window_is_hung(main.id()), "a window that was just made answers");

        // Not the title: a longer one that starts the same way is not cut down to a match.
        let longer = unique_title("longer");
        let _longer = TestWindow::new(&format!("{longer} - Explorer"), None, true);
        assert_eq!(main_window_titled(pid, &longer), None);

        // Not visible, not unowned, not of that process.
        let hidden_title = unique_title("hidden");
        let _hidden = TestWindow::new(&hidden_title, None, false);
        assert_eq!(main_window_titled(pid, &hidden_title), None);
        let owned_title = unique_title("owned");
        let _owned = TestWindow::new(&owned_title, Some(&main), true);
        assert_eq!(
            main_window_titled(pid, &owned_title),
            None,
            "a window with an owner is a dialog, not the main window"
        );
        assert_eq!(main_window_titled(pid.wrapping_add(4), &title), None);

        // The production lookup uses the real title, which no test window has.
        assert_eq!(main_window_of(pid), None);
    }

    #[test]
    fn the_title_that_is_looked_for_is_the_one_the_main_window_has() {
        // A wrong title would turn every second start into a 25 s silent wait and a box.
        let conf: serde_json::Value = serde_json::from_str(include_str!("../tauri.conf.json")).expect("tauri.conf.json");
        let windows = conf["app"]["windows"].as_array().expect("app.windows");
        let main = windows
            .iter()
            .find(|w| w["label"].as_str().unwrap_or("main") == "main")
            .expect("the main window");
        assert_eq!(main["title"].as_str(), Some(MAIN_WINDOW_TITLE));
        assert!(
            main["visible"].as_bool().unwrap_or(true),
            "a window created hidden cannot be found or raised"
        );
    }

    #[test]
    fn a_raise_that_does_not_return_in_time_is_given_up_on() {
        assert!(finishes_within(Duration::from_secs(5), || {}));
        assert!(!finishes_within(Duration::from_millis(20), || std::thread::sleep(Duration::from_millis(
            400
        ))));
        // A handle that is no window: the calls fail harmlessly and the raise returns at once.
        assert!(bring_to_front_within(Duration::from_secs(5), 0));
    }

    // --- start lock ---

    #[test]
    fn starts_on_one_port_take_turns() {
        let name = format!("Local\\Sigil-start-test-{}", std::process::id());
        let first = match create_lock(&name) {
            Turn::Taken(handle) => handle,
            other => panic!("{other:?}"),
        };
        assert_eq!(create_lock(&name), Turn::Held, "a second start does not get it");
        assert_eq!(create_lock(&name), Turn::Held, "and looking does not keep the lock object alive");

        let held = Held::new();
        held.keep(first);
        assert_eq!(create_lock(&name), Turn::Held, "kept while the start runs");
        held.release();
        held.release(); // idempotent
        let second = match create_lock(&name) {
            Turn::Taken(handle) => handle,
            other => panic!("the turn was given up, so the next start takes it: {other:?}"),
        };
        close_handle(second);

        assert_eq!(lock_name(3030), "Local\\Sigil-start-3030");
        assert_ne!(lock_name(3030), lock_name(3031), "other ports do not wait for each other");
    }

    #[test]
    fn a_start_waits_for_its_turn_and_goes_on_without_a_lock_that_stays_taken() {
        let l = fast();
        // The start in front is done after a few looks.
        let looks = Cell::new(0u32);
        let got = take_turn(&l, || {
            looks.set(looks.get() + 1);
            if looks.get() <= 3 {
                Turn::Held
            } else {
                Turn::Taken(7)
            }
        });
        assert_eq!((got, looks.get()), (Some(7), 4));

        // Taken at once: no wait at all.
        assert_eq!(take_turn(&l, || Turn::Taken(8)), Some(8));

        // A lock that cannot be used here never delays a start.
        let looks = Cell::new(0u32);
        assert_eq!(
            take_turn(&l, || {
                looks.set(looks.get() + 1);
                Turn::Unavailable
            }),
            None
        );
        assert_eq!(looks.get(), 1);

        // A lock that stays taken is given up on after `turn`.
        let started = Instant::now();
        assert_eq!(take_turn(&l, || Turn::Held), None);
        assert!(started.elapsed() >= l.turn, "{:?}", started.elapsed());
    }

    // --- log ---

    #[test]
    fn the_start_up_log_line_goes_to_the_app_log_in_its_shape() {
        let line = format_log_line(
            "2026-10-12T10:20:30.123456Z",
            "Sigil did not start: Port 3030 is not available!\n\nAnother program is using it.",
        );
        assert_eq!(
            line,
            "2026-10-12T10:20:30.123456Z  WARN main sigil::startup: Sigil did not start: Port 3030 is not available! Another program is using it.\n"
        );

        let dir = std::env::temp_dir().join(format!("sigil-precheck-log-{}", std::process::id()));
        let path = dir.join("logs").join("sigil.log");
        let _ = fs::remove_dir_all(&dir);
        append_log_line(&path, "first\n").unwrap();
        append_log_line(&path, "second\n").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "first\nsecond\n", "appended, not replaced");
        let _ = fs::remove_dir_all(&dir);

        // The folder is the one Tauri logs to: `<LOCALAPPDATA>\<bundle identifier>\logs\sigil.log`.
        let conf = include_str!("../tauri.conf.json");
        assert!(
            conf.contains(&format!("\"identifier\": \"{LOG_FOLDER}\"")),
            "the bundle identifier changed; update LOG_FOLDER"
        );
        if let Some(path) = log_file_path() {
            assert!(path.ends_with(Path::new(LOG_FOLDER).join("logs").join("sigil.log")), "{path:?}");
        }
    }
}
