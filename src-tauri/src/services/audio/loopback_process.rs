//! Per-process loopback capture: what one process tree plays, e.g. `VRChat.exe` (P6-A).
//!
//! Source ids `"process:<exe name>"` ([`super::loopback::PROCESS_PREFIX`], `SourceSpec::Process`)
//! are routed here by the hub's cpal backend (`stream.rs`):
//! - [`resolve`] finds the running process (the root of its same-name process tree) and returns a
//!   device named `process:<exe>` with `loopback: true` and the fixed [`CAPTURE_FORMAT`] (the
//!   process-loopback virtual device has no mix format; Windows converts to what we ask for). It
//!   fails while the process is not running or on Windows older than build [`MIN_WINDOWS_BUILD`],
//!   so the caller falls back to device loopback (`speech::session::plan_source`).
//! - [`open`] runs the capture on its own thread (`sigil-capture-process`) through the `wasapi`
//!   crate's application-loopback client (`AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK`,
//!   `PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE`, event-driven shared mode) and pushes
//!   blocks into the hub's [`StreamSink`] under the same real-time rules as `stream.rs`: MMCSS on
//!   the capture thread only, no allocation / locking / logging / IPC per block (one scratch buffer
//!   sized at open), QPC capture times.
//!
//! # Behaviour a consumer can rely on
//! - **Silence:** like device loopback, the capture may deliver nothing while the program is silent.
//!   After [`IDLE_AFTER`] without packets the thread delivers up to [`IDLE_FILL`] of real-time-paced
//!   silence, so voice detection closes an utterance cut off by the silence even for consumers that
//!   only idle-fill device loopback. Packets flagged silent are delivered as zeros.
//! - **The program exits:** the stream stays up and waits for a process with the same name (checked
//!   every [`RESCAN_INTERVAL`]), then captures it under the same hub epoch, so closing and starting
//!   the game neither raises an `audio:stream_error` nor needs a restart. Only a real capture
//!   failure (the audio service, an activation refused for the new process) reports an error once
//!   (`ErrorSink`); the hub then retries like a removed device.
//!
//! # Frozen contract (P6-S0; change only through INTEGRATE; structs may gain fields)
//! - Rust: [`supported`]`() -> bool`, [`is_process_device`]`(&ResolvedDevice) -> bool`,
//!   [`resolve`]`(exe) -> Result<ResolvedDevice, String>` (the device is named `process:<exe>`,
//!   `loopback: true`), [`open`]`(ResolvedDevice, StreamSink, ErrorSink) -> Result<Box<dyn CaptureStream>, String>`.
//! - IPC `plugin:audio|list_loopback_processes -> LoopbackProcessList` (TS
//!   `RustLoopbackProcessList`): whether per-process capture works here, and the running
//!   processes that can be captured (for the Others picker). P6-A additions (optional, backward
//!   compatible): the arguments `source` / `fallback` ask for the Others source plan
//!   ([`LoopbackProcessList::plan`], `speech::session::SourcePlan`), `processes: false` skips the
//!   program list (the Others service's periodic plan check), rows say whether the program has an
//!   audio session ([`LoopbackProcess::audio`]), and the result names the Windows build.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::command;

use super::hub::{CaptureStream, ErrorSink, ResolvedDevice, StreamFormat, StreamSink};
use super::loopback::PROCESS_PREFIX;
use crate::services::speech::session::{self, ProcessProbe, SourcePlan};

/// First Windows build with the process-loopback activation type (Windows 10 21H2 server /
/// Windows 11).
pub const MIN_WINDOWS_BUILD: u32 = 20348;
/// What the process-loopback client is asked to deliver (interleaved f32): the virtual device has no
/// mix format of its own, and 48 kHz stereo is what games render at (the hub downmixes and
/// resamples per consumer).
pub const CAPTURE_FORMAT: StreamFormat = StreamFormat {
    sample_rate: 48_000,
    channels: 2,
};
/// No packets for this long counts as silence (the idle fill starts).
pub const IDLE_AFTER: Duration = Duration::from_millis(300);
/// Silence delivered per silent stretch (enough for voice detection's longest default silence).
pub const IDLE_FILL: Duration = Duration::from_millis(2_500);
/// How often a waiting stream looks for the program again after it exited.
pub const RESCAN_INTERVAL: Duration = Duration::from_secs(1);
/// Longest program list returned (audio programs first).
const MAX_LIST: usize = 64;
/// Window titles are cut to this many characters.
const MAX_TITLE_CHARS: usize = 120;
/// How long opening the capture may take before it is abandoned.
const OPEN_TIMEOUT: Duration = Duration::from_secs(5);
const LOG: &str = "sigil::audio::process";

/// Shell and system hosts that own windows but are never "a program the user listens to": left out
/// of the picker unless they hold an audio session.
const WINDOW_ONLY_DENY: &[&str] = &[
    "explorer.exe",
    "ApplicationFrameHost.exe",
    "TextInputHost.exe",
    "ShellExperienceHost.exe",
    "StartMenuExperienceHost.exe",
    "SearchHost.exe",
    "SearchApp.exe",
    "SystemSettings.exe",
    "LockApp.exe",
    "dwm.exe",
    "csrss.exe",
    "winlogon.exe",
    "svchost.exe",
];

/// A process the Others picker can offer (TS `RustLoopbackProcess`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoopbackProcess {
    pub pid: u32,
    /// Executable file name (`VRChat.exe`): what `process:<exe>` names.
    pub exe: String,
    /// Main window title, when it has one.
    pub title: Option<String>,
    /// The program (or a child of the same name) holds an audio session on an output device: it plays
    /// or recently played sound (P6-A addition).
    #[serde(default)]
    pub audio: bool,
}

/// Result of `list_loopback_processes` (TS `RustLoopbackProcessList`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoopbackProcessList {
    /// Per-process capture works on this PC (Windows build and API available).
    pub supported: bool,
    pub processes: Vec<LoopbackProcess>,
    /// Windows build number (P6-A addition; `None` off Windows or when unknown).
    #[serde(default)]
    pub windows_build: Option<u32>,
    /// The Others source plan for the `source` argument (P6-A addition; absent without it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<SourcePlan>,
}

// ---------------------------------------------------------------------------------------------
// Names and process trees (pure)
// ---------------------------------------------------------------------------------------------

/// Canonical program name for `name`: the file name (directories dropped), trimmed, with `.exe`
/// added when it has no extension (`VRChat` → `VRChat.exe`). `None` when nothing usable is left.
pub fn normalize_exe(name: &str) -> Option<String> {
    let base = name.trim().rsplit(['\\', '/']).next().unwrap_or("").trim();
    if base.is_empty() || base.chars().all(|c| c == '.') || base.chars().count() > 255 {
        return None;
    }
    if base.chars().any(|c| c.is_control() || matches!(c, ':' | '*' | '?' | '"' | '<' | '>' | '|')) {
        return None;
    }
    let has_extension = base
        .rsplit_once('.')
        .is_some_and(|(stem, ext)| !stem.is_empty() && !ext.is_empty() && !ext.contains(' '));
    Some(if has_extension { base.to_string() } else { format!("{base}.exe") })
}

/// Whether two executable names are the same file name (Windows compares names case-insensitively).
pub fn same_exe(a: &str, b: &str) -> bool {
    a.chars().flat_map(char::to_lowercase).eq(b.chars().flat_map(char::to_lowercase))
}

/// One row of a process snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProcEntry {
    pub pid: u32,
    pub parent: u32,
    pub exe: String,
}

/// The process to capture for `exe`: the root of a same-name process tree (a child of the same name
/// is covered by capturing the tree), preferring one with a window (`has_window`), then the lowest
/// id. `None` when no process has that name.
pub(crate) fn find_root<'a>(procs: &'a [ProcEntry], exe: &str, has_window: &dyn Fn(u32) -> bool) -> Option<&'a ProcEntry> {
    let named: Vec<&ProcEntry> = procs.iter().filter(|p| same_exe(&p.exe, exe)).collect();
    let is_named = |pid: u32| named.iter().any(|p| p.pid == pid);
    let pick = |list: &mut dyn Iterator<Item = &'a ProcEntry>| list.min_by_key(|p| (!has_window(p.pid), p.pid));
    let mut roots = named
        .iter()
        .copied()
        .filter(|p| p.parent == 0 || p.parent == p.pid || !is_named(p.parent));
    // A parent cycle (reused ids) leaves no root: any process of that name will do.
    pick(&mut roots).or_else(|| pick(&mut named.iter().copied()))
}

/// The root of `pid`'s same-name process tree (parents with the same executable name).
fn root_of<'a>(by_pid: &HashMap<u32, &'a ProcEntry>, entry: &'a ProcEntry) -> &'a ProcEntry {
    let mut current = entry;
    for _ in 0..64 {
        match by_pid.get(&current.parent) {
            Some(&parent) if parent.pid != current.pid && same_exe(&parent.exe, &current.exe) => current = parent,
            _ => break,
        }
    }
    current
}

fn clip_title(title: &str) -> Option<String> {
    let t = title.trim();
    if t.is_empty() {
        return None;
    }
    Some(if t.chars().count() > MAX_TITLE_CHARS {
        let mut s: String = t.chars().take(MAX_TITLE_CHARS - 1).collect();
        s.push('…');
        s
    } else {
        t.to_string()
    })
}

/// The picker's program list from a snapshot: programs holding an audio session (`audio`) or a
/// titled top-level window (`titles`), one row per executable name (the root of its tree), without
/// Sigil itself and the shell's window hosts. Audio programs first, then by name.
pub(crate) fn build_list(
    procs: &[ProcEntry],
    titles: &HashMap<u32, String>,
    audio: &HashSet<u32>,
    own_pid: u32,
) -> Vec<LoopbackProcess> {
    let by_pid: HashMap<u32, &ProcEntry> = procs.iter().map(|p| (p.pid, p)).collect();
    let own_exe = by_pid.get(&own_pid).map(|p| p.exe.clone());
    let mut rows: HashMap<String, LoopbackProcess> = HashMap::new();
    let candidates: HashSet<u32> = audio.iter().chain(titles.keys()).copied().collect();
    for pid in candidates {
        if pid == 0 || pid == 4 || pid == own_pid {
            continue;
        }
        let Some(entry) = by_pid.get(&pid).copied() else { continue };
        if own_exe.as_deref().is_some_and(|own| same_exe(own, &entry.exe)) {
            continue;
        }
        let has_audio = audio.contains(&pid);
        if !has_audio && WINDOW_ONLY_DENY.iter().any(|d| same_exe(d, &entry.exe)) {
            continue;
        }
        let root = root_of(&by_pid, entry);
        let title = titles.get(&root.pid).or_else(|| titles.get(&pid)).and_then(|t| clip_title(t));
        let key = root.exe.to_lowercase();
        let candidate = LoopbackProcess {
            pid: root.pid,
            exe: root.exe.clone(),
            title,
            audio: has_audio,
        };
        match rows.get_mut(&key) {
            None => {
                rows.insert(key, candidate);
            }
            Some(row) if row.pid == candidate.pid => {
                row.audio |= candidate.audio;
                if row.title.is_none() {
                    row.title = candidate.title;
                }
            }
            Some(row) => {
                // Another instance of the same program: keep the one that plays sound, then the one
                // with a window, then the lowest id.
                let rank = |r: &LoopbackProcess| (!r.audio, r.title.is_none(), r.pid);
                if rank(&candidate) < rank(row) {
                    *row = candidate;
                }
            }
        }
    }
    let mut list: Vec<LoopbackProcess> = rows.into_values().collect();
    list.sort_by_key(|a| (!a.audio, a.exe.to_lowercase()));
    list.truncate(MAX_LIST);
    list
}

/// A snapshot as a [`ProcessProbe`] for the session planner.
pub(crate) struct SnapshotProbe {
    pub supported: bool,
    pub procs: Vec<ProcEntry>,
}

impl ProcessProbe for SnapshotProbe {
    fn supported(&self) -> bool {
        self.supported
    }

    fn running(&self, exe: &str) -> Option<String> {
        find_root(&self.procs, exe, &|_| false).map(|p| p.exe.clone())
    }
}

/// A fresh probe of this PC (one process snapshot; empty when it fails).
#[cfg_attr(not(test), allow(dead_code))] // `speech::session::plan_source` (Rust callers, tests).
pub(crate) fn system_probe() -> SnapshotProbe {
    SnapshotProbe {
        supported: supported(),
        procs: sys::snapshot().unwrap_or_default(),
    }
}

// ---------------------------------------------------------------------------------------------
// Frozen API
// ---------------------------------------------------------------------------------------------

/// The Windows build number (cached; `None` off Windows or when it cannot be read).
pub fn windows_build() -> Option<u32> {
    static BUILD: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *BUILD.get_or_init(sys::windows_build)
}

/// Whether per-process loopback works on this PC.
pub fn supported() -> bool {
    cfg!(windows) && windows_build().is_some_and(|b| b >= MIN_WINDOWS_BUILD)
}

/// Whether `device` was resolved by [`resolve`] (a `process:<exe>` loopback device).
pub fn is_process_device(device: &ResolvedDevice) -> bool {
    device.loopback
        && device
            .name
            .get(..PROCESS_PREFIX.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(PROCESS_PREFIX))
}

/// What [`resolve`] hands to [`open`].
struct ProcessTarget {
    pid: u32,
    exe: String,
}

/// Find the running process `exe` and its capture format.
pub fn resolve(exe: &str) -> Result<ResolvedDevice, String> {
    let wanted = normalize_exe(exe).ok_or_else(|| format!("'{exe}' is not a program name"))?;
    if !supported() {
        return Err(unsupported_message());
    }
    let procs = sys::snapshot()?;
    let titles = sys::window_titles();
    let root = find_root(&procs, &wanted, &|pid| titles.contains_key(&pid))
        .ok_or_else(|| format!("{wanted} is not running"))?;
    Ok(ResolvedDevice {
        name: format!("{PROCESS_PREFIX}{}", root.exe),
        loopback: true,
        format: CAPTURE_FORMAT,
        handle: Some(Box::new(ProcessTarget {
            pid: root.pid,
            exe: root.exe.clone(),
        })),
    })
}

fn unsupported_message() -> String {
    match windows_build() {
        Some(build) => format!(
            "Capturing one program's sound needs Windows 11 or Windows 10 build {MIN_WINDOWS_BUILD}+ (this PC: build {build})"
        ),
        None => "Capturing one program's sound needs Windows 11 or Windows 10 build 20348+".to_string(),
    }
}

/// Start capturing `device` (from [`resolve`]) into `sink`.
pub fn open(device: ResolvedDevice, sink: StreamSink, errors: ErrorSink) -> Result<Box<dyn CaptureStream>, String> {
    if device.format != CAPTURE_FORMAT {
        return Err(format!("Unexpected capture format for '{}'", device.name));
    }
    let target = match device.handle.and_then(|h| h.downcast::<ProcessTarget>().ok()) {
        Some(target) => *target,
        None => {
            let exe = device.name.get(PROCESS_PREFIX.len()..).unwrap_or("");
            let resolved = resolve(exe)?;
            match resolved.handle.and_then(|h| h.downcast::<ProcessTarget>().ok()) {
                Some(target) => *target,
                None => return Err(format!("{exe} is not running")),
            }
        }
    };
    sys::open_capture(target, sink, errors)
}

/// The processes whose audio can be captured. With `source` (an Others capture source id), also the
/// Others source plan (`fallback`: capture the default output while a requested program is not
/// running; default on). `processes: false` skips the program list (a plan check: one process
/// snapshot, no window or audio-session enumeration).
#[command]
pub async fn list_loopback_processes(
    source: Option<String>,
    fallback: Option<bool>,
    processes: Option<bool>,
) -> Result<LoopbackProcessList, String> {
    tauri::async_runtime::spawn_blocking(move || list_now(source.as_deref(), fallback.unwrap_or(true), processes.unwrap_or(true)))
        .await
        .map_err(|e| format!("Listing programs failed: {e}"))
}

fn list_now(source: Option<&str>, fallback: bool, with_processes: bool) -> LoopbackProcessList {
    let supported = supported();
    let procs = sys::snapshot().unwrap_or_else(|e| {
        tracing::warn!(target: LOG, "process snapshot failed: {e}");
        Vec::new()
    });
    let processes = if supported && with_processes {
        let titles = sys::window_titles();
        let audio = sys::audio_session_pids();
        build_list(&procs, &titles, &audio, std::process::id())
    } else {
        Vec::new()
    };
    let plan = source.map(|s| session::plan_source_with(s, fallback, &SnapshotProbe { supported, procs }));
    LoopbackProcessList {
        supported,
        processes,
        windows_build: windows_build(),
        plan,
    }
}

// ---------------------------------------------------------------------------------------------
// Capture timing (pure)
// ---------------------------------------------------------------------------------------------

/// Longest believable age of a packet when it is read (older stamps are driver garbage).
const MAX_CAPTURE_AGE_NS: i64 = 1_000_000_000;

/// QPC capture time (ns) of a packet stamped `qpc_100ns` (WASAPI's `u64QPCPosition`, 100 ns units),
/// read at `now_ns`: the stamp when it is plausible, else the read time.
pub(crate) fn packet_qpc_ns(qpc_100ns: u64, timestamp_error: bool, now_ns: i64) -> i64 {
    if timestamp_error || qpc_100ns == 0 {
        return now_ns;
    }
    match i64::try_from(qpc_100ns).ok().and_then(|t| t.checked_mul(100)) {
        Some(ns) if ns <= now_ns && now_ns - ns <= MAX_CAPTURE_AGE_NS => ns,
        _ => now_ns,
    }
}

/// Paces the silence a quiet process stream delivers (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IdleFill {
    /// QPC time (ns) just past the last audio delivered (real or filled).
    end_ns: i64,
    /// Silence delivered since the last real packet.
    filled_ns: i64,
}

impl IdleFill {
    pub(crate) fn new(now_ns: i64) -> Self {
        Self { end_ns: now_ns, filled_ns: 0 }
    }

    /// Real audio ended at `end_ns`.
    pub(crate) fn audio(&mut self, end_ns: i64) {
        self.end_ns = self.end_ns.max(end_ns);
        self.filled_ns = 0;
    }

    /// Frames of silence to deliver now (at `rate`), starting at the returned QPC time; `None` when
    /// nothing is due.
    pub(crate) fn due(&mut self, now_ns: i64, rate: u32) -> Option<(i64, usize)> {
        let budget = i64::try_from(IDLE_FILL.as_nanos()).unwrap_or(i64::MAX);
        let after = i64::try_from(IDLE_AFTER.as_nanos()).unwrap_or(i64::MAX);
        let quiet = now_ns.saturating_sub(self.end_ns);
        if self.filled_ns >= budget || (self.filled_ns == 0 && quiet < after) || quiet <= 0 {
            return None;
        }
        let span = quiet.min(budget - self.filled_ns);
        let frames = usize::try_from(i128::from(span) * i128::from(rate) / 1_000_000_000).unwrap_or(0);
        if frames == 0 {
            return None;
        }
        let start = self.end_ns;
        let delivered = (i128::from(frames as u64) * 1_000_000_000 / i128::from(rate.max(1))) as i64;
        self.end_ns += delivered;
        self.filled_ns += delivered;
        Some((start, frames))
    }
}

// ---------------------------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------------------------

#[cfg(windows)]
mod sys {
    use std::collections::{HashMap, HashSet};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Instant;

    use wasapi::{AudioCaptureClient, AudioClient, Direction, Handle, SampleType, StreamMode, WaveFormat};
    use windows::core::{s, w, Interface, BOOL};
    use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND, LPARAM, WAIT_OBJECT_0};
    use windows::Win32::Media::Audio::{
        eRender, IAudioSessionControl2, IAudioSessionManager2, IMMDeviceEnumerator, MMDeviceEnumerator,
        DEVICE_STATE_ACTIVE,
    };
    use windows::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_ALL, COINIT_MULTITHREADED};
    use windows::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
    };
    use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
    use windows::Win32::System::Threading::{OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE};
    use windows::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GetWindow, GetWindowLongW, GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId,
        IsWindowVisible, GWL_EXSTYLE, GW_OWNER, WS_EX_TOOLWINDOW,
    };

    use super::{
        find_root, normalize_exe, packet_qpc_ns, IdleFill, ProcEntry, ProcessTarget, CAPTURE_FORMAT, LOG,
        OPEN_TIMEOUT, RESCAN_INTERVAL,
    };
    use crate::services::audio::hub::{now_qpc_ns, CaptureStream, ErrorSink, StreamSink};
    use crate::services::rt_thread;

    /// Event wait per loop turn (also the stop latency and the idle-fill pacing).
    const WAIT_MS: u32 = 100;
    /// How often a running capture checks whether its process is still alive.
    const ALIVE_CHECK_NS: i64 = 250_000_000;
    /// Shared-mode buffer asked for (100 ns units): room for scheduling hiccups of the capture thread.
    const BUFFER_HNS: i64 = 1_000_000;
    /// Scratch buffer: 2 s of interleaved samples (packets are about 10 ms).
    const SCRATCH_SECONDS: usize = 2;

    fn wide_to_string(wide: &[u16]) -> String {
        let end = wide.iter().position(|&c| c == 0).unwrap_or(wide.len());
        String::from_utf16_lossy(&wide[..end])
    }

    struct OwnedHandle(HANDLE);

    impl Drop for OwnedHandle {
        fn drop(&mut self) {
            // SAFETY: the handle was returned by a successful open and is closed once.
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }

    /// Every running process (id, parent id, executable name).
    pub fn snapshot() -> Result<Vec<ProcEntry>, String> {
        // SAFETY: plain Win32 calls with a correctly sized entry; the handle is closed by the guard.
        unsafe {
            let snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0).map_err(|e| format!("Process snapshot: {e}"))?;
            let _guard = OwnedHandle(snap);
            let mut entry = PROCESSENTRY32W {
                dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
                ..Default::default()
            };
            let mut out = Vec::with_capacity(256);
            if Process32FirstW(snap, &mut entry).is_ok() {
                loop {
                    out.push(ProcEntry {
                        pid: entry.th32ProcessID,
                        parent: entry.th32ParentProcessID,
                        exe: wide_to_string(&entry.szExeFile),
                    });
                    if Process32NextW(snap, &mut entry).is_err() {
                        break;
                    }
                }
            }
            Ok(out)
        }
    }

    unsafe extern "system" fn collect_window(hwnd: HWND, lparam: LPARAM) -> BOOL {
        // SAFETY: `lparam` is the `&mut Vec` passed by `window_titles` for the duration of EnumWindows.
        let out = unsafe { &mut *(lparam.0 as *mut Vec<(u32, String)>) };
        // SAFETY: plain user32 queries on a window handle EnumWindows handed us.
        unsafe {
            if !IsWindowVisible(hwnd).as_bool() || GetWindow(hwnd, GW_OWNER).is_ok_and(|owner| !owner.is_invalid()) {
                return BOOL(1);
            }
            if (GetWindowLongW(hwnd, GWL_EXSTYLE) as u32) & WS_EX_TOOLWINDOW.0 != 0 {
                return BOOL(1);
            }
            let len = GetWindowTextLengthW(hwnd);
            if len <= 0 {
                return BOOL(1);
            }
            let mut buf = vec![0u16; len as usize + 1];
            let n = GetWindowTextW(hwnd, &mut buf);
            if n <= 0 {
                return BOOL(1);
            }
            let mut pid = 0u32;
            GetWindowThreadProcessId(hwnd, Some(&mut pid as *mut u32));
            let title = String::from_utf16_lossy(&buf[..n as usize]);
            if pid != 0 && title != "Program Manager" {
                out.push((pid, title));
            }
        }
        BOOL(1)
    }

    /// Titles of visible, unowned, non-tool top-level windows by process (first window wins).
    pub fn window_titles() -> HashMap<u32, String> {
        let mut found: Vec<(u32, String)> = Vec::new();
        // SAFETY: the callback only touches `found` through the pointer, during this call.
        let _ = unsafe { EnumWindows(Some(collect_window), LPARAM(&mut found as *mut Vec<(u32, String)> as isize)) };
        let mut map = HashMap::new();
        for (pid, title) in found {
            map.entry(pid).or_insert(title);
        }
        map
    }

    /// Process ids holding an audio session on an active output device (any state but expired).
    /// Runs on its own thread with its own COM apartment.
    pub fn audio_session_pids() -> HashSet<u32> {
        let worker = thread::Builder::new().name("sigil-audio-sessions".into()).spawn(|| {
            // SAFETY: COM init on a fresh thread, balanced below when it succeeded.
            let com = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }.is_ok();
            let result = enumerate_sessions();
            if com {
                // SAFETY: balances the successful CoInitializeEx; every COM object above is dropped.
                unsafe { CoUninitialize() };
            }
            result
        });
        match worker.map(|h| h.join()) {
            Ok(Ok(Ok(pids))) => pids,
            Ok(Ok(Err(e))) => {
                tracing::debug!(target: LOG, "audio session list failed: {e}");
                HashSet::new()
            }
            _ => HashSet::new(),
        }
    }

    fn enumerate_sessions() -> windows::core::Result<HashSet<u32>> {
        let mut pids = HashSet::new();
        // SAFETY: COM calls on interfaces created in this apartment. `IMMDevice::Activate` needs the
        // windows features `Win32_System_Com_StructuredStorage` and `Win32_System_Variant`, which the
        // `wasapi` dependency enables on the same windows 0.61 crate (feature unification).
        unsafe {
            let enumerator: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
            let devices = enumerator.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE)?;
            for i in 0..devices.GetCount()? {
                let Ok(device) = devices.Item(i) else { continue };
                let Ok(manager) = device.Activate::<IAudioSessionManager2>(CLSCTX_ALL, None) else { continue };
                let Ok(sessions) = manager.GetSessionEnumerator() else { continue };
                for j in 0..sessions.GetCount().unwrap_or(0) {
                    let Ok(control) = sessions.GetSession(j) else { continue };
                    let Ok(control2) = control.cast::<IAudioSessionControl2>() else { continue };
                    if let Ok(pid) = control2.GetProcessId() {
                        if pid != 0 {
                            pids.insert(pid);
                        }
                    }
                }
            }
        }
        Ok(pids)
    }

    #[repr(C)]
    struct OsVersionInfoW {
        size: u32,
        major: u32,
        minor: u32,
        build: u32,
        platform: u32,
        csd: [u16; 128],
    }

    /// The real build number (`RtlGetVersion`, not subject to manifest-based version lies).
    pub fn windows_build() -> Option<u32> {
        // SAFETY: ntdll is loaded in every process; `RtlGetVersion(PRTL_OSVERSIONINFOW) -> NTSTATUS`.
        unsafe {
            let ntdll = GetModuleHandleW(w!("ntdll.dll")).ok()?;
            let proc = GetProcAddress(ntdll, s!("RtlGetVersion"))?;
            let f: unsafe extern "system" fn(*mut OsVersionInfoW) -> i32 = std::mem::transmute(proc);
            let mut info = OsVersionInfoW {
                size: std::mem::size_of::<OsVersionInfoW>() as u32,
                major: 0,
                minor: 0,
                build: 0,
                platform: 0,
                csd: [0; 128],
            };
            (f(&mut info) == 0 && info.major >= 10).then_some(info.build)
        }
    }

    // -----------------------------------------------------------------------------------------
    // The capture thread
    // -----------------------------------------------------------------------------------------

    struct ComGuard(bool);

    impl ComGuard {
        fn mta() -> Self {
            Self(wasapi::initialize_mta().is_ok())
        }
    }

    impl Drop for ComGuard {
        fn drop(&mut self) {
            if self.0 {
                wasapi::deinitialize();
            }
        }
    }

    /// Exit watch of the captured process (`SYNCHRONIZE` handle).
    struct ProcessWatch(OwnedHandle);

    impl ProcessWatch {
        fn open(pid: u32) -> Option<Self> {
            // SAFETY: plain OpenProcess; the handle is owned and closed by the guard.
            unsafe { OpenProcess(PROCESS_SYNCHRONIZE, false, pid) }
                .ok()
                .map(|h| Self(OwnedHandle(h)))
        }

        fn exited(&self) -> bool {
            // SAFETY: a valid process handle; a zero timeout never blocks.
            unsafe { WaitForSingleObject(self.0 .0, 0) == WAIT_OBJECT_0 }
        }
    }

    // `wasapi::Handle` (0.19) is a single-`HANDLE` struct without `Drop`: it never closes its event.
    // [`own_event`] takes the raw handle out of it; this keeps that cast honest if the layout changes.
    const _: () = assert!(std::mem::size_of::<Handle>() == std::mem::size_of::<HANDLE>());

    /// Take ownership of the event `set_get_eventhandle` created, so it is closed with the session
    /// (one leaked kernel event per open otherwise: every start, re-attach and retry).
    fn own_event(handle: Handle) -> OwnedHandle {
        // SAFETY: `Handle` holds exactly one `HANDLE` (same size, asserted above, so its only field
        // sits at offset 0) and has no other state. In wasapi 0.19 it has no `Drop`, and a later
        // `Drop` is not run by `transmute`, so the event is closed exactly once, by `OwnedHandle`.
        OwnedHandle(unsafe { std::mem::transmute::<Handle, HANDLE>(handle) })
    }

    /// One activated process-loopback client. Fields drop in order: the client is released
    /// before its event is closed.
    struct Session {
        client: AudioClient,
        capture: AudioCaptureClient,
        event: OwnedHandle,
        watch: Option<ProcessWatch>,
    }

    impl Session {
        fn open(pid: u32) -> Result<Self, String> {
            let format = WaveFormat::new(
                32,
                32,
                &SampleType::Float,
                CAPTURE_FORMAT.sample_rate as usize,
                usize::from(CAPTURE_FORMAT.channels),
                None,
            );
            let mut client = AudioClient::new_application_loopback_client(pid, true)
                .map_err(|e| format!("Could not capture the program's sound: {e}"))?;
            let mode = StreamMode::EventsShared {
                autoconvert: true,
                buffer_duration_hns: BUFFER_HNS,
            };
            client
                .initialize_client(&format, &Direction::Capture, &mode)
                .map_err(|e| format!("Could not start capturing the program's sound: {e}"))?;
            // Owned from here on: closed on every early return below too.
            let event = own_event(client.set_get_eventhandle().map_err(|e| format!("Capture event: {e}"))?);
            let capture = client.get_audiocaptureclient().map_err(|e| format!("Capture client: {e}"))?;
            client.start_stream().map_err(|e| format!("Could not start the capture: {e}"))?;
            Ok(Self {
                client,
                capture,
                event,
                watch: ProcessWatch::open(pid),
            })
        }
    }

    impl Drop for Session {
        fn drop(&mut self) {
            let _ = self.client.stop_stream();
        }
    }

    fn f32_bytes_mut(samples: &mut [f32]) -> &mut [u8] {
        // SAFETY: `u8` has alignment 1 and every byte pattern is a valid `f32`; same memory span.
        unsafe { std::slice::from_raw_parts_mut(samples.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(samples)) }
    }

    /// The running capture on its owner thread (wasapi objects are created and dropped there).
    struct ProcessCapture {
        stop: Option<mpsc::Sender<Option<mpsc::Sender<()>>>>,
    }

    impl CaptureStream for ProcessCapture {
        fn stop(mut self: Box<Self>, released: Option<mpsc::Sender<()>>) {
            if let Some(tx) = self.stop.take() {
                if let Err(mpsc::SendError(Some(released))) = tx.send(released) {
                    // The thread already exited: nothing is held any more.
                    let _ = released.send(());
                }
            } else if let Some(released) = released {
                let _ = released.send(());
            }
        }
    }

    impl Drop for ProcessCapture {
        fn drop(&mut self) {
            if let Some(tx) = self.stop.take() {
                let _ = tx.send(None);
            }
        }
    }

    pub fn open_capture(target: ProcessTarget, sink: StreamSink, errors: ErrorSink) -> Result<Box<dyn CaptureStream>, String> {
        // Resolve avrt.dll off the capture thread, so its first MMCSS call never loads a DLL.
        rt_thread::preload();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
        let (stop_tx, stop_rx) = mpsc::channel::<Option<mpsc::Sender<()>>>();
        thread::Builder::new()
            .name("sigil-capture-process".into())
            .spawn(move || run(target, sink, errors, ready_tx, stop_rx))
            .map_err(|e| format!("Failed to start the capture thread: {e}"))?;
        match ready_rx.recv_timeout(OPEN_TIMEOUT) {
            Ok(Ok(())) => Ok(Box::new(ProcessCapture { stop: Some(stop_tx) })),
            Ok(Err(e)) => Err(e),
            // Dropping `stop_tx` makes a late-starting capture end itself at once.
            Err(_) => Err("Timed out starting the program capture".to_string()),
        }
    }

    enum Mode {
        Capturing(Session),
        /// The process exited: look for a new one with the same name at `next_scan`.
        Waiting { next_scan: Instant },
    }

    fn run(
        target: ProcessTarget,
        mut sink: StreamSink,
        mut errors: ErrorSink,
        ready: mpsc::Sender<Result<(), String>>,
        stop: mpsc::Receiver<Option<mpsc::Sender<()>>>,
    ) {
        // Declared first so it is dropped last, after every COM object.
        let _com = ComGuard::mta();
        let channels = usize::from(CAPTURE_FORMAT.channels);
        let rate = CAPTURE_FORMAT.sample_rate;
        let mut mode = match Session::open(target.pid) {
            Ok(session) => Mode::Capturing(session),
            Err(e) => {
                let _ = ready.send(Err(e));
                return;
            }
        };
        // The scratch buffer is the only allocation of the capture path (sized once, here).
        let mut buf = vec![0f32; rate as usize * channels * SCRATCH_SECONDS];
        if ready.send(Ok(())).is_err() {
            return; // the opener gave up (timeout): end quietly
        }
        tracing::info!(target: LOG, "capturing {} (pid {})", target.exe, target.pid);
        let mut idle = IdleFill::new(now_qpc_ns());
        let mut last_alive_check = now_qpc_ns();
        let mut released = None;
        loop {
            match stop.try_recv() {
                Ok(r) => {
                    released = r;
                    break;
                }
                Err(mpsc::TryRecvError::Disconnected) => break,
                Err(mpsc::TryRecvError::Empty) => {}
            }
            match &mut mode {
                Mode::Capturing(session) => {
                    // A timeout is normal (the program is silent).
                    // SAFETY: the session's own event handle, open until the session drops.
                    let _ = unsafe { WaitForSingleObject(session.event.0, WAIT_MS) };
                    rt_thread::prepare_audio_callback_thread_once();
                    match drain(session, &mut buf, channels, &mut sink, &mut idle) {
                        Ok(()) => {}
                        Err(e) => {
                            if session.watch.as_ref().is_some_and(ProcessWatch::exited) {
                                mode = wait_for_restart(&target);
                                continue;
                            }
                            errors.report(e);
                            break;
                        }
                    }
                    let now = now_qpc_ns();
                    if now - last_alive_check >= ALIVE_CHECK_NS {
                        last_alive_check = now;
                        if session.watch.as_ref().is_some_and(ProcessWatch::exited) {
                            mode = wait_for_restart(&target);
                            continue;
                        }
                    }
                    fill_silence(&mut idle, &mut buf, channels, rate, &mut sink);
                }
                Mode::Waiting { next_scan } => {
                    thread::sleep(std::time::Duration::from_millis(u64::from(WAIT_MS)));
                    fill_silence(&mut idle, &mut buf, channels, rate, &mut sink);
                    if Instant::now() < *next_scan {
                        continue;
                    }
                    *next_scan = Instant::now() + RESCAN_INTERVAL;
                    let Some(pid) = find_again(&target.exe) else { continue };
                    match Session::open(pid) {
                        Ok(session) => {
                            tracing::info!(target: LOG, "{} is back (pid {pid}): capturing again", target.exe);
                            mode = Mode::Capturing(session);
                        }
                        Err(e) => {
                            errors.report(e);
                            break;
                        }
                    }
                }
            }
        }
        drop(mode);
        tracing::info!(target: LOG, "stopped capturing {}", target.exe);
        if let Some(released) = released {
            // `_com` drops after this send, but every COM object is gone already.
            let _ = released.send(());
        }
    }

    fn wait_for_restart(target: &ProcessTarget) -> Mode {
        tracing::info!(target: LOG, "{} exited: waiting for it to start again", target.exe);
        Mode::Waiting {
            next_scan: Instant::now() + RESCAN_INTERVAL,
        }
    }

    fn find_again(exe: &str) -> Option<u32> {
        let wanted = normalize_exe(exe)?;
        let procs = snapshot().ok()?;
        let titles = window_titles();
        find_root(&procs, &wanted, &|pid| titles.contains_key(&pid)).map(|p| p.pid)
    }

    /// Deliver every packet waiting in the capture client. Real-time path: no allocation unless a
    /// packet is larger than the 2 s scratch buffer (never seen; grown rather than dropped).
    fn drain(
        session: &mut Session,
        buf: &mut Vec<f32>,
        channels: usize,
        sink: &mut StreamSink,
        idle: &mut IdleFill,
    ) -> Result<(), String> {
        loop {
            let frames = match session.capture.get_next_packet_size() {
                Ok(Some(n)) => n as usize,
                Ok(None) => 0,
                Err(e) => return Err(format!("Program capture failed: {e}")),
            };
            if frames == 0 {
                return Ok(());
            }
            let samples = frames * channels;
            if samples > buf.len() {
                buf.resize(samples, 0.0);
            }
            let (read, info) = session
                .capture
                .read_from_device(f32_bytes_mut(&mut buf[..samples]))
                .map_err(|e| format!("Program capture failed: {e}"))?;
            if read == 0 {
                return Ok(());
            }
            let n = (read as usize * channels).min(samples);
            if info.flags.silent {
                buf[..n].fill(0.0);
            }
            let now = now_qpc_ns();
            let start = packet_qpc_ns(info.timestamp, info.flags.timestamp_error, now);
            sink.push(&buf[..n], start);
            idle.audio(start + CAPTURE_FORMAT.frames_to_ns(u64::from(read)));
        }
    }

    /// Paced silence while the program is quiet (see [`IdleFill`]).
    fn fill_silence(idle: &mut IdleFill, buf: &mut [f32], channels: usize, rate: u32, sink: &mut StreamSink) {
        let Some((mut start, mut frames)) = idle.due(now_qpc_ns(), rate) else { return };
        let chunk_frames = (buf.len() / channels).max(1);
        while frames > 0 {
            let n = frames.min(chunk_frames);
            let block = &mut buf[..n * channels];
            block.fill(0.0);
            sink.push(block, start);
            start += CAPTURE_FORMAT.frames_to_ns(n as u64);
            frames -= n;
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessHandleCount};

        fn handle_count() -> u32 {
            let mut n = 0u32;
            // SAFETY: this process's pseudo handle and a valid out pointer.
            unsafe { GetProcessHandleCount(GetCurrentProcess(), &mut n) }.expect("handle count");
            n
        }

        /// Opening and dropping capture sessions leaks no handles (each used to leak its event).
        /// Real API (build 20348+ and the audio service): `cargo test loopback_process -- --ignored`.
        #[test]
        #[ignore]
        fn sessions_close_their_event() {
            if !super::super::supported() {
                eprintln!("per-process loopback is not supported on this build");
                return;
            }
            let _com = ComGuard::mta();
            let pid = std::process::id();
            // Warm up: the first activation loads the audio stack and keeps a few handles for good.
            for _ in 0..3 {
                drop(Session::open(pid).expect("open a capture of this process"));
            }
            let before = handle_count();
            const OPENS: u32 = 30;
            for _ in 0..OPENS {
                drop(Session::open(pid).expect("open a capture of this process"));
            }
            let after = handle_count();
            assert!(after < before + OPENS / 3, "handles grew from {before} to {after} over {OPENS} opens");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Other platforms
// ---------------------------------------------------------------------------------------------

#[cfg(not(windows))]
mod sys {
    use std::collections::{HashMap, HashSet};

    use super::{ProcEntry, ProcessTarget};
    use crate::services::audio::hub::{CaptureStream, ErrorSink, StreamSink};

    pub fn snapshot() -> Result<Vec<ProcEntry>, String> {
        Ok(Vec::new())
    }

    pub fn window_titles() -> HashMap<u32, String> {
        HashMap::new()
    }

    pub fn audio_session_pids() -> HashSet<u32> {
        HashSet::new()
    }

    pub fn windows_build() -> Option<u32> {
        None
    }

    pub fn open_capture(target: ProcessTarget, _sink: StreamSink, _errors: ErrorSink) -> Result<Box<dyn CaptureStream>, String> {
        Err(format!("Capturing {} is only supported on Windows", target.exe))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(pid: u32, parent: u32, exe: &str) -> ProcEntry {
        ProcEntry {
            pid,
            parent,
            exe: exe.into(),
        }
    }

    #[test]
    fn program_names_are_normalized() {
        assert_eq!(normalize_exe("VRChat.exe").as_deref(), Some("VRChat.exe"));
        assert_eq!(normalize_exe("  VRChat ").as_deref(), Some("VRChat.exe"));
        assert_eq!(normalize_exe(r"C:\Games\VRChat\VRChat.exe").as_deref(), Some("VRChat.exe"));
        assert_eq!(normalize_exe("steam/steam.exe").as_deref(), Some("steam.exe"));
        assert_eq!(normalize_exe("my.app").as_deref(), Some("my.app"));
        assert_eq!(normalize_exe("Game v1. 2").as_deref(), Some("Game v1. 2.exe"));
        assert_eq!(normalize_exe("ゲーム").as_deref(), Some("ゲーム.exe"));
        for bad in ["", "   ", ".", "..", r"C:\Games\", "a|b.exe", "a?.exe", "bad\u{0}.exe"] {
            assert_eq!(normalize_exe(bad), None, "{bad:?}");
        }
        assert!(same_exe("VRChat.exe", "vrchat.EXE"));
        assert!(same_exe("Ä.exe", "ä.exe"));
        assert!(!same_exe("VRChat.exe", "VRChat2.exe"));
    }

    #[test]
    fn the_root_of_a_same_name_tree_is_captured() {
        let procs = vec![
            p(1, 0, "explorer.exe"),
            p(100, 1, "chrome.exe"),
            p(101, 100, "chrome.exe"),
            p(102, 100, "chrome.exe"),
            p(200, 1, "VRChat.exe"),
            p(300, 1, "steam.exe"),
            p(301, 300, "steamwebhelper.exe"),
        ];
        let none = |_: u32| false;
        assert_eq!(find_root(&procs, "chrome.exe", &none).map(|e| e.pid), Some(100));
        assert_eq!(find_root(&procs, "vrchat.exe", &none).map(|e| e.pid), Some(200));
        assert_eq!(find_root(&procs, "steamwebhelper.exe", &none).map(|e| e.pid), Some(301));
        assert_eq!(find_root(&procs, "missing.exe", &none), None);
        // Two separate instances: the one with a window wins, else the lowest id.
        let two = vec![p(50, 1, "game.exe"), p(40, 2, "game.exe")];
        assert_eq!(find_root(&two, "game.exe", &none).map(|e| e.pid), Some(40));
        assert_eq!(find_root(&two, "game.exe", &|pid| pid == 50).map(|e| e.pid), Some(50));
        // A parent cycle (reused ids) still finds the program.
        let cycle = vec![p(7, 8, "x.exe"), p(8, 7, "x.exe")];
        assert_eq!(find_root(&cycle, "x.exe", &none).map(|e| e.pid), Some(7));
    }

    #[test]
    fn the_picker_lists_audio_programs_first_without_sigil_or_shell_hosts() {
        let procs = vec![
            p(1, 0, "explorer.exe"),
            p(10, 1, "Sigil.exe"),
            p(11, 10, "Sigil.exe"),
            p(100, 1, "chrome.exe"),
            p(101, 100, "chrome.exe"),
            p(200, 1, "VRChat.exe"),
            p(300, 1, "notepad.exe"),
            p(400, 1, "Discord.exe"),
            p(401, 400, "Discord.exe"),
            p(500, 1, "TextInputHost.exe"),
        ];
        let titles: HashMap<u32, String> = [
            (1, "Desktop".to_string()),
            (10, "Sigil".to_string()),
            (100, "A video - Chrome".to_string()),
            (200, "VRChat".to_string()),
            (300, "  ".to_string()),
            (500, "Windows Input Experience".to_string()),
        ]
        .into_iter()
        .collect();
        // The audio session lives in a child process of the program.
        let audio: HashSet<u32> = [101, 200, 401, 11, 0, 4, 999].into_iter().collect();
        let list = build_list(&procs, &titles, &audio, 10);
        let rows: Vec<(&str, u32, bool, Option<&str>)> =
            list.iter().map(|r| (r.exe.as_str(), r.pid, r.audio, r.title.as_deref())).collect();
        assert_eq!(
            rows,
            vec![
                ("chrome.exe", 100, true, Some("A video - Chrome")),
                ("Discord.exe", 400, true, None),
                ("VRChat.exe", 200, true, Some("VRChat")),
                ("notepad.exe", 300, false, None),
            ]
        );
        // Long titles are clipped.
        let long: HashMap<u32, String> = [(300, "x".repeat(500))].into_iter().collect();
        let list = build_list(&procs, &long, &HashSet::new(), 10);
        assert_eq!(list[0].title.as_ref().map(|t| t.chars().count()), Some(MAX_TITLE_CHARS));
    }

    #[test]
    fn packet_times_fall_back_to_the_read_time() {
        let now = 50_000_000_000i64;
        assert_eq!(packet_qpc_ns(499_000_000, false, now), 49_900_000_000);
        assert_eq!(packet_qpc_ns(499_000_000, true, now), now, "timestamp error");
        assert_eq!(packet_qpc_ns(0, false, now), now, "no stamp");
        assert_eq!(packet_qpc_ns(100, false, now), now, "implausibly old");
        assert_eq!(packet_qpc_ns(600_000_000, false, now), now, "in the future");
        assert_eq!(packet_qpc_ns(u64::MAX, false, now), now, "overflow");
    }

    #[test]
    fn silence_is_filled_after_a_pause_paced_and_bounded() {
        let ms = 1_000_000i64;
        let rate = 48_000;
        let mut idle = IdleFill::new(0);
        assert_eq!(idle.due(200 * ms, rate), None, "a short pause is not silence yet");
        let (start, frames) = idle.due(300 * ms, rate).expect("300 ms of quiet");
        assert_eq!((start, frames), (0, 14_400));
        // Paced: only the time that passed since is delivered next.
        assert_eq!(idle.due(400 * ms, rate), Some((300 * ms, 4_800)));
        assert_eq!(idle.due(400 * ms, rate), None, "nothing new is due");
        // Bounded by IDLE_FILL per quiet stretch.
        let total = 14_400 + 4_800 + idle.due(10_000 * ms, rate).map(|d| d.1).unwrap_or(0);
        assert_eq!(total, 120_000, "2.5 s at 48 kHz");
        assert_eq!(idle.due(20_000 * ms, rate), None);
        // Real audio starts a new stretch.
        idle.audio(20_000 * ms);
        assert_eq!(idle.due(20_100 * ms, rate), None);
        assert!(idle.due(20_400 * ms, rate).is_some());
    }

    #[test]
    fn process_devices_are_recognised_and_the_wire_format_is_camel_case() {
        let device = ResolvedDevice {
            name: "process:VRChat.exe".into(),
            loopback: true,
            format: CAPTURE_FORMAT,
            handle: None,
        };
        assert!(is_process_device(&device));
        assert_eq!(device.physical_key(), "loopback:process:VRChat.exe");
        let speakers = ResolvedDevice {
            name: "Speakers".into(),
            loopback: true,
            format: CAPTURE_FORMAT,
            handle: None,
        };
        assert!(!is_process_device(&speakers));
        let list = LoopbackProcessList {
            supported: true,
            processes: vec![LoopbackProcess {
                pid: 7,
                exe: "VRChat.exe".into(),
                title: None,
                audio: true,
            }],
            windows_build: Some(26_200),
            plan: None,
        };
        assert_eq!(
            serde_json::to_string(&list).unwrap(),
            r#"{"supported":true,"processes":[{"pid":7,"exe":"VRChat.exe","title":null,"audio":true}],"windowsBuild":26200}"#
        );
        // Rows from before P6-A (no `audio`) still parse.
        let old: LoopbackProcess = serde_json::from_str(r#"{"pid":1,"exe":"a.exe","title":"A"}"#).unwrap();
        assert!(!old.audio);
    }

    #[test]
    fn opening_a_device_of_another_format_is_refused() {
        // `open` checks the format before touching the system.
        let device = ResolvedDevice {
            name: "process:VRChat.exe".into(),
            loopback: true,
            format: StreamFormat {
                sample_rate: 44_100,
                channels: 2,
            },
            handle: None,
        };
        let (cmd_tx, cmd_rx) = rtrb::RingBuffer::new(4);
        let (retired_tx, _retired_rx) = rtrb::RingBuffer::new(4);
        drop(cmd_tx);
        let sink = StreamSink::new(
            Vec::with_capacity(crate::services::audio::hub::FANOUT_SLOT_CAPACITY),
            cmd_rx,
            retired_tx,
            Default::default(),
            std::sync::Arc::new(crate::services::audio::hub::StreamHealth::new()),
            2,
        );
        let (tx, _rx) = std::sync::mpsc::channel();
        let errors = ErrorSink::new("s".into(), 1, tx);
        assert!(open(device, sink, errors).is_err());
    }

    /// A process source through a real hub (scripted backend): what the hub and the speech pipeline
    /// rely on. The requested key is kept, the physical stream is the process's, a program that is
    /// not running fails the attach (the caller falls back), silence never counts as a stall, and
    /// the source belongs to the other players' mute group.
    #[test]
    fn process_sources_through_the_hub() {
        use crate::services::audio::hub::{CaptureHub, ConsumerOptions, HubConfig, MuteGroup, SourceSpec};
        use std::sync::{mpsc, Arc, Mutex};

        struct Held(Mutex<Option<mpsc::Sender<Option<mpsc::Sender<()>>>>>);
        impl CaptureStream for Held {
            fn stop(self: Box<Self>, released: Option<mpsc::Sender<()>>) {
                if let Some(tx) = self.0.lock().unwrap().take() {
                    let _ = tx.send(released);
                }
            }
        }
        struct Backend {
            opens: Arc<Mutex<u32>>,
        }
        impl crate::services::audio::hub::CaptureBackend for Backend {
            fn resolve(&self, spec: &SourceSpec) -> Result<ResolvedDevice, String> {
                match spec {
                    SourceSpec::Process(exe) if same_exe(exe, "VRChat.exe") => Ok(ResolvedDevice {
                        name: "process:VRChat.exe".into(),
                        loopback: true,
                        format: CAPTURE_FORMAT,
                        handle: None,
                    }),
                    SourceSpec::Process(exe) => Err(format!("{exe} is not running")),
                    _ => Err("not a process".into()),
                }
            }
            fn open(&self, device: ResolvedDevice, mut sink: StreamSink, _errors: ErrorSink) -> Result<Box<dyn CaptureStream>, String> {
                assert!(is_process_device(&device));
                *self.opens.lock().unwrap() += 1;
                let (tx, rx) = mpsc::channel::<Option<mpsc::Sender<()>>>();
                std::thread::spawn(move || {
                    // One 10 ms block, then silence (nothing at all), like a quiet game.
                    sink.push(&[0.25f32; 960], crate::services::audio::hub::now_qpc_ns());
                    if let Ok(Some(released)) = rx.recv() {
                        let _ = released.send(());
                    }
                });
                Ok(Box::new(Held(Mutex::new(Some(tx)))))
            }
        }

        let opens = Arc::new(Mutex::new(0));
        let config = HubConfig {
            tick: Duration::from_millis(10),
            stall_timeout: Duration::from_millis(40),
            first_data_timeout: Duration::from_millis(40),
            ..HubConfig::default()
        };
        let hub = CaptureHub::new(Arc::new(Backend { opens: Arc::clone(&opens) }), config).unwrap();
        let spec = SourceSpec::parse(Some("process:vrchat.exe"));
        assert_eq!(crate::services::audio::stt_mute_group(&spec), MuteGroup::Others);
        let mut consumer = hub.attach(spec, ConsumerOptions::speech("others")).expect("attach");
        assert_eq!(consumer.source(), "process:vrchat.exe");
        // Per-process capture counts as loopback (the speech front end's idle fill applies).
        assert!(consumer.is_loopback());
        let mut out = Vec::new();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while std::time::Instant::now() < deadline {
            consumer.wait(Duration::from_millis(20));
            if consumer.read(&mut out).frames_in > 0 {
                break;
            }
        }
        consumer.flush(&mut out);
        assert_eq!(out.len(), 160, "10 ms at 16 kHz mono");
        // Long silence (many stall timeouts): never restarted.
        std::thread::sleep(Duration::from_millis(300));
        let status = hub.status().unwrap();
        let stream = status.streams.iter().find(|s| s.key == "loopback:process:VRChat.exe").expect("process stream");
        assert!(stream.loopback);
        assert_eq!((stream.generation, stream.failures, *opens.lock().unwrap()), (1, 0, 1));
        // A program that is not running fails the attach: the caller falls back.
        let err = hub
            .attach(SourceSpec::Process("NotRunning.exe".into()), ConsumerOptions::speech("others"))
            .err()
            .expect("not running");
        assert!(err.contains("not running"), "{err}");
        let _ = consumer.close_and_wait();
        hub.shutdown();
    }

    /// Real system checks: the build is readable and the snapshot sees this process.
    #[cfg(windows)]
    #[test]
    fn the_system_probe_sees_this_process() {
        assert!(windows_build().is_some_and(|b| b >= 10_000));
        let procs = sys::snapshot().expect("snapshot");
        let me = std::process::id();
        assert!(procs.iter().any(|p| p.pid == me));
        let probe = system_probe();
        let own = procs.iter().find(|p| p.pid == me).unwrap();
        assert!(probe.running(&own.exe).is_some());
        assert_eq!(probe.running("surely-not-running-sigil-test.exe"), None);
    }

    /// Captures this test process's own sound tree through the real API (needs Windows build
    /// 20348+ and an active output device): `cargo test loopback_process -- --ignored`.
    #[cfg(windows)]
    #[test]
    #[ignore]
    fn real_process_capture_opens_and_stops() {
        use crate::services::audio::hub::{CaptureHub, ConsumerOptions, HubConfig, SourceSpec};
        if !supported() {
            eprintln!("per-process loopback is not supported on this build");
            return;
        }
        let procs = sys::snapshot().unwrap();
        let me = procs.iter().find(|p| p.pid == std::process::id()).unwrap().exe.clone();
        let hub = CaptureHub::new(std::sync::Arc::new(crate::services::audio::stream::CpalBackend), HubConfig::default()).unwrap();
        let mut consumer = hub
            .attach(SourceSpec::Process(me.clone()), ConsumerOptions::speech("test"))
            .expect("attach to our own process");
        assert_eq!(consumer.source(), format!("process:{me}"));
        // Silence is idle-filled: something arrives within a second even though nothing plays.
        let mut out = Vec::new();
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while out.is_empty() && std::time::Instant::now() < deadline {
            consumer.wait(Duration::from_millis(100));
            consumer.read(&mut out);
        }
        assert!(!out.is_empty(), "idle fill delivers silence");
        assert!(out.iter().all(|s| *s == 0.0));
        let _ = consumer.close_and_wait();
        hub.shutdown();
    }
}
