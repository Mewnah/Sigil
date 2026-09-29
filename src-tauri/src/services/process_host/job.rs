//! Windows Job Objects: every helper process (and every isolated worker) runs in its own job.
//!
//! - `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`: the job handle lives only in Sigil (it is created
//!   non-inheritable), so when Sigil exits or is killed (Task Manager, a crash) the kernel closes the
//!   handle and terminates every process in the job, grandchildren included.
//! - `JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION`: a crashing helper dies immediately instead of
//!   sitting behind a Windows Error Reporting dialog (which would also block the restart policy).
//! - Optional `JOB_OBJECT_LIMIT_JOB_MEMORY` (committed memory of the whole job; memory-mapped model
//!   files are not commit and do not count) and `JOB_OBJECT_LIMIT_PRIORITY_CLASS`.
//! - A private I/O completion port receives the job's notifications, so a memory-limit hit is
//!   reported as the reason a helper died instead of a bare exit code.
//!
//! Processes are created suspended, assigned, then resumed ([`resume_suspended`]), so a helper can
//! never start a grandchild outside the job.

use super::HelperPriority;
use std::ffi::c_void;
use std::os::windows::io::RawHandle;
use std::sync::OnceLock;
use std::time::Duration;
use windows::core::{s, w};
use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::System::JobObjects::{
    AssignProcessToJobObject, JobObjectAssociateCompletionPortInformation, JobObjectBasicAccountingInformation, JobObjectBasicProcessIdList,
    JobObjectExtendedLimitInformation, QueryInformationJobObject, SetInformationJobObject, TerminateJobObject, JOBOBJECT_ASSOCIATE_COMPLETION_PORT,
    JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT, JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION,
    JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOB_OBJECT_LIMIT_PRIORITY_CLASS,
};
use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
use windows::Win32::System::Threading::{
    ProcessPowerThrottling, SetProcessInformation, WaitForSingleObject, BELOW_NORMAL_PRIORITY_CLASS, IDLE_PRIORITY_CLASS,
    PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, PROCESS_POWER_THROTTLING_STATE,
};
#[cfg(test)]
use windows::Win32::{
    Foundation::WAIT_TIMEOUT,
    System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE},
};

// `CreateJobObjectW` is gated on the `Win32_Security` feature in windows 0.61 and the completion-port
// calls on `Win32_System_IO`; neither feature is enabled, and both are plain kernel32 exports.
#[link(name = "kernel32")]
extern "system" {
    fn CreateJobObjectW(attributes: *const c_void, name: *const u16) -> *mut c_void;
    fn CreateIoCompletionPort(file: *mut c_void, existing: *mut c_void, key: usize, threads: u32) -> *mut c_void;
    fn GetQueuedCompletionStatus(port: *mut c_void, bytes: *mut u32, key: *mut usize, overlapped: *mut *mut c_void, millis: u32) -> i32;
}

/// `INVALID_HANDLE_VALUE` for `CreateIoCompletionPort(file = INVALID_HANDLE_VALUE, ...)`.
const INVALID_HANDLE: *mut c_void = -1isize as *mut c_void;
/// Job notification message ids (winnt.h `JOB_OBJECT_MSG_*`).
const MSG_NEW_PROCESS: u32 = 6;
const MSG_ABNORMAL_EXIT_PROCESS: u32 = 8;
const MSG_PROCESS_MEMORY_LIMIT: u32 = 9;
const MSG_JOB_MEMORY_LIMIT: u32 = 10;
/// Smallest memory cap accepted (a cap below this would stop any process from starting).
pub const MIN_MEMORY_LIMIT_MB: u64 = 16;

/// A kernel handle closed on drop. Kernel handles may be used from any thread.
struct OwnedRaw(*mut c_void);

// SAFETY: a job / completion-port handle is a process-wide kernel object reference; every API used
// on it is thread-safe.
unsafe impl Send for OwnedRaw {}
unsafe impl Sync for OwnedRaw {}

impl OwnedRaw {
    fn handle(&self) -> HANDLE {
        HANDLE(self.0)
    }
}

impl Drop for OwnedRaw {
    fn drop(&mut self) {
        // SAFETY: the handle was returned by CreateJobObjectW / CreateIoCompletionPort and is closed once.
        unsafe {
            let _ = CloseHandle(HANDLE(self.0));
        }
    }
}

/// Limits of one job.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JobLimits {
    /// Cap on the committed memory of all processes in the job (MB, at least [`MIN_MEMORY_LIMIT_MB`]).
    pub memory_limit_mb: Option<u64>,
    /// Priority class forced on every process in the job.
    pub priority: HelperPriority,
}

/// Counters of a job (from `JobObjectBasicAccountingInformation` / `JobObjectExtendedLimitInformation`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JobStats {
    pub active_processes: u32,
    pub total_processes: u32,
    /// Peak committed memory of the whole job, in bytes.
    pub peak_job_memory_bytes: u64,
}

/// Notifications drained from the job's completion port since the last call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JobEvents {
    /// An allocation failed because of the job's memory cap.
    pub memory_limit_hit: bool,
    /// Processes that ended with an exception code (crash, fail-fast).
    pub abnormal_exits: u32,
    /// Processes that joined the job (the helper and its children).
    pub new_processes: u32,
}

/// A Job Object with `KILL_ON_JOB_CLOSE`. Dropping it closes the handle, which kills every process
/// still in the job.
pub struct Job {
    // Field order matters: the job closes (and kills its processes) before its completion port.
    job: OwnedRaw,
    port: Option<OwnedRaw>,
    limits: JobLimits,
}

impl std::fmt::Debug for Job {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Job")
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

fn win_err(what: &str) -> String {
    format!("{what}: {}", std::io::Error::last_os_error())
}

impl Job {
    /// Create an anonymous, non-inheritable job with the kill-on-close policy and `limits`.
    pub fn create(limits: JobLimits) -> Result<Job, String> {
        // SAFETY: NULL attributes (default security, handle not inheritable) and no name.
        let raw = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if raw.is_null() {
            return Err(win_err("CreateJobObjectW failed"));
        }
        let job = OwnedRaw(raw);

        let mut flags: JOB_OBJECT_LIMIT = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        if let Some(mb) = limits.memory_limit_mb {
            let mb = mb.max(MIN_MEMORY_LIMIT_MB);
            flags |= JOB_OBJECT_LIMIT_JOB_MEMORY;
            info.JobMemoryLimit = usize::try_from(mb.saturating_mul(1024 * 1024)).unwrap_or(usize::MAX);
        }
        let class = match limits.priority {
            HelperPriority::Normal => None,
            HelperPriority::BelowNormal => Some(BELOW_NORMAL_PRIORITY_CLASS.0),
            HelperPriority::Idle => Some(IDLE_PRIORITY_CLASS.0),
        };
        if let Some(class) = class {
            flags |= JOB_OBJECT_LIMIT_PRIORITY_CLASS;
            info.BasicLimitInformation.PriorityClass = class;
        }
        info.BasicLimitInformation.LimitFlags = flags;
        // SAFETY: pointer and size describe a live JOBOBJECT_EXTENDED_LIMIT_INFORMATION.
        unsafe {
            SetInformationJobObject(
                job.handle(),
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        }
        .map_err(|e| format!("SetInformationJobObject(limits) failed: {e}"))?;

        // Notifications are best effort: without them only the "why did it die" detail is lost.
        let port = Self::attach_port(&job);
        Ok(Job { job, port, limits })
    }

    fn attach_port(job: &OwnedRaw) -> Option<OwnedRaw> {
        // SAFETY: creating a fresh completion port not bound to any file.
        let raw = unsafe { CreateIoCompletionPort(INVALID_HANDLE, std::ptr::null_mut(), 0, 1) };
        if raw.is_null() {
            return None;
        }
        let port = OwnedRaw(raw);
        let assoc = JOBOBJECT_ASSOCIATE_COMPLETION_PORT {
            CompletionKey: std::ptr::null_mut(),
            CompletionPort: port.handle(),
        };
        // SAFETY: pointer and size describe a live JOBOBJECT_ASSOCIATE_COMPLETION_PORT; the port
        // outlives the job (field order of `Job`).
        let ok = unsafe {
            SetInformationJobObject(
                job.handle(),
                JobObjectAssociateCompletionPortInformation,
                &assoc as *const _ as *const c_void,
                std::mem::size_of::<JOBOBJECT_ASSOCIATE_COMPLETION_PORT>() as u32,
            )
        };
        ok.is_ok().then_some(port)
    }

    /// Put a process into the job. The process should still be suspended (see [`resume_suspended`]).
    pub fn assign(&self, process: RawHandle) -> Result<(), String> {
        // SAFETY: both handles are valid; the process handle comes from a live `std::process::Child`.
        unsafe { AssignProcessToJobObject(self.job.handle(), HANDLE(process)) }.map_err(|e| format!("AssignProcessToJobObject failed: {e}"))
    }

    /// Terminate every process in the job (the helper and all its children). Idempotent.
    pub fn terminate(&self, exit_code: u32) -> bool {
        // SAFETY: valid job handle.
        unsafe { TerminateJobObject(self.job.handle(), exit_code) }.is_ok()
    }

    pub fn stats(&self) -> JobStats {
        let mut stats = JobStats::default();
        let mut acct = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION::default();
        // SAFETY: pointer and size describe a writable JOBOBJECT_BASIC_ACCOUNTING_INFORMATION.
        let ok = unsafe {
            QueryInformationJobObject(
                Some(self.job.handle()),
                JobObjectBasicAccountingInformation,
                &mut acct as *mut _ as *mut c_void,
                std::mem::size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
                None,
            )
        };
        if ok.is_ok() {
            stats.active_processes = acct.ActiveProcesses;
            stats.total_processes = acct.TotalProcesses;
        }
        let mut ext = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        // SAFETY: as above, for JOBOBJECT_EXTENDED_LIMIT_INFORMATION.
        let ok = unsafe {
            QueryInformationJobObject(
                Some(self.job.handle()),
                JobObjectExtendedLimitInformation,
                &mut ext as *mut _ as *mut c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                None,
            )
        };
        if ok.is_ok() {
            stats.peak_job_memory_bytes = ext.PeakJobMemoryUsed as u64;
        }
        stats
    }

    /// Ids of the processes currently in the job (at most 256; helpers have a handful).
    pub fn process_ids(&self) -> Vec<u32> {
        const MAX: usize = 256;
        // JOBOBJECT_BASIC_PROCESS_ID_LIST: two DWORD counts in the first usize, then the ids.
        let mut buf = vec![0usize; MAX + 1];
        // SAFETY: the buffer is writable, usize-aligned and its size is passed.
        let ok = unsafe {
            QueryInformationJobObject(
                Some(self.job.handle()),
                JobObjectBasicProcessIdList,
                buf.as_mut_ptr() as *mut c_void,
                (buf.len() * std::mem::size_of::<usize>()) as u32,
                None,
            )
        };
        if ok.is_err() {
            return Vec::new();
        }
        let listed = ((buf[0] as u64) >> 32) as usize; // NumberOfProcessIdsInList (little-endian)
        buf[1..=listed.min(MAX)]
            .iter()
            .map(|&pid| pid as u32)
            .collect()
    }

    /// Drain the job's notifications (never blocks).
    pub fn take_events(&self) -> JobEvents {
        let mut events = JobEvents::default();
        let Some(port) = &self.port else {
            return events;
        };
        loop {
            let mut msg = 0u32;
            let mut key = 0usize;
            let mut overlapped: *mut c_void = std::ptr::null_mut();
            // SAFETY: valid port handle and out-pointers; timeout 0 never blocks.
            let ok = unsafe { GetQueuedCompletionStatus(port.0, &mut msg, &mut key, &mut overlapped, 0) };
            if ok == 0 {
                break; // WAIT_TIMEOUT: queue empty
            }
            match msg {
                MSG_JOB_MEMORY_LIMIT | MSG_PROCESS_MEMORY_LIMIT => events.memory_limit_hit = true,
                MSG_ABNORMAL_EXIT_PROCESS => events.abnormal_exits += 1,
                MSG_NEW_PROCESS => events.new_processes += 1,
                _ => {}
            }
        }
        events
    }
}

// ---------------------------------------------------------------------------------------------
// Process helpers
// ---------------------------------------------------------------------------------------------

/// `NtResumeProcess(HANDLE) -> NTSTATUS` (ntdll; stable since Windows XP). std does not expose the
/// main-thread handle of a suspended child on stable Rust, and resuming the whole process is exactly
/// what a `CREATE_SUSPENDED` child needs.
type NtResumeProcessFn = unsafe extern "system" fn(HANDLE) -> i32;

fn nt_resume_process() -> Option<NtResumeProcessFn> {
    static CELL: OnceLock<Option<NtResumeProcessFn>> = OnceLock::new();
    *CELL.get_or_init(|| unsafe {
        // SAFETY: ntdll is mapped into every process; GetModuleHandleW never loads anything.
        let module = GetModuleHandleW(w!("ntdll.dll")).ok()?;
        let proc = GetProcAddress(module, s!("NtResumeProcess"))?;
        // SAFETY: the export has exactly this signature; ntdll is never unloaded.
        Some(std::mem::transmute::<unsafe extern "system" fn() -> isize, NtResumeProcessFn>(proc))
    })
}

/// Whether processes can be created suspended and resumed after joining their job.
pub fn can_resume_suspended() -> bool {
    nt_resume_process().is_some()
}

/// Resume a process created with `CREATE_SUSPENDED`.
pub fn resume_suspended(process: RawHandle) -> Result<(), String> {
    let resume = nt_resume_process().ok_or("NtResumeProcess is not available")?;
    // SAFETY: valid process handle with PROCESS_SUSPEND_RESUME (CreateProcess returns full access).
    let status = unsafe { resume(HANDLE(process)) };
    if status < 0 {
        return Err(format!("NtResumeProcess failed (NTSTATUS {:#010X})", status as u32));
    }
    Ok(())
}

/// Wait until the process behind `process` exits (true) or `timeout` passes (false).
pub fn wait_process(process: RawHandle, timeout: Duration) -> bool {
    let millis = u32::try_from(timeout.as_millis()).unwrap_or(u32::MAX - 1);
    // SAFETY: valid process handle with SYNCHRONIZE access.
    unsafe { WaitForSingleObject(HANDLE(process), millis) == WAIT_OBJECT_0 }
}

/// Whether a process with this id is still running (false when it exited or never existed).
#[cfg(test)]
pub fn pid_alive(pid: u32) -> bool {
    // SAFETY: plain OpenProcess; the handle is closed before returning.
    unsafe {
        let Ok(h) = OpenProcess(PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, false, pid) else {
            return false;
        };
        let alive = WaitForSingleObject(h, 0) == WAIT_TIMEOUT;
        let _ = CloseHandle(h);
        alive
    }
}

/// Wait up to `timeout` for the process with this id to exit. True when it is gone.
#[cfg(test)]
pub fn wait_pid_exit(pid: u32, timeout: Duration) -> bool {
    // SAFETY: plain OpenProcess; the handle is closed before returning.
    unsafe {
        let Ok(h) = OpenProcess(PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, false, pid) else {
            return true;
        };
        let millis = u32::try_from(timeout.as_millis()).unwrap_or(u32::MAX - 1);
        let gone = WaitForSingleObject(h, millis) == WAIT_OBJECT_0;
        let _ = CloseHandle(h);
        gone
    }
}

/// Opt a child process out of Windows 11 EcoQoS (efficiency cores, reduced clocks while Sigil is in
/// the background). Best effort; false on Windows versions without the policy.
pub fn opt_out_eco_qos(process: RawHandle) -> bool {
    let state = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        StateMask: 0,
    };
    // SAFETY: pointer and size describe a live PROCESS_POWER_THROTTLING_STATE; the handle has
    // PROCESS_SET_INFORMATION (CreateProcess returns full access).
    unsafe {
        SetProcessInformation(
            HANDLE(process),
            ProcessPowerThrottling,
            &state as *const _ as *const c_void,
            std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
        )
        .is_ok()
    }
}

/// Human-readable exit code: decimal for ordinary codes, hex plus a hint for NTSTATUS failures.
pub fn describe_exit_code(code: i32) -> String {
    let u = code as u32;
    if u < 0xC000_0000 {
        return format!("exit code {code}");
    }
    let hint = match u {
        0xC000_0005 => Some("access violation (crash)"),
        0xC000_0017 | 0xC000_012D => Some("out of memory"),
        0xC000_001D => Some("illegal instruction; the build may need a newer CPU"),
        0xC000_007B => Some("bad executable image or 32/64-bit mismatch"),
        0xC000_0135 => Some("a required DLL was not found"),
        0xC000_0139 => Some("a DLL entry point was not found (DLL version mismatch)"),
        0xC000_0142 => Some("a DLL failed to initialize"),
        0xC000_00FD => Some("stack overflow"),
        0xC000_0409 => Some("fail-fast abort (out of memory or an internal check failed)"),
        0xC000_013A => Some("interrupted"),
        _ => None,
    };
    match hint {
        Some(h) => format!("exit code {u:#010X}: {h}"),
        None => format!("exit code {u:#010X}"),
    }
}

/// Whether an exit code means the Windows loader refused the program before it ran (a required
/// DLL or DLL export is missing, or the image is invalid for this machine). Starting it again
/// cannot help. (0xC0000142, "a DLL failed to initialize", is not included: it can be transient.)
pub fn is_loader_failure(code: i32) -> bool {
    matches!(code as u32, 0xC000_0135 | 0xC000_0139 | 0xC000_007B)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_exit_codes() {
        assert_eq!(describe_exit_code(0), "exit code 0");
        assert_eq!(describe_exit_code(3), "exit code 3");
        assert_eq!(
            describe_exit_code(0xC000_0135u32 as i32),
            "exit code 0xC0000135: a required DLL was not found"
        );
        assert_eq!(describe_exit_code(0xC123_4567u32 as i32), "exit code 0xC1234567");
        assert!(is_loader_failure(0xC000_0135u32 as i32));
        assert!(is_loader_failure(0xC000_0139u32 as i32));
        assert!(is_loader_failure(0xC000_007Bu32 as i32));
        assert!(!is_loader_failure(0xC000_0142u32 as i32));
        assert!(!is_loader_failure(0xC000_0005u32 as i32));
        assert!(!is_loader_failure(1));
    }

    #[test]
    fn job_with_limits_is_created_and_reports_stats() {
        let job = Job::create(JobLimits {
            memory_limit_mb: Some(1),
            priority: HelperPriority::BelowNormal,
        })
        .expect("job");
        let stats = job.stats();
        assert_eq!(stats.active_processes, 0);
        assert!(job.process_ids().is_empty());
        assert_eq!(job.take_events(), JobEvents::default());
        assert!(job.terminate(1), "terminating an empty job succeeds");
        assert!(can_resume_suspended());
    }

    #[test]
    fn pid_alive_is_false_for_unknown_pids() {
        assert!(pid_alive(std::process::id()));
        // PID 0 is the idle process, never openable; odd ids are never used by Windows.
        assert!(!pid_alive(0));
        assert!(!pid_alive(3));
    }
}
