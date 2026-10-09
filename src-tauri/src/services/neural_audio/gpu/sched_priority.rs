//! Cross-process GPU scheduling priority of the Sigil process (P3B-G; T2 / T7 W9, an experiment).
//!
//! CUDA stream priorities and the high-priority DirectML queue order work only *within* this
//! process. A game renders in another process, so the only scheduling lever that reaches it is the
//! process's GPU scheduling priority class, `D3DKMTSetProcessSchedulingPriorityClass` (`gdi32.dll`,
//! the WDDM kernel thunk): the GPU scheduler then prefers this process's work queues (CUDA's and
//! DirectML's alike) over normal-priority processes'. Sigil never asks for `REALTIME` (it needs a
//! privilege and can starve the desktop) and never requests admin: a class the OS refuses is
//! reported and nothing changes.
//!
//! `SIGIL_GPU_PROCESS_PRIORITY=normal|above_normal|high` ([`ENV`]) applies a class once, when the
//! first session plan is built ([`apply_from_env_once`]); unset (the default, [`DEFAULT`]) leaves the
//! OS's choice. The class applies to the whole process, so Sigil's GPU STT and TTS count against the
//! game too: P3B-G measures it with the out-of-process load (its achieved duty is the game's
//! frame-time proxy), and only a p99 gain with a duty drop of at most 1 % may make it a default.

use std::sync::OnceLock;

/// `SIGIL_GPU_PROCESS_PRIORITY` (see the module docs).
pub const ENV: &str = "SIGIL_GPU_PROCESS_PRIORITY";

/// A GPU scheduling priority class Sigil may ask for (`D3DKMT_SCHEDULINGPRIORITYCLASS`; never
/// `REALTIME`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuProcessPriority {
    Normal,
    AboveNormal,
    High,
}

/// The class applied without [`ENV`]: none (the OS's own). P3B-I0 kept it: under the out-of-process load
/// the class was applied but had no measurable effect (CUDA call p99 +1.1 % [-1.6, +2.9], DirectML +2.9 %
/// [-0.7, +4.3], NEUTRAL), so it stays an opt-in experiment.
pub const DEFAULT: Option<GpuProcessPriority> = None;

impl GpuProcessPriority {
    /// `normal` / `above_normal` / `high` (case-insensitive; `realtime` is refused: `None`).
    pub fn parse(s: &str) -> Option<GpuProcessPriority> {
        match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "normal" => Some(GpuProcessPriority::Normal),
            "above_normal" | "abovenormal" => Some(GpuProcessPriority::AboveNormal),
            "high" => Some(GpuProcessPriority::High),
            _ => None,
        }
    }

    /// The `D3DKMT_SCHEDULINGPRIORITYCLASS` value.
    pub fn class(self) -> i32 {
        match self {
            GpuProcessPriority::Normal => CLASS_NORMAL,
            GpuProcessPriority::AboveNormal => CLASS_ABOVE_NORMAL,
            GpuProcessPriority::High => CLASS_HIGH,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            GpuProcessPriority::Normal => "normal",
            GpuProcessPriority::AboveNormal => "above_normal",
            GpuProcessPriority::High => "high",
        }
    }
}

/// `D3DKMT_SCHEDULINGPRIORITYCLASS_*`.
pub const CLASS_IDLE: i32 = 0;
pub const CLASS_BELOW_NORMAL: i32 = 1;
pub const CLASS_NORMAL: i32 = 2;
pub const CLASS_ABOVE_NORMAL: i32 = 3;
pub const CLASS_HIGH: i32 = 4;
pub const CLASS_REALTIME: i32 = 5;

/// Name of a `D3DKMT_SCHEDULINGPRIORITYCLASS` value.
pub fn class_name(class: i32) -> &'static str {
    match class {
        CLASS_IDLE => "idle",
        CLASS_BELOW_NORMAL => "below_normal",
        CLASS_NORMAL => "normal",
        CLASS_ABOVE_NORMAL => "above_normal",
        CLASS_HIGH => "high",
        CLASS_REALTIME => "realtime",
        _ => "unknown",
    }
}

/// The class `SIGIL_GPU_PROCESS_PRIORITY` asks for (`None`: unset, empty or unknown).
pub fn from_env() -> Option<GpuProcessPriority> {
    std::env::var(ENV).ok().and_then(|v| GpuProcessPriority::parse(&v)).or(DEFAULT)
}

/// What [`apply_from_env_once`] did: `None` when nothing was asked, else the class asked for and
/// the outcome (the class the OS reports afterwards, or why it refused).
pub fn applied() -> Option<&'static (GpuProcessPriority, Result<i32, String>)> {
    APPLIED.get().and_then(Option::as_ref)
}

static APPLIED: OnceLock<Option<(GpuProcessPriority, Result<i32, String>)>> = OnceLock::new();

/// Apply [`from_env`] once per process (the first session plan; later calls are free). Logs the
/// outcome.
pub fn apply_from_env_once() {
    APPLIED.get_or_init(|| {
        let wanted = from_env()?;
        let outcome = set(wanted).and_then(|()| current());
        match &outcome {
            Ok(class) => tracing::info!(
                target: "sigil::gpu",
                "{ENV}: the process's GPU scheduling priority class is {} (asked for {})",
                class_name(*class),
                wanted.as_str()
            ),
            Err(why) => tracing::warn!(target: "sigil::gpu", "{ENV}={}: {why}; the GPU scheduling priority is unchanged", wanted.as_str()),
        }
        Some((wanted, outcome))
    });
}

/// The process's GPU scheduling priority class (`D3DKMTGetProcessSchedulingPriorityClass`).
pub fn current() -> Result<i32, String> {
    imp::get()
}

/// Set the process's GPU scheduling priority class (no admin, never `REALTIME`); `Err` with the
/// NTSTATUS when the OS refuses (e.g. a privilege it requires).
pub fn set(priority: GpuProcessPriority) -> Result<(), String> {
    imp::set(priority.class())
}

/// Set a raw class (restoring a value read with [`current`]); `REALTIME` is refused here too.
pub fn set_class(class: i32) -> Result<(), String> {
    if !(CLASS_IDLE..CLASS_REALTIME).contains(&class) {
        return Err(format!("GPU scheduling priority class {class} ({}) is not allowed", class_name(class)));
    }
    imp::set(class)
}

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;
    use std::sync::OnceLock;

    type SetFn = unsafe extern "system" fn(*mut c_void, i32) -> i32;
    type GetFn = unsafe extern "system" fn(*mut c_void, *mut i32) -> i32;

    /// `STATUS_PRIVILEGE_NOT_HELD`.
    const STATUS_PRIVILEGE_NOT_HELD: u32 = 0xC000_0061;

    struct Thunks {
        set: SetFn,
        get: GetFn,
        _lib: libloading::Library,
    }

    fn thunks() -> Result<&'static Thunks, String> {
        static T: OnceLock<Result<Thunks, String>> = OnceLock::new();
        T.get_or_init(|| {
            use libloading::os::windows::{Library, LOAD_LIBRARY_SEARCH_SYSTEM32};
            // SAFETY: the system's gdi32.dll from System32 only.
            let lib: libloading::Library = unsafe { Library::load_with_flags("gdi32.dll", LOAD_LIBRARY_SEARCH_SYSTEM32) }
                .map(Into::into)
                .map_err(|e| format!("gdi32.dll could not be loaded ({e})"))?;
            // SAFETY: the documented signatures of these exports (d3dkmthk.h).
            unsafe {
                let set = *lib
                    .get::<SetFn>(b"D3DKMTSetProcessSchedulingPriorityClass\0")
                    .map_err(|e| format!("gdi32.dll has no D3DKMTSetProcessSchedulingPriorityClass ({e})"))?;
                let get = *lib
                    .get::<GetFn>(b"D3DKMTGetProcessSchedulingPriorityClass\0")
                    .map_err(|e| format!("gdi32.dll has no D3DKMTGetProcessSchedulingPriorityClass ({e})"))?;
                Ok(Thunks { set, get, _lib: lib })
            }
        })
        .as_ref()
        .map_err(Clone::clone)
    }

    fn current_process() -> *mut c_void {
        // SAFETY: returns the calling process's pseudo-handle (never closed).
        unsafe { windows::Win32::System::Threading::GetCurrentProcess() }.0
    }

    fn status(what: &str, code: i32) -> String {
        if code as u32 == STATUS_PRIVILEGE_NOT_HELD {
            format!("{what}: Windows requires a privilege Sigil does not request (STATUS_PRIVILEGE_NOT_HELD)")
        } else {
            format!("{what} failed (NTSTATUS {:#010X})", code as u32)
        }
    }

    pub(super) fn get() -> Result<i32, String> {
        let t = thunks()?;
        let mut class = 0i32;
        // SAFETY: the process pseudo-handle and a valid out pointer.
        let code = unsafe { (t.get)(current_process(), &mut class) };
        if code < 0 {
            return Err(status("D3DKMTGetProcessSchedulingPriorityClass", code));
        }
        Ok(class)
    }

    pub(super) fn set(class: i32) -> Result<(), String> {
        let t = thunks()?;
        // SAFETY: the process pseudo-handle and a class value below REALTIME (checked by callers).
        let code = unsafe { (t.set)(current_process(), class) };
        if code < 0 {
            return Err(status("D3DKMTSetProcessSchedulingPriorityClass", code));
        }
        Ok(())
    }
}

#[cfg(not(windows))]
mod imp {
    pub(super) fn get() -> Result<i32, String> {
        Err("GPU scheduling priority classes exist only on Windows".into())
    }

    pub(super) fn set(_class: i32) -> Result<(), String> {
        Err("GPU scheduling priority classes exist only on Windows".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priorities_parse_and_never_reach_realtime() {
        assert_eq!(GpuProcessPriority::parse("HIGH"), Some(GpuProcessPriority::High));
        assert_eq!(GpuProcessPriority::parse("above-normal"), Some(GpuProcessPriority::AboveNormal));
        assert_eq!(GpuProcessPriority::parse(" normal "), Some(GpuProcessPriority::Normal));
        assert_eq!(GpuProcessPriority::parse("realtime"), None);
        assert_eq!(GpuProcessPriority::parse(""), None);
        for p in [GpuProcessPriority::Normal, GpuProcessPriority::AboveNormal, GpuProcessPriority::High] {
            assert_eq!(GpuProcessPriority::parse(p.as_str()), Some(p));
            assert!(p.class() < CLASS_REALTIME);
        }
        assert!(set_class(CLASS_REALTIME).unwrap_err().contains("not allowed"));
        assert!(set_class(-1).is_err());
        assert_eq!(DEFAULT, None, "no class is applied unless asked for");
    }

    /// Without admin rights: the class can be read, `normal` and `high` are either applied or
    /// refused with a clean reason (never a crash or a prompt), and the original class is restored.
    #[cfg(windows)]
    #[test]
    fn the_class_changes_without_admin_or_refuses_cleanly() {
        let original = match current() {
            Ok(c) => c,
            Err(why) => {
                eprintln!("no D3DKMT thunks here ({why}); skipped");
                return;
            }
        };
        eprintln!("GPU scheduling priority class: {} ({original})", class_name(original));
        for p in [GpuProcessPriority::Normal, GpuProcessPriority::High] {
            match set(p) {
                Ok(()) => {
                    let now = current().unwrap();
                    eprintln!("{} -> {}", p.as_str(), class_name(now));
                    assert_eq!(now, p.class());
                }
                Err(why) => {
                    eprintln!("{} refused: {why}", p.as_str());
                    assert!(why.contains("NTSTATUS") || why.contains("privilege"), "{why}");
                }
            }
        }
        if original < CLASS_REALTIME {
            set_class(original).unwrap();
            assert_eq!(current().unwrap(), original);
        }
    }
}
