//! Real-time thread helpers: MMCSS for audio callback threads, EcoQoS opt-out, worker priority.
//!
//! Owned by P0-A; scaffolded by P0-S0. FROZEN SIGNATURES (change only through an INTEGRATE step):
//!
//! ```ignore
//! pub fn preload();                                   // resolve avrt.dll off the audio thread
//! pub fn register_mmcss_pro_audio_once() -> bool;     // callback threads only, once per thread
//! pub fn opt_out_power_throttling() -> bool;          // current thread: EcoQoS off (HighQoS)
//! pub fn opt_out_process_power_throttling() -> bool;  // whole process: EcoQoS off
//! pub enum WorkerPriority { Auto, ProtectGame, ProtectVoice }  // serde: auto | protect_game | protect_voice
//! pub fn set_worker_priority(priority: WorkerPriority, gpu_ep: bool) -> bool;  // current thread
//! ```
//!
//! Added by P1-C (capture hub): `prepare_audio_callback_thread_once() -> bool` (MMCSS "Pro Audio"
//! plus the thread's EcoQoS opt-out, once per callback thread).
//!
//! Rules (plan "Real-time hygiene" / "Voice-changer worker priority"):
//! - MMCSS "Pro Audio" goes on audio *callback* threads only (capture, output, monitor), never on
//!   inference workers. Call [`preload`] from a normal thread before opening streams, then
//!   [`register_mmcss_pro_audio_once`] at the top of each cpal callback: after the first call per
//!   thread it is a thread-local flag check (no allocation, lock, syscall or logging).
//! - Worker priority: `auto` = NORMAL on a GPU EP, BELOW_NORMAL on CPU; `protect_voice` = NORMAL
//!   plus HighQoS; `protect_game` = BELOW_NORMAL. `auto` and `protect_game` leave EcoQoS to the
//!   system (undoing an earlier `protect_voice` on a live switch). Admin rights are never requested.
//! - The process-level EcoQoS opt-out is called once at startup (P0-H `diagnostics::qos` may
//!   delegate to [`opt_out_process_power_throttling`]).
//!
//! `avrt.dll` is resolved lazily from System32 (never a load-time import, never from the app
//! directory). Every function returns `false` when the OS call is unavailable or fails; callers
//! treat that as "best effort, carry on".
#![allow(dead_code)] // Wired up by P0-A (voice changer) and P0-C (audio capture).

use serde::{Deserialize, Serialize};

/// Voice-changer worker thread priority policy (persisted as a snake_case string).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerPriority {
    /// NORMAL on a GPU execution provider, BELOW_NORMAL on CPU.
    #[default]
    Auto,
    /// BELOW_NORMAL: the game keeps its frame rate; conversion may glitch under CPU load.
    ProtectGame,
    /// NORMAL plus HighQoS (EcoQoS opt-out): conversion wins ties with the game.
    ProtectVoice,
}

impl WorkerPriority {
    /// Every accepted value, in UI order.
    pub const ALL: [WorkerPriority; 3] = [WorkerPriority::Auto, WorkerPriority::ProtectGame, WorkerPriority::ProtectVoice];

    /// The persisted / IPC string.
    pub fn as_str(self) -> &'static str {
        match self {
            WorkerPriority::Auto => "auto",
            WorkerPriority::ProtectGame => "protect_game",
            WorkerPriority::ProtectVoice => "protect_voice",
        }
    }

    /// Parse a persisted / IPC string; anything unknown maps to [`WorkerPriority::Auto`].
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "protect_game" => WorkerPriority::ProtectGame,
            "protect_voice" => WorkerPriority::ProtectVoice,
            _ => WorkerPriority::Auto,
        }
    }
}

#[cfg(windows)]
mod imp {
    use std::cell::Cell;
    use std::sync::OnceLock;
    use windows::core::{s, w, PCWSTR};
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_SYSTEM32};
    use windows::Win32::System::Threading::{
        GetCurrentProcess, GetCurrentThread, ProcessPowerThrottling, SetProcessInformation, SetThreadInformation,
        SetThreadPriority, ThreadPowerThrottling, PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED, PROCESS_POWER_THROTTLING_STATE, THREAD_POWER_THROTTLING_CURRENT_VERSION,
        THREAD_POWER_THROTTLING_EXECUTION_SPEED, THREAD_POWER_THROTTLING_STATE, THREAD_PRIORITY, THREAD_PRIORITY_BELOW_NORMAL,
        THREAD_PRIORITY_NORMAL,
    };

    /// `AvSetMmThreadCharacteristicsW(PCWSTR task, *mut u32 index) -> HANDLE` (NULL on failure).
    type AvSetMmThreadCharacteristicsW = unsafe extern "system" fn(PCWSTR, *mut u32) -> HANDLE;

    static AV_SET: OnceLock<Option<AvSetMmThreadCharacteristicsW>> = OnceLock::new();

    thread_local! {
        /// Whether this thread already attempted MMCSS registration (success or not).
        static MMCSS_ATTEMPTED: Cell<bool> = const { Cell::new(false) };
        static MMCSS_OK: Cell<bool> = const { Cell::new(false) };
        /// Whether this thread already ran `prepare_audio_callback_thread_once` (result cached).
        static CALLBACK_PREPARED: Cell<Option<bool>> = const { Cell::new(None) };
    }

    pub fn prepare_audio_callback_thread_once() -> bool {
        if let Some(ok) = CALLBACK_PREPARED.with(Cell::get) {
            return ok;
        }
        let mmcss = register_mmcss_pro_audio_once();
        // HighQoS for the callback thread: Windows 11 must not park it on efficiency cores or
        // throttle it while the app is in the background (a game has focus).
        let _ = opt_out_power_throttling();
        CALLBACK_PREPARED.with(|c| c.set(Some(mmcss)));
        mmcss
    }

    fn resolve_av_set() -> Option<AvSetMmThreadCharacteristicsW> {
        *AV_SET.get_or_init(|| unsafe {
            let module = LoadLibraryExW(w!("avrt.dll"), None, LOAD_LIBRARY_SEARCH_SYSTEM32).ok()?;
            let proc = GetProcAddress(module, s!("AvSetMmThreadCharacteristicsW"))?;
            // SAFETY: the export has exactly this signature (avrt.h); the module is never freed.
            Some(std::mem::transmute::<unsafe extern "system" fn() -> isize, AvSetMmThreadCharacteristicsW>(proc))
        })
    }

    pub fn preload() {
        let _ = resolve_av_set();
    }

    pub fn register_mmcss_pro_audio_once() -> bool {
        if MMCSS_ATTEMPTED.with(|a| a.replace(true)) {
            return MMCSS_OK.with(Cell::get);
        }
        let ok = match resolve_av_set() {
            Some(av_set) => {
                let mut task_index = 0u32;
                // SAFETY: valid static wide string and out pointer. The registration lives until the
                // thread exits (the OS reverts it then), so the handle is intentionally not kept.
                let handle = unsafe { av_set(w!("Pro Audio"), &mut task_index) };
                !handle.is_invalid()
            }
            None => false,
        };
        MMCSS_OK.with(|c| c.set(ok));
        ok
    }

    pub fn opt_out_power_throttling() -> bool {
        let state = THREAD_POWER_THROTTLING_STATE {
            Version: THREAD_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: THREAD_POWER_THROTTLING_EXECUTION_SPEED,
            StateMask: 0,
        };
        // SAFETY: pointer and size describe a live THREAD_POWER_THROTTLING_STATE.
        unsafe {
            SetThreadInformation(
                GetCurrentThread(),
                ThreadPowerThrottling,
                &state as *const _ as *const core::ffi::c_void,
                std::mem::size_of::<THREAD_POWER_THROTTLING_STATE>() as u32,
            )
            .is_ok()
        }
    }

    /// Hand the calling thread's EcoQoS state back to the system (undo [`opt_out_power_throttling`]).
    pub fn reset_power_throttling() -> bool {
        let state = THREAD_POWER_THROTTLING_STATE {
            Version: THREAD_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: 0,
            StateMask: 0,
        };
        // SAFETY: pointer and size describe a live THREAD_POWER_THROTTLING_STATE.
        unsafe {
            SetThreadInformation(
                GetCurrentThread(),
                ThreadPowerThrottling,
                &state as *const _ as *const core::ffi::c_void,
                std::mem::size_of::<THREAD_POWER_THROTTLING_STATE>() as u32,
            )
            .is_ok()
        }
    }

    pub fn opt_out_process_power_throttling() -> bool {
        let state = PROCESS_POWER_THROTTLING_STATE {
            Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            StateMask: 0,
        };
        // SAFETY: pointer and size describe a live PROCESS_POWER_THROTTLING_STATE.
        unsafe {
            SetProcessInformation(
                GetCurrentProcess(),
                ProcessPowerThrottling,
                &state as *const _ as *const core::ffi::c_void,
                std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
            )
            .is_ok()
        }
    }

    pub fn set_current_thread_priority(normal: bool) -> bool {
        let p: THREAD_PRIORITY = if normal { THREAD_PRIORITY_NORMAL } else { THREAD_PRIORITY_BELOW_NORMAL };
        // SAFETY: pseudo-handle of the calling thread.
        unsafe { SetThreadPriority(GetCurrentThread(), p).is_ok() }
    }

    /// `(priority, EcoQoS control mask)` of the calling thread; the mask is `None` where the OS
    /// cannot report it.
    #[cfg(test)]
    pub fn current_thread_policy() -> (i32, Option<u32>) {
        use windows::Win32::System::Threading::{GetThreadInformation, GetThreadPriority};
        let mut state = THREAD_POWER_THROTTLING_STATE {
            Version: THREAD_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: 0,
            StateMask: 0,
        };
        // SAFETY: pseudo-handle of the calling thread; pointer and size describe `state`.
        unsafe {
            let prio = GetThreadPriority(GetCurrentThread());
            let mask = GetThreadInformation(
                GetCurrentThread(),
                ThreadPowerThrottling,
                &mut state as *mut _ as *mut core::ffi::c_void,
                std::mem::size_of::<THREAD_POWER_THROTTLING_STATE>() as u32,
            )
            .ok()
            .map(|_| state.ControlMask);
            (prio, mask)
        }
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn preload() {}
    pub fn register_mmcss_pro_audio_once() -> bool {
        false
    }
    pub fn prepare_audio_callback_thread_once() -> bool {
        false
    }
    pub fn opt_out_power_throttling() -> bool {
        false
    }
    pub fn reset_power_throttling() -> bool {
        false
    }
    pub fn opt_out_process_power_throttling() -> bool {
        false
    }
    pub fn set_current_thread_priority(_normal: bool) -> bool {
        false
    }
}

/// Resolve the MMCSS entry point ahead of time. Call from a normal (non-real-time) thread before
/// opening audio streams, so the first callback never loads a DLL. Idempotent.
pub fn preload() {
    imp::preload()
}

/// Register the calling **audio callback** thread with the MMCSS "Pro Audio" task, once per thread.
/// Returns whether the thread is registered. Never call it from an inference worker.
pub fn register_mmcss_pro_audio_once() -> bool {
    imp::register_mmcss_pro_audio_once()
}

/// Prepare the calling **audio callback** thread once: MMCSS "Pro Audio" plus the thread's EcoQoS
/// opt-out. After the first call per thread it is a thread-local check (no allocation, lock,
/// syscall or logging), so it can sit at the top of every cpal callback. Returns whether MMCSS is
/// active. Call [`preload`] from a normal thread before opening the stream.
pub fn prepare_audio_callback_thread_once() -> bool {
    imp::prepare_audio_callback_thread_once()
}

/// Opt the calling thread out of EcoQoS power throttling (Windows 11 "HighQoS").
pub fn opt_out_power_throttling() -> bool {
    imp::opt_out_power_throttling()
}

/// Opt the whole process out of EcoQoS execution-speed throttling (once, at startup).
pub fn opt_out_process_power_throttling() -> bool {
    imp::opt_out_process_power_throttling()
}

/// Apply the worker priority policy to the calling (inference worker) thread. `gpu_ep` is whether
/// the session actually runs on a GPU execution provider (use the *active* EP, not the requested).
/// Returns `false` when any OS call failed. Never registers MMCSS.
///
/// The policy is complete on every call, so a live switch leaves nothing of the previous one
/// behind: `auto` and `protect_game` hand the thread's EcoQoS state back to the system (a thread
/// that was `protect_voice` before stays HighQoS otherwise). That reset is best effort (Windows
/// versions without thread power throttling refuse it, and there is nothing to undo there).
pub fn set_worker_priority(priority: WorkerPriority, gpu_ep: bool) -> bool {
    match priority {
        WorkerPriority::Auto => {
            let _ = imp::reset_power_throttling();
            imp::set_current_thread_priority(gpu_ep)
        }
        WorkerPriority::ProtectGame => {
            let _ = imp::reset_power_throttling();
            imp::set_current_thread_priority(false)
        }
        WorkerPriority::ProtectVoice => {
            let prio = imp::set_current_thread_priority(true);
            imp::opt_out_power_throttling() && prio
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{register_mmcss_pro_audio_once, WorkerPriority};

    #[test]
    fn worker_priority_strings_round_trip() {
        for p in WorkerPriority::ALL {
            assert_eq!(WorkerPriority::from_str_lossy(p.as_str()), p);
            assert_eq!(serde_json::to_string(&p).unwrap(), format!("\"{}\"", p.as_str()));
        }
        assert_eq!(WorkerPriority::from_str_lossy("bogus"), WorkerPriority::Auto);
        assert_eq!(WorkerPriority::from_str_lossy(" PROTECT_VOICE "), WorkerPriority::ProtectVoice);
    }

    /// A live switch away from `protect_voice` leaves nothing of it behind: the priority drops and
    /// the thread's EcoQoS state goes back to system-managed (control mask 0).
    #[cfg(windows)]
    #[test]
    fn switching_away_from_protect_voice_restores_default_qos() {
        use super::{imp, set_worker_priority};
        const BELOW_NORMAL: i32 = -1;
        const NORMAL: i32 = 0;
        std::thread::spawn(|| {
            assert!(set_worker_priority(WorkerPriority::ProtectVoice, false));
            let (prio, mask) = imp::current_thread_policy();
            assert_eq!(prio, NORMAL);
            let Some(mask) = mask else {
                return; // This Windows cannot report EcoQoS state; the priority part is covered.
            };
            assert_ne!(mask, 0, "protect_voice opts the thread out of EcoQoS");
            set_worker_priority(WorkerPriority::ProtectGame, true);
            assert_eq!(imp::current_thread_policy(), (BELOW_NORMAL, Some(0)));
            set_worker_priority(WorkerPriority::ProtectVoice, false);
            set_worker_priority(WorkerPriority::Auto, true);
            assert_eq!(imp::current_thread_policy(), (NORMAL, Some(0)));
        })
        .join()
        .unwrap();
    }

    #[test]
    fn mmcss_registration_is_idempotent_per_thread() {
        // Runs on a throwaway thread so the test runner's threads keep their scheduling class.
        std::thread::spawn(|| {
            let first = register_mmcss_pro_audio_once();
            assert_eq!(register_mmcss_pro_audio_once(), first);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn callback_thread_preparation_is_cached_per_thread() {
        use super::prepare_audio_callback_thread_once;
        std::thread::spawn(|| {
            let first = prepare_audio_callback_thread_once();
            assert_eq!(prepare_audio_callback_thread_once(), first);
            // The MMCSS part is the same one-shot registration.
            assert_eq!(register_mmcss_pro_audio_once(), first);
            #[cfg(windows)]
            if let (_, Some(mask)) = super::imp::current_thread_policy() {
                assert_ne!(mask, 0, "callback threads are opted out of EcoQoS");
            }
        })
        .join()
        .unwrap();
    }
}
