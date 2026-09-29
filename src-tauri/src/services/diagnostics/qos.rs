//! Process-level quality of service: opt Sigil out of Windows 11 EcoQoS.
//!
//! Windows 11 puts background / occluded processes on efficiency cores at reduced clocks (EcoQoS)
//! and, from 22H2, ignores their timer-resolution requests while minimized. Streamers keep Sigil
//! minimized behind OBS and VRChat, where both would stretch voice-changer and STT deadlines. One
//! `SetProcessInformation(ProcessPowerThrottling)` call at startup turns both policies off for the
//! whole process ("HighQoS"). Admin rights are never needed. Per-thread QoS for the voice-changer
//! worker stays in [`crate::services::rt_thread`].

use serde::Serialize;

/// What [`opt_out_power_throttling`] managed to change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct QosReport {
    /// EcoQoS execution-speed throttling disabled for the process.
    pub execution_speed: bool,
    /// Timer-resolution requests honoured while minimized/occluded (Windows 11 22H2+).
    pub timer_resolution: bool,
}

/// Opt the whole process out of EcoQoS execution-speed throttling and out of the "ignore timer
/// resolution" policy. Idempotent and best effort: older Windows versions reject the
/// timer-resolution bit, in which case only execution speed is changed.
pub fn opt_out_power_throttling() -> QosReport {
    let report = imp::apply();
    tracing::info!(
        target: "sigil::diagnostics",
        "process QoS: EcoQoS throttling {}, timer-resolution throttling {}",
        if report.execution_speed { "disabled" } else { "unchanged (not supported)" },
        if report.timer_resolution { "disabled" } else { "unchanged (not supported)" }
    );
    report
}

#[cfg(windows)]
mod imp {
    use super::QosReport;
    use windows::Win32::System::Threading::{
        GetCurrentProcess, ProcessPowerThrottling, SetProcessInformation, PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED, PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
        PROCESS_POWER_THROTTLING_STATE,
    };

    fn set(control_mask: u32) -> bool {
        // ControlMask = policies Sigil takes control of; StateMask = 0 = all of them OFF.
        let state = PROCESS_POWER_THROTTLING_STATE {
            Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: control_mask,
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

    pub fn apply() -> QosReport {
        if set(PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION) {
            return QosReport {
                execution_speed: true,
                timer_resolution: true,
            };
        }
        // Windows 10 / Windows 11 before 22H2 reject the timer-resolution bit as a whole call.
        QosReport {
            execution_speed: crate::services::rt_thread::opt_out_process_power_throttling(),
            timer_resolution: false,
        }
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn apply() -> super::QosReport {
        super::QosReport::default()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn opt_out_is_idempotent() {
        let first = super::opt_out_power_throttling();
        assert_eq!(super::opt_out_power_throttling(), first);
        // A timer-resolution opt-out implies the (older) execution-speed one succeeded too.
        assert!(!first.timer_resolution || first.execution_speed);
    }
}
