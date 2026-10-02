//! Other players' speech session helpers (P6-A): which capture source the `loopback` pipeline runs
//! on.
//!
//! The `loopback` pipeline ([`super::engine_api::PIPELINE_LOOPBACK`], `MuteGroup::Others`) runs
//! next to the user's own `mic` pipeline with its own epoch, engines and latency budget. Its
//! source is normally `process:VRChat.exe` (per-process loopback, `audio::loopback_process`): only
//! the game's sound, so the user's own TTS, music and voice-changer monitor never become "other
//! players". [`plan_source_with`] decides what to capture now:
//!
//! | Requested | Situation | Captured now | `fallback` | `watch` |
//! |---|---|---|---|---|
//! | `process:<exe>` | running | `process:<exe as running>` | – | `<exe>` |
//! | `process:<exe>` | not running, fallback on | `loopback:default` | `not_running` | `<exe>` |
//! | `process:<exe>` | not running, fallback off | nothing (wait) | `not_running` | `<exe>` |
//! | `process:<exe>` | Windows older than build 20348, fallback on | `loopback:default` | `unsupported` | – |
//! | `process:<exe>` | Windows older than build 20348, fallback off | nothing | `unsupported` | – |
//! | `loopback:default` / `loopback:<output>` | – | as requested | – | – |
//! | `""` / `default` (the default *microphone*) | – | `loopback:default` | – | – |
//! | `<input name>` (e.g. a virtual cable) | – | as requested | – | – |
//!
//! The Others service (TS) asks for the plan through `plugin:audio|list_loopback_processes
//! { source, fallback }`, starts the pipeline on `source`, and while `watch` is set asks again
//! every few seconds: when the program starts (or exits, with the fallback on) the plan changes and
//! the service moves the pipeline to the new source. A process stream that loses its program
//! keeps waiting for it in the hub (no error), so with the fallback off nothing has to restart.
//! Its text is published by the TS Others service on `others.stt` (and translated in the reverse
//! direction on `others.translation`), never on `text.*`, so it never reaches the chatbox, TTS or
//! history.

use serde::{Deserialize, Serialize};

use crate::services::audio::loopback::{SourceSpec, DEFAULT_LOOPBACK_KEY, PROCESS_PREFIX};
use crate::services::audio::loopback_process::{self, normalize_exe};

/// Pipeline key of the other players' pipeline (same as `engine_api::PIPELINE_LOOPBACK`).
#[cfg_attr(not(test), allow(dead_code))] // The TS Others service starts the pipeline; Rust callers may use it.
pub const PIPELINE_ID: &str = super::engine_api::PIPELINE_LOOPBACK;

/// Why a plan captures something other than the requested program. Wire: snake_case.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackReason {
    /// Per-process capture needs Windows 11 or Windows 10 build 20348+.
    Unsupported,
    /// The requested program is not running.
    NotRunning,
}

/// What the other players' pipeline captures now (TS `OthersSourcePlan`). Wire: camelCase.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourcePlan {
    /// The requested source, normalized (`process:VRChat.exe`, `loopback:default`, ...).
    pub requested: String,
    /// The capture source id to start the pipeline with; `None` = capture nothing for now.
    pub source: Option<String>,
    /// Set when `source` is not the requested program.
    pub fallback: Option<FallbackReason>,
    /// Program to watch for: plan again when it starts or exits (per-process requests on PCs that
    /// support them).
    pub watch: Option<String>,
    /// The requested program is running now (per-process requests; `false` otherwise).
    pub running: bool,
}

/// What the planner needs to know about the PC (the system probe in the app, a fake in tests).
pub trait ProcessProbe {
    /// Per-process loopback works on this PC.
    fn supported(&self) -> bool;
    /// The executable name as it runs, when a process named `exe` is running.
    fn running(&self, exe: &str) -> Option<String>;
}

/// The plan for `requested` (an Others capture source id) on the PC `probe` describes; `fallback`:
/// capture the default output while a requested program cannot be captured.
pub fn plan_source_with(requested: &str, fallback: bool, probe: &dyn ProcessProbe) -> SourcePlan {
    let device = |source: String| SourcePlan {
        requested: source.clone(),
        source: Some(source),
        fallback: None,
        watch: None,
        running: false,
    };
    match SourceSpec::parse(Some(requested)) {
        SourceSpec::Process(name) => {
            let Some(exe) = normalize_exe(&name) else {
                // Nothing usable after `process:`: same as `process:` (the default output).
                return device(DEFAULT_LOOPBACK_KEY.to_string());
            };
            let requested = format!("{PROCESS_PREFIX}{exe}");
            let fallback_source = fallback.then(|| DEFAULT_LOOPBACK_KEY.to_string());
            if !probe.supported() {
                return SourcePlan {
                    requested,
                    source: fallback_source,
                    fallback: Some(FallbackReason::Unsupported),
                    watch: None,
                    running: false,
                };
            }
            match probe.running(&exe) {
                Some(running) => SourcePlan {
                    requested,
                    source: Some(format!("{PROCESS_PREFIX}{running}")),
                    fallback: None,
                    watch: Some(exe),
                    running: true,
                },
                None => SourcePlan {
                    requested,
                    source: fallback_source,
                    fallback: Some(FallbackReason::NotRunning),
                    watch: Some(exe),
                    running: false,
                },
            }
        }
        // Other players are never the user's own default microphone: an empty or "default"
        // setting means what the default output plays.
        SourceSpec::DefaultInput | SourceSpec::DefaultLoopback => device(DEFAULT_LOOPBACK_KEY.to_string()),
        spec @ (SourceSpec::Loopback(_) | SourceSpec::Input(_)) => device(spec.key()),
    }
}

/// [`plan_source_with`] on this PC (one process snapshot). The app plans through
/// `list_loopback_processes` (which reuses its own snapshot); this is the entry point for Rust
/// callers.
#[cfg_attr(not(test), allow(dead_code))]
pub fn plan_source(requested: &str, fallback: bool) -> SourcePlan {
    plan_source_with(requested, fallback, &loopback_process::system_probe())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        supported: bool,
        running: &'static [&'static str],
    }

    impl ProcessProbe for Fake {
        fn supported(&self) -> bool {
            self.supported
        }
        fn running(&self, exe: &str) -> Option<String> {
            self.running
                .iter()
                .find(|r| loopback_process::same_exe(r, exe))
                .map(|r| r.to_string())
        }
    }

    const PC: Fake = Fake {
        supported: true,
        running: &["VRChat.exe", "chrome.exe"],
    };
    const IDLE_PC: Fake = Fake {
        supported: true,
        running: &[],
    };
    const OLD_PC: Fake = Fake {
        supported: false,
        running: &["VRChat.exe"],
    };

    fn plan(source: Option<&str>, fallback: Option<FallbackReason>, watch: Option<&str>, running: bool, requested: &str) -> SourcePlan {
        SourcePlan {
            requested: requested.into(),
            source: source.map(Into::into),
            fallback,
            watch: watch.map(Into::into),
            running,
        }
    }

    #[test]
    fn a_running_program_is_captured_and_watched() {
        assert_eq!(
            plan_source_with("process:VRChat.exe", true, &PC),
            plan(Some("process:VRChat.exe"), None, Some("VRChat.exe"), true, "process:VRChat.exe")
        );
        // Names are normalized and the running process's own spelling is used.
        assert_eq!(
            plan_source_with(" PROCESS: vrchat ", false, &PC),
            plan(Some("process:VRChat.exe"), None, Some("vrchat.exe"), true, "process:vrchat.exe")
        );
    }

    #[test]
    fn a_program_that_is_not_running_falls_back_or_waits() {
        assert_eq!(
            plan_source_with("process:VRChat.exe", true, &IDLE_PC),
            plan(Some("loopback:default"), Some(FallbackReason::NotRunning), Some("VRChat.exe"), false, "process:VRChat.exe")
        );
        assert_eq!(
            plan_source_with("process:VRChat.exe", false, &IDLE_PC),
            plan(None, Some(FallbackReason::NotRunning), Some("VRChat.exe"), false, "process:VRChat.exe")
        );
    }

    #[test]
    fn old_windows_falls_back_without_watching() {
        assert_eq!(
            plan_source_with("process:VRChat.exe", true, &OLD_PC),
            plan(Some("loopback:default"), Some(FallbackReason::Unsupported), None, false, "process:VRChat.exe")
        );
        assert_eq!(
            plan_source_with("process:VRChat.exe", false, &OLD_PC),
            plan(None, Some(FallbackReason::Unsupported), None, false, "process:VRChat.exe")
        );
    }

    #[test]
    fn devices_are_captured_as_requested_and_never_the_default_microphone() {
        for (requested, source) in [
            ("loopback:default", "loopback:default"),
            ("LOOPBACK:", "loopback:default"),
            ("loopback:Speakers (Realtek)", "loopback:Speakers (Realtek)"),
            ("process:", "loopback:default"),
            ("process: \\ ", "loopback:default"),
            ("", "loopback:default"),
            ("default", "loopback:default"),
            ("CABLE Output (VB-Audio)", "CABLE Output (VB-Audio)"),
        ] {
            for probe in [&PC, &IDLE_PC, &OLD_PC] {
                for fallback in [true, false] {
                    assert_eq!(
                        plan_source_with(requested, fallback, probe),
                        plan(Some(source), None, None, false, source),
                        "{requested:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_plan_wire_format_is_camel_case() {
        let p = plan_source_with("process:VRChat.exe", true, &IDLE_PC);
        assert_eq!(
            serde_json::to_string(&p).unwrap(),
            r#"{"requested":"process:VRChat.exe","source":"loopback:default","fallback":"not_running","watch":"VRChat.exe","running":false}"#
        );
        assert_eq!(PIPELINE_ID, "loopback");
    }

    #[test]
    fn every_planned_source_maps_to_the_others_mute_group_or_an_explicit_input() {
        use crate::services::audio::hub::MuteGroup;
        for (requested, probe) in [("process:VRChat.exe", &PC), ("process:VRChat.exe", &IDLE_PC), ("loopback:x", &PC)] {
            let source = plan_source_with(requested, true, probe).source.unwrap();
            assert_eq!(crate::services::speech::mute::mute_group(&source), MuteGroup::Others, "{source}");
        }
    }

    #[test]
    fn the_system_planner_runs() {
        // Whatever this PC is, a device request never depends on it.
        assert_eq!(plan_source("loopback:default", true).source.as_deref(), Some("loopback:default"));
        let p = plan_source("process:surely-not-running-sigil-test.exe", true);
        assert_eq!(p.source.as_deref(), Some("loopback:default"));
        assert!(p.fallback.is_some());
    }
}
