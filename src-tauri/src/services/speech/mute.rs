//! The Rust STT mute gate for speech pipelines (P2-A). Internal to the pipeline: no frozen API.
//!
//! Gating happens upstream of everything, in the capture hub: while a pipeline's mute group is
//! muted (`MuteGroup::Stt` for microphone sources, `MuteGroup::Others` for loopback sources; the
//! TS `audioDevices` service mirrors the STT hard mute with `plugin:audio|set_capture_mute`), the
//! hub withholds the audio from the pipeline's consumer entirely, so muted speech never reaches the
//! segmenter or an engine and is never decoded or billed. The consumer reports a discontinuity
//! (`ReadInfo::gap_after`) at the mute edge; this module decides what that discontinuity means for
//! the utterance that was open:
//!
//! - **the mute started** ([`GapAction::DropOpen`]): the open utterance is dropped, never decoded
//!   (`speech:utterance` kind `dropped`, reason `muted`, when it had been announced with
//!   `speech_start`; an unconfirmed onset is discarded silently). Utterances that closed before the
//!   mute were spoken unmuted and are still decoded;
//! - **anything else** (stream restart, move to another device, producer overrun, drop-oldest)
//!   ([`GapAction::CloseOpen`]): the open utterance is closed there as `forced` and decoded, so an
//!   utterance never spans two devices or a hole in the audio.
//!
//! The unmute edge needs no action: the hub reports a discontinuity only after audio went out, and
//! none does while muted.

use crate::services::audio::hub::{self, MuteGroup, SourceSpec};

/// What the pipeline does with the open utterance at a capture discontinuity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GapAction {
    /// The mute started: drop the open utterance (reason `muted`).
    DropOpen,
    /// Close the open utterance (as `forced`) and decode it.
    CloseOpen,
}

/// The action for a discontinuity seen while the source's mute group is `muted_now`.
pub(crate) fn gap_action(muted_now: bool) -> GapAction {
    if muted_now {
        GapAction::DropOpen
    } else {
        GapAction::CloseOpen
    }
}

/// The mute group that gates a pipeline source id (`""` / device name → STT, `loopback:*` →
/// other players).
pub(crate) fn mute_group(source: &str) -> MuteGroup {
    crate::services::audio::stt_mute_group(&SourceSpec::parse(Some(source)))
}

/// Whether `group` is muted in the app's capture hub now (`false` when the hub is unavailable).
pub(crate) fn hub_muted(group: MuteGroup) -> bool {
    hub::global().map(|h| h.is_muted(group)).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mute_edges_drop_and_other_gaps_close() {
        assert_eq!(gap_action(true), GapAction::DropOpen);
        assert_eq!(gap_action(false), GapAction::CloseOpen);
    }

    #[test]
    fn sources_map_to_their_mute_groups() {
        assert_eq!(mute_group(""), MuteGroup::Stt);
        assert_eq!(mute_group("Microphone (USB)"), MuteGroup::Stt);
        assert_eq!(mute_group("loopback:default"), MuteGroup::Others);
        assert_eq!(mute_group("LOOPBACK:Speakers"), MuteGroup::Others);
    }
}
