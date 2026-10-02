//! The microphone while TTS plays into the voice changer (P6-J): a policy at the capture hub's
//! `voice_changer` mute gate.
//!
//! The policy mirrors the voice changer's own "microphone while TTS plays" setting (its aux mic
//! gate: `none` / `duck` / `mute`; the TS TTS service forwards it with `set_mic_policy`):
//! - `mute`: while TTS audio flows into the voice changer (from its first block until every block
//!   of it was mixed), the sink closes the hub's `voice_changer` gate, so the microphone does not
//!   reach the voice changer at all. The voice changer keeps running on synthesized silence
//!   meanwhile (that is what it does while its gate is closed), and the TTS is mixed into it as
//!   before. After the last audio the gate stays closed for a hold: [`DEFAULT_HOLD`] (it covers the
//!   voice changer's own release and the gapless join to an already synthesized next sentence), or
//!   [`DEFAULT_BRIDGE`] while more speech for the voice changer is already queued (the next
//!   sentence of the same reply is still synthesizing), so the microphone does not open between
//!   the sentences of one reply. The bridge only keeps a gate the sink still holds closed: speech
//!   that is merely queued never closes an open gate (the microphone stays open until its audio
//!   arrives), and the bridge is bounded, so a producer that never delivers cannot keep the
//!   microphone closed.
//! - Onset lead: the voice changer treats each edge of the hub gate as a capture gap (it restarts
//!   its model stream and denoiser and trims its output catch-up). So at the onset of a reply (the
//!   gate not held yet) the sink closes the gate first and feeds the first TTS block
//!   [`DEFAULT_LEAD`] later ([`MicGate::onset_wait`]): the restart lands on the silence before the
//!   speech, not on its first words. Later sentences of the reply join gaplessly (the gate is
//!   already held), and the opening edge comes after the hold, on silence again.
//! - `duck` and `none`: nothing at the hub (its gate is on / off only). The voice changer's mixer
//!   ducks the microphone itself while aux audio plays.
//!
//! The voice changer's mixer applies its gate as well (with a 5 ms attack), so the two agree; the
//! hub gate adds that no microphone audio enters the voice changer while TTS speaks through it,
//! including the pauses inside one TTS reply (the mixer's own hold is 250 ms).
//!
//! Ownership: the sink opens the gate only when it closed it. Nothing else drives the
//! `voice_changer` group today; a second writer would need a reference-counted gate in the hub.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// What happens to the microphone at the hub while TTS plays into the voice changer. Wire:
/// `none` | `duck` | `mute` (the voice changer's aux mic gate values).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MicPolicy {
    #[default]
    None,
    /// The voice changer ducks the microphone itself; nothing at the hub.
    Duck,
    /// The hub gate closes while TTS plays into the voice changer.
    Mute,
}

impl MicPolicy {
    /// Parse a wire value; anything unknown is [`MicPolicy::None`] (like the voice changer).
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "mute" => MicPolicy::Mute,
            "duck" => MicPolicy::Duck,
            _ => MicPolicy::None,
        }
    }
}

/// Microphone hold after the last TTS audio was mixed: the voice changer's own gate holds 250 ms,
/// then releases over about 80 ms.
pub(crate) const DEFAULT_HOLD: Duration = Duration::from_millis(330);
/// Longest hold while more speech for the voice changer is queued (the next sentence of the reply
/// is still being synthesized).
pub(crate) const DEFAULT_BRIDGE: Duration = Duration::from_secs(2);
/// At a reply's onset, how long the hub gate is closed before the first TTS block goes into the
/// voice changer. The hub applies a mute at its next device block (~10 ms) and wakes the voice
/// changer with a mute marker, which records the gap and restarts its stream on the next block it
/// converts (512 frames, ~11 ms, by default); 120 ms covers that for blocks up to ~4096 frames at
/// 48 kHz with margin. Larger blocks (very weak hardware) restart on the block that also starts
/// the speech, which then converts from a fresh stream.
pub(crate) const DEFAULT_LEAD: Duration = Duration::from_millis(120);

/// What the sink sends into the voice changer at one scheduler pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MicFlow {
    /// Nothing, and nothing more is queued for it.
    Idle,
    /// Nothing right now, but a begun utterance will play into it (its sentence is still being
    /// synthesized, or is waiting for its prebuffer).
    MoreQueued,
    /// TTS audio is queued in it or converted for it.
    Playing,
}

/// The policy's holds after the last audio (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MicHolds {
    /// Nothing more queued.
    pub hold: Duration,
    /// More speech for the voice changer is queued.
    pub bridge: Duration,
}

/// The hub gate the policy drives (the `voice_changer` mute group; a fake in tests).
pub(crate) trait MicGatePort: Send + Sync {
    fn is_muted(&self) -> bool;
    fn set_muted(&self, muted: bool);
}

/// The app's capture hub, `voice_changer` group. Only touched while TTS flows into the voice
/// changer, which means the voice changer (and so the hub) is running.
pub(crate) struct HubVoiceChangerGate;

impl MicGatePort for HubVoiceChangerGate {
    fn is_muted(&self) -> bool {
        use crate::services::audio::hub::{self, MuteGroup};
        hub::global().is_ok_and(|h| h.is_muted(MuteGroup::VoiceChanger))
    }

    fn set_muted(&self, muted: bool) {
        use crate::services::audio::hub::{self, MuteGroup};
        match hub::global() {
            Ok(h) => h.set_muted(MuteGroup::VoiceChanger, muted),
            Err(e) => tracing::warn!(target: "sigil::tts_sink", "capture hub unavailable for the TTS microphone gate: {e}"),
        }
    }
}

/// The policy's state (driven by the sink's scheduler).
#[derive(Debug, Default)]
pub(crate) struct MicGate {
    policy: MicPolicy,
    /// The sink closed the hub gate (and must open it again).
    owned: bool,
    /// When the sink closed the gate it holds (set with `owned`).
    closed_at: Option<Instant>,
    /// Last moment TTS audio was flowing into the voice changer.
    last_flow: Option<Instant>,
}

impl MicGate {
    pub(crate) fn policy(&self) -> MicPolicy {
        self.policy
    }

    pub(crate) fn set_policy(&mut self, policy: MicPolicy) {
        self.policy = policy;
    }

    /// Whether the sink holds the hub gate closed now.
    #[cfg(test)]
    pub(crate) fn engaged(&self) -> bool {
        self.owned
    }

    /// One decision at `now`. `flow`: what goes into the voice changer now. Returns how long until
    /// the next decision is due while the gate is held without audio (the hold running out), else
    /// `None`.
    pub(crate) fn update(&mut self, flow: MicFlow, now: Instant, holds: MicHolds, port: &dyn MicGatePort) -> Option<Duration> {
        let flowing = flow == MicFlow::Playing;
        if flowing {
            self.last_flow = Some(now);
        }
        // The bridge keeps a gate the sink still holds closed between the sentences of one reply;
        // it never closes an open one (queued speech that is still synthesizing leaves the
        // microphone open until its audio arrives).
        let bridging = flow == MicFlow::MoreQueued && self.owned;
        let hold = if bridging { holds.hold.max(holds.bridge) } else { holds.hold };
        let held_for = self.last_flow.map(|t| now.saturating_duration_since(t));
        // Flowing audio always wants it closed (also with a zero hold: a held onset waits for it).
        let want = self.policy == MicPolicy::Mute && (flowing || held_for.is_some_and(|d| d < hold));
        if want {
            // (Re)close it: also after someone else opened it while TTS still plays.
            if !port.is_muted() {
                port.set_muted(true);
                if !self.owned {
                    self.owned = true;
                    self.closed_at = Some(now);
                }
            }
        } else {
            self.release(port);
        }
        if !self.owned || flowing {
            return None;
        }
        held_for.map(|d| hold.saturating_sub(d).max(Duration::from_millis(1)))
    }

    /// At the onset of TTS into the voice changer (nothing of the utterance went in yet): how long
    /// its first block must still wait, so that the hub gate has been closed for `lead` before it
    /// (see the module docs: the gate's edge restarts the voice changer's stream, and that should
    /// land on silence, not on the first words). `None`: feed it now (another policy than `mute`,
    /// no lead, the sink closed the gate at least `lead` ago, or someone else holds it closed).
    /// Before the gate is closed the answer is the whole `lead`: the scheduler pass that finds the
    /// first block converted also closes the gate.
    pub(crate) fn onset_wait(&self, now: Instant, lead: Duration, port: &dyn MicGatePort) -> Option<Duration> {
        if self.policy != MicPolicy::Mute || lead.is_zero() {
            return None;
        }
        if self.owned {
            let ready = self.closed_at.map_or(now, |at| at + lead);
            return (now < ready).then(|| ready - now);
        }
        // Muted by someone else: its edge is theirs, not this speech's onset.
        (!port.is_muted()).then_some(lead)
    }

    /// Open the gate if the sink closed it.
    pub(crate) fn release(&mut self, port: &dyn MicGatePort) {
        if self.owned {
            self.owned = false;
            self.closed_at = None;
            port.set_muted(false);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A hub gate that counts its changes.
    #[derive(Clone, Default)]
    pub(crate) struct FakeGate(pub Arc<(AtomicBool, AtomicUsize)>);

    impl FakeGate {
        pub(crate) fn muted(&self) -> bool {
            self.0 .0.load(Ordering::SeqCst)
        }

        pub(crate) fn changes(&self) -> usize {
            self.0 .1.load(Ordering::SeqCst)
        }
    }

    impl MicGatePort for FakeGate {
        fn is_muted(&self) -> bool {
            self.muted()
        }

        fn set_muted(&self, muted: bool) {
            if self.0 .0.swap(muted, Ordering::SeqCst) != muted {
                self.0 .1.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    const HOLDS: MicHolds = MicHolds {
        hold: Duration::from_millis(300),
        bridge: Duration::from_millis(2_000),
    };
    const HOLD: Duration = HOLDS.hold;
    use MicFlow::{Idle, MoreQueued, Playing};

    #[test]
    fn wire_values_match_the_voice_changer() {
        assert_eq!(serde_json::to_string(&MicPolicy::Mute).unwrap(), r#""mute""#);
        assert_eq!(serde_json::from_str::<MicPolicy>(r#""duck""#).unwrap(), MicPolicy::Duck);
        assert_eq!(MicPolicy::from_str_lossy(" MUTE "), MicPolicy::Mute);
        assert_eq!(MicPolicy::from_str_lossy("loud"), MicPolicy::None);
        assert_eq!(MicPolicy::default(), MicPolicy::None);
        // The defaults outlast the voice changer's own gate (250 ms hold + ~80 ms release).
        assert!(DEFAULT_HOLD >= Duration::from_millis(330) && DEFAULT_BRIDGE > DEFAULT_HOLD);
        // The onset lead outlasts the hub's device period plus a default voice-changer block, and
        // stays well below the prebuffer-scale latencies of the sink.
        assert!(DEFAULT_LEAD >= Duration::from_millis(100) && DEFAULT_LEAD < DEFAULT_HOLD);
    }

    /// Speech that is merely queued (its sentence still synthesizing) after the gate opened must not
    /// close it again: the microphone stays open until that speech's audio arrives.
    #[test]
    fn queued_speech_never_closes_an_open_gate() {
        let gate = FakeGate::default();
        let mut m = MicGate::default();
        m.set_policy(MicPolicy::Mute);
        let t0 = Instant::now();
        m.update(Playing, t0, HOLDS, &gate);
        assert_eq!(m.update(Idle, t0 + HOLD, HOLDS, &gate), None);
        assert!(!gate.muted() && gate.changes() == 2, "reply 1 done: open after the hold");
        // Reply 2 begins within the bridge's reach of reply 1's last audio, still synthesizing.
        for ms in [400, 1_000, 1_500, 1_999] {
            assert_eq!(m.update(MoreQueued, t0 + Duration::from_millis(ms), HOLDS, &gate), None);
            assert!(!gate.muted() && !m.engaged(), "queued at +{ms} ms: the microphone stays open");
        }
        assert_eq!(gate.changes(), 2, "no edges without TTS audio");
        // Its audio arrives: closed once, and bridged between its own sentences again.
        let t1 = t0 + Duration::from_millis(2_500);
        m.update(Playing, t1, HOLDS, &gate);
        assert!(gate.muted() && m.engaged());
        assert!(m
            .update(MoreQueued, t1 + Duration::from_secs(1), HOLDS, &gate)
            .is_some());
        assert!(gate.muted());
        assert_eq!(gate.changes(), 3);
    }

    /// At a reply's onset the first block waits until the gate has been closed for the lead; once
    /// the gate is held, later sentences go in at once.
    #[test]
    fn the_onset_waits_for_the_lead_after_the_gate_closed() {
        const LEAD: Duration = Duration::from_millis(120);
        let gate = FakeGate::default();
        let mut m = MicGate::default();
        let t0 = Instant::now();
        assert_eq!(m.onset_wait(t0, LEAD, &gate), None, "policy none: no lead");
        m.set_policy(MicPolicy::Mute);
        assert_eq!(m.onset_wait(t0, Duration::ZERO, &gate), None, "no lead configured");
        assert_eq!(m.onset_wait(t0, LEAD, &gate), Some(LEAD), "not closed yet: the whole lead");
        m.update(Playing, t0, HOLDS, &gate);
        assert!(gate.muted());
        assert_eq!(m.onset_wait(t0, LEAD, &gate), Some(LEAD));
        assert_eq!(
            m.onset_wait(t0 + LEAD - Duration::from_millis(1), LEAD, &gate),
            Some(Duration::from_millis(1))
        );
        assert_eq!(m.onset_wait(t0 + LEAD, LEAD, &gate), None);
        // Still held long after (the next sentence of the reply): no new lead, also after a
        // re-close of a gate someone else opened meanwhile.
        gate.set_muted(false);
        m.update(Playing, t0 + Duration::from_millis(500), HOLDS, &gate);
        assert!(gate.muted());
        assert_eq!(m.onset_wait(t0 + Duration::from_millis(500), LEAD, &gate), None);
        // Released: the next reply's onset waits again.
        let t1 = t0 + Duration::from_millis(500) + HOLD;
        m.update(Idle, t1, HOLDS, &gate);
        assert!(!gate.muted());
        assert_eq!(m.onset_wait(t1, LEAD, &gate), Some(LEAD));
        // A gate someone else holds closed: its edge is not this onset's, so no lead.
        gate.set_muted(true);
        assert_eq!(m.onset_wait(t1, LEAD, &gate), None);
        // Another policy never waits.
        gate.set_muted(false);
        m.set_policy(MicPolicy::Duck);
        assert_eq!(m.onset_wait(t1, LEAD, &gate), None);
    }

    #[test]
    fn mute_closes_while_flowing_and_holds_across_gaps() {
        let gate = FakeGate::default();
        let mut m = MicGate::default();
        m.set_policy(MicPolicy::Mute);
        let t0 = Instant::now();
        assert_eq!(m.update(Idle, t0, HOLDS, &gate), None, "nothing played yet");
        assert_eq!(m.update(MoreQueued, t0, HOLDS, &gate), None, "queued but not playing yet");
        assert!(!gate.muted());
        assert_eq!(m.update(Playing, t0, HOLDS, &gate), None);
        assert!(gate.muted() && m.engaged());
        // A 200 ms gap between sentences: still closed, the next decision at the hold's end.
        let wait = m.update(Idle, t0 + Duration::from_millis(200), HOLDS, &gate);
        assert_eq!(wait, Some(Duration::from_millis(100)));
        assert!(gate.muted());
        assert_eq!(m.update(Playing, t0 + Duration::from_millis(250), HOLDS, &gate), None);
        // Audio ends: open once the hold ran out, exactly once.
        let end = t0 + Duration::from_millis(250);
        assert!(m.update(Idle, end + Duration::from_millis(299), HOLDS, &gate).is_some());
        assert!(gate.muted());
        assert_eq!(m.update(Idle, end + HOLD, HOLDS, &gate), None);
        assert!(!gate.muted() && !m.engaged());
        assert_eq!(gate.changes(), 2, "one close, one open");
    }

    /// While the next sentence of the reply is still synthesizing, the gate stays closed for up to
    /// the bridge, not just the hold; a bridge that runs out opens it.
    #[test]
    fn the_bridge_keeps_the_gate_closed_between_the_sentences_of_a_reply() {
        let gate = FakeGate::default();
        let mut m = MicGate::default();
        m.set_policy(MicPolicy::Mute);
        let t0 = Instant::now();
        m.update(Playing, t0, HOLDS, &gate);
        // 1 s without audio, the next sentence queued: still closed, next decision at the bridge's end.
        let wait = m.update(MoreQueued, t0 + Duration::from_secs(1), HOLDS, &gate);
        assert_eq!(wait, Some(Duration::from_secs(1)));
        assert!(gate.muted());
        // Its audio arrives: still closed (no flicker), and the clock restarts.
        let t1 = t0 + Duration::from_millis(1_500);
        assert_eq!(m.update(Playing, t1, HOLDS, &gate), None);
        assert_eq!(gate.changes(), 1);
        // A sentence that never delivers: the bridge runs out and the gate opens.
        assert!(m.update(MoreQueued, t1 + HOLDS.bridge - Duration::from_millis(1), HOLDS, &gate).is_some());
        assert!(gate.muted());
        assert_eq!(m.update(MoreQueued, t1 + HOLDS.bridge, HOLDS, &gate), None);
        assert!(!gate.muted() && !m.engaged());
        // Nothing queued any more: only the short hold applies (the stop case).
        m.update(Playing, t1 + Duration::from_secs(5), HOLDS, &gate);
        assert!(gate.muted());
        m.update(Idle, t1 + Duration::from_secs(5) + HOLD, HOLDS, &gate);
        assert!(!gate.muted());
        assert_eq!(gate.changes(), 4);
    }

    /// A zero hold still closes the gate while audio flows (a held onset waits for that closing), and
    /// opens it as soon as nothing flows.
    #[test]
    fn flowing_audio_closes_the_gate_even_with_a_zero_hold() {
        let gate = FakeGate::default();
        let mut m = MicGate::default();
        m.set_policy(MicPolicy::Mute);
        let zero = MicHolds {
            hold: Duration::ZERO,
            bridge: Duration::ZERO,
        };
        let t0 = Instant::now();
        assert_eq!(m.update(Playing, t0, zero, &gate), None);
        assert!(gate.muted() && m.engaged());
        let lead = Duration::from_millis(120);
        assert_eq!(m.onset_wait(t0 + lead, lead, &gate), None);
        m.update(MoreQueued, t0 + lead, zero, &gate);
        assert!(!gate.muted() && !m.engaged());
    }

    #[test]
    fn duck_and_none_never_touch_the_hub() {
        for policy in [MicPolicy::None, MicPolicy::Duck] {
            let gate = FakeGate::default();
            let mut m = MicGate::default();
            m.set_policy(policy);
            assert_eq!(m.update(Playing, Instant::now(), HOLDS, &gate), None);
            assert_eq!(m.update(MoreQueued, Instant::now(), HOLDS, &gate), None);
            assert_eq!(gate.changes(), 0);
        }
    }

    #[test]
    fn a_gate_closed_by_someone_else_is_left_alone() {
        let gate = FakeGate::default();
        gate.set_muted(true);
        let mut m = MicGate::default();
        m.set_policy(MicPolicy::Mute);
        let t0 = Instant::now();
        m.update(Playing, t0, HOLDS, &gate);
        assert!(!m.engaged(), "not ours");
        m.update(Idle, t0 + HOLD * 2, HOLDS, &gate);
        assert!(gate.muted(), "never opened by the sink");
    }

    #[test]
    fn switching_the_policy_off_opens_at_once_and_reopened_gates_close_again() {
        let gate = FakeGate::default();
        let mut m = MicGate::default();
        m.set_policy(MicPolicy::Mute);
        let t0 = Instant::now();
        m.update(Playing, t0, HOLDS, &gate);
        assert!(gate.muted());
        // Someone opened the hub gate while TTS still plays: closed again.
        gate.set_muted(false);
        m.update(Playing, t0 + Duration::from_millis(10), HOLDS, &gate);
        assert!(gate.muted() && m.engaged());
        // The user picks "duck" while TTS plays: the hub gate opens at once.
        m.set_policy(MicPolicy::Duck);
        assert_eq!(m.update(Playing, t0 + Duration::from_millis(20), HOLDS, &gate), None);
        assert!(!gate.muted() && !m.engaged());
        // release() is idempotent and only opens what the sink closed.
        m.release(&gate);
        assert!(!gate.muted());
        assert_eq!(m.policy(), MicPolicy::Duck);
    }
}
