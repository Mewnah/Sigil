//! Silent-microphone detection for the `mic` pipeline (W3).
//!
//! A microphone that delivers digital silence looks healthy to the capture hub: callbacks keep
//! arriving, so the stall watchdog stays quiet, and the pipeline just hears nothing. Typical causes:
//! Windows' "Let desktop apps access your microphone" turned off (after an update or by policy), a
//! hardware mute switch on a headset, a driver that returns zeros, or "System default" following
//! Windows to a virtual or muted endpoint. [`SignalWatch`] sees the pipeline's 16 kHz mono feed on
//! the capture thread (never in the audio callback, so nothing here allocates, locks or logs on a
//! real-time path) and reports two changes, each once:
//!
//! - [`SignalChange::Silent`]: the feed stayed dead for [`SILENCE_AFTER_SAMPLES`] (5 s of captured
//!   audio; counted in samples, not wall-clock time, so a stalled capture thread does not fake it).
//!   Once the stream has delivered real signal ("heard"), the wait is [`SILENCE_AFTER_SIGNAL_SAMPLES`]
//!   (60 s) instead: a noise gate or denoiser (NVIDIA Broadcast, Krisp, a headset's DSP gate) sends
//!   exact zeros between phrases, and a pause is not a broken microphone. A privacy-off or dead device
//!   never produced signal, so it is still caught after 5 s, and a mid-session headset mute after a
//!   minute;
//! - [`SignalChange::Restored`]: real signal came back, or the capture was muted (a muted source is
//!   silent by design, so the warning ends there).
//!
//! "Dead" is what an unfed converter looks like, not "quiet": a chunk whose samples all lie within
//! [`SILENT_PEAK`] of zero (one 16-bit step) *and* barely differ from each other ([`SILENT_SPAN`], half
//! a step). Exact zeros, or a stuck +-1 step, are dead. A real microphone is not, even in a quiet
//! room, with its Windows input volume turned far down, or on a clean low-gain interface: its noise
//! floor flips the lowest bits, so a chunk has samples of different values. Sixteen-bit noise
//! attenuated to a few steps (a mic at 10 % input level) therefore reads as alive. Any chunk that is
//! not dead restarts the count, so a noise-gated virtual microphone that clicks now and then is alive,
//! and one that sits at exact zeros between phrases is reported only after the long wait above (the
//! report is a status line, never a restart or a dialog).
//!
//! **Unverified on real hardware**: the headline case (microphone privacy off) is detected only if
//! Windows fills the packet with zeros. WASAPI flags such packets `AUDCLNT_BUFFERFLAGS_SILENT`, whose
//! contents are formally undefined, and the capture goes through cpal 0.15, which does not read the
//! flag. Check once by hand on a PC with the privacy switch off (Settings -> Privacy -> Microphone ->
//! "Let desktop apps access your microphone"): start captions with `SIGIL_LOG=sigil=debug` and look for
//! the `no signal from` line after 5 s. If it never comes, the packets are not zero and the capture
//! path has to pass the flag along.
//!
//! The report never changes what the pipeline does: audio keeps flowing to the segmenter, and
//! the supervisor's stream handling is untouched.

use crate::services::speech_core::SAMPLE_RATE;

/// Captured audio (16 kHz samples) of a dead feed before it is reported, on a stream that never
/// delivered real signal: 5 s.
pub(crate) const SILENCE_AFTER_SAMPLES: u64 = 5 * SAMPLE_RATE as u64;
/// The same once the stream delivered real signal (a noise gate closing between phrases): 60 s.
pub(crate) const SILENCE_AFTER_SIGNAL_SAMPLES: u64 = 60 * SAMPLE_RATE as u64;
/// Largest absolute sample value of a dead chunk (full scale is 1.0): one 16-bit step.
pub(crate) const SILENT_PEAK: f32 = 1.0 / 32_768.0;
/// Largest difference between the samples of a dead chunk: half a 16-bit step. A feed that only
/// flips its lowest bit is a noise floor, not a dead converter.
pub(crate) const SILENT_SPAN: f32 = 0.5 / 32_768.0;
/// Live audio, without a dead chunk in between, that makes a stream "heard" and clears a reported
/// silence: 100 ms. One click in an otherwise dead feed is not a microphone.
pub(crate) const RESTORE_AFTER_SAMPLES: u64 = SAMPLE_RATE as u64 / 10;

/// What [`SignalWatch`] tells the pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SignalChange {
    /// The feed has been dead for the wait that applies ([`SignalWatch::threshold_secs`]).
    Silent,
    /// The silence reported before is over.
    Restored,
}

/// Whether `chunk` (not empty) looks like a dead feed: every sample within [`SILENT_PEAK`] of zero
/// and no two further apart than [`SILENT_SPAN`]. NaN is not dead (a broken driver is not a muted
/// microphone).
fn is_dead(chunk: &[f32]) -> bool {
    let mut lowest = f32::INFINITY;
    let mut highest = f32::NEG_INFINITY;
    for &sample in chunk {
        if sample.is_nan() || sample.abs() > SILENT_PEAK {
            return false;
        }
        lowest = lowest.min(sample);
        highest = highest.max(sample);
    }
    highest - lowest <= SILENT_SPAN
}

/// Counts consecutive dead audio on the capture thread.
#[derive(Debug, Default)]
pub(crate) struct SignalWatch {
    /// Consecutive samples of dead chunks.
    silent_samples: u64,
    /// Consecutive samples of live chunks.
    live_samples: u64,
    /// This stream delivered real signal for at least [`RESTORE_AFTER_SAMPLES`] in a row.
    heard: bool,
    /// The UI was told `Silent` and not yet `Restored`.
    told_silent: bool,
    /// The capture moved to another device while `told_silent`: the next `Silent` is sent again
    /// (it names the new device).
    rearm: bool,
    /// How many times `Silent` was returned.
    reports: u32,
}

impl SignalWatch {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Seconds of dead audio before `Silent` right now: the long wait once signal was heard.
    pub(crate) fn threshold_secs(&self) -> u64 {
        self.threshold() / u64::from(SAMPLE_RATE)
    }

    /// How many `Silent` changes were returned so far (the pipeline logs the first at info level only).
    pub(crate) fn reports(&self) -> u32 {
        self.reports
    }

    fn threshold(&self) -> u64 {
        if self.heard {
            SILENCE_AFTER_SIGNAL_SAMPLES
        } else {
            SILENCE_AFTER_SAMPLES
        }
    }

    /// Feed the next audio of the capture (any length). Returns a change at most once per call.
    pub(crate) fn observe(&mut self, chunk: &[f32]) -> Option<SignalChange> {
        if chunk.is_empty() {
            return None;
        }
        let len = chunk.len() as u64;
        if is_dead(chunk) {
            self.live_samples = 0;
            self.silent_samples = self.silent_samples.saturating_add(len);
            if self.silent_samples >= self.threshold() && (!self.told_silent || self.rearm) {
                self.told_silent = true;
                self.rearm = false;
                self.reports = self.reports.saturating_add(1);
                return Some(SignalChange::Silent);
            }
            return None;
        }
        self.silent_samples = 0;
        self.live_samples = self.live_samples.saturating_add(len);
        if self.live_samples < RESTORE_AFTER_SAMPLES {
            return None;
        }
        self.heard = true;
        if self.told_silent {
            self.told_silent = false;
            self.rearm = false;
            return Some(SignalChange::Restored);
        }
        None
    }

    /// The audio is discontinuous without a new device behind it (an overrun dropped some): count
    /// afresh. What was heard on this stream still counts, and a silence already reported stays
    /// reported.
    pub(crate) fn gap(&mut self) {
        self.silent_samples = 0;
        self.live_samples = 0;
    }

    /// The stream restarted or moved to another device: count afresh, and forget what the old one
    /// delivered (the new device has to prove itself; a silence on it is reported after 5 s). A
    /// silence already reported is reported again when it lasts on the new device, which may be
    /// another one than the first report named.
    pub(crate) fn restarted(&mut self) {
        self.gap();
        self.heard = false;
        self.rearm = self.told_silent;
    }

    /// The source is muted now (nothing is delivered): a silence reported before ends, since a muted
    /// microphone is silent by design. The count starts afresh after the unmute.
    pub(crate) fn muted(&mut self) -> Option<SignalChange> {
        self.gap();
        self.rearm = false;
        std::mem::take(&mut self.told_silent).then_some(SignalChange::Restored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK: usize = 320; // 20 ms
    /// One 16-bit step.
    const STEP: f32 = 1.0 / 32_768.0;

    /// Feeds `samples` of `value` in 20 ms chunks; the changes it saw.
    fn feed(watch: &mut SignalWatch, value: f32, samples: u64) -> Vec<SignalChange> {
        let chunk = vec![value; CHUNK];
        let mut out = Vec::new();
        let mut fed = 0;
        while fed < samples {
            out.extend(watch.observe(&chunk));
            fed += CHUNK as u64;
        }
        out
    }

    /// Feeds `samples` of 16-bit noise in 20 ms chunks: every sample a whole number of steps, drawn
    /// from `-peak..=peak` with `zero_in` out of every `zero_in + 1` draws at zero (a noise floor that
    /// is mostly below one step, as at a low Windows input level).
    fn feed_noise(watch: &mut SignalWatch, peak: i32, zero_in: u32, samples: u64) -> Vec<SignalChange> {
        let mut state = 0x2545_F491_u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state >> 8
        };
        let mut out = Vec::new();
        let mut fed = 0;
        while fed < samples {
            let chunk: Vec<f32> = (0..CHUNK)
                .map(|_| {
                    if next() % (zero_in + 1) != 0 {
                        0.0
                    } else {
                        (next() % (2 * peak as u32 + 1)) as i32 as f32 * STEP - peak as f32 * STEP
                    }
                })
                .collect();
            out.extend(watch.observe(&chunk));
            fed += CHUNK as u64;
        }
        out
    }

    /// A stream that delivered 200 ms of real signal.
    fn heard_watch() -> SignalWatch {
        let mut watch = SignalWatch::new();
        assert!(feed(&mut watch, 0.2, 2 * RESTORE_AFTER_SAMPLES).is_empty());
        watch
    }

    #[test]
    fn exact_zeros_are_reported_once_after_five_seconds() {
        let mut watch = SignalWatch::new();
        assert!(feed(&mut watch, 0.0, SILENCE_AFTER_SAMPLES - 2 * CHUNK as u64).is_empty(), "not yet");
        assert_eq!(feed(&mut watch, 0.0, 4 * CHUNK as u64), [SignalChange::Silent]);
        assert!(
            feed(&mut watch, 0.0, 10 * SILENCE_AFTER_SAMPLES).is_empty(),
            "reported once, not every chunk"
        );
        assert_eq!((watch.reports(), watch.threshold_secs()), (1, 5));
    }

    #[test]
    fn a_quiet_room_is_not_silence() {
        // -60 dBFS and -80 dBFS noise: far below speech, above a dead feed.
        for level in [1.0e-3_f32, 1.0e-4] {
            let mut watch = SignalWatch::new();
            let chunk: Vec<f32> = (0..CHUNK)
                .map(|i| if i % 2 == 0 { level } else { -level })
                .collect();
            let mut changes = Vec::new();
            for _ in 0..(60 * SAMPLE_RATE as usize / CHUNK) {
                changes.extend(watch.observe(&chunk));
            }
            assert!(changes.is_empty(), "noise at {level} must not read as no signal");
        }
    }

    #[test]
    fn sixteen_bit_noise_turned_far_down_is_not_silence() {
        // A USB mic whose noise floor is a few steps, at 10-20 % Windows input volume: the samples
        // are mostly 0 with an occasional +-1 or +-2. The old absolute test (peak within two steps)
        // read all of this as a dead feed five seconds after Start, in a quiet room, before anyone spoke.
        for (peak, zero_in) in [(1, 4), (1, 20), (2, 6), (3, 3)] {
            let mut watch = SignalWatch::new();
            let changes = feed_noise(&mut watch, peak, zero_in, 60 * SAMPLE_RATE as u64);
            assert!(changes.is_empty(), "noise of +-{peak} steps (1 in {} nonzero) is a live microphone", zero_in + 1);
        }
    }

    #[test]
    fn a_dither_of_half_a_step_is_noise_too() {
        // 24-bit interface at low gain: fractional steps (-95 dBFS-ish), not whole ones.
        let mut watch = SignalWatch::new();
        let chunk: Vec<f32> = (0..CHUNK).map(|i| if i % 2 == 0 { 0.4 * STEP } else { -0.4 * STEP }).collect();
        let mut changes = Vec::new();
        for _ in 0..(30 * SAMPLE_RATE as usize / CHUNK) {
            changes.extend(watch.observe(&chunk));
        }
        assert!(changes.is_empty(), "{changes:?}");
    }

    #[test]
    fn a_dead_converter_is_silence_exact_zeros_or_a_stuck_step() {
        for value in [0.0, STEP, -STEP] {
            let mut watch = SignalWatch::new();
            assert_eq!(feed(&mut watch, value, SILENCE_AFTER_SAMPLES), [SignalChange::Silent], "stuck at {value}");
        }
        // A resampler's rounding ripple on a stuck feed is not variation.
        let mut watch = SignalWatch::new();
        let chunk: Vec<f32> = (0..CHUNK).map(|i| 0.1 * STEP * (i % 3) as f32).collect();
        let mut changes = Vec::new();
        for _ in 0..(5 * SAMPLE_RATE as usize / CHUNK) {
            changes.extend(watch.observe(&chunk));
        }
        assert_eq!(changes, [SignalChange::Silent]);
    }

    #[test]
    fn one_click_restarts_the_count() {
        let mut watch = SignalWatch::new();
        assert!(feed(&mut watch, 0.0, SILENCE_AFTER_SAMPLES - CHUNK as u64).is_empty());
        let mut click = vec![0.0; CHUNK];
        click[100] = 0.05;
        assert_eq!(watch.observe(&click), None);
        // Another almost-five seconds: still not reported, the click restarted the count.
        assert!(feed(&mut watch, 0.0, SILENCE_AFTER_SAMPLES - CHUNK as u64).is_empty());
        // And five seconds are enough: a click is not a heard microphone, so no minute-long wait.
        assert_eq!(feed(&mut watch, 0.0, 2 * CHUNK as u64), [SignalChange::Silent]);
    }

    #[test]
    fn real_signal_clears_a_report_after_a_hundred_milliseconds_and_only_then() {
        let mut watch = SignalWatch::new();
        assert_eq!(feed(&mut watch, 0.0, SILENCE_AFTER_SAMPLES), [SignalChange::Silent]);
        // One loud chunk (20 ms) followed by silence again is a click, not a microphone.
        assert_eq!(watch.observe(&vec![0.2; CHUNK]), None);
        assert_eq!(watch.observe(&vec![0.0; CHUNK]), None);
        assert!(
            feed(&mut watch, 0.2, RESTORE_AFTER_SAMPLES - CHUNK as u64).is_empty(),
            "the click's 20 ms were forgotten"
        );
        assert_eq!(feed(&mut watch, 0.2, 2 * CHUNK as u64), [SignalChange::Restored]);
        assert!(feed(&mut watch, 0.2, SILENCE_AFTER_SAMPLES).is_empty(), "restored once");
        // And a later silence is a new report, after the long wait of a stream that was heard.
        assert!(feed(&mut watch, 0.0, SILENCE_AFTER_SIGNAL_SAMPLES - CHUNK as u64).is_empty());
        assert_eq!(feed(&mut watch, 0.0, 2 * CHUNK as u64), [SignalChange::Silent]);
        assert_eq!(watch.reports(), 2);
    }

    #[test]
    fn a_click_inside_a_reported_silence_does_not_report_it_again() {
        let mut watch = SignalWatch::new();
        assert_eq!(feed(&mut watch, 0.0, SILENCE_AFTER_SAMPLES), [SignalChange::Silent]);
        assert_eq!(watch.observe(&vec![0.2; CHUNK]), None);
        assert!(
            feed(&mut watch, 0.0, 3 * SILENCE_AFTER_SAMPLES).is_empty(),
            "the UI still shows the first report"
        );
    }

    #[test]
    fn a_noise_gate_that_closes_between_phrases_is_not_a_silent_microphone() {
        // NVIDIA Broadcast, Krisp, a headset's gate: exact zeros between phrases.
        let mut watch = heard_watch();
        assert_eq!(watch.threshold_secs(), 60);
        for pause_secs in [6_u64, 20, 45, 59] {
            assert!(
                feed(&mut watch, 0.0, pause_secs * SAMPLE_RATE as u64).is_empty(),
                "a {pause_secs} s pause is not a broken microphone"
            );
            // The next phrase: nothing to clear, nothing to announce.
            assert!(feed(&mut watch, 0.3, 2 * RESTORE_AFTER_SAMPLES).is_empty());
        }
        assert_eq!(watch.reports(), 0);
    }

    #[test]
    fn a_headset_muted_during_the_session_is_still_caught_after_a_minute() {
        let mut watch = heard_watch();
        assert!(feed(&mut watch, 0.0, SILENCE_AFTER_SIGNAL_SAMPLES - CHUNK as u64).is_empty());
        assert_eq!(feed(&mut watch, 0.0, 2 * CHUNK as u64), [SignalChange::Silent]);
        assert_eq!(feed(&mut watch, 0.3, 2 * RESTORE_AFTER_SAMPLES), [SignalChange::Restored]);
    }

    #[test]
    fn a_stream_restart_forgets_what_the_old_device_delivered_but_a_gap_does_not() {
        // Windows moved the default to another device: that one has to prove itself.
        let mut moved = heard_watch();
        moved.restarted();
        assert_eq!(moved.threshold_secs(), 5);
        assert_eq!(feed(&mut moved, 0.0, SILENCE_AFTER_SAMPLES), [SignalChange::Silent]);

        // An overrun dropped some audio on the same device: the gate behaviour it showed still holds.
        let mut overrun = heard_watch();
        overrun.gap();
        assert_eq!(overrun.threshold_secs(), 60);
        assert!(feed(&mut overrun, 0.0, 30 * SAMPLE_RATE as u64).is_empty());
    }

    #[test]
    fn a_gap_without_a_new_device_neither_clears_nor_repeats_a_report() {
        let mut watch = SignalWatch::new();
        assert_eq!(feed(&mut watch, 0.0, SILENCE_AFTER_SAMPLES), [SignalChange::Silent]);
        watch.gap();
        assert!(feed(&mut watch, 0.0, 3 * SILENCE_AFTER_SAMPLES).is_empty(), "same device: reported once");
        assert_eq!(feed(&mut watch, 0.3, 2 * RESTORE_AFTER_SAMPLES), [SignalChange::Restored]);
    }

    #[test]
    fn a_device_move_reports_a_lasting_silence_again_and_a_signal_on_the_new_device_clears_it() {
        let mut watch = SignalWatch::new();
        assert_eq!(feed(&mut watch, 0.0, SILENCE_AFTER_SAMPLES), [SignalChange::Silent]);
        watch.restarted();
        assert!(
            feed(&mut watch, 0.0, SILENCE_AFTER_SAMPLES - CHUNK as u64).is_empty(),
            "the count starts afresh"
        );
        assert_eq!(feed(&mut watch, 0.0, 2 * CHUNK as u64), [SignalChange::Silent], "names the new device");
        // A restart before any report changes nothing about the report to come.
        let mut fresh = SignalWatch::new();
        feed(&mut fresh, 0.0, SILENCE_AFTER_SAMPLES / 2);
        fresh.restarted();
        assert!(feed(&mut fresh, 0.0, SILENCE_AFTER_SAMPLES / 2).is_empty());
        // Signal on the new device clears the old report.
        watch.restarted();
        assert_eq!(feed(&mut watch, 0.3, RESTORE_AFTER_SAMPLES + CHUNK as u64), [SignalChange::Restored]);
    }

    #[test]
    fn a_mute_ends_a_report_and_the_count_starts_afresh_after_it() {
        let mut watch = SignalWatch::new();
        assert_eq!(watch.muted(), None, "nothing was reported");
        assert_eq!(feed(&mut watch, 0.0, SILENCE_AFTER_SAMPLES), [SignalChange::Silent]);
        assert_eq!(watch.muted(), Some(SignalChange::Restored));
        assert_eq!(watch.muted(), None, "once");
        assert!(feed(&mut watch, 0.0, SILENCE_AFTER_SAMPLES - CHUNK as u64).is_empty());
        assert_eq!(feed(&mut watch, 0.0, 2 * CHUNK as u64), [SignalChange::Silent]);
    }

    #[test]
    fn empty_reads_and_garbage_are_harmless() {
        let mut watch = SignalWatch::new();
        assert_eq!(watch.observe(&[]), None);
        // NaN is not silence (a broken driver is not a muted microphone): no report.
        assert!(feed(&mut watch, f32::NAN, 10 * SILENCE_AFTER_SAMPLES).is_empty());
    }
}
