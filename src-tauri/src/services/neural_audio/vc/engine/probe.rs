//! Benchmark hooks of the template engine (P3B; the fidelity scorer v2 and its oracle ablations).
//!
//! Every hook is off by default, so a shipping run never pays for it (each call site first checks
//! [`EngineProbe::active`], which is a field read) and the engine's output is bit-identical with
//! the probe off. Only a benchmark turns hooks on ([`EngineProbe::configure`], reached through
//! `SigilRvcEngine::probe_mut` before streaming):
//! - oracle F0 and oracle content overrides keyed by absolute frame (the F0 on the 10 ms grid, the
//!   content features on the stream's 320-sample grid), replacing what the engine computed
//!   ([`OracleF0`], [`OracleContent`]);
//! - a fed-F0 tap: the `pitchf` the generator decoded every returned frame with, on the absolute
//!   10 ms grid ([`ModelTelemetry::fed_f0`]); a frame returned by two calls (the SOLA tail) keeps the
//!   later call's value, and frames of a call skipped as silence read 0;
//! - the stage record of every call ([`ModelTelemetry::last_stages`], [`EngineProbe::calls`]).
//!
//! A stream reset ([`EngineProbe::reset_stream`]) starts a new recording: the harness warms a voice
//! up at the stream's first positions and resets it before streaming, so the warm-up's ticks never
//! reach the recording. The overrides stay keyed to absolute frames.
//!
//! # Frozen by P3B-S0 (signatures and call sites; P3B-B implements)
//! The engine calls, on every tick path (`tick.rs`; P3B-D keeps them on the overlapped path too):
//! - [`EngineProbe::override_content`]`(pos_16k, feats, frames50, emb) -> bool` right after the
//!   content encoder (before retrieval and protect's copy);
//! - [`EngineProbe::override_f0`]`(abs_start_frame, f0) -> bool` on the context's F0 (Hz, before
//!   the pitch shift);
//! - [`EngineProbe::tap_fed_f0`]`(abs_start_frame, hz)` with the shifted F0 of the returned frames,
//!   and [`EngineProbe::tap_unvoiced`]`(abs_start_frame, frames)` for a tick skipped as silence;
//! - [`EngineProbe::record_stages`]`(&StageRecord)` once per call (`convert_inner`);
//! - [`EngineProbe::reset_stream`]`()` on a stream reset.
//!
//! # Added by P3B-B
//! [`ProbeSetup`], [`OracleF0`], [`OracleContent`], [`EngineProbe::configure`],
//! [`EngineProbe::disable`], [`EngineProbe::setup`], [`EngineProbe::calls`],
//! [`EngineProbe::applied`] ([`ProbeCounts`]).

use std::sync::Arc;

use super::super::traits::{FedF0, ModelTelemetry, StageRecord};

/// 16 kHz samples per content frame (50 Hz).
const CONTENT_HOP_16K: u64 = 320;
/// Longest fed-F0 recording (frames): one hour; a longer stream keeps its first hour.
pub const MAX_RECORDED_FRAMES: usize = 360_000;

/// An oracle F0 contour (Hz before the pitch shift, 0 = unvoiced) on the absolute 10 ms grid:
/// `hz[i]` is absolute frame `start_frame + i` (`stream_pos_16k / 160` units).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OracleF0 {
    pub start_frame: u64,
    pub hz: Vec<f32>,
}

impl OracleF0 {
    /// The oracle's value at absolute frame `frame`, if it covers it.
    pub fn at(&self, frame: u64) -> Option<f32> {
        frame
            .checked_sub(self.start_frame)
            .and_then(|i| usize::try_from(i).ok())
            .and_then(|i| self.hz.get(i).copied())
    }
}

/// Oracle content features (`frames × emb`, row-major, 50 Hz): frame `k` is the encoder's frame of
/// the absolute 16 kHz samples from `origin_16k + 320·k` (its 400-sample receptive field). The
/// origin must sit on the stream's 320-sample content grid (the engine's contexts start on it).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OracleContent {
    pub origin_16k: u64,
    pub emb: usize,
    pub feats: Vec<f32>,
}

impl OracleContent {
    pub fn frames(&self) -> usize {
        self.feats.len() / self.emb.max(1)
    }

    /// The oracle frame starting at absolute sample `pos_16k`, if it has one there.
    fn frame_at(&self, pos_16k: u64) -> Option<&[f32]> {
        let off = pos_16k.checked_sub(self.origin_16k)?;
        if off % CONTENT_HOP_16K != 0 {
            return None;
        }
        let k = usize::try_from(off / CONTENT_HOP_16K).ok()?;
        (k < self.frames()).then(|| &self.feats[k * self.emb..(k + 1) * self.emb])
    }
}

/// What a benchmark asks the probe to do (all off by default).
#[derive(Clone, Debug, Default)]
pub struct ProbeSetup {
    /// Record the F0 fed to the generator ([`ModelTelemetry::fed_f0`]).
    pub fed_f0: bool,
    /// Keep every call's stage record ([`EngineProbe::calls`]); the last one is kept whenever the
    /// probe is active.
    pub stages: bool,
    /// Replace the context's F0 with this contour where it has frames (ablations).
    pub oracle_f0: Option<Arc<OracleF0>>,
    /// Replace the content features with these where they have frames (ablations).
    pub oracle_content: Option<Arc<OracleContent>>,
}

impl ProbeSetup {
    fn any(&self) -> bool {
        self.fed_f0 || self.stages || self.oracle_f0.is_some() || self.oracle_content.is_some()
    }
}

/// How often the overrides applied since the probe was configured.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProbeCounts {
    /// Calls whose content features the oracle replaced (wholly or in part).
    pub content_calls: u64,
    /// Content frames replaced.
    pub content_frames: u64,
    /// Calls whose F0 the oracle replaced (wholly or in part).
    pub f0_calls: u64,
    /// F0 frames replaced.
    pub f0_frames: u64,
    /// Calls whose content grid did not line up with the oracle's origin (nothing replaced).
    pub content_misaligned: u64,
}

/// The engine's benchmark hooks (all off; see the module docs).
#[derive(Debug, Default)]
pub struct EngineProbe {
    /// Any hook is on (the tick path checks this first).
    active: bool,
    setup: ProbeSetup,
    /// Fed F0 (Hz after the shift) from absolute frame `fed_start`.
    fed_start: u64,
    fed: Vec<f32>,
    /// Frames dropped beyond [`MAX_RECORDED_FRAMES`].
    fed_dropped: u64,
    last: Option<StageRecord>,
    calls: Vec<StageRecord>,
    counts: ProbeCounts,
}

impl EngineProbe {
    /// Whether any hook is on.
    #[inline]
    pub fn active(&self) -> bool {
        self.active
    }

    /// Turn hooks on (or off) for the next calls; the recording starts empty.
    pub fn configure(&mut self, setup: ProbeSetup) {
        self.active = setup.any();
        self.setup = setup;
        self.clear_recording();
        self.counts = ProbeCounts::default();
    }

    /// Every hook off (the shipping state); the recording is dropped.
    pub fn disable(&mut self) {
        self.configure(ProbeSetup::default());
    }

    /// The hooks in effect.
    pub fn setup(&self) -> &ProbeSetup {
        &self.setup
    }

    /// Every call's stage record since the last reset (when `ProbeSetup::stages`).
    pub fn calls(&self) -> &[StageRecord] {
        &self.calls
    }

    /// How often the overrides applied.
    pub fn applied(&self) -> ProbeCounts {
        self.counts
    }

    /// Frames the fed-F0 recording dropped beyond its hour.
    pub fn fed_dropped_frames(&self) -> u64 {
        self.fed_dropped
    }

    fn clear_recording(&mut self) {
        self.fed.clear();
        self.fed_start = 0;
        self.fed_dropped = 0;
        self.last = None;
        self.calls.clear();
    }

    /// Replace the content features of this tick (`frames50 × emb`, 50 Hz frame `k` covering the
    /// absolute 16 kHz samples from `pos_16k + 320k`) with the oracle's; `true` when applied.
    #[inline]
    pub fn override_content(&mut self, pos_16k: u64, feats: &mut [f32], frames50: usize, emb: usize) -> bool {
        let Some(oracle) = self.setup.oracle_content.as_ref() else {
            return false;
        };
        if oracle.emb != emb || emb == 0 {
            return false;
        }
        if pos_16k.abs_diff(oracle.origin_16k) % CONTENT_HOP_16K != 0 {
            self.counts.content_misaligned += 1;
            return false;
        }
        let mut replaced = 0u64;
        for (k, dst) in feats.chunks_exact_mut(emb).take(frames50).enumerate() {
            if let Some(src) = oracle.frame_at(pos_16k + CONTENT_HOP_16K * k as u64) {
                dst.copy_from_slice(src);
                replaced += 1;
            }
        }
        if replaced > 0 {
            self.counts.content_calls += 1;
            self.counts.content_frames += replaced;
        }
        replaced > 0
    }

    /// Replace the context's F0 (Hz before the shift; `f0[i]` = absolute frame
    /// `abs_start_frame + i`) with the oracle's where it has frames; `true` when applied.
    #[inline]
    pub fn override_f0(&mut self, abs_start_frame: u64, f0: &mut [f32]) -> bool {
        let Some(oracle) = self.setup.oracle_f0.as_ref() else {
            return false;
        };
        let mut replaced = 0u64;
        for (i, v) in f0.iter_mut().enumerate() {
            if let Some(hz) = oracle.at(abs_start_frame + i as u64) {
                *v = if hz.is_finite() && hz > 0.0 { hz } else { 0.0 };
                replaced += 1;
            }
        }
        if replaced > 0 {
            self.counts.f0_calls += 1;
            self.counts.f0_frames += replaced;
        }
        replaced > 0
    }

    /// Record the F0 fed to the generator for the returned frames (`hz[i]` = absolute frame
    /// `abs_start_frame + i`, after the pitch shift).
    #[inline]
    pub fn tap_fed_f0(&mut self, abs_start_frame: u64, hz: &[f32]) {
        if self.setup.fed_f0 {
            self.write_fed(abs_start_frame, hz.len(), |i| hz[i]);
        }
    }

    /// Record `frames` unvoiced frames from `abs_start_frame` (a tick skipped as silence).
    #[inline]
    pub fn tap_unvoiced(&mut self, abs_start_frame: u64, frames: usize) {
        if self.setup.fed_f0 {
            self.write_fed(abs_start_frame, frames, |_| 0.0);
        }
    }

    /// Write `n` frames from absolute frame `start` (later calls overwrite; gaps read 0).
    fn write_fed(&mut self, start: u64, n: usize, value: impl Fn(usize) -> f32) {
        if n == 0 {
            return;
        }
        if self.fed.is_empty() {
            self.fed_start = start;
        } else if start < self.fed_start {
            // Earlier than anything recorded (not expected from the harness): grow the front.
            let grow = (self.fed_start - start) as usize;
            if self.fed.len() + grow > MAX_RECORDED_FRAMES {
                self.fed_dropped += n as u64;
                return;
            }
            self.fed.splice(0..0, std::iter::repeat_n(0.0, grow));
            self.fed_start = start;
        }
        let off = (start - self.fed_start) as usize;
        let end = (off + n).min(MAX_RECORDED_FRAMES);
        if end < off + n {
            self.fed_dropped += (off + n - end) as u64;
        }
        if end <= off {
            return;
        }
        if self.fed.len() < end {
            self.fed.resize(end, 0.0);
        }
        for (i, slot) in self.fed[off..end].iter_mut().enumerate() {
            let v = value(i);
            *slot = if v.is_finite() && v > 0.0 { v } else { 0.0 };
        }
    }

    /// The stage record of the call that just finished.
    #[inline]
    pub fn record_stages(&mut self, record: &StageRecord) {
        self.last = Some(*record);
        if self.setup.stages && self.calls.len() < MAX_RECORDED_FRAMES {
            self.calls.push(*record);
        }
    }

    /// The stream restarted: a new recording starts (the warm-up's ticks are dropped); the
    /// overrides stay keyed to absolute frames.
    #[inline]
    pub fn reset_stream(&mut self) {
        if self.active {
            self.clear_recording();
        }
    }
}

impl ModelTelemetry for EngineProbe {
    fn fed_f0(&self) -> Option<FedF0<'_>> {
        (self.setup.fed_f0 && !self.fed.is_empty()).then(|| FedF0 {
            start_frame: self.fed_start,
            hz: &self.fed,
        })
    }

    fn last_stages(&self) -> Option<StageRecord> {
        if self.active {
            self.last
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{levers_all, tiny_engine, tiny_slot, voice_16k};
    use super::*;
    use crate::services::neural_audio::vc::traits::{ConversionCtx, VoiceConversionModel};
    use crate::services::neural_audio::VcStreamHarness;

    #[test]
    fn an_inert_probe_records_nothing_and_reports_nothing() {
        let mut p = EngineProbe::default();
        assert!(!p.active());
        let mut f0 = [100.0f32; 4];
        assert!(!p.override_f0(10, &mut f0));
        let mut feats = [1.0f32; 8];
        assert!(!p.override_content(0, &mut feats, 2, 4));
        p.tap_fed_f0(10, &[120.0, 130.0]);
        p.tap_unvoiced(12, 3);
        p.record_stages(&StageRecord::default());
        assert!(p.fed_f0().is_none() && p.last_stages().is_none() && p.calls().is_empty());
        assert_eq!((f0, feats), ([100.0; 4], [1.0; 8]));
    }

    #[test]
    fn the_fed_track_is_absolute_and_later_calls_win() {
        let mut p = EngineProbe::default();
        p.configure(ProbeSetup {
            fed_f0: true,
            stages: true,
            ..ProbeSetup::default()
        });
        assert!(p.active());
        p.tap_fed_f0(100, &[1.0, 2.0, 3.0, f32::NAN]);
        // The next call returns frames 102.. (a SOLA tail of two frames): its values win.
        p.tap_fed_f0(102, &[30.0, 40.0, 50.0]);
        // A call skipped as silence.
        p.tap_unvoiced(105, 2);
        // A gap (no call returned frame 107) reads 0; then frame 108.
        p.tap_fed_f0(108, &[80.0]);
        let f = p.fed_f0().unwrap();
        assert_eq!(f.start_frame, 100);
        assert_eq!(f.hz, &[1.0, 2.0, 30.0, 40.0, 50.0, 0.0, 0.0, 0.0, 80.0]);
        // An earlier frame grows the front.
        p.tap_fed_f0(98, &[9.0]);
        let f = p.fed_f0().unwrap();
        assert_eq!((f.start_frame, f.hz[0], f.hz[1], f.hz[2]), (98, 9.0, 0.0, 1.0));
        p.record_stages(&StageRecord {
            stream_pos_16k: 320,
            ..StageRecord::default()
        });
        assert_eq!(p.last_stages().unwrap().stream_pos_16k, 320);
        assert_eq!(p.calls().len(), 1);
        // A stream reset starts a new recording.
        p.reset_stream();
        assert!(p.fed_f0().is_none() && p.calls().is_empty() && p.last_stages().is_none());
        p.disable();
        assert!(!p.active());
    }

    #[test]
    fn overrides_apply_on_their_absolute_grids_only() {
        let mut p = EngineProbe::default();
        p.configure(ProbeSetup {
            oracle_f0: Some(Arc::new(OracleF0 {
                start_frame: 50,
                hz: vec![200.0, 0.0, 210.0, f32::NAN],
            })),
            oracle_content: Some(Arc::new(OracleContent {
                origin_16k: 640,
                emb: 2,
                feats: vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            })),
            ..ProbeSetup::default()
        });
        // F0: frames 48..54; the oracle covers 50..53 (NaN -> unvoiced).
        let mut f0 = [7.0f32; 6];
        assert!(p.override_f0(48, &mut f0));
        assert_eq!(f0, [7.0, 7.0, 200.0, 0.0, 210.0, 0.0]);
        // Content: a window from 320 covers oracle frames from 640 = its frame 1.
        let mut feats = [9.0f32; 8];
        assert!(p.override_content(320, &mut feats, 4, 2));
        assert_eq!(feats, [9.0, 9.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        // Off the 320-sample grid: nothing replaced, counted as misaligned.
        let mut off = [9.0f32; 4];
        assert!(!p.override_content(160, &mut off, 2, 2));
        assert_eq!(off, [9.0; 4]);
        // Another feature width: not this oracle's.
        assert!(!p.override_content(640, &mut [0.0; 6], 2, 3));
        let c = p.applied();
        assert_eq!((c.f0_calls, c.f0_frames, c.content_calls, c.content_frames, c.content_misaligned), (1, 4, 1, 3, 1));
        // Nothing to override outside the oracles.
        assert!(!p.override_f0(1_000, &mut [1.0; 3]));
        assert!(!p.override_content(64_000, &mut [0.0; 4], 2, 2));
    }

    /// The fixture voice through the production harness: with the fed-F0 tap on, the audio is
    /// bit-identical to the probe off (the tap only records), and the tap covers the stream on the
    /// absolute 10 ms grid with the pitch shift applied.
    #[test]
    fn the_tap_records_without_changing_the_audio() {
        let s = tiny_slot("probe-tap");
        let render = |tap: bool| -> (Vec<f32>, Option<(u64, Vec<f32>)>) {
            let mut engine = tiny_engine(&s, levers_all(true));
            if tap {
                engine.probe_mut().configure(ProbeSetup {
                    fed_f0: true,
                    stages: true,
                    ..ProbeSetup::default()
                });
            }
            let shared = std::sync::Arc::new(std::sync::Mutex::new(engine));
            let mut h = VcStreamHarness::new(Box::new(Shared(shared.clone())), 48_000, 48_000, 1, 1, 1_600, 320, 1_920).unwrap();
            h.set_controls(12.0, 0.0, 0.33, 0);
            let input: Vec<f32> = voice_16k(16_000 * 2).iter().flat_map(|&x| [x, x, x]).collect();
            let mut out = Vec::new();
            for chunk in input.as_chunks::<1_920>().0 {
                out.extend(h.push_and_convert(chunk).unwrap());
            }
            let e = shared.lock().unwrap();
            let fed = e.telemetry().and_then(|t| t.fed_f0()).map(|f| (f.start_frame, f.hz.to_vec()));
            (out, fed)
        };
        let (plain, none) = render(false);
        let (tapped, fed) = render(true);
        assert!(none.is_none());
        assert_eq!(plain.len(), tapped.len());
        assert!(plain.iter().zip(&tapped).all(|(a, b)| a.to_bits() == b.to_bits()), "the tap changed the audio");
        let (start, hz) = fed.expect("the tap recorded the stream");
        // Two seconds of stream: about 200 frames recorded (minus the harness's priming).
        assert!(hz.len() >= 150, "{} frames from {start}", hz.len());
        let voiced: Vec<f32> = hz.iter().copied().filter(|&v| v > 0.0).collect();
        assert!(voiced.len() > hz.len() / 3, "{voiced:?}");
        // The test voice glides 100-180 Hz; an octave up (+12 st) it is fed at 200-360 Hz (DIO).
        let mut sorted = voiced.clone();
        sorted.sort_by(f32::total_cmp);
        let median = sorted[sorted.len() / 2];
        assert!((180.0..400.0).contains(&median), "median fed F0 {median} Hz");
    }

    /// The oracle F0 reaches the generator: a constant oracle contour is what the tap records
    /// (shifted), and the audio changes.
    #[test]
    fn an_oracle_f0_is_what_the_generator_is_fed() {
        let s = tiny_slot("probe-oracle");
        let audio = voice_16k(16_000 * 2);
        let run = |oracle: Option<f32>| -> (Vec<f32>, Vec<f32>) {
            let mut e = tiny_engine(&s, levers_all(true));
            e.set_silence_skip(false);
            e.probe_mut().configure(ProbeSetup {
                fed_f0: true,
                oracle_f0: oracle.map(|hz| {
                    Arc::new(OracleF0 {
                        start_frame: 0,
                        hz: vec![hz; 400],
                    })
                }),
                ..ProbeSetup::default()
            });
            let p = *e.plan();
            let mut out = Vec::new();
            let mut pos = 0usize;
            while pos + p.context_16k <= audio.len() {
                let ctx = ConversionCtx {
                    mono_16k: &audio[pos..pos + p.context_16k],
                    new_frames: p.want_16k,
                    pitch_shift_semitones: 0.0,
                    index_rate: 0.0,
                    protect: 0.5,
                    speaker_id: 0,
                    stream_pos_16k: pos as u64,
                    reference: None,
                    singing: Default::default(),
                };
                out.extend(e.convert(&ctx).unwrap());
                pos += p.block_16k;
            }
            let fed = e.telemetry().and_then(|t| t.fed_f0()).map(|f| f.hz.to_vec()).unwrap_or_default();
            (out, fed)
        };
        let (own, own_fed) = run(None);
        let (oracle, oracle_fed) = run(Some(150.0));
        assert!(!own_fed.is_empty() && !oracle_fed.is_empty());
        assert!(oracle_fed.iter().all(|&v| v == 150.0), "{oracle_fed:?}");
        assert!(own_fed.iter().any(|&v| v != 150.0));
        assert_ne!(own, oracle, "the oracle F0 must change the audio");
    }

    /// The oracle content reaches the generator at the tick level, which the ablation rows trust
    /// (their `applied` counters are the only evidence): an oracle on the stream's 320-sample grid
    /// replaces frames, is counted and changes the audio; one that misses the grid replaces nothing,
    /// is counted as misaligned and leaves the audio bit-identical. If a later engine change moves or
    /// guards the `override_content` call, this fails instead of the ablations quietly reading
    /// "recovered +0.00".
    #[test]
    fn an_oracle_content_changes_the_audio_only_on_the_streams_grid() {
        let s = tiny_slot("probe-oracle-content");
        let audio = voice_16k(16_000 * 2);
        let emb = crate::services::neural_audio::vc::engine::generator::tests::tiny_voice().template.layout.arch.emb_channels as usize;
        let run = |origin: Option<u64>| -> (Vec<f32>, ProbeCounts) {
            let mut e = tiny_engine(&s, levers_all(true));
            e.set_silence_skip(false);
            e.probe_mut().configure(ProbeSetup {
                stages: true,
                oracle_content: origin.map(|origin_16k| {
                    Arc::new(OracleContent {
                        origin_16k,
                        emb,
                        feats: (0..emb * 150).map(|i| 0.4 * ((i % 17) as f32 / 17.0 - 0.5)).collect(),
                    })
                }),
                ..ProbeSetup::default()
            });
            let p = *e.plan();
            let mut out = Vec::new();
            let mut pos = 0usize;
            while pos + p.context_16k <= audio.len() {
                let ctx = ConversionCtx {
                    mono_16k: &audio[pos..pos + p.context_16k],
                    new_frames: p.want_16k,
                    pitch_shift_semitones: 0.0,
                    index_rate: 0.0,
                    protect: 0.5,
                    speaker_id: 0,
                    stream_pos_16k: pos as u64,
                    reference: None,
                    singing: Default::default(),
                };
                out.extend(e.convert(&ctx).unwrap());
                pos += p.block_16k;
            }
            assert_eq!(p.block_16k % 320, 0, "the stream's block keeps the 320-sample grid");
            (out, e.probe().applied())
        };
        let (plain, none) = run(None);
        assert_eq!(none, ProbeCounts::default(), "no oracle, nothing counted");
        let (on_grid, counts) = run(Some(0));
        assert!(counts.content_calls > 0 && counts.content_frames > 0, "{counts:?}");
        assert_eq!(counts.content_misaligned, 0, "{counts:?}");
        assert_eq!(plain.len(), on_grid.len());
        assert!(plain.iter().zip(&on_grid).any(|(a, b)| a.to_bits() != b.to_bits()), "the oracle content must change the audio");
        // Origin 160: every call's grid misses it; nothing replaced, nothing changed.
        let (off_grid, missed) = run(Some(160));
        assert_eq!((missed.content_calls, missed.content_frames), (0, 0));
        assert!(missed.content_misaligned > 0, "{missed:?}");
        assert!(plain.iter().zip(&off_grid).all(|(a, b)| a.to_bits() == b.to_bits()), "a missed oracle must not touch the audio");
    }

    /// A model handle over a shared engine (the harness owns its model; the test reads the probe
    /// back afterwards).
    struct Shared(std::sync::Arc<std::sync::Mutex<super::super::SigilRvcEngine>>);

    impl VoiceConversionModel for Shared {
        fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
            self.0.lock().unwrap().convert(ctx)
        }
        fn input_sample_rate(&self) -> u32 {
            self.0.lock().unwrap().input_sample_rate()
        }
        fn output_sample_rate(&self) -> u32 {
            self.0.lock().unwrap().output_sample_rate()
        }
        fn min_context_samples(&self) -> usize {
            self.0.lock().unwrap().min_context_samples()
        }
        fn warmup_at(&mut self, ctx: &ConversionCtx<'_>) -> Result<(), String> {
            self.0.lock().unwrap().warmup_at(ctx)
        }
        fn reset_stream(&mut self) {
            self.0.lock().unwrap().reset_stream();
        }
        fn fixed_block(&self) -> Option<usize> {
            self.0.lock().unwrap().fixed_block()
        }
        fn configure_stream(&mut self, block: usize, want: usize, extra: usize) -> Result<usize, String> {
            self.0.lock().unwrap().configure_stream(block, want, extra)
        }
    }
}
