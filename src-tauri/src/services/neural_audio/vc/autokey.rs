//! Auto-key (P3-D): suggests a transpose (semitones) that moves the speaker's voiced F0 median
//! into the voice's range. A suggestion only; the user applies it (never changed silently).
//!
//! The speaker's F0 goes into a 10-cent log-frequency histogram (40 Hz - 2.56 kHz) with a
//! [`HALF_LIFE_FRAMES`] forgetting half-life, so the suggestion follows whoever is speaking now.
//! Allocation-free after construction.
//!
//! # Frozen by P3-S0 (signatures)
//! [`AutoKey::new`], [`AutoKey::push`] (voiced frames of one tick), [`AutoKey::suggestion`] (`None`
//! until enough voiced audio was heard), [`AutoKey::reset`].
//!
//! # Added by P3-D
//! [`AutoKey::median_hz`], [`AutoKey::voiced_frames`], [`MIN_VOICED_FRAMES`],
//! [`MAX_SUGGESTION_SEMITONES`], [`HALF_LIFE_FRAMES`].

// No production caller yet: the worker feeds it the running F0 and the voice changer inspector shows
// the suggestion (requested from P3-E / P3-G through INTEGRATE); the tests exercise everything.
#![allow(dead_code)]

/// Voiced 10 ms frames needed before a suggestion (3 s of speech).
pub const MIN_VOICED_FRAMES: usize = 300;
/// Largest suggested transpose.
pub const MAX_SUGGESTION_SEMITONES: f32 = 24.0;
/// Old frames count half after this many new voiced frames (60 s of speech).
pub const HALF_LIFE_FRAMES: f32 = 6_000.0;

const LOW_HZ: f32 = 40.0;
const OCTAVES: usize = 6;
const BIN_CENTS: usize = 10;
const BINS: usize = OCTAVES * 1200 / BIN_CENTS;

pub struct AutoKey {
    voiced_frames: usize,
    bins: Vec<f32>,
    /// Weight of the next frame (grows instead of decaying every bin).
    increment: f32,
    total: f32,
}

impl Default for AutoKey {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoKey {
    pub fn new() -> AutoKey {
        AutoKey {
            voiced_frames: 0,
            bins: vec![0.0; BINS],
            increment: 1.0,
            total: 0.0,
        }
    }

    /// Add one tick's F0 frames (Hz, 0 = unvoiced).
    pub fn push(&mut self, f0_hz: &[f32]) {
        let growth = (std::f32::consts::LN_2 / HALF_LIFE_FRAMES).exp();
        for &hz in f0_hz {
            if !(hz.is_finite() && (50.0..=2_500.0).contains(&hz)) {
                continue;
            }
            let bin = ((1200.0 * (hz / LOW_HZ).log2()) / BIN_CENTS as f32) as usize;
            if bin >= BINS {
                continue;
            }
            self.bins[bin] += self.increment;
            self.total += self.increment;
            self.increment *= growth;
            self.voiced_frames += 1;
            if self.increment > 1e12 {
                let scale = 1.0 / self.increment;
                for b in &mut self.bins {
                    *b *= scale;
                }
                self.total *= scale;
                self.increment = 1.0;
            }
        }
    }

    /// Voiced frames heard since the last reset.
    pub fn voiced_frames(&self) -> usize {
        self.voiced_frames
    }

    /// The speaker's weighted median F0 (Hz), once enough voiced audio was heard.
    pub fn median_hz(&self) -> Option<f32> {
        if self.voiced_frames < MIN_VOICED_FRAMES || self.total <= 0.0 {
            return None;
        }
        let half = 0.5 * self.total;
        let mut acc = 0.0f32;
        for (i, &b) in self.bins.iter().enumerate() {
            acc += b;
            if acc >= half {
                let cents = (i as f32 + 0.5) * BIN_CENTS as f32;
                return Some(LOW_HZ * (cents / 1200.0).exp2());
            }
        }
        None
    }

    /// Suggested transpose toward `target_median_hz` (the voice's typical F0), rounded to whole
    /// semitones.
    pub fn suggestion(&self, target_median_hz: f32) -> Option<f32> {
        if !(target_median_hz.is_finite() && target_median_hz > 0.0) {
            return None;
        }
        let m = self.median_hz()?;
        let st = (12.0 * (target_median_hz / m).log2()).round();
        Some(st.clamp(-MAX_SUGGESTION_SEMITONES, MAX_SUGGESTION_SEMITONES) + 0.0)
    }

    pub fn reset(&mut self) {
        self.voiced_frames = 0;
        self.bins.fill(0.0);
        self.increment = 1.0;
        self.total = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suggests_whole_semitones_after_enough_speech() {
        let mut k = AutoKey::new();
        // A 120 Hz speaker with some spread and unvoiced frames.
        let tick: Vec<f32> = (0..10)
            .map(|i| if i % 3 == 0 { 0.0 } else { 120.0 * ((i as f32 - 5.0) * 0.01).exp2() })
            .collect();
        for _ in 0..40 {
            k.push(&tick);
        }
        assert!(k.suggestion(220.0).is_none(), "not enough yet: {}", k.voiced_frames());
        for _ in 0..40 {
            k.push(&tick);
        }
        let m = k.median_hz().unwrap();
        assert!((1200.0 * (m / 120.0).log2()).abs() < 15.0, "{m}");
        // 120 -> 220 Hz is 10.49 semitones -> 10.
        assert_eq!(k.suggestion(220.0), Some(10.0));
        assert_eq!(k.suggestion(120.0), Some(0.0));
        assert_eq!(k.suggestion(60.0), Some(-12.0));
        // Clamped, and junk targets give nothing.
        assert_eq!(k.suggestion(120.0 * 64.0), Some(MAX_SUGGESTION_SEMITONES));
        assert_eq!(k.suggestion(0.0), None);
        assert_eq!(k.suggestion(f32::NAN), None);
        k.reset();
        assert!(k.median_hz().is_none());
    }

    #[test]
    fn follows_a_new_speaker_and_ignores_junk() {
        let mut k = AutoKey::new();
        k.push(&[f32::NAN, -5.0, 10.0, 9_000.0]);
        assert_eq!(k.voiced_frames(), 0);
        for _ in 0..(HALF_LIFE_FRAMES as usize / 100) {
            k.push(&[110.0; 100]);
        }
        assert!((k.median_hz().unwrap() - 110.0).abs() < 2.0);
        // Twice as much of a 250 Hz speaker afterwards: the median follows (recent frames weigh more).
        for _ in 0..(2 * HALF_LIFE_FRAMES as usize / 100) {
            k.push(&[250.0; 100]);
        }
        assert!((k.median_hz().unwrap() - 250.0).abs() < 4.0, "{:?}", k.median_hz());
        // Long runs rescale without overflow.
        for _ in 0..3_000 {
            k.push(&[250.0; 100]);
        }
        assert!(k.median_hz().unwrap().is_finite());
    }
}
