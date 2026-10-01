//! Cepstral-envelope formant shift (P3-D): warps the spectral envelope by ±12 semitones while the
//! harmonics (F0) stay put; plan acceptance: F0 preserved within 5 cents. It replaces the one-pole
//! tilt (`formant_shift`) on the RVC path; the classic / ONNX / sidecar modes keep the tilt. Driven
//! by `VoiceControls::formant_semitones` through the post stage (`engine::post`).
//!
//! # How
//! Short-time Fourier analysis / resynthesis (Hann windows of 20 ms, hop 5 ms, weighted overlap-add
//! normalised per sample). Per frame:
//! 1. the log-magnitude spectrum's **real cepstrum** is liftered to the spectral envelope `E(f)`: a
//!    flat-top lifter below [`DEFAULT_LIFTER_MS`], shortened to 0.6 of the pitch period when the
//!    cepstrum shows a clear pitch peak (so a high voice's harmonic ripple never leaks into the
//!    envelope);
//! 2. the envelope is warped, `E'(f) = E(f / a)` with `a = 2^(semitones / 12)`;
//! 3. every bin is scaled by `exp(E'(f) - E(f))` (bounded to +18 / -30 dB) and the frame keeps its
//!    energy (±6 dB), so the knob moves formants without changing loudness;
//! 4. the phase is untouched, so every harmonic stays at its frequency: the F0 is preserved.
//!
//! Each call processes one buffer on its own (the voice engine's tick, whose ends are reflected for
//! analysis), with the frame grid anchored to the **end** of the buffer: overlapping ticks whose
//! hop is a multiple of 10 ms analyse their shared audio with identical frames, so the SOLA seam
//! joins two matching versions of it. The shifter therefore adds no latency.
//!
//! # Frozen by P3-S0 (signatures)
//! [`FormantShifter::new`], [`FormantShifter::process`] (in place, allocation-free after
//! construction, 0 semitones = bypass), [`FormantShifter::latency_samples`],
//! [`FormantShifter::reset`].
//!
//! # Added by P3-D
//! [`FormantShifter::with_capacity`], [`FORMANT_SEMITONES_MAX`], [`DEFAULT_LIFTER_MS`].

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// Largest shift, in semitones (params `formant_semitones` range).
pub const FORMANT_SEMITONES_MAX: f32 = 12.0;
/// Envelope lifter cutoff (quefrency) when no pitch period shortens it.
pub const DEFAULT_LIFTER_MS: f32 = 1.5;
/// Shortest lifter (keeps a few formant peaks even for a 1 kHz voice).
const MIN_LIFTER_MS: f32 = 0.5;
/// Shifts smaller than this are a bypass (bit-exact).
const BYPASS_SEMITONES: f32 = 0.01;
/// Largest boost / cut of one bin (natural log).
const MAX_BOOST_LN: f32 = 2.072_327; // +18 dB
const MAX_CUT_LN: f32 = 3.453_878; // -30 dB
/// Per-frame energy correction bound (±6 dB in amplitude).
const MAX_ENERGY_SCALE: f32 = 2.0;
/// A cepstral peak this strong (natural-log units) marks a voiced frame whose period shortens the
/// lifter.
const PITCH_PEAK_MIN: f32 = 0.08;

/// Frame geometry for a sample rate: hop 5 ms, window 20 ms.
fn geometry(sample_rate: u32) -> (usize, usize) {
    let hop = ((sample_rate as f32 / 200.0).round() as usize).max(8);
    (hop * 4, hop)
}

/// torch `reflect` index (no edge repeat) for any offset.
fn reflect_index(i: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let mut j = i.rem_euclid(period);
    if j >= n as isize {
        j = period - j;
    }
    j as usize
}

pub struct FormantShifter {
    sample_rate: u32,
    n_fft: usize,
    hop: usize,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    // ---- scratch ----
    padded: Vec<f32>,
    out: Vec<f32>,
    wsum: Vec<f32>,
    frame: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    work: Vec<Complex<f32>>,
    cepstrum: Vec<f32>,
    envelope: Vec<f32>,
    warped: Vec<f32>,
    fwd_scratch: Vec<Complex<f32>>,
    inv_scratch: Vec<Complex<f32>>,
}

impl FormantShifter {
    #[allow(dead_code)] // Frozen P3-S0 API (the post stage pre-sizes with with_capacity).
    pub fn new(sample_rate: u32) -> FormantShifter {
        FormantShifter::with_capacity(sample_rate, 0)
    }

    /// Pre-size the scratch for buffers of up to `max_samples` (longer ones grow it once).
    pub fn with_capacity(sample_rate: u32, max_samples: usize) -> FormantShifter {
        let sample_rate = sample_rate.max(8_000);
        let (n_fft, hop) = geometry(sample_rate);
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(n_fft);
        let inv = planner.plan_fft_inverse(n_fft);
        // Periodic Hann: with a hop of a quarter window, sum(w^2) is constant (1.5).
        let window = (0..n_fft)
            .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n_fft as f64).cos()) as f32)
            .collect();
        let bins = n_fft / 2 + 1;
        let cap = max_samples + 2 * n_fft + hop;
        FormantShifter {
            sample_rate,
            n_fft,
            hop,
            frame: fwd.make_input_vec(),
            spectrum: fwd.make_output_vec(),
            work: fwd.make_output_vec(),
            fwd_scratch: fwd.make_scratch_vec(),
            inv_scratch: inv.make_scratch_vec(),
            cepstrum: vec![0.0; n_fft],
            envelope: vec![0.0; bins],
            warped: vec![0.0; bins],
            padded: Vec::with_capacity(cap),
            out: Vec::with_capacity(cap),
            wsum: Vec::with_capacity(cap),
            fwd,
            inv,
            window,
        }
    }

    /// Shift the envelope of `audio` by `semitones` (0 = bypass; clamped to ±12).
    pub fn process(&mut self, audio: &mut [f32], semitones: f32) -> Result<(), String> {
        if !semitones.is_finite() || semitones.abs() < BYPASS_SEMITONES || audio.len() < 2 {
            return Ok(());
        }
        let alpha = (semitones.clamp(-FORMANT_SEMITONES_MAX, FORMANT_SEMITONES_MAX) / 12.0).exp2();
        let (n, hop) = (self.n_fft, self.hop);
        let len = audio.len();
        let pad = n;
        // Reflect-padded copy: [pad | audio | pad + hop], so the frame grid anchored to the end can
        // start a hop early.
        let total = len + 2 * pad + hop;
        self.padded.clear();
        self.padded.extend((0..total).map(|p| {
            let v = audio[reflect_index(p as isize - pad as isize, len)];
            if v.is_finite() {
                v
            } else {
                0.0
            }
        }));
        self.out.clear();
        self.out.resize(total, 0.0);
        self.wsum.clear();
        self.wsum.resize(total, 0.0);
        // Frame centres at end + half - m * hop (end = pad + len; half is two hops, so the grid is
        // anchored to the end), every frame that reaches into the audio.
        let end = pad + len;
        let half = n / 2;
        let mut centre = end + half;
        while centre + half > pad {
            if centre >= half && centre + half <= total {
                self.frame_at(centre - half, alpha)?;
            }
            if centre < hop {
                break;
            }
            centre -= hop;
        }
        for (i, dst) in audio.iter_mut().enumerate() {
            let w = self.wsum[pad + i];
            if w > 1e-6 {
                *dst = self.out[pad + i] / w;
            }
        }
        Ok(())
    }

    /// Analyse, reshape and overlap-add the frame starting at `start` of `padded`.
    fn frame_at(&mut self, start: usize, alpha: f32) -> Result<(), String> {
        let n = self.n_fft;
        for ((dst, &x), &w) in self
            .frame
            .iter_mut()
            .zip(&self.padded[start..start + n])
            .zip(&self.window)
        {
            *dst = x * w;
        }
        let energy_in: f32 = self.frame.iter().map(|v| v * v).sum();
        if energy_in <= 1e-12 {
            // Silent frame: nothing to reshape; its window weight still counts.
            for (acc, &w) in self.wsum[start..start + n].iter_mut().zip(&self.window) {
                *acc += w * w;
            }
            return Ok(());
        }
        self.fwd
            .process_with_scratch(&mut self.frame, &mut self.spectrum, &mut self.fwd_scratch)
            .map_err(|e| format!("formant FFT: {e}"))?;
        self.envelope_of_spectrum()?;
        // Warp the envelope: E'(k) = E(k / alpha).
        let bins = self.envelope.len();
        let last = (bins - 1) as f32;
        for (k, dst) in self.warped.iter_mut().enumerate() {
            let src = (k as f32 / alpha).min(last);
            let i0 = src.floor() as usize;
            let i1 = (i0 + 1).min(bins - 1);
            let t = src - i0 as f32;
            *dst = self.envelope[i0] + (self.envelope[i1] - self.envelope[i0]) * t;
        }
        let mut energy_out = 0.0f32;
        let mut energy_spec = 0.0f32;
        for ((c, &e), &e2) in self
            .spectrum
            .iter_mut()
            .zip(&self.envelope)
            .zip(&self.warped)
        {
            energy_spec += c.norm_sqr();
            let g = (e2 - e).clamp(-MAX_CUT_LN, MAX_BOOST_LN).exp();
            *c *= g;
            energy_out += c.norm_sqr();
        }
        let scale = if energy_out > 1e-20 {
            (energy_spec / energy_out)
                .sqrt()
                .clamp(1.0 / MAX_ENERGY_SCALE, MAX_ENERGY_SCALE)
        } else {
            1.0
        };
        // realfft needs exactly real DC / Nyquist bins.
        self.spectrum[0].im = 0.0;
        if let Some(c) = self.spectrum.last_mut() {
            c.im = 0.0;
        }
        self.inv
            .process_with_scratch(&mut self.spectrum, &mut self.frame, &mut self.inv_scratch)
            .map_err(|e| format!("formant IFFT: {e}"))?;
        let norm = scale / n as f32;
        for (i, (&y, &w)) in self.frame.iter().zip(&self.window).enumerate() {
            self.out[start + i] += y * norm * w;
            self.wsum[start + i] += w * w;
        }
        Ok(())
    }

    /// The liftered log-magnitude envelope of `spectrum` into `envelope` (natural log).
    fn envelope_of_spectrum(&mut self) -> Result<(), String> {
        let n = self.n_fft;
        let peak = self
            .spectrum
            .iter()
            .map(|c| c.norm_sqr())
            .fold(0.0f32, f32::max)
            .sqrt();
        let floor = (peak * 1e-4).max(1e-9);
        for (w, c) in self.work.iter_mut().zip(&self.spectrum) {
            *w = Complex::new(c.norm().max(floor).ln(), 0.0);
        }
        self.work[0].im = 0.0;
        if let Some(c) = self.work.last_mut() {
            c.im = 0.0;
        }
        self.inv
            .process_with_scratch(&mut self.work, &mut self.cepstrum, &mut self.inv_scratch)
            .map_err(|e| format!("formant cepstrum: {e}"))?;
        let inv_n = 1.0 / n as f32;
        for v in self.cepstrum.iter_mut() {
            *v *= inv_n;
        }
        // Lifter cutoff: the default, or 0.6 of a clearly detected pitch period.
        let sr = self.sample_rate as f32;
        let default_q = sr * DEFAULT_LIFTER_MS / 1000.0;
        let min_q = sr * MIN_LIFTER_MS / 1000.0;
        let q_lo = (sr / 1100.0).floor() as usize;
        let q_hi = ((sr / 60.0) as usize).min(n / 2 - 1);
        let mut best = (0usize, PITCH_PEAK_MIN);
        for q in q_lo.max(2)..=q_hi {
            if self.cepstrum[q] > best.1 {
                best = (q, self.cepstrum[q]);
            }
        }
        let cutoff = if best.0 > 0 { (0.6 * best.0 as f32).min(default_q) } else { default_q }.max(min_q);
        // Flat-top lifter: 1 below 0.7 * cutoff, raised-cosine to 0 at the cutoff (symmetric).
        let flat = 0.7 * cutoff;
        for q in 1..n {
            let d = q.min(n - q) as f32;
            let l = if d <= flat {
                1.0
            } else if d >= cutoff {
                0.0
            } else {
                0.5 * (1.0 + (std::f32::consts::PI * (d - flat) / (cutoff - flat)).cos())
            };
            self.cepstrum[q] *= l;
        }
        self.fwd
            .process_with_scratch(&mut self.cepstrum, &mut self.work, &mut self.fwd_scratch)
            .map_err(|e| format!("formant envelope: {e}"))?;
        for (e, c) in self.envelope.iter_mut().zip(&self.work) {
            *e = c.re;
        }
        Ok(())
    }

    /// Added latency in samples: none (each call is analysed on its own, see the module docs).
    #[allow(dead_code)] // Frozen P3-S0 API.
    pub fn latency_samples(&self) -> usize {
        0
    }

    /// Nothing is carried between calls; kept for the post stage's reset path.
    pub fn reset(&mut self) {}

    #[allow(dead_code)]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A voiced vowel-like tone: harmonics of `f0` shaped by resonances at `formants` (Hz).
    pub(crate) fn vowel(f0: f32, sr: u32, len: usize, formants: &[f32]) -> Vec<f32> {
        let harmonics = ((sr as f32 / 2.0 - 200.0) / f0) as usize;
        let amp = |f: f32| -> f32 {
            formants
                .iter()
                .map(|&fc| 1.0 / (1.0 + ((f - fc) / 120.0).powi(2)))
                .sum::<f32>()
                + 0.02
        };
        (0..len)
            .map(|i| {
                let t = i as f32 / sr as f32;
                (1..=harmonics)
                    .map(|k| 0.05 * amp(f0 * k as f32) * (std::f32::consts::TAU * f0 * k as f32 * t).sin())
                    .sum()
            })
            .collect()
    }

    /// Frequency (Hz) of the strongest spectral peak between `lo` and `hi` Hz, with quadratic
    /// interpolation on a zero-padded FFT.
    pub(crate) fn peak_hz(x: &[f32], sr: u32, lo: f32, hi: f32) -> f32 {
        let n = (x.len() * 8).next_power_of_two();
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);
        let mut input = fft.make_input_vec();
        for (i, (dst, &v)) in input.iter_mut().zip(x).enumerate() {
            let w = 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / x.len() as f32).cos();
            *dst = v * w;
        }
        let mut spec = fft.make_output_vec();
        fft.process(&mut input, &mut spec).unwrap();
        let mag: Vec<f32> = spec.iter().map(|c| c.norm()).collect();
        let bin = |hz: f32| (hz * n as f32 / sr as f32) as usize;
        let k = (bin(lo)..bin(hi))
            .max_by(|&a, &b| mag[a].total_cmp(&mag[b]))
            .unwrap();
        let (a, b, c) = (mag[k - 1].ln(), mag[k].ln(), mag[k + 1].ln());
        let delta = 0.5 * (a - c) / (a - 2.0 * b + c);
        (k as f32 + delta) * sr as f32 / n as f32
    }

    /// Spectral centroid (Hz) of `x`.
    fn centroid(x: &[f32], sr: u32) -> f32 {
        let n = x.len().next_power_of_two();
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);
        let mut input = fft.make_input_vec();
        input[..x.len()].copy_from_slice(x);
        let mut spec = fft.make_output_vec();
        fft.process(&mut input, &mut spec).unwrap();
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (k, c) in spec.iter().enumerate() {
            let p = f64::from(c.norm_sqr());
            num += p * k as f64;
            den += p;
        }
        (num / den) as f32 * sr as f32 / n as f32
    }

    fn cents(a: f32, b: f32) -> f32 {
        1200.0 * (a / b).log2()
    }

    #[test]
    fn bypass_is_bit_exact_and_silence_stays_silent() {
        let x = vowel(150.0, 40_000, 4000, &[600.0, 1400.0]);
        let mut y = x.clone();
        let mut fs = FormantShifter::new(40_000);
        fs.process(&mut y, 0.0).unwrap();
        assert_eq!(x, y);
        fs.process(&mut y, 0.005).unwrap();
        assert_eq!(x, y);
        fs.process(&mut y, f32::NAN).unwrap();
        assert_eq!(x, y);
        let mut s = vec![0.0f32; 3000];
        fs.process(&mut s, 7.0).unwrap();
        assert!(s.iter().all(|&v| v == 0.0));
        assert_eq!(fs.latency_samples(), 0);
    }

    /// Plan acceptance: the F0 is preserved within 5 cents at every shift, while the envelope
    /// (spectral centroid) moves the way the knob says.
    #[test]
    fn preserves_f0_within_5_cents_and_moves_the_envelope() {
        for &(sr, f0) in &[(40_000u32, 150.0f32), (48_000, 220.0), (32_000, 110.0)] {
            let x = vowel(f0, sr, sr as usize / 2, &[700.0, 1500.0, 2600.0]);
            let c0 = centroid(&x, sr);
            // Measure on a harmonic that stays strong at every shift (the 3rd).
            let h = 3.0;
            let reference = peak_hz(&x, sr, f0 * (h - 0.5), f0 * (h + 0.5)) / h;
            assert!(cents(reference, f0).abs() < 1.0, "reference {reference}");
            let mut last_centroid = None;
            for st in [-12.0f32, -6.0, -2.0, 2.0, 6.0, 12.0] {
                let mut y = x.clone();
                let mut fs = FormantShifter::new(sr);
                fs.process(&mut y, st).unwrap();
                assert!(y.iter().all(|v| v.is_finite()));
                let got = peak_hz(&y, sr, f0 * (h - 0.5), f0 * (h + 0.5)) / h;
                assert!(cents(got, reference).abs() < 5.0, "{sr} Hz, {f0} Hz, {st} st: F0 {got} vs {reference}");
                let c = centroid(&y, sr);
                if st < 0.0 {
                    assert!(c < c0, "{st} st: centroid {c} vs {c0}");
                } else {
                    assert!(c > c0, "{st} st: centroid {c} vs {c0}");
                }
                if let Some(prev) = last_centroid {
                    assert!(c > prev, "centroid not monotonic in the shift: {c} after {prev}");
                }
                last_centroid = Some(c);
                // Loudness is kept within a few dB.
                let rms = |v: &[f32]| (v.iter().map(|s| s * s).sum::<f32>() / v.len() as f32).sqrt();
                let db = 20.0 * (rms(&y) / rms(&x)).log10();
                assert!(db.abs() < 4.0, "{st} st: level change {db} dB");
            }
        }
    }

    /// Overlapping buffers whose hop is a multiple of 10 ms produce the same audio where they
    /// overlap (away from the reflected ends), so SOLA joins matching versions.
    #[test]
    fn overlapping_ticks_agree_in_their_shared_audio() {
        let sr = 40_000;
        let x = vowel(180.0, sr, 30_000, &[500.0, 1800.0]);
        let (len, hop) = (6_000usize, 1_600usize); // 150 ms ticks, 40 ms hop
        let mut fs = FormantShifter::with_capacity(sr, len);
        let mut a = x[..len].to_vec();
        let mut b = x[hop..hop + len].to_vec();
        fs.process(&mut a, 5.0).unwrap();
        fs.process(&mut b, 5.0).unwrap();
        let margin = 1_000; // 25 ms from either end
        let mut worst = 0.0f32;
        for i in margin..len - hop - margin {
            worst = worst.max((a[hop + i] - b[i]).abs());
        }
        let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(worst < 1e-3 * peak, "overlap differs by {worst} (peak {peak})");
    }

    #[test]
    fn short_and_odd_buffers_are_safe() {
        let mut fs = FormantShifter::new(40_000);
        for len in [2usize, 3, 50, 799, 800, 801, 2_001] {
            let mut y: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin() * 0.3).collect();
            fs.process(&mut y, -9.0).unwrap();
            assert!(y.iter().all(|v| v.is_finite()), "len {len}");
        }
        let mut y = vec![0.1, f32::NAN, 0.2, f32::INFINITY];
        fs.process(&mut y, 4.0).unwrap();
        assert!(y[0].is_finite() && y[2].is_finite());
    }
}
