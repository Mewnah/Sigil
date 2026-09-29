//! Autocorrelation F0 (dio-like weak-CPU escape hatch). BSD-friendly; not full WORLD.

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use super::f0_post::smooth_f0;
use super::traits::PitchExtractor;

const SR: u32 = 16_000;
const HOP: usize = 160;
const FRAME: usize = 1024;
/// FFT length for a *linear* (non-circular) autocorrelation of up to `FRAME` samples.
const NFFT: usize = 2 * FRAME;
/// Normalized-autocorrelation peak needed to call a frame voiced.
const VOICING_THRESHOLD: f32 = 0.45;
/// A local maximum within this fraction of the best correlation counts as the period (see [`pick_lag`]).
const PEAK_RATIO: f32 = 0.9;

/// McLeod-style key maximum over `corr[i]` = normalized ACF at lag `min_lag + i`: the shortest interior
/// local maximum within [`PEAK_RATIO`] of the best (else the best itself). A periodic signal correlates
/// ~1 at every multiple of its period, so a plain argmax often lands on a subharmonic (a 150 Hz sine
/// read as 50 Hz). `None` when the best correlation is not voiced.
fn pick_lag(corr: &[f32], min_lag: usize) -> Option<usize> {
    let (best_i, best) = corr
        .iter()
        .copied()
        .enumerate()
        .fold((0usize, 0.0f32), |a, (i, c)| if c > a.1 { (i, c) } else { a });
    if best <= VOICING_THRESHOLD {
        return None;
    }
    let thr = best * PEAK_RATIO;
    let i = (1..corr.len().saturating_sub(1))
        .find(|&i| corr[i] >= thr && corr[i] > corr[i - 1] && corr[i] >= corr[i + 1])
        .unwrap_or(best_i);
    Some(min_lag + i)
}

/// Cheap CPU pitch extractor for Streamer fallback / weak machines.
///
/// Frame `j` is centred on sample `160 * j` (the RVC / WORLD `frame_period = 10 ms` grid) and
/// analyses up to `FRAME` samples around it; windows at the ends of the buffer are simply shorter.
/// `f0` returns exactly `len / 160` frames. The per-lag normalized autocorrelation
/// `sum(x[i]·x[i+lag]) / sqrt(sum(x[i]²) · sum(x[i+lag]²))` is computed with one real FFT pair
/// per frame plus prefix sums of squares (O(N log N) instead of O(lags · N)).
pub struct DioPitchExtractor {
    min_hz: f32,
    max_hz: f32,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    time_buf: Vec<f32>,
    spec_buf: Vec<Complex<f32>>,
    fwd_scratch: Vec<Complex<f32>>,
    inv_scratch: Vec<Complex<f32>>,
    acf: Vec<f32>,
    energy_prefix: Vec<f64>,
    /// Normalized ACF per candidate lag (`min_lag..=max_lag`), reused across frames.
    corr: Vec<f32>,
}

impl Default for DioPitchExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl DioPitchExtractor {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(NFFT);
        let inv = planner.plan_fft_inverse(NFFT);
        let time_buf = fwd.make_input_vec();
        let spec_buf = fwd.make_output_vec();
        let fwd_scratch = fwd.make_scratch_vec();
        let inv_scratch = inv.make_scratch_vec();
        Self {
            min_hz: 50.0,
            max_hz: 1100.0,
            fwd,
            inv,
            time_buf,
            spec_buf,
            fwd_scratch,
            inv_scratch,
            acf: vec![0.0; NFFT],
            energy_prefix: Vec::with_capacity(FRAME + 1),
            corr: Vec::with_capacity(FRAME / 2),
        }
    }

    /// Linear autocorrelation `acf[lag] = sum_{i < len-lag} x[i]·x[i+lag]` for `lag < len`.
    fn autocorrelate(&mut self, frame: &[f32]) -> Result<(), String> {
        self.time_buf[..frame.len()].copy_from_slice(frame);
        self.time_buf[frame.len()..].fill(0.0);
        self.fwd
            .process_with_scratch(&mut self.time_buf, &mut self.spec_buf, &mut self.fwd_scratch)
            .map_err(|e| format!("dio FFT: {e}"))?;
        for c in self.spec_buf.iter_mut() {
            *c = Complex::new(c.norm_sqr(), 0.0);
        }
        self.inv
            .process_with_scratch(&mut self.spec_buf, &mut self.acf, &mut self.inv_scratch)
            .map_err(|e| format!("dio IFFT: {e}"))?;
        // realfft's inverse is unnormalized.
        let scale = 1.0 / NFFT as f32;
        for v in self.acf[..frame.len()].iter_mut() {
            *v *= scale;
        }
        Ok(())
    }

    fn frame_f0(&mut self, frame: &[f32]) -> Result<f32, String> {
        let n = frame.len();
        let min_lag = (SR as f32 / self.max_hz).round() as usize;
        let max_lag = ((SR as f32 / self.min_hz).round() as usize).min(n / 2);
        if n <= min_lag || max_lag < min_lag {
            return Ok(0.0);
        }
        let energy: f32 = frame.iter().map(|x| x * x).sum::<f32>() / n as f32;
        if energy < 1e-6 {
            return Ok(0.0);
        }
        self.autocorrelate(frame)?;
        self.energy_prefix.clear();
        self.energy_prefix.push(0.0);
        let mut acc = 0.0f64;
        for &x in frame {
            acc += f64::from(x) * f64::from(x);
            self.energy_prefix.push(acc);
        }
        let total = self.energy_prefix[n];
        self.corr.clear();
        for lag in min_lag..=max_lag {
            let head = self.energy_prefix[n - lag];
            let tail = total - self.energy_prefix[lag];
            let denom = (head * tail).sqrt();
            self.corr.push(if denom <= 1e-12 {
                0.0
            } else {
                (f64::from(self.acf[lag]) / denom) as f32
            });
        }
        Ok(pick_lag(&self.corr, min_lag).map_or(0.0, |lag| SR as f32 / lag as f32))
    }
}

impl PitchExtractor for DioPitchExtractor {
    fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
        let len = mono_16k.len();
        let n_frames = len / HOP;
        let mut out = vec![0.0f32; n_frames];
        for (fi, frame_f0) in out.iter_mut().enumerate() {
            let center = fi * HOP;
            let start = center.saturating_sub(FRAME / 2);
            let end = (center + FRAME / 2).min(len);
            *frame_f0 = self.frame_f0(&mono_16k[start..end])?;
        }
        smooth_f0(&mut out, 3);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(hz: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * hz * i as f32 / SR as f32).sin())
            .collect()
    }

    /// The original O(lags·N) normalized autocorrelation, kept as the reference.
    fn brute_force_f0(frame: &[f32], min_hz: f32, max_hz: f32) -> f32 {
        let min_lag = (SR as f32 / max_hz).round() as usize;
        let max_lag = ((SR as f32 / min_hz).round() as usize).min(frame.len() / 2);
        let energy: f32 = frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32;
        if frame.len() <= min_lag || max_lag < min_lag || energy < 1e-6 {
            return 0.0;
        }
        let corr: Vec<f32> = (min_lag..=max_lag)
            .map(|lag| {
                let (mut num, mut d1, mut d2) = (0.0f32, 0.0f32, 0.0f32);
                for i in 0..frame.len() - lag {
                    num += frame[i] * frame[i + lag];
                    d1 += frame[i] * frame[i];
                    d2 += frame[i + lag] * frame[i + lag];
                }
                let denom = (d1 * d2).sqrt();
                if denom <= 1e-12 {
                    0.0
                } else {
                    num / denom
                }
            })
            .collect();
        pick_lag(&corr, min_lag).map_or(0.0, |lag| SR as f32 / lag as f32)
    }

    /// Sum of the first `harmonics` harmonics of `hz` with falling amplitude.
    fn harmonic_tone(hz: f32, harmonics: usize, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / SR as f32;
                (1..=harmonics)
                    .map(|k| 0.4 / k as f32 * (2.0 * std::f32::consts::PI * hz * k as f32 * t).sin())
                    .sum()
            })
            .collect()
    }

    #[test]
    fn dio_finds_the_fundamental_not_a_subharmonic() {
        // A plain ACF argmax read a 150 Hz sine as 50 Hz and a 120 Hz harmonic tone as ~60 Hz.
        let mut dio = DioPitchExtractor::new();
        for hz in [120.0f32, 150.0, 220.0] {
            for (kind, sig) in [("sine", sine(hz, FRAME)), ("6-harmonic", harmonic_tone(hz, 6, FRAME))] {
                let f0 = dio.frame_f0(&sig).unwrap();
                assert!((f0 - hz).abs() <= 0.03 * hz, "{kind} {hz} Hz read as {f0} Hz");
                let slow = brute_force_f0(&sig, 50.0, 1100.0);
                assert!((slow - hz).abs() <= 0.03 * hz, "brute force: {kind} {hz} Hz read as {slow} Hz");
            }
            // And through the full frame grid (windows, smoothing).
            let track = dio.f0(&harmonic_tone(hz, 6, 3200)).unwrap();
            for (i, &v) in track.iter().enumerate() {
                assert!((v - hz).abs() <= 0.03 * hz, "{hz} Hz frame {i}: {v}");
            }
        }
    }

    #[test]
    fn pick_lag_prefers_the_shortest_strong_peak() {
        // Peaks at i = 3 (0.93) and i = 8 (1.0): 0.93 >= 0.9 * 1.0, so the shorter lag wins.
        let corr = [0.1, 0.5, 0.8, 0.93, 0.6, 0.2, 0.5, 0.9, 1.0, 0.7];
        assert_eq!(pick_lag(&corr, 20), Some(23));
        // A weak early peak is ignored.
        let corr = [0.1, 0.5, 0.6, 0.5, 0.2, 0.5, 0.9, 1.0, 0.7];
        assert_eq!(pick_lag(&corr, 20), Some(27));
        // Unvoiced.
        assert_eq!(pick_lag(&[0.1, 0.3, 0.2], 20), None);
        assert_eq!(pick_lag(&[], 20), None);
    }

    #[test]
    fn dio_returns_len_over_hop_frames() {
        let mut dio = DioPitchExtractor::new();
        for len in [0usize, 159, 160, 720, 1200, 1194, 3200] {
            let f0 = dio.f0(&sine(197.0, len)).unwrap();
            assert_eq!(f0.len(), len / HOP, "len {len}");
        }
    }

    #[test]
    fn dio_tracks_a_sine_including_the_newest_frames() {
        let mut dio = DioPitchExtractor::new();
        let f0 = dio.f0(&sine(197.0, 1600)).unwrap();
        assert_eq!(f0.len(), 10);
        for (i, &v) in f0.iter().enumerate() {
            assert!((v - 197.0).abs() < 4.0, "frame {i}: {v}");
        }
    }

    #[test]
    fn fft_autocorrelation_matches_brute_force() {
        let mut dio = DioPitchExtractor::new();
        let mut seed = 0x1234_5678u32;
        let mut noise = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let signals: Vec<Vec<f32>> = vec![
            sine(137.0, FRAME),
            sine(310.0, 700),
            sine(95.0, 600).iter().map(|s| s + 0.05 * noise()).collect(),
            (0..FRAME).map(|_| noise()).collect(),
        ];
        for sig in signals {
            let fast = dio.frame_f0(&sig).unwrap();
            let slow = brute_force_f0(&sig, 50.0, 1100.0);
            assert!((fast - slow).abs() < 1e-3, "fast {fast} vs brute force {slow}");
            // And the raw autocorrelation agrees lag by lag.
            dio.autocorrelate(&sig).unwrap();
            for lag in [0usize, 1, 17, 160, sig.len() / 2] {
                let exact: f32 = (0..sig.len() - lag).map(|i| sig[i] * sig[i + lag]).sum();
                assert!(
                    (dio.acf[lag] - exact).abs() < 1e-3 * exact.abs().max(1.0),
                    "lag {lag}: {} vs {exact}",
                    dio.acf[lag]
                );
            }
        }
    }
}
