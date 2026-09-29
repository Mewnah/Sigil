//! RMVPE ONNX pitch extractor (default for Low latency / Quality GPU presets).
//!
//! Two export contracts are supported, chosen from the graph's declared input at load:
//! - RVC's `rmvpe.onnx`: log-mel `[1, 128, T]` in, salience `[1, T, 360]` out. The log-mel is
//!   computed here exactly like RVC's `MelSpectrogram` (16 kHz, n_fft = win = 1024, hop 160,
//!   periodic Hann, centred reflect padding, 128 HTK/Slaney mels over 30..8000 Hz,
//!   `log(clamp(x, 1e-5))`), with `T` zero-padded to a multiple of 32 as in `mel2hidden`.
//! - waveform exports (`[1, N]`, optional `threshold`), whose output is either salience or F0 in Hz.
//!
//! Salience is decoded like RVC `to_local_average_cents`: argmax ± 4 bins, salience-weighted mean
//! of `cents = 20 * bin + 1997.3794084376191`, 0 when the peak is <= 0.03, `f0 = 10 * 2^(c/1200)`.

use std::sync::Arc;

use ort::session::Session;
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use crate::services::neural_audio::ort_session::{session_builder_for_vc, VcSessionOptions};

use super::f0_post::smooth_f0;
use super::pitch_onnx::{flat_f0, waveform_shape, MainInput, PitchInputs};
use super::traits::PitchExtractor;

const SR: u32 = 16_000;
const HOP: usize = 160;
const N_BINS: usize = 360;
const N_FFT: usize = 1024;
const N_MELS: usize = 128;
const MEL_FMIN: f64 = 30.0;
const MEL_FMAX: f64 = 8000.0;
const MEL_CLAMP: f32 = 1e-5;
/// RVC `infer_from_audio(audio, thred=0.03)`.
const SALIENCE_THRESHOLD: f32 = 0.03;
/// Value w-okada feeds its waveform-input RMVPE exports' `threshold` input.
const WAVEFORM_EXPORT_THRESHOLD: f32 = 0.3;
const CENTS_OFFSET: f32 = 1_997.379_4;

pub struct RmvpePitchExtractor {
    session: Session,
    inputs: PitchInputs,
    mel: Option<LogMel>,
}

impl RmvpePitchExtractor {
    pub fn load(model_path: &str, execution_provider_hint: &str, opts: &VcSessionOptions) -> Result<Self, String> {
        let path = model_path.trim();
        if path.is_empty() {
            return Err("RMVPE model path is empty".into());
        }
        let session = session_builder_for_vc(execution_provider_hint, opts)?
            .commit_from_file(path)
            .map_err(|e| format!("Failed to load RMVPE ONNX {}: {}", path, e))?;
        let inputs = PitchInputs::inspect(&session, "RMVPE", path, WAVEFORM_EXPORT_THRESHOLD)?;
        let mel = match inputs.main {
            MainInput::Waveform { .. } => None,
            MainInput::LogMel { n_mels } => {
                if n_mels.is_some_and(|m| m != N_MELS) {
                    return Err(format!(
                        "RMVPE ONNX {} expects a {:?} mel input; Sigil computes RVC's {}-band \
                         log-mel [1, {}, T]",
                        path,
                        inputs.main_dims(),
                        N_MELS,
                        N_MELS
                    ));
                }
                Some(LogMel::new())
            }
        };
        Ok(Self { session, inputs, mel })
    }
}

impl PitchExtractor for RmvpePitchExtractor {
    fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
        if mono_16k.is_empty() {
            return Ok(Vec::new());
        }
        let n = mono_16k.len();
        let (outputs, frames) = match (&mut self.mel, self.inputs.main) {
            (Some(mel), _) => {
                let (frames, padded_frames, data) = mel.compute(mono_16k)?;
                let outputs = self
                    .inputs
                    .run(&mut self.session, vec![1, N_MELS, padded_frames], &data, "RMVPE")?;
                (outputs, frames)
            }
            (None, MainInput::Waveform { rank }) => {
                let outputs = self
                    .inputs
                    .run(&mut self.session, waveform_shape(rank, n), mono_16k, "RMVPE")?;
                (outputs, 1 + n / HOP)
            }
            (None, MainInput::LogMel { .. }) => {
                return Err("RMVPE mel front end missing".into());
            }
        };
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("RMVPE output: {}", e))?;
        let dims: Vec<usize> = shape.iter().map(|d| (*d).max(0) as usize).collect();
        let mut f0 = decode_output(&dims, data, frames)?;
        smooth_f0(&mut f0, 3);
        Ok(f0)
    }
}

/// Decode an RMVPE output into F0 (Hz). Salience `[1, T, 360]` / `[T, 360]` is decoded and cut to
/// `frames` rows (RVC `hidden[:, :n_frames]`); a direct F0 output `[1, T]` / `[T]` passes through.
fn decode_output(dims: &[usize], data: &[f32], frames: usize) -> Result<Vec<f32>, String> {
    let salience_rows = match dims {
        [1, t, b] if *b == N_BINS => Some(*t),
        [t, b] if *b == N_BINS && (*t != 1 || frames <= 1) => Some(*t),
        _ => None,
    };
    if let Some(t) = salience_rows {
        let rows = t.min(frames).min(data.len() / N_BINS);
        return Ok(data[..rows * N_BINS]
            .chunks_exact(N_BINS)
            .map(|row| {
                // A NaN near the peak (or an overflowing weighted mean) is unvoiced, never NaN F0.
                let f = salience_to_f0(row);
                if f.is_finite() {
                    f
                } else {
                    0.0
                }
            })
            .collect());
    }
    let mut f0 = flat_f0(dims, data).ok_or_else(|| {
        format!(
            "RMVPE output shape {dims:?} unsupported; expected salience [1, T, 360] / [T, 360] \
             or F0 [1, T] / [T]"
        )
    })?;
    for v in f0.iter_mut() {
        if !v.is_finite() || *v < 0.0 {
            *v = 0.0;
        }
    }
    Ok(f0)
}

/// RVC `to_local_average_cents` + `decode` for one salience row.
fn salience_to_f0(row: &[f32]) -> f32 {
    let (center, peak) = row
        .iter()
        .copied()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (i, v)| if v > best.1 { (i, v) } else { best });
    if peak <= SALIENCE_THRESHOLD {
        return 0.0;
    }
    let start = center.saturating_sub(4);
    let end = (center + 5).min(row.len());
    let mut weighted = 0.0f32;
    let mut weight = 0.0f32;
    for (bin, &s) in row.iter().enumerate().take(end).skip(start) {
        weighted += s * (20.0 * bin as f32 + CENTS_OFFSET);
        weight += s;
    }
    if weight <= 0.0 {
        return 0.0;
    }
    let cents = weighted / weight;
    10.0 * (cents / 1200.0).exp2()
}

/// RVC `MelSpectrogram(is_half, 128, 16000, 1024, 160, None, 30, 8000)` + `mel2hidden` padding.
struct LogMel {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    /// Per mel band: first FFT bin and its non-zero weights.
    bands: Vec<(usize, Vec<f32>)>,
    frame: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    magnitude: Vec<f32>,
}

fn hz_to_mel_htk(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz_htk(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

/// librosa `filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=True, norm="slaney")`.
fn mel_filterbank() -> Vec<(usize, Vec<f32>)> {
    let n_freqs = N_FFT / 2 + 1;
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * f64::from(SR) / N_FFT as f64)
        .collect();
    let (mel_lo, mel_hi) = (hz_to_mel_htk(MEL_FMIN), hz_to_mel_htk(MEL_FMAX));
    let mel_f: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz_htk(mel_lo + (mel_hi - mel_lo) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    (0..N_MELS)
        .map(|i| {
            let (lo, center, hi) = (mel_f[i], mel_f[i + 1], mel_f[i + 2]);
            let enorm = 2.0 / (hi - lo);
            let weights: Vec<f64> = fft_freqs
                .iter()
                .map(|&f| {
                    let lower = (f - lo) / (center - lo);
                    let upper = (hi - f) / (hi - center);
                    lower.min(upper).max(0.0) * enorm
                })
                .collect();
            let first = weights.iter().position(|&w| w > 0.0).unwrap_or(0);
            let last = weights.iter().rposition(|&w| w > 0.0).unwrap_or(0);
            let band: Vec<f32> = if last >= first && weights[first] > 0.0 {
                weights[first..=last].iter().map(|&w| w as f32).collect()
            } else {
                Vec::new()
            };
            (first, band)
        })
        .collect()
}

/// torch `reflect` padding index (no edge repeat) for any offset.
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

impl LogMel {
    fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        // torch.hann_window(1024) is periodic.
        let window = (0..N_FFT)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32)
            .collect();
        let frame = fft.make_input_vec();
        let spectrum = fft.make_output_vec();
        let scratch = fft.make_scratch_vec();
        Self {
            fft,
            window,
            bands: mel_filterbank(),
            frame,
            spectrum,
            scratch,
            magnitude: vec![0.0; N_FFT / 2 + 1],
        }
    }

    /// Returns `(frames, padded_frames, data)` with `data` = `[N_MELS, padded_frames]` row-major.
    fn compute(&mut self, audio: &[f32]) -> Result<(usize, usize, Vec<f32>), String> {
        let n = audio.len();
        let frames = 1 + n / HOP;
        let padded_frames = 32 * ((frames - 1) / 32 + 1);
        let mut data = vec![0.0f32; N_MELS * padded_frames];
        let pad = (N_FFT / 2) as isize;
        for t in 0..frames {
            let origin = (t * HOP) as isize - pad;
            for (k, (dst, &w)) in self.frame.iter_mut().zip(self.window.iter()).enumerate() {
                *dst = audio[reflect_index(origin + k as isize, n)] * w;
            }
            self.fft
                .process_with_scratch(&mut self.frame, &mut self.spectrum, &mut self.scratch)
                .map_err(|e| format!("RMVPE mel FFT: {e}"))?;
            for (m, c) in self.magnitude.iter_mut().zip(self.spectrum.iter()) {
                *m = (c.re * c.re + c.im * c.im).sqrt();
            }
            for (band_idx, (first, weights)) in self.bands.iter().enumerate() {
                let energy: f32 = weights
                    .iter()
                    .zip(&self.magnitude[*first..])
                    .map(|(w, m)| w * m)
                    .sum();
                data[band_idx * padded_frames + t] = energy.max(MEL_CLAMP).ln();
            }
        }
        Ok((frames, padded_frames, data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hz(cents: f32) -> f32 {
        10.0 * (cents / 1200.0).exp2()
    }

    #[test]
    fn salience_decode_matches_rvc_local_average() {
        // Symmetric peak at bin 100 → exactly the bin's cents.
        let mut row = vec![0.0f32; N_BINS];
        row[99] = 0.5;
        row[100] = 0.9;
        row[101] = 0.5;
        let expected = hz(20.0 * 100.0 + CENTS_OFFSET);
        assert!((salience_to_f0(&row) - expected).abs() < 1e-3);
        assert!((expected - 100.64).abs() < 0.05, "{expected}");

        // Asymmetric window: weighted mean over argmax ± 4 only (bin 110 is outside).
        let mut row = vec![0.0f32; N_BINS];
        row[200] = 0.8;
        row[203] = 0.4;
        row[210] = 0.7;
        let cents = (0.8 * (20.0 * 200.0 + CENTS_OFFSET) + 0.4 * (20.0 * 203.0 + CENTS_OFFSET)) / 1.2;
        assert!((salience_to_f0(&row) - hz(cents)).abs() < 1e-2);

        // Edge bins: the window is clipped (RVC pads zeros).
        let mut row = vec![0.0f32; N_BINS];
        row[0] = 0.6;
        row[2] = 0.2;
        let cents = (0.6 * CENTS_OFFSET + 0.2 * (40.0 + CENTS_OFFSET)) / 0.8;
        assert!((salience_to_f0(&row) - hz(cents)).abs() < 1e-3);

        // Peak at or below the 0.03 threshold → unvoiced.
        let mut row = vec![0.0f32; N_BINS];
        row[150] = 0.03;
        assert_eq!(salience_to_f0(&row), 0.0);
    }

    #[test]
    fn decode_output_handles_salience_and_flat_shapes() {
        let mut sal = vec![0.0f32; 3 * N_BINS];
        sal[N_BINS + 100] = 1.0;
        // [1, T, 360] cut to `frames` rows.
        let f0 = decode_output(&[1, 3, N_BINS], &sal, 2).unwrap();
        assert_eq!(f0.len(), 2);
        assert_eq!(f0[0], 0.0);
        assert!((f0[1] - hz(20.0 * 100.0 + CENTS_OFFSET)).abs() < 1e-3);
        // [T, 360].
        assert_eq!(decode_output(&[3, N_BINS], &sal, 3).unwrap().len(), 3);
        // A flat [1, T] F0 output is taken as Hz, not collapsed into one salience row.
        let flat = vec![100.0f32, 0.0, 220.0, 230.0];
        assert_eq!(decode_output(&[1, 4], &flat, 5).unwrap(), flat);
        assert_eq!(decode_output(&[4], &flat, 5).unwrap(), flat);
        assert!(decode_output(&[2, 2, 2], &[0.0; 8], 2).is_err());
    }

    #[test]
    fn decode_output_maps_non_finite_salience_to_unvoiced() {
        let mut sal = vec![0.0f32; 3 * N_BINS];
        // Row 0: NaN next to the peak poisons the weighted mean.
        sal[100] = 0.9;
        sal[101] = f32::NAN;
        // Row 1: all NaN (no peak at all).
        for v in &mut sal[N_BINS..2 * N_BINS] {
            *v = f32::NAN;
        }
        // Row 2: infinite salience.
        sal[2 * N_BINS + 50] = f32::INFINITY;
        let f0 = decode_output(&[1, 3, N_BINS], &sal, 3).unwrap();
        assert_eq!(f0.len(), 3);
        assert!(f0.iter().all(|v| v.is_finite()), "{f0:?}");
        assert_eq!(f0[0], 0.0);
        assert_eq!(f0[1], 0.0);
        // A flat F0 output with NaN / negative values is sanitized too.
        assert_eq!(decode_output(&[3], &[f32::NAN, -1.0, 150.0], 3).unwrap(), vec![0.0, 0.0, 150.0]);
    }

    #[test]
    fn mel_filterbank_matches_librosa_htk_slaney() {
        let bands = mel_filterbank();
        assert_eq!(bands.len(), N_MELS);
        // Band 0 spans 30 Hz .. ~58.7 Hz (centre ~44.2 Hz): FFT bins 2 (31.25 Hz) and 3 (46.875 Hz).
        let (first, w) = &bands[0];
        assert_eq!(*first, 2);
        assert_eq!(w.len(), 2);
        assert!((w[0] - 0.006_12).abs() < 2e-4, "{w:?}");
        assert!((w[1] - 0.057_0).abs() < 1e-3, "{w:?}");
        // Last band ends at fmax = 8000 Hz = bin 512, where its weight is (numerically) zero.
        let (first, w) = &bands[N_MELS - 1];
        assert!(first + w.len() <= 513);
        if first + w.len() == 513 {
            assert!(w[w.len() - 1] < 1e-6, "{:?}", &w[w.len() - 3..]);
        }
        assert!(bands.iter().all(|(_, w)| w.iter().all(|&x| x >= 0.0)));
    }

    #[test]
    fn log_mel_frame_count_padding_and_silence_floor() {
        let mut mel = LogMel::new();
        let (frames, padded, data) = mel.compute(&vec![0.0f32; 1600]).unwrap();
        assert_eq!(frames, 11); // 1 + 1600 / 160 (centred STFT)
        assert_eq!(padded, 32);
        assert_eq!(data.len(), N_MELS * 32);
        for band in 0..N_MELS {
            for t in 0..padded {
                let v = data[band * padded + t];
                if t < frames {
                    assert!((v - MEL_CLAMP.ln()).abs() < 1e-4);
                } else {
                    assert_eq!(v, 0.0, "mel2hidden pads with constant 0");
                }
            }
        }
        // A 1 kHz tone peaks in the band containing 1 kHz.
        let tone: Vec<f32> = (0..3200)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SR as f32).sin())
            .collect();
        let (frames, padded, data) = mel.compute(&tone).unwrap();
        let t = frames / 2;
        let best = (0..N_MELS)
            .max_by(|&a, &b| {
                data[a * padded + t]
                    .partial_cmp(&data[b * padded + t])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let mel_lo = hz_to_mel_htk(MEL_FMIN);
        let mel_hi = hz_to_mel_htk(MEL_FMAX);
        let centre = mel_to_hz_htk(mel_lo + (mel_hi - mel_lo) * (best + 1) as f64 / 129.0);
        assert!((centre - 1000.0).abs() < 40.0, "band {best} centre {centre}");
    }

    #[test]
    fn reflect_padding_matches_torch() {
        // torch reflect: [x2, x1 | x0 x1 x2 x3 | x2 x1]
        let n = 4;
        let idx: Vec<usize> = (-2..6).map(|i| reflect_index(i, n)).collect();
        assert_eq!(idx, vec![2, 1, 0, 1, 2, 3, 2, 1]);
    }
}
