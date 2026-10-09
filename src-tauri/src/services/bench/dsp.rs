//! Offline DSP for the benchmark's quality metrics (P3-F): short-time power spectra, mel
//! filterbanks, mel-cepstra (MCD), the Kaldi fbank front end of CAM++, log-energy envelopes and
//! time alignment of two renditions of the same speech.
//!
//! Nothing here runs on a real-time thread: it is written for clarity and exactness (f64 maths, a
//! 10 s clip takes a few milliseconds in a release build). The Kaldi fbank reproduces
//! `torchaudio.compliance.kaldi.fbank` with the defaults CAM++ was trained on (the export contract
//! in `scripts/vc-templates/export_bench.py`); a unit test pins it against values computed with
//! torchaudio.

use realfft::RealFftPlanner;

use super::e2e_loopback::xcorr_lags;

/// Hann window of `n` samples, symmetric (`periodic = False`: 0 at both ends).
pub fn hann_symmetric(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let d = (n - 1) as f64;
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / d).cos())
        .collect()
}

/// Power spectra (`|rfft|²`, `n_fft / 2 + 1` bins) of frames of `x`: `win` samples every `hop`,
/// starting at sample 0 while the frame fits (Kaldi's `snip_edges`), each multiplied by `window`
/// (`win` long) and zero-padded to `n_fft`. A clip shorter than one frame gives no frames.
pub fn power_spectra(x: &[f32], win: usize, hop: usize, n_fft: usize, window: &[f64]) -> Vec<Vec<f64>> {
    frames(x.len(), win, hop)
        .map(|start| {
            let frame: Vec<f64> = x[start..start + win].iter().map(|&v| f64::from(v)).collect();
            power_spectrum(&frame, window, n_fft)
        })
        .collect()
}

/// Frame starts: `0, hop, 2 hop, ...` while `start + win <= len`.
pub fn frames(len: usize, win: usize, hop: usize) -> impl Iterator<Item = usize> {
    let count = if win == 0 || len < win { 0 } else { 1 + (len - win) / hop.max(1) };
    (0..count).map(move |i| i * hop.max(1))
}

/// `|rfft(frame · window, n_fft)|²`.
fn power_spectrum(frame: &[f64], window: &[f64], n_fft: usize) -> Vec<f64> {
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut input = fft.make_input_vec();
    for ((d, &s), &w) in input.iter_mut().zip(frame).zip(window) {
        *d = s * w;
    }
    let mut spectrum = fft.make_output_vec();
    // The buffers have exactly the planned lengths.
    fft.process(&mut input, &mut spectrum).expect("fft sizes");
    spectrum.iter().map(|c| c.norm_sqr()).collect()
}

/// Kaldi / HTK mel scale (`1127 ln(1 + f / 700)`).
pub fn mel(hz: f64) -> f64 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// Triangular mel filters, linear in mel (Kaldi's `get_mel_banks`, no VTLN), over the FFT bins
/// `0..n_fft / 2` (the Nyquist bin gets no weight, as in Kaldi). `filters[b][k]` weights bin `k`.
pub fn mel_filters(n_mels: usize, n_fft: usize, sample_rate: u32, low_hz: f64, high_hz: f64) -> Vec<Vec<f64>> {
    let bins = n_fft / 2 + 1;
    let bin_width = f64::from(sample_rate) / n_fft as f64;
    let (lo, hi) = (mel(low_hz), mel(high_hz));
    let delta = (hi - lo) / (n_mels + 1) as f64;
    (0..n_mels)
        .map(|b| {
            let left = lo + b as f64 * delta;
            let center = left + delta;
            let right = center + delta;
            (0..bins)
                .map(|k| {
                    if k >= n_fft / 2 {
                        return 0.0;
                    }
                    let m = mel(bin_width * k as f64);
                    let up = (m - left) / (center - left);
                    let down = (right - m) / (right - center);
                    up.min(down).max(0.0)
                })
                .collect()
        })
        .collect()
}

/// Real cepstrum of a log spectrum sampled on `K = log_bands.len()` bands centred at
/// `ω_k = π (k + ½) / K`: `c_n = (1/K) Σ_k x_k cos(n ω_k)`, the discrete form of
/// `c_n = (1/π) ∫₀^π ln|X(ω)| cos(nω) dω`, so `ln|X(ω)| = c_0 + 2 Σ_{n≥1} c_n cos(nω)` and the MCD
/// constant `10 / ln 10 · √2` applies. (The orthonormal DCT-II is `√(2K)` times larger for
/// `n ≥ 1`, which would inflate MCD about 9× with 40 bands.) First `n_out` coefficients.
pub fn cepstrum(log_bands: &[f64], n_out: usize) -> Vec<f64> {
    let k = log_bands.len();
    if k == 0 {
        return vec![0.0; n_out];
    }
    let kf = k as f64;
    (0..n_out)
        .map(|n| {
            log_bands
                .iter()
                .enumerate()
                .map(|(i, &v)| v * (std::f64::consts::PI * n as f64 * (i as f64 + 0.5) / kf).cos())
                .sum::<f64>()
                / kf
        })
        .collect()
}

/// Mel-cepstrum analysis settings (MCD).
#[derive(Debug, Clone, Copy)]
pub struct MelCepConfig {
    pub frame_ms: f64,
    pub hop_ms: f64,
    pub n_mels: usize,
    /// Coefficients kept (`c0 .. c{order}`); MCD uses `c1 .. c{order}`.
    pub order: usize,
    pub low_hz: f64,
    /// Upper band edge (capped at Nyquist).
    pub high_hz: f64,
    /// Dynamic range of each frame's log-mel spectrum (dB below its loudest band): bands with
    /// (nearly) no energy clamp to this floor, so resampler noise floors and empty bands of
    /// band-limited audio do not dominate the distance, and a gain change moves only `c0`.
    pub range_db: f64,
}

impl Default for MelCepConfig {
    fn default() -> Self {
        Self {
            frame_ms: 25.0,
            hop_ms: 10.0,
            n_mels: 40,
            order: 24,
            low_hz: 20.0,
            high_hz: 8_000.0,
            range_db: 60.0,
        }
    }
}

/// Mel-cepstra of `x` (frames × `order + 1`: the real [`cepstrum`] of the natural-log *amplitude*
/// mel spectrum, so the MCD constant `10 / ln 10 · √2` applies and the scale matches SPTK-style
/// mel-cepstra; every band floored at `range_db` under the frame's loudest band) and each frame's
/// energy in dB.
pub fn mel_cepstra(x: &[f32], sample_rate: u32, cfg: &MelCepConfig) -> (Vec<Vec<f64>>, Vec<f64>) {
    let sr = f64::from(sample_rate);
    let win = ((cfg.frame_ms / 1000.0) * sr).round().max(16.0) as usize;
    let hop = ((cfg.hop_ms / 1000.0) * sr).round().max(1.0) as usize;
    let n_fft = win.next_power_of_two();
    let window = hann_symmetric(win);
    let high = cfg.high_hz.min(sr / 2.0);
    let filters = mel_filters(cfg.n_mels, n_fft, sample_rate, cfg.low_hz, high);
    let spectra = power_spectra(x, win, hop, n_fft, &window);
    let mut ceps = Vec::with_capacity(spectra.len());
    let mut energy = Vec::with_capacity(spectra.len());
    for p in &spectra {
        let total: f64 = p.iter().sum::<f64>() / (n_fft as f64 * win as f64);
        energy.push(10.0 * (total + 1e-12).log10());
        let mel: Vec<f64> = filters.iter().map(|f| f.iter().zip(p).map(|(w, e)| w * e).sum::<f64>()).collect();
        let loudest = mel.iter().cloned().fold(0.0f64, f64::max);
        let floor = (loudest * 10f64.powf(-cfg.range_db / 10.0)).max(1e-30);
        let log_mel: Vec<f64> = mel.iter().map(|&e| 0.5 * e.max(floor).ln()).collect();
        ceps.push(cepstrum(&log_mel, cfg.order + 1));
    }
    (ceps, energy)
}

// ---------------------------------------------------------------------------------------------
// F0-adaptive spectral envelope (scorer v2's envelope MCD)
// ---------------------------------------------------------------------------------------------

/// Lowest F0 the envelope analysis adapts to (WORLD's default floor): the longest window is 3 / 71 s.
pub const ENV_F0_FLOOR: f64 = 71.0;
/// Highest F0 it adapts to.
pub const ENV_F0_CEIL: f64 = 800.0;
/// F0 of unvoiced frames: a three-period window of 120 Hz is the STFT MCD's 25 ms frame (WORLD's
/// 500 Hz default gives 6 ms windows, whose noise estimates vary too much between renditions).
pub const ENV_F0_UNVOICED: f64 = 120.0;
/// CheapTrick's compensation lifter coefficient.
const ENV_Q1: f64 = -0.15;
/// FFT size of the envelope analysis at 16 kHz (`2^(1 + floor(log2(3 · 16000 / 71)))`).
pub const ENV_FFT: usize = 1024;

/// CheapTrick-style power envelope (WORLD, Morise 2015) at 16 kHz.
pub struct EnvelopeAnalyzer {
    fwd: std::sync::Arc<dyn realfft::RealToComplex<f64>>,
    inv: std::sync::Arc<dyn realfft::ComplexToReal<f64>>,
    frame: Vec<f64>,
    spec: Vec<realfft::num_complex::Complex<f64>>,
    cep: Vec<f64>,
}

impl Default for EnvelopeAnalyzer {
    fn default() -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let fwd = planner.plan_fft_forward(ENV_FFT);
        let inv = planner.plan_fft_inverse(ENV_FFT);
        let frame = fwd.make_input_vec();
        let spec = fwd.make_output_vec();
        let cep = inv.make_output_vec();
        Self { fwd, inv, frame, spec, cep }
    }
}

impl EnvelopeAnalyzer {
    /// The power envelope (`ENV_FFT / 2 + 1` bins) of 16 kHz `x` around sample `centre` at `f0` Hz
    /// (0 or not finite = unvoiced): a Hann window of three periods normalised to unit energy, the
    /// windowed mean removed, `|FFT|²`, a rectangular smoothing over `2 f0 / 3` (mirrored at DC and
    /// Nyquist), then the log spectrum liftered by `sin(π f0 q) / (π f0 q)` and the compensation
    /// lifter `(1 − 2 q1) + 2 q1 cos(2π f0 q)` (q1 = −0.15). Samples outside `x` are zero. No
    /// dither and no DC correction (a deterministic distance, not a vocoder).
    pub fn power(&mut self, x: &[f32], centre: isize, f0: f64, out: &mut Vec<f64>) {
        const SR: f64 = 16_000.0;
        let f0 = if f0.is_finite() && f0 > 0.0 { f0.clamp(ENV_F0_FLOOR, ENV_F0_CEIL) } else { ENV_F0_UNVOICED };
        let half = (1.5 * SR / f0).round() as isize;
        let len = (2 * half + 1) as usize;
        let mut w: Vec<f64> = (-half..=half)
            .map(|i| 0.5 * (std::f64::consts::PI * i as f64 / (1.5 * SR / f0)).cos() + 0.5)
            .collect();
        let norm = w.iter().map(|v| v * v).sum::<f64>().sqrt();
        w.iter_mut().for_each(|v| *v /= norm);
        let seg = |i: isize| -> f64 {
            let p = centre + i;
            if p >= 0 && (p as usize) < x.len() {
                f64::from(x[p as usize])
            } else {
                0.0
            }
        };
        let mut wf: Vec<f64> = (-half..=half).zip(&w).map(|(i, &wi)| seg(i) * wi).collect();
        let (sum_wf, sum_w) = (wf.iter().sum::<f64>(), w.iter().sum::<f64>());
        for (v, &wi) in wf.iter_mut().zip(&w) {
            *v -= wi * (sum_wf / sum_w);
        }
        self.frame.iter_mut().for_each(|v| *v = 0.0);
        for (d, &s) in self.frame.iter_mut().zip(wf.iter().take(len.min(ENV_FFT))) {
            *d = s;
        }
        self.fwd.process(&mut self.frame, &mut self.spec).expect("fft sizes");
        let p: Vec<f64> = self.spec.iter().map(|c| c.norm_sqr()).collect();
        let smoothed = linear_smoothing(&p, 2.0 * f0 / 3.0, SR, ENV_FFT);
        // Liftering: real cepstrum (c2r of the log spectrum), lifters, back (r2c).
        let bins = ENV_FFT / 2 + 1;
        for (c, &v) in self.spec.iter_mut().zip(&smoothed) {
            *c = realfft::num_complex::Complex::new(v.max(1e-20).ln(), 0.0);
        }
        self.spec[0].im = 0.0;
        self.spec[bins - 1].im = 0.0;
        self.inv.process(&mut self.spec, &mut self.cep).expect("ifft sizes");
        let n = ENV_FFT as f64;
        for (i, c) in self.cep.iter_mut().enumerate() {
            let q = (i.min(ENV_FFT - i)) as f64 / SR;
            let smooth = if q == 0.0 {
                1.0
            } else {
                let a = std::f64::consts::PI * f0 * q;
                a.sin() / a
            };
            let comp = (1.0 - 2.0 * ENV_Q1) + 2.0 * ENV_Q1 * (2.0 * std::f64::consts::PI * f0 * q).cos();
            *c *= smooth * comp / n;
        }
        self.frame.copy_from_slice(&self.cep);
        self.fwd.process(&mut self.frame, &mut self.spec).expect("fft sizes");
        out.clear();
        out.extend(self.spec.iter().map(|c| c.re.exp()));
    }
}

/// WORLD's `LinearSmoothing`: the power spectrum `p` (`n_fft / 2 + 1` bins at `sr`) averaged over a
/// rectangle `width_hz` wide, mirrored at DC and Nyquist, with the cumulative integral linearly
/// interpolated at fractional edges.
fn linear_smoothing(p: &[f64], width_hz: f64, sr: f64, n_fft: usize) -> Vec<f64> {
    let bins = n_fft / 2 + 1;
    let df = sr / n_fft as f64;
    let half_w = width_hz / 2.0;
    let pad = ((half_w / df).ceil() as usize + 2).min(bins - 2);
    // ext = p[pad..=1] (reversed), p, p[bins-2 ..= bins-1-pad] (reversed).
    let mut ext: Vec<f64> = (1..=pad).rev().map(|j| p[j]).collect();
    ext.extend_from_slice(&p[..bins]);
    ext.extend((1..=pad).map(|j| p[bins - 1 - j]));
    let mut cum = Vec::with_capacity(ext.len() + 1);
    cum.push(0.0);
    let mut acc = 0.0;
    for &v in &ext {
        acc += v * df;
        cum.push(acc);
    }
    // Edge `e` of the extended grid sits at `(e - pad) df - df / 2`.
    let interp = |f: f64| -> f64 {
        let pos = (f + df / 2.0) / df + pad as f64;
        if pos <= 0.0 {
            return cum[0];
        }
        let last = (cum.len() - 1) as f64;
        if pos >= last {
            return cum[cum.len() - 1];
        }
        let i = pos.floor() as usize;
        let t = pos - i as f64;
        cum[i] * (1.0 - t) + cum[i + 1] * t
    };
    (0..bins)
        .map(|k| {
            let f = k as f64 * df;
            (interp(f + half_w) - interp(f - half_w)) / width_hz
        })
        .collect()
}

/// Mel-cepstra of the F0-adaptive envelope of 16 kHz `x` on the STFT MCD's frame grid (frame `i`
/// spans `[160 i, 160 i + 400)`, centred at `160 i + 200`, its F0 the 10 ms analysis frame nearest
/// the centre: `f0[(160 i + 280) / 160]`, 0 = unvoiced), through the same 40 Kaldi mel bands and
/// cepstrum as [`mel_cepstra`]; and the energy (dB) of the plain 25 ms frames (the speech mask).
pub fn envelope_mel_cepstra(x: &[f32], f0: &[f32], cfg: &MelCepConfig) -> (Vec<Vec<f64>>, Vec<f64>) {
    const SR: u32 = 16_000;
    let (win, hop) = (400usize, 160usize);
    let filters = mel_filters(cfg.n_mels, ENV_FFT, SR, cfg.low_hz, cfg.high_hz.min(f64::from(SR) / 2.0));
    let window = hann_symmetric(win);
    let mut analyzer = EnvelopeAnalyzer::default();
    let mut env = Vec::with_capacity(ENV_FFT / 2 + 1);
    let mut ceps = Vec::new();
    let mut energy = Vec::new();
    for s in frames(x.len(), win, hop) {
        let k = (s + 280) / hop;
        let fv = f0.get(k).map_or(0.0, |&v| f64::from(v));
        analyzer.power(x, (s + 200) as isize, fv, &mut env);
        ceps.push(log_mel_cepstrum(&filters, &env, cfg));
        let frame: Vec<f64> = x[s..s + win].iter().map(|&v| f64::from(v)).collect();
        let p = power_spectrum(&frame, &window, 512);
        energy.push(10.0 * (p.iter().sum::<f64>() / (512.0 * win as f64) + 1e-12).log10());
    }
    (ceps, energy)
}

/// [`envelope_mel_cepstra`]'s cepstra of frames of 16 kHz `x` centred at `centres` (samples of `x`;
/// zeros outside it) with F0 `f0s` (Hz per frame, 0 = unvoiced): the offset-corrected envelope MCD
/// reads the test at per-frame shifts.
pub fn envelope_mel_cepstra_at(x: &[f32], centres: &[i64], f0s: &[f64], cfg: &MelCepConfig) -> Vec<Vec<f64>> {
    const SR: u32 = 16_000;
    let filters = mel_filters(cfg.n_mels, ENV_FFT, SR, cfg.low_hz, cfg.high_hz.min(f64::from(SR) / 2.0));
    let mut analyzer = EnvelopeAnalyzer::default();
    let mut env = Vec::with_capacity(ENV_FFT / 2 + 1);
    centres
        .iter()
        .zip(f0s)
        .map(|(&c, &f)| {
            analyzer.power(x, c as isize, f, &mut env);
            log_mel_cepstrum(&filters, &env, cfg)
        })
        .collect()
}

/// [`mel_cepstra`]'s cepstra of the `frame_ms` windows of `x` starting at `starts` (samples of `x`;
/// zeros outside it), with the same per-frame arithmetic: starts `0, hop, 2 hop, ...` give
/// [`mel_cepstra`]'s values bit for bit.
pub fn mel_cepstra_at(x: &[f32], sample_rate: u32, starts: &[i64], cfg: &MelCepConfig) -> Vec<Vec<f64>> {
    let sr = f64::from(sample_rate);
    let win = ((cfg.frame_ms / 1000.0) * sr).round().max(16.0) as usize;
    let n_fft = win.next_power_of_two();
    let window = hann_symmetric(win);
    let filters = mel_filters(cfg.n_mels, n_fft, sample_rate, cfg.low_hz, cfg.high_hz.min(sr / 2.0));
    starts
        .iter()
        .map(|&s| {
            let frame: Vec<f64> = (0..win as i64)
                .map(|k| usize::try_from(s + k).ok().and_then(|i| x.get(i)).map_or(0.0, |&v| f64::from(v)))
                .collect();
            log_mel_cepstrum(&filters, &power_spectrum(&frame, &window, n_fft), cfg)
        })
        .collect()
}

/// The cepstrum of the log-amplitude mel spectrum of power spectrum `p` (each band floored at
/// `cfg.range_db` under the loudest), as [`mel_cepstra`] computes it per frame.
fn log_mel_cepstrum(filters: &[Vec<f64>], p: &[f64], cfg: &MelCepConfig) -> Vec<f64> {
    let mel: Vec<f64> = filters.iter().map(|f| f.iter().zip(p).map(|(w, e)| w * e).sum::<f64>()).collect();
    let loudest = mel.iter().cloned().fold(0.0f64, f64::max);
    let floor = (loudest * 10f64.powf(-cfg.range_db / 10.0)).max(1e-30);
    let log_mel: Vec<f64> = mel.iter().map(|&e| 0.5 * e.max(floor).ln()).collect();
    cepstrum(&log_mel, cfg.order + 1)
}

/// `x` padded with `left` / `right` samples of reflection on each side (`numpy.pad(mode =
/// "reflect")`: no edge repeat; repeated reflection for pads longer than the clip).
pub fn reflect_pad(x: &[f32], left: usize, right: usize) -> Vec<f32> {
    let n = x.len();
    if n <= 1 {
        return std::iter::repeat_n(x.first().copied().unwrap_or(0.0), left + n + right).collect();
    }
    let period = 2 * (n as isize - 1);
    let at = |i: isize| -> f32 {
        let mut j = i.rem_euclid(period);
        if j >= n as isize {
            j = period - j;
        }
        x[j as usize]
    };
    (-(left as isize)..(n + right) as isize).map(at).collect()
}

// ---------------------------------------------------------------------------------------------
// Kaldi fbank (CAM++ front end)
// ---------------------------------------------------------------------------------------------

/// `torchaudio.compliance.kaldi.fbank(x * 32768, num_mel_bins=n_mels, sample_frequency=16000,
/// dither=0)`: 25 ms Povey-windowed frames every 10 ms (snip edges), DC removed per frame,
/// pre-emphasis 0.97 (first sample replicated), 512-point power spectrum, `n_mels` Kaldi mel
/// filters from 20 Hz to Nyquist, natural log floored at f32 epsilon. `x` is 16 kHz audio in
/// [-1, 1]. Returns `frames × n_mels`, row-major.
pub fn kaldi_fbank(x: &[f32], n_mels: usize) -> (Vec<f32>, usize) {
    const SR: u32 = 16_000;
    const WIN: usize = 400;
    const HOP: usize = 160;
    const N_FFT: usize = 512;
    const PREEMPH: f64 = 0.97;
    let povey: Vec<f64> = hann_symmetric(WIN).into_iter().map(|w| w.powf(0.85)).collect();
    let filters = mel_filters(n_mels, N_FFT, SR, 20.0, f64::from(SR) / 2.0);
    let eps = f64::from(f32::EPSILON);
    let mut out = Vec::new();
    let mut count = 0;
    let mut frame = vec![0.0f64; WIN];
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut input = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();
    for start in frames(x.len(), WIN, HOP) {
        for (d, &s) in frame.iter_mut().zip(&x[start..start + WIN]) {
            *d = f64::from(s) * 32_768.0;
        }
        let mean = frame.iter().sum::<f64>() / WIN as f64;
        frame.iter_mut().for_each(|v| *v -= mean);
        // Pre-emphasis with the first sample replicated (torchaudio pads with `replicate`).
        for i in (1..WIN).rev() {
            frame[i] -= PREEMPH * frame[i - 1];
        }
        frame[0] -= PREEMPH * frame[0];
        input.iter_mut().for_each(|v| *v = 0.0);
        for ((d, &s), &w) in input.iter_mut().zip(&frame).zip(&povey) {
            *d = s * w;
        }
        fft.process(&mut input, &mut spectrum).expect("fft sizes");
        for f in &filters {
            let e: f64 = f.iter().zip(&spectrum).map(|(w, c)| w * c.norm_sqr()).sum();
            out.push(e.max(eps).ln() as f32);
        }
        count += 1;
    }
    (out, count)
}

// ---------------------------------------------------------------------------------------------
// Envelopes and alignment
// ---------------------------------------------------------------------------------------------

/// Log energy (dB) of consecutive `hop`-sample frames.
pub fn log_energy_envelope(x: &[f32], hop: usize) -> Vec<f64> {
    x.chunks(hop.max(1))
        .map(|c| {
            let e = c.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / c.len().max(1) as f64;
            10.0 * (e + 1e-12).log10()
        })
        .collect()
}

/// How two renditions of the same audio line up.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Alignment {
    /// `test[n + lag] ≈ reference[n]` (samples; may be negative, sub-sample when refined).
    pub lag: f64,
    /// Normalized correlation at `lag` (waveform when refined, else the energy envelopes).
    pub correlation: f64,
    /// Whether the waveforms confirmed the envelope estimate (similar signals: same model).
    pub waveform: bool,
}

/// Normalized cross-correlation of two equal-length sequences.
fn ncc(a: &[f64], b: &[f64]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa <= 0.0 || bb <= 0.0 {
        0.0
    } else {
        ab / (aa * bb).sqrt()
    }
}

/// Envelope correlation at an integer frame lag (`test[i + lag]` against `reference[i]`).
fn envelope_ncc(reference: &[f64], test: &[f64], lag: isize) -> (f64, usize) {
    let start = (-lag).max(0) as usize;
    let end = (reference.len() as isize).min(test.len() as isize - lag).max(0) as usize;
    if end <= start + 4 {
        return (f64::NEG_INFINITY, 0);
    }
    let a = &reference[start..end];
    let b = &test[(start as isize + lag) as usize..(end as isize + lag) as usize];
    let ma = a.iter().sum::<f64>() / a.len() as f64;
    let mb = b.iter().sum::<f64>() / b.len() as f64;
    let a: Vec<f64> = a.iter().map(|v| v - ma).collect();
    let b: Vec<f64> = b.iter().map(|v| v - mb).collect();
    (ncc(&a, &b), a.len())
}

/// Align `test` to `reference` (same rate `sample_rate`), searching `|lag| <= max_lag` samples:
/// first on 10 ms log-energy envelopes (robust to a change of voice), then, when
/// `refine_waveform`, on the waveforms within one envelope frame of it (sample-accurate for two
/// outputs of the same model). The envelope floor is 60 dB under each clip's peak frame, so
/// silence does not dominate.
pub fn align(reference: &[f32], test: &[f32], sample_rate: u32, max_lag: usize, refine_waveform: bool) -> Option<Alignment> {
    let hop = (sample_rate as usize / 100).max(1);
    let env = |x: &[f32]| {
        let e = log_energy_envelope(x, hop);
        let peak = e.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        e.into_iter().map(|v| v.max(peak - 60.0)).collect::<Vec<f64>>()
    };
    let (er, et) = (env(reference), env(test));
    if er.len() < 8 || et.len() < 8 {
        return None;
    }
    let max_frames = (max_lag / hop) as isize + 1;
    // Require the overlap to cover half the shorter clip, so edge lags with a few frames of
    // overlap cannot win by chance.
    let min_overlap = er.len().min(et.len()) / 2;
    let mut best: Option<(isize, f64)> = None;
    for lag in -max_frames..=max_frames {
        let (c, n) = envelope_ncc(&er, &et, lag);
        if n >= min_overlap.max(4) && best.is_none_or(|(_, b)| c > b) {
            best = Some((lag, c));
        }
    }
    let (lag_frames, env_corr) = best?;
    let coarse = lag_frames * hop as isize;
    let fallback = Alignment {
        lag: coarse as f64,
        correlation: env_corr,
        waveform: false,
    };
    if !refine_waveform {
        return Some(fallback);
    }
    // Waveform refinement: lags coarse - hop ..= coarse + hop, computed as non-negative lags of a
    // shifted pair (xcorr_lags searches 0..=max).
    let lo = coarse - hop as isize;
    let (r_seg, t_seg): (&[f32], &[f32]) = if lo >= 0 {
        (reference, test.get(lo as usize..).unwrap_or(&[]))
    } else {
        // test[n + lag] ~ reference[n] with lag = lo + k  <=>  test'[n + k] ~ reference'[n] where
        // reference' = reference[-lo..] and test' = test.
        (reference.get((-lo) as usize..).unwrap_or(&[]), test)
    };
    // A bounded reference window (the loudest ~4 s) keeps the FFT small on long clips; it must
    // fit in the test with the searched lags after it (a short clip whose test ends early, as a
    // recorded model stream does, still gets a window: P3B-B).
    let avail = t_seg.len().saturating_sub(2 * hop).min(r_seg.len());
    let win = avail.min(sample_rate as usize * 4);
    if win < hop * 4 {
        return Some(fallback);
    }
    let start = loudest_window(&r_seg[..avail], win, hop);
    let reference_win = &r_seg[start..start + win];
    let recorded = &t_seg[start.min(t_seg.len())..];
    let lags = xcorr_lags(reference_win, recorded, 2 * hop);
    if lags.is_empty() {
        return Some(fallback);
    }
    let (k, _) = lags
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("non-empty");
    let lag = lo + k as isize;
    // Normalized correlation at the integer lag, then parabolic sub-sample refinement.
    let corr_at = |lag: isize| -> f64 {
        let s = (-lag).max(0) as usize;
        let e = (reference.len() as isize).min(test.len() as isize - lag).max(0) as usize;
        if e <= s + 16 {
            return 0.0;
        }
        let a: Vec<f64> = reference[s..e].iter().map(|&v| f64::from(v)).collect();
        let b: Vec<f64> = test[(s as isize + lag) as usize..(e as isize + lag) as usize]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        ncc(&a, &b)
    };
    let c0 = corr_at(lag);
    if c0 < 0.3 {
        return Some(fallback);
    }
    let frac = if k > 0 && k + 1 < lags.len() {
        let (ym, y0, yp) = (lags[k - 1], lags[k], lags[k + 1]);
        let d = ym - 2.0 * y0 + yp;
        if d.abs() > 1e-12 {
            (0.5 * (ym - yp) / d).clamp(-0.5, 0.5)
        } else {
            0.0
        }
    } else {
        0.0
    };
    Some(Alignment {
        lag: lag as f64 + frac,
        correlation: c0,
        waveform: true,
    })
}

/// Start of the `win`-sample window of `x` with the most energy (searched on a `hop` grid).
fn loudest_window(x: &[f32], win: usize, hop: usize) -> usize {
    if x.len() <= win {
        return 0;
    }
    let env: Vec<f64> = x
        .chunks(hop.max(1))
        .map(|c| c.iter().map(|&v| f64::from(v) * f64::from(v)).sum())
        .collect();
    let span = win / hop.max(1);
    let mut sum: f64 = env.iter().take(span).sum();
    let (mut best, mut best_at) = (sum, 0usize);
    for i in span..env.len() {
        sum += env[i] - env[i - span];
        if sum > best && (i + 1 - span) * hop + win <= x.len() {
            best = sum;
            best_at = i + 1 - span;
        }
    }
    best_at * hop
}

/// Overlapping parts of `reference` and `test` for `test[n + lag] ≈ reference[n]` (integer lag).
#[cfg(test)]
pub fn overlap<'a>(reference: &'a [f32], test: &'a [f32], lag: isize) -> (&'a [f32], &'a [f32]) {
    let start = (-lag).max(0) as usize;
    let end = (reference.len() as isize).min(test.len() as isize - lag).max(0) as usize;
    if end <= start {
        return (&[], &[]);
    }
    (&reference[start..end], &test[(start as isize + lag) as usize..(end as isize + lag) as usize])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::bench::signals::{synth, SplitMix64};
    use crate::services::bench::spec::SignalKind;

    /// The cepstrum has the textbook scale: `ln|1 - a e^{-jω}| = -Σ aⁿ/n cos(nω)`, so
    /// `c_n = -aⁿ / (2n)` and `c_0 = 0`; a constant log spectrum is `c_0` alone.
    #[test]
    fn cepstrum_has_the_real_cepstrum_scale_and_frames_snip_edges() {
        let k = 40;
        let a: f64 = 0.5;
        let bands: Vec<f64> = (0..k)
            .map(|i| {
                let w = std::f64::consts::PI * (i as f64 + 0.5) / k as f64;
                ((1.0 - a * w.cos()).powi(2) + (a * w.sin()).powi(2)).sqrt().ln()
            })
            .collect();
        let c = cepstrum(&bands, 25);
        assert!(c[0].abs() < 1e-12, "c0 {}", c[0]);
        for (n, &v) in c.iter().enumerate().skip(1) {
            let want = -a.powi(n as i32) / (2.0 * n as f64);
            assert!((v - want).abs() < 1e-12, "c{n}: {v} vs {want}");
        }
        let flat = cepstrum(&[1.5; 40], 4);
        assert!((flat[0] - 1.5).abs() < 1e-12 && flat[1..].iter().all(|v| v.abs() < 1e-12), "{flat:?}");
        assert_eq!(frames(1000, 400, 160).collect::<Vec<_>>(), vec![0, 160, 320, 480]);
        assert_eq!(frames(399, 400, 160).count(), 0);
    }

    #[test]
    fn mel_filters_cover_the_band_and_skip_nyquist() {
        let f = mel_filters(80, 512, 16_000, 20.0, 8_000.0);
        assert_eq!(f.len(), 80);
        assert!(f.iter().all(|row| row.len() == 257 && row[256] == 0.0));
        // Every filter has weight, peaks at most 1, and their centres rise.
        let centres: Vec<usize> = f
            .iter()
            .map(|row| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0)
            .collect();
        assert!(f.iter().all(|row| row.iter().sum::<f64>() > 0.0 && row.iter().all(|&w| w <= 1.0)));
        assert!(centres.windows(2).all(|w| w[0] <= w[1]), "{centres:?}");
    }

    /// Values from torchaudio 2.3.1 (`torchaudio.compliance.kaldi.fbank(x * 32768,
    /// num_mel_bins=80, sample_frequency=16000, dither=0.0)`) for
    /// `x[n] = 0.3 sin(2π 440 n / 16000) + 0.1 sin(2π 3000 n / 16000)`, 0.1 s.
    #[test]
    fn kaldi_fbank_matches_torchaudio() {
        let x: Vec<f32> = (0..1_600)
            .map(|n| {
                let t = n as f64 / 16_000.0;
                (0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin() + 0.1 * (2.0 * std::f64::consts::PI * 3_000.0 * t).sin()) as f32
            })
            .collect();
        let (fb, frames) = kaldi_fbank(&x, 80);
        assert_eq!(frames, 8);
        assert_eq!(fb.len(), 8 * 80);
        for &(frame, bin, want) in TORCHAUDIO_FBANK {
            let got = f64::from(fb[frame * 80 + bin]);
            // torchaudio computes in f32: bins with almost no energy (log < 2, near Nyquist here)
            // carry its rounding; the speech bins match to 2e-3.
            let tol = if want > 2.0 { 2e-3 } else { 1e-2 };
            assert!((got - want).abs() < tol, "frame {frame} bin {bin}: {got} vs torchaudio {want}");
        }
    }

    /// `(frame, bin, value)` probes of [`kaldi_fbank_matches_torchaudio`].
    const TORCHAUDIO_FBANK: &[(usize, usize, f64)] = &[
        (0, 0, 8.219269752502441),
        (0, 5, 10.669108390808105),
        (0, 13, 23.430021286010742),
        (0, 27, 6.799883842468262),
        (0, 40, 4.735416412353516),
        (0, 55, 12.977045059204102),
        (0, 70, 1.5985246896743774),
        (0, 79, -1.1124238967895508),
        (3, 0, 8.825907707214355),
        (3, 5, 10.710000038146973),
        (3, 13, 23.429906845092773),
        (3, 27, 6.7551188468933105),
        (3, 40, 5.149441719055176),
        (3, 55, 12.975822448730469),
        (3, 70, 1.5122716426849365),
        (3, 79, -1.2947748899459839),
        (7, 0, 9.171269416809082),
        (7, 5, 10.758438110351562),
        (7, 13, 23.429790496826172),
        (7, 27, 6.5424041748046875),
        (7, 40, 4.614044666290283),
        (7, 55, 12.976795196533203),
        (7, 70, 1.5375685691833496),
        (7, 79, -1.404876947402954)
    ];

    #[test]
    fn mel_cepstra_of_identical_clips_agree_and_energy_tracks_level() {
        let v = synth(SignalKind::Voice, 16_000, 2.0, 0.5, 3);
        let (a, ea) = mel_cepstra(&v, 16_000, &MelCepConfig::default());
        let (b, _) = mel_cepstra(&v, 16_000, &MelCepConfig::default());
        assert_eq!(a, b);
        assert_eq!(a[0].len(), 25);
        let quiet: Vec<f32> = v.iter().map(|x| x * 0.1).collect();
        let (c, eq) = mel_cepstra(&quiet, 16_000, &MelCepConfig::default());
        // -20 dB: energies drop by 20 dB, cepstra beyond c0 are gain-invariant.
        let i = ea.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
        assert!((ea[i] - eq[i] - 20.0).abs() < 0.05, "{} vs {}", ea[i], eq[i]);
        for d in 1..25 {
            assert!((a[i][d] - c[i][d]).abs() < 1e-6, "c{d}");
        }
    }

    /// `numpy.pad(x, (3, 5), mode="reflect")`, longer pads reflect again, and the smoothing keeps a
    /// flat spectrum flat (mirrored edges included).
    #[test]
    fn reflect_pad_and_linear_smoothing() {
        assert_eq!(reflect_pad(&[1.0, 2.0, 3.0, 4.0], 3, 5), vec![4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0]);
        assert_eq!(reflect_pad(&[1.0, 2.0], 3, 0), vec![2.0, 1.0, 2.0, 1.0, 2.0]);
        assert_eq!(reflect_pad(&[7.0], 1, 1), vec![7.0; 3]);
        let flat = vec![2.0f64; ENV_FFT / 2 + 1];
        let s = linear_smoothing(&flat, 100.0, 16_000.0, ENV_FFT);
        assert!(s.iter().all(|v| (v - 2.0).abs() < 1e-9), "{:?}", &s[..4]);
        // A single bin spreads over the rectangle, keeping its integral.
        let mut spike = vec![0.0f64; ENV_FFT / 2 + 1];
        spike[100] = 1.0;
        let s = linear_smoothing(&spike, 200.0, 16_000.0, ENV_FFT);
        let df = 16_000.0 / ENV_FFT as f64;
        assert!((s.iter().sum::<f64>() * df - df).abs() < 1e-9);
        assert!(s[100] > 0.0 && s[90] == 0.0 && s[110] == 0.0);
    }

    /// The fbank code is pinned: the CAM++ front end of a fixed signal hashes to the same value in
    /// every build (values rounded to 1e-4, so a libm's last-bit difference does not count). A
    /// change of the fbank changes every speaker score, so it must change this test too.
    #[test]
    fn kaldi_fbank_is_pinned() {
        let x: Vec<f32> = (0..16_000)
            .map(|n| {
                let t = n as f64 / 16_000.0;
                (0.25 * (2.0 * std::f64::consts::PI * 180.0 * t).sin() + 0.08 * (2.0 * std::f64::consts::PI * 1_234.0 * t).sin()) as f32
            })
            .collect();
        let (fb, frames) = kaldi_fbank(&x, 80);
        assert_eq!(frames, 98);
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for v in &fb {
            for b in ((f64::from(*v) * 1e4).round() as i64).to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        assert_eq!(h, KALDI_FBANK_PIN, "kaldi_fbank changed: {h:#018x}");
    }

    /// See [`kaldi_fbank_is_pinned`].
    const KALDI_FBANK_PIN: u64 = 0xf5b8_8b7e_5f1c_ba19;

    /// A delayed copy and a louder, slightly noisy copy are found at the right lag, positive and
    /// negative, to the sample.
    #[test]
    fn align_finds_integer_lags_both_ways() {
        let v = synth(SignalKind::Voice, 16_000, 4.0, 0.5, 11);
        let mut rng = SplitMix64::new(4);
        for lag in [0isize, 37, 1_234, -815] {
            let test: Vec<f32> = if lag >= 0 {
                std::iter::repeat_n(0.0, lag as usize).chain(v.iter().copied()).collect()
            } else {
                v[(-lag) as usize..].to_vec()
            };
            let test: Vec<f32> = test.iter().map(|x| 1.7 * x + 0.001 * rng.range(-1.0, 1.0) as f32).collect();
            let a = align(&v, &test, 16_000, 4_000, true).expect("aligned");
            assert!(a.waveform, "{a:?}");
            assert!((a.lag - lag as f64).abs() < 0.6, "lag {lag}: {a:?}");
            assert!(a.correlation > 0.99, "{a:?}");
        }
        // Different "voices" (the envelope alone): still within one frame.
        let other: Vec<f32> = std::iter::repeat_n(0.0, 480)
            .chain(synth(SignalKind::Voice, 16_000, 4.0, 0.5, 11).iter().map(|x| x.signum() * x.abs().sqrt() * 0.3))
            .collect();
        let a = align(&v, &other, 16_000, 4_000, false).unwrap();
        assert!((a.lag - 480.0).abs() <= 160.0, "{a:?}");
        let (r, t) = overlap(&v, &other, 480);
        assert_eq!(r.len(), t.len());
        assert_eq!(r.len(), v.len());
        // A short clip whose test ends early (a recorded model stream stops at its last call): the
        // waveform refinement still runs, on the part of the clip the test holds.
        let short = synth(SignalKind::Voice, 16_000, 3.0, 0.5, 12);
        let truncated: Vec<f32> = std::iter::repeat_n(0.0, 2_000).chain(short[..40_000].iter().copied()).collect();
        let a = align(&short, &truncated, 16_000, 8_000, true).expect("aligned");
        assert!(a.waveform && (a.lag - 2_000.0).abs() < 0.6 && a.correlation > 0.99, "{a:?}");
    }
}
