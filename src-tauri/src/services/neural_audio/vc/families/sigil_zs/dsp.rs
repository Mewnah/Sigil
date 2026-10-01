//! Host-side arithmetic of the Sigil-ZS loop, each reproducing what upstream (or the exported graph)
//! computes, so the Rust port matches `scripts/vc-families/seed_vc/sampler.py`:
//!
//! - [`linspace_f32`]: the flow's time grid, `torch.linspace(0, 1, n + 1)` in f32 bit for bit;
//! - [`resample_torchaudio`]: `torchaudio.functional.resample` (Hann-windowed sinc, width 6,
//!   rolloff 0.99, the kernel in f32): upstream's 22.05 → 16 kHz step for the reference;
//! - [`fbank_centered`]: CAM++'s input, `torchaudio.compliance.kaldi.fbank(x, 80 bins, dither 0)`
//!   of the unscaled waveform minus its mean over frames (upstream feeds [-1, 1] audio, not ×32768);
//! - the NSF phase helpers ([`nsf_start_cycles`], [`nsf_cycles_at`], [`harm_phase_for_anchor`]):
//!   the `hift` / `hift_nsf` graphs' per-frame wrapped phase in f32, for partial decode (carry the
//!   window-start phase to the slice) and the phase-continuous join (anchor the phase where two
//!   ticks meet).

use super::preset::{HOP, N_HARM, SR_OUT};

/// `torch.linspace(0, 1, n_steps + 1)` in float32, bit for bit: `step = 1 / n` in f32; the first
/// half counts up from 0, the second half down from 1.
pub fn linspace_f32(n_steps: u32) -> Vec<f32> {
    let n_steps = n_steps.max(1);
    let steps = n_steps as usize + 1;
    let step = 1.0f32 / n_steps as f32;
    let half = steps / 2;
    (0..steps)
        .map(|i| if i < half { i as f32 * step } else { 1.0f32 - step * (steps - i - 1) as f32 })
        .collect()
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// `torchaudio.functional.resample(x, orig, new)` with its defaults (`sinc_interp_hann`,
/// `lowpass_filter_width` 6, `rolloff` 0.99): the polyphase kernel computed in f32 as torch does,
/// the convolution accumulated in f64; `ceil(len · new / orig)` samples.
pub fn resample_torchaudio(x: &[f32], orig: u32, new: u32) -> Vec<f32> {
    if orig == new || x.is_empty() || orig == 0 || new == 0 {
        return x.to_vec();
    }
    let g = gcd(u64::from(orig), u64::from(new));
    let o = (u64::from(orig) / g) as usize;
    let n = (u64::from(new) / g) as usize;
    const WIDTH: f32 = 6.0;
    let base = o.min(n) as f64 * 0.99;
    let width = (f64::from(WIDTH) * o as f64 / base).ceil() as usize;
    let taps = 2 * width + o;
    let base32 = base as f32;
    let scale = (base / o as f64) as f32;
    let pi = std::f32::consts::PI;
    // kernel[j][k]: phase j (0..n), tap k; t = (-j / n + (k - width) / o) * base, clamped.
    let mut kernel = vec![0.0f32; n * taps];
    for j in 0..n {
        let tj = -(j as f32) / n as f32;
        for k in 0..taps {
            let idx = (k as f32 - width as f32) / o as f32;
            let mut t = (tj + idx) * base32;
            t = t.clamp(-WIDTH, WIDTH);
            let w = (t * pi / WIDTH / 2.0).cos();
            let window = w * w;
            let tp = t * pi;
            let s = if tp == 0.0 { 1.0 } else { tp.sin() / tp };
            kernel[j * taps + k] = s * (window * scale);
        }
    }
    let len = x.len();
    let blocks = len / o + 1;
    let target = (n as u64 * len as u64).div_ceil(o as u64) as usize;
    let mut out = Vec::with_capacity(blocks * n);
    // x padded with `width` zeros in front and `width + o` behind.
    let at = |i: usize| -> f64 { i.checked_sub(width).and_then(|p| x.get(p)).map_or(0.0, |&v| f64::from(v)) };
    for b in 0..blocks {
        let start = b * o;
        for j in 0..n {
            let row = &kernel[j * taps..(j + 1) * taps];
            let acc: f64 = row.iter().enumerate().map(|(k, &w)| f64::from(w) * at(start + k)).sum();
            out.push(acc as f32);
        }
        if out.len() >= target {
            break;
        }
    }
    out.truncate(target);
    out
}

/// CAM++'s input for a 16 kHz clip: Kaldi fbank (80 bins, 25 ms Povey frames every 10 ms, dither 0)
/// of the unscaled waveform, minus each bin's mean over frames. Returns `(frames × 80, frames)`.
pub fn fbank_centered(x16k: &[f32]) -> (Vec<f32>, usize) {
    // `bench::dsp::kaldi_fbank` scales by 32768 (torchaudio's int16 convention); upstream feeds the
    // [-1, 1] waveform. Dividing by 2^15 first is exact, so the two scalings cancel bit for bit.
    let scaled: Vec<f32> = x16k.iter().map(|&v| v / 32_768.0).collect();
    let (mut fb, frames) = crate::services::bench::dsp::kaldi_fbank(&scaled, 80);
    if frames == 0 {
        return (fb, 0);
    }
    for b in 0..80 {
        let mean = (0..frames).map(|f| f64::from(fb[f * 80 + b])).sum::<f64>() / frames as f64;
        for f in 0..frames {
            fb[f * 80 + b] = (f64::from(fb[f * 80 + b]) - mean) as f32;
        }
    }
    (fb, frames)
}

#[inline]
fn frac32(x: f32) -> f32 {
    x - x.floor()
}

/// Per-sample phase increment (cycles) of harmonic `h` (0-based) at `f0`, as the graphs compute it.
#[inline]
fn inc(f0: f32, h: usize) -> f32 {
    (f0 * (h + 1) as f32) / SR_OUT as f32
}

/// Each harmonic's phase (cycles, wrapped) before the first sample of `frame`, relative to the
/// window's first sample (`f0` per 256-sample frame): what `harm_phase` must advance by (× 2π) for
/// a slice starting at `frame` to continue the whole window's excitation (`sampler.py`
/// `nsf_start_cycles`).
pub fn nsf_start_cycles(f0: &[f32], frame: usize) -> [f32; N_HARM] {
    let mut out = [0.0f32; N_HARM];
    for (h, o) in out.iter_mut().enumerate() {
        // start_j = frac(cumsum(adv)_j - adv_j) = frac(sum_{i<j} adv_i), the f32 cumsum in order.
        let mut cum = 0.0f32;
        let mut adv_j = 0.0f32;
        for (i, &f) in f0.iter().enumerate().take(frame + 1) {
            let adv = frac32(inc(f, h) * HOP as f32);
            cum += adv;
            if i == frame {
                adv_j = adv;
            }
        }
        *o = frac32(cum - adv_j);
    }
    out
}

/// Phase (cycles in `[0, 1)`) of excitation sample `n` of a window that starts at `harm_phase`
/// (radians): the graph's sine argument is `2π frac(start_j + inc_j (n % 256 + 1)) + harm_phase`
/// with `j = n / 256` (`sampler.py` `nsf_cycles_at`).
pub fn nsf_cycles_at(f0: &[f32], harm_phase: &[f32; N_HARM], n: usize) -> [f64; N_HARM] {
    let j = (n / HOP).min(f0.len().saturating_sub(1));
    let k = (n % HOP + 1) as f32;
    let start = nsf_start_cycles(f0, j);
    let mut out = [0.0f64; N_HARM];
    for h in 0..N_HARM {
        let fj = f0.get(j).copied().unwrap_or(0.0);
        let ph = f64::from(frac32(start[h] + inc(fj, h) * k));
        let c = ph + f64::from(harm_phase[h]) / std::f64::consts::TAU;
        out[h] = c - c.floor();
    }
    out
}

/// `harm_phase` (radians, each in `[-π, π)`) for a window with this F0 whose excitation sample `n`
/// must have phase `target` (cycles): the phase-continuous NSF lever (`sampler.py`
/// `harm_phase_for_anchor`).
pub fn harm_phase_for_anchor(f0: &[f32], n: usize, target: &[f64; N_HARM]) -> [f32; N_HARM] {
    let own = nsf_cycles_at(f0, &[0.0; N_HARM], n);
    let mut out = [0.0f32; N_HARM];
    for h in 0..N_HARM {
        let mut d = target[h] - own[h];
        d -= (d + 0.5).floor();
        out[h] = (std::f64::consts::TAU * d) as f32;
    }
    out
}

/// `harm_phase` carried from the window start to the first sample of `frame` (partial decode
/// without an anchor: the slice continues the whole window's excitation, `sampler.py`'s parity
/// path).
pub fn carry_phase(f0: &[f32], harm_phase: &[f32; N_HARM], frame: usize) -> [f32; N_HARM] {
    let start = nsf_start_cycles(f0, frame);
    let mut out = [0.0f32; N_HARM];
    for h in 0..N_HARM {
        out[h] = (f64::from(harm_phase[h]) + std::f64::consts::TAU * f64::from(start[h])) as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_is_torchs() {
        // torch.linspace(0, 1, 7) in float32 (values recorded with make_vectors.py / sampler.py).
        let l = linspace_f32(6);
        assert_eq!(l.len(), 7);
        assert_eq!((l[0], l[6]), (0.0, 1.0));
        assert_eq!(l[1], 1.0f32 / 6.0);
        assert_eq!(l[4], 1.0f32 - (1.0f32 / 6.0) * 2.0, "the second half counts down from 1");
        for n in [1u32, 4, 10, 30] {
            let l = linspace_f32(n);
            assert!(l.windows(2).all(|w| w[1] > w[0]));
            assert_eq!(*l.last().unwrap(), 1.0);
        }
    }

    #[test]
    fn resample_lengths_and_passband() {
        let tone: Vec<f32> = (0..22_050).map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / 22_050.0).sin() * 0.5).collect();
        let y = resample_torchaudio(&tone, 22_050, 16_000);
        assert_eq!(y.len(), (22_050usize * 320).div_ceil(441));
        // A 440 Hz tone keeps its level and frequency (zero crossings) away from the edges.
        let mid = &y[2_000..14_000];
        let rms = (mid.iter().map(|v| v * v).sum::<f32>() / mid.len() as f32).sqrt();
        assert!((rms - 0.5 / 2f32.sqrt()).abs() < 0.01, "{rms}");
        let crossings = mid.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
        let expect = 440.0 * mid.len() as f32 / 16_000.0;
        assert!((crossings as f32 - expect).abs() <= 1.0, "{crossings} vs {expect}");
        assert_eq!(resample_torchaudio(&tone[..10], 16_000, 16_000), tone[..10].to_vec());
    }

    #[test]
    fn fbank_is_centered_and_unscaled() {
        let x: Vec<f32> = (0..16_000).map(|i| ((i as f32 * 0.05).sin() * 0.3) + ((i * 7919 % 101) as f32 / 101.0 - 0.5) * 0.01).collect();
        let (fb, frames) = fbank_centered(&x);
        assert_eq!(frames, (16_000 - 400) / 160 + 1);
        for b in 0..80 {
            let mean: f64 = (0..frames).map(|f| f64::from(fb[f * 80 + b])).sum::<f64>() / frames as f64;
            assert!(mean.abs() < 1e-4, "bin {b}: {mean}");
        }
        // Unscaled: the same as bench's fbank of x / 32768 minus its mean.
        let (raw, _) = crate::services::bench::dsp::kaldi_fbank(&x.iter().map(|v| v / 32_768.0).collect::<Vec<_>>(), 80);
        let mean0 = (0..frames).map(|f| raw[f * 80]).sum::<f32>() / frames as f32;
        assert!((fb[0] - (raw[0] - mean0)).abs() < 1e-4);
    }

    #[test]
    fn nsf_phase_helpers_agree_with_each_other() {
        let f0 = [0.0f32, 0.0, 110.0, 180.5, 220.25, 0.0, 300.0, 1000.0];
        let hp = [0.0f32, 0.5, -1.0, 2.0, -3.0, 1.5, 0.25, -0.75, 3.1];
        // The phase at a frame start equals the carried phase evaluated at the slice's sample 0 minus
        // one increment (sample 0 of a slice already advances one sample, as in the graph).
        let target: [f64; N_HARM] = std::array::from_fn(|h| h as f64 * 0.1);
        let anchored = harm_phase_for_anchor(&f0, 700, &target);
        let back = nsf_cycles_at(&f0, &anchored, 700);
        for h in 0..N_HARM {
            let d = (back[h] - target[h]).rem_euclid(1.0);
            assert!(d.min(1.0 - d) < 1e-5, "harmonic {h}: {} vs {}", back[h], target[h]);
            assert!(anchored[h].abs() <= std::f32::consts::PI + 1e-6);
        }
        // Carrying the phase to frame 3 and evaluating there equals evaluating the whole window.
        let carried = carry_phase(&f0, &hp, 3);
        let whole = nsf_cycles_at(&f0, &hp, 3 * HOP + 17);
        let slice = nsf_cycles_at(&f0[3..], &carried, 17);
        for h in 0..N_HARM {
            let d = (whole[h] - slice[h]).rem_euclid(1.0);
            assert!(d.min(1.0 - d) < 1e-4, "harmonic {h}: {} vs {}", whole[h], slice[h]);
        }
        // Unvoiced frames advance nothing.
        assert_eq!(nsf_start_cycles(&[0.0; 4], 3), [0.0; N_HARM]);
    }
}
