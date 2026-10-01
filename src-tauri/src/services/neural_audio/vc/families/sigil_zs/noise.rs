//! Sigil-ZS randomness (ADR 0006 "Coherent ticks"): the flow's starting noise `z` and the NSF source
//! noise drawn per absolute mel frame on the content grid, so the frames two overlapping ticks share
//! get the same draws, and the harmonic start phases.
//!
//! Upstream draws a fresh `z`, fresh harmonic phases and fresh sine noise every tick, so the two
//! ticks of a SOLA crossfade are two renditions of the same audio. With the `anchored_noise` lever,
//! tick `k`'s mel frame `j` takes the draws of absolute grid frame `k · F + j` (`F` =
//! `Geometry::grid_frames`), generated once per frame by the engine's counter-based generator
//! (`engine::noise`, the same one the RVC templates use) and cached, so a tick only draws the `F`
//! frames it has not seen. Off, every tick draws from a fresh, far-away span of the same generator
//! (upstream's behaviour: independent draws per tick).

use crate::services::neural_audio::vc::engine::noise::{AnchoredNoise, NoiseFrames, NoiseStream};

use super::preset::{HOP, MEL_BINS, N_HARM};

/// Values of the NSF source noise per mel frame: 9 harmonics × 256 samples (harmonic-major).
const SINE_PER_FRAME: usize = N_HARM * HOP;
/// Offset of the per-tick fresh draws (legacy path): far from any grid frame a stream reaches.
const FRESH_BASE: u64 = 1 << 38;
/// Frames between two fresh ticks' spans (more than any tick's window).
const FRESH_STRIDE: u64 = 1 << 16;

/// The noise of one voice.
pub struct ZsNoise {
    gen: AnchoredNoise,
    seed: u64,
    z: NoiseFrames,
    sine: NoiseFrames,
    /// `[frame][9 × 256]` scratch of the sine window before it is laid out harmonic-major.
    sine_frames: Vec<f32>,
    /// Frames the caches hold.
    cap: usize,
    fresh_tick: u64,
}

impl ZsNoise {
    /// Caches sized for windows of up to `max_frames` mel frames.
    pub fn new(seed: u64, max_frames: usize) -> ZsNoise {
        let gen = AnchoredNoise::new(seed);
        let cap = max_frames.max(1) * 2;
        ZsNoise {
            gen,
            seed,
            z: NoiseFrames::new(gen, NoiseStream::Flow, MEL_BINS, cap),
            sine: NoiseFrames::new(gen, NoiseStream::Source, SINE_PER_FRAME, cap),
            sine_frames: vec![0.0; max_frames.max(1) * SINE_PER_FRAME],
            cap,
            fresh_tick: 0,
        }
    }

    /// Make the caches hold windows of `frames` mel frames (allocates only when they grow).
    pub fn reserve(&mut self, frames: usize) {
        if frames > self.cap {
            let cap = frames.max(1) * 2;
            self.z = NoiseFrames::new(self.gen, NoiseStream::Flow, MEL_BINS, cap);
            self.sine = NoiseFrames::new(self.gen, NoiseStream::Source, SINE_PER_FRAME, cap);
            self.cap = cap;
        }
        if self.sine_frames.len() < frames * SINE_PER_FRAME {
            self.sine_frames.resize(frames * SINE_PER_FRAME, 0.0);
        }
    }

    /// The first grid frame of this tick's window: `k · F` when anchored, else a fresh span.
    pub fn window_origin(&mut self, anchored: bool, tick: u64, grid_frames: usize) -> u64 {
        if anchored {
            tick.wrapping_mul(grid_frames as u64)
        } else {
            self.fresh_tick = self.fresh_tick.wrapping_add(1);
            FRESH_BASE + (self.fresh_tick % (1 << 20)) * FRESH_STRIDE
        }
    }

    /// `z` of the tick's `n_mel` frames from grid frame `origin`, channel-major `[80][n_mel]`.
    pub fn fill_z(&mut self, origin: u64, n_mel: usize, out: &mut [f32]) {
        self.z.copy_window(origin, n_mel, 1.0, true, &mut out[..MEL_BINS * n_mel]);
    }

    /// The NSF source noise of frames `[first, first + frames)` of the window that starts at grid
    /// frame `origin`, laid out as the graphs take it: `[9][frames × 256]`.
    pub fn fill_sine(&mut self, origin: u64, first: usize, frames: usize, out: &mut [f32]) {
        let n = frames * SINE_PER_FRAME;
        if self.sine_frames.len() < n {
            self.sine_frames.resize(n, 0.0);
        }
        self.sine.copy_window(origin + first as u64, frames, 1.0, false, &mut self.sine_frames[..n]);
        let row = frames * HOP;
        for f in 0..frames {
            let src = &self.sine_frames[f * SINE_PER_FRAME..(f + 1) * SINE_PER_FRAME];
            for h in 0..N_HARM {
                out[h * row + f * HOP..h * row + (f + 1) * HOP].copy_from_slice(&src[h * HOP..(h + 1) * HOP]);
            }
        }
    }

    /// Harmonic start phases (radians) for a window without an anchor: the fundamental at 0 and the
    /// harmonics uniform in `[-π, π)` (upstream `SineGen`), fixed per voice when anchored, per tick
    /// otherwise.
    pub fn base_phase(&self, anchored: bool) -> [f32; N_HARM] {
        let key = if anchored { 0 } else { self.fresh_tick };
        let mut out = [0.0f32; N_HARM];
        for (h, o) in out.iter_mut().enumerate().skip(1) {
            let u = unit_uniform(self.seed ^ 0x0005_EED0_FA5E, key.wrapping_mul(16).wrapping_add(h as u64));
            *o = ((u - 0.5) * std::f64::consts::TAU) as f32;
        }
        out
    }
}

/// A uniform in `[0, 1)` from `(seed, index)` (SplitMix64).
fn unit_uniform(seed: u64, index: u64) -> f64 {
    let mut z = seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Overlapping windows of consecutive anchored ticks share their draws frame for frame; fresh
    /// ticks never do.
    #[test]
    fn anchored_windows_share_their_overlap() {
        let (n_mel, f) = (41usize, 9usize);
        let mut a = ZsNoise::new(7, n_mel);
        let mut z0 = vec![0.0; MEL_BINS * n_mel];
        let mut z1 = z0.clone();
        let o0 = a.window_origin(true, 10, f);
        a.fill_z(o0, n_mel, &mut z0);
        let o1 = a.window_origin(true, 11, f);
        a.fill_z(o1, n_mel, &mut z1);
        assert_eq!(o1 - o0, f as u64);
        for c in 0..MEL_BINS {
            for j in 0..n_mel - f {
                assert_eq!(z1[c * n_mel + j], z0[c * n_mel + j + f], "channel {c} frame {j}");
            }
        }
        // Unit Gaussian.
        let mean = z0.iter().map(|&v| f64::from(v)).sum::<f64>() / z0.len() as f64;
        let var = z0.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / z0.len() as f64;
        assert!(mean.abs() < 0.08 && (var - 1.0).abs() < 0.12, "{mean} {var}");
        // Sine noise: frame f of tick 11's window is frame f + F of tick 10's.
        let (first, frames) = (13usize, 28usize);
        let mut s0 = vec![0.0; N_HARM * frames * HOP];
        let mut s1 = s0.clone();
        a.fill_sine(o0, first, frames, &mut s0);
        a.fill_sine(o1, first, frames, &mut s1);
        let row = frames * HOP;
        for h in 0..N_HARM {
            for i in 0..(frames - f) * HOP {
                assert_eq!(s1[h * row + i], s0[h * row + i + f * HOP]);
            }
        }
        assert_ne!(s0[0], s0[row], "harmonics draw independently");
        // Fresh ticks: no shared draws, a new phase set per tick.
        let mut b = ZsNoise::new(7, n_mel);
        let p0 = b.window_origin(false, 10, f);
        let ph0 = b.base_phase(false);
        let p1 = b.window_origin(false, 11, f);
        let ph1 = b.base_phase(false);
        assert!(p1.abs_diff(p0) > (n_mel as u64) * 4 && p0 > 1 << 30);
        assert_ne!(ph0, ph1);
        assert_eq!(ph0[0], 0.0, "the fundamental starts at 0, as upstream");
        let fixed = a.base_phase(true);
        assert_eq!(fixed, a.base_phase(true));
        assert!(fixed[1..].iter().all(|p| p.abs() <= std::f32::consts::PI));
    }
}
