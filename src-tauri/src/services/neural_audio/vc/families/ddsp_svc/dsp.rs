//! The DDSP-SVC tick's host-side signal processing (P5-A), each piece checked against upstream's
//! Python (`gui_reflow.py` `SvcDDSP.infer`, `ddsp/vocoder.py`; the reference loop is
//! `scripts/vc-families/ddsp_svc/sampler.py`):
//! - [`tta2x_interleave`]: `Audio2ContentVec768L12TTA2X`'s unit sequence (one unit per 10 ms);
//! - [`mel2ph_into`]: `Units_Encoder.encode`'s frame → unit alignment (`torch.round`, half to even);
//! - [`f0_frames_into`]: `F0_Extractor.extract`'s post-processing of an RMVPE / FCPE contour (unvoiced
//!   frames interpolated on the 10 ms grid, resampled to the model's frames, the unvoiced mask
//!   resampled and thresholded at 0.5, then `uv_interp` and the `f0_min` floor), with no upper
//!   clamp (the singing range reaches D#7; only the vocoder's source Nyquist bounds it);
//! - [`VolumeScratch::envelope_into`]: `Volume_Extractor.extract` (RMS-like envelope, reflect
//!   padded) computed on the 16 kHz stream over the same time spans as upstream's 44.1 kHz windows;
//! - [`gate_mask_into`]: `SvcDDSP.infer`'s volume gate (`ddsp.core.upsample` of the 0/1 frame mask);
//! - [`NsfPhase`]: the vocoder's NSF phase at a frame, so consecutive ticks continue the excitation.
//!
//! Frames are anchored to absolute time: frame `k` is model-rate samples `[k·hop, (k+1)·hop)`,
//! its features are taken at sample `k·hop` (upstream's convention within a window), and positions
//! inside a window are measured from the 16 kHz context's first sample.

/// `torch.round` / `np.round`: half to even.
pub fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// Unit counts of the two ContentVec passes over `n` samples of 16 kHz audio: the audio itself
/// (`a`, frames centred at 200 + 320 t) and the audio from sample 160 on (`b`, centred at
/// 360 + 320 t). Interleaved `a0 b0 a1 b1 ...` they are upstream's tta2x sequence (its delayed
/// pass's first frame is dropped by upstream; its last is dropped or padded as here).
pub fn tta2x_counts(n: usize) -> (usize, usize) {
    let frames = |len: usize| if len >= 400 { (len - 400) / 320 + 1 } else { 0 };
    let a = frames(n);
    let b = frames(n.saturating_sub(160)).min(a);
    (a, b)
}

/// Interleave the two passes (`a` and `b` time-major `[frames × c]`) into `out`
/// (`(na + nb) × c`, `a0 b0 a1 b1 ...`; `nb` is `na` or `na − 1`).
pub fn tta2x_interleave(a: &[f32], na: usize, b: &[f32], nb: usize, c: usize, out: &mut Vec<f32>) -> Result<usize, String> {
    if a.len() < na * c || b.len() < nb * c || !(nb == na || nb + 1 == na) {
        return Err(format!("tta2x: {na} and {nb} frames of {c} channels do not interleave"));
    }
    out.clear();
    for t in 0..na {
        out.extend_from_slice(&a[t * c..(t + 1) * c]);
        if t < nb {
            out.extend_from_slice(&b[t * c..(t + 1) * c]);
        }
    }
    Ok(na + nb)
}

/// `mel2ph` of the window's frames: frame `i` (absolute `k0 + i`, at model-rate sample
/// `(k0 + i)·hop`) takes the unit nearest its time from the encoder segment's start
/// (`seg_start` model-rate samples), `round(t / unit_period)` clamped to `[0, units)`.
#[allow(clippy::too_many_arguments)]
pub fn mel2ph_into(k0: i64, frames: usize, hop: usize, seg_start: i64, sample_rate: u32, unit_period_s: f64, units: usize, out: &mut [i64]) {
    let last = units.saturating_sub(1) as f64;
    for (i, o) in out.iter_mut().take(frames).enumerate() {
        let t = ((k0 + i as i64) * hop as i64 - seg_start) as f64 / f64::from(sample_rate);
        *o = round_half_even(t / unit_period_s).clamp(0.0, last) as i64;
    }
}

/// Linear interpolation of `ys` sampled at `0, 1, 2, ...` at position `x`, held flat outside
/// (`np.interp`).
fn interp(ys: &[f64], x: f64) -> f64 {
    match ys.len() {
        0 => 0.0,
        1 => ys[0],
        n => {
            if x <= 0.0 {
                return ys[0];
            }
            if x >= (n - 1) as f64 {
                return ys[n - 1];
            }
            let i = x.floor() as usize;
            let f = x - i as f64;
            ys[i] * (1.0 - f) + ys[i + 1] * f
        }
    }
}

/// Fill the zeros of `f0` by linear interpolation between voiced values, held flat beyond the
/// first / last voiced one (`f0[uv] = np.interp(where(uv), where(~uv), f0[~uv])`); no change when
/// nothing is voiced.
fn fill_unvoiced(f0: &mut [f64]) {
    let voiced: Vec<usize> = (0..f0.len()).filter(|&i| f0[i] > 0.0).collect();
    let (Some(&first), Some(&last)) = (voiced.first(), voiced.last()) else {
        return;
    };
    let mut prev = first;
    for &v in &voiced[1..] {
        if v > prev + 1 {
            let (a, b) = (f0[prev], f0[v]);
            let span = (v - prev) as f64;
            for (k, x) in f0[prev + 1..v].iter_mut().enumerate() {
                *x = a + (b - a) * (k + 1) as f64 / span;
            }
        }
        prev = v;
    }
    let (fa, fb) = (f0[first], f0[last]);
    f0[..first].iter_mut().for_each(|x| *x = fa);
    f0[last + 1..].iter_mut().for_each(|x| *x = fb);
}

/// Scratch of [`f0_frames_into`] (kept across ticks: no allocation once sized).
#[derive(Default)]
pub struct F0Scratch {
    filled: Vec<f64>,
    uv: Vec<f64>,
    frames: Vec<f64>,
}

/// Upstream's F0 post-processing (module docs) of a 10 ms contour `f0_10ms` (frame `j` at
/// `j · 10 ms` from the context's start, 0 = unvoiced) onto the window's `frames` model frames,
/// frame `i` at `pos_10ms[i]` (its time in 10 ms units from the context's start). Unvoiced frames
/// take the interpolated pitch (DDSP-SVC always synthesises a pitch; the volume gate silences),
/// frames below `f0_min` are raised to it, and everything is shifted by `semitones`. The result is
/// clamped only to `max_hz` (the vocoder source's Nyquist frequency).
pub fn f0_frames_into(f0_10ms: &[f32], pos_10ms: &[f64], f0_min: f32, semitones: f32, max_hz: f32, scratch: &mut F0Scratch, out: &mut [f32]) {
    let n = pos_10ms.len().min(out.len());
    let s = scratch;
    s.filled.clear();
    s.filled.extend(f0_10ms.iter().map(|&x| if x.is_finite() && x > 0.0 { f64::from(x) } else { 0.0 }));
    s.uv.clear();
    s.uv.extend(s.filled.iter().map(|&x| if x > 0.0 { 0.0 } else { 1.0 }));
    fill_unvoiced(&mut s.filled);
    s.frames.clear();
    for &p in &pos_10ms[..n] {
        let v = interp(&s.filled, p);
        let unvoiced = interp(&s.uv, p) > 0.5;
        s.frames.push(if unvoiced { 0.0 } else { v });
    }
    // uv_interp: unvoiced frames take the neighbours' pitch; then the f0_min floor.
    fill_unvoiced(&mut s.frames);
    let shift = 2f64.powf(f64::from(semitones) / 12.0);
    let floor = f64::from(f0_min.max(1.0));
    for (o, &v) in out[..n].iter_mut().zip(&s.frames) {
        let hz = v.max(floor) * shift;
        *o = hz.min(f64::from(max_hz)) as f32;
    }
}

/// Upstream `Volume_Extractor.extract` on the 16 kHz stream (module docs): one value per frame,
/// `sqrt(max(E[x²] − E[x]², 0))` over the frame's window, the stream reflected past its last sample.
#[derive(Default)]
pub struct VolumeScratch {
    sum: Vec<f64>,
    sum2: Vec<f64>,
}

impl VolumeScratch {
    /// `x16`: the 16 kHz context (sample 0 at model-rate sample `seg_start`); frame `i` of the
    /// window is centred at model-rate sample `(k0 + i)·hop` with a window of `win` model-rate
    /// samples (`[c − win/2, c + win − win/2)`).
    #[allow(clippy::too_many_arguments)]
    pub fn envelope_into(&mut self, x16: &[f32], seg_start: i64, k0: i64, hop: usize, win: usize, sample_rate: u32, frames: usize, out: &mut [f32]) {
        let n = x16.len();
        if n < 2 {
            out.iter_mut().take(frames).for_each(|v| *v = 0.0);
            return;
        }
        // Reflected tail (and head) long enough for the windows at the window's edges.
        let ratio = 16_000.0 / f64::from(sample_rate);
        let pad = ((win as f64) * ratio).ceil() as usize + 2;
        let total = n + 2 * pad;
        let at = |i: usize| -> f64 {
            // padded index i -> source index with reflection (no edge repeat, like np.pad reflect)
            let j = i as i64 - pad as i64;
            let m = n as i64;
            let mut r = j;
            while r < 0 || r >= m {
                r = if r < 0 { -r } else { 2 * (m - 1) - r };
            }
            f64::from(x16[r as usize])
        };
        self.sum.clear();
        self.sum2.clear();
        self.sum.push(0.0);
        self.sum2.push(0.0);
        let (mut s, mut s2) = (0.0f64, 0.0f64);
        for i in 0..total {
            let v = at(i);
            s += v;
            s2 += v * v;
            self.sum.push(s);
            self.sum2.push(s2);
        }
        // Prefix integral at a fractional padded position (zero-order hold of the 16 kHz samples).
        let integral = |p: &[f64], u: f64| -> f64 {
            let u = u.clamp(0.0, total as f64);
            let i = (u.floor() as usize).min(total - 1);
            let f = u - i as f64;
            p[i] + f * (p[i + 1] - p[i])
        };
        let half = (win / 2) as f64;
        for (i, o) in out.iter_mut().take(frames).enumerate() {
            let c = ((k0 + i as i64) * hop as i64 - seg_start) as f64;
            let a = (c - half) * ratio + pad as f64;
            let b = (c - half + win as f64) * ratio + pad as f64;
            let len = (b - a).max(1e-9);
            let mean = (integral(&self.sum, b) - integral(&self.sum, a)) / len;
            let ms = (integral(&self.sum2, b) - integral(&self.sum2, a)) / len;
            *o = (ms - mean * mean).max(0.0).sqrt() as f32;
        }
    }
}

/// `SvcDDSP.infer`'s gate over the decoded frames: per frame `volume > 10^(threshold/20)`, the
/// 0/1 mask linearly upsampled by `hop` (`ddsp.core.upsample`: frame `k`'s value at its first
/// sample, ramping to frame `k + 1`'s; the last frame holds). `volume` covers the whole window
/// (`T` frames); `out` receives the last `frames · hop` samples, for the last `frames` frames.
pub fn gate_mask_into(volume: &[f32], threshold_db: f32, frames: usize, hop: usize, out: &mut [f32]) {
    let t = volume.len();
    let thr = 10f32.powf(threshold_db / 20.0);
    let m = |k: usize| -> f32 {
        let k = k.min(t - 1);
        if volume[k] > thr {
            1.0
        } else {
            0.0
        }
    };
    let first = t - frames;
    for f in 0..frames {
        let (a, b) = (m(first + f), m(first + f + 1));
        let dst = &mut out[f * hop..(f + 1) * hop];
        if a == b {
            dst.iter_mut().for_each(|v| *v = a);
        } else {
            for (s, v) in dst.iter_mut().enumerate() {
                let lam = s as f32 / hop as f32;
                *v = a * (1.0 - lam) + b * lam;
            }
        }
    }
}

/// Whether [`gate_mask_into`] keeps any sample of frames `from..=to` of the window: their own mask
/// or the next frame's, which the last one's ramp reaches (the window's last frame holds).
pub fn gate_open(volume: &[f32], threshold_db: f32, from: usize, to: usize) -> bool {
    let t = volume.len();
    if t == 0 {
        return false;
    }
    let thr = 10f32.powf(threshold_db / 20.0);
    let last = to.saturating_add(1).min(t - 1);
    (from.min(last)..=last).any(|k| volume[k] > thr)
}

/// The phase-continuous NSF lever of the vocoder (the graph's `phase0` input): the phase (cycles)
/// at the start of each tick's first decoded frame equals the phase the previous tick's excitation
/// had at that absolute frame. Per frame the graph advances its sine by
/// `wrap(upp·s0 + 0.5·(upp − 1)·ds0)` with `s0 = f0 / source_sr` and `ds0` the next frame's `s0`
/// minus this one's (0 for the window's last frame), which this replays in f64.
#[derive(Clone, Debug)]
pub struct NsfPhase {
    upp: usize,
    source_sr: f64,
    /// `(absolute first frame, its phase, the F0 of the decoded frames)` of the last tick.
    last: Option<(i64, f64, Vec<f32>)>,
}

impl NsfPhase {
    pub fn new(upp: usize, source_sr: f64) -> NsfPhase {
        NsfPhase { upp: upp.max(1), source_sr: source_sr.max(1.0), last: None }
    }

    /// Wrapped phase increment of frame `i` of `f0` (the graph's `rad2`).
    pub fn increment(&self, f0: &[f32], i: usize) -> f64 {
        let s0 = f64::from(f0[i]) / self.source_sr;
        let ds0 = if i + 1 < f0.len() { f64::from(f0[i + 1]) / self.source_sr - s0 } else { 0.0 };
        let inc = s0 * self.upp as f64 + 0.5 * ds0 * (self.upp as f64 - 1.0);
        (inc + 0.5).rem_euclid(1.0) - 0.5
    }

    /// The phase0 of a tick whose decoded frames start at absolute frame `first` (0 when the
    /// previous tick does not reach it: the first tick, a gap, a jump back).
    pub fn phase_at(&self, first: i64) -> f64 {
        let Some((prev, phase, f0)) = &self.last else {
            return 0.0;
        };
        let d = first - prev;
        if d < 0 || d as usize >= f0.len() {
            return 0.0;
        }
        let mut p = *phase;
        for i in 0..d as usize {
            p += self.increment(f0, i);
        }
        p.rem_euclid(1.0)
    }

    /// Record this tick's first frame, phase0 and decoded F0 (the buffer is reused).
    pub fn commit(&mut self, first: i64, phase0: f64, f0: &[f32]) {
        match &mut self.last {
            Some((f, p, buf)) => {
                *f = first;
                *p = phase0;
                buf.clear();
                buf.extend_from_slice(f0);
            }
            None => self.last = Some((first, phase0, f0.to_vec())),
        }
    }

    pub fn reset(&mut self) {
        self.last = None;
    }

    /// The last tick's first decoded frame and decoded F0 (tests).
    #[cfg(test)]
    pub fn last_decoded(&self) -> Option<(i64, &[f32])> {
        self.last.as_ref().map(|(first, _, f0)| (*first, f0.as_slice()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_to_even() {
        for (x, want) in [(0.5, 0.0), (1.5, 2.0), (2.5, 2.0), (-0.5, -0.0), (-1.5, -2.0), (1.2, 1.0), (2.6, 3.0)] {
            assert_eq!(round_half_even(x), want, "{x}");
        }
    }

    /// `sampler.tta2x` / upstream: for every length the interleaved count equals upstream's.
    #[test]
    fn tta2x_counts_match_upstream() {
        for n in (400..4000).step_by(7) {
            let up_a = (n - 400) / 320 + 1;
            let up_s = (n + 160 - 400) / 320 + 1;
            let upstream = if up_s > up_a { 2 * up_s - 2 } else { 2 * up_a - 1 };
            let (a, b) = tta2x_counts(n);
            assert_eq!(a + b, upstream, "n = {n}");
        }
        let a = [1.0, 1.0, 2.0, 2.0];
        let b = [9.0, 9.0];
        let mut out = Vec::new();
        assert_eq!(tta2x_interleave(&a, 2, &b, 1, 2, &mut out).unwrap(), 3);
        assert_eq!(out, [1.0, 1.0, 9.0, 9.0, 2.0, 2.0]);
        assert!(tta2x_interleave(&a, 2, &b, 3, 2, &mut out).is_err());
    }

    /// `sampler.mel2ph` (frames from the segment's start): round(512/441 · i), clamped.
    #[test]
    fn mel2ph_matches_upstream_alignment() {
        let mut out = vec![0i64; 40];
        mel2ph_into(0, 40, 512, 0, 44_100, 0.01, 30, &mut out);
        for (i, &v) in out.iter().enumerate() {
            let want = round_half_even(i as f64 * 512.0 / 441.0).min(29.0) as i64;
            assert_eq!(v, want, "frame {i}");
        }
        // Absolute frames, a segment that starts 1000 samples into frame 3.
        mel2ph_into(5, 3, 512, 512 * 3 + 1000, 44_100, 0.01, 100, &mut out[..3]);
        assert_eq!(&out[..3], &[round_half_even((2560.0 - 2536.0) / 441.0) as i64, 1, 2]);
    }

    /// `F0_Extractor.extract(uv_interp=True)` on a contour with gaps (numpy reference values).
    #[test]
    fn f0_post_processing_matches_upstream() {
        // 10 ms contour: unvoiced, 100, 0, 200, 0, 0 (voiced frames 1 and 3).
        let f0 = [0.0, 100.0, 0.0, 200.0, 0.0, 0.0];
        // Model frames at 0, 0.8, 1.6, ... (10 ms units).
        let pos: Vec<f64> = (0..8).map(|i| i as f64 * 0.8).collect();
        let mut out = vec![0.0f32; 8];
        f0_frames_into(&f0, &pos, 65.0, 0.0, 2500.0, &mut F0Scratch::default(), &mut out);
        // numpy (the upstream code on these arrays): rmvpe branch, then uv_interp and the floor.
        let want = [100.0, 100.0, 133.3333, 166.6667, 200.0, 200.0, 200.0, 200.0];
        for (o, w) in out.iter().zip(&want) {
            assert!((f64::from(*o) - w).abs() < 1e-3, "{out:?} vs {want:?}");
        }
        // Silence: every frame at f0_min; shifting and the Nyquist clamp.
        f0_frames_into(&[0.0; 6], &pos, 65.0, 0.0, 2500.0, &mut F0Scratch::default(), &mut out);
        assert!(out.iter().all(|&v| v == 65.0));
        f0_frames_into(&[1000.0; 6], &pos, 65.0, 12.0, 2500.0, &mut F0Scratch::default(), &mut out);
        assert!(out.iter().all(|&v| (v - 2000.0).abs() < 1e-2), "{out:?}");
        f0_frames_into(&[2000.0; 6], &pos, 65.0, 12.0, 2500.0, &mut F0Scratch::default(), &mut out);
        assert!(out.iter().all(|&v| v == 2500.0));
    }

    /// The 16 kHz volume envelope against upstream's 44.1 kHz `Volume_Extractor` on a band-limited
    /// signal (both sampled from the same continuous sum of tones).
    #[test]
    fn volume_envelope_matches_upstreams_44k_windows() {
        let sig = |t: f64| 0.3 * (2.0 * std::f64::consts::PI * 220.0 * t).sin() * (1.0 + 0.5 * (7.0 * t).sin()) + 0.05 * (2.0 * std::f64::consts::PI * 1300.0 * t).cos();
        let n16 = 16_000;
        let x16: Vec<f32> = (0..n16).map(|i| sig(i as f64 / 16_000.0) as f32).collect();
        let n44 = n16 * 441 / 160;
        let x44: Vec<f64> = (0..n44).map(|i| sig(i as f64 / 44_100.0)).collect();
        let (hop, win) = (512usize, 1024usize);
        let frames = n44 / hop + 1;
        // upstream: reflect pad win/2 and (win+1)/2, frame n = padded[n*hop .. n*hop + win]
        let reflect = |i: i64| -> f64 {
            let m = n44 as i64;
            let mut r = i;
            while r < 0 || r >= m {
                r = if r < 0 { -r } else { 2 * (m - 1) - r };
            }
            x44[r as usize]
        };
        let mut got = vec![0.0f32; frames];
        VolumeScratch::default().envelope_into(&x16, 0, 0, hop, win, 44_100, frames, &mut got);
        let mut worst = 0.0f64;
        for (k, g) in got.iter().enumerate() {
            let c = (k * hop) as i64;
            let w: Vec<f64> = (c - 512..c + 512).map(reflect).collect();
            let mean = w.iter().sum::<f64>() / w.len() as f64;
            let ms = w.iter().map(|v| v * v).sum::<f64>() / w.len() as f64;
            let want = (ms - mean * mean).max(0.0).sqrt();
            worst = worst.max((f64::from(*g) - want).abs() / want.max(1e-3));
        }
        assert!(worst < 0.02, "worst relative error {worst}");
    }

    #[test]
    fn gate_mask_ramps_like_ddsp_core_upsample() {
        let volume = [0.0, 0.1, 0.1, 0.0];
        let mut out = vec![9.0f32; 3 * 4];
        gate_mask_into(&volume, -40.0, 3, 4, &mut out);
        // frames 1, 2, 3: [1,1] -> 1s, [1,0] -> ramp down, [0, hold 0] -> 0s
        assert_eq!(&out[..4], &[1.0; 4]);
        assert_eq!(&out[4..8], &[1.0, 0.75, 0.5, 0.25]);
        assert_eq!(&out[8..], &[0.0; 4]);
        assert!(gate_open(&volume, -40.0, 2, 3));
        assert!(!gate_open(&volume, -40.0, 3, 3));
        assert!(!gate_open(&[0.001; 4], -40.0, 0, 3));
        // An onset in the frame after the last one asked about: that frame's ramp is non-zero.
        assert!(gate_open(&[0.0, 0.0, 0.0, 0.1], -40.0, 1, 2));
        assert!(!gate_open(&[], -40.0, 0, 0));
    }

    /// `gate_open` says "closed" exactly when every sample `gate_mask_into` gives frames
    /// `from..=to` is zero (so skipping a closed tick returns what the gated tick would), for every
    /// on/off pattern of six frames.
    #[test]
    fn gate_open_agrees_with_the_mask() {
        let (t, hop) = (6usize, 4usize);
        for bits in 0u32..(1 << t) {
            let volume: Vec<f32> = (0..t).map(|k| if bits >> k & 1 == 1 { 0.1 } else { 0.0 }).collect();
            let mut mask = vec![0.0f32; t * hop];
            gate_mask_into(&volume, -40.0, t, hop, &mut mask);
            for from in 0..t {
                for to in from..t {
                    let any = mask[from * hop..(to + 1) * hop].iter().any(|&m| m > 0.0);
                    assert_eq!(gate_open(&volume, -40.0, from, to), any, "{volume:?} frames {from}..={to}");
                }
            }
        }
    }

    /// The tracker reproduces the phase the graph accumulates (`cumsum(rad2)`) from an earlier
    /// first frame, so two windows that overlap agree on the excitation's phase.
    #[test]
    fn nsf_phase_continues_across_windows() {
        let mut nsf = NsfPhase::new(64, 5512.5);
        let f0: Vec<f32> = (0..20).map(|i| 200.0 + 10.0 * i as f32).collect();
        assert_eq!(nsf.phase_at(100), 0.0);
        nsf.commit(100, 0.25, &f0);
        let expect = (0.25 + (0..7).map(|i| nsf.increment(&f0, i)).sum::<f64>()).rem_euclid(1.0);
        assert!((nsf.phase_at(107) - expect).abs() < 1e-12);
        assert_eq!(nsf.phase_at(99), 0.0, "a jump back starts at 0");
        assert_eq!(nsf.phase_at(120), 0.0, "past the previous window");
        // One frame's increment: upp * f0 / source_sr wrapped, plus the slope term.
        let inc = nsf.increment(&[441.0, 441.0], 0);
        assert!((inc - ((64.0 * 441.0 / 5512.5 + 0.5f64).rem_euclid(1.0) - 0.5)).abs() < 1e-12);
        nsf.reset();
        assert_eq!(nsf.phase_at(107), 0.0);
    }
}
