//! Pure signal processing for reference clips (no I/O, no allocation in the per-sample loops):
//! the rumble / DC high-pass, ITU-R BS.1770-4 integrated loudness (K-weighting and the two-stage
//! gating), the normalisation gain under a peak ceiling, clipping statistics, levels, fades and the
//! crossfaded splice of kept spans.

/// A sample at or above this magnitude counts as clipped (≈ −0.017 dBFS: an int16 device clips at
/// 32767 / 32768, a float device's overload is ≥ 1.0).
pub const CLIP_THRESHOLD: f32 = 0.998;
/// The lowest level reported (dBFS / LUFS).
pub const DB_FLOOR: f32 = -120.0;
/// Corner of the rumble / DC high-pass (Hz). Below every voice's fundamental (a bass voice starts
/// around 70 Hz) and inaudible, but it removes handling noise, desk thumps and a DC offset, which
/// would otherwise eat the peak headroom of the normalisation.
pub const RUMBLE_HZ: f64 = 40.0;

/// `20 log10(amplitude)`, floored at [`DB_FLOOR`].
pub fn amp_to_db(amplitude: f64) -> f32 {
    if amplitude.is_nan() || amplitude <= 0.0 {
        return DB_FLOOR;
    }
    ((20.0 * amplitude.log10()) as f32).max(DB_FLOOR)
}

/// `10 log10(power)`, floored at [`DB_FLOOR`].
pub fn power_to_db(power: f64) -> f32 {
    if power.is_nan() || power <= 0.0 {
        return DB_FLOOR;
    }
    ((10.0 * power.log10()) as f32).max(DB_FLOOR)
}

/// `10^(db / 20)`.
pub fn db_to_amp(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

/// Largest absolute sample (non-finite samples count as 0).
pub fn peak(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |m, &v| if v.is_finite() { m.max(v.abs()) } else { m })
}

/// Mean square (power) of `x` (0 for an empty slice).
pub fn mean_square(x: &[f32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len() as f64
}

/// Non-finite samples become silence (one bad buffer must not poison filters and statistics).
pub fn sanitize(x: &mut [f32]) {
    for v in x.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
}

/// Clipped samples of a signal (see [`CLIP_THRESHOLD`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClipStats {
    pub clipped: usize,
    pub total: usize,
}

impl ClipStats {
    pub fn of(x: &[f32]) -> ClipStats {
        ClipStats {
            clipped: x.iter().filter(|v| v.abs() >= CLIP_THRESHOLD).count(),
            total: x.len(),
        }
    }

    pub fn add(&mut self, other: ClipStats) {
        self.clipped += other.clipped;
        self.total += other.total;
    }

    /// Fraction of clipped samples (0..1).
    pub fn ratio(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            (self.clipped as f64 / self.total as f64) as f32
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Biquads
// ---------------------------------------------------------------------------------------------

/// A second-order IIR section (transposed direct form II, f64 state), `a0` normalised to 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Biquad {
    pub b: [f64; 3],
    /// `a1`, `a2`.
    pub a: [f64; 2],
    z: [f64; 2],
}

impl Biquad {
    pub fn new(b: [f64; 3], a: [f64; 2]) -> Biquad {
        Biquad { b, a, z: [0.0; 2] }
    }

    /// RBJ cookbook high-pass at `fc` Hz with quality `q` (`FRAC_1_SQRT_2` = Butterworth).
    pub fn highpass(rate: u32, fc: f64, q: f64) -> Biquad {
        let w0 = std::f64::consts::TAU * fc / f64::from(rate.max(1));
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let a0 = 1.0 + alpha;
        Biquad::new(
            [(1.0 + cos) / 2.0 / a0, -(1.0 + cos) / a0, (1.0 + cos) / 2.0 / a0],
            [-2.0 * cos / a0, (1.0 - alpha) / a0],
        )
    }

    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[1] * y;
        y
    }
}

/// The [`RUMBLE_HZ`] Butterworth high-pass, in place.
pub fn remove_rumble(x: &mut [f32], rate: u32) {
    let mut hp = Biquad::highpass(rate, RUMBLE_HZ, std::f64::consts::FRAC_1_SQRT_2);
    for v in x.iter_mut() {
        *v = hp.process(f64::from(*v)) as f32;
    }
}

// ---------------------------------------------------------------------------------------------
// ITU-R BS.1770-4 loudness
// ---------------------------------------------------------------------------------------------

/// The two K-weighting stages (the high-shelf "pre-filter" and the RLB high-pass) at `rate`,
/// designed from their analog prototypes as libebur128 does, so any sample rate is exact (at
/// 48 kHz they equal the coefficients printed in BS.1770-4).
pub fn k_weighting(rate: u32) -> [Biquad; 2] {
    let fs = f64::from(rate.max(1));
    // Stage 1: high shelf, +4 dB above ~1.7 kHz (head acoustics).
    let f0 = 1681.974450955533;
    let gain_db = 3.999843853973347;
    let q = 0.7071752369554196;
    let k = (std::f64::consts::PI * f0 / fs).tan();
    let vh = 10f64.powf(gain_db / 20.0);
    let vb = vh.powf(0.4996667741545416);
    let a0 = 1.0 + k / q + k * k;
    let shelf = Biquad::new(
        [(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0],
        [2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
    );
    // Stage 2: RLB high-pass at ~38 Hz.
    let f0 = 38.13547087602444;
    let q = 0.5003270373238773;
    let k = (std::f64::consts::PI * f0 / fs).tan();
    let a0 = 1.0 + k / q + k * k;
    let rlb = Biquad::new([1.0, -2.0, 1.0], [2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]);
    [shelf, rlb]
}

/// Absolute gate of BS.1770-4 (LUFS).
pub const ABSOLUTE_GATE_LUFS: f64 = -70.0;
/// Relative gate below the absolute-gated loudness (LU).
pub const RELATIVE_GATE_LU: f64 = -10.0;

fn block_loudness(mean_square: f64) -> f64 {
    -0.691 + 10.0 * mean_square.max(1e-30).log10()
}

/// Integrated loudness of a mono signal (LUFS): K-weighting, 400 ms blocks with 75 % overlap, the
/// −70 LUFS absolute gate and the −10 LU relative gate. A signal shorter than one block counts as
/// one block. `None` when nothing passes the absolute gate (silence).
pub fn integrated_loudness(x: &[f32], rate: u32) -> Option<f32> {
    if x.is_empty() || rate == 0 {
        return None;
    }
    let [mut shelf, mut rlb] = k_weighting(rate);
    // Prefix sums of the K-weighted power (f64: exact enough for minutes of audio).
    let mut prefix = Vec::with_capacity(x.len() + 1);
    prefix.push(0.0f64);
    let mut acc = 0.0f64;
    for &v in x {
        let y = rlb.process(shelf.process(f64::from(v)));
        acc += y * y;
        prefix.push(acc);
    }
    let block = ((0.4 * f64::from(rate)).round() as usize).max(1);
    let step = ((0.1 * f64::from(rate)).round() as usize).max(1);
    let mut blocks = Vec::new();
    if x.len() < block {
        blocks.push(prefix[x.len()] / x.len() as f64);
    } else {
        let mut start = 0usize;
        while start + block <= x.len() {
            blocks.push((prefix[start + block] - prefix[start]) / block as f64);
            start += step;
        }
    }
    let gated: Vec<f64> = blocks.into_iter().filter(|&z| block_loudness(z) > ABSOLUTE_GATE_LUFS).collect();
    if gated.is_empty() {
        return None;
    }
    let relative = block_loudness(gated.iter().sum::<f64>() / gated.len() as f64) + RELATIVE_GATE_LU;
    let kept: Vec<f64> = gated.into_iter().filter(|&z| block_loudness(z) > relative).collect();
    if kept.is_empty() {
        return None;
    }
    Some(block_loudness(kept.iter().sum::<f64>() / kept.len() as f64) as f32)
}

/// The gain (dB) that brings a clip of loudness `lufs` and sample peak `peak_dbfs` to `target`
/// LUFS without its peak exceeding `ceiling_dbfs`, within `±max_gain_db`. A clip whose loudness is
/// unknown (`None`) only gets the peak-limited gain towards the ceiling, never more than 0 dB.
pub fn normalization_gain_db(lufs: Option<f32>, peak_dbfs: f32, target: f32, ceiling_dbfs: f32, max_gain_db: f32) -> f32 {
    let headroom = ceiling_dbfs - peak_dbfs;
    let wanted = match lufs {
        Some(l) => target - l,
        None => 0.0,
    };
    wanted.min(headroom).clamp(-max_gain_db, max_gain_db)
}

/// Multiply by `gain_db`.
pub fn apply_gain(x: &mut [f32], gain_db: f32) {
    let g = db_to_amp(f64::from(gain_db)) as f32;
    for v in x.iter_mut() {
        *v *= g;
    }
}

// ---------------------------------------------------------------------------------------------
// Fades and splicing
// ---------------------------------------------------------------------------------------------

/// Raised-cosine gain of a fade at position `i` of `n` (0 → 1).
#[inline]
fn fade_gain(i: usize, n: usize) -> f32 {
    if n == 0 {
        return 1.0;
    }
    let t = (i as f64 + 0.5) / n as f64;
    (0.5 - 0.5 * (std::f64::consts::PI * t).cos()) as f32
}

/// Fade the first `n` samples in.
pub fn fade_in(x: &mut [f32], n: usize) {
    let n = n.min(x.len());
    for (i, v) in x.iter_mut().take(n).enumerate() {
        *v *= fade_gain(i, n);
    }
}

/// Fade the last `n` samples out.
pub fn fade_out(x: &mut [f32], n: usize) {
    let n = n.min(x.len());
    for (i, v) in x.iter_mut().rev().take(n).enumerate() {
        *v *= fade_gain(i, n);
    }
}

/// Concatenate `parts` with an equal-power crossfade of up to `xfade` samples at each join (the
/// joins sit in pauses, so the fade is inaudible; it only removes the step).
pub fn splice(parts: &[&[f32]], xfade: usize) -> Vec<f32> {
    let total: usize = parts.iter().map(|p| p.len()).sum();
    let mut out: Vec<f32> = Vec::with_capacity(total);
    for part in parts {
        if part.is_empty() {
            continue;
        }
        let n = xfade.min(out.len()).min(part.len());
        if n == 0 {
            out.extend_from_slice(part);
            continue;
        }
        let base = out.len() - n;
        for (i, (o, &p)) in out[base..].iter_mut().zip(&part[..n]).enumerate() {
            let t = (i as f64 + 0.5) / n as f64;
            let fade_in = (std::f64::consts::FRAC_PI_2 * t).sin() as f32;
            let fade_out = (std::f64::consts::FRAC_PI_2 * t).cos() as f32;
            *o = *o * fade_out + p * fade_in;
        }
        out.extend_from_slice(&part[n..]);
    }
    out
}

/// Average interleaved frames to mono, appending to `out`.
pub fn downmix_into(interleaved: &[f32], channels: usize, out: &mut Vec<f32>) {
    let ch = channels.max(1);
    if ch == 1 {
        out.extend_from_slice(interleaved);
        return;
    }
    let scale = 1.0 / ch as f32;
    out.extend(interleaved.chunks_exact(ch).map(|f| f.iter().sum::<f32>() * scale));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f64, amp: f64, rate: u32, seconds: f64) -> Vec<f32> {
        let n = (seconds * f64::from(rate)) as usize;
        (0..n)
            .map(|i| (amp * (std::f64::consts::TAU * freq * i as f64 / f64::from(rate)).sin()) as f32)
            .collect()
    }

    #[test]
    fn k_weighting_matches_the_bs1770_coefficients_at_48k() {
        let [shelf, rlb] = k_weighting(48_000);
        let close = |a: f64, b: f64| (a - b).abs() < 1e-8;
        assert!(close(shelf.b[0], 1.53512485958697), "{:?}", shelf);
        assert!(close(shelf.b[1], -2.69169618940638));
        assert!(close(shelf.b[2], 1.19839281085285));
        assert!(close(shelf.a[0], -1.69065929318241));
        assert!(close(shelf.a[1], 0.73248077421585));
        assert_eq!(rlb.b, [1.0, -2.0, 1.0]);
        assert!(close(rlb.a[0], -1.99004745483398));
        assert!(close(rlb.a[1], 0.99007225036621));
    }

    /// BS.1770-4: a 997 Hz sine at 0 dBFS in one channel reads −3.01 LKFS; the reading follows the
    /// level dB for dB, and the K-weighting is rate independent.
    #[test]
    fn a_sine_reads_its_documented_loudness_at_any_rate() {
        for rate in [16_000, 22_050, 44_100, 48_000, 96_000] {
            for level_db in [0.0, -6.0, -20.0, -40.0] {
                let x = sine(997.0, db_to_amp(level_db), rate, 3.0);
                let l = integrated_loudness(&x, rate).unwrap();
                let want = level_db as f32 - 3.01;
                assert!((l - want).abs() < 0.05, "{rate} Hz, {level_db} dB: {l} LUFS (want {want})");
            }
        }
    }

    #[test]
    fn gating_ignores_silence_and_quiet_parts() {
        let rate = 48_000;
        // 2 s of tone at −20 dBFS, then 4 s of silence, then 1 s of tone at −50 dBFS: the silence
        // never passes the absolute gate and the quiet part fails the relative gate.
        let mut x = sine(997.0, db_to_amp(-20.0), rate, 2.0);
        x.extend(std::iter::repeat_n(0.0, 4 * rate as usize));
        x.extend(sine(997.0, db_to_amp(-50.0), rate, 1.0));
        let l = integrated_loudness(&x, rate).unwrap();
        // The three blocks straddling the tone's end pass both gates at 75 / 50 / 25 % power
        // (−0.34 dB on the mean); everything else is the tone itself.
        assert!((l - (-23.35)).abs() < 0.15, "{l}");
        assert_eq!(integrated_loudness(&vec![0.0; 48_000], rate), None);
        assert_eq!(integrated_loudness(&[], rate), None);
        // Shorter than one block: one block over everything.
        let short = sine(997.0, db_to_amp(-10.0), rate, 0.2);
        assert!((integrated_loudness(&short, rate).unwrap() - (-13.01)).abs() < 0.1);
    }

    #[test]
    fn the_normalisation_gain_respects_target_ceiling_and_limits() {
        // Quiet clip: gain up to the target.
        assert_eq!(normalization_gain_db(Some(-35.0), -20.0, -20.0, -1.0, 30.0), 15.0);
        // Peaky clip: the ceiling wins.
        assert_eq!(normalization_gain_db(Some(-35.0), -6.0, -20.0, -1.0, 30.0), 5.0);
        // Loud clip: attenuated.
        assert_eq!(normalization_gain_db(Some(-10.0), -0.5, -20.0, -1.0, 30.0), -10.0);
        // Very quiet: capped.
        assert_eq!(normalization_gain_db(Some(-80.0), -60.0, -20.0, -1.0, 30.0), 30.0);
        // Unknown loudness: never amplified, only pulled under the ceiling.
        assert_eq!(normalization_gain_db(None, -30.0, -20.0, -1.0, 30.0), 0.0);
        assert_eq!(normalization_gain_db(None, 2.0, -20.0, -1.0, 30.0), -3.0);
    }

    #[test]
    fn the_rumble_filter_removes_dc_and_keeps_the_voice_band() {
        let rate = 48_000;
        let mut dc: Vec<f32> = vec![0.3; rate as usize];
        remove_rumble(&mut dc, rate);
        assert!(dc[rate as usize / 2..].iter().all(|v| v.abs() < 1e-3), "DC must be gone");
        let mut voice = sine(200.0, 0.5, rate, 1.0);
        let before = mean_square(&voice[rate as usize / 2..]);
        remove_rumble(&mut voice, rate);
        let after = mean_square(&voice[rate as usize / 2..]);
        let loss_db = power_to_db(after) - power_to_db(before);
        assert!(loss_db.abs() < 0.2, "200 Hz changed by {loss_db} dB");
        let mut rumble = sine(10.0, 0.5, rate, 2.0);
        remove_rumble(&mut rumble, rate);
        assert!(peak(&rumble[rate as usize..]) < 0.05, "10 Hz must be attenuated");
    }

    #[test]
    fn clipping_levels_and_db_helpers() {
        let x = [0.1, -0.999, 1.0, 0.5, f32::NAN, -1.2];
        let s = ClipStats::of(&x);
        assert_eq!((s.clipped, s.total), (3, 6));
        assert!((s.ratio() - 0.5).abs() < 1e-6);
        assert_eq!(peak(&x), 1.2);
        assert_eq!(amp_to_db(0.0), DB_FLOOR);
        assert!((amp_to_db(0.5) - (-6.0206)).abs() < 1e-3);
        assert!((power_to_db(0.01) - (-20.0)).abs() < 1e-4);
        let mut y = x;
        sanitize(&mut y);
        assert_eq!(y[4], 0.0);
        let mut m = Vec::new();
        downmix_into(&[1.0, 0.0, 0.5, 0.5], 2, &mut m);
        assert_eq!(m, vec![0.5, 0.5]);
    }

    #[test]
    fn fades_and_splices_are_click_free_and_exact_in_length() {
        let mut x = vec![1.0f32; 100];
        fade_in(&mut x, 10);
        fade_out(&mut x, 20);
        assert!(x[0] < 0.05 && x[9] > 0.9 && x[50] == 1.0 && x[99] < 0.05);
        let a = vec![1.0f32; 50];
        let b = vec![1.0f32; 30];
        let joined = splice(&[&a, &b], 10);
        assert_eq!(joined.len(), 70, "each join overlaps by the crossfade");
        // Equal-power crossfade of identical constant signals stays within +3 dB and never dips.
        assert!(joined.iter().all(|&v| (0.99..=1.42).contains(&v)), "{joined:?}");
        let short = splice(&[&a[..3], &b], 10);
        assert_eq!(short.len(), 30, "the fade is limited by what is already there");
        assert_eq!(splice(&[], 10), Vec::<f32>::new());
    }
}
