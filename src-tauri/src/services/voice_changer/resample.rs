//! Stateful band-limited resampler for streams fed in arbitrary chunk sizes (the monitor output).
//!
//! The previous monitor path resampled every chunk on its own with linear interpolation and
//! `ceil`, so it over-produced by up to a frame per chunk whenever the monitor and main output
//! rates differ; the monitor queue then hit its cap and dropped audio (a periodic click).
//!
//! [`PolyphaseResampler`] keeps its filter history across calls and tracks the output position with
//! an exact integer phase accumulator (`from / gcd` input frames per `to / gcd` output frames), so
//! after `n` input frames it has produced exactly `ceil((n - HALF_TAPS) * to / from)` frames (for
//! `n > HALF_TAPS`), whatever the chunking — no drift, ever. Output frame `k` is the band-limited
//! input at time `k / to` (no content offset; the processing latency is `HALF_TAPS` input frames).
//! The kernel is a Kaiser-windowed sinc (64 taps, 256 interpolated phases), cutoff at 92 % of the
//! lower Nyquist frequency. Allocation-free after construction apart from the caller's output Vec.

/// Taps on each side of the interpolation point.
const HALF_TAPS: usize = 32;
const TAPS: usize = 2 * HALF_TAPS;
/// Tabulated kernel phases (linear interpolation in between).
const PHASES: usize = 256;
const KAISER_BETA: f64 = 8.0;
/// Cutoff as a fraction of the lower of the two Nyquist frequencies.
const PASSBAND: f64 = 0.92;

pub(super) struct PolyphaseResampler {
    ch: usize,
    /// Output frames per `m` input frames (both divided by their gcd).
    l: u64,
    m: u64,
    passthrough: bool,
    /// `(PHASES + 1) * TAPS` kernel rows, each normalized to unit DC gain.
    table: Vec<f32>,
    /// Interleaved input history; frame 0 is absolute input frame `buf_start`.
    buf: Vec<f32>,
    buf_start: i64,
    /// Next output position: `pos_int + pos_frac / l` input frames.
    pos_int: i64,
    pos_frac: u64,
    weights: [f32; TAPS],
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

/// Modified Bessel function of the first kind, order 0 (series; converges fast for beta <= 20).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let q = x * x / 4.0;
    for k in 1..64 {
        term *= q / (k as f64 * k as f64);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

fn kernel(d: f64, fc: f64) -> f64 {
    let u = d / HALF_TAPS as f64;
    if u.abs() >= 1.0 {
        return 0.0;
    }
    let x = std::f64::consts::PI * fc * d;
    let sinc = if x.abs() < 1e-12 { 1.0 } else { x.sin() / x };
    let window = bessel_i0(KAISER_BETA * (1.0 - u * u).sqrt()) / bessel_i0(KAISER_BETA);
    fc * sinc * window
}

impl PolyphaseResampler {
    pub(super) fn new(from_sr: u32, to_sr: u32, ch: usize) -> Self {
        let ch = ch.max(1);
        let from = u64::from(from_sr.max(1));
        let to = u64::from(to_sr.max(1));
        let g = gcd(from, to);
        let (l, m) = (to / g, from / g);
        let passthrough = l == m;
        let mut table = Vec::new();
        if !passthrough {
            let fc = PASSBAND * (l as f64 / m as f64).min(1.0);
            table.reserve((PHASES + 1) * TAPS);
            for p in 0..=PHASES {
                let phi = p as f64 / PHASES as f64;
                // Tap j multiplies input frame `floor(pos) - HALF_TAPS + 1 + j`, at distance
                // `phi + HALF_TAPS - 1 - j` from the output position.
                let row: Vec<f64> = (0..TAPS).map(|j| kernel(phi + HALF_TAPS as f64 - 1.0 - j as f64, fc)).collect();
                let sum: f64 = row.iter().sum();
                table.extend(row.iter().map(|&w| (w / sum) as f32));
            }
        }
        // History starts as the silence before the first input frame, back to the first tap of
        // output 0 (which sits exactly on input frame 0).
        let pre = HALF_TAPS - 1;
        Self {
            ch,
            l,
            m,
            passthrough,
            table,
            buf: {
                let mut b = Vec::with_capacity((pre + 4096) * ch);
                b.resize(pre * ch, 0.0);
                b
            },
            buf_start: -(pre as i64),
            pos_int: 0,
            pos_frac: 0,
            weights: [0.0; TAPS],
        }
    }

    #[cfg(test)]
    fn is_passthrough(&self) -> bool {
        self.passthrough
    }

    /// Output frames produced once `n` input frames have been pushed in total.
    #[cfg(test)]
    fn expected_total(&self, n: u64) -> u64 {
        if self.passthrough {
            return n;
        }
        let avail = n.saturating_sub(HALF_TAPS as u64);
        (avail * self.l).div_ceil(self.m)
    }

    /// Resample `input` (interleaved, whole frames) and append the output to `out`.
    pub(super) fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let ch = self.ch;
        let input = &input[..input.len() / ch * ch];
        if self.passthrough {
            out.extend_from_slice(input);
            return;
        }
        self.buf.extend_from_slice(input);
        let buf_frames = (self.buf.len() / ch) as i64;
        let end = self.buf_start + buf_frames; // one past the newest input frame
        while self.pos_int + (HALF_TAPS as i64) < end {
            let first = self.pos_int - HALF_TAPS as i64 + 1;
            let off = (first - self.buf_start) as usize * ch;
            self.load_weights();
            for c in 0..ch {
                let mut acc = 0.0f32;
                for (j, &w) in self.weights.iter().enumerate() {
                    acc += w * self.buf[off + j * ch + c];
                }
                out.push(acc);
            }
            self.pos_frac += self.m;
            self.pos_int += (self.pos_frac / self.l) as i64;
            self.pos_frac %= self.l;
        }
        // Drop history no future output needs (keep from the next output's first tap).
        let keep_from = (self.pos_int - HALF_TAPS as i64 + 1 - self.buf_start).clamp(0, buf_frames) as usize;
        if keep_from > 0 {
            self.buf.copy_within(keep_from * ch.., 0);
            self.buf.truncate(self.buf.len() - keep_from * ch);
            self.buf_start += keep_from as i64;
        }
    }

    /// Kernel for the current fractional position, interpolated between tabulated phases.
    fn load_weights(&mut self) {
        let x = self.pos_frac as f64 / self.l as f64 * PHASES as f64;
        let p0 = (x.floor() as usize).min(PHASES - 1);
        let t = (x - p0 as f64) as f32;
        let a = &self.table[p0 * TAPS..(p0 + 1) * TAPS];
        let b = &self.table[(p0 + 1) * TAPS..(p0 + 2) * TAPS];
        for ((w, &wa), &wb) in self.weights.iter_mut().zip(a).zip(b) {
            *w = wa + (wb - wa) * t;
        }
    }

    /// Advance the phase accumulator exactly like [`Self::process`] without computing samples
    /// (long exactness runs in unoptimized test builds).
    #[cfg(test)]
    fn count_only(&mut self, frames: usize) -> u64 {
        if self.passthrough {
            return frames as u64;
        }
        let end_before = self.buf_start + (self.buf.len() / self.ch) as i64;
        let end = end_before + frames as i64;
        let mut n = 0u64;
        while self.pos_int + (HALF_TAPS as i64) < end {
            n += 1;
            self.pos_frac += self.m;
            self.pos_int += (self.pos_frac / self.l) as i64;
            self.pos_frac %= self.l;
        }
        // Mirror the history bookkeeping with zero samples.
        let buf_frames = end - self.buf_start;
        let keep_from = (self.pos_int - HALF_TAPS as i64 + 1 - self.buf_start).clamp(0, buf_frames);
        self.buf_start += keep_from;
        self.buf.resize(((buf_frames - keep_from) as usize) * self.ch, 0.0);
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic chunk sizes in 1..=max.
    fn chunk_sizes(seed: u32, max: u32) -> impl FnMut() -> usize {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) % max + 1) as usize
        }
    }

    /// 44.1 kHz -> 48 kHz for 10 simulated minutes in random chunk sizes: after every call the
    /// total is exactly `ceil((n - HALF_TAPS) * 48000 / 44100)` (the old per-chunk `ceil` gained up
    /// to a frame per call).
    #[test]
    fn monitor_resampler_44k1_to_48k_is_exact_over_ten_minutes() {
        let mut r = PolyphaseResampler::new(44_100, 48_000, 2);
        let mut next = chunk_sizes(7, 1_024);
        let (mut n_in, mut n_out) = (0u64, 0u64);
        while n_in < 10 * 60 * 44_100 {
            let n = next();
            n_out += r.count_only(n);
            n_in += n as u64;
            assert_eq!(n_out, r.expected_total(n_in), "after {n_in} input frames");
        }
        // And the real sample path counts exactly like the counter (10 s, stereo).
        let mut r = PolyphaseResampler::new(44_100, 48_000, 2);
        let mut next = chunk_sizes(11, 2_048);
        let mut out = Vec::new();
        let (mut n_in, mut n_out) = (0u64, 0u64);
        let chunk = vec![0.1f32; 2 * 2_048];
        while n_in < 10 * 44_100 {
            let n = next();
            out.clear();
            r.process(&chunk[..2 * n], &mut out);
            assert_eq!(out.len() % 2, 0);
            n_out += (out.len() / 2) as u64;
            n_in += n as u64;
            assert_eq!(n_out, r.expected_total(n_in));
        }
        // DC passes at unity gain once the start-up silence has left the window.
        assert!(out.iter().all(|&v| (v - 0.1).abs() < 1e-4), "DC gain");
    }

    #[test]
    fn counts_are_exact_for_common_rate_pairs() {
        for &(from, to) in &[(48_000u32, 44_100u32), (16_000, 48_000), (96_000, 48_000), (22_050, 48_000), (48_000, 47_999)] {
            let mut r = PolyphaseResampler::new(from, to, 1);
            let mut next = chunk_sizes(from ^ to, 777);
            let (mut n_in, mut n_out) = (0u64, 0u64);
            while n_in < 30 * u64::from(from) {
                let n = next();
                n_out += r.count_only(n);
                n_in += n as u64;
                assert_eq!(n_out, r.expected_total(n_in), "{from}->{to} after {n_in}");
            }
        }
    }

    /// A 1 kHz tone comes out as the same tone at the new rate, with no content offset, and a
    /// chunked run matches a one-shot run sample for sample (the history is carried correctly).
    #[test]
    fn tone_is_band_limited_and_chunking_invariant() {
        for &(from, to) in &[(44_100u32, 48_000u32), (48_000, 44_100)] {
            let n = from as usize; // 1 s
            let tone: Vec<f32> = (0..n).map(|i| 0.5 * (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / from as f32).sin()).collect();
            let mut whole = Vec::new();
            PolyphaseResampler::new(from, to, 1).process(&tone, &mut whole);
            let mut chunked = Vec::new();
            let mut r = PolyphaseResampler::new(from, to, 1);
            let mut next = chunk_sizes(3, 500);
            let mut pos = 0;
            while pos < n {
                let k = next().min(n - pos);
                r.process(&tone[pos..pos + k], &mut chunked);
                pos += k;
            }
            assert_eq!(whole.len(), chunked.len());
            for (a, b) in whole.iter().zip(chunked.iter()) {
                assert!((a - b).abs() < 1e-6);
            }
            // Past the start-up (silence in the window), output k is the tone at time k / to.
            let mut max_err = 0.0f32;
            for (k, &v) in whole.iter().enumerate().skip(HALF_TAPS * 2) {
                let expect = 0.5 * (2.0 * std::f32::consts::PI * 1_000.0 * k as f32 / to as f32).sin();
                max_err = max_err.max((v - expect).abs());
            }
            assert!(max_err < 2e-3, "{from}->{to}: max error {max_err}");
        }
    }

    #[test]
    fn equal_rates_pass_through() {
        let mut r = PolyphaseResampler::new(48_000, 48_000, 2);
        assert!(r.is_passthrough());
        let mut out = Vec::new();
        r.process(&[0.1, 0.2, 0.3, 0.4, 0.5], &mut out);
        assert_eq!(out, vec![0.1, 0.2, 0.3, 0.4]);
    }
}
