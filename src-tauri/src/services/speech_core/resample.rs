//! Streaming conversion of interleaved device audio (any rate, any channel count) to 16 kHz mono.
//!
//! Channels are averaged first, then a band-limited rubato resampler converts the rate:
//! - [`rubato::FftFixedIn`] (synchronous, exact ratio, BlackmanHarris2-windowed filter) for the usual
//!   device rates (8/11.025/16/22.05/32/44.1/48/88.2/96/192 kHz);
//! - [`rubato::SincFixedIn`] (256-tap BlackmanHarris2 sinc) when the rate pair would need an
//!   impractically large FFT (odd rates such as 47 999 Hz);
//! - pass-through at 16 kHz.
//!
//! Both filters attenuate everything above the 8 kHz output Nyquist by well over 60 dB (see tests),
//! unlike the old nearest-neighbour decimation, which aliased 8-24 kHz content into the speech band.
//! The resampler's group delay is measured once at construction (impulse probe; rubato's reported
//! `output_delay` is not reliable for every engine) and trimmed from the start of the stream, so
//! output sample `n` corresponds to input time `n / 16000` s. Allocations happen only in `new` and
//! `flush` (and when a caller pushes a larger block than ever before).

use rubato::{
    FftFixedIn, Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use super::SAMPLE_RATE;

/// Input frames per resampler call (~21 ms at 48 kHz).
const CHUNK_FRAMES: usize = 1024;
/// Above this FFT input size the sinc resampler is used instead (latency and memory stay bounded).
const MAX_FFT_SIZE_IN: usize = 8192;

enum Engine {
    Passthrough,
    Fft(FftFixedIn<f32>),
    Sinc(SincFixedIn<f32>),
}

impl Engine {
    fn resampler(&mut self) -> Option<&mut dyn DynResampler> {
        match self {
            Engine::Passthrough => None,
            Engine::Fft(r) => Some(r),
            Engine::Sinc(r) => Some(r),
        }
    }
}

/// Object-safe subset of [`rubato::Resampler`] used here.
trait DynResampler {
    fn next_in(&self) -> usize;
    fn max_out(&self) -> usize;
    fn run(&mut self, input: &[f32], out: &mut [f32]) -> Result<(usize, usize), String>;
    fn reset_state(&mut self);
}

impl<R: Resampler<f32>> DynResampler for R {
    fn next_in(&self) -> usize {
        self.input_frames_next()
    }
    fn max_out(&self) -> usize {
        self.output_frames_max()
    }
    fn run(&mut self, input: &[f32], out: &mut [f32]) -> Result<(usize, usize), String> {
        self.process_into_buffer(&[input], &mut [out], None)
            .map_err(|e| format!("resampler: {e}"))
    }
    fn reset_state(&mut self) {
        self.reset();
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// FFT resampler for the usual rate pairs, sinc for odd ones, nothing at 16 kHz.
fn build_engine(input_rate: u32) -> Result<Engine, String> {
    let out = SAMPLE_RATE as usize;
    let inp = input_rate as usize;
    if inp == out {
        return Ok(Engine::Passthrough);
    }
    let g = gcd(inp, out);
    let fft_in = CHUNK_FRAMES.div_ceil(inp / g) * (inp / g);
    let err = |e: &dyn std::fmt::Display| format!("Resampler {inp} Hz -> {out} Hz: {e}");
    if fft_in <= MAX_FFT_SIZE_IN {
        return FftFixedIn::new(inp, out, CHUNK_FRAMES, 1, 1)
            .map(Engine::Fft)
            .map_err(|e| err(&e));
    }
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        oversampling_factor: 256,
        interpolation: SincInterpolationType::Cubic,
        window: WindowFunction::BlackmanHarris2,
    };
    SincFixedIn::new(out as f64 / inp as f64, 1.0, params, CHUNK_FRAMES, 1)
        .map(Engine::Sinc)
        .map_err(|e| err(&e))
}

/// Group delay in output samples (may be negative: rubato's sinc resampler looks ahead within its
/// chunk). Measured as the energy centroid of the response to a unit impulse placed mid-chunk,
/// minus the impulse's ideal output position; exact for these linear-phase filters, rounded.
fn measure_delay(r: &mut dyn DynResampler, max_out: usize, input_rate: u32) -> Result<i64, String> {
    let at = r.next_in() / 2;
    let ideal = at as f64 * f64::from(SAMPLE_RATE) / f64::from(input_rate);
    let mut input = vec![0.0f32; r.next_in()];
    input[at] = 1.0;
    let mut scratch = vec![0.0f32; max_out];
    let mut response = Vec::new();
    // A few chunks cover the longest filter (8192-tap FFT / 256-tap sinc) several times over.
    for _ in 0..8 {
        input.resize(r.next_in(), 0.0);
        let (_, produced) = r.run(&input, &mut scratch)?;
        response.extend_from_slice(&scratch[..produced]);
        input.iter_mut().for_each(|s| *s = 0.0);
    }
    let peak = response
        .iter()
        .enumerate()
        .fold((0, 0.0f32), |best, (i, &v)| if v.abs() > best.1 { (i, v.abs()) } else { best })
        .0;
    let (lo, hi) = (peak.saturating_sub(64), (peak + 65).min(response.len()));
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (i, &v) in response.iter().enumerate().take(hi).skip(lo) {
        let e = f64::from(v) * f64::from(v);
        num += i as f64 * e;
        den += e;
    }
    let centroid = if den > 0.0 { num / den } else { peak as f64 };
    Ok((centroid - ideal).round() as i64)
}

/// Interleaved device audio → 16 kHz mono, streaming.
pub struct StreamResampler {
    input_rate: u32,
    channels: usize,
    engine: Engine,
    /// Mono samples waiting for a full resampler chunk.
    mono: Vec<f32>,
    /// Scratch output of one resampler call.
    scratch: Vec<f32>,
    /// Output samples still to discard (the rest of the group delay at the stream start).
    skip: usize,
    /// Zero samples still to prepend (a negative group delay) at the stream start.
    lead: usize,
    /// Measured group delay in output samples.
    delay: i64,
    /// Interleaved samples of an incomplete frame carried to the next push.
    partial: Vec<f32>,
    /// Mono input frames accepted since the stream start (or the last flush/reset).
    in_frames: u64,
    /// 16 kHz samples emitted since the stream start (or the last flush/reset).
    out_frames: u64,
}

impl StreamResampler {
    pub fn new(input_rate: u32, channels: u16) -> Result<Self, String> {
        if input_rate == 0 || channels == 0 {
            return Err(format!("Unsupported capture format: {input_rate} Hz, {channels} channels"));
        }
        let engine = build_engine(input_rate)?;
        let mut this = Self {
            input_rate,
            channels: usize::from(channels),
            engine,
            mono: Vec::with_capacity(CHUNK_FRAMES * 4),
            scratch: Vec::new(),
            skip: 0,
            lead: 0,
            delay: 0,
            partial: Vec::with_capacity(usize::from(channels)),
            in_frames: 0,
            out_frames: 0,
        };
        if let Some(r) = this.engine.resampler() {
            let max_out = r.max_out();
            this.scratch = vec![0.0; max_out];
            // Measured on a separate instance so this one starts from a clean state.
            let mut probe = build_engine(input_rate)?;
            if let Some(p) = probe.resampler() {
                this.delay = measure_delay(p, max_out, input_rate)?;
            }
            this.restart_alignment();
        }
        Ok(this)
    }

    /// Append the 16 kHz mono conversion of `interleaved` to `out`. Partial frames are carried over.
    pub fn push_interleaved(&mut self, mut interleaved: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
        let ch = self.channels;
        let inv = 1.0 / ch as f32;
        let before = self.mono.len();
        if !self.partial.is_empty() {
            let take = (ch - self.partial.len()).min(interleaved.len());
            self.partial.extend_from_slice(&interleaved[..take]);
            interleaved = &interleaved[take..];
            if self.partial.len() == ch {
                // Same arithmetic as the whole-frame path, so chunking never changes the output.
                let s = self.partial.iter().sum::<f32>() * inv;
                self.partial.clear();
                self.mono.push(s);
            }
        }
        let whole = interleaved.len() / ch * ch;
        if ch == 1 {
            self.mono.extend_from_slice(&interleaved[..whole]);
        } else {
            self.mono
                .extend(interleaved[..whole].chunks_exact(ch).map(|f| f.iter().sum::<f32>() * inv));
        }
        self.partial.extend_from_slice(&interleaved[whole..]);
        self.in_frames += (self.mono.len() - before) as u64;
        self.drain(out)
    }

    /// End of stream: push out everything buffered, including the filter's delayed tail, so the
    /// total output is exactly `round(input_frames * 16000 / input_rate)` samples. The resampler is
    /// then reset for a new stream.
    pub fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), String> {
        self.partial.clear();
        self.drain(out)?;
        let rate = u64::from(self.input_rate);
        let expected = (self.in_frames * u64::from(SAMPLE_RATE) + rate / 2) / rate;
        if let Some(r) = self.engine.resampler() {
            let mut padded = vec![0.0; r.next_in().max(1)];
            // Bounded: the delayed tail is at most a few chunks long.
            for _ in 0..64 {
                if self.out_frames >= expected {
                    break;
                }
                let need = r.next_in();
                padded.clear();
                padded.resize(need, 0.0);
                let real = self.mono.len().min(need);
                padded[..real].copy_from_slice(&self.mono[..real]);
                self.mono.drain(..real);
                let (_, produced) = r.run(&padded, &mut self.scratch)?;
                if self.lead > 0 {
                    let n = self.lead.min((expected - self.out_frames) as usize);
                    out.extend(std::iter::repeat_n(0.0, n));
                    self.out_frames += n as u64;
                    self.lead = 0;
                }
                let drop = self.skip.min(produced);
                self.skip -= drop;
                let keep = (produced - drop).min((expected - self.out_frames) as usize);
                out.extend_from_slice(&self.scratch[drop..drop + keep]);
                self.out_frames += keep as u64;
            }
        }
        self.reset();
        Ok(())
    }

    /// Forget buffered audio and filter state (start of a new stream).
    pub fn reset(&mut self) {
        self.mono.clear();
        self.partial.clear();
        self.in_frames = 0;
        self.out_frames = 0;
        if let Some(r) = self.engine.resampler() {
            r.reset_state();
        }
        self.restart_alignment();
    }

    /// Delay compensation for a new stream: trim a positive delay, pad a negative one.
    fn restart_alignment(&mut self) {
        self.skip = usize::try_from(self.delay.max(0)).unwrap_or(0);
        self.lead = usize::try_from((-self.delay).max(0)).unwrap_or(0);
    }

    /// Resample every complete chunk buffered in `mono`.
    fn drain(&mut self, out: &mut Vec<f32>) -> Result<(), String> {
        let Self {
            engine,
            mono,
            scratch,
            skip,
            lead,
            out_frames,
            ..
        } = self;
        let Some(r) = engine.resampler() else {
            *out_frames += mono.len() as u64;
            out.append(mono);
            return Ok(());
        };
        let mut consumed = 0;
        while mono.len() - consumed >= r.next_in() {
            let need = r.next_in();
            let (used, produced) = r.run(&mono[consumed..consumed + need], scratch)?;
            consumed += used.max(1);
            if *lead > 0 && produced > 0 {
                out.extend(std::iter::repeat_n(0.0, *lead));
                *out_frames += *lead as u64;
                *lead = 0;
            }
            let drop = (*skip).min(produced);
            *skip -= drop;
            out.extend_from_slice(&scratch[drop..produced]);
            *out_frames += (produced - drop) as u64;
        }
        mono.drain(..consumed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, hz: f32, secs: f32, amp: f32) -> Vec<f32> {
        let n = (rate as f32 * secs) as usize;
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin())
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|s| s * s).sum::<f32>() / x.len().max(1) as f32).sqrt()
    }

    fn convert(rate: u32, channels: u16, input: &[f32], chunk: usize) -> Vec<f32> {
        let mut r = StreamResampler::new(rate, channels).unwrap();
        let mut out = Vec::new();
        for c in input.chunks(chunk) {
            r.push_interleaved(c, &mut out).unwrap();
        }
        r.flush(&mut out).unwrap();
        out
    }

    /// Middle part of the output (skips filter ramp-up/down at the edges).
    fn steady(x: &[f32]) -> &[f32] {
        &x[x.len() / 5..x.len() * 4 / 5]
    }

    #[test]
    fn twelve_khz_tone_is_attenuated_by_at_least_60_db() {
        for rate in [48_000u32, 44_100, 96_000, 47_999] {
            let input = tone(rate, 12_000.0, 1.0, 0.5);
            let out = convert(rate, 1, &input, 480);
            let atten_db = 20.0 * (rms(steady(&out)) / rms(&input)).log10();
            assert!(atten_db <= -60.0, "{rate} Hz: 12 kHz tone only {atten_db:.1} dB down");
        }
    }

    #[test]
    fn speech_band_tone_passes_with_correct_length_and_level() {
        for rate in [48_000u32, 44_100, 22_050, 8_000, 47_999] {
            let input = tone(rate, 1_000.0, 2.0, 0.5);
            let out = convert(rate, 1, &input, 441);
            // Exact length: round(input_frames * 16000 / rate).
            let expected = (input.len() as u64 * 16_000 + u64::from(rate) / 2) / u64::from(rate);
            assert_eq!(out.len() as u64, expected, "{rate} Hz");
            let gain_db = 20.0 * (rms(steady(&out)) / rms(&input)).log10();
            assert!(gain_db.abs() < 0.2, "{rate} Hz: 1 kHz gain {gain_db:.2} dB");
            // Group delay is trimmed: the output is phase-aligned with a 16 kHz rendering of the tone.
            // The delay is trimmed in whole output samples, so a sub-sample residue (up to ~0.5
            // sample) remains; 0.08 rms still rejects any whole-sample misalignment (~0.14 rms).
            let reference = tone(SAMPLE_RATE, 1_000.0, 2.2, 0.5);
            let (lo, hi) = (out.len() / 5, out.len() * 4 / 5);
            let err_at = |lag: usize| {
                let diff: Vec<f32> = out[lo..hi].iter().zip(&reference[lo + lag - 1..hi + lag - 1]).map(|(x, y)| x - y).collect();
                rms(&diff)
            };
            let exact = err_at(1);
            let best = (0..3).map(err_at).fold(f32::MAX, f32::min);
            assert!(exact < 0.08 && exact <= best, "{rate} Hz: misaligned (rms error {exact} at lag 0, best {best})");
        }
    }

    #[test]
    fn chunking_does_not_change_the_output() {
        let input: Vec<f32> = tone(48_000, 440.0, 1.0, 0.3)
            .into_iter()
            .zip(tone(48_000, 3_000.0, 1.0, 0.2))
            .flat_map(|(l, r)| [l, r])
            .collect();
        let one_shot = convert(48_000, 2, &input, input.len());
        // Odd chunk sizes split stereo frames mid-frame.
        for chunk in [1usize, 7, 97, 960, 4097] {
            assert_eq!(convert(48_000, 2, &input, chunk), one_shot, "chunk {chunk}");
        }
    }

    #[test]
    fn stereo_is_downmixed_and_16k_mono_passes_through() {
        let l = tone(16_000, 500.0, 0.5, 0.4);
        let stereo: Vec<f32> = l.iter().flat_map(|&s| [s, -s]).collect();
        // Opposite-phase channels cancel.
        assert!(rms(&convert(16_000, 2, &stereo, 333)) < 1e-6);
        assert_eq!(convert(16_000, 1, &l, 100), l);
        let mut r = StreamResampler::new(16_000, 2).unwrap();
        let mut out = Vec::new();
        r.push_interleaved(&[1.0, 0.0, 0.5], &mut out).unwrap();
        r.push_interleaved(&[0.5], &mut out).unwrap();
        assert_eq!(out, vec![0.5, 0.5], "frames split across pushes are averaged whole");
        assert!(StreamResampler::new(0, 1).is_err());
        assert!(StreamResampler::new(48_000, 0).is_err());
    }
}
