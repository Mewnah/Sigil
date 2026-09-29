//! Benchmark input signals: seeded synthetic clips, WAV read / write, offline band-limited
//! resampling, and the analytic chirp-burst train used by the latency probe and the loopback
//! meter.
//!
//! Every generator is deterministic for a given seed, so two runs of a spec feed identical audio.
//! `scripts/bench/make-test-clips.py` writes the same kinds of clips as WAV files (for the corpus
//! folder and for feeding other apps in comparisons).

use std::f64::consts::PI;
use std::path::Path;

use rubato::{calculate_cutoff, Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

use super::spec::{InputSpec, SignalKind};

/// SplitMix64: small, fast, seedable; good enough for test signals and jitter.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

pub fn db_to_amp(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

/// Scale so the absolute peak equals `peak` (silence stays silence).
fn normalize_peak(x: &mut [f32], peak: f64) {
    let max = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if max > 0.0 {
        let g = (peak / f64::from(max)) as f32;
        x.iter_mut().for_each(|v| *v *= g);
    }
}

/// Mono → `ch` interleaved (every channel the same).
pub fn to_interleaved(mono: &[f32], ch: usize) -> Vec<f32> {
    if ch <= 1 {
        return mono.to_vec();
    }
    mono.iter().flat_map(|&s| std::iter::repeat_n(s, ch)).collect()
}

/// Interleaved → mono (channel average).
pub fn to_mono(interleaved: &[f32], ch: usize) -> Vec<f32> {
    if ch <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(ch)
        .map(|f| f.iter().sum::<f32>() / ch as f32)
        .collect()
}

/// Channel of an interleaved buffer.
pub fn channel(interleaved: &[f32], ch: usize, which: usize) -> Vec<f32> {
    if ch <= 1 {
        return interleaved.to_vec();
    }
    interleaved.chunks_exact(ch).map(|f| f[which.min(ch - 1)]).collect()
}

/// The spec's input clip at `rate` Hz with `channels` channels (interleaved).
pub fn render_input(input: &InputSpec, rate: u32, channels: usize, seed: u64) -> Result<Vec<f32>, String> {
    let mono = match input {
        InputSpec::Synthetic {
            signal,
            seconds,
            level_dbfs,
        } => synth(*signal, rate, *seconds, db_to_amp(*level_dbfs), seed),
        InputSpec::Wav { path } => {
            let (data, sr, ch) = read_wav(path)?;
            let mono = to_mono(&data, ch);
            if mono.is_empty() {
                return Err(format!("{} holds no audio", path.display()));
            }
            resample(&mono, sr, rate)?
        }
    };
    Ok(to_interleaved(&mono, channels))
}

/// Generate `seconds` of a synthetic signal at `rate` with absolute peak `peak`.
pub fn synth(kind: SignalKind, rate: u32, seconds: f64, peak: f64, seed: u64) -> Vec<f32> {
    let n = (seconds * f64::from(rate)).round().max(0.0) as usize;
    let mut rng = SplitMix64::new(seed);
    let mut x = match kind {
        SignalKind::Silence => return vec![0.0; n],
        SignalKind::Voice => voice(rate, n, &mut rng),
        SignalKind::Tones => (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                [311.0f64, 523.0, 797.0].iter().map(|f| (2.0 * PI * f * t).sin()).sum::<f64>() as f32
            })
            .collect(),
        SignalKind::ChirpTrain => {
            let train = BurstTrain::irregular(seconds, 0.04, 0.35, &mut rng);
            train.render(rate, n)
        }
        SignalKind::ImpulseTrain => {
            let mut v = vec![0.0f32; n];
            let mut t = 0.1;
            while t < seconds {
                let i = (t * f64::from(rate)) as usize;
                if i < n {
                    v[i] = 1.0;
                }
                t += rng.range(0.25, 0.45);
            }
            v
        }
        SignalKind::Noise => (0..n).map(|_| rng.range(-1.0, 1.0) as f32).collect(),
    };
    normalize_peak(&mut x, peak);
    x
}

/// Vowel formants (F1, F2, F3 in Hz) of /a/, /i/, /u/, /e/, /o/ (adult male averages).
const VOWELS: [[f64; 3]; 5] = [
    [730.0, 1090.0, 2440.0],
    [270.0, 2290.0, 3010.0],
    [300.0, 870.0, 2240.0],
    [530.0, 1840.0, 2480.0],
    [570.0, 840.0, 2410.0],
];
const FORMANT_BW: [f64; 3] = [90.0, 110.0, 170.0];
const FORMANT_GAIN: [f64; 3] = [1.0, 0.6, 0.3];

/// Speech-like test voice: a band-limited glottal source (harmonics up to 4 kHz, -6 dB/oct)
/// on a gliding F0 (95-230 Hz, 5.5 Hz vibrato), shaped by formants that move from vowel to vowel
/// per syllable (~4 syllables/s), syllable envelopes, short fricative noise bursts and pauses.
/// Voiced most of the time, so every pitch extractor tracks it; not intelligible.
fn voice(rate: u32, n: usize, rng: &mut SplitMix64) -> Vec<f32> {
    let sr = f64::from(rate);
    // Syllable plan: (start s, duration s, vowel index), with pauses between phrases.
    let total = n as f64 / sr;
    let mut plan: Vec<(f64, f64, usize)> = Vec::new();
    let mut t = 0.15;
    let mut phrase_left = rng.range(1.2, 2.4);
    while t < total {
        let dur = rng.range(0.16, 0.30);
        plan.push((t, dur, (rng.next_u64() % VOWELS.len() as u64) as usize));
        t += dur + rng.range(0.01, 0.05);
        phrase_left -= dur;
        if phrase_left <= 0.0 {
            t += rng.range(0.25, 0.5);
            phrase_left = rng.range(1.2, 2.4);
        }
    }
    let base_f0 = rng.range(105.0, 145.0);
    let (p1, p2) = (rng.range(0.0, 6.0), rng.range(0.0, 6.0));
    let mut out = vec![0.0f32; n];
    let mut phase = 0.0f64;
    let mut syl = 0usize;
    let mut noise_lp = 0.0f64;
    for (i, o) in out.iter_mut().enumerate() {
        let t = i as f64 / sr;
        while syl + 1 < plan.len() && plan[syl].0 + plan[syl].1 < t {
            syl += 1;
        }
        let f0 = (base_f0
            * (1.0 + 0.28 * (2.0 * PI * 0.21 * t + p1).sin() + 0.12 * (2.0 * PI * 0.63 * t + p2).sin())
            * (1.0 + 0.012 * (2.0 * PI * 5.5 * t).sin()))
        .clamp(95.0, 230.0);
        phase = (phase + 2.0 * PI * f0 / sr) % (2.0 * PI * 1024.0);
        let Some(&(start, dur, vowel)) = plan.get(syl) else {
            continue;
        };
        let local = (t - start) / dur;
        if !(0.0..=1.0).contains(&local) {
            continue;
        }
        // Formants glide from the previous syllable's vowel over the first 30 %.
        let prev = if syl > 0 { plan[syl - 1].2 } else { vowel };
        let g = (local / 0.3).min(1.0);
        let formants: [f64; 3] = std::array::from_fn(|k| VOWELS[prev][k] + g * (VOWELS[vowel][k] - VOWELS[prev][k]));
        let env = (PI * local).sin().powf(0.6);
        let harmonics = ((4_000.0 / f0) as usize).max(1);
        let mut s = 0.0f64;
        for h in 1..=harmonics {
            let fh = h as f64 * f0;
            let mut shape = 0.0;
            for k in 0..3 {
                let d = (fh - formants[k]) / FORMANT_BW[k];
                shape += FORMANT_GAIN[k] / (1.0 + d * d);
            }
            s += (0.05 + shape) / h as f64 * (h as f64 * phase).sin();
        }
        // A fricative-like noise burst at the start of every third syllable.
        let white = rng.range(-1.0, 1.0);
        let hp = white - noise_lp;
        noise_lp = 0.7 * noise_lp + 0.3 * white;
        let fric = if syl.is_multiple_of(3) && local < 0.18 { 0.25 * hp * (1.0 - local / 0.18) } else { 0.0 };
        *o = (env * s + fric) as f32;
    }
    out
}

/// A train of Hann-windowed log chirps at given onsets. Evaluated analytically, so the same train
/// renders sample-aligned at any rate (the probe compares a 48 kHz input with a 44.1 kHz output).
#[derive(Debug, Clone)]
pub struct BurstTrain {
    /// Burst start times (s), ascending, non-overlapping.
    pub onsets_s: Vec<f64>,
    pub burst_s: f64,
    pub f_lo: f64,
    pub f_hi: f64,
    pub amp: f64,
}

impl BurstTrain {
    /// Bursts over `seconds`, spaced uniformly at random in `mean_gap * [0.8, 1.2]` (irregular
    /// spacing: no periodic ambiguity for cross-correlation). 200 Hz - 4 kHz, which passes the
    /// voice changer's 16 kHz model domain.
    pub fn irregular(seconds: f64, burst_s: f64, mean_gap: f64, rng: &mut SplitMix64) -> Self {
        let mut onsets = Vec::new();
        let mut t = 0.05;
        while t + burst_s < seconds {
            onsets.push(t);
            t += burst_s + mean_gap * rng.range(0.8, 1.2);
        }
        Self {
            onsets_s: onsets,
            burst_s,
            f_lo: 200.0,
            f_hi: 4_000.0,
            amp: 1.0,
        }
    }

    /// One burst at local time `tau` (0 outside `[0, burst_s)`).
    pub fn burst_at(&self, tau: f64) -> f64 {
        if !(0.0..self.burst_s).contains(&tau) {
            return 0.0;
        }
        let k = (self.f_hi / self.f_lo).ln() / self.burst_s;
        let phase = 2.0 * PI * self.f_lo * ((k * tau).exp() - 1.0) / k;
        let w = 0.5 - 0.5 * (2.0 * PI * tau / self.burst_s).cos();
        self.amp * w * phase.sin()
    }

    /// Render `len` samples at `rate` (time 0 = sample 0).
    pub fn render(&self, rate: u32, len: usize) -> Vec<f32> {
        let sr = f64::from(rate);
        let mut out = vec![0.0f32; len];
        for &on in &self.onsets_s {
            let first = (on * sr).ceil().max(0.0) as usize;
            let last = (((on + self.burst_s) * sr).ceil() as usize).min(len);
            for (i, o) in out.iter_mut().enumerate().take(last).skip(first) {
                *o += self.burst_at(i as f64 / sr - on) as f32;
            }
        }
        out
    }

    /// One burst sampled at `rate`, starting at sample 0 (the matched-filter template).
    pub fn template(&self, rate: u32) -> Vec<f32> {
        let sr = f64::from(rate);
        let len = (self.burst_s * sr).ceil() as usize;
        (0..len).map(|i| self.burst_at(i as f64 / sr) as f32).collect()
    }
}

/// Read a WAV file as interleaved f32: `(samples, rate, channels)`.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32, usize), String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let spec = reader.spec();
    let ch = usize::from(spec.channels.max(1));
    let data: Result<Vec<f32>, _> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample.clamp(1, 32) - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect()
        }
    };
    let data = data.map_err(|e| format!("cannot decode {}: {e}", path.display()))?;
    Ok((data, spec.sample_rate, ch))
}

/// Write interleaved f32 audio as a 32-bit float WAV.
pub fn write_wav(path: &Path, samples: &[f32], rate: u32, channels: usize) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: channels.max(1) as u16,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let err = |e: hound::Error| format!("cannot write {}: {e}", path.display());
    let mut w = hound::WavWriter::create(path, spec).map_err(err)?;
    for &s in samples {
        w.write_sample(s).map_err(err)?;
    }
    w.finalize().map_err(err)
}

/// Band-limited offline resampling of a mono signal (the harness's sinc parameters), exactly
/// `round(len * to / from)` long. rubato's `SincFixedIn` output is time-aligned with its input;
/// its `output_delay()` frames are the filter's look-ahead, which only arrive once later input
/// (here: zero padding) pushes them out, so nothing is trimmed at the start.
pub fn resample(mono: &[f32], from: u32, to: u32) -> Result<Vec<f32>, String> {
    if from == to || mono.is_empty() {
        return Ok(mono.to_vec());
    }
    const CHUNK: usize = 1024;
    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: calculate_cutoff(128, WindowFunction::BlackmanHarris2),
        oversampling_factor: 128,
        interpolation: SincInterpolationType::Cubic,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut rs = SincFixedIn::<f32>::new(f64::from(to) / f64::from(from), 1.0, params, CHUNK, 1)
        .map_err(|e| format!("resampler {from}->{to}: {e}"))?;
    let want = ((mono.len() as u64 * u64::from(to) + u64::from(from) / 2) / u64::from(from)) as usize;
    let mut out: Vec<f32> = Vec::with_capacity(want + CHUNK * 4);
    let mut buf = rs.output_buffer_allocate(true);
    let mut pos = 0usize;
    let mut chunk = vec![0.0f32; CHUNK];
    while out.len() < want {
        chunk.fill(0.0);
        if pos < mono.len() {
            let n = (mono.len() - pos).min(CHUNK);
            chunk[..n].copy_from_slice(&mono[pos..pos + n]);
        }
        pos += CHUNK;
        let (_, produced) = rs
            .process_into_buffer(&[&chunk[..]], &mut buf, None)
            .map_err(|e| format!("resampler: {e}"))?;
        out.extend_from_slice(&buf[0][..produced]);
    }
    out.truncate(want);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(x: &[f32]) -> f64 {
        (x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len().max(1) as f64).sqrt()
    }

    #[test]
    fn synthetic_clips_are_deterministic_and_at_level() {
        for kind in [
            SignalKind::Voice,
            SignalKind::Tones,
            SignalKind::ChirpTrain,
            SignalKind::ImpulseTrain,
            SignalKind::Noise,
        ] {
            let a = synth(kind, 48_000, 2.0, 0.25, 7);
            let b = synth(kind, 48_000, 2.0, 0.25, 7);
            assert_eq!(a.len(), 96_000, "{kind:?}");
            assert_eq!(a, b, "{kind:?} is not deterministic");
            let peak = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!((peak - 0.25).abs() < 1e-5, "{kind:?} peak {peak}");
            assert!(a.iter().all(|v| v.is_finite()));
        }
        assert!(synth(SignalKind::Silence, 16_000, 1.0, 0.5, 1).iter().all(|&v| v == 0.0));
        assert_ne!(synth(SignalKind::Voice, 48_000, 1.0, 0.5, 1), synth(SignalKind::Voice, 48_000, 1.0, 0.5, 2));
    }

    #[test]
    fn test_voice_is_mostly_voiced_with_pauses() {
        let v = synth(SignalKind::Voice, 16_000, 8.0, 0.5, 3);
        // 20 ms frames: most carry energy, some are pauses.
        let frames: Vec<f64> = v.chunks(320).map(rms).collect();
        let loud = frames.iter().filter(|&&r| r > 0.02).count();
        let quiet = frames.iter().filter(|&&r| r < 1e-4).count();
        assert!(loud > frames.len() / 2, "{loud} of {}", frames.len());
        assert!(quiet > 5, "no pauses ({quiet})");
    }

    #[test]
    fn burst_train_renders_aligned_at_any_rate() {
        let mut rng = SplitMix64::new(11);
        let train = BurstTrain::irregular(1.0, 0.03, 0.2, &mut rng);
        assert!(train.onsets_s.len() >= 3);
        let a = train.render(48_000, 48_000);
        let b = train.render(44_100, 44_100);
        // The same continuous signal: sample values agree at shared instants (t = k / 300 s).
        for k in 0..300 {
            let (ia, ib) = (k * 160, k * 147);
            assert!((a[ia] - b[ib]).abs() < 1e-5, "t = {k}/300: {} vs {}", a[ia], b[ib]);
        }
        let tpl = train.template(48_000);
        assert_eq!(tpl.len(), 1_440);
        let first = (train.onsets_s[0] * 48_000.0).ceil() as usize;
        assert!(rms(&a[first..first + 1_440]) > 0.2);
    }

    #[test]
    fn resampling_keeps_timing_and_length() {
        let tone: Vec<f32> = (0..44_100).map(|i| (2.0 * PI * 440.0 * i as f64 / 44_100.0).sin() as f32).collect();
        let up = resample(&tone, 44_100, 48_000).unwrap();
        assert_eq!(up.len(), 48_000);
        let analytic: Vec<f32> = (0..48_000).map(|i| (2.0 * PI * 440.0 * i as f64 / 48_000.0).sin() as f32).collect();
        let e = crate::services::bench::e2e_loopback::estimate_delay(&analytic[..40_000], &up, 50).unwrap();
        assert!(e.lag.abs() < 0.05, "resampled output is offset by {} samples", e.lag);
        // Away from the edges the resampled tone matches the analytic one at 48 kHz (no delay).
        for i in (2_000..46_000).step_by(997) {
            let want = (2.0 * PI * 440.0 * i as f64 / 48_000.0).sin() as f32;
            // The sinc filter's passband ripple is a few tenths of a percent.
            assert!((up[i] - want).abs() < 5e-3, "sample {i}: {} vs {want}", up[i]);
        }
        assert_eq!(resample(&tone, 44_100, 44_100).unwrap(), tone);
    }

    #[test]
    fn wav_roundtrip_and_channel_helpers() {
        let dir = std::env::temp_dir().join(format!("sigil-bench-wav-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.wav");
        let data: Vec<f32> = (0..200).map(|i| (i as f32 / 200.0) - 0.5).collect();
        write_wav(&path, &data, 22_050, 2).unwrap();
        let (back, sr, ch) = read_wav(&path).unwrap();
        assert_eq!((sr, ch), (22_050, 2));
        assert_eq!(back, data);
        assert_eq!(to_mono(&[1.0, 3.0, -1.0, 1.0], 2), vec![2.0, 0.0]);
        assert_eq!(to_interleaved(&[1.0, 2.0], 3), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        assert_eq!(channel(&[1.0, 3.0, -1.0, 1.0], 2, 1), vec![3.0, 1.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
