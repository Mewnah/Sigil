//! From a raw take (recorded or imported) to a reference clip: one pipeline for both sources.
//!
//! 1. Non-finite samples become silence; rates above 48 kHz are resampled to 48 kHz; rates below
//!    16 kHz are refused (a telephone-band clip cannot carry a voice's timbre).
//! 2. The rumble / DC high-pass ([`dsp::remove_rumble`]).
//! 3. Silero VAD over a 16 kHz copy, then [`trim::plan`]: the speech, its pads, long pauses spliced
//!    down, at most 10 s.
//! 4. Measurements: clipping and peak on the raw kept audio, speech-to-noise ratio from the VAD
//!    frames (noise = the median power of the non-speech frames of the whole take, speech = the
//!    mean power of the voiced frames kept), BS.1770 loudness of the clip.
//! 5. Loudness normalisation to [`TARGET_LUFS`] under a [`PEAK_CEILING_DBFS`] sample peak (never a
//!    limiter: when the peak would pass the ceiling the clip is simply quieter), then short fades.
//!
//! A take without speech is kept as it was recorded (no gain: noise is not made loud) so the user
//! can hear what went wrong.

use super::dsp::{self, ClipStats};
use super::trim::{self, TrimConfig, FRAME_SECONDS};
use super::{ClipQuality, CLIP_SECONDS};

/// Loudness of a saved clip (LUFS). Speech corpora and zero-shot / cloning references sit around
/// −20 to −23 LUFS; −20 keeps quiet voices clear without pushing peaks.
pub const TARGET_LUFS: f32 = -20.0;
/// Highest sample peak after normalisation (dBFS).
pub const PEAK_CEILING_DBFS: f32 = -1.0;
/// Most gain the normalisation applies (dB).
pub const MAX_GAIN_DB: f32 = 30.0;
/// Rates above this are resampled down to it.
pub const MAX_RATE: u32 = 48_000;
/// Rates below this are refused.
pub const MIN_RATE: u32 = 16_000;
/// Below this speech-to-noise ratio a take is `noisy` (dB).
pub const NOISY_SNR_DB: f32 = 15.0;
/// Reported when the noise floor is digital silence (dB).
pub const MAX_SNR_DB: f32 = 60.0;
/// A take quieter than this before normalisation is `too_quiet` (LUFS).
pub const TOO_QUIET_LUFS: f32 = -42.0;
/// A take with more clipped samples than this fraction is `clipped`.
pub const CLIPPED_RATIO: f32 = 1e-4;
/// Fewest clipped samples for `clipped` (one full-scale sample is not clipping).
const CLIPPED_MIN_SAMPLES: usize = 3;
/// Fades at the clip's ends and crossfades at spliced pauses (seconds).
const FADE_IN_S: f64 = 0.010;
const FADE_OUT_S: f64 = 0.020;
const HARD_CUT_FADE_S: f64 = 0.050;
const SPLICE_S: f64 = 0.010;

/// Issue codes ([`ClipQuality::issues`]). The first two block saving; the rest are warnings.
pub const ISSUE_NO_SPEECH: &str = "no_speech";
pub const ISSUE_TOO_SHORT: &str = "too_short";
pub const ISSUE_TOO_LONG: &str = "too_long";
pub const ISSUE_NOISY: &str = "noisy";
pub const ISSUE_CLIPPED: &str = "clipped";
pub const ISSUE_TOO_QUIET: &str = "too_quiet";
/// Issues that make a take unusable as a reference.
pub const BLOCKING_ISSUES: [&str; 2] = [ISSUE_NO_SPEECH, ISSUE_TOO_SHORT];

/// A voice-activity detector over 512-sample 16 kHz frames ([`crate::services::speech_core::silero`]
/// in the app; a synthetic one in tests).
pub trait Vad {
    /// Speech probability (0..1) of the next frame.
    fn prob(&mut self, frame: &[f32]) -> Result<f32, String>;
    /// Start of an unrelated stream.
    fn reset(&mut self);
}

impl Vad for crate::services::speech_core::silero::SileroVad {
    fn prob(&mut self, frame: &[f32]) -> Result<f32, String> {
        self.predict(frame)
    }
    fn reset(&mut self) {
        crate::services::speech_core::silero::SileroVad::reset(self)
    }
}

/// The library's VAD: the embedded Silero VAD v6.2 (CPU, one thread).
pub fn silero() -> Result<crate::services::speech_core::silero::SileroVad, String> {
    crate::services::speech_core::silero::SileroVad::new()
}

/// A processed take.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcessedTake {
    /// The clip: mono, normalised, faded.
    pub clip: Vec<f32>,
    pub rate: u32,
    pub quality: ClipQuality,
    /// Some kept samples were clipped (the `clipped` issue).
    pub clipped: bool,
}

/// VAD probabilities of `audio16k`, one per 512-sample frame (the last partial frame zero-padded).
pub fn vad_probs(audio16k: &[f32], vad: &mut dyn Vad) -> Result<Vec<f32>, String> {
    const N: usize = crate::services::speech_core::VAD_FRAME_SAMPLES;
    vad.reset();
    let mut probs = Vec::with_capacity(audio16k.len() / N + 1);
    let mut frame = [0f32; N];
    for chunk in audio16k.chunks(N) {
        frame[..chunk.len()].copy_from_slice(chunk);
        frame[chunk.len()..].fill(0.0);
        probs.push(vad.prob(&frame)?);
    }
    Ok(probs)
}

/// Speech-to-noise ratio (dB) from per-frame powers and VAD probabilities: noise = the median power
/// of frames below `neg_threshold` anywhere in the take, speech = the mean power of the frames at or
/// above `threshold` inside `keep` (frame indices). `None` without enough of either.
pub fn estimate_snr(powers: &[f64], probs: &[f32], keep: &dyn Fn(usize) -> bool, cfg: &TrimConfig) -> Option<f32> {
    const MIN_NOISE_FRAMES: usize = 5;
    const MIN_SPEECH_FRAMES: usize = 10;
    let mut noise: Vec<f64> = powers
        .iter()
        .zip(probs)
        .filter(|(_, &p)| p.is_finite() && p < cfg.neg_threshold)
        .map(|(&e, _)| e)
        .collect();
    let speech: Vec<f64> = powers
        .iter()
        .zip(probs)
        .enumerate()
        .filter(|(i, (_, &p))| p.is_finite() && p >= cfg.threshold && keep(*i))
        .map(|(_, (&e, _))| e)
        .collect();
    if noise.len() < MIN_NOISE_FRAMES || speech.len() < MIN_SPEECH_FRAMES {
        return None;
    }
    noise.sort_by(|a, b| a.total_cmp(b));
    let noise_p = noise[noise.len() / 2];
    let speech_p = speech.iter().sum::<f64>() / speech.len() as f64;
    if noise_p <= 1e-12 {
        return Some(MAX_SNR_DB);
    }
    let snr = 10.0 * ((speech_p - noise_p).max(1e-12) / noise_p).log10();
    Some((snr as f32).clamp(-20.0, MAX_SNR_DB))
}

fn to_samples(seconds: f64, rate: u32, len: usize) -> usize {
    ((seconds * f64::from(rate)).round().max(0.0) as usize).min(len)
}

fn ms(seconds: f64) -> u64 {
    (seconds * 1000.0).round().max(0.0) as u64
}

/// Process a mono take at `rate` (see the module docs).
pub fn process_take(mut audio: Vec<f32>, rate: u32, vad: &mut dyn Vad) -> Result<ProcessedTake, String> {
    if rate < MIN_RATE {
        return Err(format!(
            "the audio's sample rate ({rate} Hz) is too low for a reference voice (at least {MIN_RATE} Hz; a Bluetooth headset in hands-free mode often records at 8 kHz)"
        ));
    }
    if audio.is_empty() {
        return Err("the take holds no audio".into());
    }
    dsp::sanitize(&mut audio);
    let mut rate = rate;
    if rate > MAX_RATE {
        audio = crate::services::bench::signals::resample(&audio, rate, MAX_RATE)?;
        rate = MAX_RATE;
    }
    let raw = audio.clone();
    dsp::remove_rumble(&mut audio, rate);
    let take_seconds = audio.len() as f64 / f64::from(rate);

    // VAD on a 16 kHz copy.
    let audio16k = crate::services::bench::signals::resample(&audio, rate, 16_000)?;
    let probs = vad_probs(&audio16k, vad)?;
    let cfg = TrimConfig::default();
    let plan = trim::plan(&probs, take_seconds, &cfg);
    let powers: Vec<f64> = audio16k
        .chunks(crate::services::speech_core::VAD_FRAME_SAMPLES)
        .map(dsp::mean_square)
        .collect();

    let mut issues: Vec<String> = Vec::new();
    let Some(plan) = plan else {
        // No speech: keep the take as it was (faded), so it can be played back.
        let mut clip = audio;
        dsp::fade_in(&mut clip, (FADE_IN_S * f64::from(rate)) as usize);
        dsp::fade_out(&mut clip, (FADE_OUT_S * f64::from(rate)) as usize);
        let stats = ClipStats::of(&raw);
        let lufs = dsp::integrated_loudness(&clip, rate);
        issues.push(ISSUE_NO_SPEECH.into());
        let quality = ClipQuality {
            duration_ms: ms(take_seconds),
            trimmed_ms: 0,
            snr_db: None,
            peak_dbfs: dsp::amp_to_db(f64::from(dsp::peak(&raw))),
            clipped_ratio: stats.ratio(),
            lufs_before: lufs.unwrap_or(dsp::DB_FLOOR),
            lufs_after: lufs.unwrap_or(dsp::DB_FLOOR),
            ok: false,
            issues,
        };
        return Ok(ProcessedTake { clip, rate, quality, clipped: false });
    };

    // The kept spans, as sample ranges at the clip rate.
    let ranges: Vec<(usize, usize)> = plan
        .spans
        .iter()
        .map(|s| (to_samples(s.start, rate, audio.len()), to_samples(s.end, rate, audio.len())))
        .filter(|(a, b)| b > a)
        .collect();
    let mut stats = ClipStats::default();
    let mut raw_peak = 0.0f32;
    for &(a, b) in &ranges {
        stats.add(ClipStats::of(&raw[a..b]));
        raw_peak = raw_peak.max(dsp::peak(&raw[a..b]));
    }
    let parts: Vec<&[f32]> = ranges.iter().map(|&(a, b)| &audio[a..b]).collect();
    let mut clip = dsp::splice(&parts, (SPLICE_S * f64::from(rate)) as usize);
    let fade_out = if plan.hard_cut { HARD_CUT_FADE_S } else { FADE_OUT_S };
    dsp::fade_in(&mut clip, (FADE_IN_S * f64::from(rate)) as usize);
    dsp::fade_out(&mut clip, (fade_out * f64::from(rate)) as usize);

    // Speech-to-noise ratio (frames whose centre lies in a kept span count as kept).
    let keep = |i: usize| {
        let mid = (i as f64 + 0.5) * FRAME_SECONDS;
        plan.spans.iter().any(|s| mid >= s.start && mid < s.end)
    };
    let snr_db = estimate_snr(&powers, &probs, &keep, &cfg);

    // Loudness normalisation.
    let lufs_before = dsp::integrated_loudness(&clip, rate);
    let peak_db = dsp::amp_to_db(f64::from(dsp::peak(&clip)));
    let gain = dsp::normalization_gain_db(lufs_before, peak_db, TARGET_LUFS, PEAK_CEILING_DBFS, MAX_GAIN_DB);
    dsp::apply_gain(&mut clip, gain);
    let lufs_after = dsp::integrated_loudness(&clip, rate);

    let duration_s = clip.len() as f64 / f64::from(rate);
    if duration_s * 1000.0 + 0.5 < CLIP_SECONDS.0 * 1000.0 {
        issues.push(ISSUE_TOO_SHORT.into());
    }
    if plan.cut {
        issues.push(ISSUE_TOO_LONG.into());
    }
    if snr_db.is_some_and(|s| s < NOISY_SNR_DB) {
        issues.push(ISSUE_NOISY.into());
    }
    let clipped = stats.clipped >= CLIPPED_MIN_SAMPLES && stats.ratio() > CLIPPED_RATIO;
    if clipped {
        issues.push(ISSUE_CLIPPED.into());
    }
    if lufs_before.is_none_or(|l| l < TOO_QUIET_LUFS) {
        issues.push(ISSUE_TOO_QUIET.into());
    }
    let ok = !issues.iter().any(|i| BLOCKING_ISSUES.contains(&i.as_str()));
    let quality = ClipQuality {
        duration_ms: ms(duration_s),
        trimmed_ms: ms((take_seconds - duration_s).max(0.0)),
        snr_db,
        peak_dbfs: dsp::amp_to_db(f64::from(raw_peak)),
        clipped_ratio: stats.ratio(),
        lufs_before: lufs_before.unwrap_or(dsp::DB_FLOOR),
        lufs_after: lufs_after.unwrap_or(dsp::DB_FLOOR),
        ok,
        issues,
    };
    Ok(ProcessedTake { clip, rate, quality, clipped })
}

// ---------------------------------------------------------------------------------------------
// WAV encoding and file decoding
// ---------------------------------------------------------------------------------------------

/// A 16-bit PCM mono WAV of `clip` (the most widely readable format: every TTS server and tool
/// accepts it; the clip peaks at −1 dBFS, so 16 bits keep ~95 dB of range).
pub fn encode_wav(clip: &[f32], rate: u32) -> Result<Vec<u8>, String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::with_capacity(44 + clip.len() * 2));
    {
        let mut w = hound::WavWriter::new(&mut cursor, spec).map_err(|e| format!("cannot encode the clip: {e}"))?;
        let mut i16w = w.get_i16_writer(clip.len() as u32);
        for &v in clip {
            let s = if v.is_finite() { (v.clamp(-1.0, 1.0) * 32767.0).round() as i16 } else { 0 };
            i16w.write_sample(s);
        }
        i16w.flush().map_err(|e| format!("cannot encode the clip: {e}"))?;
        w.finalize().map_err(|e| format!("cannot encode the clip: {e}"))?;
    }
    Ok(cursor.into_inner())
}

/// Mono samples and rate of a WAV in memory (any PCM / float format hound reads).
pub fn decode_wav_bytes(bytes: &[u8]) -> Result<(Vec<f32>, u32), String> {
    let reader = hound::WavReader::new(std::io::Cursor::new(bytes)).map_err(|e| format!("not a WAV clip: {e}"))?;
    decode_hound(reader, f64::INFINITY)
}

fn decode_hound<R: std::io::Read>(mut reader: hound::WavReader<R>, max_seconds: f64) -> Result<(Vec<f32>, u32), String> {
    let spec = reader.spec();
    let ch = usize::from(spec.channels.max(1));
    let rate = spec.sample_rate;
    let max_samples = if max_seconds.is_finite() {
        ((max_seconds * f64::from(rate)) as usize).saturating_mul(ch)
    } else {
        usize::MAX
    };
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .take(max_samples)
            .collect::<Result<_, _>>()
            .map_err(|e| format!("cannot decode the WAV: {e}"))?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample.clamp(1, 32) - 1)) as f32;
            reader
                .samples::<i32>()
                .take(max_samples)
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()
                .map_err(|e| format!("cannot decode the WAV: {e}"))?
        }
    };
    let mut mono = Vec::with_capacity(interleaved.len() / ch);
    dsp::downmix_into(&interleaved, ch, &mut mono);
    Ok((mono, rate))
}

/// Audio file extensions the importer reads (WAV through hound; FLAC, Ogg Vorbis and MP3 through
/// rodio's decoders, already in the build).
pub const IMPORT_EXTENSIONS: [&str; 5] = ["wav", "wave", "flac", "ogg", "mp3"];
/// Longest audio read from an imported file (the first two minutes; the clip keeps ≤ 10 s).
pub const MAX_IMPORT_SECONDS: f64 = 120.0;

/// Decode the first [`MAX_IMPORT_SECONDS`] of an audio file to mono: `(samples, rate)`.
pub fn decode_file(path: &std::path::Path) -> Result<(Vec<f32>, u32), String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    if !meta.is_file() {
        return Err(format!("{} is not a file", path.display()));
    }
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if !IMPORT_EXTENSIONS.contains(&ext.as_str()) {
        return Err(format!(
            "{} is not a supported audio file (use {})",
            path.display(),
            IMPORT_EXTENSIONS.iter().map(|e| format!(".{e}")).collect::<Vec<_>>().join(", ")
        ));
    }
    if ext == "wav" || ext == "wave" {
        // hound keeps 24-bit and float precision; rodio is the fallback for WAV codecs hound
        // does not read (ADPCM, A-law, ...).
        if let Ok(reader) = hound::WavReader::open(path) {
            match decode_hound(reader, MAX_IMPORT_SECONDS) {
                Ok(r) => return Ok(r),
                Err(e) => tracing::debug!(target: "sigil::voice_library", "hound could not decode {}: {e}", path.display()),
            }
        }
    }
    decode_with_rodio(path)
}

fn decode_with_rodio(path: &std::path::Path) -> Result<(Vec<f32>, u32), String> {
    use rodio::Source;
    let file = std::fs::File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let decoder = rodio::Decoder::new(std::io::BufReader::new(file))
        .map_err(|e| format!("cannot decode {}: {e}", path.display()))?;
    let ch = usize::from(decoder.channels().max(1));
    let rate = decoder.sample_rate();
    if rate == 0 {
        return Err(format!("{} has no sample rate", path.display()));
    }
    let max_samples = ((MAX_IMPORT_SECONDS * f64::from(rate)) as usize).saturating_mul(ch);
    let interleaved: Vec<f32> = decoder.take(max_samples).map(|s| f32::from(s) / 32768.0).collect();
    if interleaved.is_empty() {
        return Err(format!("{} holds no audio", path.display()));
    }
    let mut mono = Vec::with_capacity(interleaved.len() / ch);
    dsp::downmix_into(&interleaved, ch, &mut mono);
    Ok((mono, rate))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A VAD that calls a frame speech when its RMS is above a level (synthetic signals).
    pub(crate) struct EnergyVad(pub f32);

    impl Vad for EnergyVad {
        fn prob(&mut self, frame: &[f32]) -> Result<f32, String> {
            Ok(if dsp::power_to_db(dsp::mean_square(frame)) > self.0 { 0.95 } else { 0.02 })
        }
        fn reset(&mut self) {}
    }

    /// The default synthetic VAD: speech above -55 dBFS (the clean takes' noise floor is -74 dBFS,
    /// a voice's quietest syllable trough about -43 dBFS).
    pub(crate) fn vad() -> EnergyVad {
        EnergyVad(-55.0)
    }

    /// A voice-like signal: a 150 Hz harmonic tone with 4 Hz syllable modulation.
    pub(crate) fn voice(rate: u32, seconds: f64, amp: f32) -> Vec<f32> {
        let n = (seconds * f64::from(rate)) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                let env = 0.6 + 0.4 * (std::f64::consts::TAU * 4.0 * t).sin();
                let tone: f64 = (1..=6).map(|h| (std::f64::consts::TAU * 150.0 * h as f64 * t).sin() / h as f64).sum();
                (f64::from(amp) * env * tone / 2.45) as f32
            })
            .collect()
    }

    /// Deterministic white noise at `amp` RMS.
    pub(crate) fn noise(n: usize, amp: f32, seed: u64) -> Vec<f32> {
        let mut rng = crate::services::bench::signals::SplitMix64::new(seed);
        (0..n).map(|_| ((rng.unit() * 2.0 - 1.0) * 3f64.sqrt()) as f32 * amp).collect()
    }

    fn take(rate: u32, lead: f64, speech: f64, trail: f64, speech_amp: f32, noise_amp: f32) -> Vec<f32> {
        let r = f64::from(rate);
        let mut x = vec![0.0f32; (lead * r) as usize];
        x.extend(voice(rate, speech, speech_amp));
        x.extend(std::iter::repeat_n(0.0, (trail * r) as usize));
        for (v, n) in x.iter_mut().zip(noise(((lead + speech + trail) * r) as usize + 8, noise_amp, 7)) {
            *v += n;
        }
        x
    }

    #[test]
    fn a_clean_take_is_trimmed_normalised_and_ok() {
        let rate = 48_000;
        let x = take(rate, 1.5, 6.0, 2.0, 0.1, 0.0002);
        let p = process_take(x, rate, &mut vad()).unwrap();
        let q = &p.quality;
        assert!(q.ok, "{q:?}");
        assert!(q.issues.is_empty(), "{:?}", q.issues);
        assert_eq!(p.rate, rate);
        // 6 s of speech + ~0.5 s of pads (the VAD frames round to 32 ms).
        assert!((6_300..=6_700).contains(&q.duration_ms), "{}", q.duration_ms);
        assert!((2_800..=3_200).contains(&q.trimmed_ms), "{}", q.trimmed_ms);
        assert!((p.clip.len() as i64 - q.duration_ms as i64 * 48).abs() <= 24, "duration matches the clip");
        assert!((q.lufs_after - TARGET_LUFS).abs() < 0.1, "{q:?}");
        assert!(q.lufs_before < -25.0);
        assert!(dsp::peak(&p.clip) <= dsp::db_to_amp(f64::from(PEAK_CEILING_DBFS)) as f32 + 1e-4);
        let snr = q.snr_db.unwrap();
        assert!(snr > 30.0, "{snr}");
        assert!(!p.clipped && q.clipped_ratio == 0.0);
        // Faded ends.
        assert!(p.clip[0].abs() < 1e-3 && p.clip[p.clip.len() - 1].abs() < 1e-3);
    }

    #[test]
    fn noisy_clipped_and_quiet_takes_are_flagged_but_usable() {
        let rate = 44_100;
        // Noise at -40 dBFS under a voice at about -35 dBFS.
        let noisy = take(rate, 1.0, 5.0, 1.0, 0.05, 0.01);
        let q = process_take(noisy, rate, &mut EnergyVad(-37.0)).unwrap().quality;
        assert!(q.ok && q.issues.contains(&ISSUE_NOISY.to_string()), "{q:?}");
        assert!(q.snr_db.unwrap() < NOISY_SNR_DB);

        let mut hot = take(rate, 1.0, 5.0, 1.0, 1.6, 0.0002);
        for v in &mut hot {
            *v = v.clamp(-1.0, 1.0);
        }
        let p = process_take(hot, rate, &mut vad()).unwrap();
        assert!(p.clipped && p.quality.issues.contains(&ISSUE_CLIPPED.to_string()), "{:?}", p.quality);
        assert!(p.quality.ok);

        let quiet = take(rate, 1.0, 5.0, 1.0, 0.004, 0.00001);
        let q = process_take(quiet, rate, &mut EnergyVad(-80.0)).unwrap().quality;
        assert!(q.ok && q.issues.contains(&ISSUE_TOO_QUIET.to_string()), "{q:?}");
    }

    #[test]
    fn short_silent_and_long_takes() {
        let rate = 48_000;
        let short = take(rate, 1.0, 1.5, 1.0, 0.1, 0.0002);
        let q = process_take(short, rate, &mut vad()).unwrap().quality;
        assert!(!q.ok && q.issues == vec![ISSUE_TOO_SHORT.to_string()], "{q:?}");

        let silent = noise(3 * rate as usize, 0.0005, 3);
        let p = process_take(silent.clone(), rate, &mut vad()).unwrap();
        assert!(!p.quality.ok && p.quality.issues == vec![ISSUE_NO_SPEECH.to_string()]);
        assert_eq!(p.clip.len(), silent.len(), "a take without speech is kept whole");
        assert!(dsp::peak(&p.clip) < 0.01, "and is not amplified");

        let long = take(rate, 0.5, 14.0, 0.5, 0.1, 0.0002);
        let q = process_take(long, rate, &mut vad()).unwrap().quality;
        assert!(q.ok && q.issues.contains(&ISSUE_TOO_LONG.to_string()), "{q:?}");
        assert!((8_500..=10_000).contains(&q.duration_ms), "{}", q.duration_ms);
    }

    #[test]
    fn rates_are_bounded() {
        assert!(process_take(vec![0.1; 8_000], 8_000, &mut vad()).is_err());
        assert!(process_take(Vec::new(), 48_000, &mut vad()).is_err());
        let x = take(96_000, 1.0, 4.0, 1.0, 0.1, 0.0002);
        let p = process_take(x, 96_000, &mut vad()).unwrap();
        assert_eq!(p.rate, MAX_RATE);
        assert!(p.quality.ok, "{:?}", p.quality);
        let x = take(16_000, 1.0, 4.0, 1.0, 0.1, 0.0002);
        assert_eq!(process_take(x, 16_000, &mut vad()).unwrap().rate, 16_000);
    }

    #[test]
    fn snr_needs_enough_frames_and_handles_digital_silence() {
        let cfg = TrimConfig::default();
        let all = |_: usize| true;
        let mut powers = vec![1e-6; 20];
        powers.extend(vec![1e-2; 40]);
        let mut probs = vec![0.0f32; 20];
        probs.extend(vec![0.9; 40]);
        let snr = estimate_snr(&powers, &probs, &all, &cfg).unwrap();
        assert!((snr - 40.0).abs() < 0.01, "{snr}");
        assert!(estimate_snr(&powers[15..], &probs[15..], &all, &cfg).is_some());
        assert_eq!(estimate_snr(&powers[17..], &probs[17..], &all, &cfg), None, "3 noise frames are too few");
        let zeros = vec![0.0; 20];
        let mut p2 = zeros.clone();
        p2.extend(vec![1e-2; 40]);
        assert_eq!(estimate_snr(&p2, &probs, &all, &cfg), Some(MAX_SNR_DB));
        assert_eq!(estimate_snr(&powers, &probs, &|_| false, &cfg), None, "no kept speech");
    }

    #[test]
    fn wav_round_trip_and_file_decoding() {
        let rate = 22_050;
        let clip: Vec<f32> = voice(rate, 0.5, 0.5);
        let bytes = encode_wav(&clip, rate).unwrap();
        assert_eq!(&bytes[..4], b"RIFF");
        let (back, r) = decode_wav_bytes(&bytes).unwrap();
        assert_eq!((back.len(), r), (clip.len(), rate));
        assert!(clip.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1.0 / 16_000.0));

        let dir = std::env::temp_dir().join(format!("sigil-vl-decode-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A stereo float WAV is read at full precision and downmixed.
        let stereo: Vec<f32> = clip.iter().flat_map(|&v| [v, 0.0]).collect();
        let wav = dir.join("Take One.wav");
        crate::services::bench::signals::write_wav(&wav, &stereo, rate, 2).unwrap();
        let (mono, r) = decode_file(&wav).unwrap();
        assert_eq!((mono.len(), r), (clip.len(), rate));
        assert!((mono[100] - clip[100] / 2.0).abs() < 1e-6);
        // Unsupported or missing files are refused by name.
        let txt = dir.join("notes.txt");
        std::fs::write(&txt, b"hello").unwrap();
        assert!(decode_file(&txt).unwrap_err().contains("not a supported audio file"));
        assert!(decode_file(&dir.join("missing.wav")).is_err());
        assert!(decode_file(&dir).is_err());
        // A file with an audio extension but no audio in it.
        let fake = dir.join("fake.mp3");
        std::fs::write(&fake, b"not audio at all").unwrap();
        assert!(decode_file(&fake).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
