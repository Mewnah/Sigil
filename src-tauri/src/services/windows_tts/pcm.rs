//! PCM helpers of the SAPI renderer (pure, unit-tested): the stream format Sigil asks SAPI for,
//! 16-bit little-endian bytes to `f32` (SAPI may split a sample across two writes), and the
//! assembler that pushes whole frames into the sink in small pieces.

use crate::services::tts_sink::SinkError;

/// Where rendered audio goes: the TTS sink utterance in the app, a buffer in tests.
pub(crate) trait PcmOut: Send {
    /// Queue interleaved samples (whole frames). Blocks (bounded) for back-pressure;
    /// `Err(SinkError::Cancelled)` once the utterance was cancelled.
    fn push(&mut self, samples: &[f32], sample_rate: u32, channels: u16) -> Result<(), SinkError>;
}

/// `WAVEFORMATEX::wFormatTag` values an engine may report.
pub(crate) const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// Sample rates accepted from an engine (the sink's own limits are wider).
const MIN_ENGINE_RATE: u32 = 8_000;
const MAX_ENGINE_RATE: u32 = 192_000;

/// The format of the stream SAPI renders into: 16-bit little-endian PCM, `rate` Hz, `channels`
/// interleaved. Sigil always asks for 16-bit at the engine's own rate and channel count, so SAPI
/// never resamples (the sink does, band-limited) and only converts the sample width when an engine
/// is not 16-bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PcmFormat {
    pub rate: u32,
    pub channels: u16,
}

/// Used when an engine does not report its format: SAPI's own default stream format
/// (`SPSF_22kHz16BitMono`).
pub(crate) const FALLBACK_FORMAT: PcmFormat = PcmFormat {
    rate: 22_050,
    channels: 1,
};

/// Bytes per sample of the render stream.
pub(crate) const BYTES_PER_SAMPLE: u16 = 2;

impl PcmFormat {
    /// The render format for what an engine reports from `ISpTTSEngine::GetOutputFormat`, or
    /// `None` when that is not a PCM / float wave format Sigil can use (then [`FALLBACK_FORMAT`]).
    pub(crate) fn from_engine(tag: u16, channels: u16, rate: u32, bits: u16) -> Option<PcmFormat> {
        let known_tag = matches!(tag, WAVE_FORMAT_PCM | WAVE_FORMAT_IEEE_FLOAT | WAVE_FORMAT_EXTENSIBLE);
        let sane = (1..=2).contains(&channels) && (MIN_ENGINE_RATE..=MAX_ENGINE_RATE).contains(&rate) && bits > 0;
        (known_tag && sane).then_some(PcmFormat { rate, channels })
    }

    /// Bytes per frame.
    pub(crate) fn block_align(&self) -> u16 {
        self.channels * BYTES_PER_SAMPLE
    }

    /// Bytes per second.
    pub(crate) fn avg_bytes_per_sec(&self) -> u32 {
        self.rate * u32::from(self.block_align())
    }

    /// Interleaved samples in `ms` milliseconds (whole frames, at least one).
    pub(crate) fn samples_for_ms(&self, ms: u32) -> usize {
        let frames = (u64::from(self.rate) * u64::from(ms) / 1_000).max(1) as usize;
        frames * usize::from(self.channels)
    }
}

/// Decodes 16-bit little-endian samples, keeping an odd trailing byte for the next write.
#[derive(Debug, Default)]
pub(crate) struct Pcm16Decoder {
    carry: Option<u8>,
}

impl Pcm16Decoder {
    /// Append the samples in `bytes` (after any byte left over from the previous call) to `out`.
    pub(crate) fn decode(&mut self, bytes: &[u8], out: &mut Vec<f32>) {
        let mut rest = bytes;
        if let Some(lo) = self.carry.take() {
            match rest.split_first() {
                Some((&hi, tail)) => {
                    out.push(sample(lo, hi));
                    rest = tail;
                }
                None => {
                    self.carry = Some(lo);
                    return;
                }
            }
        }
        let (pairs, odd) = rest.as_chunks::<2>();
        out.extend(pairs.iter().map(|p| sample(p[0], p[1])));
        self.carry = odd.first().copied();
    }

    /// Whether half a sample is waiting for its second byte.
    #[cfg(test)]
    pub(crate) fn has_partial(&self) -> bool {
        self.carry.is_some()
    }
}

fn sample(lo: u8, hi: u8) -> f32 {
    f32::from(i16::from_le_bytes([lo, hi])) / 32_768.0
}

/// Collects decoded samples and pushes whole frames once `flush_samples` are pending (and the rest
/// at [`PcmAssembler::finish`]).
pub(crate) struct PcmAssembler {
    format: PcmFormat,
    decoder: Pcm16Decoder,
    pending: Vec<f32>,
    flush_samples: usize,
    frames: u64,
}

impl PcmAssembler {
    /// `flush_ms`: audio collected before a push (smaller pushes cost more, larger ones start later).
    pub(crate) fn new(format: PcmFormat, flush_ms: u32) -> Self {
        let flush_samples = format.samples_for_ms(flush_ms);
        Self {
            format,
            decoder: Pcm16Decoder::default(),
            pending: Vec::with_capacity(flush_samples * 2),
            flush_samples,
            frames: 0,
        }
    }

    pub(crate) fn format(&self) -> PcmFormat {
        self.format
    }

    /// Frames pushed so far.
    pub(crate) fn frames(&self) -> u64 {
        self.frames
    }

    /// Decode `bytes`; push whole frames into `out` once enough are pending. Returns the frames
    /// pushed by this call.
    pub(crate) fn write(&mut self, bytes: &[u8], out: &mut dyn PcmOut) -> Result<u64, SinkError> {
        self.decoder.decode(bytes, &mut self.pending);
        if self.pending.len() >= self.flush_samples {
            self.push_whole_frames(out)
        } else {
            Ok(0)
        }
    }

    /// Push every whole frame still pending; a trailing partial frame (or half sample) is dropped.
    pub(crate) fn finish(&mut self, out: &mut dyn PcmOut) -> Result<u64, SinkError> {
        let pushed = self.push_whole_frames(out)?;
        self.pending.clear();
        self.decoder = Pcm16Decoder::default();
        Ok(pushed)
    }

    fn push_whole_frames(&mut self, out: &mut dyn PcmOut) -> Result<u64, SinkError> {
        let ch = usize::from(self.format.channels.max(1));
        let whole = self.pending.len() / ch * ch;
        if whole == 0 {
            return Ok(0);
        }
        out.push(&self.pending[..whole], self.format.rate, self.format.channels)?;
        self.pending.drain(..whole);
        let frames = (whole / ch) as u64;
        self.frames += frames;
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Collect {
        pushes: Vec<(Vec<f32>, u32, u16)>,
        fail_with: Option<SinkError>,
    }

    impl PcmOut for Collect {
        fn push(&mut self, samples: &[f32], sample_rate: u32, channels: u16) -> Result<(), SinkError> {
            if let Some(e) = self.fail_with.clone() {
                return Err(e);
            }
            self.pushes.push((samples.to_vec(), sample_rate, channels));
            Ok(())
        }
    }

    fn le(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    #[test]
    fn decodes_full_scale_and_carries_odd_bytes() {
        let bytes = le(&[0, 16_384, -16_384, i16::MAX, i16::MIN]);
        let mut whole = Vec::new();
        Pcm16Decoder::default().decode(&bytes, &mut whole);
        assert_eq!(whole, vec![0.0, 0.5, -0.5, 32_767.0 / 32_768.0, -1.0]);
        // Any split of the same bytes decodes to the same samples.
        for split in 0..=bytes.len() {
            let mut d = Pcm16Decoder::default();
            let mut out = Vec::new();
            d.decode(&bytes[..split], &mut out);
            d.decode(&bytes[split..], &mut out);
            assert_eq!(out, whole, "split at {split}");
            assert!(!d.has_partial());
        }
        // Byte by byte.
        let mut d = Pcm16Decoder::default();
        let mut out = Vec::new();
        for b in &bytes {
            d.decode(std::slice::from_ref(b), &mut out);
        }
        assert_eq!(out, whole);
        // An empty write keeps the carry.
        let mut d = Pcm16Decoder::default();
        let mut out = Vec::new();
        d.decode(&bytes[..1], &mut out);
        d.decode(&[], &mut out);
        assert!(d.has_partial() && out.is_empty());
    }

    #[test]
    fn engine_formats_are_validated() {
        assert_eq!(
            PcmFormat::from_engine(WAVE_FORMAT_PCM, 1, 16_000, 16),
            Some(PcmFormat { rate: 16_000, channels: 1 })
        );
        // 8-bit engines get 16-bit at their rate (SAPI widens the samples).
        assert_eq!(
            PcmFormat::from_engine(WAVE_FORMAT_PCM, 2, 11_025, 8),
            Some(PcmFormat { rate: 11_025, channels: 2 })
        );
        assert!(PcmFormat::from_engine(WAVE_FORMAT_EXTENSIBLE, 1, 24_000, 16).is_some());
        assert!(PcmFormat::from_engine(WAVE_FORMAT_IEEE_FLOAT, 1, 48_000, 32).is_some());
        // Compressed, silly rates, surround, no bits.
        assert_eq!(PcmFormat::from_engine(0x55, 1, 22_050, 16), None);
        assert_eq!(PcmFormat::from_engine(WAVE_FORMAT_PCM, 1, 1, 16), None);
        assert_eq!(PcmFormat::from_engine(WAVE_FORMAT_PCM, 1, 400_000, 16), None);
        assert_eq!(PcmFormat::from_engine(WAVE_FORMAT_PCM, 6, 48_000, 16), None);
        assert_eq!(PcmFormat::from_engine(WAVE_FORMAT_PCM, 0, 48_000, 16), None);
        assert_eq!(PcmFormat::from_engine(WAVE_FORMAT_PCM, 1, 48_000, 0), None);
        let f = PcmFormat { rate: 22_050, channels: 2 };
        assert_eq!((f.block_align(), f.avg_bytes_per_sec()), (4, 88_200));
        assert_eq!(f.samples_for_ms(10), 220 * 2);
        assert_eq!(f.samples_for_ms(0), 2, "at least one frame");
    }

    #[test]
    fn assembler_pushes_whole_frames_in_pieces_and_flushes_the_rest() {
        let format = PcmFormat { rate: 1_000, channels: 2 };
        let mut a = PcmAssembler::new(format, 10); // 10 frames = 20 samples per push
        let mut out = Collect::default();
        // 7 frames and half a sample: nothing pushed yet.
        let mut bytes = le(&[1; 14]);
        bytes.push(0);
        assert_eq!(a.write(&bytes, &mut out).unwrap(), 0);
        assert!(out.pushes.is_empty());
        // The half sample completes; 12.5 frames pending -> 12 frames pushed, half a frame kept.
        let mut more = vec![0u8];
        more.extend(le(&[2; 10]));
        assert_eq!(a.write(&more, &mut out).unwrap(), 12);
        assert_eq!(out.pushes.len(), 1);
        assert_eq!(out.pushes[0].0.len(), 24);
        assert_eq!((out.pushes[0].1, out.pushes[0].2), (1_000, 2));
        // finish: the dangling half frame (one sample) is dropped, nothing else pending.
        assert_eq!(a.finish(&mut out).unwrap(), 0);
        assert_eq!(a.frames(), 12);
        assert_eq!(a.format(), format);
        // Mono: finish pushes the remainder.
        let mut a = PcmAssembler::new(PcmFormat { rate: 16_000, channels: 1 }, 50);
        let mut out = Collect::default();
        assert_eq!(a.write(&le(&[100; 300]), &mut out).unwrap(), 0);
        assert_eq!(a.finish(&mut out).unwrap(), 300);
        assert_eq!(out.pushes[0].0.len(), 300);
    }

    #[test]
    fn assembler_reports_sink_errors() {
        let mut a = PcmAssembler::new(PcmFormat { rate: 8_000, channels: 1 }, 1);
        let mut out = Collect {
            fail_with: Some(SinkError::Cancelled),
            ..Collect::default()
        };
        assert_eq!(a.write(&le(&[1; 64]), &mut out), Err(SinkError::Cancelled));
        assert_eq!(a.frames(), 0);
    }
}
