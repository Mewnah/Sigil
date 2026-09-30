//! Streaming format conversion for the TTS sink: band-limited resampling (rubato's synchronous FFT
//! resampler, which keeps its filter state across calls, so audio streamed in arbitrary chunk sizes
//! has no seams), channel adaptation and the utterance gain.
//!
//! Length is exact: after [`StreamConverter::finish`] an utterance of `n` source frames produced
//! exactly `round(n * out_rate / in_rate)` output frames, time-aligned (the resampler's delay is
//! removed at the start and its tail flushed at the end). So consecutive utterances join without a
//! gap or an overlap, whatever the chunking.

use rubato::{FftFixedIn, Resampler};

use crate::services::voice_changer::playback::adapt_channels_interleaved_into;

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

/// Converts one utterance from its source format to an output format.
pub(crate) struct StreamConverter {
    in_rate: u32,
    in_ch: usize,
    out_rate: u32,
    out_ch: usize,
    gain: f32,
    /// `None` when the rates are equal (pass-through).
    resampler: Option<FftFixedIn<f32>>,
    /// Source frames waiting for a full resampler block (planar, source channels).
    pending: Vec<Vec<f32>>,
    /// Resampler output buffer (planar).
    out_buf: Vec<Vec<f32>>,
    /// Output frames still to drop at the start (the resampler's delay).
    delay_left: usize,
    in_frames: u64,
    out_frames: u64,
    /// Scratch: resampled frames interleaved in the source channel count, before adaptation.
    interleaved: Vec<f32>,
    adapted: Vec<f32>,
    finished: bool,
}

impl StreamConverter {
    /// A converter from `in_rate`/`in_ch` to `out_rate`/`out_ch` applying `gain` (clamped output).
    pub(crate) fn new(in_rate: u32, in_ch: usize, out_rate: u32, out_ch: usize, gain: f32) -> Result<Self, String> {
        let in_ch = in_ch.max(1);
        let out_ch = out_ch.max(1);
        if in_rate == 0 || out_rate == 0 {
            return Err("sample rate 0".into());
        }
        let (resampler, delay, out_buf) = if in_rate == out_rate {
            (None, 0, Vec::new())
        } else {
            // rubato's FFT resampler works in blocks of `in_rate / gcd` input frames and reports half
            // its output FFT length as the delay; an odd number of blocks would leave the audio half
            // a sample late, so use an even number (about 1024 input frames per call).
            let block_in = (in_rate / gcd(in_rate, out_rate)) as usize;
            let blocks = 1024usize.div_ceil(block_in).max(2).next_multiple_of(2);
            let rs = FftFixedIn::<f32>::new(in_rate as usize, out_rate as usize, blocks * block_in, 1, in_ch)
                .map_err(|e| format!("resampler {in_rate}->{out_rate} Hz: {e}"))?;
            let delay = rs.output_delay();
            let buf = rs.output_buffer_allocate(true);
            (Some(rs), delay, buf)
        };
        Ok(Self {
            in_rate,
            in_ch,
            out_rate,
            out_ch,
            gain: if gain.is_finite() { gain.clamp(0.0, 2.0) } else { 1.0 },
            resampler,
            pending: vec![Vec::new(); in_ch],
            out_buf,
            delay_left: delay,
            in_frames: 0,
            out_frames: 0,
            interleaved: Vec::new(),
            adapted: Vec::new(),
            finished: false,
        })
    }

    /// Whether this converter takes audio in the format `rate` / `ch`.
    pub(crate) fn accepts(&self, rate: u32, ch: usize) -> bool {
        self.in_rate == rate && self.in_ch == ch.max(1)
    }

    /// Output format `(rate, channels)`.
    #[cfg(test)]
    pub(crate) fn output_format(&self) -> (u32, usize) {
        (self.out_rate, self.out_ch)
    }

    /// Output frames the source frames seen so far map to (exact once finished).
    fn expected_out(&self) -> u64 {
        (self.in_frames * u64::from(self.out_rate) + u64::from(self.in_rate) / 2) / u64::from(self.in_rate)
    }

    /// Convert interleaved source samples (whole frames), appending output samples to `out`.
    pub(crate) fn push(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
        debug_assert!(!self.finished, "push after finish");
        let frames = input.len() / self.in_ch;
        if frames == 0 {
            return Ok(());
        }
        let input = &input[..frames * self.in_ch];
        self.in_frames += frames as u64;
        if self.resampler.is_none() {
            self.out_frames += frames as u64;
            self.emit_interleaved_source_rate(input, out);
            return Ok(());
        }
        for (c, p) in self.pending.iter_mut().enumerate() {
            p.extend(input.iter().skip(c).step_by(self.in_ch).map(|&s| if s.is_finite() { s } else { 0.0 }));
        }
        self.drain_full_blocks(out)
    }

    /// Flush the resampler's tail: after this the output holds exactly the expected frame count.
    pub(crate) fn finish(&mut self, out: &mut Vec<f32>) -> Result<(), String> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        if self.resampler.is_none() {
            return Ok(());
        }
        let expected = self.expected_out();
        // The partial last block (zero-padded by rubato), then silence until the delay is out.
        let mut guard = 0;
        while self.out_frames < expected && guard < 64 {
            let rs = self.resampler.as_mut().expect("resampler");
            let partial = self.pending[0].len();
            let result = if partial > 0 {
                let slices: Vec<&[f32]> = self.pending.iter().map(|p| p.as_slice()).collect();
                rs.process_partial_into_buffer(Some(slices.as_slice()), &mut self.out_buf, None)
            } else {
                rs.process_partial_into_buffer(None::<&[&[f32]]>, &mut self.out_buf, None)
            };
            let (used, produced) = result.map_err(|e| format!("resampler: {e}"))?;
            for p in self.pending.iter_mut() {
                p.drain(..used.min(p.len()));
            }
            self.take_output(produced, Some(expected), out);
            guard += 1;
        }
        // Rounding can leave the resampler one frame short: pad with silence.
        if self.out_frames < expected {
            let missing = (expected - self.out_frames) as usize;
            out.extend(std::iter::repeat_n(0.0, missing * self.out_ch));
            self.out_frames = expected;
        }
        Ok(())
    }

    fn drain_full_blocks(&mut self, out: &mut Vec<f32>) -> Result<(), String> {
        loop {
            let rs = self.resampler.as_mut().expect("resampler");
            let need = rs.input_frames_next();
            if self.pending[0].len() < need {
                return Ok(());
            }
            let (used, produced) = {
                let slices: Vec<&[f32]> = self.pending.iter().map(|p| &p[..need]).collect();
                rs.process_into_buffer(slices.as_slice(), &mut self.out_buf, None)
                    .map_err(|e| format!("resampler: {e}"))?
            };
            for p in self.pending.iter_mut() {
                p.drain(..used);
            }
            self.take_output(produced, None, out);
        }
    }

    /// Move `produced` resampled frames from `out_buf` to `out`, dropping the start delay and never
    /// going past `limit` output frames.
    fn take_output(&mut self, produced: usize, limit: Option<u64>, out: &mut Vec<f32>) {
        let skip = self.delay_left.min(produced);
        self.delay_left -= skip;
        let mut n = produced - skip;
        if let Some(limit) = limit {
            n = n.min(limit.saturating_sub(self.out_frames) as usize);
        }
        if n == 0 {
            return;
        }
        self.interleaved.clear();
        self.interleaved.reserve(n * self.in_ch);
        for f in skip..skip + n {
            for c in 0..self.in_ch {
                self.interleaved.push(self.out_buf[c][f]);
            }
        }
        self.out_frames += n as u64;
        let buf = std::mem::take(&mut self.interleaved);
        self.emit_interleaved_source_rate(&buf, out);
        self.interleaved = buf;
    }

    /// Channel-adapt and gain interleaved frames (source channel count) into `out`.
    fn emit_interleaved_source_rate(&mut self, frames: &[f32], out: &mut Vec<f32>) {
        let gain = self.gain;
        let finite_gain = |s: f32| if s.is_finite() { (s * gain).clamp(-1.0, 1.0) } else { 0.0 };
        if self.in_ch == self.out_ch {
            out.extend(frames.iter().map(|&s| finite_gain(s)));
            return;
        }
        self.adapted.clear();
        adapt_channels_interleaved_into(frames, self.in_ch, self.out_ch, &mut self.adapted);
        out.extend(self.adapted.iter().map(|&s| finite_gain(s)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(frames: usize, rate: u32, hz: f32) -> Vec<f32> {
        (0..frames).map(|i| 0.5 * (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin()).collect()
    }

    fn convert_in_chunks(input: &[f32], chunk: usize, mut conv: StreamConverter) -> Vec<f32> {
        let mut out = Vec::new();
        for c in input.chunks(chunk) {
            conv.push(c, &mut out).unwrap();
        }
        conv.finish(&mut out).unwrap();
        out
    }

    /// Exact output length and a clean, time-aligned tone for common TTS -> device rates, whatever
    /// the chunking (streamed audio has no seams).
    #[test]
    fn streamed_resampling_is_exact_and_seamless() {
        for &(from, to) in &[(24_000u32, 48_000u32), (22_050, 48_000), (16_000, 44_100), (48_000, 24_000), (44_100, 48_000)] {
            let frames = from as usize * 3 / 2 + 5;
            let input = tone(frames, from, 440.0);
            let expected = ((frames as u64 * u64::from(to) + u64::from(from) / 2) / u64::from(from)) as usize;
            let whole = convert_in_chunks(&input, frames, StreamConverter::new(from, 1, to, 1, 1.0).unwrap());
            let streamed = convert_in_chunks(&input, 97, StreamConverter::new(from, 1, to, 1, 1.0).unwrap());
            assert_eq!(whole.len(), expected, "{from}->{to}");
            assert_eq!(streamed.len(), expected, "{from}->{to}");
            let mut max_err = 0.0f32;
            for (k, (&a, &b)) in whole.iter().zip(&streamed).enumerate().take(expected - 1_500).skip(1_500) {
                let t = k as f32 / to as f32;
                let ideal = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
                max_err = max_err.max((a - ideal).abs()).max((b - ideal).abs());
            }
            assert!(max_err < 5e-3, "{from}->{to}: max error {max_err}");
        }
    }

    /// Same rate: a pass-through with channel adaptation, gain and clamping; NaN becomes silence.
    #[test]
    fn passthrough_adapts_channels_and_applies_gain() {
        let mut conv = StreamConverter::new(24_000, 1, 24_000, 2, 2.0).unwrap();
        let mut out = Vec::new();
        conv.push(&[0.25, -0.75, f32::NAN], &mut out).unwrap();
        conv.finish(&mut out).unwrap();
        assert_eq!(out, vec![0.5, 0.5, -1.0, -1.0, 0.0, 0.0]);
        assert_eq!(conv.output_format(), (24_000, 2));
        assert!(conv.accepts(24_000, 1) && !conv.accepts(48_000, 1));

        let mut down = StreamConverter::new(16_000, 2, 16_000, 1, 1.0).unwrap();
        let mut out = Vec::new();
        down.push(&[0.5, -0.5, 0.25, 0.25], &mut out).unwrap();
        assert_eq!(out, vec![0.0, 0.25]);
    }

    /// An utterance shorter than one resampler block still comes out with its exact length.
    #[test]
    fn tiny_utterances_keep_their_length() {
        for frames in [1usize, 7, 100] {
            let out = convert_in_chunks(&vec![0.1; frames], 3, StreamConverter::new(24_000, 1, 48_000, 2, 1.0).unwrap());
            assert_eq!(out.len(), frames * 2 * 2, "{frames}");
        }
        let mut empty = StreamConverter::new(24_000, 1, 48_000, 1, 1.0).unwrap();
        let mut out = Vec::new();
        empty.finish(&mut out).unwrap();
        assert!(out.is_empty());
    }
}
