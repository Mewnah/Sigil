//! What plays when a block's conversion fails: [`MuteConceal`] mutes it.
//!
//! A failed block (ONNX error, sidecar timeout or bad reply, RVC error) is never replaced by the
//! unconverted microphone: people use the voice changer to not sound like themselves, and a late
//! sidecar under GPU load would otherwise put bursts of the real voice on stream. Instead the
//! block fades from where the last converted block ended to silence within
//! [`CONCEAL_FADE_MS`], stays silent while failures last, and the first converted block after
//! them fades back in, so neither edge clicks. The stand-in has exactly the length the converted
//! block would have had, so the output timeline never drifts.

/// Fade length at both edges of a muted run.
pub(crate) const CONCEAL_FADE_MS: f64 = 5.0;

/// Mutes failed blocks (interleaved f32, fixed channel count) with click-free edges.
pub(crate) struct MuteConceal {
    ch: usize,
    fade_frames: usize,
    /// Last frame of the last block played (the fade-out starts there).
    last: Vec<f32>,
    muted: bool,
}

impl MuteConceal {
    pub(crate) fn new(ch: usize, rate: u32) -> Self {
        let ch = ch.max(1);
        Self {
            ch,
            fade_frames: ((CONCEAL_FADE_MS * f64::from(rate.max(1)) / 1000.0).round() as usize).max(1),
            last: vec![0.0; ch],
            muted: false,
        }
    }

    /// Whether the previous block was muted.
    #[cfg(test)]
    pub(crate) fn is_muted(&self) -> bool {
        self.muted
    }

    /// A block that converted: after a muted run it fades in; its last frame is remembered.
    pub(crate) fn pass(&mut self, block: &mut [f32]) {
        let ch = self.ch;
        let frames = block.len() / ch;
        if self.muted {
            let n = self.fade_frames.min(frames);
            for (i, frame) in block.chunks_exact_mut(ch).take(n).enumerate() {
                let g = fade_in_gain(i, n);
                frame.iter_mut().for_each(|s| *s *= g);
            }
            self.muted = false;
        }
        if frames > 0 {
            self.last
                .copy_from_slice(&block[(frames - 1) * ch..frames * ch]);
        }
    }

    /// Fill `out` (cleared first) with `frames` frames standing in for a failed block: on the
    /// first failed block a short fade from the last played frame down to silence, then silence.
    pub(crate) fn mute_into(&mut self, out: &mut Vec<f32>, frames: usize) {
        let ch = self.ch;
        out.clear();
        out.resize(frames * ch, 0.0);
        if !self.muted {
            let n = self.fade_frames.min(frames);
            for (i, frame) in out.chunks_exact_mut(ch).take(n).enumerate() {
                // Frame 0 continues the last played frame; the fade reaches silence at frame n.
                let g = 1.0 - fade_in_gain(i, n);
                for (s, &l) in frame.iter_mut().zip(&self.last) {
                    *s = l * g;
                }
            }
        }
        self.muted = true;
        self.last.fill(0.0);
    }
}

/// Raised-cosine gain for frame `i` of an `n`-frame fade-in (strictly between 0 and 1).
fn fade_in_gain(i: usize, n: usize) -> f32 {
    let t = (i as f32 + 0.5) / n.max(1) as f32;
    0.5 - 0.5 * (std::f32::consts::PI * t).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The largest sample-to-sample step of a stereo stream (either channel).
    fn max_step(stream: &[f32], ch: usize) -> f32 {
        let mut worst = 0.0f32;
        for c in 0..ch {
            let chan: Vec<f32> = stream.iter().skip(c).step_by(ch).copied().collect();
            for w in chan.windows(2) {
                worst = worst.max((w[1] - w[0]).abs());
            }
        }
        worst
    }

    /// A failed block never carries the input: it fades from the last played frame to silence
    /// (no click), stays silent while failures last, and conversion fades back in; the stand-in
    /// always has the owed length.
    #[test]
    fn failed_blocks_are_muted_with_smooth_edges() {
        let (ch, rate) = (2usize, 48_000u32);
        let mut c = MuteConceal::new(ch, rate);
        let fade = c.fade_frames;
        assert_eq!(fade, 240);
        let good = |v: f32, frames: usize| -> Vec<f32> { (0..frames).flat_map(|_| [v, -v]).collect() };
        let mut played = Vec::new();
        let mut block = good(0.8, 512);
        c.pass(&mut block);
        assert_eq!(block, good(0.8, 512), "no conceal while converting");
        played.extend_from_slice(&block);

        let mut out = Vec::new();
        for owed in [512usize, 100, 530] {
            c.mute_into(&mut out, owed);
            assert_eq!(out.len(), owed * ch);
            played.extend_from_slice(&out);
        }
        assert!(c.is_muted());
        // The first muted block starts where the converted audio ended and reaches silence
        // within the fade; the rest of the run is silent.
        let first_muted = &played[512 * ch..];
        assert!((first_muted[0] - 0.8).abs() < 0.01 && (first_muted[1] + 0.8).abs() < 0.01);
        assert!(first_muted[fade * ch..].iter().all(|&s| s == 0.0));

        let mut block = good(0.5, 512);
        c.pass(&mut block);
        assert!(!c.is_muted());
        assert!(block[0].abs() < 0.01, "fades back in: {}", block[0]);
        assert_eq!(&block[fade * ch..], &good(0.5, 512 - fade)[..]);
        played.extend_from_slice(&block);
        // No step anywhere larger than a raised-cosine fade of the full level allows.
        let allowed = 0.8 * std::f32::consts::PI / (2.0 * fade as f32) * 1.05;
        assert!(max_step(&played, ch) <= allowed, "{} > {allowed}", max_step(&played, ch));
    }

    /// A muted block shorter than the fade still ends the fade early at silence, and a zero-length
    /// stand-in is empty.
    #[test]
    fn short_and_empty_stand_ins() {
        let mut c = MuteConceal::new(1, 16_000);
        let mut b = vec![1.0f32; 160];
        c.pass(&mut b);
        let mut out = Vec::new();
        c.mute_into(&mut out, 10);
        assert_eq!(out.len(), 10);
        assert!(out[0] > 0.9 && out[9] < out[0]);
        c.mute_into(&mut out, 0);
        assert!(out.is_empty());
        c.mute_into(&mut out, 5);
        assert!(out.iter().all(|&s| s == 0.0), "only the first muted block fades");
    }
}
