//! Cutting an over-long utterance at quiet points, for recognizers that cannot decode it whole.
//!
//! Owned by P1-C. Pure signal logic (no model, no I/O) so it is tested with synthetic signals; the
//! only user today is the Moonshine path of `sherpa_offline` (see its `MOONSHINE_MAX_SEGMENT_MS`
//! for the measurements behind the limit).
//!
//! # How a cut is chosen
//! A [`Splitter`] has a maximum chunk length (the *limit*). Audio that fits is one chunk and is
//! never looked at. Otherwise the cut goes into the last half of the limit (a window that ends at
//! the limit): every 20 ms frame in that window is rated by the energy of the [`SPAN_MS`] of audio
//! around it (so a closure inside a word, a few frames long, does not look like a pause; only a
//! stretch of that length is quiet enough), the quietest frame wins (the latest one on a tie, which
//! keeps the chunk long), and the cut is the middle of the quiet stretch around it (the
//! neighbouring frames within 3 dB of the minimum), so a real pause is split in the middle and both
//! sides keep their silence. This is the same idea as `speech_core`'s forced split at the
//! lowest-probability frame, but works on raw samples because the engine has no VAD output.
//!
//! The window never reaches closer than a third of the limit to either end of the audio, which
//! gives the chunk and the rest after it at least that much (a sliver would be decoded alone and
//! could hallucinate or come back empty), so every chunk is at least a third of the limit, except
//! the whole input when it is shorter. Nothing is allocated while cutting.
//!
//! [`Splitter::halve`] cuts one chunk in two the same way, for a second attempt at a chunk the
//! recognizer answered with no text ([`worth_retrying`]).

use std::ops::Range;

/// Length of the frames whose energy picks a cut.
pub(crate) const FRAME_MS: u32 = 20;

/// Length of the audio around a frame whose energy rates it as a cut (the frame in the middle,
/// three frames on each side).
///
/// Measured with Moonshine on the English utterances over 9.3 s of the P7 corpus (56 utterances in
/// 7 recording conditions, cut at 9 s, 432 cuts): the share of cuts with speech within 60 ms on both
/// sides of the cut (inside a word, judged on the clean recording) is 9.6 % when the quietest
/// single frame is cut and 5.1 % with 140 ms (3.9 to 6.7 % for 160 to 250 ms, 5.5 % for 100 ms). The
/// word error rate moves less than the noise of that sample (19.2 % with 140 ms, 19.1 to 19.9 % for
/// the other spans, all with the retry of an empty piece).
pub(crate) const SPAN_MS: u32 = 140;

/// Smallest limit a [`Splitter`] accepts; a smaller request is raised to it (a chunk shorter than
/// this is no longer an utterance a recognizer can do much with).
pub(crate) const MIN_LIMIT_MS: u32 = 2_000;

/// Frames within this factor of the quietest frame (3 dB) belong to the quiet stretch whose middle
/// becomes the cut.
const PLATEAU_RATIO: f32 = 2.0;
/// Mean-square energy added to the plateau threshold so a pause with a noise floor (up to about
/// -60 dBFS) counts as one stretch although its frames differ.
const PLATEAU_FLOOR: f32 = 1e-6;

/// A frame is voiced when its mean-square energy is at least this (RMS 0.01, -40 dBFS).
const VOICED_ENERGY: f32 = 1e-4;
/// Shortest audio [`is_long_and_voiced`] calls long.
pub(crate) const LONG_VOICED_MIN_MS: u32 = 3_000;
/// Share of voiced frames that makes audio clearly voiced.
const VOICED_FRACTION_MIN: f32 = 0.5;

/// Shortest audio [`worth_retrying`] accepts (each half of it is still half a second or more).
pub(crate) const RETRY_MIN_MS: u32 = 1_500;
/// Share of voiced frames from which a chunk without text is worth a second attempt (a chunk of
/// speech with pauses is often only a third voiced).
const RETRY_VOICED_MIN: f32 = 0.3;

/// Mean-square energy of `frame`; a non-finite sample makes the frame infinitely loud, so it is
/// never picked as a quiet point and never panics a comparison.
fn energy(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum: f32 = frame.iter().map(|x| x * x).sum();
    let mean = sum / frame.len() as f32;
    if mean.is_finite() {
        mean
    } else {
        f32::INFINITY
    }
}

fn samples_per(sample_rate: u32, ms: u32) -> usize {
    (u64::from(sample_rate) * u64::from(ms) / 1000) as usize
}

/// Cuts audio into chunks of at most a fixed length at the quietest points near that length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Splitter {
    /// Longest chunk (samples).
    limit: usize,
    /// How far before the limit a cut may be placed (half the limit).
    window: usize,
    /// Shortest piece a cut may leave on either side (a third of the limit).
    min_tail: usize,
    /// Samples per energy frame (the step between cut candidates).
    frame: usize,
    /// Samples around a candidate whose energy rates it (at least one frame).
    span: usize,
}

impl Splitter {
    /// A splitter for chunks of at most `limit_ms` at `sample_rate`; `None` when either is zero
    /// (zero means "do not split"). The limit is raised to [`MIN_LIMIT_MS`].
    pub(crate) fn new(sample_rate: u32, limit_ms: u32) -> Option<Self> {
        if sample_rate == 0 || limit_ms == 0 {
            return None;
        }
        let limit = samples_per(sample_rate, limit_ms.max(MIN_LIMIT_MS));
        let frame = samples_per(sample_rate, FRAME_MS).max(1);
        Some(Self {
            limit,
            window: limit / 2,
            min_tail: limit / 3,
            frame,
            span: samples_per(sample_rate, SPAN_MS).max(frame),
        })
    }

    /// The longest chunk, in samples.
    #[cfg(test)]
    pub(crate) fn limit(&self) -> usize {
        self.limit
    }

    /// The shortest piece a cut leaves, in samples.
    #[cfg(test)]
    pub(crate) fn min_piece(&self) -> usize {
        self.min_tail
    }

    /// The same splitter rating a cut candidate over `span_ms` of audio and keeping `min_tail`
    /// samples on either side (experiments only; `span_ms` of one frame is the plain quietest
    /// frame).
    #[cfg(test)]
    pub(crate) fn tuned(mut self, sample_rate: u32, span_ms: u32, min_tail: usize) -> Self {
        self.span = samples_per(sample_rate, span_ms).max(self.frame);
        self.min_tail = min_tail;
        self
    }

    /// Whether `len` samples are longer than one chunk.
    pub(crate) fn needs_split(&self, len: usize) -> bool {
        len > self.limit
    }

    /// The cut inside `lo..hi`: the middle of the quiet stretch around the quietest frame (the
    /// latest on a tie). Always in `lo..=hi` for `lo <= hi`.
    fn quiet_cut(&self, audio: &[f32], lo: usize, hi: usize) -> usize {
        let frames = (hi - lo) / self.frame;
        if frames == 0 {
            return lo + (hi - lo) / 2;
        }
        // Energy of the span around frame `i` (cut off at the ends of the audio).
        let rate_at = |i: usize| {
            let middle = lo + i * self.frame + self.frame / 2;
            let from = middle.saturating_sub(self.span / 2);
            energy(&audio[from..(from + self.span).min(audio.len())])
        };

        // The quietest frame, the latest on a tie.
        let mut best = 0;
        let mut best_energy = f32::INFINITY;
        for i in 0..frames {
            let e = rate_at(i);
            if e <= best_energy {
                best = i;
                best_energy = e;
            }
        }
        // The quiet stretch around it, and its middle.
        let threshold = best_energy * PLATEAU_RATIO + PLATEAU_FLOOR;
        let mut first = best;
        while first > 0 && rate_at(first - 1) <= threshold {
            first -= 1;
        }
        let mut last = best;
        while last + 1 < frames && rate_at(last + 1) <= threshold {
            last += 1;
        }
        lo + (first + last + 1) * self.frame / 2
    }

    /// End (exclusive) of the chunk that starts at `start`: `audio.len()` when the rest fits in one
    /// chunk, else the cut described in the module docs. Always `> start` for `start <
    /// audio.len()`, the chunk is never longer than the limit, and neither the chunk nor what is
    /// left after it is shorter than a third of the limit.
    pub(crate) fn chunk_end(&self, audio: &[f32], start: usize) -> usize {
        let len = audio.len();
        if start >= len || len - start <= self.limit {
            return len;
        }
        // The latest cut: the limit, but far enough from the end to leave a real tail; the
        // earliest: half a limit before that, but far enough from the start to make a real chunk.
        // Here `len - start > limit`, so `hi >= start + limit - min_tail` and `hi - lo >= min_tail`.
        let hi = (start + self.limit).min(len - self.min_tail);
        let lo = hi.saturating_sub(self.window).max(start + self.min_tail);
        self.quiet_cut(audio, lo, hi)
    }

    /// Where to cut `audio` in two for a second attempt: the quietest point of its middle third, so
    /// that each half keeps at least a third. `None` for audio shorter than three frames.
    pub(crate) fn halve(&self, audio: &[f32]) -> Option<usize> {
        let third = audio.len() / 3;
        (third >= self.frame).then(|| self.quiet_cut(audio, third, audio.len() - third))
    }

    /// The chunks of `audio` as sample ranges (none for empty audio). One range spanning all of
    /// `audio` when it does not need splitting.
    pub(crate) fn chunks<'a>(&'a self, audio: &'a [f32]) -> Chunks<'a> {
        Chunks {
            splitter: self,
            audio,
            start: 0,
        }
    }
}

/// Iterator over the chunk ranges of [`Splitter::chunks`].
pub(crate) struct Chunks<'a> {
    splitter: &'a Splitter,
    audio: &'a [f32],
    start: usize,
}

impl Iterator for Chunks<'_> {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Range<usize>> {
        if self.start >= self.audio.len() {
            return None;
        }
        let end = self.splitter.chunk_end(self.audio, self.start);
        let range = self.start..end;
        self.start = end;
        Some(range)
    }
}

/// Share of the 20 ms frames of `audio` that are voiced (energy-based, not a VAD); `0.0` when there
/// is less than one frame.
pub(crate) fn voiced_fraction(audio: &[f32], sample_rate: u32) -> f32 {
    let frame = samples_per(sample_rate, FRAME_MS).max(1);
    let total = audio.len() / frame;
    if total == 0 {
        return 0.0;
    }
    let voiced = audio
        .chunks_exact(frame)
        .filter(|f| energy(f) >= VOICED_ENERGY)
        .count();
    voiced as f32 / total as f32
}

/// Whether `audio` lasts at least [`LONG_VOICED_MIN_MS`] and most of it is clearly voiced: a
/// recognizer that returns no text for it has most likely failed, not heard silence.
pub(crate) fn is_long_and_voiced(audio: &[f32], sample_rate: u32) -> bool {
    audio.len() >= samples_per(sample_rate, LONG_VOICED_MIN_MS) && voiced_fraction(audio, sample_rate) >= VOICED_FRACTION_MIN
}

/// Whether `audio` is long enough and voiced enough that a recognizer answering it with no text has
/// probably failed on it (not heard silence), so a second attempt on its two halves is worth the
/// time. Looser than [`is_long_and_voiced`], which decides what is worth a warning.
pub(crate) fn worth_retrying(audio: &[f32], sample_rate: u32) -> bool {
    audio.len() >= samples_per(sample_rate, RETRY_MIN_MS) && voiced_fraction(audio, sample_rate) >= RETRY_VOICED_MIN
}

/// Scripts written without spaces between words: no space is added next to them.
fn is_unspaced_script(c: char) -> bool {
    matches!(u32::from(c),
        0x0E00..=0x0EFF      // Thai, Lao
        | 0x1000..=0x109F    // Myanmar
        | 0x1780..=0x17FF    // Khmer
        | 0x3000..=0x30FF    // CJK punctuation, Hiragana, Katakana
        | 0x3400..=0x4DBF    // CJK extension A
        | 0x4E00..=0x9FFF    // CJK unified ideographs
        | 0xF900..=0xFAFF    // CJK compatibility ideographs
        | 0xFF00..=0xFFEF    // full-width forms
        | 0x20000..=0x2FA1F) // CJK extensions B..F and compatibility supplement
}

/// Whether two pieces of text meeting at `prev` | `next` need a space between them.
fn needs_space(prev: char, next: char) -> bool {
    if prev.is_whitespace() || next.is_whitespace() || is_unspaced_script(prev) || is_unspaced_script(next) {
        return false;
    }
    !(matches!(next, ',' | '.' | ';' | ':' | '!' | '?' | '%' | ')' | ']' | '}' | '…') || matches!(prev, '(' | '[' | '{'))
}

/// Append `part` (trimmed; nothing for an empty one) to `out`, with a space between them unless
/// the pieces meet punctuation or a script that has no word spaces.
pub(crate) fn append_text(out: &mut String, part: &str) {
    let part = part.trim();
    let Some(first) = part.chars().next() else { return };
    if let Some(last) = out.chars().next_back() {
        if needs_space(last, first) {
            out.push(' ');
        }
    }
    out.push_str(part);
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;
    const MS: usize = 16; // samples per millisecond

    /// Constant-energy "speech": alternating +-0.3 (mean square 0.09 in every frame).
    fn loud(ms: usize) -> Vec<f32> {
        (0..ms * MS)
            .map(|i| if i % 2 == 0 { 0.3 } else { -0.3 })
            .collect()
    }

    fn silence(ms: usize) -> Vec<f32> {
        vec![0.0; ms * MS]
    }

    fn join(parts: &[Vec<f32>]) -> Vec<f32> {
        parts.concat()
    }

    fn splitter() -> Splitter {
        Splitter::new(RATE, 8_000).unwrap()
    }

    fn ranges(s: &Splitter, audio: &[f32]) -> Vec<Range<usize>> {
        s.chunks(audio).collect()
    }

    /// The ranges tile `0..len` with no gap or overlap, none is longer than the limit.
    fn assert_tiles(s: &Splitter, audio: &[f32], chunks: &[Range<usize>]) {
        assert_eq!(chunks.first().map(|r| r.start), Some(0));
        assert_eq!(chunks.last().map(|r| r.end), Some(audio.len()));
        for pair in chunks.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }
        for r in chunks {
            assert!(r.start < r.end && r.len() <= s.limit(), "{r:?} (limit {})", s.limit());
        }
        // No sliver: with more than one chunk, every chunk is at least a third of the limit.
        if chunks.len() > 1 {
            for r in chunks {
                assert!(r.len() >= s.min_piece() && s.min_piece() == s.limit() / 3, "{r:?} of {chunks:?}");
            }
        }
    }

    #[test]
    fn a_zero_rate_or_limit_does_not_split_and_a_small_limit_is_raised() {
        assert_eq!(Splitter::new(0, 8_000), None);
        assert_eq!(Splitter::new(RATE, 0), None);
        assert_eq!(Splitter::new(RATE, 100).unwrap().limit(), 2_000 * MS);
        assert_eq!(Splitter::new(RATE, 8_000).unwrap().limit(), 8_000 * MS);
        assert_eq!(Splitter::new(48_000, 8_000).unwrap().limit(), 8_000 * 48);
    }

    #[test]
    fn audio_that_fits_is_one_chunk_and_the_limit_is_inclusive() {
        let s = splitter();
        assert!(ranges(&s, &[]).is_empty());
        let short = loud(3_000);
        assert!(!s.needs_split(short.len()));
        assert_eq!(ranges(&s, &short), vec![0..short.len()]);
        let exact = loud(8_000);
        assert!(!s.needs_split(exact.len()));
        assert_eq!(ranges(&s, &exact), vec![0..exact.len()]);
        let over = loud(8_000).into_iter().chain([0.3]).collect::<Vec<_>>();
        assert!(s.needs_split(over.len()));
        let chunks = ranges(&s, &over);
        assert_eq!(chunks.len(), 2, "{chunks:?}");
        assert_tiles(&s, &over, &chunks);
    }

    #[test]
    fn a_pause_in_the_window_is_cut_in_the_middle() {
        let s = splitter();
        // The window of the first chunk is 4.0 s..8.0 s; the pause 6.0..6.4 s is in it.
        let audio = join(&[loud(6_000), silence(400), loud(13_600)]);
        let chunks = ranges(&s, &audio);
        assert_eq!(chunks[0], 0..6_200 * MS, "{chunks:?}");
        assert_tiles(&s, &audio, &chunks);
    }

    #[test]
    fn a_pause_outside_the_window_is_ignored_and_without_a_pause_the_middle_of_the_window_is_cut() {
        let s = splitter();
        // The only pause is at 2.0 s, before the window: it never moves a cut.
        let audio = join(&[loud(2_000), silence(400), loud(17_600)]);
        let chunks = ranges(&s, &audio);
        assert_eq!(
            chunks,
            vec![0..6_000 * MS, 6_000 * MS..12_000 * MS, 12_000 * MS..20_000 * MS],
            "{chunks:?}"
        );
        assert_tiles(&s, &audio, &chunks);
        // No pause at all: the same.
        let flat = loud(20_000);
        assert_eq!(ranges(&s, &flat), chunks);
    }

    #[test]
    fn of_two_equally_quiet_pauses_the_later_one_wins() {
        let s = splitter();
        let audio = join(&[loud(5_000), silence(200), loud(1_800), silence(200), loud(12_800)]);
        // Pauses: 5.0..5.2 s and 7.0..7.2 s.
        assert_eq!(s.chunk_end(&audio, 0), 7_100 * MS);
    }

    #[test]
    fn the_quietest_pause_wins_over_a_later_shallower_one() {
        let s = splitter();
        let faint: Vec<f32> = (0..300 * MS)
            .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        // A deep pause at 5.0..5.2 s and a shallow one (-40 dBFS) at 7.0..7.3 s.
        let audio = join(&[loud(5_000), silence(200), loud(1_800), faint, loud(12_700)]);
        assert_eq!(s.chunk_end(&audio, 0), 5_100 * MS);
    }

    #[test]
    fn a_pause_at_the_very_end_of_the_window_is_used_and_the_limit_is_never_passed() {
        let s = splitter();
        // A 200 ms pause ending at the limit: the cut is in it, near its end, never past the limit.
        let audio = join(&[loud(7_800), silence(200), loud(12_000)]);
        let end = s.chunk_end(&audio, 0);
        // Only the cut candidates whose whole 140 ms span is silent (up to 7 930 ms) are quietest:
        // the cut is the middle of those, 7 900 ms.
        assert!((7_870 * MS..=7_930 * MS).contains(&end), "{} ms", end / MS);
        assert!(end <= s.limit());
    }

    #[test]
    fn a_dip_inside_speech_is_not_taken_for_a_pause() {
        let s = splitter();
        // A 160 ms pause at 5.0 s and, later, a 40 ms hole (a stop closure inside a word) at 6.5 s:
        // the quietest single frame is in the hole (the later one on a tie), but the hole is not
        // quiet over 140 ms, so the pause is cut.
        let audio = join(&[loud(5_000), silence(160), loud(1_340), silence(40), loud(13_460)]);
        assert_eq!(audio.len(), 20_000 * MS);
        let end = s.chunk_end(&audio, 0);
        assert!((5_000 * MS..=5_160 * MS).contains(&end), "{} ms", end / MS);
        // The plain quietest-frame rule (a span of one frame) takes the hole.
        let plain = s.tuned(RATE, FRAME_MS, s.limit() / 4);
        let plain_end = plain.chunk_end(&audio, 0);
        assert!((6_500 * MS..=6_540 * MS).contains(&plain_end), "{} ms", plain_end / MS);
        // Only a hole: it is still the best place there is.
        let only_hole = join(&[loud(6_500), silence(40), loud(13_460)]);
        let end = s.chunk_end(&only_hole, 0);
        assert!((6_000 * MS..=7_000 * MS).contains(&end), "{} ms", end / MS);
    }

    #[test]
    fn the_tail_is_never_a_sliver() {
        let s = splitter();
        // Just over the limit: the cut moves forward-of-limit so the tail keeps a third of it.
        for extra_ms in [1usize, 50, 400, 1_999, 2_000, 2_666, 2_667, 2_668] {
            let audio = loud(8_000 + extra_ms);
            let chunks = ranges(&s, &audio);
            assert_tiles(&s, &audio, &chunks);
            let last = chunks.last().unwrap();
            assert!(last.len() >= s.min_piece(), "{extra_ms} ms over: {chunks:?}");
            assert!(chunks[0].len() >= s.min_piece(), "{extra_ms} ms over: {chunks:?}");
        }
        // The same with a pause where the unclamped cut would have been.
        let audio = join(&[loud(7_900), silence(100), loud(500)]);
        let chunks = ranges(&s, &audio);
        assert_tiles(&s, &audio, &chunks);
        assert!(chunks.last().unwrap().len() >= s.min_piece());
        // And with a pause too close to the start: the first chunk is not a sliver either.
        let audio = join(&[loud(500), silence(200), loud(7_500), silence(100), loud(200)]);
        let chunks = ranges(&s, &audio);
        assert_tiles(&s, &audio, &chunks);
        assert!(chunks[0].len() >= s.min_piece(), "{chunks:?}");
    }

    #[test]
    fn halving_cuts_the_middle_third_at_its_quietest_point() {
        let s = splitter();
        // A 6.4 s piece: the middle third is 2.13 s..4.27 s; the pause at 3.0 s..3.2 s is cut.
        let piece = join(&[loud(3_000), silence(200), loud(3_200)]);
        // (The frames start at the middle third, not at a whole millisecond: the middle of the pause
        // is found within a frame.)
        let cut = s.halve(&piece).unwrap();
        assert!((3_080 * MS..=3_120 * MS).contains(&cut), "{} ms", cut / MS);
        // Pauses outside the middle third (at 0.5 s and 5.5 s) are ignored; without one in it the
        // cut is the middle of the third.
        let piece = join(&[loud(500), silence(200), loud(4_800), silence(200), loud(700)]);
        let cut = s.halve(&piece).unwrap();
        let (lo, hi) = (piece.len() / 3, piece.len() - piece.len() / 3);
        let frames = (hi - lo) / (FRAME_MS as usize * MS);
        assert_eq!(cut, lo + frames * FRAME_MS as usize * MS / 2, "{} ms", cut / MS);
        assert!(cut >= piece.len() / 3 && piece.len() - cut >= piece.len() / 3);
        // Too short to cut: fewer than three frames.
        assert_eq!(s.halve(&loud(50)), None);
        assert_eq!(s.halve(&[]), None);
        // Every length from three frames up keeps both halves non-empty and at least a third each.
        for len in [3 * 320, 3 * 320 + 1, 1_000, 24_000, 48_000, 100_003] {
            let audio = vec![0.3f32; len];
            let cut = s.halve(&audio).unwrap();
            assert!(cut >= len / 3 && len - cut >= len / 3, "len {len}: cut {cut}");
        }
    }

    #[test]
    fn a_very_long_input_is_cut_in_pauses_throughout() {
        let s = splitter();
        // Ten minutes: 3.0 s of speech, 0.3 s of pause, repeated. Every 4 s window holds a pause.
        let unit = join(&[loud(3_000), silence(300)]);
        let audio: Vec<f32> = unit
            .iter()
            .copied()
            .cycle()
            .take(unit.len() * 182)
            .collect();
        assert!(audio.len() > 600 * 16_000);
        let chunks = ranges(&s, &audio);
        assert_tiles(&s, &audio, &chunks);
        assert!(chunks.len() >= 75, "{}", chunks.len());
        for r in &chunks[..chunks.len() - 1] {
            let phase = r.end % unit.len();
            assert!(
                (3_000 * MS..=3_300 * MS).contains(&phase),
                "cut at {} ms is not in a pause (phase {} ms)",
                r.end / MS,
                phase / MS
            );
        }
    }

    #[test]
    fn every_length_tiles_with_chunks_within_the_limit_and_no_short_tail() {
        let s = splitter();
        let mut x = 12_345u32;
        let noise: Vec<f32> = (0..60 * 16_000)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((x >> 8) as f32 / 16_777_216.0 - 0.5) * 0.6
            })
            .collect();
        let mut len = 8_000 * MS;
        while len <= noise.len() {
            let audio = &noise[..len];
            let chunks = ranges(&s, audio);
            assert_tiles(&s, audio, &chunks);
            if chunks.len() > 1 {
                assert!(chunks.last().unwrap().len() >= s.min_piece(), "len {len}: {chunks:?}");
                assert!(chunks.iter().all(|r| r.len() >= s.min_piece()), "len {len}: {chunks:?}");
            }
            len += 5_923;
        }
    }

    #[test]
    fn non_finite_samples_neither_panic_nor_stall() {
        let s = splitter();
        let mut audio = loud(20_000);
        for i in (4_000 * MS..8_000 * MS).step_by(7) {
            audio[i] = f32::NAN;
        }
        audio[100] = f32::INFINITY;
        let chunks = ranges(&s, &audio);
        assert_tiles(&s, &audio, &chunks);
    }

    #[test]
    fn voiced_audio_is_told_from_silence_noise_and_short_clips() {
        let loud_5s = loud(5_000);
        assert!(is_long_and_voiced(&loud_5s, RATE));
        assert!((voiced_fraction(&loud_5s, RATE) - 1.0).abs() < 1e-6);
        assert!(!is_long_and_voiced(&silence(5_000), RATE));
        // -60 dBFS noise is not voice.
        let hiss: Vec<f32> = (0..5_000 * MS)
            .map(|i| if i % 2 == 0 { 0.001 } else { -0.001 })
            .collect();
        assert!(!is_long_and_voiced(&hiss, RATE));
        // Voiced but short.
        assert!(!is_long_and_voiced(&loud(2_000), RATE));
        assert!(is_long_and_voiced(&loud(3_000), RATE));
        // Speech with pauses: 60 % voiced counts, 30 % does not.
        assert!(is_long_and_voiced(&join(&[loud(3_000), silence(2_000)]), RATE));
        assert!(!is_long_and_voiced(&join(&[loud(1_500), silence(3_500)]), RATE));
        assert_eq!(voiced_fraction(&[0.5; 100], RATE), 0.0, "less than one frame");
        assert_eq!(voiced_fraction(&[], RATE), 0.0);
    }

    #[test]
    fn only_voiced_audio_of_some_length_is_worth_a_second_attempt() {
        assert!(worth_retrying(&loud(1_500), RATE));
        assert!(worth_retrying(&loud(6_000), RATE));
        assert!(!worth_retrying(&loud(1_400), RATE), "too short");
        assert!(!worth_retrying(&silence(6_000), RATE));
        let hiss: Vec<f32> = (0..6_000 * MS)
            .map(|i| if i % 2 == 0 { 0.001 } else { -0.001 })
            .collect();
        assert!(!worth_retrying(&hiss, RATE), "-60 dBFS noise is not voice");
        // Speech with long pauses: a third voiced counts (it does not for a warning), 20 % does not.
        let third = join(&[loud(2_000), silence(4_000)]);
        assert!(worth_retrying(&third, RATE) && !is_long_and_voiced(&third, RATE));
        assert!(!worth_retrying(&join(&[loud(1_200), silence(4_800)]), RATE));
    }

    fn joined(parts: &[&str]) -> String {
        let mut out = String::new();
        for p in parts {
            append_text(&mut out, p);
        }
        out
    }

    #[test]
    fn pieces_of_text_are_joined_with_single_spaces() {
        assert_eq!(
            joined(&["Hello there.", "  how are you  ", "", "   ", "Fine."]),
            "Hello there. how are you Fine."
        );
        assert_eq!(joined(&["", "only"]), "only");
        assert_eq!(joined(&["", " "]), "");
        // Punctuation that starts a piece attaches to the text before it.
        assert_eq!(joined(&["Well", ", maybe", ".", "Yes"]), "Well, maybe. Yes");
        assert_eq!(joined(&["f(", "x", ") ok"]), "f(x) ok");
        assert_eq!(joined(&["100", "% sure"]), "100% sure");
    }

    #[test]
    fn scripts_without_word_spaces_are_joined_without_one() {
        assert_eq!(joined(&["こんにちは", "世界"]), "こんにちは世界");
        assert_eq!(joined(&["今日は天気です。", "明日は雨です。"]), "今日は天気です。明日は雨です。");
        assert_eq!(joined(&["สวัสดี", "ครับ"]), "สวัสดีครับ");
        // Hangul has word spaces.
        assert_eq!(joined(&["안녕하세요", "여러분"]), "안녕하세요 여러분");
        assert_eq!(joined(&["Привет", "мир"]), "Привет мир");
    }
}
