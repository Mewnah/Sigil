//! Where the speech is: turns per-frame Silero VAD probabilities (512-sample frames at 16 kHz,
//! 32 ms) into the spans of a take to keep. Pure and deterministic, so it is tested on synthetic
//! probability tracks.
//!
//! The rules, in order:
//! 1. **Speech runs** by hysteresis: a run starts at a frame ≥ `threshold` and ends at the first
//!    frame below `neg_threshold`.
//! 2. Runs closer than `merge_gap_ms` are one segment (a short dip inside a word or between two
//!    quick words). Segments shorter than `min_run_ms` are dropped (a click, a lip smack, a cough
//!    tail).
//! 3. Each segment keeps `lead_pad_ms` before and `tail_pad_ms` after it (onsets and decays).
//!    Pauses up to `max_pause_ms` stay as they are; a longer pause is spliced down to the two pads
//!    (a reference clip should be speech, not waiting).
//! 4. A take longer than `max_clip_ms` is cut at the last speech-run end that keeps it within the
//!    limit (a pause between words); continuous speech without such a pause is cut at the least
//!    voiced frame of the last 1.5 s before the limit. Nothing is cut below `min_clip_ms`.

/// Duration of one VAD frame (512 samples at 16 kHz).
pub const FRAME_SECONDS: f64 = 512.0 / 16_000.0;

/// Trimming rules (see the module docs). The defaults are the library's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrimConfig {
    pub threshold: f32,
    pub neg_threshold: f32,
    pub merge_gap_ms: f64,
    pub min_run_ms: f64,
    pub lead_pad_ms: f64,
    pub tail_pad_ms: f64,
    pub max_pause_ms: f64,
    pub min_clip_ms: f64,
    pub max_clip_ms: f64,
}

impl Default for TrimConfig {
    fn default() -> Self {
        TrimConfig {
            threshold: 0.5,
            neg_threshold: 0.35,
            merge_gap_ms: 200.0,
            min_run_ms: 90.0,
            lead_pad_ms: 200.0,
            tail_pad_ms: 300.0,
            max_pause_ms: 900.0,
            min_clip_ms: 3_000.0,
            max_clip_ms: 10_000.0,
        }
    }
}

/// A span of the take to keep, in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Span {
    pub start: f64,
    pub end: f64,
}

impl Span {
    pub fn len(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }
}

/// What to keep of a take.
#[derive(Clone, Debug, PartialEq)]
pub struct TrimPlan {
    /// Spans in time order, non-overlapping; spliced together they are the clip.
    pub spans: Vec<Span>,
    /// The take held more than `max_clip_ms` of speech and was cut.
    pub cut: bool,
    /// The cut fell inside continuous speech (no pause was close enough).
    pub hard_cut: bool,
    /// Frames at or above `threshold` inside the kept spans.
    pub voiced_frames: usize,
}

impl TrimPlan {
    #[cfg(test)]
    pub fn duration(&self) -> f64 {
        self.spans.iter().map(Span::len).sum()
    }
}

/// A run of frames `[start, end)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Run {
    start: usize,
    end: usize,
}

/// Hysteresis speech runs (rule 1).
fn speech_runs(probs: &[f32], cfg: &TrimConfig) -> Vec<Run> {
    let mut runs = Vec::new();
    let mut open: Option<usize> = None;
    for (i, &p) in probs.iter().enumerate() {
        let p = if p.is_finite() { p } else { 0.0 };
        match open {
            None if p >= cfg.threshold => open = Some(i),
            Some(s) if p < cfg.neg_threshold => {
                runs.push(Run { start: s, end: i });
                open = None;
            }
            _ => {}
        }
    }
    if let Some(s) = open {
        runs.push(Run { start: s, end: probs.len() });
    }
    runs
}

fn frames(ms: f64) -> usize {
    (ms / 1000.0 / FRAME_SECONDS).round().max(0.0) as usize
}

/// Segments (rule 2): runs merged across short gaps, short segments dropped. Each segment keeps the
/// runs it was made of (their ends are the cut candidates).
fn segments(runs: &[Run], cfg: &TrimConfig) -> Vec<(Run, Vec<Run>)> {
    let gap = frames(cfg.merge_gap_ms);
    let mut out: Vec<(Run, Vec<Run>)> = Vec::new();
    for &r in runs {
        match out.last_mut() {
            Some((seg, parts)) if r.start <= seg.end + gap => {
                seg.end = r.end;
                parts.push(r);
            }
            _ => out.push((r, vec![r])),
        }
    }
    let min = frames(cfg.min_run_ms).max(1);
    out.retain(|(seg, _)| seg.end - seg.start >= min);
    out
}

/// Clip `spans` at time `t` (everything after `t` is dropped).
fn truncate(spans: &[Span], t: f64) -> Vec<Span> {
    spans
        .iter()
        .filter(|s| s.start < t)
        .map(|s| Span { start: s.start, end: s.end.min(t) })
        .filter(|s| s.len() > 0.0)
        .collect()
}

fn total(spans: &[Span]) -> f64 {
    spans.iter().map(Span::len).sum()
}

/// The take time at which the kept duration reaches `d` (within `spans`), or the end.
fn time_at_duration(spans: &[Span], d: f64) -> f64 {
    let mut acc = 0.0;
    for s in spans {
        if acc + s.len() >= d {
            return s.start + (d - acc);
        }
        acc += s.len();
    }
    spans.last().map_or(0.0, |s| s.end)
}

/// Plan the trim of a take of `take_seconds` whose VAD frames are `probs`. `None` when no speech
/// segment survives rules 1–2.
pub fn plan(probs: &[f32], take_seconds: f64, cfg: &TrimConfig) -> Option<TrimPlan> {
    let runs = speech_runs(probs, cfg);
    let segs = segments(&runs, cfg);
    if segs.is_empty() {
        return None;
    }
    let end_of_take = take_seconds.max(0.0);
    let t = |frame: usize| (frame as f64 * FRAME_SECONDS).min(end_of_take);
    let lead = cfg.lead_pad_ms / 1000.0;
    let tail = cfg.tail_pad_ms / 1000.0;
    let max_pause = cfg.max_pause_ms / 1000.0;

    // Rule 3: padded segments; pauses up to `max_pause` are kept whole.
    let mut spans: Vec<Span> = Vec::new();
    let mut prev_end: Option<usize> = None;
    for (seg, _) in &segs {
        let start = (t(seg.start) - lead).max(0.0);
        let end = (t(seg.end) + tail).min(end_of_take);
        match (spans.last_mut(), prev_end) {
            (Some(last), Some(pe)) if t(seg.start) - t(pe) <= max_pause || start <= last.end => last.end = end,
            _ => spans.push(Span { start, end }),
        }
        prev_end = Some(seg.end);
    }

    // Rule 4: too long.
    let max_clip = cfg.max_clip_ms / 1000.0;
    let min_clip = cfg.min_clip_ms / 1000.0;
    let mut cut = false;
    let mut hard_cut = false;
    if total(&spans) > max_clip + 1e-9 {
        cut = true;
        // Candidates: every run end, with a tail that never reaches into the next run.
        let all_runs: Vec<Run> = segs.iter().flat_map(|(_, parts)| parts.iter().copied()).collect();
        let mut best: Option<Vec<Span>> = None;
        for (i, r) in all_runs.iter().enumerate() {
            let next_start = all_runs.get(i + 1).map_or(end_of_take, |n| t(n.start));
            let room = (next_start - t(r.end)).max(0.0);
            let cut_at = t(r.end) + tail.min(room / 2.0);
            let kept = truncate(&spans, cut_at);
            let d = total(&kept);
            if d <= max_clip + 1e-9 && d >= min_clip - 1e-9 {
                best = Some(kept);
            }
        }
        spans = match best {
            Some(kept) => kept,
            None => {
                hard_cut = true;
                // Continuous speech: cut at the least voiced frame of the last 1.5 s before the
                // limit (never before `min_clip`).
                let limit = time_at_duration(&spans, max_clip);
                let earliest = time_at_duration(&spans, (max_clip - 1.5).max(min_clip));
                let first = (earliest / FRAME_SECONDS).ceil() as usize;
                let last = ((limit / FRAME_SECONDS).floor() as usize).min(probs.len());
                let mut at = limit;
                let mut lowest = f32::INFINITY;
                for (f, &p) in probs.iter().enumerate().take(last).skip(first) {
                    if p < lowest {
                        lowest = p;
                        at = (f as f64 + 0.5) * FRAME_SECONDS;
                    }
                }
                truncate(&spans, at.min(limit))
            }
        };
    }

    let voiced_frames = probs
        .iter()
        .enumerate()
        .filter(|(i, &p)| {
            let mid = (*i as f64 + 0.5) * FRAME_SECONDS;
            p.is_finite() && p >= cfg.threshold && spans.iter().any(|s| mid >= s.start && mid < s.end)
        })
        .count();
    Some(TrimPlan { spans, cut, hard_cut, voiced_frames })
}

#[cfg(test)]
mod tests {
    use super::*;

    const F: f64 = FRAME_SECONDS;

    /// A probability track from `(frames, prob)` pieces (31 frames ≈ 1 s).
    fn track(pieces: &[(usize, f32)]) -> (Vec<f32>, f64) {
        let mut probs = Vec::new();
        for &(n, p) in pieces {
            probs.extend(std::iter::repeat_n(p, n));
        }
        let secs = probs.len() as f64 * F;
        (probs, secs)
    }

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn silence_and_clicks_are_no_speech() {
        let cfg = TrimConfig::default();
        let (probs, secs) = track(&[(160, 0.02)]);
        assert_eq!(plan(&probs, secs, &cfg), None);
        // Two isolated one-frame clicks are not speech.
        let (probs, secs) = track(&[(31, 0.0), (1, 0.9), (31, 0.0), (2, 0.9), (31, 0.1)]);
        assert_eq!(plan(&probs, secs, &cfg), None);
        assert_eq!(plan(&[], 0.0, &cfg), None);
    }

    #[test]
    fn ends_are_trimmed_to_the_speech_plus_pads() {
        let cfg = TrimConfig::default();
        let (probs, secs) = track(&[(62, 0.05), (156, 0.95), (94, 0.05)]);
        let p = plan(&probs, secs, &cfg).unwrap();
        assert_eq!(p.spans.len(), 1);
        assert!(approx(p.spans[0].start, 62.0 * F - 0.2), "{:?}", p.spans);
        assert!(approx(p.spans[0].end, 218.0 * F + 0.3), "{:?}", p.spans);
        assert!(!p.cut && !p.hard_cut);
        assert_eq!(p.voiced_frames, 156);
        // Pads stop at the take's edges.
        let (probs, secs) = track(&[(125, 0.9), (3, 0.0)]);
        let p = plan(&probs, secs, &cfg).unwrap();
        assert!(approx(p.spans[0].start, 0.0) && approx(p.spans[0].end, secs), "{:?}", p.spans);
    }

    #[test]
    fn hysteresis_keeps_a_word_through_a_shallow_dip() {
        let cfg = TrimConfig::default();
        // 0.4 is below the threshold but above neg_threshold: the run continues.
        let (probs, secs) = track(&[(31, 0.0), (62, 0.9), (16, 0.4), (62, 0.9), (31, 0.0)]);
        let p = plan(&probs, secs, &cfg).unwrap();
        assert_eq!(p.spans.len(), 1);
        assert!(approx(p.spans[0].len(), 140.0 * F + 0.5), "{:?}", p.spans);
        assert_eq!(p.voiced_frames, 124, "the dip is kept but not counted as voiced");
    }

    #[test]
    fn short_pauses_stay_and_long_pauses_are_spliced_down() {
        let cfg = TrimConfig::default();
        // A 0.64 s pause is kept as is: one span.
        let (probs, secs) = track(&[(31, 0.0), (62, 0.9), (20, 0.0), (62, 0.9), (31, 0.0)]);
        let p = plan(&probs, secs, &cfg).unwrap();
        assert_eq!(p.spans.len(), 1, "{:?}", p.spans);
        assert!(approx(p.duration(), 144.0 * F + 0.5));
        // A 3 s pause becomes tail + lead (0.5 s): two spans.
        let (probs, secs) = track(&[(31, 0.0), (62, 0.9), (94, 0.0), (62, 0.9), (31, 0.0)]);
        let p = plan(&probs, secs, &cfg).unwrap();
        assert_eq!(p.spans.len(), 2, "{:?}", p.spans);
        assert!(approx(p.duration(), 124.0 * F + 1.0), "{}", p.duration());
        assert!(!p.cut);
    }

    #[test]
    fn a_long_take_is_cut_at_a_pause_within_the_limit() {
        let cfg = TrimConfig::default();
        // Words of 62 frames (1.98 s) with 16-frame (0.51 s) pauses: 7 words, ~17 s of speech.
        let mut pieces = vec![(16usize, 0.0f32)];
        for _ in 0..7 {
            pieces.push((62, 0.9));
            pieces.push((16, 0.0));
        }
        let (probs, secs) = track(&pieces);
        let p = plan(&probs, secs, &cfg).unwrap();
        assert!(p.cut && !p.hard_cut);
        // Word k ends at frame 78 k; word 4 ends at 9.98 s, and the clip keeps half the pause
        // after it (0.256 s < the 0.3 s tail pad).
        let end = p.spans.last().unwrap().end;
        assert!(approx(end, 312.0 * F + 8.0 * F), "{end}");
        let d = p.duration();
        assert!((9.5..=10.0).contains(&d), "{d}");
        assert_eq!(p.voiced_frames, 4 * 62);
    }

    #[test]
    fn continuous_speech_is_hard_cut_at_its_softest_point() {
        let cfg = TrimConfig::default();
        // 14 s of speech without a pause; a softer stretch around 9 s.
        let (mut probs, secs) = track(&[(16, 0.0), (440, 0.95), (16, 0.0)]);
        let soft = 280;
        for p in &mut probs[soft..soft + 6] {
            *p = 0.6;
        }
        probs[soft + 3] = 0.55;
        let p = plan(&probs, secs, &cfg).unwrap();
        assert!(p.cut && p.hard_cut);
        let end = p.spans.last().unwrap().end;
        assert!(approx(end, (soft + 3) as f64 * F + F / 2.0), "{end}");
        assert!((8.5..=10.0).contains(&p.duration()), "{}", p.duration());
    }

    #[test]
    fn a_short_take_is_planned_and_left_to_the_length_check() {
        let cfg = TrimConfig::default();
        let (probs, secs) = track(&[(16, 0.0), (31, 0.9), (16, 0.0)]);
        let p = plan(&probs, secs, &cfg).unwrap();
        assert!(approx(p.duration(), 31.0 * F + 0.5), "{}", p.duration());
        assert!(!p.cut);
    }

    #[test]
    fn non_finite_probabilities_count_as_silence() {
        let cfg = TrimConfig::default();
        let (mut probs, secs) = track(&[(31, 0.0), (125, 0.9), (31, 0.0)]);
        probs[0] = f32::NAN;
        probs[3] = f32::INFINITY;
        let p = plan(&probs, secs, &cfg).unwrap();
        assert!(approx(p.spans[0].start, 31.0 * F - 0.2), "{:?}", p.spans);
        assert_eq!(p.voiced_frames, 125);
    }
}

