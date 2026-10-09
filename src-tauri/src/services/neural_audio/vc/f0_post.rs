//! Median smoother + confidence gate for F0 contours (hop-160 grid).
//!
//! Two post-processing chains live here:
//! - [`smooth_f0`]: the legacy chain of FCPE / RMVPE / Dio (neighbour octave gate, then a
//!   scipy-style median). Unchanged by P3, so the proven path sounds exactly as before.
//! - [`F0Post`] with [`F0PostConfig`] (P3-D, the pitch bank: SwiftF0, CREPE-tiny, PESTO and F0
//!   fusion): an **octave island gate** that folds short runs sitting an octave (or two) away from
//!   the frames on both sides back by whole octaves, instead of averaging them away; a median whose
//!   edges replicate the end frames (the newest frame of a streaming slice is the one the voice is
//!   decoded with, so it is never pulled toward a zero pad); and the F0 range of the voice
//!   ([`F0Limits`]: speech 50-1100 Hz, singing up to 2.5 kHz). Runs that touch the newest edge are
//!   never folded: without the audio after them a real leap and an octave error look the same, so
//!   the gate leaves them to the extractor (F0 fusion arbitrates those with a second extractor).

use super::engine::options::F0Range;

/// F0 search range of a voice, in Hz. Frames outside it are unvoiced.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct F0Limits {
    pub min_hz: f32,
    pub max_hz: f32,
}

impl F0Limits {
    /// RVC's speech range (`f0_min = 50`, `f0_max = 1100`).
    pub const SPEECH: F0Limits = F0Limits {
        min_hz: 50.0,
        max_hz: 1100.0,
    };
    /// Singing: up to 2.5 kHz (the extractors themselves top out near 2 kHz: SwiftF0 2093.75 Hz,
    /// CREPE / RMVPE about 2006 Hz).
    pub const SINGING: F0Limits = F0Limits {
        min_hz: 50.0,
        max_hz: 2500.0,
    };

    pub fn for_range(range: F0Range) -> F0Limits {
        match range {
            F0Range::Speech => F0Limits::SPEECH,
            F0Range::Singing => F0Limits::SINGING,
        }
    }

    pub fn contains(&self, hz: f32) -> bool {
        hz >= self.min_hz && hz <= self.max_hz
    }
}

/// Configuration of [`F0Post`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct F0PostConfig {
    pub limits: F0Limits,
    /// Longest run of 10 ms frames the octave gate folds (0 = gate off). A run longer than this is
    /// taken as real (a register change, a leap in a song).
    pub max_island_frames: usize,
    /// How close (cents) a run must sit to a whole number of octaves from its neighbours to count as
    /// an octave error.
    pub octave_tolerance_cents: f32,
    /// Adjacent voiced frames further apart than this (cents) start a new run.
    pub jump_cents: f32,
    /// Unvoiced frames that may separate two runs of one voiced segment (the gate looks across
    /// them).
    pub max_gap_frames: usize,
    /// Odd median window over the contour, unvoiced frames included (1 = off); the edges replicate
    /// the end frames.
    pub median_frames: usize,
}

impl F0PostConfig {
    /// The pitch bank's defaults for `range`. Singing folds only shorter islands (a sung leap that
    /// comes back quickly is still music).
    pub fn for_range(range: F0Range) -> F0PostConfig {
        match range {
            F0Range::Speech => F0PostConfig {
                limits: F0Limits::SPEECH,
                max_island_frames: 5,
                octave_tolerance_cents: 250.0,
                jump_cents: 600.0,
                max_gap_frames: 3,
                median_frames: 3,
            },
            F0Range::Singing => F0PostConfig {
                limits: F0Limits::SINGING,
                max_island_frames: 3,
                octave_tolerance_cents: 150.0,
                jump_cents: 600.0,
                max_gap_frames: 2,
                median_frames: 3,
            },
        }
    }

    /// Range limits only (no gate, no median): for contours that were already post-processed.
    #[allow(dead_code)] // Used by tests (stand-in extractors) and callers with smoothed contours.
    pub fn limits_only(limits: F0Limits) -> F0PostConfig {
        F0PostConfig {
            limits,
            max_island_frames: 0,
            octave_tolerance_cents: 0.0,
            jump_cents: 600.0,
            max_gap_frames: 0,
            median_frames: 1,
        }
    }
}

/// A run of voiced frames: `[start, end)` plus its median pitch (log2 Hz).
#[derive(Clone, Copy, Debug)]
struct Run {
    start: usize,
    end: usize,
    log2: f32,
}

/// The pitch bank's post-processing (see the module docs). Allocation-free once its scratch has
/// grown to the longest contour it has processed.
#[derive(Default)]
pub struct F0Post {
    runs: Vec<Run>,
    tmp: Vec<f32>,
    win: Vec<f32>,
}

impl F0Post {
    pub fn new() -> F0Post {
        F0Post::default()
    }

    /// Octave island gate, median and range limits, in place.
    pub fn process(&mut self, f0: &mut [f32], cfg: &F0PostConfig) {
        for v in f0.iter_mut() {
            if !(v.is_finite() && *v > 0.0) {
                *v = 0.0;
            }
        }
        if cfg.max_island_frames > 0 {
            // Two passes: folding one island can make its neighbour an island of its own.
            for _ in 0..2 {
                if !self.octave_gate(f0, cfg) {
                    break;
                }
            }
        }
        if cfg.median_frames > 1 {
            self.median(f0, cfg.median_frames);
        }
        for v in f0.iter_mut() {
            if *v > 0.0 && !cfg.limits.contains(*v) {
                *v = 0.0;
            }
        }
    }

    /// Split `f0` into runs of voiced frames (see [`F0PostConfig::jump_cents`]).
    fn collect_runs(&mut self, f0: &[f32], cfg: &F0PostConfig) {
        self.runs.clear();
        self.tmp.clear();
        let jump = cfg.jump_cents / 1200.0;
        let mut i = 0;
        while i < f0.len() {
            if f0[i] <= 0.0 {
                i += 1;
                continue;
            }
            let start = i;
            let mut prev = f0[i].log2();
            i += 1;
            while i < f0.len() && f0[i] > 0.0 && (f0[i].log2() - prev).abs() <= jump {
                prev = f0[i].log2();
                i += 1;
            }
            self.tmp.clear();
            self.tmp.extend(f0[start..i].iter().map(|v| v.log2()));
            let mid = self.tmp.len() / 2;
            let (_, &mut log2, _) = self.tmp.select_nth_unstable_by(mid, f32::total_cmp);
            self.runs.push(Run { start, end: i, log2 });
        }
    }

    /// One pass of the octave island gate; returns whether anything was folded.
    fn octave_gate(&mut self, f0: &mut [f32], cfg: &F0PostConfig) -> bool {
        self.collect_runs(f0, cfg);
        let n = f0.len();
        let tol = cfg.octave_tolerance_cents / 1200.0;
        let jump = cfg.jump_cents / 1200.0;
        let near_octaves = |d: f32| -> Option<i32> {
            let k = d.round();
            (k != 0.0 && k.abs() <= 2.0 && (d - k).abs() <= tol).then_some(k as i32)
        };
        let mut folded = false;
        for k in 0..self.runs.len() {
            let r = self.runs[k];
            let len = r.end - r.start;
            if len > cfg.max_island_frames {
                continue;
            }
            // Never fold a run that reaches the newest frame: nothing after it confirms an error.
            if r.end >= n {
                continue;
            }
            let joined = |a: &Run, b: &Run| b.start.saturating_sub(a.end) <= cfg.max_gap_frames;
            let prev = (k > 0).then(|| self.runs[k - 1]).filter(|p| joined(p, &r));
            let next = self.runs.get(k + 1).copied().filter(|q| joined(&r, q));
            let shift = match (prev, next) {
                (Some(p), Some(q)) => {
                    // Both sides agree with each other and put this run a whole octave away.
                    if (p.log2 - q.log2).abs() > jump {
                        continue;
                    }
                    let reference = 0.5 * (p.log2 + q.log2);
                    match near_octaves(r.log2 - reference) {
                        Some(o) if (p.end - p.start) + (q.end - q.start) > len => o,
                        _ => continue,
                    }
                }
                // The first or last run of a segment (an onset or an ending): only a clearly longer
                // run on its other side decides.
                (None, Some(q)) | (Some(q), None) => {
                    if (q.end - q.start) < 3 * len {
                        continue;
                    }
                    match near_octaves(r.log2 - q.log2) {
                        Some(o) => o,
                        None => continue,
                    }
                }
                (None, None) => continue,
            };
            let scale = (-(shift as f32)).exp2();
            for v in &mut f0[r.start..r.end] {
                if *v > 0.0 {
                    *v *= scale;
                }
            }
            self.runs[k].log2 -= shift as f32;
            folded = true;
        }
        folded
    }

    /// Median over an odd window, unvoiced (0) frames included, the edges replicating the end
    /// frames.
    fn median(&mut self, f0: &mut [f32], window: usize) {
        let w = window.clamp(1, 9) | 1;
        let half = w / 2;
        let n = f0.len();
        if n == 0 || half == 0 {
            return;
        }
        self.tmp.clear();
        self.tmp.extend_from_slice(f0);
        for (i, out) in f0.iter_mut().enumerate() {
            self.win.clear();
            for k in 0..w {
                let j = (i + k).saturating_sub(half).min(n - 1);
                self.win.push(self.tmp[j]);
            }
            self.win.sort_unstable_by(f32::total_cmp);
            *out = self.win[half];
        }
    }
}

/// Cents between two positive frequencies (`a` over `b`).
pub fn cents(a: f32, b: f32) -> f32 {
    1200.0 * (a / b).log2()
}

/// Replace octave-jump frames with their neighbours' mean, then median-smooth.
///
/// The median runs over the full window *including unvoiced (0.0) frames* and zero-pads both
/// ends, exactly like `scipy.signal.medfilt` in reference RVC: voiced runs keep their length, an
/// isolated spurious voiced frame is removed instead of spread to its neighbours, and the edge
/// frames (the newest one included) are smoothed like any other instead of passed through.
/// Non-finite or negative input frames are treated as unvoiced (0.0).
pub fn smooth_f0(f0: &mut [f32], window: usize) {
    if f0.is_empty() {
        return;
    }
    for v in f0.iter_mut() {
        if !(v.is_finite() && *v > 0.0) {
            *v = 0.0;
        }
    }
    let w = window.clamp(1, 5);
    // Confidence: a voiced frame more than ~an octave away from *every* voiced neighbour is an
    // octave error; replace it with the neighbours' mean. (Requiring disagreement with all
    // neighbours keeps a correct frame next to a spike from being pulled toward the spike.)
    let raw = f0.to_vec();
    let jumps = |a: f32, b: f32| a / b > 1.9 || b / a > 1.9;
    for i in 0..f0.len() {
        let cur = raw[i];
        if cur <= 0.0 {
            continue;
        }
        let prev = if i > 0 { raw[i - 1] } else { 0.0 };
        let next = raw.get(i + 1).copied().unwrap_or(0.0);
        let mut sum = 0.0f32;
        let mut count = 0usize;
        let mut all_jump = true;
        for nb in [prev, next] {
            if nb > 0.0 {
                sum += nb;
                count += 1;
                all_jump &= jumps(cur, nb);
            }
        }
        if count > 0 && all_jump {
            f0[i] = sum / count as f32;
        }
    }
    // Median over an odd window (zeros included), zero-padded past both ends (scipy medfilt).
    let half = w / 2;
    if half == 0 {
        return;
    }
    let src = f0.to_vec();
    let n = src.len();
    let mut win: Vec<f32> = Vec::with_capacity(2 * half + 1);
    for (i, out) in f0.iter_mut().enumerate() {
        win.clear();
        // Padded index `i + k` maps to source index `i + k - half`.
        for k in 0..=2 * half {
            let j = i + k;
            win.push(if j < half || j - half >= n { 0.0 } else { src[j - half] });
        }
        win.sort_by(f32::total_cmp);
        *out = win[half];
    }
}

/// Kill switch of the streaming smoother's edges (P3B-F): `SIGIL_VC_F0_EDGE=zero` makes
/// [`smooth_f0_stream`] the P3 smoother ([`smooth_f0`]: the median zero-pads past both ends);
/// anything else (or unset) replicates the end frames. Read once per process.
pub const F0_EDGE_ENV: &str = "SIGIL_VC_F0_EDGE";

/// Whether [`smooth_f0_stream`] zero-pads its median like P3 ([`F0_EDGE_ENV`]`=zero`).
pub fn stream_edges_zero() -> bool {
    static ZERO: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ZERO.get_or_init(|| {
        std::env::var(F0_EDGE_ENV)
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "zero" | "legacy" | "p3"))
            .unwrap_or(false)
    })
}

/// The streaming extractors' smoother (RMVPE and FCPE on the tick path; P3B-S0 call sites,
/// P3B-F implementation): [`smooth_f0`]'s octave-jump repair and median over the contour,
/// unvoiced frames included, but the median's window **replicates the end frames** instead of
/// zero-padding past them, and it allocates nothing.
///
/// The newest frame of a streaming call is the one the voice is decoded with; a zero pad there
/// pulls an onset at the newest frame to unvoiced (median of `[0, f, 0]`) and a voiced newest
/// frame down to `min(previous, newest)`. Interior frames (`window / 2` or more from both ends)
/// come out exactly as from [`smooth_f0`]; only the edge frames differ.
///
/// Measured (P3B-F, `engine::f0_cache::tests::lead_sweep_against_whole_clip_f0`, table
/// geometry, 7 speech clips, 6 geometries x 3 readout lags): RMVPE's contour of an `n`-frame call
/// has `n + 1` frames (FCPE's mel export too) and the F0 cache keeps the first `n`, so the zero pad
/// lands on the dropped extra frame: on the cached frames the two edge rules are bit-identical. The
/// edge rule matters only to callers that keep the whole contour (a whole-utterance call's first
/// and last frame). What this change buys on the tick path is the allocation-free smoother (the
/// last allocation of RMVPE's `f0_into`). `SIGIL_VC_F0_EDGE=zero` ([`F0_EDGE_ENV`]) is the switch
/// back to the P3 smoother.
///
/// Whole-utterance callers keep [`smooth_f0`] (scipy `medfilt`, reference RVC).
///
/// # Frozen by P3B-S0 (signature; P3B-F implements)
pub fn smooth_f0_stream(f0: &mut [f32], window: usize) {
    if stream_edges_zero() {
        smooth_f0(f0, window);
    } else {
        smooth_f0_replicated(f0, window);
    }
}

/// [`smooth_f0_stream`]'s edge-replicating smoother, allocation-free (fixed scratch on the stack).
fn smooth_f0_replicated(f0: &mut [f32], window: usize) {
    let n = f0.len();
    if n == 0 {
        return;
    }
    for v in f0.iter_mut() {
        if !(v.is_finite() && *v > 0.0) {
            *v = 0.0;
        }
    }
    // Octave-jump repair against the raw neighbours (as `smooth_f0`): `prev_raw` keeps the value
    // frame `i - 1` had before it was repaired; frame `i + 1` is not repaired yet when read.
    let jumps = |a: f32, b: f32| a / b > 1.9 || b / a > 1.9;
    let mut prev_raw = 0.0f32;
    for i in 0..n {
        let cur = f0[i];
        let next = if i + 1 < n { f0[i + 1] } else { 0.0 };
        if cur > 0.0 {
            let (mut sum, mut count, mut all_jump) = (0.0f32, 0usize, true);
            for nb in [prev_raw, next] {
                if nb > 0.0 {
                    sum += nb;
                    count += 1;
                    all_jump &= jumps(cur, nb);
                }
            }
            if count > 0 && all_jump {
                f0[i] = sum / count as f32;
            }
        }
        prev_raw = cur;
    }
    let w = window.clamp(1, 5);
    let half = w / 2;
    if half == 0 {
        return;
    }
    // Median with the window clamped to the contour (edges replicate). `before` holds the values
    // frames `i - half .. i` had before this pass overwrote them (at most two).
    let mut before = [0.0f32; 2];
    let mut win = [0.0f32; 5];
    for i in 0..n {
        for (k, slot) in win.iter_mut().enumerate().take(2 * half + 1) {
            let j = (i + k).saturating_sub(half).min(n - 1);
            *slot = if j < i { before[half - (i - j)] } else { f0[j] };
        }
        let wv = &mut win[..2 * half + 1];
        wv.sort_unstable_by(f32::total_cmp);
        let out = wv[half];
        // Shift the history of original values, newest last.
        if half == 2 {
            before[0] = before[1];
        }
        before[half - 1] = f0[i];
        f0[i] = out;
    }
}

#[cfg(test)]
mod tests {
    use super::smooth_f0;

    #[test]
    fn median_removes_isolated_voiced_frame() {
        let mut f0 = vec![0.0, 0.0, 200.0, 0.0, 0.0];
        smooth_f0(&mut f0, 3);
        assert_eq!(f0, vec![0.0; 5]);
    }

    #[test]
    fn median_keeps_voiced_run_length() {
        let mut f0 = vec![0.0, 0.0, 120.0, 121.0, 122.0, 0.0, 0.0];
        smooth_f0(&mut f0, 3);
        let voiced: Vec<bool> = f0.iter().map(|&v| v > 0.0).collect();
        assert_eq!(voiced, vec![false, false, true, true, true, false, false], "{f0:?}");
    }

    #[test]
    fn median_fills_single_dropout_inside_voiced_run() {
        let mut f0 = vec![110.0, 111.0, 0.0, 112.0, 113.0];
        smooth_f0(&mut f0, 3);
        assert_eq!(f0[2], 111.0);
        // Edges are zero-padded like scipy medfilt: medfilt([110, 111, 0, 112, 113], 3) =
        // [110, 110, 111, 112, 112].
        assert_eq!(f0, vec![110.0, 110.0, 111.0, 112.0, 112.0]);
    }

    #[test]
    fn edge_frames_are_smoothed_like_scipy_medfilt() {
        // The newest frame is filtered too: an isolated voiced frame at the end is removed ...
        let mut f0 = vec![0.0, 0.0, 0.0, 0.0, 180.0];
        smooth_f0(&mut f0, 3);
        assert_eq!(f0, vec![0.0; 5]);
        // ... and a voiced end frame becomes the lower of itself and its inner neighbour
        // (median of the two and the zero pad).
        let mut f0 = vec![0.0, 150.0, 152.0, 151.0, 153.0];
        smooth_f0(&mut f0, 3);
        assert_eq!(f0, vec![0.0, 150.0, 151.0, 152.0, 151.0]);
        // Window 5: two zeros of padding at each end; a steady run still survives at the edges
        // (at most two of the five window values are pad zeros).
        let mut f0 = vec![100.0; 6];
        smooth_f0(&mut f0, 5);
        assert_eq!(f0, vec![100.0; 6]);
        let mut f0 = vec![100.0, 101.0, 102.0, 103.0];
        smooth_f0(&mut f0, 5);
        // [0,0,100,101,102] -> 100; [0,100,101,102,103] -> 101; [100,101,102,103,0] -> 101;
        // [101,102,103,0,0] -> 101.
        assert_eq!(f0, vec![100.0, 101.0, 101.0, 101.0]);
    }

    #[test]
    fn non_finite_frames_are_unvoiced() {
        let mut f0 = vec![120.0, f32::NAN, 121.0, f32::INFINITY, 122.0, -5.0, 123.0];
        smooth_f0(&mut f0, 3);
        assert!(f0.iter().all(|v| v.is_finite() && *v >= 0.0), "{f0:?}");
        let mut f0 = vec![f32::NAN; 4];
        smooth_f0(&mut f0, 3);
        assert_eq!(f0, vec![0.0; 4]);
    }

    #[test]
    fn octave_jump_is_replaced_before_median() {
        let mut f0 = vec![100.0, 100.0, 400.0, 100.0, 100.0];
        smooth_f0(&mut f0, 3);
        assert!(f0.iter().all(|&v| (v - 100.0).abs() < 1e-3), "{f0:?}");
    }
}

#[cfg(test)]
mod stream_tests {
    use super::{smooth_f0, smooth_f0_replicated};

    /// Deterministic contours with voiced runs, dropouts, isolated frames and octave jumps.
    fn contours() -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for len in 1..40usize {
            let f0: Vec<f32> = (0..len)
                .map(|_| match next() % 9 {
                    0 | 1 => 0.0,
                    2 => 400.0,
                    3 => f32::NAN,
                    k => 140.0 + k as f32 * 3.0,
                })
                .collect();
            out.push(f0);
        }
        out
    }

    /// Interior frames (`window / 2` or more from both ends) come out exactly as from `smooth_f0`;
    /// only the edges differ.
    #[test]
    fn interior_frames_equal_smooth_f0() {
        for window in [1usize, 3, 5] {
            let half = window / 2;
            for f0 in contours() {
                let (mut a, mut b) = (f0.clone(), f0.clone());
                smooth_f0(&mut a, window);
                smooth_f0_replicated(&mut b, window);
                let n = f0.len();
                if n > 2 * half {
                    assert_eq!(&a[half..n - half], &b[half..n - half], "window {window}: {f0:?}");
                }
                assert!(b.iter().all(|v| v.is_finite() && *v >= 0.0));
            }
        }
    }

    /// The edges replicate the end frames: an onset at the newest frame stays voiced (the zero pad
    /// unvoiced it), a voiced newest frame keeps its value, an isolated voiced frame inside the
    /// contour is still removed, and a single interior dropout is still filled.
    #[test]
    fn edges_replicate_the_end_frames() {
        let mut f0 = vec![0.0, 0.0, 0.0, 150.0, 151.0];
        let mut zero = f0.clone();
        smooth_f0(&mut zero, 3);
        smooth_f0_replicated(&mut f0, 3);
        assert_eq!(zero[4], 150.0, "P3: the newest frame pulled to min(prev, newest)");
        assert_eq!(f0, vec![0.0, 0.0, 0.0, 150.0, 151.0]);
        let mut f0 = vec![0.0, 0.0, 0.0, 0.0, 180.0];
        smooth_f0_replicated(&mut f0, 3);
        assert_eq!(f0[4], 180.0, "an onset at the newest frame is kept");
        let mut f0 = vec![0.0, 0.0, 200.0, 0.0, 0.0];
        smooth_f0_replicated(&mut f0, 3);
        assert_eq!(f0, vec![0.0; 5]);
        let mut f0 = vec![110.0, 111.0, 0.0, 112.0, 113.0];
        smooth_f0_replicated(&mut f0, 3);
        // P3 (zero pad): [110, 110, 111, 112, 112]; the newest frame now keeps its value.
        assert_eq!(f0, vec![110.0, 110.0, 111.0, 112.0, 113.0]);
        // Window 5 at both ends.
        let mut f0 = vec![100.0, 101.0, 102.0, 103.0];
        smooth_f0_replicated(&mut f0, 5);
        // [100,100,100,101,102] -> 100; [100,100,101,102,103] -> 101; [100,101,102,103,103] -> 102;
        // [101,102,103,103,103] -> 103.
        assert_eq!(f0, vec![100.0, 101.0, 102.0, 103.0]);
        // The octave repair still runs first.
        let mut f0 = vec![100.0, 100.0, 400.0, 100.0, 100.0];
        smooth_f0_replicated(&mut f0, 3);
        assert!(f0.iter().all(|&v| (v - 100.0).abs() < 1e-3), "{f0:?}");
    }

    #[test]
    fn the_stream_smoother_is_allocation_free() {
        use crate::services::neural_audio::gpu::alloc_probe::allocations_during;
        let mut f0: Vec<f32> = (0..64)
            .map(|i| if i % 11 < 2 { 0.0 } else { 150.0 + i as f32 })
            .collect();
        // The kill switch's first read (a process-wide `OnceLock`) may allocate.
        super::smooth_f0_stream(&mut f0, 3);
        let ((), n) = allocations_during(|| {
            for w in [3usize, 5] {
                smooth_f0_replicated(&mut f0, w);
            }
            super::smooth_f0_stream(&mut f0, 3);
        });
        // With `SIGIL_VC_F0_EDGE=zero` set, `smooth_f0_stream` is P3's (allocating) smoother.
        if !super::stream_edges_zero() {
            assert_eq!(n, 0);
        }
    }
}

#[cfg(test)]
mod post_tests {
    use super::*;

    fn speech() -> F0PostConfig {
        F0PostConfig::for_range(F0Range::Speech)
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn folds_an_interior_octave_island_back() {
        // A 3-frame octave-up error inside a steady 150 Hz run is folded by one octave (not
        // averaged away, not unvoiced).
        let mut f0 = vec![150.0, 151.0, 152.0, 304.0, 306.0, 308.0, 154.0, 155.0, 156.0, 157.0];
        F0Post::new().process(
            &mut f0,
            &F0PostConfig {
                median_frames: 1,
                ..speech()
            },
        );
        let want = [150.0, 151.0, 152.0, 152.0, 153.0, 154.0, 154.0, 155.0, 156.0, 157.0];
        for (a, b) in f0.iter().zip(want) {
            assert!(close(*a, b), "{f0:?}");
        }
        // Octave down and two octaves up are folded too.
        let mut f0 = vec![200.0; 12];
        f0[4] = 100.0;
        f0[5] = 100.5;
        f0[9] = 800.0;
        F0Post::new().process(
            &mut f0,
            &F0PostConfig {
                median_frames: 1,
                ..speech()
            },
        );
        assert!(f0.iter().all(|&v| (v - 200.0).abs() <= 1.01), "{f0:?}");
    }

    #[test]
    fn keeps_real_leaps_and_the_newest_edge() {
        // A register change that stays (longer than the island limit) is real.
        let mut f0 = vec![150.0; 6];
        f0.extend(vec![300.0; 8]);
        let before = f0.clone();
        F0Post::new().process(
            &mut f0,
            &F0PostConfig {
                median_frames: 1,
                ..speech()
            },
        );
        assert_eq!(f0, before);
        // A jump at the newest frames is not folded: nothing after it confirms an error.
        let mut f0 = vec![150.0; 10];
        f0.extend([300.0, 300.0]);
        let before = f0.clone();
        F0Post::new().process(
            &mut f0,
            &F0PostConfig {
                median_frames: 1,
                ..speech()
            },
        );
        assert_eq!(f0, before);
        // Neither is a jump that is not near a whole octave (a fifth).
        let mut f0 = vec![150.0; 12];
        f0[5] = 225.0;
        f0[6] = 225.0;
        let before = f0.clone();
        F0Post::new().process(
            &mut f0,
            &F0PostConfig {
                median_frames: 1,
                ..speech()
            },
        );
        assert_eq!(f0, before);
    }

    #[test]
    fn looks_across_short_unvoiced_gaps_and_at_segment_ends() {
        // Island separated by one unvoiced frame on each side.
        let mut f0 = vec![120.0, 121.0, 122.0, 0.0, 244.0, 246.0, 0.0, 123.0, 124.0, 125.0, 126.0];
        F0Post::new().process(
            &mut f0,
            &F0PostConfig {
                median_frames: 1,
                ..speech()
            },
        );
        assert!(close(f0[4], 122.0) && close(f0[5], 123.0), "{f0:?}");
        // An onset that starts an octave high for two frames, then a long steady run.
        let mut f0 = vec![0.0, 0.0, 400.0, 402.0, 200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0];
        F0Post::new().process(
            &mut f0,
            &F0PostConfig {
                median_frames: 1,
                ..speech()
            },
        );
        assert!(close(f0[2], 200.0) && close(f0[3], 201.0), "{f0:?}");
        // An island in a different segment (a long gap) is left alone.
        let mut f0 = vec![120.0, 121.0, 122.0, 0.0, 0.0, 0.0, 0.0, 0.0, 244.0, 246.0, 0.0, 0.0];
        let before = f0.clone();
        F0Post::new().process(
            &mut f0,
            &F0PostConfig {
                median_frames: 1,
                ..speech()
            },
        );
        assert_eq!(f0, before);
    }

    #[test]
    fn median_replicates_the_edges_and_limits_apply() {
        // The newest frame keeps its value (no pull toward a zero pad), unlike smooth_f0, which
        // gives [0, 150, 151, 152, 151] here.
        let mut f0 = vec![0.0, 150.0, 152.0, 151.0, 153.0];
        F0Post::new().process(&mut f0, &speech());
        assert_eq!(f0, vec![0.0, 150.0, 151.0, 152.0, 153.0]);
        // Isolated voiced frames and single dropouts behave like the legacy median.
        let mut f0 = vec![0.0, 0.0, 200.0, 0.0, 0.0, 110.0, 111.0, 0.0, 112.0, 113.0, 113.0];
        F0Post::new().process(&mut f0, &speech());
        assert_eq!(&f0[..5], &[0.0; 5]);
        assert_eq!(f0[7], 111.0);
        // Range: 1500 Hz is outside speech, inside singing; 40 Hz is outside both.
        let mut f0 = vec![1500.0, 1500.0, 1500.0, 40.0, 40.0, 40.0];
        F0Post::new().process(&mut f0, &F0PostConfig::limits_only(F0Limits::SPEECH));
        assert_eq!(f0, vec![0.0; 6]);
        let mut f0 = vec![1500.0, 1500.0, 1500.0, 40.0, 40.0, 40.0];
        F0Post::new().process(&mut f0, &F0PostConfig::for_range(F0Range::Singing));
        assert_eq!(f0, vec![1500.0, 1500.0, 1500.0, 0.0, 0.0, 0.0]);
        // Non-finite input never survives.
        let mut f0 = vec![f32::NAN, 120.0, f32::INFINITY, -3.0, 121.0];
        F0Post::new().process(&mut f0, &speech());
        assert!(f0.iter().all(|v| v.is_finite() && *v >= 0.0), "{f0:?}");
        assert!(close(cents(200.0, 100.0), 1200.0));
    }

    #[test]
    fn an_octave_error_across_an_octave_trap_signal_is_repaired() {
        // Octave traps: alternating half / double errors of 1-4 frames over a slow glide.
        let truth: Vec<f32> = (0..60).map(|i| 140.0 * (i as f32 * 0.004).exp2()).collect();
        let mut f0 = truth.clone();
        for &(s, len, factor) in &[(5usize, 1usize, 2.0f32), (14, 3, 0.5), (27, 4, 2.0), (40, 2, 4.0), (50, 1, 0.5)] {
            for v in &mut f0[s..s + len] {
                *v *= factor;
            }
        }
        F0Post::new().process(
            &mut f0,
            &F0PostConfig {
                median_frames: 1,
                ..speech()
            },
        );
        for (i, (a, b)) in f0.iter().zip(&truth).enumerate() {
            assert!(cents(*a, *b).abs() < 1.0, "frame {i}: {a} vs {b}");
        }
    }

    #[test]
    fn process_is_allocation_free_once_warm() {
        use crate::services::neural_audio::gpu::alloc_probe::allocations_during;
        let mut post = F0Post::new();
        let base: Vec<f32> = (0..60)
            .map(|i| if i % 17 < 3 { 0.0 } else { 150.0 + i as f32 })
            .collect();
        let mut f0 = base.clone();
        post.process(&mut f0, &speech());
        let mut f0 = base.clone();
        let ((), n) = allocations_during(|| post.process(&mut f0, &speech()));
        assert_eq!(n, 0);
    }
}
