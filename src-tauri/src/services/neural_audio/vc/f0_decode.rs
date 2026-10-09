//! Cross-tick F0 decoding (P3B-F): the F0 cache's provisional frames decoded again from the
//! extractor's per-frame salience (`PitchExtractor::last_salience`, RMVPE's 360 bins of 20 cents)
//! with a pitch path that continues the committed contour, and voicing hysteresis. Behind the
//! `f0_decode` lever (off by default; never for `F0Range::Singing`, whose leaps and vibrato it must
//! not smooth).
//!
//! # What it does, per tick (`engine::f0_cache` calls it after the incremental extraction)
//! The F0 cache re-extracts its provisional span every tick (the new block plus the margin) from a
//! slice whose salience the extractor keeps. Every older frame is committed: it was fed to the
//! generator and never changes. The decoder replaces the span's contour by a decode that
//! continues the committed frame before it (the **anchor**):
//! - **Voicing hysteresis** with two salience thresholds. After an unvoiced frame (decoded, or the
//!   anchor), a frame the extractor voices starts voicing only when its peak salience is at least
//!   [`DecodeParams::voice_on`] (0.05 by default, against RMVPE's own 0.03: a weak, isolated peak
//!   does not start voicing); once voiced, the voicing lasts down to the extractor's own threshold,
//!   and optionally below it to [`DecodeParams::voice_off`] (0 by default: sustaining below RMVPE's
//!   threshold raised the V/UV error on every corpus of the sweep). The onset delay this adds is
//!   reported by the sweep (0.16-0.25 frames on average with the defaults).
//! - **A fixed-lag Viterbi** over the span's voiced runs: per frame, the extractor's own value
//!   (cost 0) and the other salience peaks of at least [`DecodeParams::candidate_ratio`] of the
//!   frame's best (cost `ln(best / peak)`, value: RVC's local average ±4 bins around the peak); a
//!   transition from one frame to the next costs [`DecodeParams::cost_per_semitone`] per semitone
//!   beyond [`DecodeParams::free_cents`]; the first frame of the span pays the transition from the
//!   anchor. The path is traced back from the newest frame, so every provisional frame is decided
//!   with the evidence of the frames after it (the fixed lag is the cache's margin). Committed frames
//!   never change: only the span the cache re-extracted this tick (8 provisional frames plus the new
//!   block) is written.
//!
//! The decode is exactly the extractor's contour wherever its own value is the cheapest path (no
//! competing peak, or no jump to pay for), so a clean contour passes through bit for bit.
//!
//! "Over the last 32 frames": the trellis starts at the anchor; the committed frames before it are
//! fixed values, so by the Markov property only the anchor constrains the path (decoding a longer
//! window with those frames clamped gives the same result at the cost of more work).
//!
//! # Cost
//! Per frame: one scan of the 360 bins for the peaks, a local average per candidate, and
//! `candidates²` transitions. Allocation-free once its scratch has grown (the first whole-window
//! call at load). Measured: 0.098 ms per tick at the median (p99 0.16-0.45 ms) in the unoptimized
//! test build (`engine::f0_cache::tests::lead_sweep_against_whole_clip_f0`, the cache's update
//! without the model, 40 ms blocks); in the release engine the F0 stage with and without it is
//! NEUTRAL in the interleaved gate (DirectML f0 p50 4.41 -> 4.39 ms, CI -5.5 .. +3.9 %; CUDA 3.28 ->
//! 3.18 ms).
//!
//! # Measured, and why the lever stays off
//! On the CPU sweep (synthetic speech with exact truth) the defaults cut the fine and the gross
//! pitch error (`DecodeParams::speech`). In the engine on 8 real speech clips x 3 offsets
//! (DirectML, scorer v2, fed F0 against whole-clip RMVPE of the input;
//! `scripts/bench/specs/p3b/f/run_fidelity.ps1 -EnvB SIGIL_VC_F0_DECODE=1`) it moved nothing beyond
//! noise in the pitch rows (fed V/UV 3.99 -> 3.82 %, better on 5 of 8 clips; fine error 23.57 ->
//! 23.91 cents; GPE 0.71 -> 0.78 %; onsets 0.44 frames later, toward the reference's: the scorer's
//! signed lag is -0.69 frames without the decoder, -0.25 with it) and lowered the speaker similarity
//! (CAM++ v2 0.896 -> 0.882, CI -0.027 .. -0.001; MCD +0.009 dB, within noise). CUDA repeats it
//! (the same fed-F0 rows; CAM++ v2 0.893 -> 0.877, CI -0.024 .. -0.008; MCD +0.015 dB, CI +0.006 ..
//! +0.024). So it is not a default for any preset (rule R6); it stays an opt-in for voices whose
//! extractor makes octave errors the salience can correct.
//!
//! # Frozen by P3B-S0 (signatures; P3B-F implements)
//! - [`available`]`() -> bool`: whether this build runs the decoder (the `f0_decode` lever is
//!   offered and resolves on only when it is `true`; `voice_changer::capabilities`).

use super::engine::options::F0Range;
use super::f0_post::F0Limits;
use super::pitch_onnx::{SALIENCE_BINS, SALIENCE_CENTS_OFFSET};

/// Whether this build runs cross-tick F0 decoding (the `f0_decode` lever).
pub fn available() -> bool {
    true
}

/// Development / benchmark override of the `f0_decode` lever (P3B-F; one exe A/Bs it):
/// `SIGIL_VC_F0_DECODE=1|on` decodes, `0|off` does not, whatever the lever says. Unset: the lever.
/// Singing voices never decode. Read when the F0 cache is built (load time).
pub const DECODE_ENV: &str = "SIGIL_VC_F0_DECODE";

/// [`DECODE_ENV`] as an override (`None` = not set or not understood).
pub fn decode_env_override() -> Option<bool> {
    match std::env::var(DECODE_ENV)
        .ok()?
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "1" | "on" | "true" | "yes" => Some(true),
        "0" | "off" | "false" | "no" => Some(false),
        _ => None,
    }
}

/// Most candidates per frame (the extractor's value and the strongest other peaks).
pub const MAX_CANDIDATES: usize = 4;

/// Tunables of the decoder (module docs). [`DecodeParams::speech`] is the lever's default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeParams {
    /// Peak salience a frame the extractor voices needs to start voicing after an unvoiced frame
    /// (0 = the extractor's own decision: no added onset delay).
    pub voice_on: f32,
    /// Peak salience at which a frame the extractor left unvoiced keeps the voicing of the frame
    /// before it (0 = no hysteresis: the extractor's voicing everywhere).
    pub voice_off: f32,
    /// Transition cost per semitone of pitch change between consecutive voiced frames beyond
    /// [`free_cents`](Self::free_cents) (0 = no pitch decoding: the extractor's values).
    pub cost_per_semitone: f32,
    /// Pitch change per frame that costs nothing (intonation, vibrato).
    pub free_cents: f32,
    /// A salience peak is a candidate when it is at least this share of the frame's best.
    pub candidate_ratio: f32,
    /// Candidates per frame, the extractor's value included (1..=[`MAX_CANDIDATES`]).
    pub candidates: usize,
    /// The voice's F0 range: peaks outside it are not candidates.
    pub limits: F0Limits,
}

impl DecodeParams {
    /// The lever's parameters for speech: voicing starts at a peak salience of 0.05 and is kept down
    /// to the extractor's own threshold (RMVPE: 0.03), a pitch change costs 0.25 per semitone beyond
    /// 50 cents, and candidates are the peaks of at least a tenth of the frame's best.
    ///
    /// Chosen on the P3B-F sweep's tuning corpus (`engine::f0_cache::tests::lead_sweep_against_whole_clip_f0`,
    /// table cache, synthetic speech seeds 1-27 with exact truth, CPU RMVPE, the default 64-frame
    /// geometry, readout lag 0) and checked on the held-out corpus and the 7 real speech clips: of 11
    /// variants this is the only one that lowers both the fine and the gross pitch error on all three
    /// (tuning: fine 11.08 -> 10.84 cents, GPE 5.05 -> 4.83 %; held-out: 13.33 -> 11.76 cents,
    /// 6.23 -> 5.58 %; real speech against whole-clip RMVPE: 32.97 -> 31.74 cents, 2.35 -> 2.19 %),
    /// at a cost in voicing: V/UV -0.07 / +0.08 / +0.42 pt, onsets 0.16-0.25 frames later on average
    /// (under one frame). Sustaining voicing below the extractor's threshold (`voice_off` > 0) raised
    /// the V/UV error on every corpus (+0.2 to +1.6 pt), so it is off.
    pub fn speech() -> DecodeParams {
        DecodeParams {
            voice_on: 0.05,
            voice_off: 0.0,
            cost_per_semitone: 0.25,
            free_cents: 50.0,
            candidate_ratio: 0.1,
            candidates: MAX_CANDIDATES,
            limits: F0Limits::SPEECH,
        }
    }

    /// The decoder's parameters for a voice's F0 range; `None` for singing (never decoded: a sung
    /// leap or a wide vibrato is music, not an error).
    pub fn for_range(range: F0Range) -> Option<DecodeParams> {
        match range {
            F0Range::Speech => Some(DecodeParams::speech()),
            F0Range::Singing => None,
        }
    }
}

/// The committed frame before a decoded span: its F0 (Hz, 0 = unvoiced); `None` when the span
/// starts the cache (nothing committed before it).
pub type Anchor = Option<f32>;

/// One candidate of a frame: its F0 (Hz) and emission cost.
#[derive(Clone, Copy, Debug, Default)]
struct Candidate {
    hz: f32,
    cost: f32,
}

/// The decoder and its scratch (allocation-free once grown).
#[derive(Debug, Default)]
pub struct F0Decoder {
    /// `frames × MAX_CANDIDATES` candidates, and how many each frame has.
    cand: Vec<Candidate>,
    count: Vec<u8>,
    /// Peak salience per frame.
    best: Vec<f32>,
    /// Viterbi: accumulated cost and back pointer per candidate.
    acc: Vec<f32>,
    back: Vec<u8>,
    /// Decoded voicing per frame.
    voiced: Vec<bool>,
}

/// Salience-weighted mean of the 20-cent bins `center ± 4` (RVC `to_local_average_cents`, the same
/// arithmetic as `pitch_onnx::local_average_f0`, so the argmax gives RMVPE's own value), in Hz.
fn local_average_at(row: &[f32], center: usize) -> f32 {
    let start = center.saturating_sub(4);
    let end = (center + 5).min(row.len());
    let mut weighted = 0.0f32;
    let mut weight = 0.0f32;
    for (bin, &s) in row.iter().enumerate().take(end).skip(start) {
        weighted += s * (20.0 * bin as f32 + SALIENCE_CENTS_OFFSET);
        weight += s;
    }
    if weight.is_nan() || weight <= 0.0 {
        return 0.0;
    }
    let hz = 10.0 * (weighted / weight / 1200.0).exp2();
    if hz.is_finite() {
        hz
    } else {
        0.0
    }
}

/// Cents from `a` to `b` (both positive).
fn cents(a: f32, b: f32) -> f32 {
    1200.0 * (b / a).log2()
}

impl F0Decoder {
    /// Decode `f0` (the extractor's contour of the span, Hz, 0 = unvoiced; replaced in place) from
    /// `salience` (`f0.len()` rows of [`SALIENCE_BINS`], row `i` for frame `i`), continuing
    /// `anchor`. Returns `false` (and leaves `f0` as it was) when the salience does not fit.
    pub fn decode(&mut self, anchor: Anchor, f0: &mut [f32], salience: &[f32], params: &DecodeParams) -> bool {
        let n = f0.len();
        if n == 0 {
            return true;
        }
        if salience.len() < n * SALIENCE_BINS {
            return false;
        }
        let k_max = params.candidates.clamp(1, MAX_CANDIDATES);
        self.cand.clear();
        self.cand.resize(n * MAX_CANDIDATES, Candidate::default());
        self.count.clear();
        self.count.resize(n, 0);
        self.best.clear();
        self.voiced.clear();
        // 1. Voicing (hysteresis) and the candidates of every voiced frame.
        let mut prev_voiced = anchor.is_some_and(|h| h > 0.0);
        for (i, v) in f0.iter().enumerate() {
            let row = &salience[i * SALIENCE_BINS..(i + 1) * SALIENCE_BINS];
            let (arg, best) = row
                .iter()
                .copied()
                .enumerate()
                .fold((0usize, 0.0f32), |b, (j, s)| if s.is_finite() && s > b.1 { (j, s) } else { b });
            self.best.push(best);
            let own = v.is_finite() && *v > 0.0;
            let voiced = if own {
                prev_voiced || best >= params.voice_on
            } else {
                prev_voiced && params.voice_off > 0.0 && best >= params.voice_off
            };
            self.voiced.push(voiced);
            prev_voiced = voiced;
            if !voiced {
                continue;
            }
            let cands = &mut self.cand[i * MAX_CANDIDATES..(i + 1) * MAX_CANDIDATES];
            let mut count = 0usize;
            // The extractor's own value (a frame it voiced), else the argmax decode (a frame the
            // hysteresis keeps voiced).
            let first = if own { *v } else { local_average_at(row, arg) };
            if first > 0.0 {
                cands[0] = Candidate { hz: first, cost: 0.0 };
                count = 1;
            }
            if params.cost_per_semitone > 0.0 && best > 0.0 && count < k_max {
                // The strongest other peaks (local maxima of at least `candidate_ratio` of the best)
                // in the voice's range, in one scan: the top `k_max - 1` by salience.
                let floor = best * params.candidate_ratio;
                // Up to MAX_CANDIDATES peaks (the one the extractor's own value sits on is among
                // them and is skipped below).
                let mut top = [(0usize, 0.0f32); MAX_CANDIDATES];
                let mut found = 0usize;
                for j in 0..SALIENCE_BINS {
                    let s = row[j];
                    if !(s.is_finite() && s >= floor) || (found == MAX_CANDIDATES && s <= top[MAX_CANDIDATES - 1].1) {
                        continue;
                    }
                    let left = if j > 0 { row[j - 1] } else { f32::NEG_INFINITY };
                    let right = if j + 1 < SALIENCE_BINS { row[j + 1] } else { f32::NEG_INFINITY };
                    if !(s > left && s >= right) {
                        continue;
                    }
                    // Insert by salience (descending), dropping the weakest when full.
                    let mut at = found.min(MAX_CANDIDATES - 1);
                    if found < MAX_CANDIDATES {
                        found += 1;
                    }
                    while at > 0 && top[at - 1].1 < s {
                        top[at] = top[at - 1];
                        at -= 1;
                    }
                    top[at] = (j, s);
                }
                for &(j, s) in &top[..found] {
                    if count >= k_max {
                        break;
                    }
                    let peak_hz = 10.0 * ((20.0 * j as f32 + SALIENCE_CENTS_OFFSET) / 1200.0).exp2();
                    if !params.limits.contains(peak_hz)
                        || cands[..count]
                            .iter()
                            .any(|c| cents(c.hz, peak_hz).abs() < 100.0)
                    {
                        continue;
                    }
                    let hz = local_average_at(row, j);
                    if hz <= 0.0 {
                        continue;
                    }
                    cands[count] = Candidate {
                        hz,
                        cost: (best / s).ln().max(0.0),
                    };
                    count += 1;
                }
            }
            self.count[i] = count as u8;
            if count == 0 {
                // Nothing to voice the frame with.
                self.voiced[i] = false;
                prev_voiced = false;
            }
        }
        // 2. The pitch path over each voiced run (the anchor joins the run that starts the span).
        self.acc.clear();
        self.acc.resize(n * MAX_CANDIDATES, 0.0);
        self.back.clear();
        self.back.resize(n * MAX_CANDIDATES, 0);
        let trans = |a: f32, b: f32| {
            let c = cents(a, b).abs() - params.free_cents;
            if c > 0.0 {
                params.cost_per_semitone * c / 100.0
            } else {
                0.0
            }
        };
        let mut i = 0usize;
        while i < n {
            if !self.voiced[i] {
                f0[i] = 0.0;
                i += 1;
                continue;
            }
            let start = i;
            let mut end = i;
            while end < n && self.voiced[end] {
                end += 1;
            }
            let from = if start == 0 { anchor.filter(|h| *h > 0.0) } else { None };
            for f in start..end {
                let k = self.count[f] as usize;
                for c in 0..k {
                    let cand = self.cand[f * MAX_CANDIDATES + c];
                    let (cost, back) = if f == start {
                        (cand.cost + from.map_or(0.0, |h| trans(h, cand.hz)), 0u8)
                    } else {
                        let kp = self.count[f - 1] as usize;
                        let mut best = (f32::INFINITY, 0u8);
                        for p in 0..kp {
                            let prev = self.cand[(f - 1) * MAX_CANDIDATES + p];
                            let v = self.acc[(f - 1) * MAX_CANDIDATES + p] + trans(prev.hz, cand.hz);
                            if v < best.0 {
                                best = (v, p as u8);
                            }
                        }
                        (cand.cost + best.0, best.1)
                    };
                    self.acc[f * MAX_CANDIDATES + c] = cost;
                    self.back[f * MAX_CANDIDATES + c] = back;
                }
            }
            // Trace back from the newest frame's cheapest candidate (ties: the extractor's value).
            let last = end - 1;
            let mut c = 0usize;
            for j in 1..self.count[last] as usize {
                if self.acc[last * MAX_CANDIDATES + j] < self.acc[last * MAX_CANDIDATES + c] {
                    c = j;
                }
            }
            let mut f = last;
            loop {
                f0[f] = self.cand[f * MAX_CANDIDATES + c].hz;
                if f == start {
                    break;
                }
                c = self.back[f * MAX_CANDIDATES + c] as usize;
                f -= 1;
            }
            i = end;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A salience row with Gaussian-ish peaks at `(bin, height)`.
    fn row(peaks: &[(usize, f32)]) -> Vec<f32> {
        let mut r = vec![0.0f32; SALIENCE_BINS];
        for &(b, h) in peaks {
            for (j, v) in r.iter_mut().enumerate() {
                let d = j as f32 - b as f32;
                *v = v.max(h * (-d * d / 4.0).exp());
            }
        }
        r
    }

    /// Bin of `hz` on the 20-cent grid.
    fn bin(hz: f32) -> usize {
        ((1200.0 * (hz / 10.0).log2() - SALIENCE_CENTS_OFFSET) / 20.0).round() as usize
    }

    fn rows(frames: &[Vec<(usize, f32)>]) -> Vec<f32> {
        frames.iter().flat_map(|p| row(p)).collect()
    }

    /// The extractor's contour of rows: argmax local average, unvoiced at or under 0.03.
    fn own(sal: &[f32]) -> Vec<f32> {
        sal.chunks_exact(SALIENCE_BINS)
            .map(|r| {
                let (hz, peak) = super::super::pitch_onnx::local_average_f0(r);
                if peak <= 0.03 {
                    0.0
                } else {
                    hz
                }
            })
            .collect()
    }

    #[test]
    fn a_clean_contour_passes_through_bit_for_bit() {
        let b = bin(150.0);
        let frames: Vec<Vec<(usize, f32)>> = (0..12).map(|i| vec![(b + i / 3, 0.9)]).collect();
        let sal = rows(&frames);
        let mut f0 = own(&sal);
        let want = f0.clone();
        let mut d = F0Decoder::default();
        assert!(d.decode(Some(want[0]), &mut f0, &sal, &DecodeParams::speech()));
        assert_eq!(f0, want);
        // Unvoiced frames with no salience stay unvoiced; the anchor alone voices nothing.
        let mut z = vec![0.0f32; 4];
        assert!(d.decode(Some(150.0), &mut z, &rows(&[vec![], vec![], vec![], vec![]]), &DecodeParams::speech()));
        assert_eq!(z, vec![0.0; 4]);
    }

    #[test]
    fn a_short_octave_error_with_a_competing_peak_follows_the_anchor() {
        let b = bin(140.0);
        let up = b + 60; // +1200 cents
                         // The newest two frames' argmax jumps an octave; the true pitch is still a strong peak.
        let mut frames: Vec<Vec<(usize, f32)>> = (0..10).map(|_| vec![(b, 0.9)]).collect();
        frames.push(vec![(up, 0.8), (b, 0.5)]);
        frames.push(vec![(up, 0.8), (b, 0.5)]);
        let sal = rows(&frames);
        let mut f0 = own(&sal);
        assert!(cents(140.0, f0[11]) > 1100.0, "the stand-in extractor errs");
        let mut d = F0Decoder::default();
        d.decode(Some(140.0), &mut f0, &sal, &DecodeParams::speech());
        for (i, v) in f0.iter().enumerate() {
            assert!(cents(140.0, *v).abs() < 30.0, "frame {i}: {v}");
        }
    }

    #[test]
    fn a_real_leap_is_followed_when_nothing_supports_the_old_pitch() {
        let b = bin(140.0);
        let up = bin(280.0);
        let mut frames: Vec<Vec<(usize, f32)>> = (0..6).map(|_| vec![(b, 0.9)]).collect();
        frames.extend((0..6).map(|_| vec![(up, 0.9)]));
        let sal = rows(&frames);
        let mut f0 = own(&sal);
        let want = f0.clone();
        let mut d = F0Decoder::default();
        d.decode(Some(140.0), &mut f0, &sal, &DecodeParams::speech());
        assert_eq!(f0, want, "no candidate near the old pitch: the leap stands");
        // Even a leap at the newest frame alone stands.
        let mut frames: Vec<Vec<(usize, f32)>> = (0..11).map(|_| vec![(b, 0.9)]).collect();
        frames.push(vec![(up, 0.9)]);
        let sal = rows(&frames);
        let mut f0 = own(&sal);
        let want = f0.clone();
        d.decode(Some(140.0), &mut f0, &sal, &DecodeParams::speech());
        assert_eq!(f0, want);
    }

    #[test]
    fn hysteresis_has_two_thresholds() {
        let b = bin(200.0);
        let mut d = F0Decoder::default();
        // Sustain below the extractor's threshold (voice_off): voiced, a dip to 0.02 / 0.015 (under
        // RMVPE's 0.03), voiced again; under voice_off the voicing ends.
        let sustain = DecodeParams {
            voice_off: 0.01,
            ..DecodeParams::speech()
        };
        let frames = vec![vec![(b, 0.9)], vec![(b, 0.02)], vec![(b, 0.015)], vec![(b, 0.9)], vec![(b, 0.005)]];
        let sal = rows(&frames);
        let mut f0 = own(&sal);
        assert_eq!(f0[1], 0.0);
        d.decode(None, &mut f0, &sal, &sustain);
        assert!(f0[1] > 0.0 && f0[2] > 0.0, "{f0:?}");
        assert_eq!(f0[4], 0.0, "below voice_off the voicing ends");
        // ... and a voiced anchor sustains the first frame of the span.
        let frames = vec![vec![(b, 0.02)], vec![(b, 0.9)]];
        let sal = rows(&frames);
        let mut f0 = own(&sal);
        d.decode(Some(200.0), &mut f0, &sal, &sustain);
        assert!(f0[0] > 0.0);
        // The defaults keep the extractor's own voicing once voiced (no sustain below it).
        let frames = vec![vec![(b, 0.9)], vec![(b, 0.02)], vec![(b, 0.9)]];
        let sal = rows(&frames);
        let mut f0 = own(&sal);
        let want = f0.clone();
        d.decode(None, &mut f0, &sal, &DecodeParams::speech());
        assert_eq!(f0, want);
        // An onset needs voice_on (0.05): a 0.04 peak after an unvoiced frame waits a frame, a
        // 0.04 peak inside a voiced run is kept (the extractor's own threshold, 0.03).
        let frames = vec![vec![], vec![(b, 0.04)], vec![(b, 0.9)], vec![(b, 0.04)]];
        let sal = rows(&frames);
        let mut f0 = own(&sal);
        assert!(f0[1] > 0.0 && f0[3] > 0.0);
        d.decode(Some(0.0), &mut f0, &sal, &DecodeParams::speech());
        assert_eq!(f0[1], 0.0, "{f0:?}");
        assert!(f0[2] > 0.0 && f0[3] > 0.0, "{f0:?}");
        // voice_on 0: the extractor's onsets exactly (no delay).
        let p = DecodeParams {
            voice_on: 0.0,
            ..DecodeParams::speech()
        };
        let mut f0 = own(&sal);
        let want = f0.clone();
        d.decode(Some(0.0), &mut f0, &sal, &p);
        assert_eq!(f0, want);
        // No hysteresis and no pitch cost: the extractor's contour exactly.
        let neutral = DecodeParams {
            voice_off: 0.0,
            cost_per_semitone: 0.0,
            ..DecodeParams::speech()
        };
        let frames = vec![vec![(b, 0.9)], vec![(b, 0.02)], vec![(b + 60, 0.8), (b, 0.5)]];
        let sal = rows(&frames);
        let mut f0 = own(&sal);
        let want = f0.clone();
        d.decode(Some(200.0), &mut f0, &sal, &neutral);
        assert_eq!(f0, want);
    }

    #[test]
    fn short_salience_is_refused_and_singing_never_decodes() {
        let mut d = F0Decoder::default();
        let mut f0 = vec![100.0f32; 3];
        assert!(!d.decode(None, &mut f0, &[0.5; SALIENCE_BINS * 2], &DecodeParams::speech()));
        assert_eq!(f0, vec![100.0; 3]);
        assert!(DecodeParams::for_range(F0Range::Singing).is_none());
        assert!(DecodeParams::for_range(F0Range::Speech).is_some());
        assert!(available());
    }

    #[test]
    fn decoding_is_allocation_free_once_warm() {
        use crate::services::neural_audio::gpu::alloc_probe::allocations_during;
        let b = bin(160.0);
        let frames: Vec<Vec<(usize, f32)>> = (0..12)
            .map(|i| {
                if i % 5 == 4 {
                    vec![(b + 60, 0.7), (b, 0.6)]
                } else {
                    vec![(b, 0.8), (b + 30, 0.3)]
                }
            })
            .collect();
        let sal = rows(&frames);
        let base = own(&sal);
        let mut d = F0Decoder::default();
        let mut f0 = base.clone();
        d.decode(Some(160.0), &mut f0, &sal, &DecodeParams::speech());
        let mut f0 = base.clone();
        let (ok, allocs) = allocations_during(|| d.decode(Some(160.0), &mut f0, &sal, &DecodeParams::speech()));
        assert!(ok);
        assert_eq!(allocs, 0);
    }
}
