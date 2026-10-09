//! Per-engine content-encoder state (P3B-D): the voice's hold on its fixed-shape content plan, the
//! content window with real left context, and window-independent normalisation.
//!
//! - **Plan** ([`ContentStats::lease`]): the voice's [`ContentLease`] on the pooled encoder's
//!   fixed-shape plan, and whether the plan path is switched on for it (`content::CONTENT_PLAN_ENV`,
//!   resolved at the voice's first tick, which is load time).
//! - **Content window** ([`ContentWindow`]): with a content context longer than the
//!   stream's context (`SigilRvcRequest::content_context_16k`, the `content_context_ms` param), the
//!   encoder sees the newest `content_window_16k` samples of the stream, kept here from tick to
//!   tick (the harness's context stays the `enc` window). Without one, the window *is* the context
//!   and nothing is copied. After a gap or a reset the samples before the context are silence (the
//!   stream's own priming silence), never zero padding of the plan's shape.
//! - **Running normalisation** (the `content_norm` lever): per-engine running statistics of the
//!   encoder's first GroupNorm, handed to the encoder's `stream` variant every call, so every window
//!   normalises like a training slice instead of by its own statistics; at a speech onset after a
//!   long pause they glide from the window's own statistics to the running ones as the pause leaves
//!   the window ([`ONSET_BLEND_ENV`]). The engine computes the statistics itself from the first
//!   convolution's weights (`conv0.f32.bin`), so the `stream` graph needs no extra outputs.
//!
//! # Frozen by P3B-S0 (signatures; P3B-D implements)
//! - [`ContentStats`] (`Default`: inert), the engine's `content_stats` field;
//!   [`ContentStats::active`] (whether calls normalise with running statistics) and
//!   [`ContentStats::reset`] (called on every stream reset: forgets the window history and the
//!   statistics; the plan lease stays, sessions are not stream state).

use super::content::{ContentLease, SharedEncoder, NORM_CHANNELS};

/// Kept time of the running statistics, in 20 ms entries: RVC-family voices train on ~3.7 s slices.
pub const NORM_SLICE_ENTRIES: usize = 185;
/// A silent run longer than this many 20 ms entries (0.5 s) is not part of a training slice (RVC's
/// slicer cuts there); shorter pauses are kept.
pub const NORM_LONG_SILENCE: usize = 25;
/// Fewest kept entries (0.5 s) for running statistics; below it the window's own statistics are
/// used (stream start, after a reset or a gap).
pub const NORM_MIN_ENTRIES: usize = 25;
/// Speech gate of an entry (RMS, dBFS).
pub const NORM_GATE_DB: f32 = -50.0;
/// A channel's running variance is at least this share of the mean channel variance (clamps the
/// inverse standard deviation of a channel that barely moved in the kept audio).
pub const NORM_VAR_FLOOR: f64 = 0.01;
/// History kept (20 ms entries): ten seconds, so the last speech before a long pause still counts.
const NORM_RING: usize = 512;
/// Samples per entry: one 50 Hz content frame.
const ENTRY: usize = 320;
/// The first convolution: kernel and stride at 16 kHz.
const CONV0_K: usize = 10;
const CONV0_S: usize = 5;
const FRAMES_PER_ENTRY: usize = ENTRY / CONV0_S;
/// GroupNorm's epsilon.
const GN_EPS: f64 = 1e-5;

/// Running statistics of the content encoder's first GroupNorm (the `content_norm` lever): the
/// engine computes the first convolution's output of the newest audio itself (512 channels,
/// kernel 10, stride 5: about 0.65 MMAC per 40 ms), keeps per-20 ms sums and sums of squares for
/// the last ten seconds, and hands the encoder's `stream` variant the mean and inverse standard
/// deviation of the kept audio: the last 3.7 s of it with silences longer than 0.5 s removed and
/// shorter pauses kept, like an RVC training slice. With fewer than 0.5 s kept (a fresh stream)
/// it hands over the window's own statistics (over the window's whole 20 ms entries: what the
/// stock graph computes, less the first entry's partial frames). At a speech onset after a long
/// silence the statistics glide from the window's own to the running ones as the silence leaves
/// the window ([`ONSET_BLEND_ENV`]). Resets on a stream reset and on gaps. Allocation-free per tick.
pub struct RunningNorm {
    /// The first convolution's weights, transposed: `w[k * 512 + c]`.
    w: Vec<f32>,
    /// Per entry (ring of `NORM_RING`): 512 sums, then 512 sums of squares.
    sums: Vec<f32>,
    speech: Vec<bool>,
    /// Absolute index of the next entry to compute (`None`: no history).
    next: Option<u64>,
    /// Entries held (the newest is `next - 1`).
    held: usize,
    // ---- scratch ----
    acc: Vec<f32>,
    run_len: Vec<u16>,
    total: Vec<f64>,
    /// The window's own sums (the onset blend).
    win: Vec<f64>,
    /// The statistics of the last [`update`](Self::update): means then inverse standard deviations.
    out: Vec<f32>,
    /// The last statistics were running ones (`false`: the window's own fallback).
    pub last_running: bool,
    /// The share of the window's own statistics in the last running statistics: the part of the
    /// content window a long silence still fills (0 in steady speech; see [`ONSET_BLEND_ENV`]).
    pub last_onset_share: f32,
    /// The onset blend is on ([`onset_blend_switch`], read at construction).
    onset_blend: bool,
}

/// Switch of the running statistics' onset blend (P3B-D): `0` / `off` / `false` / `no` uses the
/// running statistics alone, also at speech onsets after a pause (for A/B runs from one
/// executable). On (the default): while a long silence (one the training slice drops) still fills
/// part of the content window, i.e. at a speech onset after a pause, the statistics move from the
/// window's own (the stock graph's) to the running ones as the silence leaves the window: the means
/// blend linearly and the inverse standard deviations geometrically, weighted by the share of the
/// window the silence fills. Running statistics alone muted the first ~50 ms of speech after a
/// 1.2 s pause (2 of 14 onsets of the 8-clip corpus, by 30-40 dB), where the window's own
/// statistics keep it (an onset that `enc`'s 0.44 s window barely holds). In steady speech (no long
/// silence in the window) the statistics are the running ones either way.
pub const ONSET_BLEND_ENV: &str = "SIGIL_VC_CONTENT_NORM_ONSET";

/// Whether the onset blend is on ([`ONSET_BLEND_ENV`]; on unless switched off). Reads the
/// environment: call it at load time, not per tick.
pub fn onset_blend_switch() -> bool {
    match std::env::var(ONSET_BLEND_ENV) {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no"),
        Err(_) => true,
    }
}

impl std::fmt::Debug for RunningNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunningNorm").field("held", &self.held).field("next", &self.next).finish()
    }
}

impl RunningNorm {
    /// `conv0`: the first convolution's weights as the export writes them (`conv0.f32.bin`:
    /// `[512, 1, 10]` row-major f32).
    pub fn new(conv0: &[f32]) -> Result<RunningNorm, String> {
        if conv0.len() != NORM_CHANNELS * CONV0_K {
            return Err(format!("the first convolution has {} weights, expected {}", conv0.len(), NORM_CHANNELS * CONV0_K));
        }
        let mut w = vec![0.0f32; NORM_CHANNELS * CONV0_K];
        for c in 0..NORM_CHANNELS {
            for k in 0..CONV0_K {
                w[k * NORM_CHANNELS + c] = conv0[c * CONV0_K + k];
            }
        }
        Ok(RunningNorm {
            w,
            sums: vec![0.0; NORM_RING * 2 * NORM_CHANNELS],
            speech: vec![false; NORM_RING],
            next: None,
            held: 0,
            acc: vec![0.0; NORM_CHANNELS],
            run_len: vec![0; NORM_RING],
            total: vec![0.0; 2 * NORM_CHANNELS],
            win: vec![0.0; 2 * NORM_CHANNELS],
            out: vec![0.0; 2 * NORM_CHANNELS],
            last_running: false,
            last_onset_share: 0.0,
            onset_blend: onset_blend_switch(),
        })
    }

    /// Switch the onset blend on or off (tests and benchmarks; [`ONSET_BLEND_ENV`] sets it at
    /// construction).
    pub fn set_onset_blend(&mut self, on: bool) {
        self.onset_blend = on;
    }

    /// Forget the history (stream reset).
    pub fn reset(&mut self) {
        self.next = None;
        self.held = 0;
    }

    /// Entries held (20 ms each).
    pub fn held(&self) -> usize {
        self.held
    }

    /// Take in the audio of `mono` (first sample at absolute 16 kHz index `pos`; consecutive calls
    /// overlap, only new entries are computed) and return the statistics for a window of
    /// `window_entries` entries ending at its end: `(mean, inv_std)`, [`NORM_CHANNELS`] each.
    pub fn update(&mut self, mono: &[f32], pos: u64, window_entries: usize) -> (&[f32], &[f32]) {
        self.advance(mono, pos);
        self.aggregate(window_entries);
        let (m, v) = self.out.split_at(NORM_CHANNELS);
        (m, v)
    }

    /// Take in the audio of `mono` without computing statistics (ticks skipped as silence keep
    /// the history continuous: a short pause belongs to the slice, a long one is dropped from it).
    pub fn advance(&mut self, mono: &[f32], pos: u64) {
        let end = pos + mono.len() as u64;
        // Entry k: the first convolution's frames starting at 320 k - 10 + 5 j (j < 64), complete
        // once the audio reaches 320 k + 315.
        let first = (pos + CONV0_K as u64).div_ceil(ENTRY as u64);
        let last_excl = if end >= 315 { (end - 315) / ENTRY as u64 + 1 } else { 0 };
        let start = match self.next {
            Some(n) if n >= first && n <= last_excl => n,
            // No history, a gap or a jump back: start over from what this context holds.
            _ => {
                self.reset();
                first
            }
        };
        for k in start..last_excl {
            self.compute_entry(mono, pos, k);
        }
        if last_excl > start || self.next.is_none() {
            self.next = Some(last_excl.max(start));
        }
    }

    fn compute_entry(&mut self, mono: &[f32], pos: u64, k: u64) {
        let slot = (k % NORM_RING as u64) as usize;
        let base = (k * ENTRY as u64 - CONV0_K as u64 - pos) as usize;
        let c = NORM_CHANNELS;
        let (sum, sq) = self.sums[slot * 2 * c..(slot + 1) * 2 * c].split_at_mut(c);
        sum.fill(0.0);
        sq.fill(0.0);
        for j in 0..FRAMES_PER_ENTRY {
            let s = base + j * CONV0_S;
            self.acc.fill(0.0);
            for t in 0..CONV0_K {
                let x = mono[s + t];
                let wt = &self.w[t * c..(t + 1) * c];
                for (a, &w) in self.acc.iter_mut().zip(wt) {
                    *a += w * x;
                }
            }
            for ((s1, s2), &a) in sum.iter_mut().zip(sq.iter_mut()).zip(&self.acc) {
                *s1 += a;
                *s2 += a * a;
            }
        }
        // The speech gate over the entry's own 20 ms.
        let from = base + CONV0_K;
        let seg = &mono[from..(from + ENTRY).min(mono.len())];
        let p = seg.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / seg.len().max(1) as f64;
        let db = 10.0 * (p + 1e-12).log10();
        self.speech[slot] = db > f64::from(NORM_GATE_DB);
        self.held = (self.held + 1).min(NORM_RING);
    }

    /// The statistics of the kept entries (or of the newest `window_entries` as the fallback) into
    /// `out`.
    fn aggregate(&mut self, window_entries: usize) {
        let c = NORM_CHANNELS;
        let held = self.held;
        let newest = self.next.unwrap_or(0);
        // Entry i of the held run (0 = oldest) is absolute entry newest - held + i.
        let slot_of = |i: usize| ((newest - held as u64 + i as u64) % NORM_RING as u64) as usize;
        // Silent-run lengths (0 for speech).
        let mut i = 0;
        while i < held {
            if self.speech[slot_of(i)] {
                self.run_len[i] = 0;
                i += 1;
                continue;
            }
            let mut j = i;
            while j < held && !self.speech[slot_of(j)] {
                j += 1;
            }
            let len = (j - i).min(u16::MAX as usize) as u16;
            for r in &mut self.run_len[i..j] {
                *r = len;
            }
            i = j;
        }
        self.total.fill(0.0);
        let mut kept = 0usize;
        for i in (0..held).rev() {
            if kept == NORM_SLICE_ENTRIES {
                break;
            }
            if self.run_len[i] as usize > NORM_LONG_SILENCE {
                continue;
            }
            let s = slot_of(i);
            for (t, &v) in self.total.iter_mut().zip(&self.sums[s * 2 * c..(s + 1) * 2 * c]) {
                *t += f64::from(v);
            }
            kept += 1;
        }
        self.last_running = kept >= NORM_MIN_ENTRIES;
        // The window's own sums (the newest entries): the fallback below the minimum, and the
        // onset blend's share while a long silence still fills part of the window.
        let w = window_entries.min(held);
        let long_in_window = self.run_len[held - w..held]
            .iter()
            .filter(|&&r| r as usize > NORM_LONG_SILENCE)
            .count();
        let share = if !self.last_running {
            1.0
        } else if self.onset_blend && w > 0 {
            long_in_window as f64 / w as f64
        } else {
            0.0
        };
        self.last_onset_share = if self.last_running { share as f32 } else { 0.0 };
        if share > 0.0 {
            self.win.fill(0.0);
            for i in held - w..held {
                let s = slot_of(i);
                for (t, &v) in self.win.iter_mut().zip(&self.sums[s * 2 * c..(s + 1) * 2 * c]) {
                    *t += f64::from(v);
                }
            }
        }
        // Per side: the mean and the (floored) variance of each channel; then the means blend
        // linearly and the inverse standard deviations geometrically (their logarithms linearly).
        // A share of speech variance in a silent window's statistics would otherwise dominate them
        // (speech is ~40 dB above the floor), so a moment blend left the onset far from the
        // window's own normalisation (feature cosine 0.26 to the stock graph at an onset, 0.96-0.99
        // with the geometric blend; `scripts/bench/specs/p3b/d/README.md`).
        if share < 1.0 {
            moments_to_stats(&mut self.total, (kept * FRAMES_PER_ENTRY).max(1) as f64);
        }
        if share > 0.0 {
            moments_to_stats(&mut self.win, (w * FRAMES_PER_ENTRY).max(1) as f64);
        }
        for ch in 0..c {
            let (m, inv) = if share <= 0.0 {
                (self.total[ch], self.total[c + ch])
            } else if share >= 1.0 {
                (self.win[ch], self.win[c + ch])
            } else {
                (
                    (1.0 - share) * self.total[ch] + share * self.win[ch],
                    (self.total[c + ch].ln() * (1.0 - share) + self.win[c + ch].ln() * share).exp(),
                )
            };
            self.out[ch] = m as f32;
            self.out[c + ch] = inv as f32;
        }
    }
}

/// Sums and sums of squares over `frames` frames (`[sum; C] ++ [sum of squares; C]`) into means and
/// inverse standard deviations in place: the biased variance, floored at [`NORM_VAR_FLOOR`] of the
/// mean channel variance, then `1 / sqrt(var + eps)` like GroupNorm.
fn moments_to_stats(acc: &mut [f64], frames: f64) {
    let c = acc.len() / 2;
    let mut mean_var = 0.0;
    for ch in 0..c {
        let m = acc[ch] / frames;
        let v = (acc[c + ch] / frames - m * m).max(0.0);
        acc[ch] = m;
        acc[c + ch] = v;
        mean_var += v;
    }
    let floor = NORM_VAR_FLOOR * mean_var / c as f64;
    for v in &mut acc[c..] {
        *v = 1.0 / (v.max(floor) + GN_EPS).sqrt();
    }
}

/// The voice's running normalisation: the encoder's `stream` variant (pooled) and the statistics.
pub struct NormState {
    pub(super) encoder: SharedEncoder,
    pub(super) lease: Option<ContentLease>,
    pub(super) running: RunningNorm,
}

impl std::fmt::Debug for NormState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NormState").field("lease", &self.lease).field("running", &self.running).finish()
    }
}

/// Whether the plan path applies to the voice (resolved at its first tick).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlanSwitch {
    /// Not resolved yet (before the first tick).
    #[default]
    Unresolved,
    /// The kill switch is set, or the encoder runs on the CPU: the dynamic session.
    Off,
    /// Fixed-shape plans on the encoder's GPU rung.
    On,
}

/// Per-engine content state (see the module docs).
#[derive(Debug, Default)]
pub struct ContentStats {
    /// Whether the voice runs content plans.
    pub(super) plan: PlanSwitch,
    /// The plan runs without a CUDA graph although the `cuda_graphs` lever is on
    /// (`content::CONTENT_GRAPH_ENV`; resolved with `plan`).
    pub(super) no_graph: bool,
    /// The voice's hold on its plan (kept across stream resets; replaced when the window or the
    /// levers change).
    pub(super) lease: Option<ContentLease>,
    /// The content window when it is longer than the stream's context.
    pub(super) ring: ContentWindow,
    /// Running normalisation (the `content_norm` lever resolved on and the encoder's `stream`
    /// variant loaded); `None`: windows normalise themselves (the stock graph).
    pub(super) norm: Option<NormState>,
}

impl Drop for ContentStats {
    /// A voice that goes away hands back its plan lease and its `stream` variant encoder, then lets
    /// the pool free what no other hot voice uses (the engine's own `Drop` releases the pool before
    /// its fields drop, when this state still holds them).
    fn drop(&mut self) {
        let held = self.lease.take().is_some() | self.norm.take().is_some();
        if held {
            super::content::ContentEncoderPool::global().release_unused();
        }
    }
}

impl ContentStats {
    /// Whether calls normalise with the running statistics.
    #[inline]
    pub fn active(&self) -> bool {
        self.norm.is_some()
    }

    /// Forget the window history and the statistics (stream gap, restart). The plan leases stay.
    #[inline]
    pub fn reset(&mut self) {
        self.ring.reset();
        if let Some(n) = self.norm.as_mut() {
            n.running.reset();
        }
    }

    /// Whether the voice runs a fixed-shape plan (its `stream` variant's when normalising).
    pub fn runs_plan(&self) -> bool {
        match &self.norm {
            Some(n) => n.lease.as_ref().is_some_and(ContentLease::fixed),
            None => self.lease.as_ref().is_some_and(ContentLease::fixed),
        }
    }
}

/// The content window kept from tick to tick when it is longer than the stream's context.
#[derive(Debug, Default)]
pub struct ContentWindow {
    /// Oldest sample first.
    buf: Vec<f32>,
    /// Absolute 16 kHz position just after the newest sample (`None`: no history).
    end: Option<u64>,
}

impl ContentWindow {
    /// Forget the history (the buffer keeps its capacity).
    pub fn reset(&mut self) {
        self.end = None;
    }

    /// The content window of this tick: the newest `len` samples of the stream ending where `mono`
    /// (the context, first sample at absolute 16 kHz index `pos`) ends. With `len <= mono.len()`
    /// that is a slice of `mono` (no copy); otherwise the window kept from earlier ticks, advanced by
    /// the samples `mono` adds (silence before the context when there is no history: the stream
    /// started, had a gap or jumped). Allocation-free once the window has its length.
    pub fn window<'a>(&'a mut self, mono: &'a [f32], pos: u64, len: usize) -> &'a [f32] {
        if len <= mono.len() {
            return &mono[mono.len() - len..];
        }
        self.advance(mono, pos, len);
        &self.buf
    }

    /// Advance the kept window by `mono` (also on ticks that skip inference, so the history stays
    /// continuous); a no-op when the window is the context.
    pub fn advance(&mut self, mono: &[f32], pos: u64, len: usize) {
        if len <= mono.len() {
            return;
        }
        let end = pos + mono.len() as u64;
        match self.end {
            Some(prev) if self.buf.len() == len && end > prev && end - prev <= mono.len() as u64 => {
                let delta = (end - prev) as usize;
                self.buf.copy_within(delta.., 0);
                let n = self.buf.len();
                self.buf[n - delta..].copy_from_slice(&mono[mono.len() - delta..]);
            }
            // The same span again (a repeated call): the window already ends here.
            Some(prev) if self.buf.len() == len && end == prev => {}
            _ => {
                self.buf.clear();
                self.buf.resize(len - mono.len(), 0.0);
                self.buf.extend_from_slice(mono);
            }
        }
        self.end = Some(end);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic first-convolution weights (`[512, 1, 10]`).
    pub(crate) fn conv0_weights() -> Vec<f32> {
        (0..NORM_CHANNELS * CONV0_K)
            .map(|i| (((i * 7919) % 211) as f32 / 105.0 - 1.0) * 0.3)
            .collect()
    }

    /// Mean and variance (f64) of the first convolution's output over its frames starting at
    /// `starts` (absolute samples of `audio`).
    fn direct_stats(w: &[f32], audio: &[f32], starts: &[usize]) -> (Vec<f64>, Vec<f64>) {
        let c = NORM_CHANNELS;
        let (mut s, mut q) = (vec![0.0f64; c], vec![0.0f64; c]);
        for &st in starts {
            for ch in 0..c {
                let y: f64 = (0..CONV0_K).map(|k| f64::from(w[ch * CONV0_K + k]) * f64::from(audio[st + k])).sum();
                s[ch] += y;
                q[ch] += y * y;
            }
        }
        let n = starts.len() as f64;
        let mean: Vec<f64> = s.iter().map(|v| v / n).collect();
        let var: Vec<f64> = q.iter().zip(&mean).map(|(v, m)| (v / n - m * m).max(0.0)).collect();
        (mean, var)
    }

    /// Frame starts of entries `ks` (absolute).
    fn starts_of(ks: impl Iterator<Item = usize>) -> Vec<usize> {
        ks.flat_map(|k| (0..FRAMES_PER_ENTRY).map(move |j| k * ENTRY - CONV0_K + j * CONV0_S))
            .collect()
    }

    fn check(got: (&[f32], &[f32]), want: &(Vec<f64>, Vec<f64>), what: &str) {
        let (m, v) = want;
        let mean_var = v.iter().sum::<f64>() / v.len() as f64;
        for ch in 0..NORM_CHANNELS {
            let floor = NORM_VAR_FLOOR * mean_var;
            let inv = 1.0 / (v[ch].max(floor) + GN_EPS).sqrt();
            assert!((f64::from(got.0[ch]) - m[ch]).abs() <= 1e-4 * (1.0 + m[ch].abs()), "{what}: mean[{ch}] {} vs {}", got.0[ch], m[ch]);
            assert!((f64::from(got.1[ch]) - inv).abs() <= 1e-3 * inv, "{what}: inv_std[{ch}] {} vs {inv}", got.1[ch]);
        }
    }

    /// A tone burst at `db` dBFS for `n` samples.
    fn tone(n: usize, db: f32, f: f32, phase0: usize) -> Vec<f32> {
        let a = 10f32.powf(db / 20.0) * std::f32::consts::SQRT_2;
        (0..n)
            .map(|i| a * (std::f32::consts::TAU * f * (i + phase0) as f32 / 16_000.0).sin())
            .collect()
    }

    /// What the statistics of a context ending at `end` (a multiple of 320) must be, from the
    /// stream's audio since sample 0: entries 1.. (entry `k` holds the first convolution's frames
    /// starting at `320 k - 10 + 5 j`, `j < 64`), gated per 20 ms, silent runs longer than
    /// `NORM_LONG_SILENCE` dropped, the newest `NORM_SLICE_ENTRIES` kept ones; below
    /// `NORM_MIN_ENTRIES` the window's own (its newest `window` entries); with `blend`, moved toward
    /// the window's own by the share of the window a long silence fills (means linearly, inverse
    /// standard deviations geometrically). Returns `(mean, inverse std)`, the share and whether the
    /// statistics are running ones.
    fn expected(w: &[f32], audio: &[f32], end: usize, window: usize, blend: bool) -> ((Vec<f64>, Vec<f64>), f64, bool) {
        let q = end / ENTRY;
        let all: Vec<usize> = (1..q).collect();
        let mut long = vec![false; all.len()];
        let mut i = 0;
        while i < all.len() {
            if gate(audio, all[i]) {
                i += 1;
                continue;
            }
            let mut j = i;
            while j < all.len() && !gate(audio, all[j]) {
                j += 1;
            }
            if j - i > NORM_LONG_SILENCE {
                long[i..j].iter_mut().for_each(|l| *l = true);
            }
            i = j;
        }
        let kept: Vec<usize> = all.iter().zip(&long).filter(|(_, l)| !**l).map(|(k, _)| *k).collect();
        let slice: Vec<usize> = kept[kept.len().saturating_sub(NORM_SLICE_ENTRIES)..].to_vec();
        let wn = window.min(all.len());
        let win: Vec<usize> = all[all.len() - wn..].to_vec();
        let window_stats = direct_stats(w, audio, &starts_of(win.into_iter()));
        if kept.len() < NORM_MIN_ENTRIES {
            return (to_inv(&window_stats), 1.0, false);
        }
        let running = direct_stats(w, audio, &starts_of(slice.into_iter()));
        let share = if blend {
            long[all.len() - wn..].iter().filter(|l| **l).count() as f64 / wn as f64
        } else {
            0.0
        };
        let (rm, ri) = to_inv(&running);
        let (wm, wi) = to_inv(&window_stats);
        let mean = rm.iter().zip(&wm).map(|(r, w)| (1.0 - share) * r + share * w).collect();
        let inv = ri.iter().zip(&wi).map(|(r, w)| (r.ln() * (1.0 - share) + w.ln() * share).exp()).collect();
        ((mean, inv), share, true)
    }

    /// `(mean, var)` to `(mean, inverse std)` with the variance floor and GroupNorm's epsilon.
    fn to_inv((m, v): &(Vec<f64>, Vec<f64>)) -> (Vec<f64>, Vec<f64>) {
        let floor = NORM_VAR_FLOOR * v.iter().sum::<f64>() / v.len() as f64;
        (m.clone(), v.iter().map(|x| 1.0 / (x.max(floor) + GN_EPS).sqrt()).collect())
    }

    /// [`check`] against expected means and inverse standard deviations.
    fn check_inv(got: (&[f32], &[f32]), (m, inv): &(Vec<f64>, Vec<f64>), what: &str) {
        for ch in 0..NORM_CHANNELS {
            assert!((f64::from(got.0[ch]) - m[ch]).abs() <= 1e-4 * (1.0 + m[ch].abs()), "{what}: mean[{ch}] {} vs {}", got.0[ch], m[ch]);
            assert!((f64::from(got.1[ch]) - inv[ch]).abs() <= 1e-3 * inv[ch], "{what}: inv_std[{ch}] {} vs {}", got.1[ch], inv[ch]);
        }
    }

    /// The speech gate of entry `k` (its own 20 ms).
    fn gate(audio: &[f32], k: usize) -> bool {
        let seg = &audio[k * ENTRY..(k * ENTRY + ENTRY).min(audio.len())];
        let p = seg.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / seg.len() as f64;
        10.0 * (p + 1e-12).log10() > f64::from(NORM_GATE_DB)
    }

    /// The running statistics are exactly the first convolution's statistics over the training
    /// slice: the window's own at the stream's start; then the kept entries (a short pause kept, a
    /// long one dropped), at most 3.7 s of them; at a speech onset after a long pause, blended with
    /// the window's own for the share of the window the pause still fills (with the onset blend
    /// off: the slice's alone); a reset or a gap starts over.
    #[test]
    fn running_statistics_model_training_slices() {
        let w = conv0_weights();
        let mut norm = RunningNorm::new(&w).unwrap();
        norm.set_onset_blend(true);
        let mut plain = RunningNorm::new(&w).unwrap();
        plain.set_onset_blend(false);
        // 3 s of speech, a 0.3 s pause (kept), 2 s of speech, a 1.2 s pause (dropped), 1 s of
        // speech: entries are 320 samples, the stream's ticks 640.
        let mut audio = tone(48_000, -20.0, 180.0, 0);
        audio.extend(vec![0.0f32; 4_800]);
        audio.extend(tone(32_000, -18.0, 230.0, 7));
        audio.extend(vec![0.0f32; 19_200]);
        audio.extend(tone(16_000, -24.0, 140.0, 3));
        let (ctx, block) = (7_040usize, 640usize);
        let entries = ctx / ENTRY;
        let tick = |pos: usize, norm: &mut RunningNorm, plain: &mut RunningNorm| -> (Vec<f32>, Vec<f32>) {
            plain.update(&audio[pos..pos + ctx], pos as u64, entries);
            let (m, v) = norm.update(&audio[pos..pos + ctx], pos as u64, entries);
            (m.to_vec(), v.to_vec())
        };
        // The first tick: 21 entries (< 0.5 s): the window's own statistics.
        let mut pos = 0usize;
        let (m, v) = tick(pos, &mut norm, &mut plain);
        let (want, share, running) = expected(&w, &audio, ctx, entries, true);
        assert_eq!((share, running), (1.0, false));
        check_inv((&m, &v), &want, "window fallback");
        assert!(!norm.last_running);
        // Tick on to 2 s: running statistics over every entry so far (no pause yet).
        while pos + ctx < 32_000 {
            pos += block;
            tick(pos, &mut norm, &mut plain);
        }
        assert!(norm.last_running);
        let (m, v) = tick(pos, &mut norm, &mut plain);
        let (want, share, _) = expected(&w, &audio, pos + ctx, entries, true);
        assert_eq!(share, 0.0);
        assert_eq!(norm.last_onset_share, 0.0);
        check_inv((&m, &v), &want, "running, 2 s");
        // 0.2 s into the speech after the long pause: the pause still fills half the window, so
        // the window's own statistics take that share (the onset blend); without the blend, the
        // slice's alone.
        let resume = 48_000 + 4_800 + 32_000 + 19_200;
        while pos + ctx < resume + 3_200 {
            pos += block;
            tick(pos, &mut norm, &mut plain);
        }
        let (m, v) = tick(pos, &mut norm, &mut plain);
        let (want, share, running) = expected(&w, &audio, pos + ctx, entries, true);
        assert!(running && (0.4..1.0).contains(&share), "share {share}");
        assert!((f64::from(norm.last_onset_share) - share).abs() < 1e-6, "{} vs {share}", norm.last_onset_share);
        check_inv((&m, &v), &want, "onset blend after a long pause");
        let (want_plain, _, _) = expected(&w, &audio, pos + ctx, entries, false);
        let (pm, pv) = plain.update(&audio[pos..pos + ctx], pos as u64, entries);
        check_inv((pm, pv), &want_plain, "running alone after a long pause (blend off)");
        assert_eq!(plain.last_onset_share, 0.0);
        // Once the pause has left the window: the slice's statistics (the short pause in it, the
        // long one not, the newest 185 kept entries), the same with and without the blend.
        while pos + ctx < resume + 9_600 {
            pos += block;
            tick(pos, &mut norm, &mut plain);
        }
        let (m, v) = tick(pos, &mut norm, &mut plain);
        let (want, share, _) = expected(&w, &audio, pos + ctx, entries, true);
        assert_eq!(share, 0.0);
        assert_eq!(norm.last_onset_share, 0.0);
        check_inv((&m, &v), &want, "slice after a long pause");
        let (pm, _) = plain.update(&audio[pos..pos + ctx], pos as u64, entries);
        assert_eq!(pm, m.as_slice(), "in steady speech the blend changes nothing");
        // The slice reaches back over the 2 s of speech into the short pause (entries 150..165).
        let short_pause = (48_000 / ENTRY + 1)..(52_800 / ENTRY - 1);
        assert!(short_pause.clone().all(|k| !gate(&audio, k)));
        let newest = (pos + ctx) / ENTRY - 1;
        let long_pause = 60;
        assert!(newest + 1 - NORM_SLICE_ENTRIES - long_pause < short_pause.start, "the short pause is part of the slice");
        // A gap (the stream jumps three contexts ahead): the window's own statistics again.
        let far = pos + 4 * ctx;
        norm.update(&audio[pos..pos + ctx], far as u64, entries);
        assert!(!norm.last_running, "a gap starts over");
        assert_eq!(norm.held(), entries - 1);
        // A reset: the same.
        norm.reset();
        assert_eq!(norm.held(), 0);
        let (m, v) = norm.update(&audio[..ctx], 0, entries);
        check((m, v), &direct_stats(&w, &audio, &starts_of(1..22)), "after a reset");
        assert!(!norm.last_running);
    }

    /// The statistics update makes no heap allocation per tick.
    #[test]
    fn running_statistics_are_allocation_free() {
        use crate::services::neural_audio::gpu::alloc_probe::allocations_during;
        let mut norm = RunningNorm::new(&conv0_weights()).unwrap();
        let audio = tone(16_000 * 4, -20.0, 200.0, 0);
        let mut pos = 0usize;
        for _ in 0..10 {
            norm.update(&audio[pos..pos + 7_040], pos as u64, 22);
            pos += 640;
        }
        let ((), allocs) = allocations_during(|| {
            for _ in 0..20 {
                let (m, _) = norm.update(&audio[pos..pos + 7_040], pos as u64, 22);
                std::hint::black_box(m[0]);
                pos += 640;
            }
        });
        assert_eq!(allocs, 0);
        assert!(RunningNorm::new(&[0.0; 10]).is_err());
    }

    /// The kept window equals the stream's newest samples tick after tick, skipped ticks included;
    /// a gap or a reset starts over with silence before the context.
    #[test]
    fn the_window_follows_the_stream() {
        let stream: Vec<f32> = (0..60_000).map(|i| i as f32).collect();
        let (ctx, block, len) = (7_040usize, 640usize, 16_000usize);
        let mut s = ContentWindow::default();
        // Ticks start where the context fits; the first one has no history.
        let first = 12_000usize;
        let w = s.window(&stream[first..first + ctx], first as u64, len).to_vec();
        assert_eq!(w.len(), len);
        assert!(w[..len - ctx].iter().all(|&x| x == 0.0), "silence before the first context");
        assert_eq!(&w[len - ctx..], &stream[first..first + ctx]);
        let mut pos = first;
        for tick in 1..20 {
            pos += block;
            if tick % 5 == 0 {
                // A tick skipped as silence still advances the history.
                s.advance(&stream[pos..pos + ctx], pos as u64, len);
                continue;
            }
            let w = s.window(&stream[pos..pos + ctx], pos as u64, len);
            let end = pos + ctx;
            let want: Vec<f32> = (0..len).map(|i| if end - len + i >= first { stream[end - len + i] } else { 0.0 }).collect();
            assert_eq!(w, want.as_slice(), "tick {tick}");
        }
        // A context that overlaps the kept window continues it (the stream is the same).
        let (prev_end, pos) = (pos + ctx, pos + 3 * block);
        assert!(pos + ctx - prev_end <= ctx);
        let w = s.window(&stream[pos..pos + ctx], pos as u64, len).to_vec();
        assert_eq!(w.as_slice(), &stream[pos + ctx - len..pos + ctx]);
        // A gap longer than the context: history starts over.
        let w = s.window(&stream[45_000..45_000 + ctx], 45_000, len).to_vec();
        assert!(w[..len - ctx].iter().all(|&x| x == 0.0));
        assert_eq!(&w[len - ctx..], &stream[45_000..45_000 + ctx]);
        s.reset();
        let w = s.window(&stream[45_640..45_640 + ctx], 45_640, len).to_vec();
        assert!(w[..len - ctx].iter().all(|&x| x == 0.0), "a reset forgets the history");
        // No longer window than the context: a slice of the context, nothing kept.
        let mono = &stream[..ctx];
        assert_eq!(s.window(mono, 0, ctx).as_ptr(), mono.as_ptr());
    }
}
