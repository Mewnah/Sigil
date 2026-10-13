//! Output-buffer floor of a Sigil template voice that converts in bursts (pure rule; no I/O).
//!
//! # The failure this sizes away
//! A template voice runs its model once per engine block `B` (120 ms in the benchmark's CPU rows,
//! 40 ms on the reference GPU shape) and the worker is synchronous. With the device chunk `D`
//! (512 frames at 48 kHz = 10.67 ms) shorter than `B`, `B / D - 1` ticks cost almost nothing and
//! push one chunk each; the tick that completes a block runs the model (85-110 ms on the CPU at
//! 120 ms blocks, more under a game) and pushes nothing, so the device drains the ring for that long.
//! If the output queue stands shorter than the call, the device runs dry, `OutputPlayback` starts a
//! recovery, and while recovering every push trims the queue to `ring_target()` (the gate level plus
//! the last chunk, a capture burst and a margin: 57 ms at the 35 / 70 ms default), which discards the
//! catch-up burst (about 45 ms of every 120 ms block). The next call dries the ring again, so the
//! recovery never ends: 35-52 % of the converted audio was lost on the CPU rows (35.4-50.5 % on the
//! pool and 6-thread rows, 51.5 % on one thread), with the `output_max_queue_ms` cap trimming
//! nothing (the trims are all recovery trims). Raising only the queue cap therefore does nothing; the
//! prefill is the lever, because the gate and `ring_target()` follow it.
//!
//! # The rule
//! For an engine block of `B` ms on a device chunk of `D` ms ([`min_output_buffer_ms`]):
//! - `n = ceil((B - 0.4) / D) - 1` pushes happen before the first model call
//!   ([`pushes_before_first_call`]; 0.4 ms is the fixed-block mode's 4 sample lead plus the input
//!   resampler's 2 sample jitter at 16 kHz). `n * D` is the **first-call limit**: the prefill gate
//!   must open before that call, or the queue stands 20-40 ms higher for the whole run.
//! - `cover = ceil5(0.9 * B)`: the call's p99 of any configuration that keeps up (real-time factor
//!   at most 0.9; measured 0.74-0.89 on the benchmark rig), on a 5 ms grid.
//! - `prefill = min(cover, floor5(n * D - 2))` (2 ms of margin against the first-call trap), and no
//!   floor at all when that is not above the 35 ms default.
//! - `queue = ceil(prefill + 2 * D + 4)`: the level after a catch-up (at most the prefill plus two
//!   chunks) plus 4 ms, so the cap never trims a healthy cycle.
//!
//! At 120 ms blocks on 512 frames at 48 kHz that is 110 / 136 ms.
//!
//! # Which stored values it raises
//! Nothing stored is rewritten, and a larger stored value always wins. Below the floor:
//! - the **stock buffers** (the Low latency 35 / 70 ms and the Stable 80 / 150 ms templates: numbers
//!   the app wrote for short blocks, not a choice about this voice) are raised to it;
//! - a **custom** buffer (the user typed the numbers) is the user's choice and stays, unless the run
//!   proves that it cannot play a block: [`StoredBufferProof`] watches the model calls of the run, and
//!   once the median of the last [`CALL_WINDOW`] calls is longer than the stored prefill (the queue
//!   the output keeps standing while a call runs and nothing is pushed), the device runs dry in more
//!   than half of the blocks, so the floor applies from then on (latched, so the delay does not flap;
//!   a new stored value is judged anew). A custom value that holds the typical call is never touched,
//!   however far under the floor it is. The criterion is the knee of the burst-worker simulation
//!   (`playback::tests::the_proof_criterion_is_the_knee_of_the_burst_worker`): wherever it says the
//!   buffer cannot play, the simulated output loses audio and runs dry; it says nothing for the
//!   borderline values just under the call, which the user keeps.
//!
//! # The device callback
//! The rule has no term for the playback device's callback period: `OutputPlayback` stands the queue's
//! valley a callback above the gate level by itself (`ring_target`, `drift_slack`), so a longer period raises
//! the standing queue without a larger prefill. The burst-worker simulation plays 98 ms calls at 110 /
//! 136 ms with no dry callback on 10, 20 and 40 ms callbacks
//! (`playback::tests::a_longer_device_callback_needs_no_larger_floor`), so none is added.
//!
//! # When the cover is longer than the first-call limit
//! When the call is longer than about 108 ms (a real-time factor above 0.9 at 120 ms blocks, such as
//! the 4-thread policy at 0.96 or FCPE on 6 threads at 0.97), the trap guard wins and the prefill stays
//! at the limit. Raising it past the limit costs a standing +20-40 ms in every block of the run (the
//! gate opens after the first call, replayed: a 299 ms median against 278); keeping it costs a dry
//! hold and one recovery in each call that outruns the queue, which the simulation puts at 1.4 % of
//! the audio when every call is 10 ms longer than the prefill (100 / 125 ms at 110 ms calls) and
//! nothing for the occasional slow call. The rows re-run with the floor are the 6-thread SwiftF0
//! ones; the 4-thread and FCPE shapes were not, and a PC that needs them should run Autotune (a
//! longer block).
//!
//! It applies to CPU generators only (a GPU call is a few ms: the 40 ms reference shape gets no
//! floor: `n = 3`, limit 30 ms, below the default), is calibrated on one rig (the cover term is an
//! RTF assumption, so faster CPUs are over-provisioned at long blocks for the stock buffers), and
//! cannot help a CPU whose call is longer than the block (no buffer can). `SIGIL_VC_BLOCK_BUFFER=0`
//! switches it off.

use super::params::{clamp_output_buffer_ms, PREFILL_MS_LOW};

/// The kill switch: `0` / `off` / `false` / `no` (any case) restores the user's buffer for every voice.
pub(crate) const ENV: &str = "SIGIL_VC_BLOCK_BUFFER";

/// Grid the prefill snaps to (ms).
const GRID_MS: f64 = 5.0;
/// The conversion's assumed real-time factor (call time over block time) the cover is sized for. The
/// benchmark rig measures 0.74-0.89 idle on the Streamer shape.
pub(crate) const COVER_RTF: f64 = 0.9;
/// Fixed-block mode's early / late slack before the first model call (ms): `FIXED_LEAD_16K` (4) plus
/// the input resampler's jitter (2) samples at 16 kHz, rounded up.
const FIRST_CALL_SLACK_MS: f64 = 0.4;
/// Margin under the first-call limit (ms): the trap guard.
const TRAP_MARGIN_MS: f64 = 2.0;
/// Headroom of the queue cap over the post-catch-up level (ms).
const QUEUE_HEADROOM_MS: f64 = 4.0;
/// Tolerance of the grid and block-count roundings (floating point: `0.9 * B` is not exact).
const EPS: f64 = 1e-9;

/// Whether the rule is on for the value of [`ENV`] (`None` = unset = on).
pub(crate) fn enabled_from(value: Option<&str>) -> bool {
    let value = value.map(|v| v.trim().to_ascii_lowercase());
    !matches!(value.as_deref(), Some("0" | "off" | "false" | "no"))
}

/// Whether the rule is on in this process.
pub(crate) fn enabled() -> bool {
    enabled_from(std::env::var(ENV).ok().as_deref())
}

fn ceil5(ms: f64) -> f64 {
    (ms / GRID_MS - EPS).ceil() * GRID_MS
}

fn floor5(ms: f64) -> f64 {
    (ms / GRID_MS + EPS).floor() * GRID_MS
}

/// Pushes of the worker before the first model call of a stream with engine blocks of `block_ms` on
/// device chunks of `chunk_ms` (`None` when a block does not span two chunks). The first call runs
/// on the push that completes the first block; those before it cost nothing.
pub(crate) fn pushes_before_first_call(block_ms: f64, chunk_ms: f64) -> Option<u32> {
    if !(block_ms.is_finite() && chunk_ms.is_finite()) || chunk_ms <= 0.0 || block_ms <= chunk_ms {
        return None;
    }
    let n = ((block_ms - FIRST_CALL_SLACK_MS) / chunk_ms - EPS).ceil() - 1.0;
    (n >= 1.0).then_some(n as u32)
}

/// The floor `(prefill, max queue)` in ms a voice needs, or `None` (see the module docs). `on_cpu`
/// says the voice's generator runs on the CPU, `enabled` is [`enabled`] (passed in so tests never
/// touch the environment).
pub(crate) fn min_output_buffer_ms(block_ms: f64, chunk_ms: f64, on_cpu: bool, enabled: bool) -> Option<(f64, f64)> {
    if !enabled || !on_cpu {
        return None;
    }
    let n = pushes_before_first_call(block_ms, chunk_ms)?;
    let first_call_ms = f64::from(n) * chunk_ms;
    let cover = ceil5(COVER_RTF * block_ms);
    let limit = floor5(first_call_ms - TRAP_MARGIN_MS);
    let prefill = cover.min(limit);
    if prefill.partial_cmp(&PREFILL_MS_LOW) != Some(std::cmp::Ordering::Greater) {
        return None;
    }
    let queue = (prefill + 2.0 * chunk_ms + QUEUE_HEADROOM_MS - EPS).ceil();
    Some(clamp_output_buffer_ms(prefill, queue))
}

/// Model calls remembered for the median (odd, so the median is one of them).
pub(crate) const CALL_WINDOW: usize = 33;
/// Calls the window needs before its median counts as evidence (about 2 s of 120 ms blocks).
pub(crate) const MIN_CALLS: usize = 16;
/// A tick is a model call when it cost at least this many chunks of audio time (the ticks between
/// two calls cost a few hundred microseconds).
pub(crate) const CALL_TICK_CHUNKS: f64 = 3.0;

/// Whether a stored output prefill cannot play a voice's blocks: the typical (median) model call is
/// longer than the queue the prefill keeps standing while the call runs and nothing is pushed, so the
/// device runs dry in more than half of the blocks and the output's recovery trims the catch-up of
/// each (see the module docs).
pub(crate) fn prefill_cannot_hold_call(prefill_ms: f64, median_call_ms: f64) -> bool {
    prefill_ms < median_call_ms
}

/// The evidence behind raising a **custom** stored buffer to the floor (see the module docs): the
/// model calls of the run so far, and the verdict, latched per stored value. Worker-owned and
/// allocation-free.
#[derive(Clone, Debug)]
pub(crate) struct StoredBufferProof {
    calls: [f32; CALL_WINDOW],
    len: usize,
    next: usize,
    /// The stored prefill the verdict is about (ms).
    judged_prefill_ms: f64,
    raised: bool,
}

impl Default for StoredBufferProof {
    fn default() -> Self {
        Self {
            calls: [0.0; CALL_WINDOW],
            len: 0,
            next: 0,
            judged_prefill_ms: f64::NAN,
            raised: false,
        }
    }
}

impl StoredBufferProof {
    /// Forget everything (another voice plays).
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    /// One tick of the worker that took `tick_ms` for the model, on device chunks of `chunk_ms`; only
    /// the ticks that ran a call are kept.
    pub(crate) fn note_tick(&mut self, tick_ms: f64, chunk_ms: f64) {
        if !(tick_ms.is_finite() && chunk_ms.is_finite()) || chunk_ms <= 0.0 || tick_ms < CALL_TICK_CHUNKS * chunk_ms {
            return;
        }
        self.calls[self.next] = tick_ms as f32;
        self.next = (self.next + 1) % CALL_WINDOW;
        self.len = (self.len + 1).min(CALL_WINDOW);
    }

    /// Median of the remembered calls (ms), once there are [`MIN_CALLS`] of them.
    pub(crate) fn median_call_ms(&self) -> Option<f64> {
        if self.len < MIN_CALLS {
            return None;
        }
        let mut sorted = [0.0f32; CALL_WINDOW];
        sorted[..self.len].copy_from_slice(&self.calls[..self.len]);
        sorted[..self.len].sort_unstable_by(f32::total_cmp);
        Some(f64::from(sorted[self.len / 2]))
    }

    /// Whether the stored custom `prefill_ms` has been proven unable to hold the voice's calls. Once
    /// true it stays true for that stored value; a different value starts a new judgement.
    pub(crate) fn judge(&mut self, prefill_ms: f64) -> bool {
        if self.judged_prefill_ms.to_bits() != prefill_ms.to_bits() {
            self.judged_prefill_ms = prefill_ms;
            self.raised = false;
        }
        if !self.raised {
            if let Some(median) = self.median_call_ms() {
                self.raised = prefill_cannot_hold_call(prefill_ms, median);
            }
        }
        self.raised
    }

    /// The latched verdict (tests; the worker reads the verdict `judge` returns).
    #[cfg(test)]
    pub(crate) fn raised(&self) -> bool {
        self.raised
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::vc::stream::FIXED_LEAD_16K;

    /// 512 frames at 48 kHz.
    const D_512_48K: f64 = 512.0 * 1000.0 / 48_000.0;

    fn rule(b: f64, d: f64) -> Option<(f64, f64)> {
        min_output_buffer_ms(b, d, true, true)
    }

    /// The table the rule generates for the benchmark's device chunk, written from the formula (not
    /// copied from a design document): only the 120 ms row is pinned by value.
    #[test]
    fn the_table_follows_the_formula_and_pins_the_120_ms_row() {
        assert_eq!(rule(120.0, D_512_48K), Some((110.0, 136.0)));
        for b in (10..=200).step_by(10).map(f64::from) {
            let d = D_512_48K;
            let n = ((b - 0.4) / d).ceil() - 1.0;
            let limit = (((n * d) - 2.0) / 5.0 + 1e-9).floor() * 5.0;
            let cover = ((0.9 * b) / 5.0 - 1e-9).ceil() * 5.0;
            let prefill = cover.min(limit);
            let want = (prefill > 35.0).then(|| (prefill, (prefill + 2.0 * d + 4.0 - 1e-9).ceil()));
            assert_eq!(rule(b, d), want, "block {b} ms");
        }
        // Blocks of 40 ms and less get none: the reference GPU shape (n = 3, limit 30) and the app's.
        for b in [10.0, 20.0, 30.0, 40.0] {
            assert_eq!(rule(b, D_512_48K), None, "block {b} ms");
        }
    }

    #[test]
    fn other_device_chunks_follow_the_same_rule() {
        // 44.1 kHz capture: 11.61 ms chunks, ten pushes before the first call.
        let d441 = 512.0 * 1000.0 / 44_100.0;
        assert_eq!(pushes_before_first_call(120.0, d441), Some(10));
        assert_eq!(rule(120.0, d441), Some((110.0, 138.0)));
        // A chunk of exactly 10 ms: the first-call limit (110 ms) is the binding term.
        assert_eq!(pushes_before_first_call(120.0, 10.0), Some(11));
        assert_eq!(rule(120.0, 10.0), Some((105.0, 129.0)));
    }

    /// The Streamer's automatic blocks (120 ms from 7 physical cores, 160 ms with 6, 200 ms with 5 or fewer) get a valid floor on
    /// the benchmark's chunk and on the other device chunks: the prefill stays at or under the first-call limit (the gate opens
    /// before the first model call), is above the 35 ms default, covers 0.9 of the block when the limit allows, and the pair sits
    /// inside `clamp_output_buffer_ms` (the clamp changes nothing).
    #[test]
    fn the_streamers_small_pc_blocks_get_a_valid_floor() {
        use crate::services::neural_audio::vc::engine::presets::cpu_streamer_block_for_cores;
        // The floor of each tier on the benchmark's chunk (the 6-core tier's 160 ms block gets 145 / 171 ms).
        assert_eq!(cpu_streamer_block_for_cores(6), 160.0);
        assert_eq!(rule(cpu_streamer_block_for_cores(6), D_512_48K), Some((145.0, 171.0)));
        assert_eq!(rule(160.0, D_512_48K), Some((145.0, 171.0)));
        assert_eq!(rule(200.0, D_512_48K), Some((180.0, 206.0)));
        for cores in [0u32, 1, 4, 5, 6, 7, 8, 16] {
            let b = cpu_streamer_block_for_cores(cores);
            for (frames, rate) in [(512.0, 48_000.0), (512.0, 44_100.0), (480.0, 48_000.0), (256.0, 48_000.0), (1_024.0, 48_000.0), (1_024.0, 44_100.0)] {
                let d = frames * 1000.0 / rate;
                let (prefill, queue) = rule(b, d).unwrap_or_else(|| panic!("no floor for {b} ms blocks on {frames} frames at {rate} Hz"));
                let n = pushes_before_first_call(b, d).expect("a block of several chunks");
                let limit = f64::from(n) * d - TRAP_MARGIN_MS;
                assert!(prefill <= limit + 1e-9, "{b} ms / {frames}@{rate}: prefill {prefill} over the first-call limit {limit}");
                assert!(prefill > PREFILL_MS_LOW, "{b} ms / {frames}@{rate}: prefill {prefill}");
                assert!(prefill >= ceil5(COVER_RTF * b).min(floor5(limit)) - 1e-9, "{b} ms / {frames}@{rate}: prefill {prefill} does not cover the call");
                assert!(queue >= prefill + 2.0 * d + QUEUE_HEADROOM_MS - 1.0, "{b} ms / {frames}@{rate}: queue {queue}");
                assert_eq!(clamp_output_buffer_ms(prefill, queue), (prefill, queue), "{b} ms / {frames}@{rate}: the clamp moved the floor");
                assert!(prefill <= 300.0 && queue <= 600.0, "{b} ms / {frames}@{rate}: {prefill} / {queue}");
            }
        }
        // Not on a GPU rung, and not with the rule off.
        for b in [160.0, 200.0] {
            assert_eq!(min_output_buffer_ms(b, D_512_48K, false, true), None);
            assert_eq!(min_output_buffer_ms(b, D_512_48K, true, false), None);
        }
    }

    /// Tripwire: the first model call happens at push 11 on 120 ms / 512 frames and at push 3 on 40 ms,
    /// which is what the recorded timelines show. If the fixed-block lead or the resampler jitter
    /// change, the rule's 0.4 ms must follow them.
    #[test]
    fn pushes_before_the_first_call_match_the_recorded_timelines() {
        assert_eq!(pushes_before_first_call(120.0, D_512_48K), Some(11));
        assert_eq!(pushes_before_first_call(40.0, D_512_48K), Some(3));
        assert_eq!(pushes_before_first_call(20.0, D_512_48K), Some(1));
        assert_eq!(pushes_before_first_call(10.0, D_512_48K), None, "a block shorter than a chunk");
        assert!(
            (FIXED_LEAD_16K + 2) as f64 / 16.0 <= FIRST_CALL_SLACK_MS,
            "the fixed-block lead plus the resampler jitter no longer fit the rule's slack"
        );
    }

    #[test]
    fn nothing_applies_off_the_cpu_or_when_disabled_or_unusable() {
        for b in (10..=200).step_by(10).map(f64::from) {
            assert_eq!(min_output_buffer_ms(b, D_512_48K, false, true), None, "GPU rung, block {b}");
            assert_eq!(min_output_buffer_ms(b, D_512_48K, true, false), None, "disabled, block {b}");
        }
        assert!(rule(120.0, D_512_48K).is_some());
        for d in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(rule(120.0, d), None, "chunk {d}");
        }
        for b in [f64::NAN, f64::INFINITY, 0.0, -5.0] {
            assert_eq!(rule(b, D_512_48K), None, "block {b}");
        }
        // A block that does not span two chunks.
        assert_eq!(rule(10.0, 10.0), None);
        assert_eq!(rule(10.0, 20.0), None);
        assert_eq!(rule(120.0, 120.0), None);
        assert_eq!(rule(120.0, 130.0), None);
    }

    /// The documented values (`0`, `off`, `false`, `no`) switch it off in any case and with padding,
    /// like the other engine switches; anything else leaves it on.
    #[test]
    fn the_kill_switch_reads_its_values() {
        assert!(enabled_from(None));
        assert!(enabled_from(Some("1")));
        assert!(enabled_from(Some("on")));
        assert!(enabled_from(Some("ON")));
        assert!(enabled_from(Some("")));
        assert!(enabled_from(Some("maybe")));
        for off in ["0", "off", "false", "no", " 0 ", "OFF", "Off", "False", "FALSE", "No", "NO", " Off\t"] {
            assert!(!enabled_from(Some(off)), "{off:?}");
        }
    }

    /// The invariants over blocks of 10-200 ms and device chunks of 2-130 ms.
    #[test]
    fn the_rule_keeps_its_invariants() {
        let mut found = 0usize;
        let mut d = 2.0;
        while d <= 130.0 {
            let mut previous = 0.0f64;
            for b in (10..=200).step_by(10).map(f64::from) {
                let got = rule(b, d);
                if b <= d {
                    assert_eq!(got, None, "block {b} <= chunk {d}");
                }
                let prefill_now = got.map_or(0.0, |g| g.0);
                assert!(prefill_now >= previous, "chunk {d}: prefill fell from {previous} to {prefill_now} at block {b}");
                previous = prefill_now;
                let Some((prefill, queue)) = got else { continue };
                found += 1;
                let n = pushes_before_first_call(b, d).expect("a floor needs a first call");
                assert_eq!(prefill % 5.0, 0.0, "grid: block {b} chunk {d}");
                assert!(prefill > 35.0 && prefill <= 300.0, "range: {prefill} (block {b}, chunk {d})");
                assert!(
                    prefill <= f64::from(n) * d - 2.0 + 1e-6,
                    "first-call trap: prefill {prefill} > {n} x {d} - 2 (block {b})"
                );
                assert!(queue >= prefill + 2.0 * d + 3.9, "queue {queue} under prefill {prefill} + 2 x {d} + 4");
                assert!(queue <= 600.0, "queue {queue}");
                assert_eq!(clamp_output_buffer_ms(prefill, queue), (prefill, queue), "the clamp is the identity");
            }
            d += 0.5;
        }
        assert!(found > 500, "the sweep must reach floors, not only Nones: {found}");
    }

    fn feed(p: &mut StoredBufferProof, calls_ms: &[f64], chunk_ms: f64) {
        for &c in calls_ms {
            // Two cheap ticks between calls, as the worker has them.
            p.note_tick(0.05, chunk_ms);
            p.note_tick(0.3, chunk_ms);
            p.note_tick(c, chunk_ms);
        }
    }

    /// Only ticks that ran a call count, the median needs enough of them, and the window rolls.
    #[test]
    fn the_proof_keeps_calls_only_and_needs_enough_of_them() {
        let mut p = StoredBufferProof::default();
        for _ in 0..1_000 {
            p.note_tick(0.2, D_512_48K);
            p.note_tick(31.0, D_512_48K); // under three chunks (32 ms): not a call
            p.note_tick(f64::NAN, D_512_48K);
            p.note_tick(90.0, f64::NAN);
        }
        assert_eq!(p.median_call_ms(), None, "no call seen");
        feed(&mut p, &[90.0; MIN_CALLS - 1], D_512_48K);
        assert_eq!(p.median_call_ms(), None, "one call short of the evidence");
        feed(&mut p, &[90.0], D_512_48K);
        assert_eq!(p.median_call_ms(), Some(90.0));
        // The window holds the latest calls: an earlier regime drops out.
        feed(&mut p, &[60.0; CALL_WINDOW], D_512_48K);
        assert_eq!(p.median_call_ms(), Some(60.0));
        // A median, not a mean or a maximum: a few long calls do not move it.
        feed(&mut p, &[200.0, 210.0, 220.0, 230.0], D_512_48K);
        assert_eq!(p.median_call_ms(), Some(60.0));
    }

    /// The criterion agrees with the replay of the recorded 120 ms timelines (6 threads, 512 frames at
    /// 48 kHz; replayed through the output queue, not run): paced calls have a median of about 78 ms and
    /// game calls about 98 ms. 60 / 120 played 84 / 71 % (paced / game), Stable's 80 / 150 98 / 88 % and
    /// 100 / 125 at least 98.3 % in every pass. (The simulation test in `playback` pins the same knee
    /// against the output queue itself.)
    #[test]
    fn the_criterion_matches_the_replayed_knee() {
        let (paced, game) = (78.0, 98.0);
        assert!(prefill_cannot_hold_call(60.0, paced), "60 / 120 paced lost 16 %");
        assert!(prefill_cannot_hold_call(60.0, game));
        assert!(!prefill_cannot_hold_call(80.0, paced), "80 / 150 paced played 98 %");
        assert!(prefill_cannot_hold_call(80.0, game), "80 / 150 under the game lost 12 %");
        assert!(!prefill_cannot_hold_call(100.0, paced));
        assert!(!prefill_cannot_hold_call(100.0, game), "100 / 125 played at least 98.3 % everywhere");
        assert!(!prefill_cannot_hold_call(110.0, game));
        assert!(prefill_cannot_hold_call(35.0, 35.1), "the default holds no call longer than itself");
        assert!(!prefill_cannot_hold_call(35.0, 35.0));
    }

    #[test]
    fn the_verdict_latches_per_stored_value_and_resets_with_the_voice() {
        let d = D_512_48K;
        let mut p = StoredBufferProof::default();
        assert!(!p.judge(60.0), "no calls, no evidence: the stored value stands");
        feed(&mut p, &[85.0; 20], d);
        assert!(p.judge(60.0), "a median call of 85 ms against a queue of 60 ms");
        assert!(p.raised());
        // Latched: faster calls later do not flap it back.
        feed(&mut p, &[40.0; CALL_WINDOW], d);
        assert_eq!(p.median_call_ms(), Some(40.0));
        assert!(p.judge(60.0), "latched for the same stored value");
        // The user edits the value: judged anew, on the calls of now.
        assert!(!p.judge(61.0), "a new stored value is judged on the current calls (40 ms)");
        assert!(!p.raised());
        feed(&mut p, &[90.0; CALL_WINDOW], d);
        assert!(p.judge(61.0));
        assert!(!p.judge(100.0), "a value that holds the 90 ms median stands");
        // Another voice: nothing carries over.
        p.reset();
        assert!(!p.raised() && p.median_call_ms().is_none());
        assert!(!p.judge(61.0));
    }
}
