//! Choosing a Sigil-ZS voice's stream configuration from measured ticks (P5-B): the decision the
//! per-PC autotuner makes for this family (the autotuner measures, this module decides), and the
//! output buffer a running voice needs.
//!
//! What P5-B's benchmark showed (RTX 4090, `docs/benchmarks/VC_BENCHMARKS.md` → Sigil-ZS):
//! - A tick is one burst of GPU work per 100 ms block (15-35 ms at full clocks). An otherwise idle
//!   NVIDIA GPU lowers its clocks for such a duty cycle (P3, about 900 MHz instead of 2.5-2.8 GHz),
//!   which stretches the tick to 35-60 ms with spikes to 80-105 ms; a game keeps the clocks up.
//! - The output buffer must hold more audio than the slowest tick takes, or playback runs dry while
//!   the tick computes: the voice changer's default 35 ms prefill underran 145-530 times in 20 s;
//!   80 ms (its Stable template) ran clean except at clock-dip spikes.
//! - `sigil_low` (6 steps, CFG, 4 s prompt) matches upstream's offline speaker similarity;
//!   `sigil_lean` is the lighter fallback; `upstream_readme` (180 ms blocks) adds 80 ms of delay and
//!   is never chosen automatically.
//!
//! - The output arrives in bursts of one block, so a prefill of a whole block or more holds playback
//!   for a second burst: on the RTX 4090 with 512-frame device blocks (`sigil_low` on CUDA, paced,
//!   P5-B fix) the latency read 247.9 ms at 80 and 85 ms of prefill, 257.9 ms at 90 and 95 ms (one
//!   device block more) and 297.9 ms at 100 ms. The prefill stays [`BURST_HEADROOM_MS`] below the
//!   block ([`burst_buffer_ms`]); a slower tick does not keep up anyway ([`keeps_up`]).
//!
//! So: the best preset (in [`PREFERENCE`] order) whose p99 tick fits the block with headroom, on
//! the execution provider with the lowest p99 tick, and an output prefill of that p99 plus a margin
//! (below one burst).
//!
//! Until a running voice's buffer follows its own measured ticks, [`min_output_buffer_ms`] gives
//! the family's minimum per preset and execution provider from the dev rig's idle-clock p99 ticks
//! ([`IDLE_TICK_P99_MS`]). The Stable template's 80 / 150 ms (the minimum P5-I0 shipped for every
//! Sigil-ZS voice) is less than `sigil_low` on CUDA needs: in P5-B's final batch it underran 75
//! times in 20 s on the idle RTX 4090, with a p99 tick of 82 ms (in later idle runs the clocks did
//! not dip and 80 ms ran clean, so the spikes come and go). Its minimum is 95 / 165 ms, which costs
//! one device block (10.7 ms) of latency.
//!
//! Callers (requested from the integration, `families::min_output_buffer_ms` and the voice
//! changer's worker; until then only the tests call this module): the factory's minimum buffer
//! of a Sigil-ZS voice ([`min_output_buffer_ms`] through `sigil_zs::min_output_buffer_ms`),
//! `bench::autotune` (a family stage: measure each preferred preset on each GPU rung, then
//! [`choose`]) and the worker's output buffer for a running zero-shot voice ([`burst_buffer_ms`] of
//! its measured ticks after the warm-up).
#![cfg_attr(not(test), allow(dead_code))]

use super::preset::{self, UNIT_16K};

/// Presets the tuner may choose, best quality first (`upstream_readme` is a manual choice).
pub const PREFERENCE: [&str; 2] = ["sigil_low", "sigil_lean"];
/// The p99 tick must stay under this share of the block (the rest is headroom for a game).
pub const MAX_TICK_SHARE: f64 = 0.85;
/// Output prefill above the p99 tick (scheduling jitter, the device period).
pub const PREFILL_MARGIN_MS: f64 = 15.0;
/// The voice changer's default prefill and its upper bound (`params::clamp_output_buffer_ms`).
pub const PREFILL_MIN_MS: f64 = 35.0;
pub const PREFILL_MAX_MS: f64 = 300.0;
/// Output queue above the prefill (the Stable template's 80 / 150).
pub const QUEUE_ABOVE_PREFILL_MS: f64 = 70.0;
/// The smallest buffer any Sigil-ZS voice plays with: the Stable template (P5-I0).
pub const FAMILY_FLOOR_MS: (f64, f64) = (80.0, 150.0);
/// How far below one burst (block) the prefill stays (see the module docs).
pub const BURST_HEADROOM_MS: f64 = 5.0;

/// p99 tick (ms) per `(preset, EP)` at the preset's defaults on the dev rig with the GPU otherwise
/// idle (its clocks dip between the bursts): RTX 4090, P5-B's final benchmark batch, paced 20 s,
/// the slowest of the runs per row (`docs/benchmarks/VC_BENCHMARKS.md` → Sigil-ZS).
pub const IDLE_TICK_P99_MS: [(&str, &str, f64); 6] = [
    ("sigil_low", "cuda", 82.3),
    ("sigil_low", "dml", 65.5),
    ("sigil_lean", "cuda", 61.6),
    ("sigil_lean", "dml", 61.6),
    ("upstream_readme", "cuda", 105.0),
    ("upstream_readme", "dml", 106.8),
];

/// The smallest output buffer `(prefill, max queue)` (ms) a voice of `preset` on `ep` plays with:
/// [`burst_buffer_ms`] of its [`IDLE_TICK_P99_MS`] row (the slowest measured EP's for an EP
/// without a row, `sigil_low`'s for an unknown preset) and its block, never below
/// [`FAMILY_FLOOR_MS`].
pub fn min_output_buffer_ms(preset: &str, ep: &str) -> (f64, f64) {
    let preset = if IDLE_TICK_P99_MS.iter().any(|(p, _, _)| *p == preset) { preset } else { PREFERENCE[0] };
    let rows = || IDLE_TICK_P99_MS.iter().filter(move |(p, _, _)| *p == preset);
    let p99 = match rows().find(|(_, e, _)| *e == ep) {
        Some(&(_, _, t)) => t,
        None => rows().map(|&(_, _, t)| t).fold(0.0, f64::max),
    };
    let block_ms = (preset::preset(preset).block_units * UNIT_16K) as f64 / 16.0;
    let (prefill, queue) = burst_buffer_ms(p99, block_ms);
    (prefill.max(FAMILY_FLOOR_MS.0), queue.max(FAMILY_FLOOR_MS.1))
}

/// [`output_buffer_ms`] for a voice whose output arrives in bursts of `block_ms`: the prefill stays
/// [`BURST_HEADROOM_MS`] below one burst (never below the voice changer's default), because a
/// prefill of a whole burst holds playback for the next one (see the module docs).
pub fn burst_buffer_ms(tick_ms_p99: f64, block_ms: f64) -> (f64, f64) {
    let (prefill, _) = output_buffer_ms(tick_ms_p99);
    let below_burst = if block_ms.is_finite() { (block_ms - BURST_HEADROOM_MS).max(PREFILL_MIN_MS) } else { PREFILL_MAX_MS };
    let prefill = prefill.min(below_burst);
    (prefill, prefill + QUEUE_ABOVE_PREFILL_MS)
}

/// One measured configuration: a preset on an execution provider, its tick times on the paced
/// (wall) clock of this PC (ms).
#[derive(Clone, Debug, PartialEq)]
pub struct TickMeasurement {
    pub preset: String,
    pub ep: String,
    pub block_ms: f64,
    pub tick_ms_p50: f64,
    pub tick_ms_p99: f64,
}

/// What the voice should run with.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamChoice {
    pub preset: String,
    pub ep: String,
    pub output_prefill_ms: f64,
    pub output_max_queue_ms: f64,
    /// Why the best-quality preset was not chosen, when it was not.
    pub notes: Vec<String>,
}

/// The output prefill a voice whose p99 tick is `tick_ms_p99` needs: the tick plus
/// [`PREFILL_MARGIN_MS`], rounded up to 5 ms, within the voice changer's range (never below its
/// default). Returns `(prefill, max_queue)`.
pub fn output_buffer_ms(tick_ms_p99: f64) -> (f64, f64) {
    let need = if tick_ms_p99.is_finite() { tick_ms_p99.max(0.0) + PREFILL_MARGIN_MS } else { PREFILL_MAX_MS };
    let prefill = ((need / 5.0).ceil() * 5.0).clamp(PREFILL_MIN_MS, PREFILL_MAX_MS);
    (prefill, prefill + QUEUE_ABOVE_PREFILL_MS)
}

/// Whether a measurement keeps up with its block with headroom.
pub fn keeps_up(m: &TickMeasurement) -> bool {
    m.block_ms > 0.0 && m.tick_ms_p99.is_finite() && m.tick_ms_p99 <= MAX_TICK_SHARE * m.block_ms
}

/// The configuration to run (see the module docs); `None` when no preferred preset keeps up on any
/// measured execution provider (the PC is too slow for live zero-shot conversion).
pub fn choose(measurements: &[TickMeasurement]) -> Option<StreamChoice> {
    let mut notes = Vec::new();
    for preset in PREFERENCE {
        let best = measurements
            .iter()
            .filter(|m| m.preset == preset && keeps_up(m))
            .min_by(|a, b| a.tick_ms_p99.total_cmp(&b.tick_ms_p99));
        match best {
            Some(m) => {
                let (output_prefill_ms, output_max_queue_ms) = burst_buffer_ms(m.tick_ms_p99, m.block_ms);
                return Some(StreamChoice { preset: preset.to_string(), ep: m.ep.clone(), output_prefill_ms, output_max_queue_ms, notes });
            }
            None => {
                let fastest = measurements.iter().filter(|m| m.preset == preset).map(|m| m.tick_ms_p99).min_by(f64::total_cmp);
                notes.push(match fastest {
                    Some(t) => format!("{preset}: the p99 tick ({t:.0} ms) does not fit {:.0} % of the block", MAX_TICK_SHARE * 100.0),
                    None => format!("{preset}: not measured"),
                });
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(preset: &str, ep: &str, p99: f64) -> TickMeasurement {
        TickMeasurement { preset: preset.into(), ep: ep.into(), block_ms: 100.0, tick_ms_p50: p99 * 0.8, tick_ms_p99: p99 }
    }

    #[test]
    fn the_buffer_covers_the_slowest_tick() {
        assert_eq!(output_buffer_ms(10.0), (35.0, 105.0), "never below the default");
        assert_eq!(output_buffer_ms(65.5), (85.0, 155.0));
        assert_eq!(output_buffer_ms(82.3), (100.0, 170.0));
        assert_eq!(output_buffer_ms(500.0), (300.0, 370.0));
        assert_eq!(output_buffer_ms(f64::NAN), (300.0, 370.0));
    }

    /// The family's minimum buffer covers each preset's idle-clock p99 tick on the dev rig plus the
    /// margin, never below the Stable template, and stays a valid voice-changer buffer.
    /// Below one burst: a 100 ms prefill for 100 ms blocks cost 50 ms of latency on the dev rig,
    /// 95 ms cost one device block.
    #[test]
    fn burst_buffers_stay_below_one_block() {
        assert_eq!(burst_buffer_ms(82.3, 100.0), (95.0, 165.0), "not 100: a whole burst");
        assert_eq!(burst_buffer_ms(65.5, 100.0), (85.0, 155.0));
        assert_eq!(burst_buffer_ms(105.0, 180.0), (120.0, 190.0));
        assert_eq!(burst_buffer_ms(10.0, 100.0), (35.0, 105.0), "never below the default");
        assert_eq!(burst_buffer_ms(500.0, 100.0), (95.0, 165.0), "a tick that slow does not keep up anyway");
        assert_eq!(burst_buffer_ms(f64::NAN, 100.0), (95.0, 165.0));
        assert_eq!(burst_buffer_ms(50.0, 20.0), (35.0, 105.0), "the default wins over a tiny block");
        assert_eq!(burst_buffer_ms(82.3, f64::INFINITY), output_buffer_ms(82.3));
    }

    #[test]
    fn the_minimum_buffer_covers_each_presets_idle_ticks() {
        assert_eq!(min_output_buffer_ms("sigil_low", "cuda"), (95.0, 165.0), "80 / 150 ms underran 75 times in 20 s");
        assert_eq!(min_output_buffer_ms("sigil_low", "dml"), (85.0, 155.0));
        assert_eq!(min_output_buffer_ms("sigil_lean", "cuda"), (80.0, 150.0));
        assert_eq!(min_output_buffer_ms("sigil_lean", "dml"), (80.0, 150.0));
        assert_eq!(min_output_buffer_ms("upstream_readme", "cuda"), (120.0, 190.0), "180 ms blocks");
        assert_eq!(min_output_buffer_ms("upstream_readme", "dml"), (125.0, 195.0));
        assert_eq!(min_output_buffer_ms("sigil_low", "cpu"), (95.0, 165.0), "no row: the slowest measured EP");
        assert_eq!(min_output_buffer_ms("bogus", "dml"), min_output_buffer_ms("sigil_low", "dml"));
        for (preset, ep, p99) in IDLE_TICK_P99_MS {
            let (prefill, queue) = min_output_buffer_ms(preset, ep);
            let block_ms = (preset::preset(preset).block_units * UNIT_16K) as f64 / 16.0;
            // The p99 tick plus the margin, or as much of it as fits below one burst.
            let want = (p99 + PREFILL_MARGIN_MS).min(block_ms - BURST_HEADROOM_MS);
            assert!(prefill >= want && prefill >= FAMILY_FLOOR_MS.0 && prefill < block_ms, "{preset} {ep}");
            assert!(prefill > p99, "{preset} {ep}: the prefill covers the p99 tick");
            // `params::clamp_output_buffer_ms`: prefill 5-300 ms, queue at least prefill + 5 ms.
            assert!(prefill <= PREFILL_MAX_MS && queue >= prefill + 5.0 && queue <= 600.0, "{preset} {ep}");
        }
        for preset in super::super::PRESETS {
            for ep in ["cuda", "dml"] {
                assert!(IDLE_TICK_P99_MS.iter().any(|(p, e, _)| *p == preset && *e == ep), "{preset} {ep} measured");
            }
        }
    }

    /// P5-B's dev-rig measurements (paced, idle GPU): `sigil_low` keeps up on both EPs, and the one
    /// with the lower p99 tick wins.
    #[test]
    fn the_best_preset_that_keeps_up_on_its_fastest_ep() {
        let rig = [m("sigil_low", "cuda", 82.3), m("sigil_low", "dml", 65.5), m("sigil_lean", "cuda", 61.6), m("sigil_lean", "dml", 61.6)];
        let c = choose(&rig).unwrap();
        assert_eq!((c.preset.as_str(), c.ep.as_str(), c.output_prefill_ms), ("sigil_low", "dml", 85.0));
        assert!(c.notes.is_empty());
        // A slower PC: `sigil_low` does not fit, `sigil_lean` does (noted why).
        let slow = [m("sigil_low", "cuda", 95.0), m("sigil_lean", "cuda", 70.0), m("upstream_readme", "cuda", 60.0)];
        let c = choose(&slow).unwrap();
        assert_eq!((c.preset.as_str(), c.output_prefill_ms), ("sigil_lean", 85.0));
        assert!(c.notes[0].starts_with("sigil_low: the p99 tick (95 ms)"), "{:?}", c.notes);
        // Too slow for either (and `upstream_readme` is never picked automatically).
        assert_eq!(choose(&[m("sigil_lean", "dml", 120.0), m("upstream_readme", "dml", 50.0)]), None);
        assert_eq!(choose(&[]), None);
    }
}
