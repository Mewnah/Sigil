//! Adaptive quality governor (P3-E): under load it steps quality down (retrieval off, then less
//! context, then the light F0 extractor, then a lighter denoiser) and back up with hysteresis, so
//! the voice degrades gracefully instead of stuttering. It never replaces the latency budget,
//! which still drops stale audio (plan T7); the `governor` lever switches it (off = fixed quality).
//!
//! # Rules
//! The worker reports every converted block ([`TickLoad`]: compute time, block length, the capture
//! backlog after the budget and what the budget dropped at that wake).
//! - **Pressure** (a block is "hot"): the budget dropped input, or the backlog exceeds half the
//!   budget, or compute took more than [`HOT_RTF`] of the block. [`GovernorConfig::step_down_after`]
//!   consecutive hot blocks step one level down. A budget drop steps down at once (audio was lost).
//! - **Comfort** (a block is "calm"): no drop, a backlog below two blocks and compute below
//!   [`CALM_RTF`] of the block. [`GovernorConfig::step_up_after`] consecutive calm blocks step one
//!   level up.
//! - **Hysteresis**: after any step the governor waits [`SETTLE_BLOCKS`] blocks before judging again
//!   (the new level's cost must show first). A level that had to be left again soon after stepping
//!   up to it (within [`RELAPSE_WINDOW_BLOCKS`]) doubles the calm run needed to try it again (up to
//!   [`MAX_UP_BACKOFF`]x), so a PC that is just at the edge does not oscillate; a long calm run
//!   resets the back-off.
//! - A step that changes nothing in the running pipeline (the model and the denoiser report whether
//!   they applied it, [`Governor::observe_with`]) is skipped over, so the governor never "uses up"
//!   a down step on a no-op.
//!
//! # Levels
//! | level | retrieval | context | light F0 | light denoiser |
//! |---|---|---|---|---|
//! | 0 | on | 100 % | no | no |
//! | 1 | off | 100 % | no | no |
//! | 2 | off | 50 % | no | no |
//! | 3 | off | 50 % | yes | no |
//! | 4 | off | 50 % | yes | yes |
//!
//! # Frozen by P3-S0 (signatures; P3-E implements)
//! - [`available`], [`QualityLevel`] (with [`QualityLevel::FULL`]; applied to the model through
//!   `VoiceConversionModel::apply_quality`), [`GovernorConfig`], [`TickLoad`],
//!   [`Governor::new`], [`Governor::observe`] (returns the new level when it changes),
//!   [`Governor::level`], [`Governor::reset`].
//!
//! # Added by P3-E
//! [`GovernorConfig::for_run`], [`Governor::observe_with`], [`QualityLevel::at`],
//! [`QualityLevel::LOWEST`], [`Governor::steps`].

use serde::Serialize;

/// Whether the governor runs in this build (the `governor` lever).
pub fn available() -> bool {
    true
}

/// Compute above this share of the block is pressure.
pub const HOT_RTF: f64 = 0.85;
/// Compute below this share of the block (and no backlog) is comfort.
pub const CALM_RTF: f64 = 0.55;
/// Blocks after a step before the governor judges again.
pub const SETTLE_BLOCKS: u32 = 8;
/// A level left again within this many blocks of stepping up to it counts as a relapse.
pub const RELAPSE_WINDOW_BLOCKS: u32 = 400;
/// Largest multiplier of the calm run needed to step up after relapses.
pub const MAX_UP_BACKOFF: u32 = 16;

/// One quality step. Level 0 is full quality; higher levels are cheaper.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct QualityLevel {
    pub level: u8,
    /// Retrieval (index blend) on.
    pub retrieval: bool,
    /// Share of the preset's extra context in use (1.0 = all of it); the engine maps it onto its
    /// pre-planned shapes.
    pub context_scale: f32,
    /// Use the light F0 extractor (SwiftF0) instead of the configured one.
    pub light_f0: bool,
    /// Use the lighter denoiser (or none).
    pub light_denoiser: bool,
}

impl QualityLevel {
    pub const FULL: QualityLevel = QualityLevel {
        level: 0,
        retrieval: true,
        context_scale: 1.0,
        light_f0: false,
        light_denoiser: false,
    };

    /// The cheapest level.
    pub const LOWEST: u8 = 4;

    /// The level `n` of the table in the module docs (clamped to [`Self::LOWEST`]).
    pub fn at(n: u8) -> QualityLevel {
        let n = n.min(Self::LOWEST);
        QualityLevel {
            level: n,
            retrieval: n < 1,
            context_scale: if n >= 2 { 0.5 } else { 1.0 },
            light_f0: n >= 3,
            light_denoiser: n >= 4,
        }
    }
}

impl Default for QualityLevel {
    fn default() -> Self {
        QualityLevel::FULL
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GovernorConfig {
    /// Latency budget (ms) of the run.
    pub budget_ms: f64,
    /// Block length (ms).
    pub block_ms: f64,
    /// Consecutive over-budget blocks before stepping down.
    pub step_down_after: u32,
    /// Consecutive comfortable blocks before stepping back up.
    pub step_up_after: u32,
}

impl GovernorConfig {
    /// The voice changer's defaults for a run: step down after about 150 ms of pressure (at least
    /// 3 blocks), step up after about 5 s of comfort (at least 50 blocks).
    pub fn for_run(block_ms: f64, budget_ms: f64) -> GovernorConfig {
        let block = if block_ms.is_finite() && block_ms > 0.0 { block_ms } else { 10.0 };
        GovernorConfig {
            budget_ms: if budget_ms.is_finite() && budget_ms > 0.0 { budget_ms } else { 300.0 },
            block_ms: block,
            step_down_after: ((150.0 / block).ceil() as u32).max(3),
            step_up_after: ((5_000.0 / block).ceil() as u32).max(50),
        }
    }
}

/// What one tick cost (worker side, after the block).
#[derive(Clone, Copy, Debug, Default)]
pub struct TickLoad {
    pub compute_ms: f64,
    pub block_ms: f64,
    pub backlog_ms: f64,
    pub dropped_ms: f64,
}

pub struct Governor {
    config: GovernorConfig,
    level: QualityLevel,
    hot_run: u32,
    calm_run: u32,
    /// Blocks to wait before judging again.
    settle: u32,
    /// Blocks since the last step up (for relapse detection), `None` before any.
    since_up: Option<u32>,
    /// Multiplier of `step_up_after` (relapse back-off).
    up_backoff: u32,
    steps: u64,
}

impl Governor {
    pub fn new(config: GovernorConfig) -> Governor {
        Governor {
            config,
            level: QualityLevel::FULL,
            hot_run: 0,
            calm_run: 0,
            settle: 0,
            since_up: None,
            up_backoff: 1,
            steps: 0,
        }
    }

    fn is_hot(&self, t: &TickLoad) -> bool {
        let block = if t.block_ms > 0.0 { t.block_ms } else { self.config.block_ms };
        t.dropped_ms > 0.0 || t.backlog_ms > self.config.budget_ms * 0.5 || t.compute_ms > block * HOT_RTF
    }

    fn is_calm(&self, t: &TickLoad) -> bool {
        let block = if t.block_ms > 0.0 { t.block_ms } else { self.config.block_ms };
        t.dropped_ms <= 0.0 && t.backlog_ms < 2.0 * block && t.compute_ms < block * CALM_RTF
    }

    /// Record one tick; `Some(level)` when the quality level changes.
    pub fn observe(&mut self, tick: &TickLoad) -> Option<QualityLevel> {
        self.observe_with(tick, |_| true)
    }

    /// [`Self::observe`], applying a candidate level through `apply` (whether any stage of the
    /// pipeline changed for it). A down step that changes nothing continues to the next level (up to
    /// [`QualityLevel::LOWEST`]); an up step always lands (levels above are never skipped, so the
    /// pipeline ends up exactly at the level it reports). Returns the level the pipeline now runs,
    /// when it changed.
    pub fn observe_with(&mut self, tick: &TickLoad, mut apply: impl FnMut(&QualityLevel) -> bool) -> Option<QualityLevel> {
        if let Some(n) = self.since_up.as_mut() {
            *n = n.saturating_add(1);
            if *n > RELAPSE_WINDOW_BLOCKS {
                // Stayed up for a long time: the back-off is forgotten.
                self.since_up = None;
                self.up_backoff = 1;
            }
        }
        let hot = self.is_hot(tick);
        let calm = self.is_calm(tick);
        self.hot_run = if hot { self.hot_run.saturating_add(1) } else { 0 };
        self.calm_run = if calm { self.calm_run.saturating_add(1) } else { 0 };
        if self.settle > 0 {
            self.settle -= 1;
            // Audio was lost: do not wait for the settle period to end.
            if !(tick.dropped_ms > 0.0 && self.level.level < QualityLevel::LOWEST) {
                return None;
            }
        }
        let step_down = self.level.level < QualityLevel::LOWEST
            && (tick.dropped_ms > 0.0 || self.hot_run >= self.config.step_down_after.max(1));
        if step_down {
            if self.since_up.is_some_and(|n| n <= RELAPSE_WINDOW_BLOCKS) {
                self.up_backoff = (self.up_backoff * 2).min(MAX_UP_BACKOFF);
            }
            self.since_up = None;
            let mut n = self.level.level;
            let mut landed = None;
            while n < QualityLevel::LOWEST {
                n += 1;
                let candidate = QualityLevel::at(n);
                if apply(&candidate) {
                    landed = Some(candidate);
                    break;
                }
            }
            self.hot_run = 0;
            self.calm_run = 0;
            self.settle = SETTLE_BLOCKS;
            return match landed {
                Some(level) => {
                    self.level = level;
                    self.steps += 1;
                    Some(level)
                }
                None => {
                    // Nothing further changes anything: stay (and stop trying until calm again).
                    self.level = QualityLevel::at(QualityLevel::LOWEST);
                    None
                }
            };
        }
        let up_after = self.config.step_up_after.max(1).saturating_mul(self.up_backoff);
        if self.level.level > 0 && self.calm_run >= up_after {
            let level = QualityLevel::at(self.level.level - 1);
            apply(&level);
            self.level = level;
            self.steps += 1;
            self.calm_run = 0;
            self.hot_run = 0;
            self.settle = SETTLE_BLOCKS;
            self.since_up = Some(0);
            return Some(level);
        }
        None
    }

    pub fn level(&self) -> QualityLevel {
        self.level
    }

    /// The run's latency budget changed (a hot param): the backlog-pressure rule follows it. A value
    /// that is not a positive number is ignored.
    pub fn set_budget_ms(&mut self, budget_ms: f64) {
        if budget_ms.is_finite() && budget_ms > 0.0 {
            self.config.budget_ms = budget_ms;
        }
    }

    /// Level changes so far.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Back to full quality (restart).
    pub fn reset(&mut self) {
        *self = Governor::new(self.config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: f64 = 10.0;

    fn cfg() -> GovernorConfig {
        GovernorConfig::for_run(BLOCK, 300.0)
    }

    fn tick(compute_ms: f64) -> TickLoad {
        TickLoad {
            compute_ms,
            block_ms: BLOCK,
            backlog_ms: 5.0,
            dropped_ms: 0.0,
        }
    }

    #[test]
    fn levels_follow_the_table() {
        assert_eq!(QualityLevel::at(0), QualityLevel::FULL);
        let l1 = QualityLevel::at(1);
        assert!(!l1.retrieval && l1.context_scale == 1.0 && !l1.light_f0 && !l1.light_denoiser);
        let l2 = QualityLevel::at(2);
        assert!(!l2.retrieval && l2.context_scale == 0.5 && !l2.light_f0);
        assert!(QualityLevel::at(3).light_f0 && !QualityLevel::at(3).light_denoiser);
        assert!(QualityLevel::at(4).light_denoiser);
        assert_eq!(QualityLevel::at(200).level, QualityLevel::LOWEST);
        let c = GovernorConfig::for_run(40.0, 300.0);
        assert_eq!((c.step_down_after, c.step_up_after), (4, 125));
        let c = GovernorConfig::for_run(f64::NAN, -1.0);
        assert_eq!((c.block_ms, c.budget_ms), (10.0, 300.0));
    }

    /// Sustained pressure steps down one level per `step_down_after` hot blocks (after settling);
    /// sustained comfort steps back up; a lone spike does nothing.
    #[test]
    fn steps_down_under_load_and_up_when_calm() {
        let mut g = Governor::new(cfg());
        // A single slow block is not pressure enough.
        assert_eq!(g.observe(&tick(9.5)), None);
        for _ in 0..10 {
            assert_eq!(g.observe(&tick(3.0)), None);
        }
        assert_eq!(g.level().level, 0);
        // Sustained: level 1 after `step_down_after` blocks.
        let mut changes = Vec::new();
        for _ in 0..200 {
            if let Some(l) = g.observe(&tick(9.5)) {
                changes.push(l.level);
            }
        }
        assert_eq!(changes, vec![1, 2, 3, 4], "one level at a time, down to the lowest");
        assert_eq!(g.level().level, QualityLevel::LOWEST);
        // Comfort: back up one level per calm run.
        let mut ups = Vec::new();
        for _ in 0..(4 * (cfg().step_up_after + SETTLE_BLOCKS + 2)) {
            if let Some(l) = g.observe(&tick(2.0)) {
                ups.push(l.level);
            }
        }
        assert_eq!(ups, vec![3, 2, 1, 0]);
        assert_eq!(g.level(), QualityLevel::FULL);
        assert_eq!(g.steps(), 8);
    }

    /// Lost audio steps down at once, even while settling from a previous step.
    #[test]
    fn a_budget_drop_steps_down_at_once() {
        let mut g = Governor::new(cfg());
        let drop = TickLoad {
            dropped_ms: 150.0,
            ..tick(3.0)
        };
        assert_eq!(g.observe(&drop).map(|l| l.level), Some(1));
        assert_eq!(g.observe(&drop).map(|l| l.level), Some(2), "not held back by the settle period");
        // A backlog over half the budget is pressure too.
        let mut g = Governor::new(cfg());
        let backlog = TickLoad {
            backlog_ms: 200.0,
            ..tick(1.0)
        };
        let mut level = 0;
        for _ in 0..cfg().step_down_after {
            if let Some(l) = g.observe(&backlog) {
                level = l.level;
            }
        }
        assert_eq!(level, 1);
    }

    /// Steps that change nothing in the pipeline are skipped (the next level that does applies).
    #[test]
    fn no_op_steps_are_skipped() {
        let mut g = Governor::new(cfg());
        let mut tried = Vec::new();
        let drop = TickLoad {
            dropped_ms: 10.0,
            ..tick(3.0)
        };
        // Only the denoiser step (level 4) does anything here.
        let got = g.observe_with(&drop, |l| {
            tried.push(l.level);
            l.light_denoiser
        });
        assert_eq!(got.map(|l| l.level), Some(4));
        assert_eq!(tried, vec![1, 2, 3, 4]);
        // Nothing applies at all: no level change is reported and the governor stops at the bottom.
        let mut g = Governor::new(cfg());
        assert_eq!(g.observe_with(&drop, |_| false), None);
        assert_eq!(g.level().level, QualityLevel::LOWEST);
    }

    /// A PC at the edge: every step up is followed by pressure. The calm run needed to try again
    /// doubles each time, so the level changes ever more rarely instead of oscillating.
    #[test]
    fn relapses_back_off_the_next_step_up() {
        let mut g = Governor::new(cfg());
        let drop = TickLoad {
            dropped_ms: 10.0,
            ..tick(3.0)
        };
        g.observe(&drop);
        assert_eq!(g.level().level, 1);
        let mut up_waits = Vec::new();
        for _ in 0..4 {
            // Calm until it steps up.
            let mut n = 0u32;
            loop {
                n += 1;
                if g.observe(&tick(2.0)).is_some() {
                    break;
                }
                assert!(n < 100_000);
            }
            up_waits.push(n);
            assert_eq!(g.level().level, 0);
            // The full-quality level cannot keep up: back down at once.
            for _ in 0..SETTLE_BLOCKS {
                g.observe(&tick(2.0));
            }
            assert_eq!(g.observe(&drop).map(|l| l.level), Some(1));
        }
        for w in up_waits.windows(2) {
            assert!(w[1] >= 2 * w[0] - SETTLE_BLOCKS - 2, "back-off grows: {up_waits:?}");
        }
        g.reset();
        assert_eq!(g.level(), QualityLevel::FULL);
        assert_eq!(g.steps(), 0);
    }

    /// The latency budget is a hot param: the backlog-pressure rule follows a budget changed while
    /// running (a backlog well inside a raised budget is no pressure; a lowered budget is), and a
    /// value that is not a positive number is ignored.
    #[test]
    fn the_backlog_rule_follows_the_live_budget() {
        let backlog = |ms: f64| TickLoad {
            backlog_ms: ms,
            ..tick(2.0)
        };
        let mut g = Governor::new(GovernorConfig::for_run(BLOCK, 100.0));
        assert!(g.is_hot(&backlog(60.0)), "60 ms of backlog is pressure on a 100 ms budget");
        g.set_budget_ms(2_000.0);
        assert!(!g.is_hot(&backlog(60.0)), "and none on a 2 s budget");
        for _ in 0..1_000 {
            assert!(g.observe(&backlog(300.0)).is_none(), "300 ms of a 2 s budget never steps down");
        }
        assert_eq!(g.level(), QualityLevel::FULL);
        g.set_budget_ms(100.0);
        let steps = (0..100).filter_map(|_| g.observe(&backlog(300.0))).count();
        assert!(steps > 0 && g.level().level > 0, "back to 100 ms: the same backlog is pressure");
        for bad in [0.0, -5.0, f64::NAN, f64::INFINITY] {
            g.set_budget_ms(bad);
            assert_eq!(g.config.budget_ms, 100.0);
        }
    }
}
