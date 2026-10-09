//! How a session plan's thread waits for its CUDA work (P3B-G; T4's GPU-preset CPU row and T2).
//!
//! Every CUDA plan run ends in a host wait for its stream. With the driver's default scheduling
//! (`CU_CTX_SCHED_AUTO`: one context and more logical CPUs than contexts) `cuStreamSynchronize`
//! **spins**: the worker burns a core for the whole GPU time of the run, which is why the CUDA rung
//! used 0.40 cores paced against DirectML's 0.12 (DirectML waits on a fence). The plans therefore
//! record an event created with `CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING` after each run's
//! last copy and wait on it in one of three ways ([`WaitMode`], `SIGIL_CUDA_TUNING=wait=...`):
//!
//! | mode | wait | CPU | wake-up |
//! |------|------|-----|---------|
//! | `spin` (legacy) | `cuStreamSynchronize` | the whole GPU time of the run | immediate |
//! | `block` | `cuEventSynchronize` on the blocking event | none while waiting | an OS wake-up (tens of µs) |
//! | `hybrid` | poll the event while the run is expected to finish within the spin budget, then block | at most the budget per run | immediate when the run ends inside the budget |
//!
//! The hybrid wait sizes its spin from the plan's previous runs: it keeps an average of the time
//! from a run's launch to its completion, and spins only when the expected remaining time at the
//! start of the wait is within [`HYBRID_SPIN_US`] (a wait that starts late, after the thread did
//! other work, finishes spinning; a long wait blocks at once). The spin is bounded by twice the
//! budget, then the thread blocks.
//!
//! The waits are allocation-free (hot path; checked by `gpu::tests`).
//!
//! `SIGIL_CUDA_CTX_SCHED=auto|spin|yield|blocking` (P3B-G, a process-wide experiment, never a
//! default) sets the primary context's scheduling flags once per device when the first plan stream
//! retains it, so ONNX Runtime's own syncs (non-graph CUDA sessions) wait that way too. It applies to
//! every CUDA user of the device in the process (CUDA STT / TTS included), so it stays an opt-in
//! until its effect on their p95 is measured.

use std::cell::Cell;
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Instant;

use super::cuda_driver::{ContextScope, CuEvent, CuStream, PrimaryContext, CTX_SCHED_AUTO, CTX_SCHED_BLOCKING_SYNC, CTX_SCHED_SPIN, CTX_SCHED_YIELD};

/// How a plan's thread waits for its CUDA work (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitMode {
    /// `cuStreamSynchronize` (spins under the driver's default scheduling): the P3 behaviour.
    Spin,
    /// Block on a `CU_EVENT_BLOCKING_SYNC` event.
    Block,
    /// Spin while the run is expected to end within [`HYBRID_SPIN_US`], then block.
    Hybrid,
}

impl WaitMode {
    /// The default wait of every CUDA plan: `Hybrid` (P3B-I0, from P3B-G's interleaved gates,
    /// `scripts/bench/specs/p3b/g/README.md`). Against `Spin` (P3's behaviour, still reachable with
    /// `SIGIL_CUDA_TUNING=wait=spin`) the worker's CPU fell on every row (paced 0.629 -> 0.042 cores,
    /// offline 1.306 -> 0.215, under the in-process 70 % load 0.449 -> 0.050, out-of-process load
    /// 0.723 -> 0.047, silent row 0.0088 -> 0.0081) while call p95 / p99 stayed NEUTRAL and the
    /// over-budget ratio stayed 0.
    pub const DEFAULT: WaitMode = WaitMode::Hybrid;

    /// `spin` / `block` / `hybrid` (case-insensitive); `None` for anything else.
    pub fn parse(s: &str) -> Option<WaitMode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "spin" => Some(WaitMode::Spin),
            "block" | "blocking" => Some(WaitMode::Block),
            "hybrid" => Some(WaitMode::Hybrid),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            WaitMode::Spin => "spin",
            WaitMode::Block => "block",
            WaitMode::Hybrid => "hybrid",
        }
    }
}

/// The hybrid wait's spin budget (µs): it spins only when the run is expected to end within this.
/// An OS wake-up after a blocking wait costs tens of microseconds, so waits shorter than about
/// this are cheaper spun than blocked; longer ones block.
pub const HYBRID_SPIN_US: f64 = 250.0;

/// Weight of the newest run in the hybrid wait's running average of launch-to-completion times.
const EMA_WEIGHT: f64 = 0.25;

/// Why a wait failed (a CUDA error code, described by the caller's cold path).
pub type WaitError = i32;

/// One plan's wait state (interior mutability: the plan waits from `&self` accessors too).
/// Allocation-free.
pub struct Waiter {
    mode: WaitMode,
    spin_budget_us: f64,
    /// Running average of launch → observed completion (µs); `None` before the first run.
    expected_us: Cell<Option<f64>>,
    launched: Cell<Option<Instant>>,
}

impl Waiter {
    pub fn new(mode: WaitMode, spin_budget_us: f64) -> Waiter {
        Waiter {
            mode,
            spin_budget_us: if spin_budget_us.is_finite() { spin_budget_us.max(0.0) } else { HYBRID_SPIN_US },
            expected_us: Cell::new(None),
            launched: Cell::new(None),
        }
    }

    pub fn mode(&self) -> WaitMode {
        self.mode
    }

    /// The running average of the plan's launch-to-completion time (µs), once it has run.
    pub fn expected_us(&self) -> Option<f64> {
        self.expected_us.get()
    }

    /// Note that the plan's work was just enqueued.
    #[inline]
    pub fn launched(&self) {
        self.launched.set(Some(Instant::now()));
    }

    /// Wait until `done` (recorded on `stream` after the run's last copy) has completed. `Spin`
    /// synchronises the whole stream (the legacy wait); the others wait on the event. Returns
    /// whether the thread had to wait (`false`: the run had already completed).
    #[inline]
    pub fn wait(&self, scope: &ContextScope<'_>, stream: CuStream, done: CuEvent) -> Result<bool, WaitError> {
        if scope.query_event(done)? {
            self.observe();
            return Ok(false);
        }
        match self.mode {
            WaitMode::Spin => scope.sync(stream)?,
            WaitMode::Block => scope.sync_event(done)?,
            WaitMode::Hybrid => self.hybrid(scope, done)?,
        }
        self.observe();
        Ok(true)
    }

    fn hybrid(&self, scope: &ContextScope<'_>, done: CuEvent) -> Result<(), WaitError> {
        let since = self.launched.get().map(|t| t.elapsed().as_secs_f64() * 1e6);
        let remaining = match (self.expected_us.get(), since) {
            (Some(e), Some(s)) => e - s,
            // No history yet: block (the first runs are load-time runs).
            _ => f64::INFINITY,
        };
        if remaining <= self.spin_budget_us {
            let start = Instant::now();
            let limit = 2.0 * self.spin_budget_us;
            loop {
                if scope.query_event(done)? {
                    return Ok(());
                }
                if start.elapsed().as_secs_f64() * 1e6 > limit {
                    break;
                }
                std::hint::spin_loop();
            }
        }
        scope.sync_event(done)
    }

    /// Fold the run's launch-to-completion time into the running average.
    #[inline]
    fn observe(&self) {
        if let Some(t) = self.launched.take() {
            let us = t.elapsed().as_secs_f64() * 1e6;
            let next = match self.expected_us.get() {
                Some(e) => e + EMA_WEIGHT * (us - e),
                None => us,
            };
            self.expected_us.set(Some(next));
        }
    }
}

/// `SIGIL_CUDA_CTX_SCHED` (see the module docs).
pub const CTX_SCHED_ENV: &str = "SIGIL_CUDA_CTX_SCHED";

/// The scheduling flags `SIGIL_CUDA_CTX_SCHED` asks for (`None`: unset or unknown).
pub fn ctx_sched_from_env() -> Option<u32> {
    parse_ctx_sched(&std::env::var(CTX_SCHED_ENV).ok()?)
}

/// `auto` / `spin` / `yield` / `blocking` → `CTX_SCHED_*`.
pub fn parse_ctx_sched(v: &str) -> Option<u32> {
    match v.trim().to_ascii_lowercase().as_str() {
        "auto" => Some(CTX_SCHED_AUTO),
        "spin" => Some(CTX_SCHED_SPIN),
        "yield" => Some(CTX_SCHED_YIELD),
        "blocking" | "block" | "blocking_sync" => Some(CTX_SCHED_BLOCKING_SYNC),
        _ => None,
    }
}

/// Apply `SIGIL_CUDA_CTX_SCHED` to `ctx`'s device once per process (the first plan stream on it).
/// Logs what the driver reports afterwards. A no-op without the variable.
pub fn apply_ctx_sched_once(ctx: &PrimaryContext) {
    let Some(sched) = ctx_sched_from_env() else {
        return;
    };
    static DONE: OnceLock<Mutex<Vec<i32>>> = OnceLock::new();
    let mut done = DONE.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap_or_else(PoisonError::into_inner);
    if done.contains(&ctx.ordinal()) {
        return;
    }
    done.push(ctx.ordinal());
    match ctx.set_sched_flags(sched) {
        Ok(()) => tracing::info!(
            target: "sigil::gpu",
            "{CTX_SCHED_ENV}: CUDA device {} scheduling flags set to {sched:#x} (reads {:?})",
            ctx.ordinal(),
            ctx.sched_flags()
        ),
        Err(why) => tracing::warn!(target: "sigil::gpu", "{CTX_SCHED_ENV}: CUDA device {}: {why}", ctx.ordinal()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wait_modes_parse_and_default_to_hybrid() {
        assert_eq!(WaitMode::parse("Spin"), Some(WaitMode::Spin));
        assert_eq!(WaitMode::parse(" block "), Some(WaitMode::Block));
        assert_eq!(WaitMode::parse("blocking"), Some(WaitMode::Block));
        assert_eq!(WaitMode::parse("hybrid"), Some(WaitMode::Hybrid));
        assert_eq!(WaitMode::parse("fast"), None);
        for m in [WaitMode::Spin, WaitMode::Block, WaitMode::Hybrid] {
            assert_eq!(WaitMode::parse(m.as_str()), Some(m));
        }
        assert_eq!(WaitMode::DEFAULT, WaitMode::Hybrid);
        let w = Waiter::new(WaitMode::Hybrid, f64::NAN);
        assert_eq!(w.spin_budget_us, HYBRID_SPIN_US);
        assert_eq!(Waiter::new(WaitMode::Block, -5.0).spin_budget_us, 0.0);
    }

    #[test]
    fn the_running_average_follows_observed_runs() {
        let w = Waiter::new(WaitMode::Hybrid, HYBRID_SPIN_US);
        assert!(w.expected_us().is_none());
        w.observe();
        assert!(w.expected_us().is_none(), "nothing launched, nothing observed");
        w.launched();
        w.observe();
        let first = w.expected_us().unwrap();
        assert!((0.0..1e6).contains(&first), "{first}");
        w.launched.set(Some(Instant::now() - std::time::Duration::from_millis(4)));
        w.observe();
        let second = w.expected_us().unwrap();
        // 25 % of the way from the first observation to about 4000 µs.
        assert!(second > first + 0.2 * (4000.0 - first) && second < first + 0.3 * (4100.0 - first), "{first} -> {second}");
    }

    #[test]
    fn ctx_sched_values_parse() {
        assert_eq!(parse_ctx_sched("blocking"), Some(CTX_SCHED_BLOCKING_SYNC));
        assert_eq!(parse_ctx_sched(" Block "), Some(CTX_SCHED_BLOCKING_SYNC));
        assert_eq!(parse_ctx_sched("yield"), Some(CTX_SCHED_YIELD));
        assert_eq!(parse_ctx_sched("spin"), Some(CTX_SCHED_SPIN));
        assert_eq!(parse_ctx_sched("auto"), Some(CTX_SCHED_AUTO));
        assert_eq!(parse_ctx_sched("realtime"), None);
    }
}
