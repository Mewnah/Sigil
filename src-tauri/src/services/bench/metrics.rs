//! Benchmark statistics and process meters.
//!
//! - [`Summary`]: count / mean / stddev / min / p50 / p90 / p95 / p99 / p99.9 / max of a sample,
//!   with linearly interpolated percentiles (Hyndman-Fan type 7, NumPy's default), so small
//!   samples give stable, comparable numbers between runs.
//! - [`Metrics`]: the flat `name -> value` map every run reports; thresholds and the perf gate
//!   read it by name (see [`super::report`] for the names).
//! - [`process_cpu_time`] / [`thread_cpu_time`]: CPU time for average cores used. On x86-64
//!   Windows from the cycle counters (`QueryProcessCycleTime` / `QueryThreadCycleTime`, invariant
//!   TSC ticks while a thread runs) over the calibrated TSC rate: exact for bursty real-time work.
//!   `GetProcessTimes` / `GetThreadTimes` only charge the thread running at each scheduler tick
//!   (~15.6 ms), so a worker that computes 0.3 ms and sleeps between ticks can show 0 CPU; they
//!   remain the fallback.
//! - [`process_vram_mb`]: this process's local VRAM over every hardware DXGI adapter
//!   (`IDXGIAdapter3::QueryVideoMemoryInfo`, through `diagnostics::gpu_mem`).
//! - [`process_cycles`] / [`thread_cycles`]: the raw cycle counters (TSC reference cycles while a
//!   thread runs; P3B-A reports them next to the CPU times).
//! - [`paired`] (P3B-A): the paired statistics of interleaved A/B runs (log ratios, a t-interval
//!   and the exact Wilcoxon signed-rank / Hodges-Lehmann interval, the gate's verdict rule). The
//!   perf gate (`scripts/bench/run-perf-gate.ps1`) implements the same arithmetic and checks it
//!   against the golden values of these tests in its `-SelfTest`; P3B-C's compare v2 reuses it.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Flat metric map of one run (stable names; see `report::known_metric`).
pub type Metrics = BTreeMap<String, f64>;

/// Distribution summary of a sample.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub count: usize,
    pub mean: f64,
    pub stddev: f64,
    pub min: f64,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub p999: f64,
    pub max: f64,
}

/// Suffixes a [`Summary`] adds under a metric prefix (`<prefix>_<suffix>`).
pub const SUMMARY_SUFFIXES: [&str; 9] = ["mean", "stddev", "min", "p50", "p90", "p95", "p99", "p999", "max"];

impl Summary {
    /// Summary of the finite values of `values`; `None` when there are none.
    pub fn of(values: &[f64]) -> Option<Summary> {
        let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            return None;
        }
        v.sort_by(f64::total_cmp);
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        // Population standard deviation (the sample is the whole run, not an estimate).
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
        Some(Summary {
            count: v.len(),
            mean,
            stddev: var.max(0.0).sqrt(),
            min: v[0],
            p50: percentile_sorted(&v, 50.0),
            p90: percentile_sorted(&v, 90.0),
            p95: percentile_sorted(&v, 95.0),
            p99: percentile_sorted(&v, 99.0),
            p999: percentile_sorted(&v, 99.9),
            max: v[v.len() - 1],
        })
    }

    /// Value of a suffix from [`SUMMARY_SUFFIXES`].
    pub fn get(&self, suffix: &str) -> Option<f64> {
        Some(match suffix {
            "mean" => self.mean,
            "stddev" => self.stddev,
            "min" => self.min,
            "p50" => self.p50,
            "p90" => self.p90,
            "p95" => self.p95,
            "p99" => self.p99,
            "p999" => self.p999,
            "max" => self.max,
            _ => return None,
        })
    }

    /// Insert `<prefix>_<suffix>` for every suffix.
    pub fn insert_into(&self, prefix: &str, metrics: &mut Metrics) {
        for suffix in SUMMARY_SUFFIXES {
            if let Some(v) = self.get(suffix) {
                metrics.insert(format!("{prefix}_{suffix}"), v);
            }
        }
    }
}

/// Insert the summary of `values` under `prefix` (nothing when the sample is empty).
/// Worker time of the ticks that ran the model (per-block compute of a fixed-block engine).
pub fn call_compute(timeline: &[super::report::TickRecord]) -> Vec<f64> {
    timeline.iter().filter(|t| t.calls > 0).map(|t| t.compute_ms).collect()
}

/// The worker times the real-time budget is judged on (`rtf_*`, `over_budget_ratio`): the ticks
/// that ran the model (a fixed-block engine's accumulating ticks would dilute them), or every tick
/// when none did.
pub fn budget_basis(timeline: &[super::report::TickRecord]) -> Vec<f64> {
    let calls = call_compute(timeline);
    if calls.is_empty() {
        timeline.iter().map(|t| t.compute_ms).collect()
    } else {
        calls
    }
}

pub fn insert_summary(metrics: &mut Metrics, prefix: &str, values: &[f64]) -> Option<Summary> {
    let s = Summary::of(values)?;
    s.insert_into(prefix, metrics);
    Some(s)
}

/// Insert the per-component compute summaries of a template voice built directly
/// (`content_ms_*`, `f0_ms_*`, `enc_ms_*`, `flowdec_ms_*`; nothing when there are none).
pub fn insert_components(metrics: &mut Metrics, components: &[crate::services::neural_audio::vc::engine::TickStats]) {
    if components.is_empty() {
        return;
    }
    let pick = |f: fn(&crate::services::neural_audio::vc::engine::TickStats) -> f64| -> Vec<f64> { components.iter().map(f).collect() };
    insert_summary(metrics, "content_ms", &pick(|t| t.content_ms));
    insert_summary(metrics, "f0_ms", &pick(|t| t.f0_ms));
    insert_summary(metrics, "enc_ms", &pick(|t| t.enc_ms));
    insert_summary(metrics, "flowdec_ms", &pick(|t| t.flowdec_ms));
}

/// Percentile `q` (0-100) of ascending, finite `sorted` data, linearly interpolated between the
/// closest ranks (`h = (n - 1) q / 100`). `NaN` for an empty slice.
pub fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    match sorted.len() {
        0 => f64::NAN,
        1 => sorted[0],
        n => {
            let h = (n - 1) as f64 * (q.clamp(0.0, 100.0) / 100.0);
            let lo = h.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
        }
    }
}

/// Kernel + user CPU time of this process (see the module docs for the source).
pub fn process_cpu_time() -> Option<Duration> {
    imp::process_cpu_time()
}

/// Kernel + user CPU time of the calling thread.
pub fn thread_cpu_time() -> Option<Duration> {
    imp::thread_cpu_time()
}

/// Calibrate the CPU-time source now (one ~25 ms measurement per process), so the first
/// measurement of a run does not pay for it.
pub fn prepare_cpu_meter() {
    let _ = imp::cycles_per_second();
}

/// Raw cycle count of this process (`QueryProcessCycleTime`: reference cycles of the invariant TSC
/// while its threads ran; `None` off Windows).
pub fn process_cycles() -> Option<u64> {
    imp::process_cycles()
}

/// Raw cycle count of the calling thread (`QueryThreadCycleTime`).
pub fn thread_cycles() -> Option<u64> {
    imp::thread_cycles()
}

/// Reference cycles per second of the cycle counters (the TSC rate), when calibrated.
pub fn cycles_per_second() -> Option<f64> {
    imp::cycles_per_second()
}

/// The cycle counters over a run (`details.cycles`): process and worker reference cycles (TSC
/// ticks while their threads ran), per model call, and the TSC rate. They count at the nominal
/// rate whatever the core's real clock, so the effective clock comes from the run state's PDH
/// window (`state.cpu.effective_mhz`).
pub fn cycle_details(before: (Option<u64>, Option<u64>), after: (Option<u64>, Option<u64>), calls: usize) -> serde_json::Value {
    let d = |a: Option<u64>, b: Option<u64>| a.zip(b).map(|(a, b)| b.saturating_sub(a));
    let process = d(before.0, after.0);
    let worker = d(before.1, after.1);
    serde_json::json!({
        "process_ref_cycles": process,
        "worker_ref_cycles": worker,
        "worker_ref_cycles_per_call": worker.filter(|_| calls > 0).map(|w| w as f64 / calls as f64),
        "tsc_hz": cycles_per_second(),
    })
}

/// Average cores used: CPU time over wall time.
pub fn cores(cpu: Duration, wall: Duration) -> f64 {
    if wall.is_zero() {
        0.0
    } else {
        cpu.as_secs_f64() / wall.as_secs_f64()
    }
}

/// This process's local VRAM use summed over every hardware DXGI adapter (MiB), or `None` when
/// DXGI reports nothing (no GPU, non-Windows).
pub fn process_vram_mb() -> Option<u64> {
    let adapters = crate::services::diagnostics::gpu_mem::adapters();
    let mut any = false;
    let mut total = 0u64;
    for a in adapters.iter().filter(|a| !a.software) {
        if let Some(v) = a.vram {
            any = true;
            total += v.process_usage_mb;
        }
    }
    any.then_some(total)
}

/// Peak-tracking VRAM sampler (sampled outside timed regions; DXGI queries take ~0.1 ms).
#[derive(Debug, Clone, Copy, Default)]
pub struct VramPeak {
    pub peak_mb: Option<u64>,
}

impl VramPeak {
    pub fn sample(&mut self) -> Option<u64> {
        let now = process_vram_mb();
        if let Some(mb) = now {
            self.peak_mb = Some(self.peak_mb.map_or(mb, |p| p.max(mb)));
        }
        now
    }
}

#[cfg(windows)]
mod imp {
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};
    use windows::Win32::Foundation::FILETIME;
    use windows::Win32::System::Threading::{GetCurrentProcess, GetCurrentThread, GetProcessTimes, GetThreadTimes};
    use windows::Win32::System::WindowsProgramming::{QueryProcessCycleTime, QueryThreadCycleTime};

    /// TSC ticks per second, measured once against the monotonic clock (x86-64 with an invariant
    /// TSC, which every CPU Windows 11 supports has). `None` elsewhere: then tick-based times.
    pub fn cycles_per_second() -> Option<f64> {
        static HZ: OnceLock<Option<f64>> = OnceLock::new();
        *HZ.get_or_init(|| {
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: RDTSC has no preconditions on x86-64.
                let read = || unsafe { core::arch::x86_64::_rdtsc() };
                let (t0, c0) = (Instant::now(), read());
                std::thread::sleep(Duration::from_millis(25));
                let (c1, dt) = (read(), t0.elapsed());
                let hz = c1.wrapping_sub(c0) as f64 / dt.as_secs_f64();
                // Sanity: 100 MHz .. 20 GHz.
                (1e8..2e10).contains(&hz).then_some(hz)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                None
            }
        })
    }

    fn from_cycles(cycles: u64, hz: f64) -> Duration {
        Duration::from_secs_f64(cycles as f64 / hz)
    }

    fn filetime_100ns(ft: &FILETIME) -> u64 {
        (u64::from(ft.dwHighDateTime) << 32) | u64::from(ft.dwLowDateTime)
    }

    fn total(kernel: &FILETIME, user: &FILETIME) -> Duration {
        Duration::from_nanos((filetime_100ns(kernel) + filetime_100ns(user)).saturating_mul(100))
    }

    pub fn process_cpu_time() -> Option<Duration> {
        if let Some(hz) = cycles_per_second() {
            let mut cycles = 0u64;
            // SAFETY: the current process's pseudo-handle is always valid; `cycles` is a valid
            // out pointer.
            if unsafe { QueryProcessCycleTime(GetCurrentProcess(), &mut cycles) }.is_ok() {
                return Some(from_cycles(cycles, hz));
            }
        }
        process_times()
    }

    pub fn thread_cpu_time() -> Option<Duration> {
        if let Some(hz) = cycles_per_second() {
            let mut cycles = 0u64;
            // SAFETY: as above, for the current thread's pseudo-handle.
            if unsafe { QueryThreadCycleTime(GetCurrentThread(), &mut cycles) }.is_ok() {
                return Some(from_cycles(cycles, hz));
            }
        }
        thread_times()
    }

    pub fn process_cycles() -> Option<u64> {
        let mut cycles = 0u64;
        // SAFETY: the current process's pseudo-handle is always valid; valid out pointer.
        unsafe { QueryProcessCycleTime(GetCurrentProcess(), &mut cycles) }.ok()?;
        Some(cycles)
    }

    pub fn thread_cycles() -> Option<u64> {
        let mut cycles = 0u64;
        // SAFETY: the current thread's pseudo-handle is always valid; valid out pointer.
        unsafe { QueryThreadCycleTime(GetCurrentThread(), &mut cycles) }.ok()?;
        Some(cycles)
    }

    /// Tick-based process CPU time (the fallback).
    pub fn process_times() -> Option<Duration> {
        let (mut c, mut e, mut k, mut u) = (FILETIME::default(), FILETIME::default(), FILETIME::default(), FILETIME::default());
        // SAFETY: the pseudo-handle of the current process is always valid; all out pointers are
        // valid FILETIMEs on this stack frame.
        unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) }.ok()?;
        Some(total(&k, &u))
    }

    /// Tick-based thread CPU time (the fallback).
    pub fn thread_times() -> Option<Duration> {
        let (mut c, mut e, mut k, mut u) = (FILETIME::default(), FILETIME::default(), FILETIME::default(), FILETIME::default());
        // SAFETY: as above, for the current thread's pseudo-handle.
        unsafe { GetThreadTimes(GetCurrentThread(), &mut c, &mut e, &mut k, &mut u) }.ok()?;
        Some(total(&k, &u))
    }
}

#[cfg(not(windows))]
mod imp {
    use std::time::Duration;

    pub fn cycles_per_second() -> Option<f64> {
        None
    }

    pub fn process_cpu_time() -> Option<Duration> {
        None
    }

    pub fn thread_cpu_time() -> Option<Duration> {
        None
    }

    pub fn process_cycles() -> Option<u64> {
        None
    }

    pub fn thread_cycles() -> Option<u64> {
        None
    }
}

// ---------------------------------------------------------------------------------------------
// Paired statistics (P3B-A)
// ---------------------------------------------------------------------------------------------

/// Paired statistics of interleaved A/B runs (the perf gate's arithmetic).
///
/// Every pair `i` gives `r_i = ln(B_i / A_i)` for a lower-is-better metric (`ln(A_i / B_i)` for a
/// higher-is-better one), so `r > 0` is always "B is worse" and ratios combine symmetrically. The
/// interval on the centre of the `r_i`:
/// - `n >= 6`: the exact Wilcoxon signed-rank interval around the Hodges-Lehmann estimate (the
///   order statistics of the Walsh averages; exact coverage of at least 95 %, robust to the single
///   slow run a desktop GPU produces now and then);
/// - `n < 6`: the Student t-interval of the mean (the exact interval cannot reach 95 % there).
///
/// Verdict (plan P3B-A, rule R3): **REGRESSED** when the median pair is worse than
/// `max_regression_pct` and the interval's lower bound is above 0; **INCONCLUSIVE** when the median
/// is worse than the limit but the lower bound is at or below 0 (run more pairs); **IMPROVED** when
/// the upper bound is below 0 and the median pair is at least [`IMPROVED_FLOOR_PCT`] better (a
/// clear but negligible shift, like the A/A's false IMPROVED at -0.87 %, is no evidence);
/// **NEUTRAL** otherwise. No bootstrap: it under-covers at 5-7 pairs.
pub mod paired {
    // The perf gate (PowerShell) runs this arithmetic today; the tests pin it to the gate's golden
    // values, and P3B-C's compare v2 reuses it.
    #![allow(dead_code)]

    use serde::Serialize;

    /// The median improvement (percent) an interval entirely below 0 needs to read IMPROVED
    /// (`-ImprovedFloorPct` of the perf gate).
    pub const IMPROVED_FLOOR_PCT: f64 = 1.0;

    /// The gate's verdict on one metric.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum Verdict {
        Regressed,
        Inconclusive,
        Improved,
        Neutral,
    }

    /// Statistics of the log ratios of one metric.
    #[derive(Clone, Copy, Debug, PartialEq, Serialize)]
    pub struct PairedStats {
        pub n: usize,
        /// Median of `r` and the same as a percent change of B against A (`exp(median) - 1`).
        pub median: f64,
        pub median_pct: f64,
        pub mean: f64,
        /// Sample standard deviation of `r` (n - 1).
        pub sd: f64,
        /// Student t 95 % interval of the mean of `r`.
        pub t_lo: f64,
        pub t_hi: f64,
        /// Hodges-Lehmann estimate and the exact Wilcoxon interval with its coverage (`None` below
        /// 6 pairs).
        pub hl: f64,
        pub w_lo: Option<f64>,
        pub w_hi: Option<f64>,
        pub w_coverage: Option<f64>,
        /// The interval the verdict used (`wilcoxon` or `t`), in `r` units and as percent.
        pub method: &'static str,
        pub lo: f64,
        pub hi: f64,
        pub lo_pct: f64,
        pub hi_pct: f64,
        /// Half the interval's width as a percent of the ratio (the gate's resolution).
        pub half_width_pct: f64,
        pub verdict: Verdict,
    }

    /// `r_i` of pairs `(a_i, b_i)` (see the module docs); `None` when a value is not positive and
    /// finite (a ratio of zeros means nothing: those metrics use absolute deltas instead).
    pub fn log_ratios(a: &[f64], b: &[f64], higher_is_better: bool) -> Option<Vec<f64>> {
        if a.len() != b.len() || a.is_empty() {
            return None;
        }
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                (x.is_finite() && y.is_finite() && x > 0.0 && y > 0.0).then(|| if higher_is_better { (x / y).ln() } else { (y / x).ln() })
            })
            .collect()
    }

    /// Two-sided 95 % quantile of Student's t with `df` degrees of freedom (`t_{0.975, df}`).
    pub fn t975(df: usize) -> f64 {
        const TABLE: [f64; 30] = [
            12.706_205, 4.302_653, 3.182_446, 2.776_445, 2.570_582, 2.446_912, 2.364_624, 2.306_004, 2.262_157, 2.228_139, 2.200_985,
            2.178_813, 2.160_369, 2.144_787, 2.131_450, 2.119_905, 2.109_816, 2.100_922, 2.093_024, 2.085_963, 2.079_614, 2.073_873,
            2.068_658, 2.063_899, 2.059_539, 2.055_529, 2.051_831, 2.048_407, 2.045_230, 2.042_272,
        ];
        match df {
            0 => f64::INFINITY,
            d if d <= 30 => TABLE[d - 1],
            d => {
                // Cornish-Fisher expansion around the normal quantile (error < 1e-4 above 30 df).
                let z = 1.959_963_985_f64;
                let v = d as f64;
                z + (z.powi(3) + z) / (4.0 * v) + (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / (96.0 * v * v)
            }
        }
    }

    /// `P(T+ <= k)` for `k = 0..=n(n+1)/2` of the Wilcoxon signed-rank statistic with `n` pairs
    /// (exact, by counting the subsets of `1..=n` with each rank sum).
    pub fn signed_rank_cdf(n: usize) -> Vec<f64> {
        let m = n * (n + 1) / 2;
        let mut counts = vec![0.0f64; m + 1];
        counts[0] = 1.0;
        for rank in 1..=n {
            for s in (rank..=m).rev() {
                counts[s] += counts[s - rank];
            }
        }
        let total = 2f64.powi(n as i32);
        let mut acc = 0.0;
        counts
            .iter()
            .map(|c| {
                acc += c;
                acc / total
            })
            .collect()
    }

    fn median_of(v: &mut [f64]) -> f64 {
        v.sort_by(f64::total_cmp);
        let n = v.len();
        if n == 0 {
            f64::NAN
        } else if n % 2 == 1 {
            v[n / 2]
        } else {
            0.5 * (v[n / 2 - 1] + v[n / 2])
        }
    }

    /// Sorted Walsh averages `(r_i + r_j) / 2`, `i <= j`.
    fn walsh(r: &[f64]) -> Vec<f64> {
        let mut w = Vec::with_capacity(r.len() * (r.len() + 1) / 2);
        for i in 0..r.len() {
            for j in i..r.len() {
                w.push(0.5 * (r[i] + r[j]));
            }
        }
        w.sort_by(f64::total_cmp);
        w
    }

    /// Hodges-Lehmann estimate: the median of the Walsh averages.
    pub fn hodges_lehmann(r: &[f64]) -> f64 {
        median_of(&mut walsh(r))
    }

    /// The exact Wilcoxon signed-rank interval of level `1 - alpha` (`(lo, hi, coverage)`): the
    /// `c+1`-th smallest and largest Walsh averages, `c` the largest rank sum with
    /// `P(T+ <= c) <= alpha / 2`; `None` when no such `c` exists (fewer than 6 pairs at 95 %).
    pub fn wilcoxon_interval(r: &[f64], alpha: f64) -> Option<(f64, f64, f64)> {
        let n = r.len();
        if n == 0 {
            return None;
        }
        let cdf = signed_rank_cdf(n);
        let c = cdf.iter().take_while(|&&p| p <= alpha / 2.0 + 1e-12).count();
        // `c` values 0..c-1 satisfy the bound; the critical value is c - 1.
        let crit = c.checked_sub(1)?;
        let w = walsh(r);
        let m = w.len();
        let coverage = 1.0 - 2.0 * cdf[crit];
        Some((w[crit], w[m - 1 - crit], coverage))
    }

    /// The statistics and verdict of `r` (see the module docs). `max_regression_pct` is the plan's
    /// 5 %; the improvement floor is [`IMPROVED_FLOOR_PCT`].
    pub fn analyze(r: &[f64], max_regression_pct: f64) -> Option<PairedStats> {
        analyze_with_floor(r, max_regression_pct, IMPROVED_FLOOR_PCT)
    }

    /// [`analyze`] with an explicit improvement floor (percent; 0 = any interval below 0).
    pub fn analyze_with_floor(r: &[f64], max_regression_pct: f64, improved_floor_pct: f64) -> Option<PairedStats> {
        let n = r.len();
        if n == 0 || r.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let mean = r.iter().sum::<f64>() / n as f64;
        let sd = if n > 1 {
            (r.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64).sqrt()
        } else {
            0.0
        };
        let half = if n > 1 { t975(n - 1) * sd / (n as f64).sqrt() } else { f64::INFINITY };
        let (t_lo, t_hi) = (mean - half, mean + half);
        let median = median_of(&mut r.to_vec());
        let hl = hodges_lehmann(r);
        let w = wilcoxon_interval(r, 0.05);
        let (method, lo, hi) = match w {
            Some((lo, hi, _)) => ("wilcoxon", lo, hi),
            None => ("t", t_lo, t_hi),
        };
        let pct = |x: f64| (x.exp() - 1.0) * 100.0;
        let median_pct = pct(median);
        let verdict = if median_pct > max_regression_pct && lo > 0.0 {
            Verdict::Regressed
        } else if median_pct > max_regression_pct {
            Verdict::Inconclusive
        } else if hi < 0.0 && median_pct < -improved_floor_pct {
            Verdict::Improved
        } else {
            Verdict::Neutral
        };
        Some(PairedStats {
            n,
            median,
            median_pct,
            mean,
            sd,
            t_lo,
            t_hi,
            hl,
            w_lo: w.map(|x| x.0),
            w_hi: w.map(|x| x.1),
            w_coverage: w.map(|x| x.2),
            method,
            lo,
            hi,
            lo_pct: pct(lo),
            hi_pct: pct(hi),
            half_width_pct: if lo.is_finite() && hi.is_finite() { 0.5 * (pct(hi) - pct(lo)) } else { f64::INFINITY },
            verdict,
        })
    }

    /// Pairs needed for a t-interval half-width of `target_pct` given the per-pair spread `sd` of
    /// `r` (an A/A run's), within `min..=max`.
    pub fn rounds_for(sd: f64, target_pct: f64, min: usize, max: usize) -> usize {
        let target = (1.0 + target_pct / 100.0).ln();
        if !(sd.is_finite() && sd > 0.0 && target > 0.0) {
            return min.min(max);
        }
        let mut n = min.max(2);
        while n < max && t975(n - 1) * sd / (n as f64).sqrt() > target {
            n += 1;
        }
        n.min(max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_interpolate_between_ranks() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        let s = Summary::of(&v).unwrap();
        assert_eq!(s.count, 100);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 100.0);
        assert!((s.mean - 50.5).abs() < 1e-12);
        // NumPy: np.percentile(np.arange(1, 101), [50, 90, 95, 99, 99.9])
        assert!((s.p50 - 50.5).abs() < 1e-9, "{}", s.p50);
        assert!((s.p90 - 90.1).abs() < 1e-9, "{}", s.p90);
        assert!((s.p95 - 95.05).abs() < 1e-9, "{}", s.p95);
        assert!((s.p99 - 99.01).abs() < 1e-9, "{}", s.p99);
        assert!((s.p999 - 99.901).abs() < 1e-9, "{}", s.p999);
        // Population stddev of 1..=100.
        assert!((s.stddev - 28.866_070_047_722_12).abs() < 1e-9, "{}", s.stddev);
    }

    #[test]
    fn percentiles_ignore_order_and_non_finite_values() {
        let v = [9.0, f64::NAN, 1.0, 5.0, f64::INFINITY, 3.0, 7.0];
        let s = Summary::of(&v).unwrap();
        assert_eq!(s.count, 5);
        assert_eq!(s.p50, 5.0);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 9.0);
        // h = 4 * 0.95 = 3.8 -> 7 + 0.8 * 2
        assert!((s.p95 - 8.6).abs() < 1e-12);
        assert!(Summary::of(&[]).is_none());
        assert!(Summary::of(&[f64::NAN]).is_none());
        let one = Summary::of(&[4.0]).unwrap();
        assert_eq!((one.p50, one.p99, one.stddev), (4.0, 4.0, 0.0));
        assert!(percentile_sorted(&[], 50.0).is_nan());
    }

    #[test]
    fn a_p99_outlier_moves_only_the_tail() {
        // 999 blocks at 2 ms and one 50 ms spike: p50/p95/p99 stay at 2 ms, p99.9 and max see it.
        let mut v = vec![2.0; 999];
        v.push(50.0);
        let s = Summary::of(&v).unwrap();
        assert_eq!((s.p50, s.p95, s.p99), (2.0, 2.0, 2.0));
        assert!(s.p999 > 2.0 && s.p999 < 50.0, "{}", s.p999);
        assert_eq!(s.max, 50.0);
    }

    #[test]
    fn summary_metrics_use_prefixed_names() {
        let mut m = Metrics::new();
        assert!(insert_summary(&mut m, "block_ms", &[]).is_none());
        assert!(m.is_empty());
        insert_summary(&mut m, "block_ms", &[1.0, 2.0, 3.0]).unwrap();
        for suffix in SUMMARY_SUFFIXES {
            assert!(m.contains_key(&format!("block_ms_{suffix}")), "{suffix}");
        }
        assert_eq!(m["block_ms_p50"], 2.0);
    }

    /// Cycle-based CPU time: a 150 ms spin reads as ~150 ms, and 200 bursts of 0.5 ms between
    /// 2 ms sleeps (a real-time worker's pattern, which tick sampling misses) read as ~100 ms.
    #[cfg(all(windows, target_arch = "x86_64"))]
    #[test]
    fn cycle_cpu_time_counts_short_bursts() {
        prepare_cpu_meter();
        assert!(imp::cycles_per_second().is_some());
        let spin = |d: Duration| {
            let s = std::time::Instant::now();
            let mut x = 0.0f64;
            while s.elapsed() < d {
                x = std::hint::black_box(x + 1.0).sqrt();
            }
        };
        let t0 = thread_cpu_time().unwrap();
        spin(Duration::from_millis(150));
        let busy = thread_cpu_time().unwrap() - t0;
        // Preemption by parallel tests (and concurrent builds on a dev rig) can only lower it.
        assert!(busy <= Duration::from_millis(165) && busy >= Duration::from_millis(5), "{busy:?}");
        let t1 = thread_cpu_time().unwrap();
        for _ in 0..200 {
            spin(Duration::from_micros(500));
            std::thread::sleep(Duration::from_millis(2));
        }
        let bursts = thread_cpu_time().unwrap() - t1;
        // Preemption by the parallel test suite can only lower it; tick sampling would read about 0
        // or whole 15.6 ms quanta.
        assert!(bursts >= Duration::from_millis(2) && bursts <= Duration::from_millis(160), "{bursts:?}");
    }

    /// Golden pair sets shared with the perf gate's `-SelfTest` (`run-perf-gate.ps1`): the same
    /// numbers must come out of both implementations.
    mod paired_golden {
        use super::super::paired::*;

        const AA: [f64; 7] = [0.010, -0.020, 0.015, -0.005, 0.000, 0.020, -0.010];
        const REG: [f64; 7] = [0.090, 0.100, 0.110, 0.095, 0.105, 0.098, 0.102];
        const INC: [f64; 7] = [0.080, 0.070, -0.050, 0.060, -0.040, 0.090, 0.065];
        const IMP: [f64; 7] = [-0.100, -0.080, -0.120, -0.090, -0.110, -0.095, -0.105];
        const SMALL: [f64; 5] = [0.08, 0.09, 0.07, 0.085, 0.075];
        /// A clear (every pair below 0) but negligible (-0.5 %) improvement: the same set the
        /// perf gate's self-test checks.
        const TINY: [f64; 7] = [-0.004, -0.006, -0.005, -0.0045, -0.0055, -0.005, -0.0048];

        fn close(a: f64, b: f64) -> bool {
            (a - b).abs() < 1e-8
        }

        #[test]
        fn verdicts_follow_the_rule() {
            let aa = analyze(&AA, 5.0).unwrap();
            assert_eq!((aa.verdict, aa.method), (Verdict::Neutral, "wilcoxon"));
            assert!(close(aa.lo, -0.0125) && close(aa.hi, 0.015) && close(aa.hl, 0.00125), "{aa:?}");
            assert!(close(aa.w_coverage.unwrap(), 0.953125));
            assert!(close(aa.sd, 0.014351041) && close(aa.t_lo, -0.011843929), "{aa:?}");
            let reg = analyze(&REG, 5.0).unwrap();
            assert_eq!(reg.verdict, Verdict::Regressed);
            assert!(close(reg.lo, 0.094) && close(reg.hi, 0.106) && close(reg.median_pct, 10.517091808), "{reg:?}");
            // The median is above +5 % but the interval reaches below 0: more pairs are needed.
            let inc = analyze(&INC, 5.0).unwrap();
            assert_eq!(inc.verdict, Verdict::Inconclusive);
            assert!(close(inc.lo, -0.04) && close(inc.hi, 0.08) && close(inc.median_pct, 6.715902438), "{inc:?}");
            let imp = analyze(&IMP, 5.0).unwrap();
            assert_eq!(imp.verdict, Verdict::Improved);
            assert!(close(imp.lo, -0.1125) && close(imp.hi, -0.0875));
            // Five pairs: the exact interval cannot reach 95 %, the t-interval decides.
            let small = analyze(&SMALL, 5.0).unwrap();
            assert_eq!((small.verdict, small.method, small.w_lo), (Verdict::Regressed, "t", None));
            assert!(close(small.lo, 0.070183785) && close(small.hi, 0.089816215), "{small:?}");
            // The limit applies to the median: at a 12 % limit the same 10 % change is NEUTRAL.
            assert_eq!(analyze(&REG, 12.0).unwrap().verdict, Verdict::Neutral);
            // IMPROVED needs an effect-size floor: an interval entirely below 0 around a -0.5 %
            // median is NEUTRAL (the A/A's false IMPROVED), and reads IMPROVED only without the floor.
            let tiny = analyze(&TINY, 5.0).unwrap();
            assert!(tiny.hi < 0.0 && tiny.median_pct > -IMPROVED_FLOOR_PCT, "{tiny:?}");
            assert_eq!(tiny.verdict, Verdict::Neutral);
            assert_eq!(analyze_with_floor(&TINY, 5.0, 0.0).unwrap().verdict, Verdict::Improved);
            assert_eq!(analyze(&IMP, 5.0).unwrap().verdict, Verdict::Improved, "a -9.5 % shift stays IMPROVED");
            assert!(analyze(&[], 5.0).is_none() && analyze(&[f64::NAN, 0.1], 5.0).is_none());
        }

        #[test]
        fn the_signed_rank_distribution_is_exact() {
            let c7 = signed_rank_cdf(7);
            assert_eq!(c7.len(), 29);
            assert!(close(c7[0], 0.0078125) && close(c7[2], 0.0234375) && close(c7[3], 0.0390625));
            assert!(close(*c7.last().unwrap(), 1.0));
            let c6 = signed_rank_cdf(6);
            assert!(close(c6[0], 0.015625) && close(c6[1], 0.03125));
            // n = 6 is the smallest set with a 95 % exact interval (the whole Walsh range).
            let r6 = [0.1, -0.2, 0.3, 0.05, -0.05, 0.0];
            let (lo, hi, cov) = wilcoxon_interval(&r6, 0.05).unwrap();
            assert!(close(lo, -0.2) && close(hi, 0.3) && close(cov, 0.96875));
            assert!(wilcoxon_interval(&r6[..5], 0.05).is_none());
        }

        #[test]
        fn t_quantiles_and_round_sizing() {
            assert!(close(t975(1), 12.706205) && close(t975(6), 2.446912) && close(t975(30), 2.042272));
            // Above the table: within 1e-3 of the exact 2.0211 (40 df) and 1.9840 (100 df).
            assert!((t975(40) - 2.021_075).abs() < 1e-3 && (t975(100) - 1.983_972).abs() < 1e-3);
            assert!(t975(0).is_infinite());
            // A 6 % per-pair spread needs about 17 pairs for a +-3 % half-width.
            let n = rounds_for(0.06, 3.0, 5, 40);
            assert!((15..=19).contains(&n), "{n}");
            assert_eq!(rounds_for(0.06, 3.0, 5, 10), 10, "capped");
            assert_eq!(rounds_for(0.001, 3.0, 7, 40), 7, "at least the minimum");
            assert_eq!(rounds_for(f64::NAN, 3.0, 7, 40), 7);
        }

        #[test]
        fn log_ratios_point_worse_up() {
            let r = log_ratios(&[10.0, 20.0], &[11.0, 18.0], false).unwrap();
            assert!(close(r[0], (1.1f64).ln()) && r[1] < 0.0);
            let h = log_ratios(&[0.9, 0.9], &[0.8, 0.95], true).unwrap();
            assert!(h[0] > 0.0 && h[1] < 0.0, "a higher-is-better drop is worse");
            assert!(log_ratios(&[0.0], &[1.0], false).is_none());
            assert!(log_ratios(&[1.0], &[1.0, 2.0], false).is_none());
        }
    }

    #[cfg(windows)]
    #[test]
    fn cpu_meters_advance_with_work() {
        let p0 = process_cpu_time().expect("GetProcessTimes");
        let t0 = thread_cpu_time().expect("GetThreadTimes");
        let start = std::time::Instant::now();
        let mut x = 0.0f64;
        while start.elapsed() < Duration::from_millis(60) {
            x = std::hint::black_box(x + 1.0).sqrt();
        }
        let t1 = thread_cpu_time().unwrap();
        let p1 = process_cpu_time().unwrap();
        // Windows accounts CPU time in scheduler quanta (~15.6 ms); 60 ms of spinning shows up.
        assert!(t1 > t0, "{t0:?} -> {t1:?}");
        assert!(p1 >= p0 + (t1 - t0) / 2, "{p0:?} -> {p1:?}");
        assert!(cores(Duration::from_millis(500), Duration::from_secs(1)) == 0.5);
        assert_eq!(cores(Duration::from_millis(5), Duration::ZERO), 0.0);
    }
}
