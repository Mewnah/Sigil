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
        // Preemption by parallel tests can only lower it.
        assert!(busy <= Duration::from_millis(165) && busy >= Duration::from_millis(20), "{busy:?}");
        let t1 = thread_cpu_time().unwrap();
        for _ in 0..200 {
            spin(Duration::from_micros(500));
            std::thread::sleep(Duration::from_millis(2));
        }
        let bursts = thread_cpu_time().unwrap() - t1;
        // Preemption by the parallel test suite can only lower it; tick sampling would read about 0
        // or whole 15.6 ms quanta.
        assert!(bursts >= Duration::from_millis(10) && bursts <= Duration::from_millis(160), "{bursts:?}");
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
