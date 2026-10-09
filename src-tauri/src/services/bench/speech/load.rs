//! What surrounds a timing run: the load simulator (the bench's contention profiles), the GPU timing
//! lock check, and the CPU / VRAM / RAM meters.
//!
//! - [`resolve_load`] turns a run's `load_profile` and `load` into the bench's [`LoadSpec`]: the
//!   profile's parts come from `bench::contention` (never mirrored here), and a `load` member replaces
//!   the profile's GPU part and / or CPU part individually (`custom` and `none` have no parts of their
//!   own).
//! - [`check_lock`] verifies, never takes, the dev rig's GPU timing lock (`scripts/bench/gpu-lock.ps1`):
//!   the lock file must exist and name the run's tag. The campaign takes the lock with
//!   `gpu-lock.ps1 -Run P7-A -Command "..."`.
//! - [`Window`] measures process CPU time over a stretch of work with the load simulator's own CPU
//!   (its running tally) subtracted; [`Vram`] samples this process's VRAM through DXGI and, for engines
//!   that run in a helper process, the device-wide usage through NVML.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::Value;

use super::report::LockBlock;
use super::spec::{LoadProfile, RunDef};
use crate::services::bench::contention;
use crate::services::bench::load_sim::LoadCpu;
use crate::services::bench::metrics;
use crate::services::bench::spec::{ContentionRun, LoadSpec};

// ---------------------------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------------------------

/// The load a run applies, and where it came from.
#[derive(Clone, Debug)]
pub struct ResolvedLoad {
    pub profile: LoadProfile,
    pub spec: LoadSpec,
}

impl ResolvedLoad {
    pub fn is_idle(&self) -> bool {
        self.spec.gpu.is_none() && self.spec.cpu.is_none()
    }
}

/// The profile's load on `gpu_ep` (`cuda` / `dml` / `cpu`), through the bench's own table.
fn profile_load(profile: LoadProfile, gpu_ep: &str) -> Result<LoadSpec, String> {
    if matches!(profile, LoadProfile::None | LoadProfile::Custom) {
        return Ok(LoadSpec::default());
    }
    let run: ContentionRun = serde_json::from_value(serde_json::json!({
        "name": "speech-bench",
        "profile": profile.as_str(),
        "seconds": 60.0,
    }))
    .map_err(|e| format!("load profile {}: {e}", profile.as_str()))?;
    // The D3D12 profiles run on DirectML whatever the engines use (a game renders through D3D12).
    let ep = if matches!(profile, LoadProfile::H2Load70Dml | LoadProfile::H2Load70Oop) { "dml" } else { gpu_ep };
    Ok(contention::expand(&run, ep).load)
}

/// The load of `run`. `gpu_ep` is the EP the profile's GPU load runs on when the profile does not
/// force one: the engines' own GPU rung (`cuda`), else DirectML.
pub fn resolve_load(run: &RunDef, gpu_ep: &str) -> Result<ResolvedLoad, String> {
    let mut spec = profile_load(run.load_profile, gpu_ep)?;
    if let Some(over) = &run.load {
        if over.gpu.is_some() {
            spec.gpu = over.gpu.clone();
        }
        if over.cpu.is_some() {
            spec.cpu = over.cpu.clone();
        }
    }
    if run.load_profile == LoadProfile::Custom && spec.gpu.is_none() && spec.cpu.is_none() {
        return Err(format!("run {}: load_profile custom needs a load (gpu and / or cpu)", run.name));
    }
    if let Some(cpu) = &spec.cpu {
        if cpu.duty <= 0.0 || cpu.duty > 1.0 || cpu.period_ms <= 0.0 {
            return Err(format!("run {}: load.cpu needs 0 < duty <= 1 and a positive period_ms", run.name));
        }
    }
    if let Some(gpu) = &spec.gpu {
        if gpu.duty <= 0.0 || gpu.duty > 1.0 || gpu.period_ms <= 0.0 {
            return Err(format!("run {}: load.gpu needs 0 < duty <= 1 and a positive period_ms", run.name));
        }
    }
    Ok(ResolvedLoad {
        profile: run.load_profile,
        spec,
    })
}

// ---------------------------------------------------------------------------------------------
// GPU timing lock
// ---------------------------------------------------------------------------------------------

/// The lock file's owner: the `tag` of a JSON lock, else the (trimmed, 80 character) text of a lock
/// written by hand: the same reading as `gpu-lock.ps1`'s `Read-Lock`.
pub fn lock_owner(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let owner = match serde_json::from_str::<Value>(trimmed) {
        Ok(Value::Object(map)) => map.get("tag").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| trimmed.to_string()),
        _ => trimmed.to_string(),
    };
    Some(owner.chars().take(80).collect())
}

/// Whether `owner` is `tag`: equal, or a hand-written lock that starts with the tag and a space
/// (`P7-A 18:02`).
pub fn owner_matches(owner: &str, tag: &str) -> bool {
    owner == tag || owner.strip_prefix(tag).is_some_and(|rest| rest.starts_with(char::is_whitespace))
}

/// `$DefaultLockPath` of `gpu-lock.ps1`, read from the script (searched from `start` upwards, then
/// from the working directory), so the runner and the script agree without a copy of the path.
pub fn default_lock_path(start: &Path) -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = start.ancestors().map(Path::to_path_buf).collect();
    if let Ok(cwd) = std::env::current_dir() {
        roots.extend(cwd.ancestors().map(Path::to_path_buf));
    }
    for root in roots {
        let script = root.join("scripts").join("bench").join("gpu-lock.ps1");
        if let Ok(text) = std::fs::read_to_string(&script) {
            if let Some(path) = parse_default_lock_path(&text) {
                return Some(path);
            }
        }
    }
    None
}

fn parse_default_lock_path(script: &str) -> Option<PathBuf> {
    script.lines().find_map(|line| {
        let rest = line.trim().strip_prefix("$DefaultLockPath")?.trim_start().strip_prefix('=')?.trim();
        let quote = rest.chars().next().filter(|c| matches!(c, '\'' | '"'))?;
        let inner = &rest[1..];
        inner.find(quote).map(|end| PathBuf::from(&inner[..end]))
    })
}

/// Check the lock for `run`. The block is `None` when the run neither requires the lock nor is a speed
/// run; otherwise it records the owner found. `Err` when the lock is required and not held by the tag.
pub fn check_lock(run: &RunDef, expanded_path: Option<PathBuf>, spec_dir: &Path) -> (Option<LockBlock>, Result<(), String>) {
    let required = run.lock_required();
    if !required && run.gpu_lock.is_none() && !run.mode.is_speed() {
        return (None, Ok(()));
    }
    let tag = run.lock_tag();
    let path = expanded_path
        .or_else(|| std::env::var_os("SIGIL_GPU_LOCK").filter(|v| !v.is_empty()).map(PathBuf::from))
        .or_else(|| default_lock_path(spec_dir));
    let (held_by, problem) = match &path {
        None => (None, Some("no GPU lock path: set gpu_lock.path, $env:SIGIL_GPU_LOCK, or run from the repository (scripts/bench/gpu-lock.ps1)".to_string())),
        Some(p) => match std::fs::read_to_string(p) {
            Ok(text) => match lock_owner(&text) {
                Some(owner) if owner_matches(&owner, &tag) => (Some(owner), None),
                Some(owner) => {
                    let msg = format!("the GPU timing lock is held by '{owner}', not '{tag}'");
                    (Some(owner), Some(msg))
                }
                None => (None, Some("the GPU timing lock file is empty".to_string())),
            },
            Err(_) => (None, Some(format!("the GPU timing lock is not held (no lock file named '{tag}'); run the spec under gpu-lock.ps1 -Run {tag}"))),
        },
    };
    let verified = problem.is_none();
    let block = LockBlock {
        required,
        tag: tag.clone(),
        held_by,
        verified,
    };
    let result = match problem {
        Some(p) if required => Err(format!("{p} (gpu_lock.require)")),
        _ => Ok(()),
    };
    (Some(block), result)
}

// ---------------------------------------------------------------------------------------------
// Meters
// ---------------------------------------------------------------------------------------------

/// A stretch of work measured for wall time and process CPU time, minus the load simulator's CPU.
#[derive(Clone, Debug)]
pub struct Window {
    started: Instant,
    cpu: Option<Duration>,
    load: Option<(LoadCpu, f64)>,
}

impl Window {
    /// Start now; `tally` is the running load simulator's CPU tally, if one runs.
    pub fn start(tally: Option<&LoadCpu>) -> Window {
        Window {
            started: Instant::now(),
            cpu: metrics::process_cpu_time(),
            load: tally.map(|t| (t.clone(), t.ms())),
        }
    }

    pub fn wall(&self) -> Duration {
        self.started.elapsed()
    }

    /// Process CPU time since the start, the load's own subtracted (never negative); `None` where the
    /// platform has no CPU clock.
    pub fn cpu_ms(&self) -> Option<f64> {
        let now = metrics::process_cpu_time()?;
        let spent = now.checked_sub(self.cpu?)?.as_secs_f64() * 1000.0;
        let load = self.load.as_ref().map_or(0.0, |(t, at_start)| (t.ms() - at_start).max(0.0));
        Some((spent - load).max(0.0))
    }

    /// Average cores used so far (`cpu / wall`).
    pub fn cores(&self) -> Option<f64> {
        let wall = self.wall().as_secs_f64();
        self.cpu_ms().filter(|_| wall > 0.0).map(|ms| ms / 1000.0 / wall)
    }
}

/// VRAM and RAM peaks of one engine.
#[derive(Clone, Debug, Default)]
pub struct Vram {
    /// Sample VRAM at all (a CPU-only run never touches the GPU APIs).
    pub enabled: bool,
    /// The engine runs in a helper process: use the device-wide usage (NVML) against the baseline.
    pub device_wide: bool,
    baseline_mb: Option<u64>,
    peak_mb: Option<u64>,
    /// RAM (working set) peak, MiB.
    pub ram_peak_mb: Option<f64>,
}

impl Vram {
    pub fn new(enabled: bool, device_wide: bool) -> Vram {
        Vram {
            enabled,
            device_wide,
            ..Vram::default()
        }
    }

    fn read(&self) -> Option<u64> {
        if !self.enabled {
            return None;
        }
        if self.device_wide {
            crate::services::sysinfo::nvml::shared(false).ok().and_then(|n| n.memory(0)).map(|m| m.used_mb)
        } else {
            metrics::process_vram_mb()
        }
    }

    /// Take the "before the engine loaded" reading.
    pub fn mark_baseline(&mut self) {
        self.baseline_mb = self.read();
        self.peak_mb = self.baseline_mb;
    }

    /// Sample (outside timed regions).
    pub fn sample(&mut self) -> Option<u64> {
        let now = self.read();
        if let Some(v) = now {
            self.peak_mb = Some(self.peak_mb.map_or(v, |p| p.max(v)));
        }
        if let Some(m) = crate::services::speech_rt::sherpa::probe::memory_sample() {
            self.ram_peak_mb = Some(self.ram_peak_mb.map_or(m.rss_mb, |p| p.max(m.rss_mb)));
        }
        now
    }

    /// Current reading minus the baseline (VRAM the engine added right now).
    pub fn delta_now(&self) -> Option<f64> {
        let (now, base) = (self.read()?, self.baseline_mb?);
        Some(now.saturating_sub(base) as f64)
    }

    /// `(vram_peak_mb, vram_delta_mb)`: for a helper process the peak is the device-wide growth over
    /// the baseline (the helper's own usage), for this process the absolute process peak.
    pub fn peaks(&self) -> (Option<f64>, Option<f64>) {
        match (self.peak_mb, self.baseline_mb) {
            (Some(p), Some(b)) => {
                let delta = p.saturating_sub(b) as f64;
                (Some(if self.device_wide { delta } else { p as f64 }), Some(delta))
            }
            _ => (None, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::bench::spec::{CpuLoadSpec, GpuLoadSpec};

    fn run(json: serde_json::Value) -> RunDef {
        let mut base = serde_json::json!({"name": "r", "mode": "cpu_speed"});
        base.as_object_mut().unwrap().extend(json.as_object().unwrap().clone());
        serde_json::from_value(base).unwrap()
    }

    #[test]
    fn profiles_expand_through_the_bench_table() {
        let none = resolve_load(&run(serde_json::json!({})), "cuda").unwrap();
        assert!(none.is_idle());
        let h3 = resolve_load(&run(serde_json::json!({"load_profile": "h3_cpu_game"})), "cuda").unwrap();
        let cpu = h3.spec.cpu.as_ref().unwrap();
        assert_eq!((cpu.threads, cpu.duty), (4, 0.6));
        assert!(h3.spec.gpu.is_none());
        let h1 = resolve_load(&run(serde_json::json!({"load_profile": "h1_vrchat_obs"})), "cuda").unwrap();
        let g = h1.spec.gpu.as_ref().unwrap();
        assert_eq!((g.duty, g.execution_provider.as_str(), g.size, g.depth, g.out_of_process), (0.7, "cuda", 256, 8, false));
        assert_eq!(h1.spec.cpu.as_ref().unwrap().threads, 3);
        // The D3D12 profiles ignore the engines' EP.
        let oop = resolve_load(&run(serde_json::json!({"load_profile": "h2_load70_oop"})), "cuda").unwrap();
        let g = oop.spec.gpu.as_ref().unwrap();
        assert_eq!((g.execution_provider.as_str(), g.out_of_process), ("dml", true));
    }

    #[test]
    fn a_load_member_replaces_the_profile_part_by_part() {
        let r = run(serde_json::json!({
            "load_profile": "h1_vrchat_obs",
            "load": {"gpu": {"duty": 0.7, "period_ms": 16.7, "execution_provider": "dml", "size": 256, "depth": 8, "out_of_process": true}}
        }));
        let l = resolve_load(&r, "cuda").unwrap();
        let g = l.spec.gpu.as_ref().unwrap();
        assert!(g.out_of_process && g.execution_provider == "dml" && g.period_ms == 16.7);
        assert_eq!(l.spec.cpu.as_ref().unwrap().threads, 3, "the profile's CPU burners stay");
        // Profile none with a load: the load applies.
        let l = resolve_load(&run(serde_json::json!({"load": {"cpu": {"threads": 2, "duty": 0.5}}})), "dml").unwrap();
        assert_eq!(l.spec.cpu.as_ref().unwrap().threads, 2);
        // custom needs something.
        assert!(resolve_load(&run(serde_json::json!({"load_profile": "custom"})), "dml").is_err());
        assert!(resolve_load(&run(serde_json::json!({"load": {"cpu": {"threads": 1, "duty": 1.5}}})), "dml").is_err());
        let g = GpuLoadSpec {
            duty: 0.0,
            period_ms: 20.0,
            execution_provider: "dml".into(),
            device_index: 0,
            size: 8,
            depth: 1,
            out_of_process: false,
        };
        let bad = RunDef {
            load: Some(LoadSpec { gpu: Some(g), cpu: None }),
            ..run(serde_json::json!({}))
        };
        assert!(resolve_load(&bad, "dml").is_err());
        let _ = CpuLoadSpec { threads: 1, duty: 1.0, period_ms: 10.0 };
    }

    #[test]
    fn lock_owners_read_json_and_hand_written_locks() {
        assert_eq!(lock_owner(r#"{"tag":"P7-A","pid":12,"start_unix_ms":1}"#).as_deref(), Some("P7-A"));
        assert_eq!(lock_owner("P7-A 18:02\n").as_deref(), Some("P7-A 18:02"));
        assert_eq!(lock_owner("   "), None);
        assert!(owner_matches("P7-A", "P7-A"));
        assert!(owner_matches("P7-A 18:02", "P7-A"));
        assert!(!owner_matches("P7-A2", "P7-A"), "a tag is not a prefix match");
        assert!(!owner_matches("P3B-A", "P7-A"));
    }

    #[test]
    fn the_script_default_path_is_read_from_gpu_lock_ps1() {
        let script = "param()\n$DefaultLockPath = 'C:\\x\\gpu_timing.lock'\n$Other = 1\n";
        assert_eq!(parse_default_lock_path(script), Some(PathBuf::from("C:\\x\\gpu_timing.lock")));
        assert_eq!(parse_default_lock_path("$DefaultLockPath = \"/tmp/l\""), Some(PathBuf::from("/tmp/l")));
        assert_eq!(parse_default_lock_path("# nothing"), None);
    }

    #[test]
    fn the_lock_check_requires_the_tag_for_speed_runs_only() {
        let dir = std::env::temp_dir().join(format!("sigil-speech-lock-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let lock = dir.join("gpu.lock");
        let _ = std::fs::remove_file(&lock);
        let speed = run(serde_json::json!({"gpu_lock": {"tag": "P7-A"}}));
        let (block, res) = check_lock(&speed, Some(lock.clone()), &dir);
        assert!(res.is_err() && !block.as_ref().unwrap().verified, "no lock file: refused");
        std::fs::write(&lock, r#"{"tag":"P3B-A","pid":1}"#).unwrap();
        let (block, res) = check_lock(&speed, Some(lock.clone()), &dir);
        let err = res.unwrap_err();
        assert!(err.contains("held by 'P3B-A'") && err.contains("P7-A"), "{err}");
        assert_eq!(block.unwrap().held_by.as_deref(), Some("P3B-A"));
        std::fs::write(&lock, r#"{"tag":"P7-A","pid":1}"#).unwrap();
        let (block, res) = check_lock(&speed, Some(lock.clone()), &dir);
        assert!(res.is_ok());
        let block = block.unwrap();
        assert!(block.verified && block.required && block.held_by.as_deref() == Some("P7-A"));
        // require=false records the owner but never refuses; accuracy runs have no block at all.
        std::fs::write(&lock, "someone else").unwrap();
        let relaxed = run(serde_json::json!({"gpu_lock": {"require": false}}));
        let (block, res) = check_lock(&relaxed, Some(lock.clone()), &dir);
        assert!(res.is_ok() && !block.unwrap().verified);
        let accuracy = run(serde_json::json!({"mode": "accuracy"}));
        let (block, res) = check_lock(&accuracy, Some(lock), &dir);
        assert!(block.is_none() && res.is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_cpu_window_never_goes_negative_with_a_large_load_tally() {
        let tally = LoadCpu::default();
        let w = Window::start(Some(&tally));
        let mut x = 0u64;
        for i in 0..2_000_000u64 {
            x = x.wrapping_add(i * i);
        }
        std::hint::black_box(x);
        if let Some(ms) = w.cpu_ms() {
            assert!(ms >= 0.0);
        }
        assert!(w.wall() > Duration::ZERO);
    }
}
