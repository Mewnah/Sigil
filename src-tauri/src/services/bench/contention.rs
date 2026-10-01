//! Contention mode (P3-F): the paced harness alongside a GPU hammer (the `bench/gpu-load`
//! convolution graph on the EP's normal-priority stream / queue at a set duty), CPU burner threads
//! and a 3 s stall, reporting p50 / p99 compute and latency, the over-budget ratio and the stall
//! recovery with the paced run's metric names. The H1 numbers of the scorecard come from here
//! until a friend sends a live-record export (plan "Real RTX 3060/4060 numbers under VRChat+OBS
//! load": provisional).
//!
//! # Profiles
//! | profile | GPU load | CPU load | stall |
//! |---|---|---|---|
//! | `h1_vrchat_obs` | conv hammer 256 x 256 x 8 layers, 70 % duty, 16.7 ms period (a 60 fps frame) | 3 threads at 50 % (game, render, OBS) | 3 s at 40 % of the run |
//! | `h2_load70` | conv hammer, 70 % duty (plan H2) | - | - |
//! | `h3_cpu_game` | - | 4 threads at 60 % (VRChat on a 6-core CPU) | - |
//! | `stall` | - | - | 3 s at 40 % of the run |
//! | `custom` | `load` only | | `stalls` only |
//!
//! `load` / `stalls` in the run replace the profile's. The GPU load runs on the engine's own GPU
//! EP (DirectML when the engine runs on the CPU) unless `gpu_execution_provider` names one. A run
//! too short for its stall (the stall must start 1 s after the warm-up and end 1 s before the
//! end) drops it.
//!
//! # Frozen by P3-S0
//! [`available`] (the spec schema of the mode is P3-F's, in `spec.rs`).

use std::sync::atomic::AtomicBool;

use super::paced::{run_paced, PacedInput};
use super::report::RunOutcome;
use super::spec::{validate_paced, BenchSpec, ContentionRun, CpuLoadSpec, GpuLoadSpec, LoadSpec, PacedRun, StallSpec};
use super::Prepared;

/// Whether the contention mode runs in this build.
#[allow(dead_code)] // P3-S0 frozen API (the spec mode runs whenever this build has it).
pub fn available() -> bool {
    true
}

/// The profiles, in documentation order.
pub const PROFILES: [&str; 5] = ["h1_vrchat_obs", "h2_load70", "h3_cpu_game", "stall", "custom"];

/// One frame at 60 fps.
const FRAME_MS: f64 = 1000.0 / 60.0;
/// Injected stall of the profiles.
const STALL_MS: f64 = 3_000.0;

fn conv_hammer(duty: f64, ep: &str) -> GpuLoadSpec {
    GpuLoadSpec {
        duty,
        period_ms: FRAME_MS,
        execution_provider: ep.to_string(),
        device_index: 0,
        size: 256,
        depth: 8,
    }
}

fn burners(threads: usize, duty: f64) -> CpuLoadSpec {
    CpuLoadSpec {
        threads,
        duty,
        period_ms: FRAME_MS,
    }
}

/// The `h2_load70` profile's load (the conv hammer at 70 % duty on `gpu_ep`): also what the
/// autotuner measures the GPU levers under.
pub(crate) fn h2_load(gpu_ep: &str) -> LoadSpec {
    LoadSpec {
        gpu: Some(conv_hammer(0.7, gpu_ep)),
        cpu: None,
    }
}

/// The profile's stall, when the run is long enough for it.
fn profile_stall(c: &ContentionRun) -> Vec<StallSpec> {
    let at = (c.seconds * 0.4).max(c.warmup_s + super::spec::STALL_BASELINE_S);
    if at + STALL_MS / 1000.0 + 1.0 <= c.seconds {
        vec![StallSpec { at_s: at, ms: STALL_MS }]
    } else {
        Vec::new()
    }
}

/// Load and stalls of `c` (profile, then the run's overrides) with the GPU load on `gpu_ep`.
fn parts_on(c: &ContentionRun, gpu_ep: &str) -> (LoadSpec, Vec<StallSpec>) {
    let (load, stalls) = match c.profile.trim() {
        "h1_vrchat_obs" => (
            LoadSpec {
                gpu: Some(conv_hammer(0.7, gpu_ep)),
                cpu: Some(burners(3, 0.5)),
            },
            profile_stall(c),
        ),
        "h2_load70" => (h2_load(gpu_ep), Vec::new()),
        "h3_cpu_game" => (
            LoadSpec {
                gpu: None,
                cpu: Some(burners(4, 0.6)),
            },
            Vec::new(),
        ),
        "stall" => (LoadSpec::default(), profile_stall(c)),
        _ => (LoadSpec::default(), Vec::new()),
    };
    (c.load.clone().unwrap_or(load), c.stalls.clone().unwrap_or(stalls))
}

/// Load and stalls of `c` (for spec validation: which metrics the run produces).
pub(crate) fn profile_parts(c: &ContentionRun) -> (LoadSpec, Vec<StallSpec>) {
    parts_on(c, "dml")
}

/// The paced run `c` expands to.
pub(crate) fn expand(c: &ContentionRun, gpu_ep: &str) -> PacedRun {
    let (load, stalls) = parts_on(c, gpu_ep);
    PacedRun {
        name: c.name.clone(),
        seconds: c.seconds,
        clock: c.clock,
        warmup_s: c.warmup_s,
        load,
        gpu_load_kind: c.gpu_load_kind,
        stalls,
        thresholds: c.thresholds.clone(),
    }
}

/// Spec validation of a contention run.
pub(crate) fn validate(c: &ContentionRun) -> Result<(), String> {
    if !PROFILES.contains(&c.profile.trim()) {
        return Err(format!("profile '{}' must be one of {}", c.profile, PROFILES.join(", ")));
    }
    if c.profile.trim() == "custom" && c.load.is_none() && c.stalls.as_ref().is_none_or(Vec::is_empty) {
        return Err("a custom contention run needs load or stalls".into());
    }
    match c.gpu_execution_provider.trim().to_ascii_lowercase().as_str() {
        "" | "cpu" | "cuda" | "dml" | "directml" | "gpu" => {}
        other => return Err(format!("gpu_execution_provider '{other}' must be cuda, dml or cpu")),
    }
    validate_paced(&expand(c, "dml"))
}

/// The EP the GPU load runs on by default: the engine's own GPU rung (the TensorRT accelerators
/// run on the CUDA rung), else DirectML (an engine on the CPU still shares the GPU with a game).
fn load_ep_for(engine_active: Option<&str>) -> &'static str {
    match engine_active {
        Some("cuda" | "tensorrt_rtx" | "tensorrt") => "cuda",
        _ => "dml",
    }
}

/// Run `c` on the spec's engine.
pub(super) fn run(
    c: &ContentionRun,
    spec: &BenchSpec,
    p: &Prepared,
    write_files: bool,
    cancel: &AtomicBool,
    progress: &(dyn Fn(f64, &str) + Sync),
) -> Result<RunOutcome, String> {
    let gpu_ep = match c.gpu_execution_provider.trim().to_ascii_lowercase().as_str() {
        "" => load_ep_for(p.engine.info.execution_provider_active.as_deref()).to_string(),
        other => other.to_string(),
    };
    let paced = expand(c, &gpu_ep);
    let mut outcome = run_paced(&PacedInput {
        stream: &spec.stream,
        run: &paced,
        engine: &p.engine,
        input: &p.input,
        probe: p.probe,
        record_audio: spec.output.write_audio && write_files,
        cancel,
        progress,
    })?;
    if let Some(obj) = outcome.details.as_object_mut() {
        obj.insert("profile".into(), serde_json::Value::String(c.profile.clone()));
        obj.insert("gpu_load_execution_provider".into(), serde_json::Value::String(gpu_ep));
        obj.insert("stalls_injected".into(), serde_json::to_value(&paced.stalls).unwrap_or_default());
    }
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::super::spec::GpuLoadKind;
    use super::*;

    fn run_spec(v: serde_json::Value) -> ContentionRun {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn profiles_expand_to_their_loads_and_overrides_win() {
        let h1 = run_spec(serde_json::json!({"name": "c", "seconds": 20}));
        let p = expand(&h1, "cuda");
        let g = p.load.gpu.as_ref().unwrap();
        assert_eq!((p.gpu_load_kind, g.execution_provider.as_str(), g.duty), (GpuLoadKind::Conv, "cuda", 0.7));
        assert_eq!(p.load.cpu.as_ref().unwrap().threads, 3);
        assert_eq!(p.stalls.len(), 1);
        assert!((p.stalls[0].at_s - 8.0).abs() < 1e-9 && p.stalls[0].ms == 3_000.0);
        assert!(validate(&h1).is_ok());
        // Too short for the stall: dropped, and still valid.
        let short = run_spec(serde_json::json!({"name": "c", "seconds": 4}));
        assert!(expand(&short, "dml").stalls.is_empty());
        assert!(validate(&short).is_ok());
        let h3 = run_spec(serde_json::json!({"name": "c", "profile": "h3_cpu_game"}));
        let p = expand(&h3, "dml");
        assert!(p.load.gpu.is_none() && p.load.cpu.as_ref().unwrap().threads == 4 && p.stalls.is_empty());
        let over = run_spec(serde_json::json!({"name": "c", "profile": "h2_load70", "stalls": [{"at_s": 3, "ms": 500}],
                                               "load": {"cpu": {"threads": 1}}}));
        let p = expand(&over, "dml");
        assert!(p.load.gpu.is_none(), "load replaces the profile's");
        assert_eq!(p.stalls.len(), 1);
        assert!(validate(&run_spec(serde_json::json!({"name": "c", "profile": "nope"}))).is_err());
        assert!(validate(&run_spec(serde_json::json!({"name": "c", "profile": "custom"}))).is_err());
        assert!(available());
    }

    /// The load shares the engine's GPU: a TensorRT engine runs on the CUDA rung.
    #[test]
    fn the_gpu_load_follows_the_engine_rung() {
        for (active, ep) in [
            (Some("cuda"), "cuda"),
            (Some("tensorrt_rtx"), "cuda"),
            (Some("tensorrt"), "cuda"),
            (Some("dml"), "dml"),
            (Some("cpu"), "dml"),
            (None, "dml"),
        ] {
            assert_eq!(load_ep_for(active), ep, "{active:?}");
        }
    }

    /// A contention spec on the stub runs end to end on the simulated clock: the stall is injected
    /// and recovered within the budget (plan T7), with the paced metric names.
    #[test]
    fn contention_runs_through_the_paced_driver() {
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "c", "engine": {"kind": "stub", "compute_ms": 3.0},
            "stream": {"output_prefill_ms": 80, "output_max_queue_ms": 150},
            "runs": [{"mode": "contention", "name": "stall", "profile": "stall", "seconds": 9, "clock": "simulated",
                      "thresholds": {"stall_recovery_blocks": {"max": 2}, "backlog_over_budget_ms": {"max": 0}}}]
        }))
        .unwrap();
        let cancel = AtomicBool::new(false);
        let progress = |_: &super::super::Progress| {};
        let r = super::super::run_spec(&spec, None, &super::super::BenchContext { cancel: &cancel, progress: &progress });
        let run = &r.runs[0];
        assert_eq!(run.status, super::super::report::RunStatus::Passed, "{:?} {:?}", run.error, run.thresholds);
        assert_eq!(run.mode.as_str(), "contention");
        assert_eq!(run.metrics["stall_recovery_blocks"], 0.0);
        assert_eq!(run.details["profile"], "stall");
    }
}
