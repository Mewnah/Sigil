//! Headless entry points: `Sigil.exe --headless <command> [args...]`.
//!
//! `main()` checks [`headless_args`] before anything else — before the port pre-check, the
//! "port in use" dialog and Tauri — so a headless child (benchmark, model fetch, isolated import
//! worker, EP probe) never binds the pubsub port, opens a window or conflicts with a running Sigil.
//!
//! # Frozen contract (P0-S0; later phases add subcommands, never rename them)
//! - `--headless` must be the **first** argument; everything after it goes to the headless parser.
//! - [`headless_args`] → `Some(rest)` for a headless invocation, [`run`] → process exit code
//!   (0 success, 1 command failed, 2 usage error; `help` / `--version` exit 0).
//! - Subcommands (kebab-case), each dispatching to a function owned by its workstream:
//!   - `ep-probe [--out <file>]` → `diagnostics::headless_ep_probe` (P0-H).
//!   - `fetch <id>... [--mirror <m>] [--custom-url <url>] | fetch --list` →
//!     `model_store::headless_fetch` (P1-A).
//!   - `bench <spec.json> [--out <dir>]` → `bench::headless_bench` (P1-F).
//!   - `engine-probe <engine> [model] [wav] [--variant <v>] [--runtime-dir <dir>] [--out <file>]
//!     [-- <extra>...]` → `speech_rt::headless_engine_probe` (P1-G1 sherpa, P1-G2 transcribe,
//!     P1-D1 sigil_engine DLLs such as `echo`).
//!   - `vc-import-worker <job.json>` → `neural_audio::vc::import::run_import_worker` (P1-G3 / P3-B).
//!   - `sysinfo [--out <file>]` → `resources::headless_report` (P1-D2): system probe, VRAM
//!     budget, claims and every recommendation as JSON.
//!   - `stt-worker [<spec.json>]` → `speech::worker_proc::headless_stt_worker` (P2-D; arm added
//!     by P2-S0): one speech engine in an isolated worker process.
//!   - `mt-bench <spec.json> [--out <dir>]` → `mt_local::bench::headless_mt_bench` (P4-D; arm
//!     added by P4-S0): machine-translation chrF / latency table over a local corpus.
//!
//!   Every handler returns `Result<String, String>`: `Ok(text)` is printed to stdout (exit 0),
//!   `Err(msg)` to stderr (exit 1). The P1 arms were added by P1-S0b (`sysinfo` by P1-I0).
//! - Output goes to stdout / stderr; in release builds (GUI subsystem) the parent console is
//!   attached first so the output reaches the invoking terminal.

use clap::{Parser, Subcommand};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// First argument that selects headless mode.
pub const HEADLESS_FLAG: &str = "--headless";

/// The arguments after `--headless` when this process was started headless.
pub fn headless_args() -> Option<Vec<OsString>> {
    let mut args = std::env::args_os().skip(1);
    match args.next() {
        Some(first) if first == HEADLESS_FLAG => Some(args.collect()),
        _ => None,
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "Sigil --headless",
    version,
    about = "Sigil headless tools (no window, no server, no port)",
    no_binary_name = true,
    arg_required_else_help = true
)]
struct HeadlessCli {
    #[command(subcommand)]
    command: HeadlessCommand,
}

#[derive(Subcommand, Debug)]
enum HeadlessCommand {
    /// Probe ONNX Runtime execution providers, GPU driver, DXGI adapters and CUDA / cuDNN DLLs,
    /// and print the report as JSON.
    EpProbe {
        /// Also write the JSON report to this file.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Install model-store catalog entries (and their requirements), one download at a time.
    Fetch {
        /// Catalog ids, e.g. `base.test-small`.
        #[arg(required_unless_present = "list")]
        ids: Vec<String>,
        /// Preferred mirror: auto, huggingface, hf-mirror, modelscope, github-release, custom.
        #[arg(long)]
        mirror: Option<String>,
        /// Base URL of the custom mirror.
        #[arg(long)]
        custom_url: Option<String>,
        /// Print the catalog and install states instead of fetching.
        #[arg(long)]
        list: bool,
    },
    /// Run a benchmark spec through the real voice-changer harness and write report.json / report.md.
    Bench {
        /// Benchmark spec (JSON).
        spec: PathBuf,
        /// Output directory for the reports.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Load a native speech runtime (sherpa, transcribe, or a sigil_engine DLL such as echo) and
    /// run a smoke test; prints a JSON report.
    EngineProbe {
        /// Engine: sherpa, transcribe, echo, ...
        engine: String,
        /// Model file or folder.
        model: Option<PathBuf>,
        /// WAV file to run through the model.
        wav: Option<PathBuf>,
        /// Force a runtime variant: cpu, cpu-avx2, cpu-avx512, vulkan, cuda13.
        #[arg(long)]
        variant: Option<String>,
        /// Load the runtime from this folder instead of the installed pack.
        #[arg(long)]
        runtime_dir: Option<PathBuf>,
        /// Also write the JSON report to this file.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Engine-specific arguments after `--`.
        #[arg(last = true)]
        extra: Vec<String>,
    },
    /// Import an untrusted RVC .pth/.zip described by a job file (isolated worker process).
    VcImportWorker {
        /// Job description (JSON).
        job: PathBuf,
    },
    /// Probe GPUs, drivers, CPU and memory, and print the VRAM budget, claims and the model-size
    /// recommendations as JSON (no ONNX Runtime, no GPU runtime, no CUDA context).
    Sysinfo {
        /// Also write the JSON report to this file.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Host one speech engine in an isolated worker process (spawned by the speech plugin through
    /// the process host; talks over stdio).
    SttWorker {
        /// Worker spec (JSON); optional when the parent sends it over stdio.
        spec: Option<PathBuf>,
    },
    /// Score local machine-translation engines (NLLB, Bergamot, llama-server LLMs) on a local
    /// corpus: chrF and latency per engine and direction (report.json / report.md).
    MtBench {
        /// Benchmark spec (JSON).
        spec: PathBuf,
        /// Output directory for the reports.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Run the benchmark's GPU load in this separate process (a game's own process and queue;
    /// started by a benchmark with `load.gpu.out_of_process`). P3B.
    GpuLoad {
        /// Load spec (JSON).
        spec: PathBuf,
    },
}

/// Parse and run one headless command; returns the process exit code.
pub fn run(args: Vec<OsString>) -> i32 {
    attach_parent_console();
    let cli = match HeadlessCli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(e) => {
            let _ = e.print();
            return e.exit_code();
        }
    };
    match cli.command {
        HeadlessCommand::EpProbe { out } => report(
            "ep-probe",
            crate::services::diagnostics::headless_ep_probe(),
            out.as_deref(),
        ),
        HeadlessCommand::Fetch {
            ids,
            mirror,
            custom_url,
            list,
        } => {
            let args = crate::services::model_store::FetchArgs {
                ids,
                mirror,
                custom_url,
                list,
            };
            report("fetch", crate::services::model_store::headless_fetch(&args), None)
        }
        HeadlessCommand::Bench { spec, out } => report(
            "bench",
            crate::services::bench::headless_bench(&spec, out.as_deref()),
            None,
        ),
        HeadlessCommand::EngineProbe {
            engine,
            model,
            wav,
            variant,
            runtime_dir,
            out,
            extra,
        } => {
            let args = crate::services::speech_rt::EngineProbeArgs {
                engine,
                model,
                wav,
                variant,
                runtime_dir,
                extra,
            };
            report(
                "engine-probe",
                crate::services::speech_rt::headless_engine_probe(&args),
                out.as_deref(),
            )
        }
        HeadlessCommand::VcImportWorker { job } => report(
            "vc-import-worker",
            crate::services::neural_audio::vc::import::run_import_worker(&job),
            None,
        ),
        HeadlessCommand::Sysinfo { out } => report(
            "sysinfo",
            crate::services::resources::headless_report(),
            out.as_deref(),
        ),
        HeadlessCommand::SttWorker { spec } => report(
            "stt-worker",
            crate::services::speech::worker_proc::headless_stt_worker(spec.as_deref()),
            None,
        ),
        HeadlessCommand::MtBench { spec, out } => report(
            "mt-bench",
            crate::services::mt_local::bench::headless_mt_bench(&spec, out.as_deref()),
            None,
        ),
        HeadlessCommand::GpuLoad { spec } => report("gpu-load", crate::services::bench::gpu_load_child::run(&spec), None),
    }
}

/// Print a handler result (stdout on success, stderr on failure), optionally also writing the
/// output to `out`; returns the exit code.
fn report(command: &str, result: Result<String, String>, out: Option<&Path>) -> i32 {
    match result {
        Ok(text) => {
            println!("{text}");
            if let Some(path) = out {
                if let Err(e) = std::fs::write(path, text.as_bytes()) {
                    eprintln!("cannot write {}: {e}", path.display());
                    return 1;
                }
            }
            0
        }
        Err(e) => {
            eprintln!("{command}: {e}");
            1
        }
    }
}

/// Release builds use the GUI subsystem (no console): attach to the parent's console so headless
/// output is visible. A no-op when a console is already present (debug builds) or there is none.
fn attach_parent_console() {
    #[cfg(windows)]
    {
        use windows::Win32::System::Console::{AttachConsole, ATTACH_PARENT_PROCESS};
        // SAFETY: plain Win32 call; failure (already attached / no parent console) is ignored.
        unsafe {
            let _ = AttachConsole(ATTACH_PARENT_PROCESS);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{HeadlessCli, HeadlessCommand};
    use clap::Parser;

    #[test]
    fn parses_ep_probe_and_help() {
        let cli = HeadlessCli::try_parse_from(["ep-probe", "--out", "x.json"]).unwrap();
        assert!(matches!(cli.command, HeadlessCommand::EpProbe { out: Some(ref p) } if p.ends_with("x.json")));
        let help = HeadlessCli::try_parse_from(["help"]).unwrap_err();
        assert_eq!(help.exit_code(), 0);
        let bad = HeadlessCli::try_parse_from(["no-such-command"]).unwrap_err();
        assert_eq!(bad.exit_code(), 2);
    }

    #[test]
    fn parses_p1_subcommands() {
        let cli = HeadlessCli::try_parse_from(["fetch", "base.test-small", "--mirror", "hf-mirror"]).unwrap();
        assert!(matches!(
            cli.command,
            HeadlessCommand::Fetch { ref ids, mirror: Some(ref m), list: false, .. }
                if ids == &["base.test-small".to_string()] && m == "hf-mirror"
        ));
        let cli = HeadlessCli::try_parse_from(["fetch", "--list"]).unwrap();
        assert!(matches!(cli.command, HeadlessCommand::Fetch { ref ids, list: true, .. } if ids.is_empty()));
        assert_eq!(HeadlessCli::try_parse_from(["fetch"]).unwrap_err().exit_code(), 2);

        let cli = HeadlessCli::try_parse_from(["bench", "specs/stub-smoke.json", "--out", "bench-results/x"]).unwrap();
        assert!(matches!(
            cli.command,
            HeadlessCommand::Bench { ref spec, out: Some(ref o) }
                if spec.ends_with("stub-smoke.json") && o.ends_with("x")
        ));

        let cli = HeadlessCli::try_parse_from([
            "engine-probe",
            "sherpa",
            "model.onnx",
            "a.wav",
            "--variant",
            "cpu",
            "--",
            "--threads",
            "2",
        ])
        .unwrap();
        match cli.command {
            HeadlessCommand::EngineProbe {
                engine,
                model,
                wav,
                variant,
                runtime_dir,
                out,
                extra,
            } => {
                assert_eq!(engine, "sherpa");
                assert!(model.unwrap().ends_with("model.onnx"));
                assert!(wav.unwrap().ends_with("a.wav"));
                assert_eq!(variant.as_deref(), Some("cpu"));
                assert!(runtime_dir.is_none() && out.is_none());
                assert_eq!(extra, vec!["--threads".to_string(), "2".to_string()]);
            }
            other => panic!("unexpected {other:?}"),
        }
        let cli = HeadlessCli::try_parse_from(["engine-probe", "echo"]).unwrap();
        assert!(matches!(cli.command, HeadlessCommand::EngineProbe { ref engine, model: None, .. } if engine == "echo"));

        let cli = HeadlessCli::try_parse_from(["vc-import-worker", "job.json"]).unwrap();
        assert!(matches!(cli.command, HeadlessCommand::VcImportWorker { ref job } if job.ends_with("job.json")));

        let cli = HeadlessCli::try_parse_from(["sysinfo", "--out", "sys.json"]).unwrap();
        assert!(matches!(cli.command, HeadlessCommand::Sysinfo { out: Some(ref p) } if p.ends_with("sys.json")));
        let cli = HeadlessCli::try_parse_from(["sysinfo"]).unwrap();
        assert!(matches!(cli.command, HeadlessCommand::Sysinfo { out: None }));
    }

    #[test]
    fn parses_p2_subcommands() {
        let cli = HeadlessCli::try_parse_from(["stt-worker", "spec.json"]).unwrap();
        assert!(matches!(cli.command, HeadlessCommand::SttWorker { spec: Some(ref p) } if p.ends_with("spec.json")));
        let cli = HeadlessCli::try_parse_from(["stt-worker"]).unwrap();
        assert!(matches!(cli.command, HeadlessCommand::SttWorker { spec: None }));
    }

    #[test]
    fn parses_p4_subcommands() {
        let cli = HeadlessCli::try_parse_from(["mt-bench", "mt.json", "--out", "bench-results/mt"]).unwrap();
        assert!(matches!(
            cli.command,
            HeadlessCommand::MtBench { ref spec, out: Some(ref o) }
                if spec.ends_with("mt.json") && o.ends_with("mt")
        ));
        let cli = HeadlessCli::try_parse_from(["mt-bench", "mt.json"]).unwrap();
        assert!(matches!(cli.command, HeadlessCommand::MtBench { out: None, .. }));
        assert_eq!(HeadlessCli::try_parse_from(["mt-bench"]).unwrap_err().exit_code(), 2);
    }

    /// P3B-S0: the out-of-process GPU load arm parses its spec path.
    #[test]
    fn gpu_load_parses() {
        let cli = HeadlessCli::try_parse_from(["gpu-load", "load.json"]).unwrap();
        assert!(matches!(cli.command, HeadlessCommand::GpuLoad { ref spec } if spec.ends_with("load.json")));
        assert_eq!(HeadlessCli::try_parse_from(["gpu-load"]).unwrap_err().exit_code(), 2);
    }

    #[test]
    fn handler_results_map_to_exit_codes() {
        // Stubs return Err until their workstream lands; the exit-code contract is what matters.
        assert_eq!(super::report("x", Err("nope".into()), None), 1);
        assert_eq!(super::report("x", Ok("{}".into()), None), 0);
    }
}
