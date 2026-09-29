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
//! - Subcommands (kebab-case): `ep-probe [--out <file>]` (P0-H). Planned: `bench <spec.json> --out <dir>`
//!   (P1-F), `fetch <id>` (P1-A), `vc-import-worker <job.json>` (P3-B).
//! - Output goes to stdout / stderr; in release builds (GUI subsystem) the parent console is
//!   attached first so the output reaches the invoking terminal.

use clap::{Parser, Subcommand};
use std::ffi::OsString;
use std::path::PathBuf;

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
        HeadlessCommand::EpProbe { out } => match crate::services::diagnostics::headless_ep_probe() {
            Ok(json) => {
                println!("{json}");
                if let Some(path) = out {
                    if let Err(e) = std::fs::write(&path, json.as_bytes()) {
                        eprintln!("cannot write {}: {e}", path.display());
                        return 1;
                    }
                }
                0
            }
            Err(e) => {
                eprintln!("ep-probe: {e}");
                1
            }
        },
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
}
