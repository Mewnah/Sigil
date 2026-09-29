//! Speech runtimes loaded lazily from runtime packs (no plugin of its own; the P2 `speech` plugin
//! drives them through its AsrEngine trait):
//! - [`sherpa`]: sherpa-onnx C API (CPU tier) through libloading; owned by P1-G1.
//! - [`transcribe`]: transcribe.cpp as `sigil_transcribe.dll` (sigil_engine ABI v1, GPU tier);
//!   owned by P1-G2.
//!
//! Scaffolded by P1-S0b. This file (module list + the engine-probe dispatcher) is a scaffold file;
//! the engine modules are the workstreams'.
//!
//! # Frozen contract (P1-S0b; change only through an INTEGRATE step)
//! - `Sigil.exe --headless engine-probe <engine> [model] [wav] [--variant <v>] [--runtime-dir <dir>]
//!   [--out <file>] [-- <extra args>...]` parses into [`EngineProbeArgs`] and runs
//!   [`headless_engine_probe`], which dispatches on `engine`:
//!   `sherpa` → [`sherpa::headless_probe`], `transcribe` → [`transcribe::headless_probe`], any other
//!   name (e.g. `echo`) → `native_runtime::headless_probe` (a sigil_engine DLL named
//!   `sigil_<engine>.dll`). Each returns the JSON report printed to stdout (Err → exit code 1).
//! - Rules for both runtimes: nothing is load-time linked; a missing pack or DLL is an `Err`, never
//!   a panic; the app starts without any pack; sherpa's bundled ORT stays CPU-only inside its DLL.

pub mod sherpa;
pub mod transcribe;

use std::path::PathBuf;

/// Arguments of `Sigil.exe --headless engine-probe`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EngineProbeArgs {
    /// Engine name: `sherpa`, `transcribe`, or a sigil_engine DLL name such as `echo`.
    pub engine: String,
    /// Model file or folder to load.
    pub model: Option<PathBuf>,
    /// 16 kHz (or any rate; engines resample) WAV to run through the model.
    pub wav: Option<PathBuf>,
    /// Runtime variant to force (`cpu`, `cpu-avx2`, `cpu-avx512`, `vulkan`, `cuda13`).
    pub variant: Option<String>,
    /// Load the runtime from this folder instead of the installed pack (development).
    pub runtime_dir: Option<PathBuf>,
    /// Engine-specific extra arguments (after `--`).
    pub extra: Vec<String>,
}

/// Body of `Sigil.exe --headless engine-probe`: returns the JSON report.
pub fn headless_engine_probe(args: &EngineProbeArgs) -> Result<String, String> {
    match args.engine.as_str() {
        "sherpa" | "sherpa-onnx" => sherpa::headless_probe(args),
        "transcribe" => transcribe::headless_probe(args),
        "" => Err("engine-probe: no engine given".into()),
        _ => crate::services::native_runtime::headless_probe(args),
    }
}
