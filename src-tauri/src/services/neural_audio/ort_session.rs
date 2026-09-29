//! Shared ONNX Runtime session builder for the voice changer (ONNX chunk mode and RVC).
//!
//! Every session built here sleeps instead of busy-spinning (`allow_spinning=0`, sequential
//! exec, Level3 graph opt) so idle ORT workers coexist with OBS/VRChat and in-process KittenTTS.
//! The intra/inter thread caps (= 1) are applied only for the RVC "Streamer" preset
//! (`VcSessionOptions::streamer_profile`); all other sessions keep ONNX Runtime's default thread
//! pool. The GPU device index and the CUDA VRAM cap are passed as execution-provider options
//! (no process environment variables are involved).

use std::sync::Once;

use ort::ep::{ArenaExtendStrategy, CPU, CUDA};
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;

static ORT_ENV_INIT: Once = Once::new();

/// Ensure a process-wide ORT environment exists before the first session.
/// Safe to call repeatedly; configuration sticks on first commit.
pub fn ensure_ort_environment() {
    ORT_ENV_INIT.call_once(|| {
        // Application-owned init (libraries must not commit env themselves).
        if !ort::init().with_name("sigil").commit() {
            eprintln!(
                "[ORT] environment was already configured before Sigil's init; the 'sigil' \
                 environment options were not applied"
            );
        }
    });
}

/// Per-session options for voice-changer ONNX graphs (RVC HuBERT/generator, pitch extractors).
#[derive(Clone, Debug, Default)]
pub struct VcSessionOptions {
    /// Device ordinal for the CUDA or DirectML execution provider (ignored on CPU).
    pub cuda_device_index: i32,
    /// RVC "Streamer" preset: cap intra/inter op threads to one each.
    pub streamer_profile: bool,
    /// CUDA arena memory limit in MiB (CUDA only; ignored for DirectML/CPU).
    pub vram_cap_mb: Option<u32>,
}

/// Operator-facing hint when CUDA/DirectML registration fails (drivers missing, no GPU, etc.).
pub fn format_onnx_ep_error(requested_ep: &str, err: &impl std::fmt::Display) -> String {
    let detail = err.to_string();
    let lower = detail.to_lowercase();
    let ep_label = match requested_ep {
        "cuda" | "gpu" => "CUDA",
        "dml" => "DirectML",
        _ => "CPU",
    };
    let driver_hint = match requested_ep {
        "cuda" | "gpu" => {
            " Install NVIDIA drivers and ensure a CUDA-capable GPU is available, or select CPU in the inspector."
        }
        "dml" => " DirectML requires a compatible Windows GPU and drivers, or select CPU in the inspector.",
        _ => "",
    };
    if lower.contains("cuda")
        || lower.contains("directml")
        || lower.contains("execution provider")
        || lower.contains("failed to create")
        || lower.contains("not found")
    {
        format!(
            "ONNX {} execution provider unavailable: {}.{}",
            ep_label, detail, driver_hint
        )
    } else {
        format!("ONNX execution providers ({}): {}", ep_label, detail)
    }
}

/// Coexistence options for every VC session: sleep instead of busy-spin (preset-independent).
fn apply_common_session_options(builder: SessionBuilder) -> Result<SessionBuilder, String> {
    builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("ONNX graph opt: {}", e))?
        .with_parallel_execution(false)
        .map_err(|e| format!("ONNX parallel exec: {}", e))?
        .with_memory_pattern(true)
        .map_err(|e| format!("ONNX memory pattern: {}", e))?
        .with_intra_op_spinning(false)
        .map_err(|e| format!("ONNX intra spinning: {}", e))?
        .with_inter_op_spinning(false)
        .map_err(|e| format!("ONNX inter spinning: {}", e))
}

/// Streamer-only thread caps: one thread per pool.
fn apply_streamer_thread_caps(builder: SessionBuilder) -> Result<SessionBuilder, String> {
    builder
        .with_intra_threads(1)
        .map_err(|e| format!("ONNX intra threads: {}", e))?
        .with_inter_threads(1)
        .map_err(|e| format!("ONNX inter threads: {}", e))
}

/// CUDA EP on the selected device; with a VRAM cap, the arena is limited to that many bytes
/// and grows only by what is requested (power-of-two growth would overshoot the cap).
fn cuda_provider(opts: &VcSessionOptions) -> CUDA {
    let cuda = CUDA::default().with_device_id(opts.cuda_device_index.max(0));
    match opts.vram_cap_mb {
        Some(mb) if mb > 0 => cuda
            .with_memory_limit((mb as usize).saturating_mul(1024 * 1024))
            .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested),
        _ => cuda,
    }
}

/// Build a session with EP selection, spinning off, and ONNX Runtime's default threading
/// (device 0, no VRAM cap). Same as [`session_builder_for_vc`] with default options.
pub fn session_builder_with_ep(execution_provider_hint: &str) -> Result<SessionBuilder, String> {
    session_builder_for_vc(execution_provider_hint, &VcSessionOptions::default())
}

/// Session builder for voice-changer graphs.
///
/// - Every session: Level3 graph optimization, sequential execution, memory pattern on,
///   intra/inter op spinning off. `streamer_profile` additionally caps intra/inter threads to 1.
/// - `"cuda"` / `"gpu"`: CUDA EP on `cuda_device_index` (+ `vram_cap_mb` arena limit), then CPU.
/// - `"dml"` (Windows): DirectML EP on `cuda_device_index`, then CPU (VRAM cap not applicable).
/// - anything else: CPU.
pub fn session_builder_for_vc(
    execution_provider_hint: &str,
    opts: &VcSessionOptions,
) -> Result<SessionBuilder, String> {
    ensure_ort_environment();

    let builder = Session::builder().map_err(|e| format!("ONNX session builder: {}", e))?;
    let builder = apply_common_session_options(builder)?;
    let builder = if opts.streamer_profile {
        apply_streamer_thread_caps(builder)?
    } else {
        builder
    };

    let p = execution_provider_hint.trim().to_lowercase();
    let b = match p.as_str() {
        "cuda" | "gpu" => {
            builder.with_execution_providers([cuda_provider(opts).build(), CPU::default().build()])
        }
        "dml" => {
            #[cfg(windows)]
            {
                builder.with_execution_providers([
                    ort::ep::DirectML::default()
                        .with_device_id(opts.cuda_device_index.max(0))
                        .build(),
                    CPU::default().build(),
                ])
            }
            #[cfg(not(windows))]
            {
                builder.with_execution_providers([CPU::default().build()])
            }
        }
        _ => builder.with_execution_providers([CPU::default().build()]),
    };
    b.map_err(|e| format_onnx_ep_error(&p, &e))
}
