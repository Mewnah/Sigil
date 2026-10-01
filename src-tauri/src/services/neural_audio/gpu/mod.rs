//! GPU layer of the Sigil Voice Engine (P3-C2; plan "VC best engine" → Components → GPU layer).
//!
//! - [`plan`]: [`SessionPlan`], one session at fixed shapes with its I/O buffers: the legacy host
//!   path, IOBinding with host tensors bound once (CPU / DirectML / TensorRT rungs, and CUDA without
//!   a graph) or with device-resident buffers at fixed addresses replayed as a CUDA graph captured
//!   and checked at load (CUDA, when every input is read from device memory; [`io_binding`],
//!   [`cuda_graph`]). Zero Rust heap allocations per tick on the IOBinding paths; only the
//!   CUDA-graph path calls no ONNX Runtime function per tick and so makes no host allocation at all
//!   (ONNX Runtime allocates on its own C++ heap inside every run).
//! - [`cuda_driver`]: the CUDA driver API subset (`nvcuda.dll` through libloading): primary
//!   contexts, streams, async copies, pinned host memory.
//! - [`cuda_stream`]: the highest-priority CUDA stream, handed to ORT as the compute stream.
//! - [`dml_queue`]: DirectML on a high-priority D3D12 queue through `OrtDmlApi` (DML1).
//! - [`trt`]: TensorRT-RTX (plugin EP pack) or classic TensorRT, with an engine cache and build
//!   progress.
//! - [`arena`]: the VRAM ledger (plans hold what they measured), the shared environment CUDA arena
//!   and the VRAM cap.
//!
//! Every session still goes through `ort_session` (EP ladder, preflight, canary, CPU retry,
//! `error_on_failure`, outcome records; `build_session_ext` applies the GPU extras per rung). A lever
//! that is off or unavailable leaves the legacy behaviour; what a plan really uses is
//! [`SessionPlan::levers`], and why a requested lever is off is in [`SessionPlan::notes`].
//!
//! # Frozen by P3-S0 (signatures; P3-C2 implements; change only through INTEGRATE)
//! - [`GpuLevers`] (+ [`GpuLevers::from_effective`]), `plan::{TensorSpec, ExternalWeights,
//!   PlanSpec, SessionPlan}` with `SessionPlan::{build, outcome, levers, input_f32, input_i64,
//!   run, output}`.
//! - `available()` of [`io_binding`], [`cuda_graph`], [`cuda_stream`], [`dml_queue`] and [`trt`]
//!   (cheap: no session builds; `voice_changer::capabilities` calls them).
//! - `trt::{TRT_PROGRESS_EVENT, TrtProgress, engine_cache_dir}`, `arena::{VramLedger, VramReport,
//!   ledger}`.
//!
//! # Added by P3-C2
//! - [`GpuLevers::tf32`] (from the precision request: every request but `fp32` allows TF32).
//! - `SessionPlan::{notes, residency, warm_up, vram_mb, set_output_download, copy_rows_to,
//!   pinned_host_buffers}`, [`Residency`], [`RowCopy`], `plan::CudaTuning` (`SIGIL_CUDA_TUNING`),
//!   `plan::DEVICE_IO_WITHOUT_GRAPHS`, `io_binding::inputs_on_device`, `dml_queue::DmlQueue::
//!   check_not_removed`.
//! - `arena::{VramLedger::hold, VramHold, VramProbe, ensure_cuda_env_arena}`,
//!   `trt::{TrtRequest, pe_imports, TRT_DIR_ENV}`, `cuda_stream::PriorityStream::{normal, priority,
//!   device}`, [`cuda_driver`], `cuda_graph::CapturedGraph`.
//! - Test builds: `alloc_probe` is the crate's single `#[global_allocator]` (a counting allocator;
//!   `alloc_probe::allocations_during`). Other allocation-free checks reuse it.

// P3-E's factory and the pitch bank drive these plans; parts of the API (row copies between
// plans, output-download control, pinned-host accessors, the VRAM ledger snapshot and the shared
// CUDA arena) serve tests, the benchmark and follow-up work, so the module keeps its dead-code
// allowance.
#![allow(dead_code)]

pub mod arena;
pub mod cuda_driver;
pub mod cuda_graph;
pub mod cuda_stream;
pub mod dml_queue;
pub mod io_binding;
pub mod plan;
pub mod trt;

#[cfg(test)]
pub(crate) mod alloc_probe;
#[cfg(test)]
mod tests;

#[allow(unused_imports)] // P3-S0 frozen re-exports for P3-C1.
pub use plan::{ExternalWeights, PlanSpec, Residency, RowCopy, SessionPlan, TensorSpec};

use super::vc::engine::options::{Accelerator, EffectiveLevers, Lever, PrecisionMode};

/// The GPU levers of one session plan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuLevers {
    pub io_binding: bool,
    /// Needs `io_binding` (a plan without it runs eager).
    pub cuda_graphs: bool,
    /// Highest-priority CUDA stream / high-priority DirectML queue.
    pub priority: bool,
    pub accelerator: Accelerator,
    /// TF32 tensor-core math on the CUDA rung (P3-C2): allowed unless the precision request is
    /// `fp32` (bit parity with PyTorch fp32, the 1e-4 gate). TF32 keeps the error near 1e-3, inside
    /// the fp16 budget the GPU path already uses, at less than half the cost (benchmarked).
    pub tf32: bool,
}

impl GpuLevers {
    pub fn from_effective(levers: &EffectiveLevers) -> GpuLevers {
        GpuLevers {
            io_binding: levers.levers.get(Lever::IoBinding),
            cuda_graphs: levers.levers.get(Lever::CudaGraphs),
            priority: levers.levers.get(Lever::GpuPriority),
            accelerator: levers.accelerator,
            tf32: levers.precision != PrecisionMode::Fp32,
        }
    }
}
