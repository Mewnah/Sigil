//! Session plans (P3-C2): one ONNX Runtime session at fixed shapes with its I/O buffers, run once
//! per tick without allocating.
//!
//! Every plan is built through `ort_session::build_session_ext` (full EP ladder, in-memory external
//! weights, dimension overrides, the rung's [`GpuExtras`]) and then runs one of three I/O paths:
//!
//! | path | when | per tick |
//! |------|------|----------|
//! | **host** (legacy) | `io_binding` off | host tensors wrap the plan's buffers; outputs are copied out of new ORT values |
//! | **bound** | `io_binding` on, CPU / DirectML / TensorRT rung, CUDA without a graph | host tensors bound once; ORT reads and writes them in place on the CPU, and uploads only the inputs written since the last run on a GPU |
//! | **device** | `io_binding` + a captured CUDA graph, CUDA rung, every input read from device memory | device tensors bound once at fixed addresses; the session runs on the plan's stream (ORT's `user_compute_stream`); dirty inputs go up from pinned host mirrors, then the CUDA graph, then the read outputs come down, one host wait (P3B-G: on an event, [`cuda_wait`]; or none with [`SessionPlan::run_async`] until the buffers are touched) |
//!
//! The bound and device paths make **no Rust heap allocation per tick** (`run`, `input_f32`,
//! `input_i64` and `output` are allocation-free there, checked by
//! `run_is_allocation_free_on_the_bound_path` and the CUDA / DirectML tests). Only the device path
//! with a CUDA graph is free of host allocations altogether (plan T2): it calls no ONNX Runtime
//! function per tick. The bound path calls the raw `RunWithBinding` C API, inside which ONNX Runtime
//! allocates on its own C++ heap every run (execution frame, feeds and fetches; on a GPU rung also
//! a new device value per re-bound input), which the counting allocator does not see.
//!
//! A CUDA plan keeps device-resident buffers only with a captured graph
//! ([`DEVICE_IO_WITHOUT_GRAPHS`]; `SIGIL_CUDA_TUNING=devio=1` keeps them without one for
//! benchmarks) and only when ORT reads every input from device memory
//! (`io_binding::inputs_on_device`: ORT copies a bound value at bind time, so an input a CUDA kernel
//! takes on the CPU would keep its bind-time copy); otherwise it binds host tensors.
//!
//! Levers (from [`GpuLevers`]; each falls back to the legacy behaviour with a note in
//! [`SessionPlan::notes`]):
//! - `io_binding`: the bound / device paths above.
//! - `cuda_graphs` (needs `io_binding` and the CUDA rung): the plan's build records the session's
//!   run into a CUDA graph itself (thread-local capture mode, so other CUDA users in the process are
//!   never disturbed, unlike ORT's `enable_cuda_graph`), checks that replays reproduce regular runs
//!   on two non-zero input patterns with the outputs poisoned first (a graph that recorded nothing
//!   or part of the run fails), and then replays it every tick without calling ORT
//!   ([`cuda_graph`]). A capture that fails its check leaves the plan running the session normally
//!   on bound host tensors, with a note. Shapes are fixed per plan, so a preset change (new
//!   `ShapePlan`) builds new plans, which capture new graphs.
//! - `gpu_priority`: on CUDA the plan's stream is created at the device's highest priority
//!   ([`PriorityStream`]); on DirectML the session runs on the adapter's high-priority D3D12 queue
//!   ([`dml_queue`]).
//! - accelerator `tensorrt_rtx` (CUDA requests): TensorRT-RTX (or classic TensorRT) with the engine
//!   cache ([`trt`]); the outcome then reads `tensorrt_rtx` (or a fallback from it with its reason).
//! - accelerator `dml_hp`: the high-priority DirectML queue, like `gpu_priority` on DirectML.
//!
//! CUDA sessions of a plan use TF32 only when [`GpuLevers::tf32`] allows it (every precision request
//! but `fp32`): TF32 puts fp32 RVC graphs near 1e-3 of PyTorch (10× over the 1e-4 fp32 parity gate,
//! P1 outcomes, but inside the fp16 budget the GPU path already accepts) and halves the CUDA cost
//! of the default voice (benchmarked by `gpu_lever_bench`); an `fp32` request gets bit parity. Plan
//! sessions grow their CUDA arena only by what they request (T3).
//!
//! What a plan holds on the GPU is measured at build (process VRAM delta on its adapter, through
//! DXGI, from before its stream is created to after its first run) and held in [`arena::ledger`]
//! under its component until the plan is dropped. The plan's CUDA stream (and the CUDA context it
//! retains) is created only when the CUDA rung's preflight and canary pass.
//!
//! # Frozen by P3-S0 (signatures; P3-C2 implements the levers)
//! [`TensorSpec`], [`ExternalWeights`], [`PlanSpec`], [`SessionPlan::build`],
//! [`SessionPlan::outcome`], [`SessionPlan::levers`], [`SessionPlan::input_f32`],
//! [`SessionPlan::input_i64`], [`SessionPlan::run`], [`SessionPlan::output`].
//!
//! # Added by P3-C2
//! [`SessionPlan::notes`], [`SessionPlan::residency`] ([`Residency`]), [`SessionPlan::warm_up`],
//! [`SessionPlan::vram_mb`], [`SessionPlan::set_output_download`], [`SessionPlan::copy_rows_to`]
//! ([`RowCopy`]: an output window straight into another plan's input, device to device when both
//! plans are device-resident on one GPU, e.g. `enc`'s `m_p` / `logs_p` into `flowdec`).
//!
//! # Added by P3B-G
//! - Asynchronous device runs: [`SessionPlan::run_async`] enqueues uploads, the graph and the
//!   downloads and records the plan's `done` event (`CU_EVENT_BLOCKING_SYNC`); [`SessionPlan::wait`]
//!   (and every host access to the plan's buffers: `input_f32`, `input_i64`, `output`, ...) waits
//!   for it the way [`CudaTuning::wait`] says (`cuda_wait`: spin / block / hybrid);
//!   [`SessionPlan::poll`] settles a run that already finished. A device-to-device
//!   [`SessionPlan::copy_rows_to`] makes the destination's stream wait for the source's `done`
//!   event (no host wait) and records the source's `read_done` event after the copy, which the
//!   source's next run waits for on its stream before it overwrites its outputs. Both directions of
//!   that chain are covered by `gpu::tests::event_chain_orders_device_copies_without_host_waits`.
//! - GPU time per run (CUDA events around the graph replay, `telemetry::record_gpu_time` under the
//!   plan's component) while `telemetry::gpu_timing_enabled`; host waits are counted only when the
//!   run had not finished yet (`telemetry::record_sync`), plus the eager waits of host-I/O plans on
//!   a GPU rung (P3B-A's request).
//! - [`SHRINK_BEFORE_CAPTURE`] / `SIGIL_CUDA_TUNING=shrink=1`: the capture's first warm-up run gives
//!   the cuDNN search workspace back to the driver (arena shrinkage) before the second warm-up and
//!   the capture; never on or after the capture.
//! - [`SessionPlan::build_with`] (explicit [`CudaTuning`]), [`SessionPlan::set_vram_mb`],
//!   [`SessionPlan::arena_stats`], [`SessionPlan::wait_mode`]; the plans' arena statistics after
//!   their first runs (`telemetry::arena_mark_session`), and `SIGIL_GPU_PROCESS_PRIORITY`
//!   (`sched_priority`) applied at the first plan build.

use std::borrow::Cow;
use std::cell::Cell;
use std::ffi::{c_void, CString};
use std::sync::Arc;

use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{IoBinding, Session, SessionInputValue};
use ort::value::{DynValue, Tensor, TensorElementType, TensorRef};
use sha2::{Digest, Sha256};

use super::arena::{self, VramHold, VramProbe};
use super::cuda_driver::{
    ContextScope, CuDevicePtr, CuStream, CudaDriver, CudaEvent, PrimaryContext, EVENT_BLOCKING_SYNC, EVENT_DEFAULT, EVENT_DISABLE_TIMING,
};
use super::cuda_graph::{self, CapturedGraph};
use super::cuda_wait::{self, WaitMode, Waiter};
use super::cuda_stream::PriorityStream;
use super::dml_queue::DmlQueue;
use super::io_binding;
use super::trt::{self, TrtRequest};
use super::GpuLevers;
use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::ort_session::{
    build_session_ext, check_gpu_ep, device_for, ep_label, ep_ladder, normalize_ep, CudaConvSearch, GpuExtras, ModelSource,
    VcSessionOptions, EP_CPU, EP_CUDA, EP_DML, EP_TRT_CLASSIC, EP_TRT_RTX,
};
use crate::services::neural_audio::telemetry::{self, LoadStage, SyncKind};
use crate::services::neural_audio::vc::engine::options::Accelerator;

/// One input or output: name, element type and the fixed shape of the plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    /// `Float32` or `Int64` (the voice-engine graphs use only these at their I/O).
    pub dtype: TensorElementType,
    pub shape: Vec<usize>,
}

impl TensorSpec {
    pub fn f32(name: &str, shape: &[usize]) -> TensorSpec {
        TensorSpec { name: name.to_string(), dtype: TensorElementType::Float32, shape: shape.to_vec() }
    }

    pub fn i64(name: &str, shape: &[usize]) -> TensorSpec {
        TensorSpec { name: name.to_string(), dtype: TensorElementType::Int64, shape: shape.to_vec() }
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes of one element (`Float32` / `Int64`; others are refused at build).
    fn elem_bytes(&self) -> usize {
        if self.dtype == TensorElementType::Int64 {
            8
        } else {
            4
        }
    }

    fn bytes(&self) -> usize {
        self.len() * self.elem_bytes()
    }
}

/// ONNX external data supplied from memory (`location` = `file` in the graph; ADR 0004). Only
/// the prefix the graph reads needs to be passed.
#[derive(Clone, Copy, Debug)]
pub struct ExternalWeights<'a> {
    pub file: &'a str,
    pub image: &'a [u8],
}

/// Everything a plan is built from.
pub struct PlanSpec<'a> {
    /// Diagnostics component (`vc.enc`, `vc.flowdec`, `vc.knn`, ...).
    pub component: &'a str,
    /// Requested EP ladder id (`cpu` / `cuda` / `dml`).
    pub ep: &'a str,
    pub opts: &'a VcSessionOptions,
    pub model: ModelSource<'a>,
    pub weights: Option<ExternalWeights<'a>>,
    /// Free-dimension overrides (`("T", 100)`, ...).
    pub dims: &'a [(&'a str, i64)],
    pub inputs: &'a [TensorSpec],
    pub outputs: &'a [TensorSpec],
    pub levers: GpuLevers,
}

/// CUDA-rung session tuning of every plan (the benchmark decides these defaults, see
/// [`CudaTuning::DEFAULT`]). `SIGIL_CUDA_TUNING` overrides them for development and benchmarking:
/// comma-separated `tf32=0|1`, `nhwc=0|1`, `nc1d=0|1`, `algo=exhaustive|heuristic|default`,
/// `maxws=0|1`, `devio=0|1`, and (P3B-G) `wait=spin|block|hybrid`, `spinus=<µs>` (the hybrid
/// wait's spin budget) and `shrink=0|1` (give the cuDNN search workspace back before the graph
/// capture); unknown keys are ignored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaTuning {
    /// TF32 math (`use_tf32`) regardless of the levers (`None` = [`GpuLevers::tf32`]).
    pub tf32: Option<bool>,
    /// `prefer_nhwc` (`None` = ORT's default).
    pub prefer_nhwc: Option<bool>,
    /// `cudnn_conv1d_pad_to_nc1d` (`None` = ORT's default).
    pub conv1d_nc1d: Option<bool>,
    /// `cudnn_conv_algo_search` (`None` = ORT's default, exhaustive).
    pub algo: Option<CudaConvSearch>,
    /// `cudnn_conv_use_max_workspace` (`None` = ORT's default, on).
    pub max_workspace: Option<bool>,
    /// Keep device-resident buffers on CUDA when no graph is captured (`None` =
    /// [`DEVICE_IO_WITHOUT_GRAPHS`]).
    pub device_io: Option<bool>,
    /// How the plan's thread waits for its CUDA work (`None` = [`WaitMode::DEFAULT`]; P3B-G).
    pub wait: Option<WaitMode>,
    /// The hybrid wait's spin budget in µs (`None` = [`cuda_wait::HYBRID_SPIN_US`]).
    pub spin_us: Option<f64>,
    /// Shrink the session's CUDA arena after the first warm-up run of a graph capture (`None` =
    /// [`SHRINK_BEFORE_CAPTURE`]; P3B-G, T3).
    pub shrink: Option<bool>,
}

/// Whether a CUDA-graph plan gives the arena regions its first warm-up run left free (the cuDNN
/// algorithm search's workspace) back to the driver before the second warm-up and the capture
/// (P3B-G, T3). The second warm-up runs with the algorithms the search cached, so the arena grows
/// back to exactly what the chosen kernels and activations need, and the capture allocates nothing.
/// Never on or after the capture: replays write into addresses captured from the arena.
///
/// On since P3B-I0 (P3B-G, 7 ABBA pairs on the levers row): -40 MB of VRAM on every pair (1380 ->
/// 1340 MB), call p95 +2.1 % [-8.8, +7.5] NEUTRAL, graphs kept and the replays identical
/// (`gpu::tests::shrink_before_capture_keeps_graphs`). `SIGIL_CUDA_TUNING=shrink=0` turns it off.
pub const SHRINK_BEFORE_CAPTURE: bool = true;

/// Whether a CUDA plan without a CUDA graph (the lever off, or a capture that failed its check)
/// keeps device-resident buffers. Off: such plans bind host tensors (the bound path), because the
/// device path is not faster without a graph (`gpu_lever_bench`, release build, see
/// `docs/benchmarks/VC_BENCHMARKS.md`) and the bound path does not depend on ORT running on the
/// plan's stream.
pub const DEVICE_IO_WITHOUT_GRAPHS: bool = false;

/// Development override of [`CudaTuning`].
pub const CUDA_TUNING_ENV: &str = "SIGIL_CUDA_TUNING";

impl CudaTuning {
    pub const DEFAULT: CudaTuning = CudaTuning {
        tf32: None,
        prefer_nhwc: None,
        conv1d_nc1d: None,
        algo: None,
        max_workspace: None,
        device_io: None,
        wait: None,
        spin_us: None,
        shrink: None,
    };

    /// [`CudaTuning::DEFAULT`] with the overrides of `spec` (the `SIGIL_CUDA_TUNING` syntax).
    pub fn parse(spec: &str) -> CudaTuning {
        let mut t = CudaTuning::DEFAULT;
        for part in spec.split(',') {
            let Some((k, v)) = part.split_once('=') else { continue };
            let v = v.trim().to_ascii_lowercase();
            let on = matches!(v.as_str(), "1" | "on" | "true" | "yes");
            match k.trim().to_ascii_lowercase().as_str() {
                "tf32" => t.tf32 = Some(on),
                "nhwc" => t.prefer_nhwc = Some(on),
                "nc1d" => t.conv1d_nc1d = Some(on),
                "maxws" => t.max_workspace = Some(on),
                "devio" => t.device_io = Some(on),
                "wait" => t.wait = WaitMode::parse(&v).or(t.wait),
                "spinus" => t.spin_us = v.parse::<f64>().ok().filter(|x| x.is_finite() && *x >= 0.0).or(t.spin_us),
                "shrink" => t.shrink = Some(on),
                "algo" => {
                    t.algo = match v.as_str() {
                        "exhaustive" => Some(CudaConvSearch::Exhaustive),
                        "heuristic" => Some(CudaConvSearch::Heuristic),
                        "default" => Some(CudaConvSearch::Default),
                        _ => t.algo,
                    }
                }
                _ => {}
            }
        }
        t
    }

    /// The tuning plans use: the environment override, else the default.
    pub fn current() -> CudaTuning {
        match std::env::var(CUDA_TUNING_ENV) {
            Ok(spec) if !spec.trim().is_empty() => CudaTuning::parse(&spec),
            _ => CudaTuning::DEFAULT,
        }
    }

    /// See [`DEVICE_IO_WITHOUT_GRAPHS`].
    pub fn device_io_without_graphs(&self) -> bool {
        self.device_io.unwrap_or(DEVICE_IO_WITHOUT_GRAPHS)
    }

    /// The wait of the plans' CUDA work (see [`cuda_wait`]).
    pub fn wait_mode(&self) -> WaitMode {
        self.wait.unwrap_or(WaitMode::DEFAULT)
    }

    /// See [`SHRINK_BEFORE_CAPTURE`].
    pub fn shrink_before_capture(&self) -> bool {
        self.shrink.unwrap_or(SHRINK_BEFORE_CAPTURE)
    }
}

/// Where a plan's I/O lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    /// Legacy: host tensors created per run.
    Host,
    /// Host tensors bound once (IOBinding; ORT uploads them on a GPU rung).
    Bound,
    /// Device tensors at fixed addresses (IOBinding on CUDA; CUDA graphs possible).
    Device,
}

/// A strided window of rows from an output into an input (see [`SessionPlan::copy_rows_to`]): row
/// `r` of the destination (`dst_stride` elements apart) gets `row_len` elements of the source row
/// `r` (`src_stride` apart) starting at column `src_offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowCopy {
    pub rows: usize,
    pub row_len: usize,
    pub src_stride: usize,
    pub src_offset: usize,
    pub dst_stride: usize,
}

impl RowCopy {
    fn check(&self, src_len: usize, dst_len: usize) -> Result<(), String> {
        let src_end = self
            .rows
            .checked_sub(1)
            .map(|r| r * self.src_stride + self.src_offset + self.row_len)
            .unwrap_or(0);
        let dst_end = self.rows.checked_sub(1).map(|r| r * self.dst_stride + self.row_len).unwrap_or(0);
        if self.row_len > self.dst_stride || src_end > src_len || dst_end > dst_len || self.src_offset + self.row_len > self.src_stride {
            return Err(format!("row copy {self:?} does not fit a {src_len}-value source and a {dst_len}-value destination"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Host path (legacy)
// ---------------------------------------------------------------------------------------------

enum HostBuffer {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

impl HostBuffer {
    fn for_spec(spec: &TensorSpec) -> Result<HostBuffer, String> {
        match spec.dtype {
            TensorElementType::Float32 => Ok(HostBuffer::F32(vec![0.0; spec.len()])),
            TensorElementType::Int64 => Ok(HostBuffer::I64(vec![0; spec.len()])),
            other => Err(format!("input '{}': element type {other:?} is not supported by session plans", spec.name)),
        }
    }
}

struct HostIo {
    inputs: Vec<(TensorSpec, HostBuffer)>,
    outputs: Vec<(TensorSpec, Vec<f32>)>,
}

impl HostIo {
    fn new(inputs: &[TensorSpec], outputs: &[TensorSpec]) -> Result<HostIo, String> {
        Ok(HostIo {
            inputs: inputs
                .iter()
                .map(|s| HostBuffer::for_spec(s).map(|b| (s.clone(), b)))
                .collect::<Result<Vec<_>, _>>()?,
            outputs: outputs.iter().map(|s| (s.clone(), vec![0.0; s.len()])).collect(),
        })
    }

    fn run(&mut self, session: &mut Session) -> Result<(), String> {
        let mut feed: Vec<(String, SessionInputValue<'_>)> = Vec::with_capacity(self.inputs.len());
        for (spec, buf) in &self.inputs {
            let shape = spec.shape.clone();
            let value: SessionInputValue<'_> = match buf {
                HostBuffer::F32(b) => TensorRef::from_array_view((shape, b.as_slice()))
                    .map_err(|e| format!("input '{}': {e}", spec.name))?
                    .into(),
                HostBuffer::I64(b) => TensorRef::from_array_view((shape, b.as_slice()))
                    .map_err(|e| format!("input '{}': {e}", spec.name))?
                    .into(),
            };
            feed.push((spec.name.clone(), value));
        }
        let out = session.run(feed).map_err(|e| format!("run: {e}"))?;
        for (spec, buf) in &mut self.outputs {
            let (_, data) = out[spec.name.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("output '{}': {e}", spec.name))?;
            if data.len() != buf.len() {
                return Err(format!("output '{}': {} values, the plan expects {}", spec.name, data.len(), buf.len()));
            }
            buf.copy_from_slice(data);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Bound path (host tensors bound once)
// ---------------------------------------------------------------------------------------------

/// A host tensor bound to the plan's binding; `ptr` is its data (stable: the tensor owns it).
struct BoundTensor {
    spec: TensorSpec,
    name: CString,
    value: DynValue,
    ptr: *mut u8,
    /// Written since the last run (GPU rungs re-bind it so ORT uploads the new data).
    dirty: bool,
}

impl BoundTensor {
    fn new(spec: &TensorSpec) -> Result<BoundTensor, String> {
        let shape: Vec<i64> = spec.shape.iter().map(|&d| d as i64).collect();
        let mut value = match spec.dtype {
            TensorElementType::Float32 => Tensor::from_array((shape, vec![0f32; spec.len()])).map(Tensor::into_dyn),
            TensorElementType::Int64 => Tensor::from_array((shape, vec![0i64; spec.len()])).map(Tensor::into_dyn),
            other => return Err(format!("'{}': element type {other:?} is not supported by session plans", spec.name)),
        }
        .map_err(|e| format!("'{}': {e}", spec.name))?;
        let ptr = match spec.dtype {
            TensorElementType::Int64 => value.try_extract_tensor_mut::<i64>().map(|(_, d)| d.as_mut_ptr().cast::<u8>()),
            _ => value.try_extract_tensor_mut::<f32>().map(|(_, d)| d.as_mut_ptr().cast::<u8>()),
        }
        .map_err(|e| format!("'{}': {e}", spec.name))?;
        Ok(BoundTensor { spec: spec.clone(), name: io_binding::c_name(&spec.name)?, value, ptr, dirty: true })
    }
}

struct BoundIo {
    // Declared first: the binding's references go before the values.
    binding: IoBinding,
    inputs: Vec<BoundTensor>,
    outputs: Vec<BoundTensor>,
    /// GPU rung: ORT copies an input when it is bound, so written inputs are re-bound before a run.
    rebind: bool,
}

impl BoundIo {
    fn new(session: &Session, inputs: &[TensorSpec], outputs: &[TensorSpec], rebind: bool) -> Result<BoundIo, String> {
        let mut binding = session.create_binding().map_err(|e| format!("IOBinding: {e}"))?;
        let inputs = inputs.iter().map(BoundTensor::new).collect::<Result<Vec<_>, _>>()?;
        let outputs = outputs.iter().map(BoundTensor::new).collect::<Result<Vec<_>, _>>()?;
        for t in &inputs {
            io_binding::bind_input(&mut binding, &t.name, &t.value).map_err(|e| format!("bind input '{}': {e}", t.spec.name))?;
        }
        for t in &outputs {
            io_binding::bind_output(&mut binding, &t.name, &t.value).map_err(|e| format!("bind output '{}': {e}", t.spec.name))?;
        }
        let mut io = BoundIo { binding, inputs, outputs, rebind };
        for t in &mut io.inputs {
            t.dirty = false;
        }
        Ok(io)
    }

    fn run(&mut self, session: &mut Session) -> Result<(), String> {
        if self.rebind {
            for t in self.inputs.iter_mut().filter(|t| t.dirty) {
                io_binding::bind_input(&mut self.binding, &t.name, &t.value).map_err(|e| format!("input '{}': {e}", t.spec.name))?;
                t.dirty = false;
            }
        }
        io_binding::run(session, &self.binding).map_err(|e| format!("run: {e}"))
    }
}

// ---------------------------------------------------------------------------------------------
// Device path (CUDA)
// ---------------------------------------------------------------------------------------------

/// Host memory for one tensor: page-locked when the driver allows (faster async copies), else a
/// heap buffer. 8-byte aligned either way.
struct HostMirror {
    ptr: *mut u8,
    bytes: usize,
    pinned: Option<Arc<PrimaryContext>>,
    _heap: Vec<u64>,
}

impl HostMirror {
    fn new(bytes: usize, ctx: &Arc<PrimaryContext>) -> HostMirror {
        match ctx.alloc_host(bytes) {
            Ok(p) => {
                // SAFETY: fresh allocation of `bytes` (at least 1) bytes.
                unsafe { std::ptr::write_bytes(p.cast::<u8>(), 0, bytes) };
                HostMirror { ptr: p.cast(), bytes, pinned: Some(Arc::clone(ctx)), _heap: Vec::new() }
            }
            Err(e) => {
                tracing::debug!(target: "sigil::gpu", "pinned host memory unavailable ({e}); using pageable memory");
                let mut heap = vec![0u64; bytes.div_ceil(8).max(1)];
                HostMirror { ptr: heap.as_mut_ptr().cast(), bytes, pinned: None, _heap: heap }
            }
        }
    }

    fn is_pinned(&self) -> bool {
        self.pinned.is_some()
    }
}

impl Drop for HostMirror {
    fn drop(&mut self) {
        if let Some(ctx) = &self.pinned {
            ctx.free_host(self.ptr.cast::<c_void>());
        }
    }
}

struct DeviceTensor {
    spec: TensorSpec,
    name: CString,
    value: DynValue,
    dptr: CuDevicePtr,
    host: HostMirror,
    /// Input: the host mirror is newer than the device (uploaded before the next run). Output:
    /// unused.
    dirty: bool,
    /// Input: the device holds data the host mirror does not (a device row copy wrote it); the
    /// mirror is refreshed from the device before the caller touches it, so a later upload never
    /// overwrites the copied rows with stale host data.
    device_newer: bool,
    /// Output: copy it to the host mirror after each run.
    download: bool,
}

impl DeviceTensor {
    /// The host mirror as `len` f32 values (outputs, f32 inputs).
    ///
    /// # Safety
    /// The tensor must be f32 and no device copy into the mirror may be in flight.
    unsafe fn mirror_f32(&self) -> &[f32] {
        // SAFETY: per the caller; the mirror holds `len` 4-byte values.
        unsafe { std::slice::from_raw_parts(self.host.ptr.cast::<f32>(), self.spec.len()) }
    }
}

/// Bits of the NaN the graph check writes into the device outputs before a replay: a replay that
/// does not compute an output leaves it, and no regular run produces it where the reference is
/// finite.
const CHECK_SENTINEL: u32 = 0x7fc0_0b0e;

/// Deterministic check input `j` of pattern `seed` (1, 2, ...): f32 in [-1, 1), i64 in {0, 1}
/// (a valid index for any embedding or gather).
fn check_value(seed: u32, j: usize) -> u32 {
    // splitmix32-style hash of (seed, j).
    let mut x = (j as u32).wrapping_mul(0x9e37_79b9) ^ seed.wrapping_mul(0x85eb_ca6b);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

struct DeviceIo {
    // Drop order: the graph (it references the buffers), the binding, the device tensors (freed
    // through the allocator), the host mirrors (need the context), the events, and the allocator
    // last.
    graph: Option<CapturedGraph>,
    binding: IoBinding,
    inputs: Vec<DeviceTensor>,
    outputs: Vec<DeviceTensor>,
    ctx: Arc<PrimaryContext>,
    /// The plan's stream: ORT's compute stream, the copies' stream and the graph's stream.
    stream: CuStream,
    /// P3B-G: recorded after each run's last copy (`CU_EVENT_BLOCKING_SYNC | DISABLE_TIMING`): what
    /// the host waits on ([`Waiter`]) and what a consumer's stream waits on before copying this
    /// plan's outputs ([`SessionPlan::copy_rows_to`]).
    done: CudaEvent,
    /// P3B-G: recorded on a consumer's stream after its device row copy of this plan's outputs;
    /// this plan's next run waits for it on the device before it overwrites them.
    read_done: CudaEvent,
    /// P3B-G: start / end timing events of a run (created when GPU timing is first asked for).
    timing: Option<(CudaEvent, CudaEvent)>,
    /// The pending run recorded the timing events.
    timed: Cell<bool>,
    /// The last completed run was timed (its timing events hold its start and end).
    last_timed: Cell<bool>,
    /// A run is enqueued and not yet waited for ([`run_async`](Self::run_async)).
    pending: Cell<bool>,
    waiter: Waiter,
    /// The plan's diagnostics component (GPU time per run, `telemetry::record_gpu_time`).
    component: &'static str,
    _allocator: Allocator,
}

impl DeviceIo {
    /// Device tensors for every input and output, bound once. The caller has checked that the
    /// session reads every input from device memory (`io_binding::inputs_on_device`): ORT copies a
    /// bound value to the device its consumers read it from at bind time, so a host-read input
    /// bound here would keep that bind-time copy forever.
    fn new(
        session: &Session,
        device: i32,
        inputs: &[TensorSpec],
        outputs: &[TensorSpec],
        stream: &PriorityStream,
        waiter: Waiter,
        component: &'static str,
    ) -> Result<DeviceIo, String> {
        let ctx = Arc::clone(stream.context());
        let info = MemoryInfo::new(AllocationDevice::CUDA, device, AllocatorType::Device, MemoryType::Default)
            .map_err(|e| format!("CUDA memory info: {e}"))?;
        let allocator = Allocator::new(session, info).map_err(|e| format!("the session's CUDA allocator: {e}"))?;
        let mut binding = session.create_binding().map_err(|e| format!("IOBinding: {e}"))?;
        let make = |spec: &TensorSpec| -> Result<DeviceTensor, String> {
            let shape: Vec<i64> = spec.shape.iter().map(|&d| d as i64).collect();
            let mut value = match spec.dtype {
                TensorElementType::Float32 => Tensor::<f32>::new(&allocator, shape).map(Tensor::into_dyn),
                TensorElementType::Int64 => Tensor::<i64>::new(&allocator, shape).map(Tensor::into_dyn),
                other => return Err(format!("'{}': element type {other:?} is not supported by session plans", spec.name)),
            }
            .map_err(|e| format!("device tensor '{}': {e}", spec.name))?;
            if value.memory_info().allocation_device() != AllocationDevice::CUDA {
                return Err(format!("device tensor '{}' is not in CUDA memory", spec.name));
            }
            let dptr = value.data_ptr_mut() as CuDevicePtr;
            if dptr == 0 && spec.bytes() > 0 {
                return Err(format!("device tensor '{}' has no memory", spec.name));
            }
            Ok(DeviceTensor {
                spec: spec.clone(),
                name: io_binding::c_name(&spec.name)?,
                value,
                dptr,
                host: HostMirror::new(spec.bytes(), &ctx),
                dirty: true,
                device_newer: false,
                download: true,
            })
        };
        let inputs = inputs.iter().map(make).collect::<Result<Vec<_>, _>>()?;
        let outputs = outputs.iter().map(make).collect::<Result<Vec<_>, _>>()?;
        for t in &inputs {
            io_binding::bind_input(&mut binding, &t.name, &t.value).map_err(|e| format!("bind input '{}': {e}", t.spec.name))?;
        }
        for t in &outputs {
            io_binding::bind_output(&mut binding, &t.name, &t.value).map_err(|e| format!("bind output '{}': {e}", t.spec.name))?;
        }
        let done = CudaEvent::new(&ctx, EVENT_BLOCKING_SYNC | EVENT_DISABLE_TIMING)?;
        let read_done = CudaEvent::new(&ctx, EVENT_DISABLE_TIMING)?;
        Ok(DeviceIo {
            graph: None,
            binding,
            inputs,
            outputs,
            ctx,
            stream: stream.handle(),
            done,
            read_done,
            timing: None,
            timed: Cell::new(false),
            last_timed: Cell::new(false),
            pending: Cell::new(false),
            waiter,
            component,
            _allocator: allocator,
        })
    }

    fn pinned(&self) -> bool {
        self.inputs.iter().chain(&self.outputs).all(|t| t.host.is_pinned())
    }

    fn cuda_err(&self, what: &str, code: i32) -> String {
        format!("{what}: {}", self.ctx.driver().describe(code))
    }

    /// Enqueue the uploads of the inputs written since the last run; `true` when anything went up.
    fn upload(&mut self, scope: &ContextScope<'_>) -> Result<bool, String> {
        let mut any = false;
        for t in self.inputs.iter_mut().filter(|t| t.dirty) {
            // SAFETY: `dptr` is a device tensor of `bytes` this plan owns; the pinned / heap mirror
            // stays alive and unwritten until the run is waited for (every host access to an input
            // mirror waits first: `input_bytes`, `fill_inputs`).
            let r = unsafe { scope.copy_htod(t.dptr, t.host.ptr.cast::<c_void>(), t.host.bytes, self.stream) };
            if let Err(code) = r {
                return Err(format!("upload '{}': {}", t.spec.name, self.ctx.driver().describe(code)));
            }
            t.dirty = false;
            any = true;
        }
        Ok(any)
    }

    /// Enqueue the downloads of the outputs the caller reads (no wait).
    fn download(&self, scope: &ContextScope<'_>) -> Result<(), String> {
        for t in self.outputs.iter().filter(|t| t.download) {
            // SAFETY: the device output holds `bytes` written by the run on this stream; the mirror
            // has room for them and is not read before the run is waited for (`output` waits).
            let r = unsafe { scope.copy_dtoh(t.host.ptr.cast::<c_void>(), t.dptr, t.host.bytes, self.stream) };
            if let Err(code) = r {
                return Err(format!("download '{}': {}", t.spec.name, self.ctx.driver().describe(code)));
            }
        }
        Ok(())
    }

    /// One tick: uploads, the graph (or an ORT run), downloads, then the wait.
    fn run(&mut self, session: &mut Session) -> Result<(), String> {
        self.run_async(session)?;
        self.wait()
    }

    /// [`run`](Self::run) without the wait: the run is enqueued on the plan's stream and
    /// [`wait`](Self::wait) (or any host access to the plan's buffers) completes it.
    fn run_async(&mut self, session: &mut Session) -> Result<(), String> {
        let graph = self.graph.take();
        let r = self.launch(session, graph.as_ref(), false, None);
        self.graph = graph;
        r
    }

    /// Enqueue one run: `graph` replayed instead of the ORT run when given; `poison` first fills
    /// the device outputs with [`CHECK_SENTINEL`] (the graph check); `options` are the ORT run's
    /// options (the arena shrink of the first capture warm-up). Waits for a previous run first.
    fn launch(
        &mut self,
        session: &mut Session,
        graph: Option<&CapturedGraph>,
        poison: bool,
        options: Option<&ort::session::RunOptions>,
    ) -> Result<(), String> {
        self.wait()?;
        let r = self.enqueue(session, graph, poison, options);
        if r.is_err() {
            // Copies enqueued before the failure may still read the host mirrors: let them finish
            // before the caller touches any buffer again (best effort; the error is what counts).
            if let Ok(scope) = self.ctx.enter() {
                let _ = scope.sync(self.stream);
            }
        }
        r
    }

    /// The body of [`launch`](Self::launch) (no wait for a previous run).
    fn enqueue(
        &mut self,
        session: &mut Session,
        graph: Option<&CapturedGraph>,
        poison: bool,
        options: Option<&ort::session::RunOptions>,
    ) -> Result<(), String> {
        let ctx = Arc::clone(&self.ctx);
        let scope = ctx.enter().map_err(|r| self.cuda_err("cuCtxPushCurrent", r))?;
        // A consumer's device copy of the previous outputs comes before this run overwrites them.
        scope
            .stream_wait_event(self.stream, self.read_done.raw())
            .map_err(|code| self.cuda_err("cuStreamWaitEvent", code))?;
        let timed = graph.is_some() && telemetry::gpu_timing_enabled() && self.ensure_timing();
        if timed {
            if let Some((t0, _)) = &self.timing {
                scope.record_event(t0.raw(), self.stream).map_err(|code| self.cuda_err("cuEventRecord", code))?;
            }
        }
        let uploaded = self.upload(&scope)?;
        if poison {
            for t in &self.outputs {
                // SAFETY: an f32 device output of `len` 4-byte words this plan owns.
                if let Err(code) = unsafe { scope.memset_d32(t.dptr, CHECK_SENTINEL, t.spec.len(), self.stream) } {
                    return Err(format!("poison '{}': {}", t.spec.name, self.ctx.driver().describe(code)));
                }
            }
        }
        match graph {
            Some(graph) => {
                if let Err(code) = graph.launch(&scope, self.stream) {
                    return Err(self.cuda_err("graph launch", code));
                }
            }
            None => {
                // ORT enqueues on the plan's stream, after the uploads, when the provider applied
                // `user_compute_stream`; waiting for the uploads keeps the run correct when it did
                // not (ORT synchronises its own stream at the end of the run).
                if uploaded {
                    telemetry::record_sync(SyncKind::UploadWait);
                    scope.sync(self.stream).map_err(|code| self.cuda_err("stream sync", code))?;
                }
                match options {
                    Some(o) => io_binding::run_with(session, &self.binding, o),
                    None => io_binding::run(session, &self.binding),
                }
                .map_err(|e| format!("run: {e}"))?
            }
        }
        self.download(&scope)?;
        if timed {
            if let Some((_, t1)) = &self.timing {
                scope.record_event(t1.raw(), self.stream).map_err(|code| self.cuda_err("cuEventRecord", code))?;
            }
        }
        scope.record_event(self.done.raw(), self.stream).map_err(|code| self.cuda_err("cuEventRecord", code))?;
        self.timed.set(timed);
        self.pending.set(true);
        self.waiter.launched();
        Ok(())
    }

    /// The timing events, created on first use; `false` when the driver refused them.
    fn ensure_timing(&mut self) -> bool {
        if self.timing.is_none() {
            match (CudaEvent::new(&self.ctx, EVENT_DEFAULT), CudaEvent::new(&self.ctx, EVENT_DEFAULT)) {
                (Ok(a), Ok(b)) => self.timing = Some((a, b)),
                (Err(why), _) | (_, Err(why)) => {
                    tracing::debug!(target: "sigil::gpu", "plan [{}]: no GPU timing events ({why})", self.component);
                    return false;
                }
            }
        }
        true
    }

    /// Complete the pending run, if any: wait for it ([`Waiter`]: spin, block or hybrid), count
    /// the host wait (`kind`) when the run had not finished yet, and report its GPU time when it
    /// was timed. Allocation-free.
    fn wait_as(&self, kind: SyncKind) -> Result<(), String> {
        if !self.pending.get() {
            return Ok(());
        }
        let scope = self.ctx.enter().map_err(|r| self.cuda_err("cuCtxPushCurrent", r))?;
        let waited = self
            .waiter
            .wait(&scope, self.stream, self.done.raw())
            .map_err(|code| self.cuda_err("plan wait", code))?;
        if waited {
            telemetry::record_sync(kind);
        }
        self.pending.set(false);
        self.finish_timing(&scope);
        Ok(())
    }

    /// [`wait_as`](Self::wait_as) for a run's end.
    fn wait(&self) -> Result<(), String> {
        self.wait_as(SyncKind::RunEnd)
    }

    /// Complete the pending run without waiting when it has already finished (its GPU time is
    /// reported then); `true` when no run is pending afterwards. Allocation-free.
    fn poll(&self) -> Result<bool, String> {
        if !self.pending.get() {
            return Ok(true);
        }
        let scope = self.ctx.enter().map_err(|r| self.cuda_err("cuCtxPushCurrent", r))?;
        if !scope.query_event(self.done.raw()).map_err(|code| self.cuda_err("cuEventQuery", code))? {
            return Ok(false);
        }
        self.pending.set(false);
        self.finish_timing(&scope);
        Ok(true)
    }

    fn finish_timing(&self, scope: &ContextScope<'_>) {
        let timed = self.timed.replace(false);
        self.last_timed.set(timed);
        if !timed {
            return;
        }
        if let Some((t0, t1)) = &self.timing {
            if let Ok(ms) = scope.elapsed_ms(t0.raw(), t1.raw()) {
                telemetry::record_gpu_time(self.component, f64::from(ms));
            }
        }
    }

    /// Make input `i`'s host mirror current after a device row copy, so the caller can read or
    /// write it (synchronous; nothing to do otherwise). Allocation-free.
    fn refresh_mirror(&mut self, i: usize) -> Result<(), String> {
        if !self.inputs[i].device_newer {
            return Ok(());
        }
        let ctx = Arc::clone(&self.ctx);
        let scope = ctx.enter().map_err(|r| self.cuda_err("cuCtxPushCurrent", r))?;
        let t = &mut self.inputs[i];
        telemetry::record_sync(SyncKind::MirrorRefresh);
        // SAFETY: the device tensor holds `bytes`; the mirror has room for them and is not touched
        // before the sync below.
        let r = unsafe { scope.copy_dtoh(t.host.ptr.cast::<c_void>(), t.dptr, t.host.bytes, self.stream) }
            .and_then(|()| scope.sync(self.stream));
        if let Err(code) = r {
            return Err(format!("refresh '{}': {}", t.spec.name, ctx.driver().describe(code)));
        }
        t.device_newer = false;
        Ok(())
    }

    /// Strided device-to-device copy of `copy`'s rows from `src`'s output `src_index` into input
    /// `i`, on this plan's stream, ordered after `src`'s last run on the device (its `done` event)
    /// and marked for `src`'s next run (its `read_done` event): neither plan's host thread waits.
    /// Host writes made to that input since the last run go up first, so the rows overwrite only
    /// their own part. Allocation-free.
    fn copy_rows_from(&mut self, src: &DeviceIo, src_index: usize, copy: &RowCopy, i: usize) -> Result<(), String> {
        // A pending run of this plan read the input before: the copy follows it on the same
        // stream, so the host does not wait for it.
        let ctx = Arc::clone(&self.ctx);
        let scope = ctx.enter().map_err(|r| self.cuda_err("cuCtxPushCurrent", r))?;
        let stream = self.stream;
        let t = &mut self.inputs[i];
        if t.dirty {
            // SAFETY: as in `upload`; the mirror is not written again before the next wait
            // (`refresh_mirror` synchronises before any host access).
            if let Err(code) = unsafe { scope.copy_htod(t.dptr, t.host.ptr.cast::<c_void>(), t.host.bytes, stream) } {
                return Err(format!("upload '{}': {}", t.spec.name, ctx.driver().describe(code)));
            }
            t.dirty = false;
        }
        // RAW: after the source's run (a no-op when it already completed or never ran).
        scope
            .stream_wait_event(stream, src.done.raw())
            .map_err(|code| ctx.driver().describe(code))?;
        let from = src.outputs[src_index].dptr;
        // SAFETY: the caller checked both ranges against the two device tensors' lengths.
        let r = unsafe {
            scope.copy_2d_dtod(
                t.dptr,
                copy.dst_stride * 4,
                from + (copy.src_offset * 4) as u64,
                copy.src_stride * 4,
                copy.row_len * 4,
                copy.rows,
                stream,
            )
        };
        if let Err(code) = r {
            return Err(ctx.driver().describe(code));
        }
        // WAR: the source's next run waits for this copy before it overwrites its outputs.
        scope
            .record_event(src.read_done.raw(), stream)
            .map_err(|code| ctx.driver().describe(code))?;
        t.device_newer = true;
        Ok(())
    }

    /// Strided device-to-host copy of `copy`'s rows of output `src` (not downloaded by the run) into
    /// `dst` (another plan's host input buffer, checked by the caller); synchronous, ordered after
    /// this plan's last run on its stream. Allocation-free.
    fn read_rows(&self, src: &DeviceTensor, copy: &RowCopy, dst: *mut f32) -> Result<(), String> {
        {
            let scope = self.ctx.enter().map_err(|r| self.cuda_err("cuCtxPushCurrent", r))?;
            // SAFETY: the source range lies in the device output (checked by the caller), `dst` has
            // room for every destination row and is not touched before the wait below. The copy
            // follows this plan's last run on its stream.
            unsafe {
                scope.copy_2d_dtoh(
                    dst.cast::<c_void>(),
                    copy.dst_stride * 4,
                    src.dptr + (copy.src_offset * 4) as u64,
                    copy.src_stride * 4,
                    copy.row_len * 4,
                    copy.rows,
                    self.stream,
                )
            }
            .and_then(|()| scope.record_event(self.done.raw(), self.stream))
            .map_err(|code| self.ctx.driver().describe(code))?;
            if !self.pending.replace(true) {
                self.waiter.launched();
            }
        }
        self.wait_as(SyncKind::RowRead)
    }

    /// Fill the input mirrors with check pattern `seed` (0 = the zeros a plan starts with) and mark
    /// them for upload.
    fn fill_inputs(&mut self, seed: u32) -> Result<(), String> {
        self.wait()?;
        for t in &mut self.inputs {
            let n = t.spec.len();
            if t.spec.dtype == TensorElementType::Int64 {
                // SAFETY: an i64 mirror of `n` values (8-byte aligned), no copy in flight.
                let v = unsafe { std::slice::from_raw_parts_mut(t.host.ptr.cast::<i64>(), n) };
                for (j, x) in v.iter_mut().enumerate() {
                    *x = if seed == 0 { 0 } else { i64::from(check_value(seed, j) & 1) };
                }
            } else {
                // SAFETY: an f32 mirror of `n` values (4-byte aligned), no copy in flight.
                let v = unsafe { std::slice::from_raw_parts_mut(t.host.ptr.cast::<f32>(), n) };
                for (j, x) in v.iter_mut().enumerate() {
                    *x = if seed == 0 { 0.0 } else { (check_value(seed, j) >> 8) as f32 / (1u32 << 23) as f32 - 1.0 };
                }
            }
            t.dirty = true;
            t.device_newer = false;
        }
        Ok(())
    }

    /// The outputs of the last run (downloaded, f32).
    fn snapshot_outputs(&self) -> Result<Vec<Vec<f32>>, String> {
        self.wait()?;
        // SAFETY: f32 outputs downloaded by the last run, which the wait above completed.
        Ok(self.outputs.iter().map(|t| unsafe { t.mirror_f32() }.to_vec()).collect())
    }

    /// Record the session's run into a CUDA graph (see `cuda_graph`): warm-up runs, the recorded
    /// run, then [`check_graph`](Self::check_graph). The inputs are zeros again afterwards (what a
    /// plan starts with). On `Err` no graph is kept. With `shrink` (P3B-G, the CUDA device) the first
    /// warm-up run shrinks the session's arena when it ends (the cuDNN search's workspace goes back
    /// to the driver); the second runs with the cached algorithms, so the arena grows back to what
    /// the chosen kernels need before the capture, which then allocates nothing. ONNX Runtime
    /// validates the shrink request against the session's arena allocator and refuses a session
    /// that has none (a plugin EP's allocator, the registered provider pack): that run is repeated
    /// without the shrink instead of costing the plan its graph for 40 MB, and `Ok` carries a note
    /// saying so.
    fn capture(&mut self, session: &mut Session, shrink: Option<i32>) -> Result<Option<String>, String> {
        for t in &mut self.outputs {
            t.download = true;
        }
        let mut note = None;
        for k in 0..cuda_graph::WARMUP_RUNS {
            if k == 0 {
                note = first_warmup(shrink, |device| match device {
                    Some(device) => self.shrinking_run(session, device),
                    None => self.run(session),
                })?;
            } else {
                self.run(session)?;
            }
        }
        let mut options = ort::session::RunOptions::new().map_err(|e| format!("run options: {e}"))?;
        options.disable_device_sync().map_err(|e| format!("run options: {e}"))?;
        let graph = {
            let binding = &self.binding;
            CapturedGraph::capture(&self.ctx, self.stream, || io_binding::run_with(session, binding, &options))?
        };
        let checked = self.check_graph(session, &graph);
        self.fill_inputs(0)?;
        checked?;
        self.graph = Some(graph);
        Ok(note)
    }

    /// One regular run that shrinks the session's device arena (`gpu:<device>`) when it ends.
    fn shrinking_run(&mut self, session: &mut Session, device: i32) -> Result<(), String> {
        let mut options = ort::session::RunOptions::new().map_err(|e| format!("run options: {e}"))?;
        options
            .set("memory.enable_memory_arena_shrinkage", format!("gpu:{device}"))
            .map_err(|e| format!("run options: {e}"))?;
        self.launch(session, None, false, Some(&options))?;
        self.wait()
    }

    /// Whether `graph` replays the whole run for any inputs: two regular runs on two non-zero check
    /// patterns give the references, then each pattern is replayed with the device outputs set to
    /// [`CHECK_SENTINEL`] first. A graph that recorded nothing leaves the sentinel; one that recorded
    /// only part of the run (kernels ORT enqueued on another stream ran eagerly instead) replays the
    /// first pattern on intermediates the second regular run left, and differs; one that baked in
    /// host-side values (a node that ran on the CPU during the capture) differs on the second
    /// pattern.
    fn check_graph(&mut self, session: &mut Session, graph: &CapturedGraph) -> Result<(), String> {
        let mut references = Vec::with_capacity(2);
        for seed in [1, 2] {
            self.fill_inputs(seed)?;
            self.launch(session, None, false, None)
                .and_then(|()| self.wait())
                .map_err(|e| format!("the check run failed: {e}"))?;
            references.push(self.snapshot_outputs()?);
        }
        // Every output must be checkable: a replay that leaves the sentinel is told apart only where
        // the reference is finite.
        for (k, t) in self.outputs.iter().enumerate() {
            if !t.spec.is_empty() && !references.iter().any(|r| r[k].iter().any(|v| v.is_finite())) {
                return Err(format!("output '{}' has no finite value on the check inputs", t.spec.name));
            }
        }
        for (seed, reference) in [1u32, 2].into_iter().zip(&references) {
            self.fill_inputs(seed)?;
            self.launch(session, Some(graph), true, None)
                .and_then(|()| self.wait())
                .map_err(|e| format!("the graph replay failed: {e}"))?;
            let same = self.outputs.iter().zip(reference).all(|(t, r)| {
                // SAFETY: f32 outputs downloaded by the replay, which the wait above completed.
                cuda_graph::outputs_match(r, unsafe { t.mirror_f32() })
            });
            if !same {
                return Err(format!("the graph replay does not reproduce the regular run (check pattern {seed})"));
            }
        }
        Ok(())
    }
}

impl Drop for DeviceIo {
    fn drop(&mut self) {
        // A run or copies enqueued by `copy_rows_to` may still be in flight when a plan is dropped
        // without another wait: wait for them before the buffers go (the stream outlives this
        // object). A consumer's device copy of this plan's outputs runs on the consumer's stream:
        // wait for it too (`read_done`; an event never recorded is complete).
        if let Ok(scope) = self.ctx.enter() {
            let _ = scope.sync(self.stream);
            let _ = scope.sync_event(self.read_done.raw());
        }
    }
}

enum Io {
    Host(HostIo),
    Bound(BoundIo),
    Device(DeviceIo),
}

impl Io {
    /// One run. `gpu_rung`: the session runs on a GPU EP, where ORT's run ends in a host wait for
    /// the device (counted like the device path's waits, `telemetry::SyncKind::RunEnd`).
    fn run(&mut self, session: &mut Session, gpu_rung: bool) -> Result<(), String> {
        match self {
            Io::Host(h) => h.run(session).map(|()| count_eager_wait(gpu_rung)),
            Io::Bound(b) => b.run(session).map(|()| count_eager_wait(gpu_rung)),
            Io::Device(d) => d.run(session),
        }
    }

    /// [`run`](Self::run) without the device path's final wait (the other paths are synchronous).
    fn run_async(&mut self, session: &mut Session, gpu_rung: bool) -> Result<(), String> {
        match self {
            Io::Device(d) => d.run_async(session),
            other => other.run(session, gpu_rung),
        }
    }
}

/// A host-I/O plan run on a GPU rung waited for the device inside ONNX Runtime (P3B-A's sync count).
#[inline]
fn count_eager_wait(gpu_rung: bool) {
    if gpu_rung {
        telemetry::record_sync(SyncKind::RunEnd);
    }
}

/// One session at fixed shapes.
pub struct SessionPlan {
    // Drop order: the I/O (graph, binding and buffers) before the session, the session before the
    // stream and queue it runs on, the VRAM hold last.
    io: Io,
    session: Session,
    stream: Option<PriorityStream>,
    _dml_queue: Option<Arc<DmlQueue>>,
    outcome: EpOutcome,
    levers: GpuLevers,
    notes: Vec<String>,
    vram: Option<VramHold>,
    /// Device index of the rung the session runs on (CUDA ordinal, DXGI adapter, 0 on the CPU).
    device: i32,
}

// SAFETY: the raw pointers in the I/O paths point into buffers the plan owns (host tensors, pinned
// or heap mirrors) or are CUDA handles; the plan is used by one thread at a time (`&mut self` for
// every mutation), and ORT sessions and bindings are Send.
unsafe impl Send for SessionPlan {}

impl SessionPlan {
    /// Build the session on the best rung of `spec.ep`'s ladder and allocate the I/O buffers (and,
    /// with the `cuda_graphs` lever on CUDA, capture the graph). A plan on a GPU rung has run once
    /// when this returns (zero inputs; the graph check's runs on the device path), so its VRAM hold
    /// includes the first run's activations, workspaces and arena peak.
    pub fn build(spec: &PlanSpec<'_>) -> Result<SessionPlan, String> {
        Self::build_with(spec, CudaTuning::current())
    }

    /// [`build`](Self::build) with explicit CUDA tuning instead of `SIGIL_CUDA_TUNING` (tests and
    /// benchmarks compare tunings in one process).
    pub(crate) fn build_with(spec: &PlanSpec<'_>, tuning: CudaTuning) -> Result<SessionPlan, String> {
        // P3B-G: `SIGIL_GPU_PROCESS_PRIORITY` (an experiment; once per process).
        super::sched_priority::apply_from_env_once();
        let wants = spec.levers;
        for out in spec.outputs {
            if out.dtype != TensorElementType::Float32 {
                return Err(format!("output '{}': only f32 outputs are supported by session plans", out.name));
            }
        }
        for input in spec.inputs {
            if !matches!(input.dtype, TensorElementType::Float32 | TensorElementType::Int64) {
                return Err(format!("input '{}': element type {:?} is not supported by session plans", input.name, input.dtype));
            }
        }
        let base = normalize_ep(spec.ep);
        let cuda_on_ladder = ep_ladder(base).contains(&EP_CUDA);
        let cuda_device = spec.opts.cuda_device_index.max(0);
        let mut notes: Vec<String> = Vec::new();

        // What the plan adds on its adapter is measured from before anything is created for it:
        // the plan's stream and the CUDA context it retains count, and a stream dropped after a
        // fallback to DirectML gives back what it took.
        let probe = arena::dxgi_adapter_for(base, device_for(base, base, spec.opts)).and_then(VramProbe::start);

        // A plan stream (and the CUDA context it retains) only when the CUDA rung can run: the
        // preflight and canary say so (cached; the ladder runs the same checks and records why
        // CUDA is skipped). Otherwise no CUDA context is created and torn down per plan.
        let cuda_ready = cuda_on_ladder && (wants.priority || wants.io_binding) && check_gpu_ep(EP_CUDA, cuda_device).is_ok();
        // Device buffers need the CUDA driver and a plan stream (ORT's compute stream).
        let driver_ok = wants.io_binding
            && cuda_ready
            && match CudaDriver::get() {
                Ok(_) => true,
                Err(why) => {
                    notes.push(format!("io_binding: device-resident buffers need the CUDA driver ({why}); host buffers are bound instead"));
                    false
                }
            };
        if wants.cuda_graphs && !wants.io_binding {
            notes.push("cuda_graphs: needs io_binding (fixed device addresses); running without CUDA graphs".into());
        }
        let stream = if cuda_ready && (wants.priority || driver_ok) {
            let made = if wants.priority { PriorityStream::highest(cuda_device) } else { PriorityStream::normal(cuda_device) };
            match made {
                Ok(s) => Some(s),
                Err(why) if wants.priority => {
                    notes.push(format!("gpu_priority: the highest-priority CUDA stream could not be created ({why}); CUDA runs on its default stream"));
                    None
                }
                Err(why) => {
                    notes.push(format!("io_binding: the plan's CUDA stream could not be created ({why}); host buffers are bound instead"));
                    None
                }
            }
        } else {
            None
        };
        let device_io_possible = driver_ok && stream.is_some();
        let trt_request = match wants.accelerator {
            Accelerator::TensorRtRtx if base == EP_CUDA && trt::available() => Some(Arc::new(TrtRequest {
                model_key: model_key(spec),
                fp16: false,
            })),
            Accelerator::TensorRtRtx => {
                notes.push(if base == EP_CUDA {
                    "tensorrt_rtx: the TensorRT runtime pack is not installed or not usable; running on CUDA".to_string()
                } else {
                    format!("tensorrt_rtx: applies to CUDA requests only; running on {}", ep_label(base))
                });
                None
            }
            _ => None,
        };
        let extras = GpuExtras {
            cuda_stream: stream.as_ref().map(PriorityStream::raw),
            cuda_tf32: Some(tuning.tf32.unwrap_or(wants.tf32)),
            cuda_same_as_requested: true,
            cuda_conv_max_workspace: tuning.max_workspace,
            cuda_prefer_nhwc: tuning.prefer_nhwc,
            cuda_conv1d_pad_to_nc1d: tuning.conv1d_nc1d,
            cuda_conv_algo_search: tuning.algo,
            cuda_env_arena: false,
            dml_high_priority: wants.priority || wants.accelerator == Accelerator::DmlHighPriority,
            // P3B-G: decided per session by `build_session_ext` (`SIGIL_VC_DML_SHARED`).
            dml_shared: false,
            tensorrt_rtx: trt_request,
            skip_outcome_record: false,
        };
        let configure = |mut builder: ort::session::builder::SessionBuilder, _ep: &str| {
            if let Some(w) = spec.weights {
                builder = builder
                    .with_external_initializer_file_in_memory(w.file, Cow::Owned(w.image.to_vec()))
                    .map_err(|e| format!("external weights: {e}"))?;
            }
            for (name, size) in spec.dims {
                builder = builder
                    .with_dimension_override(*name, *size)
                    .map_err(|e| format!("dimension override {name}: {e}"))?;
            }
            Ok(builder)
        };
        let built = build_session_ext(Some(spec.component), spec.ep, spec.opts, spec.model, &extras, &configure)?;
        let active: &str = &built.outcome.active;
        let on_cuda = active == EP_CUDA;
        let on_trt = built.applied.tensorrt_rtx;
        debug_assert_eq!(on_trt, active == EP_TRT_RTX || active == EP_TRT_CLASSIC);
        notes.extend(built.applied.notes.iter().cloned());

        // Only a session on CUDA / TensorRT runs on the plan's stream; drop it otherwise.
        let stream = match stream {
            Some(s) if on_cuda || on_trt => Some(s),
            Some(_) if wants.priority => {
                notes.push(format!("gpu_priority: the session runs on {}, not CUDA", ep_label(active)));
                None
            }
            _ => None,
        };
        let mut session = built.session;

        // Device-resident buffers pay off only with a graph (`DEVICE_IO_WITHOUT_GRAPHS`), and need
        // every input read from device memory: ORT copies a bound value to the device its consumers
        // read it from at bind time, so an input a CUDA kernel takes on the CPU (a Reshape shape,
        // Slice bounds, ...) would keep its bind-time copy.
        let device_io = wants.io_binding
            && on_cuda
            && device_io_possible
            && stream.is_some()
            && (wants.cuda_graphs || tuning.device_io_without_graphs())
            && match io_binding::inputs_on_device(&session) {
                Ok(placement) => {
                    let host_read: Vec<&str> = placement
                        .iter()
                        .filter(|(name, on_device)| !on_device && spec.inputs.iter().any(|s| s.name == *name))
                        .map(|(name, _)| name.as_str())
                        .collect();
                    if host_read.is_empty() {
                        true
                    } else {
                        notes.push(format!(
                            "io_binding: the model reads {} on the CPU; host buffers are bound instead (no device-resident buffers or CUDA graphs)",
                            host_read.iter().map(|n| format!("'{n}'")).collect::<Vec<_>>().join(", ")
                        ));
                        false
                    }
                }
                Err(why) => {
                    notes.push(format!("io_binding: where the model reads its inputs is unknown ({why}); host buffers are bound instead"));
                    false
                }
            };
        let mut io = match &stream {
            _ if !wants.io_binding => Io::Host(HostIo::new(spec.inputs, spec.outputs)?),
            Some(s) if device_io => match DeviceIo::new(
                &session,
                built.device,
                spec.inputs,
                spec.outputs,
                s,
                Waiter::new(tuning.wait_mode(), tuning.spin_us.unwrap_or(cuda_wait::HYBRID_SPIN_US)),
                telemetry::intern_component(spec.component),
            ) {
                Ok(io) => Io::Device(io),
                Err(why) => {
                    notes.push(format!("io_binding: device buffers could not be set up ({why}); host buffers are bound instead"));
                    Io::Bound(BoundIo::new(&session, spec.inputs, spec.outputs, true)?)
                }
            },
            _ => Io::Bound(BoundIo::new(&session, spec.inputs, spec.outputs, active != EP_CPU)?),
        };
        let mut graphs_on = false;
        let mut keep_device = true;
        // P3B-S0 telemetry: the first runs and the graph capture of this plan.
        let first_runs_span = telemetry::load_span(LoadStage::FirstRunsCapture);
        match &mut io {
            Io::Device(d) => {
                if wants.cuda_graphs {
                    let shrink = tuning.shrink_before_capture().then_some(built.device);
                    #[cfg(test)]
                    let shrink = shrink.map(|device| test_hooks::shrink_device().unwrap_or(device));
                    match d.capture(&mut session, shrink) {
                        Ok(note) => {
                            graphs_on = true;
                            notes.extend(note);
                        }
                        Err(why) => notes.push(format!("cuda_graphs: graph capture failed ({why}); running without CUDA graphs")),
                    }
                }
                // Without a graph (a capture that failed its check) the plan binds host tensors:
                // the path that is also correct whatever stream ORT runs on.
                keep_device = graphs_on || (!wants.cuda_graphs && tuning.device_io_without_graphs());
            }
            _ if wants.cuda_graphs && wants.io_binding && !notes.iter().any(|n| n.starts_with("cuda_graphs:")) => {
                notes.push(format!("cuda_graphs: the session runs on {} without device buffers", ep_label(active)));
            }
            _ => {}
        }
        if !keep_device {
            io = Io::Bound(BoundIo::new(&session, spec.inputs, spec.outputs, true)?);
        }
        // One run on a GPU rung before the VRAM measurement (the graph check already ran the
        // session): the first run allocates activations, workspaces and the arena peak.
        if active != EP_CPU && !graphs_on {
            if let Err(why) = io.run(&mut session, true) {
                tracing::warn!(target: "sigil::gpu", "plan [{}]: the warm-up run on zero inputs failed ({why})", spec.component);
            }
        }
        drop(first_runs_span);
        // P3B-A's arena statistics (a no-op unless a benchmark records): what the session's CUDA
        // arena holds after its first runs or graph capture.
        if on_cuda {
            telemetry::arena_mark_session(spec.component, &session, built.device);
        }
        let priority =
            (wants.priority && (on_cuda || on_trt) && stream.is_some() && built.applied.cuda_stream) || built.applied.dml_queue.is_some();
        let accelerator = if on_trt {
            Accelerator::TensorRtRtx
        } else if active == EP_DML && built.applied.dml_queue.is_some() && wants.accelerator == Accelerator::DmlHighPriority {
            Accelerator::DmlHighPriority
        } else {
            Accelerator::None
        };
        if wants.priority && !priority && !notes.iter().any(|n| n.starts_with("gpu_priority:")) {
            notes.push(format!("gpu_priority: no priority scheduling on {}", ep_label(active)));
        }
        let levers = GpuLevers {
            io_binding: !matches!(io, Io::Host(_)),
            cuda_graphs: graphs_on,
            priority,
            accelerator,
            tf32: on_cuda && extras.cuda_tf32 == Some(true),
        };
        let vram = probe.filter(|_| active != EP_CPU).and_then(|p| p.delta_mb()).map(|mb| {
            let key = match spec.opts.component_prefix {
                Some(p) if !p.is_empty() => format!("{p}.{}", spec.component),
                _ => spec.component.to_string(),
            };
            arena::ledger().hold(&key, mb)
        });
        let plan = SessionPlan {
            io,
            session,
            stream,
            // The high-priority queue (or P3B-G's shared normal one) outlives the session.
            _dml_queue: built.applied.dml_queue.clone().or_else(|| built.applied.dml_shared.clone()),
            outcome: built.outcome.clone(),
            levers,
            notes,
            vram,
            device: built.device,
        };
        tracing::info!(
            target: "sigil::gpu",
            "plan [{}] on {}: {:?}, levers {:?}{}{}",
            spec.component,
            ep_label(active),
            plan.residency(),
            plan.levers,
            plan.vram_mb().map(|mb| format!(", {mb:.0} MB VRAM")).unwrap_or_default(),
            if plan.notes.is_empty() { String::new() } else { format!(" ({})", plan.notes.join("; ")) }
        );
        Ok(plan)
    }

    /// Where the session runs (and why, after a fallback).
    pub fn outcome(&self) -> &EpOutcome {
        &self.outcome
    }

    /// The GPU levers this plan really uses.
    pub fn levers(&self) -> GpuLevers {
        self.levers
    }

    /// Why requested levers are not in use (`<lever>: <reason>`), for the load warnings.
    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    /// Where the plan's I/O lives.
    pub fn residency(&self) -> Residency {
        match &self.io {
            Io::Host(_) => Residency::Host,
            Io::Bound(_) => Residency::Bound,
            Io::Device(_) => Residency::Device,
        }
    }

    /// Process VRAM the plan added on its adapter when it was built (MiB; `None` on the CPU or when
    /// DXGI cannot tell). Held in [`arena::ledger`] until the plan is dropped.
    pub fn vram_mb(&self) -> Option<f64> {
        self.vram.as_ref().map(VramHold::mb)
    }

    /// Replace the measured VRAM hold (MiB; P3B-G: plans built concurrently each measured the
    /// other's allocations too, and their builder splits a joint measurement). A no-op without a
    /// hold.
    pub fn set_vram_mb(&self, mb: f64) {
        if let Some(h) = &self.vram {
            h.update(mb);
        }
    }

    /// The session's CUDA arena statistics (`AllocatorGetStats`: `TotalAllocated`, `InUse`,
    /// `NumArenaShrinkages`, ... in bytes or counts; `None` off CUDA). Diagnostics and tests.
    pub fn arena_stats(&self) -> Option<std::collections::BTreeMap<String, String>> {
        (self.outcome.active == EP_CUDA)
            .then(|| telemetry::session_arena_stats(&self.session, self.device))
            .flatten()
    }

    /// Whether the device path's host mirrors are page-locked (diagnostics and tests).
    pub fn pinned_host_buffers(&self) -> bool {
        matches!(&self.io, Io::Device(d) if d.pinned())
    }

    /// The f32 input buffer `name` (fixed length), written before [`run`](Self::run).
    pub fn input_f32(&mut self, name: &str) -> Result<&mut [f32], String> {
        self.input_bytes(name, TensorElementType::Float32).map(|(p, n)| {
            // SAFETY: `p` points to `n` bytes of an f32 buffer the plan owns (4-byte aligned), and
            // the returned borrow is tied to `&mut self`.
            unsafe { std::slice::from_raw_parts_mut(p.cast::<f32>(), n / 4) }
        })
    }

    /// The i64 input buffer `name` (fixed length).
    pub fn input_i64(&mut self, name: &str) -> Result<&mut [i64], String> {
        self.input_bytes(name, TensorElementType::Int64).map(|(p, n)| {
            // SAFETY: as in `input_f32` (8-byte aligned i64 buffer).
            unsafe { std::slice::from_raw_parts_mut(p.cast::<i64>(), n / 8) }
        })
    }

    /// The host buffer of input `name` as `(pointer, bytes)`, marked written. Allocation-free.
    fn input_bytes(&mut self, name: &str, want: TensorElementType) -> Result<(*mut u8, usize), String> {
        let (dtype, ptr, bytes) = match &mut self.io {
            Io::Host(h) => match h.inputs.iter_mut().find(|(s, _)| s.name == name) {
                Some((s, HostBuffer::F32(b))) => (s.dtype, b.as_mut_ptr().cast::<u8>(), b.len() * 4),
                Some((s, HostBuffer::I64(b))) => (s.dtype, b.as_mut_ptr().cast::<u8>(), b.len() * 8),
                None => return Err(no_input(name)),
            },
            Io::Bound(b) => match b.inputs.iter_mut().find(|t| t.spec.name == name) {
                Some(t) => {
                    if t.spec.dtype == want {
                        t.dirty = true;
                    }
                    (t.spec.dtype, t.ptr, t.spec.bytes())
                }
                None => return Err(no_input(name)),
            },
            Io::Device(d) => match d.inputs.iter().position(|t| t.spec.name == name) {
                Some(i) => {
                    // A pending run may still be uploading this mirror: complete it first.
                    d.wait()?;
                    if d.inputs[i].spec.dtype == want {
                        // After a device row copy the mirror is stale: bring it up to date before
                        // the caller reads or writes it (the next upload sends all of it).
                        d.refresh_mirror(i)?;
                        d.inputs[i].dirty = true;
                    }
                    let t = &d.inputs[i];
                    (t.spec.dtype, t.host.ptr, t.host.bytes)
                }
                None => return Err(no_input(name)),
            },
        };
        if dtype != want {
            return Err(format!("input '{name}' is not {}", if want == TensorElementType::Int64 { "i64" } else { "f32" }));
        }
        Ok((ptr, bytes))
    }

    /// Run once with the current inputs; the outputs are then readable with
    /// [`output`](Self::output). No Rust heap allocation on the bound and device paths (ONNX
    /// Runtime still allocates on its own heap inside a run, except on a CUDA graph replay).
    pub fn run(&mut self) -> Result<(), String> {
        let gpu = self.outcome.active != EP_CPU;
        self.io.run(&mut self.session, gpu)
    }

    /// [`run`](Self::run) without the final host wait on the device path (P3B-G): the run is
    /// enqueued on the plan's CUDA stream and completes before any host access to this plan's
    /// buffers ([`input_f32`](Self::input_f32), [`output`](Self::output), ...), [`wait`](Self::wait),
    /// or, on the device, before another device-resident plan's [`copy_rows_to`](Self::copy_rows_to)
    /// copy of its outputs. The other paths run synchronously. Allocation-free like `run`.
    pub fn run_async(&mut self) -> Result<(), String> {
        let gpu = self.outcome.active != EP_CPU;
        self.io.run_async(&mut self.session, gpu)
    }

    /// Wait for a run started by [`run_async`](Self::run_async) (a no-op otherwise), the way the
    /// plan's [`WaitMode`] says (`SIGIL_CUDA_TUNING=wait=...`). Allocation-free.
    pub fn wait(&self) -> Result<(), String> {
        match &self.io {
            Io::Device(d) => d.wait(),
            _ => Ok(()),
        }
    }

    /// Settle a run started by [`run_async`](Self::run_async) that has already completed, without
    /// waiting (its GPU time is reported then); `true` when no run is pending afterwards.
    pub fn poll(&self) -> Result<bool, String> {
        match &self.io {
            Io::Device(d) => d.poll(),
            _ => Ok(true),
        }
    }

    /// Device time (ms) from the end of this plan's last run to the start of `next`'s last run, when
    /// both are device-path plans on one CUDA context whose last runs completed with GPU timing on
    /// (`telemetry::gpu_timing_enabled`): the gap between two chained plans on the GPU (P3B-G's T2
    /// probe: with the device chain it holds only the device copy; with the host path also the
    /// download, the host's wake-up, its copy and the re-upload). Negative values (the runs
    /// overlapped) read 0. Allocation-free.
    pub fn gpu_gap_ms_to(&self, next: &SessionPlan) -> Option<f64> {
        let (Io::Device(a), Io::Device(b)) = (&self.io, &next.io) else {
            return None;
        };
        if a.pending.get() || b.pending.get() || !a.last_timed.get() || !b.last_timed.get() || a.ctx.ordinal() != b.ctx.ordinal() {
            return None;
        }
        let ((_, end), (start, _)) = (a.timing.as_ref()?, b.timing.as_ref()?);
        let scope = a.ctx.enter().ok()?;
        scope.elapsed_ms(end.raw(), start.raw()).ok().map(|ms| f64::from(ms).max(0.0))
    }

    /// The wait the plan's CUDA work uses (`None` off the device path).
    pub fn wait_mode(&self) -> Option<WaitMode> {
        match &self.io {
            Io::Device(d) => Some(d.waiter.mode()),
            _ => None,
        }
    }

    /// Run `runs` times with the current inputs (e.g. to settle a CPU session's caches). The VRAM
    /// hold is not re-measured here: it is measured once at build, after the plan's first run
    /// (later measurements would include plans built since).
    pub fn warm_up(&mut self, runs: usize) -> Result<(), String> {
        for _ in 0..runs {
            self.run()?;
        }
        Ok(())
    }

    /// The f32 output `name` of the last [`run`](Self::run).
    pub fn output(&self, name: &str) -> Result<&[f32], String> {
        match &self.io {
            Io::Host(h) => h.outputs.iter().find(|(s, _)| s.name == name).map(|(_, b)| b.as_slice()).ok_or_else(|| no_output(name)),
            Io::Bound(b) => b
                .outputs
                .iter()
                .find(|t| t.spec.name == name)
                // SAFETY: an f32 tensor of `len` values the plan owns; borrow tied to `&self`.
                .map(|t| unsafe { std::slice::from_raw_parts(t.ptr.cast::<f32>(), t.spec.len()) })
                .ok_or_else(|| no_output(name)),
            Io::Device(d) => match d.outputs.iter().find(|t| t.spec.name == name) {
                Some(t) if !t.download => Err(format!("output '{name}' stays on the device (set_output_download)")),
                Some(t) => {
                    // A pending run's download completes with it.
                    d.wait()?;
                    // SAFETY: the mirror holds `len` f32 values, downloaded by the last run (waited
                    // for above).
                    Ok(unsafe { std::slice::from_raw_parts(t.host.ptr.cast::<f32>(), t.spec.len()) })
                }
                None => Err(no_output(name)),
            },
        }
    }

    /// Whether [`run`](Self::run) copies output `name` to the host (device path only; on the other
    /// paths every output is always readable and this is a no-op). An output consumed only by
    /// [`copy_rows_to`](Self::copy_rows_to) can stay on the device.
    pub fn set_output_download(&mut self, name: &str, download: bool) -> Result<(), String> {
        match &mut self.io {
            Io::Device(d) => {
                // The flag applies from the next run: settle a pending one under the old flag.
                d.wait()?;
                let t = d.outputs.iter_mut().find(|t| t.spec.name == name).ok_or_else(|| no_output(name))?;
                t.download = download;
                Ok(())
            }
            Io::Host(h) => h.outputs.iter().any(|(s, _)| s.name == name).then_some(()).ok_or_else(|| no_output(name)),
            Io::Bound(b) => b.outputs.iter().any(|t| t.spec.name == name).then_some(()).ok_or_else(|| no_output(name)),
        }
    }

    /// Copy a window of rows of this plan's last output `output` into `dst`'s input `input` (f32).
    /// When both plans are device-resident on one GPU this is a strided device-to-device copy on
    /// `dst`'s stream (the data never leaves the GPU; `dst`'s next run sees it). Otherwise it is a
    /// host copy, read straight from the device when `output` is device-resident and not downloaded
    /// ([`set_output_download`](Self::set_output_download)). On every path the destination input
    /// counts as written, and its values outside the copied rows (padding columns written through
    /// [`input_f32`](Self::input_f32)) are kept. Run `dst` before running this plan again (the
    /// device copy reads this plan's output when `dst`'s stream gets to it). Allocation-free.
    pub fn copy_rows_to(&self, output: &str, dst: &mut SessionPlan, input: &str, copy: RowCopy) -> Result<(), String> {
        if let (Io::Device(s), Io::Device(d)) = (&self.io, &mut dst.io) {
            if s.ctx.ordinal() == d.ctx.ordinal() {
                let k = s.outputs.iter().position(|t| t.spec.name == output).ok_or_else(|| no_output(output))?;
                let src = &s.outputs[k];
                let i = d.inputs.iter().position(|t| t.spec.name == input).ok_or_else(|| no_input(input))?;
                if src.spec.dtype != TensorElementType::Float32 || d.inputs[i].spec.dtype != TensorElementType::Float32 {
                    return Err("row copies are f32 only".into());
                }
                copy.check(src.spec.len(), d.inputs[i].spec.len())?;
                if copy.rows == 0 || copy.row_len == 0 {
                    return Ok(());
                }
                return d
                    .copy_rows_from(s, k, &copy, i)
                    .map_err(|e| format!("device copy '{output}' → '{input}': {e}"));
            }
        }
        if let Io::Device(s) = &self.io {
            if let Some(src) = s.outputs.iter().find(|t| t.spec.name == output && !t.download) {
                // Kept on the device by its run: read the rows from there.
                let dst_buf = dst.input_f32(input)?;
                copy.check(src.spec.len(), dst_buf.len())?;
                if copy.rows == 0 || copy.row_len == 0 {
                    return Ok(());
                }
                return s
                    .read_rows(src, &copy, dst_buf.as_mut_ptr())
                    .map_err(|e| format!("device read '{output}' → '{input}': {e}"));
            }
        }
        let src = self.output(output)?;
        let dst_buf = dst.input_f32(input)?;
        copy.check(src.len(), dst_buf.len())?;
        for r in 0..copy.rows {
            let s = r * copy.src_stride + copy.src_offset;
            let t = r * copy.dst_stride;
            dst_buf[t..t + copy.row_len].copy_from_slice(&src[s..s + copy.row_len]);
        }
        Ok(())
    }
}

/// The first warm-up run of a graph capture: plain, or with the arena shrink request (`shrink` is
/// the CUDA device; `run(Some(device))` is the shrinking run, `run(None)` the plain one). ONNX
/// Runtime validates the request against the session's arena allocator and refuses a session
/// without one for that device, which would fail the run and cost the plan its graph for a 40 MB
/// saving: a refused shrink run is repeated plain, and `Ok` carries a note saying the shrink was
/// dropped. A plain run that fails is the capture's error.
fn first_warmup(shrink: Option<i32>, mut run: impl FnMut(Option<i32>) -> Result<(), String>) -> Result<Option<String>, String> {
    let Some(device) = shrink else {
        return run(None).map(|()| None);
    };
    match run(Some(device)) {
        Ok(()) => Ok(None),
        Err(why) => {
            tracing::warn!(target: "sigil::gpu", "arena shrink before the graph capture refused ({why}); capturing without it");
            run(None)?;
            Ok(Some(format!(
                "arena shrink: ONNX Runtime refused the shrink request for device {device} ({why}); the graph was captured without it"
            )))
        }
    }
}

/// Test-only hooks (thread-local, so parallel tests do not see each other's).
#[cfg(test)]
pub(crate) mod test_hooks {
    use std::cell::Cell;

    thread_local! {
        static SHRINK_DEVICE: Cell<Option<i32>> = const { Cell::new(None) };
    }

    /// The device the capture's shrink request names instead of the plan's own (a device ONNX
    /// Runtime has no arena for: the shrink run is refused).
    pub(crate) fn shrink_device() -> Option<i32> {
        SHRINK_DEVICE.with(Cell::get)
    }

    pub(crate) fn set_shrink_device(device: Option<i32>) {
        SHRINK_DEVICE.with(|c| c.set(device));
    }
}

fn no_input(name: &str) -> String {
    format!("no input '{name}'")
}

fn no_output(name: &str) -> String {
    format!("no output '{name}'")
}

/// Digest of everything that defines a TensorRT engine on the plan's side: the model, the weights,
/// the shape overrides and the I/O (hex).
fn model_key(spec: &PlanSpec<'_>) -> String {
    let mut h = Sha256::new();
    match spec.model {
        ModelSource::Memory(b) => {
            h.update(b"mem");
            h.update(b);
        }
        ModelSource::File(p) => {
            h.update(b"file");
            h.update(p.to_string_lossy().as_bytes());
            if let Ok(m) = std::fs::metadata(p) {
                h.update(m.len().to_le_bytes());
                if let Ok(t) = m.modified().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).map_err(std::io::Error::other)) {
                    h.update(t.as_nanos().to_le_bytes());
                }
            }
        }
    }
    if let Some(w) = spec.weights {
        h.update(w.file.as_bytes());
        h.update((w.image.len() as u64).to_le_bytes());
        h.update(w.image);
    }
    for (name, size) in spec.dims {
        h.update(name.as_bytes());
        h.update(size.to_le_bytes());
    }
    for t in spec.inputs.iter().chain(spec.outputs) {
        h.update(t.name.as_bytes());
        h.update(format!("{:?}{:?}", t.dtype, t.shape).as_bytes());
    }
    hex::encode(&h.finalize()[..16])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT, INT64};

    #[test]
    fn check_patterns_are_bounded_distinct_and_index_safe() {
        let f = |seed, j| (check_value(seed, j) >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
        let a: Vec<f32> = (0..256).map(|j| f(1, j)).collect();
        let b: Vec<f32> = (0..256).map(|j| f(2, j)).collect();
        assert!(a.iter().chain(&b).all(|v| (-1.0..1.0).contains(v)));
        assert_ne!(a, b, "two different patterns");
        assert!(a.iter().any(|v| *v < -0.5) && a.iter().any(|v| *v > 0.5), "spread over the range");
        let ones = (0..256).filter(|&j| check_value(1, j) & 1 == 1).count();
        assert!((64..192).contains(&ones), "i64 patterns mix 0 and 1: {ones}");
        assert!(f32::from_bits(CHECK_SENTINEL).is_nan());
        let t = CudaTuning::parse("devio=1,tf32=0");
        assert_eq!((t.device_io, t.tf32), (Some(true), Some(false)));
        assert!(t.device_io_without_graphs());
        assert_eq!(CudaTuning::DEFAULT.device_io_without_graphs(), DEVICE_IO_WITHOUT_GRAPHS);
    }

    /// P3B-G (review fix): the capture's first warm-up carries the shrink request only when asked; a
    /// shrink run ONNX Runtime refuses is repeated plain (and says so), instead of failing the capture
    /// and leaving the plan without a graph. A plain run that fails is still the capture's error.
    #[test]
    fn a_refused_shrink_run_is_repeated_plain() {
        let calls = std::cell::RefCell::new(Vec::new());
        let ok = |d: Option<i32>| {
            calls.borrow_mut().push(d);
            Ok(())
        };
        assert_eq!(first_warmup(None, ok), Ok(None));
        assert_eq!(first_warmup(Some(0), ok), Ok(None));
        assert_eq!(*calls.borrow(), vec![None, Some(0)], "one run each, the shrink where asked");
        calls.borrow_mut().clear();
        // Refused: the same warm-up runs again without the shrink, once, and the note says why.
        let refuse = |d: Option<i32>| {
            calls.borrow_mut().push(d);
            if d.is_some() {
                Err("Did not find an arena based allocator registered for device-id combination".to_string())
            } else {
                Ok(())
            }
        };
        let note = first_warmup(Some(3), refuse).unwrap().expect("a note");
        assert_eq!(*calls.borrow(), vec![Some(3), None]);
        assert!(note.contains("arena shrink") && note.contains("device 3") && note.contains("arena based allocator"), "{note}");
        // The plain retry failing is the capture's error (no graph).
        calls.borrow_mut().clear();
        let broken = |d: Option<i32>| {
            calls.borrow_mut().push(d);
            Err::<(), _>(format!("run failed (shrink {d:?})"))
        };
        assert_eq!(first_warmup(Some(0), broken), Err("run failed (shrink None)".to_string()));
        assert_eq!(*calls.borrow(), vec![Some(0), None]);
        // Without a shrink request there is no retry.
        calls.borrow_mut().clear();
        assert_eq!(first_warmup(None, broken), Err("run failed (shrink None)".to_string()));
        assert_eq!(*calls.borrow(), vec![None]);
    }

    /// `y = relu(x * 2 + cast(k))` at `[1, 8]` (the `gpu::tests` affine model).
    fn affine_bytes() -> Vec<u8> {
        let mut g = Graph::new();
        g.input("x", FLOAT, &[Dim::Fixed(1), Dim::Fixed(8)])
            .input("k", INT64, &[Dim::Fixed(1), Dim::Fixed(8)])
            .init_f32("two", &[1], &[2.0])
            .node("Mul", &["x", "two"], &["x2"], &[])
            .node("Cast", &["k"], &["kf"], &[Attr::Int("to", 1)])
            .node("Add", &["x2", "kf"], &["s"], &[])
            .node("Relu", &["s"], &["y"], &[])
            .output("y", FLOAT, &[Dim::Fixed(1), Dim::Fixed(8)]);
        g.model_bytes(17)
    }

    /// The graph check (dev rig, CUDA): the plan's own graph passes; a graph that recorded nothing
    /// (what a session whose kernels ran on another stream records) and a graph that recorded the
    /// wrong work both fail, although the device outputs still hold a correct earlier result. The
    /// inputs are zeros again after the check.
    #[test]
    #[ignore = "needs an NVIDIA GPU, the CUDA provider next to the test executable and the NVIDIA runtime (SIGIL_APP_DATA_DIR or SIGIL_TEST_CUDA_BIN)"]
    fn graph_check_rejects_empty_and_wrong_graphs() {
        crate::services::neural_audio::ort_session::preload_test_cuda_runtime_quiet();
        let bytes = affine_bytes();
        let opts = VcSessionOptions::default();
        let inputs = [TensorSpec::f32("x", &[1, 8]), TensorSpec::i64("k", &[1, 8])];
        let outputs = [TensorSpec::f32("y", &[1, 8])];
        let mut plan = SessionPlan::build(&PlanSpec {
            component: "test.gpu.check",
            ep: "cuda",
            opts: &opts,
            model: ModelSource::Memory(&bytes),
            weights: None,
            dims: &[],
            inputs: &inputs,
            outputs: &outputs,
            levers: GpuLevers { io_binding: true, cuda_graphs: true, ..GpuLevers::default() },
        })
        .unwrap();
        if plan.outcome().active != "cuda" {
            eprintln!("CUDA is not usable here ({:?}); skipped", plan.outcome());
            return;
        }
        assert!(plan.levers().cuda_graphs, "{:?}", plan.notes());
        assert!(plan.input_f32("x").unwrap().iter().all(|&v| v == 0.0), "the check patterns do not leak");
        assert!(plan.input_i64("k").unwrap().iter().all(|&v| v == 0));
        let SessionPlan { io, session, .. } = &mut plan;
        let Io::Device(d) = io else { panic!("device path expected") };
        let own = d.graph.take().unwrap();
        d.check_graph(session, &own).expect("the plan's own graph replays the run");
        let empty = CapturedGraph::capture(&d.ctx, d.stream, || Ok(())).unwrap();
        let err = d.check_graph(session, &empty).unwrap_err();
        assert!(err.contains("does not reproduce"), "{err}");
        // A graph of the wrong work: it copies `x` into `y` (right size, wrong values).
        let (x, y) = (d.inputs[0].dptr, d.outputs[0].dptr);
        let (ctx, stream) = (Arc::clone(&d.ctx), d.stream);
        let wrong = CapturedGraph::capture(&ctx, stream, || {
            let scope = ctx.enter().map_err(|r| r.to_string())?;
            // SAFETY: two 8-value f32 device tensors of this plan.
            unsafe { scope.copy_2d_dtod(y, 32, x, 32, 32, 1, stream) }.map_err(|r| r.to_string())
        })
        .unwrap();
        let err = d.check_graph(session, &wrong).unwrap_err();
        assert!(err.contains("does not reproduce"), "{err}");
        d.fill_inputs(0).unwrap();
        d.graph = Some(own);
        drop((empty, wrong));
        // The plan still runs its own graph correctly.
        plan.input_f32("x").unwrap().copy_from_slice(&[1.0, -1.0, 0.5, 2.0, -3.0, 0.0, 4.0, -0.25]);
        plan.input_i64("k").unwrap().copy_from_slice(&[0, 1, 0, -1, 2, 0, 0, 1]);
        plan.run().unwrap();
        assert_eq!(plan.output("y").unwrap(), &[2.0, 0.0, 1.0, 3.0, 0.0, 0.0, 8.0, 0.5]);
    }
}
