//! Device-resident IOBinding at fixed shapes (P3-C2; the `io_binding` lever), used by
//! `plan::SessionPlan` when the lever is on:
//!
//! - **CUDA rung:** inputs and outputs are device tensors allocated once from the session's CUDA
//!   allocator and bound once. A tick uploads only the inputs written since the last run
//!   (pinned host mirrors, `cuMemcpyHtoDAsync` on the plan's stream), runs, and downloads only the
//!   outputs the caller reads; an output can also be copied device-to-device into another plan's
//!   input (`SessionPlan::copy_rows_to`), so `enc`'s `m_p` / `logs_p` never leave the GPU.
//!   Fixed addresses are also what CUDA graph replay needs. ORT copies a bound value to the device
//!   its consumers read it from **when it is bound**, so this path is used only when every input is
//!   read from device memory ([`inputs_on_device`]); an input a CUDA kernel takes on the CPU (a
//!   Reshape shape, Slice starts / ends, Clip bounds, Resize scales, ...) would otherwise keep its
//!   bind-time copy forever, and such plans bind host tensors instead.
//! - **DirectML / TensorRT-RTX / CPU rungs:** inputs and outputs are host tensors bound to the
//!   binding. On the CPU they are bound once (ORT reads and writes them in place, no copy); on a GPU
//!   rung the inputs written since the last run are re-bound (ORT uploads them) and the outputs are
//!   filled by ORT at the end of the run.
//!
//! Every run goes through the raw `RunWithBinding` C API ([`run`]): `ort`'s safe `run_binding`
//! builds new `Value`s for the bound outputs on every call (Rust heap allocations), which the hot
//! path must not do (plan T2). ONNX Runtime itself still allocates on its C++ heap inside every
//! `RunWithBinding` (execution frame, feeds and fetches), and on a GPU rung `BindInput` allocates
//! and copies a new device value per re-bound input: only a CUDA graph replay (`cuda_graph`), which
//! calls no ONNX Runtime function per tick, is free of host allocations altogether.
//!
//! # Frozen by P3-S0
//! - [`available`]`() -> bool` (cheap; `voice_changer::capabilities` calls it).

use std::ffi::{CStr, CString};

use ort::session::{IoBinding, Session};
use ort::value::DynValue;
use ort::AsPointer;

/// Whether session plans can bind I/O in this build: always (IOBinding is part of the linked ONNX
/// Runtime on every EP). Where it applies (GPU rungs) is `voice_changer::capabilities`'s decision.
pub fn available() -> bool {
    true
}

/// `name` as a C string for the raw binding calls (session I/O names never contain NUL).
pub(crate) fn c_name(name: &str) -> Result<CString, String> {
    CString::new(name).map_err(|_| format!("tensor name {name:?} contains a NUL byte"))
}

/// The error of an ORT status (consumes and releases it; `Ok` for success). Allocates only on
/// failure.
fn status(ptr: ort::sys::OrtStatusPtr) -> Result<(), String> {
    // SAFETY: `ptr` is a status just returned by an ORT API call (null = success); ownership moves
    // into `result_from_status`, which releases it.
    unsafe { ort::Error::result_from_status(ptr) }.map_err(|e| e.to_string())
}

/// Bind `value` to the session input `name` (ORT copies a host value to the session's device now;
/// a value already on that device is referenced). Allocation-free on the Rust side.
pub(crate) fn bind_input(binding: &mut IoBinding, name: &CStr, value: &DynValue) -> Result<(), String> {
    // SAFETY: valid binding, NUL-terminated name and a live OrtValue (ORT keeps its own reference).
    status(unsafe { (ort::api().BindInput)(binding.ptr_mut(), name.as_ptr(), value.ptr()) })
}

/// Bind the session output `name` to the pre-allocated `value` (ORT writes the result into it, and
/// copies it there when the session produces it on another device).
pub(crate) fn bind_output(binding: &mut IoBinding, name: &CStr, value: &DynValue) -> Result<(), String> {
    // SAFETY: as in `bind_input`.
    status(unsafe { (ort::api().BindOutput)(binding.ptr_mut(), name.as_ptr(), value.ptr()) })
}

/// Run `session` with `binding` and default run options (ORT synchronises the device at the end of
/// the run). No Rust heap allocation on success.
pub(crate) fn run(session: &mut Session, binding: &IoBinding) -> Result<(), String> {
    // SAFETY: a live session and binding created from it; `&mut Session` makes the call exclusive
    // (RunWithBinding is not thread-safe per session).
    status(unsafe { (ort::api().RunWithBinding)(session.ptr_mut(), std::ptr::null(), binding.ptr()) })
}

/// Where ONNX Runtime reads each session input on the rung the session runs on, in session order:
/// `(name, true)` for device (GPU) memory, `(name, false)` for host memory (inputs of kernels that
/// take them on the CPU, e.g. a Reshape shape, Slice starts / ends, Clip bounds, Resize scales or
/// Expand / Tile / Range / ConstantOfShape sizes, and inputs no node reads). This is the device a
/// bound value is copied to at bind time (`SessionGetMemoryInfoForInputs`).
pub(crate) fn inputs_on_device(session: &Session) -> Result<Vec<(String, bool)>, String> {
    use ort::sys::{OrtMemoryInfo, OrtMemoryInfoDeviceType};
    let inputs = session.inputs();
    let n = inputs.len();
    let mut infos: Vec<*const OrtMemoryInfo> = vec![std::ptr::null(); n];
    // SAFETY: a live session and an array with room for its `n` inputs; the infos written belong to
    // the session's allocators and are never released here.
    status(unsafe { (ort::api().SessionGetMemoryInfoForInputs)(session.ptr(), infos.as_mut_ptr(), n) })
        .map_err(|e| format!("input memory info: {e}"))?;
    inputs
        .iter()
        .zip(&infos)
        .map(|(input, &info)| {
            if info.is_null() {
                return Err(format!("no memory info for input '{}'", input.name()));
            }
            let mut ty = OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU;
            // SAFETY: a valid memory info of the session; writes one enum value.
            unsafe { (ort::api().MemoryInfoGetDeviceType)(info, &mut ty) };
            Ok((input.name().to_string(), ty == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU))
        })
        .collect()
}

/// [`run`] with `options` (e.g. device synchronisation disabled for a CUDA graph capture).
pub(crate) fn run_with(session: &mut Session, binding: &IoBinding, options: &ort::session::RunOptions) -> Result<(), String> {
    // SAFETY: as in `run`; the options outlive the call.
    status(unsafe { (ort::api().RunWithBinding)(session.ptr_mut(), options.ptr(), binding.ptr()) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions};
    use crate::services::neural_audio::test_onnx::{Dim, Graph, FLOAT};
    use ort::value::Tensor;

    /// Raw binds on a CPU session: a bound host input is read in place (writing the tensor after
    /// binding changes the next run's result), and a bound output is written in place.
    #[test]
    fn raw_binding_runs_in_place_on_the_cpu() {
        let mut g = Graph::new();
        g.input("x", FLOAT, &[Dim::Fixed(1), Dim::Fixed(3)])
            .init_f32("k", &[1], &[3.0])
            .node("Mul", &["x", "k"], &["y"], &[])
            .output("y", FLOAT, &[Dim::Fixed(1), Dim::Fixed(3)]);
        let bytes = g.model_bytes(17);
        let (mut session, _) = build_session(None, "cpu", &VcSessionOptions::default(), ModelSource::Memory(&bytes)).unwrap();
        let mut binding = session.create_binding().unwrap();
        let mut x = Tensor::from_array(([1usize, 3], vec![1.0f32, 2.0, 3.0])).unwrap().into_dyn();
        let y = Tensor::from_array(([1usize, 3], vec![0.0f32; 3])).unwrap().into_dyn();
        bind_input(&mut binding, &c_name("x").unwrap(), &x).unwrap();
        bind_output(&mut binding, &c_name("y").unwrap(), &y).unwrap();
        run(&mut session, &binding).unwrap();
        assert_eq!(y.try_extract_tensor::<f32>().unwrap().1, &[3.0, 6.0, 9.0]);
        x.try_extract_tensor_mut::<f32>().unwrap().1.copy_from_slice(&[-1.0, 0.0, 0.5]);
        run(&mut session, &binding).unwrap();
        assert_eq!(y.try_extract_tensor::<f32>().unwrap().1, &[-3.0, 0.0, 1.5]);
        let err = bind_input(&mut binding, &c_name("nope").unwrap(), &x).unwrap_err();
        assert!(!err.is_empty());
        assert!(c_name("a\0b").is_err());
        assert!(available());
        assert_eq!(inputs_on_device(&session).unwrap(), vec![("x".to_string(), false)], "a CPU session reads host memory");
    }

    /// `y = Reshape(x * 2, shape)`: on DirectML (Windows, hardware GPU) `x` is read from device
    /// memory and `shape` on the host (Reshape takes its shape on the CPU), which is what keeps such
    /// plans off the device-resident path. (CUDA places them the same way; its Clip, by contrast,
    /// reads its bounds from device memory in ONNX Runtime 1.28: measured on the dev rig.)
    #[cfg(windows)]
    #[test]
    fn shape_inputs_are_read_on_the_host_by_a_gpu_session() {
        use crate::services::diagnostics::gpu_mem;
        use crate::services::neural_audio::test_onnx::INT64;
        if !gpu_mem::adapters().iter().any(|a| !a.software) {
            eprintln!("no hardware GPU; skipped");
            return;
        }
        let mut g = Graph::new();
        g.input("x", FLOAT, &[Dim::Fixed(2), Dim::Fixed(3)])
            .input("shape", INT64, &[Dim::Fixed(2)])
            .init_f32("two", &[1], &[2.0])
            .node("Mul", &["x", "two"], &["x2"], &[])
            .node("Reshape", &["x2", "shape"], &["y"], &[])
            .output("y", FLOAT, &[Dim::Named("A"), Dim::Named("B")]);
        let bytes = g.model_bytes(17);
        let (session, outcome) = build_session(None, "dml", &VcSessionOptions::default(), ModelSource::Memory(&bytes)).unwrap();
        if outcome.active != "dml" {
            eprintln!("DirectML unavailable here ({outcome:?}); skipped");
            return;
        }
        let on_device = inputs_on_device(&session).unwrap();
        assert_eq!(on_device, vec![("x".to_string(), true), ("shape".to_string(), false)]);
    }
}
