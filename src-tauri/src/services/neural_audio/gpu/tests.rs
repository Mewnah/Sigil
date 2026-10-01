//! Session-plan tests (`cargo test neural_audio::gpu`): the legacy, bound and device paths agree,
//! buffers are reused (no reallocation), the IOBinding paths make no Rust heap allocation per tick
//! (the counting allocator does not see ONNX Runtime's own C++ heap), row copies land where they
//! should, inputs a GPU kernel reads on the CPU stay correct, and each GPU lever falls back with a
//! note. GPU-only checks skip (or are `#[ignore]`d with the environment they need) when the hardware
//! or runtime is missing.
//!
//! Dev-rig runs of the ignored CUDA tests and the lever benchmark need the CUDA provider next to the
//! test executable and the NVIDIA runtime pack (copy the test exe with `DirectML.dll` and the
//! `onnxruntime_providers_{cuda,shared}.dll` of `target/<profile>` into one folder; set
//! `SIGIL_APP_DATA_DIR` to an app-data folder with `runtime/nvidia-cuda13` installed):
//! `<folder>\sigil-<hash>.exe --ignored --nocapture gpu::tests::`

use super::alloc_probe::allocations_during;
use super::*;
use crate::services::neural_audio::ort_session::{ModelSource, VcSessionOptions};
use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT, INT64};

/// `y = relu(x * 2 + cast(k))`, `x: f32 [1, N]`, `k: i64 [1, N]`, `y: f32 [1, N]`.
fn affine_model() -> Vec<u8> {
    let mut g = Graph::new();
    g.input("x", FLOAT, &[Dim::Fixed(1), Dim::Named("N")])
        .input("k", INT64, &[Dim::Fixed(1), Dim::Named("N")])
        .init_f32("two", &[1], &[2.0])
        .node("Mul", &["x", "two"], &["x2"], &[])
        .node("Cast", &["k"], &["kf"], &[Attr::Int("to", 1)])
        .node("Add", &["x2", "kf"], &["s"], &[])
        .node("Relu", &["s"], &["y"], &[])
        .output("y", FLOAT, &[Dim::Fixed(1), Dim::Named("N")]);
    g.model_bytes(17)
}

fn expected_affine(x: &[f32], k: &[i64]) -> Vec<f32> {
    x.iter().zip(k).map(|(&x, &k)| (x * 2.0 + k as f32).max(0.0)).collect()
}

/// `m = x * 3`, `x, m: f32 [1, C, T]` (an `enc` stand-in).
fn enc_model() -> Vec<u8> {
    let mut g = Graph::new();
    g.input("x", FLOAT, &[Dim::Fixed(1), Dim::Named("C"), Dim::Named("T")])
        .init_f32("three", &[1], &[3.0])
        .node("Mul", &["x", "three"], &["m"], &[])
        .output("m", FLOAT, &[Dim::Fixed(1), Dim::Named("C"), Dim::Named("T")]);
    g.model_bytes(17)
}

/// `y = Slice(x * 2, starts, ends, axes = [1])`, `x: f32 [1, 16]`, `starts, ends: i64 [1]`,
/// `y: f32 [1, 4]` (`ends = starts + 4`; a w-okada style `skip_head` / `return_length` window):
/// CUDA and DirectML kernels take Slice's bounds on the CPU, so ONNX Runtime reads `starts` /
/// `ends` from host memory.
fn slice_model() -> Vec<u8> {
    let mut g = Graph::new();
    g.input("x", FLOAT, &[Dim::Fixed(1), Dim::Fixed(16)])
        .input("starts", INT64, &[Dim::Fixed(1)])
        .input("ends", INT64, &[Dim::Fixed(1)])
        .init_f32("two", &[1], &[2.0])
        .init_i64("axes", &[1], &[1])
        .node("Mul", &["x", "two"], &["x2"], &[])
        .node("Slice", &["x2", "starts", "ends", "axes"], &["y"], &[])
        .output("y", FLOAT, &[Dim::Fixed(1), Dim::Named("W")]);
    g.model_bytes(17)
}

/// `o = w + 1`, `w, o: f32 [1, C, F]` (a `flowdec` stand-in).
fn dec_model() -> Vec<u8> {
    let mut g = Graph::new();
    g.input("w", FLOAT, &[Dim::Fixed(1), Dim::Named("C"), Dim::Named("F")])
        .init_f32("one", &[1], &[1.0])
        .node("Add", &["w", "one"], &["o"], &[])
        .output("o", FLOAT, &[Dim::Fixed(1), Dim::Named("C"), Dim::Named("F")]);
    g.model_bytes(17)
}

const N: usize = 8;

fn affine_plan(ep: &str, levers: GpuLevers, opts: &VcSessionOptions, bytes: &[u8]) -> SessionPlan {
    let inputs = [TensorSpec::f32("x", &[1, N]), TensorSpec::i64("k", &[1, N])];
    let outputs = [TensorSpec::f32("y", &[1, N])];
    SessionPlan::build(&PlanSpec {
        component: "test.gpu.affine",
        ep,
        opts,
        model: ModelSource::Memory(bytes),
        weights: None,
        dims: &[("N", N as i64)],
        inputs: &inputs,
        outputs: &outputs,
        levers,
    })
    .unwrap()
}

fn io_binding() -> GpuLevers {
    GpuLevers { io_binding: true, ..GpuLevers::default() }
}

fn all_levers() -> GpuLevers {
    GpuLevers { io_binding: true, cuda_graphs: true, priority: true, ..GpuLevers::default() }
}

/// Write tick `i`'s inputs, run, and check the output against the reference.
fn tick(plan: &mut SessionPlan, i: usize) {
    let x: Vec<f32> = (0..N).map(|j| (i * N + j) as f32 * 0.25 - 3.0).collect();
    let k: Vec<i64> = (0..N).map(|j| (j as i64 % 3) - 1).collect();
    plan.input_f32("x").unwrap().copy_from_slice(&x);
    plan.input_i64("k").unwrap().copy_from_slice(&k);
    plan.run().unwrap();
    assert_eq!(plan.output("y").unwrap(), expected_affine(&x, &k).as_slice(), "tick {i}");
}

/// The legacy path runs a graph on the CPU through the ladder at the pinned shape, with the
/// I/O buffers the plan owns.
#[test]
fn legacy_plan_runs_a_fixed_shape_graph() {
    let mut g = Graph::new();
    g.input("x", FLOAT, &[Dim::Fixed(1), Dim::Named("N")])
        .init_f32("two", &[1], &[2.0])
        .node("Mul", &["x", "two"], &["y"], &[])
        .output("y", FLOAT, &[Dim::Fixed(1), Dim::Named("N")]);
    let bytes = g.model_bytes(17);
    let opts = VcSessionOptions::default();
    let inputs = [TensorSpec::f32("x", &[1, 4])];
    let outputs = [TensorSpec::f32("y", &[1, 4])];
    let mut plan = SessionPlan::build(&PlanSpec {
        component: "test.plan",
        ep: "cpu",
        opts: &opts,
        model: ModelSource::Memory(&bytes),
        weights: None,
        dims: &[("N", 4)],
        inputs: &inputs,
        outputs: &outputs,
        levers: GpuLevers::default(),
    })
    .unwrap();
    assert_eq!(plan.outcome().active, "cpu");
    plan.input_f32("x").unwrap().copy_from_slice(&[1.0, -2.0, 3.5, 0.0]);
    plan.run().unwrap();
    assert_eq!(plan.output("y").unwrap(), &[2.0, -4.0, 7.0, 0.0]);
    assert!(plan.input_i64("x").is_err());
    assert!(plan.output("z").is_err());
    assert_eq!(plan.levers(), GpuLevers::default());
    assert_eq!(plan.residency(), Residency::Host);
    assert!(plan.notes().is_empty());
    assert_eq!(plan.vram_mb(), None, "no VRAM on the CPU");
}

/// The bound path on the CPU gives the legacy path's results tick after tick, with the same
/// buffers (plan reuse: no reallocation), and reports the lever it uses.
#[test]
fn bound_plan_matches_the_legacy_plan_and_reuses_its_buffers() {
    let bytes = affine_model();
    let opts = VcSessionOptions::default();
    let mut legacy = affine_plan("cpu", GpuLevers::default(), &opts, &bytes);
    let mut bound = affine_plan("cpu", io_binding(), &opts, &bytes);
    assert_eq!(bound.residency(), Residency::Bound);
    assert!(bound.levers().io_binding && !bound.levers().cuda_graphs && !bound.levers().priority);
    let (xp, kp, yp) = (
        bound.input_f32("x").unwrap().as_ptr(),
        bound.input_i64("k").unwrap().as_ptr(),
        bound.output("y").unwrap().as_ptr(),
    );
    for i in 0..5 {
        tick(&mut legacy, i);
        tick(&mut bound, i);
    }
    assert_eq!(bound.input_f32("x").unwrap().as_ptr(), xp, "input buffers are reused");
    assert_eq!(bound.input_i64("k").unwrap().as_ptr(), kp);
    assert_eq!(bound.output("y").unwrap().as_ptr(), yp, "output buffers are reused");
    assert!(bound.input_i64("x").unwrap_err().contains("not i64"));
    assert!(bound.input_f32("k").unwrap_err().contains("not f32"));
    assert!(bound.output("nope").is_err() && bound.input_f32("nope").is_err());
    assert!(bound.set_output_download("y", false).is_ok(), "a no-op off the device path");
    assert!(bound.output("y").is_ok());
}

/// Plan T2 (Rust side): after warm-up, a tick on the bound path (write inputs, run, read the
/// output) makes no Rust heap allocation; the legacy path does (which also proves the probe sees
/// plan work). ONNX Runtime's own allocations inside `RunWithBinding` are not counted.
#[test]
fn run_is_allocation_free_on_the_bound_path() {
    let bytes = affine_model();
    let opts = VcSessionOptions::default();
    let mut bound = affine_plan("cpu", io_binding(), &opts, &bytes);
    let mut legacy = affine_plan("cpu", GpuLevers::default(), &opts, &bytes);
    let x = [0.5f32; N];
    let k = [1i64; N];
    let ticks = |plan: &mut SessionPlan| {
        for _ in 0..50 {
            plan.input_f32("x").unwrap().copy_from_slice(&x);
            plan.input_i64("k").unwrap().copy_from_slice(&k);
            plan.run().unwrap();
            std::hint::black_box(plan.output("y").unwrap()[0]);
        }
    };
    ticks(&mut bound);
    ticks(&mut legacy);
    let ((), bound_allocs) = allocations_during(|| ticks(&mut bound));
    let ((), legacy_allocs) = allocations_during(|| ticks(&mut legacy));
    eprintln!("heap allocations per 50 ticks: bound {bound_allocs}, legacy {legacy_allocs}");
    assert_eq!(bound_allocs, 0, "the bound path must not allocate per tick");
    assert!(legacy_allocs > 0, "the legacy path allocates (the probe works)");
    assert_eq!(bound.output("y").unwrap(), expected_affine(&x, &k).as_slice());
}

fn enc_dec(ep: &str, levers: GpuLevers, opts: &VcSessionOptions) -> (SessionPlan, SessionPlan) {
    enc_dec_with(ep, levers, levers, opts, 4)
}

/// `enc` (`[1, 3, 10]`) and `dec` (`[1, 3, f]`) plans with their own levers.
fn enc_dec_with(ep: &str, enc_levers: GpuLevers, dec_levers: GpuLevers, opts: &VcSessionOptions, f: usize) -> (SessionPlan, SessionPlan) {
    let (c, t) = (3usize, 10usize);
    let (enc_bytes, dec_bytes) = (enc_model(), dec_model());
    let enc = SessionPlan::build(&PlanSpec {
        component: "test.gpu.enc",
        ep,
        opts,
        model: ModelSource::Memory(&enc_bytes),
        weights: None,
        dims: &[("C", c as i64), ("T", t as i64)],
        inputs: &[TensorSpec::f32("x", &[1, c, t])],
        outputs: &[TensorSpec::f32("m", &[1, c, t])],
        levers: enc_levers,
    })
    .unwrap();
    let dec = SessionPlan::build(&PlanSpec {
        component: "test.gpu.dec",
        ep,
        opts,
        model: ModelSource::Memory(&dec_bytes),
        weights: None,
        dims: &[("C", c as i64), ("F", f as i64)],
        inputs: &[TensorSpec::f32("w", &[1, c, f])],
        outputs: &[TensorSpec::f32("o", &[1, c, f])],
        levers: dec_levers,
    })
    .unwrap();
    (enc, dec)
}

/// `enc`'s output window lands in `dec`'s input row by row (channels 3, context 10, window 4 at
/// frame 5): `o[c][j] = 3 * x[c][5 + j] + 1`.
fn check_row_copy(enc: &mut SessionPlan, dec: &mut SessionPlan) {
    let x: Vec<f32> = (0..30).map(|i| i as f32).collect();
    enc.input_f32("x").unwrap().copy_from_slice(&x);
    enc.run().unwrap();
    let copy = RowCopy { rows: 3, row_len: 4, src_stride: 10, src_offset: 5, dst_stride: 4 };
    enc.copy_rows_to("m", dec, "w", copy).unwrap();
    dec.run().unwrap();
    let want: Vec<f32> = (0..3).flat_map(|c| (0..4).map(move |j| 3.0 * (c * 10 + 5 + j) as f32 + 1.0)).collect();
    assert_eq!(dec.output("o").unwrap(), want.as_slice());
    let bad = RowCopy { src_offset: 7, ..copy };
    assert!(enc.copy_rows_to("m", dec, "w", bad).unwrap_err().contains("does not fit"));
    assert!(enc.copy_rows_to("nope", dec, "w", copy).is_err());
}

/// A row copy into a destination wider than the rows (`dec` `[1, 3, 6]`, rows of 4): padding
/// columns written through `input_f32` before the copy are kept, the copied rows are visible to a
/// read after the copy, and padding written after the copy reaches the run, on every path.
fn check_padded_row_copy(enc: &mut SessionPlan, dec: &mut SessionPlan) {
    let x: Vec<f32> = (0..30).map(|i| i as f32 * 0.5).collect();
    enc.input_f32("x").unwrap().copy_from_slice(&x);
    enc.run().unwrap();
    {
        let w = dec.input_f32("w").unwrap();
        for c in 0..3 {
            w[c * 6..c * 6 + 4].fill(-7.0);
            w[c * 6 + 4] = 100.0 + c as f32;
            w[c * 6 + 5] = 200.0 + c as f32;
        }
    }
    let copy = RowCopy { rows: 3, row_len: 4, src_stride: 10, src_offset: 5, dst_stride: 6 };
    enc.copy_rows_to("m", dec, "w", copy).unwrap();
    {
        let w = dec.input_f32("w").unwrap();
        for c in 0..3 {
            for j in 0..4 {
                assert_eq!(w[c * 6 + j], 3.0 * x[c * 10 + 5 + j], "row {c} col {j} after the copy");
            }
            assert_eq!((w[c * 6 + 4], w[c * 6 + 5]), (100.0 + c as f32, 200.0 + c as f32), "padding of row {c}");
            w[c * 6 + 5] = 300.0 + c as f32;
        }
    }
    dec.run().unwrap();
    let want: Vec<f32> = (0..3)
        .flat_map(|c| {
            let x = &x;
            (0..6).map(move |j| match j {
                4 => 101.0 + c as f32,
                5 => 301.0 + c as f32,
                _ => 3.0 * x[c * 10 + 5 + j] + 1.0,
            })
        })
        .collect();
    assert_eq!(dec.output("o").unwrap(), want.as_slice());
}

#[test]
fn row_copies_between_plans_on_the_host() {
    let opts = VcSessionOptions::default();
    let (mut enc, mut dec) = enc_dec("cpu", GpuLevers::default(), &opts);
    check_row_copy(&mut enc, &mut dec);
    let (mut enc, mut dec) = enc_dec("cpu", io_binding(), &opts);
    check_row_copy(&mut enc, &mut dec);
    for (e, d) in [(GpuLevers::default(), GpuLevers::default()), (io_binding(), io_binding()), (io_binding(), GpuLevers::default())] {
        let (mut enc, mut dec) = enc_dec_with("cpu", e, d, &opts, 6);
        check_padded_row_copy(&mut enc, &mut dec);
        // A no-op off the device path: the output stays readable and copies still work.
        enc.set_output_download("m", false).unwrap();
        check_padded_row_copy(&mut enc, &mut dec);
    }
}

/// Inputs a GPU kernel reads on the CPU (Slice's bounds) on `ep`: the plan binds host tensors (on
/// CUDA it says why it has no device-resident buffers or graph) and follows new values every tick.
/// (CUDA's Clip, by contrast, reads its bounds from device memory in ONNX Runtime 1.28.)
fn check_host_read_inputs(ep: &str, opts: &VcSessionOptions) {
    let bytes = slice_model();
    let inputs = [TensorSpec::f32("x", &[1, 16]), TensorSpec::i64("starts", &[1]), TensorSpec::i64("ends", &[1])];
    let outputs = [TensorSpec::f32("y", &[1, 4])];
    let mut plan = SessionPlan::build(&PlanSpec {
        component: "test.gpu.slice",
        ep,
        opts,
        model: ModelSource::Memory(&bytes),
        weights: None,
        dims: &[("W", 4)],
        inputs: &inputs,
        outputs: &outputs,
        levers: all_levers(),
    })
    .unwrap();
    eprintln!("slice plan on {ep}: {:?} {:?} {:?} {:?}", plan.outcome(), plan.residency(), plan.levers(), plan.notes());
    if plan.outcome().active != ep {
        eprintln!("{ep} unavailable here; skipped");
        return;
    }
    assert_eq!(plan.residency(), Residency::Bound);
    assert!(!plan.levers().cuda_graphs);
    if ep == "cuda" {
        let notes = plan.notes().join("\n");
        assert!(notes.contains("'starts'") && notes.contains("'ends'") && notes.contains("on the CPU"), "{notes}");
    }
    let x: Vec<f32> = (0..16).map(|j| j as f32 * 0.5 - 2.0).collect();
    plan.input_f32("x").unwrap().copy_from_slice(&x);
    for (i, start) in [3usize, 0, 9, 12, 5, 7].into_iter().enumerate() {
        plan.input_i64("starts").unwrap()[0] = start as i64;
        plan.input_i64("ends").unwrap()[0] = start as i64 + 4;
        plan.run().unwrap();
        let want: Vec<f32> = x[start..start + 4].iter().map(|v| v * 2.0).collect();
        assert_eq!(plan.output("y").unwrap(), want.as_slice(), "tick {i} (start {start})");
    }
    // Only the data changes: the window stays where the bounds put it.
    let x2: Vec<f32> = x.iter().map(|v| -v).collect();
    plan.input_f32("x").unwrap().copy_from_slice(&x2);
    plan.run().unwrap();
    let want: Vec<f32> = x2[7..11].iter().map(|v| v * 2.0).collect();
    assert_eq!(plan.output("y").unwrap(), want.as_slice());
}

/// GPU levers on a CPU request run nothing on the GPU: the plan binds host tensors and says why each
/// GPU lever is off.
#[test]
fn gpu_levers_on_the_cpu_fall_back_with_notes() {
    let bytes = affine_model();
    let opts = VcSessionOptions::default();
    let mut plan = affine_plan("cpu", all_levers(), &opts, &bytes);
    assert_eq!(plan.residency(), Residency::Bound);
    assert_eq!(plan.levers(), GpuLevers { io_binding: true, ..GpuLevers::default() });
    let notes = plan.notes().join("\n");
    assert!(notes.contains("cuda_graphs:") && notes.contains("gpu_priority:"), "{notes}");
    tick(&mut plan, 3);
    let graphs_only = affine_plan("cpu", GpuLevers { cuda_graphs: true, ..GpuLevers::default() }, &opts, &bytes);
    assert_eq!(graphs_only.residency(), Residency::Host);
    assert!(graphs_only.notes()[0].contains("needs io_binding"), "{:?}", graphs_only.notes());
    let trt = affine_plan(
        "cpu",
        GpuLevers { accelerator: crate::services::neural_audio::vc::engine::options::Accelerator::TensorRtRtx, ..GpuLevers::default() },
        &opts,
        &bytes,
    );
    assert!(trt.notes().iter().any(|n| n.starts_with("tensorrt_rtx:")), "{:?}", trt.notes());
    assert_eq!(trt.outcome().active, "cpu");
}

/// DirectML with the priority lever (Windows, hardware GPU): the session runs on the adapter's
/// high-priority queue, bound I/O gives the CPU results, a tick makes no Rust heap allocation, and
/// padded row copies and CPU-read inputs stay correct.
#[cfg(windows)]
#[test]
fn directml_plan_on_the_high_priority_queue() {
    use crate::services::diagnostics::gpu_mem;
    let Some(adapter) = gpu_mem::adapters().into_iter().find(|a| !a.software) else {
        eprintln!("no hardware GPU; skipped");
        return;
    };
    let bytes = affine_model();
    let opts = VcSessionOptions { cuda_device_index: adapter.index as i32, ..VcSessionOptions::default() };
    let levers = GpuLevers {
        io_binding: true,
        priority: true,
        accelerator: crate::services::neural_audio::vc::engine::options::Accelerator::DmlHighPriority,
        ..GpuLevers::default()
    };
    let mut plan = affine_plan("dml", levers, &opts, &bytes);
    eprintln!("dml plan: {:?} {:?} {:?} {:?}", plan.outcome(), plan.levers(), plan.notes(), plan.vram_mb());
    if plan.outcome().active != "dml" {
        eprintln!("DirectML unavailable here; skipped");
        return;
    }
    assert_eq!(plan.residency(), Residency::Bound);
    if dml_queue::available() {
        assert!(plan.levers().priority, "{:?}", plan.notes());
        assert_eq!(plan.levers().accelerator, crate::services::neural_audio::vc::engine::options::Accelerator::DmlHighPriority);
    }
    for i in 0..4 {
        tick(&mut plan, i);
    }
    let x = [1.5f32; N];
    let k = [-1i64; N];
    let ((), allocs) = allocations_during(|| {
        for _ in 0..20 {
            plan.input_f32("x").unwrap().copy_from_slice(&x);
            plan.input_i64("k").unwrap().copy_from_slice(&k);
            plan.run().unwrap();
            std::hint::black_box(plan.output("y").unwrap()[0]);
        }
    });
    assert_eq!(allocs, 0, "DirectML bound path allocated");
    assert_eq!(plan.output("y").unwrap(), expected_affine(&x, &k).as_slice());
    // Only inputs written since the last run are uploaded: a run without writes keeps the result.
    plan.run().unwrap();
    assert_eq!(plan.output("y").unwrap(), expected_affine(&x, &k).as_slice());
    let (mut enc, mut dec) = enc_dec("dml", levers, &opts);
    check_row_copy(&mut enc, &mut dec);
    let (mut enc, mut dec) = enc_dec_with("dml", levers, levers, &opts, 6);
    check_padded_row_copy(&mut enc, &mut dec);
    check_host_read_inputs("dml", &opts);
}

/// Whether CUDA sessions can run in this test process (provider next to the executable and the
/// NVIDIA runtime found): a CUDA plan that lands on CUDA.
fn cuda_ready(opts: &VcSessionOptions) -> bool {
    crate::services::neural_audio::ort_session::preload_test_cuda_runtime_quiet();
    let bytes = affine_model();
    let plan = affine_plan("cuda", GpuLevers::default(), opts, &bytes);
    let ok = plan.outcome().active == "cuda";
    if !ok {
        eprintln!("CUDA is not usable here ({:?}); skipped", plan.outcome());
    }
    ok
}

/// CUDA with every lever (dev rig): device-resident I/O, a captured CUDA graph, the highest-priority
/// stream; results equal the CPU's tick after tick, no heap allocation per tick, device-to-device
/// row copies, and outputs kept on the device.
#[test]
#[ignore = "needs an NVIDIA GPU, the CUDA provider next to the test executable and the NVIDIA runtime (SIGIL_APP_DATA_DIR or SIGIL_TEST_CUDA_BIN)"]
fn cuda_device_plan_with_graphs_and_priority() {
    let opts = VcSessionOptions::default();
    if !cuda_ready(&opts) {
        return;
    }
    let bytes = affine_model();
    let mut plan = affine_plan("cuda", all_levers(), &opts, &bytes);
    eprintln!(
        "cuda plan: {:?} {:?} {:?} residency {:?} pinned {} vram {:?}",
        plan.outcome(),
        plan.levers(),
        plan.notes(),
        plan.residency(),
        plan.pinned_host_buffers(),
        plan.vram_mb()
    );
    assert_eq!(plan.residency(), Residency::Device);
    assert!(plan.levers().io_binding && plan.levers().priority, "{:?}", plan.notes());
    assert!(plan.levers().cuda_graphs, "graph capture: {:?}", plan.notes());
    assert!(plan.input_f32("x").unwrap().iter().all(|&v| v == 0.0), "the graph check's inputs do not leak");
    for i in 0..10 {
        tick(&mut plan, i);
    }
    let x = [0.75f32; N];
    let k = [2i64; N];
    let ((), allocs) = allocations_during(|| {
        for _ in 0..100 {
            plan.input_f32("x").unwrap().copy_from_slice(&x);
            plan.input_i64("k").unwrap().copy_from_slice(&k);
            plan.run().unwrap();
            std::hint::black_box(plan.output("y").unwrap()[0]);
        }
    });
    assert_eq!(allocs, 0, "the CUDA device path allocated");
    assert_eq!(plan.output("y").unwrap(), expected_affine(&x, &k).as_slice());
    // Device-to-device row copies (graphs on both plans), then outputs kept on the device.
    let (mut enc, mut dec) = enc_dec("cuda", all_levers(), &opts);
    assert_eq!((enc.residency(), dec.residency()), (Residency::Device, Residency::Device));
    check_row_copy(&mut enc, &mut dec);
    enc.set_output_download("m", false).unwrap();
    check_row_copy_device_only(&mut enc, &mut dec);
    assert!(enc.output("m").unwrap_err().contains("stays on the device"));
    // Padded copies device to device, then from a device output that is not downloaded into a
    // legacy (host) and a bound destination.
    let graphs = GpuLevers { io_binding: true, cuda_graphs: true, ..GpuLevers::default() };
    for dec_levers in [graphs, GpuLevers::default(), io_binding()] {
        let (mut enc, mut dec) = enc_dec_with("cuda", graphs, dec_levers, &opts, 6);
        assert_eq!(enc.residency(), Residency::Device);
        check_padded_row_copy(&mut enc, &mut dec);
        enc.set_output_download("m", false).unwrap();
        check_padded_row_copy(&mut enc, &mut dec);
        eprintln!("padded copies into a {:?} destination: ok", dec.residency());
    }
    // Without a graph a CUDA plan binds host tensors (`DEVICE_IO_WITHOUT_GRAPHS`).
    if !super::plan::DEVICE_IO_WITHOUT_GRAPHS && super::plan::CudaTuning::current().device_io.is_none() {
        let mut bound = affine_plan("cuda", io_binding(), &opts, &bytes);
        assert_eq!(bound.residency(), Residency::Bound);
        for i in 0..5 {
            tick(&mut bound, i);
        }
    }
    // Inputs a CUDA kernel reads on the CPU keep the plan off the device path.
    check_host_read_inputs("cuda", &opts);
    // Without the priority lever the plan copies on its own stream while ORT uses its own.
    let mut plain = affine_plan("cuda", GpuLevers { io_binding: true, cuda_graphs: true, ..GpuLevers::default() }, &opts, &bytes);
    assert!(plain.levers().cuda_graphs && !plain.levers().priority);
    for i in 0..5 {
        tick(&mut plain, i);
    }
}

fn check_row_copy_device_only(enc: &mut SessionPlan, dec: &mut SessionPlan) {
    let x: Vec<f32> = (0..30).map(|i| -(i as f32)).collect();
    enc.input_f32("x").unwrap().copy_from_slice(&x);
    enc.run().unwrap();
    let copy = RowCopy { rows: 3, row_len: 4, src_stride: 10, src_offset: 2, dst_stride: 4 };
    enc.copy_rows_to("m", dec, "w", copy).unwrap();
    dec.run().unwrap();
    let want: Vec<f32> = (0..3).flat_map(|c| (0..4).map(move |j| -3.0 * (c * 10 + 2 + j) as f32 + 1.0)).collect();
    assert_eq!(dec.output("o").unwrap(), want.as_slice());
}

// ---------------------------------------------------------------------------------------------
// Dev-rig benchmark
// ---------------------------------------------------------------------------------------------

/// The MIT default voice at the Low-latency shape (40 ms block, 64 ms extra context, partial
/// decode) with deterministic tick inputs, for the dev-rig benchmarks below.
struct DefaultVoiceBench {
    voice: crate::services::neural_audio::vc::import::ImportedVoice,
    f16: Vec<u8>,
    plan: crate::services::neural_audio::vc::engine::plan::ShapePlan,
    emb: usize,
    latent: usize,
    upp: usize,
    gin: usize,
}

impl DefaultVoiceBench {
    fn load() -> Option<DefaultVoiceBench> {
        use crate::services::neural_audio::vc::engine::options::PrecisionMode;
        use crate::services::neural_audio::vc::engine::plan::ShapePlan;
        use crate::services::neural_audio::vc::engine::presets::engine_preset;
        use crate::services::neural_audio::vc::import::{import_file, writer::f16_image};
        use crate::services::neural_audio::RvcPresetProfile;
        use std::path::{Path, PathBuf};

        let Some(models) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from) else {
            eprintln!("SIGIL_TEST_MODELS_DIR is not set; skipped");
            return None;
        };
        crate::services::neural_audio::ort_session::preload_test_cuda_runtime_quiet();
        let templates = Path::new(env!("CARGO_MANIFEST_DIR")).join("resources").join("vc-templates");
        let voice = import_file(&models.join("rvc-default-voice").join("default.pth"), &templates).expect("default voice imports");
        let f16 = f16_image(&voice.weights.f32_image).expect("fp16 image");
        let arch = voice.template.layout.arch.clone();
        let preset = engine_preset(RvcPresetProfile::LowLatency, 40.0, 0, 48_000, 64.0, PrecisionMode::Auto);
        let plan = ShapePlan::new(&voice.template.layout.streaming, &preset, arch.upp * 100, true).expect("shape plan");
        Some(DefaultVoiceBench {
            emb: arch.emb_channels as usize,
            latent: arch.inter_channels as usize,
            upp: arch.upp as usize,
            gin: arch.gin_channels as usize,
            voice,
            f16,
            plan,
        })
    }

    fn generator(
        &self,
        ep: &str,
        levers: &crate::services::neural_audio::vc::engine::options::EffectiveLevers,
        opts: &VcSessionOptions,
    ) -> Result<crate::services::neural_audio::vc::engine::generator::SplitGenerator, String> {
        use crate::services::neural_audio::vc::engine::generator::SplitGenerator;
        use crate::services::neural_audio::vc::import::session::WeightImages;
        let images = WeightImages { f32: &self.voice.weights.f32_image, f16: Some(&self.f16) };
        SplitGenerator::load(&self.voice.template, &images, &self.plan, ep, opts, levers)
    }

    /// Tick inputs from `seed` (0 = the benchmark's fixed inputs): phone, pitch, z, g, f0, src.
    #[allow(clippy::type_complexity)]
    fn inputs(&self, seed: u64) -> (Vec<f32>, Vec<i64>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let (t, f) = (self.plan.enc_frames, self.plan.flow_frames);
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut rnd = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        let g: Vec<f32> = self.voice.weights.speakers[..self.gin].to_vec();
        if seed == 0 {
            let phone: Vec<f32> = (0..t * self.emb).map(|i| ((i % 97) as f32 * 0.013).sin() * 0.5).collect();
            let pitch: Vec<i64> = (0..t).map(|i| 40 + (i as i64 % 60)).collect();
            let z: Vec<f32> = (0..self.latent * f).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
            let f0: Vec<f32> = (0..f).map(|i| 140.0 + 20.0 * (i as f32 * 0.1).sin()).collect();
            let src: Vec<f32> = (0..f * self.upp).map(|i| ((i * 104_729) % 2000) as f32 / 1000.0 - 1.0).collect();
            return (phone, pitch, z, g, f0, src);
        }
        let phone: Vec<f32> = (0..t * self.emb).map(|_| rnd() * 0.5).collect();
        let pitch: Vec<i64> = (0..t).map(|_| 1 + ((rnd() + 1.0) * 127.0) as i64).collect();
        let z: Vec<f32> = (0..self.latent * f).map(|_| rnd() * 0.5).collect();
        let base = 120.0 + 60.0 * (rnd() + 1.0);
        let f0: Vec<f32> = (0..f).map(|i| base * (1.0 + 0.05 * (i as f32 * 0.3).sin())).collect();
        let src: Vec<f32> = (0..f * self.upp).map(|_| rnd()).collect();
        (phone, pitch, z, g, f0, src)
    }
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if v.is_empty() {
        return f64::NAN;
    }
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn median(v: &[f64]) -> f64 {
    percentile(&mut v.to_vec(), 0.5)
}

/// Per-tick `enc` + `flowdec` compute of the MIT default voice (v2 40k, bundled template, P3-C1's
/// `SplitGenerator` over these plans) at the Low-latency shape (40 ms block, 64 ms extra context,
/// partial decode), for each GPU lever combination, with the VRAM the plans hold. Each of
/// `SIGIL_BENCH_REPS` repetitions (default 5) runs every configuration for `SIGIL_BENCH_TICKS`
/// ticks (default 1000) in a new shuffled order (`SIGIL_BENCH_SEED`); the table gives the median
/// over the repetitions of each statistic and the spread (min–max) of the per-repetition p95.
/// `+devio` rows keep device-resident buffers on CUDA without a graph (`SIGIL_CUDA_TUNING=devio=1`,
/// what `plan::DEVICE_IO_WITHOUT_GRAPHS` decides). Prints a table (docs:
/// `docs/benchmarks/VC_BENCHMARKS.md`); asserts only that every configuration runs. Run it on an
/// optimised build (`--release` with LTO off, in a separate `CARGO_TARGET_DIR`).
///
/// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache` (holds `rvc-default-voice/default.pth`),
/// `SIGIL_BENCH_EPS=cuda,dml` (default), `SIGIL_BENCH_CONFIGS` (labels), `SIGIL_BENCH_GPU_LOAD`
/// (duty of the GPU load simulator), plus the CUDA setup in the module docs.
#[test]
#[ignore = "dev-rig benchmark: needs SIGIL_TEST_MODELS_DIR with rvc-default-voice/default.pth and a GPU"]
fn gpu_lever_bench() {
    use crate::services::neural_audio::vc::engine::generator::SplitGenerator;
    use crate::services::neural_audio::vc::engine::options::{Accelerator, EffectiveLevers, Lever, PrecisionMode};
    use std::collections::BTreeMap;
    use std::time::Instant;

    let Some(bench) = DefaultVoiceBench::load() else { return };
    let plan = &bench.plan;
    let (t, f) = (plan.enc_frames, plan.flow_frames);
    let env_num = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let ticks = env_num("SIGIL_BENCH_TICKS", 1000);
    let reps = env_num("SIGIL_BENCH_REPS", 5).max(1);
    let seed = env_num("SIGIL_BENCH_SEED", 1) as u64;
    let eps: Vec<String> = std::env::var("SIGIL_BENCH_EPS")
        .unwrap_or_else(|_| "cuda,dml".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let (phone, pitch, z, g, f0, src) = bench.inputs(0);
    let mut out = vec![0f32; plan.decode_frames * bench.upp];
    let adapter_of = |ep: &str| super::arena::dxgi_adapter_for(ep, 0);
    let process_vram = |ep: &str| -> f64 {
        adapter_of(ep)
            .and_then(crate::services::diagnostics::gpu_mem::process_vram_usage)
            .map(|u| u.process_usage_mb as f64)
            .unwrap_or(f64::NAN)
    };
    // T3 breakdown: this process's VRAM before any GPU session, and after a trivial CUDA session
    // (CUDA context, cuDNN / cuBLAS handles: the floor every CUDA engine in the process shares).
    if eps.iter().any(|e| e == "cuda") {
        let before = process_vram("cuda");
        let bytes = affine_model();
        let mut probe_plan = affine_plan("cuda", GpuLevers::default(), &VcSessionOptions::default(), &bytes);
        tick(&mut probe_plan, 0);
        eprintln!(
            "process VRAM on the NVIDIA adapter: {before:.0} MB before CUDA, {:.0} MB with a CUDA context ({})",
            process_vram("cuda"),
            probe_plan.outcome().active
        );
    }
    let base_tuning = std::env::var(super::plan::CUDA_TUNING_ENV).unwrap_or_default();
    eprintln!(
        "GPU levers, default voice v2 40k, T={t} F={f} (40 ms block, 64 ms extra, partial decode), {reps} x {ticks} ticks, enc+flowdec per tick, build {}, CUDA tuning {:?}",
        if cfg!(debug_assertions) { "debug (numbers are not representative)" } else { "optimised" },
        super::plan::CudaTuning::current()
    );
    struct Config {
        label: &'static str,
        io: bool,
        graphs: bool,
        priority: bool,
        accel: Accelerator,
        devio: bool,
    }
    let c = |label, io, graphs, priority, accel, devio| Config { label, io, graphs, priority, accel, devio };
    let configs = [
        c("legacy", false, false, false, Accelerator::None, false),
        c("io_binding", true, false, false, Accelerator::None, false),
        c("io_binding+devio", true, false, false, Accelerator::None, true),
        c("io+graphs", true, true, false, Accelerator::None, false),
        c("io+priority", true, false, true, Accelerator::None, false),
        c("io+priority+devio", true, false, true, Accelerator::None, true),
        c("all", true, true, true, Accelerator::None, false),
        c("all+accel", true, true, true, Accelerator::TensorRtRtx, false),
    ];
    let only: Option<Vec<String>> = std::env::var("SIGIL_BENCH_CONFIGS").ok().map(|v| v.split(',').map(|s| s.trim().to_string()).collect());
    // `SIGIL_BENCH_GPU_LOAD=<duty>`: the benchmark's GPU load simulator (a MatMul chain on a
    // normal-priority stream / queue of the same EP, like a game) runs while each EP is measured.
    let gpu_load: Option<f64> = std::env::var("SIGIL_BENCH_GPU_LOAD").ok().and_then(|v| v.parse().ok());
    #[derive(Default)]
    struct Runs {
        desc: String,
        p50: Vec<f64>,
        p95: Vec<f64>,
        p99: Vec<f64>,
        max: Vec<f64>,
        enc: Vec<f64>,
        fd: Vec<f64>,
        load_ms: Vec<f64>,
        vram: Vec<f64>,
        allocs: Vec<f64>,
        notes: String,
    }
    let mut rng = seed.wrapping_mul(0x2545_f491_4f6c_dd1d) | 1;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for ep in &eps {
        let load = gpu_load.map(|duty| {
            use crate::services::bench::spec::{GpuLoadSpec, LoadSpec};
            let spec = LoadSpec {
                gpu: Some(GpuLoadSpec { duty, period_ms: 20.0, execution_provider: ep.clone(), device_index: 0, size: 2048, depth: 4 }),
                cpu: None,
            };
            let handle = crate::services::bench::load_sim::LoadHandle::start(&spec).expect("GPU load starts");
            eprintln!("  GPU load: {ep}, duty {duty}");
            handle
        });
        let wanted: Vec<&Config> = configs
            .iter()
            .filter(|cfg| only.as_ref().is_none_or(|o| o.iter().any(|l| l == cfg.label)))
            // DirectML has no graphs and no device path: those rows would repeat others.
            .filter(|cfg| !(ep == "dml" && (cfg.devio || (cfg.graphs && !cfg.priority))))
            .collect();
        let mut runs: BTreeMap<&'static str, Runs> = BTreeMap::new();
        for rep in 0..reps {
            let mut order: Vec<usize> = (0..wanted.len()).collect();
            for i in (1..order.len()).rev() {
                order.swap(i, (next() % (i as u64 + 1)) as usize);
            }
            eprintln!("  {ep} repetition {}: {}", rep + 1, order.iter().map(|&i| wanted[i].label).collect::<Vec<_>>().join(", "));
            for &i in &order {
                let cfg = wanted[i];
                let accel = if ep == "dml" && cfg.accel == Accelerator::TensorRtRtx { Accelerator::DmlHighPriority } else { cfg.accel };
                let tuning = if cfg.devio { format!("{base_tuning},devio=1") } else { base_tuning.clone() };
                std::env::set_var(super::plan::CUDA_TUNING_ENV, &tuning);
                let mut levers = EffectiveLevers { precision: PrecisionMode::Auto, accelerator: accel, ..EffectiveLevers::default() };
                levers.levers.set(Lever::IoBinding, cfg.io);
                levers.levers.set(Lever::CudaGraphs, cfg.graphs);
                levers.levers.set(Lever::GpuPriority, cfg.priority);
                let opts = VcSessionOptions { component_prefix: Some("bench"), ..VcSessionOptions::default() };
                let load_started = Instant::now();
                let loaded = bench.generator(ep, &levers, &opts);
                std::env::set_var(super::plan::CUDA_TUNING_ENV, &base_tuning);
                let mut gen = match loaded {
                    Ok(g) => g,
                    Err(e) => {
                        eprintln!("| {ep} | {} | load failed: {e} |", cfg.label);
                        continue;
                    }
                };
                // Cold load of both plans, including graph capture (plan T9: a cold slot load within 2.5 s).
                let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
                let active = gen.ep_outcomes()[1].1.active.clone();
                let (mut total, mut enc_ms, mut fd_ms) = (Vec::with_capacity(ticks), Vec::with_capacity(ticks), Vec::with_capacity(ticks));
                let run_tick = |gen: &mut SplitGenerator, out: &mut [f32]| -> (f64, f64) {
                    let a = Instant::now();
                    gen.encode(&phone, &pitch).unwrap();
                    let b = Instant::now();
                    gen.decode_window(plan.window_start, &z, &g, &f0, 0.25, &src, out).unwrap();
                    ((b - a).as_secs_f64() * 1000.0, b.elapsed().as_secs_f64() * 1000.0)
                };
                for _ in 0..20 {
                    run_tick(&mut gen, &mut out);
                }
                for _ in 0..ticks {
                    let (e, d) = run_tick(&mut gen, &mut out);
                    enc_ms.push(e);
                    fd_ms.push(d);
                    total.push(e + d);
                }
                let ((), allocs) = allocations_during(|| {
                    for _ in 0..20 {
                        run_tick(&mut gen, &mut out);
                    }
                });
                let (le, lf) = gen.plan_levers();
                let r = runs.entry(cfg.label).or_default();
                r.desc = format!(
                    "{ep}→{active} | {} ({}{}{}{}{}) | {}",
                    cfg.label,
                    if lf.io_binding { "io " } else { "" },
                    if le.cuda_graphs && lf.cuda_graphs { "graphs " } else { "" },
                    if lf.priority { "prio " } else { "" },
                    if lf.accelerator != Accelerator::None { lf.accelerator.as_str() } else { "" },
                    if cfg.devio && !lf.cuda_graphs { "device-io" } else { "" },
                    gen.flowdec_precision()
                );
                r.p50.push(percentile(&mut total.clone(), 0.5));
                r.p95.push(percentile(&mut total.clone(), 0.95));
                r.p99.push(percentile(&mut total.clone(), 0.99));
                r.max.push(total.iter().cloned().fold(0.0, f64::max));
                r.enc.push(percentile(&mut enc_ms, 0.5));
                r.fd.push(percentile(&mut fd_ms, 0.5));
                r.load_ms.push(load_ms);
                r.vram.push(process_vram(&active));
                r.allocs.push(allocs as f64 / 20.0);
                r.notes = gen.notes().to_vec().join("; ");
                drop(gen);
            }
        }
        if let Some(load) = load {
            let report = load.stop();
            eprintln!("  GPU load report: {}", serde_json::to_string(&report.gpu).unwrap_or_default());
        }
        eprintln!("| EP | levers | flowdec | load ms | total p50 | total p95 (min–max) | total p99 | max | enc p50 | flowdec p50 | process VRAM MB | Rust allocs/tick | notes |");
        eprintln!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
        for cfg in &wanted {
            let Some(r) = runs.get(cfg.label) else { continue };
            let lo = r.p95.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = r.p95.iter().cloned().fold(0.0, f64::max);
            eprintln!(
                "| {} | {:.0} | {:.2} | {:.2} ({lo:.2}–{hi:.2}) | {:.2} | {:.2} | {:.2} | {:.2} | {:.0} | {:.1} | {} |",
                r.desc,
                median(&r.load_ms),
                median(&r.p50),
                median(&r.p95),
                median(&r.p99),
                median(&r.max),
                median(&r.enc),
                median(&r.fd),
                median(&r.vram),
                median(&r.allocs),
                r.notes
            );
        }
    }
}

/// The GPU levers follow inputs that change every tick (dev rig): the default voice's `enc` +
/// `flowdec` with every lever (a CUDA graph on CUDA) against the legacy path on the same EP, and an
/// RMVPE plan with a CUDA graph against the plain session plan, on new pseudo-random inputs each
/// tick. A graph that replays stale work or ignores its inputs differs from the legacy path at
/// once (the outputs of consecutive ticks differ by far more than the tolerance).
#[test]
#[ignore = "dev-rig check: needs SIGIL_TEST_MODELS_DIR with rvc-default-voice/default.pth (and upstream-assets/rmvpe/rmvpe.onnx) and a GPU"]
fn gpu_levers_follow_changing_inputs() {
    use crate::services::neural_audio::vc::engine::options::{EffectiveLevers, Lever, PrecisionMode};

    let Some(bench) = DefaultVoiceBench::load() else { return };
    let plan = &bench.plan;
    let rel = |a: &[f32], b: &[f32]| {
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
        num / b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12)
    };
    let levers = |on: bool| {
        let mut l = EffectiveLevers { precision: PrecisionMode::Auto, ..EffectiveLevers::default() };
        for lever in [Lever::IoBinding, Lever::CudaGraphs, Lever::GpuPriority] {
            l.levers.set(lever, on);
        }
        l
    };
    let opts = VcSessionOptions::default();
    for ep in ["cuda", "dml"] {
        let (mut legacy, mut levered) = match (bench.generator(ep, &levers(false), &opts), bench.generator(ep, &levers(true), &opts)) {
            (Ok(a), Ok(b)) if a.ep_outcomes()[1].1.active == ep && b.ep_outcomes()[1].1.active == ep => (a, b),
            _ => {
                eprintln!("{ep} unavailable here; skipped");
                continue;
            }
        };
        eprintln!("{ep}: levers {:?}", levered.plan_levers());
        if ep == "cuda" {
            assert!(levered.plan_levers().1.cuda_graphs, "flowdec graph: {:?}", levered.notes());
        }
        let (mut a, mut b) = (vec![0f32; plan.decode_frames * bench.upp], vec![0f32; plan.decode_frames * bench.upp]);
        let mut previous: Option<Vec<f32>> = None;
        let mut worst = 0f32;
        for tick in 1..=12u64 {
            let (phone, pitch, z, g, f0, src) = bench.inputs(tick);
            let phase = (tick as f64 * 0.37).fract();
            for (gen, out) in [(&mut legacy, &mut a), (&mut levered, &mut b)] {
                gen.encode(&phone, &pitch).unwrap();
                gen.decode_window(plan.window_start, &z, &g, &f0, phase, &src, out).unwrap();
            }
            let err = rel(&b, &a);
            let change = previous.as_ref().map(|p| rel(p, &a)).unwrap_or(f32::NAN);
            eprintln!("  {ep} tick {tick}: levers vs legacy rel L2 {err:.2e}, change from the previous tick {change:.2e}");
            assert!(err < 1e-2, "{ep} tick {tick}: the levered output diverges from the legacy path ({err})");
            if let Some(change) = previous.as_ref().map(|p| rel(p, &a)) {
                assert!(change > 0.05, "{ep} tick {tick}: the inputs barely change the output ({change}); the check is too weak");
            }
            worst = worst.max(err);
            previous = Some(a.clone());
        }
        eprintln!("{ep}: worst rel L2 {worst:.2e} over 12 changing ticks");
    }

    // RMVPE: a CUDA-graph plan against the plain plan, 32 mel frames, new log-mel input each tick.
    let Some(models) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(std::path::PathBuf::from) else { return };
    let rmvpe = models.join("upstream-assets").join("rmvpe").join("rmvpe.onnx");
    if !rmvpe.is_file() {
        eprintln!("{} not found; RMVPE part skipped", rmvpe.display());
        return;
    }
    let frames = 32usize;
    let inputs = [TensorSpec::f32("input", &[1, 128, frames])];
    let outputs = [TensorSpec::f32("output", &[1, frames, 360])];
    let build = |levers: GpuLevers| {
        SessionPlan::build(&PlanSpec {
            component: "test.gpu.rmvpe",
            ep: "cuda",
            opts: &opts,
            model: ModelSource::File(&rmvpe),
            weights: None,
            dims: &[("time", frames as i64)],
            inputs: &inputs,
            outputs: &outputs,
            levers,
        })
    };
    let (Ok(mut plain), Ok(mut graph)) = (build(GpuLevers { tf32: true, ..GpuLevers::default() }), build(GpuLevers { tf32: true, ..all_levers() })) else {
        eprintln!("RMVPE plans could not be built; skipped");
        return;
    };
    if plain.outcome().active != "cuda" || !graph.levers().cuda_graphs {
        eprintln!("RMVPE: no CUDA graph here ({:?}, {:?}); skipped", graph.outcome(), graph.notes());
        return;
    }
    let mut state = 0x5eed_u64;
    let mut previous: Option<Vec<f32>> = None;
    for tick in 0..12 {
        let mel: Vec<f32> = (0..128 * frames)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 24) as f32 * 7.0 - 6.0
            })
            .collect();
        for p in [&mut plain, &mut graph] {
            p.input_f32("input").unwrap().copy_from_slice(&mel);
            p.run().unwrap();
        }
        let (want, got) = (plain.output("output").unwrap(), graph.output("output").unwrap());
        let err = rel(got, want);
        eprintln!("  rmvpe tick {tick}: graph vs plain rel L2 {err:.2e}");
        assert!(err < 1e-2, "RMVPE tick {tick}: the graph output diverges ({err})");
        if let Some(p) = &previous {
            assert!(rel(p, want) > 1e-3, "RMVPE tick {tick}: the input barely changes the output");
        }
        previous = Some(want.to_vec());
    }
}

/// Accuracy of the GPU levers on the default voice: the same `enc` + `flowdec` tick on the CPU
/// (fp32, the reference) and on CUDA / DirectML with every lever and the precision of
/// `SIGIL_BENCH_PRECISION` (`auto` by default, `fp32` for TF32 off), as relative L2 error and max
/// |error| of the returned audio. Same environment as [`gpu_lever_bench`].
#[test]
#[ignore = "dev-rig accuracy check: needs SIGIL_TEST_MODELS_DIR with rvc-default-voice/default.pth and a GPU"]
fn gpu_lever_accuracy_on_the_default_voice() {
    use crate::services::neural_audio::vc::engine::generator::SplitGenerator;
    use crate::services::neural_audio::vc::engine::options::{EffectiveLevers, Lever, PrecisionMode};
    use crate::services::neural_audio::vc::engine::plan::ShapePlan;
    use crate::services::neural_audio::vc::engine::presets::engine_preset;
    use crate::services::neural_audio::vc::import::{import_file, session::WeightImages, writer::f16_image};
    use crate::services::neural_audio::RvcPresetProfile;
    use std::path::{Path, PathBuf};

    let Some(models) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from) else {
        eprintln!("SIGIL_TEST_MODELS_DIR is not set; skipped");
        return;
    };
    crate::services::neural_audio::ort_session::preload_test_cuda_runtime_quiet();
    let templates = Path::new(env!("CARGO_MANIFEST_DIR")).join("resources").join("vc-templates");
    let voice = import_file(&models.join("rvc-default-voice").join("default.pth"), &templates).expect("default voice imports");
    let f16 = f16_image(&voice.weights.f32_image).expect("fp16 image");
    let images = WeightImages { f32: &voice.weights.f32_image, f16: Some(&f16) };
    let arch = voice.template.layout.arch.clone();
    let (emb, c, upp, gin) = (arch.emb_channels as usize, arch.inter_channels as usize, arch.upp as usize, arch.gin_channels as usize);
    let preset = engine_preset(RvcPresetProfile::LowLatency, 40.0, 0, 48_000, 64.0, PrecisionMode::Auto);
    let plan = ShapePlan::new(&voice.template.layout.streaming, &preset, arch.upp * 100, true).expect("shape plan");
    let (t, f) = (plan.enc_frames, plan.flow_frames);
    let phone: Vec<f32> = (0..t * emb).map(|i| ((i % 97) as f32 * 0.013).sin() * 0.5).collect();
    let pitch: Vec<i64> = (0..t).map(|i| 40 + (i as i64 % 60)).collect();
    let z: Vec<f32> = (0..c * f).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
    let g: Vec<f32> = voice.weights.speakers[..gin].to_vec();
    let f0: Vec<f32> = (0..f).map(|i| 140.0 + 20.0 * (i as f32 * 0.1).sin()).collect();
    let src: Vec<f32> = (0..f * upp).map(|i| ((i * 104_729) % 2000) as f32 / 1000.0 - 1.0).collect();
    let precision = match std::env::var("SIGIL_BENCH_PRECISION").as_deref() {
        Ok("fp32") => PrecisionMode::Fp32,
        Ok("fp16") => PrecisionMode::Fp16,
        _ => PrecisionMode::Auto,
    };
    let convert = |ep: &str, precision: PrecisionMode, all: bool| -> (Vec<f32>, String, String) {
        let mut levers = EffectiveLevers { precision, ..EffectiveLevers::default() };
        for l in [Lever::IoBinding, Lever::CudaGraphs, Lever::GpuPriority] {
            levers.levers.set(l, all);
        }
        let mut gen = SplitGenerator::load(&voice.template, &images, &plan, ep, &VcSessionOptions::default(), &levers).unwrap();
        let mut out = vec![0f32; plan.decode_frames * upp];
        for _ in 0..3 {
            gen.encode(&phone, &pitch).unwrap();
            gen.decode_window(plan.window_start, &z, &g, &f0, 0.25, &src, &mut out).unwrap();
        }
        (out, gen.ep_outcomes()[1].1.active.clone(), format!("{} {:?}", gen.flowdec_precision(), gen.plan_levers().1))
    };
    let (reference, _, _) = convert("cpu", PrecisionMode::Fp32, false);
    let norm = reference.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    eprintln!("accuracy vs CPU fp32 (peak {:.3}), tuning {:?}", reference.iter().fold(0f32, |a, x| a.max(x.abs())), super::plan::CudaTuning::current());
    for ep in ["cuda", "dml"] {
        for all in [false, true] {
            let (out, active, what) = convert(ep, precision, all);
            let err: f32 = out.iter().zip(&reference).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
            let max = out.iter().zip(&reference).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            eprintln!("  {ep}→{active} [{what}] rel L2 {:.2e}, max |err| {:.2e}", err / norm, max);
            assert!(err / norm < 0.05, "{ep} {what}: the GPU output diverges from the CPU ({})", err / norm);
        }
    }
}

/// The shared environment CUDA arena (dev rig, classic CUDA provider next to the executable): two
/// sessions opt in, both run correctly on CUDA, and the arena is registered once for the device.
#[test]
#[ignore = "needs an NVIDIA GPU, the CUDA provider next to the test executable and the NVIDIA runtime (SIGIL_APP_DATA_DIR or SIGIL_TEST_CUDA_BIN)"]
fn shared_cuda_arena_serves_several_sessions() {
    use crate::services::neural_audio::ort_session::{build_session_ext, GpuExtras};
    let opts = VcSessionOptions::default();
    if !cuda_ready(&opts) {
        return;
    }
    let bytes = affine_model();
    let extras = GpuExtras { cuda_env_arena: true, ..GpuExtras::default() };
    let dims = |b: ort::session::builder::SessionBuilder, _: &str| b.with_dimension_override("N", N as i64).map_err(|e| e.to_string());
    let mut sessions = Vec::new();
    for _ in 0..2 {
        let built = build_session_ext(Some("test.gpu.env_arena"), "cuda", &opts, ModelSource::Memory(&bytes), &extras, &dims).unwrap();
        eprintln!("env arena session: {:?} applied env_arena={} notes={:?}", built.outcome, built.applied.env_arena, built.applied.notes);
        assert_eq!(built.outcome.active, "cuda");
        if built.applied.env_arena {
            assert!(super::arena::cuda_env_arena_registered(0));
        }
        sessions.push(built.session);
    }
    for s in &mut sessions {
        let x: Vec<f32> = (0..N).map(|i| i as f32 - 3.0).collect();
        let k: Vec<i64> = vec![1; N];
        let xt = ort::value::Tensor::from_array(([1usize, N], x.clone())).unwrap();
        let kt = ort::value::Tensor::from_array(([1usize, N], k.clone())).unwrap();
        let out = s.run(ort::inputs!["x" => xt, "k" => kt]).unwrap();
        let (_, y) = out["y"].try_extract_tensor::<f32>().unwrap();
        assert_eq!(y, expected_affine(&x, &k).as_slice());
    }
}
