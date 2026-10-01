//! Allocation-free hot path (plan T2: "the GPU hot path makes zero heap allocations per block,
//! checked by a counting-allocator test"), with the crate's one counting allocator
//! (`neural_audio::gpu::alloc_probe`, P3-C2).
//!
//! - The voice changer's real-time I/O path per block, exactly as the worker and the paced
//!   benchmark run it (`voice_changer::{budget, playback}` through `vc_rt`): capture push, backlog
//!   ingest and budget, block take, output queue, ring and device playout.
//! - The Sigil Voice Engine's GPU hot path per block: the split generator's `enc` and partial
//!   `flowdec` session runs at fixed shapes with IOBinding (host tensors bound once on the CPU
//!   rung, the same plan code as the GPU rungs), on the tiny template fixture. The legacy
//!   host-tensor path allocates, which proves the probe sees the work.
//!
//! Only Rust heap allocations are counted; ONNX Runtime's C++ allocations go to the CRT heap.

use crate::services::neural_audio::gpu::alloc_probe::allocations_during;

use super::vc_rt::{InputSide, OutputSide};

/// Steady state of the real-time I/O path: after warm-up, 200 blocks (capture callbacks, worker
/// wakes, block takes, output pushes and device callbacks) make no heap allocation.
#[test]
fn realtime_io_path_is_allocation_free() {
    let (rate, ch_in, ch_out, block) = (48_000u32, 1usize, 2usize, 512usize);
    let mut input = InputSide::new(rate, ch_in, block, 300.0);
    let mut output = OutputSide::new(rate, ch_out, 35.0, 70.0, rate, ch_in, 480 * ch_in);
    let capture = vec![0.1f32; 480 * ch_in];
    let converted = vec![0.2f32; block * ch_out];
    let mut work: Vec<f32> = Vec::with_capacity(block * ch_in);
    let step = |input: &mut InputSide, output: &mut OutputSide, work: &mut Vec<f32>| {
        // ~10.7 ms of device time per block: 1-2 capture callbacks, a wake, a block, a playback
        // callback (as the paced driver orders them).
        input.capture(&capture);
        input.wake(block);
        if input.take_block(work, block) {
            std::hint::black_box(&work[..]);
            output.push(&converted);
        }
        output.callback(480, None);
    };
    // Warm-up: the rings, backlog and queue reach their steady sizes.
    for _ in 0..400 {
        step(&mut input, &mut output, &mut work);
    }
    let ((), allocs) = allocations_during(|| {
        for _ in 0..200 {
            step(&mut input, &mut output, &mut work);
        }
    });
    assert_eq!(allocs, 0, "the real-time I/O path allocated {allocs} times in 200 blocks");
}

/// The SVE's per-block GPU work (`enc` over the context, `flowdec` over the return window) makes
/// no heap allocation with IOBinding, at the fixed shapes of a 40 ms stream.
#[test]
fn sve_generator_hot_path_is_allocation_free_with_io_binding() {
    use crate::services::neural_audio::ort_session::VcSessionOptions;
    use crate::services::neural_audio::vc::engine::generator::tests::tiny_voice;
    use crate::services::neural_audio::vc::engine::generator::SplitGenerator;
    use crate::services::neural_audio::vc::engine::options::{EffectiveLevers, Lever, PrecisionMode};
    use crate::services::neural_audio::vc::engine::plan::ShapePlan;
    use crate::services::neural_audio::vc::import::session::WeightImages;

    let voice = tiny_voice();
    let arch = voice.template.layout.arch.clone();
    let (emb, c, upp) = (arch.emb_channels as usize, arch.inter_channels as usize, arch.upp as usize);
    let images = WeightImages {
        f32: &voice.weights.f32_image,
        f16: None,
    };
    let plan = ShapePlan::for_stream(&voice.template.layout.streaming, (upp * 100) as u32, true, 640, 640 + 481, 1_024).unwrap();
    let opts = VcSessionOptions::default();
    let mut bound_levers = EffectiveLevers {
        precision: PrecisionMode::Fp32,
        ..EffectiveLevers::default()
    };
    bound_levers.levers.set(Lever::PartialDecode, true);
    bound_levers.levers.set(Lever::IoBinding, true);
    let mut legacy_levers = bound_levers.clone();
    legacy_levers.levers.set(Lever::IoBinding, false);
    let mut bound = SplitGenerator::load(&voice.template, &images, &plan, "cpu", &opts, &bound_levers).unwrap();
    let mut legacy = SplitGenerator::load(&voice.template, &images, &plan, "cpu", &opts, &legacy_levers).unwrap();
    let (enc_l, fd_l) = bound.plan_levers();
    assert!(enc_l.io_binding && fd_l.io_binding, "the plans bind their I/O: {enc_l:?} {fd_l:?}");
    let t = plan.enc_frames;
    let f = plan.flow_frames;
    let phone: Vec<f32> = (0..t * emb).map(|i| 0.3 * (0.01 * i as f32).sin()).collect();
    let pitch: Vec<i64> = (0..t).map(|j| 40 + (j as i64 % 100)).collect();
    let f0: Vec<f32> = (0..f).map(|j| 120.0 + j as f32).collect();
    let z = vec![0.1f32; c * f];
    let src = vec![0.05f32; f * upp];
    let g = vec![0.2f32; voice.weights.speaker_dim];
    let mut out = vec![0.0f32; plan.decode_frames * upp];
    let tick = |gen: &mut SplitGenerator, out: &mut Vec<f32>| {
        gen.encode(&phone, &pitch).unwrap();
        gen.decode_window(plan.window_start, &z, &g, &f0, 0.0, &src, out).unwrap();
        std::hint::black_box(out[0]);
    };
    for _ in 0..5 {
        tick(&mut bound, &mut out);
        tick(&mut legacy, &mut out);
    }
    let ((), bound_allocs) = allocations_during(|| {
        for _ in 0..20 {
            tick(&mut bound, &mut out);
        }
    });
    let ((), legacy_allocs) = allocations_during(|| {
        for _ in 0..20 {
            tick(&mut legacy, &mut out);
        }
    });
    eprintln!("SVE generator heap allocations per 20 blocks: IOBinding {bound_allocs}, legacy {legacy_allocs}");
    assert_eq!(bound_allocs, 0, "the bound generator path must not allocate per block");
    assert!(legacy_allocs > 0, "the legacy host-tensor path allocates (the probe sees the work)");
}
