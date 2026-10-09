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
//! - The content step (P3B-D): the content encoder's fixed-shape plan (`encode_fixed`, IOBinding,
//!   the features copied into the engine's buffer) and the overlapped F0 hand-over (post and join
//!   on the worker thread), on the stand-in encoder. The dynamic session allocates.
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

/// The SVE's content step (P3B-D) makes no heap allocation per block on its fixed-shape plan with
/// IOBinding (the encoder's run, the features copied into the engine's buffer), at the fixed window
/// of a 40 ms stream, and with a content window longer than the context (the kept window advanced
/// each block), with the tick's F0 hand-over around it (posted to the overlap helper before the
/// content encoder, joined after it; the extractor's own work runs on the helper thread, which the
/// probe does not count). The dynamic session allocates (the probe sees the work).
#[test]
fn sve_content_step_is_allocation_free_on_its_plan() {
    use crate::services::neural_audio::gpu::GpuLevers;
    use crate::services::neural_audio::ort_session::VcSessionOptions;
    use crate::services::neural_audio::vc::engine::content::{ContentEncoder, EmbedderKind};
    use crate::services::neural_audio::vc::engine::content_stats::ContentWindow;
    use crate::services::neural_audio::vc::engine::f0_cache::{F0Cache, F0CacheConfig};
    use crate::services::neural_audio::vc::engine::overlap::F0Helper;
    use crate::services::neural_audio::vc::test_models;
    use crate::services::neural_audio::vc::traits::PitchExtractor;

    /// An extractor with an allocation-free `f0_into` (P3B-E's contract for the GPU extractors).
    struct Flat;
    impl PitchExtractor for Flat {
        fn f0(&mut self, x: &[f32]) -> Result<Vec<f32>, String> {
            Ok(vec![140.0; x.len() / 160])
        }
        fn f0_into(&mut self, x: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
            out.clear();
            out.resize(x.len() / 160, 140.0);
            Ok(())
        }
    }

    let dir = std::env::temp_dir().join(format!("sigil-hot-content-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("enc.onnx");
    std::fs::write(&path, test_models::content_encoder_bytes()).unwrap();
    let opts = VcSessionOptions::default();
    let mut enc = ContentEncoder::load(EmbedderKind::ContentVec768, &path, "explicit", "cpu", &opts).unwrap();
    let levers = GpuLevers {
        io_binding: true,
        ..GpuLevers::default()
    };
    let (ctx, block) = (7_040usize, 640usize);
    let stream: Vec<f32> = (0..ctx + block * 80)
        .map(|i| 0.3 * (i as f32 * 0.021).sin())
        .collect();
    let mut helper = F0Helper::default();
    helper.configure(true, None);
    let mut pitch: Box<dyn PitchExtractor> = Box::new(Flat);
    let mut f0_cache = F0Cache::new(F0CacheConfig {
        margin_frames: 4,
        capacity_frames: 256,
    });
    let mut f0 = Vec::with_capacity(64);
    for (label, window_len) in [("context window", ctx), ("1 s content window", 16_000)] {
        let mut feats: Vec<f32> = Vec::new();
        let mut lease = None;
        let mut ring = ContentWindow::default();
        let mut tick = |k: usize, enc: &mut ContentEncoder, feats: &mut Vec<f32>| {
            let pos = k * block;
            let mono = &stream[pos..pos + ctx];
            let job = helper
                .post(&mut pitch, &mut f0_cache, mono, pos as u64, false, 44)
                .expect("the helper takes the job");
            let window = ring.window(mono, pos as u64, window_len);
            let (n, _) = enc.encode_fixed(window, levers, &mut lease, feats).unwrap();
            std::hint::black_box(feats[n * 768 - 1]);
            job.join(&mut f0).unwrap();
            std::hint::black_box(f0[43]);
        };
        // Warm-up: the plan is built, the buffers reach their sizes.
        for k in 0..5 {
            tick(k, &mut enc, &mut feats);
        }
        let ((), allocs) = allocations_during(|| {
            for k in 5..25 {
                tick(k, &mut enc, &mut feats);
            }
        });
        eprintln!("SVE content step heap allocations per 20 blocks ({label}): plan {allocs}");
        assert_eq!(allocs, 0, "{label}: the content plan step must not allocate per block");
    }
    // The dynamic session (the legacy path) allocates: the probe sees the work.
    let window = &stream[..ctx];
    let mut feats = Vec::with_capacity(32 * 768);
    enc.encode(window, &mut feats).unwrap();
    let ((), dynamic_allocs) = allocations_during(|| {
        for _ in 0..5 {
            enc.encode(window, &mut feats).unwrap();
        }
    });
    assert!(dynamic_allocs > 0, "the dynamic content session allocates (the probe sees the work)");
    let _ = std::fs::remove_dir_all(&dir);
}
