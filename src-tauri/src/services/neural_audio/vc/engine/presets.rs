//! Engine presets (P3-C1) and the benchmark-decided lever defaults.
//!
//! The UI presets (`rvc_preset`: Streamer, Low latency, Quality, Studio-TRT) map to engine shapes
//! (block, context, precision) here. [`auto_lever`] is the table that `auto` levers resolve
//! through: it starts `false` for every lever (the proven legacy path), and a lever is switched on
//! for a preset / EP only when the benchmark (`Sigil.exe --headless bench`, P3-F) shows it meets
//! the targets without a p95 regression over 5 % (plan global rule); record the numbers next to
//! the change.
//!
//! # Frozen by P3-S0 (signatures; P3-C1 implements, P3-F / P3-I0 update the table)
//! - [`EnginePreset`] fields, [`engine_preset`], [`auto_lever`]`(profile, lever, base_ep) -> bool`.
//!
//! # Added by the P3-C1 review fixes
//! - [`MAX_DERIVED_BLOCK_MS`]: a block derived from the device block is capped at the plan's
//!   longest (the fixed-block harness runs several model blocks per device block).
//! - [`byo_lever`]: the table below is measured on the template engine; a bring-your-own ONNX
//!   voice keeps its proven zero `rnd` unless the user turns `anchored_noise` on.
//!
//! # The table (P3-C1 measurements, 2026-09-29, dev rig H2: RTX 4090 / 7800X3D, debug test build)
//! `engine::tests::sve_context_scaling_bench` (MIT default voice, v2 40k, 40 ms block, dio F0;
//! component medians of 150 blocks; first with a stand-in content encoder, then with P3-A's
//! ContentVec export: fp16 on the GPU EPs, int8 on the CPU). The four algorithmic levers of the
//! template engine are on by default on every preset and EP:
//! - `partial_decode`: flowdec stays flat as the extra context grows (DirectML fp16 3.0 → 3.3 ms
//!   from 64 ms to 2 s of extra context, CPU 143 → 142 ms) where the full decode grows (DirectML
//!   3.6 → 6.6 ms, CPU 124 → 1060 ms); the returned audio equals the full decode's tail within fp
//!   error (`generator::tests::partial_decode_equals_the_full_decode_tail`, ADR 0004 parity).
//! - `anchored_noise` and `nsf_phase0`: no measurable cost (the noise of a block's new frames and a
//!   phase sum per tick); consecutive ticks decode identical audio on the frames they share away
//!   from the window edges, where fresh noise and phase 0 differ by the signal level
//!   (`generator::tests::anchored_noise_and_phase0_make_overlaps_identical`). On the default voice
//!   with ContentVec (`tests::sve_render_default_voice`, two speech clips) the SOLA joins correlate
//!   0.76 with all four levers vs 0.69 on the legacy paths (0.69 without anchored noise, 0.74
//!   without phase0); see the engine docs for the lookahead that raises it further.
//! - `f0_cache`: the extractor analyses the new block plus 20 frames instead of the whole context
//!   (24 frames vs 44 / 56 / 204 at 64 ms / 512 ms / 2 s of extra context: dio 11 ms vs 19 / 23 /
//!   87 ms in the debug build) and matches a whole-context extraction within 5 cents
//!   (`f0_cache::tests`, dio: identical).
//!
//! With all four on, block p95 grows by 3 % (DirectML) and -3 % (CUDA) from 64 ms to 512 ms of
//! extra context with the stand-in encoder, and by 10 % / -13 % with ContentVec fp16 (plan T2:
//! ≤ 30 %; ContentVec itself stays at 5-9 ms from 0.44 s to 2 s of context on the 4090). With the
//! F0 cache off it grows 22 % / 56 % (stand-in) and 13 % / 19 % (ContentVec) at 512 ms, and 3-4x
//! at 2 s.
//!
//! # The GPU levers (P3-I0, from the P3-C2 and P3-F measurements, 2026-09-30, H2)
//! - `io_binding` on for the CUDA and DirectML rungs, `cuda_graphs` on for CUDA. P3-C2
//!   (`gpu::tests::gpu_lever_bench`, default voice, enc + flowdec per 40 ms tick): CUDA p50 / p95 /
//!   p99 12.2 / 16.1 / 26.1 ms on the legacy path vs 3.3 / 4.0 / 4.2 ms with IOBinding + graphs
//!   (21.2 → 6.7 ms p95 under the 70 % GPU load simulator), output bit-identical, 38 → 0 host
//!   allocations per tick; DirectML 4.17 / 4.94 → 3.93 / 4.59 ms with IOBinding. P3-F (whole engine,
//!   release build, `docs/benchmarks/VC_BENCHMARKS.md`): CUDA per-call p95 16.2 → 10.8 ms with
//!   SwiftF0, calls over budget under the H2 load 2-62 % → 0.4 %, CPU 0.37 → 0.25 cores, VRAM
//!   +24 MB; DirectML within noise (IOBinding keeps the tick allocation-free, T2). With the levers
//!   the pitch bank also runs RMVPE as its CUDA-graph plan (P3-D: 8.5 → 3.3 ms per tick).
//! - `gpu_priority` stays off: in effect (stream priority -5 vs 0) but no gain against the
//!   in-process load (graphs alone p95 6.67 ms vs 7.51 ms with the priority stream under 70 % load),
//!   and CUDA stream priority does not reach across processes (a game). Revisit with a live-record
//!   run.
//! - `gpu_knn` (P3-D, exact GPU retrieval) and the `governor` (P3-E) stay off until the benchmark
//!   measures them on a voice with an index / under the contention runs.

use crate::services::neural_audio::vc::rvc::RvcPresetProfile;

use super::options::{Lever, LeverMode, PrecisionMode};
use super::plan::MAX_BLOCK_FRAMES;

/// Engine shapes of a run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnginePreset {
    pub profile: RvcPresetProfile,
    /// New audio per tick (ms); rounded up by the plan to the 10 ms F0 hop.
    pub block_ms: f64,
    /// Extra context before the block (ms).
    pub extra_context_ms: f64,
    pub precision: PrecisionMode,
}

/// The preset of a run: `block_ms` 0 derives the block from the device block (`chunk_frames` at
/// `device_rate`), as the legacy harness does, capped at the longest model block (200 ms): a longer
/// device block (a 12288-frame CPU buffer, a 16 kHz Bluetooth microphone with large chunks) then
/// runs several model blocks per device block through the fixed-block pre-accumulator instead of
/// refusing the voice. An explicit `block_ms` is the user's and is checked by the plan.
pub fn engine_preset(
    profile: RvcPresetProfile,
    block_ms: f64,
    chunk_frames: u32,
    device_rate: u32,
    extra_context_ms: f64,
    precision: PrecisionMode,
) -> EnginePreset {
    let block_ms = if block_ms > 0.0 {
        block_ms
    } else if device_rate > 0 {
        (f64::from(chunk_frames) * 1000.0 / f64::from(device_rate)).min(MAX_DERIVED_BLOCK_MS)
    } else {
        0.0
    };
    EnginePreset {
        profile,
        block_ms,
        extra_context_ms,
        precision,
    }
}

/// Longest block derived from the device block (the plan's longest block).
pub const MAX_DERIVED_BLOCK_MS: f64 = (MAX_BLOCK_FRAMES * 10) as f64;

/// Whether a lever the run resolved on (`resolved_on`, from the user's `mode` through
/// [`auto_lever`]) also applies to a bring-your-own ONNX voice (the legacy `RvcModel`). The table
/// above is measured on the template engine only: `anchored_noise` replaces the BYO `rnd` input's
/// proven zeros with Gaussian noise, which no benchmark has measured on a BYO voice (and the legacy
/// harness's grid is not hop-aligned, so its noise is only approximately anchored), so an `auto`
/// resolution leaves BYO voices on the zeros and only an explicit `on` enables it there. Every other
/// lever passes through (the factory turns off what the export cannot run).
pub fn byo_lever(lever: Lever, mode: LeverMode, resolved_on: bool) -> bool {
    match lever {
        Lever::AnchoredNoise => resolved_on && mode == LeverMode::On,
        _ => resolved_on,
    }
}

/// Default of an `auto` lever for `profile` on the base EP `base_ep` (`cpu` / `cuda` / `dml`).
/// Every lever starts off (legacy path) until a benchmark proves it; see the module docs for the
/// measurements behind each `true`.
pub fn auto_lever(_profile: RvcPresetProfile, lever: Lever, base_ep: &str) -> bool {
    let base = base_ep.trim();
    match lever {
        Lever::PartialDecode | Lever::AnchoredNoise | Lever::NsfPhase0 | Lever::F0Cache => true,
        Lever::IoBinding => matches!(base, "cuda" | "dml"),
        Lever::CudaGraphs => base == "cuda",
        Lever::GpuKnn | Lever::GpuPriority | Lever::Governor => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_comes_from_the_device_block_when_unset() {
        let p = engine_preset(RvcPresetProfile::Streamer, 0.0, 480, 48_000, 64.0, PrecisionMode::Auto);
        assert!((p.block_ms - 10.0).abs() < 1e-9);
        let p = engine_preset(RvcPresetProfile::Quality, 80.0, 480, 48_000, 192.0, PrecisionMode::Fp32);
        assert_eq!((p.block_ms, p.extra_context_ms, p.precision), (80.0, 192.0, PrecisionMode::Fp32));
        assert_eq!(
            engine_preset(RvcPresetProfile::Streamer, 0.0, 480, 0, 0.0, PrecisionMode::Auto).block_ms,
            0.0
        );
    }

    /// A derived block longer than the plan's longest is capped (several model blocks per device
    /// block), so the plan builds; an explicit block is left to the plan's check.
    #[test]
    fn derived_blocks_are_capped_to_the_longest_model_block() {
        use crate::services::neural_audio::vc::engine::plan::ShapePlan;
        use crate::services::neural_audio::vc::import::layout::Streaming;
        let streaming = Streaming {
            flow_margin: 24,
            dec_margin: 11,
            noise_scale: 0.66666,
            knn_k: 8,
            knn_coarse_k: 64,
            min_index_rows: 64,
        };
        // 256 ms at 48 kHz (a CPU user's 12288-frame buffer) and 224 ms on a 16 kHz device.
        for (chunk, rate) in [(12_288u32, 48_000u32), (3_584, 16_000), (48_000, 48_000)] {
            let p = engine_preset(RvcPresetProfile::Streamer, 0.0, chunk, rate, 64.0, PrecisionMode::Auto);
            assert_eq!(p.block_ms, MAX_DERIVED_BLOCK_MS, "{chunk}@{rate}");
            let plan = ShapePlan::new(&streaming, &p, 40_000, true).unwrap();
            assert_eq!(plan.block_16k, MAX_BLOCK_FRAMES * 160);
        }
        // Shorter device blocks are unchanged.
        let p = engine_preset(RvcPresetProfile::Streamer, 0.0, 9_600, 48_000, 64.0, PrecisionMode::Auto);
        assert!((p.block_ms - 200.0).abs() < 1e-9);
        let p = engine_preset(RvcPresetProfile::Streamer, 0.0, 3_840, 48_000, 64.0, PrecisionMode::Auto);
        assert!((p.block_ms - 80.0).abs() < 1e-9);
        // An explicit block over the limit still fails in the plan (the params clamp it first).
        let p = engine_preset(RvcPresetProfile::Quality, 250.0, 480, 48_000, 64.0, PrecisionMode::Auto);
        assert!(ShapePlan::new(&streaming, &p, 40_000, true).is_err());
    }

    /// `anchored_noise` reaches a bring-your-own ONNX voice only when the user turned it on; `auto`
    /// keeps the proven zero `rnd` there. Other levers pass through.
    #[test]
    fn byo_voices_get_anchored_noise_only_when_asked() {
        assert!(!byo_lever(Lever::AnchoredNoise, LeverMode::Auto, true));
        assert!(byo_lever(Lever::AnchoredNoise, LeverMode::On, true));
        assert!(!byo_lever(Lever::AnchoredNoise, LeverMode::On, false), "not runnable");
        assert!(!byo_lever(Lever::AnchoredNoise, LeverMode::Off, false));
        for lever in Lever::ALL.into_iter().filter(|&l| l != Lever::AnchoredNoise) {
            for mode in [LeverMode::Auto, LeverMode::On, LeverMode::Off] {
                for on in [true, false] {
                    assert_eq!(byo_lever(lever, mode, on), on, "{}", lever.as_str());
                }
            }
        }
    }

    #[test]
    fn only_the_measured_levers_default_on() {
        for profile in [
            RvcPresetProfile::Streamer,
            RvcPresetProfile::LowLatency,
            RvcPresetProfile::Quality,
            RvcPresetProfile::StudioTrt,
        ] {
            for ep in ["cpu", "cuda", "dml"] {
                for lever in Lever::ALL {
                    let on = auto_lever(profile, lever, ep);
                    let algorithmic = matches!(lever, Lever::PartialDecode | Lever::AnchoredNoise | Lever::NsfPhase0 | Lever::F0Cache);
                    let gpu = match lever {
                        Lever::IoBinding => ep != "cpu",
                        Lever::CudaGraphs => ep == "cuda",
                        _ => false,
                    };
                    assert_eq!(on, algorithmic || gpu, "{profile:?} {ep} {}", lever.as_str());
                }
            }
        }
    }
}
