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
//!
//! # P3B defaults (P3B-I0, 2026-10; evidence in `scripts/bench/specs/p3b/<letter>/README.md`, ADR 0010)
//! Every number is an ABBA-interleaved comparison inside one executable (`run-perf-gate.ps1`), and a
//! lever that can lower fidelity never became a default (rule R6). The engine has no preset table
//! for these; each switch lives next to its code, with its kill switch.
//!
//! On (the lossless or measured-better paths; kill switch in brackets):
//! - the content encoder as a fixed-shape session plan (`SIGIL_VC_CONTENT_PLAN=0`; CUDA content
//!   p50 4.91 -> 1.82 ms, DirectML 4.27 -> 1.49 ms, offline call p95 CUDA 11.66 -> 7.40 ms, DirectML
//!   12.17 -> 8.98 ms) and the F0 overlap helper (`SIGIL_VC_F0_OVERLAP=0`; bit-identical audio);
//! - final shapes at load and the parallel cold start (`SIGIL_VC_FINAL_SHAPES=0`,
//!   `SIGIL_VC_PARALLEL_LOAD=0`; cold load CUDA 1837 -> 1009 ms, DirectML 2504 -> 1300 ms), with
//!   `enc` / `flowdec` building on two threads;
//! - the device-resident generator chain (`SIGIL_VC_DEVICE_CHAIN=0`; bit-identical audio, 4 -> 3
//!   syncs per call), the hybrid CUDA wait (`SIGIL_CUDA_TUNING=wait=spin`; worker CPU down 84-94 %
//!   on every row, call p95 / p99 neutral), arena shrink before the graph capture
//!   (`SIGIL_CUDA_TUNING=shrink=0`; -40 MB; a capture whose shrink run is refused retries without
//!   it) and the CPU arena (`SIGIL_VC_CPU_ARENA=0`; RTF -5.9 %);
//! - F0: the parallel SwiftF0 / RMVPE fusion lane (`SIGIL_VC_FUSION_PARALLEL=0`; bit-identical to
//!   the serial path, CUDA F0 stage 4.66 -> 3.12 ms). The F0 cache itself is P3's
//!   (`F0Geometry::DEFAULT`): the primed cold start and onset continuity are opt-in, see below.
//!
//! Off (measured, but they trade fidelity, memory or safety for the number; every one is a switch):
//! - `f0_decode` and `content_norm` (levers, off in [`auto_lever`]): the cross-tick decoder did not
//!   improve pitch on real speech and lowered speaker similarity; running normalisation raises
//!   speaker cosine by about 0.01 at +0.04 dB MCD and needs an unpublished encoder variant;
//! - the primed F0 cold start and onset continuity (`SIGIL_VC_F0_PRIME=1`,
//!   `SIGIL_VC_F0_CONTINUITY=1`, or both with `SIGIL_VC_F0_GEOMETRY=primed`): they keep every
//!   extractor call inside its plan, but moved 3 to 22 of about 5,000 fed frames on the pause clips
//!   (one by an octave) and were never measured with FCPE (rule R6);
//! - the 32-frame RMVPE plan (`SIGIL_RMVPE_PLAN_FRAMES=32`, "VRAM saver": DirectML 854 -> 654 MB,
//!   but gross pitch error 4.9 -> 8.2 % on noisy and creaky synthetic speech), RMVPE fp16 under
//!   `auto` (its default-plan gate missed on DirectML), filled F0 calls
//!   (`SIGIL_VC_F0_GEOMETRY=filled`, the measured combination with priming and continuity;
//!   `SIGIL_VC_F0_LEAD=fill` alone changes only the lead; speaker cosine -0.02), 1 s of content
//!   context;
//! - the adaptive output cushion (it underran under the 70 % loads), `device_block=model_divisor`
//!   (-17.4 ms paced on CUDA, not yet measured under every load, on DirectML / CPU or on a live
//!   device), a 10 ms Low-latency lookahead (+10 ms for +0.03 speaker similarity), the 10 ms
//!   crossfade (fails the T5 v2 gate), the frame-aligned SOLA tail (cosine v2 -0.014), the GPU
//!   prewarm (no first-load gain, holds a 338 MB CUDA context), fp16 kNN storage on CUDA
//!   (rounding 2.7e-4; the exact blend stays), the GPU process priority, denormals as zero,
//!   `dynamic_block_base` and the shared DirectML queue (`SIGIL_VC_DML_SHARED=1`: the cold load is
//!   about 0.1 s faster, but the content stage's p95 regressed on the one queue, +22 % under a
//!   70 % GPU load, and the gate's 0.05 s bound was not met);
//! - the CPU int8 ContentVec stays first on VNNI CPUs: the downstream rule marginally fired
//!   (speaker cosine -0.0026 against 0.002, MCD and V/UV better), but the fp32 encoder's CPU cost
//!   was not measured and the Streamer is already over its RTF target.

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

/// The engine block (ms) the Streamer preset takes on the processor when no block is set, on a PC with
/// [`CPU_STREAMER_FULL_CORES`] or more physical cores: the 120 ms shape the CPU rows of the benchmark
/// measure (RTF p99 0.74-0.89 on the 6-thread policy; 8 cores at 120 ms dropped no input under the game
/// stand-in). The block derived from the device block (20 ms at the default 512 frames) cannot keep up
/// on the processor: a call has a large fixed part (the content encoder's context, the F0 window) however
/// short the block is: a headless paced run of the 20 ms shape on 6 threads measured a call of p50 57 ms
/// and p99 76 ms, a real-time factor of 3.8 (the 120 ms rows: p99 89-107 ms), with 26 s of 40 s of input
/// dropped; a call longer than its block falls behind for ever. The longer block adds latency, not
/// distortion (the block is only how much new audio each call takes): about 270 ms end to end at 120 ms
/// against a voice that breaks up. An explicit block (the setting, or Autotune's pick) always wins, and so
/// does a derived one of [`CPU_STREAMER_KEEP_FROM_MS`] or more. Fewer cores take a longer block, see
/// [`cpu_streamer_block_for_cores`].
pub const CPU_STREAMER_AUTO_BLOCK_MS: f64 = 120.0;

/// The automatic Streamer block on the processor of a PC with exactly [`CPU_STREAMER_SIX_CORES`] (6) physical cores:
/// 160 ms, the user's decision of 2026-10-11 between 160 and 200 ms. At 120 ms, 6 cores dropped 9.5 % of the input under
/// the game stand-in (three passes). At 160 ms the two cells on file dropped none (RTF p99 0.88 paced and 1.06 under the
/// game, 5.3 % of the blocks over budget there; 99.7 % and 99.4 % of the converted audio played; about 355 ms end to end).
/// One quick check of the rule with the PC busy (44 % total CPU) dropped 0.5 % of the input and played 99.04 % of the
/// converted audio (p50 345 ms, RTF p99 1.29); the same check at 200 ms dropped none and played 97.86 % (p50 411 ms). The
/// user judged the 160 ms run's drop to be the busy PC and chose 160 ms for its lower delay (about 60 ms less than 200 ms).
/// (`docs/benchmarks/VC_BENCHMARKS.md` -> "The Streamer's automatic block by physical cores").
pub const CPU_STREAMER_AUTO_BLOCK_6_CORES_MS: f64 = 160.0;

/// The automatic Streamer block on the processor of a PC with fewer than [`CPU_STREAMER_SIX_CORES`] physical cores
/// (5 or fewer, and 0 when the probe could not tell). At 120 ms, 5 and 4 cores dropped 13.9 % and 19.7 % of the input
/// under the game stand-in (three passes). At 200 ms none of the 4-core runs under the game dropped any input (RTF p99
/// 0.83-0.96, except 1.06 for one game cell; a single check of the rule: 0.0 % dropped, 98.90 % played, p50 421 ms). The price
/// is delay (about 390-490 ms end to end) and an output queue with little margin at 200 ms blocks (an open item of the
/// benchmark document: 0.4-2.0 s of dry output a minute and 96.4-99.4 % of the converted audio played). 5 cores were
/// never run on a longer block and take the same figure; 1 to 3 cores were not run at all. On a small PC the least input
/// dropped matters more than the delay (`docs/benchmarks/VC_BENCHMARKS.md` -> "The Streamer's automatic block by
/// physical cores").
pub const CPU_STREAMER_AUTO_BLOCK_FEW_CORES_MS: f64 = 200.0;

/// Physical cores from which the automatic block is [`CPU_STREAMER_AUTO_BLOCK_MS`] (120 ms). 7 cores were not
/// run: they sit between the 8-core rows that keep up and the 6-core rows that do not.
pub const CPU_STREAMER_FULL_CORES: u32 = 7;

/// The physical core count whose automatic block is [`CPU_STREAMER_AUTO_BLOCK_6_CORES_MS`] (160 ms): fewer cores take
/// [`CPU_STREAMER_AUTO_BLOCK_FEW_CORES_MS`] (200 ms), more take [`CPU_STREAMER_AUTO_BLOCK_MS`] (120 ms).
pub const CPU_STREAMER_SIX_CORES: u32 = 6;

/// The block (ms) the Streamer on the processor takes by itself on a PC with `physical_cores` physical cores, in three
/// tiers: 120 ms from 7 cores, 160 ms with exactly 6, 200 ms with 5 or fewer (0, the probe could not tell, counts as the
/// smallest PC, as the Streamer's thread policy does). It is the block of a derived block shorter than
/// [`CPU_STREAMER_KEEP_FROM_MS`] (see [`cpu_streamer_block`]); the inspector (`options.ts`
/// `streamerAutoBlockMs`) carries the same table, and a test of each side pins it.
pub fn cpu_streamer_block_for_cores(physical_cores: u32) -> f64 {
    if physical_cores >= CPU_STREAMER_FULL_CORES {
        CPU_STREAMER_AUTO_BLOCK_MS
    } else if physical_cores >= CPU_STREAMER_SIX_CORES {
        CPU_STREAMER_AUTO_BLOCK_6_CORES_MS
    } else {
        CPU_STREAMER_AUTO_BLOCK_FEW_CORES_MS
    }
}

/// A block derived from the device block that is this long (ms) or longer is a deliberate big device
/// chunk (the default 512 frames derive 10.7 ms) and is kept as it is: only the shorter ones are
/// raised to the block of the PC's cores ([`cpu_streamer_block_for_cores`]).
pub const CPU_STREAMER_KEEP_FROM_MS: f64 = 60.0;
const _: () = assert!(CPU_STREAMER_KEEP_FROM_MS < CPU_STREAMER_AUTO_BLOCK_MS);
// Fewer cores take a longer block: 120 ms (7 or more), 160 ms (6), 200 ms (5 or fewer).
const _: () = assert!(CPU_STREAMER_AUTO_BLOCK_MS < CPU_STREAMER_AUTO_BLOCK_6_CORES_MS);
const _: () = assert!(CPU_STREAMER_AUTO_BLOCK_6_CORES_MS < CPU_STREAMER_AUTO_BLOCK_FEW_CORES_MS);
const _: () = assert!(CPU_STREAMER_SIX_CORES < CPU_STREAMER_FULL_CORES);
// The longest block must be one the shape plan builds (the plan's longest block).
const _: () = assert!(CPU_STREAMER_AUTO_BLOCK_FEW_CORES_MS <= MAX_DERIVED_BLOCK_MS);

/// `SIGIL_VC_STREAMER_BLOCK=0|off|false|no` keeps the block derived from the device block for the
/// Streamer on the processor ([`cpu_streamer_block_for_cores`]).
pub const CPU_STREAMER_BLOCK_ENV: &str = "SIGIL_VC_STREAMER_BLOCK";

/// Whether the Streamer's automatic block on the processor applies in this process ([`CPU_STREAMER_BLOCK_ENV`] not off).
pub fn cpu_streamer_block_enabled() -> bool {
    let value = std::env::var(CPU_STREAMER_BLOCK_ENV).ok().map(|v| v.trim().to_ascii_lowercase());
    !matches!(value.as_deref(), Some("0" | "off" | "false" | "no"))
}

/// `preset` (from [`engine_preset`]) with the Streamer's processor block applied: for the Streamer on
/// the CPU rung `base_ep` (`cpu`), with no block set (`requested_block_ms` not above 0), a derived block
/// shorter than [`CPU_STREAMER_KEEP_FROM_MS`] becomes the block of this PC's physical cores
/// ([`cpu_streamer_block_for_cores`] of `ort_session::vc_physical_cores`, the count the Streamer's thread policy
/// reads: 120 ms from 7 cores, 160 ms with 6, 200 ms with 5 or fewer). Every other preset, rung, explicit block
/// (the Block size setting or Autotune's pick) and longer derived block is returned as it is. The voice changer's
/// factory and the benchmark's builds both call it, so a spec that leaves the block on automatic measures what
/// the app runs.
pub fn cpu_streamer_block(preset: EnginePreset, base_ep: &str, requested_block_ms: f64) -> EnginePreset {
    cpu_streamer_block_with(
        preset,
        base_ep,
        requested_block_ms,
        cpu_streamer_block_enabled(),
        crate::services::neural_audio::ort_session::vc_physical_cores(),
    )
}

/// The block (ms) the Streamer on the processor takes when none is set, for a device block of
/// `chunk_frames` at `device_rate` ([`cpu_streamer_block`]); `None` when the rule does not apply.
pub fn cpu_streamer_auto_block_ms(profile: RvcPresetProfile, base_ep: &str, chunk_frames: u32, device_rate: u32) -> Option<f64> {
    let derived = engine_preset(profile, 0.0, chunk_frames, device_rate, 0.0, PrecisionMode::Auto);
    let got = cpu_streamer_block(derived, base_ep, 0.0);
    (got.block_ms != derived.block_ms).then_some(got.block_ms)
}

/// [`cpu_streamer_block`] with the kill switch and the PC's physical core count passed in (tests never touch the
/// environment or the machine).
pub fn cpu_streamer_block_with(preset: EnginePreset, base_ep: &str, requested_block_ms: f64, enabled: bool, physical_cores: u32) -> EnginePreset {
    let unset = !requested_block_ms.is_finite() || requested_block_ms <= 0.0;
    if enabled
        && unset
        && preset.profile == RvcPresetProfile::Streamer
        && base_ep.trim().eq_ignore_ascii_case("cpu")
        && preset.block_ms < CPU_STREAMER_KEEP_FROM_MS
    {
        return EnginePreset { block_ms: cpu_streamer_block_for_cores(physical_cores), ..preset };
    }
    preset
}

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
        // P3B-S0: new levers start off. P3B-I0 kept them off (opt-in): see "P3B defaults" above.
        Lever::ContentNorm | Lever::F0Decode => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// P3B-I0's decisions (see "P3B defaults" above): the measured-better paths are the defaults and
    /// the levers that trade fidelity, memory or safety stay switches. A change here is a decision
    /// that needs its interleaved numbers (rule R6), not an incidental edit.
    #[test]
    fn p3b_defaults_are_the_decided_ones() {
        use crate::services::neural_audio::gpu::{cuda_wait::WaitMode, plan::SHRINK_BEFORE_CAPTURE, sched_priority};
        use crate::services::neural_audio::ort_session::{VC_CPU_ARENA_DEFAULT, VC_DENORMAL_ZERO_DEFAULT, VC_DML_SHARED_DEFAULT};
        use crate::services::neural_audio::vc::{rmvpe::FP16_UNDER_AUTO, stream::{SolaSlack, SOLA_SLACK_DEFAULT}};
        // On.
        assert_eq!(WaitMode::DEFAULT, WaitMode::Hybrid);
        let on = [
            SHRINK_BEFORE_CAPTURE,
            VC_CPU_ARENA_DEFAULT,
            super::super::generator::PARALLEL_LOAD_DEFAULT,
            super::super::load::PARALLEL_LOAD_DEFAULT,
        ];
        assert!(on.iter().all(|&b| b), "{on:?}");
        // Off.
        // The shared DirectML queue stays a switch: its content-stage p95 regressed (R3) and the
        // load gain missed T9's bound (`ort_session::VC_DML_SHARED_DEFAULT`).
        let off = [VC_DENORMAL_ZERO_DEFAULT, VC_DML_SHARED_DEFAULT, FP16_UNDER_AUTO, super::super::retrieval::KNN_FP16_ON_CUDA_DEFAULT];
        assert!(off.iter().all(|&b| !b), "{off:?}");
        assert!(sched_priority::DEFAULT.is_none());
        assert_eq!(SOLA_SLACK_DEFAULT, SolaSlack::Legacy);
        for ep in ["cpu", "cuda", "dml"] {
            assert!(!auto_lever(RvcPresetProfile::LowLatency, Lever::ContentNorm, ep), "{ep}");
            assert!(!auto_lever(RvcPresetProfile::LowLatency, Lever::F0Decode, ep), "{ep}");
        }
    }

    /// The Streamer on the processor takes a block it can keep up with when none is set; nothing else
    /// does, and nothing explicit or longer is shortened. (8 physical cores: the 120 ms block; the other core
    /// counts are in `the_cpu_streamer_block_follows_the_physical_cores`.)
    #[test]
    fn the_cpu_streamer_takes_a_long_block_only_when_none_is_set() {
        let derived = |profile, block, chunk| engine_preset(profile, block, chunk, 48_000, 64.0, PrecisionMode::Auto);
        let on = |p: EnginePreset, ep: &str, block: f64| cpu_streamer_block_with(p, ep, block, true, 8).block_ms;
        // The app's default: 512 frames at 48 kHz derive 10.67 ms (planned to 20 ms).
        let s = derived(RvcPresetProfile::Streamer, 0.0, 512);
        assert!(s.block_ms < 20.0);
        assert_eq!(on(s, "cpu", 0.0), CPU_STREAMER_AUTO_BLOCK_MS);
        assert_eq!(on(s, " CPU ", 0.0), CPU_STREAMER_AUTO_BLOCK_MS, "the rung name is trimmed and case-blind");
        assert_eq!(on(s, "cpu", -1.0), CPU_STREAMER_AUTO_BLOCK_MS);
        assert_eq!(on(s, "cpu", f64::NAN), CPU_STREAMER_AUTO_BLOCK_MS);
        // The rest of the preset is untouched.
        let got = cpu_streamer_block_with(s, "cpu", 0.0, true, 8);
        assert_eq!((got.profile, got.extra_context_ms, got.precision), (s.profile, s.extra_context_ms, s.precision));
        // A GPU rung, the other presets and the kill switch keep the derived block.
        for ep in ["cuda", "dml", "", "tensorrt_rtx"] {
            assert_eq!(on(s, ep, 0.0), s.block_ms, "rung {ep:?}");
        }
        for profile in [RvcPresetProfile::LowLatency, RvcPresetProfile::Quality, RvcPresetProfile::StudioTrt] {
            let p = derived(profile, 0.0, 512);
            assert_eq!(on(p, "cpu", 0.0), p.block_ms, "{profile:?}");
        }
        assert_eq!(cpu_streamer_block_with(s, "cpu", 0.0, false, 8).block_ms, s.block_ms, "kill switch");
        // An explicit block is the user's, shorter or longer.
        for explicit in [10.0, 20.0, 40.0, 80.0, 120.0, 200.0] {
            let p = derived(RvcPresetProfile::Streamer, explicit, 512);
            assert_eq!(on(p, "cpu", explicit), explicit, "block {explicit}");
        }
        // A deliberate big device chunk (a derived block of 60 ms or more) is kept, in either direction;
        // only a shorter one is raised.
        for (chunk, ms) in [(9_600u32, 200.0), (5_760, 120.0), (3_840, 80.0), (3_584, 74.67), (2_880, 60.0)] {
            let p = derived(RvcPresetProfile::Streamer, 0.0, chunk);
            assert!((p.block_ms - ms).abs() < 0.01, "{chunk} frames derive {}", p.block_ms);
            assert_eq!(on(p, "cpu", 0.0), p.block_ms, "{chunk} frames ({ms} ms) keep their block");
        }
        for chunk in [160u32, 512, 1_024, 2_048, 2_816] {
            let p = derived(RvcPresetProfile::Streamer, 0.0, chunk);
            assert!(p.block_ms < CPU_STREAMER_KEEP_FROM_MS, "{chunk}");
            assert_eq!(on(p, "cpu", 0.0), CPU_STREAMER_AUTO_BLOCK_MS, "{chunk} frames ({} ms) are raised", p.block_ms);
        }
        // Unknown device rate: no derived block (0), still the processor's.
        let unknown = engine_preset(RvcPresetProfile::Streamer, 0.0, 512, 0, 64.0, PrecisionMode::Auto);
        assert_eq!(unknown.block_ms, 0.0);
        assert_eq!(on(unknown, "cpu", 0.0), CPU_STREAMER_AUTO_BLOCK_MS);
        // The environment switch reads like the other engine switches.
        assert_eq!(CPU_STREAMER_BLOCK_ENV, "SIGIL_VC_STREAMER_BLOCK");
    }

    /// The automatic block follows the physical cores in three tiers: 7 or more cores 120 ms, exactly 6 cores 160 ms, 5 or
    /// fewer (and 0, an unknown count) 200 ms. Only the automatic case moves; every block is one the shape plan builds.
    #[test]
    fn the_cpu_streamer_block_follows_the_physical_cores() {
        assert_eq!(
            (CPU_STREAMER_AUTO_BLOCK_MS, CPU_STREAMER_AUTO_BLOCK_6_CORES_MS, CPU_STREAMER_AUTO_BLOCK_FEW_CORES_MS),
            (120.0, 160.0, 200.0)
        );
        assert_eq!((CPU_STREAMER_FULL_CORES, CPU_STREAMER_SIX_CORES), (7, 6));
        for cores in [7u32, 8, 10, 12, 16, 64] {
            assert_eq!(cpu_streamer_block_for_cores(cores), 120.0, "{cores} cores");
        }
        // Exactly 6 cores is the middle tier (the user's decision of 2026-10-11).
        assert_eq!(cpu_streamer_block_for_cores(6), 160.0, "6 cores");
        // 0 is a probe that could not tell: the smallest PC, like the thread policy (4 threads).
        for cores in [0u32, 1, 2, 3, 4, 5] {
            assert_eq!(cpu_streamer_block_for_cores(cores), 200.0, "{cores} cores");
        }
        // The tier edges: 5 cores 200 ms, 6 cores 160 ms, 7 cores 120 ms.
        assert_eq!((5u32..=7).map(cpu_streamer_block_for_cores).collect::<Vec<_>>(), [200.0, 160.0, 120.0]);
        // Longer blocks for fewer cores, never shorter.
        let table: Vec<f64> = (0u32..=16).map(cpu_streamer_block_for_cores).collect();
        assert!(table.windows(2).all(|w| w[0] >= w[1]), "{table:?}");
        // Every automatic block is one the plan builds (the longest is the plan's longest).
        for block in [120.0, 160.0, 200.0] {
            assert!(super::super::plan::block_frames(block).is_ok(), "{block} ms");
        }

        let derived = |block, chunk| engine_preset(RvcPresetProfile::Streamer, block, chunk, 48_000, 64.0, PrecisionMode::Auto);
        let auto = derived(0.0, 512);
        let on = |p: EnginePreset, ep: &str, block: f64, cores: u32| cpu_streamer_block_with(p, ep, block, true, cores).block_ms;
        assert_eq!((on(auto, "cpu", 0.0, 8), on(auto, "cpu", 0.0, 6), on(auto, "cpu", 0.0, 4)), (120.0, 160.0, 200.0));
        for cores in [2u32, 4, 5, 6, 7, 8, 16] {
            let want = cpu_streamer_block_for_cores(cores);
            assert_eq!(on(auto, "cpu", 0.0, cores), want, "{cores} cores");
            assert_eq!(on(auto, " CPU ", -1.0, cores), want, "{cores} cores, rung and block as written by the setting");
            // An explicit Block size, or Autotune's pick (both arrive as a block above 0), wins at any size of PC.
            for explicit in [20.0, 40.0, 80.0, 120.0, 160.0, 200.0] {
                assert_eq!(on(derived(explicit, 512), "cpu", explicit, cores), explicit, "block {explicit} on {cores} cores");
            }
            // A deliberately big device chunk (60 ms or more) is kept whatever the PC.
            for (chunk, ms) in [(2_880u32, 60.0), (3_840, 80.0), (5_760, 120.0), (9_600, 200.0)] {
                assert_eq!(on(derived(0.0, chunk), "cpu", 0.0, cores), ms, "{chunk} frames on {cores} cores");
            }
            // The GPU rungs and the other presets keep the derived block.
            for ep in ["cuda", "dml", "", "tensorrt_rtx"] {
                assert_eq!(on(auto, ep, 0.0, cores), auto.block_ms, "{ep:?} on {cores} cores");
            }
            for profile in [RvcPresetProfile::LowLatency, RvcPresetProfile::Quality, RvcPresetProfile::StudioTrt] {
                let other = engine_preset(profile, 0.0, 512, 48_000, 64.0, PrecisionMode::Auto);
                assert_eq!(on(other, "cpu", 0.0, cores), other.block_ms, "{profile:?} on {cores} cores");
            }
            assert_eq!(cpu_streamer_block_with(auto, "cpu", 0.0, false, cores).block_ms, auto.block_ms, "kill switch, {cores} cores");
        }
        // The rest of the preset is untouched at any size.
        let got = cpu_streamer_block_with(auto, "cpu", 0.0, true, 4);
        assert_eq!((got.profile, got.extra_context_ms, got.precision), (auto.profile, auto.extra_context_ms, auto.precision));
    }

    /// The inspector's table (`options.ts` `streamerAutoBlockMs`) is this one: both sides test against the same rows.
    #[test]
    fn agrees_with_the_inspectors_table() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../src/core/services/voice_changer/__tests__/streamerBlock.fixture.json");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let fixture: serde_json::Value = serde_json::from_str(&text).expect("streamerBlock.fixture.json parses");
        let rows = fixture["rows"].as_array().expect("rows is a list");
        assert!(rows.len() >= 10, "{} rows", rows.len());
        for row in rows {
            let cores = u32::try_from(row["cores"].as_u64().expect("cores")).expect("cores fit");
            let want = row["blockMs"].as_f64().expect("blockMs");
            assert_eq!(cpu_streamer_block_for_cores(cores), want, "{cores} cores");
        }
    }

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
