//! RMVPE ONNX pitch extractor (default for Low latency / Quality GPU presets).
//!
//! Two export contracts are supported, chosen from the graph's declared input at load:
//! - RVC's `rmvpe.onnx`: log-mel `[1, 128, T]` in, salience `[1, T, 360]` out. The log-mel is
//!   computed here exactly like RVC's `MelSpectrogram` (16 kHz, n_fft = win = 1024, hop 160,
//!   periodic Hann, centred reflect padding, 128 HTK/Slaney mels over 30..8000 Hz,
//!   `log(clamp(x, 1e-5))`), with `T` padded to a multiple of 32 as in `mel2hidden`, but
//!   **end-aligned** (P3 final gate, review finding 4 on T5): the audio's frames end the padded mel
//!   and the frames before them hold the log-mel floor `ln(1e-5)` (silence), where `mel2hidden`
//!   appends zeros after the audio. A zero log-mel frame is a loud flat spectrum, and with the
//!   model's BiGRU it pulled the newest frames of every short incremental call (the F0 cache's
//!   24-frame slices carried 7 such frames after their newest frame, 39 on a 64-frame plan) towards
//!   unvoiced. Measured with the F0 cache's 40 ms ticks on 7 speech clips against whole-clip F0
//!   (`engine::f0_cache::tests::lead_sweep_against_whole_clip_f0`, CPU): V/UV error 9.42 % -> 4.99 %
//!   without lookahead and 5.79 % -> 3.66 % with 20 ms, voiced RMSE (within 600 cents) 41.2 -> 40.6
//!   and 35.6 -> 25.5 cents, at the same padded size (the same cost). The last `frames` output rows
//!   are the audio's.
//! - waveform exports (`[1, N]`, optional `threshold`; w-okada's `rmvpe.onnx`), whose output is
//!   either salience or F0 in Hz. On DirectML every call runs at the first call's length
//!   (`pitch_onnx::ShapePin`): a shorter call is front-padded with silence in whole 10 ms frames and
//!   the padding's frames are dropped, as FCPE does.
//!
//! Salience is decoded like RVC `to_local_average_cents`: argmax ± 4 bins, salience-weighted mean
//! of `cents = 20 * bin + 1997.3794084376191`, 0 when the peak is <= 0.03, `f0 = 10 * 2^(c/1200)`.
//!
//! # Fixed-shape plan, right-sized and built at load (P3-D, P3B-E)
//! A log-mel export runs as a fixed-shape session plan (`gpu::plan`):
//! - on **DirectML** with the `io_binding` lever (on in the `auto` lever table): a static shape lets
//!   DirectML compile the whole graph;
//! - on **CUDA** with the `io_binding` and `cuda_graphs` levers: device-resident I/O and a captured
//!   CUDA graph, plus the plan's priority stream and TF32 per the levers.
//!
//! P3B-E changed how the plan is sized and built:
//! - **Built once, at load.** The graph metadata (input / output names, the mel time `dim_param`)
//!   comes from a protobuf walk of the file (`neural_audio::onnx_proto`), so no plain session is
//!   built first: P3 built the 361 MB model twice per load (a plain session, then the plan at the
//!   first call, a transient second copy on the GPU). Where the levers ask for no plan (the CPU
//!   rung, levers off, waveform exports) the plain session runs as before.
//! - **Sized for the whole window by default** ([`plan_frames_for`]): `32 * ceil((1 + n) / 32)` of
//!   the whole-window call's `n` audio frames ([`RmvpeLoad::max_context_16k`], unknown = 63), the
//!   size P3 grew its plan to at the first call (64 mel frames at Low latency), so the default
//!   contour is P3's. The incremental calls (24 frames at 40 ms blocks) run end-aligned in it.
//!   Measured (stream path, `tests::rmvpe_plan_size_f0`, 27 synthetic clips with ground truth,
//!   DirectML): a 32-frame plan for the same calls doubles RMVPE's gross pitch errors (GPE 4.89 % ->
//!   8.23 %; fine error and V/UV unchanged), and filling the 32 frames with real audio (a 19-frame
//!   lead) still leaves 5.67 %; on real speech the 32-frame contour differs from P3's on 1.6 % of
//!   frames in voicing. RMVPE needs more than one row at the U-Net's bottleneck.
//! - **The VRAM saver** ([`RmvpeLoad::vram_saver`], or `SIGIL_RMVPE_PLAN_FRAMES=32`; off by
//!   default because of the F0 cost above): the plan sized for the incremental calls alone (32 mel
//!   frames for blocks up to 110 ms), where the pruned t32 graph runs (below): about -190 MiB on
//!   DirectML and CUDA, and about 25 % less RMVPE time per call. `SIGIL_RMVPE_PLAN_FRAMES=<n>` (a
//!   multiple of 32) overrides the size for sweeps.
//! - **Never grows where the sizes are known.** A call longer than the plan (an onset after a
//!   silence skip, the first tick, a gap: a whole-window call) is split here into overlapping
//!   sub-calls of at most the plan's length ([`SplitCalls`]): each sub-call after the first
//!   discards a 12-frame lead, and each one before the last keeps at least 4 frames of right
//!   context (the overlap left over is spread evenly), so the result never depends on the F0
//!   cache's priming, and no plan is rebuilt on the tick path. The F0 is smoothed once over the
//!   assembled contour, like a whole call.
//! - **Grows once where they are not** ([`RmvpeLoad::sizes_unknown`]: a legacy ONNX voice, which
//!   calls with its whole rolling context, and the benchmark's shape-less bank requests): P3's
//!   behaviour, the plan is rebuilt at the first call that outgrows it (`32 * ceil(frames / 32)`,
//!   at most 256 frames; that call is the voice's load warm-up), so its contour is P3's, not a
//!   split call's with the BiGRU's context cut at the seams. Not with a window, the VRAM saver or
//!   `SIGIL_RMVPE_PLAN_FRAMES`.
//! - **No second model load.** Before the plan is built, the rung the EP ladder lands on is
//!   predicted from the cached preflights (`ort_session::check_gpu_ep`, no model session): where
//!   the levers run the plain session (the CPU, after the GPU rung failed) only the plain session
//!   is built, not a plan that would be dropped and rebuilt.
//! - **Exact kernel pruning, t32.** At 32 mel frames the U-Net's five AvgPool(2) stages leave the
//!   Intermediate block a `[1, C, 1, 4]` map, so rows 0 and 2 of its 32 3×3 kernels (31 of 512→512
//!   plus the 256→512 entry) only ever multiply the convolution's zero padding. The `t32` graph
//!   (a weightless template, `vc/graphs/rmvpe_t32.graph.onnx`, from
//!   `scripts/vc-templates/export_pitch.py --rmvpe-templates`) has `[1, 3]` kernels with pads
//!   `[0, 1, 0, 1]`; at load its weights are read from the installed `rmvpe.onnx` (the centre row of
//!   those 32 kernels, every other tensor as is; `onnx_proto::external_weights_image`). 49,545,216
//!   fewer parameters (-189 MiB of fp32 weights); the same outputs up to summation order (CPU: max
//!   |salience difference| 3.3e-7 over 20 random and speech mels, `reports/pitch_rmvpe_templates.json`).
//!   It runs only on 32-frame plans (the VRAM saver's, or a default plan whose window fits 31
//!   frames; every sub-call is at most 32 mel frames, checked per call), and only when the model
//!   store's install record vouches that the file is the catalog's `vc/pitch-rmvpe` (sha256
//!   recorded at install, size and mtime unchanged; the file is never hashed at load). Longer
//!   plans, other files, the plain session and `SIGIL_RMVPE_T32=0` run the full model.
//! - **fp16, gated.** `rmvpe_fp16` (the full graph, any plan size) / `rmvpe_t32_fp16` templates run
//!   the U-Net and the `cnn` conv in fp16 (weights rounded at load) with the reshape, the BiGRU, the
//!   final Linear, the Sigmoid and the I/O in fp32. The fp32 census of every U-Net activation stays
//!   below 1,100 (fp16 holds 65,504), so no layer needs an fp32 island. Measured
//!   (`reports/pitch_rmvpe_fp16_gate.json`, the stream-path pitch gate `tests::rmvpe_fp16_gate`):
//!   voiced F0 within 0.12-0.16 cents (p99) and 0.03 % V/UV flips of fp32 on both EPs and both plan
//!   sizes, and GPE not worse except at the default plan on DirectML, where the synthetic GPE rose
//!   4.89 -> 4.95 % (over the +0.05 pt allowance). So the voice's `fp16` precision reaches RMVPE only
//!   where the gate passed ([`fp16_gate_passed`]: CUDA at 32 and 64 frames, DirectML at 32), under
//!   `auto` only once [`FP16_UNDER_AUTO`] is flipped, and `SIGIL_RMVPE_FP16=1` forces it anywhere
//!   (`=0` never). A request that does not run (the gate, or a file the model store does not vouch
//!   for) and a plan that does run fp16 are both load notes of the voice.
//! - **Shared across voices.** Voices whose plans would be identical (same model, graph, precision,
//!   EP request, device, levers, mel length, component prefix and session options) share one plan
//!   (`Arc<Mutex<..>>`, [`shared_components`]); each voice keeps its own F0 cache, range, smoothing,
//!   mel front end and DirectML pin. The shared plan is built and warmed once, at the first voice's
//!   load, outside any tick; the tick path only locks it to run (the mel is computed before and the
//!   output decoded after, outside the lock). While the loader holds a [`WarmupScope`] (around the
//!   test conversions of a background load), the first few calls of a voice that joined an already
//!   warm plan run in the shadow of the live voice: when another voice has called since, the call
//!   waits (at most 60 ms) for that voice's next call to finish, so a background load does not take
//!   the lock just before the live tick needs it. The scope is explicit because a thread cannot tell
//!   a warm-up from a live tick (the voice's F0 helper thread runs both); without one every call
//!   just takes the lock, and a live call waits for one warm-up run (3-6 ms) at worst.
//!   `SIGIL_RMVPE_SHARE=0` gives every voice its own plan.
//!
//! A plan that cannot be built (or fails later) hands the extractor to the plain session with a
//! warning (the legacy path; the full model).

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::{Duration, Instant};

use ort::session::Session;
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::gpu::{ExternalWeights, GpuLevers, PlanSpec, SessionPlan, TensorSpec};
use crate::services::neural_audio::onnx_proto::{self, Dim, GraphInfo, DT_FLOAT};
use crate::services::neural_audio::ort_session::{build_session, check_gpu_ep, device_for, ep_ladder, normalize_ep, ModelSource, VcSessionOptions};
use crate::services::neural_audio::telemetry::{self, LoadStage, SyncKind};

use super::engine::options::PrecisionMode;
use super::f0_post::smooth_f0_stream;
use super::pitch_onnx::{flat_f0, local_average_f0, waveform_shape, MainInput, PitchInputs, ShapePin, SALIENCE_BINS};
use super::traits::PitchExtractor;

const SR: u32 = 16_000;
const HOP: usize = 160;
const N_BINS: usize = SALIENCE_BINS;
const N_FFT: usize = 1024;
const N_MELS: usize = 128;
const MEL_FMIN: f64 = 30.0;
const MEL_FMAX: f64 = 8000.0;
const MEL_CLAMP: f32 = 1e-5;
/// RVC `infer_from_audio(audio, thred=0.03)`.
const SALIENCE_THRESHOLD: f32 = 0.03;
/// Value w-okada feeds its waveform-input RMVPE exports' `threshold` input.
const WAVEFORM_EXPORT_THRESHOLD: f32 = 0.3;
#[cfg(test)]
const CENTS_OFFSET: f32 = super::pitch_onnx::SALIENCE_CENTS_OFFSET;

/// Mel frames a padded RMVPE input is a multiple of (`mel2hidden`), and the default plan.
pub const PLAN_ALIGN: usize = 32;
/// Longest plan [`RmvpeLoad`] or `SIGIL_RMVPE_PLAN_FRAMES` may ask for.
const MAX_PLAN_FRAMES: usize = 256;
/// Frames before the kept part of a sub-call after the first ([`SplitCalls`]): the F0 cache's lead.
const SPLIT_LEAD_FRAMES: usize = 12;
/// Least right context of the kept part of a sub-call before the last (past the STFT's ±512-sample
/// reflect padding at a sub-call's end, 3.2 frames).
const SPLIT_MIN_RIGHT_FRAMES: usize = 4;
/// Calls of a voice that joined a warm shared plan that count as its load's warm-ups (while a
/// [`WarmupScope`] is held).
const WARMUP_CALLS: u32 = 8;
/// Longest a warm-up waits for the live voice's next call to finish.
const SHADOW_WAIT: Duration = Duration::from_millis(60);
/// How much later than the install record a model file's mtime may be (the record is written after
/// the file).
const STORE_MTIME_SLACK_MS: u64 = 120_000;

/// [`WarmupScope`]s held now (process-wide).
static WARMING: AtomicUsize = AtomicUsize::new(0);

/// Marks a voice load's warm-up conversions (P3B-E fix). A background hot-switch load holds one for
/// its whole run (`factory::BackgroundLoad`, wired by the P3B gate); the loader holds one around the test
/// conversions of a freshly loaded voice (`warm_once`, the harness warm-up) on whatever thread
/// they run (the F0 overlap helper runs them for a GPU voice, and the same helper thread then runs
/// the live voice's F0, so a thread cannot tell a warm-up from a live tick). While a scope is
/// held, the first calls of a voice that joined a live shared plan wait for the live voice's next
/// call to finish and then run right behind it, so a background load never takes the plan's lock
/// just before the live tick needs it. Without a scope every call just takes the lock (a live call
/// then waits for one run, 3-6 ms at most).
#[must_use = "the warm-up rule applies while the scope is held"]
pub struct WarmupScope(());

/// Open a [`WarmupScope`] (dropping it closes it).
pub fn warmup_scope() -> WarmupScope {
    WARMING.fetch_add(1, Ordering::AcqRel);
    WarmupScope(())
}

/// A [`WarmupScope`] is held somewhere (tests of the holders).
#[cfg(test)]
pub(crate) fn warming_now() -> bool {
    WARMING.load(Ordering::Acquire) > 0
}

impl Drop for WarmupScope {
    fn drop(&mut self) {
        WARMING.fetch_sub(1, Ordering::AcqRel);
    }
}

/// sha256 of the catalog's `vc/pitch-rmvpe` (`model.onnx`), the file the templates were cut from.
pub const RMVPE_SHA256: &str = "5370e71ac80af8b4b7c793d27efd51fd8bf962de3a7ede0766dac0befa3660fd";
/// The model-store entry of RVC's `rmvpe.onnx`.
const RMVPE_ENTRY: &str = "vc/pitch-rmvpe";
/// External-data file name the templates record (filled in memory at load).
const TEMPLATE_WEIGHTS: &str = "rmvpe.weights";
static T32_GRAPH: &[u8] = include_bytes!("graphs/rmvpe_t32.graph.onnx");
static T32_FP16_GRAPH: &[u8] = include_bytes!("graphs/rmvpe_t32_fp16.graph.onnx");
static FP16_GRAPH: &[u8] = include_bytes!("graphs/rmvpe_fp16.graph.onnx");

/// Whether `auto` precision runs fp16 RMVPE on the GPU. Off: the stream-path pitch gate
/// (`tests::rmvpe_fp16_gate`, DirectML and CUDA) failed at the default plan on DirectML (GPE +0.057
/// pt over its +0.05 pt allowance; module docs); flip it only from a passing gate on every GPU
/// rung ([`fp16_gate_passed`] is the table of what passed). An explicit `fp16` precision runs it
/// only where the gate passed; `SIGIL_RMVPE_FP16=1` runs it regardless.
pub const FP16_UNDER_AUTO: bool = false;

/// `SIGIL_RMVPE_PLAN_FRAMES=<n>` (a multiple of 32, at most 256): the plan's mel frames, whatever
/// the expected calls (P3B-F's 32 / 64 sweep).
pub const PLAN_FRAMES_ENV: &str = "SIGIL_RMVPE_PLAN_FRAMES";
/// `SIGIL_RMVPE_T32=0`: 32-frame plans run the full model (the t32 kill switch).
pub const T32_ENV: &str = "SIGIL_RMVPE_T32";
/// `SIGIL_RMVPE_FP16=1|0`: fp16 RMVPE on every GPU plan / never, whatever the precision.
pub const FP16_ENV: &str = "SIGIL_RMVPE_FP16";
/// `SIGIL_RMVPE_SHARE=0`: every voice builds its own plan.
pub const SHARE_ENV: &str = "SIGIL_RMVPE_SHARE";

/// How RMVPE is loaded (P3B-E; [`RmvpePitchExtractor::load_for`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RmvpeLoad {
    /// Audio frames of the F0 cache's incremental calls (`PitchBankRequest::expected_call_frames`);
    /// `None` = unknown (up to 31: blocks up to 110 ms).
    pub expected_call_frames: Option<usize>,
    /// The longest whole-window call (16 kHz samples, `PitchBankRequest::max_context_16k`): the
    /// default plan holds it ([`plan_frames_for`]); `None` = the sizes are unknown: the
    /// Low-latency window (up to 63 frames, a 64-frame plan) that grows once to a longer call.
    pub max_context_16k: Option<usize>,
    /// The voice's precision request: fp16 RMVPE only under `fp16` (or `auto` once
    /// [`FP16_UNDER_AUTO`] is on).
    pub precision: PrecisionMode,
    /// Share the plan with other voices of the same configuration (default on).
    pub shared: bool,
    /// The VRAM saver (default off): the plan sized for the incremental calls (32 frames for
    /// blocks up to 110 ms), where the pruned t32 graph runs (-189 MiB). It costs F0 accuracy
    /// (module docs), so it is a switch, never a default.
    pub vram_saver: bool,
}

impl Default for RmvpeLoad {
    fn default() -> Self {
        RmvpeLoad {
            expected_call_frames: None,
            max_context_16k: None,
            precision: PrecisionMode::Auto,
            shared: true,
            vram_saver: false,
        }
    }
}

impl RmvpeLoad {
    /// The pitch bank's request (`f0_fusion::PitchBankRequest`'s P3B fields).
    pub fn from_bank(req: &super::f0_fusion::PitchBankRequest<'_>) -> RmvpeLoad {
        RmvpeLoad {
            expected_call_frames: req.expected_call_frames,
            max_context_16k: req.max_context_16k,
            precision: req.precision,
            shared: true,
            vram_saver: false,
        }
    }
}

/// The audio frames of the whole window when the pitch bank does not say (the Low-latency window
/// is 44: P3's plan, sized by that first call, was 64 mel frames).
const DEFAULT_WINDOW_FRAMES: usize = 63;

/// The mel frames of the plan for `load` (module docs), `32 * ceil((1 + n) / 32)` of:
/// - by default, the whole window (`max_context_16k`, or [`DEFAULT_WINDOW_FRAMES`]) and the
///   incremental calls: the size P3 grew its plan to at the first call, so every call fits and
///   the contour is P3's (64 frames at Low latency);
/// - with [`RmvpeLoad::vram_saver`], the incremental calls alone (32 for blocks up to 110 ms);
///
/// unless `SIGIL_RMVPE_PLAN_FRAMES` (a multiple of 32, at most 256) says otherwise.
pub fn plan_frames_for(load: &RmvpeLoad) -> usize {
    plan_frames_with(load, std::env::var(PLAN_FRAMES_ENV).ok().as_deref())
}

/// `SIGIL_RMVPE_PLAN_FRAMES` as a plan size (`None`: unset or not a multiple of 32 up to 256).
fn plan_frames_override(env: Option<&str>) -> Option<usize> {
    env.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n >= PLAN_ALIGN && *n <= MAX_PLAN_FRAMES && n % PLAN_ALIGN == 0)
}

impl RmvpeLoad {
    /// Whether the call sizes are unknown (a legacy ONNX voice, which calls with its whole rolling
    /// context; the shape-less bank requests of the benchmark): no window, no saver and no size
    /// override. Such a plan grows once to the first call that outgrows it, as P3's did
    /// ([`RmvpePitchExtractor::grow_for`]); a size the loader states is never grown.
    fn sizes_unknown(&self, env: Option<&str>) -> bool {
        self.max_context_16k.is_none() && !self.vram_saver && plan_frames_override(env).is_none()
    }
}

fn plan_frames_with(load: &RmvpeLoad, env: Option<&str>) -> usize {
    if let Some(n) = plan_frames_override(env) {
        return n;
    }
    let incremental = load.expected_call_frames.unwrap_or(PLAN_ALIGN - 1).max(1);
    let audio = if load.vram_saver {
        incremental
    } else {
        let window = load
            .max_context_16k
            .map_or(DEFAULT_WINDOW_FRAMES, |n| n / HOP);
        window.max(incremental)
    };
    (PLAN_ALIGN * (1 + audio).div_ceil(PLAN_ALIGN)).clamp(PLAN_ALIGN, MAX_PLAN_FRAMES)
}

fn env_flag(name: &str) -> Option<bool> {
    match std::env::var(name)
        .ok()?
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

/// Which RMVPE graph a plan runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RmvpeGraph {
    /// The installed `rmvpe.onnx` (any mel length).
    Full,
    /// Exact kernel-row pruning for 32-frame mels (module docs).
    T32,
}

impl RmvpeGraph {
    pub fn as_str(self) -> &'static str {
        match self {
            RmvpeGraph::Full => "full",
            RmvpeGraph::T32 => "t32",
        }
    }
}

/// The graphs to try for a plan of `frames` mel frames, best first: `t32` (fp16 first when asked)
/// when allowed, else the fp16 full graph when asked, then the full model in fp32.
fn graph_attempts(frames: usize, fp16: bool, t32_allowed: bool) -> Vec<(RmvpeGraph, bool)> {
    let mut out = Vec::with_capacity(3);
    let t32 = t32_allowed && frames == PLAN_ALIGN;
    if t32 {
        if fp16 {
            out.push((RmvpeGraph::T32, true));
        }
        out.push((RmvpeGraph::T32, false));
    } else if fp16 {
        out.push((RmvpeGraph::Full, true));
    }
    out.push((RmvpeGraph::Full, false));
    out
}

/// Where the stream-path fp16 gate passed (`reports/pitch_rmvpe_fp16_gate.json`, 34 clips): CUDA
/// at 32 and 64 mel frames, DirectML at 32 (its 64-frame GPE rose 4.89 -> 4.95 %, over the +0.05 pt
/// allowance). Other sizes were not measured. fp16 follows the voice's precision only there.
fn fp16_gate_passed(rung: &str, frames: usize) -> bool {
    match rung {
        "cuda" => frames <= 2 * PLAN_ALIGN,
        "dml" => frames <= PLAN_ALIGN,
        _ => false,
    }
}

/// Whether the request asks for fp16 RMVPE at all: `SIGIL_RMVPE_FP16` (`env`) wins, else the voice's
/// precision (`fp16`, or `auto` once [`FP16_UNDER_AUTO`] is on).
fn fp16_asked(precision: PrecisionMode, env: Option<bool>) -> bool {
    env.unwrap_or(precision == PrecisionMode::Fp16 || (precision == PrecisionMode::Auto && FP16_UNDER_AUTO))
}

/// What a plan of `frames` mel frames on `rung` does about fp16 for `precision` (`env`:
/// [`FP16_ENV`]; `trusted`: the file is the model store's verified RMVPE, the condition for every
/// graph template). `refusal` is the load note of an fp16 request that does not run.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Fp16Verdict {
    run: bool,
    refusal: Option<String>,
}

fn fp16_verdict(precision: PrecisionMode, rung: &str, frames: usize, env: Option<bool>, trusted: bool) -> Fp16Verdict {
    let off = |refusal: Option<String>| Fp16Verdict { run: false, refusal };
    if rung == "cpu" || !fp16_asked(precision, env) {
        return off(None);
    }
    if !trusted {
        return off(Some(
            "RMVPE: fp16 was asked for, but the file is not the model store's verified vc/pitch-rmvpe (its graph templates need it); the fp32 model runs".into(),
        ));
    }
    if env.is_none() && !fp16_gate_passed(rung, frames) {
        return off(Some(format!(
            "RMVPE: fp16 precision does not reach RMVPE in a {frames}-frame plan on {rung} (its accuracy gate did not pass there); the fp32 model runs (SIGIL_RMVPE_FP16=1 forces fp16)"
        )));
    }
    Fp16Verdict { run: true, refusal: None }
}

/// Whether fp16 RMVPE runs for `precision` on `rung` (`env`: [`FP16_ENV`]), the file being trusted.
#[cfg(test)]
fn fp16_wanted(precision: PrecisionMode, rung: &str, frames: usize, env: Option<bool>) -> bool {
    fp16_verdict(precision, rung, frames, env, true).run
}

// ---------------------------------------------------------------------------------------------
// Shared plans
// ---------------------------------------------------------------------------------------------

/// What makes two plans interchangeable.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CoreKey {
    /// `sha256:<hex>` for a file the model store vouches for, else `<path>|<size>|<mtime>`.
    model: String,
    graph: RmvpeGraph,
    fp16: bool,
    ep: &'static str,
    device: i32,
    levers: GpuLevers,
    frames: usize,
    prefix: Option<&'static str>,
    /// The session options as built (threads, priority, caps).
    opts: String,
}

/// The plan and its I/O names (locked per call).
struct Core {
    plan: SessionPlan,
    input: String,
    output: String,
}

/// A plan, possibly shared by several voices.
struct SharedCore {
    core: Mutex<Core>,
    /// Plan runs completed by any voice (a joining voice's warm-ups wait for it to move).
    done: AtomicU64,
    id: String,
    frames: usize,
    graph: RmvpeGraph,
    fp16: bool,
    outcome: EpOutcome,
    notes: Vec<String>,
}

/// One registry row: the plan while any voice holds it, the gate its builders queue on, and what
/// [`shared_components`] reports (copied, so listing never touches a plan).
struct RegistryEntry {
    key: CoreKey,
    core: Weak<SharedCore>,
    gate: Arc<Mutex<()>>,
    info: Option<SharedPitchComponent>,
}

static REGISTRY: Mutex<Vec<RegistryEntry>> = Mutex::new(Vec::new());
static NEXT_CORE: AtomicU64 = AtomicU64::new(1);

fn registry() -> std::sync::MutexGuard<'static, Vec<RegistryEntry>> {
    REGISTRY.lock().unwrap_or_else(PoisonError::into_inner)
}

/// An RMVPE plan held by one or more voices (P3B-E; VoiceSet's VRAM accounting counts each once).
#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)] // Read by VoiceSet's VRAM accounting (P3B-H) and tests.
pub struct SharedPitchComponent {
    /// Stable for the plan's lifetime (`rmvpe:<graph>[-fp16]:<frames>f:<ep>#<n>`).
    pub id: String,
    /// Diagnostics component (`vc.pitch`, or `<prefix>.vc.pitch`).
    pub component: String,
    /// Voices holding it.
    pub users: usize,
    /// Process VRAM the plan added when it was built (MiB; `None` on the CPU or when unknown).
    pub vram_mb: Option<f64>,
    pub mel_frames: usize,
    /// `full` / `t32`.
    pub graph: &'static str,
    pub fp16: bool,
    /// The EP the plan runs on.
    pub ep: String,
}

/// The RMVPE plans that exist now, with how many voices hold each (P3B-H's VoiceSet counts a plan
/// once against `vram_budget_mb` and frees it with its last voice). Never waits for a plan build.
#[allow(dead_code)] // For VoiceSet's VRAM accounting (P3B-H).
pub fn shared_components() -> Vec<SharedPitchComponent> {
    registry()
        .iter()
        .filter_map(|e| {
            let users = e.core.strong_count();
            let mut info = e.info.clone()?;
            (users > 0).then(|| {
                info.users = users;
                info
            })
        })
        .collect()
}

/// The plan for `key`: the live one when another voice holds it (`joined = true`), else `build`'s,
/// built once even when two voices load at the same time (they queue on the key's gate; the
/// registry lock is never held while building). `share = false` builds a private plan.
fn acquire(key: CoreKey, share: bool, build: impl FnOnce() -> Result<SharedCore, String>) -> Result<(Arc<SharedCore>, bool), String> {
    if !share {
        return Ok((Arc::new(build()?), false));
    }
    let gate = {
        let mut reg = registry();
        reg.retain(|e| e.core.strong_count() > 0 || Arc::strong_count(&e.gate) > 1);
        match reg.iter().find(|e| e.key == key) {
            Some(e) => match e.core.upgrade() {
                Some(core) => return Ok((core, true)),
                None => e.gate.clone(),
            },
            None => {
                let gate = Arc::new(Mutex::new(()));
                reg.push(RegistryEntry {
                    key: key.clone(),
                    core: Weak::new(),
                    gate: gate.clone(),
                    info: None,
                });
                gate
            }
        }
    };
    let _building = gate.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(core) = registry()
        .iter()
        .find(|e| e.key == key)
        .and_then(|e| e.core.upgrade())
    {
        return Ok((core, true));
    }
    let core = Arc::new(build()?);
    let info = SharedPitchComponent {
        id: core.id.clone(),
        component: component_name(key.prefix),
        users: 1,
        vram_mb: core
            .core
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .plan
            .vram_mb(),
        mel_frames: core.frames,
        graph: core.graph.as_str(),
        fp16: core.fp16,
        ep: core.outcome.active.clone(),
    };
    let mut reg = registry();
    match reg.iter_mut().find(|e| e.key == key) {
        Some(e) => {
            e.core = Arc::downgrade(&core);
            e.info = Some(info);
        }
        None => reg.push(RegistryEntry {
            key,
            core: Arc::downgrade(&core),
            gate: gate.clone(),
            info: Some(info),
        }),
    }
    Ok((core, false))
}

fn component_name(prefix: Option<&str>) -> String {
    match prefix {
        Some(p) if !p.is_empty() => format!("{p}.vc.pitch"),
        _ => "vc.pitch".to_string(),
    }
}

/// Run the plan once on `mel` (`N_MELS × frames`, row-major) and copy output rows `rows` (each
/// `N_BINS` values) into `out_rows`, holding the plan's lock only for the copy-in, the run and the
/// copy-out. Allocation-free once `out_rows` has its capacity.
fn run_core(shared: &SharedCore, mel: &[f32], rows: Range<usize>, out_rows: &mut Vec<f32>) -> Result<(), String> {
    let mut guard = shared
        .core
        .lock()
        .map_err(|_| "the RMVPE plan's lock is poisoned".to_string())?;
    let Core { plan, input, output } = &mut *guard;
    let dst = plan.input_f32(input.as_str())?;
    if dst.len() != mel.len() {
        return Err(format!("RMVPE plan input holds {} values, the mel {}", dst.len(), mel.len()));
    }
    dst.copy_from_slice(mel);
    plan.run()?;
    let o = plan.output(output.as_str())?;
    if o.len() < rows.end * N_BINS {
        return Err(format!("RMVPE plan output holds {} values, {} rows asked", o.len(), rows.end));
    }
    out_rows.clear();
    out_rows.extend_from_slice(&o[rows.start * N_BINS..rows.end * N_BINS]);
    drop(guard);
    shared.done.fetch_add(1, Ordering::AcqRel);
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Splitting long calls
// ---------------------------------------------------------------------------------------------

/// One sub-call of a long call: mel frames `[start, start + w)` (the last one ends at the call's
/// end), whose output frames `[keep_from, keep_to)` are kept (absolute frames of the call).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SubCall {
    start: usize,
    keep_from: usize,
    keep_to: usize,
    last: bool,
}

/// The sub-calls of a call of `total` mel frames on a `w`-frame plan (module docs): one call when
/// it fits; else the fewest sub-calls of `w` frames whose overlaps give each kept part a 12-frame
/// lead and at least 4 frames of right context, starts spread evenly from 0 to `total - w`.
/// Allocation-free.
struct SplitCalls {
    total: usize,
    w: usize,
    lead: usize,
    calls: usize,
    next: usize,
}

impl SplitCalls {
    fn new(total: usize, w: usize) -> SplitCalls {
        let w = w.max(1);
        // Plans are multiples of 32 frames; a plan too short for the 12 + 4 frames of overlap
        // (none is built) would keep a third of itself on each side.
        let (lead, right) = if w > SPLIT_LEAD_FRAMES + SPLIT_MIN_RIGHT_FRAMES {
            (SPLIT_LEAD_FRAMES, SPLIT_MIN_RIGHT_FRAMES)
        } else {
            (w / 3, w / 3)
        };
        let calls = if total <= w {
            1
        } else {
            let step = (w - lead - right).max(1);
            1 + (total - w).div_ceil(step)
        };
        SplitCalls {
            total,
            w,
            lead,
            calls,
            next: 0,
        }
    }

    fn start(&self, j: usize) -> usize {
        if self.calls <= 1 {
            return 0;
        }
        let span = self.total - self.w;
        (j * span + (self.calls - 1) / 2) / (self.calls - 1)
    }
}

impl Iterator for SplitCalls {
    type Item = SubCall;

    fn next(&mut self) -> Option<SubCall> {
        let j = self.next;
        if j >= self.calls {
            return None;
        }
        self.next += 1;
        if self.calls == 1 {
            return Some(SubCall {
                start: 0,
                keep_from: 0,
                keep_to: self.total,
                last: true,
            });
        }
        let start = self.start(j);
        let keep_from = if j == 0 { 0 } else { start + self.lead };
        let last = j + 1 == self.calls;
        let keep_to = if last { self.total } else { self.start(j + 1) + self.lead };
        Some(SubCall {
            start,
            keep_from,
            keep_to,
            last,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// The extractor
// ---------------------------------------------------------------------------------------------

/// Call counts (diagnostics and tests).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(dead_code)] // Read by tests and diagnostics.
pub struct RmvpeStats {
    /// `f0` / `f0_into` calls with audio.
    pub calls: u64,
    /// Plan runs (a split call runs the plan more than once).
    pub plan_runs: u64,
    /// Calls split into sub-calls.
    pub split_calls: u64,
    /// Plans built for a call that outgrew the previous one (call sizes unknown at load).
    pub grows: u64,
    /// Plain-session runs.
    pub plain_runs: u64,
    /// Warm-up calls that waited for the live voice's next call.
    pub shadow_waits: u64,
}

/// The plain (dynamic-shape) session and its input contract.
struct PlainSession {
    session: Session,
    inputs: PitchInputs,
}

/// This voice's hold on a (shared) plan.
struct Planned {
    shared: Arc<SharedCore>,
    /// The plan was already built by another voice when this one loaded.
    joined: bool,
    /// Calls so far (the first [`WARMUP_CALLS`] of a joined voice may be warm-ups).
    calls: u32,
    /// `done` after this voice's last plan run.
    seen: u64,
}

impl Planned {
    /// A warm-up of a voice that joined a live plan: one of its first calls while the loader holds
    /// a [`WarmupScope`] (on any thread: see there).
    fn in_warmup(&self) -> bool {
        self.joined && self.calls < WARMUP_CALLS && WARMING.load(Ordering::Acquire) > 0
    }
}

/// What a test may force (production loads use `Hooks::default()`).
#[derive(Clone, Copy, Default)]
struct Hooks {
    /// Build the plan on any rung (the CPU in tests).
    force_plan: bool,
    /// Treat the file as the catalog's verified RMVPE (dev-rig tests that hashed it themselves).
    trust_source: bool,
    /// Graph templates instead of the compiled-in ones: `(t32, t32_fp16, fp16)`.
    templates: Option<(&'static [u8], &'static [u8], &'static [u8])>,
    /// `SIGIL_RMVPE_FP16` as a test sets it (the process environment is shared by parallel tests).
    fp16_env: Option<bool>,
}

/// What a plan is built from; kept when the plan may grow later (an extractor whose call sizes
/// were unknown at load).
struct PlanSource {
    info: GraphInfo,
    contract: MelContract,
    /// The requested EP, and the rung its ladder lands on (from the cached preflights).
    base: &'static str,
    rung: &'static str,
    levers: GpuLevers,
    load: RmvpeLoad,
    hooks: Hooks,
    /// The file was vouched for by the model store at load (never re-checked on a tick).
    trusted: bool,
}

/// The rung the EP ladder of `base` lands on right now: the first whose preflight and canary pass
/// (`usable(rung, device)`; cached, no model session is built), the CPU at the end.
fn predicted_rung_with(base: &'static str, device_of: impl Fn(&'static str) -> i32, usable: impl Fn(&'static str, i32) -> bool) -> &'static str {
    ep_ladder(base)
        .iter()
        .copied()
        .find(|&rung| rung == "cpu" || usable(rung, device_of(rung)))
        .unwrap_or("cpu")
}

fn predicted_rung(base: &'static str, opts: &VcSessionOptions) -> &'static str {
    predicted_rung_with(base, |rung| device_for(rung, base, opts), |rung, device| check_gpu_ep(rung, device).is_ok())
}

pub struct RmvpePitchExtractor {
    path: PathBuf,
    /// EP of the plain session (the load's rung; after a plan, the plan's).
    plain_ep: String,
    opts: VcSessionOptions,
    /// The plain session (`None` while a plan runs every call).
    plain: Option<PlainSession>,
    planned: Option<Planned>,
    /// Set when the call sizes were unknown at load: how to build a bigger plan for a longer call.
    grow: Option<PlanSource>,
    mel: Option<LogMel>,
    /// Peak salience per frame of the last call (P3-D: the confidence F0 fusion weighs; 1 / 0 for
    /// exports that output F0 directly).
    confidence: Vec<f32>,
    /// Salience rows of the last call (`frames × 360`, aligned with its contour; empty for exports
    /// that output F0).
    salience: Vec<f32>,
    /// DirectML: every plain-session call runs at the first call's mel length (`pitch_onnx::ShapePin`).
    pin: ShapePin,
    /// Mel scratch (`N_MELS × plan frames` on the plan path).
    mel_buf: Vec<f32>,
    /// Front-padded waveform of a pinned (DirectML) waveform-export call.
    wave_buf: Vec<f32>,
    /// Output rows of one plan run, copied out under the lock.
    rows_buf: Vec<f32>,
    /// EP outcome of the load (`vc.pitch`).
    outcome: EpOutcome,
    notes: Vec<String>,
    stats: RmvpeStats,
    /// Tests: skip the streaming smoother (the allocation-free check of the extractor's own path).
    #[cfg(test)]
    skip_smoothing: bool,
}

impl RmvpePitchExtractor {
    /// Load on the EP ladder of `execution_provider_hint` with no GPU levers (the plain session on
    /// every rung, see the module docs).
    #[allow(dead_code)] // The P0 constructor; the pitch bank passes the voice's levers.
    pub fn load(model_path: &str, execution_provider_hint: &str, opts: &VcSessionOptions) -> Result<Self, String> {
        Self::load_with_levers(model_path, execution_provider_hint, opts, GpuLevers::default())
    }

    /// [`load`](Self::load) with the voice's GPU levers: on CUDA, `io_binding` plus `cuda_graphs`
    /// run a log-mel export as a CUDA-graph plan; on DirectML, `io_binding` runs it as a fixed-shape
    /// plan (module docs). The default [`RmvpeLoad`]: a 64-frame plan that grows to a longer call, fp32, shared.
    pub fn load_with_levers(model_path: &str, execution_provider_hint: &str, opts: &VcSessionOptions, levers: GpuLevers) -> Result<Self, String> {
        Self::load_for(model_path, execution_provider_hint, opts, levers, &RmvpeLoad::default())
    }

    /// [`load_with_levers`](Self::load_with_levers) with the pitch bank's request (P3B-E: the plan
    /// sized for the F0 cache's calls, the precision, sharing).
    pub fn load_for(
        model_path: &str,
        execution_provider_hint: &str,
        opts: &VcSessionOptions,
        levers: GpuLevers,
        load: &RmvpeLoad,
    ) -> Result<Self, String> {
        Self::load_inner(model_path, execution_provider_hint, opts, levers, load, Hooks::default())
    }

    fn load_inner(
        model_path: &str,
        ep_hint: &str,
        opts: &VcSessionOptions,
        levers: GpuLevers,
        load: &RmvpeLoad,
        hooks: Hooks,
    ) -> Result<Self, String> {
        let path_str = model_path.trim();
        if path_str.is_empty() {
            return Err("RMVPE model path is empty".into());
        }
        let path = Path::new(path_str);
        if let Some(problem) = super::rvc::non_onnx_file_problem(path) {
            return Err(format!("RMVPE: {problem}"));
        }
        let base = normalize_ep(ep_hint);
        let mut this = RmvpePitchExtractor {
            path: path.to_path_buf(),
            plain_ep: base.to_string(),
            opts: opts.clone(),
            plain: None,
            planned: None,
            grow: None,
            mel: None,
            confidence: Vec::new(),
            salience: Vec::new(),
            pin: ShapePin::default(),
            mel_buf: Vec::new(),
            wave_buf: Vec::new(),
            rows_buf: Vec::new(),
            outcome: EpOutcome::granted("cpu"),
            notes: Vec::new(),
            stats: RmvpeStats::default(),
            #[cfg(test)]
            skip_smoothing: false,
        };
        // The graph's contract from the file itself (no session): a log-mel export can run its plan
        // without a plain session ever being built.
        let info = match onnx_proto::read_graph_info(path) {
            Ok(info) => Some(info),
            Err(e) => {
                tracing::debug!(target: "sigil::vc", "RMVPE: {e}; ONNX Runtime reads the graph instead");
                None
            }
        };
        if let Some(why) = info.as_ref().and_then(|i| foreign_graph(i, path)) {
            return Err(format!("RMVPE: {why}"));
        }
        let contract = info.as_ref().and_then(MelContract::of);
        if let (Some(info), Some(contract)) = (info, contract) {
            if hooks.force_plan || wants_plan(base, levers) {
                // The rung the ladder lands on, from the cached preflights (no model session): where
                // the levers run the plain session (the CPU, after a GPU rung failed) no plan is
                // built, so the model is not loaded twice.
                let rung = if hooks.force_plan { base } else { predicted_rung(base, &this.opts) };
                if hooks.force_plan || wants_plan(rung, levers) {
                    let src = PlanSource {
                        info,
                        contract,
                        base,
                        rung,
                        levers,
                        load: *load,
                        hooks,
                        trusted: this.trusted(hooks),
                    };
                    this.try_plan(&src, plan_frames_for(load), true);
                    if this.planned.is_some() {
                        if load.sizes_unknown(std::env::var(PLAN_FRAMES_ENV).ok().as_deref()) {
                            this.grow = Some(src);
                        }
                        return Ok(this);
                    }
                } else {
                    this.note(format!("RMVPE: the plan would land on {rung}, where the plain session runs"));
                }
            }
        }
        this.ensure_plain()?;
        Ok(this)
    }

    /// Add a load note once.
    fn note(&mut self, note: String) {
        if !self.notes.contains(&note) {
            self.notes.push(note);
        }
    }

    /// Whether the model store vouches for the file (the graph templates need it; module docs).
    fn trusted(&self, hooks: Hooks) -> bool {
        let verdict = if hooks.trust_source { Ok(()) } else { verified_by_store(&self.path) };
        if let Err(why) = &verdict {
            tracing::info!(target: "sigil::vc", "RMVPE: the pruned / fp16 graph templates are not used ({why}); the full model runs");
        }
        verdict.is_ok()
    }

    /// Build or join a `frames`-frame plan (module docs), trying the best graph first; on success
    /// `planned` is set, otherwise the reasons are in `notes` and `planned` is as it was.
    /// `warmup`: the plan comes with a voice load (a joined plan's first calls are its warm-ups;
    /// a plan grown on a tick has none).
    fn try_plan(&mut self, src: &PlanSource, frames: usize, warmup: bool) {
        let (base, levers, load) = (src.base, src.levers, &src.load);
        if src.contract.fixed_frames.is_some_and(|t| t != frames) {
            self.note(format!(
                "RMVPE: the model's mel length is fixed at {:?} frames; the plain session runs",
                src.contract.fixed_frames
            ));
            return;
        }
        let verdict = fp16_verdict(
            load.precision,
            src.rung,
            frames,
            src.hooks.fp16_env.or_else(|| env_flag(FP16_ENV)),
            src.trusted,
        );
        if let Some(why) = verdict.refusal {
            tracing::info!(target: "sigil::vc", "{why}");
            self.note(why);
        }
        let t32_allowed = src.trusted && env_flag(T32_ENV) != Some(false);
        let share = load.shared && env_flag(SHARE_ENV) != Some(false);
        let model_id = if src.trusted {
            format!("sha256:{RMVPE_SHA256}")
        } else {
            file_identity(&self.path)
        };
        let templates = src
            .hooks
            .templates
            .unwrap_or((T32_GRAPH, T32_FP16_GRAPH, FP16_GRAPH));
        for (graph, half) in graph_attempts(frames, verdict.run, t32_allowed) {
            let key = CoreKey {
                model: model_id.clone(),
                graph,
                fp16: half,
                ep: base,
                device: self.opts.cuda_device_index,
                levers,
                frames,
                prefix: self.opts.component_prefix,
                opts: format!("{:?}", self.opts),
            };
            let template = match (graph, half) {
                (RmvpeGraph::T32, false) => Some(templates.0),
                (RmvpeGraph::T32, true) => Some(templates.1),
                (RmvpeGraph::Full, true) => Some(templates.2),
                (RmvpeGraph::Full, false) => None,
            };
            let spec = CoreSpec {
                path: &self.path,
                source: &src.info,
                contract: &src.contract,
                template,
                graph,
                fp16: half,
                ep: base,
                opts: &self.opts,
                levers,
                frames,
            };
            let acquired = acquire(key, share, || {
                let _span = telemetry::load_span(LoadStage::PitchPlan);
                build_core(&spec)
            });
            match acquired {
                Ok((core, joined)) => {
                    let active = core.outcome.active.clone();
                    if !src.hooks.force_plan && !wants_plan(&active, levers) {
                        // The prediction missed (a GPU rung passed its preflight and then failed on
                        // the model): the levers run the plain session there, which loads it again.
                        self.note(format!("RMVPE: the plan landed on {active}; the plain session runs"));
                        self.plain_ep = active;
                        return;
                    }
                    self.adopt(core, joined, load, warmup);
                    return;
                }
                Err(e) => {
                    let what = format!("{}{}", graph.as_str(), if half { " fp16" } else { "" });
                    tracing::warn!(target: "sigil::vc", "RMVPE: the {frames}-frame {what} plan could not be built ({e})");
                    self.note(format!("RMVPE: the {frames}-frame {what} plan could not be built ({e})"));
                }
            }
        }
    }

    /// Take `core` as this voice's plan and size the buffers for it.
    fn adopt(&mut self, core: Arc<SharedCore>, joined: bool, load: &RmvpeLoad, warmup: bool) {
        let w = core.frames;
        self.outcome = core.outcome.clone();
        self.plain_ep = core.outcome.active.clone();
        for n in &core.notes {
            self.note(format!("RMVPE plan: {n}"));
        }
        if core.fp16 {
            self.note(format!("RMVPE: the {w}-frame plan runs its U-Net in fp16 on {}", core.outcome.active));
        }
        if self.mel.is_none() {
            self.mel = Some(LogMel::new());
        }
        self.mel_buf = vec![0.0; N_MELS * w];
        self.rows_buf = Vec::with_capacity(w * N_BINS);
        let longest = load.max_context_16k.map_or(2 * w, |n| (1 + n / HOP).max(w));
        self.confidence = Vec::with_capacity(longest);
        self.salience = Vec::with_capacity(longest * N_BINS);
        tracing::info!(
            target: "sigil::vc",
            "RMVPE: {} {w}-frame {}{} plan on {}{}",
            if joined { "sharing the" } else { "built a" },
            core.graph.as_str(),
            if core.fp16 { " fp16" } else { "" },
            core.outcome.active,
            if joined { format!(" ({})", core.id) } else { String::new() }
        );
        let seen = core.done.load(Ordering::Acquire);
        self.planned = Some(Planned {
            shared: core,
            joined,
            calls: if warmup { 0 } else { WARMUP_CALLS },
            seen,
        });
    }

    /// Grow the plan once for a call longer than it, when the call sizes were unknown at load (a
    /// legacy voice calls with its whole rolling context): P3's behaviour, so its contour is P3's
    /// instead of a split call's (module docs). The plan is `32 * ceil(total / 32)` frames, at most
    /// [`MAX_PLAN_FRAMES`]; a longer call splits. The first such call is the voice's load warm-up,
    /// so this is built before the first tick. A failed growth keeps the old plan when another
    /// voice holds it too (it splits), else the plain session takes over, as P3's did, and is not
    /// retried.
    fn grow_for(&mut self, total: usize) {
        let Some(w) = self.planned.as_ref().map(|p| p.shared.frames) else {
            return;
        };
        if total <= w || w >= MAX_PLAN_FRAMES {
            return;
        }
        let Some(src) = self.grow.take() else {
            return;
        };
        let want = (PLAN_ALIGN * total.div_ceil(PLAN_ALIGN)).min(MAX_PLAN_FRAMES);
        tracing::info!(target: "sigil::vc", "RMVPE: a {total}-frame call outgrew the {w}-frame plan; building a {want}-frame one");
        // Free the old plan's device memory before the larger one is built, unless another voice
        // holds it (then it stays for them, and for this voice if the growth fails).
        let kept = self
            .planned
            .take()
            .filter(|p| Arc::strong_count(&p.shared) > 1);
        self.try_plan(&src, want, false);
        if self.planned.is_some() {
            self.stats.grows += 1;
            self.grow = Some(src);
        } else {
            self.planned = kept;
        }
    }

    /// The plain session, built now if missing (the load's legacy path, or after a plan failure).
    fn ensure_plain(&mut self) -> Result<(), String> {
        if self.plain.is_some() {
            return Ok(());
        }
        let path_str = self.path.to_string_lossy().into_owned();
        let (session, outcome) = {
            let _span = telemetry::load_span(LoadStage::PitchPlain);
            build_session(Some("vc.pitch"), &self.plain_ep, &self.opts, ModelSource::File(&self.path))
                .map_err(|e| format!("Failed to load RMVPE ONNX {path_str}: {e}"))?
        };
        // w-okada's fused `rmvpe2_silence.onnx`, a content encoder or a voice set as the RMVPE model
        // is named for what it is (P5-D's graph identification; load time only).
        if let Some(why) = super::rvc::foreign_graph_problem(&session, &self.path, "an RMVPE pitch model", &[super::rvc::GraphKind::PitchModel]) {
            return Err(format!("RMVPE: {why}"));
        }
        let inputs = PitchInputs::inspect(&session, "RMVPE", &path_str, WAVEFORM_EXPORT_THRESHOLD)?;
        match inputs.main {
            MainInput::Waveform { .. } => self.mel = None,
            MainInput::LogMel { n_mels } => {
                if n_mels.is_some_and(|m| m != N_MELS) {
                    return Err(format!(
                        "RMVPE ONNX {} expects a {:?} mel input; Sigil computes RVC's {}-band log-mel [1, {}, T]",
                        path_str,
                        inputs.main_dims(),
                        N_MELS,
                        N_MELS
                    ));
                }
                if self.mel.is_none() {
                    self.mel = Some(LogMel::new());
                }
            }
        }
        self.pin = ShapePin::for_ep(&outcome.active);
        if self.planned.is_none() {
            self.outcome = outcome;
        }
        self.plain = Some(PlainSession { session, inputs });
        Ok(())
    }

    /// EP outcome of the load (`vc.pitch`), for callers that report where F0 runs.
    pub fn outcome(&self) -> &EpOutcome {
        &self.outcome
    }

    /// Per-frame confidence of the last [`PitchExtractor::f0`] call, aligned with its contour: the
    /// peak salience (0..1) for salience exports, 1 / 0 (voiced / unvoiced) for exports that output
    /// F0 directly.
    pub fn last_confidence(&self) -> &[f32] {
        &self.confidence
    }

    /// Whether calls run through a fixed-shape plan.
    #[allow(dead_code)] // Diagnostics accessor (tests, the benchmark).
    pub fn planned(&self) -> bool {
        self.planned.is_some()
    }

    /// The plan's mel frames, graph and precision (`None` on the plain session).
    #[allow(dead_code)] // Diagnostics accessor (tests, the benchmark).
    pub fn plan_shape(&self) -> Option<(usize, RmvpeGraph, bool)> {
        self.planned
            .as_ref()
            .map(|p| (p.shared.frames, p.shared.graph, p.shared.fp16))
    }

    /// The id of the plan this voice holds in [`shared_components`] (`None` on the plain session or
    /// a private plan).
    #[allow(dead_code)] // For VoiceSet's VRAM accounting (P3B-H).
    pub fn shared_component_id(&self) -> Option<&str> {
        let p = self.planned.as_ref()?;
        registry()
            .iter()
            .any(|e| std::ptr::eq(e.core.as_ptr(), Arc::as_ptr(&p.shared)))
            .then_some(p.shared.id.as_str())
    }

    /// Non-fatal notes of the load (a graph or precision that fell back, the plan's lever notes).
    pub fn load_notes(&self) -> &[String] {
        &self.notes
    }

    /// Call counts since the load.
    #[allow(dead_code)] // Diagnostics accessor (tests, the benchmark).
    pub fn stats(&self) -> RmvpeStats {
        self.stats
    }

    /// The contour of `mono` through the plan into `out` (split when longer than the plan), with
    /// confidence and salience. Allocation-free once the buffers have their capacity.
    fn f0_planned_into(&mut self, mono: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
        let RmvpePitchExtractor {
            planned,
            mel,
            mel_buf,
            rows_buf,
            confidence,
            salience,
            stats,
            ..
        } = self;
        let p = planned.as_mut().ok_or("RMVPE plan missing")?;
        let mel = mel.as_mut().ok_or("RMVPE mel front end missing")?;
        let w = p.shared.frames;
        let n = mono.len();
        let total = 1 + n / HOP;
        if p.in_warmup() {
            // A background load's warm-up: when another voice is calling, run right after its call.
            let now = p.shared.done.load(Ordering::Acquire);
            if now != p.seen {
                let t0 = Instant::now();
                while p.shared.done.load(Ordering::Acquire) == now && t0.elapsed() < SHADOW_WAIT {
                    std::thread::sleep(Duration::from_micros(500));
                }
                stats.shadow_waits += 1;
            }
        }
        p.calls = p.calls.saturating_add(1);
        confidence.clear();
        salience.clear();
        let mut runs = 0u64;
        for sub in SplitCalls::new(total, w) {
            let from = sub.start * HOP;
            let to = if sub.last { n } else { (sub.start + w - 1) * HOP };
            let slice = &mono[from..to];
            let frames = 1 + slice.len() / HOP;
            // The hard check of the fixed shape (t32 is exact only at its 32 frames).
            if frames > w {
                return Err(format!("RMVPE: a {frames}-frame sub-call does not fit the {w}-frame plan"));
            }
            mel.compute_into(slice, w, mel_buf)?;
            let lead = w - frames;
            let rows = lead + (sub.keep_from - sub.start)..lead + (sub.keep_to - sub.start);
            run_core(&p.shared, mel_buf, rows, rows_buf)?;
            decode_salience_rows(rows_buf, out, confidence, salience);
            runs += 1;
        }
        p.seen = p.shared.done.load(Ordering::Acquire);
        stats.plan_runs += runs;
        if runs > 1 {
            stats.split_calls += 1;
        }
        Ok(())
    }

    /// The contour of `mono` through the plain session into `out` (the legacy path).
    fn f0_plain_into(&mut self, mono: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
        self.ensure_plain()?;
        let RmvpePitchExtractor {
            plain,
            mel,
            pin,
            mel_buf,
            wave_buf,
            confidence,
            salience,
            stats,
            ..
        } = self;
        let plain = plain.as_mut().ok_or("RMVPE session missing")?;
        let n = mono.len();
        confidence.clear();
        salience.clear();
        // Log-mel exports: the mel into the reused buffer, end-aligned in a multiple of 32 frames
        // (and in the DirectML pin).
        let mel_frames = match mel {
            Some(mel) => {
                let padded = pin.size(PLAN_ALIGN * (1 + n / HOP).div_ceil(PLAN_ALIGN));
                mel_buf.clear();
                mel_buf.resize(N_MELS * padded, 0.0);
                Some((mel.compute_into(mono, padded, mel_buf)?, padded))
            }
            None => None,
        };
        // Waveform exports on DirectML: the pinned length, front-padded with silence in whole frames
        // (the padding's frames are dropped below; the newest frames keep their real right context).
        let run_n = match mel_frames {
            Some(_) => n,
            None => pin.size_aligned(n, HOP),
        };
        let lead_frames = (run_n - n) / HOP;
        let wave: &[f32] = if run_n == n {
            mono
        } else {
            wave_buf.clear();
            wave_buf.resize(run_n - n, 0.0);
            wave_buf.extend_from_slice(mono);
            wave_buf
        };
        let outputs = match (mel_frames, plain.inputs.main) {
            (Some((_, padded)), _) => plain
                .inputs
                .run(&mut plain.session, vec![1, N_MELS, padded], mel_buf, "RMVPE")?,
            (None, MainInput::Waveform { rank }) => plain
                .inputs
                .run(&mut plain.session, waveform_shape(rank, run_n), wave, "RMVPE")?,
            (None, MainInput::LogMel { .. }) => return Err("RMVPE mel front end missing".into()),
        };
        // The plain session runs eagerly with host tensors (P3B-S0 telemetry: one GPU wait).
        telemetry::record_sync(SyncKind::EagerRun);
        stats.plain_runs += 1;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("RMVPE output: {}", e))?;
        let dims: Vec<usize> = shape.iter().map(|d| (*d).max(0) as usize).collect();
        match mel_frames {
            // The audio's frames end the end-aligned mel.
            Some((frames, _)) => decode_output_into(&dims, data, Keep::Tail(frames), out, confidence, salience)?,
            None => decode_output_into(&dims, data, Keep::Head(1 + run_n / HOP), out, confidence, salience)?,
        }
        if lead_frames > 0 {
            let lead = lead_frames.min(out.len());
            out.drain(..lead);
            confidence.drain(..lead.min(confidence.len()));
            if !salience.is_empty() {
                salience.drain(..(lead * N_BINS).min(salience.len()));
            }
        }
        Ok(())
    }

    fn smooth(&self, out: &mut [f32]) {
        #[cfg(test)]
        if self.skip_smoothing {
            return;
        }
        smooth_f0_stream(out, 3);
    }
}

impl PitchExtractor for RmvpePitchExtractor {
    fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(1 + mono_16k.len() / HOP);
        self.f0_into(mono_16k, &mut out)?;
        Ok(out)
    }

    fn f0_into(&mut self, mono_16k: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
        out.clear();
        if mono_16k.is_empty() {
            self.confidence.clear();
            self.salience.clear();
            return Ok(());
        }
        self.stats.calls += 1;
        if self.grow.is_some() {
            self.grow_for(1 + mono_16k.len() / HOP);
        }
        if self.planned.is_some() {
            match self.f0_planned_into(mono_16k, out) {
                Ok(()) => {
                    self.smooth(out);
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!(target: "sigil::vc", "RMVPE: the fixed-shape plan failed ({e}); running the plain session");
                    // Release this voice's hold on the plan (freed with its last voice).
                    self.planned = None;
                    self.grow = None;
                    out.clear();
                }
            }
        }
        self.f0_plain_into(mono_16k, out)?;
        self.smooth(out);
        Ok(())
    }

    /// The plan's audio frames (its mel frames minus the centred STFT's extra frame); a plain
    /// log-mel session pads to 32 frames anyway, so 31 there too.
    fn preferred_input_frames(&self) -> Option<usize> {
        match (&self.planned, &self.mel) {
            (Some(p), _) => Some(p.shared.frames - 1),
            (None, Some(_)) => Some(PLAN_ALIGN - 1),
            (None, None) => None,
        }
    }

    /// On a plan, its audio frames: a longer call is split here (module docs), never grown.
    fn max_call_frames(&self) -> Option<usize> {
        self.planned.as_ref().map(|p| p.shared.frames - 1)
    }

    fn last_salience(&self) -> Option<(&[f32], usize)> {
        (!self.salience.is_empty() && self.salience.len() == self.confidence.len() * N_BINS).then_some((self.salience.as_slice(), N_BINS))
    }
}

/// Whether a log-mel export on `active_ep` runs as a fixed-shape plan with `levers` (module docs):
/// DirectML with `io_binding`, CUDA with `io_binding` and `cuda_graphs`; every other case runs the
/// plain session (the legacy path).
fn wants_plan(active_ep: &str, levers: GpuLevers) -> bool {
    match active_ep {
        "dml" => levers.io_binding,
        "cuda" => levers.io_binding && levers.cuda_graphs,
        _ => false,
    }
}

// ---------------------------------------------------------------------------------------------
// The graph contract from the file, and the plan build
// ---------------------------------------------------------------------------------------------

/// RVC's `rmvpe.onnx` contract read from the protobuf: one float `[1, 128, T]` input, a float
/// output.
#[derive(Clone, Debug, PartialEq, Eq)]
struct MelContract {
    input: String,
    output: String,
    /// The mel time `dim_param` (overridden to the plan's length).
    time_dim: Option<String>,
    /// A mel length the graph fixes.
    fixed_frames: Option<usize>,
}

impl MelContract {
    fn of(info: &GraphInfo) -> Option<MelContract> {
        let [input] = info.inputs.as_slice() else {
            return None;
        };
        let output = info.outputs.first()?;
        if input.elem_type != DT_FLOAT || output.elem_type != DT_FLOAT || input.dims.len() != 3 {
            return None;
        }
        let batch_ok = matches!(input.dims[0], Dim::Fixed(1) | Dim::Param(_) | Dim::Unknown);
        let mels_ok = matches!(input.dims[1], Dim::Fixed(m) if m == N_MELS as i64) || !matches!(input.dims[1], Dim::Fixed(_));
        if !batch_ok || !mels_ok {
            return None;
        }
        let (time_dim, fixed_frames) = match &input.dims[2] {
            Dim::Param(p) if !p.is_empty() => (Some(p.clone()), None),
            Dim::Fixed(t) if *t > 0 => (None, Some(*t as usize)),
            _ => (None, None),
        };
        Some(MelContract {
            input: input.name.clone(),
            output: output.name.clone(),
            time_dim,
            fixed_frames,
        })
    }
}

/// [`super::rvc::foreign_graph_problem`] from the protobuf (no session).
fn foreign_graph(info: &GraphInfo, path: &Path) -> Option<String> {
    use super::rvc::{identify_graph, wrong_graph_message, GraphKind, GraphValue};
    let values = |v: &[onnx_proto::ValueInfo]| -> Vec<GraphValue> {
        v.iter()
            .map(|x| GraphValue {
                name: x.name.clone(),
                dtype: element_type(x.elem_type),
                dims: Some(
                    x.dims
                        .iter()
                        .map(|d| match d {
                            Dim::Fixed(v) => *v,
                            _ => -1,
                        })
                        .collect(),
                ),
            })
            .collect()
    };
    let (inputs, outputs) = (values(&info.inputs), values(&info.outputs));
    let kind = identify_graph(&inputs, &outputs);
    if kind == GraphKind::Unknown || kind == GraphKind::PitchModel {
        return None;
    }
    wrong_graph_message(path, "an RMVPE pitch model", &inputs, &outputs)
}

fn element_type(t: i32) -> Option<ort::value::TensorElementType> {
    use ort::value::TensorElementType as T;
    Some(match t {
        1 => T::Float32,
        2 => T::Uint8,
        3 => T::Int8,
        5 => T::Int16,
        6 => T::Int32,
        7 => T::Int64,
        9 => T::Bool,
        10 => T::Float16,
        11 => T::Float64,
        _ => return None,
    })
}

/// `<path>|<size>|<mtime>` of a file the model store does not vouch for (the plan key).
fn file_identity(path: &Path) -> String {
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let meta = std::fs::metadata(path).ok();
    let size = meta.as_ref().map_or(0, std::fs::Metadata::len);
    let mtime = meta
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    format!("{}|{size}|{mtime}", canon.display())
}

/// Whether the model store's install record vouches that `path` is the catalog's
/// `vc/pitch-rmvpe` (the sha256 checked at install is [`RMVPE_SHA256`], and the file's size and
/// mtime have not changed since): the condition for the graph templates. Never hashes the file.
fn verified_by_store(path: &Path) -> Result<(), String> {
    // Unit tests never consult (or create) the process's model store; the dev-rig tests that load
    // the real model hash it themselves (`Hooks::trust_source`). `store_vouches` is tested apart.
    if cfg!(test) {
        return Err("unit tests do not consult the model store".into());
    }
    let store = crate::services::model_store::global_store().map_err(|e| format!("the model store is not available ({e})"))?;
    let entry = store.entry(RMVPE_ENTRY)?;
    let (root, record, _) = store
        .installed(entry)
        .ok_or_else(|| format!("{RMVPE_ENTRY} is not installed through the model store"))?;
    let facts = InstallFacts {
        pinned: record.pinned,
        files: record
            .files
            .iter()
            .map(|f| (root.join(&f.path), f.size, f.sha256.clone()))
            .collect(),
        installed_at_unix_ms: record.installed_at_unix_ms,
    };
    store_vouches(&facts, path)
}

/// What the model store's install record of `vc/pitch-rmvpe` says (copied out of the record).
#[derive(Clone, Debug)]
struct InstallFacts {
    /// Every file was checked against a catalog hash at install.
    pinned: bool,
    /// The catalog files as installed: absolute path, size, sha256 recorded at install.
    files: Vec<(PathBuf, u64, Option<String>)>,
    installed_at_unix_ms: u64,
}

/// [`verified_by_store`]'s decision over the record's facts and the file now: the record is
/// pinned, lists `path` (by canonical path) with the catalog's [`RMVPE_SHA256`] and the file's
/// size, and the file was not modified after the install (its mtime is at most
/// [`STORE_MTIME_SLACK_MS`] later than the record, which is written after the file).
fn store_vouches(facts: &InstallFacts, path: &Path) -> Result<(), String> {
    if !facts.pinned {
        return Err(format!("the {RMVPE_ENTRY} install was not checked against the catalog hash"));
    }
    let target = std::fs::canonicalize(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (_, size, sha256) = facts
        .files
        .iter()
        .find(|(p, _, _)| std::fs::canonicalize(p).ok().as_ref() == Some(&target))
        .ok_or_else(|| format!("{} is not the model store's {RMVPE_ENTRY}", path.display()))?;
    if sha256.as_deref() != Some(RMVPE_SHA256) {
        return Err(format!("the installed {RMVPE_ENTRY} is not the version the templates were cut from"));
    }
    let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if meta.len() != *size {
        return Err(format!("{} changed size since its install", path.display()));
    }
    let modified_ms = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_millis() as u64);
    // Written before the record; a later mtime means the file was replaced after the install.
    if modified_ms > facts.installed_at_unix_ms.saturating_add(STORE_MTIME_SLACK_MS) {
        return Err(format!("{} was modified after its install", path.display()));
    }
    Ok(())
}

/// Everything one plan is built from.
struct CoreSpec<'a> {
    path: &'a Path,
    source: &'a GraphInfo,
    contract: &'a MelContract,
    /// A weightless template (filled from `path`), or `None` for the file itself.
    template: Option<&'static [u8]>,
    graph: RmvpeGraph,
    fp16: bool,
    ep: &'a str,
    opts: &'a VcSessionOptions,
    levers: GpuLevers,
    frames: usize,
}

/// Build and warm one plan (load time only).
fn build_core(spec: &CoreSpec<'_>) -> Result<SharedCore, String> {
    let frames = spec.frames;
    let t0 = Instant::now();
    let (input, output, time_dim, image) = match spec.template {
        Some(bytes) => {
            let t = onnx_proto::read_graph_info_bytes(bytes).map_err(|e| format!("graph template: {e}"))?;
            if t.metadata("sigil.source_sha256") != Some(RMVPE_SHA256) || t.metadata("sigil.weights") != Some(TEMPLATE_WEIGHTS) {
                return Err("the graph template was not cut from the catalog's rmvpe.onnx".into());
            }
            let c = MelContract::of(&t).ok_or("the graph template does not take a [1, 128, T] mel")?;
            if c.fixed_frames.is_some_and(|f| f != frames) {
                return Err(format!(
                    "the {} template runs {:?} mel frames, not {frames}",
                    spec.graph.as_str(),
                    c.fixed_frames
                ));
            }
            if spec.graph == RmvpeGraph::T32 && (frames != PLAN_ALIGN || c.fixed_frames != Some(PLAN_ALIGN)) {
                return Err("t32 runs only at 32 mel frames".into());
            }
            let image = onnx_proto::external_weights_image(&t, TEMPLATE_WEIGHTS, spec.source, spec.path)?;
            tracing::debug!(
                target: "sigil::vc",
                "RMVPE {}{} weights: {} tensors ({} centre rows, {} fp16), {:.1} MiB",
                spec.graph.as_str(),
                if spec.fp16 { " fp16" } else { "" },
                image.tensors,
                image.centre_rows,
                image.to_f16,
                image.bytes.len() as f64 / (1024.0 * 1024.0)
            );
            (c.input, c.output, c.time_dim, Some(image))
        }
        None => (
            spec.contract.input.clone(),
            spec.contract.output.clone(),
            spec.contract.time_dim.clone(),
            None,
        ),
    };
    let inputs = [TensorSpec::f32(&input, &[1, N_MELS, frames])];
    let outputs = [TensorSpec::f32(&output, &[1, frames, N_BINS])];
    let dims: Vec<(&str, i64)> = time_dim
        .iter()
        .map(|d| (d.as_str(), frames as i64))
        .collect();
    let model = match spec.template {
        Some(bytes) => ModelSource::Memory(bytes),
        None => ModelSource::File(spec.path),
    };
    let image_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut plan = SessionPlan::build(&PlanSpec {
        component: "vc.pitch",
        ep: spec.ep,
        opts: spec.opts,
        model,
        weights: image.as_ref().map(|i| ExternalWeights {
            file: TEMPLATE_WEIGHTS,
            image: &i.bytes,
        }),
        dims: &dims,
        inputs: &inputs,
        outputs: &outputs,
        levers: spec.levers,
    })?;
    // The plan copied the image (ONNX Runtime keeps its copy for the session's lifetime).
    drop(image);
    for note in plan.notes() {
        tracing::info!(target: "sigil::vc", "RMVPE plan: {note}");
    }
    // Warm once on silence (every rung; the build already ran the GPU rungs on zeros).
    plan.input_f32(&input)?.fill(MEL_CLAMP.ln());
    plan.run()?;
    tracing::info!(
        target: "sigil::vc",
        "RMVPE {frames}-frame {}{} plan: weights image {image_ms:.0} ms, session and warm-up {:.0} ms",
        spec.graph.as_str(),
        if spec.fp16 { " fp16" } else { "" },
        t0.elapsed().as_secs_f64() * 1000.0 - image_ms
    );
    let outcome = plan.outcome().clone();
    let notes = plan.notes().to_vec();
    let id = format!(
        "rmvpe:{}{}:{frames}f:{}#{}",
        spec.graph.as_str(),
        if spec.fp16 { "-fp16" } else { "" },
        outcome.active,
        NEXT_CORE.fetch_add(1, Ordering::Relaxed)
    );
    Ok(SharedCore {
        core: Mutex::new(Core { plan, input, output }),
        done: AtomicU64::new(0),
        id,
        frames,
        graph: spec.graph,
        fp16: spec.fp16,
        outcome,
        notes,
    })
}

// ---------------------------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------------------------

/// Append the F0, confidence (peak salience) and salience of salience rows (`k × 360`, row-major).
/// Allocation-free within the buffers' capacity.
fn decode_salience_rows(rows: &[f32], out: &mut Vec<f32>, confidence: &mut Vec<f32>, salience: &mut Vec<f32>) {
    for row in rows.as_chunks::<N_BINS>().0 {
        // A NaN near the peak (or an overflowing weighted mean) is unvoiced, never NaN F0.
        let f = salience_to_f0(row);
        let peak = row
            .iter()
            .copied()
            .fold(0.0f32, |a, v| if v > a { v } else { a });
        confidence.push(peak.min(1.0));
        out.push(if f.is_finite() { f } else { 0.0 });
    }
    salience.extend(rows.iter().map(|&v| if v.is_finite() { v } else { 0.0 }));
}

/// Which output rows a plain-session call keeps.
#[derive(Clone, Copy, Debug)]
enum Keep {
    /// The last `n` rows (an end-aligned mel: the rows before them are the padding's).
    Tail(usize),
    /// The first `n` rows (RVC `hidden[:, :n_frames]`).
    Head(usize),
}

/// The salience rows of a `[1, T, 360]` / `[T, 360]` output (`cut`: the frames asked for, so a
/// one-row `[1, 360]` is F0 when one frame was fed).
fn salience_rows(dims: &[usize], cut: usize) -> Option<usize> {
    match dims {
        [1, t, b] if *b == N_BINS => Some(*t),
        [t, b] if *b == N_BINS && (*t != 1 || cut <= 1) => Some(*t),
        _ => None,
    }
}

/// Decode a plain-session RMVPE output into `out` (Hz), `confidence` and (for salience) `salience`.
/// Salience `[1, T, 360]` / `[T, 360]` keeps the rows `keep` says; a direct F0 output `[1, T]` /
/// `[T]` passes through (the tail of it for [`Keep::Tail`]).
fn decode_output_into(
    dims: &[usize],
    data: &[f32],
    keep: Keep,
    out: &mut Vec<f32>,
    confidence: &mut Vec<f32>,
    salience: &mut Vec<f32>,
) -> Result<(), String> {
    let cut = match keep {
        // No cut: the padded mel has at least 32 frames, so a `[T, 360]` output is never
        // mistaken for a one-row F0.
        Keep::Tail(_) => usize::MAX,
        Keep::Head(n) => n,
    };
    if let Some(t) = salience_rows(dims, cut) {
        let t = t.min(data.len() / N_BINS);
        let rows = match keep {
            Keep::Tail(n) => t.saturating_sub(n)..t,
            Keep::Head(n) => 0..t.min(n),
        };
        decode_salience_rows(&data[rows.start * N_BINS..rows.end * N_BINS], out, confidence, salience);
        return Ok(());
    }
    let f0 = flat_f0(dims, data).ok_or_else(|| {
        format!(
            "RMVPE output shape {dims:?} unsupported; expected salience [1, T, 360] / [T, 360] \
             or F0 [1, T] / [T]"
        )
    })?;
    let from = match keep {
        Keep::Tail(n) => f0.len().saturating_sub(n),
        Keep::Head(_) => 0,
    };
    for &v in &f0[from..] {
        let v = if !v.is_finite() || v < 0.0 { 0.0 } else { v };
        out.push(v);
        confidence.push(if v > 0.0 { 1.0 } else { 0.0 });
    }
    Ok(())
}

/// [`decode_output_into`] into new vectors: salience cut to `frames` rows (the P3 tests' form).
#[cfg(test)]
fn decode_output(dims: &[usize], data: &[f32], frames: usize, confidence: &mut Vec<f32>) -> Result<Vec<f32>, String> {
    let mut out = Vec::new();
    confidence.clear();
    decode_output_into(dims, data, Keep::Head(frames), &mut out, confidence, &mut Vec::new())?;
    Ok(out)
}

/// RVC `to_local_average_cents` + `decode` for one salience row (0 at or below the 0.03 peak
/// threshold).
fn salience_to_f0(row: &[f32]) -> f32 {
    let (hz, peak) = local_average_f0(row);
    if peak <= SALIENCE_THRESHOLD {
        return 0.0;
    }
    hz
}

/// RVC `MelSpectrogram(is_half, 128, 16000, 1024, 160, None, 30, 8000)`, padded like `mel2hidden`
/// but end-aligned (module docs).
struct LogMel {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    /// Per mel band: first FFT bin and its non-zero weights.
    bands: Vec<(usize, Vec<f32>)>,
    frame: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    magnitude: Vec<f32>,
}

fn hz_to_mel_htk(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz_htk(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

/// librosa `filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=True, norm="slaney")`.
fn mel_filterbank() -> Vec<(usize, Vec<f32>)> {
    let n_freqs = N_FFT / 2 + 1;
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * f64::from(SR) / N_FFT as f64)
        .collect();
    let (mel_lo, mel_hi) = (hz_to_mel_htk(MEL_FMIN), hz_to_mel_htk(MEL_FMAX));
    let mel_f: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz_htk(mel_lo + (mel_hi - mel_lo) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    (0..N_MELS)
        .map(|i| {
            let (lo, center, hi) = (mel_f[i], mel_f[i + 1], mel_f[i + 2]);
            let enorm = 2.0 / (hi - lo);
            let weights: Vec<f64> = fft_freqs
                .iter()
                .map(|&f| {
                    let lower = (f - lo) / (center - lo);
                    let upper = (hi - f) / (hi - center);
                    lower.min(upper).max(0.0) * enorm
                })
                .collect();
            let first = weights.iter().position(|&w| w > 0.0).unwrap_or(0);
            let last = weights.iter().rposition(|&w| w > 0.0).unwrap_or(0);
            let band: Vec<f32> = if last >= first && weights[first] > 0.0 {
                weights[first..=last].iter().map(|&w| w as f32).collect()
            } else {
                Vec::new()
            };
            (first, band)
        })
        .collect()
}

/// torch `reflect` padding index (no edge repeat) for any offset.
fn reflect_index(i: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let mut j = i.rem_euclid(period);
    if j >= n as isize {
        j = period - j;
    }
    j as usize
}

impl LogMel {
    fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        // torch.hann_window(1024) is periodic.
        let window = (0..N_FFT)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32)
            .collect();
        let frame = fft.make_input_vec();
        let spectrum = fft.make_output_vec();
        let scratch = fft.make_scratch_vec();
        Self {
            fft,
            window,
            bands: mel_filterbank(),
            frame,
            spectrum,
            scratch,
            magnitude: vec![0.0; N_FFT / 2 + 1],
        }
    }

    /// Returns `(frames, padded_frames, data)` with `data` = `[N_MELS, padded_frames]` row-major.
    /// `min_padded`: pad to at least this many frames (a DirectML session's pinned shape).
    #[cfg(test)]
    fn compute(&mut self, audio: &[f32], min_padded: usize) -> Result<(usize, usize, Vec<f32>), String> {
        let frames = 1 + audio.len() / HOP;
        let padded_frames = (32 * ((frames - 1) / 32 + 1)).max(min_padded);
        let mut data = vec![0.0f32; N_MELS * padded_frames];
        self.compute_into(audio, padded_frames, &mut data)?;
        Ok((frames, padded_frames, data))
    }

    /// The log-mel of `audio` into `data` (`[N_MELS, padded_frames]` row-major, every value written),
    /// end-aligned: the audio's frames are the last `frames` columns and the columns before them
    /// hold the log-mel floor `ln(1e-5)` (silence). Returns the audio's frame count. Allocation-free.
    fn compute_into(&mut self, audio: &[f32], padded_frames: usize, data: &mut [f32]) -> Result<usize, String> {
        let n = audio.len();
        let frames = 1 + n / HOP;
        if padded_frames < frames || data.len() != N_MELS * padded_frames {
            return Err(format!(
                "RMVPE mel buffer: {} values for {frames} frames padded to {padded_frames}",
                data.len()
            ));
        }
        data.fill(MEL_CLAMP.ln());
        let lead = padded_frames - frames;
        let pad = (N_FFT / 2) as isize;
        for t in 0..frames {
            let origin = (t * HOP) as isize - pad;
            for (k, (dst, &w)) in self.frame.iter_mut().zip(self.window.iter()).enumerate() {
                *dst = audio[reflect_index(origin + k as isize, n)] * w;
            }
            self.fft
                .process_with_scratch(&mut self.frame, &mut self.spectrum, &mut self.scratch)
                .map_err(|e| format!("RMVPE mel FFT: {e}"))?;
            for (m, c) in self.magnitude.iter_mut().zip(self.spectrum.iter()) {
                *m = (c.re * c.re + c.im * c.im).sqrt();
            }
            for (band_idx, (first, weights)) in self.bands.iter().enumerate() {
                let energy: f32 = weights
                    .iter()
                    .zip(&self.magnitude[*first..])
                    .map(|(w, m)| w * m)
                    .sum();
                data[band_idx * padded_frames + lead + t] = energy.max(MEL_CLAMP).ln();
            }
        }
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hz(cents: f32) -> f32 {
        10.0 * (cents / 1200.0).exp2()
    }

    fn temp_model(tag: &str, bytes: &[u8]) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("sigil-p3b-rmvpe-{tag}-{}-{nanos}.onnx", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// Speech-like test audio: two tones with a slow level change.
    fn test_audio(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (0.3 * (i as f32 * 0.071).sin() + 0.2 * (i as f32 * 0.013).sin()) * (1.0 + (i / 3000) as f32 * 0.2))
            .collect()
    }

    fn bound() -> GpuLevers {
        GpuLevers {
            io_binding: true,
            ..GpuLevers::default()
        }
    }

    /// A test load on the CPU rung with the plan forced (it runs on DirectML / CUDA in production).
    fn planned_load(path: &Path, load: RmvpeLoad, hooks: Hooks) -> RmvpePitchExtractor {
        let hooks = Hooks { force_plan: true, ..hooks };
        RmvpePitchExtractor::load_inner(&path.to_string_lossy(), "cpu", &VcSessionOptions::default(), bound(), &load, hooks).unwrap()
    }

    fn private() -> RmvpeLoad {
        RmvpeLoad {
            shared: false,
            ..RmvpeLoad::default()
        }
    }

    /// The VRAM saver's 32-frame plan, private.
    fn saver() -> RmvpeLoad {
        RmvpeLoad {
            vram_saver: true,
            ..private()
        }
    }

    #[test]
    fn salience_decode_matches_rvc_local_average() {
        // Symmetric peak at bin 100 → exactly the bin's cents.
        let mut row = vec![0.0f32; N_BINS];
        row[99] = 0.5;
        row[100] = 0.9;
        row[101] = 0.5;
        let expected = hz(20.0 * 100.0 + CENTS_OFFSET);
        assert!((salience_to_f0(&row) - expected).abs() < 1e-3);
        assert!((expected - 100.64).abs() < 0.05, "{expected}");

        // Asymmetric window: weighted mean over argmax ± 4 only (bin 110 is outside).
        let mut row = vec![0.0f32; N_BINS];
        row[200] = 0.8;
        row[203] = 0.4;
        row[210] = 0.7;
        let cents = (0.8 * (20.0 * 200.0 + CENTS_OFFSET) + 0.4 * (20.0 * 203.0 + CENTS_OFFSET)) / 1.2;
        assert!((salience_to_f0(&row) - hz(cents)).abs() < 1e-2);

        // Edge bins: the window is clipped (RVC pads zeros).
        let mut row = vec![0.0f32; N_BINS];
        row[0] = 0.6;
        row[2] = 0.2;
        let cents = (0.6 * CENTS_OFFSET + 0.2 * (40.0 + CENTS_OFFSET)) / 0.8;
        assert!((salience_to_f0(&row) - hz(cents)).abs() < 1e-3);

        // Peak at or below the 0.03 threshold → unvoiced.
        let mut row = vec![0.0f32; N_BINS];
        row[150] = 0.03;
        assert_eq!(salience_to_f0(&row), 0.0);
    }

    #[test]
    fn decode_output_handles_salience_and_flat_shapes() {
        let mut sal = vec![0.0f32; 3 * N_BINS];
        sal[N_BINS + 100] = 1.0;
        // [1, T, 360] cut to `frames` rows.
        let f0 = decode_output(&[1, 3, N_BINS], &sal, 2, &mut Vec::new()).unwrap();
        assert_eq!(f0.len(), 2);
        assert_eq!(f0[0], 0.0);
        assert!((f0[1] - hz(20.0 * 100.0 + CENTS_OFFSET)).abs() < 1e-3);
        // [T, 360].
        assert_eq!(
            decode_output(&[3, N_BINS], &sal, 3, &mut Vec::new())
                .unwrap()
                .len(),
            3
        );
        // A flat [1, T] F0 output is taken as Hz, not collapsed into one salience row.
        let flat = vec![100.0f32, 0.0, 220.0, 230.0];
        assert_eq!(decode_output(&[1, 4], &flat, 5, &mut Vec::new()).unwrap(), flat);
        assert_eq!(decode_output(&[4], &flat, 5, &mut Vec::new()).unwrap(), flat);
        assert!(decode_output(&[2, 2, 2], &[0.0; 8], 2, &mut Vec::new()).is_err());
        // The tail of an end-aligned mel's output: the last rows, with their salience.
        let (mut out, mut conf, mut s) = (Vec::new(), Vec::new(), Vec::new());
        decode_output_into(&[1, 3, N_BINS], &sal, Keep::Tail(2), &mut out, &mut conf, &mut s).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - hz(20.0 * 100.0 + CENTS_OFFSET)).abs() < 1e-3);
        assert_eq!(s.len(), 2 * N_BINS);
        assert_eq!(s[100], 1.0);
        assert_eq!(conf, vec![1.0, 0.0]);
    }

    #[test]
    fn decode_output_maps_non_finite_salience_to_unvoiced() {
        let mut sal = vec![0.0f32; 3 * N_BINS];
        // Row 0: NaN next to the peak poisons the weighted mean.
        sal[100] = 0.9;
        sal[101] = f32::NAN;
        // Row 1: all NaN (no peak at all).
        for v in &mut sal[N_BINS..2 * N_BINS] {
            *v = f32::NAN;
        }
        // Row 2: infinite salience.
        sal[2 * N_BINS + 50] = f32::INFINITY;
        let f0 = decode_output(&[1, 3, N_BINS], &sal, 3, &mut Vec::new()).unwrap();
        assert_eq!(f0.len(), 3);
        assert!(f0.iter().all(|v| v.is_finite()), "{f0:?}");
        assert_eq!(f0[0], 0.0);
        assert_eq!(f0[1], 0.0);
        // A flat F0 output with NaN / negative values is sanitized too.
        assert_eq!(
            decode_output(&[3], &[f32::NAN, -1.0, 150.0], 3, &mut Vec::new()).unwrap(),
            vec![0.0, 0.0, 150.0]
        );
        // The shared salience never carries a non-finite value.
        let (mut out, mut conf, mut s) = (Vec::new(), Vec::new(), Vec::new());
        decode_salience_rows(&sal, &mut out, &mut conf, &mut s);
        assert!(s.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mel_filterbank_matches_librosa_htk_slaney() {
        let bands = mel_filterbank();
        assert_eq!(bands.len(), N_MELS);
        // Band 0 spans 30 Hz .. ~58.7 Hz (centre ~44.2 Hz): FFT bins 2 (31.25 Hz) and 3 (46.875 Hz).
        let (first, w) = &bands[0];
        assert_eq!(*first, 2);
        assert_eq!(w.len(), 2);
        assert!((w[0] - 0.006_12).abs() < 2e-4, "{w:?}");
        assert!((w[1] - 0.057_0).abs() < 1e-3, "{w:?}");
        // Last band ends at fmax = 8000 Hz = bin 512, where its weight is (numerically) zero.
        let (first, w) = &bands[N_MELS - 1];
        assert!(first + w.len() <= 513);
        if first + w.len() == 513 {
            assert!(w[w.len() - 1] < 1e-6, "{:?}", &w[w.len() - 3..]);
        }
        assert!(bands.iter().all(|(_, w)| w.iter().all(|&x| x >= 0.0)));
    }

    #[test]
    fn log_mel_frame_count_padding_and_silence_floor() {
        let mut mel = LogMel::new();
        let (frames, padded, data) = mel.compute(&vec![0.0f32; 1600], 0).unwrap();
        assert_eq!(frames, 11); // 1 + 1600 / 160 (centred STFT)
        assert_eq!(padded, 32);
        assert_eq!(data.len(), N_MELS * 32);
        // Silence and the padding are the same floor (end-aligned: the padding comes first).
        assert!(data.iter().all(|&v| (v - MEL_CLAMP.ln()).abs() < 1e-4));
        // A 1 kHz tone peaks in the band containing 1 kHz; its frames end the padded mel, the
        // padding before them is the silence floor.
        let tone: Vec<f32> = (0..3200)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SR as f32).sin())
            .collect();
        let (frames, padded, data) = mel.compute(&tone, 0).unwrap();
        assert_eq!((frames, padded), (21, 32));
        let lead = padded - frames;
        for band in 0..N_MELS {
            assert!(
                data[band * padded..band * padded + lead]
                    .iter()
                    .all(|&v| v == MEL_CLAMP.ln()),
                "band {band}"
            );
        }
        let t = lead + frames / 2;
        let best = (0..N_MELS)
            .max_by(|&a, &b| {
                data[a * padded + t]
                    .partial_cmp(&data[b * padded + t])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let mel_lo = hz_to_mel_htk(MEL_FMIN);
        let mel_hi = hz_to_mel_htk(MEL_FMAX);
        let centre = mel_to_hz_htk(mel_lo + (mel_hi - mel_lo) * (best + 1) as f64 / 129.0);
        assert!((centre - 1000.0).abs() < 40.0, "band {best} centre {centre}");
    }

    #[test]
    fn directml_calls_run_at_the_first_shape() {
        let mut pin = ShapePin::for_ep("dml");
        assert_eq!(pin.size(64), 64);
        assert_eq!(pin.size(32), 64);
        assert_eq!(pin.size(64), 64);
        // A larger call cannot fit: it runs at its own size, the pin stays.
        assert_eq!(pin.size(96), 96);
        assert_eq!(pin.size(40), 64);
        // Aligned pinning (FCPE pads whole 10 ms frames).
        let mut pin = ShapePin::for_ep("dml");
        assert_eq!(pin.size_aligned(7_040, 160), 7_040);
        assert_eq!(pin.size_aligned(3_840, 160), 7_040);
        assert_eq!(pin.size_aligned(3_841, 160), 3_841);
        // Other EPs never pad.
        for ep in ["cpu", "cuda"] {
            let mut pin = ShapePin::for_ep(ep);
            assert_eq!(pin.size(64), 64);
            assert_eq!(pin.size(32), 32);
        }
        // The mel pads to the pinned frame count, end-aligned: the silence floor before the audio.
        let mut mel = LogMel::new();
        let tone: Vec<f32> = (0..3_840).map(|i| (i as f32 * 0.3).sin()).collect();
        let (frames, padded, data) = mel.compute(&tone, 64).unwrap();
        assert_eq!((frames, padded, data.len()), (25, 64, N_MELS * 64));
        for band in 0..N_MELS {
            assert!(
                data[band * 64..band * 64 + 39]
                    .iter()
                    .all(|&v| v == MEL_CLAMP.ln()),
                "band {band}"
            );
        }
        assert!(
            (39..64).all(|t| (0..N_MELS).any(|b| data[b * 64 + t] > MEL_CLAMP.ln() + 1.0)),
            "the audio's frames end the mel"
        );
    }

    /// The fixed-shape plan follows the levers: DirectML with `io_binding`, CUDA with `io_binding`
    /// and `cuda_graphs`; `off` (and the CPU) is the plain session, the legacy path.
    #[test]
    fn the_plan_follows_the_levers() {
        let graphs = GpuLevers {
            cuda_graphs: true,
            ..bound()
        };
        assert!(wants_plan("dml", bound()));
        assert!(wants_plan("dml", graphs));
        assert!(!wants_plan("dml", GpuLevers::default()), "io_binding off: the plain DirectML session");
        assert!(!wants_plan("cuda", bound()));
        assert!(wants_plan("cuda", graphs));
        assert!(!wants_plan("cuda", GpuLevers::default()));
        assert!(!wants_plan("cpu", graphs));
    }

    /// The plan's mel frames, `32 * ceil((1 + n) / 32)`: by default of the whole window (P3's size;
    /// 64 at Low latency, also when the window is unknown), with the VRAM saver of the incremental
    /// calls alone (32 up to 31 frames: blocks up to 110 ms), and the sweep's override (a multiple of
    /// 32; anything else is ignored).
    #[test]
    fn plan_frames_follow_the_window_or_the_saver() {
        let load = |calls: Option<usize>, window: Option<usize>, saver: bool| RmvpeLoad {
            expected_call_frames: calls,
            max_context_16k: window.map(|w| w * HOP),
            vram_saver: saver,
            ..RmvpeLoad::default()
        };
        assert_eq!(plan_frames_with(&RmvpeLoad::default(), None), 64);
        for (calls, window, want) in [
            (24usize, Some(44usize), 64usize),
            (24, None, 64),
            (24, Some(20), 32),
            (24, Some(70), 96),
            (40, Some(30), 64),
            (500, Some(500), 256),
        ] {
            assert_eq!(plan_frames_with(&load(Some(calls), window, false), None), want, "{calls} / {window:?}");
        }
        for (n, want) in [(1usize, 32usize), (24, 32), (30, 32), (31, 32), (32, 64), (63, 64), (64, 96), (500, 256)] {
            assert_eq!(plan_frames_with(&load(Some(n), Some(1_000), true), None), want, "saver, {n} frames");
        }
        assert_eq!(plan_frames_with(&load(None, None, true), None), 32);
        assert_eq!(plan_frames_with(&load(Some(24), None, false), Some("32")), 32);
        assert_eq!(plan_frames_with(&load(Some(24), None, true), Some(" 96 ")), 96);
        for bad in ["48", "0", "512", "x", ""] {
            assert_eq!(plan_frames_with(&load(Some(24), None, true), Some(bad)), 32, "{bad:?}");
            assert_eq!(plan_frames_with(&load(Some(24), None, false), Some(bad)), 64, "{bad:?}");
        }
    }

    /// Which graphs a plan tries, best first, and when fp16 runs at all.
    #[test]
    fn graph_attempts_and_the_fp16_gate() {
        use RmvpeGraph::{Full, T32};
        assert_eq!(graph_attempts(32, false, true), vec![(T32, false), (Full, false)]);
        assert_eq!(graph_attempts(32, true, true), vec![(T32, true), (T32, false), (Full, false)]);
        assert_eq!(graph_attempts(64, true, true), vec![(Full, true), (Full, false)]);
        assert_eq!(graph_attempts(32, false, false), vec![(Full, false)]);
        assert_eq!(graph_attempts(32, true, false), vec![(Full, true), (Full, false)]);
        // fp16 only on GPU rungs, under an explicit fp16 precision where the gate passed (auto
        // follows FP16_UNDER_AUTO, there too), and the environment override wins.
        assert!(fp16_wanted(PrecisionMode::Fp16, "dml", 32, None));
        assert!(fp16_wanted(PrecisionMode::Fp16, "cuda", 32, None));
        assert!(fp16_wanted(PrecisionMode::Fp16, "cuda", 64, None));
        assert!(!fp16_wanted(PrecisionMode::Fp32, "cuda", 64, None));
        assert_eq!(fp16_wanted(PrecisionMode::Auto, "cuda", 64, None), FP16_UNDER_AUTO);
        assert!(!fp16_wanted(PrecisionMode::Fp16, "cpu", 32, Some(true)));
        assert!(fp16_wanted(PrecisionMode::Fp32, "dml", 64, Some(true)));
        assert!(!fp16_wanted(PrecisionMode::Fp16, "dml", 32, Some(false)));
    }

    /// The voice's fp16 precision does not reach RMVPE where its accuracy gate failed (DirectML at
    /// the default 64-frame plan) or was not measured (96 frames and more), and every request that
    /// does not run says why in a load note; a plan that runs fp16 has its own (in `adopt`).
    #[test]
    fn the_fp16_gate_decides_and_the_notes_say_so() {
        let v = |p, rung: &str, frames, env, trusted| fp16_verdict(p, rung, frames, env, trusted);
        // Passed: runs, no refusal.
        for (rung, frames) in [("cuda", 32usize), ("cuda", 64), ("dml", 32)] {
            assert_eq!(v(PrecisionMode::Fp16, rung, frames, None, true), Fp16Verdict { run: true, refusal: None }, "{rung} {frames}");
        }
        // Failed / unmeasured: fp32 with a note that names the plan and the override.
        for (rung, frames) in [("dml", 64usize), ("dml", 96), ("cuda", 96), ("cuda", 256)] {
            let got = v(PrecisionMode::Fp16, rung, frames, None, true);
            assert!(!got.run, "{rung} {frames}");
            let note = got.refusal.expect("a refused request is noted");
            assert!(note.contains(&format!("{frames}-frame plan on {rung}")) && note.contains("SIGIL_RMVPE_FP16=1"), "{note}");
        }
        // The environment forces it past the gate, and is never refused for it.
        assert_eq!(v(PrecisionMode::Fp32, "dml", 64, Some(true), true), Fp16Verdict { run: true, refusal: None });
        // A file the model store does not vouch for: the templates are not used, and it is said.
        for env in [None, Some(true)] {
            let got = v(PrecisionMode::Fp16, "cuda", 32, env, false);
            assert!(!got.run && got.refusal.as_deref().is_some_and(|n| n.contains("not the model store's verified")), "{got:?}");
        }
        // Nothing asked or nothing to do: silent (fp32 precision, the CPU, the kill switch, auto while
        // it is off).
        for got in [
            v(PrecisionMode::Fp32, "cuda", 32, None, true),
            v(PrecisionMode::Fp16, "cpu", 32, None, true),
            v(PrecisionMode::Fp16, "cuda", 32, Some(false), false),
            v(PrecisionMode::Fp32, "cuda", 32, None, false),
        ] {
            assert_eq!(got, Fp16Verdict { run: false, refusal: None });
        }
        if !FP16_UNDER_AUTO {
            assert_eq!(v(PrecisionMode::Auto, "cuda", 32, None, true), Fp16Verdict { run: false, refusal: None });
        }
    }

    /// Long calls are split into sub-calls of at most the plan's length: every frame is kept
    /// exactly once, in order; each sub-call after the first discards a 12-frame lead, each one
    /// before the last keeps at least 4 frames of right context, and no more sub-calls run than
    /// those rules need.
    #[test]
    fn split_calls_cover_every_frame_once_with_lead_and_right_context() {
        for w in [32usize, 64] {
            for total in 1..=400usize {
                let subs: Vec<SubCall> = SplitCalls::new(total, w).collect();
                assert!(!subs.is_empty());
                assert_eq!(subs[0].keep_from, 0, "w {w} total {total}");
                assert_eq!(subs.last().unwrap().keep_to, total);
                assert!(subs.last().unwrap().last);
                for (j, s) in subs.iter().enumerate() {
                    let end = if s.last { total } else { s.start + w };
                    assert!(end - s.start <= w, "w {w} total {total}: sub-call {j} runs {} frames", end - s.start);
                    assert!(s.start <= s.keep_from && s.keep_from < s.keep_to && s.keep_to <= end, "{s:?}");
                    if j > 0 {
                        assert_eq!(s.keep_from - s.start, SPLIT_LEAD_FRAMES, "w {w} total {total}: {s:?}");
                        assert_eq!(subs[j - 1].keep_to, s.keep_from);
                    }
                    if !s.last {
                        assert!(end - s.keep_to >= SPLIT_MIN_RIGHT_FRAMES, "w {w} total {total}: {s:?}");
                        assert_eq!(end - s.start, w, "only the last sub-call may be shorter");
                    }
                }
                let want = if total <= w {
                    1
                } else {
                    1 + (total - w).div_ceil(w - SPLIT_LEAD_FRAMES - SPLIT_MIN_RIGHT_FRAMES)
                };
                assert_eq!(subs.len(), want, "w {w} total {total}");
            }
        }
        // The engine's whole Low-latency window (44 audio frames, 45 mel frames) is two runs.
        assert_eq!(SplitCalls::new(45, 32).count(), 2);
    }

    /// A stand-in waveform RMVPE (w-okada style, `[1, N]` in, F0 `[1, N / 160]` out): each 10 ms
    /// frame's F0 is `50 + 1000 * |mean|` of its samples, so frames are independent and the zero
    /// front padding's frames read exactly 50 Hz.
    fn fake_waveform_rmvpe_file(tag: &str) -> PathBuf {
        use crate::services::neural_audio::test_onnx::{Attr, Dim as TDim, Graph, FLOAT};
        let mut g = Graph::new();
        g.input("waveform", FLOAT, &[TDim::Fixed(1), TDim::Named("n")])
            .init_i64("shape", &[3], &[1, -1, HOP as i64])
            .init_f32("scale", &[], &[1000.0])
            .init_f32("offset", &[], &[50.0])
            .node("Reshape", &["waveform", "shape"], &["frames"], &[])
            .node(
                "ReduceMean",
                &["frames"],
                &["mean"],
                &[Attr::Ints("axes", vec![2]), Attr::Int("keepdims", 0)],
            )
            .node("Abs", &["mean"], &["level"], &[])
            .node("Mul", &["level", "scale"], &["scaled"], &[])
            .node("Add", &["scaled", "offset"], &["f0"], &[])
            .output("f0", FLOAT, &[TDim::Fixed(1), TDim::Named("t")]);
        temp_model(&format!("wave-{tag}"), &g.model_bytes(17))
    }

    /// Waveform exports pin their shape on DirectML too (the incremental F0 cache's slices are
    /// shorter than its first, whole-window call): a shorter call runs front-padded to the pinned
    /// length and returns exactly the frames of the unpadded call. They always run the plain
    /// session, whatever the levers.
    #[test]
    fn waveform_exports_pin_their_shape_on_directml() {
        let path = fake_waveform_rmvpe_file("pin");
        let p = path.to_string_lossy().to_string();
        let opts = VcSessionOptions::default();
        let x: Vec<f32> = (0..16_000)
            .map(|i| 0.2 * ((i as f32) * 0.05).sin() + 0.1 * ((i / 700) as f32 * 0.3).cos())
            .collect();
        let mut plain = RmvpePitchExtractor::load(&p, "cpu", &opts).unwrap();
        assert!(matches!(
            plain.plain.as_ref().map(|s| s.inputs.main),
            Some(MainInput::Waveform { rank: 2 })
        ));
        assert_eq!(
            (plain.preferred_input_frames(), plain.max_call_frames(), plain.last_salience()),
            (None, None, None)
        );
        let forced = planned_load(&path, private(), Hooks::default());
        assert!(!forced.planned(), "a waveform export never runs a mel plan");
        let mut pinned = RmvpePitchExtractor::load(&p, "cpu", &opts).unwrap();
        pinned.pin = ShapePin::for_ep("dml");
        // The first call (the whole window) pins 7 040 samples; the slices that follow are padded.
        for (k, len) in [7_040usize, 3_840, 3_840, 7_040, 4_480]
            .into_iter()
            .enumerate()
        {
            let slice = &x[k * 640..k * 640 + len];
            let want = plain.f0(slice).unwrap();
            let got = pinned.f0(slice).unwrap();
            assert_eq!(got.len(), len / HOP, "call {k}");
            assert_eq!(got.len(), want.len(), "call {k}");
            // The same frames (ONNX Runtime's reduction may differ in the last bit between the
            // padded and the unpadded tensor).
            for (j, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() <= 1e-4 * w.abs(), "call {k} ({len} samples) frame {j}: {g} vs {w}");
            }
            assert_eq!(pinned.last_confidence(), plain.last_confidence(), "call {k}");
            assert_eq!(pinned.last_confidence().len(), got.len(), "call {k}");
            if len < 7_040 {
                assert_eq!(pinned.wave_buf.len(), 7_040, "call {k} ran at the pinned length");
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    /// A stand-in log-mel RMVPE: `output = sigmoid(transpose(input) · W)` (`[1, 128, time]` in,
    /// `[1, time, 360]` out; frames are independent, so padding never changes a frame).
    fn fake_rmvpe_file(tag: &str) -> PathBuf {
        use crate::services::neural_audio::test_onnx::{Attr, Dim as TDim, Graph, FLOAT};
        let w: Vec<f32> = (0..N_MELS * N_BINS)
            .map(|i| {
                let (m, b) = (i / N_BINS, i % N_BINS);
                if b == (3 * m + 40) % N_BINS {
                    0.05
                } else {
                    -0.002 * ((b % 7) as f32)
                }
            })
            .collect();
        let mut g = Graph::new();
        g.input("input", FLOAT, &[TDim::Fixed(1), TDim::Fixed(N_MELS as u64), TDim::Named("time")])
            .init_f32("w", &[N_MELS as u64, N_BINS as u64], &w)
            .node("Transpose", &["input"], &["t"], &[Attr::Ints("perm", vec![0, 2, 1])])
            .node("MatMul", &["t", "w"], &["logits"], &[])
            .node("Sigmoid", &["logits"], &["output"], &[])
            .output("output", FLOAT, &[TDim::Fixed(1), TDim::Named("time"), TDim::Fixed(N_BINS as u64)]);
        temp_model(&format!("mel-{tag}"), &g.model_bytes(17))
    }

    fn assert_close(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: lengths");
        for (j, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= 1e-5 * w.abs().max(1.0), "{what}: frame {j}: {g} vs {w}");
        }
    }

    /// The graph's contract comes from the file: RVC's `[1, 128, time]` mel export, with its
    /// output and the `time` dim to override; a waveform export is not a mel contract.
    #[test]
    fn the_mel_contract_is_read_from_the_file() {
        let mel = fake_rmvpe_file("contract");
        let info = onnx_proto::read_graph_info(&mel).unwrap();
        assert_eq!(
            MelContract::of(&info),
            Some(MelContract {
                input: "input".into(),
                output: "output".into(),
                time_dim: Some("time".into()),
                fixed_frames: None,
            })
        );
        assert!(foreign_graph(&info, &mel).is_none());
        let wave = fake_waveform_rmvpe_file("contract");
        assert_eq!(MelContract::of(&onnx_proto::read_graph_info(&wave).unwrap()), None);
        let _ = std::fs::remove_file(&mel);
        let _ = std::fs::remove_file(&wave);
    }

    /// Another kind of graph set as the RMVPE model is named for what it is from its protobuf,
    /// before any session is built.
    #[test]
    fn a_foreign_graph_is_refused_from_the_file() {
        use crate::services::neural_audio::test_onnx::{Dim as TDim, Graph, FLOAT};
        let mut g = Graph::new();
        g.input("source", FLOAT, &[TDim::Fixed(1), TDim::Named("n")])
            .init_i64("axes", &[1], &[2])
            .init_f32("w", &[1, 768], &vec![0.01; 768])
            .node("Unsqueeze", &["source", "axes"], &["u"], &[])
            .node("MatMul", &["u", "w"], &["feats"], &[])
            .output("feats", FLOAT, &[TDim::Fixed(1), TDim::Named("t"), TDim::Fixed(768)]);
        let path = temp_model("foreign", &g.model_bytes(17));
        let err = RmvpePitchExtractor::load(&path.to_string_lossy(), "cpu", &VcSessionOptions::default())
            .err()
            .unwrap();
        assert!(err.contains("content encoder"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    /// The plan is built at load (no plain session at all), sized for the whole window (or, with
    /// the VRAM saver, the incremental calls), and gives the plain session's contour, confidence and
    /// salience at every call length that fits.
    #[test]
    fn the_plan_is_built_at_load_without_a_plain_session() {
        let path = fake_rmvpe_file("load");
        let x = test_audio(20_000);
        let mut plain = RmvpePitchExtractor::load(&path.to_string_lossy(), "cpu", &VcSessionOptions::default()).unwrap();
        assert!(!plain.planned(), "no plan on the CPU rung without asking");
        let default = planned_load(&path, private(), Hooks::default());
        assert_eq!(default.plan_shape(), Some((64, RmvpeGraph::Full, false)));
        assert_eq!((default.preferred_input_frames(), default.max_call_frames()), (Some(63), Some(63)));
        drop(default);
        let mut planned = planned_load(&path, saver(), Hooks::default());
        assert!(planned.planned());
        assert!(planned.plain.is_none(), "the plan replaced the plain session at load");
        assert_eq!(planned.plan_shape(), Some((32, RmvpeGraph::Full, false)));
        assert_eq!((planned.preferred_input_frames(), planned.max_call_frames()), (Some(31), Some(31)));
        assert_eq!(plain.preferred_input_frames(), Some(31), "a plain mel session pads to 32 anyway");
        assert_eq!(plain.max_call_frames(), None);
        assert!(planned.f0(&[]).unwrap().is_empty());
        for (k, frames) in [24usize, 31, 1, 0, 17].into_iter().enumerate() {
            let len = frames * HOP + 37 * (k % 2);
            let slice = &x[k * 640..k * 640 + len];
            let want = plain.f0(slice).unwrap();
            let got = planned.f0(slice).unwrap();
            assert_close(&got, &want, &format!("{frames} frames"));
            assert_close(planned.last_confidence(), plain.last_confidence(), "confidence");
            let (sal, bins) = planned.last_salience().unwrap();
            assert_eq!((bins, sal.len()), (N_BINS, got.len() * N_BINS));
            assert_close(sal, plain.last_salience().unwrap().0, "salience");
        }
        assert!(planned.plain.is_none(), "every call ran the plan");
        assert_eq!(planned.stats().plan_runs, 5);
        assert_eq!(planned.stats().plain_runs, 0);
        // With the saver, expected calls longer than 31 frames size a 64-frame plan.
        let wide = planned_load(
            &path,
            RmvpeLoad {
                expected_call_frames: Some(40),
                ..saver()
            },
            Hooks::default(),
        );
        assert_eq!(wide.plan_shape(), Some((64, RmvpeGraph::Full, false)));
        let _ = std::fs::remove_file(&path);
    }

    /// A call longer than the plan (with the VRAM saver: an onset after silence, the first tick,
    /// the whole window) is split, never grown: the plan keeps its 32 frames, and the contour equals
    /// the whole call's (the stand-in's frames are independent, and every kept frame is far enough
    /// from its sub-call's edges for the mel to match).
    #[test]
    fn long_calls_are_split_never_grown() {
        let path = fake_rmvpe_file("split");
        let x = test_audio(40_000);
        let mut plain = RmvpePitchExtractor::load(&path.to_string_lossy(), "cpu", &VcSessionOptions::default()).unwrap();
        let mut planned = planned_load(&path, saver(), Hooks::default());
        let mut runs = 0u64;
        for (k, frames) in [44usize, 32, 64, 100, 200, 24].into_iter().enumerate() {
            let len = frames * HOP + 59 * (k % 2);
            let slice = &x[k * 1_000..k * 1_000 + len];
            let want = plain.f0(slice).unwrap();
            let got = planned.f0(slice).unwrap();
            assert_close(&got, &want, &format!("{frames} frames"));
            assert_close(planned.last_confidence(), plain.last_confidence(), "confidence");
            assert_eq!(planned.last_salience().unwrap().0.len(), got.len() * N_BINS);
            runs += SplitCalls::new(1 + len / HOP, 32).count() as u64;
            assert_eq!(planned.plan_shape(), Some((32, RmvpeGraph::Full, false)), "the plan never grows");
        }
        let stats = planned.stats();
        assert_eq!(stats.plan_runs, runs);
        assert_eq!(stats.split_calls, 5, "44, 32, 64, 100 and 200 frames are longer than 31");
        assert!(planned.plain.is_none());
        let _ = std::fs::remove_file(&path);
    }

    /// A plan that fails at run time hands the voice to the plain session (built then), with the
    /// same results; the voice's hold on the plan is released.
    #[test]
    fn a_failing_plan_falls_back_to_the_plain_session() {
        let path = fake_rmvpe_file("fallback");
        let x = test_audio(12_000);
        let mut plain = RmvpePitchExtractor::load(&path.to_string_lossy(), "cpu", &VcSessionOptions::default()).unwrap();
        let mut broken = planned_load(&path, private(), Hooks::default());
        // Break the plan: its input no longer exists.
        broken
            .planned
            .as_ref()
            .unwrap()
            .shared
            .core
            .lock()
            .unwrap()
            .input = "no-such-input".into();
        let want = plain.f0(&x[..7_040]).unwrap();
        assert_close(&broken.f0(&x[..7_040]).unwrap(), &want, "fallback");
        assert!(!broken.planned() && broken.plain.is_some());
        assert_eq!(broken.max_call_frames(), None);
        assert_close(
            &broken.f0(&x[640..640 + 3_840]).unwrap(),
            &plain.f0(&x[640..640 + 3_840]).unwrap(),
            "after",
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Voices with the same plan configuration share one plan: built once (the second voice joins
    /// it), counted per voice in `shared_components`, freed with its last voice; another
    /// configuration (mel length) gets its own. Each voice keeps its own contour and confidence.
    #[test]
    fn voices_share_one_plan_and_count_their_users() {
        let path = fake_rmvpe_file("share");
        let x = test_audio(16_000);
        let mut a = planned_load(&path, RmvpeLoad::default(), Hooks::default());
        let mut b = planned_load(&path, RmvpeLoad::default(), Hooks::default());
        assert!(!a.planned.as_ref().unwrap().joined && b.planned.as_ref().unwrap().joined);
        assert!(Arc::ptr_eq(&a.planned.as_ref().unwrap().shared, &b.planned.as_ref().unwrap().shared));
        let id = a.shared_component_id().unwrap().to_string();
        assert_eq!(b.shared_component_id(), Some(id.as_str()));
        let row = |id: &str| shared_components().into_iter().find(|c| c.id == id);
        let r = row(&id).unwrap();
        assert_eq!(
            (r.users, r.mel_frames, r.graph, r.fp16, r.component.as_str()),
            (2, 64, "full", false, "vc.pitch")
        );
        // Another mel length is another plan.
        let c = planned_load(
            &path,
            RmvpeLoad {
                vram_saver: true,
                ..RmvpeLoad::default()
            },
            Hooks::default(),
        );
        assert!(!c.planned.as_ref().unwrap().joined);
        assert_ne!(c.shared_component_id(), Some(id.as_str()));
        // Each voice's contour and confidence come from its own call (read under one lock).
        let (s1, s2) = (&x[..3_840], &x[5_000..5_000 + 3_840]);
        let fa = a.f0(s1).unwrap();
        let fb = b.f0(s2).unwrap();
        let mut solo = planned_load(&path, private(), Hooks::default());
        assert_close(&fa, &solo.f0(s1).unwrap(), "voice a");
        assert_close(a.last_confidence(), solo.last_confidence(), "voice a confidence");
        assert_close(&fb, &solo.f0(s2).unwrap(), "voice b");
        assert_close(b.last_confidence(), solo.last_confidence(), "voice b confidence");
        assert!(solo.shared_component_id().is_none(), "a private plan is not in the registry");
        drop(a);
        assert_eq!(row(&id).unwrap().users, 1);
        drop(b);
        assert!(row(&id).is_none(), "the plan is freed with its last voice");
        drop(c);
        let _ = std::fs::remove_file(&path);
    }

    /// Only loads that state no sizes grow their plan: a window, the VRAM saver and a valid size
    /// override all fix it.
    #[test]
    fn only_loads_without_sizes_grow() {
        let unknown = RmvpeLoad::default();
        assert!(unknown.sizes_unknown(None));
        assert!(unknown.sizes_unknown(Some("48")), "an invalid override is ignored");
        assert!(!unknown.sizes_unknown(Some("64")));
        assert!(!RmvpeLoad {
            max_context_16k: Some(1_000),
            ..unknown
        }
        .sizes_unknown(None));
        assert!(!RmvpeLoad {
            vram_saver: true,
            ..unknown
        }
        .sizes_unknown(None));
        // The call size alone says nothing about the whole window.
        assert!(RmvpeLoad {
            expected_call_frames: Some(24),
            ..unknown
        }
        .sizes_unknown(None));
    }

    /// A voice whose call sizes were unknown at load (a legacy voice, which calls with its whole
    /// rolling context) keeps P3's behaviour: the plan grows once to the first call that outgrows
    /// it, built at that call, and the contour is the plain session's, not a split call's. A known
    /// window never grows (the split test above covers the saver); a call past 256 frames grows to
    /// the bound and splits.
    #[test]
    fn unknown_call_sizes_grow_the_plan_once_like_p3() {
        let path = fake_rmvpe_file("grow");
        let x = test_audio(80_000);
        let mut plain = RmvpePitchExtractor::load(&path.to_string_lossy(), "cpu", &VcSessionOptions::default()).unwrap();
        let mut e = planned_load(&path, private(), Hooks::default());
        assert_eq!(e.plan_shape(), Some((64, RmvpeGraph::Full, false)));
        assert!(e.grow.is_some());
        let mut check = |e: &mut RmvpePitchExtractor, frames: usize, at: usize| {
            let slice = &x[at..at + frames * HOP + 59];
            assert_close(&e.f0(slice).unwrap(), &plain.f0(slice).unwrap(), &format!("{frames} frames"));
            assert_close(e.last_confidence(), plain.last_confidence(), "confidence");
        };
        // Calls that fit never grow.
        check(&mut e, 24, 0);
        check(&mut e, 62, 500);
        assert_eq!((e.plan_shape().unwrap().0, e.stats().grows), (64, 0));
        // 100 frames: P3's `32 * ceil(101 / 32)`, built at that call; no split, no more growth.
        check(&mut e, 100, 1_000);
        assert_eq!((e.plan_shape().unwrap().0, e.stats().grows, e.stats().split_calls), (128, 1, 0));
        let before = e.stats().plan_runs;
        check(&mut e, 100, 2_000);
        check(&mut e, 40, 3_000);
        assert_eq!((e.plan_shape().unwrap().0, e.stats().grows, e.stats().plan_runs), (128, 1, before + 2));
        assert!(e.planned.as_ref().unwrap().calls >= WARMUP_CALLS, "a grown plan has no warm-up phase");
        // Past the bound: the largest plan, then splits.
        check(&mut e, 400, 4_000);
        assert_eq!((e.plan_shape().unwrap().0, e.stats().grows), (256, 2));
        check(&mut e, 400, 5_000);
        assert_eq!((e.plan_shape().unwrap().0, e.stats().grows, e.stats().split_calls), (256, 2, 2));
        assert!(e.plain.is_none(), "every call ran a plan");
        // A loader that states the window gets the window's plan, which never grows.
        let mut known = planned_load(
            &path,
            RmvpeLoad {
                max_context_16k: Some(63 * HOP),
                ..private()
            },
            Hooks::default(),
        );
        assert!(known.grow.is_none());
        known.f0(&x[..100 * HOP]).unwrap();
        assert_eq!((known.plan_shape().unwrap().0, known.stats().grows, known.stats().split_calls), (64, 0, 1));
        let _ = std::fs::remove_file(&path);
    }

    /// Growth leaves other voices alone: a plan another voice still holds stays for it (and for this
    /// voice when the growth fails, which is not retried); a plan only this voice held is freed
    /// before the larger one is built.
    #[test]
    fn growing_keeps_a_plan_other_voices_hold() {
        let path = fake_rmvpe_file("grow-shared");
        let x = test_audio(30_000);
        let mut plain = RmvpePitchExtractor::load(&path.to_string_lossy(), "cpu", &VcSessionOptions::default()).unwrap();
        let mut a = planned_load(&path, RmvpeLoad::default(), Hooks::default());
        let mut b = planned_load(&path, RmvpeLoad::default(), Hooks::default());
        let small = a.shared_component_id().unwrap().to_string();
        assert_eq!(b.shared_component_id(), Some(small.as_str()));
        let row = |id: &str| shared_components().into_iter().find(|c| c.id == id);
        assert_eq!(row(&small).unwrap().users, 2);
        let slice = &x[..100 * HOP];
        assert_close(&a.f0(slice).unwrap(), &plain.f0(slice).unwrap(), "grown");
        assert_eq!(a.plan_shape().unwrap().0, 128);
        assert_ne!(a.shared_component_id(), Some(small.as_str()));
        assert_eq!(row(&small).unwrap().users, 1, "b still holds the 64-frame plan");
        assert_eq!(b.plan_shape().unwrap().0, 64);
        // b grows to the plan a built: it joins it, and the 64-frame plan is freed.
        assert_close(&b.f0(slice).unwrap(), &plain.f0(slice).unwrap(), "joined");
        assert_eq!(b.shared_component_id(), a.shared_component_id());
        assert!(row(&small).is_none(), "freed with its last voice");
        let big = a.shared_component_id().unwrap().to_string();
        assert_eq!(row(&big).unwrap().users, 2);
        drop((a, b));
        assert!(row(&big).is_none());

        // A growth that cannot be built (the file is gone) keeps the shared plan: the call splits,
        // the plan stays, and nothing is retried.
        let path2 = fake_rmvpe_file("grow-fail");
        let mut c = planned_load(&path2, RmvpeLoad::default(), Hooks::default());
        let d = planned_load(&path2, RmvpeLoad::default(), Hooks::default());
        std::fs::remove_file(&path2).unwrap();
        let want = plain.f0(&x[..100 * HOP]).unwrap();
        assert_close(&c.f0(&x[..100 * HOP]).unwrap(), &want, "split after a failed growth");
        assert_eq!((c.plan_shape().unwrap().0, c.stats().grows, c.stats().split_calls), (64, 0, 1));
        assert!(c.grow.is_none(), "not retried");
        assert!(
            c.load_notes()
                .iter()
                .any(|n| n.contains("128-frame") && n.contains("could not be built")),
            "{:?}",
            c.load_notes()
        );
        assert_eq!(d.plan_shape().unwrap().0, 64);
        let _ = std::fs::remove_file(&path);
    }

    /// The rung a plan would land on comes from the ladder's preflights, in ladder order with the
    /// rung's own device, the CPU last; the levers then decide whether that rung runs a plan at all
    /// (so a GPU request that falls to the CPU builds the model once, as the plain session).
    #[test]
    fn the_plan_waits_for_a_rung_that_runs_it() {
        let seen = std::cell::RefCell::new(Vec::new());
        let device = |rung: &'static str| if rung == "dml" { 7 } else { 1 };
        let ladder = ep_ladder("cuda");
        assert_eq!(predicted_rung_with("cuda", device, |_, _| true), ladder[0]);
        assert_eq!(predicted_rung_with("cuda", device, |_, _| false), "cpu");
        assert_eq!(predicted_rung_with("dml", device, |_, _| false), "cpu");
        assert_eq!(predicted_rung_with("cpu", device, |_, _| false), "cpu");
        let landed = predicted_rung_with("cuda", device, |rung, dev| {
            seen.borrow_mut().push((rung, dev));
            rung != "cuda"
        });
        assert_eq!(landed, ladder.get(1).copied().unwrap_or("cpu"));
        assert_eq!(seen.borrow()[0], ("cuda", 1));
        if landed == "dml" {
            assert_eq!(seen.borrow()[1], ("dml", 7), "the fallback rung is checked on its own device");
        }
        // The levers decide per rung: DirectML runs the plan with io_binding, the CPU never.
        let graphs = GpuLevers {
            cuda_graphs: true,
            ..bound()
        };
        assert!(wants_plan("dml", bound()) && !wants_plan("cpu", graphs));
        assert!(
            !wants_plan(predicted_rung_with("dml", device, |_, _| false), graphs),
            "a DirectML request that falls to the CPU builds no plan"
        );
        // The real check answers for the CPU without touching a GPU.
        assert_eq!(predicted_rung("cpu", &VcSessionOptions::default()), "cpu");
    }

    /// The model store's record decides which file the graph templates may use: pinned, listing
    /// this file (by canonical path) with the catalog's sha256 and its size, and not modified after
    /// the install (the record is written after the file, so a little later is fine).
    #[test]
    fn the_store_record_decides_which_file_the_templates_may_use() {
        let dir = std::env::temp_dir().join(format!("sigil-p3b-rmvpe-store-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        let path = dir.join("model.onnx");
        std::fs::write(&path, b"an rmvpe stand-in").unwrap();
        let other = dir.join("other.onnx");
        std::fs::write(&other, b"another file").unwrap();
        let size = std::fs::metadata(&path).unwrap().len();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let facts = |pinned: bool, file: &Path, sha: Option<&str>, size: u64, installed_at: u64| InstallFacts {
            pinned,
            files: vec![(file.to_path_buf(), size, sha.map(str::to_string))],
            installed_at_unix_ms: installed_at,
        };
        let ok = facts(true, &path, Some(RMVPE_SHA256), size, now_ms);
        assert!(store_vouches(&ok, &path).is_ok());
        // The record lists the file by another spelling of the same path.
        let respelled = dir.join("sub").join("..").join("model.onnx");
        assert!(store_vouches(&facts(true, &respelled, Some(RMVPE_SHA256), size, now_ms), &path).is_ok());
        // The record may be written a little after the file (the slack), not much later.
        assert!(store_vouches(
            &facts(true, &path, Some(RMVPE_SHA256), size, now_ms.saturating_sub(STORE_MTIME_SLACK_MS / 2)),
            &path
        )
        .is_ok());
        let reject = |f: InstallFacts, what: &str| store_vouches(&f, &path).expect_err(what);
        let late = now_ms.saturating_sub(10 * STORE_MTIME_SLACK_MS);
        assert!(reject(facts(true, &path, Some(RMVPE_SHA256), size, late), "modified after").contains("modified after"));
        assert!(reject(facts(false, &path, Some(RMVPE_SHA256), size, now_ms), "unpinned").contains("not checked against the catalog hash"));
        assert!(reject(facts(true, &path, Some("00"), size, now_ms), "other sha").contains("not the version"));
        assert!(reject(facts(true, &path, None, size, now_ms), "no sha").contains("not the version"));
        assert!(reject(facts(true, &path, Some(RMVPE_SHA256), size + 1, now_ms), "size").contains("changed size"));
        assert!(reject(facts(true, &other, Some(RMVPE_SHA256), size, now_ms), "other file").contains("not the model store's"));
        let gone = dir.join("gone.onnx");
        assert!(store_vouches(&ok, &gone).is_err(), "a file that does not exist is not vouched for");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two voices that load the same plan at the same time build it once (the second queues on the
    /// key's gate and joins); a build that fails leaves the gate usable, so the voice waiting behind
    /// it builds its own; sharing off builds a private plan.
    #[test]
    fn concurrent_loads_of_one_plan_build_it_once() {
        use std::sync::atomic::AtomicUsize;
        let path = fake_rmvpe_file("gate");
        let info = onnx_proto::read_graph_info(&path).unwrap();
        let contract = MelContract::of(&info).unwrap();
        let opts = VcSessionOptions::default();
        let key = |frames: usize| CoreKey {
            model: file_identity(&path),
            graph: RmvpeGraph::Full,
            fp16: false,
            ep: "cpu",
            device: 0,
            levers: bound(),
            frames,
            prefix: None,
            opts: format!("{opts:?}"),
        };
        let spec = |frames: usize| CoreSpec {
            path: &path,
            source: &info,
            contract: &contract,
            template: None,
            graph: RmvpeGraph::Full,
            fp16: false,
            ep: "cpu",
            opts: &opts,
            levers: bound(),
            frames,
        };
        // Four voices at once, one build.
        let builds = AtomicUsize::new(0);
        let barrier = std::sync::Barrier::new(4);
        let got: Vec<(Arc<SharedCore>, bool)> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..4)
                .map(|_| {
                    s.spawn(|| {
                        barrier.wait();
                        acquire(key(32), true, || {
                            builds.fetch_add(1, Ordering::SeqCst);
                            std::thread::sleep(Duration::from_millis(40));
                            build_core(&spec(32))
                        })
                        .unwrap()
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert_eq!(got.iter().filter(|(_, joined)| !joined).count(), 1, "one builder, three joiners");
        assert!(got.iter().all(|(core, _)| Arc::ptr_eq(core, &got[0].0)));
        assert_eq!(Arc::strong_count(&got[0].0), 4);
        // Sharing off: a plan of its own.
        let (private, joined) = acquire(key(32), false, || build_core(&spec(32))).unwrap();
        assert!(!joined && !Arc::ptr_eq(&private, &got[0].0));
        drop((got, private));
        // A failed build: the voice queued behind it builds its own.
        let builds = AtomicUsize::new(0);
        let (first, second) = std::thread::scope(|s| {
            let a = s.spawn(|| {
                acquire(key(64), true, || {
                    builds.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(60));
                    Err("the build failed".to_string())
                })
            });
            std::thread::sleep(Duration::from_millis(20));
            let b = s.spawn(|| {
                acquire(key(64), true, || {
                    builds.fetch_add(1, Ordering::SeqCst);
                    build_core(&spec(64))
                })
            });
            (a.join().unwrap(), b.join().unwrap())
        });
        assert_eq!(first.err().as_deref(), Some("the build failed"));
        let (core, joined) = second.unwrap();
        assert!(!joined, "the second voice built the plan itself");
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        // ...and a third voice joins that plan.
        let (again, joined) = acquire(key(64), true, || panic!("must not build")).unwrap();
        assert!(joined && Arc::ptr_eq(&again, &core));
        let _ = std::fs::remove_file(&path);
    }

    /// While the loader holds a warm-up scope, a voice that joined a live plan runs its first calls
    /// in the live voice's shadow, on whatever thread makes them (here: not the thread that built
    /// the extractor, like the load's F0 helper); without a scope, or after the voice's warm-up
    /// budget, every call just takes the lock.
    #[test]
    fn a_joining_voice_warms_up_in_the_live_voices_shadow() {
        let path = fake_rmvpe_file("shadow");
        let x = test_audio(16_000);
        let mut live = planned_load(&path, RmvpeLoad::default(), Hooks::default());
        // The joining voice is built on one thread (a load branch) and called from others.
        let mut joining = std::thread::scope(|s| {
            s.spawn(|| planned_load(&path, RmvpeLoad::default(), Hooks::default()))
                .join()
                .unwrap()
        });
        assert!(joining.planned.as_ref().unwrap().joined);
        let from_another_thread = |e: &mut RmvpePitchExtractor, from: usize| {
            std::thread::scope(|s| s.spawn(|| e.f0(&x[from..from + 3_840]).unwrap()).join().unwrap())
        };
        // No scope: nothing waits, although the live voice has called since the joiner's load.
        live.f0(&x[..3_840]).unwrap();
        assert!(!joining.planned.as_ref().unwrap().in_warmup());
        from_another_thread(&mut joining, 0);
        assert_eq!(joining.stats().shadow_waits, 0);
        // The loader opens a scope; the joiner's calls come from yet another thread. The live voice
        // is idle since the joiner's last call: the warm-up runs at once.
        let scope = warmup_scope();
        assert!(joining.planned.as_ref().unwrap().in_warmup());
        from_another_thread(&mut joining, 640);
        assert_eq!(joining.stats().shadow_waits, 0);
        // The live voice calls (on its own thread) after the joiner's last call: the next warm-up
        // waits for one more of its calls to finish, then runs.
        let stop = std::sync::atomic::AtomicBool::new(false);
        let shared = joining.planned.as_ref().unwrap().shared.clone();
        let seen = joining.planned.as_ref().unwrap().seen;
        std::thread::scope(|s| {
            let worker = s.spawn(|| {
                let mut n = 0;
                while !stop.load(Ordering::Relaxed) {
                    live.f0(&x[..3_840]).unwrap();
                    n += 1;
                    std::thread::sleep(Duration::from_millis(5));
                }
                n
            });
            while shared.done.load(Ordering::Acquire) == seen {
                std::thread::sleep(Duration::from_millis(1));
            }
            from_another_thread(&mut joining, 1_280);
            stop.store(true, Ordering::Relaxed);
            assert!(worker.join().unwrap() > 0);
        });
        assert_eq!(joining.stats().shadow_waits, 1);
        // After its warm-up budget the voice is live, scope or not.
        joining.planned.as_mut().unwrap().calls = WARMUP_CALLS;
        assert!(!joining.planned.as_ref().unwrap().in_warmup());
        // And a voice in its first calls is live once the scope is closed.
        joining.planned.as_mut().unwrap().calls = 0;
        assert!(joining.planned.as_ref().unwrap().in_warmup());
        drop(scope);
        assert!(!joining.planned.as_ref().unwrap().in_warmup());
        let _ = std::fs::remove_file(&path);
    }

    /// The tick path through the plan makes no heap allocation once warm (the mel, the locked
    /// run, the row copy and the decode reuse the extractor's buffers), on the default plan and on
    /// the saver's, whose whole-window calls are split into sub-calls. The streaming smoother
    /// (`f0_post::smooth_f0_stream`, P3B-F) is checked separately by its owner, so it is skipped here.
    #[test]
    fn f0_into_is_allocation_free_on_the_plan_path() {
        use crate::services::neural_audio::gpu::alloc_probe::allocations_during;
        let path = fake_rmvpe_file("alloc");
        let x = test_audio(20_000);
        for load in [private(), saver()] {
            let mut e = planned_load(
                &path,
                RmvpeLoad {
                    max_context_16k: Some(63 * HOP),
                    ..load
                },
                Hooks::default(),
            );
            e.skip_smoothing = true;
            let mut out = Vec::with_capacity(128);
            for k in 0..3 {
                e.f0_into(&x[k * 640..k * 640 + 44 * HOP], &mut out)
                    .unwrap();
                e.f0_into(&x[k * 640..k * 640 + 24 * HOP], &mut out)
                    .unwrap();
            }
            for (k, frames) in [24usize, 44, 31, 63].into_iter().enumerate() {
                let slice = &x[2_000 + k * 640..2_000 + k * 640 + frames * HOP];
                let (r, allocs) = allocations_during(|| e.f0_into(slice, &mut out));
                r.unwrap();
                assert_eq!(allocs, 0, "{:?}: {frames}-frame call allocated", e.plan_shape());
                assert_eq!(out.len(), 1 + frames);
            }
            if load.vram_saver {
                assert!(e.stats().split_calls > 0, "the saver's long calls were split");
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    /// A tiny RMVPE-like graph whose middle 3×3 conv sees a one-row map at 32 mel frames (an
    /// AvgPool over 32×32), plus its `t32`-style template: the conv as `[1, 3]` with pads
    /// `[0, 1, 0, 1]`, the kernel and the projection as external data in `rmvpe.weights`.
    fn tiny_rmvpe_pair(tag: &str) -> (PathBuf, &'static [u8]) {
        use crate::services::neural_audio::onnx_proto::test_proto as tp;
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        };
        let k3: Vec<f32> = (0..4 * 3 * 3).map(|_| rnd()).collect();
        let kb: Vec<f32> = (0..4).map(|_| 0.1 * rnd()).collect();
        let proj: Vec<f32> = (0..16 * 32 * N_BINS).map(|_| 0.2 * rnd()).collect();
        let raw = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let ints = |v: &[i64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let nodes = |conv_pads: &[i64], kernel: &[i64]| {
            vec![
                tp::node("Transpose", "/t", &["input"], &["t1"], &[tp::attr_ints("perm", &[0, 2, 1])]),
                tp::node("Reshape", "/r1", &["t1", "shape4"], &["t2"], &[]),
                tp::node(
                    "AveragePool",
                    "/pool",
                    &["t2"],
                    &["t3"],
                    &[tp::attr_ints("kernel_shape", &[32, 32]), tp::attr_ints("strides", &[32, 32])],
                ),
                tp::node(
                    "Conv",
                    "/unet/intermediate/conv",
                    &["t3", "k3", "kb"],
                    &["t4"],
                    &[tp::attr_ints("pads", conv_pads), tp::attr_ints("kernel_shape", kernel)],
                ),
                tp::node("Relu", "/relu", &["t4"], &["t5"], &[]),
                tp::node("Reshape", "/r2", &["t5", "shape2"], &["t6"], &[]),
                tp::node("MatMul", "/proj", &["t6", "proj"], &["t7"], &[]),
                tp::node("Reshape", "/r3", &["t7", "shape3"], &["t8"], &[]),
                tp::node("Sigmoid", "/sig", &["t8"], &["output"], &[]),
            ]
        };
        let small = |name: &str, v: &[i64]| tp::tensor(name, 7, &[v.len() as u64], Some(&ints(v)), None);
        let shapes = vec![
            small("shape4", &[1, 1, -1, N_MELS as i64]),
            small("shape2", &[1, 16]),
            small("shape3", &[1, 32, N_BINS as i64]),
            tp::tensor("kb", 1, &[4], Some(&raw(&kb)), None),
        ];
        let mut full_inits = shapes.clone();
        full_inits.push(tp::tensor("k3", 1, &[4, 1, 3, 3], Some(&raw(&k3)), None));
        full_inits.push(tp::tensor("proj", 1, &[16, (32 * N_BINS) as u64], Some(&raw(&proj)), None));
        let full = tp::model(
            &nodes(&[1, 1, 1, 1], &[3, 3]),
            &full_inits,
            &[tp::value_info("input", 1, &[Ok(1), Ok(N_MELS as u64), Err("time")])],
            &[tp::value_info("output", 1, &[Ok(1), Err("time"), Ok(N_BINS as u64)])],
            &[],
            11,
        );
        let mut t_inits = shapes;
        t_inits.push(tp::tensor("k3", 1, &[4, 1, 1, 3], None, Some((TEMPLATE_WEIGHTS, 0, 48))));
        t_inits.push(tp::tensor(
            "proj",
            1,
            &[16, (32 * N_BINS) as u64],
            None,
            Some((TEMPLATE_WEIGHTS, 64, (16 * 32 * N_BINS * 4) as u64)),
        ));
        let template = tp::model(
            &nodes(&[0, 1, 0, 1], &[1, 3]),
            &t_inits,
            &[tp::value_info("input", 1, &[Ok(1), Ok(N_MELS as u64), Ok(32)])],
            &[tp::value_info("output", 1, &[Ok(1), Ok(32), Ok(N_BINS as u64)])],
            &[("sigil.source_sha256", RMVPE_SHA256), ("sigil.weights", TEMPLATE_WEIGHTS)],
            11,
        );
        (temp_model(&format!("tiny-{tag}"), &full), Box::leak(template.into_boxed_slice()))
    }

    /// The t32 pruning is exact: the template filled from the full model's weights (the 3×3
    /// kernel cut to its centre row) gives the full model's salience at 32 mel frames, through the
    /// plan, the split path included; a template not cut from the catalog's model is refused.
    #[test]
    fn t32_slicing_matches_the_full_model_on_cpu() {
        let (path, template) = tiny_rmvpe_pair("t32");
        let x = test_audio(20_000);
        let trusted = Hooks {
            trust_source: true,
            templates: Some((template, template, template)),
            ..Hooks::default()
        };
        let mut t32 = planned_load(&path, saver(), trusted);
        assert_eq!(t32.plan_shape(), Some((32, RmvpeGraph::T32, false)), "{:?}", t32.load_notes());
        let mut full = planned_load(&path, saver(), Hooks::default());
        assert_eq!(
            full.plan_shape(),
            Some((32, RmvpeGraph::Full, false)),
            "an unverified file runs the full model"
        );
        // t32 runs wherever a plan is 32 frames (it is exact there): also a default plan whose whole
        // window fits 31 frames.
        let small = planned_load(
            &path,
            RmvpeLoad {
                max_context_16k: Some(20 * HOP),
                ..private()
            },
            trusted,
        );
        assert_eq!(small.plan_shape(), Some((32, RmvpeGraph::T32, false)));
        drop(small);
        let mut worst = 0.0f32;
        for (k, frames) in [31usize, 24, 44, 5].into_iter().enumerate() {
            let slice = &x[k * 2_000..k * 2_000 + frames * HOP];
            let a = t32.f0(slice).unwrap();
            let b = full.f0(slice).unwrap();
            assert_eq!(a.len(), b.len());
            let (sa, sb) = (t32.last_salience().unwrap().0, full.last_salience().unwrap().0);
            for (p, q) in sa.iter().zip(sb) {
                worst = worst.max((p - q).abs());
            }
            for (p, q) in a.iter().zip(&b) {
                assert_eq!(*p > 0.0, *q > 0.0, "{frames} frames: voicing differs");
                if *p > 0.0 {
                    assert!((1200.0 * (p / q).log2()).abs() < 0.01, "{frames} frames: {p} vs {q}");
                }
            }
        }
        assert!(worst <= 1e-5, "max |salience difference| {worst}");
        // A template that was not cut from the catalog's model is refused (the full model runs).
        let mut wrong = template.to_vec();
        let at = wrong
            .windows(RMVPE_SHA256.len())
            .position(|w| w == RMVPE_SHA256.as_bytes())
            .unwrap();
        wrong[at] = if wrong[at] == b'5' { b'6' } else { b'5' };
        let wrong: &'static [u8] = Box::leak(wrong.into_boxed_slice());
        let e = planned_load(
            &path,
            saver(),
            Hooks {
                trust_source: true,
                templates: Some((wrong, wrong, wrong)),
                ..Hooks::default()
            },
        );
        assert_eq!(e.plan_shape(), Some((32, RmvpeGraph::Full, false)));
        assert!(
            e.load_notes()
                .iter()
                .any(|n| n.contains("t32") && n.contains("not cut from")),
            "{:?}",
            e.load_notes()
        );
        let _ = std::fs::remove_file(&path);
    }

    /// The compiled-in templates are what the export script wrote: weightless (every large tensor
    /// external in `rmvpe.weights`), cut from the catalog's model, t32 fixed at 32 frames, the fp16
    /// ones with fp32 I/O.
    #[test]
    fn the_compiled_in_templates_are_weightless_and_pinned() {
        for (name, bytes, frames) in [
            ("t32", T32_GRAPH, Some(32usize)),
            ("t32_fp16", T32_FP16_GRAPH, Some(32)),
            ("fp16", FP16_GRAPH, None),
        ] {
            let info = onnx_proto::read_graph_info_bytes(bytes).unwrap();
            assert_eq!(info.metadata("sigil.source_sha256"), Some(RMVPE_SHA256), "{name}");
            assert_eq!(info.metadata("sigil.weights"), Some(TEMPLATE_WEIGHTS), "{name}");
            assert_eq!(info.metadata("sigil.variant"), Some(name));
            let c = MelContract::of(&info).unwrap();
            assert_eq!(c.fixed_frames, frames, "{name}");
            assert!(
                info.initializers
                    .iter()
                    .all(|i| i.external || i.raw_data.is_none_or(|(_, len)| len < 1024)),
                "{name}"
            );
            let external = info.initializers.iter().filter(|i| i.external).count();
            assert!(external > 150, "{name}: {external} external tensors");
            assert!(bytes.len() < 1 << 20, "{name} is {} bytes", bytes.len());
        }
    }

    #[test]
    fn reflect_padding_matches_torch() {
        // torch reflect: [x2, x1 | x0 x1 x2 x3 | x2 x1]
        let n = 4;
        let idx: Vec<usize> = (-2..6).map(|i| reflect_index(i, n)).collect();
        assert_eq!(idx, vec![2, 1, 0, 1, 2, 3, 2, 1]);
    }

    // -----------------------------------------------------------------------------------------
    // Dev rig (real model; `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache`, GPU timing lock)
    // -----------------------------------------------------------------------------------------

    /// The real `rmvpe.onnx` and whether it is the catalog's (hashed here, once: production trusts
    /// the model store's install record instead).
    fn real_model() -> Option<(PathBuf, bool)> {
        use sha2::{Digest, Sha256};
        use std::io::Read;
        let path = super::super::f0_fusion::test_model("pitch-rmvpe")?;
        let mut f = std::fs::File::open(&path).ok()?;
        let mut h = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = f.read(&mut buf).ok()?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        let sha: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        Some((path, sha == RMVPE_SHA256))
    }

    /// Real 16 kHz speech clips of the dev cache (`sherpa-onnx/clips`).
    fn real_clips() -> Vec<(String, Vec<f32>)> {
        let Some(root) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from) else {
            return Vec::new();
        };
        let dir = root.join("sherpa-onnx").join("clips");
        let mut names: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect())
            .unwrap_or_default();
        names.sort();
        names
            .into_iter()
            .filter(|p| p.extension().is_some_and(|e| e == "wav"))
            .filter_map(|p| {
                let mut r = hound::WavReader::open(&p).ok()?;
                let spec = r.spec();
                if (spec.sample_rate, spec.channels) != (16_000, 1) {
                    return None;
                }
                let x: Vec<f32> = match spec.sample_format {
                    hound::SampleFormat::Float => r.samples::<f32>().map(|v| v.unwrap_or(0.0)).collect(),
                    hound::SampleFormat::Int => r
                        .samples::<i32>()
                        .map(|v| v.unwrap_or(0) as f32 / (1u64 << (spec.bits_per_sample - 1)) as f32)
                        .collect(),
                };
                Some((p.file_stem()?.to_string_lossy().into_owned(), x))
            })
            .collect()
    }

    /// The levers a dev-rig plan runs with on `ep` (the production `auto` table).
    fn rig_levers(ep: &str) -> GpuLevers {
        GpuLevers {
            io_binding: true,
            cuda_graphs: ep == "cuda",
            tf32: ep == "cuda",
            ..GpuLevers::default()
        }
    }

    fn rig_load(path: &Path, ep: &str, frames: usize, graph: RmvpeGraph, fp16: bool) -> Result<RmvpePitchExtractor, String> {
        let trusted = graph == RmvpeGraph::T32 || fp16;
        let load = RmvpeLoad {
            expected_call_frames: Some(24.min(frames - 1)),
            max_context_16k: Some((frames - 1) * HOP),
            precision: if fp16 { PrecisionMode::Fp16 } else { PrecisionMode::Fp32 },
            shared: false,
            vram_saver: frames == PLAN_ALIGN,
        };
        let mut e = RmvpePitchExtractor::load_inner(
            &path.to_string_lossy(),
            ep,
            &VcSessionOptions {
                intra_threads: Some(2),
                ..VcSessionOptions::default()
            },
            rig_levers(ep),
            &load,
            Hooks {
                trust_source: trusted,
                // The rig measures fp16 wherever it is asked for, gate or not.
                fp16_env: fp16.then_some(true),
                ..Hooks::default()
            },
        )?;
        if e.plan_shape().map(|(_, g, h)| (g, h)) != Some((graph, fp16)) {
            return Err(format!(
                "{ep}: wanted {graph:?} fp16 {fp16}, got {:?} ({:?})",
                e.plan_shape(),
                e.load_notes()
            ));
        }
        e.stats = RmvpeStats::default();
        Ok(e)
    }

    /// Verification (2) of P3B-E: RMVPE per-call cost of the 64-frame plan (the P3 / exe-s0 size,
    /// full model) against the 32-frame plan (full and t32, and fp16), per EP, on the F0 cache's
    /// 24-frame incremental call. Rounds alternate the variant order (ABBA), and the per-round
    /// medians give paired ratios against the 64-frame plan.
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache cargo test rmvpe_plan_cost -- --ignored --nocapture`
    #[test]
    #[ignore = "dev rig: needs SIGIL_TEST_MODELS_DIR with the RMVPE model and a GPU (hold the GPU timing lock)"]
    fn rmvpe_plan_cost() {
        let Some((path, catalog)) = real_model() else {
            eprintln!("RMVPE model not found; skipped");
            return;
        };
        assert!(catalog, "{} is not the catalog's rmvpe.onnx", path.display());
        let x = super::super::f0_fusion::ab::harmonic_tone(180.0, 32_000, 0.3);
        let rounds: usize = std::env::var("SIGIL_RMVPE_COST_ROUNDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6);
        let eps: Vec<String> = std::env::var("SIGIL_RMVPE_COST_EPS")
            .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_else(|_| vec!["cuda".into(), "dml".into()]);
        for ep in eps.iter().map(String::as_str) {
            let variants = [
                ("64 full", 64usize, RmvpeGraph::Full, false),
                ("32 full", 32, RmvpeGraph::Full, false),
                ("32 t32", 32, RmvpeGraph::T32, false),
                ("32 t32 fp16", 32, RmvpeGraph::T32, true),
            ];
            let mut loaded: Vec<(&str, RmvpePitchExtractor)> = Vec::new();
            for (label, frames, graph, fp16) in variants {
                let t0 = Instant::now();
                match rig_load(&path, ep, frames, graph, fp16) {
                    Ok(e) => {
                        println!(
                            "rmvpe {ep} {label}: load {:.0} ms, outcome {}, vram {:?} MB, notes {:?}",
                            t0.elapsed().as_secs_f64() * 1000.0,
                            e.outcome().active,
                            e.planned
                                .as_ref()
                                .and_then(|p| p.shared.core.lock().ok().and_then(|c| c.plan.vram_mb())),
                            e.load_notes()
                        );
                        loaded.push((label, e));
                    }
                    Err(err) => println!("rmvpe {ep} {label}: {err}"),
                }
            }
            if loaded.is_empty() {
                continue;
            }
            let n = 24 * HOP;
            let mut medians: Vec<Vec<f64>> = vec![Vec::new(); loaded.len()];
            for round in 0..rounds {
                let order: Vec<usize> = if round % 2 == 0 {
                    (0..loaded.len()).collect()
                } else {
                    (0..loaded.len()).rev().collect()
                };
                for i in order {
                    let e = &mut loaded[i].1;
                    let mut out = Vec::with_capacity(64);
                    for k in 0..10 {
                        e.f0_into(&x[k * 160..k * 160 + n], &mut out).unwrap();
                    }
                    let mut times = Vec::with_capacity(60);
                    for k in 0..60 {
                        let t0 = Instant::now();
                        e.f0_into(&x[k * 160..k * 160 + n], &mut out).unwrap();
                        times.push(t0.elapsed().as_secs_f64() * 1000.0);
                    }
                    times.sort_by(f64::total_cmp);
                    medians[i].push(times[30]);
                }
            }
            for (i, (label, _)) in loaded.iter().enumerate() {
                let mut m = medians[i].clone();
                m.sort_by(f64::total_cmp);
                let mut ratios: Vec<f64> = medians[i]
                    .iter()
                    .zip(&medians[0])
                    .map(|(a, b)| a / b)
                    .collect();
                ratios.sort_by(f64::total_cmp);
                println!(
                    "rmvpe {ep} {label}: per-call p50 over {rounds} rounds {:.3} ms (min {:.3}, max {:.3}); paired ratio vs 64 full: median {:.3} [{:.3}, {:.3}]",
                    m[m.len() / 2],
                    m[0],
                    m[m.len() - 1],
                    ratios[ratios.len() / 2],
                    ratios[0],
                    ratios[ratios.len() - 1]
                );
            }
        }
    }

    /// One stream of `audio` through an F0 cache with the engine's Low-latency geometry (40 ms
    /// blocks, a 44-frame window): each block's frames from the tick that decoded them, plus the
    /// confidence of each call.
    fn stream_f0(e: &mut RmvpePitchExtractor, audio: &[f32]) -> (Vec<f32>, Vec<f32>) {
        stream_f0_lead(e, audio, super::super::engine::f0_cache::DEFAULT_LEAD_FRAMES, 44)
    }

    /// [`stream_f0`] with the F0 cache's lead-in (`lead` frames before the re-extracted ones) and
    /// window (`window` frames) chosen.
    fn stream_f0_lead(e: &mut RmvpePitchExtractor, audio: &[f32], lead: usize, window: usize) -> (Vec<f32>, Vec<f32>) {
        use super::super::engine::f0_cache::{F0Cache, F0CacheConfig, DEFAULT_MARGIN_FRAMES};
        let (block, ctx) = (4 * HOP, window * HOP);
        let mut cache = F0Cache::new(F0CacheConfig {
            margin_frames: DEFAULT_MARGIN_FRAMES,
            capacity_frames: 2 * window,
        })
        .with_lead_frames(lead);
        let mut f0 = vec![0.0f32; audio.len() / HOP];
        let mut conf = Vec::new();
        let mut end = ctx;
        while end <= audio.len() {
            let window = cache
                .update(e, &audio[end - ctx..end], (end - ctx) as u64)
                .unwrap();
            let first = (end - block) / HOP;
            let w0 = (end - ctx) / HOP;
            let last = (end / HOP).min(f0.len());
            f0[first..last].copy_from_slice(&window[first - w0..last - w0]);
            conf.extend_from_slice(e.last_confidence());
            end += block;
        }
        (f0, conf)
    }

    /// Frame-by-frame agreement of two contours: voiced-both |cents| (p99, max), V/UV flips (% of
    /// frames), and the confidence's |difference| (p99, max).
    fn agreement(a: &[f32], b: &[f32], ca: &[f32], cb: &[f32]) -> serde_json::Value {
        let mut cents: Vec<f64> = a
            .iter()
            .zip(b)
            .filter(|(p, q)| **p > 0.0 && **q > 0.0)
            .map(|(p, q)| f64::from(1200.0 * (p / q).log2()).abs())
            .collect();
        cents.sort_by(f64::total_cmp);
        let flips = a
            .iter()
            .zip(b)
            .filter(|(p, q)| (**p > 0.0) != (**q > 0.0))
            .count();
        let mut dc: Vec<f64> = ca
            .iter()
            .zip(cb)
            .map(|(p, q)| f64::from((p - q).abs()))
            .collect();
        dc.sort_by(f64::total_cmp);
        let q = |v: &[f64], p: f64| {
            if v.is_empty() {
                0.0
            } else {
                v[((v.len() - 1) as f64 * p).round() as usize]
            }
        };
        serde_json::json!({
            "voiced_both": cents.len(),
            "cents_p99": q(&cents, 0.99),
            "cents_max": q(&cents, 1.0),
            "vuv_flips_pct": 100.0 * flips as f64 / a.len().max(1) as f64,
            "confidence_delta_p99": q(&dc, 0.99),
            "confidence_delta_max": q(&dc, 1.0),
        })
    }

    /// Accumulated scores of contours against the synthetic truth (NaN = unscored frames).
    #[derive(Default)]
    struct Truth {
        scored: usize,
        both: usize,
        gross: usize,
        fine_sum: f64,
        fine_n: usize,
        vuv_err: usize,
    }

    impl Truth {
        fn add(&mut self, est: &[f32], truth: &[f32]) {
            for (e, t) in est.iter().zip(truth) {
                if t.is_nan() {
                    continue;
                }
                self.scored += 1;
                let (tv, ev) = (*t > 0.0, *e > 0.0);
                if tv != ev {
                    self.vuv_err += 1;
                }
                if tv && ev {
                    self.both += 1;
                    if ((e / t) - 1.0).abs() > 0.2 {
                        self.gross += 1;
                    } else {
                        self.fine_sum += f64::from(1200.0 * (e / t).log2()).abs();
                        self.fine_n += 1;
                    }
                }
            }
        }

        fn gpe_pct(&self) -> f64 {
            if self.both == 0 {
                0.0
            } else {
                100.0 * self.gross as f64 / self.both as f64
            }
        }

        fn json(&self) -> serde_json::Value {
            serde_json::json!({
                "gpe_pct": self.gpe_pct(),
                "fpe_cents": if self.fine_n == 0 { 0.0 } else { self.fine_sum / self.fine_n as f64 },
                "vuv_error_pct": if self.scored == 0 { 0.0 } else { 100.0 * self.vuv_err as f64 / self.scored as f64 },
                "voiced_both": self.both,
                "frames": self.scored,
            })
        }
    }

    /// One configuration's stream over the gate's clips: every frame's F0 and each call's
    /// confidence, and the synthetic truth scores.
    struct GateStream {
        f0: Vec<f32>,
        conf: Vec<f32>,
        truth: Truth,
    }

    /// A gate clip: name, 16 kHz audio and (synthetic clips) the true F0 per frame.
    type GateClip = (String, Vec<f32>, Option<Vec<f32>>);

    fn gate_stream(e: &mut RmvpePitchExtractor, clips: &[GateClip]) -> GateStream {
        let mut g = GateStream {
            f0: Vec::new(),
            conf: Vec::new(),
            truth: Truth::default(),
        };
        for (_, audio, truth) in clips {
            let (f, c) = stream_f0(e, audio);
            if let Some(t) = truth {
                g.truth.add(&f, t);
            }
            g.f0.extend(f);
            g.conf.extend(c);
        }
        g
    }

    /// The fp16 gate of `half` against `full` (module docs): voiced F0 delta p99 <= 5 cents, V/UV
    /// flips <= 0.2 pt, GPE not worse (+0.05 pt), confidence delta p99 <= 0.01.
    fn fp16_gate(half: &GateStream, full: &GateStream) -> (serde_json::Value, bool) {
        let a = agreement(&half.f0, &full.f0, &half.conf, &full.conf);
        let f = |k: &str| a[k].as_f64().unwrap_or(f64::INFINITY);
        let ok = f("cents_p99") <= 5.0
            && f("vuv_flips_pct") <= 0.2
            && half.truth.gpe_pct() <= full.truth.gpe_pct() + 0.05
            && f("confidence_delta_p99") <= 0.01;
        (a, ok)
    }

    /// Acceptance (the T5 rows of P3B-E) and the fp16 gate, on the stream path (the F0 cache's
    /// calls, 40 ms blocks, a 44-frame window) on DirectML and CUDA, over P3-D's 27 synthetic clips
    /// (ground truth) and the dev cache's real clips:
    /// - the default plan (64 frames, P3's size): fp16 against fp32 (the gate [`FP16_UNDER_AUTO`]
    ///   follows): voiced F0 delta p99 <= 5 cents, V/UV flips <= 0.2 pt, GPE not worse (+0.05 pt),
    ///   salience peak delta p99 <= 0.01;
    /// - the VRAM saver (32 frames): t32 against the full model (voiced p99 within 1 cent, V/UV flips
    ///   <= 0.05 pt), t32 fp16 against t32 (the same fp16 gate), and the saver against the default
    ///   plan (the cost of the saver, reported).
    ///
    /// Writes `scripts/vc-templates/reports/pitch_rmvpe_fp16_gate.json` (the evidence for keeping
    /// [`FP16_UNDER_AUTO`] off: P3B-I0).
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache cargo test rmvpe_fp16_gate -- --ignored --nocapture`
    #[test]
    #[ignore = "dev rig: needs SIGIL_TEST_MODELS_DIR with the RMVPE model and a GPU (hold the GPU timing lock)"]
    fn rmvpe_fp16_gate() {
        let Some((path, catalog)) = real_model() else {
            eprintln!("RMVPE model not found; skipped");
            return;
        };
        assert!(catalog, "{} is not the catalog's rmvpe.onnx", path.display());
        let synthetic = super::super::f0_fusion::ab::corpus(4.0);
        let real = real_clips();
        let mut clips: Vec<GateClip> = synthetic
            .iter()
            .map(|c| (c.name.clone(), c.audio.clone(), Some(c.truth.clone())))
            .collect();
        clips.extend(
            real.iter()
                .map(|(n, a)| (format!("real/{n}"), a.clone(), None)),
        );
        let mut report = serde_json::Map::new();
        let (mut default_pass, mut saver_pass, mut ran) = (true, true, 0);
        for ep in ["dml", "cuda"] {
            let variants = [
                ("64_full", 64usize, RmvpeGraph::Full, false),
                ("64_full_fp16", 64, RmvpeGraph::Full, true),
                ("32_full", 32, RmvpeGraph::Full, false),
                ("32_t32", 32, RmvpeGraph::T32, false),
                ("32_t32_fp16", 32, RmvpeGraph::T32, true),
            ];
            let mut streams: Vec<(&str, GateStream)> = Vec::new();
            let mut outcome = String::new();
            for (label, frames, graph, fp16) in variants {
                match rig_load(&path, ep, frames, graph, fp16) {
                    Ok(mut e) => {
                        outcome = e.outcome().active.clone();
                        streams.push((label, gate_stream(&mut e, &clips)));
                    }
                    Err(err) => println!("{ep} {label}: {err}"),
                }
            }
            if streams.len() < variants.len() || outcome != ep {
                println!("{ep}: not every variant ran on {ep} (outcome {outcome}); skipped");
                continue;
            }
            ran += 1;
            let get = |l: &str| &streams.iter().find(|(k, _)| *k == l).unwrap().1;
            let (d16, d16_ok) = fp16_gate(get("64_full_fp16"), get("64_full"));
            let (s16, s16_ok) = fp16_gate(get("32_t32_fp16"), get("32_t32"));
            let t32_vs_full = agreement(&get("32_t32").f0, &get("32_full").f0, &get("32_t32").conf, &get("32_full").conf);
            let ft = |k: &str| t32_vs_full[k].as_f64().unwrap_or(f64::INFINITY);
            let t32_ok = ft("cents_p99") <= 1.0 && ft("vuv_flips_pct") <= 0.05;
            let saver_vs_default = agreement(&get("32_full").f0, &get("64_full").f0, &get("32_full").conf, &get("64_full").conf);
            default_pass &= d16_ok;
            saver_pass &= s16_ok && t32_ok;
            let truth: serde_json::Map<String, serde_json::Value> = streams
                .iter()
                .map(|(k, g)| (k.to_string(), g.truth.json()))
                .collect();
            let verdict = |ok: bool| if ok { "PASS" } else { "FAIL" };
            println!("{ep}: default plan, fp16 vs fp32 {d16} -> {}", verdict(d16_ok));
            println!("{ep}: saver, t32 vs full {t32_vs_full} -> {}", verdict(t32_ok));
            println!("{ep}: saver, t32 fp16 vs t32 {s16} -> {}", verdict(s16_ok));
            println!("{ep}: saver vs default plan {saver_vs_default}");
            println!("{ep}: against the synthetic truth {}", serde_json::Value::Object(truth.clone()));
            report.insert(
                ep.to_string(),
                serde_json::json!({
                    "outcome": outcome,
                    "clips": clips.len(),
                    "synthetic_clips": synthetic.len(),
                    "real_clips": real.len(),
                    "frames": get("64_full").f0.len(),
                    "default_fp16_vs_fp32": d16,
                    "default_fp16_ok": d16_ok,
                    "saver_t32_vs_full": t32_vs_full,
                    "saver_t32_ok": t32_ok,
                    "saver_fp16_vs_t32": s16,
                    "saver_fp16_ok": s16_ok,
                    "saver_vs_default": saver_vs_default,
                    "synthetic_truth": serde_json::Value::Object(truth),
                }),
            );
        }
        report.insert("fp16_gate_passed_default_plan".into(), serde_json::Value::Bool(default_pass && ran > 0));
        report.insert("fp16_gate_passed_saver".into(), serde_json::Value::Bool(saver_pass && ran > 0));
        report.insert(
            "gate".into(),
            serde_json::json!(
                "stream path (F0 cache, 40 ms blocks, 44-frame window): fp16 vs fp32 voiced cents p99 <= 5, V/UV flips <= 0.2 pt, GPE vs the synthetic truth not worse (+0.05 pt), confidence delta p99 <= 0.01; t32 vs full (32 frames): cents p99 <= 1, flips <= 0.05 pt"
            ),
        );
        let out = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("scripts")
            .join("vc-templates")
            .join("reports")
            .join("pitch_rmvpe_fp16_gate.json");
        std::fs::write(&out, serde_json::to_string_pretty(&serde_json::Value::Object(report)).unwrap() + "\n").unwrap();
        println!("wrote {}", out.display());
    }

    /// The plan size against F0 quality on the stream path (DirectML; the 32 vs 64 question of
    /// P3B-E / P3B-F): P3's 64-frame full plan with the F0 cache's 12-frame lead, the 32-frame
    /// plan with the same lead, the 32-frame plan fed its whole shape (19 frames of lead: the
    /// `preferred_input_frames` priming), and the 64-frame plan fed its whole shape (51 frames of
    /// lead), scored against the synthetic truth (27 clips) and against each other on real speech.
    /// Writes `scripts/vc-templates/reports/pitch_rmvpe_plan_size.json`.
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache cargo test rmvpe_plan_size_f0 -- --ignored --nocapture`
    #[test]
    #[ignore = "dev rig: needs SIGIL_TEST_MODELS_DIR with the RMVPE model and a GPU (hold the GPU timing lock)"]
    fn rmvpe_plan_size_f0() {
        let Some((path, catalog)) = real_model() else {
            eprintln!("RMVPE model not found; skipped");
            return;
        };
        assert!(catalog);
        let ep = std::env::var("SIGIL_RMVPE_SIZE_EP").unwrap_or_else(|_| "dml".into());
        let synthetic = super::super::f0_fusion::ab::corpus(4.0);
        let real = real_clips();
        // (label, plan frames, lead, window)
        let variants: [(&str, usize, usize, usize); 6] = [
            ("p3_64_lead12", 64, 12, 44),
            ("32_lead12", 32, 12, 44),
            ("32_lead19_fill", 32, 19, 44),
            ("64_lead51_fill", 64, 51, 64),
            ("32_lead12_w64", 32, 12, 64),
            ("64_lead12_w64", 64, 12, 64),
        ];
        let mut report = serde_json::Map::new();
        let mut contours: Vec<(String, Vec<f32>)> = Vec::new();
        for (label, frames, lead, window) in variants {
            let graph = if frames == 32 { RmvpeGraph::T32 } else { RmvpeGraph::Full };
            let Ok(mut e) = rig_load(&path, &ep, frames, graph, false) else {
                println!("{label}: not built");
                continue;
            };
            let mut truth = Truth::default();
            // Per condition (the clip name after its `/`: clean, snr20 / 10 / 5, music-20, no-f1,
            // creak), where the octave traps are.
            let mut by_condition: Vec<(String, Truth)> = Vec::new();
            let mut real_f0 = Vec::new();
            for c in &synthetic {
                let (f, _) = stream_f0_lead(&mut e, &c.audio, lead, window);
                truth.add(&f, &c.truth);
                let condition = c.name.rsplit('/').next().unwrap_or(&c.name).to_string();
                match by_condition.iter_mut().find(|(k, _)| *k == condition) {
                    Some((_, t)) => t.add(&f, &c.truth),
                    None => {
                        let mut t = Truth::default();
                        t.add(&f, &c.truth);
                        by_condition.push((condition, t));
                    }
                }
            }
            for (_, a) in &real {
                real_f0.extend(stream_f0_lead(&mut e, a, lead, window).0);
            }
            let stats = e.stats();
            println!(
                "{ep} {label}: synthetic {} | plan runs {} split {}",
                truth.json(),
                stats.plan_runs,
                stats.split_calls
            );
            let conditions: serde_json::Map<String, serde_json::Value> = by_condition
                .iter()
                .map(|(k, t)| (k.clone(), serde_json::json!(t.gpe_pct())))
                .collect();
            println!("{ep} {label}: GPE % by condition {}", serde_json::Value::Object(conditions.clone()));
            let mut row = truth.json();
            row["gpe_pct_by_condition"] = serde_json::Value::Object(conditions);
            row["plan_runs"] = serde_json::json!(stats.plan_runs);
            row["split_calls"] = serde_json::json!(stats.split_calls);
            report.insert(label.to_string(), row);
            contours.push((label.to_string(), real_f0));
        }
        if let Some((_, base)) = contours.iter().find(|(l, _)| l == "p3_64_lead12") {
            for (label, f) in &contours {
                let n = f.len().min(base.len());
                let zeros = vec![0.0f32; n];
                let a = agreement(&f[..n], &base[..n], &zeros, &zeros);
                println!("{ep} {label} vs p3_64_lead12 on real speech: {a}");
                if let Some(row) = report.get_mut(label.as_str()) {
                    row["real_vs_p3"] = a;
                }
            }
        }
        report.insert("ep".into(), serde_json::json!(ep));
        let out = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("scripts")
            .join("vc-templates")
            .join("reports")
            .join("pitch_rmvpe_plan_size.json");
        std::fs::write(&out, serde_json::to_string_pretty(&serde_json::Value::Object(report)).unwrap() + "\n").unwrap();
        println!("wrote {}", out.display());
    }

    /// The real model on the CPU: the t32 template filled from `rmvpe.onnx` gives the full model's
    /// salience at 32 mel frames (max |difference| <= 1e-5, identical argmax on voiced rows), and
    /// the protobuf walk reads its contract.
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache cargo test real_t32_matches_full -- --ignored --nocapture`
    #[test]
    #[ignore = "dev rig: needs SIGIL_TEST_MODELS_DIR with the RMVPE model"]
    fn real_t32_matches_full_on_cpu() {
        let Some((path, catalog)) = real_model() else {
            eprintln!("RMVPE model not found; skipped");
            return;
        };
        assert!(catalog);
        let info = onnx_proto::read_graph_info(&path).unwrap();
        let contract = MelContract::of(&info).unwrap();
        assert_eq!(contract.time_dim.as_deref(), Some("time"));
        let mut t32 = planned_load(
            &path,
            saver(),
            Hooks {
                trust_source: true,
                ..Hooks::default()
            },
        );
        assert_eq!(t32.plan_shape(), Some((32, RmvpeGraph::T32, false)), "{:?}", t32.load_notes());
        let mut full = planned_load(&path, saver(), Hooks::default());
        assert_eq!(full.plan_shape(), Some((32, RmvpeGraph::Full, false)));
        let mut audio = super::super::f0_fusion::ab::corpus(2.0)
            .into_iter()
            .map(|c| c.audio)
            .collect::<Vec<_>>();
        audio.extend(real_clips().into_iter().map(|(_, a)| a));
        let (mut worst, mut argmax, mut rows) = (0.0f32, 0usize, 0usize);
        let am = |r: &[f32]| {
            r.iter()
                .enumerate()
                .fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
                .0
        };
        for (k, a) in audio.iter().enumerate() {
            let n = 31 * HOP;
            if a.len() < n + 8_000 {
                continue;
            }
            let start = (k * 3_517) % (a.len() - n);
            let slice = &a[start..start + n];
            t32.f0(slice).unwrap();
            full.f0(slice).unwrap();
            let (sa, sb) = (t32.last_salience().unwrap().0, full.last_salience().unwrap().0);
            for (ra, rb) in sa.as_chunks::<N_BINS>().0.iter().zip(sb.as_chunks::<N_BINS>().0) {
                rows += 1;
                if am(ra) != am(rb) && ra.iter().copied().fold(0.0f32, f32::max) > SALIENCE_THRESHOLD {
                    argmax += 1;
                }
                for (p, q) in ra.iter().zip(rb) {
                    worst = worst.max((p - q).abs());
                }
            }
        }
        println!("t32 vs full on the CPU: {rows} rows, max |salience difference| {worst:.3e}, argmax differs on {argmax} voiced rows");
        assert!(worst <= 1e-5 && argmax == 0);
    }

    /// Dev rig: RMVPE per-call cost of the production configuration per EP on a whole 44-frame
    /// window (now split into two 32-frame runs), then the F0 cache's 24-frame slice, then the
    /// window again.
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache cargo test rmvpe_tick_cost -- --ignored --nocapture`
    #[test]
    #[ignore = "dev rig: needs SIGIL_TEST_MODELS_DIR with the RMVPE model (hold the GPU timing lock)"]
    fn rmvpe_tick_cost() {
        let Some((path, _)) = real_model() else {
            eprintln!("RMVPE model not found; skipped");
            return;
        };
        let x = super::super::f0_fusion::ab::harmonic_tone(180.0, 16_000, 0.3);
        for (ep, levers) in [
            ("cuda", GpuLevers::default()),
            ("cuda", rig_levers("cuda")),
            ("dml", GpuLevers::default()),
            ("dml", rig_levers("dml")),
            ("cpu", GpuLevers::default()),
        ] {
            let opts = VcSessionOptions {
                intra_threads: Some(2),
                ..VcSessionOptions::default()
            };
            let Ok(mut r) = RmvpePitchExtractor::load_with_levers(&path.to_string_lossy(), ep, &opts, levers) else {
                continue;
            };
            let label = format!(
                "{ep}{}",
                r.plan_shape()
                    .map(|(f, g, h)| format!(" plan {f} {}{}", g.as_str(), if h { " fp16" } else { "" }))
                    .unwrap_or_default()
            );
            for n in [7_040usize, 3_840, 7_040] {
                let mut times = Vec::new();
                for k in 0..40 {
                    let t0 = Instant::now();
                    let f0 = r.f0(&x[k * 160..k * 160 + n]).unwrap();
                    times.push(t0.elapsed().as_secs_f64() * 1000.0);
                    assert!(!f0.is_empty());
                }
                let first = times[0];
                times.sort_by(f64::total_cmp);
                println!(
                    "rmvpe {label} {n} samples: first {first:.1} ms, p50 {:.2} ms, p95 {:.2} ms",
                    times[20], times[38]
                );
            }
        }
    }
}
