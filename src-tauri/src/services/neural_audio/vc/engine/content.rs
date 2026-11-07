//! Shared content-encoder pool (P3-C1): ContentVec 768 / layer 12, ContentVec 256 / layer 9 +
//! final_proj, SPIN-v2, SPIN (v1; P3-I0) and zh / ja / ko HuBERT, chosen per slot from its manifest / checkpoint
//! metadata; fp16 on GPUs, int8 on VNNI CPUs (`int8_gate`), shared across hot slots (one session
//! per embedder + EP + precision, plan T3: "+<= 150 MB per extra hot voice").
//!
//! # Frozen by P3-S0 (signatures; P3-C1 implements)
//! - [`EmbedderKind`] with [`EmbedderKind::from_manifest`], [`EmbedderKind::as_str`] (the manifest
//!   / catalog name) and [`EmbedderKind::frontend`] (receptive field and hop at 16 kHz: 400 / 320
//!   for every supported encoder; P3-A verifies the CJK exports).
//! - [`ContentEncoderPool::global`], [`ContentEncoderPool::acquire`] → [`SharedEncoder`];
//!   [`ContentEncoder::encode`] writes `frames × channels` features into a caller-owned buffer
//!   (no per-tick allocation) and returns the frame count.
//!
//! # Resolution (P3-C1)
//! A slot's encoder file is found by [`resolve_encoder_file`] ([`acquire_for_slot`] loads it):
//! 1. an explicit path (the UI's HuBERT path) for the ContentVec kinds, the one encoder a user can
//!    pick by hand today, when it produces the slot's feature width (a 768-channel HuBERT set for
//!    v2 voices does not serve a v1 voice: [`acquire_for_slot`] falls back to the steps below with
//!    a warning); a SPIN / CJK voice always uses its own catalog entry;
//! 2. the model-store entry of the kind (`vc/embedder-<name>`, `vc::assets`), trying the precision
//!    variants in [`variant_order`]: `model.fp16.onnx` first on GPUs, `model.int8.onnx` first on
//!    CPUs with VNNI (`int8_gate`), then `model.fp32.onnx` and `model.onnx`;
//! 3. for ContentVec, the legacy `voice-changer/base/` files (`hubert_base.onnx`, ...) next to the
//!    slot folder, so existing setups keep working with no download.
//!
//! A GPU-precision variant whose session falls back to the CPU is rebuilt from the CPU variant.
//! Encoders run through `ort_session::build_session` (EP ladder, outcome `vc.content`) with host
//! tensors; the pool shares one session per (file, EP, device, threads) between hot slots.
//!
//! # Frozen by P3B-S0 (signatures; P3B-D implements)
//! - [`runs_fixed_shape`]`(ep) -> bool`: whether the content encoder runs as a fixed-shape session
//!   plan on the rung `ep`. The factory's DirectML lookahead skip applies only while it is `false`.
//! - [`running_norm_available`]`() -> bool`: whether this install can run window-independent content
//!   normalisation (the `content_norm` lever is offered and resolves on only when it is `true`: a
//!   `stream` encoder variant is installed).
//!
//! # The fixed-shape plan (P3B-D)
//! The template engine feeds its encoder one constant-length window every tick (the context, or the
//! content window with its real left context), so on a GPU rung the encoder runs as a
//! `gpu::SessionPlan` at that length ([`ContentEncoder::encode_fixed`]):
//! - the input's free dimension is pinned (`with_dimension_override`), so ONNX Runtime folds the
//!   graph's shape arithmetic (about 130 CPU nodes per run of the dynamic ContentVec graph) and
//!   DirectML compiles the whole graph once, instead of re-planning on a new length (the 34-49 ms
//!   calls at a crossfade other than 20 ms or with lookahead);
//! - IOBinding, a thread-local CUDA graph on the plan's stream (the priority stream with
//!   `gpu_priority`), checked against regular runs at build, and the high-priority DirectML queue,
//!   following the engine's `io_binding` / `cuda_graphs` / `gpu_priority` levers;
//! - every plan is checked once, on the window that built it, and a plan that cannot be built or
//!   misses its check leaves the encoder on its dynamic session for that length, with a note. An
//!   encoder that has a dynamic session (the CPU, explicit or legacy files, the kill-switch path) is
//!   checked against it ([`PLAN_CHECK_REL_L2`], same EP, same graph). A *deferred* encoder (the
//!   production GPU path: no dynamic session exists, to spare the load its VRAM and time) is checked
//!   against a CPU-EP session of the same file ([`PLAN_CPU_CHECK_REL_L2`]), which runs on a helper
//!   thread while the plan builds, so a graph a driver compiled wrongly at its pinned length (ADR
//!   0006 finding 5) is caught on any adapter, not only on the dev rig. The verdict is remembered per
//!   (file, EP, device, plan) for the process, so only the first load of a shape pays for it; kill
//!   switch `SIGIL_VC_CONTENT_CHECK=0` ([`CONTENT_CHECK_ENV`]) leaves only the finite / not-silent test;
//! - plans are keyed by (length, levers) inside the pooled encoder, so hot voices with the same
//!   stream shape share one; each plan's VRAM is measured at build and held in the ledger
//!   (`telemetry::vram_mark` too). A voice holds a [`ContentLease`] on its plan; a plan nobody leases
//!   is dropped, and once every holder of the encoder runs a plan the dynamic session is released
//!   (rebuilt on demand by the next [`ContentEncoder::encode`] caller), so the plan does not double
//!   the encoder's VRAM.
//!
//! Only files with fp32 audio in and fp32 features out run plans (Sigil's store variants and most
//! explicit / legacy files); fp16-I/O files keep the dynamic path. Kill switch:
//! `SIGIL_VC_CONTENT_PLAN=0` ([`CONTENT_PLAN_ENV`]).
//!
//! # Precision of the store variants (P3B-D; ADR 0011)
//! [`variant_order`] documents why fp16 stays first on the GPU rungs and int8 first on VNNI CPUs
//! although both missed a plan rule: the downstream measurements decided it.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use half::f16;
use ort::session::{Session, SessionInputValue};
use ort::value::{TensorElementType, TensorRef, ValueType};

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::gpu::{GpuLevers, PlanSpec, SessionPlan, TensorSpec};
use crate::services::neural_audio::ort_session::{build_session, normalize_ep, ModelSource, VcSessionOptions, EP_CPU, EP_CUDA, EP_DML};
use crate::services::neural_audio::telemetry::{self, LoadStage, SyncKind};
use crate::services::neural_audio::vc::assets;
use crate::services::neural_audio::vc::int8_gate;
use crate::services::neural_audio::vc::rvc;

use super::options::{Accelerator, PrecisionMode};

/// Kill switch of the fixed-shape content plan: `0` / `off` / `false` / `no` keeps every content
/// encoder on its dynamic session (the P3 path), for A/B runs from one executable.
pub const CONTENT_PLAN_ENV: &str = "SIGIL_VC_CONTENT_PLAN";

/// Switch of the content plan's CUDA graph: `0` / `off` / `false` / `no` runs the plan with
/// IOBinding only (no graph capture) even when the `cuda_graphs` lever is on, for the A/B of the
/// graph's speed against the VRAM it holds (P3B-D measurement). Read at the voice's first tick.
pub const CONTENT_GRAPH_ENV: &str = "SIGIL_VC_CONTENT_GRAPH";

/// Whether the content plan may capture a CUDA graph ([`CONTENT_GRAPH_ENV`]; on unless switched
/// off). Reads the environment: call it at load time, not per tick.
pub fn content_graph_switch() -> bool {
    switch_on(std::env::var(CONTENT_GRAPH_ENV).ok().as_deref())
}

/// An on-by-default environment switch: only `0` / `off` / `false` / `no` (any case, padded) turn
/// it off; unset or anything else keeps it on.
fn switch_on(value: Option<&str>) -> bool {
    value.is_none_or(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no"))
}

/// Largest relative L2 between a new plan's output and the dynamic session's on the window that
/// built it. Both run the same graph at the same precision, so they differ only by kernel choice
/// and fusion order (well under 1e-2 even in fp16); a larger difference means a broken plan (a
/// wrong output layout, a graph that did not record everything), which is then not used.
pub const PLAN_CHECK_REL_L2: f64 = 2e-2;

/// Largest relative L2 between a *deferred* plan's output and a CPU-EP session's (the same file) on
/// the window that built it. The two run different kernels, so they differ by more than a plan and
/// a dynamic session on one EP: the CPU runs even the fp16 file in fp32, so an fp16 plan differs
/// from it by fp16's own error. Measured by `content_plan_cpu_reference_dev_rig` (RTX 4090, CUDA and
/// DirectML, 8 speech windows per length): fp16 4.9e-3 (2 s) to 1.2e-2 (the live 0.44 s window; up to
/// 7.6e-2 at 0.1 s per the fp16 stream parity run), fp32 6e-6 to 2.5e-5. A plan that reads wrong
/// features (a pinned dimension a driver compiled wrongly, a scrambled layout) is off by order 1, and
/// even a 30 % gain error reads 0.3, so 0.1 sits well above the noise and below real damage.
/// (This is a refusal limit; the stricter DirectML parity of the acceptance table, 1e-3, is asserted
/// against the same-EP dynamic session by `content_plan_dev_rig`.)
pub const PLAN_CPU_CHECK_REL_L2: f64 = 0.1;

/// Intra-op threads of the cross-check's CPU session: the check runs while the voice loads (and
/// possibly while another voice streams), so it takes a few cores, not all of them.
const CHECK_THREADS: usize = 3;

/// Whether the plan path is switched on in this process ([`CONTENT_PLAN_ENV`]; on unless set to
/// `0` / `off` / `false` / `no`). Reads the environment: call it at load time, not per tick.
pub fn content_plan_switch() -> bool {
    switch_on(std::env::var(CONTENT_PLAN_ENV).ok().as_deref())
}

/// Kill switch of the deferred plan's cross-check against a CPU session: `0` / `off` / `false` /
/// `no` leaves only the finite / not-silent test (the P3B-D integration build's behaviour), for A/B
/// runs and for a machine whose CPU cannot run the encoder file.
pub const CONTENT_CHECK_ENV: &str = "SIGIL_VC_CONTENT_CHECK";

/// Whether a deferred plan is cross-checked against a CPU session ([`CONTENT_CHECK_ENV`]; on unless
/// switched off). Reads the environment: call it at load time, not per tick.
pub fn content_check_switch() -> bool {
    switch_on(std::env::var(CONTENT_CHECK_ENV).ok().as_deref())
}

/// Whether the content encoder runs as a fixed-shape session plan (a whole-graph DirectML compile,
/// IOBinding and a CUDA graph per the levers) on the EP rung `ep`: on the GPU rungs unless the
/// kill switch is set. A voice whose encoder file has fp16 audio I/O, or whose plan could not be
/// built, still runs its dynamic session; `SigilRvcEngine::content_runs_fixed_shape` says what a
/// loaded voice really does.
pub fn runs_fixed_shape(ep: &str) -> bool {
    matches!(normalize_ep(ep), EP_CUDA | EP_DML) && content_plan_switch()
}

/// Whether this install can run window-independent content normalisation (the `content_norm`
/// lever, P3B-D): only when some content encoder has an installed `stream` variant with its
/// `conv0.f32.bin` (the variants are opt-in exports, `export_embedders.py --stream`, and no catalog
/// entry publishes them yet), so the lever is not offered where it could only turn itself off at
/// load. A voice whose own encoder lacks the variant still turns the lever off with a note.
pub fn running_norm_available() -> bool {
    EmbedderKind::ALL.into_iter().any(stream_variant_installed)
}

/// Whether `kind` has an installed `stream` variant with the first convolution's weights next to it
/// (what [`resolve_stream_variant`] needs on either rung). Only looks at the files.
fn stream_variant_installed(kind: EmbedderKind) -> bool {
    crate::services::model_store::installed_path(kind.catalog_id()).is_some_and(|dir| stream_variant_in(&dir))
}

/// Whether the install folder `dir` of an encoder holds a `stream` variant (fp16 or fp32) and
/// `conv0.f32.bin`.
fn stream_variant_in(dir: &Path) -> bool {
    dir.join("conv0.f32.bin").is_file()
        && ["stream.fp16", "stream.fp32"]
            .into_iter()
            .any(|v| dir.join(assets::file_name(Some(v))).is_file())
}

/// An installed `stream` variant of an encoder (the `content_norm` lever): the graph that takes
/// the first GroupNorm's statistics as inputs, and the first convolution's weights the engine
/// computes the running statistics with.
#[derive(Clone, Debug)]
pub struct StreamVariant {
    pub path: PathBuf,
    /// `stream.fp16` / `stream.fp32`.
    pub variant: &'static str,
    /// The fp32 variant for a GPU variant that lands on the CPU.
    pub cpu_fallback: Option<PathBuf>,
    /// `conv0.f32.bin`: `[512, 1, 10]` f32.
    pub conv0: Vec<f32>,
}

/// The `stream` variant of `kind` from the model store: `model.stream.fp16.onnx` first on GPUs
/// (unless fp32 is asked for), else `model.stream.fp32.onnx`, with `conv0.f32.bin` next to it.
pub fn resolve_stream_variant(kind: EmbedderKind, gpu: bool, precision: PrecisionMode) -> Result<StreamVariant, String> {
    let order: &[&'static str] = match (gpu, precision) {
        (true, PrecisionMode::Fp32) | (false, _) => &["stream.fp32"],
        (true, _) => &["stream.fp16", "stream.fp32"],
    };
    for &v in order {
        if let Some(path) = assets::model_file(kind.catalog_id(), Some(v)) {
            let conv0 = read_conv0(&path.with_file_name("conv0.f32.bin"))?;
            let cpu_fallback = if gpu {
                assets::model_file(kind.catalog_id(), Some("stream.fp32")).filter(|p| p != &path)
            } else {
                None
            };
            return Ok(StreamVariant {
                path,
                variant: v,
                cpu_fallback,
                conv0,
            });
        }
    }
    Err(format!(
        "the {} encoder has no stream variant installed (model.stream.fp16.onnx / model.stream.fp32.onnx with conv0.f32.bin in '{}')",
        kind.as_str(),
        kind.catalog_id()
    ))
}

/// The first convolution's weights (`conv0.f32.bin`, `[512, 1, 10]` little-endian f32).
pub fn read_conv0(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != NORM_CHANNELS * 10 * 4 {
        return Err(format!("{} holds {} bytes, expected {} (512 x 10 f32)", path.display(), bytes.len(), NORM_CHANNELS * 40));
    }
    Ok(bytes
        .as_chunks::<4>().0.iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// A content encoder family (the `embedder` of a voice).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EmbedderKind {
    /// ContentVec / HuBERT base, 768 channels from layer 12 (RVC v2).
    ContentVec768,
    /// ContentVec / HuBERT base, layer 9 through `final_proj` to 256 channels (RVC v1).
    ContentVec256,
    /// SPIN-v2 (Applio).
    SpinV2,
    /// Applio's first SPIN checkpoint (another network than SPIN-v2; `vc/embedder-spin`, P3-I0).
    Spin,
    HubertJa,
    HubertKo,
    HubertZh,
}

impl EmbedderKind {
    pub const ALL: [EmbedderKind; 7] = [
        EmbedderKind::ContentVec768,
        EmbedderKind::ContentVec256,
        EmbedderKind::SpinV2,
        EmbedderKind::Spin,
        EmbedderKind::HubertJa,
        EmbedderKind::HubertKo,
        EmbedderKind::HubertZh,
    ];

    /// The manifest / catalog name.
    pub fn as_str(self) -> &'static str {
        match self {
            EmbedderKind::ContentVec768 => "contentvec",
            EmbedderKind::ContentVec256 => "contentvec-256",
            EmbedderKind::SpinV2 => "spin-v2",
            EmbedderKind::Spin => "spin",
            EmbedderKind::HubertJa => "hubert-ja",
            EmbedderKind::HubertKo => "hubert-ko",
            EmbedderKind::HubertZh => "hubert-zh",
        }
    }

    /// Feature channels the encoder produces.
    pub fn channels(self) -> usize {
        match self {
            EmbedderKind::ContentVec256 => 256,
            _ => 768,
        }
    }

    /// Model-store id of the encoder (`vc::assets`).
    pub fn catalog_id(self) -> &'static str {
        assets::embedder_id(self.as_str()).unwrap_or(assets::EMBEDDER_CONTENTVEC)
    }

    /// The embedder a slot needs: its manifest `embedder` name, else ContentVec by channel count.
    ///
    /// Accepts the manifest names ([`as_str`](Self::as_str)) and the names RVC / Applio
    /// checkpoints record (`embedder_model`: `contentvec`, `hubert_base`, `spin`,
    /// `japanese-hubert-base`, `chinese-hubert-base`, `korean-hubert-base`, ...). The weights'
    /// channel count is authoritative for ContentVec (a v1 voice named `contentvec` uses the
    /// 256-channel encoder); the other encoders exist only with 768 channels.
    pub fn from_manifest(embedder: Option<&str>, emb_channels: u32) -> Result<EmbedderKind, String> {
        let by_channels = || match emb_channels {
            768 => Ok(EmbedderKind::ContentVec768),
            256 => Ok(EmbedderKind::ContentVec256),
            c => Err(format!("no content encoder produces {c}-channel features (RVC v1 uses 256, v2 768)")),
        };
        let Some(raw) = embedder.map(str::trim).filter(|s| !s.is_empty()) else {
            return by_channels();
        };
        let name = raw.to_ascii_lowercase().replace(['_', ' '], "-");
        let kind = match name.as_str() {
            "contentvec" | "content-vec" | "hubert" | "hubert-base" | "contentvec-768" | "vec-768-layer-12" | "contentvec-base" => {
                return by_channels();
            }
            "contentvec-256" | "vec-256-layer-9" => EmbedderKind::ContentVec256,
            "spin-v2" | "spinv2" => EmbedderKind::SpinV2,
            // Applio's first SPIN is another checkpoint than SPIN-v2 (its own catalog entry).
            "spin" | "spin-v1" | "spinv1" => EmbedderKind::Spin,
            "hubert-ja" | "japanese-hubert-base" | "rinna-hubert-base" | "japanese-hubert" => EmbedderKind::HubertJa,
            "hubert-ko" | "korean-hubert-base" | "korean-hubert" => EmbedderKind::HubertKo,
            "hubert-zh" | "chinese-hubert-base" | "chinese-hubert" => EmbedderKind::HubertZh,
            other => return Err(format!("the voice was trained with the content encoder '{other}', which Sigil does not have")),
        };
        if kind.channels() as u32 != emb_channels {
            return Err(format!(
                "the voice's {emb_channels}-channel features do not match its content encoder {} ({} channels)",
                kind.as_str(),
                kind.channels()
            ));
        }
        Ok(kind)
    }

    /// `(receptive field, hop)` of the encoder's conv front end at 16 kHz.
    pub fn frontend(self) -> (usize, usize) {
        (400, 320)
    }
}

/// Model files to try on a device class, most preferred first: a variant name (`fp16` / `fp32` /
/// `int8`, file `model.<variant>.onnx`) or `None` (`model.onnx`).
///
/// # Why `auto` still starts with fp16 on GPUs and int8 on VNNI CPUs (P3B-D, ADR 0011)
/// P3B-D's precision rules were: fp16 ContentVec failing its parity gate at stream lengths puts
/// fp32 first on the GPU, and an int8 loss above 0.002 speaker cosine / 0.03 dB MCD puts fp32 first
/// on the CPU. Both fired on the encoder's own metrics (the export's `embedder.json` even records
/// `recommended: fp32` for both classes and the catalog says DirectML "runs fp32"), and both were
/// then measured where it counts, the converted voice against an fp32-content ceiling (scorer v2,
/// 5 clips per EP, the stream and its ceiling on the same plan path; scripts/bench/specs/p3b/d/README.md):
/// - GPU fp16 minus fp32: CUDA speaker cosine +0.0008, MCD +0.0045 dB, DNSMOS +0.017; DirectML
///   +0.0001, -0.0062 dB, -0.037. The same size as the plan-vs-plain difference and inside the rules'
///   limits, in both directions, so fp16 costs no measurable fidelity. fp32 would add about 170 MB of
///   VRAM per plan (498 against 328 MB on CUDA, 381 against 208 MB on DirectML; T3 is already a
///   miss) and 1.0 ms (CUDA) / 0.5 ms (DirectML) per content call.
/// - CPU int8 minus fp32: speaker cosine -0.0026 (the rule fires, marginally), MCD -0.012 dB and V/UV
///   -0.006 (better), DNSMOS -0.04, output F0 RMSE +10 cents (2 of 5 clips better). fp32 would add
///   an unmeasured CPU cost to the Streamer, which already misses its RTF target; P7-B measures both.
///
/// So the shipped order stays; the rule's outcome is one setting away: `precision = fp32` puts fp32
/// first on both classes (the arms below), and `SIGIL_VC_INT8=0` ([`int8_gate::INT8_ENV`]) alone
/// turns the CPU's int8 off. The CPU rule fired by 0.0006 speaker cosine (0.0026 against 0.002) on
/// one rig; int8 stays the Streamer's default until P7-B has the fp32 encoder's CPU cost, and the
/// decision is recorded in ADR 0010 and ADR 0011.
pub fn variant_order(gpu: bool, precision: PrecisionMode, int8_ok: bool) -> Vec<Option<&'static str>> {
    match (gpu, precision) {
        (true, PrecisionMode::Fp32) => vec![Some("fp32"), None],
        (true, _) => vec![Some("fp16"), Some("fp32"), None],
        // CPU fp16 is ONNX Runtime's slow emulation: an explicit fp16 request runs fp32 there.
        (false, PrecisionMode::Auto) if int8_ok => vec![Some("int8"), Some("fp32"), None],
        (false, _) => vec![Some("fp32"), None],
    }
}

/// Legacy `voice-changer/base/` file names of the ContentVec encoders.
const LEGACY_768: &[&str] = &["hubert_base.onnx", "contentvec.onnx", "vec-768-layer-12.onnx", "hubert.onnx"];
const LEGACY_256: &[&str] = &["vec-256-layer-9.onnx", "hubert_base_256.onnx"];

/// Where an encoder file came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedEncoder {
    pub path: PathBuf,
    /// `fp16` / `fp32` / `int8` / `model` (store entries), `explicit` or `legacy`.
    pub variant: &'static str,
    /// Store variants of the same entry still usable on the CPU (after a GPU fallback).
    pub cpu_fallback: Option<PathBuf>,
}

fn store_variant(kind: EmbedderKind, variant: Option<&'static str>) -> Option<(PathBuf, &'static str)> {
    assets::model_file(kind.catalog_id(), variant).map(|p| (p, variant.unwrap_or("model")))
}

/// Find the encoder file of `kind` (see the module docs for the order). `slot_dir` enables the
/// legacy `voice-changer/base/` fallback (`<app_data>/voice-changer/models/<slot>` → `../../base`).
pub fn resolve_encoder_file(
    kind: EmbedderKind,
    explicit: Option<&Path>,
    slot_dir: Option<&Path>,
    gpu: bool,
    precision: PrecisionMode,
) -> Result<ResolvedEncoder, String> {
    let contentvec = matches!(kind, EmbedderKind::ContentVec768 | EmbedderKind::ContentVec256);
    if let Some(p) = explicit.filter(|p| !p.as_os_str().is_empty()) {
        if contentvec {
            if !p.is_file() {
                return Err(format!("content encoder not found: {}", p.display()));
            }
            return Ok(ResolvedEncoder {
                path: p.to_path_buf(),
                variant: "explicit",
                cpu_fallback: None,
            });
        }
    }
    let int8_ok = int8_gate::int8_content_encoder_enabled();
    for v in variant_order(gpu, precision, int8_ok) {
        if let Some((path, variant)) = store_variant(kind, v) {
            let cpu_fallback = if gpu {
                variant_order(false, precision, int8_ok)
                    .into_iter()
                    .find_map(|cv| store_variant(kind, cv))
                    .map(|(p, _)| p)
                    .filter(|p| p != &path)
            } else {
                None
            };
            return Ok(ResolvedEncoder { path, variant, cpu_fallback });
        }
    }
    if contentvec {
        let names = if kind == EmbedderKind::ContentVec768 { LEGACY_768 } else { LEGACY_256 };
        let mut dirs: Vec<PathBuf> = Vec::new();
        if let Some(slot) = slot_dir {
            if let Some(vc_root) = slot.parent().and_then(Path::parent) {
                dirs.push(vc_root.join("base"));
            }
            dirs.push(slot.to_path_buf());
        }
        for dir in dirs {
            for name in names {
                let p = dir.join(name);
                if p.is_file() {
                    return Ok(ResolvedEncoder {
                        path: p,
                        variant: "legacy",
                        cpu_fallback: None,
                    });
                }
            }
        }
    }
    Err(format!(
        "the {} content encoder this voice needs is not installed: add '{}' in Models & Runtimes",
        kind.as_str(),
        kind.catalog_id()
    ))
}

/// The pooled encoder of a slot: [`resolve_encoder_file`], then
/// [`ContentEncoderPool::acquire_resolved`]. An explicit file is checked with one encode of
/// `probe_16k` samples of silence (the plan's context length, so the probe is also a warm-up at the
/// live shape): when it does not produce `kind`'s feature width, the store / legacy resolution runs
/// instead and `warnings` says so; if nothing else resolves either, the explicit file's error is
/// returned. When the explicit file wins although a faster store variant of the kind is installed
/// (fp16 on a GPU, int8 on a VNNI CPU), `warnings` says how to use it.
#[allow(clippy::too_many_arguments)]
pub fn acquire_for_slot(
    kind: EmbedderKind,
    explicit: Option<&Path>,
    slot_dir: Option<&Path>,
    ep: &str,
    opts: &VcSessionOptions,
    precision: PrecisionMode,
    probe_16k: usize,
    warnings: &mut Vec<String>,
) -> Result<SharedEncoder, String> {
    let pool = ContentEncoderPool::global();
    let gpu = normalize_ep(ep) != "cpu";
    let resolved = resolve_encoder_file(kind, explicit, slot_dir, gpu, precision)?;
    let encoder = pool.acquire_resolved(kind, &resolved, ep, opts)?;
    if resolved.variant != "explicit" {
        return Ok(encoder);
    }
    match probe_width(&encoder, probe_16k) {
        Ok(()) => {
            let int8_ok = int8_gate::int8_content_encoder_enabled();
            let faster = variant_order(gpu, precision, int8_ok)
                .into_iter()
                .next()
                .flatten()
                .filter(|v| matches!(*v, "fp16" | "int8"))
                .and_then(|v| store_variant(kind, Some(v)))
                // The settings may name that very file (no advice then).
                .filter(|(p, _)| !same_file(p, &resolved.path));
            if let Some((_, v)) = faster {
                warnings.push(format!(
                    "content encoder: the HuBERT path from the settings ({}) runs instead of the installed {v} {} encoder ({}); clear the path to use the faster one",
                    resolved.path.display(),
                    kind.as_str(),
                    kind.catalog_id()
                ));
            }
            Ok(encoder)
        }
        Err(e) => {
            drop(encoder);
            pool.release_unused();
            let fallback = resolve_encoder_file(kind, None, slot_dir, gpu, precision).and_then(|r| {
                let enc = pool.acquire_resolved(kind, &r, ep, opts)?;
                probe_width(&enc, probe_16k)?;
                Ok((enc, r))
            });
            match fallback {
                Ok((enc, r)) => {
                    warnings.push(format!(
                        "content encoder: the HuBERT path from the settings ({}) does not fit this voice ({e}); using {} instead",
                        resolved.path.display(),
                        r.path.display()
                    ));
                    Ok(enc)
                }
                Err(_) => {
                    pool.release_unused();
                    Err(e)
                }
            }
        }
    }
}

/// [`acquire_for_slot`] for the template engine (P3B-D): on a GPU rung with the plan path on, a Sigil
/// store variant (with its export contract) is pooled without a dynamic session
/// ([`ContentEncoder::deferred`]): the engine's first tick builds its fixed-shape plan directly, so
/// the voice's cold load builds one content session instead of two. Everything else (the CPU, the
/// kill switch, an explicit or legacy file, a variant without a contract) is [`acquire_for_slot`].
#[allow(clippy::too_many_arguments)]
pub fn acquire_for_engine(
    kind: EmbedderKind,
    explicit: Option<&Path>,
    slot_dir: Option<&Path>,
    ep: &str,
    opts: &VcSessionOptions,
    precision: PrecisionMode,
    probe_16k: usize,
    warnings: &mut Vec<String>,
) -> Result<SharedEncoder, String> {
    let gpu = matches!(normalize_ep(ep), EP_CUDA | EP_DML);
    let explicit_used = matches!(kind, EmbedderKind::ContentVec768 | EmbedderKind::ContentVec256) && explicit.is_some_and(|p| !p.as_os_str().is_empty());
    if gpu && !explicit_used && content_plan_switch() {
        if let Ok(resolved) = resolve_encoder_file(kind, None, slot_dir, gpu, precision) {
            if matches!(resolved.variant, "fp16" | "fp32" | "model") {
                if let Ok(enc) = ContentEncoderPool::global().acquire_deferred(kind, &resolved, ep, opts) {
                    return Ok(enc);
                }
            }
        }
    }
    acquire_for_slot(kind, explicit, slot_dir, ep, opts, precision, probe_16k, warnings)
}

/// Whether two paths name the same file (canonical paths when both resolve; else the paths as
/// given, ignoring case and separator style on Windows).
fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => {
            let norm = |p: &Path| p.to_string_lossy().replace('\\', "/").to_ascii_lowercase();
            norm(a) == norm(b)
        }
    }
}

/// One encode of `len` samples of silence (at least the front end's 720): fails when the encoder
/// does not produce its kind's feature width.
fn probe_width(encoder: &SharedEncoder, len: usize) -> Result<(), String> {
    let probe = vec![0.0f32; len.max(720)];
    let mut feats = Vec::new();
    // The dynamic session, without counting as a dynamic-path user (a template voice that probes
    // its explicit file still runs plans afterwards).
    encoder
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .encode_dynamic(&probe, &mut feats)
        .map(|_| ())
}

/// How an encoder's graph takes its audio and returns its features (read from its dynamic session).
#[derive(Clone, Debug)]
struct EncoderIo {
    input_name: String,
    /// `[N]` (1) or `[1, N]` (2).
    input_rank: usize,
    /// The graph takes fp16 audio (converted here; fp32 is the normal case).
    half_input: bool,
    /// The input's free dimensions: `(axis, symbol)` (an empty symbol cannot be pinned by name).
    input_free: Vec<(usize, String)>,
    /// The output carrying the voice's feature width (`rvc::feature_output_index`: w-okada's
    /// three-output `contentvec-f/-h` / `hubert-jp-f/-h` serve 256- and 768-wide voices; P5-I0).
    output: usize,
    output_name: String,
    /// The features come out as fp32 (a plan needs fp32 on both sides).
    output_f32: bool,
    /// The feature output's declared dimensions (negative = free).
    output_dims: Option<Vec<i64>>,
    /// The graph is a `stream` variant: it takes the first GroupNorm's statistics as the inputs
    /// [`NORM_MEAN`] / [`NORM_INV_STD`] (`[1, 512, 1]` f32 each; see [`ContentEncoder::set_norm`]).
    norm_inputs: bool,
}

/// Input of a `stream` variant: the per-channel mean of the first convolution's output.
pub const NORM_MEAN: &str = "gn_mean";
/// Input of a `stream` variant: the per-channel `1 / sqrt(var + 1e-5)` of the first convolution's
/// output.
pub const NORM_INV_STD: &str = "gn_inv_std";
/// Channels of the first convolution (HuBERT-base front ends).
pub const NORM_CHANNELS: usize = 512;

impl EncoderIo {
    fn of(session: &Session, kind: EmbedderKind, path: &Path) -> Result<EncoderIo, String> {
        let outputs = rvc::graph_outputs(session);
        let output = rvc::feature_output_index(&outputs, kind.channels());
        let is_norm = |n: &str| n == NORM_MEAN || n == NORM_INV_STD;
        let norm_inputs = session.inputs().iter().filter(|i| is_norm(i.name())).count() == 2;
        let input = session
            .inputs()
            .iter()
            .find(|i| !is_norm(i.name()))
            .ok_or_else(|| format!("content encoder {} declares no audio input", path.display()))?;
        let input_name = input.name().to_string();
        let half_input = match input.dtype().tensor_type() {
            Some(TensorElementType::Float32) => false,
            Some(TensorElementType::Float16) => true,
            other => return Err(format!("content encoder {} takes {other:?} audio (expected float)", path.display())),
        };
        let input_rank = input.dtype().tensor_shape().map(|s| s.len()).unwrap_or(2);
        if !(1..=2).contains(&input_rank) {
            return Err(format!(
                "content encoder {} takes a rank-{input_rank} input (expected [N] or [1, N])",
                path.display()
            ));
        }
        let input_free = match input.dtype() {
            ValueType::Tensor { shape, dimension_symbols, .. } => shape
                .iter()
                .enumerate()
                .filter(|(_, d)| **d < 0)
                .map(|(i, _)| (i, dimension_symbols.get(i).cloned().unwrap_or_default()))
                .collect(),
            _ => Vec::new(),
        };
        let (output_name, output_f32, output_dims) = outputs
            .get(output)
            .map(|o| (o.name.clone(), o.dtype == Some(TensorElementType::Float32), o.dims.clone()))
            .unwrap_or_default();
        Ok(EncoderIo {
            input_name,
            input_rank,
            half_input,
            input_free,
            output,
            output_name,
            output_f32,
            output_dims,
            norm_inputs,
        })
    }

    /// The I/O of a Sigil export from its contract (`embedder.json`: `io.input` `{name, shape:
    /// [1, "N"], dtype: float32}` and `io.output` `{name, shape: [1, "T", C], dtype: float32}`).
    fn from_contract(path: &Path, kind: EmbedderKind) -> Result<EncoderIo, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        let io = v.get("io").ok_or_else(|| format!("{} has no io contract", path.display()))?;
        let side = |k: &str| -> Result<(String, Vec<serde_json::Value>, String), String> {
            let x = io.get(k).ok_or_else(|| format!("{}: no io.{k}", path.display()))?;
            let name = x.get("name").and_then(|n| n.as_str()).unwrap_or_default().to_string();
            let shape = x.get("shape").and_then(|n| n.as_array()).cloned().unwrap_or_default();
            let dtype = x.get("dtype").and_then(|n| n.as_str()).unwrap_or("float32").to_string();
            Ok((name, shape, dtype))
        };
        let (input_name, in_shape, in_dtype) = side("input")?;
        let (output_name, out_shape, out_dtype) = side("output")?;
        if input_name.is_empty() || output_name.is_empty() || in_dtype != "float32" || out_dtype != "float32" {
            return Err(format!("{}: not an fp32 [1, N] -> [1, T, C] contract", path.display()));
        }
        let input_free: Vec<(usize, String)> = in_shape
            .iter()
            .enumerate()
            .filter_map(|(i, d)| d.as_str().map(|s| (i, s.to_string())))
            .collect();
        let dims: Vec<i64> = out_shape.iter().map(|d| d.as_i64().unwrap_or(-1)).collect();
        if in_shape.len() != 2 || input_free.len() != 1 || dims.len() != 3 || dims[2] != kind.channels() as i64 {
            return Err(format!("{}: io {in_shape:?} -> {out_shape:?} is not [1, N] -> [1, T, {}]", path.display(), kind.channels()));
        }
        Ok(EncoderIo {
            input_name,
            input_rank: 2,
            half_input: false,
            input_free,
            output: 0,
            output_name,
            output_f32: true,
            output_dims: Some(dims),
            norm_inputs: false,
        })
    }

    /// The fixed output shapes a plan of `frames` frames may have, with their layouts, from the
    /// declared output dimensions (all three when they do not say).
    fn plan_output_shapes(&self, frames: usize, c: usize) -> Vec<(Vec<usize>, FeatLayout)> {
        let ci = c as i64;
        match self.output_dims.as_deref() {
            Some([1, ch, t]) if *ch == ci && *t != ci => vec![(vec![1, c, frames], FeatLayout::ChannelMajor)],
            Some([_, _, ch]) if *ch == ci => vec![(vec![1, frames, c], FeatLayout::TimeMajor)],
            Some([_, ch]) if *ch == ci => vec![(vec![frames, c], FeatLayout::TimeMajor)],
            _ => vec![
                (vec![1, frames, c], FeatLayout::TimeMajor),
                (vec![frames, c], FeatLayout::TimeMajor),
                (vec![1, c, frames], FeatLayout::ChannelMajor),
            ],
        }
    }

    /// Why this graph cannot run as a fixed-shape plan (`None`: it can).
    fn plan_blocker(&self) -> Option<String> {
        if self.half_input || !self.output_f32 {
            return Some("the encoder file has fp16 audio or feature I/O (only fp32 I/O runs as a fixed-shape plan)".into());
        }
        if self.input_free.is_empty() {
            return Some("the encoder's audio input has no free length to pin".into());
        }
        if self.input_free.iter().any(|(_, s)| s.is_empty()) {
            return Some("the encoder's audio length is an unnamed dimension (it cannot be pinned)".into());
        }
        if self.output_name.is_empty() {
            return Some("the encoder's feature output has no name".into());
        }
        None
    }
}

/// How features are laid out in a plan's output buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FeatLayout {
    /// `[1, T, C]` / `[T, C]`: copied as they are.
    TimeMajor,
    /// `[1, C, T]`: transposed into the time-major features.
    ChannelMajor,
}

/// What a plan is built for: one window length at one set of GPU levers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanKey {
    /// Audio samples (16 kHz) of every call.
    pub len: usize,
    pub levers: GpuLevers,
}

/// A voice's hold on the fixed-shape plan of its key (kept in the engine's content state). A plan
/// nobody holds is dropped by the encoder. A key whose plan was refused holds no plan: the voice
/// runs the dynamic session for it without asking again.
#[derive(Debug)]
pub struct ContentLease {
    key: PlanKey,
    token: Option<Arc<()>>,
}

impl ContentLease {
    pub fn key(&self) -> PlanKey {
        self.key
    }

    /// Whether the key runs a fixed-shape plan (`false`: the dynamic session).
    pub fn fixed(&self) -> bool {
        self.token.is_some()
    }
}

/// What [`ContentEncoder::encode_fixed`] reports the first time a voice runs a plan (or falls back
/// from one): the outcome of the session it runs on and the notes for the load warnings. Built at
/// load time only (it allocates).
#[derive(Clone, Debug, PartialEq)]
pub struct PlanReport {
    /// Whether the call ran on a fixed-shape plan (`false`: the dynamic session, see `notes`).
    pub fixed: bool,
    /// The EP outcome of the session the encoder now runs (the plan's, or the dynamic session's).
    pub outcome: EpOutcome,
    /// The levers the plan really uses (`GpuLevers::default()` on the dynamic path).
    pub levers: GpuLevers,
    /// VRAM the plan added when it was built (MiB; `None` when not measured or shared).
    pub vram_mb: Option<f64>,
    /// Why requested levers are off, or why the plan is not used.
    pub notes: Vec<String>,
}

struct FixedPlan {
    key: PlanKey,
    plan: SessionPlan,
    frames: usize,
    layout: FeatLayout,
    /// Leases on this plan (`strong_count - 1` voices run it).
    users: Arc<()>,
    vram_mb: Option<f64>,
    /// Why the plan was accepted without its CPU cross-check (`None`: checked, or switched off).
    check_note: Option<String>,
}

/// What a deferred plan's cross-check is keyed by: the encoder file (its size and time stamp too, so
/// a re-exported file is checked again), the rung and device it runs on, and the plan (window
/// length and levers).
#[derive(Clone, Debug, PartialEq, Eq)]
struct CheckKey {
    path: PathBuf,
    stamp: (u64, Option<std::time::SystemTime>),
    ep: &'static str,
    device: i32,
    plan: PlanKey,
}

/// The outcome of a cross-check, kept for the process.
#[derive(Clone, Debug)]
enum Verdict {
    /// Within the limit (the relative L2 measured).
    Pass(f64),
    /// Refused, with the reason (a rebuilt encoder does not build the same broken plan again).
    Fail(String),
}

#[cfg(test)]
thread_local! {
    /// References started by this thread (tests: a verdict on record must not start another).
    static REFERENCES_STARTED: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

fn verdicts() -> &'static Mutex<Vec<(CheckKey, Verdict)>> {
    static VERDICTS: OnceLock<Mutex<Vec<(CheckKey, Verdict)>>> = OnceLock::new();
    VERDICTS.get_or_init(|| Mutex::new(Vec::new()))
}

fn verdict_of(key: &CheckKey) -> Option<Verdict> {
    verdicts()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.clone())
}

fn remember_verdict(key: CheckKey, verdict: Verdict) {
    let mut all = verdicts().lock().unwrap_or_else(PoisonError::into_inner);
    all.retain(|(k, _)| *k != key);
    all.push((key, verdict));
}

/// A CPU session of the plan's file running the plan's window on a helper thread, started before
/// the plan builds so the two overlap (the session build and one run take about as long as the
/// plan's build, and use no VRAM).
struct ReferenceJob {
    handle: std::thread::JoinHandle<Result<Vec<f32>, String>>,
}

/// The CPU session of `path` the cross-check compares plans with, and its I/O. No component: the
/// check's session records no EP outcome of its own (the encoder's `vc.content` row describes the
/// GPU session). A few threads ([`CHECK_THREADS`]); the Streamer profile's single thread would
/// make the check as slow as the voice's load.
fn cpu_reference_session(kind: EmbedderKind, path: &Path, opts: &VcSessionOptions) -> Result<(Session, EncoderIo), String> {
    let opts = VcSessionOptions {
        streamer_profile: false,
        intra_threads: Some(CHECK_THREADS),
        ..opts.clone()
    };
    let (session, _) = build_session(None, EP_CPU, &opts, ModelSource::File(path))
        .map_err(|e| format!("the CPU session of {} could not be built: {e}", path.display()))?;
    let io = EncoderIo::of(&session, kind, path)?;
    Ok((session, io))
}

impl ReferenceJob {
    fn start(kind: EmbedderKind, path: PathBuf, opts: &VcSessionOptions, mono_16k: &[f32], norm: &[f32]) -> Result<ReferenceJob, String> {
        let opts = opts.clone();
        let (window, norm) = (mono_16k.to_vec(), norm.to_vec());
        std::thread::Builder::new()
            .name("vc-content-check".into())
            .spawn(move || {
                let (mut session, io) = cpu_reference_session(kind, &path, &opts)?;
                let (mut half, mut out) = (Vec::new(), Vec::new());
                run_session(&mut session, &io, kind, &path, &window, &norm, &mut half, &mut out)?;
                Ok(out)
            })
            .map(|handle| ReferenceJob { handle })
            .map_err(|e| format!("the check thread could not start: {e}"))
    }

    fn finish(self) -> Result<Vec<f32>, String> {
        self.handle
            .join()
            .unwrap_or_else(|_| Err("the check thread panicked".into()))
    }
}

/// The cross-check's CPU reference for each of `windows` (one session for all; fp32 audio, no
/// statistics), with the session's build time in ms: the dev-rig calibration of
/// [`PLAN_CPU_CHECK_REL_L2`] (`content_plan_cpu_reference_dev_rig`).
#[cfg(test)]
pub(crate) fn cpu_references_for_tests(
    kind: EmbedderKind,
    path: &Path,
    opts: &VcSessionOptions,
    windows: &[Vec<f32>],
) -> Result<(Vec<Vec<f32>>, f64), String> {
    let t = std::time::Instant::now();
    let (mut session, io) = cpu_reference_session(kind, path, opts)?;
    let build_ms = t.elapsed().as_secs_f64() * 1000.0;
    let mut half = Vec::new();
    let mut all = Vec::new();
    for w in windows {
        let mut out = Vec::new();
        run_session(&mut session, &io, kind, path, w, &[], &mut half, &mut out)?;
        all.push(out);
    }
    Ok((all, build_ms))
}

/// How the output of a deferred plan is checked beyond being finite and not silent.
enum CrossCheck {
    /// No check ran (switched off, or the reference could not start: the note says why, if any).
    Off(Option<String>),
    /// A passing verdict for exactly this plan is on record (the relative L2 it measured).
    Known(f64),
    /// The reference is running.
    Pending(ReferenceJob, CheckKey),
    /// The reference finished.
    Done(Result<Vec<f32>, String>, CheckKey),
}

impl CrossCheck {
    /// Wait for a running reference.
    fn settle(self) -> CrossCheck {
        match self {
            CrossCheck::Pending(job, key) => CrossCheck::Done(job.finish(), key),
            other => other,
        }
    }
}

/// Input lengths the dynamic session has run (a debug counter of the shapes a pooled session saw:
/// DirectML specialises a dynamic session to the shapes it runs). Fixed size, no allocation.
#[derive(Clone, Copy, Debug, Default)]
struct LengthLog {
    lengths: [usize; 8],
    n: usize,
    /// Calls whose length was not among the first eight.
    other: u64,
}

impl LengthLog {
    /// Record `len`; `true` when the session had not run it before.
    fn note(&mut self, len: usize) -> bool {
        if self.lengths[..self.n].contains(&len) {
            return false;
        }
        if self.n < self.lengths.len() {
            self.lengths[self.n] = len;
            self.n += 1;
        } else {
            self.other += 1;
        }
        true
    }
}

/// One loaded encoder (behind the pool's lock; shared by every hot slot using it): its dynamic
/// session and the fixed-shape plans the template engine runs (see the module docs).
pub struct ContentEncoder {
    kind: EmbedderKind,
    channels: usize,
    /// The dynamic-shape session; `None` once every holder runs a plan (rebuilt on demand).
    session: Option<Session>,
    io: EncoderIo,
    half_in: Vec<f16>,
    outcome: EpOutcome,
    path: PathBuf,
    variant: &'static str,
    // ---- P3B-D: fixed-shape plans ----
    /// The EP ladder the encoder was requested on, and the session options (plans are built on the
    /// same ladder, and the dynamic session is rebuilt with them).
    ep: String,
    opts: VcSessionOptions,
    plans: Vec<FixedPlan>,
    /// Plans that could not be built (never retried for that key), with the reason.
    refused: Vec<(PlanKey, String)>,
    /// Calls of [`encode`](Self::encode) (the dynamic path) since the encoder was loaded.
    dynamic_calls: u64,
    lengths: LengthLog,
    /// A `stream` variant's statistics for the next call: means then inverse standard deviations
    /// ([`NORM_CHANNELS`] each; [`set_norm`](Self::set_norm)).
    norm: Vec<f32>,
    /// Loaded without a dynamic session ([`ContentEncoder::deferred`]): the I/O comes from the export
    /// contract and the outcome from the first session built (a plan, or the dynamic session).
    deferred: bool,
    /// The CPU variant of the same store entry, for a GPU variant whose session lands on the CPU.
    cpu_fallback: Option<PathBuf>,
    /// The file a deferred plan's CPU cross-check runs (`None`: the encoder's own file; tests point
    /// it at another graph to see a wrong plan refused).
    check_file: Option<PathBuf>,
}

impl ContentEncoder {
    /// Build the encoder session for `path` on the EP ladder of `ep` (outcome `vc.content`).
    pub fn load(kind: EmbedderKind, path: &Path, variant: &'static str, ep: &str, opts: &VcSessionOptions) -> Result<ContentEncoder, String> {
        let (session, outcome) = Self::build_dynamic(path, ep, opts)?;
        // A pitch model, a voice or a fused pipeline set as the content encoder is named for what it
        // is (P5-D's graph identification; load time only).
        if let Some(why) = rvc::foreign_graph_problem(&session, path, "a content encoder", &[rvc::GraphKind::ContentEncoder]) {
            return Err(format!("content encoder: {why}"));
        }
        let io = EncoderIo::of(&session, kind, path)?;
        Ok(ContentEncoder {
            kind,
            channels: kind.channels(),
            session: Some(session),
            io,
            half_in: Vec::new(),
            outcome,
            path: path.to_path_buf(),
            variant,
            ep: ep.to_string(),
            opts: opts.clone(),
            plans: Vec::new(),
            refused: Vec::new(),
            dynamic_calls: 0,
            lengths: LengthLog::default(),
            norm: Vec::new(),
            deferred: false,
            cpu_fallback: None,
            check_file: None,
        })
    }

    /// An encoder of a Sigil store variant without a session yet (P3B-D): the template engine's
    /// first tick builds its fixed-shape plan directly, so a GPU voice never builds (and then
    /// releases) a dynamic session it does not use. The I/O comes from the export contract
    /// (`embedder.json` next to the file: `io.input` / `io.output` names and shapes); the outcome is
    /// the requested rung until the first session says where it runs. A dynamic caller builds the
    /// dynamic session on its first call (swapping in `cpu_fallback` when a GPU variant lands on the
    /// CPU, like [`ContentEncoderPool::acquire_resolved`]).
    pub fn deferred(
        kind: EmbedderKind,
        path: &Path,
        variant: &'static str,
        ep: &str,
        opts: &VcSessionOptions,
        cpu_fallback: Option<PathBuf>,
    ) -> Result<ContentEncoder, String> {
        if !path.is_file() {
            return Err(format!("content encoder not found: {}", path.display()));
        }
        let io = EncoderIo::from_contract(&path.with_file_name("embedder.json"), kind)?;
        let active = normalize_ep(ep);
        Ok(ContentEncoder {
            kind,
            channels: kind.channels(),
            session: None,
            io,
            half_in: Vec::new(),
            outcome: EpOutcome::new(active, active, None),
            path: path.to_path_buf(),
            variant,
            ep: ep.to_string(),
            opts: opts.clone(),
            plans: Vec::new(),
            refused: Vec::new(),
            dynamic_calls: 0,
            lengths: LengthLog::default(),
            norm: Vec::new(),
            deferred: true,
            cpu_fallback,
            check_file: None,
        })
    }

    /// Whether the encoder was loaded without a session and none has been built yet.
    pub fn is_deferred(&self) -> bool {
        self.deferred
    }

    /// Build the dynamic session when it is not loaded (released, or never built for a deferred
    /// encoder). A deferred encoder takes its outcome and I/O from it, and a GPU variant that landed
    /// on the CPU is rebuilt from the entry's CPU variant.
    fn ensure_dynamic(&mut self) -> Result<(), String> {
        if self.session.is_some() {
            return Ok(());
        }
        let (session, outcome) = Self::build_dynamic(&self.path, &self.ep, &self.opts)?;
        self.lengths = LengthLog::default();
        if !self.deferred {
            self.session = Some(session);
            return Ok(());
        }
        self.deferred = false;
        let gpu_variant_on_cpu = outcome.active == EP_CPU && normalize_ep(&self.ep) != EP_CPU && matches!(self.variant, "fp16" | "stream.fp16");
        match self.cpu_fallback.clone().filter(|_| gpu_variant_on_cpu) {
            Some(cpu) => {
                drop(session);
                let (s2, _) = Self::build_dynamic(&cpu, EP_CPU, &self.opts)?;
                let name = cpu.to_string_lossy().into_owned();
                self.variant = if name.contains(".stream.") {
                    "stream.fp32"
                } else if name.contains(".int8.") {
                    "int8"
                } else {
                    "fp32"
                };
                self.io = EncoderIo::of(&s2, self.kind, &cpu)?;
                self.path = cpu;
                self.session = Some(s2);
                // Keep the request and its fallback reason (the rebuild recorded a plain CPU session).
                self.outcome = outcome;
                crate::services::diagnostics::record_ep_outcome(&super::generator::component_key(&self.opts, "vc.content"), self.outcome.clone());
            }
            None => {
                self.io = EncoderIo::of(&session, self.kind, &self.path)?;
                self.outcome = outcome;
                self.session = Some(session);
            }
        }
        Ok(())
    }

    fn build_dynamic(path: &Path, ep: &str, opts: &VcSessionOptions) -> Result<(Session, EpOutcome), String> {
        let _span = telemetry::load_span(LoadStage::ContentSessionInit);
        build_session(Some("vc.content"), ep, opts, ModelSource::File(path)).map_err(|e| format!("content encoder {}: {e}", path.display()))
    }

    pub fn kind(&self) -> EmbedderKind {
        self.kind
    }

    /// Feature channels per frame (768 or 256).
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Where the session runs.
    pub fn outcome(&self) -> &EpOutcome {
        &self.outcome
    }

    /// The model file and its variant (`fp16`, `int8`, `explicit`, ...).
    pub fn source(&self) -> (&Path, &'static str) {
        (&self.path, self.variant)
    }

    /// Distinct input lengths the dynamic session has run (the first eight; diagnostics and the
    /// DirectML first-shape measurement).
    pub fn dynamic_lengths(&self) -> &[usize] {
        &self.lengths.lengths[..self.lengths.n]
    }

    /// Whether the dynamic session is loaded (it is released once every holder runs a plan).
    pub fn has_dynamic_session(&self) -> bool {
        self.session.is_some()
    }

    /// Fixed-shape plans currently built (one per length and lever set in use).
    pub fn plan_count(&self) -> usize {
        self.plans.len()
    }

    /// Why this encoder cannot run fixed-shape plans (`None`: it can).
    pub fn plan_blocker(&self) -> Option<String> {
        self.io.plan_blocker()
    }

    /// Whether the graph is a `stream` variant (it takes the first GroupNorm's statistics).
    pub fn takes_norm(&self) -> bool {
        self.io.norm_inputs
    }

    /// The first GroupNorm's statistics for the next calls of a `stream` variant: per-channel means
    /// and inverse standard deviations ([`NORM_CHANNELS`] each). Allocation-free once set.
    pub fn set_norm(&mut self, mean: &[f32], inv_std: &[f32]) -> Result<(), String> {
        if mean.len() != NORM_CHANNELS || inv_std.len() != NORM_CHANNELS {
            return Err(format!(
                "content statistics: {} / {} channels, the encoder takes {NORM_CHANNELS}",
                mean.len(),
                inv_std.len()
            ));
        }
        self.norm.clear();
        self.norm.extend_from_slice(mean);
        self.norm.extend_from_slice(inv_std);
        Ok(())
    }

    fn norm_ready(&self) -> Result<(), String> {
        if self.io.norm_inputs && self.norm.len() != 2 * NORM_CHANNELS {
            return Err("the stream content encoder needs its normalisation statistics (set_norm) before a call".into());
        }
        Ok(())
    }

    /// Release the dynamic session while nothing runs the dynamic path on this encoder (the voice
    /// moved to its `stream` variant). A later dynamic caller rebuilds it.
    pub fn park(&mut self) {
        if self.dynamic_calls == 0 && self.plans.is_empty() {
            self.session = None;
        }
    }

    /// Encode `mono_16k` into `out` (`frames × channels`, resized by the encoder, capacity kept
    /// across calls) with the dynamic session; returns the frame count.
    pub fn encode(&mut self, mono_16k: &[f32], out: &mut Vec<f32>) -> Result<usize, String> {
        self.dynamic_calls += 1;
        self.encode_dynamic(mono_16k, out)
    }

    fn encode_dynamic(&mut self, mono_16k: &[f32], out: &mut Vec<f32>) -> Result<usize, String> {
        let n = mono_16k.len();
        // Released while every holder ran a plan (or never built: a deferred encoder); a caller of
        // the dynamic path brings it back.
        self.ensure_dynamic()?;
        if self.lengths.note(n) && self.lengths.n > 1 {
            // A dynamic session on a new length (DirectML re-plans for it): rare, so worth a line.
            tracing::debug!(
                target: "sigil::vc",
                "content encoder {}: the dynamic session runs a new input length {n} (lengths so far {:?})",
                self.path.display(),
                self.dynamic_lengths()
            );
        }
        self.norm_ready()?;
        let session = self.session.as_mut().ok_or("the content encoder's session is not loaded")?;
        let frames = run_session(session, &self.io, self.kind, &self.path, mono_16k, &self.norm, &mut self.half_in, out)?;
        // An eager run with host tensors (P3B-S0 telemetry: one GPU wait on a GPU rung).
        telemetry::record_sync(SyncKind::EagerRun);
        Ok(frames)
    }

    /// Encode `mono_16k` with the fixed-shape plan of its length at `levers` (building it on the
    /// first call; load time): `lease` is the calling voice's hold on its plan (replaced when the
    /// key changes). Returns the frame count and, when this call built, joined or fell back from a
    /// plan, its report. A plan that cannot be built (or misses the build check) leaves this length
    /// on the dynamic session, for good, with the reason in the report. Allocation-free once the
    /// plan runs.
    pub fn encode_fixed(
        &mut self,
        mono_16k: &[f32],
        levers: GpuLevers,
        lease: &mut Option<ContentLease>,
        out: &mut Vec<f32>,
    ) -> Result<(usize, Option<PlanReport>), String> {
        let key = PlanKey { len: mono_16k.len(), levers };
        let mut report = None;
        let leased = lease.as_ref().is_some_and(|l| l.key == key);
        if !leased {
            // A new key for this voice: join (or build) its plan, let go of the old one.
            *lease = None;
            report = Some(self.join_plan(key, mono_16k, lease));
            self.prune_plans();
            self.release_dynamic_if_unused();
        } else if self.plans.len() > 1 {
            // Another voice moved off a plan (or was dropped): free what nobody leases.
            self.prune_plans();
        }
        let plan = lease
            .as_ref()
            .and_then(|l| l.token.as_ref())
            .and_then(|tok| self.plans.iter().position(|p| Arc::ptr_eq(&p.users, tok)));
        let Some(i) = plan else {
            // The dynamic path for this key (no plan for this encoder or key).
            return self.encode_dynamic(mono_16k, out).map(|n| (n, report));
        };
        self.norm_ready()?;
        let p = &mut self.plans[i];
        p.plan
            .input_f32(&self.io.input_name)?
            .copy_from_slice(mono_16k);
        if self.io.norm_inputs {
            let (m, v) = self.norm.split_at(NORM_CHANNELS);
            p.plan.input_f32(NORM_MEAN)?.copy_from_slice(m);
            p.plan.input_f32(NORM_INV_STD)?.copy_from_slice(v);
        }
        p.plan.run().map_err(|e| format!("content encoder plan: {e}"))?;
        let data = p.plan.output(&self.io.output_name)?;
        copy_features(data, p.frames, self.channels, p.layout, out);
        Ok((p.frames, report))
    }

    /// Join the plan of `key` (building it if needed) and put the lease in `lease` (a lease without
    /// a plan when the key runs the dynamic session). Returns the report for the caller's warnings.
    fn join_plan(&mut self, key: PlanKey, mono_16k: &[f32], lease: &mut Option<ContentLease>) -> PlanReport {
        let dynamic = |s: &ContentEncoder, notes: Vec<String>| PlanReport {
            fixed: false,
            outcome: s.outcome.clone(),
            levers: GpuLevers::default(),
            vram_mb: None,
            notes,
        };
        if let Some(why) = self.io.plan_blocker() {
            *lease = Some(ContentLease { key, token: None });
            return dynamic(self, vec![format!("content plan: {why}; the content encoder runs at a dynamic shape")]);
        }
        if let Some((_, why)) = self.refused.iter().find(|(k, _)| *k == key) {
            *lease = Some(ContentLease { key, token: None });
            return dynamic(self, vec![why.clone()]);
        }
        if let Some(p) = self.plans.iter().find(|p| p.key == key) {
            *lease = Some(ContentLease { key, token: Some(Arc::clone(&p.users)) });
            return PlanReport {
                fixed: true,
                outcome: p.plan.outcome().clone(),
                levers: p.plan.levers(),
                vram_mb: None,
                notes: p.plan.notes().iter().map(|n| format!("content plan: {n}")).collect(),
            };
        }
        match self.build_plan(key, mono_16k) {
            Ok(p) => {
                if self.deferred && self.session.is_none() {
                    // The plan is the encoder's first session: it says where the encoder runs.
                    self.outcome = p.plan.outcome().clone();
                }
                let mut notes: Vec<String> = p.plan.notes().iter().map(|n| format!("content plan: {n}")).collect();
                notes.extend(p.check_note.clone());
                let report = PlanReport {
                    fixed: true,
                    outcome: p.plan.outcome().clone(),
                    levers: p.plan.levers(),
                    vram_mb: p.vram_mb,
                    notes,
                };
                *lease = Some(ContentLease { key, token: Some(Arc::clone(&p.users)) });
                self.plans.push(p);
                report
            }
            Err(why) => {
                let note = format!("content plan: {why}; the content encoder runs at a dynamic shape for this stream");
                tracing::warn!(target: "sigil::vc", "content encoder {}: {note}", self.path.display());
                self.refused.push((key, note.clone()));
                *lease = Some(ContentLease { key, token: None });
                // The dynamic session serves this key: make sure it is there.
                if let Err(e) = self.ensure_dynamic() {
                    tracing::warn!(target: "sigil::vc", "content encoder: building the dynamic session failed: {e}");
                }
                dynamic(self, vec![note])
            }
        }
    }

    /// Build and check the plan of `key` on the window `mono_16k`.
    fn build_plan(&mut self, key: PlanKey, mono_16k: &[f32]) -> Result<FixedPlan, String> {
        let len = key.len;
        if len < 400 {
            return Err(format!("a {len}-sample window is shorter than the encoder's 400-sample front end"));
        }
        self.norm_ready()?;
        // The reference (and the output geometry) from the dynamic session on the same window.
        let mut reference = Vec::new();
        let reference_frames = match self.session {
            Some(_) => Some(self.encode_dynamic(mono_16k, &mut reference)?),
            None => None,
        };
        // Without a dynamic session (a deferred encoder, or one whose session was released) the
        // reference is a CPU session of the file on a helper thread, running while the plan builds;
        // a verdict on record for this exact plan (or a refusal) settles it at once.
        let mut cross = match reference_frames {
            Some(_) => CrossCheck::Off(None),
            None => self.begin_cross_check(key, mono_16k)?,
        };
        let mut check_note = None;
        let c = self.channels;
        let frames = reference_frames.unwrap_or((len - 400) / 320 + 1);
        let input_shape: Vec<usize> = if self.io.input_rank == 1 { vec![len] } else { vec![1, len] };
        let dims: Vec<(&str, i64)> = self
            .io
            .input_free
            .iter()
            .map(|(axis, sym)| (sym.as_str(), if self.io.input_rank == 2 && *axis == 0 { 1 } else { len as i64 }))
            .collect();
        // The declared output layout (every Sigil export: time-major); when the graph does not say,
        // the time-major layouts first, then channel-major.
        let session_ep = self.outcome.active.to_string();
        let mut last_err = String::new();
        for (shape, layout) in self.io.plan_output_shapes(frames, c) {
            let mut inputs = vec![TensorSpec::f32(&self.io.input_name, &input_shape)];
            if self.io.norm_inputs {
                inputs.push(TensorSpec::f32(NORM_MEAN, &[1, NORM_CHANNELS, 1]));
                inputs.push(TensorSpec::f32(NORM_INV_STD, &[1, NORM_CHANNELS, 1]));
            }
            let outputs = [TensorSpec::f32(&self.io.output_name, &shape)];
            let built = {
                let _span = telemetry::load_span(LoadStage::ContentSessionInit);
                SessionPlan::build(&PlanSpec {
                    component: "vc.content",
                    ep: &self.ep,
                    opts: &self.opts,
                    model: ModelSource::File(&self.path),
                    weights: None,
                    dims: &dims,
                    inputs: &inputs,
                    outputs: &outputs,
                    levers: key.levers,
                })
            };
            let mut plan = match built {
                Ok(p) => p,
                Err(e) => {
                    last_err = e;
                    continue;
                }
            };
            // The plan must run where the dynamic session runs (a GPU variant that lands on the
            // CPU is the dynamic path's business: it swaps in the CPU variant). A deferred encoder
            // has no session yet: its plan must not have fallen from a GPU request to the CPU.
            let misplaced = if self.deferred && self.session.is_none() {
                plan.outcome().active == EP_CPU && normalize_ep(&self.ep) != EP_CPU
            } else {
                plan.outcome().active != session_ep
            };
            if misplaced {
                return Err(format!(
                    "the plan's session runs on {} while the encoder runs on {session_ep}{}",
                    plan.outcome().active,
                    plan.outcome().reason.as_deref().map(|r| format!(" ({r})")).unwrap_or_default()
                ));
            }
            let mut run = plan.input_f32(&self.io.input_name).map(|x| x.copy_from_slice(mono_16k));
            if self.io.norm_inputs {
                let (m, v) = self.norm.split_at(NORM_CHANNELS);
                run = run
                    .and_then(|()| plan.input_f32(NORM_MEAN).map(|x| x.copy_from_slice(m)))
                    .and_then(|()| plan.input_f32(NORM_INV_STD).map(|x| x.copy_from_slice(v)));
            }
            let run = run.and_then(|()| plan.run());
            if let Err(e) = run {
                // A wrong output shape fails here; try the next layout.
                last_err = e;
                continue;
            }
            let mut got = Vec::with_capacity(frames * c);
            copy_features(plan.output(&self.io.output_name)?, frames, c, layout, &mut got);
            if reference_frames.is_some() {
                let rel = rel_l2(&got, &reference);
                if rel.is_nan() || rel > PLAN_CHECK_REL_L2 {
                    // A layout guessed wrong reads scrambled features: try the next one.
                    last_err = format!("the fixed-shape plan differs from the dynamic session by {rel:.2e} relative L2 (limit {PLAN_CHECK_REL_L2:.0e})");
                    continue;
                }
                tracing::info!(
                    target: "sigil::vc",
                    "content plan [{}] {len} samples on {}: levers {:?}, check rel L2 {rel:.2e}{}",
                    self.kind.as_str(),
                    plan.outcome().active,
                    plan.levers(),
                    plan.vram_mb().map(|mb| format!(", {mb:.0} MB VRAM")).unwrap_or_default()
                );
            } else if !got.iter().all(|v| v.is_finite()) || got.iter().all(|&v| v == 0.0) {
                // A deferred encoder has no dynamic session to compare with (its layout comes from
                // the export contract): a plan whose features are not finite, or all zero (a graph
                // that computed nothing), is refused.
                last_err = "the fixed-shape plan returned non-finite or all-zero features".into();
                continue;
            } else {
                // Finite and not silent; now against the CPU session of the same file.
                cross = cross.settle();
                let checked = match &cross {
                    CrossCheck::Known(rel) => format!(", cross-check on record (rel L2 {rel:.2e})"),
                    CrossCheck::Done(Ok(cpu), ck) => {
                        let rel = rel_l2(&got, cpu);
                        if rel.is_nan() || rel > PLAN_CPU_CHECK_REL_L2 {
                            let why = format!(
                                "the fixed-shape plan differs from a CPU session of the same file by {rel:.2e} relative L2 (limit {PLAN_CPU_CHECK_REL_L2:.0e}), so its compile on {} is not trusted",
                                plan.outcome().active
                            );
                            remember_verdict(ck.clone(), Verdict::Fail(why.clone()));
                            last_err = why;
                            continue;
                        }
                        remember_verdict(ck.clone(), Verdict::Pass(rel));
                        format!(", cross-checked against the CPU (rel L2 {rel:.2e})")
                    }
                    CrossCheck::Done(Err(e), _) => {
                        // The reference itself failed (no memory, an op the CPU cannot run): no
                        // verdict, and the plan stands on its finite, not-silent output.
                        tracing::warn!(target: "sigil::vc", "content plan [{}]: the CPU cross-check could not run: {e}", self.kind.as_str());
                        check_note = Some(format!("content plan: not cross-checked against a CPU session ({e}); only its finite, not-silent output was tested"));
                        ", cross-check failed to run".to_string()
                    }
                    CrossCheck::Off(note) => {
                        check_note = note.clone();
                        ", cross-check off".to_string()
                    }
                    CrossCheck::Pending(..) => unreachable!("settled above"),
                };
                tracing::info!(
                    target: "sigil::vc",
                    "content plan [{}] {len} samples on {} (deferred, no dynamic session): levers {:?}{}{checked}",
                    self.kind.as_str(),
                    plan.outcome().active,
                    plan.levers(),
                    plan.vram_mb().map(|mb| format!(", {mb:.0} MB VRAM")).unwrap_or_default()
                );
            }
            let vram_mb = plan.vram_mb();
            if let Some(mb) = vram_mb {
                telemetry::vram_mark("vc.content", mb);
            }
            return Ok(FixedPlan {
                key,
                plan,
                frames,
                layout,
                users: Arc::new(()),
                vram_mb,
                check_note,
            });
        }
        Err(format!("no fixed-shape plan ({last_err})"))
    }

    /// The CPU cross-check of a plan that has no dynamic session to compare with: a verdict on
    /// record for exactly this plan answers it (`Err`: a recorded refusal, the plan is not built);
    /// otherwise the reference starts on a helper thread. The kill switch
    /// ([`CONTENT_CHECK_ENV`]) or a thread that cannot start leaves only the finite / not-silent
    /// test.
    fn begin_cross_check(&self, key: PlanKey, mono_16k: &[f32]) -> Result<CrossCheck, String> {
        if !content_check_switch() {
            return Ok(CrossCheck::Off(None));
        }
        let file = self.check_file.clone().unwrap_or_else(|| self.path.clone());
        let ck = self.check_key(&file, key);
        match verdict_of(&ck) {
            Some(Verdict::Fail(why)) => return Err(why),
            Some(Verdict::Pass(rel)) => return Ok(CrossCheck::Known(rel)),
            None => {}
        }
        #[cfg(test)]
        REFERENCES_STARTED.with(|c| c.set(c.get() + 1));
        Ok(match ReferenceJob::start(self.kind, file, &self.opts, mono_16k, &self.norm) {
            Ok(job) => CrossCheck::Pending(job, ck),
            Err(e) => CrossCheck::Off(Some(format!("content plan: not cross-checked against a CPU session ({e})"))),
        })
    }

    fn check_key(&self, file: &Path, plan: PlanKey) -> CheckKey {
        let meta = std::fs::metadata(file).ok();
        CheckKey {
            path: file.to_path_buf(),
            stamp: (meta.as_ref().map_or(0, |m| m.len()), meta.and_then(|m| m.modified().ok())),
            ep: normalize_ep(&self.ep),
            device: self.opts.cuda_device_index,
            plan,
        }
    }

    /// Point the CPU cross-check at another file than the encoder's own (tests: a wrong graph must
    /// get the plan refused).
    #[cfg(test)]
    pub(crate) fn set_check_file(&mut self, file: PathBuf) {
        self.check_file = Some(file);
    }

    /// Drop plans no voice leases any more.
    fn prune_plans(&mut self) {
        self.plans.retain(|p| Arc::strong_count(&p.users) > 1);
    }

    /// Release the dynamic session while only plans use the encoder (no caller of the dynamic
    /// path, [`encode`](Self::encode), since it loaded, and no refused plan), so a plan never
    /// doubles the encoder's VRAM. A later dynamic caller (another voice family sharing the pooled
    /// encoder) rebuilds it once, at its own load, and from then on it stays.
    fn release_dynamic_if_unused(&mut self) {
        if self.session.is_some() && self.dynamic_calls == 0 && self.refused.is_empty() && !self.plans.is_empty() && release_dynamic_on(&self.outcome.active) {
            self.session = None;
            tracing::debug!(target: "sigil::vc", "content encoder {}: only fixed-shape plans run it; the dynamic session is released", self.path.display());
        }
    }
}

/// One run of an encoder session on `mono_16k` (fp32 audio converted to fp16 through `half_in` when
/// the graph takes it; the statistics `norm` for a `stream` variant): the time-major features go to
/// `out` (cleared), the frame count is returned. The dynamic path of [`ContentEncoder`] and the plan
/// cross-check's CPU reference both run through it.
#[allow(clippy::too_many_arguments)]
fn run_session(
    session: &mut Session,
    io: &EncoderIo,
    kind: EmbedderKind,
    path: &Path,
    mono_16k: &[f32],
    norm: &[f32],
    half_in: &mut Vec<f16>,
    out: &mut Vec<f32>,
) -> Result<usize, String> {
    let n = mono_16k.len();
    let err = |e: ort::Error| format!("content encoder input: {e}");
    if io.half_input {
        half_in.clear();
        half_in.extend(mono_16k.iter().map(|&x| f16::from_f32(x)));
    }
    let input: SessionInputValue<'_> = match (io.half_input, io.input_rank) {
        (true, 1) => TensorRef::from_array_view(([n], half_in.as_slice())).map_err(err)?.into(),
        (true, _) => TensorRef::from_array_view(([1, n], half_in.as_slice())).map_err(err)?.into(),
        (false, 1) => TensorRef::from_array_view(([n], mono_16k)).map_err(err)?.into(),
        (false, _) => TensorRef::from_array_view(([1, n], mono_16k)).map_err(err)?.into(),
    };
    let outputs = if io.norm_inputs {
        if norm.len() != 2 * NORM_CHANNELS {
            return Err("the stream content encoder needs its normalisation statistics (set_norm) before a call".into());
        }
        let (m, v) = norm.split_at(NORM_CHANNELS);
        let m = TensorRef::from_array_view(([1, NORM_CHANNELS, 1], m)).map_err(err)?;
        let v = TensorRef::from_array_view(([1, NORM_CHANNELS, 1], v)).map_err(err)?;
        session.run(ort::inputs![io.input_name.as_str() => input, NORM_MEAN => m, NORM_INV_STD => v])
    } else {
        session.run(ort::inputs![io.input_name.as_str() => input])
    }
    .map_err(|e| format!("content encoder run: {e}"))?;
    let value = &outputs[io.output];
    let c = kind.channels();
    let dims: Vec<usize>;
    let flat: Vec<f32>;
    let data: &[f32] = match value.try_extract_tensor::<f32>() {
        Ok((shape, data)) => {
            dims = shape.iter().map(|d| (*d).max(0) as usize).collect();
            data
        }
        Err(_) => {
            let (shape, data) = value
                .try_extract_tensor::<f16>()
                .map_err(|e| format!("content encoder output: {e}"))?;
            dims = shape.iter().map(|d| (*d).max(0) as usize).collect();
            flat = data.iter().map(|h| h.to_f32()).collect();
            &flat
        }
    };
    let (frames, layout) = feature_layout(&dims, c).ok_or_else(|| {
        format!(
            "content encoder {} returned {:?}, not {c}-channel features for a {} voice (the slot needs the '{}' encoder)",
            path.display(),
            dims,
            if c == 256 { "v1" } else { "v2" },
            kind.as_str()
        )
    })?;
    copy_features(data, frames, c, layout, out);
    Ok(frames)
}

/// `(frames, layout)` of a feature tensor with `dims` and `c` channels: `[1, T, C]` / `[T, C]`
/// (time-major) or `[1, C, T]` (channel-major).
fn feature_layout(dims: &[usize], c: usize) -> Option<(usize, FeatLayout)> {
    match dims {
        [1, ch, t] if *ch == c && *t != c => Some((*t, FeatLayout::ChannelMajor)),
        [.., t, ch] if *ch == c => Some((*t, FeatLayout::TimeMajor)),
        [ch] if *ch == c => Some((1, FeatLayout::TimeMajor)),
        _ => None,
    }
}

/// `frames × c` time-major features from `data` into `out` (cleared; no allocation once its
/// capacity fits).
fn copy_features(data: &[f32], frames: usize, c: usize, layout: FeatLayout, out: &mut Vec<f32>) {
    out.clear();
    match layout {
        FeatLayout::TimeMajor => out.extend_from_slice(&data[..frames * c]),
        FeatLayout::ChannelMajor => {
            out.resize(frames * c, 0.0);
            for ti in 0..frames {
                for ci in 0..c {
                    out[ti * c + ci] = data[ci * frames + ti];
                }
            }
        }
    }
}

/// Relative L2 distance of `a` from `b` (`‖a − b‖ / ‖b‖`; infinite for different lengths).
pub fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return f64::INFINITY;
    }
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = f64::from(x) - f64::from(y);
        num += d * d;
        den += f64::from(y) * f64::from(y);
    }
    if den == 0.0 {
        return if num == 0.0 { 0.0 } else { f64::INFINITY };
    }
    (num / den).sqrt()
}

/// The GPU levers a content plan runs with, from the engine's (TensorRT-RTX is not validated for the
/// content encoder: it stays on CUDA there).
pub fn plan_levers(effective: &super::EffectiveLevers) -> GpuLevers {
    let mut l = GpuLevers::from_effective(effective);
    if l.accelerator == Accelerator::TensorRtRtx {
        l.accelerator = Accelerator::None;
    }
    l
}

/// Whether a content encoder whose session runs on `active` (its EP outcome) should run plans:
/// GPU rungs only (the CPU Streamer keeps its dynamic session).
pub fn plan_rung(active: &str) -> bool {
    active != EP_CPU && matches!(active, EP_CUDA | EP_DML)
}

/// Whether the dynamic session of an encoder on `active` may be released while plans serve it.
/// DirectML: running a session after another DirectML session of the process was released crashed
/// onnxruntime-directml 1.24.4 (ADR 0006 finding 5); `tick_tests::content_plan_dev_rig` checks it on
/// Sigil's ONNX Runtime by releasing the dynamic session and running the plan afterwards.
fn release_dynamic_on(_active: &str) -> bool {
    true
}

/// A pooled encoder.
pub type SharedEncoder = Arc<Mutex<ContentEncoder>>;

#[derive(Clone, Debug, PartialEq, Eq)]
struct PoolKey {
    kind: EmbedderKind,
    path: PathBuf,
    ep: &'static str,
    device: i32,
    streamer: bool,
    threads: Option<usize>,
    prefix: Option<&'static str>,
}

/// Encoders shared across hot slots, keyed by kind + EP + precision.
pub struct ContentEncoderPool {
    entries: Mutex<Vec<(PoolKey, SharedEncoder)>>,
}

impl ContentEncoderPool {
    pub fn global() -> &'static ContentEncoderPool {
        static POOL: OnceLock<ContentEncoderPool> = OnceLock::new();
        POOL.get_or_init(|| ContentEncoderPool {
            entries: Mutex::new(Vec::new()),
        })
    }

    /// The pooled encoder for `kind` on `ep` (loading it on first use). `model` overrides the
    /// store-resolved file (the UI's explicit HuBERT path).
    pub fn acquire(
        &self,
        kind: EmbedderKind,
        model: Option<&Path>,
        ep: &str,
        opts: &VcSessionOptions,
        precision: PrecisionMode,
    ) -> Result<SharedEncoder, String> {
        let gpu = normalize_ep(ep) != "cpu";
        let resolved = match model {
            Some(p) => ResolvedEncoder {
                path: p.to_path_buf(),
                variant: "explicit",
                cpu_fallback: None,
            },
            None => resolve_encoder_file(kind, None, None, gpu, precision)?,
        };
        self.acquire_resolved(kind, &resolved, ep, opts)
    }

    /// [`acquire`](Self::acquire) for an already resolved file ([`resolve_encoder_file`]).
    pub fn acquire_resolved(
        &self,
        kind: EmbedderKind,
        resolved: &ResolvedEncoder,
        ep: &str,
        opts: &VcSessionOptions,
    ) -> Result<SharedEncoder, String> {
        let key = PoolKey {
            kind,
            path: resolved.path.clone(),
            ep: normalize_ep(ep),
            device: opts.cuda_device_index,
            streamer: opts.streamer_profile,
            threads: opts.intra_threads,
            prefix: opts.component_prefix,
        };
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, enc)) = entries.iter().find(|(k, _)| *k == key) {
            return Ok(enc.clone());
        }
        let mut encoder = ContentEncoder::load(kind, &resolved.path, resolved.variant, ep, opts)?;
        if encoder.outcome.active == "cpu" && key.ep != "cpu" && matches!(resolved.variant, "fp16" | "stream.fp16") {
            // The GPU variant landed on the CPU (the EP ladder fell back): ONNX Runtime's CPU fp16
            // path is slow, so run the CPU variant of the same entry.
            if let Some(cpu) = &resolved.cpu_fallback {
                let name = cpu.to_string_lossy();
                let variant = if name.contains(".stream.") {
                    "stream.fp32"
                } else if name.contains(".int8.") {
                    "int8"
                } else {
                    "fp32"
                };
                let mut rebuilt = ContentEncoder::load(kind, cpu, variant, "cpu", opts)?;
                // Keep the original request and its fallback reason (the rebuild recorded a plain
                // CPU session).
                rebuilt.outcome = encoder.outcome.clone();
                crate::services::diagnostics::record_ep_outcome(&super::generator::component_key(opts, "vc.content"), rebuilt.outcome.clone());
                encoder = rebuilt;
            }
        }
        let shared = Arc::new(Mutex::new(encoder));
        entries.push((key, shared.clone()));
        Ok(shared)
    }

    /// [`acquire_resolved`](Self::acquire_resolved) without building a session for a new entry
    /// ([`ContentEncoder::deferred`]; an existing entry is shared as it is). Fails when the file has
    /// no export contract (the caller then loads it the usual way).
    pub fn acquire_deferred(&self, kind: EmbedderKind, resolved: &ResolvedEncoder, ep: &str, opts: &VcSessionOptions) -> Result<SharedEncoder, String> {
        let key = PoolKey {
            kind,
            path: resolved.path.clone(),
            ep: normalize_ep(ep),
            device: opts.cuda_device_index,
            streamer: opts.streamer_profile,
            threads: opts.intra_threads,
            prefix: opts.component_prefix,
        };
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, enc)) = entries.iter().find(|(k, _)| *k == key) {
            return Ok(enc.clone());
        }
        let encoder = ContentEncoder::deferred(kind, &resolved.path, resolved.variant, ep, opts, resolved.cpu_fallback.clone())?;
        let shared = Arc::new(Mutex::new(encoder));
        entries.push((key, shared.clone()));
        Ok(shared)
    }

    /// Drop encoders no hot slot uses any more (frees their VRAM), and the fixed-shape plans no
    /// voice leases in the others (an encoder busy in a tick prunes its own on its next call; a
    /// voice being dropped still holds its lease here, so its plan goes on the next prune).
    ///
    /// The plans are pruned under the pool's lock, through the pool's own references: taking
    /// clones of the entries to prune them outside the lock would raise their strong counts, so a
    /// release running at the same time on another thread (two voices dropped at once) would see
    /// every encoder as still in use and keep an unused one alive. `try_lock` never blocks, so
    /// holding the pool's lock meanwhile cannot deadlock with a tick that holds an encoder.
    pub fn release_unused(&self) {
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        entries.retain(|(_, enc)| Arc::strong_count(enc) > 1);
        for (_, enc) in entries.iter() {
            match enc.try_lock() {
                Ok(mut e) => e.prune_plans(),
                Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner().prune_plans(),
                Err(std::sync::TryLockError::WouldBlock) => {}
            }
        }
    }

    /// Encoders currently pooled.
    pub fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::services::neural_audio::vc::test_models;

    #[test]
    fn embedder_names_map_to_kinds_and_check_channels() {
        assert_eq!(EmbedderKind::from_manifest(None, 768).unwrap(), EmbedderKind::ContentVec768);
        assert_eq!(EmbedderKind::from_manifest(None, 256).unwrap(), EmbedderKind::ContentVec256);
        assert!(EmbedderKind::from_manifest(None, 512).is_err());
        // RVC / Applio names; a v1 voice named contentvec uses the 256-channel encoder.
        assert_eq!(EmbedderKind::from_manifest(Some("contentvec"), 256).unwrap(), EmbedderKind::ContentVec256);
        assert_eq!(EmbedderKind::from_manifest(Some("ContentVec"), 768).unwrap(), EmbedderKind::ContentVec768);
        assert_eq!(
            EmbedderKind::from_manifest(Some("hubert_base"), 768).unwrap(),
            EmbedderKind::ContentVec768
        );
        assert_eq!(EmbedderKind::from_manifest(Some("spin_v2"), 768).unwrap(), EmbedderKind::SpinV2);
        assert_eq!(EmbedderKind::from_manifest(Some("spin"), 768).unwrap(), EmbedderKind::Spin);
        assert_eq!(EmbedderKind::from_manifest(Some("SPIN_v1"), 768).unwrap(), EmbedderKind::Spin);
        assert!(EmbedderKind::from_manifest(Some("spin"), 256).is_err());
        assert_eq!(EmbedderKind::Spin.catalog_id(), crate::services::neural_audio::vc::assets::EMBEDDER_SPIN);
        assert_eq!(
            EmbedderKind::from_manifest(Some("japanese-hubert-base"), 768).unwrap(),
            EmbedderKind::HubertJa
        );
        assert_eq!(
            EmbedderKind::from_manifest(Some("chinese-hubert-base"), 768).unwrap(),
            EmbedderKind::HubertZh
        );
        assert_eq!(
            EmbedderKind::from_manifest(Some("korean_hubert_base"), 768).unwrap(),
            EmbedderKind::HubertKo
        );
        // Manifest names round-trip.
        for kind in EmbedderKind::ALL {
            assert_eq!(EmbedderKind::from_manifest(Some(kind.as_str()), kind.channels() as u32).unwrap(), kind);
            assert!(assets::embedder_id(kind.as_str()).is_some());
            assert_eq!(kind.frontend(), (400, 320));
        }
        // A SPIN voice with 256 channels, an unknown encoder.
        assert!(EmbedderKind::from_manifest(Some("spin-v2"), 256).is_err());
        assert!(EmbedderKind::from_manifest(Some("wavlm-large"), 768)
            .unwrap_err()
            .contains("wavlm-large"));
    }

    #[test]
    fn variants_prefer_fp16_on_gpus_and_int8_on_vnni_cpus() {
        assert_eq!(variant_order(true, PrecisionMode::Auto, true), vec![Some("fp16"), Some("fp32"), None]);
        assert_eq!(variant_order(true, PrecisionMode::Fp32, true), vec![Some("fp32"), None]);
        assert_eq!(variant_order(false, PrecisionMode::Auto, true), vec![Some("int8"), Some("fp32"), None]);
        assert_eq!(variant_order(false, PrecisionMode::Auto, false), vec![Some("fp32"), None]);
        assert_eq!(variant_order(false, PrecisionMode::Fp16, true), vec![Some("fp32"), None]);
        // The outcome of the precision rules (fp32 first on the GPU when fp16 fails its parity gate,
        // on the CPU when int8 loses speaker similarity) is one setting away on both classes, and the
        // int8 environment switch alone turns the CPU's int8 off (ADR 0011).
        assert_eq!(variant_order(false, PrecisionMode::Fp32, true), vec![Some("fp32"), None]);
        assert_eq!(variant_order(true, PrecisionMode::Fp32, false), vec![Some("fp32"), None]);
        assert_eq!(variant_order(false, PrecisionMode::Auto, false)[0], Some("fp32"));
    }

    #[test]
    fn on_by_default_switches_turn_off_only_on_an_off_word() {
        assert!(switch_on(None));
        for on in ["1", "on", "yes", "true", "", "auto", "2"] {
            assert!(switch_on(Some(on)), "{on:?}");
        }
        for off in ["0", "off", "OFF", " false ", "No"] {
            assert!(!switch_on(Some(off)), "{off:?}");
        }
    }

    /// The `content_norm` lever is offered only where a `stream` variant (fp16 or fp32) and its
    /// `conv0.f32.bin` are installed: none is published in the catalog, so a shipped install does
    /// not show a control that could only turn itself off at every voice load.
    #[test]
    fn running_norm_is_offered_only_with_an_installed_stream_variant() {
        let dir = std::env::temp_dir().join(format!("sigil-content-streamfiles-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!stream_variant_in(&dir), "an empty folder");
        std::fs::write(dir.join("model.fp32.onnx"), b"x").unwrap();
        std::fs::write(dir.join("model.fp16.onnx"), b"x").unwrap();
        assert!(!stream_variant_in(&dir), "the stock variants are not stream variants");
        std::fs::write(dir.join("model.stream.fp32.onnx"), b"x").unwrap();
        assert!(!stream_variant_in(&dir), "a stream graph without its conv0 weights");
        std::fs::write(dir.join("conv0.f32.bin"), b"x").unwrap();
        assert!(stream_variant_in(&dir));
        std::fs::remove_file(dir.join("model.stream.fp32.onnx")).unwrap();
        assert!(!stream_variant_in(&dir), "conv0 alone");
        std::fs::write(dir.join("model.stream.fp16.onnx"), b"x").unwrap();
        assert!(stream_variant_in(&dir), "the fp16 stream variant alone serves a GPU");
        let _ = std::fs::remove_dir_all(&dir);
        // The offer and the resolution agree on whatever this machine has installed.
        let resolvable = EmbedderKind::ALL
            .into_iter()
            .any(|k| resolve_stream_variant(k, true, PrecisionMode::Auto).is_ok() || resolve_stream_variant(k, false, PrecisionMode::Auto).is_ok());
        assert_eq!(running_norm_available(), resolvable);
    }

    /// A stand-in encoder (a 400/320 conv) with `channels` output channels.
    fn standin(channels: usize) -> Vec<u8> {
        use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT};
        let w: Vec<f32> = (0..channels * 400).map(|i| ((i % 5) as f32 - 2.0) * 1e-3).collect();
        let mut g = Graph::new();
        g.input("source", FLOAT, &[Dim::Fixed(1), Dim::Named("n")])
            .output("feats", FLOAT, &[Dim::Fixed(1), Dim::Named("t"), Dim::Fixed(channels as u64)])
            .init_i64("axes1", &[1], &[1])
            .init_f32("conv_w", &[channels as u64, 1, 400], &w)
            .node("Unsqueeze", &["source", "axes1"], &["x3"], &[])
            .node(
                "Conv",
                &["x3", "conv_w"],
                &["c"],
                &[Attr::Ints("strides", vec![320]), Attr::Ints("kernel_shape", vec![400]), Attr::Int("group", 1)],
            )
            .node("Transpose", &["c"], &["feats"], &[Attr::Ints("perm", vec![0, 2, 1])]);
        g.model_bytes(13)
    }

    /// The UI's HuBERT path (a 768-channel file kept for v2 voices) does not block a v1 voice: the
    /// width probe fails, the slot falls back to its 256-channel encoder with a warning. A file of
    /// the right width is used as before, and with nothing else to fall back to the explicit
    /// file's error is the load error.
    #[test]
    fn explicit_encoder_of_the_wrong_width_falls_back() {
        let root = std::env::temp_dir().join(format!("sigil-content-width-{}", std::process::id()));
        let slot = root.join("voice-changer").join("models").join("v1");
        let base = root.join("voice-changer").join("base");
        std::fs::create_dir_all(&slot).unwrap();
        std::fs::create_dir_all(&base).unwrap();
        let explicit = root.join("hubert_768.onnx");
        std::fs::write(&explicit, standin(768)).unwrap();
        let opts = VcSessionOptions::default();
        // Nothing to fall back to: the width error.
        let mut warnings = Vec::new();
        let err = acquire_for_slot(EmbedderKind::ContentVec256, Some(&explicit), Some(&slot), "cpu", &opts, PrecisionMode::Auto, 1_600, &mut warnings)
            .err()
            .unwrap();
        assert!(err.contains("contentvec-256"), "{err}");
        // The legacy 256-channel file next to the slot.
        let v1 = base.join("vec-256-layer-9.onnx");
        std::fs::write(&v1, standin(256)).unwrap();
        let enc = acquire_for_slot(EmbedderKind::ContentVec256, Some(&explicit), Some(&slot), "cpu", &opts, PrecisionMode::Auto, 1_600, &mut warnings).unwrap();
        assert_eq!(enc.lock().unwrap().source(), (v1.as_path(), "legacy"));
        assert_eq!(enc.lock().unwrap().channels(), 256);
        assert!(warnings.iter().any(|w| w.contains("does not fit this voice")), "{warnings:?}");
        // The same explicit file still serves a v2 voice, with no warning.
        let mut warnings = Vec::new();
        let enc2 = acquire_for_slot(EmbedderKind::ContentVec768, Some(&explicit), Some(&slot), "cpu", &opts, PrecisionMode::Auto, 1_600, &mut warnings).unwrap();
        assert_eq!(enc2.lock().unwrap().source(), (explicit.as_path(), "explicit"));
        assert!(warnings.is_empty(), "{warnings:?}");
        drop((enc, enc2));
        ContentEncoderPool::global().release_unused();
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Explicit files win for ContentVec, the legacy base folder next to the slot is found, and a
    /// SPIN voice never runs on the UI's ContentVec file.
    #[test]
    fn encoder_files_resolve_explicit_then_store_then_legacy() {
        let root = std::env::temp_dir().join(format!("sigil-content-resolve-{}", std::process::id()));
        let slot = root.join("voice-changer").join("models").join("v");
        let base = root.join("voice-changer").join("base");
        std::fs::create_dir_all(&slot).unwrap();
        std::fs::create_dir_all(&base).unwrap();
        let explicit = root.join("mine.onnx");
        std::fs::write(&explicit, b"x").unwrap();
        let r = resolve_encoder_file(EmbedderKind::ContentVec768, Some(&explicit), Some(&slot), true, PrecisionMode::Auto).unwrap();
        assert_eq!((r.path.as_path(), r.variant), (explicit.as_path(), "explicit"));
        assert!(resolve_encoder_file(
            EmbedderKind::ContentVec768,
            Some(&root.join("missing.onnx")),
            None,
            true,
            PrecisionMode::Auto
        )
        .is_err());
        // No store entry in tests: the legacy base file.
        std::fs::write(base.join("hubert_base.onnx"), b"x").unwrap();
        let r = resolve_encoder_file(EmbedderKind::ContentVec768, None, Some(&slot), false, PrecisionMode::Auto).unwrap();
        assert_eq!((r.path, r.variant), (base.join("hubert_base.onnx"), "legacy"));
        // SPIN ignores the explicit ContentVec file and has no legacy fallback.
        let err = resolve_encoder_file(EmbedderKind::SpinV2, Some(&explicit), Some(&slot), true, PrecisionMode::Auto).unwrap_err();
        assert!(err.contains("vc/embedder-spin-v2"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A stand-in encoder whose audio input is fp16 (`half_in`) or whose features come out
    /// channel-major `[1, C, T]` (`channel_major`), with 768 channels.
    fn standin_variant(half_in: bool, channel_major: bool) -> Vec<u8> {
        use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT};
        const FLOAT16: u64 = 10;
        let w: Vec<f32> = (0..768 * 400).map(|i| ((i % 5) as f32 - 2.0) * 1e-3).collect();
        let mut g = Graph::new();
        g.input("audio", if half_in { FLOAT16 } else { FLOAT }, &[Dim::Fixed(1), Dim::Named("N")])
            .init_i64("axes1", &[1], &[1])
            .init_f32("conv_w", &[768, 1, 400], &w);
        let src = if half_in {
            g.node("Cast", &["audio"], &["a32"], &[Attr::Int("to", 1)]);
            "a32"
        } else {
            "audio"
        };
        g.node("Unsqueeze", &[src, "axes1"], &["x3"], &[]).node(
            "Conv",
            &["x3", "conv_w"],
            &[if channel_major { "feats" } else { "c" }],
            &[Attr::Ints("strides", vec![320]), Attr::Ints("kernel_shape", vec![400]), Attr::Int("group", 1)],
        );
        if channel_major {
            g.output("feats", FLOAT, &[Dim::Fixed(1), Dim::Fixed(768), Dim::Named("T")]);
        } else {
            g.node("Transpose", &["c"], &["feats"], &[Attr::Ints("perm", vec![0, 2, 1])])
                .output("feats", FLOAT, &[Dim::Fixed(1), Dim::Named("T"), Dim::Fixed(768)]);
        }
        g.model_bytes(13)
    }

    fn bound_levers() -> GpuLevers {
        GpuLevers {
            io_binding: true,
            ..GpuLevers::default()
        }
    }

    fn speechy(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                0.3 * (std::f32::consts::TAU * (130.0 + 30.0 * (t * 2.0 + seed).sin()) * t).sin() * (0.6 + 0.4 * (t * 5.0 + seed).sin())
            })
            .collect()
    }

    /// The fixed-shape plan (CPU rung with IOBinding: the same plan code as the GPU rungs) gives the
    /// dynamic session's features; voices with the same key share one plan; a voice that moves to
    /// another length lets go of its plan, which is pruned; once only plans use the encoder its
    /// dynamic session is released, and a dynamic caller brings it back.
    #[test]
    fn fixed_plans_match_and_are_shared_leased_and_pruned() {
        let dir = std::env::temp_dir().join(format!("sigil-content-plans-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("enc.onnx");
        std::fs::write(&path, test_models::content_encoder_bytes()).unwrap();
        let opts = VcSessionOptions::default();
        let mut enc = ContentEncoder::load(EmbedderKind::ContentVec768, &path, "explicit", "cpu", &opts).unwrap();
        assert!(enc.plan_blocker().is_none(), "{:?}", enc.plan_blocker());
        let (a, b) = (speechy(7_040, 0.0), speechy(7_040, 1.3));
        let (mut want, mut got) = (Vec::new(), Vec::new());
        let n = enc.encode_dynamic(&a, &mut want).unwrap();
        // Voice 1 builds the plan at 7040 samples.
        let mut lease1 = None;
        let (m, report) = enc.encode_fixed(&a, bound_levers(), &mut lease1, &mut got).unwrap();
        let report = report.expect("the first call reports the plan");
        assert!(report.fixed, "{report:?}");
        assert_eq!(report.outcome.active, "cpu");
        assert!(report.levers.io_binding);
        assert_eq!((m, got.len()), (n, n * 768));
        assert!(rel_l2(&got, &want) < 1e-6, "plan vs dynamic {}", rel_l2(&got, &want));
        assert_eq!(enc.plan_count(), 1);
        // Steady state: no report, the same features as the dynamic session on new audio.
        enc.encode_dynamic(&b, &mut want).unwrap();
        let (_, report) = enc.encode_fixed(&b, bound_levers(), &mut lease1, &mut got).unwrap();
        assert!(report.is_none());
        assert!(rel_l2(&got, &want) < 1e-6);
        // Voice 2 with the same key joins the plan.
        let mut lease2 = None;
        let (_, report) = enc.encode_fixed(&b, bound_levers(), &mut lease2, &mut got).unwrap();
        assert!(report.unwrap().fixed);
        assert_eq!(enc.plan_count(), 1, "one plan per key");
        // The only users are plans: the dynamic session went (the test's own dynamic calls above
        // used the uncounted path).
        assert!(!enc.has_dynamic_session());
        // Voice 2 moves to a longer window: a second plan; voice 1 keeps its own.
        let c = speechy(16_000, 0.4);
        let (m2, _) = enc.encode_fixed(&c, bound_levers(), &mut lease2, &mut got).unwrap();
        assert_eq!(m2, (16_000 - 400) / 320 + 1);
        assert_eq!(enc.plan_count(), 2);
        // Voice 1 goes away: its plan is pruned on the next call that looks.
        drop(lease1);
        enc.encode_fixed(&c, bound_levers(), &mut lease2, &mut got).unwrap();
        assert_eq!(enc.plan_count(), 1);
        // A dynamic-path caller (another voice family) brings the dynamic session back, for good.
        let n3 = enc.encode(&c, &mut want).unwrap();
        assert!(enc.has_dynamic_session());
        assert_eq!(n3, m2);
        assert!(rel_l2(&got, &want) < 1e-6);
        drop(lease2);
        enc.prune_plans();
        assert_eq!(enc.plan_count(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// fp16 audio I/O keeps the dynamic path (with a note, asked once per key); a channel-major
    /// `[1, C, T]` output runs as a plan and is transposed like the dynamic path's.
    #[test]
    fn fp16_io_stays_dynamic_and_channel_major_plans_transpose() {
        let dir = std::env::temp_dir().join(format!("sigil-content-variants-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let opts = VcSessionOptions::default();
        let audio = speechy(7_040, 0.7);
        let half = dir.join("half.onnx");
        std::fs::write(&half, standin_variant(true, false)).unwrap();
        let mut enc = ContentEncoder::load(EmbedderKind::ContentVec768, &half, "explicit", "cpu", &opts).unwrap();
        assert!(enc.plan_blocker().unwrap().contains("fp16"));
        let (mut want, mut got) = (Vec::new(), Vec::new());
        let mut lease = None;
        let (n, report) = enc.encode_fixed(&audio, bound_levers(), &mut lease, &mut got).unwrap();
        let report = report.unwrap();
        assert!(!report.fixed && report.notes[0].contains("fp16"), "{report:?}");
        assert!(lease.as_ref().is_some_and(|l| !l.fixed()));
        let (_, report) = enc.encode_fixed(&audio, bound_levers(), &mut lease, &mut got).unwrap();
        assert!(report.is_none(), "asked once per key");
        enc.encode_dynamic(&audio, &mut want).unwrap();
        assert_eq!(got, want);
        assert_eq!(n, (7_040 - 400) / 320 + 1);
        assert!(enc.has_dynamic_session(), "the dynamic session serves the voice");

        let cm = dir.join("channel-major.onnx");
        std::fs::write(&cm, standin_variant(false, true)).unwrap();
        let mut enc = ContentEncoder::load(EmbedderKind::ContentVec768, &cm, "explicit", "cpu", &opts).unwrap();
        let n = enc.encode_dynamic(&audio, &mut want).unwrap();
        let mut lease = None;
        let (m, report) = enc.encode_fixed(&audio, bound_levers(), &mut lease, &mut got).unwrap();
        assert!(report.unwrap().fixed);
        assert_eq!(m, n);
        assert!(rel_l2(&got, &want) < 1e-6);
        // Too short a window for the front end: refused, the dynamic path runs it.
        let mut lease = None;
        let short = speechy(320, 0.0);
        let r = enc.encode_fixed(&short, bound_levers(), &mut lease, &mut got);
        assert!(r.is_err() || !r.unwrap().1.unwrap().fixed);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A stand-in `stream` variant (the 400/320 front end of a HuBERT): the first convolution
    /// (`conv0`, 512 x 10, stride 5), normalised by the `gn_mean` / `gn_inv_std` inputs, a 1x1
    /// convolution to 8 channels and a 79-tap convolution of stride 64 to 768 channels.
    pub(crate) fn stream_standin_bytes(conv0: &[f32]) -> Vec<u8> {
        use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT};
        let w1: Vec<f32> = (0..8 * 512).map(|i| ((i % 13) as f32 - 6.0) * 2e-3).collect();
        let w2: Vec<f32> = (0..768 * 8 * 79).map(|i| ((i % 11) as f32 - 5.0) * 1e-3).collect();
        let mut g = Graph::new();
        g.input("audio", FLOAT, &[Dim::Fixed(1), Dim::Named("N")])
            .input(NORM_MEAN, FLOAT, &[Dim::Fixed(1), Dim::Fixed(512), Dim::Fixed(1)])
            .input(NORM_INV_STD, FLOAT, &[Dim::Fixed(1), Dim::Fixed(512), Dim::Fixed(1)])
            .output("feats", FLOAT, &[Dim::Fixed(1), Dim::Named("T"), Dim::Fixed(768)])
            .init_i64("axes1", &[1], &[1])
            .init_f32("conv0_w", &[512, 1, 10], conv0)
            .init_f32("w1", &[8, 512, 1], &w1)
            .init_f32("w2", &[768, 8, 79], &w2)
            .node("Unsqueeze", &["audio", "axes1"], &["x3"], &[])
            .node("Conv", &["x3", "conv0_w"], &["c0"], &[Attr::Ints("strides", vec![5]), Attr::Ints("kernel_shape", vec![10])])
            .node("Sub", &["c0", NORM_MEAN], &["d"], &[])
            .node("Mul", &["d", NORM_INV_STD], &["n"], &[])
            .node("Conv", &["n", "w1"], &["h"], &[Attr::Ints("kernel_shape", vec![1])])
            .node("Conv", &["h", "w2"], &["c"], &[Attr::Ints("strides", vec![64]), Attr::Ints("kernel_shape", vec![79])])
            .node("Transpose", &["c"], &["feats"], &[Attr::Ints("perm", vec![0, 2, 1])]);
        g.model_bytes(13)
    }

    /// A `stream` variant takes its statistics on the dynamic path and as a plan (both agree; the
    /// statistics change the features), and refuses to run without them.
    #[test]
    fn stream_variants_take_their_statistics_on_both_paths() {
        let dir = std::env::temp_dir().join(format!("sigil-content-stream-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let conv0: Vec<f32> = (0..512 * 10).map(|i| ((i * 31 % 97) as f32 / 48.0 - 1.0) * 0.2).collect();
        let path = dir.join("model.stream.fp32.onnx");
        std::fs::write(&path, stream_standin_bytes(&conv0)).unwrap();
        std::fs::write(dir.join("conv0.f32.bin"), conv0.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        assert_eq!(read_conv0(&dir.join("conv0.f32.bin")).unwrap(), conv0);
        let opts = VcSessionOptions::default();
        let mut enc = ContentEncoder::load(EmbedderKind::ContentVec768, &path, "stream.fp32", "cpu", &opts).unwrap();
        assert!(enc.takes_norm());
        assert!(enc.plan_blocker().is_none());
        let audio = speechy(7_040, 0.2);
        let mut out = Vec::new();
        assert!(enc.encode(&audio, &mut out).unwrap_err().contains("statistics"));
        assert!(enc.set_norm(&[0.0; 3], &[1.0; 3]).is_err());
        let (mean, inv) = (vec![0.01f32; 512], vec![2.0f32; 512]);
        enc.set_norm(&mean, &inv).unwrap();
        let mut want = Vec::new();
        let n = enc.encode_dynamic(&audio, &mut want).unwrap();
        assert_eq!(n, (7_040 - 400) / 320 + 1);
        let mut lease = None;
        let (m, report) = enc.encode_fixed(&audio, bound_levers(), &mut lease, &mut out).unwrap();
        assert!(report.unwrap().fixed);
        assert_eq!(m, n);
        assert!(rel_l2(&out, &want) < 1e-6, "{}", rel_l2(&out, &want));
        // Other statistics, other features (on both paths).
        enc.set_norm(&mean, &vec![1.0f32; 512]).unwrap();
        enc.encode_fixed(&audio, bound_levers(), &mut lease, &mut out).unwrap();
        assert!(rel_l2(&out, &want) > 1e-2);
        enc.encode_dynamic(&audio, &mut want).unwrap();
        assert!(rel_l2(&out, &want) < 1e-6);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A deferred encoder (no dynamic session at load) builds its plan from the export contract,
    /// gives the eager encoder's features, takes its outcome from the plan, and builds the dynamic
    /// session only for a dynamic caller. Without a contract it is refused (the caller loads it the
    /// usual way); on the CPU `acquire_for_engine` is `acquire_for_slot`.
    #[test]
    fn deferred_encoders_build_plans_without_a_dynamic_session() {
        let dir = std::env::temp_dir().join(format!("sigil-content-deferred-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.fp32.onnx");
        std::fs::write(&path, test_models::content_encoder_bytes()).unwrap();
        let opts = VcSessionOptions::default();
        assert!(ContentEncoder::deferred(EmbedderKind::ContentVec768, &path, "fp32", "cpu", &opts, None).is_err(), "no contract");
        let contract = serde_json::json!({"io": {
            "input": {"name": "source", "shape": [1, "n"], "dtype": "float32"},
            "output": {"name": "feats", "shape": [1, "t", 768], "dtype": "float32"}}});
        std::fs::write(dir.join("embedder.json"), serde_json::to_vec(&contract).unwrap()).unwrap();
        let mut d = ContentEncoder::deferred(EmbedderKind::ContentVec768, &path, "fp32", "cpu", &opts, None).unwrap();
        assert!(d.is_deferred() && !d.has_dynamic_session());
        let mut eager = ContentEncoder::load(EmbedderKind::ContentVec768, &path, "fp32", "cpu", &opts).unwrap();
        let audio = speechy(7_040, 0.9);
        let (mut want, mut got) = (Vec::new(), Vec::new());
        let n = eager.encode_dynamic(&audio, &mut want).unwrap();
        let mut lease = None;
        let (m, report) = d.encode_fixed(&audio, bound_levers(), &mut lease, &mut got).unwrap();
        assert!(report.unwrap().fixed);
        assert_eq!(m, n);
        assert!(rel_l2(&got, &want) < 1e-6);
        assert!(!d.has_dynamic_session(), "the plan ran without a dynamic session");
        assert_eq!(d.outcome().active, "cpu");
        // A dynamic caller (another voice family) builds it on its first call.
        let k = d.encode(&audio, &mut got).unwrap();
        assert_eq!(k, n);
        assert!(d.has_dynamic_session() && !d.is_deferred());
        assert!(rel_l2(&got, &want) < 1e-6);
        // The engine's acquisition on the CPU: the usual (eager) path.
        let mut warnings = Vec::new();
        let enc = acquire_for_engine(EmbedderKind::ContentVec768, Some(&path), None, "cpu", &opts, PrecisionMode::Auto, 7_040, &mut warnings).unwrap();
        assert!(enc.lock().unwrap().has_dynamic_session());
        drop(enc);
        ContentEncoderPool::global().release_unused();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A deferred encoder's plan has no dynamic session to compare with, so its features are checked
    /// against a CPU session of the file (on a helper thread): a correct plan is accepted and its
    /// verdict kept for the process (a rebuilt encoder starts no new reference), a plan that
    /// reads other features than the CPU does is refused with the dynamic session serving the voice,
    /// the refusal is remembered too, and a reference that cannot run leaves the plan standing with
    /// a note and no verdict.
    #[test]
    fn deferred_plans_are_cross_checked_against_a_cpu_session() {
        let dir = std::env::temp_dir().join(format!("sigil-content-crosscheck-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.fp32.onnx");
        std::fs::write(&path, test_models::content_encoder_bytes()).unwrap();
        // The same I/O, other weights: a plan checked against it must look wrong.
        let wrong = dir.join("wrong.onnx");
        std::fs::write(&wrong, standin(768)).unwrap();
        let contract = serde_json::json!({"io": {
            "input": {"name": "source", "shape": [1, "n"], "dtype": "float32"},
            "output": {"name": "feats", "shape": [1, "t", 768], "dtype": "float32"}}});
        std::fs::write(dir.join("embedder.json"), serde_json::to_vec(&contract).unwrap()).unwrap();
        let opts = VcSessionOptions::default();
        let deferred = || ContentEncoder::deferred(EmbedderKind::ContentVec768, &path, "fp32", "cpu", &opts, None).unwrap();
        let started = || REFERENCES_STARTED.with(std::cell::Cell::get);
        let audio = speechy(7_040, 0.5);
        let key = PlanKey { len: 7_040, levers: bound_levers() };
        let (mut want, mut got) = (Vec::new(), Vec::new());
        ContentEncoder::load(EmbedderKind::ContentVec768, &path, "fp32", "cpu", &opts)
            .unwrap()
            .encode_dynamic(&audio, &mut want)
            .unwrap();

        // 1. The plan against the CPU session of its own file: accepted, verdict on record.
        let mut enc = deferred();
        let mut lease = None;
        let (_, report) = enc.encode_fixed(&audio, bound_levers(), &mut lease, &mut got).unwrap();
        let report = report.unwrap();
        assert!(report.fixed && !report.notes.iter().any(|n| n.contains("cross-check")), "{report:?}");
        assert_eq!(started(), 1);
        assert!(rel_l2(&got, &want) < 1e-6);
        assert!(!enc.has_dynamic_session(), "the check built no dynamic session");
        let own = enc.check_key(&path, key);
        assert!(matches!(verdict_of(&own), Some(Verdict::Pass(rel)) if rel < 1e-5), "{:?}", verdict_of(&own));

        // 2. A rebuilt encoder of the same file and shape finds the verdict: no new reference.
        let mut again = deferred();
        let (_, report) = again.encode_fixed(&audio, bound_levers(), &mut None, &mut got).unwrap();
        assert!(report.unwrap().fixed);
        assert_eq!(started(), 1, "a verdict on record starts no reference");
        // Another shape of the same file is a new plan, and checked.
        let (_, report) = again
            .encode_fixed(&speechy(8_000, 0.1), bound_levers(), &mut None, &mut got)
            .unwrap();
        assert!(report.unwrap().fixed);
        assert_eq!(started(), 2);

        // 3. A plan that disagrees with the CPU session is refused, the dynamic session serves the
        // voice (same features as the eager encoder), and the refusal is recorded.
        let mut bad = deferred();
        bad.set_check_file(wrong.clone());
        let mut lease = None;
        let (n, report) = bad.encode_fixed(&audio, bound_levers(), &mut lease, &mut got).unwrap();
        let report = report.unwrap();
        assert!(!report.fixed, "{report:?}");
        assert!(report.notes.iter().any(|n| n.contains("differs from a CPU session")), "{:?}", report.notes);
        assert!(lease.as_ref().is_some_and(|l| !l.fixed()));
        assert_eq!(bad.plan_count(), 0);
        assert!(bad.has_dynamic_session(), "the voice runs the dynamic session");
        assert_eq!(n, (7_040 - 400) / 320 + 1);
        assert!(rel_l2(&got, &want) < 1e-6, "the fallback gives the encoder's own features");
        let bad_key = bad.check_key(&wrong, key);
        assert!(matches!(verdict_of(&bad_key), Some(Verdict::Fail(_))));
        let before = started();
        let mut rebuilt = deferred();
        rebuilt.set_check_file(wrong.clone());
        let (_, report) = rebuilt.encode_fixed(&audio, bound_levers(), &mut None, &mut got).unwrap();
        let report = report.unwrap();
        assert!(!report.fixed && report.notes.iter().any(|n| n.contains("differs from a CPU session")), "{report:?}");
        assert_eq!(started(), before, "a refusal on record is not checked again");

        // 4. A reference that cannot run: the plan stands on its finite output, with a note, and no
        // verdict is recorded.
        let mut blind = deferred();
        let missing = dir.join("missing.onnx");
        blind.set_check_file(missing.clone());
        let (_, report) = blind.encode_fixed(&audio, bound_levers(), &mut None, &mut got).unwrap();
        let report = report.unwrap();
        assert!(report.fixed, "{report:?}");
        assert!(report.notes.iter().any(|n| n.contains("not cross-checked")), "{:?}", report.notes);
        assert!(verdict_of(&blind.check_key(&missing, key)).is_none());
        assert!(rel_l2(&got, &want) < 1e-6);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The settings' HuBERT path naming the store's own faster file gets no "use the faster one"
    /// advice: the paths are compared as files, whatever their spelling.
    #[test]
    fn same_file_compares_files_not_spellings() {
        let dir = std::env::temp_dir().join(format!("sigil-content-same-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        let (a, b) = (dir.join("model.int8.onnx"), dir.join("model.fp32.onnx"));
        std::fs::write(&a, b"a").unwrap();
        std::fs::write(&b, b"b").unwrap();
        assert!(same_file(&a, &a));
        assert!(same_file(&a, &dir.join("sub").join("..").join("model.int8.onnx")));
        let slashed = PathBuf::from(a.to_string_lossy().replace('\\', "/"));
        assert!(same_file(&a, &slashed));
        assert!(!same_file(&a, &b));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The kill switch and the rung rule: plans run on CUDA / DirectML only, unless switched off.
    #[test]
    fn plan_rungs_and_relative_l2() {
        assert!(plan_rung("cuda") && plan_rung("dml"));
        assert!(!plan_rung("cpu") && !plan_rung("tensorrt_rtx"));
        assert!(!runs_fixed_shape("cpu"));
        assert_eq!(rel_l2(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        assert!(rel_l2(&[1.0], &[1.0, 2.0]).is_infinite());
        assert!((rel_l2(&[1.1, 0.0], &[1.0, 0.0]) - 0.1).abs() < 1e-6);
        let mut levers = super::super::EffectiveLevers {
            accelerator: Accelerator::TensorRtRtx,
            ..super::super::EffectiveLevers::default()
        };
        levers.levers.set(super::super::Lever::IoBinding, true);
        let l = plan_levers(&levers);
        assert_eq!(l.accelerator, Accelerator::None, "content stays on CUDA under TensorRT-RTX");
        assert!(l.io_binding);
    }

    /// The stand-in encoder (a 400/320 conv to 768 channels) runs through the pool: features are
    /// time-major with the conv's frame count, the session is shared, and a mismatched channel
    /// count is refused with the encoder the slot needs.
    #[test]
    fn pooled_encoder_encodes_and_is_shared() {
        let dir = std::env::temp_dir().join(format!("sigil-content-pool-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("enc.onnx");
        std::fs::write(&path, test_models::content_encoder_bytes()).unwrap();
        let pool = ContentEncoderPool {
            entries: Mutex::new(Vec::new()),
        };
        let opts = VcSessionOptions::default();
        let a = pool
            .acquire(EmbedderKind::ContentVec768, Some(&path), "cpu", &opts, PrecisionMode::Auto)
            .unwrap();
        let b = pool
            .acquire(EmbedderKind::ContentVec768, Some(&path), "cpu", &opts, PrecisionMode::Auto)
            .unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(pool.len(), 1);
        let audio: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let mut feats = Vec::with_capacity(8);
        let frames = a.lock().unwrap().encode(&audio, &mut feats).unwrap();
        assert_eq!(frames, (16_000 - 400) / 320 + 1);
        assert_eq!(feats.len(), frames * 768);
        assert_eq!(a.lock().unwrap().outcome().active, "cpu");
        // The v1 encoder kind on a 768-channel file: refused, naming the encoder the slot needs.
        let v1 = pool
            .acquire(EmbedderKind::ContentVec256, Some(&path), "cpu", &opts, PrecisionMode::Auto)
            .unwrap();
        let err = v1.lock().unwrap().encode(&audio, &mut feats).unwrap_err();
        assert!(err.contains("contentvec-256"), "{err}");
        drop((a, b, v1));
        pool.release_unused();
        assert!(pool.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
