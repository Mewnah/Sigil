//! transcribe.cpp GPU-tier engine (`sigil_transcribe.dll`): Qwen3-ASR-1.7B Q8_0 / BF16, Cohere
//! Transcribe Q4_K, Granite Speech 5.0 TurboCTC, Voxtral Mini 4B Realtime (streaming) and any
//! bring-your-own GGUF transcribe.cpp runs. Owned by P2-D.
//!
//! # Frozen signature (P2-S0)
//! `create(&EngineSpec, &EngineContext) -> Result<Box<dyn AsrEngine>, AsrError>` (called by
//! `engines::create`; blocking: loads the model).
//!
//! # Model and runtime
//! [`EngineSpec::model`] is a catalog id of `model-catalog/stt-gguf.json` (located only with
//! `model_store::installed_path`) or an absolute `.gguf` / `.bin` path
//! (`speech_rt::transcribe::models`). The runtime is the best installed
//! `runtime/sigil-transcribe-<variant>` pack (`speech_rt::transcribe::TranscribeModel`). A missing
//! model or pack is [`AsrErrorKind::NotInstalled`] whose message starts with the catalog id to
//! install (for a runtime: the pack of what the load runs on, [`effective_want`]; a CPU pack for a
//! CPU run); engines never download. A spec that names a GPU build this PC can use needs exactly
//! that build (plus the NVIDIA runtime for `cuda13`) in [`requirements`]: a CPU pack never stands
//! in for the GPU tier.
//!
//! # The resource governor decides the GPU ([`governor`])
//! [`EngineSpec::variant`]: `None` = the governor decides (`auto`); a GPU variant (`cuda13`, the
//! router's choice, or `vulkan`) = the GPU when the governor admits it; a CPU variant = that CPU
//! build. The admission and the VRAM claims are one step: the model claim
//! (`resources::claim_vram(Subsystem::SttFinal, ..)`, resized after the load and warm-up to the
//! measured process-VRAM delta within ±25 % of the estimate, or kept at the estimate when another
//! in-process claim changed during the load) plus a context claim shared by every engine of that
//! GPU runtime (pinned once the runtime was initialized: its DLL and CUDA context stay loaded).
//! A model the governor keeps off the GPU loads on the CPU variants and says why
//! ([`EngineInfo::fallback_reason`], which the pipeline shows as `engine_fallback`); with
//! `cpuFallback: false` it fails with `OutOfMemory` instead. A model the governor admitted that
//! still loads on the CPU says why too (the GPU build failed, was skipped or is not installed).
//! The GPU is never used without an admission, and a model that moves to the CPU later (the
//! wrapper's VRAM-headroom guard, a GPU failure) releases its model claim and reports
//! `engine_fallback`.
//!
//! # Decoding
//! - `transcribe`: one closed utterance. The language comes from [`DecodeOptions::language`], else
//!   the engine's `language` option; family rules are enforced here too (Cohere never for
//!   ja / zh / ko / ru / yue and never without a language; a language a detecting model does not
//!   list is dropped so it detects instead). Utterances longer than `maxSegmentSeconds` (the GPU
//!   warm-up length, 20 s: longer runs would grow ggml-cuda's pool past what the warm-up sized) are
//!   split at their quietest point (never leaving a last piece under 1 s) and joined; every piece
//!   shorter than 0.3 s is padded with silence. `lang:<code>` is tagged when the model detected
//!   the language.
//! - Streams (models whose capabilities say streaming, e.g. Voxtral Realtime): `begin_stream` /
//!   `push_stream` / `end_stream`. Partials are `committed + tentative`; with `appendOnly` (and
//!   always for Confucius4-R2T2, which needs a later transcribe.cpp than the pinned 0.2.4) only the
//!   committed text, which never retracts. An idle stream is reused for the next utterance after
//!   the VRAM-headroom check a new stream gets (once per utterance, as for finals). A stream
//!   closed by a CPU fallback is reopened and the utterance so far replayed.
//! - Cancellation: the canceller interrupts transcribe.cpp between decode steps. Qwen3-ASR 0.2.4
//!   checks only before its single-shot decode, so a call cancelled while it runs finishes, and its
//!   result is dropped here (`Cancelled`), never delivered late.
//! - One compute per model: a `transcribe` while this engine's stream is mid-utterance is
//!   [`AsrErrorKind::Busy`] (the pipeline uses separate engines for streams and finals).
//!
//! # Isolation
//! `isolation: "worker"` (or `SIGIL_STT_ISOLATION=worker`) hosts the model in a
//! `Sigil.exe --headless stt-worker` child ([`worker_proc`]): a ggml abort (a driver reset during a
//! game) then ends the worker, not Sigil, and the worker restarts (on the CPU after repeated
//! crashes; with `cpuFallback: false` it is not restarted after them). The governor decides in the
//! parent, which holds one external claim for the child, and decides again for every restart of
//! the GPU worker (the GPU with a new claim, the CPU for now, or an error without the CPU
//! fallback).
//!
//! # Engine options ([`EngineSpec::options`], camelCase, all optional; never logged)
//! `language` (default language; Cohere needs one), `deviceId` (GPU PCI bus id or name) / `device`
//! (index), `cpuFallback` (default true), `minFreeVramMb` (default 256), `nCtx`, `warmup` (default
//! true), `warmupSeconds`, `maxSegmentSeconds` (default 20), `pnc` / `itn` (`default|on|off`),
//! `noSpeechThreshold` (Whisper), streams: `commit` (`auto|on_finalize|stable_prefix`), `agreement`,
//! `appendOnly`; isolation: `isolation` (`inProcess|worker`), `workerMemoryLimitMb`.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::super::engine_api::{
    AsrEngine, AsrError, AsrErrorKind, AsrResult, Canceller, DecodeOptions, EngineCaps, EngineContext, EngineInfo, EngineKind, EngineSpec,
    EngineStatus, Hypothesis, StatusLevel, SAMPLE_RATE,
};
use super::super::events::codes;
use super::super::lang;
use super::super::worker_proc::{self, Readmission, Readmit, WorkerEngine, WorkerSetup};
use crate::services::native_runtime::{Attempt, RuntimeVariant, VariantSupport};
use crate::services::speech_rt::transcribe::governor::{self, Admission, GpuClaims, Want};
use crate::services::speech_rt::transcribe::models::{self, Family, ResolvedModel};
use crate::services::speech_rt::transcribe::{
    installed_variants, BackendChoice, ErrorKind, ModelInfo, ModelOptions, Timestamps, Toggle, TranscribeCanceller, TranscribeError,
    TranscribeModel, TranscribeOptions, TranscribeStream, DEFAULT_MIN_FREE_VRAM_MB, GPU_WARMUP_SECONDS,
};

/// Log target.
const LOG: &str = "sigil::speech::transcribe";
/// Catalog id of the NVIDIA CUDA 13 runtime the `cuda13` pack needs.
pub const NVIDIA_RUNTIME_ID: &str = "runtime/nvidia-cuda13";
/// Internal option key: the parent's admission, honoured only inside a worker process.
pub(crate) const ADMISSION_KEY: &str = "__sigilAdmission";
/// Shorter utterances (and pieces of a split one) are padded with silence to this length (some
/// models reject tiny inputs).
const MIN_AUDIO_SAMPLES: usize = SAMPLE_RATE as usize * 3 / 10;
/// Shortest last piece of a split utterance (1 s): a cut that would leave less moves earlier.
const MIN_TAIL_SAMPLES: usize = SAMPLE_RATE as usize;
/// Longest decoder prompt sent to Whisper (bytes; its prompt window is 224 tokens).
const MAX_PROMPT_BYTES: usize = 800;
/// Audio kept per open stream for a replay after a CPU fallback (60 s).
const MAX_REPLAY_SAMPLES: usize = SAMPLE_RATE as usize * 60;

/// Where the model runs relative to Sigil.exe.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Isolation {
    #[default]
    InProcess,
    Worker,
}

impl Isolation {
    fn parse(s: &str) -> Option<Isolation> {
        match s.trim().to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "worker" | "process" | "workerprocess" => Some(Isolation::Worker),
            "inprocess" | "none" | "off" => Some(Isolation::InProcess),
            _ => None,
        }
    }

    /// The default: [`worker_proc::ISOLATION_ENV`] when set, else in-process (measured stable,
    /// ADR 0003b; the worker is the documented fallback for in-process instability).
    fn default_from_env() -> Isolation {
        std::env::var(worker_proc::ISOLATION_ENV)
            .ok()
            .and_then(|v| Isolation::parse(&v))
            .unwrap_or_default()
    }
}

/// The parent's admission, forwarded to a worker process (internal).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ForwardedAdmission {
    /// GPU variant admitted, `None` = CPU.
    pub gpu: Option<String>,
    pub code: String,
    pub reason: Option<String>,
}

impl ForwardedAdmission {
    pub fn from_admission(a: &Admission) -> Self {
        Self {
            gpu: a.gpu.map(|v| v.as_str().to_string()),
            code: a.code.to_string(),
            reason: a.reason.clone(),
        }
    }

    fn to_admission(&self) -> Admission {
        let gpu = self
            .gpu
            .as_deref()
            .and_then(RuntimeVariant::parse)
            .filter(|v| v.is_gpu());
        let code = [
            governor::codes::FITS,
            governor::codes::CPU_REQUESTED,
            governor::codes::NO_GPU,
            governor::codes::GPU_UNKNOWN,
            governor::codes::INTEGRATED_GPU,
            governor::codes::SMALL_GPU,
            governor::codes::NO_GPU_RUNTIME,
            governor::codes::VC_RUNNING,
            governor::codes::GPU_FULL,
            governor::codes::WORKER_CRASHED,
        ]
        .into_iter()
        .find(|c| *c == self.code)
        .unwrap_or(governor::codes::CPU_REQUESTED);
        Admission {
            gpu,
            code,
            reason: self.reason.clone(),
            need_mb: 0,
            available_mb: 0,
        }
    }
}

/// Engine options (see the module docs). Parsed leniently: a value of the wrong type is ignored.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Options {
    pub language: Option<String>,
    pub device: Option<u32>,
    pub device_id: Option<String>,
    pub cpu_fallback: bool,
    pub min_free_vram_mb: Option<u32>,
    pub n_ctx: Option<u32>,
    pub warmup: bool,
    pub warmup_seconds: Option<f32>,
    pub max_segment_seconds: f32,
    pub punctuation: Toggle,
    pub itn: Toggle,
    pub no_speech_threshold: Option<f32>,
    pub commit: Option<String>,
    pub agreement: Option<u32>,
    pub append_only: bool,
    pub isolation: Isolation,
    pub worker_memory_limit_mb: Option<u64>,
    /// Internal: the parent's admission (worker processes only).
    pub admission: Option<ForwardedAdmission>,
}

impl Default for Options {
    fn default() -> Self {
        Options::parse(&Value::Null)
    }
}

fn toggle(v: Option<&str>) -> Toggle {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("on" | "true") => Toggle::On,
        Some("off" | "false") => Toggle::Off,
        _ => Toggle::Default,
    }
}

impl Options {
    pub fn parse(v: &Value) -> Options {
        let s = |k: &str| {
            v.get(k)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };
        let n = |k: &str| v.get(k).and_then(Value::as_f64).filter(|x| x.is_finite());
        let b = |k: &str| v.get(k).and_then(Value::as_bool);
        let commit = s("commit").filter(|c| matches!(c.as_str(), "auto" | "on_finalize" | "stable_prefix"));
        Options {
            language: s("language").and_then(|l| lang::normalize(&l)),
            device: n("device").filter(|x| *x >= 0.0).map(|x| x.min(64.0) as u32),
            device_id: s("deviceId"),
            cpu_fallback: b("cpuFallback").unwrap_or(true),
            min_free_vram_mb: n("minFreeVramMb").map(|x| x.clamp(0.0, 16_384.0) as u32),
            n_ctx: n("nCtx").filter(|x| *x >= 256.0).map(|x| x.min(1_048_576.0) as u32),
            warmup: b("warmup").unwrap_or(true),
            warmup_seconds: n("warmupSeconds").map(|x| x.clamp(0.1, 30.0) as f32),
            max_segment_seconds: n("maxSegmentSeconds")
                .map(|x| x.clamp(5.0, 30.0) as f32)
                .unwrap_or(GPU_WARMUP_SECONDS),
            punctuation: toggle(s("pnc").as_deref()),
            itn: toggle(s("itn").as_deref()),
            no_speech_threshold: n("noSpeechThreshold").map(|x| x.clamp(0.0, 1.0) as f32),
            commit,
            agreement: n("agreement").map(|x| x.clamp(0.0, 16.0) as u32),
            append_only: b("appendOnly").unwrap_or(false),
            isolation: s("isolation")
                .and_then(|i| Isolation::parse(&i))
                .unwrap_or_else(Isolation::default_from_env),
            worker_memory_limit_mb: n("workerMemoryLimitMb").map(|x| x.clamp(1024.0, 262_144.0) as u64),
            admission: v
                .get(ADMISSION_KEY)
                .and_then(|a| serde_json::from_value(a.clone()).ok()),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------------------------

/// Create the engine (blocking: loads the model).
pub fn create(spec: &EngineSpec, ctx: &EngineContext) -> Result<Box<dyn AsrEngine>, AsrError> {
    let opts = Options::parse(&spec.options);
    if opts.isolation == Isolation::Worker && !worker_proc::is_worker_process() {
        return create_in_worker(spec, ctx, &opts).map(|e| Box::new(e) as Box<dyn AsrEngine>);
    }
    TranscribeEngine::create(spec, ctx, opts).map(|e| Box::new(e) as Box<dyn AsrEngine>)
}

/// Map a wrapper error. NotInstalled messages start with a catalog id (the pipeline installs it):
/// a runtime problem names the pack this PC should use for `want` (the spec's wish as the
/// governor decided it, [`effective_want`]: a CPU build for a CPU run, never the GPU pack).
pub(crate) fn asr_error_for(e: TranscribeError, want: Want) -> AsrError {
    let kind = match e.kind {
        ErrorKind::NotInstalled => AsrErrorKind::NotInstalled,
        ErrorKind::OutOfMemory => AsrErrorKind::OutOfMemory,
        ErrorKind::InvalidInput if e.message.contains("another stream is active") => AsrErrorKind::Busy,
        ErrorKind::InvalidInput => AsrErrorKind::InvalidInput,
        ErrorKind::Unsupported => AsrErrorKind::Unsupported,
        ErrorKind::Cancelled => AsrErrorKind::Cancelled,
        ErrorKind::Load | ErrorKind::Backend | ErrorKind::Runtime => AsrErrorKind::Runtime,
    };
    let message = if kind == AsrErrorKind::NotInstalled && !e.message.contains("stt/") && !e.message.starts_with("runtime/") {
        format!("{}: {}", recommended_runtime(want).0, e.message)
    } else {
        e.message
    };
    AsrError::new(kind, message)
}

/// What a load under `admission` needs from the runtime (pure): the admitted GPU build; else the
/// CPU build the spec named, or any CPU build (a spec that wanted the GPU but was denied runs on
/// the CPU, so its missing runtime is a CPU pack).
pub(crate) fn effective_want(want: Want, admission: &Admission) -> Want {
    match (admission.gpu, want) {
        (Some(v), _) => Want::Gpu(v),
        (None, Want::Cpu(v)) => Want::Cpu(v),
        (None, _) => Want::Cpu(None),
    }
}

/// The error of a GPU denial when the spec forbids the CPU fallback (`cpuFallback: false`): out of
/// memory when the card is busy (the voice changer, other models), unsupported when this PC has no
/// GPU the engine can use.
pub(crate) fn denial_error(admission: &Admission) -> AsrError {
    let kind = match admission.code {
        governor::codes::GPU_FULL | governor::codes::VC_RUNNING => AsrErrorKind::OutOfMemory,
        _ => AsrErrorKind::Unsupported,
    };
    let reason = admission
        .reason
        .clone()
        .unwrap_or_else(|| "the GPU is not available for speech recognition".into());
    AsrError::new(kind, format!("{reason} (the CPU fallback is off)"))
}

/// The claim label of a model in the resources view.
fn claim_label(model: &ResolvedModel, worker: bool) -> String {
    format!("speech: {}{}", model.label, if worker { " (worker process)" } else { "" })
}

/// GPU variant plus the CPU ladder behind it (the governor admitted `gpu`).
fn allowed_ladder(gpu: RuntimeVariant) -> Vec<RuntimeVariant> {
    vec![gpu, RuntimeVariant::CpuAvx512, RuntimeVariant::CpuAvx2, RuntimeVariant::Cpu]
}

/// Wrapper options for `model` under `admission` (pure).
pub(crate) fn model_options(
    model: &ResolvedModel,
    spec: &EngineSpec,
    opts: &Options,
    want: Want,
    admission: &Admission,
    primary_device: Option<String>,
) -> ModelOptions {
    let mut mo = ModelOptions::new(&model.path);
    mo.threads = spec.threads.filter(|t| *t > 0).map(|t| t.min(64));
    mo.cpu_fallback = opts.cpu_fallback;
    mo.warmup = opts.warmup;
    mo.warmup_seconds = opts.warmup_seconds;
    mo.n_ctx = opts.n_ctx;
    mo.min_free_vram_mb = opts.min_free_vram_mb.unwrap_or(DEFAULT_MIN_FREE_VRAM_MB);
    mo.device = opts.device;
    mo.device_id = opts
        .device_id
        .clone()
        .or(if opts.device.is_none() { primary_device } else { None });
    match admission.gpu {
        Some(v) => {
            mo.backend = BackendChoice::Auto;
            mo.allowed_variants = Some(allowed_ladder(v));
        }
        None => {
            mo.backend = BackendChoice::Cpu;
            if let Want::Cpu(Some(v)) = want {
                mo.variant = Some(v);
            }
        }
    }
    mo
}

/// The primary GPU's stable id for the engine when the PC has several NVIDIA GPUs (CUDA's device 0
/// is not necessarily the card the governor budgets for); `None` with one (CUDA0 is it).
fn primary_device_id() -> Option<String> {
    let sys = crate::services::sysinfo::cached();
    let nvidia = sys
        .gpus
        .iter()
        .filter(|g| !g.adapter.software && g.adapter.vendor_id == crate::services::diagnostics::gpu_mem::VENDOR_NVIDIA)
        .count();
    if nvidia < 2 {
        return None;
    }
    let p = sys.primary()?;
    p.pci_bus_id
        .clone()
        .or_else(|| Some(p.adapter.name.clone()))
}

/// The runtime build `want` needs on a PC (pure; `gpu_ok(v)`: the PC can run GPU build `v` for
/// speech, `best_cpu`: its best CPU build): the CPU build the spec named; the GPU build it named
/// when the PC can use it; for auto the governor's GPU build (CUDA 13) when usable; else the best
/// CPU build (a GPU the PC cannot use means the CPU, as the governor decides).
pub(crate) fn wanted_variant(want: Want, gpu_ok: &dyn Fn(RuntimeVariant) -> bool, best_cpu: RuntimeVariant) -> RuntimeVariant {
    match want {
        Want::Cpu(Some(v)) => v,
        Want::Gpu(v) if gpu_ok(v) => v,
        Want::Auto if gpu_ok(RuntimeVariant::Cuda13) => RuntimeVariant::Cuda13,
        Want::Cpu(None) | Want::Gpu(_) | Want::Auto => best_cpu,
    }
}

/// Whether an installed build `v` satisfies `want` (pure; `runs(v)`: its code runs on this PC):
/// exactly the named CPU build; exactly the named GPU build when the PC can use it (a CPU pack must
/// not stand in for the GPU tier: that is a silent CPU run); a CPU build for a CPU request; any
/// build that runs here for auto or an unusable GPU (GPU builds carry a CPU backend).
pub(crate) fn variant_satisfies(
    want: Want,
    v: RuntimeVariant,
    runs: &dyn Fn(RuntimeVariant) -> bool,
    gpu_ok: &dyn Fn(RuntimeVariant) -> bool,
) -> bool {
    match want {
        Want::Cpu(Some(w)) => v == w,
        Want::Gpu(w) if gpu_ok(w) => v == w,
        Want::Cpu(None) => !v.is_gpu() && runs(v),
        Want::Gpu(_) | Want::Auto => runs(v),
    }
}

fn best_cpu_variant(support: &VariantSupport) -> RuntimeVariant {
    [RuntimeVariant::CpuAvx512, RuntimeVariant::CpuAvx2, RuntimeVariant::Cpu]
        .into_iter()
        .find(|v| support.cpu_supports(*v))
        .unwrap_or(RuntimeVariant::Cpu)
}

/// Whether this PC can run GPU build `variant` for a speech model: its code runs here and the
/// governor admits it on an idle card (what else uses the VRAM aside; a small, integrated or
/// ineligible GPU is never admitted).
fn gpu_usable(variant: RuntimeVariant, support: &VariantSupport) -> bool {
    variant.is_gpu()
        && support.supports(variant)
        && governor::admit(
            Want::Gpu(variant),
            0,
            "",
            true,
            &crate::services::sysinfo::cached(),
            &crate::services::resources::ResourceSnapshot::default(),
        )
        .gpu
        .is_some()
}

/// The runtime pack this PC would use for `want` (catalog id, and whether the NVIDIA runtime is
/// needed too).
pub(crate) fn recommended_runtime(want: Want) -> (String, bool) {
    let support = VariantSupport::detect();
    let variant = wanted_variant(want, &|v| gpu_usable(v, &support), best_cpu_variant(&support));
    (format!("runtime/sigil-transcribe-{}", variant.as_str()), variant == RuntimeVariant::Cuda13)
}

/// Catalog ids the engine of `spec` needs and whether each is installed: the model (catalog ids
/// only) and the runtime pack this PC would use (plus the NVIDIA runtime for `cuda13`). An installed
/// variant satisfies the runtime as [`variant_satisfies`] says (a spec that names a GPU build this
/// PC can use needs exactly that build). Used by `engines::requirements`.
pub(crate) fn requirements(spec: &EngineSpec) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    let model = spec.model.trim();
    if !model.is_empty() && !Path::new(model).is_absolute() && model.contains('/') {
        out.push((model.to_string(), crate::services::model_store::installed_path(model).is_some()));
    }
    let want = Want::from_variant(spec.variant.as_deref());
    let support = VariantSupport::detect();
    let installed = installed_variants();
    let runs = |v: RuntimeVariant| support.supports(v);
    let gpu_ok = |v: RuntimeVariant| gpu_usable(v, &support);
    let (id, cuda) = match installed
        .iter()
        .find(|(v, _, _)| variant_satisfies(want, *v, &runs, &gpu_ok))
    {
        Some((v, id, _)) => (id.clone(), *v == RuntimeVariant::Cuda13),
        None => recommended_runtime(want),
    };
    let have = installed.iter().any(|(_, i, _)| *i == id);
    out.push((id, have));
    if cuda {
        out.push((
            NVIDIA_RUNTIME_ID.to_string(),
            crate::services::model_store::installed_path(NVIDIA_RUNTIME_ID).is_some(),
        ));
    }
    out
}

/// Default committed-memory cap of a worker hosting a model file of `file_bytes`: three times the
/// file (weights, a CPU copy after a fallback, compute buffers) plus 4 GB for the runtimes and the
/// CUDA driver's host allocations, at least 8 GB. A safety net against runaway allocations, not a
/// budget.
pub(crate) fn worker_memory_limit_mb(file_bytes: u64) -> u64 {
    let file_mb = file_bytes / (1024 * 1024);
    (file_mb.saturating_mul(3) + 4096).max(8192)
}

/// The spec a worker child hosts under the parent's admission `a` (pure): in-process there, with
/// the admission forwarded and, on the GPU, the GPU the parent budgeted for (`device_id`).
pub(crate) fn child_spec(spec: &EngineSpec, device_id: Option<&str>, a: &Admission) -> EngineSpec {
    let mut s = spec.clone();
    let mut o = match &spec.options {
        Value::Object(m) => m.clone(),
        _ => serde_json::Map::new(),
    };
    o.insert("isolation".into(), Value::String("inProcess".into()));
    if let (Some(id), Some(_)) = (device_id, a.gpu) {
        o.insert("deviceId".into(), Value::String(id.to_string()));
    }
    o.insert(
        ADMISSION_KEY.into(),
        serde_json::to_value(ForwardedAdmission::from_admission(a)).unwrap_or(Value::Null),
    );
    s.options = Value::Object(o);
    s
}

/// The CPU spec a GPU worker restarts with after repeated crashes (pure): only with the CPU
/// fallback (`cpuFallback: false` has none, so the worker stops instead of starting children
/// that refuse the CPU).
pub(crate) fn crash_cpu_spec(spec: &EngineSpec, opts: &Options, admission: &Admission, title: &str) -> Option<EngineSpec> {
    (admission.gpu.is_some() && opts.cpu_fallback).then(|| {
        child_spec(
            spec,
            None,
            &Admission::forced_cpu(
                governor::codes::WORKER_CRASHED,
                format!("{title} runs on the CPU: its GPU worker process crashed repeatedly"),
            ),
        )
    })
}

/// A restart's governor decision as the worker takes it (pure): the GPU with the new claims; the
/// CPU for now with the reason; or, without the CPU fallback, the denial as an error.
pub(crate) fn readmission(
    admission: &Admission,
    claims: GpuClaims,
    spec: &EngineSpec,
    device_id: Option<&str>,
    cpu_fallback: bool,
    title: &str,
) -> Readmission {
    if admission.gpu.is_some() {
        return Readmission::Gpu {
            spec: child_spec(spec, device_id, admission),
            claims,
        };
    }
    if !cpu_fallback {
        return Readmission::Refused(denial_error(admission));
    }
    Readmission::Cpu {
        spec: child_spec(spec, None, admission),
        reason: admission
            .reason
            .clone()
            .unwrap_or_else(|| format!("{title} runs on the CPU for now: the resource governor keeps it off the GPU")),
    }
}

/// The parent side of worker isolation: admission and the external claim here, the model in a
/// `--headless stt-worker` child. Every restart of the GPU worker is admitted again.
fn create_in_worker(spec: &EngineSpec, ctx: &EngineContext, opts: &Options) -> Result<WorkerEngine, AsrError> {
    let want = Want::from_variant(spec.variant.as_deref());
    let model = models::resolve(&spec.model).map_err(|e| asr_error_for(e, want))?;
    let label = claim_label(&model, true);
    let (admission, claims) = governor::admit_and_claim(want, model.gpu_mb, &model.title, &label, true);
    if admission.gpu.is_none() && admission.reason.is_some() && !opts.cpu_fallback {
        return Err(denial_error(&admission));
    }
    // The GPU the governor budgets for, decided here (the child never probes the system for it).
    let device_id = if admission.gpu.is_some() && opts.device_id.is_none() && opts.device.is_none() {
        primary_device_id()
    } else {
        None
    };
    let gpu_spec = child_spec(spec, device_id.as_deref(), &admission);
    let cpu_spec = crash_cpu_spec(spec, opts, &admission, &model.title);
    let readmit = admission.gpu.is_some().then(|| {
        let (spec, title, gpu_mb, cpu_fallback) = (spec.clone(), model.title.clone(), model.gpu_mb, opts.cpu_fallback);
        let device_id = device_id.clone();
        Box::new(move || {
            let (a, claims) = governor::admit_and_claim(want, gpu_mb, &title, &label, true);
            readmission(&a, claims, &spec, device_id.as_deref(), cpu_fallback, &title)
        }) as Readmit
    });
    let memory_limit_mb = opts
        .worker_memory_limit_mb
        .unwrap_or_else(|| worker_memory_limit_mb(model.file_bytes));
    WorkerEngine::spawn(WorkerSetup {
        spec: gpu_spec,
        readmit,
        cpu_spec,
        ctx: ctx.clone(),
        claims,
        memory_limit_mb: Some(memory_limit_mb),
        label: model.label.clone(),
    })
}

// ---------------------------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------------------------

/// The language to decode with: `Ok(Some(code))` = hint, `Ok(None)` = the model detects. The bool
/// is true when a hint was dropped because a detecting model does not list it.
pub(crate) fn decode_language(
    family: Family,
    title: &str,
    requested: Option<&str>,
    fallback: Option<&str>,
    languages: &[String],
    detects: bool,
) -> Result<(Option<String>, bool), AsrError> {
    let lang = requested
        .and_then(lang::normalize)
        .or_else(|| fallback.and_then(lang::normalize));
    match lang {
        Some(l) => {
            if family.never_languages().contains(&l.as_str()) {
                return Err(AsrError::unsupported(format!(
                    "{title} is never used for '{l}' (it drops sentences in that language); use Qwen3-ASR or the CPU route"
                )));
            }
            if languages.is_empty() || languages.contains(&l) {
                Ok((Some(l), false))
            } else if detects {
                Ok((None, true))
            } else {
                Err(AsrError::unsupported(format!(
                    "{title} does not transcribe '{l}' (it knows {})",
                    languages.join(", ")
                )))
            }
        }
        None if languages.len() == 1 => Ok((Some(languages[0].clone()), false)),
        None if family.needs_language() || (!detects && languages.len() > 1) => Err(AsrError::invalid_input(format!(
            "{title} cannot detect the spoken language; choose the language (it knows {})",
            languages.join(", ")
        ))),
        None => Ok((None, false)),
    }
}

/// Split `audio` into pieces of at most `max` samples, each cut at the quietest 20 ms window of the
/// last 30 % of the piece, never leaving a last piece shorter than [`MIN_TAIL_SAMPLES`] (or 30 %
/// of `max` when that is less): a cut that would is moved earlier (pure; returns ranges covering
/// all of `audio`).
pub(crate) fn split_points(audio: &[f32], max: usize) -> Vec<std::ops::Range<usize>> {
    const WIN: usize = SAMPLE_RATE as usize / 50;
    let max = max.max(WIN * 10);
    let min_tail = MIN_TAIL_SAMPLES.min(max * 3 / 10);
    let mut out = Vec::new();
    let mut start = 0;
    while audio.len() - start > max {
        let search_from = start + max * 7 / 10;
        // `len - start > max` and `min_tail <= 0.3 max` keep this at or after `search_from`.
        let search_to = (start + max).min(audio.len() - min_tail);
        let mut best = search_to;
        let mut best_energy = f32::INFINITY;
        let mut w = search_from;
        while w + WIN <= search_to {
            let e: f32 = audio[w..w + WIN].iter().map(|x| x * x).sum();
            if e < best_energy {
                best_energy = e;
                best = w + WIN / 2;
            }
            w += WIN / 2;
        }
        out.push(start..best);
        start = best;
    }
    out.push(start..audio.len());
    out
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0xAC00..=0xD7AF | 0x3000..=0x303F | 0xFF00..=0xFFEF)
}

/// Join transcripts of consecutive pieces: a space between words, none between CJK text (pure).
pub(crate) fn join_texts(parts: &[String]) -> String {
    let mut out = String::new();
    for p in parts.iter().map(|p| p.trim()).filter(|p| !p.is_empty()) {
        if let (Some(a), Some(b)) = (out.chars().last(), p.chars().next()) {
            if !(is_cjk(a) || is_cjk(b)) {
                out.push(' ');
            }
        }
        out.push_str(p);
    }
    out
}

/// Whisper's decoder prompt from the prompt and hotwords (pure; bounded, UTF-8 safe).
pub(crate) fn whisper_prompt(options: &DecodeOptions) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(p) = options.prompt.as_deref().map(str::trim).filter(|p| !p.is_empty()) {
        parts.push(p.to_string());
    }
    let words: Vec<&str> = options
        .hotwords
        .iter()
        .map(|h| h.trim())
        .filter(|h| !h.is_empty())
        .collect();
    if !words.is_empty() {
        parts.push(format!("{}.", words.join(", ")));
    }
    let mut prompt = parts.join(" ");
    if prompt.len() > MAX_PROMPT_BYTES {
        let mut cut = MAX_PROMPT_BYTES;
        while !prompt.is_char_boundary(cut) {
            cut -= 1;
        }
        prompt.truncate(cut);
    }
    (!prompt.is_empty()).then_some(prompt)
}

/// Why a model the governor admitted to GPU build `variant` runs on the CPU when no GPU attempt
/// failed (pure): the build was skipped (its reason from the variant walk) or is not installed.
pub(crate) fn gpu_not_used_reason(title: &str, variant: RuntimeVariant, attempts: &[Attempt]) -> String {
    match attempts.iter().find(|a| a.variant == Some(variant)) {
        Some(a) => format!("{title} runs on the CPU: the {} runtime was not used ({})", variant.label(), a.error),
        None => format!(
            "{title} runs on the CPU: the {} runtime (runtime/sigil-transcribe-{}) is not installed",
            variant.label(),
            variant.as_str()
        ),
    }
}

/// `EngineInfo::vram_mb` of a model on the CPU (pure): 0, except in a worker process whose GPU
/// runtime was initialized: that process still holds the runtime's CUDA context, which the parent
/// keeps claiming (it reports 0 itself; `worker_proc`'s VRAM docs). In-process the context has its
/// own pinned claim.
pub(crate) fn cpu_vram_mb(in_worker: bool, gpu_initialized: bool) -> u64 {
    if in_worker && gpu_initialized {
        models::GPU_CONTEXT_MB
    } else {
        0
    }
}

/// `EngineInfo::device` of a loaded model.
fn device_of(info: &ModelInfo) -> String {
    match info.device.as_ref().and_then(|d| d.kind.as_deref()) {
        Some(k) if info.on_gpu() => k.to_ascii_lowercase(),
        _ => "cpu".into(),
    }
}

/// Capabilities of a loaded model under its family's rules.
pub(crate) fn caps_of(info: &ModelInfo, family: Family) -> EngineCaps {
    let never = family.never_languages();
    let mut languages: Vec<String> = Vec::new();
    for code in info.capabilities.language_codes() {
        if let Some(l) = lang::normalize(code) {
            if !never.contains(&l.as_str()) && !languages.contains(&l) {
                languages.push(l);
            }
        }
    }
    let detects = info.capabilities.language_detect;
    let prompt = family.takes_prompt() && info.capabilities.features.initial_prompt;
    EngineCaps {
        streaming: info.capabilities.streaming,
        utterance: true,
        needs_language: family.needs_language() || (!detects && languages.len() > 1),
        detects_language: detects,
        hotwords: prompt,
        prompt,
        tags: detects,
        languages,
    }
}

// ---------------------------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------------------------

/// Cancellation shared with the canceller: a generation counter (results of a call during which it
/// changed are dropped) plus the native cancel.
struct CancelState {
    gen: AtomicU64,
    native: TranscribeCanceller,
}

/// One open stream.
struct StreamState {
    stream: TranscribeStream,
    options: TranscribeOptions,
    /// Pushed since `begin_stream` (mid-utterance: one compute per model).
    active: bool,
    /// The utterance so far, for a replay after a CPU fallback closed the stream.
    replay: Vec<f32>,
    replay_ok: bool,
    last: String,
    language: Option<String>,
}

/// The in-process transcribe.cpp engine.
pub struct TranscribeEngine {
    // Field order is drop order: the stream and the model release their memory before the claims.
    stream: Option<StreamState>,
    model: TranscribeModel,
    info: EngineInfo,
    family: Family,
    title: String,
    opts: Options,
    /// What the load runs on as the governor decided ([`effective_want`]), for error messages.
    want: Want,
    /// Hosted by a `--headless stt-worker` child.
    in_worker: bool,
    cancel: Arc<CancelState>,
    ctx: EngineContext,
    /// A dropped language hint was reported once.
    hint_note: bool,
    claims: GpuClaims,
}

impl std::fmt::Debug for TranscribeEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranscribeEngine")
            .field("model", &self.info.model)
            .field("device", &self.info.device)
            .finish_non_exhaustive()
    }
}

fn report(ctx: &EngineContext, code: &str, detail: impl Into<String>, model_id: Option<&str>, level: StatusLevel) {
    ctx.status.report(EngineStatus {
        code: code.to_string(),
        detail: detail.into(),
        progress: None,
        level,
        model_id: model_id.map(str::to_string),
    });
}

fn runtime_label(model: &TranscribeModel) -> String {
    let info = model.info();
    let version = if info.upstream_version.is_empty() {
        model.runtime().version.clone()
    } else {
        info.upstream_version.clone()
    };
    let variant = model
        .runtime()
        .variant
        .map_or("runtime folder", |v| v.as_str());
    format!("transcribe.cpp {version} {variant}")
}

impl TranscribeEngine {
    /// Load `spec`'s model on the device the governor admits (blocking).
    pub(crate) fn create(spec: &EngineSpec, ctx: &EngineContext, opts: Options) -> Result<TranscribeEngine, AsrError> {
        let want = Want::from_variant(spec.variant.as_deref());
        let model = models::resolve(&spec.model).map_err(|e| asr_error_for(e, want))?;
        let in_worker = worker_proc::is_worker_process();
        let (admission, mut claims) = match (&opts.admission, in_worker) {
            // The parent decided and holds the (external) claim for this whole process.
            (Some(fwd), true) => (fwd.to_admission(), GpuClaims::none()),
            _ => governor::admit_and_claim(want, model.gpu_mb, &model.title, &claim_label(&model, false), false),
        };
        if admission.gpu.is_none() && admission.reason.is_some() && !opts.cpu_fallback {
            return Err(denial_error(&admission));
        }
        // Missing runtimes are named for what this load really needs (a CPU pack for a CPU run).
        let want = effective_want(want, &admission);
        let primary = if admission.gpu.is_some() && opts.device_id.is_none() && opts.device.is_none() {
            primary_device_id()
        } else {
            None
        };
        let mo = model_options(&model, spec, &opts, want, &admission, primary);
        let place = match admission.gpu {
            Some(v) => format!("on the GPU ({})", v.label()),
            None => "on the processor".to_string(),
        };
        report(
            ctx,
            codes::LOADING_MODEL,
            format!("Loading {} {place}", model.title),
            model.catalog_id.as_deref(),
            StatusLevel::Info,
        );
        let adapter = admission.gpu.and_then(|_| governor::primary_adapter());
        let before = adapter.and_then(governor::sample);
        // Other in-process claims around the load: a load that overlapped another one (their
        // claims change) is measured together with it, so the estimate is kept then.
        let ledger_before = before.map(|_| governor::ledger_fingerprint(&crate::services::resources::snapshot()));
        let t0 = Instant::now();
        let loaded = TranscribeModel::open(&mo).map_err(|e| asr_error_for(e, want))?;
        let load_ms = t0.elapsed().as_millis();
        let on_gpu = loaded.info().on_gpu();
        let gpu_initialized = loaded.runtime().gpu_backend_initialized;
        let family = match Family::from_arch(loaded.info().arch.as_deref().unwrap_or("")) {
            Family::Other => model.family,
            f => f,
        };
        // A bring-your-own file is named after its family in messages ("Qwen3-ASR (my.gguf)").
        let title = if model.catalog_id.is_none() && family != Family::Other {
            format!("{} ({})", family.label(), model.title)
        } else {
            model.title.clone()
        };

        // Claims: the measured footprint on the GPU; on the CPU only a context the load left behind.
        let mut vram_mb: u64;
        if on_gpu {
            let measured_adapter = crate::services::sysinfo::cached();
            let device_adapter = loaded
                .info()
                .device
                .as_ref()
                .and_then(|d| governor::adapter_of(d, &measured_adapter))
                .or(adapter);
            let after = device_adapter.and_then(governor::sample);
            let fresh_process = in_worker;
            match (before, after) {
                (Some(b), Some(a)) if b.adapter == a.adapter => {
                    let created = claims.context_created() || fresh_process;
                    let quiet =
                        fresh_process || ledger_before.as_ref() == Some(&governor::ledger_fingerprint(&crate::services::resources::snapshot()));
                    let measured = if quiet {
                        governor::measured_model_mb(b.process_mb, a.process_mb, created, model.gpu_mb)
                    } else {
                        None
                    };
                    match measured {
                        Some(mb) => {
                            claims.resize_model(mb);
                            vram_mb = mb + if created || !claims.is_empty() { models::GPU_CONTEXT_MB } else { 0 };
                        }
                        None => vram_mb = model.gpu_mb + models::GPU_CONTEXT_MB,
                    }
                    tracing::info!(
                        target: LOG,
                        "{} on {}: process VRAM {} -> {} MiB (claimed model {} MB, runtime context {} MB{}{})",
                        model.label,
                        loaded.info().backend,
                        b.process_mb,
                        a.process_mb,
                        claims.model_mb(),
                        models::GPU_CONTEXT_MB,
                        if created { ", created by this load" } else { ", shared" },
                        if quiet { "" } else { "; other VRAM claims changed during the load, so the estimate is kept" }
                    );
                }
                _ => {
                    vram_mb = model.gpu_mb + models::GPU_CONTEXT_MB;
                    tracing::info!(
                        target: LOG,
                        "{} on {}: VRAM not measurable here; claimed the estimate ({} MB + {} MB context)",
                        model.label,
                        loaded.info().backend,
                        model.gpu_mb,
                        models::GPU_CONTEXT_MB
                    );
                }
            }
            if !claims.is_empty() {
                vram_mb = claims.total_mb();
            }
            // The GPU DLL (and its CUDA context) now stays loaded for the process lifetime.
            claims.pin_context();
        } else {
            // A GPU runtime this load initialized (its devices queried, a model tried on it) keeps
            // its DLL and CUDA context for the process lifetime although the model ended on the
            // CPU: its context claim stays (pinned); only the model claim goes.
            if gpu_initialized {
                claims.pin_context();
            }
            claims.release();
            vram_mb = cpu_vram_mb(in_worker, gpu_initialized);
        }
        let fallback_reason = if on_gpu {
            None
        } else {
            loaded
                .fallback_reason()
                .map(str::to_string)
                .or(admission.reason.clone())
                .or_else(|| {
                    // Admitted to the GPU, yet no GPU attempt failed: the GPU build was not
                    // installed (or skipped). Never a silent CPU run.
                    admission
                        .gpu
                        .map(|v| gpu_not_used_reason(&model.title, v, &loaded.runtime().attempts))
                })
        };
        if let Some(why) = fallback_reason.as_deref().filter(|_| admission.gpu.is_some()) {
            tracing::warn!(target: LOG, "{}: {why}", model.label);
        }
        let caps = caps_of(loaded.info(), family);
        let info = EngineInfo {
            kind: EngineKind::Transcribe,
            model: model.label.clone(),
            runtime: runtime_label(&loaded),
            device: device_of(loaded.info()),
            caps,
            vram_mb,
            fallback_reason,
        };
        tracing::info!(
            target: LOG,
            "{} ready on {} ({}, {} ms{}; governor: {})",
            model.label,
            info.device,
            info.runtime,
            load_ms,
            if vram_mb > 0 { format!(", ~{vram_mb} MB VRAM") } else { String::new() },
            admission.code
        );
        let cancel = Arc::new(CancelState {
            gen: AtomicU64::new(0),
            native: loaded.canceller(),
        });
        Ok(TranscribeEngine {
            stream: None,
            model: loaded,
            info,
            family,
            title,
            opts,
            want,
            in_worker,
            cancel,
            ctx: ctx.clone(),
            hint_note: false,
            claims,
        })
    }

    /// A wrapper error of this engine, missing runtimes named for what it runs on.
    fn error(&self, e: TranscribeError) -> AsrError {
        asr_error_for(e, self.want)
    }

    /// After every call: a model that moved to the CPU (VRAM headroom lost, a GPU failure) releases
    /// its claims and says why. (Its context claim was pinned when it loaded on the GPU: the DLL
    /// and its CUDA context stay.)
    fn sync_device(&mut self) {
        if self.model.info().on_gpu() || self.info.device == "cpu" {
            return;
        }
        self.claims.release();
        let reason = self
            .model
            .fallback_reason()
            .map(str::to_string)
            .unwrap_or_else(|| format!("{} moved to the CPU", self.title));
        tracing::warn!(target: LOG, "{}: {reason}", self.info.model);
        self.info.device = "cpu".into();
        self.info.vram_mb = cpu_vram_mb(self.in_worker, true);
        self.info.runtime = runtime_label(&self.model);
        self.info.fallback_reason = Some(reason.clone());
        report(&self.ctx, codes::ENGINE_FALLBACK, reason, None, StatusLevel::Warning);
    }

    fn language_for(&mut self, requested: Option<&str>) -> Result<Option<String>, AsrError> {
        let (lang, dropped) = decode_language(
            self.family,
            &self.title,
            requested,
            self.opts.language.as_deref(),
            &self.info.caps.languages,
            self.info.caps.detects_language,
        )?;
        if dropped && !self.hint_note {
            self.hint_note = true;
            tracing::info!(
                target: LOG,
                "{}: {:?} is not one of its languages; it detects the language instead",
                self.info.model,
                requested
            );
        }
        Ok(lang)
    }

    fn decode_options(&self, language: Option<String>, options: &DecodeOptions) -> TranscribeOptions {
        let whisper = self.family == Family::Whisper;
        TranscribeOptions {
            language,
            timestamps: Timestamps::None,
            punctuation: self.opts.punctuation,
            itn: self.opts.itn,
            prompt: if self.info.caps.prompt { whisper_prompt(options) } else { None },
            temperature_fallback: whisper.then_some(false),
            no_speech_threshold: if whisper { self.opts.no_speech_threshold } else { None },
            ..TranscribeOptions::default()
        }
    }

    fn cancelled_since(&self, gen: u64) -> bool {
        self.cancel.gen.load(Ordering::SeqCst) != gen
    }

    fn stream_options(&self, language: Option<String>) -> TranscribeOptions {
        let commit = self
            .opts
            .commit
            .clone()
            .or_else(|| self.opts.append_only.then(|| "stable_prefix".to_string()));
        TranscribeOptions {
            language,
            timestamps: Timestamps::None,
            punctuation: self.opts.punctuation,
            itn: self.opts.itn,
            commit,
            agreement: self.opts.agreement,
            ..TranscribeOptions::default()
        }
    }

    /// Partials never retract: the `appendOnly` option, or a family whose streaming is append-only
    /// by design (Confucius4-R2T2, committed by its own stable-prefix decoding).
    fn append_only(&self) -> bool {
        self.opts.append_only || self.family.append_only()
    }

    /// The partial to show for an update.
    fn shown(&self, update: &crate::services::speech_rt::transcribe::StreamUpdate) -> String {
        let text = if self.append_only() { &update.committed } else { &update.partial };
        text.trim().to_string()
    }
}

impl AsrEngine for TranscribeEngine {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn transcribe(&mut self, audio: &[f32], options: &DecodeOptions) -> Result<AsrResult, AsrError> {
        let gen = self.cancel.gen.load(Ordering::SeqCst);
        let t0 = Instant::now();
        if audio.is_empty() {
            return Ok(AsrResult::default());
        }
        if self.stream.as_ref().is_some_and(|s| s.active) {
            return Err(AsrError::new(
                AsrErrorKind::Busy,
                "this engine's stream is mid-utterance (one decode per model at a time)",
            ));
        }
        let language = self.language_for(options.language.as_deref())?;
        let topts = self.decode_options(language.clone(), options);
        let mut max = (self.opts.max_segment_seconds * SAMPLE_RATE as f32) as usize;
        let model_max_ms = self.model.info().max_audio_ms();
        if model_max_ms > 0 {
            max = max.min((model_max_ms as usize).saturating_mul(SAMPLE_RATE as usize) / 1000);
        }
        let mut texts = Vec::new();
        let mut detected: Option<String> = None;
        let mut padded = Vec::new();
        for range in split_points(audio, max) {
            let piece = &audio[range];
            // Every piece, not only a short utterance: the last piece of a split one can be short.
            let piece = if piece.len() < MIN_AUDIO_SAMPLES {
                padded.clear();
                padded.extend_from_slice(piece);
                padded.resize(MIN_AUDIO_SAMPLES, 0.0);
                &padded[..]
            } else {
                piece
            };
            let result = self.model.transcribe(piece, &topts);
            self.sync_device();
            let t = result.map_err(|e| self.error(e))?;
            if self.cancelled_since(gen) {
                return Err(AsrError::cancelled());
            }
            if t.truncated {
                tracing::warn!(target: LOG, "{}: transcript truncated at the model's generation limit", self.info.model);
            }
            if detected.is_none() && t.language_source == "detected" {
                detected = lang::normalize(&t.language);
            }
            texts.push(t.text);
        }
        if self.cancelled_since(gen) {
            return Err(AsrError::cancelled());
        }
        let lang = language.clone().or(detected.clone());
        let tags = match (&language, &detected) {
            (None, Some(d)) => vec![format!("lang:{d}")],
            _ => Vec::new(),
        };
        Ok(AsrResult {
            text: join_texts(&texts),
            lang,
            tags,
            no_speech_prob: None,
            confidence: None,
            decode_ms: t0.elapsed().as_millis().min(u128::from(u32::MAX)) as u32,
        })
    }

    fn begin_stream(&mut self, options: &DecodeOptions) -> Result<(), AsrError> {
        if !self.info.caps.streaming {
            return Err(AsrError::unsupported(format!("{} does not stream; it decodes utterances", self.title)));
        }
        let language = self.language_for(options.language.as_deref())?;
        let topts = self.stream_options(language.clone());
        // Reuse an idle stream with the same options; otherwise (a second begin aborts the first).
        let mut reusable = self
            .stream
            .as_ref()
            .is_some_and(|s| !s.active && s.options == topts && s.stream.is_open());
        if reusable {
            // The VRAM-headroom guard runs per utterance, as for finals: `open_stream` checks only
            // when a stream opens. A move to the CPU closes the stream, which then reopens there.
            let checked = self.model.check_headroom();
            self.sync_device();
            checked.map_err(|e| self.error(e))?;
            reusable = self.stream.as_ref().is_some_and(|s| s.stream.is_open());
        }
        if !reusable {
            self.stream = None;
            let stream = self.model.open_stream(&topts);
            self.sync_device();
            let stream = stream.map_err(|e| self.error(e))?;
            self.stream = Some(StreamState {
                stream,
                options: topts,
                active: false,
                replay: Vec::new(),
                replay_ok: true,
                last: String::new(),
                language,
            });
        } else if let Some(s) = self.stream.as_mut() {
            s.language = language;
        }
        if let Some(s) = self.stream.as_mut() {
            s.replay.clear();
            s.replay_ok = true;
            s.last.clear();
        }
        Ok(())
    }

    fn push_stream(&mut self, audio: &[f32]) -> Result<Option<Hypothesis>, AsrError> {
        let gen = self.cancel.gen.load(Ordering::SeqCst);
        if self.stream.is_none() {
            return Err(AsrError::invalid_input("no stream is open (begin_stream first)"));
        }
        if audio.is_empty() {
            return Ok(None);
        }
        let update = {
            let s = self.stream.as_mut().expect("checked");
            s.active = true;
            if s.replay_ok {
                if s.replay.len() + audio.len() <= MAX_REPLAY_SAMPLES {
                    s.replay.extend_from_slice(audio);
                } else {
                    s.replay_ok = false;
                    s.replay = Vec::new();
                }
            }
            s.stream.push(audio)
        };
        self.sync_device();
        let update = match update {
            Err(e) if e.kind == ErrorKind::Backend && self.stream.as_ref().is_some_and(|s| !s.stream.is_open()) => {
                // A CPU fallback closed the stream: reopen on the CPU copy and replay the utterance.
                let (opts, replay, replay_ok) = {
                    let s = self.stream.as_ref().expect("checked");
                    (s.options.clone(), s.replay.clone(), s.replay_ok)
                };
                if !replay_ok {
                    self.stream = None;
                    return Err(AsrError::runtime(format!(
                        "{}: the stream was closed by a move to the CPU mid-utterance",
                        self.info.model
                    )));
                }
                let mut stream = self.model.open_stream(&opts).map_err(|e| self.error(e))?;
                let r = stream.push(&replay);
                if let Some(s) = self.stream.as_mut() {
                    s.stream = stream;
                }
                r
            }
            other => other,
        };
        match update {
            Ok(update) => {
                if self.cancelled_since(gen) {
                    return Err(AsrError::cancelled());
                }
                let Some(u) = update else { return Ok(None) };
                let text = self.shown(&u);
                let s = self.stream.as_mut().expect("checked");
                if text == s.last {
                    return Ok(None);
                }
                s.last = text.clone();
                Ok(Some(Hypothesis {
                    text,
                    lang: s.language.clone(),
                }))
            }
            Err(e) => {
                // A failed push ends the utterance's claim on the model (the shim marks it FAILED).
                if let Some(s) = self.stream.as_mut() {
                    s.active = false;
                }
                Err(self.error(e))
            }
        }
    }

    fn end_stream(&mut self) -> Result<AsrResult, AsrError> {
        let gen = self.cancel.gen.load(Ordering::SeqCst);
        let t0 = Instant::now();
        let Some(s) = self.stream.as_mut() else {
            return Err(AsrError::invalid_input("no stream is open"));
        };
        if !s.active {
            // Nothing pushed since begin: nothing recognized.
            return Ok(AsrResult {
                lang: s.language.clone(),
                ..AsrResult::default()
            });
        }
        let result = s.stream.finish();
        s.active = false;
        s.replay.clear();
        s.last.clear();
        let hint = s.language.clone();
        self.sync_device();
        let f = result.map_err(|e| self.error(e))?;
        if self.cancelled_since(gen) {
            return Err(AsrError::cancelled());
        }
        let detected = lang::normalize(&f.language).filter(|_| hint.is_none());
        Ok(AsrResult {
            text: f.text.trim().to_string(),
            lang: hint.clone().or(detected.clone()),
            tags: detected.map(|d| vec![format!("lang:{d}")]).unwrap_or_default(),
            no_speech_prob: None,
            confidence: None,
            decode_ms: t0.elapsed().as_millis().min(u128::from(u32::MAX)) as u32,
        })
    }

    fn abort_stream(&mut self) {
        // transcribe.cpp has no reset: closing the session is the cheap way to drop the utterance.
        if self.stream.as_ref().is_some_and(|s| s.active) {
            self.stream = None;
        }
    }

    fn canceller(&self) -> Option<Canceller> {
        let state = Arc::clone(&self.cancel);
        Some(Canceller::new(move || {
            state.gen.fetch_add(1, Ordering::SeqCst);
            state.native.cancel();
        }))
    }
}

#[cfg(test)]
mod tests;
