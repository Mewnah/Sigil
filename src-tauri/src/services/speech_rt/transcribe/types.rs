//! Options sent to, and results parsed from, `sigil_transcribe.dll` (the JSON contract documented
//! at the top of `native/engines/transcribe/shim.cpp`). Every result type tolerates missing fields
//! (`#[serde(default)]`) so a newer engine that adds fields never breaks an older Sigil.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::path::PathBuf;

use crate::services::native_runtime::RuntimeVariant;

// ---------------------------------------------------------------------------------------------
// Opening a model
// ---------------------------------------------------------------------------------------------

/// Which compute backend the model should run on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendChoice {
    /// Best installed variant this PC supports (CUDA 13, then Vulkan, then the CPU variants), each
    /// on its own backend; a GPU failure falls through to the next variant.
    #[default]
    Auto,
    /// GPU variants only (CUDA 13, then Vulkan); an error when none can run.
    Gpu,
    /// The best CPU variant only (no GPU driver is touched).
    Cpu,
}

/// Precision of the attention K/V cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvType {
    /// f16 for quantized models, f32 for f32 models (transcribe.cpp's default).
    #[default]
    Auto,
    F16,
    F32,
}

impl KvType {
    fn as_str(self) -> &'static str {
        match self {
            KvType::Auto => "auto",
            KvType::F16 => "f16",
            KvType::F32 => "f32",
        }
    }
}

/// How to load a model (see `TranscribeModel::open`).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelOptions {
    /// GGUF model (or a legacy whisper.cpp `.bin`).
    pub model: PathBuf,
    pub backend: BackendChoice,
    /// Only this runtime variant (no fallback between variants).
    pub variant: Option<RuntimeVariant>,
    /// Load `sigil_transcribe.dll` from this folder instead of the installed packs (development,
    /// engine probes). Also taken from `SIGIL_TRANSCRIBE_RUNTIME_DIR`.
    pub runtime_dir: Option<PathBuf>,
    /// CPU threads for the ops that run on the CPU; `None` = [`default_threads`].
    pub threads: Option<u32>,
    /// GPU only: index of the device among the devices of the first GPU backend that loads (e.g.
    /// the second CUDA GPU = 1). The wrapper turns it into the device's stable id (PCI bus id) for
    /// any later GPU backend, whose device order differs; CPU backends never get it.
    pub device: Option<u32>,
    /// GPU only: stable device id as `{"op":"devices"}` reports it (`device_id`, the PCI bus id, or
    /// the `description`); wins over [`Self::device`]. A backend without that device fails over
    /// to the next variant instead of picking another GPU.
    pub device_id: Option<String>,
    pub kv_type: KvType,
    /// Cap on the decoder context in tokens (bounds KV-cache memory); `None` = the model's maximum.
    pub n_ctx: Option<u32>,
    /// When the GPU backend reports a failure at load or at run time (a status code:
    /// `SIGIL_E_BACKEND_UNAVAILABLE` / out of device memory), or the device lacks the VRAM
    /// headroom below, reload the model on the CPU backend of the same runtime and report why
    /// ([`super::TranscribeModel::fallback_reason`]). CUDA errors that ggml-cuda cannot return (a
    /// failed memory-pool growth, a lost device) abort the process instead and cannot be recovered
    /// in-process; the headroom checks exist to keep runs away from them (ADR 0003b).
    pub cpu_fallback: bool,
    /// Run a warm-up decode after loading (GPU kernels / pipelines, graph and memory-pool sizing).
    pub warmup: bool,
    /// Audio length of the warm-up; `None` = [`GPU_WARMUP_SECONDS`] on a GPU (the longest
    /// utterance Sigil sends, so ggml-cuda's pool is sized at load time), [`CPU_WARMUP_SECONDS`]
    /// on the CPU. Clamped to the model's input limit and to 0.1..=30 s.
    pub warmup_seconds: Option<f32>,
    /// GPU only: free device memory (MB) required before loading (on top of the model's estimated
    /// need) and before every run; below it the model moves to the CPU (`cpu_fallback`) or the
    /// call fails with `OutOfMemory`. 0 disables the checks. Default [`DEFAULT_MIN_FREE_VRAM_MB`].
    pub min_free_vram_mb: u32,
}

impl ModelOptions {
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            backend: BackendChoice::Auto,
            variant: None,
            runtime_dir: None,
            threads: None,
            device: None,
            device_id: None,
            kv_type: KvType::Auto,
            n_ctx: None,
            cpu_fallback: true,
            warmup: true,
            warmup_seconds: None,
            min_free_vram_mb: DEFAULT_MIN_FREE_VRAM_MB,
        }
    }

    /// The GPU device this model asks for (`device_id` wins over `device`).
    pub(crate) fn device_ref(&self) -> Option<DeviceRef> {
        self.device_id
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(|id| DeviceRef::Id(id.to_string()))
            .or(self.device.map(DeviceRef::Index))
    }
}

/// The `device` member of a create config (GPU backends only).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DeviceRef {
    /// Index among the backend's own devices.
    Index(u32),
    /// Stable id (`device_id` / `description`), the same on every backend.
    Id(String),
}

impl DeviceRef {
    /// The id a device found through `self` keeps on every backend: its PCI bus id, else its
    /// description (the backend-specific `name`, e.g. `CUDA0`, never). `None`: keep `self`.
    pub(crate) fn stable_for(device: &DeviceInfo) -> Option<DeviceRef> {
        [&device.device_id, &device.description]
            .into_iter()
            .flatten()
            .map(|s| s.trim())
            .find(|s| !s.is_empty())
            .map(|s| DeviceRef::Id(s.to_string()))
    }
}

/// Whether `backend` (a `sigil_engine_create` backend name) runs on a GPU.
pub(crate) fn is_gpu_backend(backend: &str) -> bool {
    matches!(backend, "cuda" | "vulkan")
}

/// Pick the device of `backend` that `device` names (the backend's first device when `None`) from
/// a `{"op":"devices"}` answer.
pub(crate) fn select_device(devices: &[DeviceInfo], backend: &str, device: Option<&DeviceRef>) -> Option<DeviceInfo> {
    let mut of_kind = devices
        .iter()
        .filter(|d| d.kind.as_deref() == Some(backend));
    match device {
        None => of_kind.next(),
        Some(DeviceRef::Index(i)) => of_kind.nth(*i as usize),
        Some(DeviceRef::Id(id)) => of_kind.find(|d| {
            d.device_id
                .as_deref()
                .is_some_and(|v| v.eq_ignore_ascii_case(id))
                || d.name.as_deref() == Some(id.as_str())
                || d.description.as_deref() == Some(id.as_str())
        }),
    }
    .cloned()
}

/// Longest utterance Sigil sends in one call (P2's VAD cap is 15 to 20 s): GPU warm-ups decode this
/// much, so ggml-cuda's memory pool and the compute graphs reach their working size at load time
/// instead of growing on the first long utterance (a failed pool growth aborts the process).
pub const GPU_WARMUP_SECONDS: f32 = 20.0;
/// CPU warm-up length (CPU buffers grow through malloc, whose failure is reported, not fatal).
pub const CPU_WARMUP_SECONDS: f32 = 1.0;
/// Default [`ModelOptions::min_free_vram_mb`]. Qwen3-ASR-1.7B Q8_0 grew 92 to 144 MB after a 1 s
/// warm-up on 11 to 28 s clips; a warm-up at [`GPU_WARMUP_SECONDS`] leaves less to grow.
pub const DEFAULT_MIN_FREE_VRAM_MB: u32 = 256;

/// Seconds of audio the warm-up decodes (see [`ModelOptions::warmup_seconds`]).
pub(crate) fn warmup_seconds(requested: Option<f32>, on_gpu: bool, max_audio_ms: i64) -> f64 {
    let default = if on_gpu { GPU_WARMUP_SECONDS } else { CPU_WARMUP_SECONDS };
    let mut s = f64::from(
        requested
            .filter(|s| s.is_finite() && *s > 0.0)
            .unwrap_or(default),
    );
    if max_audio_ms > 0 {
        s = s.min(max_audio_ms as f64 / 1000.0);
    }
    s.clamp(0.1, 30.0)
}

const MIB: u64 = 1024 * 1024;

/// Device memory (MB) a model needs besides the CUDA context: its weights plus about half again for
/// compute buffers and the KV cache (measured, ADR 0003b: Qwen3-ASR-1.7B Q8_0, a 2084 MiB file,
/// took about 2970 MiB after load and warm-up; whisper-tiny about its file size).
pub(crate) fn estimated_model_vram_mb(model_file_bytes: u64) -> u64 {
    (model_file_bytes / MIB).saturating_mul(3) / 2
}

/// Why `device` cannot take `need_mb` more (a message), or `None` when it can or its free memory is
/// unknown (0).
pub(crate) fn vram_shortfall(device: &DeviceInfo, need_mb: u64) -> Option<String> {
    if device.memory_free == 0 || need_mb == 0 {
        return None;
    }
    let free_mb = device.memory_free / MIB;
    (free_mb < need_mb).then(|| {
        format!(
            "only {free_mb} MB of {} MB VRAM is free on {} ({} MB needed)",
            device.memory_total / MIB,
            device
                .description
                .as_deref()
                .or(device.name.as_deref())
                .unwrap_or("the GPU"),
            need_mb
        )
    })
}

/// Default CPU threads: GPU backends only run small host-side ops (2 threads); the CPU backend
/// uses about half the logical processors (the physical cores on SMT CPUs), 2 to 6, so a
/// CPU-bound game keeps its cores.
pub fn default_threads(gpu: bool) -> u32 {
    if gpu {
        return 2;
    }
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4) as u32;
    (logical / 2).clamp(2, 6)
}

/// `sigil_engine_create` config JSON. `backend`: `None` = the variant's own backend. The device
/// (`device`, else the options' own) is sent only to an explicit GPU backend: the shim counts
/// devices per backend kind, so a GPU index on the CPU backend (which has one device) would fail
/// the CPU fallback itself.
pub(crate) fn create_config(opts: &ModelOptions, backend: Option<&str>, threads: u32, device: Option<&DeviceRef>) -> String {
    let mut cfg = Map::new();
    cfg.insert("model".into(), Value::String(opts.model.to_string_lossy().into_owned()));
    if let Some(b) = backend {
        cfg.insert("backend".into(), Value::String(b.into()));
        if is_gpu_backend(b) {
            match device.cloned().or_else(|| opts.device_ref()) {
                Some(DeviceRef::Index(i)) => {
                    cfg.insert("device".into(), json!(i));
                }
                Some(DeviceRef::Id(id)) => {
                    cfg.insert("device".into(), Value::String(id));
                }
                None => {}
            }
        }
    }
    cfg.insert("threads".into(), json!(threads.min(64)));
    cfg.insert("kv_type".into(), Value::String(opts.kv_type.as_str().into()));
    if let Some(n) = opts.n_ctx {
        cfg.insert("n_ctx".into(), json!(n));
    }
    Value::Object(cfg).to_string()
}

// ---------------------------------------------------------------------------------------------
// Transcribing
// ---------------------------------------------------------------------------------------------

/// Timestamp granularity to request (clamped by the engine to what the model can produce).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Timestamps {
    None,
    #[default]
    Segment,
    Word,
    Token,
    /// The finest the model supports.
    Auto,
}

impl Timestamps {
    fn as_str(self) -> &'static str {
        match self {
            Timestamps::None => "none",
            Timestamps::Segment => "segment",
            Timestamps::Word => "word",
            Timestamps::Token => "token",
            Timestamps::Auto => "auto",
        }
    }
}

/// A `default` / `on` / `off` model toggle (punctuation, inverse text normalization, diarization).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Toggle {
    #[default]
    Default,
    On,
    Off,
}

impl Toggle {
    fn as_str(self) -> &'static str {
        match self {
            Toggle::Default => "default",
            Toggle::On => "on",
            Toggle::Off => "off",
        }
    }
}

/// Per-utterance options.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TranscribeOptions {
    /// Language hint (`ja`, `en`, ...); `None` or `auto` = detect (when the model can).
    pub language: Option<String>,
    /// Translate into `target_language` (English for Whisper) instead of transcribing.
    pub translate: bool,
    pub target_language: Option<String>,
    pub timestamps: Timestamps,
    /// Also return word timings (when the model produces them).
    pub words: bool,
    pub punctuation: Toggle,
    pub itn: Toggle,
    /// Whisper only: initial prompt (names, vocabulary).
    pub prompt: Option<String>,
    /// Whisper only: `Some(false)` disables the temperature-fallback re-decodes (bounded latency).
    pub temperature_fallback: Option<bool>,
    /// Whisper only: no-speech probability above which a window is dropped (default 0.6).
    pub no_speech_threshold: Option<f32>,
    /// Streams only: `auto` / `on_finalize` / `stable_prefix`.
    pub commit: Option<String>,
}

impl TranscribeOptions {
    /// Options JSON for `sigil_engine_process` / `sigil_stream_open` (only the set fields).
    pub(crate) fn to_json(&self) -> String {
        let mut o = Map::new();
        if let Some(lang) = self
            .language
            .as_deref()
            .map(str::trim)
            .filter(|l| !l.is_empty())
        {
            o.insert("language".into(), Value::String(lang.to_string()));
        }
        if self.translate {
            o.insert("task".into(), Value::String("translate".into()));
        }
        if let Some(t) = self.target_language.as_deref().filter(|t| !t.is_empty()) {
            o.insert("target_language".into(), Value::String(t.to_string()));
        }
        o.insert("timestamps".into(), Value::String(self.timestamps.as_str().into()));
        if self.words {
            o.insert("words".into(), Value::Bool(true));
        }
        if self.punctuation != Toggle::Default {
            o.insert("pnc".into(), Value::String(self.punctuation.as_str().into()));
        }
        if self.itn != Toggle::Default {
            o.insert("itn".into(), Value::String(self.itn.as_str().into()));
        }
        if let Some(p) = self
            .prompt
            .as_deref()
            .map(str::trim)
            .filter(|p| !p.is_empty())
        {
            o.insert("prompt".into(), Value::String(p.to_string()));
        }
        if let Some(f) = self.temperature_fallback {
            o.insert("temperature_fallback".into(), Value::Bool(f));
        }
        if let Some(t) = self.no_speech_threshold.filter(|t| t.is_finite()) {
            o.insert("no_speech_threshold".into(), json!(t.clamp(0.0, 1.0)));
        }
        if let Some(c) = self.commit.as_deref().filter(|c| !c.is_empty()) {
            o.insert("commit".into(), Value::String(c.to_string()));
        }
        Value::Object(o).to_string()
    }
}

// ---------------------------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------------------------

/// A compute device as the engine reports it (`{"op":"devices"}` lists them; `null` entries are
/// devices the engine could not describe).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeviceList {
    pub devices: Vec<Option<DeviceInfo>>,
}

/// A compute device as the engine reports it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeviceInfo {
    pub name: Option<String>,
    pub description: Option<String>,
    /// `cpu`, `cuda`, `vulkan`, ...
    pub kind: Option<String>,
    /// PCI bus id when the backend reports one.
    pub device_id: Option<String>,
    #[serde(rename = "type")]
    pub device_type: Option<String>,
    /// Bytes (device-wide for a discrete GPU, system RAM for the CPU); 0 = unknown.
    pub memory_total: u64,
    /// Free bytes when the engine answered (a snapshot).
    pub memory_free: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Features {
    pub initial_prompt: bool,
    pub temperature_fallback: bool,
    pub long_form: bool,
    pub cancellation: bool,
    pub pnc: bool,
    pub itn: bool,
    pub diarization: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Capabilities {
    pub sample_rate: i64,
    pub languages: Vec<Option<String>>,
    pub language_detect: bool,
    pub translate: bool,
    pub streaming: bool,
    pub spec_decode: bool,
    pub max_timestamps: String,
    /// Longest clip one call accepts (0 = no practical limit).
    pub max_audio_ms: i64,
    pub features: Features,
}

impl Capabilities {
    /// Language codes the model supports.
    pub fn language_codes(&self) -> Vec<&str> {
        self.languages.iter().filter_map(|l| l.as_deref()).collect()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Limits {
    pub n_ctx: i64,
    pub max_audio_ms: i64,
    pub max_kv_bytes: i64,
}

/// `{"op":"info"}` of a loaded model.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub variant: String,
    /// The backend the model actually runs on (`CPU`, `CUDA0`, `Vulkan0`, ...).
    pub backend: String,
    pub requested_backend: String,
    pub upstream_version: String,
    pub upstream_commit: String,
    pub arch: Option<String>,
    pub model_variant: Option<String>,
    pub model_name: Option<String>,
    pub device: Option<DeviceInfo>,
    pub capabilities: Capabilities,
    pub limits: Option<Limits>,
    pub threads: i64,
    pub load_ms: f64,
}

impl ModelInfo {
    /// Whether the model runs on a GPU backend.
    pub fn on_gpu(&self) -> bool {
        self.device
            .as_ref()
            .and_then(|d| d.kind.as_deref())
            .is_some_and(|k| !matches!(k, "cpu" | "accel"))
    }

    /// The effective input limit in ms (0 = none).
    pub fn max_audio_ms(&self) -> i64 {
        match &self.limits {
            Some(l) if l.max_audio_ms > 0 => l.max_audio_ms,
            _ => self.capabilities.max_audio_ms,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    /// 1-based speaker (diarizing models only).
    pub speaker: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Word {
    pub start_ms: i64,
    pub end_ms: i64,
    pub segment: i64,
    pub text: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Timings {
    pub load_ms: f64,
    pub mel_ms: f64,
    pub encode_ms: f64,
    pub decode_ms: f64,
    pub wall_ms: f64,
}

/// Result of one offline transcription.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Transcript {
    pub text: String,
    /// The hint, the detected language, or empty.
    pub language: String,
    /// `hint`, `detected` or `none`.
    pub language_source: String,
    /// Granularity actually returned.
    pub timestamps: String,
    pub segments: Vec<Segment>,
    pub words: Vec<Word>,
    /// The decode hit the model's context / generation cap: `text` is incomplete.
    pub truncated: bool,
    pub audio_ms: f64,
    pub wall_ms: f64,
    /// `wall_ms / audio_ms` (lower is faster; < 1 is faster than real time).
    pub rtf: f64,
    pub timings: Timings,
    pub backend: String,
}

/// A streaming hypothesis update (`sigil_stream_push`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct StreamUpdate {
    /// `committed + tentative`: what a caption should show (committed never flickers).
    pub partial: String,
    pub committed: String,
    pub tentative: String,
    /// The model's raw current hypothesis.
    pub text: String,
    pub revision: i64,
    pub input_ms: i64,
    pub committed_audio_ms: i64,
    pub buffered_ms: i64,
    pub committed_changed: bool,
}

/// The final result of one streamed utterance (`sigil_stream_finish`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct StreamFinal {
    /// The authoritative final transcript.
    pub text: String,
    pub committed: String,
    pub truncated: bool,
    pub language: String,
    pub input_ms: i64,
}
