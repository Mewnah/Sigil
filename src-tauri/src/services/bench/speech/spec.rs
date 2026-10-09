//! `sigil-speech-bench-spec/1`: the input of `Sigil.exe --headless speech-bench` (frozen by P7-S0,
//! `scripts/speech-bench/schemas/speech-bench-spec.schema.json`).
//!
//! - Every struct denies unknown fields; `$comment` is allowed anywhere (removed before the strict
//!   parse, kept in the spec the report echoes and hashes).
//! - [`SpeechSpec::parse`] also applies the rules a schema cannot express (the Rust twin of
//!   `scripts/speech-bench/validate.py::spec_errors`): unique engine and run names, a backend that
//!   belongs to its task, runs that name existing engines whose task the mode takes, no secret-looking
//!   option names, STT runs with a corpus.
//! - Path fields expand `${NAME}` from the environment ([`expand_vars`]; `SIGIL_P7_ROOT` and
//!   `SIGIL_P7_CORPUS` have defaults) and are relative to the spec's folder ([`Paths`]).
//! - [`canonical_json`] / [`spec_sha256`]: the report's `spec_sha256` (sorted keys, no whitespace,
//!   UTF-8), the same bytes as `scripts/speech-bench/score/common.py::canonical_sha256`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::services::bench::spec::{GpuLoadKind, LoadSpec};

/// Schema id of the spec.
pub const SPEC_SCHEMA: &str = "sigil-speech-bench-spec/1";
/// Default `seed`.
pub const DEFAULT_SEED: u64 = 20_261_002;
/// Default `gpu_lock.tag`.
pub const DEFAULT_LOCK_TAG: &str = "P7-A";

// ---------------------------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------------------------

/// The whole spec.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechSpec {
    pub schema: String,
    /// Names the default output folder (`bench-results/speech/<name>-<timestamp>/`).
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Seed of every pseudo-random choice (item order of paced feeds, load jitter).
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub corpus: Option<CorpusDef>,
    pub engines: Vec<EngineDef>,
    pub runs: Vec<RunDef>,
    #[serde(default)]
    pub output: OutputDef,
}

fn default_seed() -> u64 {
    DEFAULT_SEED
}

/// The audio corpus and the default item selection.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusDef {
    /// Path of `corpus.json` (`${SIGIL_P7_CORPUS}` expands).
    pub manifest: String,
    #[serde(default)]
    pub select: Select,
}

/// Which corpus items a run feeds (filters combine with AND).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Select {
    /// A named subset of the manifest: its condition list.
    #[serde(default)]
    pub subset: Option<String>,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub conditions: Vec<String>,
    #[serde(default)]
    pub kinds: Vec<ItemKind>,
    /// Utterances per language, by rank, applied before the conditions.
    #[serde(default)]
    pub limit_per_language: Option<usize>,
    /// Explicit item ids (override every other filter when non-empty).
    #[serde(default)]
    pub ids: Vec<String>,
}

/// `synthetic` (built from FLEURS + bleed) or `real` (drop-in clips).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ItemKind {
    Synthetic,
    Real,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    Stt,
    Mt,
    Tts,
}

impl Task {
    pub fn as_str(self) -> &'static str {
        match self {
            Task::Stt => "stt",
            Task::Mt => "mt",
            Task::Tts => "tts",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionProvider {
    #[default]
    Auto,
    Cpu,
    Cuda,
    Dml,
    Vulkan,
}

impl ExecutionProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            ExecutionProvider::Auto => "auto",
            ExecutionProvider::Cpu => "cpu",
            ExecutionProvider::Cuda => "cuda",
            ExecutionProvider::Dml => "dml",
            ExecutionProvider::Vulkan => "vulkan",
        }
    }

    /// A GPU was asked for explicitly.
    pub fn wants_gpu(self) -> bool {
        matches!(self, ExecutionProvider::Cuda | ExecutionProvider::Dml | ExecutionProvider::Vulkan)
    }
}

/// STT: pass the item's language to the engine, or let it detect.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LanguageMode {
    #[default]
    Given,
    Auto,
}

/// One engine a run may use.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineDef {
    /// Unique; names hypothesis / audio files and report rows.
    pub id: String,
    pub task: Task,
    pub backend: String,
    /// Catalog id (never downloaded by the runner) or an absolute path (bring-your-own).
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub execution_provider: ExecutionProvider,
    #[serde(default)]
    pub variant: Option<String>,
    #[serde(default)]
    pub threads: Option<u32>,
    /// STT: the corpus languages it is fed; TTS: the prompt languages it renders.
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub language_mode: LanguageMode,
    /// MT: the directions it translates.
    #[serde(default)]
    pub directions: Vec<String>,
    /// Backend-specific settings (no secrets: `apiKeyEnv` names a variable).
    #[serde(default = "empty_object")]
    pub options: Value,
}

fn empty_object() -> Value {
    Value::Object(serde_json::Map::new())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Accuracy,
    CpuSpeed,
    GpuUnderLoad,
    MtQuality,
    TtsListening,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Accuracy => "accuracy",
            Mode::CpuSpeed => "cpu_speed",
            Mode::GpuUnderLoad => "gpu_under_load",
            Mode::MtQuality => "mt_quality",
            Mode::TtsListening => "tts_listening",
        }
    }

    /// The tasks a run of this mode takes.
    pub fn tasks(self) -> &'static [Task] {
        match self {
            Mode::Accuracy => &[Task::Stt],
            Mode::CpuSpeed | Mode::GpuUnderLoad => &[Task::Stt, Task::Mt, Task::Tts],
            Mode::MtQuality => &[Task::Mt],
            Mode::TtsListening => &[Task::Tts],
        }
    }

    /// A timing mode: warm-up and repeats apply, the GPU lock defaults to required.
    pub fn is_speed(self) -> bool {
        matches!(self, Mode::CpuSpeed | Mode::GpuUnderLoad)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FeedKind {
    /// The whole clip through `AsrEngine::transcribe`.
    #[default]
    Utterance,
    /// `begin_stream` / `push_stream(chunk_ms)` / `end_stream`.
    Stream,
    /// The real speech pipeline (Silero VAD segmentation, router, two-pass).
    Pipeline,
}

impl FeedKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FeedKind::Utterance => "utterance",
            FeedKind::Stream => "stream",
            FeedKind::Pipeline => "pipeline",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pace {
    /// As fast as the engine takes it (accuracy, compute).
    #[default]
    Asap,
    /// Chunks at wall-clock pace (latency to final, first partial).
    Realtime,
}

/// How audio reaches an STT engine.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Feed {
    #[serde(default)]
    pub kind: FeedKind,
    #[serde(default = "default_chunk_ms")]
    pub chunk_ms: u32,
    #[serde(default)]
    pub pace: Pace,
    /// Silence fed before the clip (pipeline: VAD pre-roll).
    #[serde(default)]
    pub lead_silence_ms: u32,
    /// Silence fed after the clip so a VAD can close the utterance.
    #[serde(default = "default_tail_silence_ms")]
    pub tail_silence_ms: u32,
}

fn default_chunk_ms() -> u32 {
    100
}

fn default_tail_silence_ms() -> u32 {
    800
}

impl Default for Feed {
    fn default() -> Self {
        Feed {
            kind: FeedKind::Utterance,
            chunk_ms: default_chunk_ms(),
            pace: Pace::Asap,
            lead_silence_ms: 0,
            tail_silence_ms: default_tail_silence_ms(),
        }
    }
}

/// `load_profile`: the bench contention profiles (without the stall).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadProfile {
    #[default]
    None,
    H1VrchatObs,
    H2Load70,
    H2Load70Dml,
    H2Load70Oop,
    H3CpuGame,
    Custom,
}

impl LoadProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            LoadProfile::None => "none",
            LoadProfile::H1VrchatObs => "h1_vrchat_obs",
            LoadProfile::H2Load70 => "h2_load70",
            LoadProfile::H2Load70Dml => "h2_load70_dml",
            LoadProfile::H2Load70Oop => "h2_load70_oop",
            LoadProfile::H3CpuGame => "h3_cpu_game",
            LoadProfile::Custom => "custom",
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoLoad {
    pub voice_changer_spec: String,
}

/// The dev rig's GPU timing lock (checked, never taken).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuLockDef {
    /// `None` = required for the speed modes, not otherwise.
    #[serde(default)]
    pub require: Option<bool>,
    #[serde(default = "default_lock_tag")]
    pub tag: String,
    #[serde(default)]
    pub path: Option<String>,
}

fn default_lock_tag() -> String {
    DEFAULT_LOCK_TAG.to_string()
}

impl Default for GpuLockDef {
    fn default() -> Self {
        GpuLockDef {
            require: None,
            tag: default_lock_tag(),
            path: None,
        }
    }
}

/// `{min, max}` of one aggregate key.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limit {
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Intelligibility {
    pub engine: String,
}

fn yes() -> bool {
    true
}

/// `tts_listening` and TTS speed runs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TtsOptions {
    /// JSON lines `{id, lang, text, kind?, voice?, reference_voice?}`.
    pub prompts: String,
    #[serde(default)]
    pub limit_per_language: Option<usize>,
    #[serde(default = "yes")]
    pub write_audio: bool,
    #[serde(default)]
    pub intelligibility: Option<Intelligibility>,
    #[serde(default)]
    pub dnsmos: bool,
    #[serde(default)]
    pub speaker_similarity: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MtContext {
    #[default]
    None,
    Previous,
}

/// `mt_quality` and MT speed runs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MtOptions {
    /// An mt-bench corpus (JSON lines `{id, source, target, src, ref | refs, ...}`).
    pub set: String,
    #[serde(default)]
    pub directions: Vec<String>,
    /// Lines per direction (file order).
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub context: MtContext,
}

/// One run.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunDef {
    pub name: String,
    pub mode: Mode,
    /// Engine ids; default: every engine whose task the mode takes.
    #[serde(default)]
    pub engines: Vec<String>,
    #[serde(default)]
    pub select: Option<Select>,
    #[serde(default)]
    pub feed: Feed,
    /// Untimed items per engine first.
    #[serde(default = "default_warmup")]
    pub warmup: u32,
    /// Timed passes over the selection (speed modes).
    #[serde(default = "default_repeat")]
    pub repeat: u32,
    #[serde(default)]
    pub load_profile: LoadProfile,
    #[serde(default)]
    pub load: Option<LoadSpec>,
    #[serde(default = "default_conv")]
    pub gpu_load_kind: GpuLoadKind,
    #[serde(default)]
    pub external_load: String,
    #[serde(default)]
    pub co_load: Option<CoLoad>,
    #[serde(default)]
    pub gpu_lock: Option<GpuLockDef>,
    /// Per item; an item over it is cancelled and reported as an error.
    #[serde(default = "default_timeout_s")]
    pub timeout_s: f64,
    #[serde(default)]
    pub tts: Option<TtsOptions>,
    #[serde(default)]
    pub mt: Option<MtOptions>,
    #[serde(default = "default_state_period_ms")]
    pub state_period_ms: u64,
    #[serde(default)]
    pub thresholds: BTreeMap<String, Limit>,
}

fn default_warmup() -> u32 {
    1
}
fn default_repeat() -> u32 {
    1
}
fn default_conv() -> GpuLoadKind {
    GpuLoadKind::Conv
}
fn default_timeout_s() -> f64 {
    120.0
}
fn default_state_period_ms() -> u64 {
    1000
}

impl RunDef {
    /// Whether the GPU lock must be held by the tag.
    pub fn lock_required(&self) -> bool {
        self.gpu_lock
            .as_ref()
            .and_then(|l| l.require)
            .unwrap_or(self.mode.is_speed())
    }

    pub fn lock_tag(&self) -> String {
        self.gpu_lock.as_ref().map_or_else(default_lock_tag, |l| l.tag.clone())
    }

    pub fn lock_path(&self) -> Option<String> {
        self.gpu_lock.as_ref().and_then(|l| l.path.clone())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExportDef {
    /// Keep hypotheses / rendered text of synthetic items (never of real clips) in the export.
    #[serde(default)]
    pub include_text: bool,
    /// Keep per-item timings (false: aggregates only).
    #[serde(default = "yes")]
    pub include_items: bool,
}

impl Default for ExportDef {
    fn default() -> Self {
        ExportDef {
            include_text: false,
            include_items: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputDef {
    #[serde(default = "yes")]
    pub write_hypotheses: bool,
    #[serde(default = "yes")]
    pub write_markdown: bool,
    #[serde(default)]
    pub export: ExportDef,
}

impl Default for OutputDef {
    fn default() -> Self {
        OutputDef {
            write_hypotheses: true,
            write_markdown: true,
            export: ExportDef::default(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Parsing and validation
// ---------------------------------------------------------------------------------------------

/// The backends of each task.
pub fn backends(task: Task) -> &'static [&'static str] {
    match task {
        Task::Stt => &["whisper", "http", "sherpa-online", "sherpa-offline", "transcribe", "pipeline", "crispasr", "hypotheses"],
        Task::Mt => &["nllb", "openai", "hypotheses", "bergamot", "llama-server"],
        Task::Tts => &["local-tts", "tts-helper", "openai-speech", "rendered"],
    }
}

static NAME_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$").expect("name pattern"));
static LANG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$").expect("lang pattern"));
static DIRECTION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[a-z]{2,3}(-[A-Z][a-z]{3})?(-[A-Z]{2})?-[a-z]{2,3}(-[A-Z][a-z]{3})?(-[A-Z]{2})?$").expect("direction pattern")
});
static SECRET_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)(api[_-]?key|token|secret|password)$").expect("secret pattern"));
/// A catalog id as the model store writes it (`stt/qwen3-asr-0.6b-int8`).
static CATALOG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[a-z0-9-]+/[A-Za-z0-9._-]+$").expect("catalog pattern"));

/// Whether `model` is a model-store catalog id (not a path).
pub fn is_catalog_id(model: &str) -> bool {
    CATALOG_RE.is_match(model)
}

/// Remove every `$comment` member (anywhere) from a copy of the spec.
pub fn strip_comments(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.remove("$comment");
            for v in map.values_mut() {
                strip_comments(v);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(strip_comments),
        _ => {}
    }
}

impl SpeechSpec {
    /// Parse and validate spec text; returns the typed spec and the spec as written (with its
    /// `$comment` members), which the report echoes and hashes.
    pub fn parse(text: &str) -> Result<(SpeechSpec, Value), String> {
        let raw: Value = serde_json::from_str(text).map_err(|e| format!("not valid JSON: {e}"))?;
        let mut strict = raw.clone();
        strip_comments(&mut strict);
        let spec: SpeechSpec = serde_json::from_value(strict).map_err(|e| format!("invalid speech-bench spec: {e}"))?;
        spec.validate()?;
        Ok((spec, raw))
    }

    /// Every problem of the spec, one string each (schema constraints serde cannot express, plus the
    /// cross-reference rules of `validate.py::spec_errors`).
    pub fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.schema != SPEC_SCHEMA {
            out.push(format!("schema must be \"{SPEC_SCHEMA}\", not \"{}\"", self.schema));
        }
        if !NAME_RE.is_match(&self.name) {
            out.push(format!("name {:?} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,63}}", self.name));
        }
        if self.engines.is_empty() {
            out.push("engines: at least one engine".into());
        }
        if self.runs.is_empty() {
            out.push("runs: at least one run".into());
        }
        let mut engines: BTreeMap<&str, &EngineDef> = BTreeMap::new();
        for e in &self.engines {
            if !NAME_RE.is_match(&e.id) {
                out.push(format!("engine id {:?} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,63}}", e.id));
            }
            if engines.insert(e.id.as_str(), e).is_some() {
                out.push(format!("engines: duplicate id {:?}", e.id));
            }
            if !backends(e.task).contains(&e.backend.as_str()) {
                out.push(format!(
                    "engine {}: backend {:?} is not a {} backend ({})",
                    e.id,
                    e.backend,
                    e.task.as_str(),
                    backends(e.task).join(", ")
                ));
            }
            if let Some(t) = e.threads {
                if !(1..=64).contains(&t) {
                    out.push(format!("engine {}: threads {t} must be 1-64", e.id));
                }
            }
            for l in &e.languages {
                if !LANG_RE.is_match(l) {
                    out.push(format!("engine {}: language {l:?} is not a lower-case BCP-47 base code", e.id));
                }
            }
            for d in &e.directions {
                if !DIRECTION_RE.is_match(d) {
                    out.push(format!("engine {}: direction {d:?} must look like en-ja", e.id));
                }
            }
            match &e.options {
                Value::Object(map) => {
                    for k in map.keys() {
                        if SECRET_RE.is_match(k) && k != "apiKeyEnv" {
                            out.push(format!(
                                "engine {}: options.{k} looks like a secret (name an environment variable with apiKeyEnv instead)",
                                e.id
                            ));
                        }
                    }
                }
                _ => out.push(format!("engine {}: options must be an object", e.id)),
            }
        }
        let mut names = BTreeSet::new();
        for r in &self.runs {
            if !NAME_RE.is_match(&r.name) {
                out.push(format!("run name {:?} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,63}}", r.name));
            }
            if !names.insert(r.name.as_str()) {
                out.push(format!("runs: duplicate name {:?}", r.name));
            }
            let tasks = r.mode.tasks();
            let chosen: Vec<&str> = if r.engines.is_empty() {
                self.engines.iter().filter(|e| tasks.contains(&e.task)).map(|e| e.id.as_str()).collect()
            } else {
                r.engines.iter().map(String::as_str).collect()
            };
            if chosen.is_empty() {
                out.push(format!("run {}: no engine of task {:?} in the spec", r.name, tasks.iter().map(|t| t.as_str()).collect::<Vec<_>>()));
            }
            for id in &chosen {
                match engines.get(id) {
                    None => out.push(format!("run {}: unknown engine {id:?}", r.name)),
                    Some(e) if !tasks.contains(&e.task) => out.push(format!(
                        "run {}: engine {id} is {}, mode {} takes {:?}",
                        r.name,
                        e.task.as_str(),
                        r.mode.as_str(),
                        tasks.iter().map(|t| t.as_str()).collect::<Vec<_>>()
                    )),
                    Some(_) => {}
                }
            }
            if matches!(r.mode, Mode::Accuracy | Mode::CpuSpeed | Mode::GpuUnderLoad)
                && chosen.iter().any(|id| engines.get(id).is_some_and(|e| e.task == Task::Stt))
                && self.corpus.is_none()
            {
                out.push(format!("run {}: STT runs need the spec's corpus", r.name));
            }
            if r.mode == Mode::MtQuality && r.mt.is_none() {
                out.push(format!("run {}: mt_quality needs the run's mt block", r.name));
            }
            if r.mode == Mode::TtsListening && r.tts.is_none() {
                out.push(format!("run {}: tts_listening needs the run's tts block", r.name));
            }
            if let Some(eng) = r.tts.as_ref().and_then(|t| t.intelligibility.as_ref()).map(|i| i.engine.as_str()) {
                if engines.get(eng).map(|e| e.task) != Some(Task::Stt) {
                    out.push(format!("run {}: tts.intelligibility.engine {eng:?} is not an stt engine of the spec", r.name));
                }
            }
            if !(r.timeout_s.is_finite() && r.timeout_s > 0.0) {
                out.push(format!("run {}: timeout_s must be positive", r.name));
            }
            if !(10..=2000).contains(&r.feed.chunk_ms) {
                out.push(format!("run {}: feed.chunk_ms must be 10-2000", r.name));
            }
            if r.repeat == 0 {
                out.push(format!("run {}: repeat must be at least 1", r.name));
            }
            for (metric, limit) in &r.thresholds {
                if limit.min.is_none() && limit.max.is_none() {
                    out.push(format!("run {}: threshold {metric} needs min or max", r.name));
                }
            }
            if let Some(sel) = &r.select {
                for l in &sel.languages {
                    if !LANG_RE.is_match(l) {
                        out.push(format!("run {}: select language {l:?} is not a lower-case BCP-47 base code", r.name));
                    }
                }
            }
            if let Some(mt) = &r.mt {
                for d in &mt.directions {
                    if !DIRECTION_RE.is_match(d) {
                        out.push(format!("run {}: mt direction {d:?} must look like en-ja", r.name));
                    }
                }
            }
        }
        out
    }

    fn validate(&self) -> Result<(), String> {
        let problems = self.problems();
        if problems.is_empty() {
            Ok(())
        } else {
            Err(format!("invalid speech-bench spec:\n  {}", problems.join("\n  ")))
        }
    }

    /// The engine `id`.
    pub fn engine(&self, id: &str) -> Option<&EngineDef> {
        self.engines.iter().find(|e| e.id == id)
    }

    /// The engines `run` uses, in the run's order (default: every engine of a task the mode takes).
    pub fn run_engines<'a>(&'a self, run: &RunDef) -> Vec<&'a EngineDef> {
        if run.engines.is_empty() {
            let tasks = run.mode.tasks();
            self.engines.iter().filter(|e| tasks.contains(&e.task)).collect()
        } else {
            run.engines.iter().filter_map(|id| self.engine(id)).collect()
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------------------------

/// `${NAME}` expansion over `lookup`, with the P7 defaults for `SIGIL_P7_ROOT` and `SIGIL_P7_CORPUS`
/// (`%LOCALAPPDATA%/sigil-dev-cache/p7`, `${SIGIL_P7_ROOT}/corpus`). An unknown variable is an error.
pub fn expand_vars(text: &str, lookup: &dyn Fn(&str) -> Option<String>) -> Result<String, String> {
    expand_depth(text, lookup, 0)
}

fn expand_depth(text: &str, lookup: &dyn Fn(&str) -> Option<String>, depth: u32) -> Result<String, String> {
    if depth > 4 {
        return Err(format!("variables nest too deeply in {text:?}"));
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let end = after.find('}').ok_or_else(|| format!("unterminated ${{ in {text:?}"))?;
        let name = &after[..end];
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(format!("bad variable name ${{{name}}} in {text:?}"));
        }
        let value = lookup(name)
            .filter(|v| !v.is_empty())
            .or_else(|| default_var(name, lookup))
            .ok_or_else(|| format!("the variable ${{{name}}} is not set (in {text:?})"))?;
        out.push_str(&expand_depth(&value, lookup, depth + 1)?);
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

fn default_var(name: &str, lookup: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    match name {
        "SIGIL_P7_ROOT" => lookup("LOCALAPPDATA")
            .filter(|v| !v.is_empty())
            .map(|d| format!("{}/sigil-dev-cache/p7", d.replace('\\', "/").trim_end_matches('/'))),
        "SIGIL_P7_CORPUS" => Some("${SIGIL_P7_ROOT}/corpus".to_string()),
        _ => None,
    }
}

/// Path resolution of a spec: `${NAME}` expansion over the environment, then relative to the spec
/// file's folder.
#[derive(Clone, Debug)]
pub struct Paths {
    base: PathBuf,
}

impl Paths {
    pub fn new(spec_file: &Path) -> Paths {
        let base = spec_file.parent().map(Path::to_path_buf).unwrap_or_default();
        Paths { base }
    }

    #[cfg(test)]
    pub fn with_base(base: impl Into<PathBuf>) -> Paths {
        Paths { base: base.into() }
    }

    /// The expanded text of a path field.
    pub fn expand(&self, text: &str) -> Result<String, String> {
        expand_vars(text, &|name| std::env::var(name).ok())
    }

    /// An expanded path field, relative to the spec's folder when not absolute.
    pub fn resolve(&self, text: &str) -> Result<PathBuf, String> {
        let expanded = self.expand(text)?;
        let p = PathBuf::from(&expanded);
        Ok(if p.is_absolute() { p } else { self.base.join(p) })
    }
}

// ---------------------------------------------------------------------------------------------
// Canonical JSON and the spec hash
// ---------------------------------------------------------------------------------------------

/// The canonical JSON of `value`: object keys sorted, no whitespace, non-ASCII kept (UTF-8), the same
/// text as Python's `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
pub fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).unwrap_or_else(|_| "\"\"".into()));
                out.push(':');
                write_canonical(&map[k.as_str()], out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(v, out);
            }
            out.push(']');
        }
        other => out.push_str(&serde_json::to_string(other).unwrap_or_else(|_| "null".into())),
    }
}

/// sha256 (lower-case hex) of the canonical JSON of `value`.
pub fn canonical_sha256(value: &Value) -> String {
    hex_digest(canonical_json(value).as_bytes())
}

/// The report's `spec_sha256`.
pub fn spec_sha256(spec_as_written: &Value) -> String {
    canonical_sha256(spec_as_written)
}

/// Lower-case hex sha256 of `bytes`.
pub fn hex_digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
