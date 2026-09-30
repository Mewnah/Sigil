//! sherpa-onnx utterance engines (`OfflineRecognizer`: Parakeet Ultra / Parakeet TDT v3,
//! Qwen3-ASR-0.6B, SenseVoice, Moonshine; `SpokenLanguageIdentification`: Whisper base language
//! ID), the sherpa runtime preload, and the helpers both sherpa engine modules share.
//!
//! Owned by P2-B.
//!
//! # Frozen signatures (P2-S0)
//! - `create(&EngineSpec, &EngineContext) -> Result<Box<dyn AsrEngine>, AsrError>` (called by
//!   `engines::create`; blocking).
//! - `setup(&AppHandle)`: called once from `engines::setup` at plugin setup.
//!
//! # Models
//! [`EngineSpec::model`] is a catalog id from `model-catalog/stt-sherpa.json`
//! (`speech_rt::sherpa::catalog`), located only with `model_store::installed_path`; a missing
//! model or runtime pack is [`AsrErrorKind::NotInstalled`] naming the id (engines never download).
//! An absolute folder path is a bring-your-own sherpa package: detected, then loaded once in an
//! isolated child (`probe::validate_in_child`, with the exact final config) before it is loaded
//! here, because sherpa-onnx aborts the process on some malformed models.
//!
//! # What each family does with [`DecodeOptions`]
//! | Family | `language` | `hotwords` / `prompt` | result |
//! |---|---|---|---|
//! | Parakeet (NeMo TDT) | ignored (the model detects among its 25 languages); reported back when it is one of them | modified beam search with a context graph (a `bpe.vocab` is derived from `tokens.txt`); catalog Parakeet models decode with beam search from the start, so the first custom terms need no reload | text |
//! | Qwen3-ASR | forced through the per-stream option as the English name (`Japanese`); without one the model transcribes in the language it hears | per-stream prompt context (terms, then the prompt) | text; `lang` only when forced (sherpa-onnx 1.13.8 does not return the detected language, so auto mode runs the language-ID engine first) |
//! | SenseVoice | the per-call option is ignored (sherpa ignores it too); `options.language` at creation pins the model config's language, else auto | none | text, detected `lang` + `lang:` tag, `emotion:` / `event:` tags |
//! | Moonshine | English only | none | text, `lang` = `en` |
//! | Language ID | — | — | empty text, `lang` + `lang:<code>` tag |
//!
//! # Engine options ([`EngineSpec::options`], camelCase, all optional; never logged)
//! `language` (default language / SenseVoice's pinned language), `hotwords` (string array used when
//! a call brings none), `hotwordsScore`, `prompt` (Qwen3-ASR default context), `decoding`
//! (`"greedy"` | `"beam"`; transducers: see `starts_on_beam_search`), `maxActivePaths`,
//! `maxNewTokens` (Qwen3-ASR; `max_total_len` grows with it), `useItn` (SenseVoice),
//! `tailPaddingMs` (streaming), and for bring-your-own folders the detection hints `kind`, `mode`,
//! `fp32`.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use super::super::engine_api::{
    AsrEngine, AsrError, AsrErrorKind, AsrResult, Canceller, DecodeOptions, EngineCaps, EngineContext, EngineInfo, EngineKind, EngineSpec,
    EngineStatus, StatusLevel, SAMPLE_RATE,
};
use super::super::events::codes;
use super::super::lang;
use crate::services::speech_rt::sherpa::catalog::{self, qwen3_max_total_len, CatalogModel, Family, HotwordSupport, Task, QWEN3_MAX_NEW_TOKENS};
use crate::services::speech_rt::sherpa::detect::{detect, DetectHints, SherpaModel};
use crate::services::speech_rt::sherpa::hotwords::{self, default_cache_dir, ensure_bpe_vocab};
use crate::services::speech_rt::sherpa::languages::{qwen3_language_code, qwen3_language_name, SENSE_VOICE_LANGUAGES};
use crate::services::speech_rt::sherpa::lib::{runtime_dir, SherpaRuntime, EXPECTED_VERSION, MAX_THREADS, RUNTIME_ID};
use crate::services::speech_rt::sherpa::lid::SpokenLanguageId;
use crate::services::speech_rt::sherpa::offline::{
    self as sherpa_offline, HotwordsMode, OfflineModel, OfflineRecognizer, OfflineRecognizerConfig, OfflineResult,
};
use crate::services::speech_rt::sherpa::online::Decoding;
use crate::services::speech_rt::sherpa::probe::validate_in_child;

/// Environment kill switch for the runtime preload at plugin setup (`0` / `false` / `off`).
pub const PRELOAD_ENV: &str = "SIGIL_SHERPA_PRELOAD";

/// Status codes the sherpa engines report through [`EngineContext::status`] (besides
/// `events::codes`); the TS driver shows `detail` for codes it does not localize.
pub mod status_codes {
    /// A streaming model that needs the spoken language got none (or one it does not know) and
    /// runs on its automatic language detection, which is less accurate.
    pub use crate::services::speech::events::codes::SHERPA_LANGUAGE_AUTO as LANGUAGE_AUTO;
    /// Hotwords could not be enabled for this model; recognition continues without them.
    pub use crate::services::speech::events::codes::SHERPA_HOTWORDS_UNAVAILABLE as HOTWORDS_UNAVAILABLE;
}

// ---------------------------------------------------------------------------------------------
// Plugin setup: load the runtime early
// ---------------------------------------------------------------------------------------------

/// Plugin setup hook (returns at once). When the sherpa runtime pack is installed, a background
/// thread loads it now: `SherpaRuntime::load` refuses to run once another `onnxruntime.dll` is in
/// the process (Windows would bind sherpa to it), so loading first is what keeps it usable. The
/// pack is installed only when the user chose a sherpa engine (or local TTS), so an installed pack
/// is the signal that one is configured. Costs ~3 ms and ~4 MB; no model is loaded.
pub fn setup<R: tauri::Runtime>(app: &tauri::AppHandle<R>) {
    let _ = app;
    if preload_disabled(std::env::var(PRELOAD_ENV).ok().as_deref()) {
        tracing::info!(target: "sigil::sherpa", "sherpa-onnx preload disabled by {PRELOAD_ENV}");
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("sigil-sherpa-preload".into())
        .spawn(|| {
            if SherpaRuntime::loaded().is_some() || runtime_dir().is_none() {
                return;
            }
            match SherpaRuntime::installed() {
                Ok(rt) => tracing::info!(
                    target: "sigil::sherpa",
                    "sherpa-onnx {} preloaded for the speech engines",
                    rt.info().version
                ),
                Err(e) => tracing::warn!(target: "sigil::sherpa", "sherpa-onnx preload failed: {e}"),
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(target: "sigil::sherpa", "cannot start the sherpa-onnx preload thread: {e}");
    }
}

fn preload_disabled(v: Option<&str>) -> bool {
    matches!(v.map(|s| s.trim().to_ascii_lowercase()).as_deref(), Some("0" | "false" | "off" | "no"))
}

// ---------------------------------------------------------------------------------------------
// Shared helpers (also used by `sherpa_online`)
// ---------------------------------------------------------------------------------------------

/// `EngineInfo::runtime` of every sherpa engine.
pub(super) fn runtime_label() -> String {
    format!("sherpa-onnx {EXPECTED_VERSION}")
}

/// The loaded sherpa runtime, loading the installed pack on first use. A missing pack is
/// `NotInstalled` naming [`RUNTIME_ID`].
pub(super) fn runtime(ctx: &EngineContext) -> Result<Arc<SherpaRuntime>, AsrError> {
    if let Some(rt) = SherpaRuntime::loaded() {
        return Ok(rt);
    }
    if runtime_dir().is_none() {
        return Err(AsrError::not_installed(format!(
            "{RUNTIME_ID}: the sherpa-onnx speech runtime is not installed"
        )));
    }
    report(
        ctx,
        codes::LOADING_RUNTIME,
        "Loading the sherpa-onnx speech runtime",
        Some(RUNTIME_ID),
        StatusLevel::Info,
    );
    SherpaRuntime::installed().map_err(|e| AsrError::runtime(format!("{RUNTIME_ID}: {e}")))
}

/// Report a status line (never with secrets).
pub(super) fn report(ctx: &EngineContext, code: &str, detail: impl Into<String>, model_id: Option<&str>, level: StatusLevel) {
    ctx.status.report(EngineStatus {
        code: code.to_string(),
        detail: detail.into(),
        progress: None,
        level,
        model_id: model_id.map(str::to_string),
    });
}

/// Engine options (see the module docs). Parsed leniently: a value of the wrong type is ignored.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Options {
    pub language: Option<String>,
    pub hotwords: Vec<String>,
    pub hotwords_score: Option<f32>,
    pub prompt: Option<String>,
    /// `decoding: "beam"`.
    pub beam_search: bool,
    /// `decoding: "greedy"`: a transducer keeps greedy search (and so takes no hotwords).
    pub greedy: bool,
    pub max_active_paths: Option<u32>,
    pub max_new_tokens: Option<i32>,
    pub use_itn: Option<bool>,
    pub tail_padding_ms: Option<u32>,
    pub kind: Option<String>,
    pub mode: Option<String>,
    pub fp32: bool,
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
        let hotwords = match v.get("hotwords") {
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect(),
            Some(Value::String(s)) => s.lines().map(str::to_string).collect(),
            _ => Vec::new(),
        };
        Options {
            language: s("language").and_then(|l| lang::normalize(&l)),
            hotwords,
            hotwords_score: n("hotwordsScore").map(|x| x.clamp(0.0, 10.0) as f32),
            prompt: s("prompt"),
            beam_search: matches!(s("decoding").as_deref(), Some("beam" | "modified_beam_search" | "beam_search")),
            greedy: matches!(s("decoding").as_deref(), Some("greedy" | "greedy_search")),
            max_active_paths: n("maxActivePaths").map(|x| x.clamp(1.0, 64.0) as u32),
            max_new_tokens: n("maxNewTokens").map(|x| x.clamp(16.0, 1024.0) as i32),
            use_itn: v.get("useItn").and_then(Value::as_bool),
            tail_padding_ms: n("tailPaddingMs").map(|x| x.clamp(0.0, 3000.0) as u32),
            kind: s("kind"),
            mode: s("mode"),
            fp32: v.get("fp32").and_then(Value::as_bool).unwrap_or(false),
        }
    }
}

/// Threads for this engine: the spec's, else the family default; clamped to `1..=MAX_THREADS`.
pub(super) fn engine_threads(spec: &EngineSpec, default: u32) -> u32 {
    spec.threads.unwrap_or(default).clamp(1, MAX_THREADS)
}

/// A model resolved from an [`EngineSpec`].
pub(super) struct Resolved {
    /// Catalog id, or the folder name of a bring-your-own model (`EngineInfo::model`).
    pub label: String,
    pub catalog: Option<&'static CatalogModel>,
    pub config: SherpaModel,
    /// A bring-your-own folder: must pass [`validate_byo`] before it is loaded here.
    pub byo: bool,
}

/// Resolve `spec.model` (catalog id or absolute folder) into a config for `want`.
pub(super) fn resolve(spec: &EngineSpec, want: Task, opts: &Options) -> Result<Resolved, AsrError> {
    let model = spec.model.trim();
    if model.is_empty() {
        return Err(AsrError::invalid_input("no sherpa model given"));
    }
    let resolved = if let Some(cm) = catalog::find(model) {
        let dir = crate::services::model_store::installed_path(cm.id)
            .ok_or_else(|| AsrError::not_installed(format!("{}: the model is not installed", cm.id)))?;
        Resolved {
            label: cm.id.to_string(),
            catalog: Some(cm),
            config: cm.config(&dir),
            byo: false,
        }
    } else {
        let path = Path::new(model);
        if !path.is_absolute() {
            return Err(AsrError::invalid_input(format!(
                "{model} is not a sherpa-onnx catalog model (and not a model folder)"
            )));
        }
        if !path.exists() {
            return Err(AsrError::invalid_input(format!("the model folder {} does not exist", path.display())));
        }
        // The folder's own name decides streaming vs utterance (an explicit `mode` option wins);
        // a mismatch with the engine is refused below rather than loading a streaming encoder into
        // an utterance recognizer.
        let hints = DetectHints {
            kind: opts.kind.clone(),
            mode: opts.mode.clone(),
            prefer_fp32: opts.fp32,
            vocoder: None,
        };
        let hints = if want == Task::LanguageId && hints.kind.is_none() {
            DetectHints {
                kind: Some("lid".into()),
                ..hints
            }
        } else {
            hints
        };
        let config = detect(path, &hints).map_err(AsrError::invalid_input)?;
        Resolved {
            label: path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| model.to_string()),
            catalog: None,
            config,
            byo: true,
        }
    };
    let task = match &resolved.config {
        SherpaModel::Online(_) => Task::Online,
        SherpaModel::Offline(_) => Task::Offline,
        SherpaModel::LanguageId(_) => Task::LanguageId,
        other => {
            return Err(AsrError::unsupported(format!(
                "{} is a {} model, not speech recognition",
                resolved.label,
                other.label()
            )));
        }
    };
    let offline_side = |t: Task| matches!(t, Task::Offline | Task::LanguageId);
    let fits = task == want || (offline_side(task) && offline_side(want));
    if !fits {
        let (is, needs) = match task {
            Task::Online => ("a streaming", "sherpa-online"),
            _ => ("an utterance", "sherpa-offline"),
        };
        return Err(AsrError::unsupported(format!(
            "{} is {is} model; run it with the {needs} engine",
            resolved.label
        )));
    }
    Ok(resolved)
}

/// Configs (by content) that already passed the isolated load check in this process.
fn validated() -> &'static Mutex<BTreeSet<String>> {
    static SET: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
    &SET
}

/// A fingerprint of `config` and the size / modification time of every file it names, so an
/// edited model folder is checked again.
fn fingerprint(config: &SherpaModel) -> String {
    let json = serde_json::to_string(config).unwrap_or_default();
    let mut fp = json.clone();
    // Every string in the config that names an existing file contributes its size and mtime.
    fn walk(v: &Value, out: &mut String) {
        match v {
            Value::String(s) => {
                if let Ok(md) = std::fs::metadata(s) {
                    let mtime = md
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_nanos())
                        .unwrap_or(0);
                    out.push_str(&format!("|{}:{}", md.len(), mtime));
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            Value::Object(o) => o.values().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    if let Ok(v) = serde_json::from_str::<Value>(&json) {
        walk(&v, &mut fp);
    }
    fp
}

/// Load a bring-your-own `config` once in an isolated child first (sherpa-onnx can abort the
/// process on malformed models); cached per config and file state for the process.
pub(super) fn validate_byo(config: &SherpaModel, label: &str, ctx: &EngineContext) -> Result<(), AsrError> {
    config.validate().map_err(AsrError::invalid_input)?;
    let fp = fingerprint(config);
    if validated()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .contains(&fp)
    {
        return Ok(());
    }
    report(
        ctx,
        codes::LOADING_MODEL,
        format!("Checking the model {label} in isolation"),
        None,
        StatusLevel::Info,
    );
    validate_in_child(config, Duration::from_secs(180))
        .map_err(|e| AsrError::invalid_input(format!("{label} failed the isolated load check: {e}")))?;
    validated()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(fp);
    Ok(())
}

/// The catalog model's language when it has exactly one.
pub(super) fn only_language(catalog: Option<&CatalogModel>) -> Option<String> {
    match catalog.map(|c| c.languages) {
        Some([one]) => Some(one.to_string()),
        _ => None,
    }
}

/// Normalized language of a DecodeOptions / option value.
pub(super) fn requested_language(options: &DecodeOptions, default: &Option<String>) -> Option<String> {
    options
        .language
        .as_deref()
        .and_then(lang::normalize)
        .or_else(|| default.clone())
}

fn elapsed_ms(t0: Instant) -> u32 {
    t0.elapsed().as_millis().min(u128::from(u32::MAX)) as u32
}

// ---------------------------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------------------------

/// Create the engine (blocking: loads the model). A language-ID model gives a language-ID engine
/// (empty text, `lang` + `lang:<code>` tag); every other model an utterance recognizer.
pub fn create(spec: &EngineSpec, ctx: &EngineContext) -> Result<Box<dyn AsrEngine>, AsrError> {
    let opts = Options::parse(&spec.options);
    let resolved = resolve(spec, Task::Offline, &opts)?;
    let rt = runtime(ctx)?;
    match resolved.config {
        SherpaModel::LanguageId(ref c) => {
            let mut cfg = c.clone();
            cfg.num_threads = Some(engine_threads(spec, catalog_threads(resolved.catalog, 2)));
            let config = SherpaModel::LanguageId(cfg.clone());
            if resolved.byo {
                validate_byo(&config, &resolved.label, ctx)?;
            }
            report(
                ctx,
                codes::LOADING_MODEL,
                format!("Loading {}", resolved.label),
                model_id(&resolved),
                StatusLevel::Info,
            );
            let lid = SpokenLanguageId::new(&rt, &cfg).map_err(|e| AsrError::runtime(format!("{}: {e}", resolved.label)))?;
            Ok(Box::new(SherpaLidEngine {
                info: EngineInfo {
                    kind: EngineKind::SherpaOffline,
                    model: resolved.label,
                    runtime: runtime_label(),
                    device: "cpu".into(),
                    caps: EngineCaps {
                        utterance: true,
                        detects_language: true,
                        tags: true,
                        ..EngineCaps::default()
                    },
                    vram_mb: 0,
                    fallback_reason: None,
                },
                lid,
                cancel: Arc::new(AtomicBool::new(false)),
            }))
        }
        SherpaModel::Offline(ref c) => {
            let engine = SherpaOfflineEngine::new(spec, ctx, rt, &resolved, c.clone(), opts)?;
            Ok(Box::new(engine))
        }
        _ => Err(AsrError::unsupported(format!("{} is not an utterance model", resolved.label))),
    }
}

fn model_id(r: &Resolved) -> Option<&str> {
    r.catalog.map(|c| c.id)
}

fn catalog_threads(catalog: Option<&CatalogModel>, fallback: u32) -> u32 {
    catalog
        .map(|c| c.family.default_threads())
        .unwrap_or(fallback)
}

// ---------------------------------------------------------------------------------------------
// Utterance engine
// ---------------------------------------------------------------------------------------------

/// Whether an offline transducer starts on modified beam search (the only decoding that takes
/// hotwords): when asked (`decoding: "beam"`, or hotwords in the engine options), and always for a
/// catalog model with transducer hotwords (Parakeet). Otherwise the custom terms the pipeline sends
/// with each utterance would switch the recognizer over on the first utterance of every session,
/// which reloads the ~600 MB model on the final-decode thread (1.2-1.4 s). Beam search costs next
/// to nothing on Parakeet (measured RTF 0.068 vs 0.065 greedy, ADR 0002). `decoding: "greedy"`
/// keeps greedy search. Other (bring-your-own) transducers switch when hotwords first arrive.
fn starts_on_beam_search(catalog: Option<&CatalogModel>, opts: &Options) -> bool {
    !opts.greedy
        && (opts.beam_search
            || !opts.hotwords.is_empty()
            || catalog.is_some_and(|c| c.family.hotwords() == HotwordSupport::TransducerBeamSearch))
}

#[cfg(test)]
thread_local! {
    /// Test hook: recognizer loads on this thread fail (simulates a reload that runs out of memory).
    static FAIL_LOADS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Test hook: recognizer loads attempted on this thread.
    static LOAD_ATTEMPTS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Which offline family a config is (catalog or bring-your-own).
fn offline_family(catalog: Option<&CatalogModel>, config: &OfflineRecognizerConfig) -> Option<Family> {
    if let Some(c) = catalog {
        return Some(c.family);
    }
    match (&config.model, config.model_type.as_deref()) {
        (OfflineModel::SenseVoice { .. }, _) => Some(Family::SenseVoice),
        (OfflineModel::Qwen3Asr { .. }, _) => Some(Family::Qwen3Asr),
        (OfflineModel::Moonshine { .. }, _) => Some(Family::Moonshine),
        (OfflineModel::Transducer { .. }, Some("nemo_transducer")) => Some(Family::ParakeetTdt),
        _ => None,
    }
}

/// The sherpa-onnx `OfflineRecognizer` engine.
pub struct SherpaOfflineEngine {
    info: EngineInfo,
    label: String,
    model_id: Option<&'static str>,
    rt: Arc<SherpaRuntime>,
    catalog: Option<&'static CatalogModel>,
    family: Option<Family>,
    config: OfflineRecognizerConfig,
    recognizer: Option<OfflineRecognizer>,
    opts: Options,
    /// Hotwords cannot be enabled for this model (reported once; never retried).
    hotwords_failed: bool,
    /// A Qwen3-ASR result hit `max_new_tokens` (logged once).
    truncation_logged: bool,
    cancel: Arc<AtomicBool>,
    ctx: EngineContext,
}

impl SherpaOfflineEngine {
    fn new(
        spec: &EngineSpec,
        ctx: &EngineContext,
        rt: Arc<SherpaRuntime>,
        resolved: &Resolved,
        mut config: OfflineRecognizerConfig,
        opts: Options,
    ) -> Result<Self, AsrError> {
        let family = offline_family(resolved.catalog, &config);
        config.num_threads = Some(engine_threads(spec, catalog_threads(resolved.catalog, 2)));
        match &mut config.model {
            OfflineModel::SenseVoice { language, use_itn, .. } => {
                // SenseVoice reads its language from the model config only (per-stream options are
                // ignored), so a pinned language is set at creation.
                *language = opts
                    .language
                    .clone()
                    .filter(|l| SENSE_VOICE_LANGUAGES.contains(&l.as_str()));
                if let Some(itn) = opts.use_itn {
                    *use_itn = itn;
                }
            }
            OfflineModel::Qwen3Asr {
                max_new_tokens, max_total_len, ..
            } => {
                // Catalog and bring-your-own configs alike: a cap sized for the longest utterance
                // and a KV context that fits it (sherpa's defaults, 128 / 512, cut long captions
                // and drop the audio of long utterances).
                let new_tokens = opts
                    .max_new_tokens
                    .or(*max_new_tokens)
                    .filter(|n| *n > 0)
                    .unwrap_or(QWEN3_MAX_NEW_TOKENS);
                *max_new_tokens = Some(new_tokens);
                *max_total_len = Some(max_total_len.unwrap_or(0).max(qwen3_max_total_len(new_tokens)));
            }
            _ => {}
        }
        if let Some(score) = opts.hotwords_score {
            config.hotwords_score = score;
        }
        let transducer = matches!(config.model, OfflineModel::Transducer { .. });
        let caps = EngineCaps {
            streaming: false,
            utterance: true,
            languages: resolved
                .catalog
                .map(|c| c.languages.iter().map(|l| l.to_string()).collect())
                .unwrap_or_default(),
            needs_language: false,
            detects_language: family.is_some_and(Family::detects_language),
            hotwords: transducer || family.is_some_and(|f| f.hotwords() != HotwordSupport::None),
            prompt: family.is_some_and(Family::prompt),
            tags: family.is_some_and(Family::tags),
        };
        let mut engine = SherpaOfflineEngine {
            info: EngineInfo {
                kind: EngineKind::SherpaOffline,
                model: resolved.label.clone(),
                runtime: runtime_label(),
                device: "cpu".into(),
                caps,
                vram_mb: 0,
                fallback_reason: None,
            },
            label: resolved.label.clone(),
            model_id: resolved.catalog.map(|c| c.id),
            rt,
            catalog: resolved.catalog,
            family,
            config,
            recognizer: None,
            opts,
            hotwords_failed: false,
            truncation_logged: false,
            cancel: Arc::new(AtomicBool::new(false)),
            ctx: ctx.clone(),
        };
        let greedy = engine.config.clone();
        if transducer && starts_on_beam_search(resolved.catalog, &engine.opts) {
            if let Err(e) = engine.enable_beam_search() {
                engine.hotwords_unavailable(&e);
            }
        }
        if resolved.byo {
            validate_byo(&SherpaModel::Offline(engine.config.clone()), &engine.label, ctx)?;
        }
        if let Err(e) = engine.load() {
            if !engine.config.decoding.is_beam_search() || greedy.decoding.is_beam_search() || resolved.byo {
                return Err(e);
            }
            // The beam-search recognizer did not load: greedy search, without custom terms.
            engine.config = greedy;
            engine.hotwords_unavailable(&e.message);
            engine.load()?;
        }
        Ok(engine)
    }

    /// (Re)create the recognizer from `self.config`.
    fn load(&mut self) -> Result<(), AsrError> {
        report(
            &self.ctx,
            codes::LOADING_MODEL,
            format!("Loading {}", self.label),
            self.model_id,
            StatusLevel::Info,
        );
        // Free the old model first: two Parakeet encoders at once would double the memory peak.
        self.recognizer = None;
        #[cfg(test)]
        {
            LOAD_ATTEMPTS.with(|n| n.set(n.get() + 1));
            if FAIL_LOADS.with(std::cell::Cell::get) {
                return Err(AsrError::runtime(format!("{}: simulated load failure", self.label)));
            }
        }
        let rec = OfflineRecognizer::new(&self.rt, &self.config).map_err(|e| AsrError::runtime(format!("{}: {e}", self.label)))?;
        self.recognizer = Some(rec);
        Ok(())
    }

    /// Switch a transducer config to modified beam search with a `bpe.vocab` (needed to encode
    /// hotwords for SentencePiece models). Does not reload.
    fn enable_beam_search(&mut self) -> Result<(), String> {
        if self.config.decoding.is_beam_search() {
            return Ok(());
        }
        let tokens = self
            .config
            .tokens
            .clone()
            .ok_or("the model has no tokens.txt")?;
        let sentencepiece = self.family == Some(Family::ParakeetTdt)
            || self.config.model_type.as_deref() == Some("nemo_transducer")
            || self
                .config
                .modeling_unit
                .as_deref()
                .is_some_and(|u| u.contains("bpe"));
        let mut next = self.config.clone();
        if sentencepiece {
            let vocab = match self.config.bpe_vocab.clone() {
                Some(v) => v,
                None => ensure_bpe_vocab(&tokens, &default_cache_dir())?,
            };
            next.bpe_vocab = Some(vocab);
            next.modeling_unit = Some("bpe".into());
        }
        next.decoding = Decoding::ModifiedBeamSearch {
            max_active_paths: self.opts.max_active_paths.unwrap_or(4),
        };
        next.validate()?;
        self.config = next;
        Ok(())
    }

    fn hotwords_unavailable(&mut self, why: &str) {
        if !self.hotwords_failed {
            self.hotwords_failed = true;
            tracing::warn!(target: "sigil::sherpa", "{}: hotwords unavailable: {why}", self.label);
            report(
                &self.ctx,
                status_codes::HOTWORDS_UNAVAILABLE,
                format!("Custom terms are not available for {}: {why}", self.label),
                self.model_id,
                StatusLevel::Warning,
            );
        }
    }

    /// Switch a greedy transducer to modified beam search for hotwords (reloads the model). Tried
    /// once per engine: any failure keeps (or restores) greedy search and turns hotwords off for the
    /// rest of the session, so later utterances never pay for another attempt. `Err` only when
    /// the greedy recognizer cannot be loaded again (the next utterance retries that load).
    fn upgrade_to_beam_search(&mut self) -> Result<(), AsrError> {
        let greedy = self.config.clone();
        if let Err(e) = self.enable_beam_search() {
            self.hotwords_unavailable(&e);
            return Ok(());
        }
        let Err(e) = self.load() else { return Ok(()) };
        self.config = greedy;
        self.hotwords_unavailable(&e.message);
        self.load()
    }

    /// The per-call sherpa options for `options` (switching a greedy transducer to beam search the
    /// first time hotwords arrive, see [`Self::upgrade_to_beam_search`]).
    fn call_options(&mut self, options: &DecodeOptions) -> Result<sherpa_offline::DecodeOptions, AsrError> {
        let phrases: Vec<String> = if options.hotwords.is_empty() {
            self.opts.hotwords.clone()
        } else {
            options.hotwords.clone()
        };
        let mut out = sherpa_offline::DecodeOptions::default();
        let transducer = matches!(self.config.model, OfflineModel::Transducer { .. });
        if transducer && !self.hotwords_failed {
            if let Some(text) = hotwords::transducer_hotwords(&phrases) {
                if !self.config.decoding.is_beam_search() {
                    if self.opts.greedy {
                        self.hotwords_unavailable("the engine is set to greedy search");
                    } else {
                        self.upgrade_to_beam_search()?;
                    }
                }
                if self.config.hotwords_mode() == HotwordsMode::ContextGraph {
                    out.hotwords = Some(text);
                }
            }
        }
        if self.config.hotwords_mode() == HotwordsMode::PromptContext {
            let prompt = options.prompt.as_deref().or(self.opts.prompt.as_deref());
            out.hotwords = hotwords::qwen3_context(&phrases, prompt);
        }
        if let OfflineModel::Qwen3Asr { .. } = self.config.model {
            out.language = requested_language(options, &self.opts.language)
                .and_then(|l| qwen3_language_name(&l))
                .map(str::to_string);
        }
        Ok(out)
    }

    /// Map a sherpa result to the engine result (language, tags).
    fn result(&self, r: OfflineResult, options: &DecodeOptions, forced: Option<&str>, decode_ms: u32) -> AsrResult {
        let mut tags = Vec::new();
        let family = self.family;
        let detected = r.lang.as_deref().and_then(|l| {
            if family == Some(Family::Qwen3Asr) {
                qwen3_language_code(l).map(str::to_string)
            } else {
                lang::normalize(l)
            }
        });
        let lang = match family {
            Some(Family::SenseVoice) => detected.clone(),
            Some(Family::Qwen3Asr) => forced.map(str::to_string).or_else(|| detected.clone()),
            _ => only_language(self.catalog)
                .or_else(|| requested_language(options, &self.opts.language).filter(|l| self.catalog.is_some_and(|c| c.supports_language(l)))),
        };
        if let Some(d) = &detected {
            if forced.is_none() {
                tags.push(format!("lang:{d}"));
            }
        }
        if let Some(e) = r.emotion.as_deref().and_then(emotion_tag) {
            tags.push(format!("emotion:{e}"));
        }
        if let Some(e) = r.event.as_deref().and_then(event_tag) {
            tags.push(format!("event:{e}"));
        }
        AsrResult {
            text: r.text,
            lang,
            tags,
            no_speech_prob: None,
            confidence: None,
            decode_ms,
        }
    }
}

/// SenseVoice emotion → contract value (`engine_api`); unknown → none.
pub(crate) fn emotion_tag(raw: &str) -> Option<&'static str> {
    Some(match raw.trim().to_ascii_lowercase().as_str() {
        "happy" => "happy",
        "sad" => "sad",
        "angry" => "angry",
        "neutral" => "neutral",
        "fearful" | "fear" => "fearful",
        "disgusted" | "disgust" => "disgusted",
        "surprised" | "surprise" => "surprised",
        _ => return None,
    })
}

/// SenseVoice audio event → contract value; plain speech and unknown events give no tag.
pub(crate) fn event_tag(raw: &str) -> Option<&'static str> {
    Some(match raw.trim().to_ascii_lowercase().as_str() {
        "bgm" | "music" => "bgm",
        "applause" => "applause",
        "laughter" | "laugh" => "laughter",
        "cry" | "crying" => "cry",
        "sneeze" => "sneeze",
        "breath" | "breathing" => "breath",
        "cough" => "cough",
        _ => return None,
    })
}

impl AsrEngine for SherpaOfflineEngine {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn transcribe(&mut self, audio: &[f32], options: &DecodeOptions) -> Result<AsrResult, AsrError> {
        self.cancel.store(false, Ordering::SeqCst);
        let t0 = Instant::now();
        if !sherpa_offline::long_enough(audio.len(), SAMPLE_RATE) {
            return Ok(AsrResult::default());
        }
        let call = self.call_options(options)?;
        if self.recognizer.is_none() {
            // A reload failed on an earlier utterance: one attempt per utterance (the pipeline
            // ends the session after five failures in a row).
            self.load()?;
        }
        if self.cancel.load(Ordering::SeqCst) {
            return Err(AsrError::cancelled());
        }
        let rec = self
            .recognizer
            .as_ref()
            .ok_or_else(|| AsrError::runtime(format!("{} is not loaded", self.label)))?;
        let r = rec
            .transcribe(SAMPLE_RATE, audio, &call)
            .map_err(|e| AsrError::runtime(format!("{}: {e}", self.label)))?;
        if self.cancel.load(Ordering::SeqCst) {
            return Err(AsrError::cancelled());
        }
        if let OfflineModel::Qwen3Asr {
            max_new_tokens: Some(cap), ..
        } = self.config.model
        {
            if !self.truncation_logged && r.tokens.len() >= cap as usize {
                self.truncation_logged = true;
                tracing::warn!(
                    target: "sigil::sherpa",
                    "{}: a {} ms utterance reached the {cap}-token limit; its caption may be cut off",
                    self.label,
                    audio.len() as u64 * 1000 / u64::from(SAMPLE_RATE)
                );
            }
        }
        let forced = match self.family {
            Some(Family::Qwen3Asr) => call.language.as_deref().and_then(qwen3_language_code),
            Some(Family::SenseVoice) => match &self.config.model {
                OfflineModel::SenseVoice { language: Some(l), .. } => SENSE_VOICE_LANGUAGES.iter().copied().find(|x| x == l),
                _ => None,
            },
            _ => None,
        };
        Ok(self.result(r, options, forced, elapsed_ms(t0)))
    }

    fn canceller(&self) -> Option<Canceller> {
        Some(Canceller::from_flag(self.cancel.clone()))
    }
}

// ---------------------------------------------------------------------------------------------
// Spoken language identification
// ---------------------------------------------------------------------------------------------

/// Spoken language ID as an engine: `transcribe` returns empty text with `lang` and a
/// `lang:<code>` tag (the pipeline uses it to pick the final engine's language in `"auto"` mode).
pub struct SherpaLidEngine {
    info: EngineInfo,
    lid: SpokenLanguageId,
    cancel: Arc<AtomicBool>,
}

/// Longest audio language ID looks at: Whisper's window is 30 s; a few seconds decide it.
const LID_MAX_SECONDS: usize = 10;

impl AsrEngine for SherpaLidEngine {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn transcribe(&mut self, audio: &[f32], _options: &DecodeOptions) -> Result<AsrResult, AsrError> {
        self.cancel.store(false, Ordering::SeqCst);
        let t0 = Instant::now();
        if !sherpa_offline::long_enough(audio.len(), SAMPLE_RATE) {
            return Ok(AsrResult::default());
        }
        let clip = &audio[..audio.len().min(SAMPLE_RATE as usize * LID_MAX_SECONDS)];
        let raw = self
            .lid
            .identify(SAMPLE_RATE, clip)
            .map_err(|e| AsrError::new(AsrErrorKind::Runtime, format!("{}: {e}", self.info.model)))?;
        if self.cancel.load(Ordering::SeqCst) {
            return Err(AsrError::cancelled());
        }
        let lang = lang::normalize(&raw);
        Ok(AsrResult {
            text: String::new(),
            tags: lang.iter().map(|l| format!("lang:{l}")).collect(),
            lang,
            no_speech_prob: None,
            confidence: None,
            decode_ms: elapsed_ms(t0),
        })
    }

    fn canceller(&self) -> Option<Canceller> {
        Some(Canceller::from_flag(self.cancel.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn options_parse_leniently() {
        let o = Options::parse(&json!({
            "language": "ja-JP", "hotwords": ["Sigil", "VRChat"], "hotwordsScore": 99.0, "decoding": "beam",
            "maxActivePaths": 1000, "maxNewTokens": 5, "useItn": false, "tailPaddingMs": 400, "kind": "sense_voice",
            "prompt": "  ", "apiKey": "secret"
        }));
        assert_eq!(o.language.as_deref(), Some("ja"));
        assert_eq!(o.hotwords, vec!["Sigil", "VRChat"]);
        assert_eq!(o.hotwords_score, Some(10.0));
        assert!(o.beam_search);
        assert_eq!(o.max_active_paths, Some(64));
        assert_eq!(o.max_new_tokens, Some(16));
        assert_eq!(o.use_itn, Some(false));
        assert_eq!(o.tail_padding_ms, Some(400));
        assert_eq!(o.kind.as_deref(), Some("sense_voice"));
        assert_eq!(o.prompt, None);
        let o = Options::parse(&json!({ "language": "auto", "hotwords": "a\nb", "maxNewTokens": "x", "decoding": 3 }));
        assert_eq!(o.language, None);
        assert_eq!(o.hotwords, vec!["a", "b"]);
        assert_eq!(o.max_new_tokens, None);
        assert!(!o.beam_search && !o.greedy);
        assert_eq!(Options::parse(&Value::Null), Options::default());
        let o = Options::parse(&json!({ "decoding": "greedy" }));
        assert!(o.greedy && !o.beam_search);
    }

    /// Catalog Parakeet starts on beam search, so the pipeline's custom terms never reload it;
    /// `decoding: "greedy"` keeps greedy search; other models only when asked.
    #[test]
    fn parakeet_starts_on_beam_search_unless_greedy_is_asked_for() {
        let default = Options::default();
        for id in [catalog::ids::PARAKEET_ULTRA, catalog::ids::PARAKEET_V3] {
            let c = catalog::find(id);
            assert!(starts_on_beam_search(c, &default), "{id}");
            assert!(!starts_on_beam_search(c, &Options::parse(&json!({ "decoding": "greedy" }))), "{id}");
            assert!(!starts_on_beam_search(c, &Options::parse(&json!({ "decoding": "greedy", "hotwords": ["Sigil"] }))));
        }
        // A bring-your-own transducer: only when asked for, or with hotwords up front.
        assert!(!starts_on_beam_search(None, &default));
        assert!(starts_on_beam_search(None, &Options::parse(&json!({ "decoding": "beam" }))));
        assert!(starts_on_beam_search(None, &Options::parse(&json!({ "hotwords": ["Sigil"] }))));
    }

    #[test]
    fn threads_follow_the_spec_and_are_capped() {
        let mut spec = EngineSpec::new(EngineKind::SherpaOffline, catalog::ids::SENSE_VOICE);
        assert_eq!(engine_threads(&spec, 2), 2);
        spec.threads = Some(16);
        assert_eq!(engine_threads(&spec, 2), MAX_THREADS);
        spec.threads = Some(0);
        assert_eq!(engine_threads(&spec, 2), 1);
    }

    #[test]
    fn preload_kill_switch() {
        assert!(preload_disabled(Some("0")) && preload_disabled(Some(" OFF ")) && preload_disabled(Some("false")));
        assert!(!preload_disabled(None) && !preload_disabled(Some("1")) && !preload_disabled(Some("")));
    }

    #[test]
    fn sense_voice_tags_map_to_the_contract() {
        assert_eq!(emotion_tag("HAPPY"), Some("happy"));
        assert_eq!(emotion_tag("neutral"), Some("neutral"));
        assert_eq!(emotion_tag("emo_unknown"), None);
        assert_eq!(event_tag("Speech"), None);
        assert_eq!(event_tag("BGM"), Some("bgm"));
        assert_eq!(event_tag("Laughter"), Some("laughter"));
        assert_eq!(event_tag("event_unk"), None);
    }

    #[test]
    fn unknown_ids_relative_paths_and_empty_models_are_invalid() {
        let opts = Options::default();
        for model in ["", "stt/not-a-sherpa-model", "relative/folder"] {
            let spec = EngineSpec::new(EngineKind::SherpaOffline, model);
            let e = resolve(&spec, Task::Offline, &opts).err().unwrap();
            assert_eq!(e.kind, AsrErrorKind::InvalidInput, "{model}: {e}");
        }
        let missing = std::env::temp_dir().join("sigil-no-such-sherpa-model-folder");
        let spec = EngineSpec::new(EngineKind::SherpaOffline, missing.display().to_string());
        assert_eq!(resolve(&spec, Task::Offline, &opts).err().unwrap().kind, AsrErrorKind::InvalidInput);
    }

    #[test]
    fn a_streaming_folder_is_refused_by_the_utterance_engine_and_vice_versa() {
        let root = std::env::temp_dir().join(format!("sigil-sherpa-engine-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let online = root.join("sherpa-onnx-streaming-zipformer-en");
        std::fs::create_dir_all(&online).unwrap();
        for f in ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"] {
            std::fs::write(online.join(f), b"x").unwrap();
        }
        let spec = EngineSpec::new(EngineKind::SherpaOffline, online.display().to_string());
        let e = resolve(&spec, Task::Offline, &Options::default())
            .err()
            .unwrap();
        assert_eq!(e.kind, AsrErrorKind::Unsupported, "{e}");
        assert!(e.message.contains("sherpa-online"), "{e}");
        let r = resolve(&spec, Task::Online, &Options::default()).unwrap();
        assert!(r.byo && r.catalog.is_none() && matches!(r.config, SherpaModel::Online(_)));
        assert_eq!(r.label, "sherpa-onnx-streaming-zipformer-en");
        // An explicit `mode` overrides the folder name.
        let offline = Options {
            mode: Some("offline".into()),
            ..Options::default()
        };
        assert!(matches!(resolve(&spec, Task::Offline, &offline).unwrap().config, SherpaModel::Offline(_)));
        let e = resolve(&spec, Task::Online, &offline).err().unwrap();
        assert!(e.message.contains("sherpa-offline"), "{e}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_catalog_model_that_is_not_installed_names_its_id() {
        // Uses the process's app data folder: models a developer installed there are skipped.
        for id in [catalog::ids::QWEN3_ASR_0_6B, catalog::ids::LID_WHISPER_BASE, catalog::ids::PARAKEET_ULTRA] {
            if crate::services::model_store::installed_path(id).is_some() {
                continue;
            }
            let spec = EngineSpec::new(EngineKind::SherpaOffline, id);
            let e = resolve(&spec, Task::Offline, &Options::default())
                .err()
                .unwrap();
            assert_eq!(e.kind, AsrErrorKind::NotInstalled, "{e}");
            assert!(e.message.contains(id), "{e}");
        }
    }

    #[test]
    fn a_missing_runtime_pack_is_not_installed() {
        if runtime_dir().is_some() || SherpaRuntime::loaded().is_some() {
            return; // a developer PC with the pack installed
        }
        let e = runtime(&EngineContext::headless()).err().unwrap();
        assert_eq!(e.kind, AsrErrorKind::NotInstalled);
        assert!(e.message.contains(RUNTIME_ID), "{e}");
    }

    // -----------------------------------------------------------------------------------------
    // Real models (ignored; see `speech_rt::sherpa::testkit` for the environment)
    // -----------------------------------------------------------------------------------------

    use crate::services::speech_rt::sherpa::catalog::ids;
    use crate::services::speech_rt::sherpa::testkit::{clip_16k, dll_lock, installed, runtime_installed, squash};

    /// The engine for catalog model `id` through the factory the pipeline uses (`None`: skipped).
    fn real_engine(id: &str, options: Value) -> Option<Box<dyn AsrEngine>> {
        if !runtime_installed() || installed(id).is_none() {
            return None;
        }
        let mut spec = EngineSpec::new(EngineKind::SherpaOffline, id);
        spec.options = options;
        Some(super::super::create(&spec, &EngineContext::headless()).unwrap_or_else(|e| panic!("{id}: {e}")))
    }

    fn lang_opts(lang: Option<&str>) -> DecodeOptions {
        DecodeOptions {
            language: lang.map(str::to_string),
            ..DecodeOptions::default()
        }
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/sense-voice-small-int8, and the test clips"]
    fn real_sense_voice_finals_carry_language_emotion_and_event_tags() {
        let _g = dll_lock();
        let Some(mut e) = real_engine(ids::SENSE_VOICE, Value::Null) else {
            return;
        };
        let caps = e.info().caps.clone();
        assert!(caps.utterance && caps.tags && caps.detects_language && !caps.streaming && !caps.hotwords);
        assert_eq!(e.info().runtime, runtime_label());
        for (clip, lang, needle) in [
            ("en.wav", "en", "tribalchieftain"),
            ("ja.wav", "ja", "弁当"),
            ("zh.wav", "zh", ""),
            ("yue.wav", "yue", ""),
            ("ko.wav", "ko", ""),
        ] {
            let Some(audio) = clip_16k(clip) else { return };
            let r = e.transcribe(&audio, &DecodeOptions::default()).unwrap();
            eprintln!("{clip}: {:?} {:?} {:?} {} ms", r.text, r.lang, r.tags, r.decode_ms);
            assert!(!r.text.is_empty(), "{clip}");
            assert!(squash(&r.text).contains(needle), "{clip}: {}", r.text);
            assert_eq!(r.lang.as_deref(), Some(lang), "{clip}");
            assert!(r.tags.contains(&format!("lang:{lang}")), "{clip}: {:?}", r.tags);
            assert!(r.tags.iter().any(|t| t.starts_with("emotion:")), "{clip}: {:?}", r.tags);
            assert!(r
                .tags
                .iter()
                .all(|t| t.starts_with("lang:") || t.starts_with("emotion:") || t.starts_with("event:")));
        }
        // A pinned language comes from the engine options (per-call hints are ignored by the
        // model); the result then carries no `lang:` detection tag.
        let Some(mut zh) = real_engine(ids::SENSE_VOICE, serde_json::json!({ "language": "zh" })) else {
            return;
        };
        let Some(audio) = clip_16k("zh.wav") else { return };
        let r = zh.transcribe(&audio, &lang_opts(Some("en"))).unwrap();
        assert_eq!(r.lang.as_deref(), Some("zh"));
        assert!(!r.tags.iter().any(|t| t.starts_with("lang:")), "{:?}", r.tags);
        // Too short to decode: empty, no native call.
        assert_eq!(
            e.transcribe(&[0.0; 800], &DecodeOptions::default())
                .unwrap(),
            AsrResult::default()
        );
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/qwen3-asr-0.6b-int8, and the test clips"]
    fn real_qwen3_asr_takes_the_language_and_context() {
        let _g = dll_lock();
        let Some(mut e) = real_engine(ids::QWEN3_ASR_0_6B, Value::Null) else {
            return;
        };
        let caps = e.info().caps.clone();
        // sherpa-onnx 1.13.8 does not return the language Qwen3-ASR heard: the router runs the
        // language-ID engine for it in auto mode (see `Family::detects_language`).
        assert!(caps.hotwords && caps.prompt && !caps.detects_language && caps.languages.len() == 30);
        let Some(ja) = clip_16k("ja-qwen.wav") else { return };
        let r = e.transcribe(&ja, &lang_opts(Some("ja"))).unwrap();
        eprintln!("ja forced: {:?} {:?} {} ms", r.text, r.lang, r.decode_ms);
        assert!(r.text.contains("運動神経"), "{}", r.text);
        assert_eq!(r.lang.as_deref(), Some("ja"));
        let auto = e.transcribe(&ja, &DecodeOptions::default()).unwrap();
        eprintln!("ja auto: {:?} {:?} {:?}", auto.text, auto.lang, auto.tags);
        assert!(auto.text.contains("運動神経"), "{}", auto.text);
        // If a sherpa-onnx upgrade starts reporting the detected language, `detects_language`
        // (and the routing table's `detectsLanguage`) can become true again.
        assert_eq!(auto.lang, None, "sherpa now reports Qwen3-ASR's language");
        assert!(!auto.tags.iter().any(|t| t.starts_with("lang:")), "{:?}", auto.tags);
        let Some(ru) = clip_16k("ru.wav") else { return };
        let plain = e.transcribe(&ru, &lang_opts(Some("ru"))).unwrap();
        let ctx = DecodeOptions {
            language: Some("ru".into()),
            hotwords: vec!["зоопарке".into(), "вольера".into()],
            prompt: None,
        };
        let biased = e.transcribe(&ru, &ctx).unwrap();
        eprintln!("ru: {:?}\nru + context: {:?}", plain.text, biased.text);
        assert!(squash(&plain.text).contains("барсук"), "{}", plain.text);
        // Measured: the context fixes "вольера" (plain decode: "ливера"); Qwen3-ASR treats it as
        // context, not a hard constraint.
        assert!(!squash(&plain.text).contains("вольера"), "{}", plain.text);
        assert!(squash(&biased.text).contains("вольера"), "context terms steer Qwen3-ASR: {}", biased.text);
        assert_eq!(biased.lang.as_deref(), Some("ru"));
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/parakeet-ultra-int8 / stt/parakeet-tdt-0.6b-v3-int8, and the test clips"]
    fn real_parakeet_decodes_european_languages_and_biases_hotwords() {
        let _g = dll_lock();
        for id in [ids::PARAKEET_ULTRA, ids::PARAKEET_V3] {
            let Some(mut e) = real_engine(id, Value::Null) else { continue };
            assert!(e.info().caps.hotwords && !e.info().caps.tags);
            for (clip, lang, needle) in [
                ("en-jfk.wav", "en", "asknotwhatyourcountry"),
                ("de.wav", "de", "wurst"),
                ("es.wav", "es", "país"),
                ("fr.wav", "fr", "pays"),
                ("ru.wav", "ru", "барсук"),
            ] {
                let Some(audio) = clip_16k(clip) else { return };
                let r = e.transcribe(&audio, &lang_opts(Some(lang))).unwrap();
                eprintln!("{id} {clip}: {:?} {:?} {} ms", r.text, r.lang, r.decode_ms);
                assert!(squash(&r.text).contains(needle), "{id} {clip}: {}", r.text);
                assert_eq!(r.lang.as_deref(), Some(lang), "the router's language is reported back");
            }
            // Japanese is not a Parakeet language: not reported as the result's language.
            let Some(en) = clip_16k("en-jfk.wav") else { return };
            assert_eq!(e.transcribe(&en, &lang_opts(Some("ja"))).unwrap().lang, None);
            // The first hotwords switch the recognizer to beam search with a derived bpe.vocab;
            // the transcript stays right.
            let hot = DecodeOptions {
                language: Some("en".into()),
                hotwords: vec!["country".into(), "Sigil".into()],
                prompt: None,
            };
            let r = e.transcribe(&en, &hot).unwrap();
            eprintln!("{id} en + hotwords: {:?}", r.text);
            assert!(squash(&r.text).contains("asknotwhatyourcountry"), "{}", r.text);
            let Some(ru) = clip_16k("ru.wav") else { return };
            let plain = e.transcribe(&ru, &lang_opts(Some("ru"))).unwrap();
            let biased = e
                .transcribe(
                    &ru,
                    &DecodeOptions {
                        language: Some("ru".into()),
                        hotwords: vec!["вольера".into(), "зоопарке".into()],
                        prompt: None,
                    },
                )
                .unwrap();
            eprintln!("{id} ru beam: {:?}\n{id} ru + hotwords: {:?}", plain.text, biased.text);
            assert!(squash(&biased.text).contains("барсук"), "{}", biased.text);
        }
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/qwen3-asr-0.6b-int8 (its test_wavs)"]
    fn real_qwen3_asr_transcribes_a_60_second_utterance_to_the_end() {
        let _g = dll_lock();
        let Some(mut e) = real_engine(ids::QWEN3_ASR_0_6B, Value::Null) else {
            return;
        };
        // 88 s of fast conversational English, cut to the longest utterance the pipeline closes.
        // sherpa's defaults decoded it to a single token; at 1024 context it stopped after 227.
        let Some(mut audio) = clip_16k("noise1-en.wav") else { return };
        audio.truncate(SAMPLE_RATE as usize * 60);
        let r = e.transcribe(&audio, &lang_opts(Some("en"))).unwrap();
        eprintln!("60 s in {} ms: {:?}", r.decode_ms, r.text);
        assert!(r.text.len() > 900, "{}", r.text);
        assert!(squash(&r.text).starts_with("mygirls"), "{}", r.text);
        // The last words of the minute are there.
        assert!(squash(&r.text).contains("youshotme"), "{}", r.text);
        // 51 s of Mandarin lyrics: the end of the song, too.
        let Some(zh) = clip_16k("qiqiu1.wav") else { return };
        let r = e.transcribe(&zh, &lang_opts(Some("zh"))).unwrap();
        eprintln!("51 s zh: {:?}", r.text);
        assert!(r.text.contains("慢慢死去"), "{}", r.text);
    }

    /// A catalog Parakeet engine made the way the pipeline makes it, as its concrete type.
    fn real_parakeet(id: &str, options: Value) -> Option<SherpaOfflineEngine> {
        if !runtime_installed() || installed(id).is_none() {
            return None;
        }
        let mut spec = EngineSpec::new(EngineKind::SherpaOffline, id);
        spec.options = options;
        let ctx = EngineContext::headless();
        let opts = Options::parse(&spec.options);
        let resolved = resolve(&spec, Task::Offline, &opts).unwrap();
        let SherpaModel::Offline(config) = resolved.config.clone() else {
            panic!("{id} is an utterance model")
        };
        let rt = runtime(&ctx).unwrap();
        Some(SherpaOfflineEngine::new(&spec, &ctx, rt, &resolved, config, opts).unwrap_or_else(|e| panic!("{id}: {e}")))
    }

    fn loads() -> u32 {
        LOAD_ATTEMPTS.with(std::cell::Cell::get)
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/parakeet-ultra-int8, and the test clips"]
    fn real_parakeet_takes_custom_terms_without_a_reload() {
        let _g = dll_lock();
        FAIL_LOADS.with(|f| f.set(false));
        let before = loads();
        let Some(mut e) = real_parakeet(ids::PARAKEET_ULTRA, Value::Null) else {
            return;
        };
        assert_eq!(loads() - before, 1, "one load at creation");
        assert!(e.config.decoding.is_beam_search() && e.config.bpe_vocab.is_some());
        let Some(en) = clip_16k("en-jfk.wav") else { return };
        let hot = DecodeOptions {
            language: Some("en".into()),
            hotwords: vec!["country".into(), "Sigil".into()],
            prompt: None,
        };
        for _ in 0..2 {
            let r = e.transcribe(&en, &hot).unwrap();
            assert!(squash(&r.text).contains("asknotwhatyourcountry"), "{}", r.text);
        }
        assert_eq!(loads() - before, 1, "custom terms never reload the model");
        assert!(!e.hotwords_failed);
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/parakeet-ultra-int8, and the test clips"]
    fn real_failed_beam_search_reload_is_tried_once() {
        let _g = dll_lock();
        FAIL_LOADS.with(|f| f.set(false));
        // A greedy transducer that switches when hotwords arrive (the bring-your-own path): made
        // greedy, then allowed to switch.
        let Some(mut e) = real_parakeet(ids::PARAKEET_ULTRA, json!({ "decoding": "greedy" })) else {
            return;
        };
        assert!(!e.config.decoding.is_beam_search());
        e.opts.greedy = false;
        let Some(en) = clip_16k("en-jfk.wav") else { return };
        let hot = DecodeOptions {
            language: Some("en".into()),
            hotwords: vec!["country".into()],
            prompt: None,
        };
        // Both the beam-search load and the greedy reload fail: the utterance fails once.
        FAIL_LOADS.with(|f| f.set(true));
        let before = loads();
        assert!(e.transcribe(&en, &hot).is_err());
        assert_eq!(loads() - before, 2, "beam search, then greedy");
        assert!(e.hotwords_failed && !e.config.decoding.is_beam_search() && e.recognizer.is_none());
        // Later utterances retry only the greedy load, never beam search again.
        assert!(e.transcribe(&en, &hot).is_err());
        assert_eq!(loads() - before, 3);
        FAIL_LOADS.with(|f| f.set(false));
        let r = e.transcribe(&en, &hot).unwrap();
        assert!(squash(&r.text).contains("asknotwhatyourcountry"), "{}", r.text);
        assert_eq!(loads() - before, 4);
        e.transcribe(&en, &hot).unwrap();
        assert_eq!(loads() - before, 4, "no reloads once it works");
        assert!(!e.config.decoding.is_beam_search());

        // Explicit greedy search: custom terms are unavailable, never a reload.
        let Some(mut g) = real_parakeet(ids::PARAKEET_ULTRA, json!({ "decoding": "greedy" })) else {
            return;
        };
        let before = loads();
        g.transcribe(&en, &hot).unwrap();
        assert_eq!(loads(), before);
        assert!(g.hotwords_failed && !g.config.decoding.is_beam_search());
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/moonshine-base-en-int8, and the test clips"]
    fn real_moonshine_transcribes_english() {
        let _g = dll_lock();
        let Some(mut e) = real_engine(ids::MOONSHINE_BASE_EN, Value::Null) else {
            return;
        };
        assert_eq!(e.info().caps.languages, vec!["en".to_string()]);
        let Some(audio) = clip_16k("en.wav") else { return };
        let r = e.transcribe(&audio, &DecodeOptions::default()).unwrap();
        eprintln!("moonshine: {:?} {} ms", r.text, r.decode_ms);
        assert!(squash(&r.text).contains("tribalchieftain"), "{}", r.text);
        assert_eq!(r.lang.as_deref(), Some("en"));
        // Silence gives no text (sherpa 1.13.8 fixed Moonshine v2 hallucinating on silence).
        let silence = vec![0.0f32; SAMPLE_RATE as usize * 2];
        assert_eq!(
            e.transcribe(&silence, &DecodeOptions::default())
                .unwrap()
                .text,
            ""
        );
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/lid-whisper-base-int8, and the test clips"]
    fn real_language_id_names_the_spoken_language() {
        let _g = dll_lock();
        let Some(mut e) = real_engine(ids::LID_WHISPER_BASE, Value::Null) else {
            return;
        };
        assert!(e.info().caps.detects_language && e.info().caps.languages.is_empty());
        for (clip, lang) in [
            ("en.wav", "en"),
            ("ja.wav", "ja"),
            ("zh.wav", "zh"),
            ("ru.wav", "ru"),
            ("de.wav", "de"),
            ("es.wav", "es"),
        ] {
            let Some(audio) = clip_16k(clip) else { return };
            let r = e.transcribe(&audio, &DecodeOptions::default()).unwrap();
            eprintln!("lid {clip}: {:?} {} ms", r.lang, r.decode_ms);
            assert_eq!(r.text, "");
            assert_eq!(r.lang.as_deref(), Some(lang), "{clip}");
            assert_eq!(r.tags, vec![format!("lang:{lang}")]);
        }
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/sense-voice-small-int8, and the test clips"]
    fn real_stale_cancel_never_eats_the_next_utterance() {
        let _g = dll_lock();
        let Some(mut e) = real_engine(ids::SENSE_VOICE, Value::Null) else {
            return;
        };
        let Some(audio) = clip_16k("en.wav") else { return };
        let c = e.canceller().expect("canceller");
        // A cancel that arrives between calls is for the decode that was running then; the next
        // utterance starts clean.
        c.cancel();
        let r = e.transcribe(&audio, &DecodeOptions::default()).unwrap();
        assert!(!r.text.is_empty());
    }
}
