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
//! | Moonshine | English only | none | text, `lang` = `en`; v2 utterances over `MOONSHINE_MAX_SEGMENT_MS` are decoded in pieces cut at quiet points (see there) |
//! | Language ID | — | — | empty text, `lang` + `lang:<code>` tag |
//!
//! # Engine options ([`EngineSpec::options`], camelCase, all optional; never logged)
//! `language` (default language / SenseVoice's pinned language), `hotwords` (string array used when
//! a call brings none), `hotwordsScore`, `prompt` (Qwen3-ASR default context), `decoding`
//! (`"greedy"` | `"beam"`; transducers: see `starts_on_beam_search`), `maxActivePaths`,
//! `maxNewTokens` (Qwen3-ASR; `max_total_len` grows with it), `useItn` (SenseVoice),
//! `moonshineMaxSegmentMs` (Moonshine: longest piece decoded in one call, `2000..=9000`, which can
//! only lower the default; `0` never splits), `tailPaddingMs` (streaming), and for bring-your-own
//! folders the detection hints `kind`, `mode`, `fp32`.

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
use super::long_utterance::{self, Splitter};
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
    /// `moonshineMaxSegmentMs`: `Some(0)` never splits, else clamped to
    /// `2000..=`[`MOONSHINE_MAX_SEGMENT_MS`] (a longer piece brings back the empty results the
    /// split exists to avoid).
    pub max_segment_ms: Option<u32>,
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
            max_segment_ms: n("moonshineMaxSegmentMs").map(|x| {
                if x < 1.0 {
                    0
                } else {
                    x.clamp(2_000.0, f64::from(MOONSHINE_MAX_SEGMENT_MS)) as u32
                }
            }),
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

/// Longest stretch of audio (ms) a Moonshine v2 model decodes in one call; a longer utterance is
/// cut at quiet points into pieces of at most this length ([`long_utterance::Splitter`]), each
/// decoded on its own, and the texts are joined. `moonshineMaxSegmentMs` can only lower it.
///
/// Why: sherpa-onnx 1.13.8 returns an EMPTY result for Moonshine v2 (encoder + merged decoder) audio
/// of 148 351 samples or more at 16 kHz (9.272 s): ONNX Runtime throws "Attempting to broadcast an
/// axis by a dimension other than 1" inside `offline-recognizer-moonshine-v2-impl` `DecodeStream`
/// and sherpa swallows it. Measured with `stt/moonshine-base-en-int8` (the only Moonshine catalog
/// model; CPU): 148 350 samples decode, 148 351 do not, to the sample, for different utterances,
/// gain, speaking rate (0.85x / 1.25x) and zero or noise padding before or after the speech, and
/// every longer length fails (checked up to 52 s), so only the length matters. The speech
/// pipeline closes utterances at up to 18 s by default (60 s at most), which is how long unbroken
/// speech lost its captions.
///
/// The limit is exact, so the margin below it is small: 9 s leaves 4 350 samples (0.27 s). A wider
/// margin (the first version cut at 8 s) cuts every utterance between the two lengths for nothing,
/// and a cut costs accuracy (words at the seam come out doubled, split or wrong): on the benchmark
/// corpus the 8-9 s English utterances scored a word error rate of 11 % decoded whole and 19 % cut
/// at 8 s. A shifted limit is caught by the ignored real-model test
/// `real_moonshine_sherpa_length_limit_is_still_where_the_split_assumes`, and a piece that still
/// comes back empty is decoded again in two halves ([`decode_in_chunks`]).
pub(super) const MOONSHINE_MAX_SEGMENT_MS: u32 = 9_000;

/// How `audio_len` samples are decoded by a model `model`: `Some` splitter when the audio must be
/// cut into pieces, `None` for the single-call path every short utterance (and every other family)
/// takes.
///
/// The limit is measured for the v2 layout (encoder + merged decoder, `merged_decoder` set) only;
/// the failure sits in sherpa-onnx's v2 implementation. A bring-your-own v1 folder (preprocessor +
/// uncached/cached decoders) takes the single call unless `moonshineMaxSegmentMs` asks for a limit.
fn split_for(model: &OfflineModel, opts: &Options, audio_len: usize) -> Option<Splitter> {
    let OfflineModel::Moonshine { merged_decoder, .. } = model else {
        return None;
    };
    let limit_ms = match opts.max_segment_ms {
        Some(ms) => ms,
        None if merged_decoder.is_some() => MOONSHINE_MAX_SEGMENT_MS,
        None => return None,
    };
    Splitter::new(SAMPLE_RATE, limit_ms).filter(|s| s.needs_split(audio_len))
}

/// Whether `text` has nothing to caption.
fn is_blank(text: &str) -> bool {
    text.trim().is_empty()
}

/// Whether an empty result for clearly voiced audio is worth a warning for `family`. Only for
/// Moonshine, whose empty results are the known trace of a failure sherpa-onnx swallows; other
/// families answer non-speech (music, applause, noise) with no text on purpose (SenseVoice with an
/// event tag, Parakeet), so a warning would be a false alarm for a streamer with loud game audio.
fn warns_on_empty_voiced(family: Option<Family>) -> bool {
    family == Some(Family::Moonshine)
}

/// Writes the warning "clearly voiced audio decoded to no text" at most once per
/// [`EmptyDecodeLog::INTERVAL`], counting what it skipped meanwhile.
///
/// sherpa-onnx swallows the errors of a model run and answers with an empty result (which is how
/// the Moonshine length limit went unnoticed), so an empty result for audio like this is the only
/// trace of such a failure. A permanent once-per-engine flag would hide every failure after the
/// first. Only lengths and shares are logged, never audio or text.
#[derive(Debug, Default)]
struct EmptyDecodeLog {
    /// When the last warning was written.
    last: Option<Instant>,
    /// Empty results seen since then that were not written.
    skipped: u32,
}

impl EmptyDecodeLog {
    const INTERVAL: Duration = Duration::from_secs(60);

    /// Whether a warning has been written.
    #[cfg(test)]
    fn has_warned(&self) -> bool {
        self.last.is_some()
    }

    /// `piece` of audio decoded to no text at `now`. Writes the warning when the piece is long and
    /// clearly voiced and the last warning is older than the interval; `true` when it did.
    fn note(&mut self, now: Instant, label: &str, piece: &[f32]) -> bool {
        if !long_utterance::is_long_and_voiced(piece, SAMPLE_RATE) {
            return false;
        }
        if self.last.is_some_and(|last| now.saturating_duration_since(last) < Self::INTERVAL) {
            self.skipped = self.skipped.saturating_add(1);
            return false;
        }
        let skipped = std::mem::take(&mut self.skipped);
        self.last = Some(now);
        let more = if skipped > 0 {
            format!(" ({skipped} more since the last warning)")
        } else {
            String::new()
        };
        tracing::warn!(
            target: "sigil::sherpa",
            "{label}: {} ms of audio with {:.0}% voiced frames decoded to no text; the model may have failed inside sherpa-onnx{more}",
            piece.len() as u64 * 1000 / u64::from(SAMPLE_RATE),
            long_utterance::voiced_fraction(piece, SAMPLE_RATE) * 100.0
        );
        true
    }
}

/// Add `part` to `merged`: its text after the text so far, and its language when none is set yet.
fn append_part(merged: &mut OfflineResult, part: OfflineResult) {
    if merged.lang.is_none() {
        merged.lang = part.lang;
    }
    long_utterance::append_text(&mut merged.text, &part.text);
}

/// A second attempt at `piece`, which decoded to no text: its two halves (cut at the quietest point
/// of the middle third) decoded one after the other. `None` when the piece is too short or too
/// quiet to be worth it, or when the halves have no text either.
///
/// Why: pieces of a cut utterance come back empty several times as often as whole utterances do
/// (measured 2.8 % of the pieces against 0.4 % of whole 3-8 s utterances, the short ones most
/// often), each losing a few seconds of words, and decoding the two halves recovered most of them.
fn retry_in_halves(
    piece: &[f32],
    splitter: &Splitter,
    cancel: &AtomicBool,
    decode: &mut impl FnMut(&[f32]) -> Result<OfflineResult, AsrError>,
) -> Result<Option<OfflineResult>, AsrError> {
    if !long_utterance::worth_retrying(piece, SAMPLE_RATE) {
        return Ok(None);
    }
    let Some(cut) = splitter.halve(piece) else {
        return Ok(None);
    };
    let mut joined = OfflineResult::default();
    for half in [&piece[..cut], &piece[cut..]] {
        let part = decode(half)?;
        if cancel.load(Ordering::SeqCst) {
            return Err(AsrError::cancelled());
        }
        append_part(&mut joined, part);
    }
    Ok((!is_blank(&joined.text)).then_some(joined))
}

/// Decode `audio` one piece at a time (`decode` runs once per piece, in order) and join the texts.
/// A piece with no text is decoded again in two halves ([`retry_in_halves`]); `on_empty` gets each
/// piece that still has none. The cancel flag is checked after every decode. The result carries the
/// text and the first language any piece reported; tokens and timestamps are per piece and are not
/// merged.
fn decode_in_chunks(
    audio: &[f32],
    splitter: &Splitter,
    cancel: &AtomicBool,
    mut decode: impl FnMut(&[f32]) -> Result<OfflineResult, AsrError>,
    mut on_empty: impl FnMut(&[f32]),
) -> Result<OfflineResult, AsrError> {
    let mut merged = OfflineResult::default();
    for range in splitter.chunks(audio) {
        let piece = &audio[range];
        let mut part = decode(piece)?;
        if cancel.load(Ordering::SeqCst) {
            return Err(AsrError::cancelled());
        }
        if is_blank(&part.text) {
            match retry_in_halves(piece, splitter, cancel, &mut decode)? {
                Some(again) => part = again,
                None => on_empty(piece),
            }
        }
        append_part(&mut merged, part);
    }
    Ok(merged)
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
    /// Rate limit of the warning for long, clearly voiced audio that decoded to no text.
    empty_voiced: EmptyDecodeLog,
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
            empty_voiced: EmptyDecodeLog::default(),
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
        // One native call per piece (a single piece unless the audio is split, see `split_for`).
        let label = self.label.as_str();
        let warns = warns_on_empty_voiced(self.family);
        let notes = &mut self.empty_voiced;
        let decode = |piece: &[f32]| -> Result<OfflineResult, AsrError> {
            rec.transcribe(SAMPLE_RATE, piece, &call)
                .map_err(|e| AsrError::runtime(format!("{label}: {e}")))
        };
        let on_empty = |piece: &[f32]| {
            if warns {
                notes.note(Instant::now(), label, piece);
            }
        };
        let r = match split_for(&self.config.model, &self.opts, audio.len()) {
            Some(splitter) => decode_in_chunks(audio, &splitter, &self.cancel, decode, on_empty)?,
            None => {
                let mut on_empty = on_empty;
                let r = decode(audio)?;
                if is_blank(&r.text) {
                    on_empty(audio);
                }
                r
            }
        };
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

    #[test]
    fn the_moonshine_segment_option_is_clamped_and_zero_means_never_split() {
        let ms = |v: Value| Options::parse(&json!({ "moonshineMaxSegmentMs": v })).max_segment_ms;
        assert_eq!(ms(json!(6000)), Some(6_000));
        assert_eq!(ms(json!(500)), Some(2_000));
        assert_eq!(ms(json!(MOONSHINE_MAX_SEGMENT_MS)), Some(MOONSHINE_MAX_SEGMENT_MS));
        // The option can only lower the limit: a longer piece brings the empty results back.
        assert_eq!(ms(json!(MOONSHINE_MAX_SEGMENT_MS + 1)), Some(MOONSHINE_MAX_SEGMENT_MS));
        assert_eq!(ms(json!(12_000)), Some(MOONSHINE_MAX_SEGMENT_MS));
        assert_eq!(ms(json!(10_000_000)), Some(MOONSHINE_MAX_SEGMENT_MS));
        assert_eq!(ms(json!(0)), Some(0));
        assert_eq!(ms(json!(-5)), Some(0));
        assert_eq!(ms(json!("x")), None);
        assert_eq!(Options::parse(&Value::Null).max_segment_ms, None);
    }

    /// The default limit sits just below the first length sherpa-onnx 1.13.8 fails on (148 351
    /// samples at 16 kHz, see `MOONSHINE_MAX_SEGMENT_MS`): far enough for a quarter of a second of
    /// margin, not much further (every 100 ms of margin cuts the utterances of that length for
    /// nothing, and a cut costs accuracy).
    #[test]
    fn the_default_moonshine_limit_sits_just_below_the_measured_failure() {
        let rate = SAMPLE_RATE as usize;
        let limit = MOONSHINE_MAX_SEGMENT_MS as usize * rate / 1000;
        assert!(limit + rate / 4 <= 148_350, "{limit} samples leave less than 0.25 s of margin");
        assert!(limit + rate / 2 >= 148_350, "{limit} samples leave more than 0.5 s of margin");
        assert_eq!(Splitter::new(SAMPLE_RATE, MOONSHINE_MAX_SEGMENT_MS).unwrap().limit(), limit);
    }

    /// A Moonshine config as sherpa-onnx loads it: v2 (encoder + merged decoder) or v1
    /// (preprocessor + uncached and cached decoders).
    fn moonshine(v2: bool) -> OfflineModel {
        let path = |n: &str| Some(std::path::PathBuf::from(n));
        OfflineModel::Moonshine {
            preprocessor: if v2 { None } else { path("preprocess.onnx") },
            encoder: std::path::PathBuf::from("encode.onnx"),
            uncached_decoder: if v2 { None } else { path("uncached_decode.onnx") },
            cached_decoder: if v2 { None } else { path("cached_decode.onnx") },
            merged_decoder: if v2 { path("decoder_model_merged.ort") } else { None },
        }
    }

    /// Short utterances (up to the limit, inclusive) and every other model keep the single native
    /// call: no splitter at all.
    #[test]
    fn only_long_moonshine_v2_audio_is_split() {
        let rate = SAMPLE_RATE as usize;
        let default = Options::default();
        let v2 = moonshine(true);
        let limit = MOONSHINE_MAX_SEGMENT_MS as usize * rate / 1000;
        // Up to the limit, one call; this includes the 8 to 9 s utterances, which decode whole
        // (cutting them scored worse).
        for len in [0, 1, rate, 5 * rate, 8 * rate, 8 * rate + rate / 2, limit - 1, limit] {
            assert_eq!(split_for(&v2, &default, len), None, "{len} samples");
        }
        for len in [limit + 1, 10 * rate, 18 * rate, 60 * rate] {
            let s = split_for(&v2, &default, len).expect("split");
            assert_eq!(s.limit(), limit);
        }
        let other_models = [
            OfflineModel::Dolphin { model: "m.onnx".into() },
            OfflineModel::Omnilingual { model: "m.onnx".into() },
            OfflineModel::FireRedAsrCtc { model: "m.onnx".into() },
        ];
        for model in &other_models {
            assert_eq!(split_for(model, &default, 60 * rate), None, "{}", model.family());
        }
        // The option moves the limit; 0 turns splitting off.
        let four = Options::parse(&json!({ "moonshineMaxSegmentMs": 4000 }));
        assert_eq!(split_for(&v2, &four, 5 * rate).unwrap().limit(), 4 * rate);
        assert_eq!(split_for(&v2, &four, 4 * rate), None);
        let off = Options::parse(&json!({ "moonshineMaxSegmentMs": 0 }));
        assert_eq!(split_for(&v2, &off, 60 * rate), None);
        // ... but never beyond the measured limit: 12 s from the option is 9 s.
        let twelve = Options::parse(&json!({ "moonshineMaxSegmentMs": 12_000 }));
        assert_eq!(split_for(&v2, &twelve, 10 * rate).unwrap().limit(), limit);
        assert_eq!(split_for(&v2, &twelve, 9 * rate), None);
    }

    /// The limit was measured for the v2 layout (merged decoder): a bring-your-own v1 folder is
    /// not cut unless the option asks for it.
    #[test]
    fn a_moonshine_v1_folder_is_split_only_when_asked() {
        let rate = SAMPLE_RATE as usize;
        let v1 = moonshine(false);
        assert_eq!(split_for(&v1, &Options::default(), 60 * rate), None);
        let five = Options::parse(&json!({ "moonshineMaxSegmentMs": 5000 }));
        assert_eq!(split_for(&v1, &five, 12 * rate).unwrap().limit(), 5 * rate);
        assert_eq!(split_for(&v1, &five, 5 * rate), None);
        let off = Options::parse(&json!({ "moonshineMaxSegmentMs": 0 }));
        assert_eq!(split_for(&v1, &off, 60 * rate), None);
    }

    /// Fake recognizer: the text of a piece names its length in milliseconds.
    fn fake_piece(piece: &[f32]) -> OfflineResult {
        OfflineResult {
            text: format!("piece {}", piece.len() * 1000 / SAMPLE_RATE as usize),
            ..OfflineResult::default()
        }
    }

    /// `ms` of loud, clearly voiced audio.
    fn tone(ms: usize) -> Vec<f32> {
        (0..ms * SAMPLE_RATE as usize / 1000)
            .map(|i| if i % 2 == 0 { 0.3 } else { -0.3 })
            .collect()
    }

    /// `ms` of digital silence.
    fn pause(ms: usize) -> Vec<f32> {
        vec![0.0f32; ms * SAMPLE_RATE as usize / 1000]
    }

    fn millis(samples: usize) -> usize {
        samples * 1000 / SAMPLE_RATE as usize
    }

    #[test]
    fn long_audio_is_decoded_piece_by_piece_and_the_texts_are_joined() {
        let rate = SAMPLE_RATE as usize;
        // 18 s with a pause at 6.0..6.4 s and one at 12.0..12.4 s.
        let audio = [tone(6_000), pause(400), tone(5_600), pause(400), tone(5_600)].concat();
        let splitter = Splitter::new(SAMPLE_RATE, MOONSHINE_MAX_SEGMENT_MS).unwrap();
        let cancel = AtomicBool::new(false);
        let mut seen = Vec::new();
        let r = decode_in_chunks(
            &audio,
            &splitter,
            &cancel,
            |piece| {
                seen.push(piece.len());
                let mut r = fake_piece(piece);
                if seen.len() == 2 {
                    r.lang = Some("en".into());
                }
                Ok(r)
            },
            |_| panic!("no piece is empty"),
        )
        .unwrap();
        // Cut in the middle of each pause, every piece within the limit, nothing lost or repeated.
        assert_eq!(seen, vec![6_200 * rate / 1000, 6_000 * rate / 1000, 5_800 * rate / 1000]);
        assert_eq!(seen.iter().sum::<usize>(), audio.len());
        assert_eq!(r.text, "piece 6200 piece 6000 piece 5800");
        assert_eq!(r.lang.as_deref(), Some("en"), "the first language a piece reports");
    }

    /// Pieces of 6.2 s, 6.0 s (silence, 5.6 s of speech, silence) and 5.8 s.
    fn three_pieces() -> (Vec<f32>, Splitter) {
        let audio = [tone(6_000), pause(400), tone(5_600), pause(400), tone(5_600)].concat();
        (audio, Splitter::new(SAMPLE_RATE, MOONSHINE_MAX_SEGMENT_MS).unwrap())
    }

    #[test]
    fn a_piece_without_text_is_decoded_again_as_two_halves() {
        let (audio, splitter) = three_pieces();
        let cancel = AtomicBool::new(false);
        let mut seen = Vec::new();
        let mut empties = Vec::new();
        let r = decode_in_chunks(
            &audio,
            &splitter,
            &cancel,
            |piece| {
                seen.push(millis(piece.len()));
                // The second piece comes back empty as a whole; its halves decode.
                Ok(if millis(piece.len()) == 6_000 { OfflineResult::default() } else { fake_piece(piece) })
            },
            |piece| empties.push(millis(piece.len())),
        )
        .unwrap();
        assert_eq!(seen, vec![6_200, 6_000, 3_000, 3_000, 5_800], "whole, whole (empty), two halves, whole");
        assert!(empties.is_empty(), "the halves had text: {empties:?}");
        assert_eq!(r.text, "piece 6200 piece 3000 piece 3000 piece 5800");
    }

    #[test]
    fn a_piece_whose_halves_have_no_text_either_is_reported_once() {
        let (audio, splitter) = three_pieces();
        let cancel = AtomicBool::new(false);
        let mut seen = Vec::new();
        let mut empties = Vec::new();
        let r = decode_in_chunks(
            &audio,
            &splitter,
            &cancel,
            |piece| {
                seen.push(millis(piece.len()));
                // Everything inside the second piece is empty (whole and halves).
                Ok(if millis(piece.len()) <= 6_000 && seen.len() >= 2 && seen.len() <= 4 { OfflineResult::default() } else { fake_piece(piece) })
            },
            |piece| empties.push(millis(piece.len())),
        )
        .unwrap();
        assert_eq!(seen, vec![6_200, 6_000, 3_000, 3_000, 5_800]);
        assert_eq!(empties, vec![6_000], "the piece, not its halves, is reported");
        assert_eq!(r.text, "piece 6200 piece 5800", "the other pieces keep their text");
    }

    #[test]
    fn silent_or_short_pieces_are_not_decoded_again() {
        let cancel = AtomicBool::new(false);
        // Silence: every piece is empty and none is retried (one decode each).
        let silent = pause(20_000);
        let splitter = Splitter::new(SAMPLE_RATE, MOONSHINE_MAX_SEGMENT_MS).unwrap();
        let pieces = splitter.chunks(&silent).count();
        assert!(pieces >= 3);
        let (mut decodes, mut empties) = (0, 0);
        let r = decode_in_chunks(
            &silent,
            &splitter,
            &cancel,
            |_| {
                decodes += 1;
                Ok(OfflineResult::default())
            },
            |_| empties += 1,
        )
        .unwrap();
        assert_eq!((decodes, empties, r.text.as_str()), (pieces, pieces, ""));
        // Voiced pieces under 1.5 s (a 2 s limit cuts 2.3 s into two) are not retried either.
        let short = tone(2_300);
        let small = Splitter::new(SAMPLE_RATE, 2_000).unwrap();
        let pieces = small.chunks(&short).count();
        assert_eq!(pieces, 2);
        let (mut decodes, mut empties) = (0, 0);
        decode_in_chunks(
            &short,
            &small,
            &cancel,
            |piece| {
                assert!(piece.len() < 3 * SAMPLE_RATE as usize / 2);
                decodes += 1;
                Ok(OfflineResult::default())
            },
            |_| empties += 1,
        )
        .unwrap();
        assert_eq!((decodes, empties), (pieces, pieces));
    }

    /// What `tracing` events of `f` print (warn and up, no timestamps).
    fn captured_log(f: impl FnOnce()) -> String {
        #[derive(Clone, Default)]
        struct Sink(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for Sink {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let sink = Sink::default();
        let writer = sink.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .without_time()
            .with_max_level(tracing::Level::WARN)
            .finish();
        tracing::subscriber::with_default(subscriber, f);
        let out = sink.0.lock().unwrap().clone();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn an_empty_decode_of_long_voiced_audio_is_logged_without_content_and_rate_limited() {
        let voiced = tone(12_000);
        let t0 = Instant::now();
        let mut notes = EmptyDecodeLog::default();
        assert!(!notes.has_warned());
        let secs = Duration::from_secs;
        let log = captured_log(|| {
            assert!(notes.note(t0, "stt/moonshine-base-en-int8", &voiced));
            // Inside the interval: counted, not written.
            assert!(!notes.note(t0 + secs(1), "stt/moonshine-base-en-int8", &voiced));
            assert!(!notes.note(t0 + secs(59), "stt/moonshine-base-en-int8", &voiced));
            // Later: written again, with the count. A failure after the first is not lost.
            assert!(notes.note(t0 + secs(61), "stt/moonshine-base-en-int8", &voiced));
            assert!(!notes.note(t0 + secs(62), "stt/moonshine-base-en-int8", &voiced));
            assert!(notes.note(t0 + secs(200), "stt/moonshine-base-en-int8", &voiced));
        });
        assert!(notes.has_warned());
        let lines: Vec<&str> = log.lines().collect();
        assert_eq!(lines.len(), 3, "{log}");
        assert!(lines[0].contains("WARN") && lines[0].contains("sigil::sherpa"), "{log}");
        assert!(lines[0].contains("stt/moonshine-base-en-int8") && lines[0].contains("12000 ms") && lines[0].contains("100% voiced"), "{log}");
        assert!(!lines[0].contains("more since"), "{log}");
        assert!(lines[1].contains("(2 more since the last warning)"), "{log}");
        assert!(lines[2].contains("(1 more since the last warning)"), "{log}");
        // Silence, short audio and quiet noise are not a warning, and do not count or start the
        // interval.
        let hiss: Vec<f32> = (0..12 * SAMPLE_RATE as usize).map(|i| if i % 2 == 0 { 0.001 } else { -0.001 }).collect();
        let quiet = |piece: &[f32]| {
            let mut fresh = EmptyDecodeLog::default();
            let log = captured_log(|| assert!(!fresh.note(t0, "m", piece)));
            (log, fresh.has_warned(), fresh.skipped)
        };
        assert_eq!(quiet(&pause(12_000)), (String::new(), false, 0));
        assert_eq!(quiet(&voiced[..2 * SAMPLE_RATE as usize]), (String::new(), false, 0));
        assert_eq!(quiet(&hiss), (String::new(), false, 0));
    }

    /// SenseVoice, Parakeet and the rest answer music and noise with no text on purpose; only
    /// Moonshine's empty results are a failure sign.
    #[test]
    fn only_moonshine_warns_about_empty_results_for_voiced_audio() {
        assert!(warns_on_empty_voiced(Some(Family::Moonshine)));
        for family in [Some(Family::SenseVoice), Some(Family::Qwen3Asr), Some(Family::ParakeetTdt), Some(Family::WhisperLid), None] {
            assert!(!warns_on_empty_voiced(family), "{family:?}");
        }
    }

    #[test]
    fn a_failed_or_cancelled_piece_ends_the_utterance() {
        let audio = vec![0.1f32; 30 * SAMPLE_RATE as usize];
        let splitter = Splitter::new(SAMPLE_RATE, 8_000).unwrap();
        let cancel = AtomicBool::new(false);
        let mut calls = 0;
        let err = decode_in_chunks(
            &audio,
            &splitter,
            &cancel,
            |piece| {
                calls += 1;
                if calls == 2 {
                    Err(AsrError::runtime("boom"))
                } else {
                    Ok(fake_piece(piece))
                }
            },
            |_| {},
        )
        .unwrap_err();
        assert_eq!((calls, err.kind), (2, AsrErrorKind::Runtime));
        // A cancel that arrives while a piece decodes stops before the next one.
        let mut calls = 0;
        let err = decode_in_chunks(
            &audio,
            &splitter,
            &cancel,
            |piece| {
                calls += 1;
                cancel.store(true, Ordering::SeqCst);
                Ok(fake_piece(piece))
            },
            |_| {},
        )
        .unwrap_err();
        assert_eq!((calls, err.kind), (1, AsrErrorKind::Cancelled));
    }

    #[test]
    fn a_failed_or_cancelled_retry_ends_the_utterance() {
        let (audio, splitter) = three_pieces();
        let cancel = AtomicBool::new(false);
        // The first half of the second piece fails.
        let mut calls = 0;
        let err = decode_in_chunks(
            &audio,
            &splitter,
            &cancel,
            |piece| {
                calls += 1;
                match calls {
                    2 => Ok(OfflineResult::default()),
                    3 => Err(AsrError::runtime("boom")),
                    _ => Ok(fake_piece(piece)),
                }
            },
            |_| {},
        )
        .unwrap_err();
        assert_eq!((calls, err.kind), (3, AsrErrorKind::Runtime));
        // A cancel that arrives while the first half decodes stops before the second.
        let mut calls = 0;
        let err = decode_in_chunks(
            &audio,
            &splitter,
            &cancel,
            |piece| {
                calls += 1;
                if calls == 3 {
                    cancel.store(true, Ordering::SeqCst);
                }
                Ok(if calls == 2 { OfflineResult::default() } else { fake_piece(piece) })
            },
            |_| {},
        )
        .unwrap_err();
        assert_eq!((calls, err.kind), (3, AsrErrorKind::Cancelled));
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

    // -----------------------------------------------------------------------------------------
    // Moonshine: utterances beyond sherpa-onnx's length limit
    // -----------------------------------------------------------------------------------------

    /// FLEURS English clips (16 kHz, read speech) with their reference transcripts from the P7
    /// benchmark corpus at `SIGIL_TEST_MODELS_DIR/p7/corpus` (`None`, with a skip message, when it
    /// is not there). Only clean clips of at least `min_seconds`, longest first, at most `max` of
    /// them.
    fn corpus_english(min_seconds: f64, max: usize) -> Option<Vec<(Vec<f32>, String)>> {
        corpus_english_between(min_seconds, f64::INFINITY, max)
    }

    /// Like [`corpus_english`], for clips of `min_seconds` up to `max_seconds`.
    fn corpus_english_between(min_seconds: f64, max_seconds: f64, max: usize) -> Option<Vec<(Vec<f32>, String)>> {
        corpus_clips(&["clean"], min_seconds, max_seconds, max, true)
    }

    /// English corpus clips recorded in one of `conditions` (`clean`, `mic_codec`, `room_light`,
    /// `babble_snr20`, `music_snr20`, `sfx_snr20`, `vrchat_mix`, ...) of `min_seconds` up to
    /// `max_seconds`, at most `max` of them: the longest first, or in the order of the manifest.
    fn corpus_clips(conditions: &[&str], min_seconds: f64, max_seconds: f64, max: usize, longest_first: bool) -> Option<Vec<(Vec<f32>, String)>> {
        let root = std::path::PathBuf::from(std::env::var_os("SIGIL_TEST_MODELS_DIR")?)
            .join("p7")
            .join("corpus");
        let Some(manifest) = std::fs::read(root.join("corpus.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        else {
            eprintln!("{} has no corpus.json; skipped", root.display());
            return None;
        };
        let mut items: Vec<(f64, String, String)> = manifest["items"]
            .as_array()?
            .iter()
            .filter(|i| i["language"] == "en" && conditions.iter().any(|c| i["condition"] == *c))
            .filter_map(|i| Some((i["seconds"].as_f64()?, i["audio"].as_str()?.to_string(), i["reference"].as_str()?.to_string())))
            .filter(|(seconds, _, reference)| (min_seconds..=max_seconds).contains(seconds) && !reference.trim().is_empty())
            .collect();
        if longest_first {
            items.sort_by(|a, b| b.0.total_cmp(&a.0));
        }
        items.truncate(max);
        let mut out = Vec::new();
        for (_, audio, reference) in items {
            let (mono, rate) = crate::services::speech_rt::sherpa::probe::read_wav(&root.join(audio)).expect("readable corpus clip");
            assert_eq!(rate, SAMPLE_RATE, "the corpus is 16 kHz");
            out.push((mono, reference));
        }
        Some(out)
    }

    /// Lower-case words of `text`: letters, digits and apostrophes; everything else separates.
    fn words(text: &str) -> Vec<String> {
        text.to_lowercase()
            .replace('\u{2019}', "'")
            .split(|c: char| !(c.is_alphanumeric() || c == '\''))
            .filter(|w| !w.is_empty())
            .map(str::to_string)
            .collect()
    }

    /// Word error rate of `hypothesis` against `reference` (edit distance over words).
    fn wer(reference: &str, hypothesis: &str) -> f64 {
        let (r, h) = (words(reference), words(hypothesis));
        let mut row: Vec<usize> = (0..=h.len()).collect();
        for (i, rw) in r.iter().enumerate() {
            let mut diag = row[0];
            row[0] = i + 1;
            for (j, hw) in h.iter().enumerate() {
                let up = row[j + 1];
                row[j + 1] = (diag + usize::from(rw != hw)).min(up + 1).min(row[j] + 1);
                diag = up;
            }
        }
        row[h.len()] as f64 / r.len().max(1) as f64
    }

    #[test]
    fn wer_counts_word_edits() {
        assert_eq!(wer("The cat sat.", "the cat sat"), 0.0);
        assert!((wer("a b c d", "a x c") - 0.5).abs() < 1e-9);
        assert_eq!(wer("a b", ""), 1.0);
        assert_eq!(wer("don't stop", "Don\u{2019}t stop!"), 0.0);
    }

    /// The Moonshine engine as its concrete type (`real_parakeet` builds any catalog offline model).
    fn real_moonshine(options: Value) -> Option<SherpaOfflineEngine> {
        real_parakeet(ids::MOONSHINE_BASE_EN, options)
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/moonshine-base-en-int8, and SIGIL_TEST_MODELS_DIR with the p7 corpus"]
    fn real_moonshine_sherpa_length_limit_is_still_where_the_split_assumes() {
        let _g = dll_lock();
        let Some(e) = real_moonshine(Value::Null) else { return };
        let Some(clips) = corpus_english(15.0, 1) else { return };
        let audio = &clips[0].0;
        let raw = |n: usize| {
            e.recognizer
                .as_ref()
                .unwrap()
                .transcribe(SAMPLE_RATE, &audio[..n], &sherpa_offline::DecodeOptions::default())
                .unwrap()
                .text
        };
        // Measured with sherpa-onnx 1.13.8 (see MOONSHINE_MAX_SEGMENT_MS). If this fails because
        // the longer decode now works, sherpa fixed the bug: the split can go (or the limit grow).
        assert!(!raw(148_350).is_empty(), "148350 samples must decode");
        assert!(raw(148_351).is_empty(), "sherpa-onnx decodes 148351 samples now: re-measure the Moonshine limit");
        assert!(raw(audio.len()).is_empty(), "{} samples", audio.len());
        // The default limit leaves a small margin below the measurement.
        let limit = MOONSHINE_MAX_SEGMENT_MS as usize * SAMPLE_RATE as usize / 1000;
        assert!(limit + SAMPLE_RATE as usize / 4 <= 148_350, "at least a quarter of a second of margin");
        assert!(!raw(limit).is_empty(), "the default limit must decode");
        // With splitting off (`moonshineMaxSegmentMs: 0`) the engine hits the limit and says so
        // (once): the empty result for a long, clearly voiced utterance is flagged.
        let Some(mut off) = real_moonshine(json!({ "moonshineMaxSegmentMs": 0 })) else { return };
        assert!(!off.empty_voiced.has_warned());
        assert_eq!(off.transcribe(audio, &DecodeOptions::default()).unwrap().text, "");
        assert!(off.empty_voiced.has_warned(), "the empty decode of {} voiced samples went unnoticed", audio.len());
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/moonshine-base-en-int8, and SIGIL_TEST_MODELS_DIR with the p7 corpus"]
    fn real_moonshine_transcribes_clips_of_15_seconds_and_more() {
        let _g = dll_lock();
        let Some(mut e) = real_moonshine(Value::Null) else { return };
        let Some(clips) = corpus_english(15.0, 6) else { return };
        assert!(!clips.is_empty());
        let mut rates = Vec::new();
        for (audio, reference) in &clips {
            assert!(audio.len() >= 15 * SAMPLE_RATE as usize);
            let r = e.transcribe(audio, &DecodeOptions::default()).unwrap();
            let rate = wer(reference, &r.text);
            eprintln!("{:.1} s: WER {rate:.3} in {} ms\n  ref: {reference}\n  got: {}", audio.len() as f64 / 16_000.0, r.decode_ms, r.text);
            assert!(!r.text.trim().is_empty(), "no text for a {:.1} s clip", audio.len() as f64 / 16_000.0);
            assert_eq!(r.lang.as_deref(), Some("en"));
            rates.push(rate);
        }
        assert!(!e.empty_voiced.has_warned(), "nothing decoded to an empty text");
        let mean = rates.iter().sum::<f64>() / rates.len() as f64;
        eprintln!("WERs {rates:.3?}, mean {mean:.3}");
        assert!(rates.iter().all(|r| *r <= 0.40), "{rates:.3?}");
        assert!(mean <= 0.25, "mean WER {mean:.3}");
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/moonshine-base-en-int8, and SIGIL_TEST_MODELS_DIR with the p7 corpus"]
    fn real_moonshine_transcribes_a_minute_of_speech_and_follows_the_option() {
        let _g = dll_lock();
        let Some(mut e) = real_moonshine(Value::Null) else { return };
        let Some(clips) = corpus_english(14.0, 4) else { return };
        // The longest utterance the speech pipeline can close is 60 s: four clips back to back.
        let audio: Vec<f32> = clips.iter().flat_map(|(a, _)| a.iter().copied()).collect();
        let reference = clips.iter().map(|(_, r)| r.as_str()).collect::<Vec<_>>().join(" ");
        assert!(audio.len() > 55 * SAMPLE_RATE as usize, "{} s", audio.len() / SAMPLE_RATE as usize);
        let r = e.transcribe(&audio, &DecodeOptions::default()).unwrap();
        let rate = wer(&reference, &r.text);
        eprintln!("{:.1} s: WER {rate:.3} in {} ms\n  got: {}", audio.len() as f64 / 16_000.0, r.decode_ms, r.text);
        assert!(rate <= 0.25, "WER {rate:.3}: {}", r.text);
        // A smaller limit from the engine options cuts more often and still reads right.
        let Some(mut small) = real_moonshine(json!({ "moonshineMaxSegmentMs": 4000 })) else { return };
        let r = small.transcribe(&audio, &DecodeOptions::default()).unwrap();
        let rate = wer(&reference, &r.text);
        eprintln!("4 s pieces: WER {rate:.3} in {} ms", r.decode_ms);
        assert!(rate <= 0.30, "WER {rate:.3}: {}", r.text);
    }

    /// Pieces of a cut utterance come back empty from time to time (the corpus has a few); decoding
    /// them again as two halves brings their words back.
    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/moonshine-base-en-int8, and SIGIL_TEST_MODELS_DIR with the p7 corpus"]
    fn real_moonshine_pieces_that_come_back_empty_are_decoded_again() {
        let _g = dll_lock();
        let Some(mut e) = real_moonshine(Value::Null) else { return };
        let conditions = ["clean", "mic_codec", "room_light", "babble_snr20", "music_snr20", "sfx_snr20", "vrchat_mix"];
        let Some(clips) = corpus_clips(&conditions, 9.3, f64::INFINITY, usize::MAX, false) else { return };
        let splitter = Splitter::new(SAMPLE_RATE, MOONSHINE_MAX_SEGMENT_MS).unwrap();
        let (mut found, mut recovered_words) = (0, 0);
        for (audio, reference) in &clips {
            // Every piece decoded alone, as the engine decodes it first.
            let alone: Vec<String> = splitter
                .chunks(audio)
                .map(|r| {
                    let rec = e.recognizer.as_ref().unwrap();
                    rec.transcribe(SAMPLE_RATE, &audio[r], &sherpa_offline::DecodeOptions::default())
                        .unwrap()
                        .text
                })
                .collect();
            if alone.iter().all(|t| !is_blank(t)) {
                continue;
            }
            found += 1;
            let alone_words: usize = alone.iter().map(|t| words(t).len()).sum();
            let via_engine = e.transcribe(audio, &DecodeOptions::default()).unwrap();
            let engine_words = words(&via_engine.text).len();
            eprintln!(
                "{:.1} s: {} of {} pieces empty; {alone_words} words alone, {engine_words} through the engine (reference {})",
                audio.len() as f64 / 16_000.0,
                alone.iter().filter(|t| is_blank(t)).count(),
                alone.len(),
                words(reference).len()
            );
            assert!(engine_words >= alone_words, "the retry must not lose words");
            recovered_words += engine_words - alone_words;
            if found == 4 {
                break;
            }
        }
        assert!(found >= 1, "no clip of the corpus has a piece that decodes to no text any more");
        assert!(recovered_words > 0, "{found} clips with empty pieces and none of their words came back");
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/moonshine-base-en-int8, and SIGIL_TEST_MODELS_DIR with the p7 corpus"]
    fn real_moonshine_utterances_up_to_the_limit_are_decoded_whole() {
        let _g = dll_lock();
        let Some(mut e) = real_moonshine(Value::Null) else { return };
        // The 8 to 9 s clips that cutting at 8 s used to cut for nothing.
        let Some(clips) = corpus_english_between(8.0, MOONSHINE_MAX_SEGMENT_MS as f64 / 1000.0, 8) else { return };
        assert!(clips.len() >= 3, "{} clips of 8 to 9 s", clips.len());
        for (audio, reference) in &clips {
            let direct = e
                .recognizer
                .as_ref()
                .unwrap()
                .transcribe(SAMPLE_RATE, audio, &sherpa_offline::DecodeOptions::default())
                .unwrap();
            let via_engine = e.transcribe(audio, &DecodeOptions::default()).unwrap();
            eprintln!("{:.2} s: WER {:.3}", audio.len() as f64 / 16_000.0, wer(reference, &via_engine.text));
            assert_eq!(via_engine.text, direct.text, "{:.2} s was cut", audio.len() as f64 / 16_000.0);
            assert!(!direct.text.is_empty());
        }
        assert!(!e.empty_voiced.has_warned());
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/moonshine-base-en-int8, and SIGIL_TEST_MODELS_DIR with the p7 corpus"]
    fn real_moonshine_short_utterances_take_the_unchanged_single_call() {
        let _g = dll_lock();
        let Some(mut e) = real_moonshine(Value::Null) else { return };
        let Some(clips) = corpus_english(0.0, 100) else { return };
        let limit = MOONSHINE_MAX_SEGMENT_MS as usize * SAMPLE_RATE as usize / 1000;
        let mut checked = 0;
        for (audio, _) in clips.iter().filter(|(a, _)| a.len() <= limit).take(8) {
            let direct = e
                .recognizer
                .as_ref()
                .unwrap()
                .transcribe(SAMPLE_RATE, audio, &sherpa_offline::DecodeOptions::default())
                .unwrap();
            let via_engine = e.transcribe(audio, &DecodeOptions::default()).unwrap();
            assert_eq!(via_engine.text, direct.text);
            assert!(!direct.text.is_empty());
            checked += 1;
        }
        assert!(checked >= 4, "{checked} clips of at most the limit");
        // A silent long input is not an error and not logged as a failure.
        let silence = vec![0.0f32; SAMPLE_RATE as usize * 20];
        assert_eq!(e.transcribe(&silence, &DecodeOptions::default()).unwrap().text, "");
        assert!(!e.empty_voiced.has_warned());
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
