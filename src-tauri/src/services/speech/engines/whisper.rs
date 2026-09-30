//! Whisper (whisper-rs / whisper.cpp, CPU) utterance engine, including Anime-Whisper BYO models.
//!
//! Owned by P2-A. The P0 hardening lives here: one reused `WhisperState` per model, one decode
//! window per utterance, `suppress_nst`, a token cap that scales with the utterance, the prompt /
//! hotwords tokenized once per (model, prompt), OpenAI's start-of-transcript no-speech probability
//! (utterances above the threshold return no text), and an abort callback so a stop never waits
//! for a long decode. The hallucination / repeat filter runs in the pipeline on every final
//! (`speech_core::hallucination`, with this engine's no-speech probability).
//!
//! Models: `EngineSpec::model` is a catalog id (`whisper/<id>`, or a bare `<id>` as the Whisper
//! plugin persists it), resolved with `model_store::installed_path` (the store installs and adopts
//! them in the legacy `<app_data>/whisper/` folder), falling back to a file already in that folder
//! that the store did not adopt (a custom or older conversion under a catalog name); or the
//! absolute path of a bring-your-own ggml `.bin`. `options.preset: "anime-whisper"` marks a BYO
//! file as Anime-Whisper (Japanese only, no prompt). The engine never downloads.
//!
//! `services/whisper.rs` keeps model management (catalog, downloads through the store, import) and
//! runs its legacy `start_recording` through a speech pipeline with [`WhisperEngine::shared`] over
//! the model it loaded.
//!
//! # Frozen signature (P2-S0)
//! `create(&EngineSpec, &EngineContext) -> Result<Box<dyn AsrEngine>, AsrError>` (called by
//! `engines::create`; blocking).

use std::ffi::{c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::Instant;

use serde::Deserialize;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use super::super::engine_api::{
    AsrEngine, AsrError, AsrResult, Canceller, DecodeOptions, EngineCaps, EngineContext, EngineInfo, EngineKind,
    EngineSpec, EngineStatus,
};
use super::super::events::codes;
use super::super::worker::scope;
use crate::services::model_store;
use crate::services::speech_core::SAMPLE_RATE;
use crate::services::whisper::{
    bring_your_own_missing, get_model_info, validate_whisper_bin_src, WhisperModelInfo, ANIME_WHISPER_ID,
    WHISPER_FOLDER,
};

/// Log target (shared with the Whisper plugin).
const LOG: &str = "sigil::stt::whisper";
/// Runtime label for status and logs.
pub(crate) const RUNTIME: &str = "whisper.cpp (whisper-rs 0.16)";
/// Longest prompt kept, in bytes (whisper.cpp keeps at most the last 223 tokens anyway).
pub(crate) const MAX_PROMPT_BYTES: usize = 1024;
/// Prompt tokens fed to the decoder (`n_text_ctx / 2 - 1`).
const MAX_PROMPT_TOKENS: usize = 223;
/// One whisper.cpp decode window (30 s at 16 kHz).
const WHISPER_WINDOW_SAMPLES: usize = 30 * SAMPLE_RATE as usize;
/// Default no-speech probability above which an utterance returns no text.
pub(crate) const DEFAULT_NO_SPEECH_THRESHOLD: f32 = 0.6;

/// The loaded model plus the one `WhisperState` reused for every utterance (created lazily; both
/// are replaced together on a model switch, so the old model's memory is freed at once).
pub(crate) struct LoadedModel {
    ctx: WhisperContext,
    state: Option<whisper_rs::WhisperState>,
    accepts_prompt: bool,
    /// `(prompt text, its tokens)` for this model's vocabulary.
    prompt_tokens: Option<(String, Vec<c_int>)>,
}

impl LoadedModel {
    pub(crate) fn new(ctx: WhisperContext, accepts_prompt: bool) -> Self {
        Self {
            ctx,
            state: None,
            accepts_prompt,
            prompt_tokens: None,
        }
    }
}

/// A model slot shared between an engine and its owner (the Whisper plugin's loaded model).
pub(crate) type SharedModel = Arc<Mutex<Option<LoadedModel>>>;

/// Load a ggml Whisper model on the CPU (blocking; seconds for large models).
pub(crate) fn load_context(path: &Path) -> Result<WhisperContext, String> {
    WhisperContext::new_with_params(path, WhisperContextParameters::default())
        .map_err(|e| format!("Failed to load context: {e}"))
}

/// Default inference threads: half the logical CPUs (about the physical cores), 1..=4 (VRChat is
/// CPU-bound).
pub(crate) fn default_threads() -> u32 {
    let logical = thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(4);
    (logical / 2).clamp(1, 4)
}

/// Threads for a request (`None` / 0: the default), clamped to the CPU and 16.
pub(crate) fn resolve_threads(requested: Option<u32>) -> c_int {
    let threads = requested.filter(|&n| n > 0).unwrap_or_else(default_threads);
    let available = thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(4);
    threads.clamp(1, available.max(1)).min(16) as c_int
}

/// No-speech threshold from a request (`None` / NaN: the default), clamped to 0.05..=1.
pub(crate) fn resolve_no_speech_threshold(value: Option<f32>) -> f32 {
    value
        .filter(|v| v.is_finite())
        .map_or(DEFAULT_NO_SPEECH_THRESHOLD, |v| v.clamp(0.05, 1.0))
}

/// The decoder prompt: the free-form prompt, then the hotwords as a comma-separated list; no NUL,
/// at most [`MAX_PROMPT_BYTES`] (cut on a char boundary).
pub(crate) fn compose_prompt(prompt: Option<&str>, hotwords: &[String]) -> String {
    let mut out = prompt.unwrap_or("").trim().to_string();
    let words: Vec<&str> = hotwords.iter().map(|w| w.trim()).filter(|w| !w.is_empty()).collect();
    if !words.is_empty() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&words.join(", "));
        out.push('.');
    }
    // NUL would abort tokenization.
    out.retain(|c| c != '\0');
    if out.len() > MAX_PROMPT_BYTES {
        let mut cut = MAX_PROMPT_BYTES;
        while !out.is_char_boundary(cut) {
            cut -= 1;
        }
        out.truncate(cut);
    }
    out
}

/// Whisper's per-pass token cap for an utterance of `samples` (16 kHz): generous for fast speech
/// (about 12 tokens/s + 32), far below a runaway repetition loop (224).
pub(crate) fn max_tokens_for(samples: usize) -> c_int {
    let secs = samples as f32 / SAMPLE_RATE as f32;
    ((secs * 12.0).ceil() as c_int + 32).clamp(48, 220)
}

/// Whisper's language codes (whisper.cpp / OpenAI tokenizer order; `yue` only in large-v3).
pub(crate) const WHISPER_LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi",
    "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml",
    "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs",
    "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am",
    "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

/// Sigil codes ([`lang::normalize`](crate::services::speech::lang::normalize)) that Whisper (and
/// the OpenAI transcription API) spell differently.
const SIGIL_TO_WHISPER: &[(&str, &str)] = &[
    // Norwegian Bokmål: Whisper's Norwegian is `no` (Nynorsk `nn` exists as is).
    ("nb", "no"),
    // Filipino / Tagalog.
    ("fil", "tl"),
    // Javanese: Whisper keeps the deprecated ISO code.
    ("jv", "jw"),
];

/// Whisper's code for a language tag (`auto`, empty or a language Whisper does not know: `None`,
/// the model detects). Cantonese (`yue`) exists only in large-v3 models; others get Chinese.
pub(crate) fn whisper_code(tag: &str, large_v3: bool) -> Option<&'static str> {
    let code = crate::services::speech::lang::normalize(tag)?;
    let code = SIGIL_TO_WHISPER
        .iter()
        .find(|(sigil, _)| *sigil == code)
        .map_or(code.as_str(), |(_, whisper)| whisper);
    let code = if code == "yue" && !large_v3 { "zh" } else { code };
    WHISPER_LANGUAGES.iter().copied().find(|w| *w == code)
}

/// Whisper language code for `configured` (`auto`, empty or unknown codes: detect); see
/// [`whisper_code`].
pub(crate) fn whisper_language(configured: &str, large_v3: bool) -> Option<&'static str> {
    whisper_code(configured, large_v3)
}

/// Abort flag handed to whisper.cpp: inference stops as soon as the engine's cancel generation
/// moves on (a stop, or a cancel from the two-pass queue).
struct AbortToken {
    generation: Arc<AtomicU64>,
    mine: u64,
}

impl AbortToken {
    fn aborted(&self) -> bool {
        self.generation.load(Ordering::Relaxed) != self.mine
    }
}

/// `ggml_abort_callback`: `user_data` is the decode's `AbortToken`.
unsafe extern "C" fn abort_trampoline(user_data: *mut c_void) -> bool {
    // SAFETY: set only by `decode`, which passes a pointer to an `AbortToken` that outlives the
    // `full()` call that invokes this callback.
    let token = unsafe { &*(user_data as *const AbortToken) };
    token.aborted()
}

/// OpenAI's no-speech probability: P(`<|nospeech|>`) at the `<|startoftranscript|>` position, from one
/// extra decoder step over the encoder output `full()` left in `wstate` (a single 30 s window for a
/// VAD utterance).
///
/// whisper.cpp's own per-segment value cannot be used: it reads row 0 of the logits buffer, but a
/// multi-token prompt decode (initial prompt, or the language/task tokens of every multilingual
/// model) writes only the last row, so row 0 holds stale logits of an earlier decode. Measured on
/// base.en: this probe gives 0.94 on silence and 0.0006 on speech with or without a prompt, while
/// whisper.cpp's value simply repeated the previous utterance's number.
fn sot_no_speech_prob(ctx: &WhisperContext, wstate: &mut whisper_rs::WhisperState, threads: c_int) -> Option<f32> {
    wstate.decode(&[ctx.token_sot()], 0, threads.max(1) as usize).ok()?;
    let logits = wstate.get_logits().ok()?;
    let nosp = usize::try_from(ctx.token_nosp()).ok()?;
    let target = *logits.get(nosp)?;
    let max = logits.iter().copied().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
    if !target.is_finite() || !max.is_finite() {
        return None;
    }
    let sum: f64 = logits.iter().filter(|v| v.is_finite()).map(|&v| f64::from(v - max).exp()).sum();
    (sum > 0.0).then(|| (f64::from(target - max).exp() / sum) as f32)
}

/// Result of one decode.
pub(crate) struct Decoded {
    /// Raw text (trimmed); empty when the no-speech probability is above the threshold.
    pub text: String,
    /// Language Whisper decoded (configured or detected).
    pub language: Option<&'static str>,
    /// No-speech probability ([`sot_no_speech_prob`]; 0 when unavailable).
    pub no_speech_prob: f32,
}

/// Settings of one decode.
struct DecodeRequest<'a> {
    threads: c_int,
    language: Option<&'static str>,
    prompt: &'a str,
    no_speech_threshold: f32,
}

/// Transcribe one 16 kHz mono utterance on `loaded`, reusing its `WhisperState`.
fn decode(loaded: &mut LoadedModel, req: &DecodeRequest<'_>, samples: &[f32], abort: &AbortToken) -> Result<Decoded, AsrError> {
    if samples.len() < SAMPLE_RATE as usize / 5 {
        // whisper.cpp needs >= 100 ms; the VAD never closes anything this short.
        return Ok(Decoded {
            text: String::new(),
            language: None,
            no_speech_prob: 1.0,
        });
    }
    let LoadedModel {
        ctx,
        state: wstate_slot,
        accepts_prompt,
        prompt_tokens,
    } = loaded;

    // Prompt tokens for this model's vocabulary, tokenized once per (model, prompt).
    let use_prompt = *accepts_prompt && !req.prompt.is_empty();
    if use_prompt && prompt_tokens.as_ref().is_none_or(|(text, _)| text != req.prompt) {
        // Every token covers at least one byte, so `len + 8` can never overflow whisper-rs' buffer.
        match ctx.tokenize(req.prompt, req.prompt.len() + 8) {
            Ok(mut tokens) => {
                if tokens.len() > MAX_PROMPT_TOKENS {
                    tokens.drain(..tokens.len() - MAX_PROMPT_TOKENS);
                }
                *prompt_tokens = Some((req.prompt.to_string(), tokens));
            }
            Err(e) => {
                tracing::warn!(target: LOG, "initial prompt ignored (tokenize failed: {e})");
                *prompt_tokens = Some((req.prompt.to_string(), Vec::new()));
            }
        }
    }

    if wstate_slot.is_none() {
        *wstate_slot = Some(
            ctx.create_state()
                .map_err(|e| AsrError::runtime(format!("Failed to create Whisper state: {e}")))?,
        );
    }
    let wstate = wstate_slot.as_mut().expect("state created above");

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(req.threads);
    params.set_translate(false);
    params.set_language(req.language);
    // Each utterance stands alone: carrying the previous text as context propagates hallucinations.
    params.set_no_context(true);
    // One decode window per utterance (utterances are <= 18 s by default, merged batches <= 28 s):
    // whisper.cpp would otherwise re-encode a short tail after the last timestamp, which is where
    // "you" / "Thank you." hallucinations come from, and the no-speech probe below must see the
    // utterance's own encoder output.
    params.set_single_segment(true);
    params.set_suppress_nst(true);
    params.set_suppress_blank(true);
    // Disables whisper.cpp's internal no-speech skip, which acts on its stale value (see
    // `sot_no_speech_prob`) and could discard real speech after a noisy utterance; the probe below
    // replaces it. Temperature fallback still runs on low average log-probability.
    params.set_no_speech_thold(1.0);
    params.set_max_tokens(max_tokens_for(samples.len()));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    if use_prompt {
        if let Some((_, tokens)) = prompt_tokens.as_ref().filter(|(_, t)| !t.is_empty()) {
            params.set_tokens(tokens);
        }
    }
    // SAFETY: `abort` outlives `full()` below, the only call that uses the callback.
    unsafe {
        params.set_abort_callback(Some(abort_trampoline));
        params.set_abort_callback_user_data(abort as *const AbortToken as *mut c_void);
    }

    if let Err(e) = wstate.full(params, samples) {
        // Never reuse a state whose decode failed or was aborted midway.
        *wstate_slot = None;
        return Err(if abort.aborted() {
            AsrError::cancelled()
        } else {
            AsrError::runtime(format!("Whisper inference failed: {e}"))
        });
    }

    // Concatenate raw bytes: a multi-byte character can be split across segments, and segment text
    // already carries its own leading space where the script uses spaces (none for CJK).
    let mut bytes = Vec::new();
    for segment in wstate.as_iter() {
        if let Ok(b) = segment.to_bytes() {
            bytes.extend_from_slice(b);
        }
    }
    let detected = whisper_rs::get_lang_str(wstate.full_lang_id_from_state());
    // After the segments are read: this extra decode reuses the state's decoder buffers. It sees the
    // last encoded window only, so it is meaningful for utterances that fit one 30 s window (every
    // VAD utterance and merged batch); longer ones are never dropped on it.
    let no_speech_prob = if samples.len() <= WHISPER_WINDOW_SAMPLES {
        sot_no_speech_prob(ctx, wstate, req.threads).unwrap_or(0.0)
    } else {
        0.0
    };
    if no_speech_prob > req.no_speech_threshold {
        // The utterance as a whole is judged non-speech (breath, keyboard, music the VAD let through).
        bytes.clear();
    }
    Ok(Decoded {
        text: String::from_utf8_lossy(&bytes).trim().to_string(),
        language: detected.or(req.language),
        no_speech_prob,
    })
}

/// Engine options (`EngineSpec::options`, camelCase; every field optional).
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct WhisperOptions {
    /// Utterances whose no-speech probability is above this return no text (default 0.6; 1
    /// disables).
    no_speech_threshold: Option<f32>,
    /// `"anime-whisper"`: a bring-your-own file is Anime-Whisper (Japanese only, no prompt).
    preset: Option<String>,
}

/// A resolved model file.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedModel {
    pub path: PathBuf,
    /// Catalog id (`whisper/<id>`) or the file name of a bring-your-own model.
    pub label: String,
    /// The Whisper plugin's preset, when the model is one of them.
    pub info: Option<WhisperModelInfo>,
}

/// `<app_data>/whisper/<file>`, when it exists.
fn legacy_file(filename: &str) -> Option<PathBuf> {
    let path = crate::services::diagnostics::app_data_dir()?.join(WHISPER_FOLDER).join(filename);
    path.is_file().then_some(path)
}

/// Catalog id of a Whisper model spec (`whisper/<id>` or a bare persisted id); `None` for paths.
pub(crate) fn catalog_id(model: &str) -> Option<String> {
    let m = model.trim();
    if m.is_empty() || Path::new(m).is_absolute() {
        return None;
    }
    let id = m.strip_prefix("whisper/").unwrap_or(m);
    let info = get_model_info(id).ok()?;
    (!info.bring_your_own).then(|| model_store::whisper_entry_id(info.id))
}

/// Find the file of `model` (see the module docs). Never downloads.
pub(crate) fn resolve_model(model: &str) -> Result<ResolvedModel, AsrError> {
    let m = model.trim();
    if m.is_empty() {
        return Err(AsrError::invalid_input("no Whisper model was given"));
    }
    let path = Path::new(m);
    if path.is_absolute() {
        if !path.is_file() {
            return Err(AsrError::invalid_input(format!(
                "the Whisper model file '{}' does not exist",
                path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()
            )));
        }
        return Ok(ResolvedModel {
            path: path.to_path_buf(),
            label: path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            info: None,
        });
    }
    let id = m.strip_prefix("whisper/").unwrap_or(m);
    let info = get_model_info(id).map_err(AsrError::invalid_input)?;
    if info.bring_your_own {
        return legacy_file(info.filename)
            .map(|path| ResolvedModel {
                path,
                label: info.id.to_string(),
                info: Some(info),
            })
            .ok_or_else(|| AsrError::not_installed(bring_your_own_missing(&info)));
    }
    let entry = model_store::whisper_entry_id(info.id);
    if let Some(path) = model_store::installed_path(&entry)
        .map(|root| root.join(info.filename))
        .filter(|p| p.is_file())
        .or_else(|| legacy_file(info.filename))
    {
        return Ok(ResolvedModel {
            path,
            label: entry,
            info: Some(info),
        });
    }
    Err(AsrError::not_installed(entry))
}

/// Whether a model id / file name is an English-only model (`*.en`, `*.en-q5_1`, ...).
fn is_english_only(label: &str) -> bool {
    let name = label.rsplit('/').next().unwrap_or(label).to_ascii_lowercase();
    name.contains(".en")
}

/// The [`EngineInfo`] of a Whisper model.
pub(crate) fn engine_info(label: &str, accepts_prompt: bool, forced_language: Option<&str>) -> EngineInfo {
    let languages = match forced_language {
        Some(l) => vec![l.to_string()],
        None if is_english_only(label) => vec!["en".to_string()],
        None => Vec::new(),
    };
    EngineInfo {
        kind: EngineKind::Whisper,
        model: label.to_string(),
        runtime: RUNTIME.to_string(),
        device: "cpu".to_string(),
        caps: EngineCaps {
            streaming: false,
            utterance: true,
            detects_language: languages.is_empty(),
            languages,
            needs_language: false,
            hotwords: accepts_prompt,
            prompt: accepts_prompt,
            tags: false,
        },
        vram_mb: 0,
        fallback_reason: None,
    }
}

/// The Whisper engine.
pub(crate) struct WhisperEngine {
    model: SharedModel,
    info: EngineInfo,
    threads: c_int,
    no_speech_threshold: f32,
    /// Single-language fine-tunes (Anime-Whisper: Japanese) override the requested language.
    forced_language: Option<&'static str>,
    large_v3: bool,
    cancel: Arc<AtomicU64>,
}

impl WhisperEngine {
    /// An engine over a model loaded by someone else (the Whisper plugin's shared slot). The model
    /// may be replaced in the slot between utterances.
    pub(crate) fn shared(
        model: SharedModel,
        info: EngineInfo,
        threads: Option<u32>,
        no_speech_threshold: Option<f32>,
        forced_language: Option<&'static str>,
    ) -> Self {
        let large_v3 = info.model.contains("large-v3");
        Self {
            model,
            info,
            threads: resolve_threads(threads),
            no_speech_threshold: resolve_no_speech_threshold(no_speech_threshold),
            forced_language,
            large_v3,
            cancel: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl AsrEngine for WhisperEngine {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn transcribe(&mut self, audio: &[f32], options: &DecodeOptions) -> Result<AsrResult, AsrError> {
        let started = Instant::now();
        let abort = AbortToken {
            generation: Arc::clone(&self.cancel),
            mine: self.cancel.load(Ordering::SeqCst),
        };
        // After the snapshot: a stop that came before it is seen here, any later one aborts the
        // decode through the generation.
        if scope::abandoned() {
            return Err(AsrError::cancelled());
        }
        let language = self
            .forced_language
            .or_else(|| options.language.as_deref().and_then(|l| whisper_language(l, self.large_v3)));
        let prompt = compose_prompt(options.prompt.as_deref(), &options.hotwords);
        let request = DecodeRequest {
            threads: self.threads,
            language,
            prompt: &prompt,
            no_speech_threshold: self.no_speech_threshold,
        };
        let decoded = {
            let mut guard = self.model.lock().unwrap_or_else(PoisonError::into_inner);
            let loaded = guard
                .as_mut()
                .ok_or_else(|| AsrError::not_installed("the Whisper model is not loaded"))?;
            decode(loaded, &request, audio, &abort)?
        };
        Ok(AsrResult {
            text: decoded.text,
            lang: decoded.language.map(str::to_string),
            tags: Vec::new(),
            no_speech_prob: Some(decoded.no_speech_prob),
            confidence: None,
            decode_ms: started.elapsed().as_millis().min(u128::from(u32::MAX)) as u32,
        })
    }

    fn canceller(&self) -> Option<Canceller> {
        let generation = Arc::clone(&self.cancel);
        Some(Canceller::new(move || {
            generation.fetch_add(1, Ordering::SeqCst);
        }))
    }
}

/// Create the engine (blocking: loads the model).
pub fn create(spec: &EngineSpec, ctx: &EngineContext) -> Result<Box<dyn AsrEngine>, AsrError> {
    let options: WhisperOptions = if spec.options.is_null() {
        WhisperOptions::default()
    } else {
        serde_json::from_value(spec.options.clone())
            .map_err(|e| AsrError::invalid_input(format!("invalid Whisper engine options: {e}")))?
    };
    let resolved = resolve_model(&spec.model)?;
    let anime = options.preset.as_deref().is_some_and(|p| p.trim().eq_ignore_ascii_case(ANIME_WHISPER_ID))
        || resolved.info.as_ref().is_some_and(|i| i.id == ANIME_WHISPER_ID);
    let (accepts_prompt, forced_language) = match (&resolved.info, anime) {
        (_, true) => (false, Some("ja")),
        (Some(info), false) => (info.accepts_prompt, info.language),
        (None, false) => (true, None),
    };
    validate_whisper_bin_src(&resolved.path).map_err(AsrError::invalid_input)?;
    ctx.status.report(EngineStatus {
        code: codes::LOADING_MODEL.to_string(),
        detail: format!("loading Whisper {}", resolved.label),
        model_id: resolved.label.contains('/').then(|| resolved.label.clone()),
        ..EngineStatus::default()
    });
    let started = Instant::now();
    let context = load_context(&resolved.path).map_err(AsrError::runtime)?;
    tracing::info!(
        target: LOG,
        "loaded {} in {} ms",
        resolved.label,
        started.elapsed().as_millis()
    );
    let info = engine_info(&resolved.label, accepts_prompt, forced_language);
    let model: SharedModel = Arc::new(Mutex::new(Some(LoadedModel::new(context, accepts_prompt))));
    Ok(Box::new(WhisperEngine::shared(
        model,
        info,
        spec.threads,
        options.no_speech_threshold,
        forced_language,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::speech_core::hallucination::clean_transcript;
    use std::time::Duration;

    #[test]
    fn prompt_composition_matches_the_legacy_decode_options() {
        assert_eq!(compose_prompt(None, &[]), "");
        assert_eq!(
            compose_prompt(Some("  Chat in VRChat. "), &["Sigil".into(), " ".into(), "Kokoro".into()]),
            "Chat in VRChat. Sigil, Kokoro."
        );
        assert_eq!(compose_prompt(None, &["a".into()]), "a.");
        let long = compose_prompt(Some(&format!("a\0b{}", "語".repeat(600))), &[]);
        assert!(!long.contains('\0') && long.starts_with("ab"));
        assert!(long.len() <= MAX_PROMPT_BYTES);
    }

    #[test]
    fn threads_and_thresholds_resolve_to_safe_values() {
        assert!((1..=4).contains(&resolve_threads(None)));
        assert_eq!(resolve_threads(Some(1)), 1);
        assert!(resolve_threads(Some(99)) <= 16);
        assert_eq!(resolve_no_speech_threshold(None), 0.6);
        assert_eq!(resolve_no_speech_threshold(Some(f32::NAN)), 0.6);
        assert_eq!(resolve_no_speech_threshold(Some(0.0)), 0.05);
        assert_eq!(resolve_no_speech_threshold(Some(3.0)), 1.0);
    }

    #[test]
    fn token_cap_scales_with_utterance_length() {
        assert_eq!(max_tokens_for(8_000), 48);
        assert_eq!(max_tokens_for(16_000 * 5), 92);
        assert_eq!(max_tokens_for(16_000 * 18), 220);
        assert_eq!(max_tokens_for(16_000 * 60), 220);
    }

    #[test]
    fn whisper_language_codes_are_validated() {
        assert_eq!(whisper_language("auto", false), None);
        assert_eq!(whisper_language("", false), None);
        assert_eq!(whisper_language("en", false), Some("en"));
        assert_eq!(whisper_language("zh-TW", false), Some("zh"));
        assert_eq!(whisper_language("pt_BR", false), Some("pt"));
        assert_eq!(whisper_language("xx", false), None, "unknown codes auto-detect instead of failing");
        assert_eq!(whisper_language("yue", false), Some("zh"), "Cantonese needs a large-v3 model");
        assert_eq!(whisper_language("yue", true), Some("yue"));
        // Sigil's codes that Whisper spells differently (stt_data.ts lists nb-NO, fil-PH, jv-ID).
        assert_eq!(whisper_language("nb-NO", false), Some("no"));
        assert_eq!(whisper_language("nn", false), Some("nn"));
        assert_eq!(whisper_language("fil-PH", false), Some("tl"));
        assert_eq!(whisper_language("tgl", false), Some("tl"));
        assert_eq!(whisper_language("jv-ID", false), Some("jw"));
        assert_eq!(whisper_language("jw", false), Some("jw"));
        assert_eq!(whisper_language("iw", false), Some("he"));
        assert_eq!(whisper_language("haw", false), Some("haw"));
        assert_eq!(whisper_language("zh-yue", true), Some("yue"));
    }

    #[test]
    fn the_language_table_matches_whisper_cpp() {
        assert_eq!(WHISPER_LANGUAGES.len(), 100);
        for code in WHISPER_LANGUAGES {
            let id = whisper_rs::get_lang_id(code).unwrap_or_else(|| panic!("whisper.cpp does not know '{code}'"));
            assert_eq!(whisper_rs::get_lang_str(id), Some(*code));
        }
        for (sigil, whisper) in SIGIL_TO_WHISPER {
            assert!(WHISPER_LANGUAGES.contains(whisper), "{sigil} -> {whisper}");
            assert_eq!(crate::services::speech::lang::normalize(sigil).as_deref(), Some(*sigil), "a Sigil code");
        }
    }

    #[test]
    fn a_decode_for_an_ended_session_is_never_started() {
        let model: SharedModel = Arc::new(Mutex::new(None));
        let mut e = WhisperEngine::shared(model, engine_info("whisper/base", true, None), Some(1), None, None);
        let ended: scope::Live = Arc::new(|| false);
        let (r, _) = scope::run(None, Some(ended), || e.transcribe(&[0.0; 16_000], &DecodeOptions::default()));
        assert!(r.unwrap_err().is_cancelled(), "cancelled before the model is even looked at");
        let live: scope::Live = Arc::new(|| true);
        let (r, _) = scope::run(None, Some(live), || e.transcribe(&[0.0; 16_000], &DecodeOptions::default()));
        assert!(!r.unwrap_err().is_cancelled());
    }

    #[test]
    fn model_specs_resolve_to_catalog_ids_or_errors() {
        assert_eq!(catalog_id("base.en").as_deref(), Some("whisper/base.en"));
        assert_eq!(catalog_id("whisper/small-q5_1").as_deref(), Some("whisper/small-q5_1"));
        assert_eq!(catalog_id("medium-q5_1").as_deref(), Some("whisper/medium-q5_0"), "legacy alias");
        assert_eq!(catalog_id(ANIME_WHISPER_ID), None, "bring your own");
        assert_eq!(catalog_id("huge-v9"), None);
        let abs = std::env::temp_dir().join("sigil-no-such-model.bin");
        assert_eq!(catalog_id(abs.to_str().unwrap()), None);
        assert_eq!(resolve_model("").unwrap_err().kind, crate::services::speech::engine_api::AsrErrorKind::InvalidInput);
        assert!(resolve_model("huge-v9").unwrap_err().message.contains("huge-v9"));
        let err = resolve_model(abs.to_str().unwrap()).unwrap_err();
        assert!(err.message.contains("sigil-no-such-model.bin"), "{err}");
        assert!(!err.message.contains(&std::env::temp_dir().to_string_lossy().to_string()), "no full paths in messages");
    }

    #[test]
    fn engine_info_reflects_model_languages_and_prompts() {
        let en = engine_info("whisper/base.en-q5_1", true, None);
        assert_eq!(en.caps.languages, vec!["en".to_string()]);
        assert!(!en.caps.detects_language && en.caps.prompt && en.caps.utterance && !en.caps.streaming);
        let multi = engine_info("whisper/small-q5_1", true, None);
        assert!(multi.caps.languages.is_empty() && multi.caps.detects_language);
        let anime = engine_info("anime-whisper", false, Some("ja"));
        assert_eq!(anime.caps.languages, vec!["ja".to_string()]);
        assert!(!anime.caps.prompt && !anime.caps.hotwords);
        assert_eq!(anime.device, "cpu");
    }

    #[test]
    fn an_empty_slot_is_not_installed_and_cancel_is_generation_based() {
        let model: SharedModel = Arc::new(Mutex::new(None));
        let mut e = WhisperEngine::shared(model, engine_info("whisper/base", true, None), Some(1), None, None);
        let err = e.transcribe(&[0.0; 16_000], &DecodeOptions::default()).unwrap_err();
        assert_eq!(err.kind, crate::services::speech::engine_api::AsrErrorKind::NotInstalled);
        // Too-short audio never reaches the model.
        let r = e.transcribe(&[0.0; 100], &DecodeOptions::default());
        assert!(r.is_err(), "the slot is checked first");
        let before = e.cancel.load(Ordering::SeqCst);
        e.canceller().unwrap().cancel();
        assert_eq!(e.cancel.load(Ordering::SeqCst), before + 1);
    }

    fn real_model() -> Option<PathBuf> {
        let dir = PathBuf::from(std::env::var("SIGIL_TEST_MODELS_DIR").ok()?);
        ["ggml-tiny.en.bin", "ggml-base.en.bin", "ggml-base.bin", "ggml-tiny.bin"]
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.exists())
    }

    /// Real-model check (manual): `SIGIL_TEST_MODELS_DIR` must hold `ggml-tiny.en.bin`,
    /// `ggml-base.en.bin`, `ggml-base.bin` or `ggml-tiny.bin`. Run with
    /// `cargo test real_model -- --ignored`.
    #[test]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR with a ggml Whisper model"]
    fn real_model_transcribes_speech_and_filters_silence() {
        let Some(path) = real_model() else { return };
        let spec = EngineSpec {
            options: serde_json::json!({ "noSpeechThreshold": 0.6 }),
            threads: Some(4),
            ..EngineSpec::new(EngineKind::Whisper, path.to_string_lossy())
        };
        let mut engine = create(&spec, &EngineContext::headless()).unwrap();
        let options = DecodeOptions {
            language: Some("en".into()),
            hotwords: vec!["weather".into()],
            prompt: None,
        };
        let speech = crate::services::speech_core::silero::tests::speech_fixture();
        let r = engine.transcribe(&speech, &options).unwrap();
        let text = r.text.to_lowercase();
        assert!(text.contains("weather"), "{text}");
        assert!(r.no_speech_prob.unwrap() < 0.1, "speech no-speech prob {:?}", r.no_speech_prob);
        // The same WhisperState is reused for the next utterance.
        let again = engine.transcribe(&speech, &options).unwrap();
        assert_eq!(again.text.to_lowercase(), text);

        // Silence or noise that slipped past the VAD produces no caption, even with a prompt
        // (whisper.cpp alone would emit "you" / "Thank you." here on multilingual models).
        let silence = vec![0.0f32; 16_000 * 3];
        let s = engine.transcribe(&silence, &options).unwrap();
        assert!(s.no_speech_prob.unwrap() > 0.6, "silence no-speech prob {:?}", s.no_speech_prob);
        assert_eq!(clean_transcript(&s.text, Some("en"), s.no_speech_prob.unwrap()), None);
        let mut seed = 7u32;
        let noise: Vec<f32> = (0..16_000 * 3)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed >> 8) as f32 / 16_777_216.0 - 0.5) * 0.01
            })
            .collect();
        let n = engine.transcribe(&noise, &options).unwrap();
        assert_eq!(clean_transcript(&n.text, Some("en"), n.no_speech_prob.unwrap()), None);
        // Speech right after noise is still transcribed (no stale no-speech value carries over).
        assert!(!engine.transcribe(&speech, &options).unwrap().text.is_empty());

        // A cancel aborts inference promptly; the next decode works again.
        let canceller = engine.canceller().unwrap();
        let long: Vec<f32> = speech.iter().cycle().take(16_000 * 15).copied().collect();
        let cancel_later = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            canceller.cancel();
        });
        let started = Instant::now();
        let result = engine.transcribe(&long, &options);
        cancel_later.join().unwrap();
        if let Err(e) = &result {
            assert!(e.is_cancelled(), "{e}");
        }
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(!engine.transcribe(&speech, &options).unwrap().text.is_empty());
    }
}
