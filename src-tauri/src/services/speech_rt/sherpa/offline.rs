//! Non-streaming ASR: `SherpaOnnxOfflineRecognizer` (SenseVoice, Qwen3-ASR, Parakeet / NeMo
//! transducers, Moonshine, Whisper, Canary, Cohere Transcribe, CTC families).
//!
//! [`OfflineRecognizer::transcribe`] runs one whole utterance (the P2 pipeline cuts utterances
//! with Sigil's Silero segmenter). Every native call is serialized per recognizer, so the type is
//! `Send + Sync`; decode concurrency comes from ONNX Runtime's intra-op threads.

use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::bindings::{SherpaOnnxOfflineRecognizer, SherpaOnnxOfflineRecognizerConfig, SherpaOnnxOfflineRecognizerResult, SherpaOnnxOfflineStream};
use super::lib::{
    c_float, floats, opt_str, owned_str, rate_i32, require_dir, require_files, strings, threads, CStrArena, CallLock, SherpaRuntime,
    CPU_PROVIDER,
};
use super::online::{default_feature_dim, default_hotwords_score, default_sample_rate, Decoding};

/// Model files and options of a non-streaming recognizer, by sherpa family.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum OfflineModel {
    /// SenseVoiceSmall (zh/en/ja/ko/yue + emotion and audio-event tags).
    SenseVoice {
        model: PathBuf,
        /// `auto`, `zh`, `en`, `ja`, `ko` or `yue` (default `auto`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        language: Option<String>,
        /// Inverse text normalization (numbers, punctuation).
        #[serde(default = "yes")]
        use_itn: bool,
    },
    /// Transducer (zipformer, NeMo Parakeet TDT: set `model_type` = `nemo_transducer`).
    Transducer {
        encoder: PathBuf,
        decoder: PathBuf,
        joiner: PathBuf,
    },
    Paraformer {
        model: PathBuf,
    },
    NemoCtc {
        model: PathBuf,
    },
    Whisper {
        encoder: PathBuf,
        decoder: PathBuf,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        language: Option<String>,
        /// `transcribe` (default) or `translate`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        task: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tail_paddings: Option<i32>,
    },
    /// Moonshine v1 (preprocessor + encoder + uncached/cached decoders) or v2 (encoder + merged
    /// decoder).
    Moonshine {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        preprocessor: Option<PathBuf>,
        encoder: PathBuf,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        uncached_decoder: Option<PathBuf>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cached_decoder: Option<PathBuf>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        merged_decoder: Option<PathBuf>,
    },
    /// Qwen3-ASR (conv frontend + encoder + KV-cache decoder + tokenizer folder). Detects the
    /// language itself.
    Qwen3Asr {
        conv_frontend: PathBuf,
        encoder: PathBuf,
        decoder: PathBuf,
        /// Folder with `vocab.json` / `merges.txt`.
        tokenizer: PathBuf,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_total_len: Option<i32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_new_tokens: Option<i32>,
        /// Comma-separated hotwords.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        hotwords: Option<String>,
    },
    Canary {
        encoder: PathBuf,
        decoder: PathBuf,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        src_lang: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tgt_lang: Option<String>,
        #[serde(default = "yes")]
        use_pnc: bool,
    },
    CohereTranscribe {
        encoder: PathBuf,
        decoder: PathBuf,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        language: Option<String>,
        #[serde(default = "yes")]
        use_punct: bool,
        #[serde(default = "yes")]
        use_itn: bool,
    },
    FireRedAsr {
        encoder: PathBuf,
        decoder: PathBuf,
    },
    FireRedAsrCtc {
        model: PathBuf,
    },
    Dolphin {
        model: PathBuf,
    },
    ZipformerCtc {
        model: PathBuf,
    },
    WenetCtc {
        model: PathBuf,
    },
    Omnilingual {
        model: PathBuf,
    },
}

fn yes() -> bool {
    true
}

impl OfflineModel {
    pub fn family(&self) -> &'static str {
        match self {
            OfflineModel::SenseVoice { .. } => "sense_voice",
            OfflineModel::Transducer { .. } => "transducer",
            OfflineModel::Paraformer { .. } => "paraformer",
            OfflineModel::NemoCtc { .. } => "nemo_ctc",
            OfflineModel::Whisper { .. } => "whisper",
            OfflineModel::Moonshine { .. } => "moonshine",
            OfflineModel::Qwen3Asr { .. } => "qwen3_asr",
            OfflineModel::Canary { .. } => "canary",
            OfflineModel::CohereTranscribe { .. } => "cohere_transcribe",
            OfflineModel::FireRedAsr { .. } => "fire_red_asr",
            OfflineModel::FireRedAsrCtc { .. } => "fire_red_asr_ctc",
            OfflineModel::Dolphin { .. } => "dolphin",
            OfflineModel::ZipformerCtc { .. } => "zipformer_ctc",
            OfflineModel::WenetCtc { .. } => "wenet_ctc",
            OfflineModel::Omnilingual { .. } => "omnilingual",
        }
    }

    /// Whether the family reads a `tokens.txt` (Qwen3-ASR uses its tokenizer folder instead).
    pub fn needs_tokens(&self) -> bool {
        !matches!(self, OfflineModel::Qwen3Asr { .. })
    }

    fn files(&self) -> Vec<&Path> {
        let mut v: Vec<&PathBuf> = match self {
            OfflineModel::SenseVoice { model, .. }
            | OfflineModel::Paraformer { model }
            | OfflineModel::NemoCtc { model }
            | OfflineModel::FireRedAsrCtc { model }
            | OfflineModel::Dolphin { model }
            | OfflineModel::ZipformerCtc { model }
            | OfflineModel::WenetCtc { model }
            | OfflineModel::Omnilingual { model } => vec![model],
            OfflineModel::Transducer { encoder, decoder, joiner } => vec![encoder, decoder, joiner],
            OfflineModel::Whisper { encoder, decoder, .. }
            | OfflineModel::Canary { encoder, decoder, .. }
            | OfflineModel::CohereTranscribe { encoder, decoder, .. }
            | OfflineModel::FireRedAsr { encoder, decoder } => vec![encoder, decoder],
            OfflineModel::Moonshine {
                preprocessor,
                encoder,
                uncached_decoder,
                cached_decoder,
                merged_decoder,
            } => {
                let mut v = vec![encoder];
                v.extend(
                    preprocessor
                        .iter()
                        .chain(uncached_decoder)
                        .chain(cached_decoder)
                        .chain(merged_decoder),
                );
                v
            }
            OfflineModel::Qwen3Asr {
                conv_frontend,
                encoder,
                decoder,
                ..
            } => vec![conv_frontend, encoder, decoder],
        };
        v.dedup();
        v.into_iter().map(PathBuf::as_path).collect()
    }
}

/// Everything needed to create an [`OfflineRecognizer`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OfflineRecognizerConfig {
    pub model: OfflineModel,
    /// `tokens.txt` (all families but Qwen3-ASR).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<PathBuf>,
    /// Explicit sherpa model type, e.g. `nemo_transducer` for Parakeet TDT.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modeling_unit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bpe_vocab: Option<PathBuf>,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_feature_dim")]
    pub feature_dim: u32,
    /// ONNX Runtime threads (default 2, at most 4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_threads: Option<u32>,
    #[serde(default)]
    pub decoding: Decoding,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hotwords_file: Option<PathBuf>,
    #[serde(default = "default_hotwords_score")]
    pub hotwords_score: f32,
    #[serde(default)]
    pub blank_penalty: f32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rule_fsts: Vec<PathBuf>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rule_fars: Vec<PathBuf>,
}

impl OfflineRecognizerConfig {
    /// A config with defaults for everything but the model files.
    pub fn new(model: OfflineModel, tokens: Option<PathBuf>) -> Self {
        Self {
            model,
            tokens,
            model_type: None,
            modeling_unit: None,
            bpe_vocab: None,
            sample_rate: default_sample_rate(),
            feature_dim: default_feature_dim(),
            num_threads: None,
            decoding: Decoding::default(),
            hotwords_file: None,
            hotwords_score: default_hotwords_score(),
            blank_penalty: 0.0,
            rule_fsts: Vec::new(),
            rule_fars: Vec::new(),
        }
    }

    /// Check that sherpa supports the option combination and every referenced file exists (before
    /// any native call: sherpa answers unsupported combinations with `exit(-1)`).
    pub fn validate(&self) -> Result<(), String> {
        let what = format!("sherpa {} model", self.model.family());
        self.check_decoding(&what)?;
        let mut files = self.model.files();
        match (&self.tokens, self.model.needs_tokens()) {
            (Some(t), _) => files.push(t),
            (None, true) => return Err(format!("{what}: tokens.txt is required")),
            (None, false) => {}
        }
        files.extend(self.bpe_vocab.as_deref());
        files.extend(self.hotwords_file.as_deref());
        files.extend(self.rule_fsts.iter().map(PathBuf::as_path));
        files.extend(self.rule_fars.iter().map(PathBuf::as_path));
        require_files(&what, files)?;
        if let OfflineModel::Qwen3Asr { tokenizer, .. } = &self.model {
            require_dir(&what, tokenizer)?;
        }
        if self.sample_rate == 0 || self.feature_dim == 0 {
            return Err(format!("{what}: sample rate and feature dimension must be positive"));
        }
        Ok(())
    }

    /// sherpa-onnx 1.13.8 runs modified beam search (and so hotwords) only on transducers (icefall
    /// / k2 and NeMo Parakeet). Every other family's constructor calls `exit(-1)` on
    /// `modified_beam_search` (SenseVoice: "Only greedy_search is supported"), and the default
    /// `CreateStream(hotwords)` does too ("Only transducer models support contextual biasing").
    fn check_decoding(&self, what: &str) -> Result<(), String> {
        let qwen3_hint = if matches!(self.model, OfflineModel::Qwen3Asr { .. }) {
            " (Qwen3-ASR takes hotwords in its model config instead)"
        } else {
            ""
        };
        if !self.decoding.is_beam_search() {
            if self.hotwords_file.is_some() {
                return Err(format!("{what}: a hotwords file needs modified beam search{qwen3_hint}"));
            }
            return Ok(());
        }
        if !matches!(self.model, OfflineModel::Transducer { .. }) {
            return Err(format!(
                "{what}: modified beam search and hotwords need a transducer model; the {} family decodes with greedy \
                 search only{qwen3_hint}",
                self.model.family()
            ));
        }
        if let Some(unit) = self.modeling_unit.as_deref().filter(|u| u.contains("bpe")) {
            if self.bpe_vocab.is_none() {
                return Err(format!("{what}: modeling unit {unit} needs bpe_vocab for hotwords"));
            }
        }
        Ok(())
    }

    /// Whether streams of this (validated) config may carry per-utterance hotwords: a transducer
    /// decoded with modified beam search.
    pub fn supports_hotwords(&self) -> bool {
        matches!(self.model, OfflineModel::Transducer { .. }) && self.decoding.is_beam_search()
    }

    /// `Err` (with the reason) unless [`supports_hotwords`](Self::supports_hotwords).
    pub(crate) fn require_hotwords_support(&self) -> Result<(), String> {
        if self.supports_hotwords() {
            return Ok(());
        }
        let family = self.model.family();
        Err(match self.model {
            OfflineModel::Qwen3Asr { .. } => {
                "Qwen3-ASR takes hotwords only in its model config (`hotwords` of the qwen3_asr model), not per utterance".to_string()
            }
            OfflineModel::Transducer { .. } => format!("the {family} model takes hotwords only when it decodes with modified beam search"),
            _ => format!("the {family} model does not take hotwords (only transducer models decoded with modified beam search do)"),
        })
    }

    /// The C config (pointers into `arena`).
    pub(crate) fn to_raw(&self, arena: &mut CStrArena) -> Result<SherpaOnnxOfflineRecognizerConfig, String> {
        let mut c = SherpaOnnxOfflineRecognizerConfig::default();
        c.feat_config.sample_rate = self.sample_rate as i32;
        c.feat_config.feature_dim = self.feature_dim as i32;
        let m = &mut c.model_config;
        match &self.model {
            OfflineModel::SenseVoice { model, language, use_itn } => {
                m.sense_voice.model = arena.path(model)?;
                m.sense_voice.language = arena.str(language.as_deref().unwrap_or("auto"))?;
                m.sense_voice.use_itn = i32::from(*use_itn);
            }
            OfflineModel::Transducer { encoder, decoder, joiner } => {
                m.transducer.encoder = arena.path(encoder)?;
                m.transducer.decoder = arena.path(decoder)?;
                m.transducer.joiner = arena.path(joiner)?;
            }
            OfflineModel::Paraformer { model } => m.paraformer.model = arena.path(model)?,
            OfflineModel::NemoCtc { model } => m.nemo_ctc.model = arena.path(model)?,
            OfflineModel::Whisper {
                encoder,
                decoder,
                language,
                task,
                tail_paddings,
            } => {
                m.whisper.encoder = arena.path(encoder)?;
                m.whisper.decoder = arena.path(decoder)?;
                m.whisper.language = arena.opt(language.as_deref())?;
                m.whisper.task = arena.str(task.as_deref().unwrap_or("transcribe"))?;
                m.whisper.tail_paddings = tail_paddings.unwrap_or(-1);
            }
            OfflineModel::Moonshine {
                preprocessor,
                encoder,
                uncached_decoder,
                cached_decoder,
                merged_decoder,
            } => {
                m.moonshine.preprocessor = arena.opt_path(preprocessor.as_deref())?;
                m.moonshine.encoder = arena.path(encoder)?;
                m.moonshine.uncached_decoder = arena.opt_path(uncached_decoder.as_deref())?;
                m.moonshine.cached_decoder = arena.opt_path(cached_decoder.as_deref())?;
                m.moonshine.merged_decoder = arena.opt_path(merged_decoder.as_deref())?;
            }
            OfflineModel::Qwen3Asr {
                conv_frontend,
                encoder,
                decoder,
                tokenizer,
                max_total_len,
                max_new_tokens,
                hotwords,
            } => {
                let q = &mut m.qwen3_asr;
                q.conv_frontend = arena.path(conv_frontend)?;
                q.encoder = arena.path(encoder)?;
                q.decoder = arena.path(decoder)?;
                q.tokenizer = arena.path(tokenizer)?;
                // 0 lets sherpa use the model's defaults.
                q.max_total_len = max_total_len.unwrap_or(0);
                q.max_new_tokens = max_new_tokens.unwrap_or(0);
                // Effectively greedy and reproducible (captions must be deterministic). sherpa
                // replaces 0 with its own defaults (1e-6 / 0.8 / 42), so the values are explicit.
                q.temperature = 1e-6;
                q.top_p = 1.0;
                q.seed = 42;
                q.hotwords = arena.opt(hotwords.as_deref())?;
            }
            OfflineModel::Canary {
                encoder,
                decoder,
                src_lang,
                tgt_lang,
                use_pnc,
            } => {
                m.canary.encoder = arena.path(encoder)?;
                m.canary.decoder = arena.path(decoder)?;
                m.canary.src_lang = arena.opt(src_lang.as_deref())?;
                m.canary.tgt_lang = arena.opt(tgt_lang.as_deref())?;
                m.canary.use_pnc = i32::from(*use_pnc);
            }
            OfflineModel::CohereTranscribe {
                encoder,
                decoder,
                language,
                use_punct,
                use_itn,
            } => {
                m.cohere_transcribe.encoder = arena.path(encoder)?;
                m.cohere_transcribe.decoder = arena.path(decoder)?;
                m.cohere_transcribe.language = arena.opt(language.as_deref())?;
                m.cohere_transcribe.use_punct = i32::from(*use_punct);
                m.cohere_transcribe.use_itn = i32::from(*use_itn);
            }
            OfflineModel::FireRedAsr { encoder, decoder } => {
                m.fire_red_asr.encoder = arena.path(encoder)?;
                m.fire_red_asr.decoder = arena.path(decoder)?;
            }
            OfflineModel::FireRedAsrCtc { model } => m.fire_red_asr_ctc.model = arena.path(model)?,
            OfflineModel::Dolphin { model } => m.dolphin.model = arena.path(model)?,
            OfflineModel::ZipformerCtc { model } => m.zipformer_ctc.model = arena.path(model)?,
            OfflineModel::WenetCtc { model } => m.wenet_ctc.model = arena.path(model)?,
            OfflineModel::Omnilingual { model } => m.omnilingual.model = arena.path(model)?,
        }
        m.tokens = arena.opt_path(self.tokens.as_deref())?;
        m.num_threads = threads(self.num_threads);
        m.debug = 0;
        m.provider = CPU_PROVIDER.as_ptr();
        m.model_type = arena.opt(self.model_type.as_deref())?;
        m.modeling_unit = arena.opt(self.modeling_unit.as_deref())?;
        m.bpe_vocab = arena.opt_path(self.bpe_vocab.as_deref())?;
        c.decoding_method = arena.str(self.decoding.method())?;
        c.max_active_paths = self.decoding.max_active_paths();
        c.hotwords_file = arena.opt_path(self.hotwords_file.as_deref())?;
        c.hotwords_score = c_float(self.hotwords_score, default_hotwords_score());
        c.blank_penalty = if self.blank_penalty.is_finite() { self.blank_penalty } else { 0.0 };
        c.rule_fsts = arena.path_list(&self.rule_fsts)?;
        c.rule_fars = arena.path_list(&self.rule_fars)?;
        Ok(c)
    }
}

/// Utterances shorter than this return an empty result without calling the model.
pub const MIN_UTTERANCE_MS: u64 = 100;

/// Whether `samples` at `sample_rate` last at least [`MIN_UTTERANCE_MS`] (below that, some model
/// families throw inside ONNX Runtime on zero feature frames, and a C++ exception cannot cross
/// sherpa's C API).
pub(crate) fn long_enough(samples: usize, sample_rate: u32) -> bool {
    sample_rate > 0 && (samples as u64).saturating_mul(1000) >= u64::from(sample_rate) * MIN_UTTERANCE_MS
}

/// Whether an offline stream with this accepted audio (`(samples, sample rate)`) may run the model.
pub(crate) fn should_decode(accepted: Option<(usize, u32)>) -> bool {
    accepted.is_some_and(|(samples, rate)| long_enough(samples, rate))
}

/// Per-call options of [`OfflineRecognizer::transcribe`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeOptions {
    /// Language hint, passed as the per-stream `language` option. Only families that read that
    /// option use it; SenseVoice does not (measured, ADR 0002): its language is part of the model
    /// config, so force one with a recognizer created for it. `None` keeps the model config's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Per-utterance hotwords. Only a transducer decoded with modified beam search takes them
    /// ([`OfflineRecognizerConfig::supports_hotwords`]); on any other model non-empty hotwords are an
    /// `Err` (sherpa would `exit(-1)`). Qwen3-ASR takes hotwords in its model config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hotwords: Option<String>,
}

/// One timed segment (Whisper / Qwen3-ASR / Cohere segment output).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Segment {
    pub start: f32,
    pub duration: f32,
    pub text: String,
}

/// Result of one utterance.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OfflineResult {
    pub text: String,
    /// Detected or configured language (`en`, `zh`, ...), tags like `<|en|>` normalized.
    pub lang: Option<String>,
    /// SenseVoice emotion tag, lower case (`neutral`, `happy`, ...).
    pub emotion: Option<String>,
    /// SenseVoice audio-event tag, lower case (`speech`, `bgm`, `laughter`, ...).
    pub event: Option<String>,
    pub tokens: Vec<String>,
    /// Token start times in seconds (empty when the model has none).
    pub timestamps: Vec<f32>,
    pub segments: Vec<Segment>,
}

/// Trimmed transcript; punctuation-only output (SenseVoice answers silence with `。`) is empty.
pub fn speech_text(raw: String) -> String {
    let t = raw.trim();
    if t.chars().any(char::is_alphanumeric) {
        t.to_string()
    } else {
        String::new()
    }
}

/// `<|en|>` → `en`, `<|NEUTRAL|>` → `neutral`; empty / placeholder tags → `None`.
pub fn normalize_tag(raw: Option<String>) -> Option<String> {
    let raw = raw?;
    let t = raw
        .trim()
        .trim_start_matches("<|")
        .trim_end_matches("|>")
        .trim();
    if t.is_empty() || t.eq_ignore_ascii_case("nospeech") || t.eq_ignore_ascii_case("null") {
        return None;
    }
    Some(t.to_lowercase())
}

struct RecognizerInner {
    rt: Arc<SherpaRuntime>,
    raw: NonNull<SherpaOnnxOfflineRecognizer>,
    lock: CallLock,
}

// SAFETY: no thread affinity; every call into the recognizer (and its streams) holds `lock`.
unsafe impl Send for RecognizerInner {}
unsafe impl Sync for RecognizerInner {}

impl Drop for RecognizerInner {
    fn drop(&mut self) {
        let _g = self.lock.lock();
        // SAFETY: created by SherpaOnnxCreateOfflineRecognizer; streams hold an Arc to this inner.
        unsafe { (self.rt.api().SherpaOnnxDestroyOfflineRecognizer)(self.raw.as_ptr()) }
    }
}

/// A loaded non-streaming model.
#[derive(Clone)]
pub struct OfflineRecognizer {
    inner: Arc<RecognizerInner>,
    config: Arc<OfflineRecognizerConfig>,
}

impl std::fmt::Debug for OfflineRecognizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OfflineRecognizer")
            .field("model", &self.config.model)
            .finish()
    }
}

impl OfflineRecognizer {
    /// Load the model (blocking; seconds for large models, run it off async threads).
    pub fn new(rt: &Arc<SherpaRuntime>, config: &OfflineRecognizerConfig) -> Result<Self, String> {
        config.validate()?;
        let mut arena = CStrArena::default();
        let raw_cfg = config.to_raw(&mut arena)?;
        // SAFETY: `raw_cfg` points into `arena`, alive for the call; the C API copies everything.
        let raw = unsafe { (rt.api().SherpaOnnxCreateOfflineRecognizer)(&raw_cfg) };
        let raw = NonNull::new(raw.cast_mut()).ok_or_else(|| {
            format!(
                "sherpa-onnx could not load the {} model; check that the files belong together and match \
                 sherpa-onnx {}",
                config.model.family(),
                rt.info().version
            )
        })?;
        Ok(Self {
            inner: Arc::new(RecognizerInner {
                rt: rt.clone(),
                raw,
                lock: CallLock::default(),
            }),
            config: Arc::new(config.clone()),
        })
    }

    pub fn config(&self) -> &OfflineRecognizerConfig {
        &self.config
    }

    /// Transcribe one utterance of mono samples in `[-1, 1]` at `sample_rate` (sherpa resamples
    /// to the model rate). Blocking.
    pub fn transcribe(&self, sample_rate: u32, samples: &[f32], options: &DecodeOptions) -> Result<OfflineResult, String> {
        rate_i32(sample_rate)?;
        let hotwords = options.hotwords.as_deref().filter(|h| !h.trim().is_empty());
        if hotwords.is_some() {
            self.config.require_hotwords_support()?;
        }
        // Too short for one feature frame after padding (see `long_enough`): the stream would skip
        // the decode as well, but this also skips creating it.
        if !long_enough(samples.len(), sample_rate) {
            return Ok(OfflineResult::default());
        }
        let mut stream = match hotwords {
            Some(h) => self.create_stream_with_hotwords(h)?,
            None => self.create_stream()?,
        };
        if let Some(lang) = options.language.as_deref().filter(|l| !l.is_empty()) {
            stream.set_option("language", lang)?;
        }
        stream.accept_waveform(sample_rate, samples)?;
        stream.decode();
        stream.result()
    }

    pub fn create_stream(&self) -> Result<OfflineStream, String> {
        let _g = self.inner.lock.lock();
        // SAFETY: valid recognizer, call serialized.
        let raw = unsafe { (self.inner.rt.api().SherpaOnnxCreateOfflineStream)(self.inner.raw.as_ptr()) };
        self.wrap_stream(raw)
    }

    /// A stream biased toward `hotwords` (one phrase per line). `Err` unless the model is a
    /// transducer decoded with modified beam search (sherpa would `exit(-1)`).
    pub fn create_stream_with_hotwords(&self, hotwords: &str) -> Result<OfflineStream, String> {
        self.config.require_hotwords_support()?;
        let mut arena = CStrArena::default();
        let words = arena.str(hotwords)?;
        let _g = self.inner.lock.lock();
        // SAFETY: valid recognizer and C string, call serialized.
        let raw = unsafe {
            (self
                .inner
                .rt
                .api()
                .SherpaOnnxCreateOfflineStreamWithHotwords)(self.inner.raw.as_ptr(), words)
        };
        self.wrap_stream(raw)
    }

    fn wrap_stream(&self, raw: *const SherpaOnnxOfflineStream) -> Result<OfflineStream, String> {
        let raw = NonNull::new(raw.cast_mut()).ok_or("sherpa-onnx could not create an ASR stream")?;
        Ok(OfflineStream {
            rec: self.inner.clone(),
            raw,
            accepted: None,
            decoded: false,
        })
    }
}

/// One utterance. Audio is accepted once (sherpa's offline contract). [`decode`](Self::decode)
/// runs the model only on at least [`MIN_UTTERANCE_MS`] of accepted audio; otherwise the result is
/// empty.
pub struct OfflineStream {
    rec: Arc<RecognizerInner>,
    raw: NonNull<SherpaOnnxOfflineStream>,
    /// `(samples, sample rate)` of the accepted utterance.
    accepted: Option<(usize, u32)>,
    decoded: bool,
}

// SAFETY: only touched through `&mut self` or under the recognizer's call lock.
unsafe impl Send for OfflineStream {}
unsafe impl Sync for OfflineStream {}

impl Drop for OfflineStream {
    fn drop(&mut self) {
        let _g = self.rec.lock.lock();
        // SAFETY: created by this recognizer, destroyed once.
        unsafe { (self.rec.rt.api().SherpaOnnxDestroyOfflineStream)(self.raw.as_ptr()) }
    }
}

impl OfflineStream {
    /// Provide the whole utterance (at most once per stream; at most `i32::MAX` samples).
    pub fn accept_waveform(&mut self, sample_rate: u32, samples: &[f32]) -> Result<(), String> {
        if self.accepted.is_some() {
            return Err("an offline ASR stream accepts audio only once".into());
        }
        if samples.len() > i32::MAX as usize {
            return Err("the utterance is too long for one offline decode".into());
        }
        let rate = rate_i32(sample_rate)?;
        let _g = self.rec.lock.lock();
        // SAFETY: valid stream; `samples` outlives the call; the length fits i32 (checked).
        unsafe { (self.rec.rt.api().SherpaOnnxAcceptWaveformOffline)(self.raw.as_ptr(), rate, samples.as_ptr(), samples.len() as i32) };
        self.accepted = Some((samples.len(), sample_rate));
        Ok(())
    }

    pub fn set_option(&mut self, key: &str, value: &str) -> Result<(), String> {
        let mut arena = CStrArena::default();
        let (k, v) = (arena.str(key)?, arena.str(value)?);
        let _g = self.rec.lock.lock();
        // SAFETY: valid stream and C strings (copied by sherpa).
        unsafe { (self.rec.rt.api().SherpaOnnxOfflineStreamSetOption)(self.raw.as_ptr(), k, v) };
        Ok(())
    }

    /// Run the model on the accepted audio (blocking). Returns whether it ran: with nothing
    /// accepted, or less than [`MIN_UTTERANCE_MS`], the model is skipped (it could throw inside ONNX
    /// Runtime on zero feature frames, which would abort the process) and the result stays empty.
    pub fn decode(&mut self) -> bool {
        if !should_decode(self.accepted) {
            return false;
        }
        let _g = self.rec.lock.lock();
        // SAFETY: valid recognizer and a stream of that recognizer, call serialized.
        unsafe { (self.rec.rt.api().SherpaOnnxDecodeOfflineStream)(self.rec.raw.as_ptr(), self.raw.as_ptr()) }
        self.decoded = true;
        true
    }

    /// The decoded result; empty when [`decode`](Self::decode) did not run the model.
    pub fn result(&self) -> Result<OfflineResult, String> {
        if !self.decoded {
            return Ok(OfflineResult::default());
        }
        let api = self.rec.rt.api();
        let _g = self.rec.lock.lock();
        // SAFETY: valid stream; the snapshot is copied and destroyed right away.
        unsafe {
            let r: *const SherpaOnnxOfflineRecognizerResult = (api.SherpaOnnxGetOfflineStreamResult)(self.raw.as_ptr());
            if r.is_null() {
                return Err("sherpa-onnx returned no result".into());
            }
            let res = &*r;
            let seg_texts = strings(res.segment_texts_arr, res.segment_count);
            let seg_starts = floats(res.segment_timestamps, res.segment_count);
            let seg_durations = floats(res.segment_durations, res.segment_count);
            let segments = seg_texts
                .into_iter()
                .enumerate()
                .map(|(i, text)| Segment {
                    start: seg_starts.get(i).copied().unwrap_or(0.0),
                    duration: seg_durations.get(i).copied().unwrap_or(0.0),
                    text: text.trim().to_string(),
                })
                .collect();
            let out = OfflineResult {
                text: speech_text(owned_str(res.text)),
                lang: normalize_tag(opt_str(res.lang)),
                emotion: normalize_tag(opt_str(res.emotion)),
                event: normalize_tag(opt_str(res.event)),
                tokens: strings(res.tokens_arr, res.count),
                timestamps: floats(res.timestamps, res.count),
                segments,
            };
            (api.SherpaOnnxDestroyOfflineRecognizerResult)(r);
            Ok(out)
        }
    }
}
