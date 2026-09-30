//! Streaming ASR: `SherpaOnnxOnlineRecognizer` / `SherpaOnnxOnlineStream` (Nemotron 3.5 streaming,
//! streaming zipformer / paraformer / NeMo CTC).
//!
//! One [`OnlineRecognizer`] (the loaded model) serves any number of [`OnlineStream`]s (one per
//! audio source). Streams keep their recognizer alive; every native call is serialized per
//! recognizer (`CallLock`), so both types are `Send + Sync`.

use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::bindings::{SherpaOnnxOnlineRecognizer, SherpaOnnxOnlineRecognizerConfig, SherpaOnnxOnlineRecognizerResult, SherpaOnnxOnlineStream};
use super::lib::{
    c_float, floats, owned_str, rate_i32, require_files, strings, threads_or, CStrArena, CallLock, SherpaRuntime, CPU_PROVIDER,
    MAX_SAMPLES_PER_CALL, STREAMING_DEFAULT_THREADS,
};
use super::onnx_info::graph_output_count;

/// Model files of a streaming recognizer, by sherpa family.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum OnlineModel {
    /// Streaming transducer (zipformer, NeMo cache-aware FastConformer incl. Nemotron 3.5 ASR).
    Transducer {
        encoder: PathBuf,
        decoder: PathBuf,
        joiner: PathBuf,
    },
    Paraformer {
        encoder: PathBuf,
        decoder: PathBuf,
    },
    Zipformer2Ctc {
        model: PathBuf,
    },
    NemoCtc {
        model: PathBuf,
    },
    ToneCtc {
        model: PathBuf,
    },
}

impl OnlineModel {
    pub fn family(&self) -> &'static str {
        match self {
            OnlineModel::Transducer { .. } => "transducer",
            OnlineModel::Paraformer { .. } => "paraformer",
            OnlineModel::Zipformer2Ctc { .. } => "zipformer2_ctc",
            OnlineModel::NemoCtc { .. } => "nemo_ctc",
            OnlineModel::ToneCtc { .. } => "tone_ctc",
        }
    }

    fn files(&self) -> Vec<&Path> {
        match self {
            OnlineModel::Transducer { encoder, decoder, joiner } => vec![encoder, decoder, joiner],
            OnlineModel::Paraformer { encoder, decoder } => vec![encoder, decoder],
            OnlineModel::Zipformer2Ctc { model } | OnlineModel::NemoCtc { model } | OnlineModel::ToneCtc { model } => vec![model],
        }
        .into_iter()
        .map(PathBuf::as_path)
        .collect()
    }
}

/// Endpoint rules (trailing silence / utterance length, in seconds). Sigil's own Silero segmenter
/// is the primary endpointer; these are sherpa's decoder-side rules for standalone use.
///
/// Every value must be positive and finite ([`OnlineRecognizerConfig::validate`]): the C API reads
/// 0 as "unset" and substitutes its default (2.4 / 1.2 / 20 s). A rule fires once its threshold is
/// reached, so a very large value keeps it from firing.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EndpointRules {
    /// Endpoint after this much trailing silence when nothing was decoded.
    pub rule1_min_trailing_silence: f32,
    /// Endpoint after this much trailing silence once something was decoded.
    pub rule2_min_trailing_silence: f32,
    /// Endpoint once the utterance is this long.
    pub rule3_min_utterance_length: f32,
}

impl Default for EndpointRules {
    fn default() -> Self {
        // sherpa-onnx's defaults.
        Self {
            rule1_min_trailing_silence: 2.4,
            rule2_min_trailing_silence: 1.2,
            rule3_min_utterance_length: 20.0,
        }
    }
}

/// Decoding search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum Decoding {
    #[default]
    GreedySearch,
    /// Needed for hotwords. Transducers only, and on the streaming side not NeMo transducers
    /// (Nemotron): every other model is refused by `validate` (sherpa would `exit(-1)`).
    ModifiedBeamSearch { max_active_paths: u32 },
}

impl Decoding {
    pub fn is_beam_search(&self) -> bool {
        matches!(self, Decoding::ModifiedBeamSearch { .. })
    }

    pub(crate) fn method(&self) -> &'static str {
        match self {
            Decoding::GreedySearch => "greedy_search",
            Decoding::ModifiedBeamSearch { .. } => "modified_beam_search",
        }
    }

    pub(crate) fn max_active_paths(&self) -> i32 {
        match self {
            Decoding::GreedySearch => 4,
            Decoding::ModifiedBeamSearch { max_active_paths } => (*max_active_paths).clamp(1, 64) as i32,
        }
    }
}

/// Everything needed to create an [`OnlineRecognizer`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OnlineRecognizerConfig {
    pub model: OnlineModel,
    pub tokens: PathBuf,
    /// Explicit sherpa model type when the model metadata does not carry one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_type: Option<String>,
    /// `cjkchar`, `bpe` or `cjkchar+bpe` (hotwords only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modeling_unit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bpe_vocab: Option<PathBuf>,
    /// Feature sample rate (the model's; input audio of any rate is resampled by sherpa).
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_feature_dim")]
    pub feature_dim: u32,
    /// ONNX Runtime threads (default 1: a decode runs every chunk and extra workers spin after
    /// each run; at most 4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_threads: Option<u32>,
    #[serde(default)]
    pub decoding: Decoding,
    /// `None`: sherpa's endpointing off (Sigil's VAD decides).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<EndpointRules>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hotwords_file: Option<PathBuf>,
    #[serde(default = "default_hotwords_score")]
    pub hotwords_score: f32,
    #[serde(default)]
    pub blank_penalty: f32,
    /// Inverse-text-normalization rule FSTs / FARs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rule_fsts: Vec<PathBuf>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rule_fars: Vec<PathBuf>,
}

pub(crate) fn default_sample_rate() -> u32 {
    16_000
}
pub(crate) fn default_feature_dim() -> u32 {
    80
}
pub(crate) fn default_hotwords_score() -> f32 {
    1.5
}

impl OnlineRecognizerConfig {
    /// A config with defaults for everything but the model files.
    pub fn new(model: OnlineModel, tokens: impl Into<PathBuf>) -> Self {
        Self {
            model,
            tokens: tokens.into(),
            model_type: None,
            modeling_unit: None,
            bpe_vocab: None,
            sample_rate: default_sample_rate(),
            feature_dim: default_feature_dim(),
            num_threads: None,
            decoding: Decoding::default(),
            endpoint: None,
            hotwords_file: None,
            hotwords_score: default_hotwords_score(),
            blank_penalty: 0.0,
            rule_fsts: Vec::new(),
            rule_fars: Vec::new(),
        }
    }

    /// Check every referenced file exists and that sherpa supports the option combination (before
    /// any native call: sherpa answers unsupported ones with `exit(-1)`).
    pub fn validate(&self) -> Result<(), String> {
        let what = format!("sherpa streaming {} model", self.model.family());
        let mut files = self.model.files();
        files.push(&self.tokens);
        files.extend(self.bpe_vocab.as_deref());
        files.extend(self.hotwords_file.as_deref());
        files.extend(self.rule_fsts.iter().map(PathBuf::as_path));
        files.extend(self.rule_fars.iter().map(PathBuf::as_path));
        require_files(&what, files)?;
        if self.sample_rate == 0 || self.feature_dim == 0 {
            return Err(format!("{what}: sample rate and feature dimension must be positive"));
        }
        if let Some(r) = self.endpoint {
            let rules = [r.rule1_min_trailing_silence, r.rule2_min_trailing_silence, r.rule3_min_utterance_length];
            if rules.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
                return Err(format!(
                    "{what}: endpoint rules must be positive seconds (sherpa-onnx reads 0 as its default; a large value keeps \
                     a rule from firing)"
                ));
            }
        }
        self.check_decoding(&what)
    }

    /// sherpa-onnx 1.13.8 runs modified beam search (and so hotwords) only on streaming icefall /
    /// k2 transducers, whose decoder has one output. `OnlineRecognizerTransducerNeMoImpl`
    /// (Nemotron, cache-aware FastConformer: decoders with more outputs) and the other families
    /// call `exit(-1)` on `modified_beam_search`, and their `CreateStream(hotwords)` does too.
    fn check_decoding(&self, what: &str) -> Result<(), String> {
        if !self.decoding.is_beam_search() {
            if self.hotwords_file.is_some() {
                return Err(format!("{what}: a hotwords file needs modified beam search"));
            }
            return Ok(());
        }
        let OnlineModel::Transducer { decoder, .. } = &self.model else {
            return Err(format!(
                "{what}: modified beam search and hotwords need a transducer model; the {} family decodes with greedy \
                 search only",
                self.model.family()
            ));
        };
        if let Some(unit) = self.modeling_unit.as_deref().filter(|u| u.contains("bpe")) {
            if self.bpe_vocab.is_none() {
                return Err(format!("{what}: modeling unit {unit} needs bpe_vocab for hotwords"));
            }
        }
        let outputs = graph_output_count(decoder).map_err(|e| format!("{what}: {e}"))?;
        if outputs != 1 {
            return Err(format!(
                "{what}: {} is a NeMo transducer decoder (Nemotron, FastConformer); sherpa-onnx streams those with greedy \
                 search only, so modified beam search and hotwords are unavailable",
                decoder.display()
            ));
        }
        Ok(())
    }

    /// Whether streams of this (validated) config may carry hotwords: a k2 transducer decoded with
    /// modified beam search.
    pub fn supports_hotwords(&self) -> bool {
        matches!(self.model, OnlineModel::Transducer { .. }) && self.decoding.is_beam_search()
    }

    /// The C config (pointers into `arena`).
    pub(crate) fn to_raw(&self, arena: &mut CStrArena) -> Result<SherpaOnnxOnlineRecognizerConfig, String> {
        let mut c = SherpaOnnxOnlineRecognizerConfig::default();
        c.feat_config.sample_rate = self.sample_rate as i32;
        c.feat_config.feature_dim = self.feature_dim as i32;
        let m = &mut c.model_config;
        match &self.model {
            OnlineModel::Transducer { encoder, decoder, joiner } => {
                m.transducer.encoder = arena.path(encoder)?;
                m.transducer.decoder = arena.path(decoder)?;
                m.transducer.joiner = arena.path(joiner)?;
            }
            OnlineModel::Paraformer { encoder, decoder } => {
                m.paraformer.encoder = arena.path(encoder)?;
                m.paraformer.decoder = arena.path(decoder)?;
            }
            OnlineModel::Zipformer2Ctc { model } => m.zipformer2_ctc.model = arena.path(model)?,
            OnlineModel::NemoCtc { model } => m.nemo_ctc.model = arena.path(model)?,
            OnlineModel::ToneCtc { model } => m.t_one_ctc.model = arena.path(model)?,
        }
        m.tokens = arena.path(&self.tokens)?;
        m.num_threads = threads_or(self.num_threads, STREAMING_DEFAULT_THREADS);
        m.provider = CPU_PROVIDER.as_ptr();
        m.debug = 0;
        m.model_type = arena.opt(self.model_type.as_deref())?;
        m.modeling_unit = arena.opt(self.modeling_unit.as_deref())?;
        m.bpe_vocab = arena.opt_path(self.bpe_vocab.as_deref())?;
        c.decoding_method = arena.str(self.decoding.method())?;
        c.max_active_paths = self.decoding.max_active_paths();
        if let Some(rules) = self.endpoint {
            c.enable_endpoint = 1;
            c.rule1_min_trailing_silence = rules.rule1_min_trailing_silence;
            c.rule2_min_trailing_silence = rules.rule2_min_trailing_silence;
            c.rule3_min_utterance_length = rules.rule3_min_utterance_length;
        }
        c.hotwords_file = arena.opt_path(self.hotwords_file.as_deref())?;
        c.hotwords_score = c_float(self.hotwords_score, default_hotwords_score());
        c.blank_penalty = if self.blank_penalty.is_finite() { self.blank_penalty } else { 0.0 };
        c.rule_fsts = arena.path_list(&self.rule_fsts)?;
        c.rule_fars = arena.path_list(&self.rule_fars)?;
        Ok(c)
    }
}

/// Silence fed around a stream's audio (see [`StreamPadding::for_encoder`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamPadding {
    /// Before the first audio.
    pub head_ms: u32,
    /// After the last audio, before the input is finished.
    pub tail_ms: u32,
}

/// Tail padding when the model does not say how much it needs (sherpa-onnx's streaming examples
/// pad 0.66 s).
pub const DEFAULT_TAIL_PADDING_MS: u32 = 660;
/// Head padding of NeMo cache-aware streams: without it the first word is often lost when speech
/// starts right away (measured on Nemotron 3.5: "No preguntes ..." lost "No", "Alles hat ein Ende"
/// its first words); sherpa-onnx's own CLI pads 0.3 s since its Nemotron 3.5 support. It costs no
/// wall-clock latency (the padding is there at once).
pub const NEMO_HEAD_PADDING_MS: u32 = 300;

/// sherpa decodes a NeMo cache-aware stream only in whole windows (`window_size` feature frames of
/// 10 ms), also after the input is finished, so up to a window of real audio at the end is never
/// decoded without enough padding (measured at 560 ms chunks with 0.3 s: "... tu país" came out
/// as "... tu pa"). One full window plus a frame of slack flushes it.
pub fn window_padding_ms(window_frames: u32) -> u32 {
    window_frames.saturating_mul(10).saturating_add(30)
}

impl StreamPadding {
    /// Padding for a streaming model whose encoder metadata is `encoder` (`None`: unreadable or
    /// not a transducer): NeMo cache-aware encoders (`window_size` metadata) get
    /// [`NEMO_HEAD_PADDING_MS`] and one window of tail; others none and
    /// [`DEFAULT_TAIL_PADDING_MS`].
    pub fn for_encoder(encoder: Option<&super::onnx_info::OnnxSummary>) -> Self {
        let window = encoder
            .and_then(|s| s.meta("window_size"))
            .and_then(|w| w.trim().parse::<u32>().ok())
            .filter(|w| (1..=1000).contains(w));
        match window {
            Some(w) => Self {
                head_ms: NEMO_HEAD_PADDING_MS,
                tail_ms: window_padding_ms(w),
            },
            None => Self {
                head_ms: 0,
                tail_ms: DEFAULT_TAIL_PADDING_MS,
            },
        }
    }

    /// For the encoder of `config` (reads its metadata; cheap, no initializers).
    pub fn for_config(config: &OnlineRecognizerConfig) -> Self {
        let summary = match &config.model {
            OnlineModel::Transducer { encoder, .. } => super::onnx_info::summary(encoder).ok(),
            _ => None,
        };
        Self::for_encoder(summary.as_ref())
    }
}

/// Snapshot of a stream's hypothesis.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OnlineResult {
    pub text: String,
    pub tokens: Vec<String>,
    /// Token start times in seconds (empty when the model has none).
    pub timestamps: Vec<f32>,
}

struct RecognizerInner {
    rt: Arc<SherpaRuntime>,
    raw: NonNull<SherpaOnnxOnlineRecognizer>,
    lock: CallLock,
}

// SAFETY: the native recognizer has no thread affinity, and every call into it holds `lock`.
unsafe impl Send for RecognizerInner {}
unsafe impl Sync for RecognizerInner {}

impl Drop for RecognizerInner {
    fn drop(&mut self) {
        let _g = self.lock.lock();
        // SAFETY: created by SherpaOnnxCreateOnlineRecognizer; every stream holds an Arc to this
        // inner, so no stream outlives it.
        unsafe { (self.rt.api().SherpaOnnxDestroyOnlineRecognizer)(self.raw.as_ptr()) }
    }
}

/// A loaded streaming model.
#[derive(Clone)]
pub struct OnlineRecognizer {
    inner: Arc<RecognizerInner>,
    config: Arc<OnlineRecognizerConfig>,
}

impl std::fmt::Debug for OnlineRecognizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnlineRecognizer")
            .field("model", &self.config.model)
            .finish()
    }
}

impl OnlineRecognizer {
    /// Load the model (blocking; seconds for large models, run it off async threads).
    pub fn new(rt: &Arc<SherpaRuntime>, config: &OnlineRecognizerConfig) -> Result<Self, String> {
        config.validate()?;
        let mut arena = CStrArena::default();
        let raw_cfg = config.to_raw(&mut arena)?;
        // SAFETY: `raw_cfg` points into `arena`, alive for the call; the C API copies everything.
        let raw = unsafe { (rt.api().SherpaOnnxCreateOnlineRecognizer)(&raw_cfg) };
        let raw = NonNull::new(raw.cast_mut()).ok_or_else(|| {
            format!(
                "sherpa-onnx could not load the streaming {} model (tokens {}); check that the files belong \
                 together and match sherpa-onnx {}",
                config.model.family(),
                config.tokens.display(),
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

    pub fn config(&self) -> &OnlineRecognizerConfig {
        &self.config
    }

    /// A new decoding state (one per audio source / utterance sequence).
    pub fn create_stream(&self) -> Result<OnlineStream, String> {
        let _g = self.inner.lock.lock();
        // SAFETY: valid recognizer, call serialized.
        let raw = unsafe { (self.inner.rt.api().SherpaOnnxCreateOnlineStream)(self.inner.raw.as_ptr()) };
        self.wrap_stream(raw)
    }

    /// A stream biased toward `hotwords` (one phrase per line). Only a k2 transducer decoded with
    /// modified beam search takes them ([`OnlineRecognizerConfig::supports_hotwords`]); any other
    /// recognizer returns `Err` (sherpa would `exit(-1)`).
    pub fn create_stream_with_hotwords(&self, hotwords: &str) -> Result<OnlineStream, String> {
        if !self.config.supports_hotwords() {
            return Err(format!(
                "the streaming {} model does not take hotwords: they need a transducer model decoded with modified beam search",
                self.config.model.family()
            ));
        }
        let mut arena = CStrArena::default();
        let words = arena.str(hotwords)?;
        let _g = self.inner.lock.lock();
        // SAFETY: valid recognizer and C string, call serialized.
        let raw = unsafe { (self.inner.rt.api().SherpaOnnxCreateOnlineStreamWithHotwords)(self.inner.raw.as_ptr(), words) };
        self.wrap_stream(raw)
    }

    fn wrap_stream(&self, raw: *const SherpaOnnxOnlineStream) -> Result<OnlineStream, String> {
        let raw = NonNull::new(raw.cast_mut()).ok_or("sherpa-onnx could not create a streaming ASR stream")?;
        Ok(OnlineStream {
            rec: self.inner.clone(),
            raw,
            input: InputState::default(),
        })
    }
}

/// Input side of an [`OnlineStream`]. `input_finished` is terminal: sherpa's `Reset` keeps the
/// feature extractor (and its input-finished flag), and kaldi-native-fbank aborts the process on
/// audio after `InputFinished` (`KNF_LOG(FATAL)`), so a finished stream refuses audio and resets
/// for good; the next utterance needs a new stream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct InputState {
    finished: bool,
}

const FINISHED: &str = "this streaming ASR stream's input is finished; create a new stream for more audio";

impl InputState {
    /// Audio may be appended.
    pub(crate) fn accept(&self) -> Result<(), String> {
        if self.finished {
            Err(FINISHED.into())
        } else {
            Ok(())
        }
    }

    /// Mark the input finished; `true` the first time (when the native call is due).
    pub(crate) fn finish(&mut self) -> bool {
        !std::mem::replace(&mut self.finished, true)
    }

    /// A reset starts a new utterance only on an open stream.
    pub(crate) fn reset(&self) -> Result<(), String> {
        self.accept()
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.finished
    }
}

/// One streaming decoding state. `Send + Sync` (all methods take `&mut self` or lock).
pub struct OnlineStream {
    rec: Arc<RecognizerInner>,
    raw: NonNull<SherpaOnnxOnlineStream>,
    input: InputState,
}

// SAFETY: the stream is only touched through `&mut self` or under the recognizer's call lock.
unsafe impl Send for OnlineStream {}
unsafe impl Sync for OnlineStream {}

impl Drop for OnlineStream {
    fn drop(&mut self) {
        let _g = self.rec.lock.lock();
        // SAFETY: created by this recognizer, destroyed once.
        unsafe { (self.rec.rt.api().SherpaOnnxDestroyOnlineStream)(self.raw.as_ptr()) }
    }
}

impl OnlineStream {
    /// Append mono samples in `[-1, 1]` at `sample_rate` (sherpa resamples to the model rate).
    /// `Err` after [`input_finished`](Self::input_finished): a finished stream never takes audio
    /// again (not even after [`reset`](Self::reset)); create a new one with
    /// [`OnlineRecognizer::create_stream`].
    pub fn accept_waveform(&mut self, sample_rate: u32, samples: &[f32]) -> Result<(), String> {
        self.input.accept()?;
        if samples.is_empty() {
            return Ok(());
        }
        let rate = rate_i32(sample_rate)?;
        let _g = self.rec.lock.lock();
        for chunk in samples.chunks(MAX_SAMPLES_PER_CALL) {
            // SAFETY: valid stream; `chunk` outlives the call and its length fits i32.
            unsafe { (self.rec.rt.api().SherpaOnnxOnlineStreamAcceptWaveform)(self.raw.as_ptr(), rate, chunk.as_ptr(), chunk.len() as i32) }
        }
        Ok(())
    }

    /// Signal the end of input (flushes the model's right context on the next decode). Terminal:
    /// decode and read the result, then drop the stream; more audio needs a new stream.
    pub fn input_finished(&mut self) {
        if !self.input.finish() {
            return;
        }
        let _g = self.rec.lock.lock();
        // SAFETY: valid stream.
        unsafe { (self.rec.rt.api().SherpaOnnxOnlineStreamInputFinished)(self.raw.as_ptr()) };
    }

    /// Whether [`input_finished`](Self::input_finished) was called.
    pub fn is_finished(&self) -> bool {
        self.input.is_finished()
    }

    /// Decode every chunk that is ready; returns how many decode steps ran.
    pub fn decode(&mut self) -> usize {
        self.decode_cancellable(None).unwrap_or(0)
    }

    /// [`decode`](Self::decode) that stops between decode steps once `cancel` is set (`None` when
    /// it stopped early; the stream stays usable, the skipped chunks decode on the next call).
    pub fn decode_cancellable(&mut self, cancel: Option<&std::sync::atomic::AtomicBool>) -> Option<usize> {
        let api = self.rec.rt.api();
        let mut steps = 0;
        loop {
            if cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed)) {
                return None;
            }
            let _g = self.rec.lock.lock();
            // SAFETY: valid recognizer and stream of that recognizer, call serialized.
            unsafe {
                if (api.SherpaOnnxIsOnlineStreamReady)(self.rec.raw.as_ptr(), self.raw.as_ptr()) == 0 {
                    break;
                }
                (api.SherpaOnnxDecodeOnlineStream)(self.rec.raw.as_ptr(), self.raw.as_ptr());
            }
            steps += 1;
        }
        Some(steps)
    }

    /// The current hypothesis.
    pub fn result(&self) -> Result<OnlineResult, String> {
        let api = self.rec.rt.api();
        let _g = self.rec.lock.lock();
        // SAFETY: valid recognizer and stream; the snapshot is copied and destroyed right away.
        unsafe {
            let r: *const SherpaOnnxOnlineRecognizerResult = (api.SherpaOnnxGetOnlineStreamResult)(self.rec.raw.as_ptr(), self.raw.as_ptr());
            if r.is_null() {
                return Err("sherpa-onnx returned no streaming result".into());
            }
            let out = OnlineResult {
                text: super::offline::speech_text(owned_str((*r).text)),
                tokens: strings((*r).tokens_arr, (*r).count),
                timestamps: floats((*r).timestamps, (*r).count),
            };
            (api.SherpaOnnxDestroyOnlineRecognizerResult)(r);
            Ok(out)
        }
    }

    /// Whether sherpa's endpoint rules fired (only with [`OnlineRecognizerConfig::endpoint`]).
    pub fn is_endpoint(&self) -> bool {
        let _g = self.rec.lock.lock();
        // SAFETY: valid recognizer and stream.
        unsafe { (self.rec.rt.api().SherpaOnnxOnlineStreamIsEndpoint)(self.rec.raw.as_ptr(), self.raw.as_ptr()) != 0 }
    }

    /// Start a new utterance on this stream after an endpoint ([`is_endpoint`](Self::is_endpoint)).
    /// `Err` after [`input_finished`](Self::input_finished): sherpa's reset keeps the feature
    /// extractor's input-finished state, so such a stream can never take audio again; create a new
    /// stream instead.
    pub fn reset(&mut self) -> Result<(), String> {
        self.input.reset()?;
        let _g = self.rec.lock.lock();
        // SAFETY: valid recognizer and stream.
        unsafe { (self.rec.rt.api().SherpaOnnxOnlineStreamReset)(self.rec.raw.as_ptr(), self.raw.as_ptr()) };
        Ok(())
    }

    /// Model-specific per-stream option (e.g. `is_final` for streaming paraformer).
    pub fn set_option(&mut self, key: &str, value: &str) -> Result<(), String> {
        let mut arena = CStrArena::default();
        let (k, v) = (arena.str(key)?, arena.str(value)?);
        let _g = self.rec.lock.lock();
        // SAFETY: valid stream and C strings (copied by sherpa).
        unsafe { (self.rec.rt.api().SherpaOnnxOnlineStreamSetOption)(self.raw.as_ptr(), k, v) };
        Ok(())
    }
}
