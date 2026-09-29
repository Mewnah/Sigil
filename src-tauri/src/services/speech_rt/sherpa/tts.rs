//! Offline TTS: `SherpaOnnxOfflineTts` (Kokoro, Kitten, Piper / VITS, Matcha) with sherpa's C
//! espeak-ng.
//!
//! [`OfflineTts::generate`] synthesizes a text; sentences arrive through an optional chunk
//! callback as they are generated (sherpa batches `max_num_sentences` at a time), and a cancel
//! flag stops generation at the next batch boundary (a flag already set returns at once, without
//! synthesizing). espeak-ng keeps process-global state, so generation is serialized across every
//! sherpa TTS instance in the process.

use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::bindings::{SherpaOnnxGenerationConfig, SherpaOnnxOfflineTts, SherpaOnnxOfflineTtsConfig};
use super::lib::{c_float, floats, require_dir, require_files, threads, CStrArena, CallLock, SherpaRuntime, CPU_PROVIDER};

/// sherpa-onnx's default pause scale (silences in the generated audio shrink to 20%).
pub const DEFAULT_SILENCE_SCALE: f32 = 0.2;
/// Shortest pause scale sherpa-onnx applies (1% of the natural silence), what "no pause" maps to.
///
/// Neither 0 nor a tiny positive value works: the C API turns an exact 0 into its 0.2 default
/// (`SHERPA_ONNX_OR`), and `ScaleSilence` (offline-tts.cc) skips scaling altogether outside
/// `[0.01, 10]` ("Silence scale ... is not in [0.01, 10.00]. Skip scaling."), which keeps the
/// model's full pauses, longer than the default (measured with Kokoro).
pub const MIN_SILENCE_SCALE: f32 = 0.01;
/// Longest pause scale sherpa-onnx applies.
pub const MAX_SILENCE_SCALE: f32 = 10.0;

/// Pause scale as the C API must see it: `None` / non-finite → [`DEFAULT_SILENCE_SCALE`], anything
/// else clamped to `MIN_SILENCE_SCALE..=MAX_SILENCE_SCALE` (so 0, "no pause", is the shortest pause
/// sherpa applies).
pub(crate) fn silence_scale(v: Option<f32>) -> f32 {
    match v {
        Some(x) if x.is_finite() => x.clamp(MIN_SILENCE_SCALE, MAX_SILENCE_SCALE),
        _ => DEFAULT_SILENCE_SCALE,
    }
}

/// A VITS / Matcha noise scale: `None` / non-finite → `default`, negatives → 0, and 0 ("no
/// sampling noise") passed as `lib::C_FLOAT_ZERO` (the C API turns an exact 0 into its default).
pub(crate) fn noise_scale(v: Option<f32>, default: f32) -> f32 {
    match v {
        Some(x) if x.is_finite() => c_float(x.max(0.0), default),
        _ => default,
    }
}

/// Model files and options of a TTS model, by sherpa family.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum TtsModel {
    /// Kokoro 82M (v0.19 English, v1.0 / v1.1 multi-lang with lexicons).
    Kokoro {
        model: PathBuf,
        voices: PathBuf,
        tokens: PathBuf,
        /// `espeak-ng-data` folder.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data_dir: Option<PathBuf>,
        /// Lexicons (multi-lang models: `lexicon-us-en.txt`, `lexicon-zh.txt`).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        lexicons: Vec<PathBuf>,
        /// espeak language for multi-lang models (e.g. `en-us`, `es`, `fr-fr`); empty = model default.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lang: Option<String>,
    },
    Kitten {
        model: PathBuf,
        voices: PathBuf,
        tokens: PathBuf,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data_dir: Option<PathBuf>,
    },
    /// VITS, including Piper voices (`data_dir` = espeak-ng-data) and lexicon-based VITS.
    Vits {
        model: PathBuf,
        tokens: PathBuf,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lexicon: Option<PathBuf>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data_dir: Option<PathBuf>,
        /// Sampling noise (default 0.667; 0 = none).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        noise_scale: Option<f32>,
        /// Duration noise (default 0.8; 0 = none).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        noise_scale_w: Option<f32>,
    },
    Matcha {
        acoustic_model: PathBuf,
        vocoder: PathBuf,
        tokens: PathBuf,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lexicon: Option<PathBuf>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data_dir: Option<PathBuf>,
        /// Sampling noise (default 0.667; 0 = none).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        noise_scale: Option<f32>,
    },
}

impl TtsModel {
    pub fn family(&self) -> &'static str {
        match self {
            TtsModel::Kokoro { .. } => "kokoro",
            TtsModel::Kitten { .. } => "kitten",
            TtsModel::Vits { .. } => "vits",
            TtsModel::Matcha { .. } => "matcha",
        }
    }

    fn files(&self) -> Vec<&Path> {
        let v: Vec<&PathBuf> = match self {
            TtsModel::Kokoro {
                model,
                voices,
                tokens,
                lexicons,
                ..
            } => [model, voices, tokens]
                .into_iter()
                .chain(lexicons)
                .collect(),
            TtsModel::Kitten { model, voices, tokens, .. } => vec![model, voices, tokens],
            TtsModel::Vits { model, tokens, lexicon, .. } => [model, tokens].into_iter().chain(lexicon).collect(),
            TtsModel::Matcha {
                acoustic_model,
                vocoder,
                tokens,
                lexicon,
                ..
            } => [acoustic_model, vocoder, tokens]
                .into_iter()
                .chain(lexicon)
                .collect(),
        };
        v.into_iter().map(PathBuf::as_path).collect()
    }

    fn data_dir(&self) -> Option<&Path> {
        match self {
            TtsModel::Kokoro { data_dir, .. }
            | TtsModel::Kitten { data_dir, .. }
            | TtsModel::Vits { data_dir, .. }
            | TtsModel::Matcha { data_dir, .. } => data_dir.as_deref(),
        }
    }
}

/// Everything needed to create an [`OfflineTts`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TtsConfig {
    pub model: TtsModel,
    /// ONNX Runtime threads (default 2, at most 4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_threads: Option<u32>,
    /// Sentences synthesized per batch (each batch is one chunk callback); default 1 so the first
    /// sentence plays as early as possible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_num_sentences: Option<u32>,
    /// Pause scale: silences in the generated audio are scaled by it (sherpa default 0.2; clamped
    /// to 0.01..=10, so 0 = the shortest pause sherpa applies).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub silence_scale: Option<f32>,
    /// Text-normalization rule FSTs (multi-lang Kokoro: `date-zh.fst`, `number-zh.fst`, ...).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rule_fsts: Vec<PathBuf>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rule_fars: Vec<PathBuf>,
}

impl TtsConfig {
    pub fn new(model: TtsModel) -> Self {
        Self {
            model,
            num_threads: None,
            max_num_sentences: None,
            silence_scale: None,
            rule_fsts: Vec::new(),
            rule_fars: Vec::new(),
        }
    }

    /// Check every referenced file / folder exists (before any native call).
    pub fn validate(&self) -> Result<(), String> {
        let what = format!("sherpa {} voice", self.model.family());
        let mut files = self.model.files();
        files.extend(self.rule_fsts.iter().map(PathBuf::as_path));
        files.extend(self.rule_fars.iter().map(PathBuf::as_path));
        require_files(&what, files)?;
        if let Some(d) = self.model.data_dir() {
            require_dir(&what, d)?;
        }
        Ok(())
    }

    /// The C config (pointers into `arena`).
    pub(crate) fn to_raw(&self, arena: &mut CStrArena) -> Result<SherpaOnnxOfflineTtsConfig, String> {
        let mut c = SherpaOnnxOfflineTtsConfig::default();
        let m = &mut c.model;
        match &self.model {
            TtsModel::Kokoro {
                model,
                voices,
                tokens,
                data_dir,
                lexicons,
                lang,
            } => {
                let k = &mut m.kokoro;
                k.model = arena.path(model)?;
                k.voices = arena.path(voices)?;
                k.tokens = arena.path(tokens)?;
                k.data_dir = arena.opt_path(data_dir.as_deref())?;
                k.lexicon = arena.path_list(lexicons)?;
                k.lang = arena.opt(lang.as_deref())?;
                k.length_scale = 1.0;
            }
            TtsModel::Kitten {
                model,
                voices,
                tokens,
                data_dir,
            } => {
                let k = &mut m.kitten;
                k.model = arena.path(model)?;
                k.voices = arena.path(voices)?;
                k.tokens = arena.path(tokens)?;
                k.data_dir = arena.opt_path(data_dir.as_deref())?;
                k.length_scale = 1.0;
            }
            TtsModel::Vits {
                model,
                tokens,
                lexicon,
                data_dir,
                noise_scale,
                noise_scale_w,
            } => {
                let v = &mut m.vits;
                v.model = arena.path(model)?;
                v.tokens = arena.path(tokens)?;
                v.lexicon = arena.opt_path(lexicon.as_deref())?;
                v.data_dir = arena.opt_path(data_dir.as_deref())?;
                v.noise_scale = self::noise_scale(*noise_scale, 0.667);
                v.noise_scale_w = self::noise_scale(*noise_scale_w, 0.8);
                v.length_scale = 1.0;
            }
            TtsModel::Matcha {
                acoustic_model,
                vocoder,
                tokens,
                lexicon,
                data_dir,
                noise_scale,
            } => {
                let t = &mut m.matcha;
                t.acoustic_model = arena.path(acoustic_model)?;
                t.vocoder = arena.path(vocoder)?;
                t.tokens = arena.path(tokens)?;
                t.lexicon = arena.opt_path(lexicon.as_deref())?;
                t.data_dir = arena.opt_path(data_dir.as_deref())?;
                t.noise_scale = self::noise_scale(*noise_scale, 0.667);
                t.length_scale = 1.0;
            }
        }
        m.num_threads = threads(self.num_threads);
        m.debug = 0;
        m.provider = CPU_PROVIDER.as_ptr();
        c.max_num_sentences = self.max_num_sentences.unwrap_or(1).clamp(1, 32) as i32;
        c.silence_scale = silence_scale(self.silence_scale);
        c.rule_fsts = arena.path_list(&self.rule_fsts)?;
        c.rule_fars = arena.path_list(&self.rule_fars)?;
        Ok(c)
    }
}

/// Per-call options of [`OfflineTts::generate`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerateOptions {
    /// Speaker / voice id (`0..num_speakers`).
    pub sid: u32,
    /// Speech rate (1.0 normal, larger is faster).
    pub speed: f32,
    /// Pause scale (see [`TtsConfig::silence_scale`]; 0 = shortest pause); `None` keeps the
    /// config's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub silence_scale: Option<f32>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            sid: 0,
            speed: 1.0,
            silence_scale: None,
        }
    }
}

/// Synthesized mono audio.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeneratedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Generation stopped early (cancel flag or the chunk callback returned `false`).
    pub cancelled: bool,
}

/// espeak-ng (inside sherpa) keeps process-global state: one generation at a time per process.
static GENERATE_LOCK: Mutex<()> = Mutex::new(());

/// A loaded TTS model. `Send + Sync`.
pub struct OfflineTts {
    rt: Arc<SherpaRuntime>,
    raw: NonNull<SherpaOnnxOfflineTts>,
    lock: CallLock,
    sample_rate: u32,
    num_speakers: u32,
    config: TtsConfig,
}

// SAFETY: no thread affinity; every call into the engine holds `lock` (and generation also the
// process-wide GENERATE_LOCK).
unsafe impl Send for OfflineTts {}
unsafe impl Sync for OfflineTts {}

impl std::fmt::Debug for OfflineTts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OfflineTts")
            .field("model", &self.config.model)
            .field("sample_rate", &self.sample_rate)
            .field("num_speakers", &self.num_speakers)
            .finish()
    }
}

impl Drop for OfflineTts {
    fn drop(&mut self) {
        let _g = self.lock.lock();
        // SAFETY: created by SherpaOnnxCreateOfflineTts, destroyed once.
        unsafe { (self.rt.api().SherpaOnnxDestroyOfflineTts)(self.raw.as_ptr()) }
    }
}

/// Receives each generated batch of samples; returns `false` to stop generation.
pub type ChunkCallback<'f> = dyn FnMut(&[f32]) -> bool + 'f;

struct CallbackState<'c, 'f> {
    cancel: Option<&'c AtomicBool>,
    on_chunk: Option<&'f mut ChunkCallback<'f>>,
    stopped: bool,
    panicked: bool,
}

/// `SherpaOnnxGeneratedAudioProgressCallbackWithArg`: forwards each batch to the Rust callback.
/// Never unwinds into C (a panic stops generation and is reported afterwards).
unsafe extern "C" fn on_progress(samples: *const f32, n: i32, _progress: f32, arg: *mut c_void) -> i32 {
    if arg.is_null() {
        return 1;
    }
    // SAFETY: `arg` is the `CallbackState` owned by `generate`, alive for the whole native call.
    let state = unsafe { &mut *(arg as *mut CallbackState<'_, '_>) };
    let keep_going = catch_unwind(AssertUnwindSafe(|| {
        if state.cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return false;
        }
        match state.on_chunk.as_mut() {
            Some(cb) if !samples.is_null() && n > 0 => {
                // SAFETY: sherpa passes `n` valid samples for the duration of the callback.
                let chunk = unsafe { std::slice::from_raw_parts(samples, n as usize) };
                cb(chunk)
            }
            _ => true,
        }
    }));
    match keep_going {
        Ok(true) => 1,
        Ok(false) => {
            state.stopped = true;
            0
        }
        Err(_) => {
            state.panicked = true;
            state.stopped = true;
            0
        }
    }
}

impl OfflineTts {
    /// Load the voice (blocking).
    pub fn new(rt: &Arc<SherpaRuntime>, config: &TtsConfig) -> Result<Self, String> {
        config.validate()?;
        let mut arena = CStrArena::default();
        let raw_cfg = config.to_raw(&mut arena)?;
        // espeak-ng initialisation inside the constructor touches the same global state.
        let _gen = GENERATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: `raw_cfg` points into `arena`, alive for the call; the C API copies everything.
        let raw = unsafe { (rt.api().SherpaOnnxCreateOfflineTts)(&raw_cfg) };
        let raw = NonNull::new(raw.cast_mut()).ok_or_else(|| {
            format!(
                "sherpa-onnx could not load the {} voice; check that the files belong together and match \
                 sherpa-onnx {}",
                config.model.family(),
                rt.info().version
            )
        })?;
        // SAFETY: valid engine.
        let (rate, speakers) = unsafe {
            (
                (rt.api().SherpaOnnxOfflineTtsSampleRate)(raw.as_ptr()),
                (rt.api().SherpaOnnxOfflineTtsNumSpeakers)(raw.as_ptr()),
            )
        };
        let tts = Self {
            rt: rt.clone(),
            raw,
            lock: CallLock::default(),
            sample_rate: rate.max(0) as u32,
            num_speakers: speakers.max(0) as u32,
            config: config.clone(),
        };
        if tts.sample_rate == 0 {
            return Err(format!("the {} voice reports no sample rate", config.model.family()));
        }
        Ok(tts)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Voices / speakers of the model (1 for single-speaker models).
    pub fn num_speakers(&self) -> u32 {
        self.num_speakers
    }

    pub fn config(&self) -> &TtsConfig {
        &self.config
    }

    /// Synthesize `text` (blocking). `on_chunk` receives each generated batch as it is ready and
    /// returns `false` to stop; `cancel` stops at the next batch boundary. A stop keeps the audio
    /// generated so far (`cancelled = true`). A `cancel` already set (before the call, or while it
    /// waited for the process-wide TTS lock) returns at once with no audio and `cancelled = true`,
    /// without synthesizing anything. The callback runs on this thread while generation
    /// holds the process-wide TTS lock: hand the samples off (channel / ring buffer) and never call
    /// back into a sherpa TTS from it.
    pub fn generate<'f>(
        &self,
        text: &str,
        options: &GenerateOptions,
        cancel: Option<&AtomicBool>,
        on_chunk: Option<&'f mut ChunkCallback<'f>>,
    ) -> Result<GeneratedAudio, String> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(GeneratedAudio {
                samples: Vec::new(),
                sample_rate: self.sample_rate,
                cancelled: false,
            });
        }
        if self.num_speakers > 0 && options.sid >= self.num_speakers {
            return Err(format!(
                "voice {} does not exist (this model has {} voices)",
                options.sid, self.num_speakers
            ));
        }
        if !(options.speed.is_finite() && options.speed > 0.0) {
            return Err(format!("invalid speech rate {}", options.speed));
        }
        let mut arena = CStrArena::default();
        let c_text = arena.str(text)?;
        let gen_cfg = SherpaOnnxGenerationConfig {
            silence_scale: silence_scale(options.silence_scale.or(self.config.silence_scale)),
            speed: options.speed.clamp(0.25, 4.0),
            sid: options.sid as i32,
            ..SherpaOnnxGenerationConfig::default()
        };
        let mut state = CallbackState {
            cancel,
            on_chunk,
            stopped: false,
            panicked: false,
        };
        let cancelled_early = || GeneratedAudio {
            samples: Vec::new(),
            sample_rate: self.sample_rate,
            cancelled: true,
        };
        // Skipped utterances never synthesize a first batch (seconds of CPU on Kokoro int8) or hold
        // the process-wide lock another request is waiting for.
        if is_cancelled(cancel) {
            return Ok(cancelled_early());
        }
        let api = self.rt.api();
        let _gen = GENERATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if is_cancelled(cancel) {
            return Ok(cancelled_early());
        }
        let _g = self.lock.lock();
        // SAFETY: valid engine, C string and config alive for the call; `state` outlives the call
        // and the callback only runs during it (on this thread).
        let audio = unsafe {
            (api.SherpaOnnxOfflineTtsGenerateWithConfig)(
                self.raw.as_ptr(),
                c_text,
                &gen_cfg,
                Some(on_progress),
                (&mut state as *mut CallbackState<'_, '_>).cast::<c_void>(),
            )
        };
        if state.panicked {
            if !audio.is_null() {
                // SAFETY: returned by the generate call above, freed once.
                unsafe { (api.SherpaOnnxDestroyOfflineTtsGeneratedAudio)(audio) };
            }
            return Err("the TTS chunk callback panicked".into());
        }
        if audio.is_null() {
            if state.stopped {
                return Ok(GeneratedAudio {
                    samples: Vec::new(),
                    sample_rate: self.sample_rate,
                    cancelled: true,
                });
            }
            return Err(format!(
                "sherpa-onnx produced no audio for {:?} (unsupported characters for this voice?)",
                text.chars().take(40).collect::<String>()
            ));
        }
        // SAFETY: non-null result; samples copied before it is destroyed.
        let out = unsafe {
            let a = &*audio;
            let samples = floats(a.samples, a.n);
            let rate = if a.sample_rate > 0 { a.sample_rate as u32 } else { self.sample_rate };
            (api.SherpaOnnxDestroyOfflineTtsGeneratedAudio)(audio);
            GeneratedAudio {
                samples,
                sample_rate: rate,
                cancelled: state.stopped,
            }
        };
        Ok(out)
    }
}

fn is_cancelled(cancel: Option<&AtomicBool>) -> bool {
    cancel.is_some_and(|c| c.load(Ordering::Relaxed))
}
