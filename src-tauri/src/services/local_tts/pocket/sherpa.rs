//! Kyutai Pocket TTS through sherpa-onnx's C API (`SherpaOnnxOfflineTtsPocketModelConfig`,
//! sherpa-onnx 1.13.8, the same runtime pack as Kokoro): the vendored bindings, the runtime and the
//! arena / lock helpers of `speech_rt::sherpa` with a Pocket-specific config and the generation
//! options sherpa's Pocket implementation reads (`offline-tts-pocket-impl.h`):
//!
//! - The reference voice is passed as audio on every call (`SherpaOnnxGenerationConfig::
//!   reference_audio`); sherpa encodes it once and keeps the embedding in its own cache
//!   (`voice_embedding_cache_capacity`, keyed by a hash of the samples), so passing the same samples
//!   again costs nothing. The C API does not expose the embedding, so nothing is stored in the
//!   voice library for Pocket.
//! - Options go in the `extra` JSON: `temperature` (default 0.7), `seed` (-1 = random),
//!   `max_reference_audio_len` (seconds; longer references are cut). `num_steps` is the flow's
//!   decode steps (0 = sherpa's 5).
//! - Audio arrives through the progress callback one decoder chunk (15 frames, 1.2 s) at a time,
//!   after the language model has generated the whole sentence. The pause scale is 1 so the
//!   streamed chunks equal the returned audio (sherpa rescales pauses only in the returned buffer).
//! - Pocket has no speech-rate control and speaks English (the 2026-01-26 model).

use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::services::speech_rt::sherpa::bindings::{SherpaOnnxGenerationConfig, SherpaOnnxOfflineTts, SherpaOnnxOfflineTtsConfig};
use crate::services::speech_rt::sherpa::lib::{floats, rate_i32, require_files, threads_or, CStrArena, CallLock, CPU_PROVIDER};
use crate::services::speech_rt::sherpa::SherpaRuntime;

/// Pocket TTS output rate (sherpa's `SampleRate()` for Pocket).
pub const SAMPLE_RATE: u32 = 24_000;
/// Default inference threads (Pocket is built for two cores).
pub const DEFAULT_THREADS: u32 = 2;
/// Voice embeddings sherpa keeps (one per distinct reference clip).
const VOICE_CACHE_CAPACITY: i32 = 8;
/// Longest reference sherpa encodes (seconds); the voice library keeps clips of at most 10 s.
pub const MAX_REFERENCE_SECONDS: f32 = 10.0;

/// The files of a Pocket TTS package (`sherpa-onnx-pocket-tts-*`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PocketFiles {
    pub lm_flow: PathBuf,
    pub lm_main: PathBuf,
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub text_conditioner: PathBuf,
    pub vocab_json: PathBuf,
    pub token_scores_json: PathBuf,
}

/// `<stem>.int8.onnx` or `<stem>.onnx` (in the preferred order) when present, else any
/// `<stem>*.onnx` (sorted).
fn find_graph(dir: &Path, stem: &str, prefer_int8: bool) -> Option<PathBuf> {
    let (int8, full) = (format!("{stem}.int8.onnx"), format!("{stem}.onnx"));
    let names = if prefer_int8 { [int8, full] } else { [full, int8] };
    for name in names {
        let p = dir.join(&name);
        if p.is_file() {
            return Some(p);
        }
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(stem) && n.ends_with(".onnx"))
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

impl PocketFiles {
    /// Find the package files in `dir`; with both precisions present, `prefer_int8` picks.
    pub fn discover(dir: &Path, prefer_int8: bool) -> Result<Self, String> {
        let graph =
            |stem: &str| find_graph(dir, stem, prefer_int8).ok_or_else(|| format!("the Pocket TTS package in {} has no {stem} model", dir.display()));
        let file = |name: &str| {
            let p = dir.join(name);
            if p.is_file() {
                Ok(p)
            } else {
                Err(format!("the Pocket TTS package in {} has no {name}", dir.display()))
            }
        };
        Ok(Self {
            lm_flow: graph("lm_flow")?,
            lm_main: graph("lm_main")?,
            encoder: graph("encoder")?,
            decoder: graph("decoder")?,
            text_conditioner: graph("text_conditioner")?,
            vocab_json: file("vocab.json")?,
            token_scores_json: file("token_scores.json")?,
        })
    }

    fn all(&self) -> [&Path; 7] {
        [
            &self.lm_flow,
            &self.lm_main,
            &self.encoder,
            &self.decoder,
            &self.text_conditioner,
            &self.vocab_json,
            &self.token_scores_json,
        ]
    }
}

/// The package's sample voices (`test_wavs/*.wav`, sorted by name): Pocket's built-in voices.
pub fn builtin_voices(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir.join("test_wavs"))
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_file() && p.extension().is_some_and(|e| e.eq_ignore_ascii_case("wav")))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// Per-call options (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PocketOptions {
    /// Sampling temperature of the latent flow (sherpa default 0.7).
    pub temperature: f32,
    /// Random seed; `None` = a new one per call.
    pub seed: Option<i64>,
    /// Flow decode steps; `None` = sherpa's default (5).
    pub steps: Option<u32>,
}

impl Default for PocketOptions {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            seed: None,
            steps: None,
        }
    }
}

/// Temperature range accepted (sherpa takes its square root as the noise deviation).
pub const TEMPERATURE_RANGE: (f32, f32) = (0.05, 2.0);
/// Decode steps accepted.
pub const STEPS_RANGE: (u32, u32) = (1, 16);

impl PocketOptions {
    /// The `extra` JSON of the generation config.
    pub fn extra_json(&self) -> String {
        let t = if self.temperature.is_finite() {
            self.temperature
                .clamp(TEMPERATURE_RANGE.0, TEMPERATURE_RANGE.1)
        } else {
            0.7
        };
        let seed = self.seed.map_or(-1, |s| s.clamp(0, i64::from(i32::MAX)));
        serde_json::json!({
            "temperature": t,
            "seed": seed,
            "max_reference_audio_len": MAX_REFERENCE_SECONDS,
        })
        .to_string()
    }

    fn num_steps(&self) -> i32 {
        self.steps
            .map_or(0, |s| s.clamp(STEPS_RANGE.0, STEPS_RANGE.1) as i32)
    }
}

/// A loaded Pocket TTS model. `Send + Sync`: every native call holds `lock`.
pub struct PocketTts {
    rt: Arc<SherpaRuntime>,
    raw: NonNull<SherpaOnnxOfflineTts>,
    lock: CallLock,
    sample_rate: u32,
}

// SAFETY: no thread affinity; every call into the engine holds `lock`.
unsafe impl Send for PocketTts {}
unsafe impl Sync for PocketTts {}

impl Drop for PocketTts {
    fn drop(&mut self) {
        let _g = self.lock.lock();
        // SAFETY: created by SherpaOnnxCreateOfflineTts, destroyed once.
        unsafe { (self.rt.api().SherpaOnnxDestroyOfflineTts)(self.raw.as_ptr()) }
    }
}

/// Callback state of one generation.
struct CallbackState<'f> {
    cancel: &'f AtomicBool,
    on_chunk: &'f mut dyn FnMut(&[f32]) -> bool,
    pushed: usize,
    stopped: bool,
    panicked: bool,
}

/// `SherpaOnnxGeneratedAudioProgressCallbackWithArg`: forwards each decoder chunk. Never unwinds
/// into C (a panic stops generation and is reported afterwards).
unsafe extern "C" fn on_progress(samples: *const f32, n: i32, _progress: f32, arg: *mut c_void) -> i32 {
    if arg.is_null() {
        return 1;
    }
    // SAFETY: `arg` is the `CallbackState` owned by `generate`, alive for the whole native call.
    let state = unsafe { &mut *(arg as *mut CallbackState<'_>) };
    let keep_going = catch_unwind(AssertUnwindSafe(|| {
        if state.cancel.load(Ordering::Relaxed) {
            return false;
        }
        if samples.is_null() || n <= 0 {
            return true;
        }
        // SAFETY: sherpa passes `n` valid samples for the duration of the callback.
        let chunk = unsafe { std::slice::from_raw_parts(samples, n as usize) };
        state.pushed += chunk.len();
        (state.on_chunk)(chunk)
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

impl PocketTts {
    /// Load the model (blocking) on `threads` CPU threads (1-4).
    pub fn new(rt: &Arc<SherpaRuntime>, files: &PocketFiles, threads: Option<u32>) -> Result<Self, String> {
        require_files("Pocket TTS", files.all())?;
        let mut arena = CStrArena::default();
        let mut c = SherpaOnnxOfflineTtsConfig::default();
        let p = &mut c.model.pocket;
        p.lm_flow = arena.path(&files.lm_flow)?;
        p.lm_main = arena.path(&files.lm_main)?;
        p.encoder = arena.path(&files.encoder)?;
        p.decoder = arena.path(&files.decoder)?;
        p.text_conditioner = arena.path(&files.text_conditioner)?;
        p.vocab_json = arena.path(&files.vocab_json)?;
        p.token_scores_json = arena.path(&files.token_scores_json)?;
        p.voice_embedding_cache_capacity = VOICE_CACHE_CAPACITY;
        c.model.num_threads = threads_or(threads, DEFAULT_THREADS);
        c.model.debug = 0;
        c.model.provider = CPU_PROVIDER.as_ptr();
        c.max_num_sentences = 1;
        c.silence_scale = 1.0;
        // SAFETY: `c` points into `arena`, alive for the call; the C API copies every string.
        let raw = unsafe { (rt.api().SherpaOnnxCreateOfflineTts)(&c) };
        let raw = NonNull::new(raw.cast_mut()).ok_or_else(|| {
            format!(
                "sherpa-onnx could not load the Pocket TTS model; check that the files belong together and match \
                 sherpa-onnx {}",
                rt.info().version
            )
        })?;
        // SAFETY: valid engine.
        let rate = unsafe { (rt.api().SherpaOnnxOfflineTtsSampleRate)(raw.as_ptr()) };
        let tts = Self {
            rt: rt.clone(),
            raw,
            lock: CallLock::default(),
            sample_rate: rate.max(0) as u32,
        };
        if tts.sample_rate == 0 {
            return Err("the Pocket TTS model reports no sample rate".into());
        }
        Ok(tts)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Synthesize `text` in the voice of `reference` (mono, `reference_rate` Hz, at most
    /// [`MAX_REFERENCE_SECONDS`] used), blocking. `on_chunk` receives the audio as it is decoded
    /// and returns `false` to stop; `cancel` stops at the next chunk. `Ok(false)` = stopped early.
    pub fn generate(
        &self,
        text: &str,
        reference: &[f32],
        reference_rate: u32,
        options: &PocketOptions,
        cancel: &AtomicBool,
        on_chunk: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<bool, String> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(true);
        }
        if reference.is_empty() {
            return Err("Pocket TTS needs a reference voice".into());
        }
        if cancel.load(Ordering::Acquire) {
            return Ok(false);
        }
        // sherpa encodes at most this much; never hand it more than that.
        let keep = ((MAX_REFERENCE_SECONDS * reference_rate as f32) as usize).min(reference.len());
        let reference = &reference[..keep];
        let mut arena = CStrArena::default();
        let c_text = arena.str(text)?;
        let c_extra = arena.str(&options.extra_json())?;
        let cfg = SherpaOnnxGenerationConfig {
            silence_scale: 1.0,
            speed: 1.0,
            sid: 0,
            reference_audio: reference.as_ptr(),
            reference_audio_len: i32::try_from(reference.len()).map_err(|_| "the reference voice is too long")?,
            reference_sample_rate: rate_i32(reference_rate)?,
            num_steps: options.num_steps(),
            extra: c_extra,
            ..SherpaOnnxGenerationConfig::default()
        };
        let mut state = CallbackState {
            cancel,
            on_chunk,
            pushed: 0,
            stopped: false,
            panicked: false,
        };
        let api = self.rt.api();
        let _g = self.lock.lock();
        // SAFETY: valid engine; the text, extra JSON, reference samples and config outlive the call;
        // `state` outlives the call and the callback only runs during it (on this thread).
        let audio = unsafe {
            (api.SherpaOnnxOfflineTtsGenerateWithConfig)(
                self.raw.as_ptr(),
                c_text,
                &cfg,
                Some(on_progress),
                (&mut state as *mut CallbackState<'_>).cast::<c_void>(),
            )
        };
        // The audio the callback did not deliver (none in practice: sherpa calls back for every
        // chunk). Copied before the native buffer is freed.
        let rest = if audio.is_null() {
            Vec::new()
        } else {
            // SAFETY: non-null result of the call above; copied, then destroyed once.
            unsafe {
                let a = &*audio;
                let all = floats(a.samples, a.n);
                (api.SherpaOnnxDestroyOfflineTtsGeneratedAudio)(audio);
                all
            }
        };
        if state.panicked {
            return Err("the Pocket TTS audio callback panicked".into());
        }
        if state.stopped {
            return Ok(false);
        }
        if audio.is_null() && state.pushed == 0 {
            return Err(format!(
                "Pocket TTS produced no audio for {:?}",
                text.chars().take(40).collect::<String>()
            ));
        }
        if state.pushed == 0 && !rest.is_empty() && !(state.on_chunk)(&rest) {
            return Ok(false);
        }
        Ok(true)
    }
}
