//! Kyutai Pocket TTS: voice cloning on the CPU, in process (P6-I). The three commands are frozen by
//! P6-S0 and registered in [`super::commands`] (`plugin:local-tts|pocket_tts_*`).
//!
//! - **Runtime.** sherpa-onnx 1.13.8's Pocket implementation through its C API ([`sherpa`]), the
//!   same runtime pack (`runtime/sherpa-onnx`) and process as Kokoro; CPU only, two threads by
//!   default (Pocket is a 100M-parameter model built for two cores). Models are the sherpa packages
//!   of `model-catalog/tts-cloning.json` ([`MODELS`]; int8 is the default, fp32 the alternative).
//! - **Voices.** A request's `referenceVoiceId` is a shared voice-library voice
//!   (`voice_library::reference_clip`, decoded at 24 kHz; the voice in use stays decoded with the model);
//!   `""` speaks with the package's first sample voice (`test_wavs/`). sherpa caches the voice
//!   embedding itself (see [`sherpa`]), so nothing is written to the library for Pocket
//!   ([`CACHE_BACKEND`] is reserved for a future embedding cache).
//! - **Worker.** Requests run on the shared cloning worker ([`worker`], also used by
//!   `chatterbox_ort`): ordered, cancellable, an empty text (Start) loads the model, prepares the
//!   voice and holds the model until `pocket_tts_unload` (Stop); a model loaded only for a panel
//!   preview is freed after [`IDLE_UNLOAD`] without requests.
//! - **Options** (`request.options`): `temperature` (0.05-2, default 0.7), `seed` (integer; none =
//!   random per sentence), `steps` (flow decode steps 1-16, default sherpa's 5). Pocket has no
//!   speech-rate control (`speed` is ignored) and speaks English.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use once_cell::sync::Lazy;
use tauri::command;

use super::commands::{CloningTtsRequest, SpeakReport};
use super::engine::AppSink;
use crate::services::neural_audio::vc::families::reference::ReferenceClip;

mod sherpa;
pub(crate) mod worker;

#[cfg(test)]
mod tests;

use worker::{answer, option_f32, option_i64, CloneBackend, CloneWorker};

/// Voice-library cache backend name (`voices/<id>/cache/pocket-tts.bin`), reserved: sherpa keeps
/// Pocket's voice embedding internally and does not expose it, so nothing is cached yet.
#[allow(dead_code)] // Frozen by P6-S0; kept for a future embedding cache.
pub const CACHE_BACKEND: &str = "pocket-tts";
/// Default model (`model-catalog/tts-cloning.json`).
pub const DEFAULT_MODEL: &str = "tts/pocket-tts-en-int8";
/// Catalog ids this engine runs.
pub const MODELS: [&str; 2] = [DEFAULT_MODEL, "tts/pocket-tts-en"];
/// A loaded model is freed after this long without requests, unless a Start holds it.
pub const IDLE_UNLOAD: Duration = Duration::from_secs(10 * 60);
/// Warm-up phrase (English).
const WARM_UP_TEXT: &str = "Hello.";

/// The model of a request: [`DEFAULT_MODEL`] for `""`, else one of [`MODELS`].
pub fn resolve_model(model: &str) -> Result<&'static str, String> {
    let m = model.trim();
    if m.is_empty() {
        return Ok(DEFAULT_MODEL);
    }
    MODELS
        .iter()
        .copied()
        .find(|id| *id == m)
        .ok_or_else(|| format!("{m} is not a Pocket TTS model (expected one of {})", MODELS.join(", ")))
}

/// Inference threads of a request (`None` = the default; 1-4).
pub fn resolve_threads(threads: Option<u32>) -> u32 {
    threads.unwrap_or(sherpa::DEFAULT_THREADS).clamp(1, 4)
}

/// The generation options of a request (see the module docs).
pub fn options_of(request: &CloningTtsRequest) -> sherpa::PocketOptions {
    let o = &request.options;
    let defaults = sherpa::PocketOptions::default();
    sherpa::PocketOptions {
        temperature: option_f32(o, "temperature")
            .map(|t| t.clamp(sherpa::TEMPERATURE_RANGE.0, sherpa::TEMPERATURE_RANGE.1))
            .unwrap_or(defaults.temperature),
        seed: option_i64(o, "seed").filter(|s| *s >= 0),
        steps: option_i64(o, "steps")
            .filter(|s| *s > 0)
            .map(|s| (s.min(i64::from(sherpa::STEPS_RANGE.1)) as u32).max(sherpa::STEPS_RANGE.0)),
    }
}

/// A reference voice decoded for the model.
struct PreparedVoice {
    /// Voice-library id, or `""` for the built-in voice.
    id: String,
    /// sha256 of the clip file.
    sha256: String,
    samples: Arc<Vec<f32>>,
}

/// The loaded Pocket model.
pub(crate) struct PocketModel {
    tts: sherpa::PocketTts,
    dir: PathBuf,
    /// The last voice used (a voice switch decodes the new clip once).
    voice: Option<PreparedVoice>,
}

/// Where library voices come from (the app's data folder; tests use their own).
#[derive(Clone, Debug, Default)]
pub(crate) struct PocketBackend {
    app_data: Option<PathBuf>,
}

impl PocketBackend {
    /// The app's backend (voices from the app data folder).
    pub fn app() -> Self {
        Self { app_data: None }
    }

    #[cfg(test)]
    pub fn with_app_data(dir: &Path) -> Self {
        Self {
            app_data: Some(dir.to_path_buf()),
        }
    }

    fn app_data(&self) -> Result<PathBuf, String> {
        self.app_data
            .clone()
            .or_else(crate::services::diagnostics::app_data_dir)
            .ok_or_else(|| "the voice library is not available (no app data folder)".to_string())
    }

    /// The clip of `request`'s voice: a library voice, or the package's first sample voice.
    fn reference_clip(&self, dir: &Path, voice_id: &str) -> Result<ReferenceClip, String> {
        let id = voice_id.trim();
        if id.is_empty() {
            let wav = sherpa::builtin_voices(dir)
                .into_iter()
                .next()
                .ok_or("this Pocket TTS package has no built-in voice; choose a voice from your voice library")?;
            return ReferenceClip::from_wav(&wav);
        }
        if !crate::services::voice_library::valid_voice_id(id) {
            return Err(format!("invalid voice id {id:?}"));
        }
        crate::services::voice_library::reference_clip(&self.app_data()?, id)
    }

    /// The decoded reference of `voice_id` (cached in the model while the clip is unchanged).
    fn voice(&self, model: &mut PocketModel, voice_id: &str) -> Result<Arc<Vec<f32>>, String> {
        let clip = self.reference_clip(&model.dir, voice_id)?;
        if let Some(v) = &model.voice {
            if v.id == voice_id.trim() && v.sha256 == clip.sha256 {
                return Ok(v.samples.clone());
            }
        }
        let samples = clip.load_mono(sherpa::SAMPLE_RATE)?;
        if samples.len() < sherpa::SAMPLE_RATE as usize / 2 {
            return Err(format!("the reference voice {:?} is too short", clip.name));
        }
        let samples = Arc::new(samples);
        model.voice = Some(PreparedVoice {
            id: voice_id.trim().to_string(),
            sha256: clip.sha256,
            samples: samples.clone(),
        });
        Ok(samples)
    }
}

impl CloneBackend for PocketBackend {
    type Model = PocketModel;

    fn name(&self) -> &'static str {
        "pocket"
    }

    fn key(&self, request: &CloningTtsRequest) -> Result<String, String> {
        Ok(format!("{}|t{}", resolve_model(&request.model)?, resolve_threads(request.threads)))
    }

    fn load(&self, request: &CloningTtsRequest, _cancel: &AtomicBool) -> Result<PocketModel, String> {
        let id = resolve_model(&request.model)?;
        let dir = crate::services::model_store::installed_path(id)
            .ok_or_else(|| format!("{id} is not installed; download it in the Pocket TTS settings or Models & Runtimes"))?;
        let files = sherpa::PocketFiles::discover(&dir, id == DEFAULT_MODEL)?;
        let rt = crate::services::speech_rt::sherpa::SherpaRuntime::installed()?;
        let tts = sherpa::PocketTts::new(&rt, &files, Some(resolve_threads(request.threads)))?;
        Ok(PocketModel { tts, dir, voice: None })
    }

    fn warm_up(&self, model: &mut PocketModel, request: &CloningTtsRequest, cancel: &AtomicBool) -> Result<(), String> {
        let reference = self.voice(model, &request.reference_voice_id)?;
        // One short phrase: sherpa encodes the voice (cached afterwards) and ONNX Runtime sets up
        // its kernels, so the first sentence starts as fast as the next ones.
        let started = std::time::Instant::now();
        let mut discard = |_: &[f32]| true;
        model
            .tts
            .generate(WARM_UP_TEXT, &reference, sherpa::SAMPLE_RATE, &options_of(request), cancel, &mut discard)?;
        tracing::debug!(target: "sigil::tts", "Pocket TTS: warm-up took {} ms", started.elapsed().as_millis());
        Ok(())
    }

    fn synthesize(
        &self,
        model: &mut PocketModel,
        request: &CloningTtsRequest,
        text: &str,
        cancel: &AtomicBool,
        out: &mut dyn FnMut(&[f32], u32) -> bool,
    ) -> Result<bool, String> {
        let reference = self.voice(model, &request.reference_voice_id)?;
        let rate = model.tts.sample_rate();
        let mut on_chunk = |samples: &[f32]| out(samples, rate);
        model
            .tts
            .generate(text, &reference, sherpa::SAMPLE_RATE, &options_of(request), cancel, &mut on_chunk)
    }
}

/// The app's Pocket worker (thread started on the first request).
static WORKER: Lazy<CloneWorker<PocketBackend>> = Lazy::new(|| CloneWorker::new(Arc::new(PocketBackend::app()), Arc::new(AppSink), IDLE_UNLOAD));

/// Speak one sentence into the begun `tts_sink` utterance `utterance_id`; `Err("cancelled")` when
/// it was cancelled. An empty `request.text` only loads the model and prepares the voice (Start);
/// `utterance_id` may then be empty.
#[command]
pub async fn pocket_tts_speak(request: CloningTtsRequest, utterance_id: String) -> Result<SpeakReport, String> {
    answer(WORKER.speak(request, utterance_id)).await
}

/// Cancel one request (or every request when `utterance_id` is `None`).
#[command]
pub async fn pocket_tts_cancel(utterance_id: Option<String>) -> Result<(), String> {
    WORKER.cancel(utterance_id.as_deref());
    Ok(())
}

/// Free the model.
#[command]
pub async fn pocket_tts_unload() -> Result<(), String> {
    answer(WORKER.unload()).await
}
