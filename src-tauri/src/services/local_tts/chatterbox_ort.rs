//! Chatterbox Multilingual V3 on Sigil's ONNX Runtime: the in-process voice-cloning fallback in 23
//! languages (P6-I). The three commands are frozen by P6-S0 and registered in [`super::commands`]
//! (`plugin:local-tts|chatterbox_local_*`). The older `chatterbox` backend (a user-run HTTP server)
//! is unrelated and stays.
//!
//! - **Model.** Resemble AI's Chatterbox Multilingual V3 (MIT; T3 0.5B Llama backbone, S3Gen flow,
//!   HiFT vocoder) as KitsuMate's split-speed ONNX export (`tts/chatterbox-mtl-v3` in
//!   `model-catalog/tts-cloning.json`, about 1.75 GB; `tts/chatterbox-mtl-v3-gpu-flow` adds the
//!   fp16 flow step for graphics cards). How it runs: [`model`]; the text frontend: [`text`]; the
//!   language model and its in-place cache: [`lm`]; token sampling: [`sampling`].
//! - **Voices.** Always a shared voice-library voice (`referenceVoiceId`; the model has no built-in
//!   voice): the speech encoder turns its clip into conditioning once, cached in the library under
//!   [`CACHE_BACKEND`] (and the newest four in memory).
//! - **Execution.** `options.ep`: `auto` (CUDA on NVIDIA, else DirectML on the discrete card with
//!   the most memory when it has 2 GB or more, never integrated graphics, else the CPU), `cuda`,
//!   `dml` (the discrete card with the most memory) or `cpu`; every session goes through
//!   `ort_session`'s ladder on that one device and records its outcome (`tts.chatterbox`,
//!   `tts.chatterbox.flow`, `tts.chatterbox.vocoder`).
//! - **Lifetime.** A Start (an empty-text request) loads the model and holds it until
//!   `chatterbox_local_unload` (Stop), however long the gaps between sentences: a reload costs
//!   seconds on a GPU and much longer on the CPU. A model loaded only for a panel preview is freed
//!   after [`IDLE_UNLOAD`].
//! - **Options** (`request.options`, the reference implementation's defaults): `exaggeration`
//!   (0.25-2, 0.5), `cfgWeight` (0-1.5, 0.5; 0 turns guidance off, which Resemble recommends when
//!   the reference voice speaks another language), `temperature` (0.05-2, 0.8),
//!   `repetitionPenalty` (1-3, 2.0), `minP` (0-1, 0.05), `topP` (0.05-1, 1), `flowSteps` (1-32;
//!   default 10 on a GPU, 6 on the CPU), `seed` (integer; none = random): it seeds the speech-token
//!   sampling and the flow's starting noise (drawn in Rust instead of by the export's unseeded
//!   `RandomNormalLike`), so a seeded sentence comes out the same on the same device. `speed` is
//!   ignored: the model has no rate control.
//! - **Language.** `request.language` when Chatterbox speaks it, else the script of the sentence,
//!   else English (Latin-script languages need the tag: the TS backend passes the caption's
//!   language or the panel's choice).
//! - **Known limits.** The export has no attention outputs, so the reference implementation's
//!   alignment guard (which forces the stop token when the speech runs past the text) is replaced by
//!   a length limit from the text ([`text::max_speech_tokens`]). The model does not stream: each
//!   sentence (or first clause) plays once it is fully generated.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use once_cell::sync::Lazy;
use tauri::command;

use super::commands::{CloningTtsRequest, SpeakReport};
use super::engine::AppSink;
use super::pocket::worker::{answer, option_f32, option_i64, option_str, CloneBackend, CloneWorker};

mod embed;
mod lm;
pub(crate) mod model;
mod sampling;
pub(crate) mod text;
mod voice;

#[cfg(test)]
mod testonnx;
#[cfg(test)]
mod tests;

use model::{ChatterboxModel, GenOptions, ModelFiles};
use sampling::SamplingParams;

/// Voice-library cache backend name (`voices/<id>/cache/chatterbox-ort.bin`).
pub const CACHE_BACKEND: &str = "chatterbox-ort";
/// The model (`model-catalog/tts-cloning.json`).
pub const MODEL_ID: &str = "tts/chatterbox-mtl-v3";
/// The optional fp16 flow step for graphics cards.
pub const GPU_FLOW_ID: &str = "tts/chatterbox-mtl-v3-gpu-flow";
/// A loaded model is freed after this long without requests (it holds 2-3 GB), unless a Start
/// holds it (the worker keeps a started backend's model until `chatterbox_local_unload`).
pub const IDLE_UNLOAD: Duration = Duration::from_secs(10 * 60);

/// The model of a request (`""` = [`MODEL_ID`]).
pub fn resolve_model(model: &str) -> Result<&'static str, String> {
    match model.trim() {
        "" | MODEL_ID => Ok(MODEL_ID),
        other => Err(format!("{other} is not a Chatterbox Multilingual model (expected {MODEL_ID})")),
    }
}

/// CPU threads of a request (`None` = [`model::DEFAULT_THREADS`]; 1-8).
pub fn resolve_threads(threads: Option<u32>) -> u32 {
    threads.unwrap_or(model::DEFAULT_THREADS).clamp(1, 8)
}

fn clamp_or(v: Option<f32>, lo: f32, hi: f32, default: f32) -> f32 {
    v.map_or(default, |x| x.clamp(lo, hi))
}

/// The generation options of a request (see the module docs).
pub fn options_of(request: &CloningTtsRequest) -> GenOptions {
    let o = &request.options;
    let d = GenOptions::default();
    let s = SamplingParams::default();
    GenOptions {
        exaggeration: clamp_or(option_f32(o, "exaggeration"), 0.25, 2.0, d.exaggeration),
        sampling: SamplingParams {
            cfg_weight: clamp_or(option_f32(o, "cfgWeight"), 0.0, 1.5, s.cfg_weight),
            temperature: clamp_or(option_f32(o, "temperature"), 0.05, 2.0, s.temperature),
            repetition_penalty: clamp_or(option_f32(o, "repetitionPenalty"), 1.0, 3.0, s.repetition_penalty),
            min_p: clamp_or(option_f32(o, "minP"), 0.0, 1.0, s.min_p),
            top_p: clamp_or(option_f32(o, "topP"), 0.05, 1.0, s.top_p),
        },
        flow_steps: option_i64(o, "flowSteps")
            .filter(|&n| n > 0)
            .map(|n| n.min(32) as usize),
        flow_guidance: d.flow_guidance,
        seed: option_i64(o, "seed").filter(|&n| n >= 0).map(|n| n as u64),
    }
}

/// Where library voices come from (the app's data folder; tests use their own).
#[derive(Clone, Debug, Default)]
pub(crate) struct ChatterboxBackend {
    app_data: Option<PathBuf>,
    /// Model folders for tests (`None` = the model store).
    #[cfg(test)]
    dirs: Option<(PathBuf, Option<PathBuf>)>,
}

impl ChatterboxBackend {
    pub fn app() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub fn for_test(app_data: &std::path::Path, model_dir: &std::path::Path) -> Self {
        Self {
            app_data: Some(app_data.to_path_buf()),
            dirs: Some((model_dir.to_path_buf(), None)),
        }
    }

    fn app_data(&self) -> Result<PathBuf, String> {
        self.app_data
            .clone()
            .or_else(crate::services::diagnostics::app_data_dir)
            .ok_or_else(|| "the voice library is not available (no app data folder)".to_string())
    }

    /// The installed model folder and the optional fp16 flow folder.
    fn dirs(&self, id: &str) -> Result<(PathBuf, Option<PathBuf>), String> {
        #[cfg(test)]
        if let Some(d) = &self.dirs {
            return Ok(d.clone());
        }
        let dir = crate::services::model_store::installed_path(id)
            .ok_or_else(|| format!("{id} is not installed; download it in the Chatterbox settings or Models & Runtimes"))?;
        Ok((dir, crate::services::model_store::installed_path(GPU_FLOW_ID)))
    }
}

impl CloneBackend for ChatterboxBackend {
    type Model = ChatterboxModel;

    fn name(&self) -> &'static str {
        "chatterbox"
    }

    fn key(&self, request: &CloningTtsRequest) -> Result<String, String> {
        // The resolved placement (EP and device), so a model follows the card `auto` / `dml` picks.
        let place = model::requested_placement(option_str(&request.options, "ep"));
        Ok(format!(
            "{}|{}|t{}",
            resolve_model(&request.model)?,
            place.label(),
            resolve_threads(request.threads)
        ))
    }

    fn load(&self, request: &CloningTtsRequest, _cancel: &AtomicBool) -> Result<ChatterboxModel, String> {
        let id = resolve_model(&request.model)?;
        let (dir, gpu_flow) = self.dirs(id)?;
        let files = ModelFiles::discover(&dir, gpu_flow.as_deref())?;
        let place = model::requested_placement(option_str(&request.options, "ep"));
        ChatterboxModel::load(files, place, resolve_threads(request.threads))
    }

    fn warm_up(&self, model: &mut ChatterboxModel, request: &CloningTtsRequest, cancel: &AtomicBool) -> Result<(), String> {
        let cond = model.conditionals(&self.app_data()?, &request.reference_voice_id)?;
        if model.on_gpu() {
            // One short sentence sets up the GPU kernels so the first real one starts promptly
            // (on the CPU that costs seconds and gains little, so it is skipped there).
            let mut opts = options_of(request);
            opts.flow_steps = Some(2);
            let started = std::time::Instant::now();
            model.synthesize("Hi.", "en", &cond, &opts, cancel)?;
            tracing::debug!(
                target: "sigil::tts",
                "Chatterbox: warm-up on {} took {} ms",
                model.lm_ep(),
                started.elapsed().as_millis()
            );
        }
        Ok(())
    }

    fn synthesize(
        &self,
        model: &mut ChatterboxModel,
        request: &CloningTtsRequest,
        text: &str,
        cancel: &AtomicBool,
        out: &mut dyn FnMut(&[f32], u32) -> bool,
    ) -> Result<bool, String> {
        let cond = model.conditionals(&self.app_data()?, &request.reference_voice_id)?;
        let lang = text::resolve_language(request.language.as_deref(), text);
        speak_text(model, text, lang, &cond, &options_of(request), cancel, out, 0)
    }
}

/// Speak `text` in `lang`, splitting it in two (at a clause, a space, else anywhere) while it has
/// more text tokens than one run takes; each piece's audio goes to `out` as soon as it is ready.
/// `Ok(false)` = cancelled or `out` stopped.
#[allow(clippy::too_many_arguments)]
fn speak_text(
    model: &mut ChatterboxModel,
    text: &str,
    lang: &str,
    cond: &voice::Conditionals,
    opts: &GenOptions,
    cancel: &AtomicBool,
    out: &mut dyn FnMut(&[f32], u32) -> bool,
    depth: u32,
) -> Result<bool, String> {
    match model.synthesize(text, lang, cond, opts, cancel) {
        Ok(None) => Ok(false),
        Ok(Some(audio)) if audio.is_empty() => Ok(true),
        Ok(Some(audio)) => Ok(out(&audio, model::SAMPLE_RATE)),
        Err(e) if e.starts_with(text::TOO_LONG) && depth < 8 => {
            let half = text.chars().count().div_ceil(2);
            let pieces = super::text::split_long(text, half.max(1));
            if pieces.len() < 2 {
                return Err(e);
            }
            for piece in &pieces {
                if !speak_text(model, piece, lang, cond, opts, cancel, out, depth + 1)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Err(e) => Err(e),
    }
}

/// The app's Chatterbox worker (thread started on the first request).
static WORKER: Lazy<CloneWorker<ChatterboxBackend>> =
    Lazy::new(|| CloneWorker::new(Arc::new(ChatterboxBackend::app()), Arc::new(AppSink), IDLE_UNLOAD));

/// Speak one sentence into the begun `tts_sink` utterance `utterance_id`; `Err("cancelled")` when
/// it was cancelled. An empty `request.text` only loads the model and prepares the voice (Start);
/// `utterance_id` may then be empty.
#[command]
pub async fn chatterbox_local_speak(request: CloningTtsRequest, utterance_id: String) -> Result<SpeakReport, String> {
    answer(WORKER.speak(request, utterance_id)).await
}

/// Cancel one request (or every request when `utterance_id` is `None`).
#[command]
pub async fn chatterbox_local_cancel(utterance_id: Option<String>) -> Result<(), String> {
    WORKER.cancel(utterance_id.as_deref());
    Ok(())
}

/// Free the model and its GPU memory.
#[command]
pub async fn chatterbox_local_unload() -> Result<(), String> {
    answer(WORKER.unload()).await
}
