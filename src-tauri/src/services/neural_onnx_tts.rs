//! In-process ONNX speech TTS: Hub download or local **speech package** folder (see
//! `docs/OPERATOR_REFERENCE.md#local-speech-package-onnx`).
//! Implementation uses KittenTTS, vendored in `services/local_tts/kitten` (see its NOTICE).
//!
//! # Legacy (P2-H)
//! Since P2 the `neuralOnnx` TTS backend speaks through the `local-tts` plugin
//! (`services/local_tts`): its Hub setting becomes `hub:<repo>` (KittenML 0.8 repos run as the
//! catalog Kitten packages once installed) and its folder setting a folder model, both played
//! through the TTS sink with the top-level route. Nothing in the app calls this plugin any more;
//! it stays registered for one release so older frontends keep working, and can then be retired
//! (main.rs registration, `capabilities/neural-onnx-tts.json`, `plugin-commands/neural-onnx-tts.txt`).
//!
//! # Threading
//! Every command is `async`. Model loading (which may download from the Hugging Face Hub) and
//! synthesis run on Tauri's blocking pool via `spawn_blocking`, never on the main thread, so loading
//! Kitten no longer freezes the UI.
//!
//! Settings live in a small config mutex that is never held across I/O or inference; the model has
//! its own mutex, held while loading or synthesizing. Setters that change the model source bump a
//! `generation` counter instead of waiting for the model mutex: a model loaded for an older
//! generation is stale and is reloaded (lazily, on a blocking thread) by the next load or speak.
//! Setters therefore return immediately even while a long synthesis or download is running. A
//! setter that leaves the effective source unchanged (the inspector re-sends it on every mount)
//! does not bump the generation, so the loaded model stays.
//!
//! # Execution provider
//! Kitten sessions are built on the CPU EP through Sigil's ORT factory, so the UI's EP hint cannot
//! select a GPU yet. Each load records the truthful outcome under the `tts.kitten` diagnostics
//! component (requested = hint, active = `cpu`).
//!
//! # Hub models
//! A Hub repo already in the legacy hf-hub cache is used in place; otherwise its files are fetched
//! one at a time into `<app_data>/models/kitten/<org>--<name>/` (see `local_tts::kitten::hub`).

use crate::services::diagnostics::{record_ep_outcome, EpOutcome};
use crate::services::local_tts::kitten::{self, hub};
use crate::services::voice_changer::{push_aux_wav_bytes, VoiceChangerState};
use hound::{SampleFormat, WavSpec, WavWriter};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, TryLockError};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime, State,
};

const DEFAULT_HUB_ID: &str = "KittenML/kitten-tts-mini-0.8";
const DEFAULT_VOICE: &str = "Jasper";
const SUPPORTED_LOCAL_FORMAT: &str = "speech_package_v1";
/// Diagnostics component key (frozen by P0-S0).
const EP_COMPONENT: &str = "tts.kitten";

/// User-facing settings pushed by the TS service. Cheap to clone; never locked across I/O.
#[derive(Clone, Debug)]
struct TtsConfig {
    hub_id: String,
    voice: String,
    speed: f32,
    /// Last EP hint from the UI (`cpu` / `cuda` / `dml`). Recorded only: Kitten sessions run on the
    /// CPU EP, so the hint does not select a device yet.
    execution_provider_hint: String,
    /// When true, load from `local_model_dir` instead of Hugging Face.
    use_local: bool,
    local_model_dir: String,
    /// `speech_package_v1` today; other values reserved for future formats.
    local_package_format: String,
    /// Bumped whenever the model source changes; a model loaded for an older value is stale.
    generation: u64,
    /// Generation of the model currently loaded (`None` = nothing current is loaded).
    loaded_generation: Option<u64>,
    /// Voices of the current model (valid while `loaded_generation == Some(generation)`).
    voices: Vec<String>,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            hub_id: DEFAULT_HUB_ID.to_string(),
            voice: DEFAULT_VOICE.to_string(),
            speed: 1.0,
            execution_provider_hint: "cpu".to_string(),
            use_local: false,
            local_model_dir: String::new(),
            local_package_format: SUPPORTED_LOCAL_FORMAT.to_string(),
            generation: 0,
            loaded_generation: None,
            voices: Vec::new(),
        }
    }
}

impl TtsConfig {
    fn is_current_loaded(&self) -> bool {
        self.loaded_generation == Some(self.generation)
    }

    /// What [`load_model_from`] would load: the Hub id, or the local folder and its format. Fields
    /// of the inactive source are left out, so e.g. setting the package format while on the Hub
    /// does not count as a change.
    fn model_source(&self) -> ModelSource {
        if self.use_local {
            ModelSource::Local {
                dir: self.local_model_dir.trim().to_string(),
                format: self.local_package_format.trim().to_lowercase(),
            }
        } else {
            ModelSource::Hub(self.hub_id.clone())
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ModelSource {
    Hub(String),
    Local { dir: String, format: String },
}

struct LoadedModel {
    tts: kitten::KittenTTS,
    generation: u64,
}

#[derive(Clone)]
pub struct NeuralOnnxTtsState {
    config: Arc<Mutex<TtsConfig>>,
    model: Arc<Mutex<Option<LoadedModel>>>,
}

impl NeuralOnnxTtsState {
    fn new() -> Self {
        Self {
            config: Arc::new(Mutex::new(TtsConfig::default())),
            model: Arc::new(Mutex::new(None)),
        }
    }

    fn config(&self) -> MutexGuard<'_, TtsConfig> {
        self.config.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The model slot. A panic during a previous load or synthesis poisons the mutex; the model may
    /// then be half-initialized, so it is dropped and the next call reloads it.
    fn model(&self) -> MutexGuard<'_, Option<LoadedModel>> {
        match self.model.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                *guard = None;
                self.model.clear_poison();
                self.config().loaded_generation = None;
                guard
            }
        }
    }

    /// Apply a change to the model source and, when the effective source actually changed, mark any
    /// loaded model stale. Re-sending the current settings (the TTS inspector does on every mount)
    /// keeps the loaded model. Never waits for the model mutex; a stale model is freed now when
    /// idle, else by the next load.
    fn invalidate(&self, change: impl FnOnce(&mut TtsConfig)) {
        {
            let mut cfg = self.config();
            let before = cfg.model_source();
            change(&mut cfg);
            if cfg.model_source() == before {
                return;
            }
            cfg.generation = cfg.generation.wrapping_add(1);
            cfg.loaded_generation = None;
            cfg.voices.clear();
        }
        match self.model.try_lock() {
            Ok(mut guard) => *guard = None,
            Err(TryLockError::Poisoned(poisoned)) => {
                *poisoned.into_inner() = None;
                self.model.clear_poison();
            }
            Err(TryLockError::WouldBlock) => {} // busy: the next ensure_loaded replaces it
        }
    }

    fn set_hub_id(&self, hub_id: &str) -> Result<(), String> {
        let id = hub_id.trim().to_string();
        if id.is_empty() {
            return Err("hub_id must not be empty".into());
        }
        self.invalidate(|c| {
            c.hub_id = id;
            c.use_local = false;
        });
        Ok(())
    }

    fn set_model_source(&self, source: &str, local_dir: Option<String>) -> Result<(), String> {
        match source.trim().to_lowercase().as_str() {
            "hub" => self.invalidate(|c| {
                c.use_local = false;
                c.local_model_dir.clear();
            }),
            "local" => self.invalidate(|c| {
                c.use_local = true;
                if let Some(d) = local_dir {
                    c.local_model_dir = d;
                }
            }),
            _ => return Err("model source must be 'hub' or 'local'".into()),
        }
        Ok(())
    }

    fn set_local_package_format(&self, format: &str) -> Result<(), String> {
        let f = format.trim().to_lowercase();
        if f.is_empty() {
            return Err("local package format must not be empty".into());
        }
        self.invalidate(|c| c.local_package_format = f);
        Ok(())
    }

    fn set_voice_speed(&self, voice: &str, speed: f32) {
        let mut cfg = self.config();
        cfg.voice = if voice.trim().is_empty() {
            DEFAULT_VOICE.to_string()
        } else {
            voice.to_string()
        };
        cfg.speed = sanitize_speed(speed);
    }

    fn set_execution_hint(&self, hint: &str) -> Result<(), String> {
        let h = normalize_ep_hint(hint).ok_or("execution hint must not be empty")?;
        let loaded = {
            let mut cfg = self.config();
            cfg.execution_provider_hint = h.clone();
            cfg.is_current_loaded()
        };
        if loaded {
            record_ep_outcome(EP_COMPONENT, kitten_ep_outcome(&h));
        }
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.config().is_current_loaded()
    }

    fn voices(&self) -> Vec<String> {
        let cfg = self.config();
        if cfg.is_current_loaded() {
            cfg.voices.clone()
        } else {
            Vec::new()
        }
    }

    /// Blocking: return the model slot holding a model for the current generation, loading it first
    /// if needed (the Hub path may download). Concurrent callers wait here and reuse the result.
    fn ensure_loaded(&self) -> Result<MutexGuard<'_, Option<LoadedModel>>, String> {
        let mut slot = self.model();
        let snapshot = self.config().clone();
        if slot.as_ref().map(|m| m.generation) == Some(snapshot.generation) {
            return Ok(slot);
        }
        // Free the stale model before loading the next one (two models in RAM is pointless).
        *slot = None;
        let tts = load_model_from(&snapshot)?;
        let voices = tts.available_voices.clone();
        {
            let mut cfg = self.config();
            // A setter may have run during the load; then this model is already stale and the
            // next call reloads it, so do not advertise it as current.
            if cfg.generation == snapshot.generation {
                cfg.loaded_generation = Some(snapshot.generation);
                cfg.voices = voices;
            }
        }
        record_ep_outcome(EP_COMPONENT, kitten_ep_outcome(&snapshot.execution_provider_hint));
        *slot = Some(LoadedModel {
            tts,
            generation: snapshot.generation,
        });
        Ok(slot)
    }

    /// Blocking: synthesize `text` as a 24 kHz mono f32 WAV (empty text → empty bytes, no load).
    fn synthesize_wav_bytes(&self, text: &str) -> Result<Vec<u8>, String> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let (voice, speed) = {
            let cfg = self.config();
            (cfg.voice.clone(), cfg.speed)
        };
        let slot = self.ensure_loaded()?;
        let model = slot.as_ref().ok_or("Model not loaded")?;
        let audio = model
            .tts
            .generate(text, &voice, speed, true)
            .map_err(|e| format!("Synthesis failed: {e}"))?;
        drop(slot);
        encode_wav_f32(&audio, kitten::SAMPLE_RATE)
    }
}

/// A finite speed in Kitten's useful range (0.5–2.0); anything else becomes 1.0.
fn sanitize_speed(speed: f32) -> f32 {
    if speed.is_finite() && speed > 0.0 {
        speed.clamp(0.5, 2.0)
    } else {
        1.0
    }
}

/// Lower-cased EP id (`directml` → `dml`), or `None` when blank.
fn normalize_ep_hint(hint: &str) -> Option<String> {
    let h = hint.trim().to_lowercase();
    match h.as_str() {
        "" => None,
        "directml" => Some("dml".to_string()),
        _ => Some(h),
    }
}

fn kitten_ep_outcome(hint: &str) -> EpOutcome {
    let requested = normalize_ep_hint(hint).unwrap_or_else(|| "cpu".to_string());
    if requested == "cpu" {
        EpOutcome::granted("cpu")
    } else {
        EpOutcome::fallback(
            requested,
            "cpu",
            "Kitten TTS runs on the CPU ONNX Runtime execution provider; GPU execution is not available for this backend yet.",
        )
    }
}

fn encode_wav_f32(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, String> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut cursor = Cursor::new(Vec::with_capacity(44 + samples.len() * 4));
    {
        let mut writer = WavWriter::new(&mut cursor, spec).map_err(|e| e.to_string())?;
        for s in samples {
            writer.write_sample(*s).map_err(|e| e.to_string())?;
        }
        writer.finalize().map_err(|e| e.to_string())?;
    }
    Ok(cursor.into_inner())
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NeuralOnnxVoice {
    pub name: String,
}

fn load_from_local_dir(dir: &Path) -> Result<kitten::KittenTTS, String> {
    hub::load_from_dir(dir).map_err(|e| format!("Failed to load local speech model from folder: {e:#}"))
}

/// Blocking: load the model `cfg` describes (local folder or Hugging Face Hub download).
fn load_model_from(cfg: &TtsConfig) -> Result<kitten::KittenTTS, String> {
    if cfg.use_local {
        let fmt = cfg.local_package_format.trim().to_lowercase();
        if fmt != SUPPORTED_LOCAL_FORMAT {
            return Err(format!(
                "Unsupported local speech folder format '{}'. Only speech_package_v1 is available (see docs/OPERATOR_REFERENCE.md#local-speech-package-onnx).",
                fmt
            ));
        }
        let trimmed = cfg.local_model_dir.trim();
        if trimmed.is_empty() {
            return Err("Local model folder path is empty. Choose a folder or switch to Hugging Face.".into());
        }
        load_from_local_dir(Path::new(trimmed))
    } else {
        hub::load_from_hub(&cfg.hub_id).map_err(|e| format!("Failed to load speech model from Hub: {e:#}"))
    }
}

/// Run blocking model work off the async runtime and the main thread.
async fn blocking<T: Send + 'static>(
    job: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(job)
        .await
        .map_err(|e| format!("Local speech worker failed: {e}"))?
}

#[command]
async fn neural_onnx_tts_set_execution_hint<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
    hint: String,
) -> Result<(), String> {
    state.set_execution_hint(&hint)
}

#[command]
async fn neural_onnx_tts_set_hub_id<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
    hub_id: String,
) -> Result<(), String> {
    state.set_hub_id(&hub_id)
}

/// `source`: `"hub"` or `"local"`. For `local`, pass the absolute folder path (may be empty until user picks a folder).
#[command]
async fn neural_onnx_tts_set_model_source<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
    source: String,
    local_dir: Option<String>,
) -> Result<(), String> {
    state.set_model_source(&source, local_dir)
}

#[command]
async fn neural_onnx_tts_set_local_package_format<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
    format: String,
) -> Result<(), String> {
    state.set_local_package_format(&format)
}

#[command]
async fn neural_onnx_tts_set_voice_speed<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
    voice: String,
    speed: f32,
) -> Result<(), String> {
    state.set_voice_speed(&voice, speed);
    Ok(())
}

/// Load (or keep) the model for the current settings. May download from the Hub; runs on the
/// blocking pool so the UI stays responsive.
#[command]
async fn neural_onnx_tts_load_model<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
    execution_provider: Option<String>,
) -> Result<(), String> {
    if let Some(ep) = execution_provider.as_deref().and_then(normalize_ep_hint) {
        state.config().execution_provider_hint = ep;
    }
    let st = state.inner().clone();
    blocking(move || st.ensure_loaded().map(|_| ())).await
}

#[command]
async fn neural_onnx_tts_list_voices<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
) -> Result<Vec<NeuralOnnxVoice>, String> {
    Ok(state
        .voices()
        .into_iter()
        .map(|name| NeuralOnnxVoice { name })
        .collect())
}

#[command]
async fn neural_onnx_tts_speak<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
    text: String,
) -> Result<Vec<u8>, String> {
    let st = state.inner().clone();
    blocking(move || st.synthesize_wav_bytes(&text)).await
}

/// One IPC hop: synthesize WAV and mix into the voice changer (no base64 / webview round-trip).
#[command]
async fn neural_onnx_tts_speak_into_voice_changer<R: Runtime>(
    app: AppHandle<R>,
    onnx: State<'_, NeuralOnnxTtsState>,
    text: String,
) -> Result<(), String> {
    let st = onnx.inner().clone();
    blocking(move || {
        let wav = st.synthesize_wav_bytes(&text)?;
        if wav.is_empty() {
            return Ok(());
        }
        let vc = app
            .try_state::<VoiceChangerState>()
            .ok_or("The voice changer is not available.")?;
        push_aux_wav_bytes(&vc, wav)
    })
    .await
}

#[command]
async fn neural_onnx_tts_is_loaded<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
) -> Result<bool, String> {
    Ok(state.is_loaded())
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("neural-onnx-tts")
        .setup(|app, _api| {
            app.manage(NeuralOnnxTtsState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            neural_onnx_tts_set_execution_hint,
            neural_onnx_tts_set_hub_id,
            neural_onnx_tts_set_model_source,
            neural_onnx_tts_set_local_package_format,
            neural_onnx_tts_set_voice_speed,
            neural_onnx_tts_load_model,
            neural_onnx_tts_list_voices,
            neural_onnx_tts_speak,
            neural_onnx_tts_speak_into_voice_changer,
            neural_onnx_tts_is_loaded,
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn setters_invalidate_by_generation() {
        let s = NeuralOnnxTtsState::new();
        let g0 = s.config().generation;
        s.set_hub_id("org/model").unwrap();
        s.set_model_source("local", Some("C:/models/kitten".into())).unwrap();
        assert_eq!(s.config().generation, g0 + 2);
        // Same format after normalization: not a change.
        s.set_local_package_format("SPEECH_PACKAGE_V1").unwrap();
        assert_eq!(s.config().generation, g0 + 2);
        s.set_local_package_format("piper_v1").unwrap();
        let cfg = s.config().clone();
        assert_eq!(cfg.generation, g0 + 3);
        assert!(cfg.use_local);
        assert_eq!(cfg.local_model_dir, "C:/models/kitten");
        assert_eq!(cfg.local_package_format, "piper_v1");
        assert_eq!(cfg.hub_id, "org/model");
        assert!(!s.is_loaded());
        assert!(s.voices().is_empty());
    }

    /// Pretend the current generation's model finished loading (a real Kitten model needs a
    /// download, so only the bookkeeping is simulated).
    fn mark_loaded(s: &NeuralOnnxTtsState) {
        let mut cfg = s.config();
        cfg.loaded_generation = Some(cfg.generation);
        cfg.voices = vec!["Jasper".into()];
    }

    #[test]
    fn resending_the_same_source_keeps_the_loaded_model() {
        // Review finding: opening the TTS inspector re-sends set_model_source('hub') and
        // set_local_package_format('speech_package_v1'), which used to drop the running model.
        let s = NeuralOnnxTtsState::new();
        s.set_hub_id(DEFAULT_HUB_ID).unwrap();
        mark_loaded(&s);
        let g = s.config().generation;
        for _ in 0..2 {
            s.set_model_source("hub", None).unwrap();
            s.set_local_package_format("speech_package_v1").unwrap();
            s.set_hub_id(DEFAULT_HUB_ID).unwrap();
        }
        // The package format only matters for local folders, so changing it on the Hub is no change.
        s.set_local_package_format("piper_v1").unwrap();
        assert_eq!(s.config().generation, g);
        assert!(s.is_loaded());
        assert_eq!(s.voices(), ["Jasper"]);

        // Local folder: re-sending the same folder (or none, which keeps it) is not a change either.
        s.set_local_package_format("speech_package_v1").unwrap();
        s.set_model_source("local", Some("D:/kitten".into())).unwrap();
        mark_loaded(&s);
        let g = s.config().generation;
        s.set_model_source("local", Some("D:/kitten".into())).unwrap();
        s.set_model_source("local", None).unwrap();
        s.set_local_package_format(" Speech_Package_V1 ").unwrap();
        assert_eq!(s.config().generation, g);
        assert!(s.is_loaded());

        // A real change still invalidates.
        s.set_model_source("local", Some("D:/other".into())).unwrap();
        assert_eq!(s.config().generation, g + 1);
        assert!(!s.is_loaded());
        assert!(s.voices().is_empty());
    }

    #[test]
    fn setters_validate_input() {
        let s = NeuralOnnxTtsState::new();
        assert!(s.set_hub_id("  ").is_err());
        assert!(s.set_model_source("cloud", None).is_err());
        assert!(s.set_local_package_format(" ").is_err());
        assert!(s.set_execution_hint("").is_err());
        s.set_model_source("hub", None).unwrap();
        assert!(!s.config().use_local);
    }

    #[test]
    fn voice_and_speed_do_not_invalidate() {
        let s = NeuralOnnxTtsState::new();
        let g = s.config().generation;
        s.set_voice_speed("  ", f32::NAN);
        assert_eq!(s.config().voice, DEFAULT_VOICE);
        assert_eq!(s.config().speed, 1.0);
        s.set_voice_speed("Bella", 9.0);
        assert_eq!(s.config().speed, 2.0);
        assert_eq!(s.config().generation, g);
    }

    #[test]
    fn setters_never_wait_for_a_running_synthesis() {
        // Simulate a long synthesis / Hub download holding the model mutex.
        let s = NeuralOnnxTtsState::new();
        let busy = s.clone();
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _slot = busy.model.lock().unwrap();
            held_tx.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(1500));
        });
        held_rx.recv().unwrap();
        let started = Instant::now();
        s.set_model_source("local", Some("D:/voices".into())).unwrap();
        s.set_hub_id("KittenML/kitten-tts-nano-0.8").unwrap();
        s.set_voice_speed("Luna", 1.1);
        s.set_execution_hint("cuda").unwrap();
        assert!(!s.is_loaded());
        assert!(s.voices().is_empty());
        assert!(started.elapsed() < Duration::from_millis(500), "setters blocked on the model lock");
        worker.join().unwrap();
    }

    #[test]
    fn empty_text_does_not_load_a_model() {
        let s = NeuralOnnxTtsState::new();
        assert_eq!(s.synthesize_wav_bytes("   ").unwrap(), Vec::<u8>::new());
        assert!(!s.is_loaded());
    }

    #[test]
    fn local_load_errors_are_reported_and_leave_state_unloaded() {
        let s = NeuralOnnxTtsState::new();
        s.set_model_source("local", Some(String::new())).unwrap();
        let err = s.ensure_loaded().err().unwrap();
        assert!(err.contains("folder path is empty"), "{err}");

        s.set_local_package_format("piper_v1").unwrap();
        s.set_model_source("local", Some("C:/does/not/matter".into())).unwrap();
        let err = s.ensure_loaded().err().unwrap();
        assert!(err.contains("Unsupported local speech folder format"), "{err}");

        s.set_local_package_format("speech_package_v1").unwrap();
        let missing = std::env::temp_dir().join("sigil-p0f-missing-speech-package");
        s.set_model_source("local", Some(missing.display().to_string())).unwrap();
        let err = s.ensure_loaded().err().unwrap();
        assert!(err.contains("config.json"), "{err}");
        assert!(!s.is_loaded());
    }

    #[test]
    fn local_package_loads_and_speaks_offline() {
        let dir = crate::services::local_tts::test_model::write_package("plugin");
        let s = NeuralOnnxTtsState::new();
        s.set_model_source("local", Some(dir.display().to_string())).unwrap();
        s.set_voice_speed("Luna", 1.0);
        s.set_execution_hint("dml").unwrap();
        let wav = s.synthesize_wav_bytes("Hello from Sigil.").expect("speak");
        assert!(s.is_loaded());
        let mut voices = s.voices();
        voices.sort();
        assert_eq!(voices, ["Jasper", "Luna"]);
        let mut r = hound::WavReader::new(Cursor::new(wav)).unwrap();
        assert_eq!(r.spec().sample_rate, kitten::SAMPLE_RATE);
        let samples: Vec<f32> = r.samples::<f32>().map(Result::unwrap).collect();
        assert!(!samples.is_empty());
        assert!(samples.iter().all(|x| (x - crate::services::local_tts::test_model::LEVEL).abs() < 1e-6));
        // The outcome is truthful: a DirectML hint still runs on the CPU, with the reason.
        let o = crate::services::diagnostics::ep_outcome(EP_COMPONENT).expect("recorded");
        assert_eq!((o.requested.as_str(), o.active.as_str()), ("dml", "cpu"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn poisoned_model_mutex_recovers() {
        let s = NeuralOnnxTtsState::new();
        let p = s.clone();
        let _ = std::thread::spawn(move || {
            let _slot = p.model.lock().unwrap();
            panic!("simulated Kitten panic");
        })
        .join();
        assert!(s.model.is_poisoned());
        assert!(s.model().is_none());
        assert!(!s.model.is_poisoned());
        // Setters also survive a poisoned model mutex.
        s.set_hub_id("a/b").unwrap();
    }

    #[test]
    fn ep_outcome_is_truthful() {
        assert_eq!(kitten_ep_outcome("cpu"), EpOutcome::granted("cpu"));
        let o = kitten_ep_outcome("DirectML");
        assert_eq!(o.requested, "dml");
        assert_eq!(o.active, "cpu");
        assert!(o.fell_back() && o.reason.is_some());
        assert_eq!(normalize_ep_hint(" CUDA ").as_deref(), Some("cuda"));
        assert_eq!(normalize_ep_hint(" "), None);
    }

    #[test]
    fn wav_encoding_round_trips() {
        let samples = [0.0f32, 0.5, -0.25, 1.0];
        let wav = encode_wav_f32(&samples, kitten::SAMPLE_RATE).unwrap();
        let mut r = hound::WavReader::new(Cursor::new(wav)).unwrap();
        assert_eq!(r.spec().sample_rate, kitten::SAMPLE_RATE);
        let back: Vec<f32> = r.samples::<f32>().map(Result::unwrap).collect();
        assert_eq!(back, samples);
    }
}
