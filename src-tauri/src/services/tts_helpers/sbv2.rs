//! `sbv2` TTS engine: Style-Bert-VITS2 JP-Extra voices and AivisSpeech `.aivmx` voices, running on
//! Sigil's own ONNX Runtime (P6-G).
//!
//! # Why in-process (P6-G deviation from the plan's helper process)
//! The plan ran `sbv2_core`'s server as a helper because `sbv2_core` links its own ONNX Runtime
//! (ort rc.9) and two static ORTs clash in one executable. The engine is small (a Japanese
//! frontend, a BERT and a VITS graph), so P6-G ports it instead (see [`g2p`], [`text`], [`synth`])
//! onto Sigil's ORT rc.13, which removes the clash without a second runtime or a Sigil-built
//! helper pack: the only downloads are the published BERT model and the jpreprocess dictionary
//! (catalog fragment `runtime-sbv2.json`), plus the user's own voice files. The engine runs on its
//! own worker thread ([`engine`]) like the other in-process voices (`local_tts`), with the EP
//! ladder of `ort_session` (CPU by default; DirectML / CUDA on request, with a VRAM claim).
//! The frozen helper interface is kept: [`start`] loads the engine and returns a
//! [`TtsHelperEndpoint`] whose `base_url` is empty (no server); TypeScript speaks with
//! [`sbv2_speak`] / [`sbv2_cancel`] (registered in `tts_helpers/mod.rs` and
//! `plugin-commands/tts-helpers.txt` through INTEGRATE).
//!
//! # Voices
//! User files dropped on the panel ([`sbv2_import_model`]): `.aivmx` (AivisHub) or a plain
//! Style-Bert-VITS2 ONNX export (`model_<name>.onnx` + `style_vectors_<name>.json`, `config.json`),
//! copied into `app_data/tts/sbv2/<id>/` after a format check ([`format`], [`library`]); never
//! executed, never unpickled. `.aivm` (Safetensors) and PyTorch checkpoints are refused with how to
//! get the ONNX version. The AivisSpeech preset of the VOICEVOX backend (P0) stays the fallback.
//!
//! # Frozen contract (P6-S0; structs may gain fields)
//! - `info()`, `start(app, config)`, `stop()`, `status()` (see [`super`]). `config`:
//!   `{ model: <imported id>, ep?: "cpu" | "dml" | "cuda", threads?: 1..8 }`.
//! - `plugin:tts-helpers|sbv2_import_model { path } -> Sbv2Model`, `sbv2_list_models -> Sbv2Model[]`,
//!   `sbv2_remove_model { id } -> ()`.
//! - P6-G additions (registered at INTEGRATE): `sbv2_speak { request: Sbv2SpeakRequest, utteranceId }
//!   -> SpeakReport` (pushes the sentence into the begun `tts_sink` utterance; `Err("cancelled")`
//!   when cancelled), `sbv2_cancel { utteranceId? } -> ()`.

pub mod engine;
pub mod format;
pub mod g2p;
pub mod library;
mod mora;
pub mod synth;
pub mod text;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, Weak};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{command, AppHandle, Emitter, Runtime};

use self::engine::{AppSink, Backend, Cancel, LoadSpec, Loaded, SpeakJob, Worker};
use self::format::SpeakerInfo;
use self::library::StoredModel;
use self::synth::{Bert, CharTokenizer, SynthParams, Voice};
use super::{TtsHelperEndpoint, TtsHelperEngine, TtsHelperInfo, TtsHelperState, TtsHelperStatus, STATE_EVENT};
use crate::services::neural_audio::ort_session::{normalize_ep, EP_CPU};
use crate::services::tts_sink::SpeakReport;

const LOG: &str = "sigil::tts::sbv2";

/// This module's engine.
pub const ENGINE: TtsHelperEngine = TtsHelperEngine::Sbv2;
/// Catalog id of the Japanese BERT (DeBERTa-v2 large, character tokenizer) every voice needs.
pub const BERT_MODEL_ID: &str = "tts/sbv2-bert-ja";
/// Catalog id of the jpreprocess NAIST-JDIC dictionary (OpenJTalk's, built for jpreprocess 0.15).
pub const DICTIONARY_ID: &str = "text/jpreprocess-naist-jdic";
/// Files of the BERT entry, in preference order (the fp16 export AivisSpeech uses; an fp32
/// `model.onnx` placed by hand also works).
const BERT_FILES: [&str; 2] = ["model_fp16.onnx", "model.onnx"];
const TOKENIZER_FILE: &str = "tokenizer.json";
/// Text spoken once after a voice loads (primes ONNX Runtime's buffers for the first sentence).
const WARM_UP_TEXT: &str = "こんにちは。";
/// Sentences longer than this are spoken in clauses (bounded memory and time per run).
const MAX_CHUNK_CHARS: usize = 160;
/// Pause between the clauses of a long sentence (seconds).
const CLAUSE_PAUSE_S: f32 = 0.12;
/// Provisional VRAM of the engine on a GPU EP (BERT fp16 + one voice); P7 measures it.
const GPU_VRAM_MB: u64 = 1400;
/// Peak of every spoken sentence (see [`normalize_peak`]); a little under full scale so that
/// resampling to the output device's rate does not clip.
const OUTPUT_PEAK: f32 = 0.97;

/// An imported voice model (TS `RustSbv2Model`; P6-G added the fields after `styles`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Sbv2Model {
    /// Folder id under `app_data/tts/sbv2/`.
    pub id: String,
    /// Display name (from the model's metadata, else the file name).
    pub name: String,
    /// `aivmx` or `sbv2`.
    pub format: String,
    /// Speaker names.
    pub speakers: Vec<String>,
    /// Style names (every speaker's, first seen first).
    pub styles: Vec<String>,
    /// Speakers with their own styles and ids (P6-G).
    #[serde(default)]
    pub voices: Vec<SpeakerInfo>,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub creators: Vec<String>,
    #[serde(default)]
    pub version: String,
    /// `Style-Bert-VITS2 (JP-Extra)` or `Style-Bert-VITS2`.
    #[serde(default)]
    pub architecture: String,
    #[serde(default)]
    pub sample_rate: u32,
    #[serde(default)]
    pub size_bytes: u64,
    #[serde(default)]
    pub source_name: String,
}

impl From<&StoredModel> for Sbv2Model {
    fn from(m: &StoredModel) -> Self {
        let mut styles: Vec<String> = Vec::new();
        for s in m.speakers.iter().flat_map(|s| s.styles.iter()) {
            if !styles.contains(&s.name) {
                styles.push(s.name.clone());
            }
        }
        Self {
            id: m.id.clone(),
            name: m.name.clone(),
            format: m.format.clone(),
            speakers: m.speakers.iter().map(|s| s.name.clone()).collect(),
            styles,
            voices: m.speakers.clone(),
            description: m.description.clone(),
            creators: m.creators.clone(),
            version: m.version.clone(),
            architecture: m.architecture.clone(),
            sample_rate: m.sample_rate,
            size_bytes: m.size_bytes,
            source_name: m.source_name.clone(),
        }
    }
}

/// `tts_helper_start`'s `config` for this engine.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct StartConfig {
    pub model: String,
    /// `cpu` (default), `dml` or `cuda` (`gpu` = `cuda`).
    pub ep: String,
    /// Inference threads (CPU); `None` / 0 = automatic.
    pub threads: Option<u32>,
}

/// `sbv2_speak`'s `request` (TS `Sbv2SpeakRequest` in `services/sbv2.ts`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Sbv2SpeakRequest {
    /// Imported model id; `""` = the started one.
    pub model: String,
    pub text: String,
    /// Speaker name (`""` = the first).
    pub speaker: String,
    /// Style name (`""` or unknown = the neutral style, id 0).
    pub style: String,
    /// Style strength (Style-Bert-VITS2's style weight; 0–20, 1 = as trained).
    pub style_weight: f32,
    /// Tempo variation (SDP ratio, 0–1).
    pub sdp_ratio: f32,
    pub noise: f32,
    pub noise_w: f32,
    /// Speaking rate (1 = normal; 0.5–2).
    pub speed: f32,
}

impl Default for Sbv2SpeakRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            text: String::new(),
            speaker: String::new(),
            style: String::new(),
            style_weight: 1.0,
            sdp_ratio: synth::DEFAULT_SDP_RATIO,
            noise: synth::DEFAULT_NOISE,
            noise_w: synth::DEFAULT_NOISE_W,
            speed: 1.0,
        }
    }
}

fn finite_or(v: f32, fallback: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        fallback
    }
}

impl From<Sbv2SpeakRequest> for SpeakJob {
    fn from(r: Sbv2SpeakRequest) -> Self {
        SpeakJob {
            model: r.model.trim().to_string(),
            text: r.text,
            speaker: r.speaker.trim().to_string(),
            style: r.style.trim().to_string(),
            style_weight: finite_or(r.style_weight, 1.0).clamp(0.0, 20.0),
            sdp_ratio: finite_or(r.sdp_ratio, synth::DEFAULT_SDP_RATIO).clamp(0.0, 1.0),
            noise: finite_or(r.noise, synth::DEFAULT_NOISE).clamp(0.0, 2.0),
            noise_w: finite_or(r.noise_w, synth::DEFAULT_NOISE_W).clamp(0.0, 2.0),
            speed: finite_or(r.speed, 1.0).clamp(0.5, 2.0),
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Paths
// -------------------------------------------------------------------------------------------------

fn app_data() -> Result<PathBuf, String> {
    crate::services::diagnostics::app_data_dir().ok_or_else(|| "no app data folder".to_string())
}

fn library_root() -> Result<PathBuf, String> {
    Ok(library::library_dir(&app_data()?))
}

fn not_downloaded(id: &str, what: &str) -> String {
    format!("{what} is not downloaded yet ({id}); download it in the voice panel or Models & Runtimes")
}

/// The BERT graph and tokenizer of an installed BERT entry folder.
fn bert_files(dir: &Path) -> Result<(PathBuf, PathBuf), String> {
    let model = BERT_FILES
        .iter()
        .map(|f| dir.join(f))
        .find(|p| p.is_file())
        .ok_or_else(|| format!("{} holds no BERT model ({})", dir.display(), BERT_FILES.join(" or ")))?;
    let tokenizer = dir.join(TOKENIZER_FILE);
    if !tokenizer.is_file() {
        return Err(format!("{} holds no {TOKENIZER_FILE}", dir.display()));
    }
    Ok((model, tokenizer))
}

/// Folders of the runtime assets (BERT, dictionary).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assets {
    pub bert_dir: PathBuf,
    pub dictionary_dir: PathBuf,
}

fn installed_assets() -> Result<Assets, String> {
    let bert_dir = crate::services::model_store::installed_path(BERT_MODEL_ID)
        .ok_or_else(|| not_downloaded(BERT_MODEL_ID, "The Japanese BERT model for Style-Bert-VITS2 voices"))?;
    let dictionary_dir = crate::services::model_store::installed_path(DICTIONARY_ID)
        .ok_or_else(|| not_downloaded(DICTIONARY_ID, "The Japanese dictionary"))?;
    Ok(Assets { bert_dir, dictionary_dir })
}

// -------------------------------------------------------------------------------------------------
// The Japanese frontend, shared
// -------------------------------------------------------------------------------------------------

/// A loaded dictionary frontend and the folder it came from.
type FrontendSlot = Option<(PathBuf, Weak<g2p::Frontend>)>;

/// The loaded dictionary frontend, kept while anyone holds it (the dictionary is about 80 MB).
static SHARED_FRONTEND: LazyLock<Mutex<FrontendSlot>> = LazyLock::new(|| Mutex::new(None));

/// The Japanese frontend on the dictionary in `dir`, loaded once while anyone holds it.
pub fn shared_frontend(dir: &Path) -> Result<Arc<g2p::Frontend>, String> {
    let mut slot = SHARED_FRONTEND.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((path, weak)) = slot.as_ref() {
        if path == dir {
            if let Some(fe) = weak.upgrade() {
                return Ok(fe);
            }
        }
    }
    let fe = Arc::new(g2p::Frontend::load(dir)?);
    *slot = Some((dir.to_path_buf(), Arc::downgrade(&fe)));
    Ok(fe)
}

/// A Japanese reader on the installed dictionary ([`g2p::Frontend::kana_reading`]), for TTS
/// frontends that read kanji wrongly (sherpa-onnx's Kokoro gives them Mandarin readings; P2 outcome
/// note for P6-G). `None` when the dictionary (`text/jpreprocess-naist-jdic`) is not installed or
/// does not load. Hold the `Arc` while the voice is loaded: the dictionary stays in memory as long.
pub fn japanese_reader() -> Option<Arc<g2p::Frontend>> {
    let dir = crate::services::model_store::installed_path(DICTIONARY_ID)?;
    match shared_frontend(&dir) {
        Ok(fe) => Some(fe),
        Err(e) => {
            tracing::warn!(target: LOG, "the Japanese dictionary does not load: {e}");
            None
        }
    }
}

// -------------------------------------------------------------------------------------------------
// The engine
// -------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
struct BaseKey {
    assets: Assets,
    ep: String,
    threads: usize,
}

/// Frontend, tokenizer and BERT (shared by every voice).
struct Base {
    key: BaseKey,
    frontend: Arc<g2p::Frontend>,
    tokenizer: CharTokenizer,
    bert: Bert,
}

struct LoadedVoice {
    model: StoredModel,
    ep: String,
    threads: usize,
    voice: Voice,
}

/// The real engine behind the worker.
pub struct RealBackend {
    /// Where the runtime assets come from (`None` = the model store).
    assets: Option<Assets>,
    /// Library folder (`None` = `app_data/tts/sbv2`).
    library: Option<PathBuf>,
    base: Option<Base>,
    voice: Option<LoadedVoice>,
    /// The last start (`None` after a stop): a sentence for another model loads it the same way,
    /// and a reset engine reloads from it. Nothing loads without a start.
    started: Option<LoadSpec>,
    vram: Option<crate::services::resources::VramClaim>,
    /// Speak a warm-up sentence after a voice loads.
    warm_up: bool,
}

impl Default for RealBackend {
    fn default() -> Self {
        Self { assets: None, library: None, base: None, voice: None, started: None, vram: None, warm_up: true }
    }
}

struct NoCancel;
impl Cancel for NoCancel {
    fn cancelled(&self) -> bool {
        false
    }
}

/// Text split into clauses of at most [`MAX_CHUNK_CHARS`] characters (at sentence and clause
/// punctuation where possible; a run without any is cut at the limit).
fn chunks(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.chars().count() <= MAX_CHUNK_CHARS {
        return if text.is_empty() { Vec::new() } else { vec![text.to_string()] };
    }
    let breaks = ['。', '！', '？', '!', '?', '.', '、', '，', ',', '；', ';', '\n', '…'];
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut last_break: Option<usize> = None;
    for c in text.chars() {
        cur.push(c);
        if breaks.contains(&c) {
            last_break = Some(cur.len());
        }
        if cur.chars().count() >= MAX_CHUNK_CHARS {
            let cut = last_break.unwrap_or(cur.len());
            let rest = cur.split_off(cut);
            if !cur.trim().is_empty() {
                out.push(cur.trim().to_string());
            }
            cur = rest;
            last_break = None;
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Scale a sentence so its peak is [`OUTPUT_PEAK`], as Style-Bert-VITS2 does with every utterance
/// (`TTSModel.infer` ends with `convert_to_16_bit_wav`, which divides by the peak; AivisSpeech plays
/// that): raw VITS output usually peaks well under full scale, and sometimes over it, where a clamp
/// would crackle. Once per sentence, after its clauses are joined, so the loudness does not jump
/// between clauses. Silence is left alone.
fn normalize_peak(audio: &mut [f32]) {
    let peak = audio.iter().fold(0f32, |m, s| if s.is_finite() { m.max(s.abs()) } else { m });
    if peak > 1e-6 {
        let gain = OUTPUT_PEAK / peak;
        for s in audio.iter_mut() {
            *s = if s.is_finite() { *s * gain } else { 0.0 };
        }
    }
}

impl RealBackend {
    fn assets(&self) -> Result<Assets, String> {
        match &self.assets {
            Some(a) => Ok(a.clone()),
            None => installed_assets(),
        }
    }

    fn library(&self) -> Result<PathBuf, String> {
        match &self.library {
            Some(l) => Ok(l.clone()),
            None => library_root(),
        }
    }

    fn ensure_base(&mut self, ep: &str, threads: usize) -> Result<(), String> {
        let key = BaseKey { assets: self.assets()?, ep: ep.to_string(), threads };
        if self.base.as_ref().is_some_and(|b| b.key == key) {
            return Ok(());
        }
        self.base = None;
        let started = std::time::Instant::now();
        let (bert_model, tokenizer) = bert_files(&key.assets.bert_dir)?;
        let frontend = shared_frontend(&key.assets.dictionary_dir)?;
        let tokenizer = CharTokenizer::load(&tokenizer)?;
        let bert = Bert::load(&bert_model, ep, threads)?;
        tracing::info!(target: LOG, "Style-Bert-VITS2 frontend and BERT ready on {} in {} ms", bert.ep, started.elapsed().as_millis());
        self.base = Some(Base { key, frontend, tokenizer, bert });
        Ok(())
    }

    /// The imported model `id` (checked before anything heavy loads).
    fn model(&self, id: &str) -> Result<(PathBuf, StoredModel), String> {
        let root = self.library()?;
        let model = library::get(&root, id)?;
        Ok((root.join(&model.id), model))
    }

    fn ensure_voice(&mut self, dir: &Path, model: StoredModel, ep: &str, threads: usize) -> Result<bool, String> {
        if self.voice.as_ref().is_some_and(|v| v.model.id == model.id && v.ep == ep && v.threads == threads) {
            return Ok(false);
        }
        self.voice = None;
        let styles = library::load_styles(dir, &model)?;
        let started = std::time::Instant::now();
        let voice = Voice::load(&dir.join(&model.model_file), styles, model.sample_rate, ep, threads)?;
        if voice.jp_extra() != model.jp_extra {
            tracing::info!(target: LOG, "voice {}: the graph's layout says JP-Extra = {}; using that", model.id, voice.jp_extra());
        }
        tracing::info!(target: LOG, "Style-Bert-VITS2 voice {} ready on {} in {} ms", model.id, voice.ep, started.elapsed().as_millis());
        self.voice = Some(LoadedVoice { model, ep: ep.to_string(), threads, voice });
        Ok(true)
    }

    fn update_vram(&mut self) {
        let gpu = self.base.as_ref().is_some_and(|b| b.bert.ep != EP_CPU) || self.voice.as_ref().is_some_and(|v| v.voice.ep != EP_CPU);
        if gpu && self.vram.is_none() {
            self.vram = Some(crate::services::resources::claim_vram(crate::services::resources::Subsystem::Tts, "sbv2", GPU_VRAM_MB));
        } else if !gpu {
            self.vram = None;
        }
    }

    /// Synthesize `text` with the loaded engine and voice.
    fn run(&mut self, text: &str, req: &SpeakJob, cancel: &dyn Cancel) -> Result<Option<Vec<f32>>, String> {
        let (Some(base), Some(lv)) = (self.base.as_mut(), self.voice.as_mut()) else {
            return Err("the Style-Bert-VITS2 voice is not started".into());
        };
        let speaker = lv
            .model
            .speakers
            .iter()
            .find(|s| !req.speaker.is_empty() && s.name == req.speaker)
            .or_else(|| lv.model.speakers.first())
            .ok_or("the voice model lists no speaker")?;
        let style = speaker
            .styles
            .iter()
            .find(|s| !req.style.is_empty() && s.name == req.style)
            .map_or(0, |s| s.id);
        let params = SynthParams {
            sid: speaker.id,
            style: style.min(lv.voice.style_count().saturating_sub(1) as u32),
            style_weight: req.style_weight,
            sdp_ratio: req.sdp_ratio,
            noise: req.noise,
            noise_w: req.noise_w,
            length_scale: 1.0 / req.speed.max(0.01),
        };
        let pause = vec![0f32; (lv.voice.sample_rate as f32 * CLAUSE_PAUSE_S) as usize];
        let mut audio: Vec<f32> = Vec::new();
        for clause in chunks(text) {
            if cancel.cancelled() {
                return Ok(None);
            }
            let analysis = base.frontend.analyze(&clause)?;
            if analysis.phones.len() <= 2 {
                continue;
            }
            let seq = analysis.sequence(lv.voice.jp_extra());
            let ids = base.tokenizer.ids(&analysis.bert_text);
            if ids.len() != analysis.word2ph.len() {
                return Err("internal error: tokens and characters do not line up".into());
            }
            let (features, dim) = base.bert.features(&ids)?;
            if cancel.cancelled() {
                return Ok(None);
            }
            let phone_level = synth::phone_level_features(&features, dim, &seq.word2ph)?;
            let wav = lv.voice.synthesize(&seq, &phone_level, dim, &params)?;
            if !audio.is_empty() && !wav.is_empty() {
                audio.extend_from_slice(&pause);
            }
            audio.extend(wav);
        }
        normalize_peak(&mut audio);
        Ok(Some(audio))
    }
}

impl Backend for RealBackend {
    fn load(&mut self, spec: &LoadSpec) -> Result<Loaded, String> {
        let ep = normalize_ep(&spec.ep).to_string();
        let threads = spec.threads.clamp(1, 8);
        let (dir, model) = self.model(&spec.model)?;
        self.started = Some(LoadSpec { model: spec.model.clone(), ep: ep.clone(), threads });
        let result = (|| {
            self.ensure_base(&ep, threads)?;
            let fresh = self.ensure_voice(&dir, model, &ep, threads)?;
            self.update_vram();
            if fresh && self.warm_up {
                let job = SpeakJob::from(Sbv2SpeakRequest { text: WARM_UP_TEXT.into(), ..Sbv2SpeakRequest::default() });
                self.run(WARM_UP_TEXT, &job, &NoCancel).map_err(|e| format!("the voice does not speak: {e}"))?;
            }
            let lv = self.voice.as_ref().expect("loaded above");
            Ok(Loaded { model: lv.model.id.clone(), sample_rate: lv.voice.sample_rate, ep: lv.voice.ep.clone() })
        })();
        if result.is_err() {
            // A failed start leaves nothing half loaded (and nothing to reload from).
            self.unload();
        }
        result
    }

    fn synthesize(&mut self, job: &SpeakJob, cancel: &dyn Cancel) -> Result<Option<(Vec<f32>, u32)>, String> {
        let Some(spec) = self.started.clone() else {
            return Err("the Style-Bert-VITS2 voice is not started".into());
        };
        let id = if job.model.is_empty() { spec.model.clone() } else { job.model.clone() };
        let (dir, model) = self.model(&id)?;
        self.ensure_base(&spec.ep, spec.threads)?;
        self.ensure_voice(&dir, model, &spec.ep, spec.threads)?;
        self.update_vram();
        let rate = self.voice.as_ref().map_or(44_100, |v| v.voice.sample_rate);
        Ok(self.run(&job.text, job, cancel)?.map(|a| (a, rate)))
    }

    fn drop_model(&mut self, model: &str) {
        if self.voice.as_ref().is_some_and(|v| v.model.id == model) {
            self.voice = None;
            self.update_vram();
        }
    }

    fn unload(&mut self) {
        self.started = None;
        self.reset();
    }

    fn reset(&mut self) {
        self.voice = None;
        self.base = None;
        self.vram = None;
    }
}

static WORKER: LazyLock<Worker> = LazyLock::new(|| Worker::new(Box::new(RealBackend::default()), Arc::new(AppSink)));

fn worker() -> &'static Worker {
    &WORKER
}

// -------------------------------------------------------------------------------------------------
// State
// -------------------------------------------------------------------------------------------------

struct StateCell {
    epoch: u64,
    status: TtsHelperStatus,
}

static STATE: LazyLock<Mutex<StateCell>> = LazyLock::new(|| {
    Mutex::new(StateCell { epoch: 0, status: TtsHelperStatus { engine: ENGINE, state: TtsHelperState::Stopped, model: None, error: None } })
});
static EPOCH_SOURCE: AtomicU64 = AtomicU64::new(0);
type Emit = Box<dyn Fn(&TtsHelperStatus) + Send + Sync>;
static EMIT: OnceLock<Emit> = OnceLock::new();

fn install_emitter<R: Runtime>(app: &AppHandle<R>) {
    let app = app.clone();
    let _ = EMIT.set(Box::new(move |s: &TtsHelperStatus| {
        let _ = app.emit(STATE_EVENT, s);
    }));
}

/// Start a new epoch in `state` (a start or a stop); returns it.
fn begin(state: TtsHelperState, model: Option<String>) -> u64 {
    let mut cell = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let epoch = EPOCH_SOURCE.fetch_add(1, Ordering::AcqRel) + 1;
    cell.epoch = epoch;
    cell.status = TtsHelperStatus { engine: ENGINE, state, model, error: None };
    if let Some(emit) = EMIT.get() {
        emit(&cell.status);
    }
    epoch
}

/// Move to `state` only while `epoch` is current (a stop meanwhile wins); emitted under the lock.
fn finish(epoch: u64, state: TtsHelperState, model: Option<String>, error: Option<String>) -> bool {
    let mut cell = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if cell.epoch != epoch {
        return false;
    }
    cell.status = TtsHelperStatus { engine: ENGINE, state, model, error };
    if let Some(emit) = EMIT.get() {
        emit(&cell.status);
    }
    true
}

// -------------------------------------------------------------------------------------------------
// Engine interface (frozen by P6-S0)
// -------------------------------------------------------------------------------------------------

/// Static facts about the engine.
pub fn info() -> TtsHelperInfo {
    let assets = installed_assets().is_ok();
    let has_voice = library_root().map(|r| !library::list(&r).is_empty()).unwrap_or(false);
    TtsHelperInfo {
        engine: ENGINE,
        title: "Style-Bert-VITS2 / AivisSpeech".into(),
        experimental: false,
        gpu: false,
        cloning: false,
        requires: vec![BERT_MODEL_ID.to_string(), DICTIONARY_ID.to_string()],
        available: assets && has_voice,
    }
}

fn parse_config(config: Value) -> Result<StartConfig, String> {
    let cfg: StartConfig = match config {
        Value::Null => StartConfig::default(),
        v => serde_json::from_value(v).map_err(|e| format!("invalid Style-Bert-VITS2 settings: {e}"))?,
    };
    let model = cfg.model.trim().to_string();
    if model.is_empty() {
        return Err("Choose a Style-Bert-VITS2 voice first (import an .aivmx file).".into());
    }
    if !library::valid_id(&model) {
        return Err(format!("invalid voice model id {model:?}"));
    }
    Ok(StartConfig { model, ..cfg })
}

/// Load the engine and the chosen voice (in this process) and return the (server-less) endpoint.
pub async fn start<R: Runtime>(app: &AppHandle<R>, config: Value) -> Result<TtsHelperEndpoint, String> {
    install_emitter(app);
    let cfg = parse_config(config)?;
    let threads = crate::services::local_tts::models::effective_threads(cfg.threads.filter(|t| *t > 0)) as usize;
    let spec = LoadSpec { model: cfg.model.clone(), ep: normalize_ep(&cfg.ep).to_string(), threads };
    let epoch = begin(TtsHelperState::Starting, Some(cfg.model.clone()));
    let result = match worker().load(spec).await {
        Ok(r) => r,
        Err(_) => Err(engine::WORKER_STOPPED.to_string()),
    };
    match result {
        Ok(loaded) => {
            if !finish(epoch, TtsHelperState::Ready, Some(loaded.model.clone()), None) {
                return Err(engine::CANCELLED.into());
            }
            tracing::info!(target: LOG, "Style-Bert-VITS2 voice {} started on {} ({} Hz)", loaded.model, loaded.ep, loaded.sample_rate);
            Ok(TtsHelperEndpoint {
                engine: ENGINE,
                base_url: String::new(),
                api_key: String::new(),
                model: loaded.model,
                pcm_sample_rate: loaded.sample_rate,
            })
        }
        Err(e) => {
            finish(epoch, TtsHelperState::Failed, Some(cfg.model), Some(e.clone()));
            Err(e)
        }
    }
}

/// Stop: cancel what is queued and free the engine (Ok when nothing runs).
pub async fn stop() -> Result<(), String> {
    begin(TtsHelperState::Stopped, None);
    let w = worker();
    w.cancel(None);
    let _ = w.unload().await;
    Ok(())
}

/// Current state.
pub fn status() -> TtsHelperStatus {
    STATE.lock().unwrap_or_else(|e| e.into_inner()).status.clone()
}

// -------------------------------------------------------------------------------------------------
// Commands
// -------------------------------------------------------------------------------------------------

/// Check a copied model before it is committed: load its graph once (CPU, one thread) to read its
/// inputs, so a model Sigil cannot run is refused at import, not at the first sentence.
fn check_import(dir: &Path, m: &StoredModel) -> Result<(), String> {
    let styles = library::load_styles(dir, m)?;
    Voice::probe(&dir.join(&m.model_file), styles)
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, String> + Send + 'static) -> Result<T, String> {
    tokio::task::spawn_blocking(f).await.map_err(|e| format!("the voice library task failed: {e}"))?
}

/// Import a dropped `.aivmx` / SBV2 model file (or folder). Imports run one at a time (a second
/// drop waits for the first; `library::import` holds the library lock and sweeps leftovers).
#[command]
pub async fn sbv2_import_model<R: Runtime>(app: AppHandle<R>, path: String) -> Result<Sbv2Model, String> {
    let _ = &app;
    let path = PathBuf::from(path.trim());
    if !path.is_absolute() {
        return Err("expected an absolute path to the voice file".into());
    }
    blocking(move || {
        let root = library_root()?;
        let m = library::import(&root, &path, &check_import)?;
        tracing::info!(target: LOG, "imported Style-Bert-VITS2 voice {} ({})", m.id, m.format);
        Ok(Sbv2Model::from(&m))
    })
    .await
}

/// Imported models.
#[command]
pub async fn sbv2_list_models<R: Runtime>(app: AppHandle<R>) -> Result<Vec<Sbv2Model>, String> {
    let _ = &app;
    blocking(|| Ok(library::list(&library_root()?).iter().map(Sbv2Model::from).collect())).await
}

/// Delete an imported model (the engine lets go of it first).
#[command]
pub async fn sbv2_remove_model<R: Runtime>(app: AppHandle<R>, id: String) -> Result<(), String> {
    let _ = &app;
    let id = id.trim().to_string();
    if !library::valid_id(&id) {
        return Err(format!("invalid voice model id {id:?}"));
    }
    let _ = worker().drop_model(id.clone()).await;
    blocking(move || library::remove(&library_root()?, &id)).await
}

/// Speak one sentence into the begun sink utterance `utterance_id` (P6-G).
#[command]
pub async fn sbv2_speak(request: Sbv2SpeakRequest, utterance_id: String) -> Result<SpeakReport, String> {
    let id = utterance_id.trim().to_string();
    if id.is_empty() {
        return Err("utteranceId must not be empty".into());
    }
    if request.text.trim().is_empty() {
        return Ok(SpeakReport::default());
    }
    match worker().speak(SpeakJob::from(request), id).await {
        Ok(r) => r,
        Err(_) => Err(engine::WORKER_STOPPED.into()),
    }
}

/// Cancel the speak job of one utterance, or every one (`None`).
#[command]
pub async fn sbv2_cancel(utterance_id: Option<String>) -> Result<(), String> {
    match utterance_id.as_deref().map(str::trim) {
        None => worker().cancel(None),
        Some("") => {}
        Some(id) => worker().cancel(Some(id)),
    }
    Ok(())
}

#[cfg(test)]
mod tests;
