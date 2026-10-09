//! TTS engines of the speech bench, through the app's own backends:
//! - `local-tts`: the in-process `local_tts` worker (Kokoro, Kitten, Piper through sherpa-onnx / Sigil's
//!   ORT, with the production chunking: the first clause of a sentence is synthesized alone), driven
//!   through `Engine::speak` into a recording sink instead of the output device;
//! - `tts-helper`: the helper-process engines of `tts_helpers` (`options.helper`: `qwen3`, `magpie`,
//!   `breeze`, `omnivoice`, `s2`), started with the same planning code as the app (`options.config` is
//!   the engine's camelCase start config) and spoken to over their OpenAI-compatible `/audio/speech`
//!   route (raw PCM, streamed: the time to first audio is the first byte of audio);
//! - `openai-speech`: any OpenAI-compatible `/v1/audio/speech` server (`options.baseUrl`, `model`,
//!   `voice`, `apiKeyEnv`);
//! - `rendered`: WAV files made elsewhere (`options.dir`, `<prompt id>.wav`).
//!
//! A rendering is mono f32 at the engine's sample rate; the runner writes it as 16-bit PCM WAV. A model
//! or runtime pack that is not installed fails the engine's rows with the engine's own message.

use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use futures::StreamExt;
use serde_json::{json, Map, Value};

use super::corpus::TtsPrompt;
use super::spec::{EngineDef, ExecutionProvider, Paths};
use super::stt::{file_name, DriverInfo};
use crate::services::local_tts::commands::LocalTtsRequest;
use crate::services::local_tts::engine::{Engine, PcmWriter, SinkPort, IDLE_UNLOAD};
use crate::services::local_tts::models::StoreLoader;
use crate::services::tts_helpers::ggml::{self, runner::rate_of_content_type};
use crate::services::tts_helpers::{TtsHelperEndpoint, TtsHelperEngine};
use crate::services::tts_sink::SinkError;

/// One rendering.
#[derive(Clone, Debug, Default)]
pub struct TtsOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Request to first audio (ms); `None` for renderings made elsewhere.
    pub ttfa_ms: Option<f64>,
    /// Request to last audio (ms).
    pub total_ms: Option<f64>,
    /// The voice that spoke.
    pub voice: Option<String>,
}

/// A TTS engine ready to render prompts.
pub trait TtsDriver {
    fn info(&self) -> &DriverInfo;

    fn render(&mut self, prompt: &TtsPrompt, timeout: Duration) -> Result<TtsOutput, String>;

    /// Whether timings are real (renderings made elsewhere have none).
    fn timed(&self) -> bool {
        true
    }
}

fn not_installed(id: &str) -> String {
    format!("not installed: {id} (Sigil.exe --headless fetch {id}, with SIGIL_APP_DATA_DIR if you use one)")
}

/// The voice for a prompt: the prompt's own, else `options.voices[lang]`, else `options.voice`.
pub fn voice_for(options: &Value, prompt: &TtsPrompt) -> String {
    if let Some(v) = prompt.voice.as_deref().filter(|v| !v.is_empty()) {
        return v.to_string();
    }
    let base = prompt.lang.split('-').next().unwrap_or("");
    options
        .get("voices")
        .and_then(|m| m.get(&prompt.lang).or_else(|| m.get(base)))
        .and_then(Value::as_str)
        .or_else(|| options.get("voice").and_then(Value::as_str))
        .unwrap_or("")
        .to_string()
}

// ---------------------------------------------------------------------------------------------
// local-tts
// ---------------------------------------------------------------------------------------------

/// What the recording sink collected for one utterance.
#[derive(Default)]
struct Recorded {
    samples: Vec<f32>,
    sample_rate: u32,
    first: Option<Instant>,
    last: Option<Instant>,
}

type Shared = Arc<Mutex<HashMap<String, Recorded>>>;

struct RecordingSink {
    utterances: Shared,
}

struct RecordingWriter {
    id: String,
    utterances: Shared,
}

impl PcmWriter for RecordingWriter {
    fn push(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), SinkError> {
        let mut map = self.utterances.lock().unwrap_or_else(PoisonError::into_inner);
        let rec = map.entry(self.id.clone()).or_default();
        let now = Instant::now();
        rec.first.get_or_insert(now);
        rec.last = Some(now);
        rec.sample_rate = sample_rate;
        rec.samples.extend_from_slice(samples);
        Ok(())
    }

    fn is_cancelled(&self) -> bool {
        false
    }
}

impl SinkPort for RecordingSink {
    fn writer(&self, utterance_id: &str) -> Result<Box<dyn PcmWriter>, SinkError> {
        Ok(Box::new(RecordingWriter {
            id: utterance_id.to_string(),
            utterances: self.utterances.clone(),
        }))
    }

    fn order(&self) -> Vec<String> {
        Vec::new()
    }

    /// Nothing is queued ahead: the first clause of a sentence is synthesized alone, as for the
    /// first sentence of a burst.
    fn queued_ms(&self) -> u32 {
        0
    }
}

pub struct LocalTtsDriver {
    engine: Engine,
    utterances: Shared,
    model: String,
    options: Value,
    speed: f32,
    threads: Option<u32>,
    info: DriverInfo,
    n: u64,
    /// The language the model is loaded for (Kokoro loads per language).
    loaded_language: String,
}

pub fn create_local(def: &EngineDef, paths: &Paths, first_language: &str) -> Result<LocalTtsDriver, String> {
    let model = super::stt::model_ref(&def.model, paths)?;
    if model.is_empty() {
        return Err("the local-tts backend needs a model (a catalog id like tts/kokoro-v1.0, or a package folder)".into());
    }
    if super::spec::is_catalog_id(&model) && crate::services::model_store::installed_path(&model).is_none() {
        return Err(not_installed(&model));
    }
    let utterances: Shared = Arc::default();
    let engine = Engine::new(Arc::new(StoreLoader::default()), Arc::new(RecordingSink { utterances: utterances.clone() }), IDLE_UNLOAD);
    let prompt = TtsPrompt {
        id: "load".into(),
        lang: first_language.to_string(),
        text: String::new(),
        kind: String::new(),
        voice: None,
        reference_voice: None,
    };
    let voice = voice_for(&def.options, &prompt);
    let started = Instant::now();
    let loaded = tauri::async_runtime::block_on(engine.load(model.clone(), voice, Some(first_language.to_string()), def.threads))
        .map_err(|_| "the local TTS worker stopped".to_string())??;
    tracing::info!(target: "sigil::bench", "local TTS {model} loaded in {} ms", started.elapsed().as_millis());
    let mut info = DriverInfo {
        device: "cpu".into(),
        threads: def.threads.map(i64::from),
        runtime: Some(format!("local TTS ({} Hz)", loaded.sample_rate)),
        ..DriverInfo::default()
    };
    if def.execution_provider.wants_gpu() {
        info.ep_reason = Some(format!("{} was requested; local TTS runs on the CPU", def.execution_provider.as_str()));
    }
    Ok(LocalTtsDriver {
        engine,
        utterances,
        model,
        options: def.options.clone(),
        speed: def.options.get("speed").and_then(Value::as_f64).map_or(1.0, |s| s as f32),
        threads: def.threads,
        info,
        n: 0,
        loaded_language: first_language.to_string(),
    })
}

impl TtsDriver for LocalTtsDriver {
    fn info(&self) -> &DriverInfo {
        &self.info
    }

    fn render(&mut self, prompt: &TtsPrompt, timeout: Duration) -> Result<TtsOutput, String> {
        self.n += 1;
        let id = format!("bench-{}-{}", self.n, prompt.id);
        let voice = voice_for(&self.options, prompt);
        // Some models load per language (Kokoro): switch before the clock starts, so a render
        // never pays for the reload.
        if !self.loaded_language.eq_ignore_ascii_case(&prompt.lang) {
            tauri::async_runtime::block_on(self.engine.load(self.model.clone(), voice.clone(), Some(prompt.lang.clone()), self.threads))
                .map_err(|_| "the local TTS worker stopped".to_string())??;
            self.loaded_language = prompt.lang.clone();
        }
        let request = LocalTtsRequest {
            model: self.model.clone(),
            voice: voice.clone(),
            speed: self.speed,
            language: Some(prompt.lang.clone()),
            text: prompt.text.clone(),
            threads: self.threads,
        };
        let submitted = Instant::now();
        let rx = self.engine.speak(request, id.clone());
        let result = tauri::async_runtime::block_on(async { tokio::time::timeout(timeout, rx).await });
        let rec = self.utterances.lock().unwrap_or_else(PoisonError::into_inner).remove(&id).unwrap_or_default();
        match result {
            Err(_) => {
                self.engine.cancel(Some(&id));
                Err(super::stt::timeout_message(timeout))
            }
            Ok(Err(_)) => Err("the local TTS worker stopped".into()),
            Ok(Ok(Err(e))) => Err(e),
            Ok(Ok(Ok(_report))) => {
                if rec.samples.is_empty() {
                    return Err("the engine produced no audio".into());
                }
                let ms = |t: Option<Instant>| t.map(|t| t.saturating_duration_since(submitted).as_secs_f64() * 1000.0);
                Ok(TtsOutput {
                    samples: rec.samples,
                    sample_rate: rec.sample_rate,
                    ttfa_ms: ms(rec.first),
                    total_ms: ms(rec.last),
                    voice: (!voice.is_empty()).then_some(voice),
                })
            }
        }
    }
}

impl Drop for LocalTtsDriver {
    fn drop(&mut self) {
        let _ = tauri::async_runtime::block_on(self.engine.unload());
    }
}

// ---------------------------------------------------------------------------------------------
// OpenAI-compatible /audio/speech over HTTP
// ---------------------------------------------------------------------------------------------

/// How the server's audio arrives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// Raw little-endian 16-bit mono.
    Pcm,
    Wav,
}

/// 16-bit little-endian PCM bytes to f32 (a trailing odd byte is dropped).
pub fn pcm16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0).collect()
}

/// Decode a WAV body (a complete file, or a streamed one with placeholder sizes): mono f32 and the rate.
pub fn decode_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32), String> {
    if let Ok(mut reader) = hound::WavReader::new(Cursor::new(bytes)) {
        let spec = reader.spec();
        let ch = usize::from(spec.channels.max(1));
        let interleaved: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
            hound::SampleFormat::Int => {
                let scale = 1.0 / (1u64 << (spec.bits_per_sample.clamp(1, 32) - 1)) as f32;
                reader.samples::<i32>().filter_map(Result::ok).map(|v| v as f32 * scale).collect()
            }
        };
        if !interleaved.is_empty() {
            let mono = interleaved.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect();
            return Ok((mono, spec.sample_rate));
        }
    }
    // A streamed WAV: find fmt and data chunks by hand and read the rest as 16-bit PCM.
    if bytes.len() < 44 || &bytes[..4] != b"RIFF" {
        return Err("the server answered with audio that is not a WAV".into());
    }
    let (mut rate, mut channels, mut pos) = (0u32, 1usize, 12usize);
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;
        let body = pos + 8;
        if id == b"fmt " && body + 16 <= bytes.len() {
            channels = usize::from(u16::from_le_bytes([bytes[body + 2], bytes[body + 3]]).max(1));
            rate = u32::from_le_bytes([bytes[body + 4], bytes[body + 5], bytes[body + 6], bytes[body + 7]]);
        } else if id == b"data" {
            let pcm = pcm16_to_f32(&bytes[body..]);
            let mono = pcm.chunks(channels).map(|f| f.iter().sum::<f32>() / channels as f32).collect();
            return if rate > 0 { Ok((mono, rate)) } else { Err("the WAV has no fmt chunk".into()) };
        }
        pos = body + size + (size & 1);
    }
    Err("the WAV has no data chunk".into())
}

/// One speech request and its audio.
pub struct HttpSpeech {
    pub url: String,
    pub key: Option<String>,
    pub body: Value,
    pub format: Format,
    pub default_rate: u32,
    pub local: bool,
}

impl HttpSpeech {
    /// POST the request and collect the audio as it streams in; the time to first audio is the first
    /// non-empty piece of the body.
    pub fn run(&self, timeout: Duration) -> Result<TtsOutput, String> {
        let client = if self.local {
            ggml::runner::loopback_client()
        } else {
            crate::services::http_client::http_client()
        };
        let mut req = client.post(&self.url).timeout(timeout).json(&self.body);
        if let Some(key) = self.key.as_deref().filter(|k| !k.is_empty()) {
            req = req.bearer_auth(key);
        }
        let (format, default_rate) = (self.format, self.default_rate);
        tauri::async_runtime::block_on(async move {
            let started = Instant::now();
            let resp = req
                .send()
                .await
                .map_err(|e| crate::services::http_client::describe_request_error("speech server", &e, timeout))?;
            if !resp.status().is_success() {
                return Err(crate::services::http_client::describe_api_error("speech server", resp).await);
            }
            let content_type = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_ascii_lowercase();
            let rate_header = rate_of_content_type(&content_type);
            let mut bytes: Vec<u8> = Vec::new();
            let (mut first, mut last) = (None::<Duration>, None::<Duration>);
            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| format!("speech stream failed: {e}"))?;
                if !chunk.is_empty() {
                    let now = started.elapsed();
                    first.get_or_insert(now);
                    last = Some(now);
                    bytes.extend_from_slice(&chunk);
                }
            }
            if bytes.is_empty() {
                return Err("the server answered with no audio".into());
            }
            if content_type.starts_with("application/json") || content_type.starts_with("text/") {
                return Err(format!("the server answered with text instead of audio: {}", crate::services::http_client::body_preview(&bytes, 200)));
            }
            // A server may answer with a WAV although raw PCM was asked.
            let is_wav = bytes.starts_with(b"RIFF") || content_type.contains("wav");
            let (samples, rate) = if is_wav || format == Format::Wav {
                decode_wav(&bytes)?
            } else {
                (pcm16_to_f32(&bytes), rate_header.unwrap_or(default_rate).max(1))
            };
            Ok(TtsOutput {
                samples,
                sample_rate: rate,
                ttfa_ms: first.map(|d| d.as_secs_f64() * 1000.0),
                total_ms: last.map(|d| d.as_secs_f64() * 1000.0),
                voice: None,
            })
        })
    }
}

/// The qwentts.cpp language name of a prompt language (`auto` for one the model lacks).
pub fn qwen_language(code: &str) -> &'static str {
    match code.split('-').next().unwrap_or("").to_ascii_lowercase().as_str() {
        "en" | "eng" => "english",
        "zh" | "cmn" | "yue" | "zho" => "chinese",
        "ja" | "jpn" => "japanese",
        "ko" => "korean",
        "de" => "german",
        "fr" => "french",
        "ru" => "russian",
        "pt" => "portuguese",
        "es" => "spanish",
        "it" => "italian",
        _ => "auto",
    }
}

/// Magpie's server language code.
pub fn magpie_language(code: &str) -> Option<&'static str> {
    Some(match code.split('-').next().unwrap_or("").to_ascii_lowercase().as_str() {
        "en" => "en-US",
        "es" => "es-US",
        "de" => "de-DE",
        "fr" => "fr-FR",
        "it" => "it-IT",
        "vi" => "vi-VN",
        "hi" => "hi-IN",
        "ko" => "ko-KR",
        "ar" => "ar-AR",
        "pt" => "pt-BR",
        _ => return None,
    })
}

/// Request members a helper engine wants for a prompt (its language spelling, defaults).
fn helper_extra(helper: TtsHelperEngine, prompt: &TtsPrompt) -> Map<String, Value> {
    let mut m = Map::new();
    match helper {
        TtsHelperEngine::Qwen3 => {
            m.insert("language".into(), json!(qwen_language(&prompt.lang)));
        }
        TtsHelperEngine::Magpie => {
            if let Some(l) = magpie_language(&prompt.lang) {
                m.insert("language".into(), json!(l));
            }
        }
        TtsHelperEngine::Omnivoice => {
            m.insert("language".into(), json!(prompt.lang.split('-').next().unwrap_or("en")));
            m.insert("seed".into(), json!(7));
        }
        TtsHelperEngine::Breeze => {
            m.insert("seed".into(), json!(42));
        }
        TtsHelperEngine::S2 | TtsHelperEngine::Sbv2 => {}
    }
    m
}

/// The request body of a prompt: OpenAI's members plus the engine's extras and `options.extra`.
pub fn speech_body(options: &Value, helper: Option<TtsHelperEngine>, prompt: &TtsPrompt, model: &str) -> Value {
    let mut body = Map::new();
    body.insert("input".into(), json!(prompt.text));
    body.insert("response_format".into(), json!(options.get("responseFormat").and_then(Value::as_str).unwrap_or("pcm")));
    if !model.is_empty() {
        body.insert("model".into(), json!(model));
    }
    let voice = voice_for(options, prompt);
    if !voice.is_empty() {
        body.insert("voice".into(), json!(voice));
    }
    if let Some(speed) = options.get("speed").and_then(Value::as_f64).filter(|s| (s - 1.0).abs() > 1e-9) {
        body.insert("speed".into(), json!(speed.clamp(0.25, 4.0)));
    }
    if options.get("stream").and_then(Value::as_bool).unwrap_or(true) {
        body.insert("stream".into(), json!(true));
    }
    if let Some(h) = helper {
        body.extend(helper_extra(h, prompt));
    }
    if let Some(extra) = options.get("extra").and_then(Value::as_object) {
        body.extend(extra.clone());
    }
    Value::Object(body)
}

// ---------------------------------------------------------------------------------------------
// tts-helper
// ---------------------------------------------------------------------------------------------

/// The helper engine of `options.helper`.
pub fn helper_engine(name: &str) -> Result<TtsHelperEngine, String> {
    TtsHelperEngine::ALL
        .into_iter()
        .find(|e| e.as_str() == name && *e != TtsHelperEngine::Sbv2)
        .ok_or_else(|| format!("options.helper must be one of qwen3, magpie, breeze, omnivoice, s2 (not {name:?})"))
}

pub struct HelperTtsDriver {
    helper: TtsHelperEngine,
    endpoint: TtsHelperEndpoint,
    options: Value,
    info: DriverInfo,
}

fn stop_helper(helper: TtsHelperEngine) {
    use crate::services::tts_helpers::{breeze, magpie, omnivoice, qwen, s2};
    let result = match helper {
        TtsHelperEngine::Qwen3 => tauri::async_runtime::block_on(qwen::stop()),
        TtsHelperEngine::Magpie => tauri::async_runtime::block_on(magpie::stop()),
        TtsHelperEngine::Breeze => tauri::async_runtime::block_on(breeze::stop()),
        TtsHelperEngine::Omnivoice => tauri::async_runtime::block_on(omnivoice::stop()),
        TtsHelperEngine::S2 => tauri::async_runtime::block_on(s2::stop()),
        TtsHelperEngine::Sbv2 => Ok(()),
    };
    if let Err(e) = result {
        tracing::warn!(target: "sigil::bench", "stopping the {} helper: {e}", helper.as_str());
    }
}

/// Start the helper `options.helper` with `options.config` (the app's own planning, without an
/// `AppHandle`: the engines' `start` functions ignore theirs) and return what it runs on.
pub fn create_helper(def: &EngineDef) -> Result<HelperTtsDriver, String> {
    use crate::services::tts_helpers::{breeze, magpie, omnivoice, qwen, s2};
    let helper = helper_engine(def.options.get("helper").and_then(Value::as_str).unwrap_or(""))?;
    let config = def.options.get("config").cloned().unwrap_or_else(|| json!({}));
    // Variant: the engine's execution provider unless the config names one.
    let mut config = config;
    if let Some(map) = config.as_object_mut() {
        if map.get("variant").and_then(Value::as_str).is_none_or(str::is_empty) {
            let v = match def.execution_provider {
                ExecutionProvider::Cpu => Some("cpu"),
                ExecutionProvider::Cuda => Some("cuda13"),
                ExecutionProvider::Vulkan => Some("vulkan"),
                ExecutionProvider::Dml | ExecutionProvider::Auto => None,
            };
            if let Some(v) = v {
                map.insert("variant".into(), json!(v));
            }
        }
    }
    let started = Instant::now();
    let (endpoint, variant, vram_mb, model) = match helper {
        TtsHelperEngine::Qwen3 => {
            let cfg: qwen::plan::QwenConfig = serde_json::from_value(config).map_err(|e| format!("options.config (Qwen3-TTS): {e}"))?;
            let (launch, _plan) = qwen::server::resolve(&cfg).map_err(super::stt::normalize_missing)?;
            let (variant, vram, model) = (launch.variant.as_str().to_string(), launch.vram_mb, launch.model.clone());
            (tauri::async_runtime::block_on(qwen::server::start(cfg)).map_err(super::stt::normalize_missing)?, variant, vram, model)
        }
        other => {
            let cfg: ggml::GgmlConfig = serde_json::from_value(config).map_err(|e| format!("options.config ({}): {e}", other.as_str()))?;
            macro_rules! ggml_start {
                ($m:ident) => {{
                    let launch = ggml::with_plan_env(|env| $m::plan(&cfg, env)).map_err(super::stt::normalize_missing)?;
                    let (variant, vram, model) = (launch.variant.as_str().to_string(), launch.vram_mb, launch.model.clone());
                    let ep = tauri::async_runtime::block_on($m::SLOT.start(move || Ok(launch)))?;
                    (ep, variant, vram, model)
                }};
            }
            match other {
                TtsHelperEngine::Magpie => ggml_start!(magpie),
                TtsHelperEngine::Breeze => ggml_start!(breeze),
                TtsHelperEngine::Omnivoice => ggml_start!(omnivoice),
                TtsHelperEngine::S2 => ggml_start!(s2),
                _ => return Err(format!("the {} helper cannot be benchmarked here", other.as_str())),
            }
        }
    };
    tracing::info!(target: "sigil::bench", "{} helper ready in {} ms", helper.as_str(), started.elapsed().as_millis());
    let device = match variant.as_str() {
        "cuda13" | "cuda" => "cuda",
        "vulkan" => "vulkan",
        _ => "cpu",
    };
    let mut info = DriverInfo {
        device: device.into(),
        variant: Some(variant.clone()),
        threads: def.threads.map(i64::from),
        vram_mb: Some(vram_mb),
        runtime: Some(format!("{} helper ({model})", helper.as_str())),
        helper_process: true,
        ..DriverInfo::default()
    };
    if def.execution_provider.wants_gpu() && device != def.execution_provider.as_str() {
        info.ep_reason = Some(format!("{} was requested; the {variant} build runs", def.execution_provider.as_str()));
    }
    Ok(HelperTtsDriver {
        helper,
        endpoint,
        options: def.options.clone(),
        info,
    })
}

impl TtsDriver for HelperTtsDriver {
    fn info(&self) -> &DriverInfo {
        &self.info
    }

    fn render(&mut self, prompt: &TtsPrompt, timeout: Duration) -> Result<TtsOutput, String> {
        let body = speech_body(&self.options, Some(self.helper), prompt, "");
        let speech = HttpSpeech {
            url: format!("{}/audio/speech", self.endpoint.base_url.trim_end_matches('/')),
            key: Some(self.endpoint.api_key.clone()),
            body,
            format: Format::Pcm,
            default_rate: if self.endpoint.pcm_sample_rate > 0 { self.endpoint.pcm_sample_rate } else { 24_000 },
            local: true,
        };
        let mut out = speech.run(timeout)?;
        let voice = voice_for(&self.options, prompt);
        out.voice = (!voice.is_empty()).then_some(voice);
        Ok(out)
    }
}

impl Drop for HelperTtsDriver {
    fn drop(&mut self) {
        stop_helper(self.helper);
    }
}

// ---------------------------------------------------------------------------------------------
// openai-speech
// ---------------------------------------------------------------------------------------------

pub struct OpenAiSpeechDriver {
    url: String,
    key: Option<String>,
    model: String,
    options: Value,
    format: Format,
    rate: u32,
    local: bool,
    info: DriverInfo,
}

pub fn create_openai_speech(def: &EngineDef) -> Result<OpenAiSpeechDriver, String> {
    let o = &def.options;
    let base = o.get("baseUrl").and_then(Value::as_str).ok_or("the openai-speech backend needs options.baseUrl")?;
    let url = crate::services::http_client::api_url(base, "/audio/speech")?;
    let key = match o.get("apiKeyEnv").and_then(Value::as_str).filter(|v| !v.is_empty()) {
        Some(var) => Some(std::env::var(var).map_err(|_| format!("the environment variable {var} (options.apiKeyEnv) is not set"))?),
        None => None,
    };
    let local = crate::services::http_client::is_local_endpoint(&url);
    let format = match o.get("responseFormat").and_then(Value::as_str).unwrap_or("pcm") {
        "wav" => Format::Wav,
        "pcm" => Format::Pcm,
        other => return Err(format!("options.responseFormat must be pcm or wav, not {other:?}")),
    };
    let model = o.get("model").and_then(Value::as_str).unwrap_or(&def.model).to_string();
    Ok(OpenAiSpeechDriver {
        url: url.to_string(),
        key,
        model,
        options: o.clone(),
        format,
        rate: o.get("pcmSampleRate").and_then(Value::as_u64).map_or(24_000, |r| r.clamp(8_000, 192_000) as u32),
        local,
        info: DriverInfo {
            device: "remote".into(),
            runtime: Some("OpenAI-compatible speech server".into()),
            ..DriverInfo::default()
        },
    })
}

impl TtsDriver for OpenAiSpeechDriver {
    fn info(&self) -> &DriverInfo {
        &self.info
    }

    fn render(&mut self, prompt: &TtsPrompt, timeout: Duration) -> Result<TtsOutput, String> {
        let mut options = self.options.clone();
        // OpenAI itself has no `stream` member; servers that stream answer progressively anyway.
        if self.url.contains("api.openai.com") && options.get("stream").is_none() {
            options["stream"] = json!(false);
        }
        let body = speech_body(&options, None, prompt, &self.model);
        let speech = HttpSpeech {
            url: self.url.clone(),
            key: self.key.clone(),
            body,
            format: self.format,
            default_rate: self.rate,
            local: self.local,
        };
        let mut out = speech.run(timeout)?;
        let voice = voice_for(&self.options, prompt);
        out.voice = (!voice.is_empty()).then_some(voice);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------
// rendered
// ---------------------------------------------------------------------------------------------

/// Renderings made elsewhere: `<dir>/<prompt id>.wav`.
pub struct RenderedDriver {
    dir: std::path::PathBuf,
    info: DriverInfo,
}

pub fn create_rendered(def: &EngineDef, paths: &Paths) -> Result<RenderedDriver, String> {
    let dir = def.options.get("dir").and_then(Value::as_str).ok_or("the rendered backend needs options.dir")?;
    let dir = paths.resolve(dir)?;
    if !dir.is_dir() {
        return Err(format!("the rendered folder {} does not exist", file_name(&dir)));
    }
    Ok(RenderedDriver {
        dir,
        info: DriverInfo {
            device: "none".into(),
            ..DriverInfo::default()
        },
    })
}

impl TtsDriver for RenderedDriver {
    fn info(&self) -> &DriverInfo {
        &self.info
    }

    fn render(&mut self, prompt: &TtsPrompt, _timeout: Duration) -> Result<TtsOutput, String> {
        let path = self.dir.join(format!("{}.wav", prompt.id));
        if !path.is_file() {
            return Err(format!("no rendering {}.wav", prompt.id));
        }
        let (samples, rate) = crate::services::speech_rt::sherpa::probe::read_wav(&path)?;
        Ok(TtsOutput {
            samples,
            sample_rate: rate,
            ..TtsOutput::default()
        })
    }

    fn timed(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------------------------
// Creation and files
// ---------------------------------------------------------------------------------------------

/// Create the TTS driver of `def` (blocking: loads the model / starts the helper). `first_language`
/// is the language of the first prompt the engine renders (the local worker loads for it).
pub fn create_driver(def: &EngineDef, paths: &Paths, first_language: &str) -> Result<Box<dyn TtsDriver>, String> {
    match def.backend.as_str() {
        "local-tts" => create_local(def, paths, first_language).map(|d| Box::new(d) as Box<dyn TtsDriver>),
        "tts-helper" => create_helper(def).map(|d| Box::new(d) as Box<dyn TtsDriver>),
        "openai-speech" => create_openai_speech(def).map(|d| Box::new(d) as Box<dyn TtsDriver>),
        "rendered" => create_rendered(def, paths).map(|d| Box::new(d) as Box<dyn TtsDriver>),
        other => Err(format!("{other} is not a TTS backend")),
    }
}

/// Write mono f32 samples as a 16-bit PCM WAV (clipped to [-1, 1]).
pub fn write_pcm16_wav(path: &Path, samples: &[f32], rate: u32) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let err = |e: hound::Error| format!("cannot write {}: {e}", file_name(path));
    let mut w = hound::WavWriter::create(path, spec).map_err(err)?;
    for &s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).map_err(err)?;
    }
    w.finalize().map_err(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prompt(id: &str, lang: &str) -> TtsPrompt {
        TtsPrompt {
            id: id.into(),
            lang: lang.into(),
            text: "Hello there.".into(),
            kind: "casual".into(),
            voice: None,
            reference_voice: None,
        }
    }

    #[test]
    fn voices_come_from_the_prompt_then_the_language_map_then_the_default() {
        let o = json!({"voice": "af_heart", "voices": {"ja": "jf_alpha", "pt-BR": "pf_dora"}});
        assert_eq!(voice_for(&o, &prompt("a", "en")), "af_heart");
        assert_eq!(voice_for(&o, &prompt("a", "ja")), "jf_alpha");
        assert_eq!(voice_for(&o, &prompt("a", "pt-BR")), "pf_dora");
        let mut p = prompt("a", "ja");
        p.voice = Some("custom".into());
        assert_eq!(voice_for(&o, &p), "custom");
        assert_eq!(voice_for(&json!({}), &prompt("a", "en")), "");
    }

    #[test]
    fn speech_bodies_follow_the_openai_shape_with_engine_extras() {
        let o = json!({"voices": {"en": "ryan"}, "speed": 1.25, "extra": {"seed": 5}});
        let b = speech_body(&o, Some(TtsHelperEngine::Qwen3), &prompt("a", "ja"), "");
        assert_eq!(b["input"], "Hello there.");
        assert_eq!(b["response_format"], "pcm");
        assert_eq!(b["stream"], true);
        assert_eq!(b["speed"], 1.25);
        assert_eq!(b["language"], "japanese");
        assert_eq!(b["seed"], 5);
        assert!(b.get("voice").is_none() && b.get("model").is_none(), "unset members are not sent");
        let b = speech_body(&o, Some(TtsHelperEngine::Qwen3), &prompt("a", "en"), "m");
        assert_eq!((b["voice"].as_str(), b["model"].as_str()), (Some("ryan"), Some("m")));
        let b = speech_body(&json!({"stream": false}), Some(TtsHelperEngine::Magpie), &prompt("a", "es"), "");
        assert_eq!((b["language"].as_str(), b.get("stream").is_none()), (Some("es-US"), true));
        assert_eq!(speech_body(&json!({}), Some(TtsHelperEngine::Omnivoice), &prompt("a", "pt-BR"), "")["language"], "pt");
        assert!(speech_body(&json!({}), Some(TtsHelperEngine::Magpie), &prompt("a", "ja"), "").get("language").is_none());
        assert_eq!(qwen_language("zh-Hans"), "chinese");
        assert_eq!(qwen_language("ar"), "auto");
    }

    #[test]
    fn pcm_and_wav_bodies_decode() {
        let pcm: Vec<u8> = [0i16, 16384, -16384, 32767].iter().flat_map(|s| s.to_le_bytes()).collect();
        let f = pcm16_to_f32(&pcm);
        assert_eq!(f.len(), 4);
        assert!((f[1] - 0.5).abs() < 1e-4 && (f[2] + 0.5).abs() < 1e-4);
        assert_eq!(pcm16_to_f32(&pcm[..3]).len(), 1, "a trailing odd byte is dropped");
        // A complete WAV through hound.
        let dir = std::env::temp_dir().join(format!("sigil-speech-tts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.wav");
        write_pcm16_wav(&path, &[0.0, 0.5, -0.5, 1.5], 22_050).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let (samples, rate) = decode_wav(&bytes).unwrap();
        assert_eq!(rate, 22_050);
        assert_eq!(samples.len(), 4);
        assert!((samples[3] - 1.0).abs() < 1e-3, "clipped to full scale");
        // A streamed WAV: placeholder sizes in the RIFF and data headers.
        let mut streamed = bytes.clone();
        streamed[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        let data_at = streamed.windows(4).position(|w| w == b"data").unwrap();
        streamed[data_at + 4..data_at + 8].copy_from_slice(&u32::MAX.to_le_bytes());
        let (s2, r2) = decode_wav(&streamed).unwrap();
        assert_eq!((s2.len(), r2), (4, 22_050));
        assert!(decode_wav(b"not a wav at all, definitely not a wav, not even close to one").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn helper_names_are_checked() {
        assert_eq!(helper_engine("qwen3").unwrap(), TtsHelperEngine::Qwen3);
        assert_eq!(helper_engine("s2").unwrap(), TtsHelperEngine::S2);
        assert!(helper_engine("sbv2").is_err(), "sbv2 runs in process and is not a helper here");
        assert!(helper_engine("").unwrap_err().contains("qwen3"));
    }

    #[test]
    fn a_rendered_folder_serves_files_by_prompt_id() {
        let dir = std::env::temp_dir().join(format!("sigil-speech-rendered-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_pcm16_wav(&dir.join("en-casual-1.wav"), &[0.1; 2400], 24_000).unwrap();
        let def: EngineDef = serde_json::from_value(json!({
            "id": "r", "task": "tts", "backend": "rendered", "options": {"dir": dir.to_string_lossy()}
        }))
        .unwrap();
        let mut d = create_rendered(&def, &Paths::with_base(std::env::temp_dir())).unwrap();
        let out = d.render(&prompt("en-casual-1", "en"), Duration::from_secs(1)).unwrap();
        assert_eq!((out.samples.len(), out.sample_rate), (2400, 24_000));
        assert!(out.ttfa_ms.is_none() && !d.timed());
        assert!(d.render(&prompt("missing", "en"), Duration::from_secs(1)).unwrap_err().contains("no rendering"));
        let bad: EngineDef = serde_json::from_value(json!({
            "id": "r", "task": "tts", "backend": "rendered", "options": {"dir": dir.join("nope").to_string_lossy()}
        }))
        .unwrap();
        assert!(create_rendered(&bad, &Paths::with_base(std::env::temp_dir())).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_local_model_fails_the_engine_with_the_catalog_id() {
        let def: EngineDef = serde_json::from_value(json!({
            "id": "k", "task": "tts", "backend": "local-tts", "model": "tts/no-such-voice-for-the-test"
        }))
        .unwrap();
        let err = create_local(&def, &Paths::with_base(std::env::temp_dir()), "en").err().unwrap();
        assert!(err.contains("not installed: tts/no-such-voice-for-the-test"), "{err}");
    }

    #[test]
    fn the_openai_speech_backend_needs_a_base_url_and_a_set_key() {
        let mk = |options: Value| -> EngineDef { serde_json::from_value(json!({"id": "o", "task": "tts", "backend": "openai-speech", "options": options})).unwrap() };
        assert!(create_openai_speech(&mk(json!({}))).err().unwrap().contains("baseUrl"));
        assert!(create_openai_speech(&mk(json!({"baseUrl": "http://127.0.0.1:8880", "responseFormat": "mp3"}))).err().unwrap().contains("pcm or wav"));
        assert!(create_openai_speech(&mk(json!({"baseUrl": "http://127.0.0.1:8880", "apiKeyEnv": "SIGIL_SPEECH_BENCH_NO_KEY"})))
            .err()
            .unwrap()
            .contains("SIGIL_SPEECH_BENCH_NO_KEY"));
        let d = create_openai_speech(&mk(json!({"baseUrl": "http://127.0.0.1:8880", "model": "kokoro", "voice": "af_heart"}))).unwrap();
        assert!(d.url.ends_with("/audio/speech") && d.local && d.model == "kokoro");
    }
}
