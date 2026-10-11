//! STT engines of the speech bench: the app's own recognizers behind one small driver trait.
//!
//! Backends (the spec's `engines[].backend`):
//! - `whisper`, `http`, `sherpa-online`, `sherpa-offline`, `transcribe`: `speech::engines::create`
//!   with an [`EngineSpec`] (the router's own engines; nothing is re-implemented), driven through
//!   `AsrEngine::transcribe` (feed `utterance`) or `begin_stream` / `push_stream` / `end_stream`
//!   (feed `stream`). An engine that cannot stream still runs a stream feed: the clip is fed at the
//!   chosen pace and decoded once at its end (the run says so in its warnings).
//! - `pipeline`: the real speech pipeline (Silero VAD segmentation, router, two-pass finals) fed from
//!   the clips ([`super::stt_pipeline`]).
//! - `crispasr`: the pinned CrispASR runtime as a `process_host` helper (`crispasr --server`), spoken
//!   to through the app's OpenAI-compatible HTTP engine (`/v1/audio/transcriptions`).
//! - `hypotheses`: JSON lines `{item, text}` made elsewhere.
//!
//! An engine whose model or runtime pack is not installed fails with the engine's own message
//! (`not_installed: <catalog id>`); the runner reports it for that engine and carries on.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::report::Segment;
use super::spec::{is_catalog_id, EngineDef, ExecutionProvider, Feed, Pace, Paths};
use crate::services::speech::engine_api::{AsrEngine, AsrError, AsrErrorKind, Canceller, DecodeOptions, EngineContext, EngineInfo, EngineKind, EngineSpec};

/// Samples per millisecond at 16 kHz.
pub const SAMPLES_PER_MS: usize = 16;

/// One item to transcribe.
pub struct SttRequest<'a> {
    /// Corpus item id.
    pub item: &'a str,
    /// 16 kHz mono audio of the clip.
    pub audio: &'a [f32],
    /// Normalized language to give the engine (`None`: let it detect).
    pub language: Option<String>,
    pub feed: &'a Feed,
    pub timeout: Duration,
}

/// What a driver returns for one item.
#[derive(Clone, Debug, Default)]
pub struct SttOutput {
    pub text: String,
    pub language: Option<String>,
    pub tags: Vec<String>,
    pub segments: Vec<Segment>,
    pub partials: Option<u64>,
    pub first_partial_ms: Option<f64>,
    pub final_latency_ms: Option<f64>,
    /// Engine busy time; `None`: the caller's wall time around the call.
    pub compute_ms: Option<f64>,
}

/// Facts about a created engine (for the report's engine row).
#[derive(Clone, Debug, Default)]
pub struct DriverInfo {
    /// `cpu`, `cuda`, `dml`, `vulkan` or `remote`.
    pub device: String,
    /// Why it runs on a lesser device than requested.
    pub ep_reason: Option<String>,
    pub variant: Option<String>,
    pub threads: Option<i64>,
    pub vram_mb: Option<u64>,
    /// Runtime and version (`sherpa-onnx 1.13.8`).
    pub runtime: Option<String>,
    pub streaming: bool,
    /// The engine runs in a helper process (VRAM is read device-wide).
    pub helper_process: bool,
    /// More facts for the report's engine row (the pipeline's plan).
    pub extra: Option<Value>,
}

/// An STT engine ready to transcribe items.
pub trait SttDriver {
    fn info(&self) -> &DriverInfo;

    /// Transcribe one item.
    fn transcribe(&mut self, req: &SttRequest<'_>) -> Result<SttOutput, String>;

    /// The driver feeds consecutive items of one language as one stream (the pipeline): the runner
    /// calls [`transcribe_group`](Self::transcribe_group) once per language run of items.
    fn grouped(&self) -> bool {
        false
    }

    /// Transcribe a group of items of one language (default: one by one).
    fn transcribe_group(&mut self, reqs: &[SttRequest<'_>]) -> Vec<Result<SttOutput, String>> {
        reqs.iter().map(|r| self.transcribe(r)).collect()
    }

    /// Things the run should say (shown once per engine in the run's warnings).
    fn notes(&self) -> Vec<String> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------------------------
// Deadlines
// ---------------------------------------------------------------------------------------------

/// Run `f`, cancelling it through `canceller` when it takes longer than `timeout`; returns its value
/// and whether the deadline fired.
pub fn with_deadline<T>(canceller: Option<Canceller>, timeout: Duration, f: impl FnOnce() -> T) -> (T, bool) {
    let fired = AtomicBool::new(false);
    std::thread::scope(|s| {
        let (tx, rx) = mpsc::channel::<()>();
        let fired_ref = &fired;
        let canceller = canceller;
        s.spawn(move || {
            if let Err(mpsc::RecvTimeoutError::Timeout) = rx.recv_timeout(timeout) {
                fired_ref.store(true, Ordering::SeqCst);
                if let Some(c) = &canceller {
                    c.cancel();
                }
            }
        });
        let out = f();
        drop(tx);
        (out, fired.load(Ordering::SeqCst))
    })
}

pub fn timeout_message(timeout: Duration) -> String {
    if timeout < Duration::from_secs(1) {
        format!("timeout after {} ms", timeout.as_millis())
    } else {
        format!("timeout after {} s", timeout.as_secs_f64().round() as u64)
    }
}

// ---------------------------------------------------------------------------------------------
// The app's engines
// ---------------------------------------------------------------------------------------------

/// Engines word a missing download as "<Title> needs downloads first: id1, id2"; the runner reports
/// it as "not installed: <ids>" with the command that installs them.
pub fn normalize_missing(message: String) -> String {
    match message.split_once("needs downloads first: ") {
        Some((_, ids)) => format!("not installed: {ids} (Sigil.exe --headless fetch {})", ids.split(", ").collect::<Vec<_>>().join(" ")),
        None => message,
    }
}

fn asr_message(e: &AsrError) -> String {
    match e.kind {
        AsrErrorKind::NotInstalled => format!("not installed: {}", e.message),
        _ => e.to_string(),
    }
}

fn silence(ms: u32) -> Vec<f32> {
    vec![0.0; ms as usize * SAMPLES_PER_MS]
}

/// A created `AsrEngine` and how it is driven.
pub struct EngineDriver {
    engine: Box<dyn AsrEngine>,
    info: DriverInfo,
    caps_streaming: bool,
    caps_utterance: bool,
    /// Held while the engine's helper process lives (stops it on drop).
    _helper: Option<HelperGuard>,
    notes: Vec<String>,
    noted_fallback: bool,
}

impl EngineDriver {
    /// Wrap a created engine; `requested` is the spec's execution provider (a lesser device than
    /// asked is recorded with its reason, never hidden).
    pub fn new(engine: Box<dyn AsrEngine>, requested: ExecutionProvider, threads: Option<u32>, helper: Option<HelperGuard>) -> EngineDriver {
        let info = driver_info(engine.info(), requested, threads, helper.is_some());
        let caps = engine.info().caps.clone();
        EngineDriver {
            engine,
            info,
            caps_streaming: caps.streaming,
            caps_utterance: caps.utterance,
            _helper: helper,
            notes: Vec::new(),
            noted_fallback: false,
        }
    }

    fn decode_options(req: &SttRequest<'_>) -> DecodeOptions {
        DecodeOptions {
            language: req.language.clone(),
            hotwords: Vec::new(),
            prompt: None,
        }
    }

    fn run_utterance(&mut self, req: &SttRequest<'_>) -> Result<SttOutput, String> {
        let opts = Self::decode_options(req);
        let started = Instant::now();
        let canceller = self.engine.canceller();
        let (result, fired) = with_deadline(canceller, req.timeout, || self.engine.transcribe(req.audio, &opts));
        let compute = started.elapsed();
        if fired {
            return Err(timeout_message(req.timeout));
        }
        let r = result.map_err(|e| asr_message(&e))?;
        let ms = compute.as_secs_f64() * 1000.0;
        Ok(SttOutput {
            text: r.text,
            language: r.lang,
            tags: r.tags,
            compute_ms: Some(ms),
            // An utterance engine's final arrives when its decode ends.
            final_latency_ms: Some(ms),
            ..SttOutput::default()
        })
    }

    /// The stream feed: lead silence, the clip and the tail silence in `chunk_ms` blocks.
    fn run_stream(&mut self, req: &SttRequest<'_>) -> Result<SttOutput, String> {
        let feed = req.feed;
        let mut padded = silence(feed.lead_silence_ms);
        padded.extend_from_slice(req.audio);
        padded.extend(silence(feed.tail_silence_ms));
        let total = Duration::from_micros((padded.len() as u64 * 1_000_000) / 16_000);
        let realtime = feed.pace == Pace::Realtime;
        let opts = Self::decode_options(req);
        let started = Instant::now();

        if !self.caps_streaming {
            if !self.noted_fallback {
                self.noted_fallback = true;
                self.notes.push(
                    "the engine does not stream: the stream feed delivered the clip at the chosen pace and decoded it once at its end (first_partial_ms is absent; final_latency_ms is the decode after the last sample)".into(),
                );
            }
            if realtime {
                if let Some(wait) = total.checked_sub(started.elapsed()) {
                    std::thread::sleep(wait);
                }
            }
            let ideal_end = started + total;
            let canceller = self.engine.canceller();
            let c0 = Instant::now();
            let (result, fired) = with_deadline(canceller, req.timeout, || self.engine.transcribe(req.audio, &opts));
            let compute = c0.elapsed();
            if fired {
                return Err(timeout_message(req.timeout));
            }
            let r = result.map_err(|e| asr_message(&e))?;
            let after_end = if realtime { Instant::now().saturating_duration_since(ideal_end) } else { compute };
            return Ok(SttOutput {
                text: r.text,
                language: r.lang,
                tags: r.tags,
                compute_ms: Some(compute.as_secs_f64() * 1000.0),
                final_latency_ms: Some(after_end.as_secs_f64() * 1000.0),
                ..SttOutput::default()
            });
        }

        let canceller = self.engine.canceller();
        let timeout = req.timeout;
        // The item's time to finish. At real time the feed itself takes `total` (the audio is spoken at
        // its own speed inside the deadline), so the engine gets its timeout on top of that; this budget
        // serves the watchdog and the in-loop check alike.
        let budget = if realtime { timeout + total } else { timeout };
        let engine = &mut self.engine;
        let (result, fired) = with_deadline(canceller, budget, || -> Result<SttOutput, String> {
            engine.begin_stream(&opts).map_err(|e| asr_message(&e))?;
            let t0 = Instant::now();
            let chunk = (feed.chunk_ms as usize * SAMPLES_PER_MS).max(1);
            let mut busy = Duration::ZERO;
            let (mut partials, mut first_partial) = (0u64, None::<Duration>);
            for (n, block) in padded.chunks(chunk).enumerate() {
                if realtime {
                    // The block is available once it has been spoken.
                    let due = Duration::from_micros((((n * chunk + block.len()) as u64) * 1_000_000) / 16_000);
                    if let Some(wait) = due.checked_sub(t0.elapsed()) {
                        std::thread::sleep(wait);
                    }
                }
                if t0.elapsed() > budget {
                    engine.abort_stream();
                    return Err(timeout_message(timeout));
                }
                let c0 = Instant::now();
                let pushed = engine.push_stream(block);
                busy += c0.elapsed();
                match pushed {
                    Ok(Some(h)) if !h.text.trim().is_empty() => {
                        partials += 1;
                        first_partial.get_or_insert_with(|| t0.elapsed());
                    }
                    Ok(_) => {}
                    Err(e) => {
                        engine.abort_stream();
                        return Err(asr_message(&e));
                    }
                }
            }
            let end_started = Instant::now();
            let finished = engine.end_stream();
            let end_took = end_started.elapsed();
            busy += end_took;
            let r = finished.map_err(|e| asr_message(&e))?;
            let after_end = if realtime { Instant::now().saturating_duration_since(t0 + total) } else { end_took };
            let lead = Duration::from_millis(u64::from(feed.lead_silence_ms));
            Ok(SttOutput {
                text: r.text,
                language: r.lang,
                tags: r.tags,
                partials: Some(partials),
                first_partial_ms: first_partial.map(|d| (if realtime { d.saturating_sub(lead) } else { d }).as_secs_f64() * 1000.0),
                final_latency_ms: Some(after_end.as_secs_f64() * 1000.0),
                compute_ms: Some(busy.as_secs_f64() * 1000.0),
                ..SttOutput::default()
            })
        });
        if fired {
            return Err(timeout_message(timeout));
        }
        result
    }
}

impl SttDriver for EngineDriver {
    fn info(&self) -> &DriverInfo {
        &self.info
    }

    fn transcribe(&mut self, req: &SttRequest<'_>) -> Result<SttOutput, String> {
        use super::spec::FeedKind;
        match req.feed.kind {
            FeedKind::Stream => self.run_stream(req),
            // An engine without utterance decoding (a streaming-only model) is fed as a stream.
            FeedKind::Utterance if !self.caps_utterance && self.caps_streaming => self.run_stream(&SttRequest {
                feed: &Feed {
                    pace: Pace::Asap,
                    lead_silence_ms: 0,
                    ..req.feed.clone()
                },
                item: req.item,
                audio: req.audio,
                language: req.language.clone(),
                timeout: req.timeout,
            }),
            _ => self.run_utterance(req),
        }
    }

    fn notes(&self) -> Vec<String> {
        self.notes.clone()
    }
}

/// `EngineInfo` of a created engine as the report's driver facts.
pub fn driver_info(info: &EngineInfo, requested: ExecutionProvider, threads: Option<u32>, helper: bool) -> DriverInfo {
    let device = match info.device.as_str() {
        "directml" => "dml".to_string(),
        other => other.to_string(),
    };
    let mut reason = info.fallback_reason.clone();
    let asked = requested.as_str();
    if reason.is_none() && requested.wants_gpu() && device != asked && device != "remote" {
        reason = Some(format!("{asked} was requested; the engine runs on {device}"));
    }
    let variant = info
        .runtime
        .split_whitespace()
        .last()
        .filter(|v| matches!(*v, "cpu" | "cpu-avx2" | "cpu-avx512" | "vulkan" | "cuda13" | "cuda"))
        .map(str::to_string);
    DriverInfo {
        device,
        ep_reason: reason,
        variant,
        threads: threads.map(i64::from),
        vram_mb: Some(info.vram_mb),
        runtime: Some(info.runtime.clone()),
        streaming: info.caps.streaming,
        helper_process: helper,
        extra: None,
    }
}

// ---------------------------------------------------------------------------------------------
// EngineSpec from a spec engine
// ---------------------------------------------------------------------------------------------

/// The speech plugin's engine kind of a backend wire name.
pub fn engine_kind(backend: &str) -> Option<EngineKind> {
    EngineKind::ALL.into_iter().find(|k| k.as_str() == backend)
}

/// The model of a spec engine: a catalog id as is, anything else is a bring-your-own path (`${NAME}`
/// expanded, relative to the spec's folder).
pub fn model_ref(model: &str, paths: &Paths) -> Result<String, String> {
    let m = model.trim();
    if m.is_empty() || is_catalog_id(m) || (!m.contains("${") && !m.contains('/') && !m.contains('\\')) {
        // Bare Whisper ids (`base.en`) and catalog ids pass through.
        return Ok(m.to_string());
    }
    Ok(paths.resolve(m)?.to_string_lossy().into_owned())
}

/// Resolve `options.apiKeyEnv` into the `apiKey` the HTTP engine reads (the key never enters the spec
/// or the report).
pub fn with_api_key(options: &Value) -> Result<Value, String> {
    let mut opts = options.clone();
    if let Some(map) = opts.as_object_mut() {
        if let Some(var) = map.remove("apiKeyEnv").and_then(|v| v.as_str().map(str::to_string)).filter(|v| !v.is_empty()) {
            let key = std::env::var(&var).map_err(|_| format!("the environment variable {var} (options.apiKeyEnv) is not set"))?;
            map.insert("apiKey".into(), json!(key));
        }
    }
    Ok(opts)
}

/// The runtime variant a Transcribe engine is asked for.
fn transcribe_variant(def: &EngineDef) -> Option<String> {
    if let Some(v) = def.variant.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
        return Some(v.to_string());
    }
    match def.execution_provider {
        ExecutionProvider::Cpu => Some("cpu-any".into()),
        ExecutionProvider::Cuda => Some("cuda13".into()),
        ExecutionProvider::Vulkan => Some("vulkan".into()),
        ExecutionProvider::Dml | ExecutionProvider::Auto => None,
    }
}

/// The `EngineSpec` of a spec engine whose backend is an `EngineKind`.
pub fn engine_spec(def: &EngineDef, paths: &Paths) -> Result<EngineSpec, String> {
    let kind = engine_kind(&def.backend).ok_or_else(|| format!("{} is not a speech engine backend", def.backend))?;
    let mut spec = EngineSpec::new(kind, model_ref(&def.model, paths)?);
    spec.threads = def.threads;
    spec.variant = match kind {
        EngineKind::Transcribe => transcribe_variant(def),
        _ => def.variant.clone().filter(|v| !v.trim().is_empty()),
    };
    spec.options = if kind == EngineKind::Http { with_api_key(&def.options)? } else { def.options.clone() };
    Ok(spec)
}

/// Create the engine of `def` (blocking: loads the model).
pub fn create_engine_driver(def: &EngineDef, paths: &Paths) -> Result<Box<dyn SttDriver>, String> {
    let spec = engine_spec(def, paths)?;
    let engine = crate::services::speech::engines::create(&spec, &EngineContext::headless()).map_err(|e| asr_message(&e))?;
    Ok(Box::new(EngineDriver::new(engine, def.execution_provider, def.threads, None)))
}

// ---------------------------------------------------------------------------------------------
// Pre-computed hypotheses
// ---------------------------------------------------------------------------------------------

/// Hypotheses made elsewhere: JSON lines `{item, text}` (also `{id, hyp}` as mt-bench writes them).
pub struct HypothesesDriver {
    hyps: HashMap<String, String>,
    info: DriverInfo,
}

/// Parse a hypotheses file.
pub fn parse_hypotheses(text: &str) -> Result<HashMap<String, String>, String> {
    let mut out = HashMap::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| format!("hypotheses line {}: {e}", n + 1))?;
        let id = v.get("item").or_else(|| v.get("id")).and_then(Value::as_str);
        // A null text is what the runner's own hyps files hold for an item that failed (`"text": null`
        // beside its error): that item has no hypothesis, and the other lines stay usable.
        let given = |key: &str| v.get(key).filter(|t| !t.is_null());
        let txt = given("text").or_else(|| given("hyp")).and_then(Value::as_str);
        let only_null = given("text").is_none() && given("hyp").is_none() && (v.get("text").is_some() || v.get("hyp").is_some());
        match (id, txt) {
            (Some(id), Some(txt)) => {
                out.insert(id.to_string(), txt.to_string());
            }
            (Some(_), None) if only_null => {}
            _ => return Err(format!("hypotheses line {}: needs {{\"item\", \"text\"}}", n + 1)),
        }
    }
    Ok(out)
}

impl HypothesesDriver {
    pub fn open(def: &EngineDef, paths: &Paths) -> Result<HypothesesDriver, String> {
        let path = def
            .options
            .get("path")
            .and_then(Value::as_str)
            .ok_or("the hypotheses backend needs options.path")?;
        let path = paths.resolve(path)?;
        let text = std::fs::read_to_string(&path).map_err(|e| format!("cannot read the hypotheses {}: {e}", file_name(&path)))?;
        Ok(HypothesesDriver::from_map(parse_hypotheses(&text)?))
    }

    pub fn from_map(hyps: HashMap<String, String>) -> HypothesesDriver {
        HypothesesDriver {
            hyps,
            info: DriverInfo {
                device: "none".into(),
                ..DriverInfo::default()
            },
        }
    }
}

impl SttDriver for HypothesesDriver {
    fn info(&self) -> &DriverInfo {
        &self.info
    }

    fn transcribe(&mut self, req: &SttRequest<'_>) -> Result<SttOutput, String> {
        self.hyps
            .get(req.item)
            .map(|t| SttOutput {
                text: t.clone(),
                ..SttOutput::default()
            })
            .ok_or_else(|| format!("no hypothesis for {}", req.item))
    }
}

pub fn file_name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

// ---------------------------------------------------------------------------------------------
// CrispASR as a helper process
// ---------------------------------------------------------------------------------------------

/// Stops a `process_host` helper when dropped.
pub struct HelperGuard {
    name: String,
}

impl HelperGuard {
    pub fn new(name: impl Into<String>) -> HelperGuard {
        HelperGuard { name: name.into() }
    }
}

impl Drop for HelperGuard {
    fn drop(&mut self) {
        let name = self.name.clone();
        if let Err(e) = tauri::async_runtime::block_on(crate::services::process_host::stop(&name)) {
            tracing::warn!(target: "sigil::bench", "stopping the helper {name}: {e}");
        }
    }
}

/// `process_host` name of the CrispASR STT helper.
pub const CRISPASR_HELPER: &str = "bench-crispasr";
const CRISPASR_PROGRAM: &str = "crispasr.exe";

/// CrispASR backends the bench maps `options.crispasrBackend` to (`--backend`).
fn crispasr_args(def: &EngineDef, model: &str) -> Result<Vec<String>, String> {
    let backend = def.options.get("crispasrBackend").and_then(Value::as_str).unwrap_or("");
    if backend.is_empty() || !backend.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err("the crispasr backend needs options.crispasrBackend (CrispASR's --backend, e.g. \"qwen3\")".into());
    }
    let mut args = vec![
        "--server".to_string(),
        "--backend".into(),
        backend.to_string(),
        "-m".into(),
        model.to_string(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        crate::services::tts_helpers::ggml::launch::port_arg(),
    ];
    if let Some(threads) = def.threads {
        args.push("-t".into());
        args.push(threads.to_string());
    }
    Ok(args)
}

/// The GGUF file of catalog model `id` for CrispASR's `-m`: the entry's file `file` (`options.modelFile`,
/// as the catalog lists it), else its first file. `-m` takes a file; the install folder fails to load.
fn crispasr_model_file(env: &crate::services::tts_helpers::ggml::launch::PlanEnv<'_>, id: &str, file: Option<&str>) -> Result<String, String> {
    env.file_of(id, file.filter(|f| !f.is_empty())).map(|p| p.to_string_lossy().into_owned())
}

/// Start the CrispASR server for `def` and return the HTTP engine that talks to it. The pack is the
/// CUDA 13 / Vulkan / CPU build that matches the engine's execution provider.
pub fn create_crispasr_driver(def: &EngineDef, paths: &Paths) -> Result<Box<dyn SttDriver>, String> {
    use crate::services::tts_helpers::ggml::{self, launch, Variant};
    // A bring-your-own model is a file; a catalog id is resolved to its file below, in the plan
    // environment (`-m` takes the GGUF, not the install folder).
    let byo_model = if is_catalog_id(&def.model) {
        None
    } else {
        let p = paths.resolve(&def.model)?;
        if !p.is_file() {
            return Err(format!("the CrispASR model {} does not exist", file_name(&p)));
        }
        Some(p.to_string_lossy().into_owned())
    };
    let packs = [
        launch::PackChoice {
            variant: Variant::Cuda13,
            runtime: "runtime/crispasr-cuda13",
        },
        launch::PackChoice {
            variant: Variant::Vulkan,
            runtime: "runtime/crispasr-vulkan",
        },
        launch::PackChoice {
            variant: Variant::Cpu,
            runtime: "runtime/crispasr-cpu",
        },
    ];
    let wanted = match def.variant.as_deref().filter(|v| !v.is_empty()) {
        Some(v) => Variant::parse(v)?,
        None => match def.execution_provider {
            ExecutionProvider::Cuda => Some(Variant::Cuda13),
            ExecutionProvider::Vulkan => Some(Variant::Vulkan),
            ExecutionProvider::Cpu => Some(Variant::Cpu),
            ExecutionProvider::Dml | ExecutionProvider::Auto => None,
        },
    };
    let (program, variant, dll_dirs, model) = ggml::with_plan_env(|env| {
        let pack = launch::pick_pack("CrispASR", &packs, wanted, env)?;
        let mut needed: Vec<&str> = Vec::new();
        if byo_model.is_none() {
            needed.push(&def.model);
        }
        needed.extend(launch::pack_ids(pack));
        env.require("CrispASR", &needed)?;
        let model = match &byo_model {
            Some(file) => file.clone(),
            None => crispasr_model_file(env, &def.model, def.options.get("modelFile").and_then(Value::as_str))?,
        };
        let program = launch::pack_program(env, pack.runtime, CRISPASR_PROGRAM)?;
        let dirs = if pack.variant == Variant::Cuda13 { (env.nvidia_dirs)() } else { Vec::new() };
        Ok((program, pack.variant, dirs, model))
    })
    .map_err(normalize_missing)?;
    let args = crispasr_args(def, &model)?;
    let mut env = vec![("CRISPASR_API_KEYS".to_string(), launch::key_env())];
    if !dll_dirs.is_empty() {
        let mut path = dll_dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(";");
        if let Some(inherited) = std::env::var_os("PATH") {
            path.push(';');
            path.push_str(&inherited.to_string_lossy());
        }
        env.push(("PATH".into(), path));
    }
    let spec = crate::services::process_host::HelperSpec {
        name: CRISPASR_HELPER.to_string(),
        cwd: program.parent().map(Path::to_path_buf),
        program,
        args,
        env,
        env_remove: vec!["CRISPASR_*".to_string()],
        port: crate::services::process_host::PortRequest::Auto,
        api_key: true,
        health: Some(crate::services::process_host::HealthCheck {
            path: "/health".into(),
            timeout_ms: 300_000,
            interval_ms: 250,
            send_api_key: true,
        }),
        priority: if variant == Variant::Cpu {
            crate::services::process_host::HelperPriority::BelowNormal
        } else {
            crate::services::process_host::HelperPriority::Normal
        },
        high_qos: true,
        ..Default::default()
    };
    let handle = tauri::async_runtime::block_on(crate::services::process_host::spawn(spec))?;
    let guard = HelperGuard::new(CRISPASR_HELPER);
    let (base, key) = match (handle.base_url.clone(), handle.api_key.clone()) {
        (Some(b), Some(k)) => (b, k),
        _ => return Err("CrispASR started without a port or key".into()),
    };
    let language = def.options.get("language").and_then(Value::as_str).unwrap_or("");
    let mut options = json!({ "baseUrl": format!("{base}/v1"), "model": "crispasr", "apiKey": key });
    if !language.is_empty() {
        options["language"] = json!(language);
    }
    let engine = crate::services::speech::engines::create(
        &EngineSpec {
            engine: EngineKind::Http,
            model: String::new(),
            threads: None,
            variant: None,
            options,
        },
        &EngineContext::headless(),
    )
    .map_err(|e| asr_message(&e))?;
    let mut driver = EngineDriver::new(engine, def.execution_provider, def.threads, Some(guard));
    // The HTTP engine reports `remote`; what matters is the pack the helper runs.
    driver.info.device = match variant {
        Variant::Cuda13 | Variant::Cuda => "cuda".into(),
        Variant::Vulkan => "vulkan".into(),
        Variant::Cpu => "cpu".into(),
    };
    driver.info.variant = Some(variant.as_str().to_string());
    driver.info.runtime = Some("CrispASR (helper process)".into());
    driver.info.ep_reason = (def.execution_provider.wants_gpu() && driver.info.device != def.execution_provider.as_str())
        .then(|| format!("{} was requested; the {} build of CrispASR runs", def.execution_provider.as_str(), driver.info.device));
    Ok(Box::new(driver))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::speech::engine_api::{AsrResult, EngineCaps, Hypothesis};
    use std::sync::{Arc, Mutex};

    /// A scripted engine: counts what it was fed and answers after a fixed delay.
    struct Fake {
        info: EngineInfo,
        delay: Duration,
        stream: Vec<String>,
        fed: Arc<Mutex<usize>>,
        calls: Arc<Mutex<Vec<String>>>,
        block_until_cancelled: bool,
        cancelled: Arc<AtomicBool>,
    }

    impl Fake {
        fn new(streaming: bool, utterance: bool) -> Fake {
            Fake {
                info: EngineInfo {
                    kind: EngineKind::Whisper,
                    model: "fake".into(),
                    runtime: "fake 1.0 cpu".into(),
                    device: "cpu".into(),
                    caps: EngineCaps {
                        streaming,
                        utterance,
                        ..EngineCaps::default()
                    },
                    vram_mb: 0,
                    fallback_reason: None,
                },
                delay: Duration::ZERO,
                stream: vec!["he".into(), "hello".into(), "hello world".into()],
                fed: Arc::new(Mutex::new(0)),
                calls: Arc::new(Mutex::new(Vec::new())),
                block_until_cancelled: false,
                cancelled: Arc::new(AtomicBool::new(false)),
            }
        }
    }

    impl AsrEngine for Fake {
        fn info(&self) -> &EngineInfo {
            &self.info
        }
        fn transcribe(&mut self, audio: &[f32], options: &DecodeOptions) -> Result<AsrResult, AsrError> {
            self.calls.lock().unwrap().push(format!("transcribe {} {:?}", audio.len(), options.language));
            if self.block_until_cancelled {
                let start = Instant::now();
                while !self.cancelled.load(Ordering::SeqCst) && start.elapsed() < Duration::from_secs(5) {
                    std::thread::sleep(Duration::from_millis(2));
                }
                return Err(AsrError::cancelled());
            }
            std::thread::sleep(self.delay);
            Ok(AsrResult {
                text: "hello world".into(),
                lang: options.language.clone(),
                ..AsrResult::default()
            })
        }
        fn begin_stream(&mut self, options: &DecodeOptions) -> Result<(), AsrError> {
            self.calls.lock().unwrap().push(format!("begin {:?}", options.language));
            *self.fed.lock().unwrap() = 0;
            Ok(())
        }
        fn push_stream(&mut self, audio: &[f32]) -> Result<Option<Hypothesis>, AsrError> {
            let mut fed = self.fed.lock().unwrap();
            *fed += audio.len();
            let n = *fed / (SAMPLES_PER_MS * 100);
            Ok(self.stream.get(n.saturating_sub(2)).filter(|_| n >= 2).map(|t| Hypothesis {
                text: t.clone(),
                lang: None,
            }))
        }
        fn end_stream(&mut self) -> Result<AsrResult, AsrError> {
            if self.block_until_cancelled {
                let start = Instant::now();
                while !self.cancelled.load(Ordering::SeqCst) && start.elapsed() < Duration::from_secs(5) {
                    std::thread::sleep(Duration::from_millis(2));
                }
                return Err(AsrError::cancelled());
            }
            std::thread::sleep(self.delay);
            self.calls.lock().unwrap().push(format!("end {}", *self.fed.lock().unwrap()));
            Ok(AsrResult {
                text: "hello world".into(),
                ..AsrResult::default()
            })
        }
        fn canceller(&self) -> Option<Canceller> {
            Some(Canceller::from_flag(self.cancelled.clone()))
        }
    }

    fn feed(kind: super::super::spec::FeedKind, pace: Pace) -> Feed {
        Feed {
            kind,
            pace,
            chunk_ms: 100,
            lead_silence_ms: 0,
            tail_silence_ms: 200,
        }
    }

    fn request<'a>(audio: &'a [f32], feed: &'a Feed, timeout: Duration) -> SttRequest<'a> {
        SttRequest {
            item: "en/1/clean",
            audio,
            language: Some("en".into()),
            feed,
            timeout,
        }
    }

    #[test]
    fn an_utterance_feed_times_the_decode_and_passes_the_language() {
        let mut fake = Fake::new(false, true);
        fake.delay = Duration::from_millis(20);
        let calls = fake.calls.clone();
        let mut d = EngineDriver::new(Box::new(fake), ExecutionProvider::Cpu, Some(2), None);
        let audio = vec![0.0; 16_000];
        let f = feed(super::super::spec::FeedKind::Utterance, Pace::Asap);
        let out = d.transcribe(&request(&audio, &f, Duration::from_secs(5))).unwrap();
        assert_eq!(out.text, "hello world");
        assert_eq!(out.language.as_deref(), Some("en"));
        let ms = out.compute_ms.unwrap();
        assert!((20.0..2_000.0).contains(&ms), "{ms}");
        assert_eq!(out.final_latency_ms, Some(ms), "an utterance engine's final arrives when its decode ends");
        assert_eq!(calls.lock().unwrap().as_slice(), ["transcribe 16000 Some(\"en\")"]);
        assert_eq!(d.info().device, "cpu");
        assert_eq!(d.info().variant.as_deref(), Some("cpu"));
        assert!(d.info().ep_reason.is_none());
    }

    #[test]
    fn a_stream_feed_pushes_chunks_with_the_silences_and_counts_partials() {
        let fake = Fake::new(true, false);
        let calls = fake.calls.clone();
        let mut d = EngineDriver::new(Box::new(fake), ExecutionProvider::Cpu, None, None);
        let audio = vec![0.0; 16_000]; // 1 s
        let f = feed(super::super::spec::FeedKind::Stream, Pace::Asap);
        let out = d.transcribe(&request(&audio, &f, Duration::from_secs(5))).unwrap();
        assert_eq!(out.text, "hello world");
        assert!(out.partials.unwrap() >= 2);
        assert!(out.first_partial_ms.is_some() && out.final_latency_ms.is_some());
        // 1 s clip + 200 ms tail.
        assert_eq!(calls.lock().unwrap().last().unwrap(), "end 19200");
        // A streaming-only engine with the utterance feed is fed as a stream.
        let fake = Fake::new(true, false);
        let calls = fake.calls.clone();
        let mut d = EngineDriver::new(Box::new(fake), ExecutionProvider::Cpu, None, None);
        let f = feed(super::super::spec::FeedKind::Utterance, Pace::Asap);
        d.transcribe(&request(&audio, &f, Duration::from_secs(5))).unwrap();
        assert!(calls.lock().unwrap()[0].starts_with("begin"));
    }

    #[test]
    fn a_realtime_stream_feed_waits_for_the_audio() {
        let fake = Fake::new(true, true);
        let mut d = EngineDriver::new(Box::new(fake), ExecutionProvider::Cpu, None, None);
        let audio = vec![0.0; 8_000]; // 0.5 s + 0.2 s tail
        let f = feed(super::super::spec::FeedKind::Stream, Pace::Realtime);
        let t0 = Instant::now();
        let out = d.transcribe(&request(&audio, &f, Duration::from_secs(5))).unwrap();
        assert!(t0.elapsed() >= Duration::from_millis(650), "{:?}", t0.elapsed());
        // The engine kept up: the final lag after the last sample is small.
        assert!(out.final_latency_ms.unwrap() < 500.0, "{:?}", out.final_latency_ms);
        assert!(out.compute_ms.unwrap() < 650.0, "compute is the engine's busy time, not the paced wall time");
    }

    #[test]
    fn a_realtime_stream_feed_is_not_timed_out_by_the_time_the_audio_takes() {
        // 0.5 s of audio + 0.2 s of tail take 0.7 s to speak; the timeout is only 0.3 s. The engine
        // keeps up, so the item passes: the deadline covers the engine's time, not the audio's.
        let fake = Fake::new(true, true);
        let mut d = EngineDriver::new(Box::new(fake), ExecutionProvider::Cpu, None, None);
        let audio = vec![0.0; 8_000];
        let f = feed(super::super::spec::FeedKind::Stream, Pace::Realtime);
        let t0 = Instant::now();
        let out = d.transcribe(&request(&audio, &f, Duration::from_millis(300)));
        assert!(t0.elapsed() >= Duration::from_millis(650), "{:?}", t0.elapsed());
        assert_eq!(out.expect("an engine that keeps up is not a timeout").text, "hello world");
    }

    #[test]
    fn a_hung_realtime_stream_still_times_out_after_the_audio_plus_the_timeout() {
        let mut fake = Fake::new(true, true);
        fake.block_until_cancelled = true;
        let mut d = EngineDriver::new(Box::new(fake), ExecutionProvider::Cpu, None, None);
        let audio = vec![0.0; 4_800]; // 0.3 s + 0.2 s tail
        let f = feed(super::super::spec::FeedKind::Stream, Pace::Realtime);
        let t0 = Instant::now();
        let err = d.transcribe(&request(&audio, &f, Duration::from_millis(200))).unwrap_err();
        assert_eq!(err, "timeout after 200 ms");
        let took = t0.elapsed();
        assert!(took >= Duration::from_millis(650) && took < Duration::from_secs(3), "0.5 s of audio, then the 0.2 s timeout: {took:?}");
    }

    #[test]
    fn an_engine_that_cannot_stream_decodes_once_and_says_so() {
        let fake = Fake::new(false, true);
        let calls = fake.calls.clone();
        let mut d = EngineDriver::new(Box::new(fake), ExecutionProvider::Cpu, None, None);
        let audio = vec![0.0; 3_200];
        let f = feed(super::super::spec::FeedKind::Stream, Pace::Asap);
        let out = d.transcribe(&request(&audio, &f, Duration::from_secs(5))).unwrap();
        assert!(out.first_partial_ms.is_none() && out.partials.is_none());
        assert_eq!(calls.lock().unwrap().len(), 1);
        d.transcribe(&request(&audio, &f, Duration::from_secs(5))).unwrap();
        assert_eq!(d.notes().len(), 1, "said once");
        assert!(d.notes()[0].contains("does not stream"));
    }

    #[test]
    fn an_item_over_its_timeout_is_cancelled_and_reported() {
        let mut fake = Fake::new(false, true);
        fake.block_until_cancelled = true;
        let mut d = EngineDriver::new(Box::new(fake), ExecutionProvider::Cpu, None, None);
        let audio = vec![0.0; 1_600];
        let f = feed(super::super::spec::FeedKind::Utterance, Pace::Asap);
        let t0 = Instant::now();
        let err = d.transcribe(&request(&audio, &f, Duration::from_millis(50))).unwrap_err();
        assert_eq!(err, "timeout after 50 ms");
        assert!(t0.elapsed() < Duration::from_secs(3));
        assert_eq!(timeout_message(Duration::from_secs(120)), "timeout after 120 s");
    }

    #[test]
    fn a_lesser_device_than_requested_is_recorded_with_a_reason() {
        let mut info = Fake::new(false, true).info;
        info.device = "directml".into();
        let d = driver_info(&info, ExecutionProvider::Cuda, None, false);
        assert_eq!(d.device, "dml");
        assert!(d.ep_reason.as_deref().unwrap().contains("cuda was requested"));
        info.fallback_reason = Some("the GPU is busy".into());
        assert_eq!(driver_info(&info, ExecutionProvider::Cuda, None, false).ep_reason.as_deref(), Some("the GPU is busy"));
        info.device = "remote".into();
        info.fallback_reason = None;
        assert!(driver_info(&info, ExecutionProvider::Cuda, None, false).ep_reason.is_none(), "a remote server has no local device to fall back from");
        info.device = "cpu".into();
        info.runtime = "transcribe.cpp 0.2.4 cuda13".into();
        assert_eq!(driver_info(&info, ExecutionProvider::Cpu, None, false).variant.as_deref(), Some("cuda13"));
    }

    #[test]
    fn engine_specs_follow_the_execution_provider_and_keys_come_from_the_environment() {
        let paths = Paths::with_base(std::env::temp_dir());
        let def = |json: Value| -> EngineDef {
            let mut base = json!({"id": "e", "task": "stt", "backend": "transcribe", "model": "stt/qwen3-asr-1.7b-q8_0"});
            base.as_object_mut().unwrap().extend(json.as_object().unwrap().clone());
            serde_json::from_value(base).unwrap()
        };
        let s = engine_spec(&def(json!({"execution_provider": "cuda"})), &paths).unwrap();
        assert_eq!((s.engine, s.variant.as_deref(), s.model.as_str()), (EngineKind::Transcribe, Some("cuda13"), "stt/qwen3-asr-1.7b-q8_0"));
        assert_eq!(engine_spec(&def(json!({"execution_provider": "cpu"})), &paths).unwrap().variant.as_deref(), Some("cpu-any"));
        assert_eq!(engine_spec(&def(json!({"execution_provider": "auto"})), &paths).unwrap().variant, None);
        assert_eq!(engine_spec(&def(json!({"variant": "vulkan"})), &paths).unwrap().variant.as_deref(), Some("vulkan"));
        let sh = engine_spec(&def(json!({"backend": "sherpa-offline", "execution_provider": "cuda", "threads": 2})), &paths).unwrap();
        assert_eq!((sh.engine, sh.variant, sh.threads), (EngineKind::SherpaOffline, None, Some(2)));
        let bring = engine_spec(&def(json!({"model": "${SIGIL_BENCH_TEST_DIR}/m.gguf"})), &Paths::with_base("."));
        assert!(bring.unwrap_err().contains("SIGIL_BENCH_TEST_DIR"), "unknown variables are errors");
        assert!(engine_spec(&def(json!({"backend": "pipeline"})), &paths).is_err());
        // The HTTP engine's key comes from the named variable and never from the spec.
        std::env::set_var("SIGIL_SPEECH_BENCH_TEST_KEY", "k-123");
        let http = engine_spec(
            &def(json!({"backend": "http", "model": "", "options": {"baseUrl": "http://localhost:1/v1", "model": "m", "apiKeyEnv": "SIGIL_SPEECH_BENCH_TEST_KEY"}})),
            &paths,
        )
        .unwrap();
        assert_eq!(http.options["apiKey"], "k-123");
        assert!(http.options.get("apiKeyEnv").is_none());
        let missing = with_api_key(&json!({"apiKeyEnv": "SIGIL_SPEECH_BENCH_NO_SUCH_VAR"})).unwrap_err();
        assert!(missing.contains("SIGIL_SPEECH_BENCH_NO_SUCH_VAR") && !missing.contains("k-123"));
    }

    #[test]
    fn a_catalog_model_reaches_crispasr_as_its_file_and_not_the_install_folder() {
        use crate::services::model_store::Catalog;
        use crate::services::tts_helpers::ggml::launch::PlanEnv;
        use crate::services::tts_helpers::ggml::GpuFacts;
        let catalog = Catalog::embedded().expect("embedded catalog");
        let id = "stt/granite-speech-5.0-turboctc-q8_0";
        let entry = catalog.get(id).expect("the entry is in the catalog");
        let dir = std::env::temp_dir().join(format!("sigil-crispasr-model-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for f in &entry.files {
            std::fs::write(dir.join(&f.path), b"fake gguf").unwrap();
        }
        let lookup = |i: &str| (i == id).then(|| dir.clone());
        let no_dirs = || Vec::<std::path::PathBuf>::new();
        let env = PlanEnv {
            catalog: &catalog,
            installed: &lookup,
            gpu: GpuFacts::default(),
            nvidia_dirs: &no_dirs,
            app_data: None,
        };
        let file = crispasr_model_file(&env, id, None).expect("the entry's first file");
        assert!(Path::new(&file).is_file(), "-m gets a file, not {file}");
        assert!(file.ends_with(&entry.files[0].path), "{file}");
        // The `-m` argument of the server command is exactly that file.
        let def: EngineDef = serde_json::from_value(json!({"id": "r2t2", "task": "stt", "backend": "crispasr", "model": id, "options": {"crispasrBackend": "qwen3"}})).unwrap();
        let args = crispasr_args(&def, &file).unwrap();
        let at = args.iter().position(|a| a == "-m").expect("-m");
        assert_eq!(args[at + 1], file);
        // `options.modelFile` picks another file of the entry; an unknown one is an error that names it.
        assert_eq!(crispasr_model_file(&env, id, Some(entry.files[0].path.as_str())).unwrap(), file);
        assert!(crispasr_model_file(&env, id, Some("nope.gguf")).unwrap_err().contains("nope.gguf"));
        // An empty `modelFile` means the default; a model that is not installed says so.
        assert_eq!(crispasr_model_file(&env, id, Some("")).unwrap(), file);
        assert!(crispasr_model_file(&env, "stt/qwen3-asr-1.7b-q8_0", None).unwrap_err().contains("not installed"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_downloads_are_reported_as_not_installed() {
        assert_eq!(
            normalize_missing("CrispASR needs downloads first: runtime/crispasr-cuda13, runtime/nvidia-cuda13".into()),
            "not installed: runtime/crispasr-cuda13, runtime/nvidia-cuda13 (Sigil.exe --headless fetch runtime/crispasr-cuda13 runtime/nvidia-cuda13)"
        );
        assert_eq!(normalize_missing("some other problem".into()), "some other problem");
    }

    #[test]
    fn a_hypotheses_file_the_runner_wrote_can_be_read_back_even_with_failed_items() {
        // The runner's own hyps lines: a failed item has `"text": null` beside its error.
        let lines = [
            r#"{"item":"en/1/clean","language":"en","condition":"clean","audio_s":3.2,"text":"hello world"}"#,
            r#"{"item":"en/2/clean","language":"en","condition":"clean","audio_s":2.0,"text":null,"error":"timeout after 120 s"}"#,
            r#"{"id":"s:en-ja:3","hyp":null}"#,
            r#"{"id":"s:en-ja:4","hyp":"fine"}"#,
        ]
        .join("\n");
        let h = parse_hypotheses(&lines).expect("a null text is a missing hypothesis, not a broken file");
        assert_eq!(h.len(), 2);
        assert_eq!((h["en/1/clean"].as_str(), h["s:en-ja:4"].as_str()), ("hello world", "fine"));
        let mut d = HypothesesDriver::from_map(h);
        let f = Feed::default();
        let req = |item| SttRequest {
            item,
            audio: &[],
            language: None,
            feed: &f,
            timeout: Duration::from_secs(1),
        };
        assert!(d.transcribe(&req("en/2/clean")).unwrap_err().contains("no hypothesis for en/2/clean"));
        assert_eq!(d.transcribe(&req("en/1/clean")).unwrap().text, "hello world");
        // Still errors: no text member at all, a text that is not a string, no item.
        assert!(parse_hypotheses(r#"{"item":"a"}"#).is_err());
        assert!(parse_hypotheses(r#"{"item":"a","text":5}"#).is_err());
        assert!(parse_hypotheses(r#"{"text":"x"}"#).is_err());
        assert!(parse_hypotheses(r#"{"text":null}"#).is_err());
    }

    #[test]
    fn hypotheses_files_parse_both_spellings() {
        let h = parse_hypotheses("# c\n{\"item\":\"a\",\"text\":\"one\"}\n{\"id\":\"b\",\"hyp\":\"two\"}\n").unwrap();
        assert_eq!(h["a"], "one");
        assert_eq!(h["b"], "two");
        assert!(parse_hypotheses("{\"item\":\"a\"}").is_err());
        let mut d = HypothesesDriver::from_map(h);
        let f = Feed::default();
        let req = |item| SttRequest {
            item,
            audio: &[],
            language: None,
            feed: &f,
            timeout: Duration::from_secs(1),
        };
        assert_eq!(d.transcribe(&req("a")).unwrap().text, "one");
        assert!(d.transcribe(&req("zz")).unwrap_err().contains("no hypothesis"));
        assert!(d.transcribe(&req("a")).unwrap().compute_ms.is_none(), "no timing for precomputed text");
    }

    #[test]
    fn model_refs_keep_catalog_ids_and_resolve_paths() {
        let paths = Paths::with_base("C:/spec");
        assert_eq!(model_ref("stt/foo", &paths).unwrap(), "stt/foo");
        assert_eq!(model_ref("base.en", &paths).unwrap(), "base.en");
        assert_eq!(model_ref("", &paths).unwrap(), "");
        // A bring-your-own model is an absolute path, or starts with ${NAME} or ./ (anything of the
        // form family/name is a catalog id).
        let p = model_ref("./models/x.gguf", &paths).unwrap();
        assert!(p.replace('\\', "/").ends_with("C:/spec/./models/x.gguf"), "{p}");
        assert_eq!(model_ref("models/x.gguf", &paths).unwrap(), "models/x.gguf");
    }
}
