//! MT engines of the speech bench, through the app's own backends:
//! - `nllb`: `mt_local::nllb::NllbModel` in this process (the catalog model of the model store, or a
//!   folder);
//! - `bergamot`: `mt_local::bergamot` (the `sigil-bergamot.exe` helper under `process_host`);
//! - `llama-server`: the bundled llama.cpp server, started with `llm_server::server::start` (the same
//!   code path and VRAM governor as the translation service), then spoken to through its
//!   OpenAI-compatible endpoint;
//! - `openai`: any OpenAI-compatible `/chat/completions` server (`options.baseUrl`; the key is read
//!   from the environment variable `options.apiKeyEnv`, never from the spec);
//! - `hypotheses`: JSON lines `{id, hyp}` (mt-bench's format) made elsewhere.
//!
//! The prompts and the output cleaning are mt-bench's (`mt_local::bench::build_prompt` /
//! `clean_llm_output`), so a bench number is the number of the translation service. A model or pack
//! that is not installed fails the engine's rows with `not installed: <catalog id>`; nothing is
//! downloaded.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::corpus::MtLine;
use super::spec::{is_catalog_id, EngineDef, ExecutionProvider, MtContext, Paths};
use super::stt::{file_name, DriverInfo};
use crate::services::mt_local::bench::{build_prompt, clean_llm_output};
use crate::services::mt_local::{flores, nllb::NllbModel, MtErrorKind, MtKind, MtRequest, NLLB_DEFAULT_MODEL};

/// A translation and its extras.
#[derive(Clone, Debug, Default)]
pub struct MtOutput {
    pub text: String,
    pub out_tokens: Option<u64>,
}

/// An MT engine ready to translate lines.
pub trait MtDriver {
    fn info(&self) -> &DriverInfo;

    /// Translate one line; `context` is the previous source line of the same document when the run
    /// asked for context (LLM backends use it; the others ignore it).
    fn translate(&mut self, line: &MtLine, context: Option<&MtLine>, timeout: Duration) -> Result<MtOutput, String>;

    /// Whether the driver measures real latency (precomputed hypotheses do not).
    fn timed(&self) -> bool {
        true
    }

    fn notes(&self) -> Vec<String> {
        Vec::new()
    }
}

fn not_installed(id: &str) -> String {
    format!("not installed: {id} (Sigil.exe --headless fetch {id}, with SIGIL_APP_DATA_DIR if you use one)")
}

// ---------------------------------------------------------------------------------------------
// NLLB
// ---------------------------------------------------------------------------------------------

pub struct NllbDriver {
    model: NllbModel,
    beam: usize,
    info: DriverInfo,
}

pub fn create_nllb(def: &EngineDef, paths: &Paths) -> Result<NllbDriver, String> {
    let model = def.model.trim();
    let (dir, id) = if model.is_empty() {
        let dir = crate::services::model_store::installed_path(NLLB_DEFAULT_MODEL).ok_or_else(|| not_installed(NLLB_DEFAULT_MODEL))?;
        (dir, NLLB_DEFAULT_MODEL.to_string())
    } else if is_catalog_id(model) {
        let dir = crate::services::model_store::installed_path(model).ok_or_else(|| not_installed(model))?;
        (dir, model.to_string())
    } else {
        let dir = paths.resolve(model)?;
        let id = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "nllb".into());
        (dir, id)
    };
    let threads = crate::services::mt_local::thread_count(def.threads);
    let beam = def
        .options
        .get("beam")
        .and_then(Value::as_u64)
        .map_or(1, |b| b as usize)
        .clamp(1, crate::services::mt_local::nllb::MAX_BEAM);
    let model = NllbModel::load(&dir, &id, threads)?;
    let mut info = DriverInfo {
        device: "cpu".into(),
        threads: Some(i64::from(threads)),
        runtime: Some("ONNX Runtime (NLLB-200)".into()),
        ..DriverInfo::default()
    };
    if def.execution_provider.wants_gpu() {
        info.ep_reason = Some(format!("{} was requested; NLLB runs on the CPU", def.execution_provider.as_str()));
    }
    Ok(NllbDriver { model, beam, info })
}

impl MtDriver for NllbDriver {
    fn info(&self) -> &DriverInfo {
        &self.info
    }

    fn translate(&mut self, line: &MtLine, _context: Option<&MtLine>, _timeout: Duration) -> Result<MtOutput, String> {
        let tgt = flores::to_flores(&line.target).ok_or_else(|| format!("NLLB has no language {}", line.target))?;
        let src = flores::to_flores(&line.source);
        self.model
            .translate(&line.src, src, tgt, self.beam, &|| false)
            .map(|text| MtOutput { text, out_tokens: None })
            .map_err(|e| e.message)
    }
}

// ---------------------------------------------------------------------------------------------
// Bergamot
// ---------------------------------------------------------------------------------------------

pub struct BergamotDriver {
    threads: u32,
    info: DriverInfo,
}

pub fn create_bergamot(def: &EngineDef) -> Result<BergamotDriver, String> {
    let threads = crate::services::mt_local::thread_count(def.threads);
    let status = tauri::async_runtime::block_on(crate::services::mt_local::bergamot::engine().start(Some(threads))).map_err(|e| match e.kind {
        MtErrorKind::NotInstalled => format!("not installed: {} (Sigil.exe --headless fetch {})", e.missing.join(", "), e.missing.join(" ")),
        _ => e.message,
    })?;
    let mut info = DriverInfo {
        device: status.device.unwrap_or_else(|| "cpu".into()),
        threads: Some(i64::from(threads)),
        runtime: Some("Bergamot (sigil-bergamot helper)".into()),
        helper_process: true,
        ..DriverInfo::default()
    };
    if def.execution_provider.wants_gpu() {
        info.ep_reason = Some(format!("{} was requested; Bergamot runs on the CPU", def.execution_provider.as_str()));
    }
    Ok(BergamotDriver { threads, info })
}

impl MtDriver for BergamotDriver {
    fn info(&self) -> &DriverInfo {
        &self.info
    }

    fn translate(&mut self, line: &MtLine, _context: Option<&MtLine>, _timeout: Duration) -> Result<MtOutput, String> {
        let request = MtRequest {
            text: line.src.clone(),
            source: line.source.clone(),
            target: line.target.clone(),
            kind: MtKind::Final,
            threads: Some(self.threads),
            ..MtRequest::default()
        };
        tauri::async_runtime::block_on(crate::services::mt_local::bergamot::translate_line(request))
            .map(|r| MtOutput { text: r.text, out_tokens: None })
            .map_err(|e| match e.kind {
                MtErrorKind::NotInstalled => format!("not installed: {}", e.missing.join(", ")),
                _ => e.message,
            })
    }
}

impl Drop for BergamotDriver {
    fn drop(&mut self) {
        tauri::async_runtime::block_on(crate::services::mt_local::bergamot::engine().stop());
    }
}

// ---------------------------------------------------------------------------------------------
// OpenAI-compatible chat servers (also the llama-server helper)
// ---------------------------------------------------------------------------------------------

/// A chat-completions translator.
pub struct ChatDriver {
    url: String,
    model: String,
    key: Option<String>,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
    timeout: Duration,
    local: bool,
    info: DriverInfo,
    /// Stops the llama-server helper on drop.
    server: bool,
}

/// How a chat request is worded and sampled.
#[derive(Clone, Debug)]
pub struct ChatSettings {
    pub model: String,
    /// `generic`, `hy-mt2` or a template (mt-bench's `prompt`).
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

/// The request body of one translation.
pub fn chat_body(settings: &ChatSettings, source: &str, target: &str, text: &str, context: Option<&MtLine>) -> Value {
    let mut user = build_prompt(&settings.prompt, source, target, text);
    if let Some(prev) = context {
        // The previous line of the same document, as context only.
        user = format!(
            "Context (the previous line, do not translate it): {}

{user}",
            prev.src.trim()
        );
    }
    json!({
        "model": settings.model,
        "messages": [{ "role": "user", "content": user }],
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "stream": false,
    })
}

impl ChatDriver {
    fn from_options(def: &EngineDef, url: String, model: String, key: Option<String>, info: DriverInfo, server: bool) -> Result<ChatDriver, String> {
        let o = &def.options;
        let max_tokens = o.get("maxTokens").and_then(Value::as_u64).map_or(512, |v| v.clamp(1, 8192) as u32);
        let temperature = o.get("temperature").and_then(Value::as_f64).map_or(0.2, |v| v.clamp(0.0, 2.0) as f32);
        let timeout = Duration::from_millis(o.get("timeoutMs").and_then(Value::as_u64).unwrap_or(120_000).clamp(1_000, 600_000));
        let local = reqwest::Url::parse(&url).map(|u| crate::services::http_client::is_local_endpoint(&u)).unwrap_or(false);
        Ok(ChatDriver {
            url,
            model,
            key,
            prompt: o.get("prompt").and_then(Value::as_str).unwrap_or("generic").to_string(),
            max_tokens,
            temperature,
            timeout,
            local,
            info,
            server,
        })
    }
}

/// An OpenAI-compatible server named in the spec.
pub fn create_openai(def: &EngineDef) -> Result<ChatDriver, String> {
    let o = &def.options;
    let base = o.get("baseUrl").and_then(Value::as_str).ok_or("the openai backend needs options.baseUrl")?;
    let model = o.get("model").and_then(Value::as_str).unwrap_or(&def.model).to_string();
    if model.is_empty() {
        return Err("the openai backend needs a model name (options.model or the engine's model)".into());
    }
    let url = crate::services::http_client::api_url(base, "/chat/completions")?.to_string();
    let key = match o.get("apiKeyEnv").and_then(Value::as_str).filter(|v| !v.is_empty()) {
        Some(var) => Some(std::env::var(var).map_err(|_| format!("the environment variable {var} (options.apiKeyEnv) is not set"))?),
        None => None,
    };
    let info = DriverInfo {
        device: "remote".into(),
        runtime: Some("OpenAI-compatible server".into()),
        ..DriverInfo::default()
    };
    ChatDriver::from_options(def, url, model, key, info, false)
}

/// The variant of the llama-server pack an execution provider asks for.
pub fn llama_variant(def: &EngineDef) -> String {
    if let Some(v) = def.variant.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
        return v.to_string();
    }
    match def.execution_provider {
        ExecutionProvider::Cpu => "cpu",
        ExecutionProvider::Cuda => "cuda13",
        ExecutionProvider::Vulkan => "vulkan",
        ExecutionProvider::Dml | ExecutionProvider::Auto => "auto",
    }
    .to_string()
}

/// Start the bundled llama-server for `def` (blocking) and return the chat driver for it.
pub fn create_llama_server(def: &EngineDef, paths: &Paths) -> Result<ChatDriver, String> {
    use crate::services::llm_server::{server, LlmServerConfig, LlmServerErrorKind};
    let model = if def.model.is_empty() || is_catalog_id(&def.model) {
        def.model.clone()
    } else {
        paths.resolve(&def.model)?.to_string_lossy().into_owned()
    };
    if model.is_empty() {
        return Err("the llama-server backend needs a model (a catalog id like mt/hy-mt2-1.8b-q4_k_m, or a .gguf path)".into());
    }
    let o = &def.options;
    let config = LlmServerConfig {
        model: model.clone(),
        variant: llama_variant(def),
        ctx_size: o.get("ctxSize").and_then(Value::as_u64).map(|v| v as u32),
        gpu_layers: o.get("gpuLayers").and_then(Value::as_u64).map(|v| v as u32),
        flash_attn: o.get("flashAttn").and_then(Value::as_str).map(str::to_string),
        // A benchmark runs what the spec asks for; the voice-changer VRAM rule protects a live
        // session that the bench does not have (a co_load run states it explicitly).
        allow_over_budget: o.get("allowOverBudget").and_then(Value::as_bool).unwrap_or(true),
    };
    let started = Instant::now();
    let started_server = tauri::async_runtime::block_on(server::start(config)).map_err(|e| match e.kind {
        LlmServerErrorKind::NotInstalled => format!("not installed: {} (Sigil.exe --headless fetch {})", e.missing.join(", "), e.missing.join(" ")),
        _ => e.message,
    })?;
    let status = &started_server.status;
    let used = status.variant.clone().unwrap_or_default();
    let device = match used.as_str() {
        "cuda13" | "cuda" => "cuda",
        "vulkan" => "vulkan",
        _ => "cpu",
    };
    let mut info = DriverInfo {
        device: device.into(),
        variant: status.variant.clone(),
        threads: def.threads.map(i64::from),
        vram_mb: Some(status.vram_mb),
        runtime: Some(format!(
            "llama-server ({} GPU layers, ctx {})",
            status.gpu_layers.map_or("?".to_string(), |l| l.to_string()),
            status.ctx_size.map_or("?".to_string(), |c| c.to_string())
        )),
        helper_process: true,
        ..DriverInfo::default()
    };
    if def.execution_provider.wants_gpu() && device != def.execution_provider.as_str() {
        info.ep_reason = Some(format!("{} was requested; llama-server runs the {used} build", def.execution_provider.as_str()));
    }
    tracing::info!(target: "sigil::bench", "llama-server ready in {} ms", started.elapsed().as_millis());
    let url = format!("{}/chat/completions", started_server.endpoint.base_url.trim_end_matches('/'));
    ChatDriver::from_options(def, url, started_server.endpoint.model.clone(), Some(started_server.endpoint.api_key.clone()), info, true)
}

impl MtDriver for ChatDriver {
    fn info(&self) -> &DriverInfo {
        &self.info
    }

    fn translate(&mut self, line: &MtLine, context: Option<&MtLine>, timeout: Duration) -> Result<MtOutput, String> {
        let settings = ChatSettings {
            model: self.model.clone(),
            prompt: self.prompt.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
        };
        let body = chat_body(&settings, &line.source, &line.target, &line.src, context);
        let timeout = timeout.min(self.timeout.max(Duration::from_secs(1)));
        let client = if self.local {
            crate::services::http_client::local_http_client()
        } else {
            crate::services::http_client::http_client()
        };
        let mut req = client.post(&self.url).timeout(timeout).json(&body);
        if let Some(key) = &self.key {
            req = req.bearer_auth(key);
        }
        tauri::async_runtime::block_on(async move {
            let resp = req
                .send()
                .await
                .map_err(|e| crate::services::http_client::describe_request_error("LLM", &e, timeout))?;
            if !resp.status().is_success() {
                return Err(crate::services::http_client::describe_api_error("LLM", resp).await);
            }
            let v: Value = resp.json().await.map_err(|e| format!("LLM: unexpected response ({e})"))?;
            let text = v["choices"][0]["message"]["content"].as_str().ok_or("LLM: the response has no message content")?;
            Ok(MtOutput {
                text: clean_llm_output(text),
                out_tokens: v["usage"]["completion_tokens"].as_u64(),
            })
        })
    }
}

impl Drop for ChatDriver {
    fn drop(&mut self) {
        if self.server {
            if let Err(e) = tauri::async_runtime::block_on(crate::services::llm_server::server::stop()) {
                tracing::warn!(target: "sigil::bench", "stopping llama-server: {}", e.message);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Pre-computed hypotheses
// ---------------------------------------------------------------------------------------------

pub struct MtHypotheses {
    hyps: HashMap<String, String>,
    info: DriverInfo,
}

pub fn parse_mt_hypotheses(text: &str) -> Result<HashMap<String, String>, String> {
    super::stt::parse_hypotheses(text)
}

pub fn create_hypotheses(def: &EngineDef, paths: &Paths) -> Result<MtHypotheses, String> {
    let path = def.options.get("path").and_then(Value::as_str).ok_or("the hypotheses backend needs options.path")?;
    let path = paths.resolve(path)?;
    let text = std::fs::read_to_string(&path).map_err(|e| format!("cannot read the hypotheses {}: {e}", file_name(&path)))?;
    Ok(MtHypotheses::from_map(parse_mt_hypotheses(&text)?))
}

impl MtHypotheses {
    pub fn from_map(hyps: HashMap<String, String>) -> MtHypotheses {
        MtHypotheses {
            hyps,
            info: DriverInfo {
                device: "none".into(),
                ..DriverInfo::default()
            },
        }
    }
}

impl MtDriver for MtHypotheses {
    fn info(&self) -> &DriverInfo {
        &self.info
    }

    fn translate(&mut self, line: &MtLine, _context: Option<&MtLine>, _timeout: Duration) -> Result<MtOutput, String> {
        self.hyps
            .get(&line.id)
            .map(|t| MtOutput { text: t.clone(), out_tokens: None })
            .ok_or_else(|| format!("no hypothesis for {}", line.id))
    }

    fn timed(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------------------------

/// Create the MT driver of `def` (blocking: loads the model / starts the helper).
pub fn create_driver(def: &EngineDef, paths: &Paths) -> Result<Box<dyn MtDriver>, String> {
    match def.backend.as_str() {
        "nllb" => create_nllb(def, paths).map(|d| Box::new(d) as Box<dyn MtDriver>),
        "bergamot" => create_bergamot(def).map(|d| Box::new(d) as Box<dyn MtDriver>),
        "llama-server" => create_llama_server(def, paths).map(|d| Box::new(d) as Box<dyn MtDriver>),
        "openai" => create_openai(def).map(|d| Box::new(d) as Box<dyn MtDriver>),
        "hypotheses" => create_hypotheses(def, paths).map(|d| Box::new(d) as Box<dyn MtDriver>),
        other => Err(format!("{other} is not an MT backend")),
    }
}

/// Whether `context: previous` applies to the backend (LLM prompts take it, the others ignore it).
pub fn takes_context(def: &EngineDef, context: MtContext) -> bool {
    context == MtContext::Previous && matches!(def.backend.as_str(), "openai" | "llama-server")
}

/// The `Path` of an MT set, for messages.
pub fn set_name(path: &Path) -> String {
    file_name(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(id: &str, src: &str) -> MtLine {
        MtLine {
            id: id.into(),
            source: "en".into(),
            target: "ja".into(),
            src: src.into(),
            refs: vec!["x".into()],
            doc: Some("d#1".into()),
        }
    }

    #[test]
    fn chat_bodies_use_the_mt_bench_prompts_and_optional_context() {
        let settings = |prompt: &str| ChatSettings {
            model: "m".into(),
            prompt: prompt.into(),
            max_tokens: 256,
            temperature: 0.2,
        };
        let body = chat_body(&settings("hy-mt2"), "en", "ja", "Hello", None);
        assert_eq!(body["model"], "m");
        assert_eq!(body["stream"], false);
        assert_eq!(body["max_tokens"], 256);
        assert_eq!(
            body["messages"][0]["content"],
            "Translate the following segment into Japanese, without additional explanation.\n\nHello"
        );
        let prev = line("p", "Earlier line.");
        let with = chat_body(&settings("generic"), "en", "ja", "Hello", Some(&prev));
        let text = with["messages"][0]["content"].as_str().unwrap();
        assert!(text.starts_with("Context (the previous line, do not translate it): Earlier line."), "{text}");
        assert!(text.ends_with("Hello"));
    }

    #[test]
    fn llama_variants_follow_the_execution_provider() {
        let def = |ep: &str, variant: Option<&str>| -> EngineDef {
            serde_json::from_value(json!({"id": "e", "task": "mt", "backend": "llama-server", "execution_provider": ep, "variant": variant})).unwrap()
        };
        assert_eq!(llama_variant(&def("cpu", None)), "cpu");
        assert_eq!(llama_variant(&def("cuda", None)), "cuda13");
        assert_eq!(llama_variant(&def("vulkan", None)), "vulkan");
        assert_eq!(llama_variant(&def("auto", None)), "auto");
        assert_eq!(llama_variant(&def("cuda", Some("vulkan"))), "vulkan");
    }

    #[test]
    fn hypotheses_translate_by_id_without_timing() {
        let mut d = MtHypotheses::from_map(parse_mt_hypotheses("{\"id\":\"a\",\"hyp\":\"one\"}\n{\"item\":\"b\",\"text\":\"two\"}").unwrap());
        assert!(!d.timed());
        assert_eq!(d.translate(&line("a", "x"), None, Duration::from_secs(1)).unwrap().text, "one");
        assert_eq!(d.translate(&line("b", "x"), None, Duration::from_secs(1)).unwrap().text, "two");
        assert!(d.translate(&line("c", "x"), None, Duration::from_secs(1)).unwrap_err().contains("no hypothesis"));
    }

    #[test]
    fn a_missing_nllb_model_fails_the_engine_with_the_catalog_id() {
        let paths = Paths::with_base(std::env::temp_dir());
        let def: EngineDef = serde_json::from_value(json!({
            "id": "n", "task": "mt", "backend": "nllb", "model": "mt/no-such-model-for-the-test"
        }))
        .unwrap();
        let err = create_nllb(&def, &paths).err().unwrap();
        assert!(err.contains("not installed: mt/no-such-model-for-the-test"), "{err}");
        let folder: EngineDef = serde_json::from_value(json!({
            "id": "n", "task": "mt", "backend": "nllb", "model": "no-such-folder"
        }))
        .unwrap();
        assert!(create_nllb(&folder, &paths).is_err());
    }

    #[test]
    fn the_openai_backend_needs_a_base_url_a_model_and_a_set_key() {
        let paths = Paths::with_base(std::env::temp_dir());
        let mk = |options: Value| -> EngineDef {
            serde_json::from_value(json!({"id": "o", "task": "mt", "backend": "openai", "options": options})).unwrap()
        };
        assert!(create_openai(&mk(json!({}))).err().unwrap().contains("baseUrl"));
        assert!(create_openai(&mk(json!({"baseUrl": "http://127.0.0.1:1/v1"}))).err().unwrap().contains("model"));
        let err = create_openai(&mk(json!({"baseUrl": "http://127.0.0.1:1/v1", "model": "m", "apiKeyEnv": "SIGIL_SPEECH_BENCH_NO_KEY"}))).err().unwrap();
        assert!(err.contains("SIGIL_SPEECH_BENCH_NO_KEY"));
        let ok = create_openai(&mk(json!({"baseUrl": "http://127.0.0.1:1", "model": "m", "prompt": "hy-mt2", "maxTokens": 99}))).unwrap();
        let v1 = create_openai(&mk(json!({"baseUrl": "http://127.0.0.1:1/v1/", "model": "m"}))).unwrap();
        assert_eq!(v1.url, "http://127.0.0.1:1/v1/chat/completions");
        assert_eq!(ok.url, "http://127.0.0.1:1/chat/completions", "the base URL is used as given (mt-bench does the same): the /v1 belongs in the spec");
        assert_eq!((ok.max_tokens, ok.prompt.as_str(), ok.local), (99, "hy-mt2", true));
        assert!(takes_context(&mk(json!({})), MtContext::Previous));
        assert!(!takes_context(&mk(json!({})), MtContext::None));
        let _ = paths;
    }
}
