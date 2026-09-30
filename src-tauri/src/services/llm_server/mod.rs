//! `llm-server` plugin: the bundled llama.cpp `llama-server` helper for local LLM translation
//! (Hy-MT2, MiLMMT, TranslateGemma, Seed-X, ... as GGUF), run through `process_host`
//! (`recipes::llama_server`: 127.0.0.1 only, random API key, kill-on-close Job Object), plus a
//! streaming HTTP proxy so the webview can reach OpenAI-compatible servers on the LAN without CORS /
//! CSP problems.
//!
//! Owned by P4-B (this whole folder, `model-catalog/llm-gguf.json`, `plugin-commands/llm-server.txt`).
//! Scaffolded by P4-S0: plugin registered in main.rs, capability `capabilities/llm-server.json`;
//! the names, arguments and wire types below are frozen. Implemented by P4-B.
//!
//! # Frozen contract (P4-S0; change only through INTEGRATE; structs may gain fields)
//! Wire format camelCase; TS mirrors `RustLlm*` in `src/types.ts`; the TS host module is
//! `src/core/services/llm_server/index.ts` (P4-B).
//! - `llm_server_start { config: LlmServerConfig }` → [`LlmServerStarted`]: resolves once the
//!   server answers `/health` (downloads nothing: `Err(not_installed)` lists the missing catalog
//!   ids — the GGUF and the runtime pack — for the TS consent UI). Refuses (`Err(refused)`) a plan
//!   the VRAM governor rejects (e.g. a 7B model on an 8 GB card while the voice changer runs)
//!   unless `config.allowOverBudget`. Starting with a different model or variant replaces the
//!   running server; the same config returns the running one.
//! - `llm_server_stop` → `()`.
//! - `llm_server_status` → [`LlmServerStatus`] (never contains the API key).
//! - `llm_server_endpoint` → `Option<`[`LlmServerEndpoint`]`>`: where to reach the running server
//!   (port and key can change after a helper restart; the key stays the same).
//! - `llm_server_logs { limit?: number }` → `process_host::HelperLogLine[]` (key redacted).
//! - `llm_server_plan { config }` → [`LlmServerPlan`]: the dry-run decision (variant, `-ngl`, ctx,
//!   flash-attn, VRAM estimate, allowed / refused with a reason code, missing ids).
//! - `list_llm_models` → [`LlmModelInfo`]`[]`: the GGUF catalog (family `mt-llm`) with prompt
//!   format, sizes and install state.
//! - `llm_proxy_chat { request: LlmProxyRequest, onEvent: Channel<LlmProxyEvent> }` → `()`:
//!   performs one HTTP request for the webview and streams the response through the channel
//!   (`response`, then `chunk`s, then `done`, or `error`); resolves after the last event. Rules:
//!   `http`/`https` URLs only, no redirects, headers passed through and never logged, 30 s stall
//!   timeout (plus `timeoutMs` overall when set), response bodies capped at 8 MB.
//! - `llm_proxy_cancel { requestId }` → `()`: aborts that proxy request (its call resolves with an
//!   `error` event `cancelled` or `Err(cancelled)`; P4-B: always the event, then `Ok`).
//! - Event [`STATE_EVENT`] `"llm_server:state"` (payload [`LlmServerStatus`]).
//! - Errors: every command returns `Result<_, LlmServerError>` (`{ kind, message, missing }`).
//! - VRAM: while the server runs it holds `resources::claim_external_vram(Subsystem::Llm, ..)`
//!   sized to the plan's estimate (0 MB on the CPU pack).
//!
//! # Implementation (P4-B)
//! - [`catalog`]: the GGUF catalog (`model-catalog/llm-gguf.json`) and its per-model routing data.
//! - [`gguf`]: bounded GGUF header reader (user files, KV-cache size).
//! - [`plan`]: the governor decision (variant, `-ngl`, context, VRAM, the 8 GB voice-changer rule).
//! - [`devices`]: pinning the server to the primary GPU; measuring its real footprint.
//! - [`server`]: lifecycle, watcher, VRAM claim, events.
//! - [`proxy`]: `llm_proxy_chat` / `llm_proxy_cancel`.
//!
//! Behaviour notes (P4-B):
//! - `start` resolves `model: "auto"` with `resources::recommend(MtLlm)`, skipping models the pinned
//!   llama.cpp cannot run (`availability: needs_runtime_update`, the 1.25-bit Hy-MT2). The plan
//!   uses the governor's room arithmetic plus the 8 GB rule: while the voice changer runs on the GPU
//!   of a card under 10 GB, models above 4.5 GB (the 7B class) are refused (`refused`,
//!   reason `vc_running`) unless `allowOverBudget`.
//! - The VRAM claim starts at the plan's estimate and is resized to the device buffers llama-server
//!   logs once it is ready (plus the runtime's base: 100 MB Vulkan, 400 MB CUDA).
//! - On a PC with more than one hardware GPU the server is pinned to the primary one
//!   (`--device <name>` from `--list-devices`, else `-sm none`).
//! - Extra server flags (checked against b11146's `--help`): `--no-webui -fit off --reasoning off
//!   --log-colors off`, `--poll 0` on full GPU offload, `--device none` on the CPU.
//! - Additive fields: [`LlmModelInfo`] gains `role`, `availability`, `directions`, `sampling`,
//!   `description` and `license` (TS mirror: `LlmModelInfoExt` in `src/core/services/llm_server`).
//! - A refused plan reports the GPU placement it refused (`gpuLayers`, `vramMb`, `device: gpu`), so
//!   `allowOverBudget` runs exactly that, with its `-ngl`, device pin and VRAM claim. A hard refusal
//!   (`no_gpu_runtime`, `needs_runtime_update`) cannot be overridden and lists nothing in `missing`.
//! - Proxy timeouts: the 30 s stall timeout applies to the body once the response headers arrived;
//!   the wait for the headers is bounded by `timeoutMs`, else 5 minutes (a non-streaming reply or a
//!   cold server loading its model sends nothing until it is done). A `llm_proxy_cancel` that
//!   arrives before its `llm_proxy_chat` is registered still cancels it (remembered for 60 s).

pub mod catalog;
pub mod devices;
pub mod gguf;
pub mod plan;
pub mod proxy;
pub mod server;
#[cfg(test)]
mod tests;

use crate::services::process_host::HelperLogLine;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tauri::{
    command,
    ipc::Channel,
    plugin::{Builder, TauriPlugin},
    Runtime,
};

/// Plugin name (`plugin:llm-server|...`, capability `llm-server:default`).
pub const PLUGIN_NAME: &str = "llm-server";
/// Event emitted when the server changes state (payload [`LlmServerStatus`]).
pub const STATE_EVENT: &str = "llm_server:state";
/// `process_host` helper name of the translation server.
pub const HELPER_NAME: &str = crate::services::process_host::recipes::LLAMA_SERVER_NAME;
/// `config.model` value that asks the governor (`resources::recommend(MtLlm)`) to choose.
pub const MODEL_AUTO: &str = "auto";
/// Catalog family of the GGUF translation models (`model-catalog/llm-gguf.json`).
pub const MODEL_FAMILY: &str = "mt-llm";

/// How to start the server. Every field is optional on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct LlmServerConfig {
    /// Catalog id (`mt/hy-mt2-1.8b-q4_k_m`, see `resources::policy::ids`), [`MODEL_AUTO`], or an
    /// absolute path to a user-supplied `.gguf` file.
    pub model: String,
    /// Runtime pack variant: `auto` (governor), `vulkan`, `cuda13` or `cpu`.
    pub variant: String,
    /// `-c`; `None` = 4096.
    pub ctx_size: Option<u32>,
    /// `-ngl`; `None` = from the VRAM plan (all layers when the model fits, 0 on the CPU pack).
    pub gpu_layers: Option<u32>,
    /// `-fa`: `auto`, `on` or `off`; `None` = `auto`.
    pub flash_attn: Option<String>,
    /// Start even when the governor refuses the plan (the user insisted).
    pub allow_over_budget: bool,
}

impl Default for LlmServerConfig {
    fn default() -> Self {
        Self {
            model: MODEL_AUTO.into(),
            variant: "auto".into(),
            ctx_size: None,
            gpu_layers: None,
            flash_attn: None,
            allow_over_budget: false,
        }
    }
}

/// Server lifecycle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LlmServerState {
    #[default]
    Stopped,
    /// Process started, model loading (health 503).
    Starting,
    Ready,
    /// Restarting after a crash (process_host restart policy).
    Restarting,
    Failed,
}

/// Server state (status replies and [`STATE_EVENT`]); never contains the API key.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct LlmServerStatus {
    pub state: LlmServerState,
    /// Catalog id (or file name of a user GGUF) being served.
    pub model: Option<String>,
    /// Runtime variant in use (`vulkan`, `cuda13`, `cpu`).
    pub variant: Option<String>,
    pub gpu_layers: Option<u32>,
    pub ctx_size: Option<u32>,
    pub flash_attn: Option<String>,
    /// `http://127.0.0.1:<port>/v1` while running.
    pub base_url: Option<String>,
    /// VRAM claimed for the server (MiB; 0 on the CPU).
    pub vram_mb: u64,
    pub restarts: u32,
    pub error: Option<String>,
}

/// Where to reach the running server (only in command replies, never in events).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct LlmServerEndpoint {
    /// OpenAI-compatible base URL, `http://127.0.0.1:<port>/v1` (raw `/completion` is at the root).
    pub base_url: String,
    /// Bearer key for every request (llama-server `--api-key` via `LLAMA_API_KEY`).
    pub api_key: String,
    /// Model name to send (`--alias`).
    pub model: String,
}

/// Reply of `llm_server_start`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct LlmServerStarted {
    pub status: LlmServerStatus,
    pub endpoint: LlmServerEndpoint,
}

/// Dry-run decision for a config (`llm_server_plan`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct LlmServerPlan {
    /// Resolved catalog id (after [`MODEL_AUTO`]).
    pub model: String,
    pub variant: String,
    pub gpu_layers: u32,
    pub ctx_size: u32,
    pub flash_attn: String,
    /// `gpu` or `cpu`.
    pub device: String,
    /// Estimated VRAM (MiB; 0 on the CPU).
    pub vram_mb: u64,
    /// VRAM the governor found available (MiB).
    pub available_mb: u64,
    pub allowed: bool,
    /// `fits`, `vc_running`, `small_gpu`, `no_gpu`, `gpu_full`, `low_memory`, ... (the
    /// `resources::policy::Reason` codes).
    pub reason_code: String,
    /// Plain-English reason (logged; the UI words it by `reason_code`).
    pub reason: String,
    /// Catalog ids to install first (model, runtime pack).
    pub missing: Vec<String>,
}

/// A supported translation direction of a model narrower than its languages (canonical BCP-47).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct LlmDirection {
    pub from: String,
    pub to: String,
}

/// The model authors' recommended sampler (sent by the translation engine; `None` = server default).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct LlmSamplingPreset {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    /// `-1` = disabled.
    pub top_k: Option<i32>,
    pub repetition_penalty: Option<f32>,
}

/// One GGUF catalog model (`list_llm_models`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct LlmModelInfo {
    pub id: String,
    pub title: String,
    /// Prompt family for the TS prompt builder: `hy-mt2`, `translategemma`, `milmmt`, `seed-x`,
    /// `lmt`, `sakura`, `vntl`, `madlad`, `generic` (the `LlmPromptFormat` values of
    /// `translation/schema/llm.ts`).
    pub prompt_format: String,
    pub size_bytes: u64,
    /// Estimated VRAM with every layer on the GPU (MiB).
    pub vram_mb: u64,
    pub installed: bool,
    /// Opt-in specialist (CAT-Translate, Shisa, VNTL, Sakura/GalTransl, MADLAD): not offered by
    /// the automatic choice.
    pub specialist: bool,
    /// Languages it is meant for (canonical BCP-47); empty = broadly multilingual.
    pub languages: Vec<String>,
    /// The governor's current choice for [`MODEL_AUTO`].
    pub recommended: bool,
    /// P4-B: `default`, `challenger`, `alternative`, `high_gpu`, `ab` (offered for comparison) or
    /// `specialist`.
    pub role: String,
    /// P4-B: `ok`, `needs_runtime_update` (Sigil's llama.cpp cannot run it yet) or `experimental`.
    pub availability: String,
    /// P4-B: supported pairs when narrower than `languages` (e.g. VNTL: ja → en only).
    pub directions: Vec<LlmDirection>,
    /// P4-B: recommended sampler.
    pub sampling: Option<LlmSamplingPreset>,
    /// P4-B: catalog description (English).
    pub description: Option<String>,
    /// P4-B: license fact from the catalog.
    pub license: String,
}

/// One HTTP request performed by `llm_proxy_chat`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct LlmProxyRequest {
    /// Caller-chosen id for `llm_proxy_cancel`.
    pub request_id: String,
    /// Absolute `http`/`https` URL (e.g. `http://192.168.1.20:11434/v1/chat/completions`).
    pub url: String,
    /// `GET` or `POST` (default `POST`).
    pub method: Option<String>,
    /// Request headers, including the auth header; never logged.
    pub headers: BTreeMap<String, String>,
    /// JSON body text.
    pub body: Option<String>,
    /// Forward the body in chunks as it arrives (SSE); otherwise one `chunk` with the whole body.
    pub stream: bool,
    /// Overall deadline; `None` = only the 30 s body stall timeout (and at most 5 minutes waiting
    /// for the response headers).
    pub timeout_ms: Option<u64>,
}

/// One event of a proxied response (tagged by `kind`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum LlmProxyEvent {
    /// Status line and headers (lower-case names), before any chunk.
    #[serde(rename_all = "camelCase")]
    Response { status: u16, headers: BTreeMap<String, String> },
    /// Raw body text (UTF-8; SSE framing is parsed by the TS client).
    #[serde(rename_all = "camelCase")]
    Chunk { data: String },
    /// The body ended.
    Done,
    /// Transport failure or cancellation (`cancelled`); no event follows.
    #[serde(rename_all = "camelCase")]
    Error { message: String },
}

/// Why a command failed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmServerErrorKind {
    /// Model or runtime pack missing: [`LlmServerError::missing`] lists the catalog ids.
    NotInstalled,
    /// The VRAM governor refused the plan (see `llm_server_plan`).
    Refused,
    /// The helper did not become ready (crash, health timeout, exposure audit).
    StartFailed,
    NotRunning,
    Cancelled,
    InvalidInput,
    /// Proxy transport failure.
    Network,
    #[default]
    Runtime,
    /// Stub until P4-B lands.
    NotImplemented,
}

/// Error of every `llm-server` command (serialized as `{ kind, message, missing }`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct LlmServerError {
    pub kind: LlmServerErrorKind,
    pub message: String,
    pub missing: Vec<String>,
}

impl LlmServerError {
    pub fn new(kind: LlmServerErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            missing: Vec::new(),
        }
    }
}

impl std::fmt::Display for LlmServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for LlmServerError {}

#[command]
async fn llm_server_start(config: LlmServerConfig) -> Result<LlmServerStarted, LlmServerError> {
    server::start(config).await
}

#[command]
async fn llm_server_stop() -> Result<(), LlmServerError> {
    server::stop().await
}

#[command]
async fn llm_server_status() -> Result<LlmServerStatus, LlmServerError> {
    Ok(server::status())
}

#[command]
async fn llm_server_endpoint() -> Result<Option<LlmServerEndpoint>, LlmServerError> {
    Ok(server::endpoint())
}

#[command]
async fn llm_server_logs(limit: Option<u32>) -> Result<Vec<HelperLogLine>, LlmServerError> {
    Ok(server::logs(limit))
}

#[command]
async fn llm_server_plan(config: LlmServerConfig) -> Result<LlmServerPlan, LlmServerError> {
    server::plan(config).await
}

#[command]
async fn list_llm_models() -> Result<Vec<LlmModelInfo>, LlmServerError> {
    tauri::async_runtime::spawn_blocking(server::list_models)
        .await
        .map_err(|e| LlmServerError::new(LlmServerErrorKind::Runtime, format!("model list task failed: {e}")))?
}

#[command]
async fn llm_proxy_chat(request: LlmProxyRequest, on_event: Channel<LlmProxyEvent>) -> Result<(), LlmServerError> {
    proxy::run(request, |event| on_event.send(event).is_ok(), |url| proxy::client_for(url).clone()).await
}

#[command]
async fn llm_proxy_cancel(request_id: String) -> Result<(), LlmServerError> {
    proxy::cancel(&request_id);
    Ok(())
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new(PLUGIN_NAME)
        .invoke_handler(tauri::generate_handler![
            llm_server_start,
            llm_server_stop,
            llm_server_status,
            llm_server_endpoint,
            llm_server_logs,
            llm_server_plan,
            list_llm_models,
            llm_proxy_chat,
            llm_proxy_cancel,
        ])
        .build()
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    fn config_defaults_to_auto_and_reads_camel_case() {
        let cfg: LlmServerConfig = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(cfg, LlmServerConfig::default());
        assert_eq!(cfg.model, MODEL_AUTO);
        let cfg: LlmServerConfig = serde_json::from_value(serde_json::json!({
            "model": "mt/hy-mt2-1.8b-q4_k_m",
            "variant": "cuda13",
            "ctxSize": 8192,
            "gpuLayers": 999,
            "flashAttn": "on",
            "allowOverBudget": true,
        }))
        .unwrap();
        assert_eq!(cfg.ctx_size, Some(8192));
        assert_eq!(cfg.gpu_layers, Some(999));
        assert!(cfg.allow_over_budget);
    }

    #[test]
    fn status_never_carries_the_key_and_endpoint_does() {
        let status = serde_json::to_value(LlmServerStatus {
            state: LlmServerState::Ready,
            base_url: Some("http://127.0.0.1:5123/v1".into()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(status["state"], "ready");
        assert_eq!(status["baseUrl"], "http://127.0.0.1:5123/v1");
        assert!(!status.to_string().to_lowercase().contains("key"));
        let started = serde_json::to_value(LlmServerStarted {
            status: LlmServerStatus::default(),
            endpoint: LlmServerEndpoint {
                base_url: "http://127.0.0.1:5123/v1".into(),
                api_key: "k".into(),
                model: "hy-mt2".into(),
            },
        })
        .unwrap();
        assert_eq!(started["endpoint"]["apiKey"], "k");
    }

    #[test]
    fn proxy_events_are_tagged_by_kind() {
        let ev = serde_json::to_value(LlmProxyEvent::Response {
            status: 200,
            headers: BTreeMap::from([("content-type".to_string(), "text/event-stream".to_string())]),
        })
        .unwrap();
        assert_eq!(ev["kind"], "response");
        assert_eq!(ev["status"], 200);
        assert_eq!(
            serde_json::to_value(LlmProxyEvent::Chunk { data: "data: {}\n\n".into() }).unwrap()["kind"],
            "chunk"
        );
        assert_eq!(serde_json::to_value(LlmProxyEvent::Done).unwrap(), serde_json::json!({ "kind": "done" }));
        let req: LlmProxyRequest = serde_json::from_value(serde_json::json!({
            "requestId": "r1",
            "url": "http://192.168.1.20:11434/v1/chat/completions",
            "headers": { "authorization": "Bearer x" },
            "body": "{}",
            "stream": true,
            "timeoutMs": 15000,
        }))
        .unwrap();
        assert_eq!(req.request_id, "r1");
        assert!(req.stream && req.method.is_none());
        let err = serde_json::to_value(LlmServerError::new(LlmServerErrorKind::Refused, "7B does not fit")).unwrap();
        assert_eq!(err["kind"], "refused");
    }
}
