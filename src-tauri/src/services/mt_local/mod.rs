//! `mt-local` plugin: local machine translation outside the webview. NLLB-200 runs in process on
//! Sigil's ONNX Runtime (CPU background profile, [`nllb`], P4-D); Mozilla Bergamot runs as the
//! `sigil-bergamot.exe` helper under `process_host` (JSON lines over stdio, [`bergamot`], P4-E).
//! Text-level logic (utterance tracking, LocalAgreement, the router and its fallback chain, prompt
//! building, glossary) stays in TypeScript (`src/core/services/translation/**`, P4-C); the TS engines
//! `translation/services/{nllb,bergamot}.ts` call the commands below.
//!
//! Scaffolded by P4-S0 (plugin registered in main.rs, capability `capabilities/mt-local.json`,
//! commands in `plugin-commands/mt-local.txt`). This file is owned by the scaffold; the engine files
//! are owned by their workstreams: `nllb.rs` (+ `nllb/tests.rs`), `tokenizer.rs`, `flores.rs`,
//! `bench.rs` (P4-D) and `bergamot.rs` with its submodules `bergamot/{catalog,client,pack,protocol,
//! tests}.rs` (P4-E; the helper's C++ source is `scripts/runtime-packs/bergamot/helper/`, ADR 0008).
//!
//! # Frozen contract (P4-S0; change only through INTEGRATE; structs may gain fields)
//! Wire format camelCase; TS mirrors `RustMt*` in `src/types.ts`.
//! - Shared types: [`MtKind`], [`MtRequest`], [`MtResponse`], [`MtEngineState`], [`MtEngineStatus`],
//!   [`MtPair`], [`MtErrorKind`], [`MtError`]. Every command returns `Result<_, MtError>`; the TS
//!   side receives the error object (`{ kind, message, missing }`) as the rejection value.
//! - Language codes in requests are canonical lower-case BCP-47 (`en`, `ja`, `zh-Hant`, `pt-BR`)
//!   produced by the TS `languages.ts`; each engine maps them itself (NLLB: FLORES-200 in
//!   `flores.rs`; Bergamot: its registry's ISO 639-1 codes). An empty `source` means "unknown":
//!   an engine that needs it fails with [`MtErrorKind::UnsupportedPair`].
//! - NLLB (`nllb.rs`, P4-D):
//!   - `nllb_load { model?: string, threads?: number }` → [`MtEngineStatus`]: loads (or keeps) the
//!     model (`model` = catalog id, default [`NLLB_DEFAULT_MODEL`]); `Err(not_installed)` names the
//!     missing catalog ids (the TS panel asks before downloading; engines never download).
//!   - `nllb_translate { request: MtRequest }` → [`MtResponse`]: loads on demand like `nllb_load`.
//!   - `nllb_cancel { requestId?: string }` → `()`: stops that request (or every pending one) at
//!     the next decoding step; the cancelled call fails with [`MtErrorKind::Cancelled`].
//!   - `nllb_unload` → `()`, `nllb_status` → [`MtEngineStatus`].
//! - Bergamot (`bergamot.rs`, P4-E):
//!   - `bergamot_start { threads?: number }` → [`MtEngineStatus`]: starts the helper
//!     (`process_host::recipes::bergamot`); idempotent.
//!   - `bergamot_translate { request: MtRequest }` → [`MtResponse`]: pivots through English when
//!     no direct model exists; starts the helper on demand.
//!   - `bergamot_pairs` → [`MtPair`]`[]`: every direction the catalog offers, with install state.
//!   - `bergamot_cancel { requestId?: string }`, `bergamot_stop` → `()`, `bergamot_status` →
//!     [`MtEngineStatus`].
//! - Event [`STATUS_EVENT`] `"mt_local:status"` (payload [`MtEngineStatus`]) on engine state changes.
//! - Headless: `Sigil.exe --headless mt-bench <spec.json> [--out <dir>]` →
//!   [`bench::headless_mt_bench`] (P4-D; chrF table over a local corpus, never downloads).
//! - Threads: every engine is a background workload: at most 2 intra-op threads unless the request
//!   asks for more (cap 4), ORT sessions only through `neural_audio::ort_session` (spinning off),
//!   inference inside `spawn_blocking`, never on the audio or main thread.

pub mod bench;
pub mod bergamot;
pub mod flores;
pub mod nllb;
pub mod tokenizer;

use serde::{Deserialize, Serialize};
use tauri::{
    plugin::{Builder, TauriPlugin},
    Runtime,
};

/// Plugin name (`plugin:mt-local|...`, capability `mt-local:default`).
pub const PLUGIN_NAME: &str = "mt-local";
/// Event emitted when an engine changes state (payload [`MtEngineStatus`]).
pub const STATUS_EVENT: &str = "mt_local:status";
/// Engine id of NLLB-200 in [`MtResponse::engine`] / [`MtEngineStatus::engine`].
pub const ENGINE_NLLB: &str = "nllb";
/// Engine id of Bergamot in [`MtResponse::engine`] / [`MtEngineStatus::engine`].
pub const ENGINE_BERGAMOT: &str = "bergamot";
/// Default NLLB catalog id (P4-D's `model-catalog/mt-nllb.json` must define it).
pub const NLLB_DEFAULT_MODEL: &str = "mt/nllb-200-distilled-600m-int8";
/// Larger NLLB variant (opt-in; the TS model picker names it, Rust accepts any `mt/nllb-*` catalog id).
#[allow(dead_code)]
pub const NLLB_LARGE_MODEL: &str = "mt/nllb-200-distilled-1.3b-int8";
/// Upper bound for per-request thread counts (background profile default is 2).
pub const MAX_THREADS: u32 = 4;

/// Whether the text is a live partial (may be superseded) or a final line (must be delivered).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MtKind {
    Interim,
    #[default]
    Final,
}

/// One translation request (one line).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MtRequest {
    pub text: String,
    /// Canonical BCP-47 source language; "" = unknown.
    pub source: String,
    /// Canonical BCP-47 target language.
    pub target: String,
    pub kind: MtKind,
    /// Caller id for `*_cancel` (the TS pipeline uses `<segmentId>#<revision>`).
    pub request_id: Option<String>,
    /// Engine model (catalog id); `None` = the engine's default or loaded model.
    pub model: Option<String>,
    /// Beam width (NLLB; 1 = greedy). Engines without beam search ignore it.
    pub beam: Option<u32>,
    /// Intra-op / worker threads (capped at [`MAX_THREADS`]); `None` = 2.
    pub threads: Option<u32>,
}

/// A finished translation.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MtResponse {
    pub text: String,
    /// [`ENGINE_NLLB`] or [`ENGINE_BERGAMOT`].
    pub engine: String,
    /// Model id(s) used; a Bergamot pivot names both directions (`en-ja via en`).
    pub model: String,
    /// Wall time inside the engine.
    pub elapsed_ms: u64,
    /// Whether the text went through an English pivot.
    pub pivot: bool,
}

/// Lifecycle of one engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MtEngineState {
    #[default]
    Idle,
    Loading,
    Ready,
    Error,
}

/// State of one engine (`*_status`, `*_load` / `*_start` replies and [`STATUS_EVENT`]).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MtEngineStatus {
    /// [`ENGINE_NLLB`] or [`ENGINE_BERGAMOT`].
    pub engine: String,
    pub state: MtEngineState,
    /// Loaded model (NLLB) or loaded directions (Bergamot, comma separated).
    pub model: Option<String>,
    /// `cpu` today (NLLB and Bergamot are CPU engines).
    pub device: Option<String>,
    pub threads: Option<u32>,
    /// Human-readable detail or the last error (never contains paths with user names or keys).
    pub detail: Option<String>,
}

/// One Bergamot direction (`bergamot_pairs`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MtPair {
    /// Canonical BCP-47 codes.
    pub source: String,
    pub target: String,
    /// Catalog ids needed for this direction (two when it pivots through English).
    pub ids: Vec<String>,
    pub installed: bool,
    pub pivot: bool,
}

/// Why a command failed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MtErrorKind {
    /// Model or runtime pack missing: [`MtError::missing`] lists the catalog ids.
    NotInstalled,
    /// The engine cannot translate this direction (or the source language is unknown).
    UnsupportedPair,
    /// Cancelled by `*_cancel` or a stop.
    Cancelled,
    /// The engine is loading or restarting; retrying later can succeed.
    Busy,
    /// Invalid request (empty text, bad option).
    InvalidInput,
    /// Engine or helper failure.
    #[default]
    Runtime,
    /// Stub until the owning workstream lands.
    NotImplemented,
}

/// Error of every `mt-local` command (serialized as `{ kind, message, missing }`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MtError {
    pub kind: MtErrorKind,
    pub message: String,
    /// Catalog ids to install ([`MtErrorKind::NotInstalled`]).
    pub missing: Vec<String>,
}

impl MtError {
    pub fn new(kind: MtErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            missing: Vec::new(),
        }
    }

    pub fn not_installed(missing: Vec<String>) -> Self {
        Self {
            kind: MtErrorKind::NotInstalled,
            message: format!("not installed: {}", missing.join(", ")),
            missing,
        }
    }

    pub fn unsupported_pair(source: &str, target: &str) -> Self {
        let source = if source.is_empty() { "?" } else { source };
        Self::new(MtErrorKind::UnsupportedPair, format!("cannot translate {source} to {target}"))
    }

    pub fn cancelled() -> Self {
        Self::new(MtErrorKind::Cancelled, "cancelled")
    }

    pub fn runtime(message: impl Into<String>) -> Self {
        Self::new(MtErrorKind::Runtime, message)
    }

    /// Stub error until `owner` (a workstream id) implements the command.
    pub fn not_implemented(owner: &str) -> Self {
        Self::new(MtErrorKind::NotImplemented, format!("not implemented yet ({owner})"))
    }

    pub fn is_cancelled(&self) -> bool {
        self.kind == MtErrorKind::Cancelled
    }
}

impl std::fmt::Display for MtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MtError {}

impl From<MtError> for String {
    fn from(e: MtError) -> String {
        e.message
    }
}

/// Clamp a requested thread count (`None` = 2, the background profile).
pub fn thread_count(requested: Option<u32>) -> u32 {
    requested.unwrap_or(2).clamp(1, MAX_THREADS)
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new(PLUGIN_NAME)
        .invoke_handler(tauri::generate_handler![
            nllb::nllb_load,
            nllb::nllb_translate,
            nllb::nllb_cancel,
            nllb::nllb_unload,
            nllb::nllb_status,
            bergamot::bergamot_start,
            bergamot::bergamot_translate,
            bergamot::bergamot_pairs,
            bergamot::bergamot_cancel,
            bergamot::bergamot_stop,
            bergamot::bergamot_status,
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_wire_format_is_camel_case_with_defaults() {
        let req: MtRequest = serde_json::from_value(serde_json::json!({
            "text": "hello",
            "source": "en",
            "target": "ja",
            "kind": "interim",
            "requestId": "mic:1:4#2",
        }))
        .unwrap();
        assert_eq!(req.kind, MtKind::Interim);
        assert_eq!(req.request_id.as_deref(), Some("mic:1:4#2"));
        assert!(req.model.is_none() && req.beam.is_none() && req.threads.is_none());
        // Missing fields default (a final line with no options).
        let req: MtRequest = serde_json::from_value(serde_json::json!({ "text": "x", "target": "de" })).unwrap();
        assert_eq!(req.kind, MtKind::Final);
        assert_eq!(req.source, "");

        let resp = serde_json::to_value(MtResponse {
            text: "こんにちは".into(),
            engine: ENGINE_BERGAMOT.into(),
            model: "en-ja".into(),
            elapsed_ms: 12,
            pivot: false,
        })
        .unwrap();
        assert_eq!(resp["elapsedMs"], 12);
        assert_eq!(resp["engine"], "bergamot");
    }

    #[test]
    fn errors_serialize_as_kind_message_missing() {
        let e = serde_json::to_value(MtError::not_installed(vec![NLLB_DEFAULT_MODEL.into()])).unwrap();
        assert_eq!(e["kind"], "not_installed");
        assert_eq!(e["missing"][0], NLLB_DEFAULT_MODEL);
        assert!(e["message"].as_str().unwrap().contains(NLLB_DEFAULT_MODEL));
        assert_eq!(serde_json::to_value(MtError::cancelled()).unwrap()["kind"], "cancelled");
        assert!(MtError::cancelled().is_cancelled());
        assert_eq!(
            serde_json::to_value(MtError::unsupported_pair("", "ja")).unwrap()["kind"],
            "unsupported_pair"
        );
        let status = serde_json::to_value(MtEngineStatus {
            engine: ENGINE_NLLB.into(),
            state: MtEngineState::Ready,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(status["state"], "ready");
    }

    #[test]
    fn thread_counts_are_capped() {
        assert_eq!(thread_count(None), 2);
        assert_eq!(thread_count(Some(0)), 1);
        assert_eq!(thread_count(Some(16)), MAX_THREADS);
    }
}
