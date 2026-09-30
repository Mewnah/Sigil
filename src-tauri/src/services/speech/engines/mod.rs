//! ASR engine factory of the `speech` plugin (owned by P2-A; dispatch table frozen by P2-S0).
//!
//! # Frozen contract (P2-S0; change only through an INTEGRATE step)
//! - [`create`] dispatches an [`EngineSpec`] on its [`EngineKind`] to the engine module's own
//!   `create(&EngineSpec, &EngineContext) -> Result<Box<dyn AsrEngine>, AsrError>`:
//!   `whisper` / `http` (P2-A), `sherpa_online` / `sherpa_offline` (P2-B), `transcribe` (P2-D).
//!   Creation blocks (model load): call it on a worker thread / `spawn_blocking`.
//! - [`setup`] runs once from the plugin's setup and forwards to `sherpa_offline::setup` (P2-B).
//! - Each engine module owns its file only; adding an engine kind is an INTEGRATE change here and
//!   in `engine_api::EngineKind`.
//!
//! Pipeline helpers (P2-A, crate-internal): [`requirements`] (catalog ids an explicit engine needs,
//! for `RoutePlan::models` / `missing`) and [`batch_policy`] (request pacing the utterance worker
//! honours: the HTTP engine's Groq billing minimum).

pub mod http;
pub mod sherpa_offline;
pub mod sherpa_online;
pub mod transcribe;
pub mod whisper;

use std::path::Path;
use std::sync::Arc;

use tauri::{AppHandle, Runtime};

use super::engine_api::{AsrEngine, AsrError, EngineContext, EngineKind, EngineSpec};
use super::worker::BatchPolicy;

/// Create the engine described by `spec` (blocking: loads the model).
pub fn create(spec: &EngineSpec, ctx: &EngineContext) -> Result<Box<dyn AsrEngine>, AsrError> {
    match spec.engine {
        EngineKind::Whisper => whisper::create(spec, ctx),
        EngineKind::Http => http::create(spec, ctx),
        EngineKind::SherpaOnline => sherpa_online::create(spec, ctx),
        EngineKind::SherpaOffline => sherpa_offline::create(spec, ctx),
        EngineKind::Transcribe => transcribe::create(spec, ctx),
    }
}

/// Plugin setup hook (must return quickly; spawn threads for slow work).
pub fn setup<R: Runtime>(app: &AppHandle<R>) {
    sherpa_offline::setup(app);
}

/// Catalog ids the engine of `spec` needs and whether each is installed (no network, a few
/// `stat` calls). Engines without catalog models (HTTP, bring-your-own paths) need none. The
/// transcribe engine also lists the runtime pack this PC would use (plus the NVIDIA runtime for
/// `cuda13`); other runtime packs are the router's to list.
pub(crate) fn requirements(spec: &EngineSpec) -> Vec<(String, bool)> {
    match spec.engine {
        EngineKind::Http => Vec::new(),
        EngineKind::Transcribe => transcribe::requirements(spec),
        EngineKind::Whisper => whisper::catalog_id(&spec.model)
            .map(|id| {
                let installed = whisper::resolve_model(&spec.model).is_ok();
                vec![(id, installed)]
            })
            .unwrap_or_default(),
        _ => {
            let model = spec.model.trim();
            if model.is_empty() || Path::new(model).is_absolute() || !model.contains('/') {
                return Vec::new();
            }
            let installed = crate::services::model_store::installed_path(model).is_some();
            vec![(model.to_string(), installed)]
        }
    }
}

/// Request pacing for the utterance worker in front of `spec`'s engine.
pub(crate) fn batch_policy(spec: &EngineSpec) -> Option<Arc<dyn BatchPolicy>> {
    match spec.engine {
        EngineKind::Http => http::batch_policy(spec),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kind dispatches to a module; a stub or a missing model answers with an error, never
    /// a panic, and no test here touches the network or a real model.
    #[test]
    fn every_kind_dispatches_without_panicking() {
        for kind in EngineKind::ALL {
            let spec = EngineSpec::new(kind, "stt/does-not-exist");
            if let Err(e) = create(&spec, &EngineContext::headless()) {
                assert!(!e.message.is_empty());
            }
        }
    }

    #[test]
    fn requirements_skip_paths_and_http() {
        assert!(requirements(&EngineSpec::new(EngineKind::Http, "")).is_empty());
        let abs = std::env::temp_dir().join("model.bin");
        assert!(requirements(&EngineSpec::new(EngineKind::Whisper, abs.to_string_lossy())).is_empty());
        assert!(requirements(&EngineSpec::new(EngineKind::Whisper, "anime-whisper")).is_empty());
        let w = requirements(&EngineSpec::new(EngineKind::Whisper, "base.en"));
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].0, "whisper/base.en");
        assert!(batch_policy(&EngineSpec::new(EngineKind::Whisper, "base.en")).is_none());
    }
}
