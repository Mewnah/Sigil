//! Mozilla Bergamot client: `sigil-bergamot.exe` (Mozilla's Bergamot translation engine, the `inference/` tree of
//! mozilla/translations, built by `scripts/runtime-packs/bergamot/build.ps1` from `scripts/runtime-packs/bergamot/helper/`) runs
//! under `process_host` as a JSON-lines stdio helper (`process_host::recipes::bergamot`, `StdioMode::Protocol` +
//! `take_stdio`; the reader thread keeps draining stdout so the helper never blocks). Per-direction models come from
//! the model store (Mozilla's registry pinned in `model-catalog/mt-bergamot.json`, ids `mt-bergamot/<src>_<tgt>`),
//! with an English pivot when no direct model exists. The runtime pack is `runtime/bergamot`
//! (`model-catalog/runtime-bergamot.json`, local-only / unpublished). Decision record: `docs/adr/0008-bergamot.md`.
//!
//! Owned by P4-E. Commands (frozen by P4-S0, see `mt_local/mod.rs`):
//! - `bergamot_start { threads? }` → [`MtEngineStatus`]: start the helper (idempotent; a different thread count
//!   restarts it).
//! - `bergamot_translate { request }` → [`MtResponse`]: route the pair (direct or through English), check the models
//!   are installed (`not_installed` names the missing ids; nothing downloads here), start the helper and load the
//!   models on demand (at most 6 directions are loaded, fewer with more worker threads; a load first unloads the
//!   least recently used ones no line is using), translate. Finals are queued before interims in the helper and get
//!   one retry if the helper crashed; interims beyond 3 in flight fail with `busy`. The same language on both sides
//!   returns the text unchanged.
//! - `bergamot_pairs` → [`MtPair`]`[]`: every direct direction plus every English pivot, with install state.
//! - `bergamot_cancel { requestId? }`: the call fails at once with `cancelled`; the helper drops it if still queued.
//! - `bergamot_stop`: cancel everything and stop the helper; a start issued right after it keeps its own helper.
//!   `bergamot_status` → [`MtEngineStatus`].
//! - Status changes are emitted as [`super::STATUS_EVENT`] (`engine: "bergamot"`).
//!
//! Module layout: [`catalog`] (directions, language codes, routes, model files), [`protocol`] (the wire format),
//! [`pack`] (runtime-pack manifest and CPU checks), [`client`] (connection and engine).

// `mt_local` allows dead code while its scaffold stubs exist; this module is complete, so it opts back in.
#![warn(dead_code)]

pub mod catalog;
pub mod client;
pub mod pack;
pub mod protocol;
#[cfg(test)]
mod tests;

use super::{MtEngineStatus, MtError, MtPair, MtRequest, MtResponse, STATUS_EVENT};
use crate::services::model_store::{self, Catalog};
use crate::services::process_host::recipes::{self, BergamotOptions, BERGAMOT_EXE, BERGAMOT_RUNTIME_ID};
use crate::services::process_host::HelperSpec;
use catalog::Directions;
use client::{Engine, EngineConfig, Environment};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use tauri::{command, AppHandle, Emitter, Runtime};

/// `tracing` target of the Bergamot client.
pub const LOG_TARGET: &str = "sigil::bergamot";
/// Translation cache entries in the helper (sentences; partial captions repeat earlier sentences).
pub const CACHE_SIZE: u32 = 1024;

/// The app's environment: the model store's catalog and installs, and the installed runtime pack.
#[derive(Default)]
struct StoreEnv {
    directions: OnceLock<Arc<Directions>>,
}

impl Environment for StoreEnv {
    fn directions(&self) -> Arc<Directions> {
        self.directions
            .get_or_init(|| match Catalog::load() {
                Ok(c) => Arc::new(Directions::from_catalog(&c)),
                Err(e) => {
                    tracing::warn!(target: LOG_TARGET, "no Bergamot directions: {e}");
                    Arc::new(Directions::default())
                }
            })
            .clone()
    }

    fn installed_path(&self, id: &str) -> Option<PathBuf> {
        model_store::installed_path(id)
    }

    fn helper_spec(&self, name: &str, threads: u32) -> Result<HelperSpec, MtError> {
        let dir = model_store::installed_path(BERGAMOT_RUNTIME_ID).ok_or_else(|| MtError::not_installed(vec![BERGAMOT_RUNTIME_ID.to_string()]))?;
        let exe = recipes::bergamot_exe(&dir);
        if !exe.is_file() {
            return Err(MtError::runtime(format!(
                "the Bergamot pack has no {BERGAMOT_EXE}: import it again in Models & Runtimes"
            )));
        }
        pack::check_pack(&dir, &exe, pack::CpuFeatures::detect())?;
        Ok(recipes::bergamot(
            &exe,
            &BergamotOptions {
                name: name.to_string(),
                threads: Some(threads),
                cache_size: Some(CACHE_SIZE),
                ..Default::default()
            },
        ))
    }
}

/// The app's engine (one helper per process).
pub fn engine() -> &'static Arc<Engine> {
    static ENGINE: OnceLock<Arc<Engine>> = OnceLock::new();
    ENGINE.get_or_init(|| Engine::new(Arc::new(StoreEnv::default()), EngineConfig::production()))
}

fn install_emitter<R: Runtime>(app: &AppHandle<R>) {
    let app = app.clone();
    engine().set_emitter(Box::new(move |status: &MtEngineStatus| {
        if let Err(e) = app.emit(STATUS_EVENT, status.clone()) {
            tracing::debug!(target: LOG_TARGET, "cannot emit {STATUS_EVENT}: {e}");
        }
    }));
}

/// Translate one line without the app (for the headless MT benchmark and other in-process callers).
#[allow(dead_code)] // For `--headless mt-bench` (a `bergamot` engine there is a P4-D / INTEGRATE follow-up).
pub async fn translate_line(request: MtRequest) -> Result<MtResponse, MtError> {
    engine().translate(request).await
}

/// Start the helper (idempotent).
#[command]
pub async fn bergamot_start<R: Runtime>(app: AppHandle<R>, threads: Option<u32>) -> Result<MtEngineStatus, MtError> {
    install_emitter(&app);
    engine().start(threads).await
}

/// Translate one line (starts the helper and loads the direction's models on demand).
#[command]
pub async fn bergamot_translate<R: Runtime>(app: AppHandle<R>, request: MtRequest) -> Result<MtResponse, MtError> {
    install_emitter(&app);
    engine().translate(request).await
}

/// Every direction the catalog offers, with install state.
#[command]
pub async fn bergamot_pairs() -> Result<Vec<MtPair>, MtError> {
    tokio::task::spawn_blocking(|| engine().pairs())
        .await
        .map_err(|e| MtError::runtime(format!("listing the Bergamot directions failed: {e}")))
}

/// Cancel one request (`request_id`) or every pending one (`None`).
#[command]
pub async fn bergamot_cancel(request_id: Option<String>) -> Result<(), MtError> {
    engine().cancel(request_id.as_deref());
    Ok(())
}

/// Stop the helper.
#[command]
pub async fn bergamot_stop<R: Runtime>(app: AppHandle<R>) -> Result<(), MtError> {
    install_emitter(&app);
    engine().stop().await;
    Ok(())
}

#[command]
pub async fn bergamot_status() -> Result<MtEngineStatus, MtError> {
    Ok(engine().status())
}
