//! `plugin:model-store|<cmd>` (list in `plugin-commands/model-store.txt`) and the plugin setup.

use super::events::TauriSink;
use super::install;
use super::store::global;
use super::{CatalogEntry, DiskUsage, EntryStatus, StoreConfig};
use std::sync::Arc;
use std::time::Duration;
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime,
};

/// Delay before the startup housekeeping (sweep, legacy adoption), so it never competes with the
/// app's own start.
const MAINTENANCE_DELAY: Duration = Duration::from_secs(3);

#[command]
async fn get_catalog() -> Result<Vec<CatalogEntry>, String> {
    Ok(global()?.catalog.entries.clone())
}

#[command]
async fn get_entry_states() -> Result<Vec<EntryStatus>, String> {
    let store = global()?;
    tauri::async_runtime::spawn_blocking(move || store.statuses())
        .await
        .map_err(|e| format!("status task failed: {e}"))
}

#[command]
async fn set_store_config(config: StoreConfig) -> Result<(), String> {
    global()?.set_config(config).map(|_| ())
}

/// Queue installs; resolves once queued (progress arrives as `models:progress` / `models:state`).
#[command]
async fn install_entries(ids: Vec<String>) -> Result<(), String> {
    if ids.is_empty() {
        return Ok(());
    }
    global()?.enqueue(&ids).map(|_| ())
}

/// Cancel a queued or running install; its partial download is kept for resume.
#[command]
async fn cancel_install(id: String) -> Result<(), String> {
    let store = global()?;
    store.entry(&id)?;
    store.cancel(&id);
    Ok(())
}

#[command]
async fn remove_entry(id: String) -> Result<EntryStatus, String> {
    install::remove(&global()?, &id).await
}

#[command]
async fn verify_entry(id: String) -> Result<EntryStatus, String> {
    install::verify(&global()?, &id).await
}

#[command]
async fn import_local(id: String, path: String) -> Result<EntryStatus, String> {
    install::import_local(&global()?, &id, &path).await
}

#[command]
async fn get_disk_usage() -> Result<DiskUsage, String> {
    let store = global()?;
    tauri::async_runtime::spawn_blocking(move || store.disk_usage())
        .await
        .map_err(|e| format!("disk usage task failed: {e}"))
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("model-store")
        .setup(|app, _api| {
            match global() {
                Ok(store) => {
                    store.set_sink(Arc::new(TauriSink(app.clone())));
                    let spawned = std::thread::Builder::new()
                        .name("sigil-model-store".into())
                        .spawn(move || {
                            std::thread::sleep(MAINTENANCE_DELAY);
                            store.maintenance();
                        });
                    if let Err(e) = spawned {
                        tracing::warn!(target: "sigil::models", "model store housekeeping not started: {e}");
                    }
                }
                Err(e) => tracing::error!(target: "sigil::models", "model store unavailable: {e}"),
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_catalog,
            get_entry_states,
            set_store_config,
            install_entries,
            cancel_install,
            remove_entry,
            verify_entry,
            import_local,
            get_disk_usage
        ])
        .build()
}
