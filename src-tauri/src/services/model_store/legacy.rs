//! Legacy folders: `whisper/` (Whisper ggml files) and `vosk-models/` (Vosk zips) predate the store.
//! Their consumers still read them directly (whisper.rs; the Vosk WASM loader through the asset
//! protocol, whose scope is `$APPDATA/vosk-models/**`), so catalog families `whisper` and `vosk`
//! install their files into those folders instead of `models/<family>/...`, and files already there
//! are **adopted**: hashed once against the catalog and registered, never downloaded again.
//! A file that does not match (another revision, a custom model under a catalog name) is left
//! alone and remembered in `models/.adoption-cache.json` until it changes, so it is not re-hashed
//! at every start.

use super::install::register_legacy_file;
use super::layout::{self, join_rel, Origin};
use super::store::Store;
use super::verify::{hash_file, Expected};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::UNIX_EPOCH;

/// Catalog family → folder under the app data folder.
pub(crate) const LEGACY_FOLDERS: &[(&str, &str)] = &[("whisper", "whisper"), ("vosk", "vosk-models")];

const ADOPTION_CACHE: &str = ".adoption-cache.json";

pub(crate) fn legacy_folder(family: &str) -> Option<&'static str> {
    LEGACY_FOLDERS.iter().find(|(f, _)| *f == family).map(|(_, d)| *d)
}

pub(crate) fn is_legacy_folder(name: &str) -> bool {
    LEGACY_FOLDERS.iter().any(|(_, d)| *d == name)
}

/// Catalog id of a Vosk model id (`vosk-model-small-en-us-0.15` → `vosk/small-en-us-0.15`).
pub(crate) fn vosk_entry_id(model_id: &str) -> String {
    let id = model_id.trim();
    format!("vosk/{}", id.strip_prefix("vosk-model-").unwrap_or(id))
}

/// Catalog id of a Whisper model id as the Whisper plugin persists it (`base.en` → `whisper/base.en`).
/// For P2-A, when Whisper downloads move onto the store.
#[allow(dead_code)]
pub(crate) fn whisper_entry_id(model_id: &str) -> String {
    format!("whisper/{}", model_id.trim())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct Seen {
    size: u64,
    mtime_ms: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct AdoptionCache {
    /// `whisper/ggml-base.bin` → size and mtime of a file that did not match its catalog entry.
    #[serde(default)]
    rejected: BTreeMap<String, Seen>,
}

fn seen(path: &Path) -> Option<Seen> {
    let m = fs::metadata(path).ok()?;
    let mtime_ms = m
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_millis() as u64;
    Some(Seen { size: m.len(), mtime_ms })
}

/// Adopt every legacy file that matches its catalog entry and is not registered yet (blocking).
/// Returns the adopted ids.
pub(crate) fn adopt_all(store: &Store) -> Vec<String> {
    let cache_path = store.layout.models.join(ADOPTION_CACHE);
    let mut cache: AdoptionCache = fs::read(&cache_path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let mut dirty = false;
    let mut adopted = Vec::new();
    for entry in &store.catalog.entries {
        let Some(folder) = legacy_folder(&entry.family) else { continue };
        let [file] = entry.files.as_slice() else { continue };
        if store.queue.get(&entry.id).is_some() || store.installed(entry).is_some() {
            continue;
        }
        let expected = Expected::of(file);
        if !expected.is_pinned() {
            continue;
        }
        let root = store.layout.app_data.join(folder);
        let Ok(path) = join_rel(&root, &file.path) else { continue };
        let Some(now) = seen(&path) else { continue };
        if now.size != file.size {
            continue;
        }
        let key = format!("{folder}/{}", file.path);
        if cache.rejected.get(&key) == Some(&now) {
            continue;
        }
        match hash_file(&path, &expected, &|| false) {
            Ok(d) if expected.matches(&d) => {
                if register_legacy_file(store, entry, &d, Origin::Adopted) {
                    adopted.push(entry.id.clone());
                    if cache.rejected.remove(&key).is_some() {
                        dirty = true;
                    }
                    // Emits the new status (installed, adopted).
                    store.set_state(&entry.id, None);
                }
            }
            Ok(d) => {
                tracing::info!(
                    target: "sigil::models",
                    "{} does not match catalog entry {} ({} expected, got {}); left as is",
                    path.display(),
                    entry.id,
                    expected.describe(),
                    d.describe()
                );
                cache.rejected.insert(key, now);
                dirty = true;
            }
            Err(_) => {}
        }
    }
    // Forget files that no longer exist.
    let before = cache.rejected.len();
    cache
        .rejected
        .retain(|k, _| store.layout.app_data.join(k.replace('/', std::path::MAIN_SEPARATOR_STR)).exists());
    dirty |= cache.rejected.len() != before;
    if dirty {
        if let Ok(b) = serde_json::to_vec_pretty(&cache) {
            let _ = layout::write_atomic(&cache_path, &b);
        }
    }
    adopted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_map_from_the_legacy_plugins() {
        assert_eq!(vosk_entry_id("vosk-model-small-en-us-0.15"), "vosk/small-en-us-0.15");
        assert_eq!(vosk_entry_id("custom-model"), "vosk/custom-model");
        assert_eq!(whisper_entry_id("large-v3-turbo-q5_0"), "whisper/large-v3-turbo-q5_0");
        assert_eq!(legacy_folder("vosk"), Some("vosk-models"));
        assert_eq!(legacy_folder("stt"), None);
        assert!(is_legacy_folder("whisper"));
        assert!(!is_legacy_folder("models"));
    }
}
