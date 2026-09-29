//! `model-store` plugin: the one verified, resumable, mirror-aware store for models, runtime packs,
//! templates and benchmark assets (catalog, Range resume, mirrors, sha256 / git-oid, safe
//! extraction, atomic install, one download at a time, cancel, local import, disk usage, adoption
//! of the legacy `whisper/` and `vosk-models/` folders).
//!
//! Owned by P1-A; scaffolded by P1-S0b (plugin registered in main.rs, capability
//! `capabilities/model-store.json`, commands in `plugin-commands/model-store.txt`, catalog embedded
//! by build.rs).
//!
//! # Modules
//! - `catalog`: override loading, runtime validation, dependency order, id suggestions.
//! - `layout`: folders, `installed.json` records, atomic writes, free space.
//! - `mirrors`: source order (catalog order, derived hf-mirror twins, preference, custom mirror).
//! - `download`: HTTP Range resume, stall timeout, failover, streaming hash.
//! - `verify`: sha256 and git blob ids. `extract`: safe zip / tar.gz / tar.bz2 extraction.
//! - `install`: install / import / verify / remove. `legacy`: `whisper/` and `vosk-models/`.
//! - `queue`: FIFO slots, per-host locks, cancel. `store`: the store. `events`, `commands`.
//!
//! # Frozen contract (P1-S0b; change only through an INTEGRATE step, additive fields are fine)
//! - Catalog: [`EMBEDDED_CATALOG_JSON`] is the merged, build-validated `model-catalog/*.json`
//!   (`{"entries":[...]}`). [`Catalog`] / [`CatalogEntry`] / [`CatalogFile`] / [`CatalogSource`] /
//!   [`ArchiveSpec`] mirror `model-catalog/_schema.json` field for field (serde names identical).
//!   [`Catalog::load`] honours [`CATALOG_OVERRIDE_ENV`] (a fragment file or a directory of
//!   fragments) and falls back to the embedded catalog; [`LOCAL_ASSETS_ENV`] serves files from a
//!   local folder instead of the network (P1-A).
//! - Rust API (no `AppHandle`: works in the app, in tests and in headless children; the models root
//!   is `<diagnostics::app_data_dir()>/models`):
//!   - `async` [`ensure`]`(ids: &[impl AsRef<str>]) -> Result<Vec<PathBuf>, String>`: install every id (and its `requires`)
//!     that is not installed yet, sequentially, and return each id's install directory in input
//!     order. Already-installed ids return immediately without network access.
//!   - [`installed_path`]`(id) -> Option<PathBuf>`: the install directory of a verified, installed
//!     entry. Consumers (native_runtime, speech_rt, whisper, vosk) locate files ONLY through this,
//!     never by recomputing the on-disk layout (P1-A owns the layout; runtime packs should follow
//!     the global rule `app_data/runtimes/<name>/<ver>/<variant>/`).
//!   - [`models_root`]`() -> Option<PathBuf>`.
//!   - [`headless_fetch`]`(&FetchArgs) -> Result<String, String>`: body of
//!     `Sigil.exe --headless fetch <id>... [--mirror <m>] [--custom-url <url>] [--list]`; returns
//!     the text printed to stdout (Err → exit code 1).
//! - IPC `plugin:model-store|<cmd>` (camelCase args on the TS side; TS types in src/types.ts):
//!   - `get_catalog() -> CatalogEntry[]`
//!   - `get_entry_states() -> EntryStatus[]` (one row per catalog entry)
//!   - `set_store_config({ config: StoreConfig })` (the TS models service pushes the root keys
//!     `modelMirror`, `modelMirrorCustomUrl`, `modelMaxConcurrentDownloads` at init and on change)
//!   - `install_entries({ ids: string[] })` (queues; resolves once queued, progress via events)
//!   - `cancel_install({ id })` (keeps the `.partial` for resume)
//!   - `remove_entry({ id }) -> EntryStatus`
//!   - `verify_entry({ id }) -> EntryStatus` (re-hash the installed files)
//!   - `import_local({ id, path }) -> EntryStatus` (offline file or archive import, verified)
//!   - `get_disk_usage() -> DiskUsage`
//! - Events: [`PROGRESS_EVENT`] `"models:progress"` (payload [`ProgressEvent`], at most 4 Hz per
//!   download) and [`STATE_EVENT`] `"models:state"` (payload [`EntryStatus`], on every state change).
//! - Network rules: one download at a time by default (`max_concurrent` 1, at most 2, one per host);
//!   30 s stall timeout, no total timeout; tests never touch the network.
//!
//! # Layout and trust (P1-A; details in `layout.rs`, `legacy.rs`, `mirrors.rs`)
//! - Record: `models/<family>/<id without a leading "<family>/">/installed.json`, written last and
//!   atomically; an entry is installed only while its record parses and every recorded file is
//!   present with its recorded size.
//! - Files: ordinary entries under `models/<family>/<id rest>/<version>/`; `kind: runtime` under
//!   `runtimes/<id rest>/<version>/<variant or "default">/`; families `whisper` and `vosk` in the
//!   legacy `whisper/` and `vosk-models/` folders (their consumers read those folders; the Vosk
//!   WASM loader's asset scope is `$APPDATA/vosk-models/**`). Files already there are adopted after
//!   a hash check (startup scan, or at install), never downloaded again.
//! - Sources: catalog order plus an automatic `hf-mirror.com` twin of every `huggingface.co` URL;
//!   a non-`auto` mirror preference moves that family first; `custom` puts
//!   `<custom_url>/<entry id>/<file path>` first. [`LOCAL_ASSETS_ENV`] is searched first as
//!   `<dir>/<entry id>/<file path>`, then `<dir>/<file path>`, then `<dir>/<source file name>`.
//! - `unpublished` entries never use catalog URLs: they install from a local import or local
//!   assets, and from a custom mirror only for files the catalog pins by hash. Unpinned files of
//!   unpublished entries (today every Sigil-built runtime pack: `runtime/sigil-transcribe-*`,
//!   `runtime/sigil-echo-*`) install **only** through Import or [`LOCAL_ASSETS_ENV`], in every build.
//! - Unpinned files (no sha256 / git_oid) of published entries: network download only in debug
//!   builds and never for executables (nor for executables inside an unpinned archive); their
//!   sha256 is recorded on first use. `size: 0` on an unpinned file means "size unknown". Local
//!   imports of unpinned files (Sigil-built packs) are trusted because the user picked them.
//! - Record content paths (extracted archive members) follow the extraction name rule (spaces,
//!   `!`, non-ASCII allowed); catalog file paths are stricter (`[A-Za-z0-9._/+-]`).
//! - An old install folder that cannot be removed (a loaded runtime DLL) is listed in
//!   `models/.pending-trash.json` and removed by the startup sweep; disk usage counts it meanwhile.

mod catalog;
mod commands;
mod download;
mod events;
mod extract;
mod install;
mod layout;
mod legacy;
mod mirrors;
mod queue;
mod store;
#[cfg(test)]
mod tests;
mod verify;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

pub use commands::init;

// Crate-internal API for the legacy STT plugins (Vosk now, Whisper in P2): they keep their own
// commands but download, import and register through the store.
pub(crate) use download::DownloadError;
pub(crate) use extract::{validate_zip, ExtractLimits};
pub(crate) use install::register_legacy_file;
pub(crate) use layout::Origin;
#[allow(unused_imports)] // whisper_entry_id: for P2-A (Whisper downloads move onto the store).
pub(crate) use legacy::{vosk_entry_id, whisper_entry_id};
pub(crate) use queue::CancelToken;
pub(crate) use store::{global as global_store, Store};
pub(crate) use verify::Digest;

/// Event: download progress of one file (payload [`ProgressEvent`], throttled to at most 4 Hz).
pub const PROGRESS_EVENT: &str = "models:progress";
/// Event: an entry changed state (payload [`EntryStatus`]).
pub const STATE_EVENT: &str = "models:state";
/// Folder under the app-data directory that holds installed models.
pub const MODELS_DIR: &str = "models";
/// Per-entry install record written last (atomically) by an install.
pub const INSTALLED_MANIFEST: &str = "installed.json";
/// Development override: a catalog fragment file or a directory of fragments replacing the embedded catalog.
pub const CATALOG_OVERRIDE_ENV: &str = "SIGIL_MODEL_CATALOG_OVERRIDE";
/// Development override: serve catalog files from this local folder instead of the network.
pub const LOCAL_ASSETS_ENV: &str = "SIGIL_LOCAL_ASSETS";

/// The merged catalog produced by build.rs from `model-catalog/*.json` (validated at build time).
pub const EMBEDDED_CATALOG_JSON: &str = include_str!(concat!(env!("OUT_DIR"), "/model_catalog.json"));

// ---------------------------------------------------------------------------------------------
// Catalog types (mirror model-catalog/_schema.json)
// ---------------------------------------------------------------------------------------------

/// `kind` of a catalog entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntryKind {
    Model,
    Runtime,
    Template,
    Bench,
}

/// `status` of a catalog entry. `Unpublished` entries (Sigil-built assets not released yet) are
/// satisfied only by local import or [`LOCAL_ASSETS_ENV`] (plus a custom mirror for their
/// hash-pinned files); their catalog URLs are never used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PublishStatus {
    Published,
    Unpublished,
}

/// Mirror family of a download source (tried in catalog order; see [`MirrorPreference`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SourceKind {
    Huggingface,
    HfMirror,
    Modelscope,
    GithubRelease,
    NvidiaRedist,
    Direct,
    Custom,
}

/// One download source, pinned to an immutable revision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogSource {
    pub kind: SourceKind,
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

/// Archive format of a download that is extracted into the entry folder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchiveFormat {
    #[serde(rename = "zip")]
    Zip,
    #[serde(rename = "tar.bz2")]
    TarBz2,
    #[serde(rename = "tar.gz")]
    TarGz,
}

/// Safe extraction settings (no traversal or links; entry, size and ratio caps).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveSpec {
    pub format: ArchiveFormat,
    /// Leading path components to drop.
    #[serde(default)]
    pub strip: u32,
    /// Glob patterns to keep (empty: everything).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub include: Vec<String>,
}

/// One file of an entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogFile {
    /// Relative install path inside the entry folder.
    pub path: String,
    /// Bytes of the downloaded file.
    pub size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Git blob id for small non-LFS files without a published sha256.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_oid: Option<String>,
    pub sources: Vec<CatalogSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archive: Option<ArchiveSpec>,
}

/// One catalog entry (model, runtime pack, template or benchmark asset).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogEntry {
    /// Stable id, never renamed once published (e.g. `stt/silero-vad-v6.2`).
    pub id: String,
    pub family: String,
    pub kind: EntryKind,
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// SPDX id or short name, a fact only (never shown as a warning).
    pub license: String,
    pub status: PublishStatus,
    /// Entries installed first (e.g. a runtime pack).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requires: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_vram_mb: Option<u64>,
    /// Runtime variant, e.g. `cpu`, `vulkan`, `cuda13`, `cpu-avx2`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    pub files: Vec<CatalogFile>,
}

/// The merged catalog.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Catalog {
    pub entries: Vec<CatalogEntry>,
}

impl Catalog {
    /// Parse a catalog document (`{"entries":[...]}`).
    pub fn from_json(text: &str) -> Result<Catalog, String> {
        serde_json::from_str(text).map_err(|e| format!("invalid model catalog: {e}"))
    }

    /// The catalog embedded at build time.
    pub fn embedded() -> Result<Catalog, String> {
        Self::from_json(EMBEDDED_CATALOG_JSON)
    }

    /// The catalog in effect: [`CATALOG_OVERRIDE_ENV`] if set (a fragment file or a folder of
    /// fragments, validated like build.rs does), else the embedded one.
    pub fn load() -> Result<Catalog, String> {
        catalog::load_effective()
    }

    /// Entry by id.
    pub fn get(&self, id: &str) -> Option<&CatalogEntry> {
        self.entries.iter().find(|e| e.id == id)
    }
}

// ---------------------------------------------------------------------------------------------
// Store state and IPC payloads
// ---------------------------------------------------------------------------------------------

/// Install state of one entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryState {
    NotInstalled,
    Queued,
    Downloading,
    Verifying,
    Extracting,
    Installed,
    Failed,
    Cancelled,
}

/// Row of `get_entry_states` and payload of [`STATE_EVENT`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryStatus {
    pub id: String,
    pub state: EntryState,
    /// Version recorded in `installed.json` (may differ from the catalog after an update).
    #[serde(default)]
    pub installed_version: Option<String>,
    /// Install directory when installed.
    #[serde(default)]
    pub installed_path: Option<String>,
    /// Bytes on disk of the installed files (0 when not installed).
    #[serde(default)]
    pub bytes_on_disk: u64,
    /// Adopted from a legacy folder (whisper/, vosk-models/) without re-downloading.
    #[serde(default)]
    pub adopted: bool,
    /// Why the last operation failed (state `failed`).
    #[serde(default)]
    pub error: Option<String>,
}

/// Payload of [`PROGRESS_EVENT`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProgressEvent {
    pub id: String,
    /// Catalog file path being downloaded.
    pub file: String,
    pub bytes_done: u64,
    /// `None` when the size is unknown.
    pub bytes_total: Option<u64>,
    /// Smoothed download speed.
    pub bytes_per_sec: f64,
    /// Mirror family serving the bytes.
    pub mirror: SourceKind,
    /// 1-based attempt number across mirrors / resumes.
    pub attempt: u32,
}

/// Preferred mirror (root key `modelMirror`). `Auto` keeps catalog order; any other value tries
/// that family first, then the remaining sources in catalog order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MirrorPreference {
    #[default]
    Auto,
    Huggingface,
    HfMirror,
    Modelscope,
    GithubRelease,
    Custom,
}

/// Argument of `set_store_config`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreConfig {
    pub mirror: MirrorPreference,
    /// Base URL for [`MirrorPreference::Custom`] (ignored otherwise). P1-A validates it (https, or
    /// http for a private LAN host, since every file is still hash-verified).
    #[serde(default)]
    pub custom_url: Option<String>,
    /// Parallel downloads, clamped to 1..=2 (never two on one host).
    pub max_concurrent: u8,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            mirror: MirrorPreference::Auto,
            custom_url: None,
            max_concurrent: 1,
        }
    }
}

/// Result of `get_disk_usage`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiskUsage {
    pub models_root: Option<String>,
    pub total_bytes: u64,
    /// Bytes per catalog family.
    pub per_family: BTreeMap<String, u64>,
    /// Free bytes on the models volume, when known.
    pub free_bytes: Option<u64>,
}

/// Arguments of `Sigil.exe --headless fetch`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FetchArgs {
    /// Catalog ids to install (with their `requires`).
    pub ids: Vec<String>,
    /// `--mirror <auto|huggingface|hf-mirror|modelscope|github-release|custom>`.
    pub mirror: Option<String>,
    /// `--custom-url <https://...>` for the custom mirror.
    pub custom_url: Option<String>,
    /// `--list`: print the catalog and install states instead of fetching.
    pub list: bool,
}

// ---------------------------------------------------------------------------------------------
// Rust API
// ---------------------------------------------------------------------------------------------

/// `<app_data>/models`.
#[allow(dead_code)] // Frozen API for P1-D1 / P1-G* / P2 consumers.
pub fn models_root() -> Option<PathBuf> {
    crate::services::diagnostics::app_data_dir().map(|d| d.join(MODELS_DIR))
}

/// Install directory of `id` when it is installed: its record is valid and every recorded file is
/// present with its recorded size (a few `stat` calls, no hashing).
#[allow(dead_code)] // Frozen API for P1-D1 / P1-G* / P2 consumers.
pub fn installed_path(id: &str) -> Option<PathBuf> {
    store::global().ok()?.installed_path(id)
}

/// Install every id (and its `requires`) that is missing, one download at a time, and return
/// each id's install directory in input order. Already-installed ids return without network
/// access. Must be awaited on a tokio runtime (Tauri's, or the headless one).
#[allow(dead_code)] // Frozen API for P1-D1 / P1-G* / P2 consumers.
pub async fn ensure<S: AsRef<str>>(ids: &[S]) -> Result<Vec<PathBuf>, String> {
    let ids: Vec<String> = ids.iter().map(|s| s.as_ref().trim().to_string()).collect();
    store::global()?.ensure(&ids).await
}

/// Body of `Sigil.exe --headless fetch`: returns the text printed to stdout (`<id><TAB><folder>` per
/// id, or the `--list` table). Progress and state lines go to stderr.
pub fn headless_fetch(args: &FetchArgs) -> Result<String, String> {
    let store = store::global()?;
    store.set_sink(Arc::new(events::StderrSink::new()));
    let mut cfg = store.config();
    if let Some(m) = &args.mirror {
        cfg.mirror = mirrors::parse_mirror(m)?;
    }
    if let Some(u) = &args.custom_url {
        cfg.custom_url = Some(u.clone());
        if args.mirror.is_none() {
            cfg.mirror = MirrorPreference::Custom;
        }
    }
    store.set_config(cfg)?;
    if args.list {
        return Ok(list_text(&store));
    }
    if args.ids.is_empty() {
        return Err("fetch: name at least one catalog id (see `fetch --list`)".into());
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("sigil-fetch")
        .enable_all()
        .build()
        .map_err(|e| format!("fetch: cannot start the async runtime: {e}"))?;
    // Typing convenience: `base.test-small` finds `base/test-small` when exactly one id matches.
    let ids: Vec<String> = args
        .ids
        .iter()
        .map(|raw| {
            let id = raw.trim();
            if store.catalog.get(id).is_some() {
                return id.to_string();
            }
            match catalog::punctuation_match(&store.catalog, id) {
                Some(m) => {
                    eprintln!("[fetch] {id}: using catalog id {m}");
                    m
                }
                None => id.to_string(),
            }
        })
        .collect();
    let s = store.clone();
    let wanted = ids.clone();
    let paths = runtime.block_on(async move { s.ensure(&wanted).await })?;
    Ok(ids
        .iter()
        .zip(paths)
        .map(|(id, p)| format!("{id}\t{}", p.display()))
        .collect::<Vec<_>>()
        .join("\n"))
}

/// `fetch --list`: one line per entry, grouped by family.
fn list_text(store: &Store) -> String {
    let name = |v: serde_json::Value| v.as_str().unwrap_or_default().to_string();
    let mut out = vec![format!(
        "{:<36} {:<8} {:<12} {:<14} {:>10}  VERSION",
        "ID", "KIND", "STATUS", "STATE", "SIZE"
    )];
    for entries in catalog::by_family(&store.catalog).into_values() {
        for e in entries {
            let st = store.status_of(e);
            let size: u64 = e.files.iter().map(|f| f.size).sum();
            out.push(format!(
                "{:<36} {:<8} {:<12} {:<14} {:>10}  {}{}",
                e.id,
                name(serde_json::to_value(e.kind).unwrap_or_default()),
                name(serde_json::to_value(e.status).unwrap_or_default()),
                name(serde_json::to_value(st.state).unwrap_or_default()),
                events::human_bytes(size),
                e.version,
                if st.adopted { " (adopted)" } else { "" }
            ));
        }
    }
    out.join("\n")
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    fn embedded_catalog_parses_and_ids_are_unique() {
        let catalog = Catalog::embedded().expect("build.rs output parses as a Catalog");
        let mut ids: Vec<_> = catalog.entries.iter().map(|e| e.id.as_str()).collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n);
    }

    #[test]
    fn catalog_wire_format_matches_schema_names() {
        let json = r#"{"entries":[{"id":"base/test","family":"base","kind":"bench","version":"1",
            "license":"MIT","status":"unpublished","files":[{"path":"a/b.onnx","size":3,
            "sources":[{"kind":"hf-mirror","url":"https://example.invalid/a"}],
            "archive":{"format":"tar.bz2","strip":1}}]}]}"#;
        let c = Catalog::from_json(json).unwrap();
        let e = c.get("base/test").unwrap();
        assert_eq!(e.kind, EntryKind::Bench);
        assert_eq!(e.status, PublishStatus::Unpublished);
        assert_eq!(e.files[0].sources[0].kind, SourceKind::HfMirror);
        assert_eq!(e.files[0].archive.as_ref().unwrap().format, ArchiveFormat::TarBz2);
        let cfg: StoreConfig =
            serde_json::from_str(r#"{"mirror":"github-release","custom_url":null,"max_concurrent":1}"#).unwrap();
        assert_eq!(cfg.mirror, MirrorPreference::GithubRelease);
        // Every value of the TS root key `modelMirror` (MODEL_MIRRORS in src/core/schema.ts).
        for v in ["auto", "huggingface", "hf-mirror", "modelscope", "github-release", "custom"] {
            let m: MirrorPreference = serde_json::from_value(serde_json::json!(v)).unwrap();
            assert_eq!(serde_json::to_value(m).unwrap(), serde_json::json!(v));
        }
        assert_eq!(
            serde_json::to_value(EntryState::NotInstalled).unwrap(),
            serde_json::json!("not_installed")
        );
    }
}
