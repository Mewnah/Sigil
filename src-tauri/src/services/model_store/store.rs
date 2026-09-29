//! The store: catalog + layout + queue + events for one app data folder.
//!
//! One global store serves the app, `--headless fetch` and the Rust API ([`global`]); tests build
//! their own on a temp folder with a loopback-friendly client.

use super::catalog::{self, ValidateOpts};
use super::download::{self, DownloadError, DownloadRequest};
use super::events::{EventSink, NullSink};
use super::extract::ExtractLimits;
use super::install::{self, record_matches};
use super::layout::{self, contents_present, read_record, InstalledRecord, Layout};
use super::legacy;
use super::mirrors::{self, ResolvedSource};
use super::queue::{CancelToken, Job, JobResult, Queue, CANCELLED};
use super::verify::Expected;
use super::{
    Catalog, CatalogEntry, CatalogFile, DiskUsage, EntryState, EntryStatus, ProgressEvent, SourceKind, StoreConfig,
    CATALOG_OVERRIDE_ENV, LOCAL_ASSETS_ENV,
};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock};
use std::time::Duration;
use tokio::sync::broadcast;

/// Tunables (tests shorten the timeouts).
#[derive(Clone, Debug)]
pub(crate) struct StoreSettings {
    /// [`LOCAL_ASSETS_ENV`]: serve files from this folder instead of the network.
    pub local_assets: Option<PathBuf>,
    /// Download unpinned (trust-on-first-use) non-executable files from the network: development
    /// builds only; release catalogs are fully pinned (build.rs).
    pub allow_tofu_network: bool,
    pub validate: ValidateOpts,
    pub stall_timeout: Duration,
    pub header_timeout: Duration,
    /// Attempts per mirror that made no progress (connect / header failures, drops and stalls
    /// before any new byte arrived). An attempt that appended bytes does not count.
    pub attempts_per_mirror: u32,
    /// Resumes per mirror after an attempt that did make progress and then dropped or stalled
    /// (bounds a server that trickles a few bytes per connection).
    pub max_resumes_per_mirror: u32,
    /// Waits before the 2nd, 3rd, ... attempt on one mirror.
    pub retry_backoff: Vec<Duration>,
    pub progress_interval: Duration,
    pub limits: ExtractLimits,
    /// Free space kept on top of what a download needs.
    pub free_space_margin: u64,
}

impl StoreSettings {
    /// Defaults of the app and `--headless fetch` (reads [`LOCAL_ASSETS_ENV`] and
    /// [`CATALOG_OVERRIDE_ENV`]).
    pub fn app_defaults() -> Self {
        let env_path = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
        Self {
            local_assets: env_path(LOCAL_ASSETS_ENV),
            allow_tofu_network: cfg!(debug_assertions),
            validate: ValidateOpts {
                allow_http_loopback: env_path(CATALOG_OVERRIDE_ENV).is_some(),
            },
            stall_timeout: crate::services::http_client::MODEL_STALL_TIMEOUT,
            header_timeout: Duration::from_secs(60),
            attempts_per_mirror: 3,
            max_resumes_per_mirror: 200,
            retry_backoff: vec![Duration::from_secs(2), Duration::from_secs(5)],
            progress_interval: super::events::PROGRESS_INTERVAL,
            limits: ExtractLimits::default(),
            free_space_margin: 64 << 20,
        }
    }
}

/// Transient (not on disk) state of an entry.
#[derive(Clone, Debug)]
struct Transient {
    state: EntryState,
    error: Option<String>,
}

/// One requested id after [`Store::enqueue`].
pub(crate) struct Enqueued {
    pub id: String,
    /// `None`: already installed and up to date.
    pub job: Option<Arc<Job>>,
    /// This call created the job (vs. joined one already queued or running).
    pub created: bool,
}

pub(crate) struct Store {
    pub settings: StoreSettings,
    pub layout: Layout,
    pub catalog: Arc<Catalog>,
    pub client: reqwest::Client,
    pub queue: Queue,
    config: Mutex<StoreConfig>,
    sink: RwLock<Arc<dyn EventSink>>,
    progress_tx: broadcast::Sender<ProgressEvent>,
    transient: Mutex<HashMap<String, Transient>>,
    /// Serializes install-record writes and deletions within this process.
    registry: Mutex<()>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

static GLOBAL: OnceLock<Result<Arc<Store>, String>> = OnceLock::new();

/// The store of this process (app data from `diagnostics::app_data_dir`, catalog from
/// [`catalog::load_effective`]). Created on first use.
pub(crate) fn global() -> Result<Arc<Store>, String> {
    GLOBAL
        .get_or_init(|| {
            let app_data = crate::services::diagnostics::app_data_dir()
                .ok_or_else(|| "the model store has no app data folder (APPDATA is not set)".to_string())?;
            let catalog = Catalog::load().inspect_err(|e| {
                tracing::error!(target: "sigil::models", "model catalog: {e}");
            })?;
            Ok(Store::new(
                &app_data,
                catalog,
                StoreSettings::app_defaults(),
                crate::services::http_client::model_download_client().clone(),
            ))
        })
        .clone()
}

/// Spawn on the current tokio runtime, or on Tauri's when called from elsewhere.
pub(crate) fn spawn_detached<F>(fut: F)
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    match tokio::runtime::Handle::try_current() {
        Ok(h) => {
            h.spawn(fut);
        }
        Err(_) => {
            tauri::async_runtime::spawn(fut);
        }
    }
}

/// Partial-download key of a catalog file: its pinned hash, else a hash of its identity.
pub(crate) fn partial_key(entry: &CatalogEntry, file: &CatalogFile) -> String {
    match Expected::of(file) {
        Expected::Sha256(h) => h,
        Expected::GitOid(o) => format!("git-{o}"),
        Expected::None => {
            let mut h = Sha256::new();
            for part in [entry.id.as_str(), file.path.as_str(), &file.size.to_string(), entry.version.as_str()] {
                h.update(part.as_bytes());
                h.update([0]);
            }
            format!("u-{}", &hex::encode(h.finalize())[..32])
        }
    }
}

impl Store {
    pub fn new(app_data: &Path, catalog: Catalog, settings: StoreSettings, client: reqwest::Client) -> Arc<Store> {
        let (progress_tx, _) = broadcast::channel(256);
        Arc::new(Store {
            settings,
            layout: Layout::new(app_data),
            catalog: Arc::new(catalog),
            client,
            queue: Queue::new(1),
            config: Mutex::new(StoreConfig::default()),
            sink: RwLock::new(Arc::new(NullSink)),
            progress_tx,
            transient: Mutex::new(HashMap::new()),
            registry: Mutex::new(()),
        })
    }

    pub fn set_sink(&self, sink: Arc<dyn EventSink>) {
        *self.sink.write().unwrap_or_else(|p| p.into_inner()) = sink;
    }

    fn sink(&self) -> Arc<dyn EventSink> {
        self.sink.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    pub fn registry_lock(&self) -> MutexGuard<'_, ()> {
        lock(&self.registry)
    }

    // -----------------------------------------------------------------------------------------
    // Config and events
    // -----------------------------------------------------------------------------------------

    pub fn config(&self) -> StoreConfig {
        lock(&self.config).clone()
    }

    /// Validate and apply a config (mirror, custom URL, parallel downloads).
    pub fn set_config(&self, cfg: StoreConfig) -> Result<StoreConfig, String> {
        let cfg = mirrors::normalize_config(cfg)?;
        self.queue.set_limit(cfg.max_concurrent as usize);
        *lock(&self.config) = cfg.clone();
        Ok(cfg)
    }

    pub fn emit_progress(&self, e: ProgressEvent) {
        self.sink().progress(&e);
        let _ = self.progress_tx.send(e);
    }

    /// Every progress event of this store (for bridges such as `vosk:download_progress`).
    pub fn subscribe_progress(&self) -> broadcast::Receiver<ProgressEvent> {
        self.progress_tx.subscribe()
    }

    /// Set (or with `None`, clear) the transient state of `id` and emit its status.
    pub fn set_state(&self, id: &str, state: Option<(EntryState, Option<String>)>) {
        {
            let mut t = lock(&self.transient);
            match state {
                Some((state, error)) => {
                    t.insert(id.to_string(), Transient { state, error });
                }
                None => {
                    t.remove(id);
                }
            }
        }
        if let Some(entry) = self.catalog.get(id) {
            self.sink().state(&self.status_of(entry));
        }
    }

    // -----------------------------------------------------------------------------------------
    // Lookups
    // -----------------------------------------------------------------------------------------

    pub fn entry(&self, id: &str) -> Result<&CatalogEntry, String> {
        let id = id.trim();
        self.catalog.get(id).ok_or_else(|| catalog::unknown_id(&self.catalog, id))
    }

    /// Root folder, record and bytes of an installed entry (record valid, files present).
    pub fn installed(&self, entry: &CatalogEntry) -> Option<(PathBuf, InstalledRecord, u64)> {
        let rec = read_record(&self.layout.record_path(entry), &entry.id)?;
        let root = self.layout.resolve_root(&rec.root)?;
        let bytes = contents_present(&root, &rec).ok()?;
        Some((root, rec, bytes))
    }

    /// Installed and matching the current catalog (hashes; version for unpinned or archives).
    pub fn is_up_to_date(&self, entry: &CatalogEntry) -> bool {
        self.installed(entry).is_some_and(|(_, rec, _)| record_matches(entry, &rec))
    }

    pub fn installed_path(&self, id: &str) -> Option<PathBuf> {
        let entry = self.catalog.get(id.trim())?;
        self.installed(entry).map(|(root, _, _)| root)
    }

    /// `(id, root)` of every catalog entry whose install record parses and resolves (files are not
    /// checked): the folders nothing may delete.
    pub fn record_roots(&self) -> Vec<(String, PathBuf)> {
        self.catalog
            .entries
            .iter()
            .filter_map(|e| {
                let rec = read_record(&self.layout.record_path(e), &e.id)?;
                Some((e.id.clone(), self.layout.resolve_root(&rec.root)?))
            })
            .collect()
    }

    pub fn status_of(&self, entry: &CatalogEntry) -> EntryStatus {
        let mut st = EntryStatus {
            id: entry.id.clone(),
            state: EntryState::NotInstalled,
            installed_version: None,
            installed_path: None,
            bytes_on_disk: 0,
            adopted: false,
            error: None,
        };
        if let Some((root, rec, bytes)) = self.installed(entry) {
            st.state = EntryState::Installed;
            st.installed_version = Some(rec.version.clone());
            st.installed_path = Some(root.display().to_string());
            st.bytes_on_disk = bytes;
            st.adopted = rec.origin == layout::Origin::Adopted;
        }
        if let Some(t) = lock(&self.transient).get(&entry.id) {
            st.state = t.state;
            st.error = t.error.clone();
        }
        st
    }

    pub fn status(&self, id: &str) -> Result<EntryStatus, String> {
        Ok(self.status_of(self.entry(id)?))
    }

    pub fn statuses(&self) -> Vec<EntryStatus> {
        self.catalog.entries.iter().map(|e| self.status_of(e)).collect()
    }

    // -----------------------------------------------------------------------------------------
    // Installing
    // -----------------------------------------------------------------------------------------

    /// Queue `ids` (and what they require, first). Already-installed entries get no job. Must run
    /// inside a tokio runtime (jobs are spawned on it).
    pub fn enqueue(self: &Arc<Self>, ids: &[String]) -> Result<Vec<Enqueued>, String> {
        let order = catalog::install_order(&self.catalog, ids)?;
        let mut jobs: HashMap<String, (Arc<Job>, bool)> = HashMap::new();
        for id in &order {
            let entry = self.catalog.get(id).expect("install_order returns catalog ids").clone();
            if let Some(job) = self.queue.get(id) {
                jobs.insert(id.clone(), (job, false));
                continue;
            }
            if self.is_up_to_date(&entry) {
                continue;
            }
            let deps: Vec<Arc<Job>> = entry
                .requires
                .iter()
                .filter_map(|r| jobs.get(r).map(|(j, _)| j.clone()).or_else(|| self.queue.get(r)))
                .collect();
            let (job, created) = self.queue.get_or_create(id, deps);
            if created {
                self.set_state(id, Some((EntryState::Queued, None)));
                let store = self.clone();
                let job2 = job.clone();
                spawn_detached(async move {
                    let result = install::run_job(&store, &job2, &entry).await;
                    store.finish_job(&job2, result);
                });
            }
            jobs.insert(id.clone(), (job, created));
        }
        Ok(ids
            .iter()
            .map(|id| {
                let id = id.trim().to_string();
                match jobs.get(&id) {
                    Some((job, created)) => Enqueued {
                        id,
                        job: Some(job.clone()),
                        created: *created,
                    },
                    None => Enqueued {
                        id,
                        job: None,
                        created: false,
                    },
                }
            })
            .collect())
    }

    pub(crate) fn finish_job(&self, job: &Job, result: JobResult) {
        match &result {
            Ok(_) => self.set_state(&job.id, None),
            Err(e) if e == CANCELLED => self.set_state(&job.id, Some((EntryState::Cancelled, None))),
            Err(e) => {
                tracing::warn!(target: "sigil::models", "{}: install failed: {e}", job.id);
                self.set_state(&job.id, Some((EntryState::Failed, Some(e.clone()))));
            }
        }
        self.queue.finish(job, result);
    }

    /// Install every id (and its `requires`) that is missing, then return each id's install folder
    /// in input order.
    pub async fn ensure(self: &Arc<Self>, ids: &[String]) -> Result<Vec<PathBuf>, String> {
        let queued = self.enqueue(ids)?;
        let mut out = Vec::with_capacity(queued.len());
        let mut errors = Vec::new();
        for q in queued {
            let result = match q.job {
                Some(job) => job.wait().await,
                None => self
                    .installed_path(&q.id)
                    .ok_or_else(|| "the install vanished while it was being used".to_string()),
            };
            match result {
                Ok(p) => out.push(p),
                Err(e) => errors.push(format!("{}: {e}", q.id)),
            }
        }
        if errors.is_empty() {
            Ok(out)
        } else {
            Err(errors.join("; "))
        }
    }

    /// Cancel a queued or running install (its partial download is kept). `false`: nothing to cancel.
    pub fn cancel(&self, id: &str) -> bool {
        match self.queue.get(id.trim()) {
            Some(job) => {
                job.cancel.cancel();
                true
            }
            None => false,
        }
    }

    /// Fetch one unpinned file from `url` into `dest` (the legacy Vosk custom-URL and unlisted-model
    /// path): same resume / stall / HTML rules and the same one-download-at-a-time slot, with
    /// `validate` run on the complete file before it is moved into place. The partial is kept on
    /// cancel (keyed by the URL).
    pub async fn fetch_unpinned(
        &self,
        label_id: &str,
        file_label: &str,
        url: &str,
        dest: &Path,
        validate: fn(&Path, &ExtractLimits) -> Result<(), String>,
        cancel: &CancelToken,
    ) -> Result<super::verify::Digest, DownloadError> {
        let parsed = reqwest::Url::parse(url.trim())
            .map_err(|e| DownloadError::Failed(format!("invalid download URL: {e}")))?;
        if !matches!(parsed.scheme(), "https" | "http") {
            return Err(DownloadError::Failed("only http:// and https:// URLs can be downloaded".into()));
        }
        let key = format!("url-{}", &hex::encode(Sha256::digest(parsed.as_str().as_bytes()))[..32]);
        let partial = self.layout.downloads().join(format!("{key}.partial"));
        let seq = self.queue.next_seq();
        let Some(_slot) = self.queue.acquire(seq, cancel).await else {
            return Err(DownloadError::Cancelled);
        };
        let sources = [ResolvedSource {
            kind: SourceKind::Direct,
            url: parsed,
            derived: true,
        }];
        let req = DownloadRequest {
            entry_id: label_id,
            file_label,
            expected: &Expected::None,
            size: None,
            sources: &sources,
            partial: partial.clone(),
            cross_mirror_resume: false,
            allow_html: false,
        };
        let done = download::download(self, &req, cancel).await?;
        tracing::info!(target: "sigil::models", "{label_id}: downloaded {file_label} ({} bytes)", done.size);
        let limits = self.settings.limits.clone();
        let dest = dest.to_path_buf();
        let path = done.path.clone();
        tokio::task::spawn_blocking(move || -> Result<(), String> {
            if let Err(e) = validate(&path, &limits) {
                download::remove_partial(&path);
                return Err(e);
            }
            layout::move_file(&path, &dest).map_err(|e| layout::describe_io("move the download into place", &e))
        })
        .await
        .map_err(|e| DownloadError::Failed(format!("install task failed: {e}")))?
        .map_err(DownloadError::Failed)?;
        Ok(done.digest)
    }

    // -----------------------------------------------------------------------------------------
    // Disk
    // -----------------------------------------------------------------------------------------

    /// Map of partial key → family for every catalog file.
    fn partial_families(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        for e in &self.catalog.entries {
            for f in &e.files {
                m.insert(partial_key(e, f), e.family.clone());
            }
        }
        m
    }

    pub fn disk_usage(&self) -> DiskUsage {
        let mut per_family: BTreeMap<String, u64> = BTreeMap::new();
        for e in &self.catalog.entries {
            if let Some((_, _, bytes)) = self.installed(e) {
                *per_family.entry(e.family.clone()).or_default() += bytes;
            }
        }
        let families = self.partial_families();
        if let Ok(rd) = fs::read_dir(self.layout.downloads()) {
            for d in rd.flatten() {
                let name = d.file_name().to_string_lossy().into_owned();
                let Some(key) = name.strip_suffix(".partial") else { continue };
                if let (Some(fam), Ok(m)) = (families.get(key), d.metadata()) {
                    *per_family.entry(fam.clone()).or_default() += m.len();
                }
            }
        }
        // Old installs still waiting for removal (a DLL was in use): on disk until the next start.
        for p in layout::read_pending_trash(&self.layout).roots {
            if let Some(dir) = self.layout.resolve_root(&p.root).filter(|d| self.layout.is_store_root(d)) {
                let bytes = layout::dir_size(&dir);
                if bytes > 0 {
                    *per_family.entry(p.family).or_default() += bytes;
                }
            }
        }
        let root = if self.layout.models.exists() { &self.layout.models } else { &self.layout.app_data };
        DiskUsage {
            models_root: Some(self.layout.models.display().to_string()),
            total_bytes: per_family.values().sum(),
            per_family,
            free_bytes: layout::free_space(root),
        }
    }

    /// Startup housekeeping (blocking; run on a background thread): sweep leftovers, adopt legacy
    /// files, refresh record versions. Never fails.
    pub fn maintenance(&self) {
        self.sweep();
        let adopted = legacy::adopt_all(self);
        if !adopted.is_empty() {
            tracing::info!(target: "sigil::models", "adopted existing model files: {}", adopted.join(", "));
        }
        self.refresh_versions();
    }

    /// Remove leftovers: old staging folders, the trash, abandoned partials, unreferenced version
    /// folders and temp files under `models/`, and install folders listed in
    /// `models/.pending-trash.json` (under `models/` or `runtimes/`) that no record uses.
    pub fn sweep(&self) {
        const STAGING_AGE: Duration = Duration::from_secs(6 * 3600);
        const UNKNOWN_PARTIAL_AGE: Duration = Duration::from_secs(7 * 24 * 3600);
        const KNOWN_PARTIAL_AGE: Duration = Duration::from_secs(30 * 24 * 3600);
        const ORPHAN_AGE: Duration = Duration::from_secs(600);
        let age = |p: &Path| {
            fs::metadata(p)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.elapsed().ok())
                .unwrap_or(Duration::MAX)
        };
        let me = format!(".{}.", std::process::id());
        if let Ok(rd) = fs::read_dir(self.layout.staging()) {
            for d in rd.flatten() {
                let name = d.file_name().to_string_lossy().into_owned();
                if !name.contains(&me) && age(&d.path()) >= STAGING_AGE {
                    let _ = super::extract::remove_tree(&d.path());
                }
            }
        }
        if let Ok(rd) = fs::read_dir(self.layout.trash()) {
            for d in rd.flatten() {
                let p = d.path();
                let _ = if p.is_dir() { super::extract::remove_tree(&p) } else { fs::remove_file(&p) };
            }
        }
        self.sweep_pending_trash();
        let known = self.partial_families();
        let running: HashSet<String> = self
            .queue
            .ids()
            .iter()
            .filter_map(|id| self.catalog.get(id))
            .flat_map(|e| e.files.iter().map(move |f| partial_key(e, f)))
            .collect();
        if let Ok(rd) = fs::read_dir(self.layout.downloads()) {
            for d in rd.flatten() {
                let name = d.file_name().to_string_lossy().into_owned();
                let key = name.strip_suffix(".partial.json").or_else(|| name.strip_suffix(".partial"));
                let Some(key) = key else {
                    if name.ends_with(".tmp") && age(&d.path()) >= ORPHAN_AGE {
                        let _ = fs::remove_file(d.path());
                    }
                    continue;
                };
                if running.contains(key) {
                    continue;
                }
                let limit = if known.contains_key(key) { KNOWN_PARTIAL_AGE } else { UNKNOWN_PARTIAL_AGE };
                if age(&d.path()) >= limit {
                    let _ = fs::remove_file(d.path());
                }
            }
        }
        // Version folders of ordinary entries that no record points at (a crash between the
        // rename into place and the record write, or an old version that was in use). Never a
        // folder that holds (or is) another entry's folder or a recorded install: catalog
        // validation forbids nested entry folders, and this keeps a bad fragment from deleting one.
        let mut protected: Vec<PathBuf> = self.catalog.entries.iter().map(|e| self.layout.entry_dir(e)).collect();
        protected.extend(self.record_roots().into_iter().map(|(_, r)| r));
        for e in &self.catalog.entries {
            if legacy::legacy_folder(&e.family).is_some() || self.queue.get(&e.id).is_some() {
                continue;
            }
            let dir = self.layout.entry_dir(e);
            let Ok(rd) = fs::read_dir(&dir) else { continue };
            for d in rd.flatten() {
                let p = d.path();
                let name = d.file_name().to_string_lossy().into_owned();
                if p.is_dir() && !protected.iter().any(|x| x.starts_with(&p)) && age(&p) >= ORPHAN_AGE {
                    let _ = layout::trash_dir(&self.layout, &p);
                } else if p.is_file() && name.ends_with(".tmp") && age(&p) >= ORPHAN_AGE {
                    let _ = fs::remove_file(&p);
                }
            }
        }
    }

    /// Retry the removal of install folders that were in use when they were replaced or removed.
    /// A folder a record points into again (or that holds one) is forgotten, never deleted.
    fn sweep_pending_trash(&self) {
        let _g = self.registry_lock();
        let mut pending = layout::read_pending_trash(&self.layout);
        if pending.roots.is_empty() {
            return;
        }
        let before = pending.roots.clone();
        let referenced: Vec<PathBuf> = self.record_roots().into_iter().map(|(_, r)| r).collect();
        let installing = !self.queue.ids().is_empty();
        pending.roots.retain(|p| {
            let Some(dir) = self.layout.resolve_root(&p.root).filter(|d| self.layout.is_store_root(d)) else {
                return false;
            };
            if !dir.exists() || referenced.iter().any(|r| r.starts_with(&dir) || dir.starts_with(r)) {
                return false;
            }
            if installing {
                // An install may be about to rename a folder into this place: leave it for later.
                return true;
            }
            match layout::trash_dir(&self.layout, &dir) {
                Ok(()) => {
                    tracing::info!(target: "sigil::models", "removed the old install {}", dir.display());
                    false
                }
                Err(e) => {
                    tracing::info!(target: "sigil::models", "{} is still in use ({e})", dir.display());
                    true
                }
            }
        });
        if pending.roots != before {
            if let Err(e) = layout::write_pending_trash(&self.layout, &pending) {
                tracing::warn!(target: "sigil::models", "could not update the pending-removal list: {e}");
            }
        }
    }

    /// Rewrite the version of records whose files still match a re-pinned catalog entry, so
    /// `installed_version != version` keeps meaning "an update is available".
    pub fn refresh_versions(&self) {
        for e in &self.catalog.entries {
            let Some((_, mut rec, _)) = self.installed(e) else { continue };
            if rec.version == e.version || !record_matches(e, &rec) {
                continue;
            }
            let _g = self.registry_lock();
            rec.version = e.version.clone();
            let _ = layout::write_record(&self.layout.record_path(e), &rec);
        }
    }
}

#[cfg(test)]
impl StoreSettings {
    /// Short timeouts and backoffs, loopback http allowed, trust-on-first-use allowed.
    pub(crate) fn test_defaults() -> Self {
        Self {
            local_assets: None,
            allow_tofu_network: true,
            validate: ValidateOpts { allow_http_loopback: true },
            stall_timeout: Duration::from_millis(600),
            header_timeout: Duration::from_secs(5),
            attempts_per_mirror: 3,
            max_resumes_per_mirror: 50,
            retry_backoff: vec![Duration::from_millis(20)],
            progress_interval: Duration::from_millis(1),
            limits: ExtractLimits::default(),
            free_space_margin: 0,
        }
    }
}

#[cfg(test)]
impl Store {
    /// A store on `app_data` with [`StoreSettings::test_defaults`] and a proxy-free client.
    pub(crate) fn for_test(app_data: &Path, catalog: Catalog) -> Arc<Store> {
        Self::with_settings(app_data, catalog, StoreSettings::test_defaults())
    }

    pub(crate) fn with_settings(app_data: &Path, catalog: Catalog, settings: StoreSettings) -> Arc<Store> {
        let client = crate::services::http_client::build_model_download_client(Duration::from_secs(5), false);
        Store::new(app_data, catalog, settings, client)
    }

    pub(crate) fn transient_state(&self, id: &str) -> Option<EntryState> {
        lock(&self.transient).get(id).map(|t| t.state)
    }
}
