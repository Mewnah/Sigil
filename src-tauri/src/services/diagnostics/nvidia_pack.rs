//! Interim NVIDIA GPU runtime pack for the CUDA execution provider (ORT 1.23 "cu12" build).
//!
//! `onnxruntime_providers_cuda.dll` (bundled next to `Sigil.exe`) imports the CUDA 12 runtime,
//! cuBLAS, cuFFT and cuDNN 9, which NVIDIA's driver does not install. This module installs them
//! from NVIDIA's official redistributable archives into
//! `app_data/runtimes/nvidia-cu12/<label>/bin/` and preloads them (full path, dependency order,
//! plus `AddDllDirectory`) right before the first CUDA session, so the provider's imports resolve
//! to the pack no matter what is on `PATH`. Nothing is load-time linked; without the pack Sigil
//! starts normally and the CUDA rung of the EP ladder falls back to DirectML with a visible reason.
//!
//! - **Pinned**: archive URLs, sizes and sha256 come from NVIDIA's `redistrib_12.9.1.json` (CUDA)
//!   and `redistrib_9.10.2.json` (cuDNN, `cuda12` variant). Bump them together with the ORT build.
//! - **Downloads**: one archive at a time (router-friendly), HTTP `Range` resume into
//!   `.downloads/<sha256>.partial`, streaming sha256, bounded retries, cancel. ~1.4 GB total.
//! - **Extraction**: only `*.dll` members, with Windows' own `System32\tar.exe` (bsdtar; refuses
//!   absolute and `..` paths) into a staging directory that is renamed into place atomically.
//!   The P1 model store replaces this interim downloader and extractor.
//! - **Removal** of a pack that this process already loaded is deferred to the next start.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use super::canary;

/// Runtime family folder under `app_data/runtimes/`.
pub const FAMILY: &str = "nvidia-cu12";
/// Pinned pack version (folder name).
pub const LABEL: &str = "cuda-12.9.1_cudnn-9.10.2";
/// Tauri event with [`PackProgress`] payloads.
pub const PROGRESS_EVENT: &str = "diagnostics:nvidia_pack";

const MANIFEST_FILE: &str = "installed.json";
const PENDING_REMOVAL_FILE: &str = ".pending-removal";
const MANIFEST_SCHEMA: u32 = 1;
const DOWNLOAD_ATTEMPTS: u32 = 6;
const EXTRACT_TIMEOUT: Duration = Duration::from_secs(20 * 60);
const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);

/// One pinned NVIDIA redistributable archive.
#[derive(Clone, Copy, Debug)]
pub struct Archive {
    pub id: &'static str,
    /// Redistributable root the manifest's `relative_path` is relative to.
    pub base: &'static str,
    /// `relative_path` from NVIDIA's redistrib JSON.
    pub rel_path: &'static str,
    pub sha256: &'static str,
    pub size: u64,
    /// DLLs taken from the archive's `bin/`.
    pub dlls: &'static [&'static str],
}

const CUDA_REDIST: &str = "https://developer.download.nvidia.com/compute/cuda/redist/";
const CUDNN_REDIST: &str = "https://developer.download.nvidia.com/compute/cudnn/redist/";

/// Pinned from redistrib_12.9.1.json (2025-06-05) and cuDNN redistrib_9.10.2.json (2025-06-06).
pub static ARCHIVES: [Archive; 4] = [
    Archive {
        id: "cuda_cudart 12.9.79",
        base: CUDA_REDIST,
        rel_path: "cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.9.79-archive.zip",
        sha256: "179e9c43b0735ffe67207b3da556eb5a0c50f3047961882b7657d3b822d34ef8",
        size: 3_521_238,
        dlls: &["cudart64_12.dll"],
    },
    Archive {
        id: "libcublas 12.9.1.4",
        base: CUDA_REDIST,
        rel_path: "libcublas/windows-x86_64/libcublas-windows-x86_64-12.9.1.4-archive.zip",
        sha256: "d534d98b0b453a98914dbf3adf47d7e84b55037abf02f87466439e1dcef581ed",
        size: 549_755_186,
        dlls: &["cublas64_12.dll", "cublasLt64_12.dll"],
    },
    Archive {
        id: "libcufft 11.4.1.4",
        base: CUDA_REDIST,
        rel_path: "libcufft/windows-x86_64/libcufft-windows-x86_64-11.4.1.4-archive.zip",
        sha256: "f26f80bb9abff3269c548e1559e8c2b4ba58ccb8acc6095bbc6404fc962d4b80",
        size: 198_361_265,
        dlls: &["cufft64_11.dll"],
    },
    Archive {
        id: "cudnn 9.10.2.21 (cuda12)",
        base: CUDNN_REDIST,
        rel_path: "cudnn/windows-x86_64/cudnn-windows-x86_64-9.10.2.21_cuda12-archive.zip",
        sha256: "c1a4567d822ebda7373fa1f19255dff4942302de741f830160b6c7d1fb31af23",
        size: 683_336_095,
        dlls: &[
            "cudnn64_9.dll",
            "cudnn_graph64_9.dll",
            "cudnn_ops64_9.dll",
            "cudnn_heuristic64_9.dll",
            "cudnn_adv64_9.dll",
            "cudnn_cnn64_9.dll",
            "cudnn_engines_precompiled64_9.dll",
            "cudnn_engines_runtime_compiled64_9.dll",
        ],
    },
];

/// Download URL of a pinned archive.
pub fn archive_url(a: &Archive) -> String {
    format!("{}{}", a.base, a.rel_path)
}

/// Load order for the preload: dependencies first. `false` = optional (loaded when it loads).
pub const PRELOAD_ORDER: [(&str, bool); 12] = [
    ("cudart64_12.dll", true),
    ("cublasLt64_12.dll", true),
    ("cublas64_12.dll", true),
    ("cufft64_11.dll", true),
    ("cudnn64_9.dll", true),
    ("cudnn_graph64_9.dll", true),
    ("cudnn_ops64_9.dll", true),
    ("cudnn_heuristic64_9.dll", true),
    ("cudnn_cnn64_9.dll", true),
    ("cudnn_engines_precompiled64_9.dll", true),
    // Needed only for RNN/attention graphs and runtime-fused engines (NVRTC); RVC uses neither.
    ("cudnn_adv64_9.dll", false),
    ("cudnn_engines_runtime_compiled64_9.dll", false),
];

/// Total download size of the pinned pack.
pub fn download_bytes() -> u64 {
    ARCHIVES.iter().map(|a| a.size).sum()
}

pub fn pack_root(app_data: &Path) -> PathBuf {
    app_data.join("runtimes").join(FAMILY)
}

pub fn install_dir(app_data: &Path) -> PathBuf {
    pack_root(app_data).join(LABEL)
}

pub fn bin_dir(app_data: &Path) -> PathBuf {
    install_dir(app_data).join("bin")
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InstalledFile {
    pub name: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InstalledSource {
    pub url: String,
    pub sha256: String,
    pub size: u64,
}

/// `installed.json` of an installed pack.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InstalledManifest {
    pub schema: u32,
    pub family: String,
    pub label: String,
    pub installed_at_unix_ms: u64,
    pub files: Vec<InstalledFile>,
    pub sources: Vec<InstalledSource>,
}

/// Pack state for the UI (`get_nvidia_pack_status`).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct NvidiaPackStatus {
    /// Pinned pack version.
    pub label: String,
    /// Bytes an install downloads.
    pub download_bytes: u64,
    /// The pinned pack is installed and complete.
    pub installed: bool,
    pub dir: Option<String>,
    /// Loaded into this process (a CUDA session was prepared).
    pub preloaded: bool,
    pub preload_error: Option<String>,
    pub installing: bool,
    /// Removal requested while loaded; completes at the next start.
    pub pending_removal: bool,
    /// Last install error in this process.
    pub last_error: Option<String>,
}

/// Payload of [`PROGRESS_EVENT`].
#[derive(Clone, Debug, Serialize)]
pub struct PackProgress {
    /// `download` / `extract` / `finalize` / `done` / `error` / `cancelled`.
    pub phase: &'static str,
    pub archive: Option<&'static str>,
    pub archive_index: usize,
    pub archive_count: usize,
    /// Overall bytes downloaded (all archives) and the pack total.
    pub done_bytes: u64,
    pub total_bytes: u64,
    pub message: Option<String>,
}

struct PreloadState {
    dir: PathBuf,
    result: Result<Vec<String>, String>,
}

static INSTALLING: AtomicBool = AtomicBool::new(false);
static CANCEL: AtomicBool = AtomicBool::new(false);

fn last_error() -> &'static Mutex<Option<String>> {
    static E: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    E.get_or_init(|| Mutex::new(None))
}

fn preload_state() -> &'static Mutex<Option<PreloadState>> {
    static P: OnceLock<Mutex<Option<PreloadState>>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(None))
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Every DLL a complete pack holds.
pub fn required_dlls() -> impl Iterator<Item = &'static str> {
    ARCHIVES.iter().flat_map(|a| a.dlls.iter().copied())
}

/// Read and check an installed pack: manifest of the pinned label, every DLL present with its
/// recorded size. (Hashes were checked at install; re-hashing 1.4 GB on every start is avoided.)
pub fn verify_installed(dir: &Path) -> Result<InstalledManifest, String> {
    let text = fs::read_to_string(dir.join(MANIFEST_FILE)).map_err(|e| format!("{MANIFEST_FILE}: {e}"))?;
    let manifest: InstalledManifest = serde_json::from_str(&text).map_err(|e| format!("{MANIFEST_FILE}: {e}"))?;
    if manifest.schema != MANIFEST_SCHEMA || manifest.family != FAMILY || manifest.label != LABEL {
        return Err(format!("{MANIFEST_FILE} describes another pack ({} {})", manifest.family, manifest.label));
    }
    for name in required_dlls() {
        let rec = manifest
            .files
            .iter()
            .find(|f| f.name.eq_ignore_ascii_case(name))
            .ok_or_else(|| format!("{name} is not in {MANIFEST_FILE}"))?;
        let len = fs::metadata(dir.join("bin").join(name)).map(|m| m.len()).map_err(|e| format!("{name}: {e}"))?;
        if len != rec.size {
            return Err(format!("{name} has {len} bytes, expected {}", rec.size));
        }
    }
    Ok(manifest)
}

fn pending_removal(dir: &Path) -> bool {
    dir.join(PENDING_REMOVAL_FILE).exists()
}

/// Current state of the pinned pack under `app_data`.
pub fn status(app_data: Option<&Path>) -> NvidiaPackStatus {
    let mut s = NvidiaPackStatus {
        label: LABEL.to_string(),
        download_bytes: download_bytes(),
        installing: INSTALLING.load(Ordering::Acquire),
        last_error: lock(last_error()).clone(),
        ..NvidiaPackStatus::default()
    };
    if let Some(app_data) = app_data {
        let dir = install_dir(app_data);
        s.pending_removal = pending_removal(&dir);
        s.installed = !s.pending_removal && verify_installed(&dir).is_ok();
        if s.installed || s.pending_removal {
            s.dir = Some(dir.display().to_string());
        }
    }
    if let Some(p) = lock(preload_state()).as_ref() {
        s.preloaded = p.result.is_ok();
        s.preload_error = p.result.as_ref().err().cloned();
    }
    s
}

fn cached_preload() -> Option<Result<(), String>> {
    lock(preload_state())
        .as_ref()
        .map(|p| p.result.as_ref().map(|_| ()).map_err(Clone::clone))
}

/// Preload the installed pack into this process (once; later calls return the first result).
/// `None` when no complete pack is installed — CUDA may still work from a system CUDA install on
/// `PATH`. Call before registering the CUDA EP. The state lock is never held while DLLs load, so
/// status queries from the UI thread stay instant.
pub fn ensure_preloaded() -> Option<Result<(), String>> {
    let app_data = super::app_data_dir()?;
    if let Some(r) = cached_preload() {
        return Some(r);
    }
    static SERIAL: Mutex<()> = Mutex::new(());
    let _serial = lock(&SERIAL);
    if let Some(r) = cached_preload() {
        return Some(r); // Another thread finished the preload while this one waited.
    }
    let dir = install_dir(&app_data);
    if pending_removal(&dir) {
        return None;
    }
    if let Err(e) = verify_installed(&dir) {
        if dir.exists() {
            tracing::warn!(target: "sigil::diagnostics", "NVIDIA runtime pack in {} is incomplete: {e}", dir.display());
        }
        return None;
    }
    let started = Instant::now();
    let result = imp::preload_dir(&dir.join("bin"));
    match &result {
        Ok(loaded) => tracing::info!(
            target: "sigil::diagnostics",
            "NVIDIA runtime pack {LABEL} preloaded ({} DLLs) in {} ms",
            loaded.len(),
            started.elapsed().as_millis()
        ),
        Err(e) => tracing::warn!(target: "sigil::diagnostics", "NVIDIA runtime pack {LABEL} failed to load: {e}"),
    }
    let out = result.as_ref().map(|_| ()).map_err(Clone::clone);
    *lock(preload_state()) = Some(PreloadState { dir, result });
    Some(out)
}

fn loaded_from(dir: &Path) -> bool {
    lock(preload_state()).as_ref().is_some_and(|p| p.dir == dir)
}

/// Remove leftovers at startup: interrupted staging directories, replaced packs and packs whose
/// removal was deferred. Never touches `.downloads` (partial downloads resume).
pub fn sweep(app_data: &Path) {
    let root = pack_root(app_data);
    let Ok(entries) = fs::read_dir(&root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        let stale = name.starts_with(".staging-") || name.starts_with(".old-") || (path.is_dir() && pending_removal(&path));
        if stale && path.is_dir() {
            match fs::remove_dir_all(&path) {
                Ok(()) => tracing::info!(target: "sigil::diagnostics", "removed {}", path.display()),
                Err(e) => tracing::warn!(target: "sigil::diagnostics", "cannot remove {}: {e}", path.display()),
            }
        }
    }
}

/// Ask a running install to stop (it keeps its partial download for a later resume).
pub fn cancel() {
    if INSTALLING.load(Ordering::Acquire) {
        CANCEL.store(true, Ordering::Release);
    }
}

/// Remove the installed pack; deferred to the next start when this process already loaded it.
pub fn remove(app_data: &Path) -> Result<NvidiaPackStatus, String> {
    if INSTALLING.load(Ordering::Acquire) {
        return Err("the NVIDIA runtime pack is being installed; cancel it first".to_string());
    }
    let dir = install_dir(app_data);
    if dir.exists() {
        let deferred = loaded_from(&dir) || {
            let old = pack_root(app_data).join(format!(".old-{}", super::logging::unix_ms_now()));
            match fs::rename(&dir, &old) {
                Ok(()) => {
                    let _ = fs::remove_dir_all(&old); // leftovers are swept at the next start
                    false
                }
                Err(_) => true,
            }
        };
        if deferred {
            fs::write(dir.join(PENDING_REMOVAL_FILE), b"remove at next start\n")
                .map_err(|e| format!("cannot mark {} for removal: {e}", dir.display()))?;
            tracing::info!(target: "sigil::diagnostics", "NVIDIA runtime pack is in use; it will be removed at the next start");
        } else {
            tracing::info!(target: "sigil::diagnostics", "NVIDIA runtime pack removed");
        }
        canary::invalidate();
    }
    Ok(status(Some(app_data)))
}

// ---------------------------------------------------------------------------------------------
// Install
// ---------------------------------------------------------------------------------------------

enum DlError {
    Cancelled,
    /// Retry (resuming the partial file).
    Transient(String),
    /// Bytes do not match the pinned hash: restart from zero.
    Corrupt(String),
    Fatal(String),
}

struct InstallGuard;

impl Drop for InstallGuard {
    fn drop(&mut self) {
        CANCEL.store(false, Ordering::Release);
        INSTALLING.store(false, Ordering::Release);
    }
}

fn cancelled() -> bool {
    CANCEL.load(Ordering::Acquire)
}

/// The downloader and extractor take the cancel flag as a parameter (production passes the
/// process-wide [`CANCEL`]) so tests running in parallel never see each other's cancellation.
fn is_set(flag: &AtomicBool) -> bool {
    flag.load(Ordering::Acquire)
}

/// sha256 of the first `len` bytes of `path`.
fn hash_prefix(path: &Path, len: u64) -> io::Result<Sha256> {
    let mut file = fs::File::open(path)?.take(len);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher)
}

pub(crate) fn sha256_file(path: &Path) -> io::Result<String> {
    let len = fs::metadata(path)?.len();
    Ok(hex::encode(hash_prefix(path, len)?.finalize()))
}

fn is_storage_full(e: &io::Error) -> bool {
    e.kind() == io::ErrorKind::StorageFull || e.raw_os_error() == Some(112) // ERROR_DISK_FULL
}

/// Parse `Content-Range: bytes <start>-<end>/<total>` → start.
fn content_range_start(value: &str) -> Option<u64> {
    let rest = value.trim().strip_prefix("bytes")?.trim_start();
    rest.split('-').next()?.trim().parse().ok()
}

async fn download_once(
    archive: &Archive,
    url: &str,
    partial: &Path,
    cancel: &AtomicBool,
    on_bytes: &mut (dyn FnMut(u64) + Send),
) -> Result<(), DlError> {
    use reqwest::header::{CONTENT_RANGE, CONTENT_TYPE, RANGE};
    use reqwest::StatusCode;
    use tokio::io::AsyncWriteExt;

    let mut existing = tokio::fs::metadata(partial).await.map(|m| m.len()).unwrap_or(0);
    if existing > archive.size {
        let _ = tokio::fs::remove_file(partial).await;
        existing = 0;
    }
    let mut hasher = if existing > 0 {
        let p = partial.to_path_buf();
        tokio::task::spawn_blocking(move || hash_prefix(&p, existing))
            .await
            .map_err(|e| DlError::Fatal(format!("hash task: {e}")))?
            .map_err(|e| DlError::Fatal(format!("cannot read {}: {e}", partial.display())))?
    } else {
        Sha256::new()
    };
    on_bytes(existing);
    if existing == archive.size {
        let digest = hex::encode(hasher.finalize());
        return if digest == archive.sha256 {
            Ok(())
        } else {
            Err(DlError::Corrupt(format!("{}: checksum mismatch", archive.id)))
        };
    }

    let mut req = super::super::http_client::http_download_get(url);
    if existing > 0 {
        req = req.header(RANGE, format!("bytes={existing}-"));
    }
    let what = format!("Downloading {}", archive.id);
    let mut resp = req.send().await.map_err(|e| {
        DlError::Transient(super::super::http_client::describe_request_error(
            &what,
            &e,
            super::super::http_client::DOWNLOAD_TIMEOUT,
        ))
    })?;
    let status = resp.status();
    let resumed = existing > 0
        && status == StatusCode::PARTIAL_CONTENT
        && resp
            .headers()
            .get(CONTENT_RANGE)
            .and_then(|v| v.to_str().ok())
            .and_then(content_range_start)
            == Some(existing);
    if status == StatusCode::RANGE_NOT_SATISFIABLE {
        let _ = tokio::fs::remove_file(partial).await;
        return Err(DlError::Transient(format!("{}: the server refused to resume; restarting", archive.id)));
    }
    if !status.is_success() {
        let detail = super::super::http_client::describe_failed_response(resp).await;
        let msg = format!("{what} failed: {detail}");
        return Err(if status.is_server_error() || status == StatusCode::TOO_MANY_REQUESTS {
            DlError::Transient(msg)
        } else {
            DlError::Fatal(msg)
        });
    }
    if resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.to_ascii_lowercase().contains("text/html"))
    {
        return Err(DlError::Fatal(format!("{what}: the server returned a web page instead of the archive")));
    }
    let mut offset = if resumed { existing } else { 0 };
    if !resumed {
        hasher = Sha256::new();
        on_bytes(0);
    }
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resumed)
        .truncate(!resumed)
        .open(partial)
        .await
        .map_err(|e| DlError::Fatal(format!("cannot write {}: {e}", partial.display())))?;
    loop {
        if is_set(cancel) {
            let _ = file.flush().await;
            return Err(DlError::Cancelled);
        }
        let chunk = match resp.chunk().await {
            Ok(Some(c)) => c,
            Ok(None) => break,
            Err(e) => {
                let _ = file.flush().await;
                return Err(DlError::Transient(super::super::http_client::describe_request_error(
                    &what,
                    &e,
                    super::super::http_client::DOWNLOAD_TIMEOUT,
                )));
            }
        };
        offset += chunk.len() as u64;
        if offset > archive.size {
            drop(file);
            let _ = tokio::fs::remove_file(partial).await;
            return Err(DlError::Corrupt(format!("{}: larger than the pinned size", archive.id)));
        }
        hasher.update(&chunk);
        file.write_all(&chunk).await.map_err(|e| {
            if is_storage_full(&e) {
                DlError::Fatal("the disk is full (the NVIDIA runtime pack needs about 3 GB free while installing)".to_string())
            } else {
                DlError::Fatal(format!("cannot write {}: {e}", partial.display()))
            }
        })?;
        on_bytes(offset);
    }
    file.flush().await.map_err(|e| DlError::Fatal(format!("flush {}: {e}", partial.display())))?;
    file.sync_all().await.map_err(|e| DlError::Fatal(format!("sync {}: {e}", partial.display())))?;
    drop(file);
    if offset != archive.size {
        return Err(DlError::Transient(format!(
            "{}: the connection ended at {offset} of {} bytes",
            archive.id, archive.size
        )));
    }
    let digest = hex::encode(hasher.finalize());
    if digest != archive.sha256 {
        let _ = tokio::fs::remove_file(partial).await;
        return Err(DlError::Corrupt(format!("{}: checksum mismatch (got {digest})", archive.id)));
    }
    Ok(())
}

/// Download `archive` from `url` to `dest` (verified), resuming `dest.partial`; stops with
/// `"cancelled"` once `cancel` is set.
async fn download_verified(
    archive: &Archive,
    url: &str,
    dest: &Path,
    cancel: &AtomicBool,
    on_bytes: &mut (dyn FnMut(u64) + Send),
) -> Result<(), String> {
    let partial = dest.with_extension("partial");
    let mut corrupt_restarts = 0;
    let mut attempt = 0;
    loop {
        attempt += 1;
        match download_once(archive, url, &partial, cancel, on_bytes).await {
            Ok(()) => {
                tokio::fs::rename(&partial, dest)
                    .await
                    .map_err(|e| format!("cannot finish {}: {e}", dest.display()))?;
                return Ok(());
            }
            Err(DlError::Cancelled) => return Err("cancelled".to_string()),
            Err(DlError::Fatal(e)) => return Err(e),
            Err(DlError::Corrupt(e)) => {
                let _ = tokio::fs::remove_file(&partial).await;
                corrupt_restarts += 1;
                if corrupt_restarts > 1 {
                    return Err(e);
                }
                tracing::warn!(target: "sigil::diagnostics", "{e}; downloading it again");
            }
            Err(DlError::Transient(e)) => {
                if attempt >= DOWNLOAD_ATTEMPTS {
                    return Err(e);
                }
                tracing::warn!(target: "sigil::diagnostics", "{e}; retrying ({attempt}/{DOWNLOAD_ATTEMPTS})");
                // Back off gently (the user's router dislikes bursts); cancel stays responsive.
                let until = Instant::now() + Duration::from_secs(2 * u64::from(attempt));
                while Instant::now() < until {
                    if is_set(cancel) {
                        return Err("cancelled".to_string());
                    }
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            }
        }
    }
}

fn system_tar() -> Result<PathBuf, String> {
    let root = std::env::var_os("SystemRoot")
        .or_else(|| std::env::var_os("windir"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
    let tar = root.join("System32").join("tar.exe");
    if tar.is_file() {
        Ok(tar)
    } else {
        Err(format!("{} is missing (Windows 10 1803 or later is required)", tar.display()))
    }
}

/// Extract every `*.dll` of `zip` below `out` with Windows' bsdtar (time-limited; stops with
/// `"cancelled"` once `cancel` is set).
fn extract_dlls(zip: &Path, out: &Path, cancel: &AtomicBool) -> Result<(), String> {
    use std::process::{Command, Stdio};
    if is_set(cancel) {
        return Err("cancelled".to_string());
    }
    let tar = system_tar()?;
    fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut cmd = Command::new(&tar);
    cmd.arg("-x")
        .arg("-f")
        .arg(zip)
        .arg("-C")
        .arg(out)
        .arg("*.dll")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = cmd.spawn().map_err(|e| format!("cannot run {}: {e}", tar.display()))?;
    let stderr = child.stderr.take();
    let reader = std::thread::spawn(move || {
        let mut s = String::new();
        if let Some(mut e) = stderr {
            let _ = e.read_to_string(&mut s);
        }
        s
    });
    let deadline = Instant::now() + EXTRACT_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if is_set(cancel) || Instant::now() > deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(if is_set(cancel) { "cancelled".to_string() } else { "extraction timed out".to_string() });
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(100)),
            Err(e) => return Err(format!("tar: {e}")),
        }
    };
    let err_text = reader.join().unwrap_or_default();
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "extracting {} failed ({status}): {}",
            zip.display(),
            super::logging::truncate_chars(err_text.trim(), 400)
        ))
    }
}

/// Find `name` (case-insensitive) anywhere below `dir`.
fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    let mut subdirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let ft = entry.file_type().ok()?;
        if ft.is_dir() {
            subdirs.push(path);
        } else if ft.is_file() && entry.file_name().to_string_lossy().eq_ignore_ascii_case(name) {
            return Some(path);
        }
    }
    subdirs.into_iter().find_map(|d| find_file(&d, name))
}

/// Move `archive.dlls` from an extraction tree into `bin`.
fn collect_dlls(archive: &Archive, extracted: &Path, bin: &Path) -> Result<(), String> {
    fs::create_dir_all(bin).map_err(|e| format!("{}: {e}", bin.display()))?;
    for name in archive.dlls {
        let src = find_file(extracted, name).ok_or_else(|| {
            format!("{name} was not found in {} (NVIDIA changed the archive layout?)", archive.id)
        })?;
        fs::rename(&src, bin.join(name)).map_err(|e| format!("{name}: {e}"))?;
    }
    Ok(())
}

fn write_manifest(staging: &Path) -> Result<(), String> {
    let bin = staging.join("bin");
    let mut files = Vec::new();
    for name in required_dlls() {
        let p = bin.join(name);
        let size = fs::metadata(&p).map_err(|e| format!("{name}: {e}"))?.len();
        let sha256 = sha256_file(&p).map_err(|e| format!("{name}: {e}"))?;
        files.push(InstalledFile {
            name: name.to_string(),
            size,
            sha256,
        });
    }
    let manifest = InstalledManifest {
        schema: MANIFEST_SCHEMA,
        family: FAMILY.to_string(),
        label: LABEL.to_string(),
        installed_at_unix_ms: super::logging::unix_ms_now(),
        files,
        sources: ARCHIVES
            .iter()
            .map(|a| InstalledSource {
                url: archive_url(a),
                sha256: a.sha256.to_string(),
                size: a.size,
            })
            .collect(),
    };
    let json = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    fs::write(staging.join(MANIFEST_FILE), json).map_err(|e| format!("{MANIFEST_FILE}: {e}"))
}

/// Download, verify and install the pinned pack. Emits [`PackProgress`] through `progress`.
/// Idempotent: returns immediately when the pack is already installed.
pub async fn install(
    app_data: PathBuf,
    progress: impl Fn(PackProgress) + Send + Sync + 'static,
) -> Result<NvidiaPackStatus, String> {
    if INSTALLING.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return Err("the NVIDIA runtime pack is already being installed".to_string());
    }
    let guard = InstallGuard;
    CANCEL.store(false, Ordering::Release);
    *lock(last_error()) = None;

    let result = install_inner(&app_data, &progress).await;
    let total = download_bytes();
    match &result {
        Ok(()) => {
            progress(PackProgress {
                phase: "done",
                archive: None,
                archive_index: ARCHIVES.len(),
                archive_count: ARCHIVES.len(),
                done_bytes: total,
                total_bytes: total,
                message: None,
            });
            tracing::info!(target: "sigil::diagnostics", "NVIDIA runtime pack {LABEL} installed in {}", install_dir(&app_data).display());
        }
        Err(e) => {
            let was_cancel = e == "cancelled";
            if !was_cancel {
                *lock(last_error()) = Some(e.clone());
                tracing::warn!(target: "sigil::diagnostics", "NVIDIA runtime pack install failed: {e}");
            }
            progress(PackProgress {
                phase: if was_cancel { "cancelled" } else { "error" },
                archive: None,
                archive_index: 0,
                archive_count: ARCHIVES.len(),
                done_bytes: 0,
                total_bytes: total,
                message: Some(e.clone()),
            });
        }
    }
    drop(guard); // `status` below must no longer report "installing"
    result.map(|()| status(Some(&app_data)))
}

async fn install_inner(app_data: &Path, progress: &(impl Fn(PackProgress) + Send + Sync)) -> Result<(), String> {
    let dir = install_dir(app_data);
    if !pending_removal(&dir) && verify_installed(&dir).is_ok() {
        return Ok(());
    }
    if loaded_from(&dir) {
        return Err("the NVIDIA runtime pack is loaded by this Sigil session; restart Sigil before reinstalling it".to_string());
    }
    let root = pack_root(app_data);
    let downloads = root.join(".downloads");
    fs::create_dir_all(&downloads).map_err(|e| format!("cannot create {}: {e}", downloads.display()))?;

    let total = download_bytes();
    let count = ARCHIVES.len();
    let mut before = 0u64;
    let mut zips = Vec::with_capacity(count);
    for (index, archive) in ARCHIVES.iter().enumerate() {
        let dest = downloads.join(format!("{}.zip", archive.sha256));
        if !dest.is_file() || fs::metadata(&dest).map(|m| m.len()).unwrap_or(0) != archive.size {
            tracing::info!(target: "sigil::diagnostics", "downloading {} ({} MB)", archive.id, archive.size / 1_000_000);
            let mut last_emit: Option<Instant> = None;
            let mut on_bytes = |done: u64| {
                let now = Instant::now();
                if last_emit.is_none_or(|t| now.duration_since(t) >= PROGRESS_INTERVAL) || done == archive.size {
                    last_emit = Some(now);
                    progress(PackProgress {
                        phase: "download",
                        archive: Some(archive.id),
                        archive_index: index,
                        archive_count: count,
                        done_bytes: before + done,
                        total_bytes: total,
                        message: None,
                    });
                }
            };
            download_verified(archive, &archive_url(archive), &dest, &CANCEL, &mut on_bytes).await?;
        }
        before += archive.size;
        zips.push(dest);
    }

    let staging = root.join(format!(".staging-{}-{}", std::process::id(), super::logging::unix_ms_now()));
    let fail = |staging: &Path, e: String| -> Result<(), String> {
        let _ = fs::remove_dir_all(staging);
        Err(e)
    };
    for (index, (archive, zip)) in ARCHIVES.iter().zip(zips).enumerate() {
        if cancelled() {
            return fail(&staging, "cancelled".to_string());
        }
        progress(PackProgress {
            phase: "extract",
            archive: Some(archive.id),
            archive_index: index,
            archive_count: count,
            done_bytes: total,
            total_bytes: total,
            message: None,
        });
        let (st, bin) = (staging.clone(), staging.join("bin"));
        let extracted = tokio::task::spawn_blocking(move || -> Result<(), String> {
            let x = st.join("x");
            extract_dlls(&zip, &x, &CANCEL)?;
            collect_dlls(archive, &x, &bin)?;
            let _ = fs::remove_dir_all(&x);
            Ok(())
        })
        .await
        .map_err(|e| format!("extraction task: {e}"))
        .and_then(|r| r);
        if let Err(e) = extracted {
            return fail(&staging, e);
        }
    }
    let st = staging.clone();
    let manifest = tokio::task::spawn_blocking(move || write_manifest(&st))
        .await
        .map_err(|e| format!("manifest task: {e}"))
        .and_then(|r| r);
    if let Err(e) = manifest {
        return fail(&staging, e);
    }
    progress(PackProgress {
        phase: "finalize",
        archive: None,
        archive_index: count,
        archive_count: count,
        done_bytes: total,
        total_bytes: total,
        message: None,
    });
    verify_installed(&staging).map_err(|e| {
        let _ = fs::remove_dir_all(&staging);
        format!("the staged pack failed verification: {e}")
    })?;
    if dir.exists() {
        let old = root.join(format!(".old-{}", super::logging::unix_ms_now()));
        fs::rename(&dir, &old).map_err(|e| {
            let _ = fs::remove_dir_all(&staging);
            format!("cannot replace {}: {e}", dir.display())
        })?;
        let _ = fs::remove_dir_all(&old);
    }
    fs::rename(&staging, &dir).map_err(|e| {
        let _ = fs::remove_dir_all(&staging);
        format!("cannot move the pack into {}: {e}", dir.display())
    })?;
    // The archives are no longer needed (~1.4 GB).
    let _ = fs::remove_dir_all(&downloads);
    canary::invalidate();
    Ok(())
}

#[cfg(windows)]
mod imp {
    use super::PRELOAD_ORDER;
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;
    use windows::core::PCWSTR;
    use windows::Win32::System::LibraryLoader::{
        AddDllDirectory, LoadLibraryExW, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
    };

    fn wide(p: &Path) -> Vec<u16> {
        p.as_os_str().encode_wide().chain(std::iter::once(0)).collect()
    }

    /// Load every pack DLL by full path in dependency order and keep them loaded for the life of
    /// the process, so the CUDA provider's imports (resolved by module name) bind to the pack.
    pub fn preload_dir(bin: &Path) -> Result<Vec<String>, String> {
        let dir = wide(bin);
        // SAFETY: NUL-terminated path; the returned cookie is intentionally never removed.
        let cookie = unsafe { AddDllDirectory(PCWSTR(dir.as_ptr())) };
        if cookie.is_null() {
            tracing::debug!(target: "sigil::diagnostics", "AddDllDirectory({}) failed", bin.display());
        }
        let mut loaded = Vec::new();
        for (name, required) in PRELOAD_ORDER {
            let path = bin.join(name);
            if !path.is_file() {
                if required {
                    return Err(format!("{name} is missing from {}", bin.display()));
                }
                continue;
            }
            let w = wide(&path);
            // SAFETY: NUL-terminated absolute path. The module is deliberately leaked (never
            // FreeLibrary'd): ONNX Runtime's CUDA provider binds to it by name later.
            match unsafe {
                LoadLibraryExW(PCWSTR(w.as_ptr()), None, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)
            } {
                Ok(_) => loaded.push(name.to_string()),
                Err(e) if required => return Err(format!("cannot load {name}: {e}")),
                Err(e) => tracing::info!(target: "sigil::diagnostics", "optional {name} not loaded: {e}"),
            }
        }
        Ok(loaded)
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn preload_dir(_bin: &std::path::Path) -> Result<Vec<String>, String> {
        Err("the NVIDIA runtime pack is Windows-only".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sigil-nvpack-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn pins_are_well_formed() {
        for a in &ARCHIVES {
            assert_eq!(a.sha256.len(), 64, "{}", a.id);
            assert!(a.sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
            assert!(a.size > 1_000_000);
            let url = archive_url(a);
            assert!(url.starts_with("https://developer.download.nvidia.com/compute/"), "{url}");
            assert!(url.ends_with("-archive.zip"));
            assert!(!a.dlls.is_empty());
        }
        // Every DLL the CUDA provider imports is in the pack and in the preload order.
        for dll in ["cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cufft64_11.dll", "cudnn64_9.dll"] {
            assert!(required_dlls().any(|d| d == dll), "{dll} missing from the pack");
            assert!(PRELOAD_ORDER.iter().any(|(d, req)| *d == dll && *req), "{dll} not preloaded");
        }
        assert_eq!(PRELOAD_ORDER.len(), required_dlls().count());
        assert!(download_bytes() > 1_000_000_000);
    }

    #[test]
    fn content_range_parsing() {
        assert_eq!(content_range_start("bytes 100-199/200"), Some(100));
        assert_eq!(content_range_start("bytes  0-9/10"), Some(0));
        assert_eq!(content_range_start("items 1-2/3"), None);
    }

    fn fake_pack(app_data: &Path) -> PathBuf {
        let staging = pack_root(app_data).join("stage");
        let bin = staging.join("bin");
        fs::create_dir_all(&bin).unwrap();
        for (i, name) in required_dlls().enumerate() {
            fs::write(bin.join(name), vec![b'x'; i + 1]).unwrap();
        }
        write_manifest(&staging).unwrap();
        let dir = install_dir(app_data);
        fs::rename(&staging, &dir).unwrap();
        dir
    }

    #[test]
    fn installed_pack_is_verified_by_manifest_and_sizes() {
        let app_data = temp("verify");
        assert!(!status(Some(&app_data)).installed);
        let dir = fake_pack(&app_data);
        let m = verify_installed(&dir).expect("complete pack verifies");
        assert_eq!(m.files.len(), required_dlls().count());
        assert!(status(Some(&app_data)).installed);
        // A truncated DLL is detected.
        fs::write(dir.join("bin").join("cudnn64_9.dll"), b"").unwrap();
        assert!(verify_installed(&dir).unwrap_err().contains("cudnn64_9.dll"));
        let _ = fs::remove_dir_all(&app_data);
    }

    #[test]
    fn remove_and_sweep() {
        let app_data = temp("remove");
        let dir = fake_pack(&app_data);
        fs::create_dir_all(pack_root(&app_data).join(".staging-1-2/bin")).unwrap();
        let s = remove(&app_data).unwrap();
        assert!(!s.installed && !s.pending_removal);
        assert!(!dir.exists());
        sweep(&app_data);
        assert!(!pack_root(&app_data).join(".staging-1-2").exists());
        let _ = fs::remove_dir_all(&app_data);
    }

    #[test]
    fn collect_finds_nested_dlls() {
        let root = temp("collect");
        let extracted = root.join("x/cudnn-windows-x86_64-9.10.2.21_cuda12-archive/bin");
        fs::create_dir_all(&extracted).unwrap();
        let cudnn = &ARCHIVES[3];
        for d in cudnn.dlls {
            fs::write(extracted.join(d), b"dll").unwrap();
        }
        collect_dlls(cudnn, &root.join("x"), &root.join("bin")).unwrap();
        for d in cudnn.dlls {
            assert!(root.join("bin").join(d).is_file());
        }
        let err = collect_dlls(&ARCHIVES[0], &root.join("x"), &root.join("bin")).unwrap_err();
        assert!(err.contains("cudart64_12.dll"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn sha256_of_file() {
        let root = temp("sha");
        let p = root.join("f");
        fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_file(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// Serve `body` on 127.0.0.1 (warp's file filter honours `Range`) and return the URL.
    async fn serve(body: &[u8], name: &str) -> (String, PathBuf) {
        use warp::Filter;
        let dir = temp(&format!("srv-{name}"));
        fs::write(dir.join("file.zip"), body).unwrap();
        let route = warp::path("file.zip").and(warp::fs::file(dir.join("file.zip")));
        let (addr, server) = warp::serve(route).bind_ephemeral(([127, 0, 0, 1], 0));
        tokio::spawn(server);
        (format!("http://{addr}/file.zip"), dir)
    }

    fn body(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 31 % 251) as u8).collect()
    }

    fn test_archive(data: &[u8]) -> Archive {
        Archive {
            id: "test archive",
            base: "unused",
            rel_path: "unused",
            sha256: Box::leak(hex::encode(Sha256::digest(data)).into_boxed_str()),
            size: data.len() as u64,
            dlls: &["x.dll"],
        }
    }

    /// Uses its own cancel flag: the process-wide `CANCEL` is never touched by tests, so the
    /// extraction test running in parallel cannot be cancelled by this one.
    #[tokio::test]
    async fn downloader_resumes_verifies_and_cancels() {
        let data = body(3 * 1024 * 1024 + 17);
        let (url, dir) = serve(&data, "dl").await;
        let archive = test_archive(&data);
        let cancel = AtomicBool::new(false);

        // Resume: a third of the file is already there; the rest comes from a Range request.
        let dest = dir.join("out.zip");
        fs::write(dest.with_extension("partial"), &data[..data.len() / 3]).unwrap();
        let mut seen = Vec::new();
        download_verified(&archive, &url, &dest, &cancel, &mut |b| seen.push(b))
            .await
            .expect("resumed download");
        assert_eq!(fs::read(&dest).unwrap(), data);
        assert!(!dest.with_extension("partial").exists());
        assert_eq!(seen.first().copied(), Some((data.len() / 3) as u64), "progress starts at the resume point");
        assert_eq!(seen.last().copied(), Some(data.len() as u64));

        // A corrupt partial (right length, wrong bytes) is detected and fetched again.
        let dest2 = dir.join("out2.zip");
        let mut bad = data.clone();
        bad[10] ^= 0xFF;
        fs::write(dest2.with_extension("partial"), &bad[..data.len() / 2]).unwrap();
        download_verified(&archive, &url, &dest2, &cancel, &mut |_| {})
            .await
            .expect("restarted download");
        assert_eq!(fs::read(&dest2).unwrap(), data);

        // Wrong pinned hash: fails after one restart, leaves nothing behind.
        let mut wrong = test_archive(&data);
        wrong.sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
        let dest3 = dir.join("out3.zip");
        let err = download_verified(&wrong, &url, &dest3, &cancel, &mut |_| {}).await.unwrap_err();
        assert!(err.contains("checksum mismatch"), "{err}");
        assert!(!dest3.exists() && !dest3.with_extension("partial").exists());

        // Cancel: the partial file is kept for a later resume.
        let dest4 = dir.join("out4.zip");
        cancel.store(true, Ordering::Release);
        let err = download_verified(&archive, &url, &dest4, &cancel, &mut |_| {}).await.unwrap_err();
        assert_eq!(err, "cancelled");
        assert!(!dest4.exists());
        assert!(!CANCEL.load(Ordering::Acquire), "the process-wide flag is untouched");

        let _ = fs::remove_dir_all(&dir);
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc ^= u32::from(b);
            for _ in 0..8 {
                crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
            }
        }
        !crc
    }

    /// Minimal stored (uncompressed) zip writer for extraction tests.
    fn write_zip(path: &Path, entries: &[(&str, &[u8])]) {
        let mut out = Vec::new();
        let mut central = Vec::new();
        for (name, data) in entries {
            let offset = out.len() as u32;
            let crc = crc32(data);
            let header = |sig: u32, central: bool| {
                let mut h = Vec::new();
                h.extend_from_slice(&sig.to_le_bytes());
                if central {
                    h.extend_from_slice(&20u16.to_le_bytes()); // version made by
                }
                h.extend_from_slice(&20u16.to_le_bytes()); // version needed
                h.extend_from_slice(&0u16.to_le_bytes()); // flags
                h.extend_from_slice(&0u16.to_le_bytes()); // stored
                h.extend_from_slice(&0u16.to_le_bytes()); // time
                h.extend_from_slice(&0x21u16.to_le_bytes()); // date 1980-01-01
                h.extend_from_slice(&crc.to_le_bytes());
                h.extend_from_slice(&(data.len() as u32).to_le_bytes());
                h.extend_from_slice(&(data.len() as u32).to_le_bytes());
                h.extend_from_slice(&(name.len() as u16).to_le_bytes());
                h.extend_from_slice(&0u16.to_le_bytes()); // extra
                if central {
                    h.extend_from_slice(&0u16.to_le_bytes()); // comment
                    h.extend_from_slice(&0u16.to_le_bytes()); // disk
                    h.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
                    h.extend_from_slice(&0u32.to_le_bytes()); // external attrs
                    h.extend_from_slice(&offset.to_le_bytes());
                }
                h.extend_from_slice(name.as_bytes());
                h
            };
            out.extend(header(0x0403_4b50, false));
            out.extend_from_slice(data);
            central.extend(header(0x0201_4b50, true));
        }
        let cd_offset = out.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&(central.len() as u32).to_le_bytes());
        out.extend_from_slice(&cd_offset.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        fs::write(path, out).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn extracts_only_dlls_with_system_tar() {
        let root = temp("tar");
        let zip = root.join("cudart.zip");
        write_zip(
            &zip,
            &[
                ("cuda_cudart-windows-x86_64-12.9.79-archive/bin/cudart64_12.dll", b"MZ-cudart"),
                ("cuda_cudart-windows-x86_64-12.9.79-archive/include/cuda_runtime.h", b"// header"),
                ("cuda_cudart-windows-x86_64-12.9.79-archive/LICENSE", b"license"),
            ],
        );
        let x = root.join("x");
        let never = AtomicBool::new(false);
        extract_dlls(&zip, &x, &never).expect("tar.exe extracts the DLLs");
        collect_dlls(&ARCHIVES[0], &x, &root.join("bin")).unwrap();
        assert_eq!(fs::read(root.join("bin/cudart64_12.dll")).unwrap(), b"MZ-cudart");
        assert!(find_file(&x, "cuda_runtime.h").is_none(), "only *.dll members are extracted");

        // An entry escaping the target directory is never written outside it.
        let evil = root.join("evil.zip");
        write_zip(&evil, &[("../escaped.dll", b"MZ-evil"), ("ok/bin/fine.dll", b"MZ")]);
        let x2 = root.join("x2");
        let _ = extract_dlls(&evil, &x2, &never);
        assert!(!root.join("escaped.dll").exists(), "path traversal must be refused");

        // A set cancel flag stops the extraction and reports it.
        let stop = AtomicBool::new(true);
        assert_eq!(extract_dlls(&zip, &root.join("x3"), &stop).unwrap_err(), "cancelled");
        let _ = fs::remove_dir_all(&root);
    }
}
