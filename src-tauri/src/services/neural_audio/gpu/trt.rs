//! TensorRT on the CUDA rung (P3-C2; the `tensorrt_rtx` accelerator): TensorRT-RTX through its ONNX
//! Runtime plugin EP, or classic TensorRT as an option, with an engine cache and build progress.
//! Plan T9: a first engine build within 60 s, then within 3 s from the cache.
//!
//! # Where TensorRT comes from
//! Nothing TensorRT-related ships in the installer. The runtime is the model-store pack
//! `runtime/tensorrt-rtx` (`vc::assets::RUNTIME_TENSORRT_RTX`, catalog fragment
//! `runtime-trt-rtx.json`, installed by local import while asset hosting is local-only), or, for
//! development, the folder named by [`TRT_DIR_ENV`]. Two routes use it:
//! - **TensorRT-RTX** (preferred): `onnxruntime_providers_nv_tensorrt_rtx.dll`, an ONNX Runtime
//!   plugin EP (`CreateEpFactories`), from the pack (or next to `Sigil.exe` in development builds),
//!   registered once with `register_ep_library` and reached through its EP device, like the CUDA
//!   provider pack (ADR 0007). It supports RTX GPUs of compute capability 8.6, 8.9 and 12.0+.
//! - **Classic TensorRT** (option): `onnxruntime_providers_tensorrt.dll` next to `Sigil.exe` (a
//!   provider-bridge library; it cannot be registered from another folder), with the CUDA EP after it
//!   for nodes TensorRT does not take.
//!
//! A route is usable only when every DLL its provider imports resolves: in the pack, next to
//! `Sigil.exe`, in the NVIDIA runtime pack or in System32. This is decided by reading the providers'
//! PE import tables ([`pe_imports`]), so [`available`] never loads anything; the result is cached
//! until the TensorRT pack or the NVIDIA runtime / CUDA provider packs change (installed, updated or
//! removed while Sigil runs). The classic provider is a provider-bridge library built for one ONNX
//! Runtime release, so a copy from another release is refused like the CUDA provider
//! (`ort_session::provider_version_mismatch`). Before registration the pack's DLLs are loaded by
//! full path in dependency order (later loads by name bind to them). A session request runs the
//! CUDA rung's preflight and canary first (`ort_session::build_session_ext`): TensorRT loads the
//! CUDA provider and the NVIDIA runtime into the process too.
//!
//! # Engine cache
//! Engines are cached per model, weights, shapes, GPU (name and compute capability), driver, ONNX
//! Runtime and pack version under [`engine_cache_dir`]`/<key>/` ([`cache_key`]). TensorRT-RTX writes
//! an EP-context model (`model_ctx.onnx`, the engine embedded) plus its runtime (JIT) cache; a hit
//! loads the context model instead of building. Classic TensorRT keeps its own engine and timing
//! caches in the entry. At most [`CACHE_MAX_ENTRIES`] entries / [`CACHE_MAX_BYTES`] are kept (least
//! recently used first out). A cache entry that fails to load is deleted and rebuilt.
//!
//! # Progress
//! Every build and cache load reports on [`TRT_PROGRESS_EVENT`] ([`TrtProgress`]): `cache_hit`,
//! `building` (twice a second while TensorRT builds; `fraction` is an estimate against the 60 s
//! target, TensorRT reports no progress of its own), `done` or `failed`.
//!
//! # Frozen by P3-S0
//! - [`available`]`() -> bool` (the pack is installed and loadable; cheap after the first call),
//!   [`TRT_PROGRESS_EVENT`] + [`TrtProgress`] (TS mirror: `src/core/services/voice_changer/ipc.ts`),
//!   [`engine_cache_dir`].

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::cuda_driver::CudaDriver;
use crate::services::diagnostics;
use crate::services::neural_audio::ort_session::{
    base_builder, commit, cuda_provider_source, explain_ep_error, provider_dir, provider_version_mismatch, runtime_version,
    CudaProviderSource, GpuExtras, ModelSource, SessionConfigure, VcSessionOptions, EP_CUDA, EP_TRT_CLASSIC, EP_TRT_RTX,
};
use crate::services::neural_audio::vc::assets::RUNTIME_TENSORRT_RTX;

/// Event with a TensorRT engine build's progress (payload [`TrtProgress`]).
pub const TRT_PROGRESS_EVENT: &str = "voice_changer:trt_progress";
/// Development override: a folder with the TensorRT runtime DLLs (and optionally the TensorRT-RTX
/// provider), used instead of the model-store pack.
pub const TRT_DIR_ENV: &str = "SIGIL_TRT_DIR";
/// The TensorRT-RTX plugin EP library.
pub const RTX_PROVIDER_DLL: &str = "onnxruntime_providers_nv_tensorrt_rtx.dll";
/// The classic TensorRT provider library (next to `Sigil.exe` only).
pub const CLASSIC_PROVIDER_DLL: &str = "onnxruntime_providers_tensorrt.dll";
/// EP name of the TensorRT-RTX plugin's devices.
const RTX_EP_NAME: &str = "NvTensorRTRTXExecutionProvider";
/// Registration name of the TensorRT-RTX library.
const RTX_LIBRARY_NAME: &str = "NvTensorRtRtx";
/// Plan T9: the first engine build takes at most this long (the progress estimate's scale).
const EXPECTED_BUILD_SECS: f64 = 60.0;
/// Cache entries kept (least recently used out first).
pub const CACHE_MAX_ENTRIES: usize = 16;
/// Cache bytes kept.
pub const CACHE_MAX_BYTES: u64 = 8 << 30;
/// Folder depth searched for DLLs in the pack.
const SEARCH_DEPTH: usize = 4;
const CTX_MODEL: &str = "model_ctx.onnx";
const RUNTIME_CACHE_DIR: &str = "runtime-cache";
const META_FILE: &str = "sigil-trt.json";

/// Whether TensorRT runs in this build: Windows, an NVIDIA adapter, and a route whose provider
/// imports all resolve (see the module docs). Cheap after the first call per pack location.
pub fn available() -> bool {
    cfg!(windows) && super::cuda_stream::nvidia_adapter_present() && discover().usable()
}

/// Progress of one engine build (snake_case, like the rest of the voice-changer plugin).
#[derive(Clone, Debug, Default, Serialize)]
pub struct TrtProgress {
    /// Session component (`vc.enc`, `vc.flowdec`, ...).
    pub component: String,
    /// `cache_hit`, `building`, `done` or `failed`.
    pub stage: String,
    /// 0..1 of the build (0 while unknown).
    pub fraction: f64,
    pub message: String,
    pub elapsed_ms: u64,
}

/// Engine cache folder: `<app_data>/voice-changer/trt-cache/`.
pub fn engine_cache_dir(app_data: &Path) -> PathBuf {
    app_data.join("voice-changer").join("trt-cache")
}

/// What a session plan asks of TensorRT.
#[derive(Clone, Debug)]
pub struct TrtRequest {
    /// Digest of everything that defines the engine on the plan's side (model bytes, weights,
    /// shapes, I/O), see `plan`.
    pub model_key: String,
    /// Classic TensorRT: build fp16 kernels (the graph's own precision otherwise).
    pub fp16: bool,
}

// ---------------------------------------------------------------------------------------------
// PE import tables
// ---------------------------------------------------------------------------------------------

/// The DLL names a PE image imports (normal and delay-load import tables, lower-case, in table
/// order). Reads only the headers and the import tables, never the whole file.
pub fn pe_imports(path: &Path) -> Result<Vec<String>, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    pe_imports_from(file).map_err(|e| format!("{}: {e}", path.display()))
}

fn pe_imports_from<R: Read + Seek>(mut r: R) -> Result<Vec<String>, String> {
    fn read_at<R: Read + Seek>(r: &mut R, off: u64, buf: &mut [u8]) -> Result<(), String> {
        r.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
        r.read_exact(buf).map_err(|_| "truncated PE image".to_string())
    }
    fn u16_at(b: &[u8], o: usize) -> Result<u16, String> {
        b.get(o..o + 2).map(|s| u16::from_le_bytes([s[0], s[1]])).ok_or_else(|| "truncated PE header".to_string())
    }
    fn u32_at(b: &[u8], o: usize) -> Result<u32, String> {
        b.get(o..o + 4)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
            .ok_or_else(|| "truncated PE header".to_string())
    }
    const HEADER: usize = 4096;
    let mut head = vec![0u8; HEADER];
    r.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let n = r.read(&mut head).map_err(|e| e.to_string())?;
    head.truncate(n);
    if head.get(0..2) != Some(b"MZ") {
        return Err("not a PE image (no MZ header)".into());
    }
    let pe = u32_at(&head, 0x3c)? as usize;
    if head.get(pe..pe + 4) != Some(b"PE\0\0") {
        return Err("not a PE image (no PE signature)".into());
    }
    let nsections = u16_at(&head, pe + 6)? as usize;
    let opt_size = u16_at(&head, pe + 20)? as usize;
    let opt = pe + 24;
    let (dirs, count_at) = match u16_at(&head, opt)? {
        0x20b => (opt + 112, opt + 108),
        0x10b => (opt + 96, opt + 92),
        m => return Err(format!("unknown optional header magic {m:#x}")),
    };
    let ndirs = u32_at(&head, count_at)? as usize;
    let dir = |i: usize| -> Result<(u32, u32), String> {
        if i >= ndirs {
            return Ok((0, 0));
        }
        Ok((u32_at(&head, dirs + i * 8)?, u32_at(&head, dirs + i * 8 + 4)?))
    };
    let (import_rva, _) = dir(1)?;
    let (delay_rva, _) = dir(13)?;
    let mut sections = Vec::with_capacity(nsections.min(96));
    let table = opt + opt_size;
    for i in 0..nsections.min(96) {
        let s = table + i * 40;
        let vsize = u32_at(&head, s + 8)?;
        let va = u32_at(&head, s + 12)?;
        let raw_size = u32_at(&head, s + 16)?;
        let raw_ptr = u32_at(&head, s + 20)?;
        sections.push((va, vsize.max(raw_size), raw_ptr));
    }
    let to_off = |rva: u32| -> Option<u64> {
        sections
            .iter()
            .find(|&&(va, size, _)| rva >= va && rva - va < size)
            .map(|&(va, _, raw)| u64::from(raw) + u64::from(rva - va))
    };
    let mut names = Vec::new();
    let read_name = |r: &mut R, rva: u32| -> Result<Option<String>, String> {
        let Some(off) = to_off(rva) else { return Ok(None) };
        let mut buf = [0u8; 260];
        r.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
        let n = r.read(&mut buf).map_err(|e| e.to_string())?;
        let end = buf[..n].iter().position(|&b| b == 0).unwrap_or(n);
        let name = String::from_utf8_lossy(&buf[..end]).to_ascii_lowercase();
        Ok((!name.is_empty()).then_some(name))
    };
    // IMAGE_IMPORT_DESCRIPTOR: 20 bytes, Name at +12; the table ends with an all-zero entry.
    if let Some(mut off) = (import_rva != 0).then(|| to_off(import_rva)).flatten() {
        for _ in 0..4096 {
            let mut d = [0u8; 20];
            read_at(&mut r, off, &mut d)?;
            let name_rva = u32_at(&d, 12)?;
            if name_rva == 0 {
                break;
            }
            if let Some(n) = read_name(&mut r, name_rva)? {
                names.push(n);
            }
            off += 20;
        }
    }
    // ImgDelayDescr: 32 bytes, DllNameRVA at +4; ends with a zero name.
    if let Some(mut off) = (delay_rva != 0).then(|| to_off(delay_rva)).flatten() {
        for _ in 0..4096 {
            let mut d = [0u8; 32];
            read_at(&mut r, off, &mut d)?;
            let name_rva = u32_at(&d, 4)?;
            if name_rva == 0 {
                break;
            }
            if let Some(n) = read_name(&mut r, name_rva)? {
                if !names.contains(&n) {
                    names.push(n);
                }
            }
            off += 32;
        }
    }
    Ok(names)
}

/// API-set contracts resolve inside the loader, never from a file.
fn is_api_set(dll: &str) -> bool {
    dll.starts_with("api-ms-win-") || dll.starts_with("ext-ms-")
}

fn system32() -> PathBuf {
    std::env::var_os("SystemRoot")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Windows"))
        .join("System32")
}

/// Every `*.dll` below `root` (at most `depth` levels), keyed by lower-case file name; the
/// shallowest wins.
fn find_dlls(root: &Path, depth: usize) -> BTreeMap<String, PathBuf> {
    fn walk(dir: &Path, depth: usize, max: usize, out: &mut BTreeMap<String, (usize, PathBuf)>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        let mut subdirs = Vec::new();
        for e in entries.flatten() {
            let Ok(ft) = e.file_type() else { continue };
            let p = e.path();
            if ft.is_dir() {
                subdirs.push(p);
            } else if ft.is_file() {
                let name = e.file_name().to_string_lossy().to_ascii_lowercase();
                if name.ends_with(".dll") && out.get(&name).is_none_or(|(d, q)| depth < *d || (depth == *d && p < *q)) {
                    out.insert(name, (depth, p));
                }
            }
        }
        if depth < max {
            subdirs.sort();
            for d in subdirs {
                walk(&d, depth + 1, max, out);
            }
        }
    }
    let mut found = BTreeMap::new();
    walk(root, 0, depth, &mut found);
    found.into_iter().map(|(k, (_, p))| (k, p)).collect()
}

/// Where imports may resolve besides the pack.
struct Resolver {
    pack: BTreeMap<String, PathBuf>,
    nvidia: BTreeMap<String, PathBuf>,
    exe_dir: Option<PathBuf>,
    system32: PathBuf,
}

impl Resolver {
    fn new(pack: BTreeMap<String, PathBuf>) -> Resolver {
        Resolver {
            pack,
            nvidia: crate::services::native_runtime::nvidia::store_dll_index().unwrap_or_default(),
            exe_dir: provider_dir(),
            system32: system32(),
        }
    }

    /// Pack DLLs to load before `provider` (dependencies first) and the imports nothing provides.
    fn plan(&self, provider: &Path) -> (Vec<PathBuf>, Vec<String>) {
        let mut order = Vec::new();
        let mut missing = BTreeSet::new();
        let mut visiting = BTreeSet::new();
        let mut done = BTreeSet::new();
        self.visit_imports(provider, &mut order, &mut missing, &mut visiting, &mut done);
        (order, missing.into_iter().collect())
    }

    fn visit_imports(
        &self,
        image: &Path,
        order: &mut Vec<PathBuf>,
        missing: &mut BTreeSet<String>,
        visiting: &mut BTreeSet<String>,
        done: &mut BTreeSet<String>,
    ) {
        let imports = match pe_imports(image) {
            Ok(i) => i,
            Err(e) => {
                missing.insert(format!("(unreadable {}: {e})", image.display()));
                return;
            }
        };
        for dll in imports {
            if is_api_set(&dll) || done.contains(&dll) || visiting.contains(&dll) {
                continue;
            }
            if let Some(path) = self.pack.get(&dll) {
                visiting.insert(dll.clone());
                self.visit_imports(path, order, missing, visiting, done);
                visiting.remove(&dll);
                done.insert(dll);
                order.push(path.clone());
            } else if self.nvidia.contains_key(&dll)
                || self.exe_dir.as_ref().is_some_and(|d| d.join(&dll).is_file())
                || self.system32.join(&dll).is_file()
            {
                done.insert(dll);
            } else {
                missing.insert(dll);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------------------------

/// A provider library whose imports resolve, with the pack DLLs to load before it.
#[derive(Clone, Debug)]
struct Route {
    provider: PathBuf,
    preload: Vec<PathBuf>,
}

/// The TensorRT runtime found on this PC.
#[derive(Clone, Debug)]
struct Discovery {
    /// Pack folder (store install or [`TRT_DIR_ENV`]).
    root: Option<PathBuf>,
    /// Pack version (its store version folder, or `dev`).
    version: String,
    rtx: Result<Route, String>,
    classic: Result<Route, String>,
}

impl Discovery {
    fn usable(&self) -> bool {
        self.rtx.is_ok() || self.classic.is_ok()
    }

    fn why_not(&self) -> String {
        match (&self.rtx, &self.classic) {
            (Err(a), Err(b)) => format!("TensorRT-RTX: {a}; classic TensorRT: {b}"),
            _ => String::new(),
        }
    }
}

fn pack_root() -> Option<(PathBuf, String)> {
    if let Some(dir) = std::env::var_os(TRT_DIR_ENV).map(PathBuf::from).filter(|d| d.is_dir()) {
        return Some((dir, "dev".to_string()));
    }
    let root = crate::services::model_store::installed_path(RUNTIME_TENSORRT_RTX)?;
    // runtimes/tensorrt-rtx/<version>/<variant>/
    let version = root
        .parent()
        .and_then(|p| p.file_name())
        .map(|v| v.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".to_string());
    Some((root, version))
}

fn route_for(provider: Option<PathBuf>, resolver: &Resolver, missing_provider: &str) -> Result<Route, String> {
    let provider = provider.ok_or_else(|| missing_provider.to_string())?;
    let (preload, missing) = resolver.plan(&provider);
    if !missing.is_empty() {
        return Err(format!("{} needs {} (not in the TensorRT runtime pack)", provider.display(), missing.join(", ")));
    }
    Ok(Route { provider, preload })
}

fn discover_now() -> Discovery {
    let root = pack_root();
    let pack = root.as_ref().map(|(r, _)| find_dlls(r, SEARCH_DEPTH)).unwrap_or_default();
    let exe_dir = provider_dir();
    let resolver = Resolver::new(pack);
    let no_pack = format!("the TensorRT runtime pack ({RUNTIME_TENSORRT_RTX}) is not installed");
    let rtx_provider = resolver
        .pack
        .get(RTX_PROVIDER_DLL)
        .cloned()
        .or_else(|| exe_dir.as_ref().map(|d| d.join(RTX_PROVIDER_DLL)).filter(|p| p.is_file()));
    let classic_provider = exe_dir.as_ref().map(|d| d.join(CLASSIC_PROVIDER_DLL)).filter(|p| p.is_file());
    let (rtx, classic) = if root.is_none() {
        (Err(no_pack.clone()), Err(no_pack))
    } else {
        (
            route_for(rtx_provider, &resolver, &format!("{RTX_PROVIDER_DLL} is not in the pack")),
            // A provider-bridge library from another ONNX Runtime release must never be loaded (no
            // ABI check there; loading it can crash the process).
            route_for(classic_provider, &resolver, &format!("{CLASSIC_PROVIDER_DLL} is not installed next to Sigil.exe")).and_then(
                |route| match provider_version_mismatch(&route.provider) {
                    Some(why) => Err(why),
                    None => Ok(route),
                },
            ),
        )
    };
    Discovery {
        version: root.as_ref().map(|(_, v)| v.clone()).unwrap_or_default(),
        root: root.map(|(r, _)| r),
        rtx,
        classic,
    }
}

/// What the routes depend on besides the files next to `Sigil.exe` and in System32: the TensorRT
/// pack's location and the NVIDIA runtime / CUDA provider packs (`nvidia::fingerprint`, the key the
/// CUDA canary's cache follows too).
fn discovery_key() -> (Option<PathBuf>, String) {
    (pack_root().map(|(r, _)| r), crate::services::native_runtime::nvidia::fingerprint())
}

/// `cache`'s value when it was computed for `key`, else `compute()` (stored for `key`).
fn cached_for<K: PartialEq, V: Clone>(cache: &Mutex<Option<(K, V)>>, key: K, compute: impl FnOnce() -> V) -> V {
    let mut cache = cache.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((k, v)) = cache.as_ref() {
        if *k == key {
            return v.clone();
        }
    }
    let v = compute();
    *cache = Some((key, v.clone()));
    v
}

/// The routes, cached per [`discovery_key`] (a pack installed, updated or removed later is picked
/// up without a restart).
fn discover() -> Discovery {
    type Key = (Option<PathBuf>, String);
    static CACHE: Mutex<Option<(Key, Discovery)>> = Mutex::new(None);
    cached_for(&CACHE, discovery_key(), discover_now)
}

/// Load `route`'s pack DLLs by full path (kept for the life of the process, so later loads by name
/// bind to them).
fn preload(route: &Route) -> Result<(), String> {
    static LOADED: Mutex<BTreeSet<PathBuf>> = Mutex::new(BTreeSet::new());
    let mut loaded = LOADED.lock().unwrap_or_else(PoisonError::into_inner);
    for path in &route.preload {
        if loaded.contains(path) {
            continue;
        }
        #[cfg(windows)]
        {
            use libloading::os::windows::{Library, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR};
            let abs = std::path::absolute(path).unwrap_or_else(|_| path.clone());
            // SAFETY: DLLs of the installed TensorRT runtime pack (model-store verified install or
            // the developer's own folder), loaded by full path in dependency order.
            let lib = unsafe { Library::load_with_flags(&abs, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) }
                .map_err(|e| format!("{} could not be loaded ({e})", abs.display()))?;
            // Deliberately never unloaded: later imports bind to it by name.
            std::mem::forget(lib);
        }
        loaded.insert(path.clone());
    }
    Ok(())
}

/// The TensorRT-RTX library registration (kept for the life of the process).
fn ensure_rtx_registered(route: &Route) -> Result<(), String> {
    static REGISTERED: Mutex<Option<(PathBuf, ort::ep::ExecutionProviderLibrary)>> = Mutex::new(None);
    let mut reg = REGISTERED.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((path, _)) = reg.as_ref() {
        if *path == route.provider {
            return Ok(());
        }
        return Err(format!("another TensorRT-RTX provider ({}) is already registered; restart Sigil", path.display()));
    }
    let _ = crate::services::native_runtime::nvidia::ensure_preloaded();
    preload(route)?;
    crate::services::neural_audio::ort_session::ensure_ort_environment();
    let env = ort::environment::Environment::current().map_err(|e| format!("ONNX Runtime environment: {e}"))?;
    let started = Instant::now();
    let library = env
        .register_ep_library(RTX_LIBRARY_NAME, &route.provider)
        .map_err(|e| explain_ep_error(EP_CUDA, &e.to_string()))?;
    tracing::info!(
        target: "sigil::gpu",
        "TensorRT-RTX provider registered from {} in {} ms",
        route.provider.display(),
        started.elapsed().as_millis()
    );
    *reg = Some((route.provider.clone(), library));
    Ok(())
}

/// Compute capabilities the TensorRT-RTX EP accepts (its own check: 8.6, 8.9 and 12.0+).
pub fn rtx_supports(major: i32, minor: i32) -> bool {
    matches!((major, minor), (8, 6) | (8, 9)) || major >= 12
}

// ---------------------------------------------------------------------------------------------
// Engine cache
// ---------------------------------------------------------------------------------------------

/// The cache key of one engine (32 hex chars): the plan's model key plus everything on this PC
/// that invalidates a TensorRT engine.
pub fn cache_key(route: &str, model_key: &str, gpu: &str, cc: (i32, i32), driver: &str, pack_version: &str, provider_stamp: &str) -> String {
    let mut h = Sha256::new();
    for part in ["sigil-trt-v1", route, model_key, gpu, &format!("{}.{}", cc.0, cc.1), driver, pack_version, provider_stamp, runtime_version()] {
        h.update(part.as_bytes());
        h.update([0u8]);
    }
    hex::encode(&h.finalize()[..16])
}

/// Size and modification time of a file (a cheap identity of the provider build).
fn file_stamp(path: &Path) -> String {
    std::fs::metadata(path)
        .map(|m| {
            let t = m.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
            format!("{}:{t}", m.len())
        })
        .unwrap_or_default()
}

fn now_unix_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

#[derive(Serialize, serde::Deserialize, Default)]
struct EntryMeta {
    route: String,
    component: String,
    gpu: String,
    driver: String,
    pack_version: String,
    ort: String,
    build_ms: u64,
    created_unix_ms: u64,
    last_used_unix_ms: u64,
}

/// One engine cache entry folder.
struct CacheEntry {
    dir: PathBuf,
}

impl CacheEntry {
    fn ctx_model(&self) -> PathBuf {
        self.dir.join(CTX_MODEL)
    }

    fn runtime_cache(&self) -> PathBuf {
        self.dir.join(RUNTIME_CACHE_DIR)
    }

    fn meta_path(&self) -> PathBuf {
        self.dir.join(META_FILE)
    }

    fn read_meta(&self) -> Option<EntryMeta> {
        serde_json::from_slice(&std::fs::read(self.meta_path()).ok()?).ok()
    }

    fn write_meta(&self, meta: &EntryMeta) {
        let tmp = self.dir.join(format!("{META_FILE}.tmp"));
        if let Ok(bytes) = serde_json::to_vec_pretty(meta) {
            if std::fs::write(&tmp, bytes).is_ok() {
                let _ = std::fs::rename(&tmp, self.meta_path());
            }
        }
    }

    /// Mark the entry used now (the eviction order).
    fn touch(&self) {
        if let Some(mut meta) = self.read_meta() {
            meta.last_used_unix_ms = now_unix_ms();
            self.write_meta(&meta);
        }
    }

    fn remove(&self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn dir_size(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else { return 0 };
    entries
        .flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_size(&e.path()),
            Ok(_) => e.metadata().map(|m| m.len()).unwrap_or(0),
            Err(_) => 0,
        })
        .sum()
}

/// Remove the least recently used entries beyond [`CACHE_MAX_ENTRIES`] / [`CACHE_MAX_BYTES`]
/// (`keep` is never removed). Returns the removed folders.
pub(crate) fn evict(root: &Path, keep: &Path, max_entries: usize, max_bytes: u64) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(root) else { return Vec::new() };
    let mut list: Vec<(u64, u64, PathBuf)> = entries
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .map(|e| {
            let dir = e.path();
            let used = CacheEntry { dir: dir.clone() }
                .read_meta()
                .map(|m| m.last_used_unix_ms.max(m.created_unix_ms))
                .unwrap_or(0);
            (used, dir_size(&dir), dir)
        })
        .collect();
    // Newest first; the rest is dropped once a limit is reached.
    list.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.2.cmp(&b.2)));
    let (mut kept, mut bytes, mut removed) = (0usize, 0u64, Vec::new());
    for (_, size, dir) in list {
        let is_keep = dir == keep;
        if is_keep || (kept < max_entries && bytes + size <= max_bytes) {
            kept += 1;
            bytes += size;
        } else {
            let _ = std::fs::remove_dir_all(&dir);
            removed.push(dir);
        }
    }
    removed
}

// ---------------------------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------------------------

fn emit(component: &str, stage: &str, fraction: f64, message: String, started: Instant) {
    let p = TrtProgress {
        component: component.to_string(),
        stage: stage.to_string(),
        fraction: fraction.clamp(0.0, 1.0),
        message,
        elapsed_ms: started.elapsed().as_millis() as u64,
    };
    diagnostics::emit(TRT_PROGRESS_EVENT, &p);
}

/// The estimated build fraction after `elapsed_s` (never 1 before the build ends).
pub fn estimated_fraction(elapsed_s: f64) -> f64 {
    (elapsed_s / EXPECTED_BUILD_SECS).clamp(0.0, 0.95)
}

/// Emits `building` twice a second until dropped.
struct Heartbeat {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Heartbeat {
    fn start(component: &str, route_label: &'static str, started: Instant) -> Heartbeat {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let component = component.to_string();
        let thread = std::thread::Builder::new()
            .name("sigil-trt-progress".into())
            .spawn(move || {
                while !flag.load(Ordering::Relaxed) {
                    let s = started.elapsed().as_secs_f64();
                    emit(
                        &component,
                        "building",
                        estimated_fraction(s),
                        format!("Building the {route_label} engine (first use of this voice on this GPU)"),
                        started,
                    );
                    for _ in 0..10 {
                        if flag.load(Ordering::Relaxed) {
                            return;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                }
            })
            .ok();
        Heartbeat { stop, thread }
    }
}

impl Drop for Heartbeat {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------------------------

/// This PC's identity for the cache key: GPU name, compute capability and driver.
fn gpu_identity(device: i32) -> Result<(String, (i32, i32), String), String> {
    let driver = CudaDriver::get()?;
    let cc = driver.compute_capability(device)?;
    let name = driver.device_name(device)?;
    let adapters = diagnostics::gpu_mem::adapters();
    let nvidia = diagnostics::gpu_mem::nvidia_adapter_indices(&adapters);
    let display_driver = nvidia
        .get(device.max(0) as usize)
        .and_then(|&i| adapters.iter().find(|a| a.index == i))
        .and_then(|a| a.nvidia_driver.clone())
        .unwrap_or_default();
    Ok((name, cc, format!("{display_driver}/{}", driver.driver_version())))
}

/// Build a session on TensorRT (TensorRT-RTX first, then classic TensorRT) for CUDA device
/// `device`, from the engine cache when possible. Returns the session and the EP id it runs on
/// (`tensorrt_rtx` / `tensorrt`); `Err` says why neither route ran (the ladder then goes on with
/// CUDA).
pub(crate) fn build_session(
    component: &str,
    device: i32,
    opts: &VcSessionOptions,
    source: ModelSource<'_>,
    extras: &GpuExtras,
    request: &TrtRequest,
    configure: &SessionConfigure<'_>,
) -> Result<(ort::session::Session, &'static str), String> {
    let found = discover();
    if !found.usable() {
        return Err(found.why_not());
    }
    let (gpu, cc, driver) = gpu_identity(device)?;
    let app_data = diagnostics::app_data_dir().ok_or_else(|| "the app data folder is unknown (engine cache)".to_string())?;
    let cache_root = engine_cache_dir(&app_data);
    let mut reasons = Vec::new();
    match &found.rtx {
        Ok(route) if rtx_supports(cc.0, cc.1) => {
            let key = cache_key("rtx", &request.model_key, &gpu, cc, &driver, &found.version, &file_stamp(&route.provider));
            let entry = CacheEntry { dir: cache_root.join(&key) };
            match build_rtx(component, device, opts, source, extras, route, &entry, configure) {
                Ok(s) => {
                    finish_entry(&entry, &cache_root, "tensorrt_rtx", component, &gpu, &driver, &found.version);
                    return Ok((s, EP_TRT_RTX));
                }
                Err(e) => reasons.push(format!("TensorRT-RTX: {e}")),
            }
        }
        Ok(_) => reasons.push(format!("TensorRT-RTX: {gpu} (compute capability {}.{}) is not supported; it needs an RTX 30 (8.6), 40 (8.9) or 50 (12.0) series GPU", cc.0, cc.1)),
        Err(e) => reasons.push(format!("TensorRT-RTX: {e}")),
    }
    match &found.classic {
        Ok(route) => {
            let key = cache_key("classic", &request.model_key, &gpu, cc, &driver, &found.version, &file_stamp(&route.provider));
            let entry = CacheEntry { dir: cache_root.join(&key) };
            match build_classic(component, device, opts, source, extras, request, route, &entry, configure) {
                Ok(s) => {
                    finish_entry(&entry, &cache_root, "tensorrt", component, &gpu, &driver, &found.version);
                    return Ok((s, EP_TRT_CLASSIC));
                }
                Err(e) => reasons.push(format!("classic TensorRT: {e}")),
            }
        }
        Err(e) => reasons.push(format!("classic TensorRT: {e}")),
    }
    Err(reasons.join("; "))
}

/// Record a used entry (its metadata) and trim the cache.
fn finish_entry(entry: &CacheEntry, root: &Path, route: &str, component: &str, gpu: &str, driver: &str, version: &str) {
    if !entry.dir.is_dir() {
        return;
    }
    let mut meta = entry.read_meta().unwrap_or_else(|| EntryMeta {
        route: route.to_string(),
        component: component.to_string(),
        gpu: gpu.to_string(),
        driver: driver.to_string(),
        pack_version: version.to_string(),
        ort: runtime_version().to_string(),
        created_unix_ms: now_unix_ms(),
        ..EntryMeta::default()
    });
    meta.last_used_unix_ms = now_unix_ms();
    entry.write_meta(&meta);
    for removed in evict(root, &entry.dir, CACHE_MAX_ENTRIES, CACHE_MAX_BYTES) {
        tracing::info!(target: "sigil::gpu", "TensorRT engine cache: removed {}", removed.display());
    }
}

fn rtx_device_options(device: i32, entry: &CacheEntry, extras: &GpuExtras) -> Vec<(String, String)> {
    const P: &str = "NvTensorRTRTXExecutionProvider.";
    let mut o = vec![
        (format!("{P}device_id"), device.max(0).to_string()),
        (format!("{P}nv_runtime_cache_path"), entry.runtime_cache().to_string_lossy().into_owned()),
    ];
    if let Some(stream) = extras.cuda_stream {
        o.push((format!("{P}has_user_compute_stream"), "1".to_string()));
        o.push((format!("{P}user_compute_stream"), stream.to_string()));
    }
    o
}

#[allow(clippy::too_many_arguments)]
fn build_rtx(
    component: &str,
    device: i32,
    opts: &VcSessionOptions,
    source: ModelSource<'_>,
    extras: &GpuExtras,
    route: &Route,
    entry: &CacheEntry,
    configure: &SessionConfigure<'_>,
) -> Result<ort::session::Session, String> {
    ensure_rtx_registered(route)?;
    let env = ort::environment::Environment::current().map_err(|e| format!("ONNX Runtime environment: {e}"))?;
    let options = rtx_device_options(device, entry, extras);
    let started = Instant::now();
    std::fs::create_dir_all(entry.runtime_cache()).map_err(|e| format!("engine cache {}: {e}", entry.dir.display()))?;
    let device_of = || {
        let devices: Vec<_> = env.devices().filter(|d| d.ep().ok() == Some(RTX_EP_NAME)).collect();
        let n = devices.len();
        devices
            .into_iter()
            .nth(device.max(0) as usize)
            .ok_or_else(|| format!("TensorRT-RTX lists {n} GPU(s); CUDA device {device} is not one of them"))
    };
    if entry.ctx_model().is_file() {
        emit(component, "cache_hit", 0.0, "Loading the cached TensorRT-RTX engine".into(), started);
        let load = base_builder(EP_CUDA, opts)
            .and_then(|b| b.with_devices([device_of()?], Some(&options)).map_err(|e| e.to_string()))
            .and_then(|b| configure(b, EP_TRT_RTX))
            .and_then(|b| commit(b, ModelSource::File(&entry.ctx_model())));
        match load {
            Ok(session) => {
                entry.touch();
                emit(component, "done", 1.0, format!("TensorRT-RTX engine loaded from the cache in {} ms", started.elapsed().as_millis()), started);
                return Ok(session);
            }
            Err(e) => {
                tracing::warn!(target: "sigil::gpu", "TensorRT-RTX cache entry {} failed to load ({e}); rebuilding", entry.dir.display());
                entry.remove();
                std::fs::create_dir_all(entry.runtime_cache()).map_err(|e| format!("engine cache {}: {e}", entry.dir.display()))?;
            }
        }
    }
    let heartbeat = Heartbeat::start(component, "TensorRT-RTX", started);
    let ctx = entry.ctx_model();
    let built = base_builder(EP_CUDA, opts)
        .and_then(|b| b.with_config_entry("ep.context_enable", "1").map_err(|e| e.to_string()))
        .and_then(|b| b.with_config_entry("ep.context_file_path", ctx.to_string_lossy()).map_err(|e| e.to_string()))
        .and_then(|b| b.with_config_entry("ep.context_embed_mode", "1").map_err(|e| e.to_string()))
        .and_then(|b| b.with_devices([device_of()?], Some(&options)).map_err(|e| e.to_string()))
        .and_then(|b| configure(b, EP_TRT_RTX))
        .and_then(|b| commit(b, source));
    drop(heartbeat);
    match built {
        Ok(session) => {
            let ms = started.elapsed().as_millis() as u64;
            if let Some(mut meta) = entry.read_meta() {
                meta.build_ms = ms;
                entry.write_meta(&meta);
            } else {
                entry.write_meta(&EntryMeta { build_ms: ms, created_unix_ms: now_unix_ms(), ..EntryMeta::default() });
            }
            let cached = if ctx.is_file() { "; cached for next time" } else { "; the engine could not be cached" };
            emit(component, "done", 1.0, format!("TensorRT-RTX engine built in {:.1} s{cached}", ms as f64 / 1000.0), started);
            tracing::info!(target: "sigil::gpu", "TensorRT-RTX engine for [{component}] built in {ms} ms{cached}");
            Ok(session)
        }
        Err(e) => {
            emit(component, "failed", 0.0, format!("TensorRT-RTX engine build failed: {e}"), started);
            entry.remove();
            Err(e)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_classic(
    component: &str,
    device: i32,
    opts: &VcSessionOptions,
    source: ModelSource<'_>,
    extras: &GpuExtras,
    request: &TrtRequest,
    route: &Route,
    entry: &CacheEntry,
    configure: &SessionConfigure<'_>,
) -> Result<ort::session::Session, String> {
    let _ = crate::services::native_runtime::nvidia::ensure_preloaded();
    preload(route)?;
    std::fs::create_dir_all(&entry.dir).map_err(|e| format!("engine cache {}: {e}", entry.dir.display()))?;
    let started = Instant::now();
    let dir = entry.dir.to_string_lossy().into_owned();
    // A cached engine is recognised by its files (classic TensorRT names them itself).
    let hit = std::fs::read_dir(&entry.dir)
        .map(|it| it.flatten().any(|e| e.file_name().to_string_lossy().ends_with(".engine")))
        .unwrap_or(false);
    let mut trt = ort::ep::TensorRT::default()
        .with_device_id(device.max(0))
        .with_engine_cache(true)
        .with_engine_cache_path(&dir)
        .with_timing_cache(true)
        .with_timing_cache_path(&dir)
        .with_fp16(request.fp16);
    if let Some(stream) = extras.cuda_stream {
        // SAFETY: the plan owns the stream and drops it after the session.
        trt = unsafe { trt.with_compute_stream(stream as *mut ()) };
    }
    let mut providers = vec![trt.build().error_on_failure()];
    // Nodes TensorRT does not take run on the CUDA EP when the classic CUDA provider is installed
    // (the registered CUDA provider pack cannot be mixed with a provider-bridge TensorRT).
    if matches!(cuda_provider_source(), Ok(CudaProviderSource::ExeDir(_))) {
        providers.push(ort::ep::CUDA::default().with_device_id(device.max(0)).build().error_on_failure());
    }
    let heartbeat = (!hit).then(|| Heartbeat::start(component, "TensorRT", started));
    if hit {
        emit(component, "cache_hit", 0.0, "Loading the cached TensorRT engine".into(), started);
    }
    let built = base_builder(EP_CUDA, opts)
        .and_then(|b| b.with_execution_providers(providers).map_err(|e| explain_ep_error(EP_CUDA, &e.to_string())))
        .and_then(|b| configure(b, EP_TRT_CLASSIC))
        .and_then(|b| commit(b, source));
    drop(heartbeat);
    match built {
        Ok(session) => {
            let ms = started.elapsed().as_millis() as u64;
            let what = if hit { "loaded from the cache" } else { "built" };
            emit(component, "done", 1.0, format!("TensorRT engine {what} in {:.1} s", ms as f64 / 1000.0), started);
            Ok(session)
        }
        Err(e) => {
            emit(component, "failed", 0.0, format!("TensorRT engine build failed: {e}"), started);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pe_imports_of_system_dlls_and_garbage() {
        #[cfg(windows)]
        {
            let k32 = system32().join("kernel32.dll");
            let imports = pe_imports(&k32).expect("kernel32 parses");
            assert!(!imports.is_empty());
            assert!(imports.iter().all(|d| d.ends_with(".dll")), "{imports:?}");
            assert!(imports.iter().any(|d| d == "ntdll.dll" || is_api_set(d)), "{imports:?}");
        }
        assert!(pe_imports_from(std::io::Cursor::new(b"not a dll at all".to_vec())).is_err());
        let mut mz = vec![0u8; 256];
        mz[0] = b'M';
        mz[1] = b'Z';
        mz[0x3c] = 0x80;
        assert!(pe_imports_from(std::io::Cursor::new(mz)).unwrap_err().contains("PE signature"));
        assert!(pe_imports(Path::new("Z:/missing/provider.dll")).is_err());
    }

    /// The providers ort-sys ships (when present next to the test binary or in `target/<profile>`):
    /// both TensorRT providers import the provider bridge, and the TensorRT-RTX one is a plugin EP.
    #[test]
    fn provider_import_tables_are_read() {
        let Some(exe_dir) = provider_dir() else { return };
        for dir in [exe_dir.clone(), exe_dir.parent().map(Path::to_path_buf).unwrap_or_default()] {
            let p = dir.join(RTX_PROVIDER_DLL);
            if p.is_file() {
                let imports = pe_imports(&p).unwrap();
                eprintln!("{RTX_PROVIDER_DLL} imports {imports:?}");
                assert!(imports.contains(&"onnxruntime_providers_shared.dll".to_string()), "{imports:?}");
                // A resolver with an empty pack reports the TensorRT runtime DLLs as missing.
                let resolver = Resolver { pack: BTreeMap::new(), nvidia: BTreeMap::new(), exe_dir: Some(dir.clone()), system32: system32() };
                let (order, missing) = resolver.plan(&p);
                assert!(order.is_empty());
                assert!(!missing.is_empty(), "the TensorRT runtime is not next to the provider");
                return;
            }
        }
        eprintln!("no {RTX_PROVIDER_DLL} next to the test binary; skipped");
    }

    /// A fake pack folder of copies of system DLLs: the resolver orders pack-internal dependencies
    /// before their users and reports what nothing provides.
    #[cfg(windows)]
    #[test]
    fn resolver_orders_pack_dependencies_first() {
        let root = std::env::temp_dir().join(format!("sigil-trt-pack-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("lib")).unwrap();
        // kernelbase.dll is imported by kernel32.dll; a "pack" copy of it must load first.
        std::fs::copy(system32().join("kernel32.dll"), root.join("provider.dll")).unwrap();
        std::fs::copy(system32().join("kernelbase.dll"), root.join("lib").join("kernelbase.dll")).unwrap();
        let pack = find_dlls(&root, SEARCH_DEPTH);
        assert!(pack.contains_key("kernelbase.dll") && pack.contains_key("provider.dll"));
        let resolver = Resolver { pack: pack.clone(), nvidia: BTreeMap::new(), exe_dir: None, system32: system32() };
        let (order, missing) = resolver.plan(&root.join("provider.dll"));
        if pe_imports(&system32().join("kernel32.dll")).unwrap().contains(&"kernelbase.dll".to_string()) {
            assert_eq!(order, vec![pack["kernelbase.dll"].clone()], "{order:?}");
        }
        assert!(missing.is_empty(), "{missing:?}");
        // Without System32 in the search, its imports are missing.
        let lonely = Resolver { pack: BTreeMap::new(), nvidia: BTreeMap::new(), exe_dir: None, system32: root.join("nowhere") };
        let (_, missing) = lonely.plan(&root.join("provider.dll"));
        assert!(!missing.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cache_keys_change_with_everything_that_invalidates_an_engine() {
        let base = cache_key("rtx", "m", "RTX 4090", (8, 9), "616.56/13040", "1.3", "1:2");
        assert_eq!(base.len(), 32);
        assert_eq!(base, cache_key("rtx", "m", "RTX 4090", (8, 9), "616.56/13040", "1.3", "1:2"));
        for other in [
            cache_key("classic", "m", "RTX 4090", (8, 9), "616.56/13040", "1.3", "1:2"),
            cache_key("rtx", "m2", "RTX 4090", (8, 9), "616.56/13040", "1.3", "1:2"),
            cache_key("rtx", "m", "RTX 4060", (8, 9), "616.56/13040", "1.3", "1:2"),
            cache_key("rtx", "m", "RTX 4090", (8, 6), "616.56/13040", "1.3", "1:2"),
            cache_key("rtx", "m", "RTX 4090", (8, 9), "620.00/13040", "1.3", "1:2"),
            cache_key("rtx", "m", "RTX 4090", (8, 9), "616.56/13040", "1.4", "1:2"),
            cache_key("rtx", "m", "RTX 4090", (8, 9), "616.56/13040", "1.3", "1:3"),
        ] {
            assert_ne!(base, other);
        }
    }

    #[test]
    fn supported_gpus_and_progress_estimate() {
        assert!(rtx_supports(8, 6) && rtx_supports(8, 9) && rtx_supports(12, 0) && rtx_supports(12, 1));
        assert!(!rtx_supports(7, 5) && !rtx_supports(8, 0) && !rtx_supports(8, 7) && !rtx_supports(9, 0));
        assert_eq!(estimated_fraction(0.0), 0.0);
        assert!((estimated_fraction(30.0) - 0.5).abs() < 1e-9);
        assert_eq!(estimated_fraction(600.0), 0.95, "never done before the build ends");
        let p = serde_json::to_value(TrtProgress { component: "vc.flowdec".into(), stage: "building".into(), ..TrtProgress::default() }).unwrap();
        for k in ["component", "stage", "fraction", "message", "elapsed_ms"] {
            assert!(p.get(k).is_some(), "{k}");
        }
        assert_eq!(engine_cache_dir(Path::new("/d")), Path::new("/d").join("voice-changer").join("trt-cache"));
    }

    /// Least recently used entries go first; the entry in use stays even past the limits.
    #[test]
    fn cache_eviction_keeps_the_newest_and_the_one_in_use() {
        let root = std::env::temp_dir().join(format!("sigil-trt-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for (i, name) in ["a", "b", "c", "d"].iter().enumerate() {
            let e = CacheEntry { dir: root.join(name) };
            std::fs::create_dir_all(&e.dir).unwrap();
            std::fs::write(e.dir.join("engine.bin"), vec![0u8; 1000]).unwrap();
            e.write_meta(&EntryMeta { last_used_unix_ms: 1000 + i as u64, ..EntryMeta::default() });
        }
        // Keep 2 entries: "d" and "c" are newest; "a" is in use, so it stays too.
        let removed = evict(&root, &root.join("a"), 2, u64::MAX);
        let names: BTreeSet<String> = removed.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, ["b".to_string()].into_iter().collect());
        assert!(root.join("a").is_dir() && root.join("c").is_dir() && root.join("d").is_dir());
        // A byte limit that fits one entry: only the newest ("d") and the one in use stay.
        let removed = evict(&root, &root.join("a"), 16, 1500);
        assert_eq!(removed.len(), 1, "{removed:?}");
        assert!(root.join("d").is_dir() && root.join("a").is_dir() && !root.join("c").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The discovery cache is recomputed when its key changes (e.g. the NVIDIA runtime pack is
    /// installed after the TensorRT pack) and reused while it does not.
    #[test]
    fn discovery_is_recomputed_when_the_runtime_packs_change() {
        let cache: Mutex<Option<(String, u32)>> = Mutex::new(None);
        let mut computed = 0;
        let mut get = |key: &str| {
            cached_for(&cache, key.to_string(), || {
                computed += 1;
                computed
            })
        };
        assert_eq!(get("trt|nvidia-missing"), 1);
        assert_eq!(get("trt|nvidia-missing"), 1, "cached");
        assert_eq!(get("trt|nvidia-13.0"), 2, "the NVIDIA pack changed");
        assert_eq!(get("trt|nvidia-13.0"), 2);
        // The real key follows the NVIDIA packs' fingerprint.
        let (_, fingerprint) = discovery_key();
        assert_eq!(fingerprint, crate::services::native_runtime::nvidia::fingerprint());
    }

    /// Without a pack there is no route, and a session request says why (never panics).
    #[test]
    fn no_pack_means_no_tensorrt() {
        if std::env::var_os(TRT_DIR_ENV).is_some() || crate::services::model_store::installed_path(RUNTIME_TENSORRT_RTX).is_some() {
            eprintln!("a TensorRT pack is installed; skipped");
            return;
        }
        assert!(!available());
        let d = discover();
        assert!(d.rtx.is_err() && d.classic.is_err());
        assert!(d.why_not().contains("runtime/tensorrt-rtx"), "{}", d.why_not());
    }
}
