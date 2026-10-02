//! What a helper is started with ([`Launch`]), the helper spec built from it, and the planning
//! helpers every engine shares: runtime pack selection, finding a pack's program, checking a
//! Sigil-built pack, catalog file paths and the NVIDIA runtime folders.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, PoisonError};

use serde::Deserialize;

use super::{GpuFacts, Variant};
use crate::services::model_store::Catalog;
use crate::services::process_host::{self, HealthCheck, HelperPriority, HelperSpec, PortRequest, RestartPolicy, StdioMode};
use crate::services::tts_helpers::TtsHelperEngine;
use crate::services::{diagnostics, native_runtime};

/// Catalog id of the NVIDIA CUDA 13 runtime pack.
pub const NVIDIA_RUNTIME_ID: &str = "runtime/nvidia-cuda13";

/// Engines whose server does not take Sigil's request shape, served through the in-process
/// [`super::adapter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AdapterKind {
    /// Breeze-TTS-2.cpp's native form request (`text`, `instruction`, `ref_audio`, ...).
    Breeze,
    /// s2.cpp's multipart `POST /generate`.
    S2,
}

/// How the runner knows the server is ready.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Readiness {
    /// `GET <path>` answers 2xx (process_host's health check).
    Health(String),
    /// The server has no health route: ready once it answers any HTTP request.
    AnyHttp,
}

/// Everything a helper is started with (engine-neutral).
#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    pub engine: TtsHelperEngine,
    pub helper_name: String,
    pub program: PathBuf,
    /// Arguments; `{port}` is substituted by the process host.
    pub args: Vec<String>,
    /// Extra environment; `{api_key}` is substituted by the process host.
    pub env: Vec<(String, String)>,
    /// Inherited variables removed first (exact names, or prefixes ending in `*`).
    pub env_remove: Vec<String>,
    /// The server takes the process host's random key (through `env`): requests send it.
    pub server_key: bool,
    pub readiness: Readiness,
    pub adapter: Option<AdapterKind>,
    /// Catalog id of the loaded model (status `model`).
    pub model: String,
    pub variant: Variant,
    /// Sample rate of the endpoint's raw PCM.
    pub pcm_rate: u32,
    /// VRAM claimed while it runs (MiB; 0 on the processor).
    pub vram_mb: u64,
    pub load_timeout_ms: u64,
    /// Folders put in front of the helper's PATH (the NVIDIA runtime for CUDA packs).
    pub dll_dirs: Vec<PathBuf>,
    /// One short request (OpenAI shape, sent through the endpoint) run after start.
    pub warmup: Option<serde_json::Value>,
    /// Summary for the log.
    pub summary: String,
}

impl Launch {
    /// Two launches that run the same server process (everything but the bookkeeping).
    pub fn same_server(&self, other: &Launch) -> bool {
        (&self.helper_name, &self.program, &self.args, &self.env, &self.env_remove, &self.dll_dirs, self.variant, self.adapter, &self.readiness)
            == (&other.helper_name, &other.program, &other.args, &other.env, &other.env_remove, &other.dll_dirs, other.variant, other.adapter, &other.readiness)
    }
}

/// The helper spec of a launch (pure):
/// - 127.0.0.1 on an automatic port, the random key only through the environment (never on the
///   command line, which every local process can read);
/// - the NVIDIA runtime folders in front of PATH;
/// - restart on failure (2 times, 1 s backoff doubling), EcoQoS off (speech is latency
///   sensitive), below-normal priority on the processor (VRChat and the voice changer first),
///   immediate stop (no state worth saving).
pub fn build_spec(l: &Launch) -> HelperSpec {
    let mut env = l.env.clone();
    if !l.dll_dirs.is_empty() {
        let mut path = l
            .dll_dirs
            .iter()
            .map(|d| d.display().to_string())
            .collect::<Vec<_>>()
            .join(";");
        if let Some(inherited) = std::env::var_os("PATH") {
            path.push(';');
            path.push_str(&inherited.to_string_lossy());
        }
        env.push(("PATH".to_string(), path));
    }
    let health = match &l.readiness {
        Readiness::Health(path) => Some(HealthCheck {
            path: path.clone(),
            timeout_ms: l.load_timeout_ms,
            interval_ms: 250,
            // Health routes of these servers are public; a key there is harmless.
            send_api_key: l.server_key,
        }),
        Readiness::AnyHttp => None,
    };
    HelperSpec {
        name: l.helper_name.clone(),
        program: l.program.clone(),
        args: l.args.clone(),
        env,
        env_remove: l.env_remove.clone(),
        cwd: l.program.parent().map(Path::to_path_buf),
        port: PortRequest::Auto,
        api_key: l.server_key,
        health,
        memory_limit_mb: None,
        restart: RestartPolicy::OnFailure {
            max_restarts: 2,
            backoff_ms: 1000,
        },
        priority: if l.variant == Variant::Cpu {
            HelperPriority::BelowNormal
        } else {
            HelperPriority::Normal
        },
        high_qos: true,
        stop_grace_ms: 0,
        stdio: StdioMode::Log,
    }
}

// ---------------------------------------------------------------------------------------------
// Planning environment
// ---------------------------------------------------------------------------------------------

/// What a plan reads from the PC (explicit for tests).
pub struct PlanEnv<'a> {
    pub catalog: &'a Catalog,
    /// Install folder of an installed catalog entry.
    pub installed: &'a dyn Fn(&str) -> Option<PathBuf>,
    pub gpu: GpuFacts,
    /// Folders with the CUDA 13 runtime DLLs (empty when the NVIDIA runtime is not installed).
    pub nvidia_dirs: &'a dyn Fn() -> Vec<PathBuf>,
    /// Sigil's app data (voice folders of the helpers); `None` when unknown.
    pub app_data: Option<PathBuf>,
}

impl PlanEnv<'_> {
    pub fn is_installed(&self, id: &str) -> bool {
        (self.installed)(id).is_some()
    }

    /// Install folder of `id`, or an error naming it.
    pub fn dir_of(&self, id: &str) -> Result<PathBuf, String> {
        (self.installed)(id).ok_or_else(|| format!("{id} is not installed"))
    }

    /// The file of catalog entry `id` whose path is `path` (or its first file), checked to exist.
    pub fn file_of(&self, id: &str, path: Option<&str>) -> Result<PathBuf, String> {
        let dir = self.dir_of(id)?;
        let entry = self.catalog.get(id).ok_or_else(|| format!("{id} is not in the catalog"))?;
        let rel = match path {
            Some(p) => entry
                .files
                .iter()
                .find(|f| f.path == p)
                .map(|f| f.path.clone())
                .ok_or_else(|| format!("{id} has no file {p}"))?,
            None => entry
                .files
                .first()
                .map(|f| f.path.clone())
                .ok_or_else(|| format!("{id} has no file in the catalog"))?,
        };
        let file = dir.join(rel);
        if file.is_file() {
            Ok(file)
        } else {
            Err(format!("{} is missing: check the files in Models & Runtimes", file.display()))
        }
    }

    /// Bytes of every file of `ids` (load timeout).
    pub fn bytes_of(&self, ids: &[&str]) -> u64 {
        ids.iter()
            .filter_map(|id| self.catalog.get(id))
            .flat_map(|e| e.files.iter().map(|f| f.size))
            .fold(0u64, u64::saturating_add)
    }

    /// The catalog footprint of `id` (MiB).
    pub fn footprint_mb(&self, id: &str) -> Option<u64> {
        self.catalog.get(id).and_then(|e| e.min_vram_mb)
    }

    /// Fail with every id of `ids` that is not installed (in order), as `<title> needs downloads first`.
    pub fn require(&self, title: &str, ids: &[&str]) -> Result<(), String> {
        let missing: Vec<&str> = ids.iter().copied().filter(|id| !self.is_installed(id)).collect();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(format!("{title} needs downloads first: {}", missing.join(", ")))
        }
    }
}

/// One runtime pack an engine can run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackChoice {
    pub variant: Variant,
    /// Catalog id of the pack.
    pub runtime: &'static str,
}

/// The pack for `choice` (`None` = automatic) among the engine's `packs` (in preference order):
/// - explicit: that variant, refused when the engine has no such pack or the GPU cannot run it;
/// - automatic: the processor without a GPU; else an installed NVIDIA pack the GPU runs (with the
///   NVIDIA runtime when it needs it), else an installed Vulkan pack, else an NVIDIA pack that is
///   half there (its pack or the NVIDIA runtime), else the first GPU pack, else the processor.
pub fn pick_pack(title: &str, packs: &[PackChoice], choice: Option<Variant>, env: &PlanEnv<'_>) -> Result<PackChoice, String> {
    let find = |v: Variant| packs.iter().copied().find(|p| p.variant == v);
    let ready = |p: &PackChoice| env.is_installed(p.runtime) && (p.variant != Variant::Cuda13 || env.is_installed(NVIDIA_RUNTIME_ID));
    if let Some(v) = choice {
        let p = find(v).ok_or_else(|| {
            let have: Vec<&str> = packs.iter().map(|p| p.variant.as_str()).collect();
            format!("{title} has no {} build (available: {})", v.as_str(), have.join(", "))
        })?;
        if v.nvidia() && !env.gpu.cuda {
            return Err(format!(
                "the {} build of {title} needs an NVIDIA GeForce GTX 16 / RTX 20 or newer with driver 580 or newer; choose Vulkan or Automatic",
                v.as_str()
            ));
        }
        if v.on_gpu() && !env.gpu.any_gpu {
            return Err(format!("no graphics card was found for the {} build of {title}; choose the processor", v.as_str()));
        }
        return Ok(p);
    }
    let cpu = find(Variant::Cpu);
    if !env.gpu.any_gpu {
        return cpu.ok_or_else(|| format!("{title} needs a graphics card, and none was found"));
    }
    let nvidia = packs.iter().copied().filter(|p| p.variant.nvidia() && env.gpu.cuda);
    if let Some(p) = nvidia.clone().find(ready) {
        return Ok(p);
    }
    if let Some(p) = find(Variant::Vulkan).filter(ready) {
        return Ok(p);
    }
    if let Some(p) = nvidia.clone().find(|p| env.is_installed(p.runtime) || (p.variant == Variant::Cuda13 && env.is_installed(NVIDIA_RUNTIME_ID))) {
        return Ok(p);
    }
    if let Some(p) = find(Variant::Vulkan).or_else(|| nvidia.clone().next()) {
        return Ok(p);
    }
    if let Some(p) = cpu {
        return Ok(p);
    }
    packs.first().copied().ok_or_else(|| format!("{title} has no runtime pack"))
}

/// The catalog ids a pack needs (the NVIDIA runtime for CUDA 13).
pub fn pack_ids(p: PackChoice) -> Vec<&'static str> {
    if p.variant == Variant::Cuda13 {
        vec![p.runtime, NVIDIA_RUNTIME_ID]
    } else {
        vec![p.runtime]
    }
}

/// `program` inside a runtime pack folder: the folder itself, then `bin/`, then any folder up to
/// three levels down (official zips keep their own top-level folder). Case-insensitive.
pub fn find_program(dir: &Path, program: &str) -> Option<PathBuf> {
    let mut level: Vec<PathBuf> = vec![dir.to_path_buf()];
    for _depth in 0..4 {
        let mut next = Vec::new();
        // `bin` folders first at every level.
        level.sort_by_key(|p| !p.file_name().is_some_and(|n| n.eq_ignore_ascii_case("bin")));
        for d in &level {
            let Ok(rd) = std::fs::read_dir(d) else { continue };
            let mut subdirs = Vec::new();
            for e in rd.flatten() {
                let Ok(ft) = e.file_type() else { continue };
                if ft.is_symlink() {
                    continue;
                }
                let name = e.file_name();
                if ft.is_file() && name.to_string_lossy().eq_ignore_ascii_case(program) {
                    return Some(e.path());
                }
                if ft.is_dir() {
                    subdirs.push(e.path());
                }
            }
            subdirs.sort();
            next.extend(subdirs);
        }
        if next.is_empty() {
            break;
        }
        level = next;
    }
    None
}

/// The program of an installed runtime pack, or an error that names the pack.
pub fn pack_program(env: &PlanEnv<'_>, runtime: &str, program: &str) -> Result<PathBuf, String> {
    let dir = env.dir_of(runtime)?;
    find_program(&dir, program).ok_or_else(|| {
        format!(
            "{runtime} in {} has no {program}: remove it in Models & Runtimes and install it again",
            dir.display()
        )
    })
}

// ---------------------------------------------------------------------------------------------
// Sigil-built packs (sigil-pack.json)
// ---------------------------------------------------------------------------------------------

/// `sigil-pack.json` of a Sigil-built runtime pack (written by the engine's `build.ps1`).
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct PackManifest {
    pub engine: String,
    pub variant: String,
    pub version: String,
    pub program: String,
    pub files: Vec<PackFile>,
    /// Arguments the build found the server needs on this variant (e.g. its GPU switch).
    pub server_args: Vec<String>,
    /// Extra arguments for cards with little video memory (layers or codec on the processor).
    pub low_vram_args: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct PackFile {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

/// sha256 of a file, cached per (path, size, modification time) for the life of the process.
pub fn sha256_cached(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    type Key = (PathBuf, u64, Option<std::time::SystemTime>);
    static CACHE: OnceLock<Mutex<HashMap<Key, String>>> = OnceLock::new();
    let md = std::fs::metadata(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let key: Key = (path.to_path_buf(), md.len(), md.modified().ok());
    let cache = CACHE.get_or_init(Default::default);
    if let Some(hit) = cache.lock().unwrap_or_else(PoisonError::into_inner).get(&key) {
        return Ok(hit.clone());
    }
    let mut f = std::fs::File::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let hex = hex::encode(hasher.finalize());
    cache
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(key, hex.clone());
    Ok(hex)
}

/// Check that `dir` holds the Sigil-built `engine` pack of `variant` whose program is `program`
/// and intact (blocking); returns the program and the manifest.
pub fn verify_sigil_pack(dir: &Path, engine: &str, variant: Variant, program: &str) -> Result<(PathBuf, PackManifest), String> {
    let manifest_path = dir.join("sigil-pack.json");
    let text = std::fs::read(&manifest_path)
        .map_err(|e| format!("the pack in {} has no readable sigil-pack.json ({e}): import the pack again", dir.display()))?;
    let m: PackManifest = serde_json::from_slice(&text).map_err(|e| format!("{} is not a pack manifest: {e}", manifest_path.display()))?;
    if m.engine != engine {
        return Err(format!("{} is not a {engine} pack (engine {:?})", dir.display(), m.engine));
    }
    if m.variant != variant.as_str() {
        return Err(format!(
            "the pack installed as the {} build is the {} build ({}): import the matching zip in Models & Runtimes",
            variant.as_str(),
            m.variant,
            m.version
        ));
    }
    let named = if m.program.is_empty() { program } else { m.program.as_str() };
    if !named.eq_ignore_ascii_case(program) {
        return Err(format!("the pack names {named:?} as its server; expected {program}"));
    }
    let exe = dir.join(program);
    let pinned = m
        .files
        .iter()
        .find(|f| f.path.eq_ignore_ascii_case(program))
        .ok_or_else(|| format!("sigil-pack.json in {} does not list {program}", dir.display()))?;
    let len = std::fs::metadata(&exe)
        .map_err(|e| format!("{} is missing ({e}): import the pack again", exe.display()))?
        .len();
    if len != pinned.size || !sha256_cached(&exe)?.eq_ignore_ascii_case(&pinned.sha256) {
        return Err(format!("{} does not match its pack manifest (damaged or replaced): import the pack again", exe.display()));
    }
    for a in m.server_args.iter().chain(&m.low_vram_args) {
        if !safe_pack_arg(a) {
            return Err(format!("sigil-pack.json in {} has an argument Sigil does not pass: {a:?}", dir.display()));
        }
    }
    Ok((exe, m))
}

/// Arguments a pack manifest may add: switches and plain values only (the host, port, model
/// paths and anything with a path or a shell character are Sigil's).
pub fn safe_pack_arg(a: &str) -> bool {
    let lower = a.to_ascii_lowercase();
    !a.is_empty()
        && a.len() <= 64
        && a.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '='))
        && !["--host", "--port", "--model", "--tokenizer", "--voice-dir", "--voice_dir", "--server"]
            .iter()
            .any(|k| lower == *k || lower.starts_with(&format!("{k}=")))
}

// ---------------------------------------------------------------------------------------------
// NVIDIA runtime folders, helper folders
// ---------------------------------------------------------------------------------------------

/// Folders holding the CUDA 13 runtime DLLs (the model-store NVIDIA pack, else the interim P0
/// pack); empty when neither is installed.
pub fn nvidia_dll_dirs() -> Vec<PathBuf> {
    const NEEDED: [&str; 3] = ["cudart64_13.dll", "cublas64_13.dll", "cublaslt64_13.dll"];
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(index) = native_runtime::nvidia::store_dll_index() {
        for name in NEEDED {
            let hit = index
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .and_then(|(_, p)| p.parent().map(Path::to_path_buf));
            if let Some(d) = hit {
                if !dirs.contains(&d) {
                    dirs.push(d);
                }
            }
        }
        if !dirs.is_empty() {
            return dirs;
        }
    }
    if let Some(app_data) = diagnostics::app_data_dir() {
        let bin = diagnostics::nvidia_pack::bin_dir(&app_data);
        if bin.join("cudart64_13.dll").is_file() {
            dirs.push(bin);
        }
    }
    dirs
}

/// `app_data/tts-helpers/<engine>/<leaf>`, created (the helpers' voice folders).
pub fn helper_dir(app_data: Option<&Path>, engine: TtsHelperEngine, leaf: &str) -> Result<PathBuf, String> {
    let root = app_data.ok_or_else(|| "Sigil's app-data folder is unknown".to_string())?;
    let dir = root.join("tts-helpers").join(engine.as_str()).join(leaf);
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    Ok(dir)
}

/// The process host's placeholder for the port (`{port}`).
pub fn port_arg() -> String {
    process_host::spawn::PORT_PLACEHOLDER.to_string()
}

/// The process host's placeholder for the key (`{api_key}`).
pub fn key_env() -> String {
    process_host::spawn::API_KEY_PLACEHOLDER.to_string()
}
