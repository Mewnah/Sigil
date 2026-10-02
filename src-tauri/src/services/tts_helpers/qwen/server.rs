//! Lifecycle of the `tts-qwen3` helper (Sigil's qwentts.cpp `sigil-tts-server`): resolve a config
//! to a talker, the tokenizer and a runtime pack; start it through `process_host` (replacing a
//! server with another launch, reusing one with the same launch, waiting for the process host's
//! restart of one that crashed); check what it loaded; warm it up
//! (time to first audio and real-time factor logged); watch it (crash, restart, port change); hold
//! and resize its VRAM claim; stop it. Status changes are emitted as `tts_helpers:state`.
//!
//! Concurrency (as `llm_server::server`): starts are serialized by an async gate (a second caller
//! with the same config gets the running server). A stop never waits for the gate: it bumps the
//! stop epoch and stops the helper, which cancels a start that is still loading (process_host's
//! last-request-wins); a start requested before a stop returns "cancelled". Every start and stop
//! bumps a generation, so a watcher or a start that finishes late never touches a newer server's
//! state, and a start that finds itself superseded after its helper became ready stops it again.

use super::plan::{self, Knobs, Mode, Plan, QwenConfig, Size, Variant, SERVER_EXE, TOKENIZER_ID};
use super::ENGINE;
use crate::services::model_store::{self, Catalog};
use crate::services::process_host::{self, HealthCheck, HelperLogLine, HelperPriority, HelperSpec, HelperState, PortRequest, RestartPolicy, StdioMode};
use crate::services::resources::{self, RecommendationTopic, Subsystem, VramClaim};
use crate::services::tts_helpers::{TtsHelperEndpoint, TtsHelperState, TtsHelperStatus, STATE_EVENT};
use crate::services::{diagnostics, sysinfo};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, Instant};

pub(super) const TARGET: &str = "sigil::tts::qwen3";
/// process_host helper name (`TtsHelperEngine::Qwen3.helper_name()`).
pub const HELPER_NAME: &str = "tts-qwen3";
/// Environment variable the server reads its key from (never a command-line argument).
pub const API_KEY_ENV: &str = "QWENTTS_API_KEY";
/// Sample rate of the server's raw PCM.
pub const PCM_RATE: u32 = 24_000;
/// How often the watcher looks at the helper.
const WATCH_INTERVAL: Duration = Duration::from_millis(400);
/// Watcher rounds between two looks at the logged buffer sizes (lazily loaded encoders add some).
const MEASURE_EVERY: u32 = 10;
/// Runtime overhead added to the logged buffers (CUDA context, Vulkan device), MiB.
pub const CUDA_BASE_MB: u64 = 400;
pub const VULKAN_BASE_MB: u64 = 100;
/// Compute buffers qwentts.cpp does not log (graph arenas of the talker, predictor and codec), MiB.
pub const COMPUTE_ALLOWANCE_MB: u64 = 150;
/// Time a warm-up request may take before the start gives up on it (the server is ready anyway).
const WARMUP_TIMEOUT: Duration = Duration::from_secs(90);

fn err(message: impl Into<String>) -> String {
    message.into()
}

// ---------------------------------------------------------------------------------------------
// Resolution: config -> plan -> launch
// ---------------------------------------------------------------------------------------------

/// Everything the helper is started with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    pub helper_name: String,
    pub exe: PathBuf,
    pub talker_path: PathBuf,
    pub codec_path: PathBuf,
    /// Talker catalog id (status `model`, server `--alias`).
    pub model: String,
    pub size: Size,
    pub mode: Mode,
    pub variant: Variant,
    /// The configured language (a qwentts name or `auto`), used by the warm-up request only. Not part
    /// of the server's identity: every speech request names its own `language`, so the server always
    /// runs with `--lang auto` and a language change never reloads the model.
    pub language: String,
    /// VRAM claimed while it runs (MiB; 0 on the CPU pack).
    pub vram_mb: u64,
    pub load_timeout_ms: u64,
    /// Folders put in front of the helper's PATH (the NVIDIA runtime for the cuda13 pack).
    pub dll_dirs: Vec<PathBuf>,
}

impl Launch {
    /// Two launches with the same key run the same server (the language is not part of it: requests
    /// carry their own).
    pub fn same_server(&self, other: &Launch) -> bool {
        (&self.helper_name, &self.exe, &self.talker_path, &self.codec_path, self.variant, &self.dll_dirs)
            == (&other.helper_name, &other.exe, &other.talker_path, &other.codec_path, other.variant, &other.dll_dirs)
    }

    /// Both run the same talker and tokenizer files (cloned voices' speaker data stays valid).
    pub fn same_talker(&self, other: &Launch) -> bool {
        self.talker_path == other.talker_path && self.codec_path == other.codec_path
    }

    /// File name of the talker (cache key of cloned voices' speaker data).
    pub fn talker_file(&self) -> String {
        file_name(&self.talker_path)
    }

    pub fn codec_file(&self) -> String {
        file_name(&self.codec_path)
    }
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The first file of catalog entry `id` inside its install folder `dir`.
fn entry_file(catalog: &Catalog, id: &str, dir: &Path) -> Option<PathBuf> {
    let entry = catalog.get(id)?;
    let f = entry.files.first()?;
    Some(dir.join(&f.path))
}

fn entry_bytes(catalog: &Catalog, id: &str) -> u64 {
    catalog
        .get(id)
        .map(|e| e.files.iter().map(|f| f.size).sum())
        .unwrap_or(0)
}

/// Folders holding the CUDA 13 runtime DLLs the cuda13 pack imports (the model-store NVIDIA pack, else
/// the interim P0 pack); empty when neither is installed.
pub fn nvidia_dll_dirs() -> Vec<PathBuf> {
    const NEEDED: [&str; 3] = ["cudart64_13.dll", "cublas64_13.dll", "cublaslt64_13.dll"];
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(index) = crate::services::native_runtime::nvidia::store_dll_index() {
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

/// `sigil-pack.json` of a runtime pack (written by `scripts/runtime-packs/qwentts/build.ps1`).
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct PackManifest {
    engine: String,
    variant: String,
    version: String,
    program: String,
    files: Vec<PackFile>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct PackFile {
    path: String,
    size: u64,
    sha256: String,
}

/// The engine name in a pack manifest.
pub const PACK_ENGINE: &str = "qwentts-server";

/// sha256 of a file, cached per (path, size, modification time) for the life of the process.
fn sha256_cached(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::collections::HashMap;
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

/// Check that `dir` holds Sigil's `variant` server pack and that its program is intact (blocking).
pub fn verify_pack(dir: &Path, variant: Variant) -> Result<PathBuf, String> {
    let manifest_path = dir.join("sigil-pack.json");
    let text = std::fs::read(&manifest_path)
        .map_err(|e| format!("the Qwen3-TTS server pack in {} has no readable sigil-pack.json ({e}): import the pack again", dir.display()))?;
    let m: PackManifest = serde_json::from_slice(&text).map_err(|e| format!("{} is not a pack manifest: {e}", manifest_path.display()))?;
    if m.engine != PACK_ENGINE {
        return Err(format!("{} is not a Qwen3-TTS server pack (engine {:?})", dir.display(), m.engine));
    }
    if m.variant != variant.as_str() {
        return Err(format!(
            "the pack installed as the {} build is the {} build ({}): import the matching zip in Models & Runtimes",
            variant.as_str(),
            m.variant,
            m.version
        ));
    }
    let program = if m.program.is_empty() { SERVER_EXE.to_string() } else { m.program.clone() };
    if program != SERVER_EXE {
        return Err(format!("the pack names {program:?} as its server; expected {SERVER_EXE}"));
    }
    let exe = dir.join(SERVER_EXE);
    let pinned = m
        .files
        .iter()
        .find(|f| f.path.eq_ignore_ascii_case(SERVER_EXE))
        .ok_or_else(|| format!("sigil-pack.json in {} does not list {SERVER_EXE}", dir.display()))?;
    let len = std::fs::metadata(&exe)
        .map_err(|e| format!("{} is missing ({e}): import the pack again", exe.display()))?
        .len();
    if len != pinned.size || !sha256_cached(&exe)?.eq_ignore_ascii_case(&pinned.sha256) {
        return Err(format!("{} does not match its pack manifest (damaged or replaced): import the pack again", exe.display()));
    }
    Ok(exe)
}

/// The launch for `config` on this PC, or why it cannot start (blocking: system probe, catalog,
/// file checks).
pub fn resolve(config: &QwenConfig) -> Result<(Launch, Plan), String> {
    let catalog = Catalog::load()?;
    let sys = sysinfo::cached();
    let snap = resources::snapshot();
    let rec = resources::recommend(RecommendationTopic::Qwen3Tts, &sys, &snap);
    resolve_with(config, &catalog, &sys, &rec, &|id| model_store::installed_path(id), &nvidia_dll_dirs)
}

/// [`resolve`] with explicit inputs (tests).
pub fn resolve_with(
    config: &QwenConfig,
    catalog: &Catalog,
    sys: &sysinfo::SystemInfo,
    rec: &resources::Recommendation,
    installed: &dyn Fn(&str) -> Option<PathBuf>,
    nvidia_dirs: &dyn Fn() -> Vec<PathBuf>,
) -> Result<(Launch, Plan), String> {
    let knobs: Knobs = plan::normalize(config)?;
    let is_installed = |id: &str| installed(id).is_some();
    let footprint = |id: &str| catalog.get(id).and_then(|e| e.min_vram_mb);
    let p = plan::plan(
        &knobs,
        &plan::Env {
            sys,
            recommendation: rec,
            installed: &is_installed,
            footprint_mb: &footprint,
        },
    );
    if let Some(why) = &p.refused {
        return Err(format!("Qwen3-TTS cannot run: {why}"));
    }
    if !p.missing.is_empty() {
        return Err(format!("Qwen3-TTS needs downloads first: {}", p.missing.join(", ")));
    }
    let talker_dir = installed(p.talker).ok_or_else(|| err(format!("{} is not installed", p.talker)))?;
    let codec_dir = installed(TOKENIZER_ID).ok_or_else(|| err(format!("{TOKENIZER_ID} is not installed")))?;
    let runtime_dir = installed(p.runtime).ok_or_else(|| err(format!("{} is not installed", p.runtime)))?;
    let talker_path = entry_file(catalog, p.talker, &talker_dir).ok_or_else(|| err(format!("{} has no file in the catalog", p.talker)))?;
    let codec_path = entry_file(catalog, TOKENIZER_ID, &codec_dir).ok_or_else(|| err(format!("{TOKENIZER_ID} has no file in the catalog")))?;
    for f in [&talker_path, &codec_path] {
        if !f.is_file() {
            return Err(format!("{} is missing: check the files in Models & Runtimes", f.display()));
        }
    }
    let exe = verify_pack(&runtime_dir, p.variant)?;
    let dll_dirs = if p.variant == Variant::Cuda13 { nvidia_dirs() } else { Vec::new() };
    if p.variant == Variant::Cuda13 && dll_dirs.is_empty() {
        return Err(format!(
            "the CUDA 13 build needs the NVIDIA GPU runtime ({}): install it in Models & Runtimes, or choose Vulkan",
            plan::NVIDIA_RUNTIME_ID
        ));
    }
    let bytes = entry_bytes(catalog, p.talker).saturating_add(entry_bytes(catalog, TOKENIZER_ID));
    Ok((
        Launch {
            helper_name: HELPER_NAME.to_string(),
            exe,
            talker_path,
            codec_path,
            model: p.talker.to_string(),
            size: p.size,
            mode: p.mode,
            variant: p.variant,
            language: knobs.language.to_string(),
            vram_mb: p.vram_mb,
            load_timeout_ms: plan::load_timeout_ms(bytes),
            dll_dirs,
        },
        p,
    ))
}

/// Default language of the server (`--lang`): always `auto`, since every speech request names its
/// own `language` (TypeScript always sends one), so the configured language never decides the server.
pub const SERVER_LANG: &str = "auto";

/// The helper spec of a launch (pure):
/// - `--host 127.0.0.1 --port {port}`, the talker and tokenizer, `--alias <talker id>`,
///   `--lang auto` ([`SERVER_LANG`]);
/// - the random key through [`API_KEY_ENV`] (never on the command line); `/health` is public;
/// - the NVIDIA runtime folders in front of PATH for the cuda13 pack;
/// - restart on failure (2 times, 1 s backoff doubling), EcoQoS off (speech is latency
///   sensitive), below-normal priority on the CPU pack (VRChat and the voice changer first),
///   immediate stop (no state worth saving).
pub fn build_spec(l: &Launch) -> HelperSpec {
    let mut env = vec![(API_KEY_ENV.to_string(), process_host::spawn::API_KEY_PLACEHOLDER.to_string())];
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
    HelperSpec {
        name: l.helper_name.clone(),
        program: l.exe.clone(),
        args: vec![
            "--model".into(),
            l.talker_path.display().to_string(),
            "--codec".into(),
            l.codec_path.display().to_string(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            process_host::spawn::PORT_PLACEHOLDER.into(),
            "--alias".into(),
            l.model.clone(),
            "--lang".into(),
            SERVER_LANG.into(),
        ],
        env,
        env_remove: vec![API_KEY_ENV.into()],
        cwd: l.exe.parent().map(Path::to_path_buf),
        port: PortRequest::Auto,
        api_key: true,
        health: Some(HealthCheck {
            path: "/health".into(),
            timeout_ms: l.load_timeout_ms,
            interval_ms: 250,
            // The port's owner is verified before the first probe; a key on /health is harmless.
            send_api_key: true,
        }),
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
// Log measurements
// ---------------------------------------------------------------------------------------------

/// What the server's load log says: the compute device of the talker and the device buffers it
/// allocated (weights and KV cache, MiB).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoadReport {
    /// `CUDA0`, `Vulkan0`, `CPU`, ... (`[Load] Talker backend: <name> (...)`).
    pub backend: Option<String>,
    pub buffers_mb: f64,
}

impl LoadReport {
    pub fn on_gpu(&self) -> bool {
        self.backend
            .as_deref()
            .is_some_and(|b| !b.to_ascii_uppercase().starts_with("CPU"))
    }
}

/// Parse qwentts.cpp's load lines: `[Load] Talker backend: CUDA0 (CPU threads: 16)`,
/// `[WeightCtx] Loaded 316 tensors, 1765.0 MB into backend`, `[KVCache] Allocated: ... -> 896 MB`.
/// The helper's log keeps the lines of every instance since it was first started: each instance
/// begins with its backend line, so only the buffers after the last one count (a restarted server
/// is never counted twice).
pub fn parse_load_report(lines: &[HelperLogLine]) -> LoadReport {
    let mut r = LoadReport::default();
    for l in lines {
        let t = l.text.trim();
        if let Some(rest) = t.split("Talker backend:").nth(1) {
            let name = rest.split_whitespace().next().unwrap_or("").to_string();
            r = LoadReport::default();
            if !name.is_empty() {
                r.backend = Some(name);
            }
        } else if t.contains("[WeightCtx]") && t.contains("into backend") {
            if let Some(mb) = number_before(t, " MB into backend") {
                r.buffers_mb += mb;
            }
        } else if t.contains("[KVCache]") && t.contains("->") {
            if let Some(mb) = t.rsplit("->").next().and_then(|s| number_before(s, " MB")) {
                r.buffers_mb += mb;
            }
        }
    }
    r
}

/// The number right before `suffix` in `text` (`"..., 1765.0 MB into backend"` -> 1765.0).
fn number_before(text: &str, suffix: &str) -> Option<f64> {
    let end = text.find(suffix)?;
    let head = &text[..end];
    let start = head
        .rfind(|c: char| !(c.is_ascii_digit() || c == '.'))
        .map(|i| i + 1)
        .unwrap_or(0);
    head[start..].parse::<f64>().ok().filter(|v| v.is_finite() && *v >= 0.0)
}

/// VRAM the helper holds according to its log (MiB), `None` until it reported a GPU backend.
pub fn measured_vram_mb(report: &LoadReport, variant: Variant) -> Option<u64> {
    if !report.on_gpu() || report.buffers_mb <= 0.0 {
        return None;
    }
    let base = match variant {
        Variant::Cuda13 => CUDA_BASE_MB,
        Variant::Vulkan => VULKAN_BASE_MB,
        Variant::Cpu => 0,
    };
    Some((report.buffers_mb.ceil() as u64).saturating_add(base).saturating_add(COMPUTE_ALLOWANCE_MB))
}

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------

struct Current {
    gen: u64,
    launch: Launch,
    status: TtsHelperStatus,
    claim: Option<VramClaim>,
    /// Largest measured footprint of this instance (MiB); reset when the helper restarts.
    measured_mb: u64,
    restarts: u32,
    rounds: u32,
}

#[derive(Default)]
struct State {
    /// Bumped by every start and stop.
    gen: u64,
    /// Bumped by every stop.
    stops: u64,
    current: Option<Current>,
    /// Shown while nothing is current (the last failure, or stopped).
    last: Option<TtsHelperStatus>,
}

fn state() -> MutexGuard<'static, State> {
    static S: OnceLock<Mutex<State>> = OnceLock::new();
    S.get_or_init(Default::default)
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

fn gate() -> &'static tokio::sync::Mutex<()> {
    static G: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    G.get_or_init(Default::default)
}

fn emit(status: &TtsHelperStatus) {
    diagnostics::emit(STATE_EVENT, status);
}

fn status_of(model: Option<&str>, state: TtsHelperState, error: Option<String>) -> TtsHelperStatus {
    TtsHelperStatus {
        engine: ENGINE,
        state,
        model: model.map(str::to_string),
        error,
    }
}

fn stopped_status() -> TtsHelperStatus {
    status_of(None, TtsHelperState::Stopped, None)
}

/// The current status (never contains the key).
pub fn status() -> TtsHelperStatus {
    let st = state();
    match &st.current {
        Some(c) => c.status.clone(),
        None => st.last.clone().unwrap_or_else(stopped_status),
    }
}

/// The stop counter: a start that read an older value is cancelled.
pub fn stop_epoch() -> u64 {
    state().stops
}

/// The running server's launch, when it is ready.
pub fn ready_launch() -> Option<Launch> {
    let st = state();
    st.current
        .as_ref()
        .filter(|c| c.status.state == TtsHelperState::Ready)
        .map(|c| c.launch.clone())
}

/// The endpoint of `launch`'s running helper (its port and key can change after a restart).
fn endpoint_of(launch: &Launch) -> Option<(TtsHelperEndpoint, u32)> {
    let h = process_host::handle(&launch.helper_name)?;
    let base = h.base_url?;
    let key = h.api_key?;
    Some((
        TtsHelperEndpoint {
            engine: ENGINE,
            base_url: format!("{base}/v1"),
            api_key: key,
            model: launch.model.clone(),
            pcm_sample_rate: PCM_RATE,
        },
        h.pid,
    ))
}

/// Where the ready server listens right now: `(endpoint, pid, launch)`.
pub fn current_target() -> Result<(TtsHelperEndpoint, u32, Launch), String> {
    let launch = ready_launch().ok_or_else(|| err("Qwen3-TTS is not running; start text to speech first"))?;
    let running = process_host::get(&launch.helper_name).is_some_and(|i| i.state == HelperState::Running);
    if !running {
        return Err(err("the Qwen3-TTS helper is restarting; try again in a moment"));
    }
    let (endpoint, pid) = endpoint_of(&launch).ok_or_else(|| err("the Qwen3-TTS helper has no endpoint"))?;
    Ok((endpoint, pid, launch))
}

// ---------------------------------------------------------------------------------------------
// HTTP to the helper (loopback only, never through a proxy)
// ---------------------------------------------------------------------------------------------

pub(super) fn client() -> &'static reqwest::Client {
    static C: OnceLock<reqwest::Client> = OnceLock::new();
    C.get_or_init(|| {
        reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(Duration::from_secs(5))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("reqwest Client builder")
    })
}

/// `GET /sigil/info` of the server (Sigil's build only).
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct ServerInfo {
    pub server: String,
    pub pack_version: String,
    pub variant: String,
    pub upstream_commit: String,
    pub model_type: String,
    pub speakers: Vec<String>,
    pub languages: Vec<String>,
    pub sample_rate: u32,
}

pub(super) async fn fetch_info(endpoint: &TtsHelperEndpoint) -> Result<ServerInfo, String> {
    let root = endpoint.base_url.trim_end_matches("/v1");
    let resp = client()
        .get(format!("{root}/sigil/info"))
        .bearer_auth(&endpoint.api_key)
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("cannot reach the Qwen3-TTS helper: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("the Qwen3-TTS helper answered {} to /sigil/info", resp.status()));
    }
    resp.json::<ServerInfo>()
        .await
        .map_err(|e| format!("unreadable /sigil/info answer: {e}"))
}

/// Timing of one synthesis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timing {
    pub first_audio_ms: f64,
    pub total_ms: f64,
    pub audio_ms: f64,
}

impl Timing {
    /// Real-time factor (synthesis time / audio time); `None` without audio.
    pub fn rtf(&self) -> Option<f64> {
        (self.audio_ms > 0.0).then(|| self.total_ms / self.audio_ms)
    }
}

/// Speak `body` (raw PCM) and time it; the audio is discarded.
pub(super) async fn timed_speech(endpoint: &TtsHelperEndpoint, body: serde_json::Value, timeout: Duration) -> Result<Timing, String> {
    use futures::StreamExt;
    let started = Instant::now();
    let resp = client()
        .post(format!("{}/audio/speech", endpoint.base_url))
        .bearer_auth(&endpoint.api_key)
        .json(&body)
        .timeout(timeout)
        .send()
        .await
        .map_err(|e| format!("speech request failed: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("speech request answered {status}: {}", text.chars().take(200).collect::<String>()));
    }
    let mut first = None;
    let mut bytes = 0usize;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("speech stream failed: {e}"))?;
        if !chunk.is_empty() && first.is_none() {
            first = Some(started.elapsed());
        }
        bytes += chunk.len();
    }
    let total = started.elapsed();
    Ok(Timing {
        first_audio_ms: first.unwrap_or(total).as_secs_f64() * 1000.0,
        total_ms: total.as_secs_f64() * 1000.0,
        audio_ms: (bytes / 2) as f64 * 1000.0 / f64::from(PCM_RATE),
    })
}

/// A short request that runs every kernel once (CUDA graphs, Vulkan pipelines), so the first
/// caption does not pay for it; logs its time to first audio and real-time factor.
async fn warm_up(endpoint: &TtsHelperEndpoint, launch: &Launch, info: Option<&ServerInfo>) {
    let mut body = serde_json::json!({
        "input": "Hello there.",
        "response_format": "pcm",
        "max_new_tokens": 12,
        "language": launch.language,
    });
    match launch.mode {
        Mode::Preset => {
            let speaker = info
                .and_then(|i| i.speakers.first().cloned())
                .unwrap_or_else(|| "ryan".to_string());
            body["voice"] = serde_json::Value::String(speaker);
        }
        Mode::Design => body["instructions"] = serde_json::Value::String("A clear, calm adult voice.".into()),
        Mode::Clone => {}
    }
    match timed_speech(endpoint, body, WARMUP_TIMEOUT).await {
        Ok(t) => tracing::info!(
            target: TARGET,
            "warm-up ({} on {}): first audio {:.0} ms, {:.0} ms of audio in {:.0} ms (RTF {})",
            launch.model,
            launch.variant.as_str(),
            t.first_audio_ms,
            t.audio_ms,
            t.total_ms,
            t.rtf().map(|r| format!("{r:.3}")).unwrap_or_else(|| "-".into())
        ),
        Err(e) => tracing::warn!(target: TARGET, "warm-up failed (the server keeps running): {e}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Start / stop
// ---------------------------------------------------------------------------------------------

fn join_error(e: impl std::fmt::Display) -> String {
    format!("Qwen3-TTS task failed: {e}")
}

/// Start (or reuse) the helper for `config`.
pub async fn start(config: QwenConfig) -> Result<TtsHelperEndpoint, String> {
    let stops_at_call = stop_epoch();
    let _gate = gate().lock().await;
    if state().stops != stops_at_call {
        return Err(err("cancelled: Qwen3-TTS was stopped"));
    }
    let cfg = config.clone();
    let (launch, p) = tauri::async_runtime::spawn_blocking(move || resolve(&cfg))
        .await
        .map_err(join_error)??;
    tracing::info!(target: TARGET, "plan: {}", p.reason);
    start_launch(launch, stops_at_call).await
}

/// How a spawn ended, decided under the state lock (acted on after it is released).
enum SpawnOutcome {
    Ready(TtsHelperEndpoint),
    Failed(TtsHelperStatus, String),
    /// A stop (or a replacement) came while it started; the helper may run anyway.
    Superseded { started: bool },
}

/// The endpoint of server generation `gen` once the process host runs it: at once when it runs,
/// after the process host's restart when it is restarting it (bounded by the load timeout), `None`
/// when it is gone, failed or replaced, or a stop came.
async fn wait_for_running(gen: u64, launch: &Launch, stops_at_call: u64) -> Option<TtsHelperEndpoint> {
    let deadline = Instant::now() + Duration::from_millis(launch.load_timeout_ms);
    let mut logged = false;
    loop {
        {
            let st = state();
            if st.stops != stops_at_call || st.current.as_ref().is_none_or(|c| c.gen != gen) {
                return None;
            }
        }
        match process_host::get(&launch.helper_name).map(|i| i.state) {
            Some(HelperState::Running) => return endpoint_of(launch).map(|(endpoint, _)| endpoint),
            Some(HelperState::Starting | HelperState::Restarting) if Instant::now() < deadline => {
                if !logged {
                    tracing::info!(target: TARGET, "{} is restarting; waiting for it", launch.model);
                    logged = true;
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            _ => return None,
        }
    }
}

/// Take the process host's restart count of the current instance: a restarted instance loaded again
/// (measured afresh) and holds none of the old one's cloned voices. True when it changed (the caller
/// then forgets the clone registry, still under the state lock).
fn note_restarts(cur: &mut Current, restarts: u32) -> bool {
    if restarts == cur.restarts {
        return false;
    }
    cur.restarts = restarts;
    cur.measured_mb = 0;
    cur.rounds = 0;
    true
}

/// The reused server of generation `gen` runs again (a restart after a crash, or it never stopped):
/// show it ready at once instead of after the watcher's next look, so a clone registration made
/// right after this start (TypeScript's next step) is not refused as "not running". Nothing changes
/// when it is no longer current or the process host no longer runs it.
fn mark_running(gen: u64, launch: &Launch) {
    let Some(info) = process_host::get(&launch.helper_name).filter(|i| i.state == HelperState::Running) else {
        return;
    };
    let changed = {
        let mut st = state();
        let Some(cur) = st.current.as_mut().filter(|c| c.gen == gen) else {
            return;
        };
        if note_restarts(cur, info.restarts) {
            super::clone::forget_all();
        }
        let before = cur.status.clone();
        cur.status.state = TtsHelperState::Ready;
        cur.status.error = None;
        (cur.status != before).then(|| cur.status.clone())
    };
    if let Some(status) = changed {
        tracing::info!(target: TARGET, "{} runs again", launch.model);
        emit(&status);
    }
}

/// Start `launch` unless the running server already is it. Caller holds the start gate (tests call
/// it directly with their own launch).
pub(super) async fn start_launch(launch: Launch, stops_at_call: u64) -> Result<TtsHelperEndpoint, String> {
    // Reuse the running server, or wait for the process host's restart of it (after a crash) instead
    // of starting another one; its new port and key are read from the process host.
    let reuse = {
        let st = state();
        st.current
            .as_ref()
            .filter(|c| c.launch.same_server(&launch) && matches!(c.status.state, TtsHelperState::Ready | TtsHelperState::Starting))
            .map(|c| (c.gen, c.launch.clone()))
    };
    if let Some((gen, running)) = reuse {
        if let Some(endpoint) = wait_for_running(gen, &running, stops_at_call).await {
            mark_running(gen, &running);
            return Ok(endpoint);
        }
    }

    // Retire whatever runs now (its VRAM is freed before the new model loads).
    let old = {
        let mut st = state();
        st.gen += 1;
        st.current.take()
    };
    if let Some(old) = old {
        tracing::info!(target: TARGET, "replacing {} ({})", old.launch.model, old.launch.variant.as_str());
        if let Err(e) = process_host::stop(&old.launch.helper_name).await {
            tracing::warn!(target: TARGET, "stopping the previous server: {e}");
        }
        drop(old);
    }

    let (gen, starting) = {
        let mut st = state();
        if st.stops != stops_at_call {
            return Err(err("cancelled: Qwen3-TTS was stopped"));
        }
        st.gen += 1;
        let claim = (launch.vram_mb > 0).then(|| resources::claim_external_vram(Subsystem::Tts, &format!("qwen3-tts: {}", launch.model), launch.vram_mb));
        let status = status_of(Some(&launch.model), TtsHelperState::Starting, None);
        st.current = Some(Current {
            gen: st.gen,
            launch: launch.clone(),
            status: status.clone(),
            claim,
            measured_mb: 0,
            restarts: 0,
            rounds: 0,
        });
        (st.gen, status)
    };
    emit(&starting);
    // A new instance has none of the old one's cloned voices.
    super::clone::forget_all();

    let spec = build_spec(&launch);
    #[cfg(test)]
    let spec = test_hooks::apply(spec);
    let started_at = Instant::now();
    let result = process_host::spawn(spec).await;

    // Check what the server loaded, then warm it up (outside the state lock).
    let checked: Result<TtsHelperEndpoint, String> = match &result {
        Ok(_) => match endpoint_of(&launch) {
            Some((endpoint, _)) => {
                let info = fetch_info(&endpoint).await;
                match &info {
                    Ok(i) if !i.model_type.is_empty() && i.model_type != launch.mode.model_type() => Err(format!(
                        "{} is a {} model, but the {} mode needs a {} model: check the files in Models & Runtimes",
                        launch.talker_file(),
                        i.model_type,
                        launch.mode.as_str(),
                        launch.mode.model_type()
                    )),
                    other => {
                        match other {
                            Ok(i) => tracing::info!(
                                target: TARGET,
                                "ready in {} ms: {} ({}, {}), {} speakers, languages {}, {} Hz, {} {} {} (qwentts.cpp {})",
                                started_at.elapsed().as_millis(),
                                launch.model,
                                launch.size.as_str(),
                                i.model_type,
                                i.speakers.len(),
                                i.languages.join(" "),
                                i.sample_rate,
                                i.server,
                                i.pack_version,
                                i.variant,
                                i.upstream_commit
                            ),
                            Err(e) => tracing::warn!(target: TARGET, "ready, but its info is not available: {e}"),
                        }
                        warm_up(&endpoint, &launch, other.as_ref().ok()).await;
                        Ok(endpoint)
                    }
                }
            }
            None => Err(err("the Qwen3-TTS helper started without a port or key")),
        },
        Err(e) => Err(e.clone()),
    };

    // Decide under the lock, act (emit, stop a helper) after releasing it.
    let (outcome, released) = {
        let mut st = state();
        let still_ours = st.gen == gen && st.current.as_ref().is_some_and(|c| c.gen == gen);
        match (checked, still_ours) {
            (Ok(endpoint), true) => {
                let cur = st.current.as_mut().expect("checked above");
                cur.status.state = TtsHelperState::Ready;
                (SpawnOutcome::Ready(endpoint), None)
            }
            (Err(e), true) => {
                let failed = st.current.take();
                let status = status_of(Some(&launch.model), TtsHelperState::Failed, Some(e.clone()));
                st.last = Some(status.clone());
                (SpawnOutcome::Failed(status, e), failed)
            }
            (_, false) => (SpawnOutcome::Superseded { started: result.is_ok() }, None),
        }
    };

    match outcome {
        SpawnOutcome::Ready(endpoint) => {
            emit(&status());
            tauri::async_runtime::spawn(watch(gen));
            Ok(endpoint)
        }
        SpawnOutcome::Failed(status, e) => {
            if result.is_ok() {
                // Started, but unusable (wrong model type, no endpoint): stop it again.
                if let Err(stop_err) = process_host::stop(&launch.helper_name).await {
                    tracing::warn!(target: TARGET, "stopping an unusable server: {stop_err}");
                }
            }
            drop(released); // the VRAM claim goes once the process is gone
            emit(&status);
            tracing::warn!(target: TARGET, "Qwen3-TTS failed to start: {e}");
            Err(e)
        }
        SpawnOutcome::Superseded { started } => {
            if started {
                tracing::info!(target: TARGET, "{} became ready after a stop; stopping it", launch.model);
                if let Err(e) = process_host::stop(&launch.helper_name).await {
                    tracing::warn!(target: TARGET, "stopping a server that was started after a stop: {e}");
                }
            }
            Err(err("cancelled: Qwen3-TTS was stopped or replaced while it started"))
        }
    }
}

/// Stop the helper (also cancels a start that is still loading).
pub async fn stop() -> Result<(), String> {
    let (name, cur) = {
        let mut st = state();
        st.stops += 1;
        st.gen += 1;
        let cur = st.current.take();
        let name = cur
            .as_ref()
            .map(|c| c.launch.helper_name.clone())
            .unwrap_or_else(|| HELPER_NAME.to_string());
        (name, cur)
    };
    let result = process_host::stop(&name).await;
    drop(cur); // releases the VRAM claim once the process is gone
    let status = {
        let mut st = state();
        if st.current.is_none() {
            st.last = None;
        }
        match &st.current {
            Some(c) => c.status.clone(),
            None => stopped_status(),
        }
    };
    emit(&status);
    super::clone::forget_all();
    result
}

fn map_state(s: Option<HelperState>) -> TtsHelperState {
    match s {
        Some(HelperState::Starting) | Some(HelperState::Restarting) => TtsHelperState::Starting,
        Some(HelperState::Running) => TtsHelperState::Ready,
        Some(HelperState::Failed) => TtsHelperState::Failed,
        Some(HelperState::Stopped) | None => TtsHelperState::Stopped,
    }
}

/// Follow the helper of generation `gen` until it stops or is replaced.
async fn watch(gen: u64) {
    loop {
        tokio::time::sleep(WATCH_INTERVAL).await;
        let (name, measure) = {
            let mut st = state();
            match st.current.as_mut() {
                Some(c) if c.gen == gen => {
                    c.rounds = c.rounds.wrapping_add(1);
                    let measure = c.launch.variant.on_gpu() && c.status.state == TtsHelperState::Ready && ((c.measured_mb == 0 && c.rounds < 30) || c.rounds.is_multiple_of(MEASURE_EVERY));
                    (c.launch.helper_name.clone(), measure)
                }
                _ => return,
            }
        };
        let report = measure.then(|| parse_load_report(&process_host::logs(&name)));
        let (changed, terminal, status, gpu_warning) = {
            let mut st = state();
            let Some(cur) = st.current.as_mut().filter(|c| c.gen == gen) else {
                return;
            };
            // Looked at under the state lock: a start's `mark_running` (which looks before it takes
            // the lock) is never overwritten by an older look ("starting" while it already runs).
            let info = process_host::get(&name);
            let before = cur.status.clone();
            let new_state = map_state(info.as_ref().map(|i| i.state));
            cur.status.state = new_state;
            let restarts = info.as_ref().map_or(0, |i| i.restarts);
            if note_restarts(cur, restarts) {
                // A restarted instance loads again: measure afresh and forget its voices.
                super::clone::forget_all();
            }
            cur.status.error = match new_state {
                TtsHelperState::Failed => info.as_ref().and_then(|i| i.error.clone()),
                _ => None,
            };
            let mut gpu_warning = None;
            if let Some(report) = report.filter(|_| new_state == TtsHelperState::Ready) {
                if report.backend.is_some() && !report.on_gpu() && cur.launch.variant.on_gpu() && cur.measured_mb == 0 {
                    // A GPU pack that found no usable GPU runs on the CPU: release the claim.
                    gpu_warning = report.backend.clone();
                    cur.measured_mb = 1;
                    if let Some(c) = &cur.claim {
                        c.resize(0);
                    }
                } else if let Some(mb) = measured_vram_mb(&report, cur.launch.variant).filter(|&mb| mb > cur.measured_mb) {
                    cur.measured_mb = mb;
                    if let Some(c) = &cur.claim {
                        c.resize(mb);
                    }
                    tracing::info!(target: TARGET, "{}: measured {mb} MiB of video memory (planned {} MiB)", cur.launch.model, cur.launch.vram_mb);
                }
            }
            let terminal = matches!(new_state, TtsHelperState::Failed | TtsHelperState::Stopped);
            let status = cur.status.clone();
            if terminal {
                let finished = st.current.take();
                st.last = Some(status.clone());
                drop(finished); // releases the claim
            }
            (status != before || terminal, terminal, status, gpu_warning)
        };
        if let Some(backend) = gpu_warning {
            tracing::warn!(target: TARGET, "the GPU build found no usable GPU and runs on {backend}: speech will be slow");
        }
        if changed {
            if status.state == TtsHelperState::Failed {
                tracing::warn!(target: TARGET, "Qwen3-TTS failed: {}", status.error.as_deref().unwrap_or("unknown error"));
                super::clone::forget_all();
            }
            emit(&status);
        }
        if terminal {
            return;
        }
    }
}

/// Test hook: rewrite the helper spec before it is spawned (the lifecycle tests run a fake server
/// built into the test binary).
#[cfg(test)]
pub mod test_hooks {
    use super::HelperSpec;
    use std::sync::Mutex;

    type Hook = Box<dyn Fn(HelperSpec) -> HelperSpec + Send + Sync>;
    static HOOK: Mutex<Option<Hook>> = Mutex::new(None);

    pub fn set(f: impl Fn(HelperSpec) -> HelperSpec + Send + Sync + 'static) {
        *HOOK.lock().unwrap() = Some(Box::new(f));
    }

    pub(super) fn apply(spec: HelperSpec) -> HelperSpec {
        match HOOK.lock().unwrap().as_ref() {
            Some(f) => f(spec),
            None => spec,
        }
    }
}
