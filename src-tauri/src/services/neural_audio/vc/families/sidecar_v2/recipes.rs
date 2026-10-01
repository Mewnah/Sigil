//! Helper-process recipes of the research sidecars (P5-E): X-VC, StreamVoiceAnon and
//! seed-vc-realtime run as Python servers from `scripts/neural-sidecar/` under `process_host`
//! (Job Object, `CREATE_NO_WINDOW`, a free `127.0.0.1` port, a random key in the environment, a
//! `/health` poll, restart on failure).
//!
//! The `HelperSpec`s are built here rather than in `process_host::recipes` (P5-S0 contract), and
//! one helper serves every voice of the same recipe, interpreter, model folder and arguments:
//! [`lease`] starts it or shares the running one (one [`HelperLease`] per helper, shared by its
//! voices, also across a restart of the process), and the last voice that drops it stops it. A
//! start waits in short slices and gives up as soon as the voice is no longer wanted.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use super::SidecarV2Options;
use crate::services::process_host::{self, HealthCheck, HelperPriority, HelperSpec, HelperState, PortRequest, RestartPolicy, StdioMode};
use crate::services::resources::{self, Subsystem, VramClaim};
use crate::services::voice_changer::factory::LOAD_CANCELLED;

/// The protocol module every sidecar script imports (it must sit next to the script).
pub const SHARED_MODULE: &str = "sigil_vc_v2.py";
/// Folder name of the sidecar scripts (in the repository: `scripts/neural-sidecar`).
pub const SCRIPTS_DIR: &str = "neural-sidecar";
/// Environment variable that carries the helper's key (never on the command line).
pub const KEY_ENV: &str = "SIGIL_SIDECAR_KEY";
/// Environment variable naming an extra scripts folder (developers, tests).
pub const SCRIPTS_ENV: &str = "SIGIL_SIDECAR_SCRIPTS";
/// The scripts' readiness route (public, like llama-server's).
pub const HEALTH_PATH: &str = "/health";
/// The scripts' WebSocket route.
pub const WS_PATH: &str = "/vc";
/// Default interpreter (looked up on PATH).
pub const DEFAULT_PYTHON: &str = "python";
/// Arguments a user may not pass: the bind address, the port and the key are Sigil's.
const RESERVED_ARGS: [&str; 5] = ["--host", "--port", "--key", "--api-key", "--allow-remote"];
/// How often a waiting start looks at the load's cancel flag.
const CANCEL_POLL: Duration = Duration::from_millis(100);
/// Margin over the recipe's load timeout for a start (process_host gives up at the health
/// timeout; this only guards against a start that hangs), and over a restart's backoff.
const START_SLACK: Duration = Duration::from_secs(30);
/// Restart policy of the recipe helpers (process_host: backoff `RESTART_BACKOFF_MS · 2^(n-1)`).
const RESTART_MAX: u32 = 2;
const RESTART_BACKOFF_MS: u64 = 2_000;

/// A research engine Sigil knows how to start.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Recipe {
    /// [`SidecarV2Options::recipe`].
    pub id: &'static str,
    /// Display name (logs, errors; UI strings come from i18n by `id`).
    pub label: &'static str,
    /// Script in the sidecar folder.
    pub script: &'static str,
    /// Frame proposed in the hello (the server answers with its own when it has one).
    pub frame_ms: f64,
    /// Converts toward a reference voice (a voice without one is refused before the helper starts).
    pub needs_reference: bool,
    /// Health timeout: interpreter start, imports and model load (several GB from disk).
    pub load_timeout_ms: u64,
    /// GPU memory claimed until the server reports its own figure (0 = none on the CPU).
    pub vram_hint_mb: u64,
    /// Runs on ONNX Runtime: on the CUDA rung the script gets the NVIDIA runtime pack's folders
    /// (`--dll-dir`), as Sigil's own CUDA sessions use them. PyTorch recipes bring their own.
    pub cuda_dll_dirs: bool,
}

/// The recipes ([`super::RECIPES`] without `custom`).
pub const TABLE: [Recipe; 3] = [
    Recipe {
        id: "xvc",
        label: "X-VC",
        script: "xvc_sidecar.py",
        // X-VC's chunkwise inference emits 120 ms per step (current 120, overlap 20, future 100).
        frame_ms: 120.0,
        needs_reference: true,
        load_timeout_ms: 300_000,
        vram_hint_mb: 5_000,
        cuda_dll_dirs: false,
    },
    Recipe {
        id: "streamvoiceanon",
        label: "StreamVoiceAnon",
        script: "streamvoiceanon_sidecar.py",
        // The authors' real-time setting: a 92 ms chunk (237 ms end to end on an RTX 3060 Laptop).
        frame_ms: 92.0,
        needs_reference: true,
        load_timeout_ms: 300_000,
        vram_hint_mb: 3_000,
        cuda_dll_dirs: false,
    },
    Recipe {
        id: "seedvc",
        label: "Seed-VC",
        script: "seedvc_sidecar.py",
        // Upstream real-time GUI default block (0.18 s).
        frame_ms: 180.0,
        needs_reference: true,
        load_timeout_ms: 180_000,
        vram_hint_mb: 1_500,
        cuda_dll_dirs: true,
    },
];

/// The recipe named `id` (`None` for `custom` and unknown names).
pub fn recipe(id: &str) -> Option<&'static Recipe> {
    TABLE.iter().find(|r| r.id == id)
}

/// Folders searched for the scripts, in order: [`SCRIPTS_ENV`], `app_data/voice-changer/sidecars`
/// (copies the user installed or edited), the bundled resources (`resources/neural-sidecar`
/// next to the executable), then, in debug builds only, the source tree.
pub fn script_dirs(app_data: Option<&Path>) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = std::env::var_os(SCRIPTS_ENV).filter(|d| !d.is_empty()) {
        dirs.push(PathBuf::from(d));
    }
    if let Some(a) = app_data {
        dirs.push(a.join("voice-changer").join("sidecars"));
    }
    if let Some(exe_dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        dirs.push(exe_dir.join("resources").join(SCRIPTS_DIR));
        dirs.push(exe_dir.join(SCRIPTS_DIR));
    }
    #[cfg(debug_assertions)]
    dirs.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("scripts").join(SCRIPTS_DIR));
    dirs
}

/// The script of a recipe (with [`SHARED_MODULE`] beside it) from [`script_dirs`].
pub fn find_script(recipe: &Recipe, app_data: Option<&Path>) -> Result<PathBuf, String> {
    script_dirs(app_data)
        .into_iter()
        .map(|d| d.join(recipe.script))
        .find(|p| p.is_file() && p.with_file_name(SHARED_MODULE).is_file())
        .ok_or_else(|| {
            format!(
                "the {} sidecar script ({} with {SHARED_MODULE}) is not installed: copy scripts/{SCRIPTS_DIR} into voice-changer/sidecars in Sigil's data folder",
                recipe.label, recipe.script
            )
        })
}

/// Validate the user's extra arguments (count, length, characters, reserved flags).
pub fn validate_args(args: &[String]) -> Result<(), String> {
    if args.len() > 64 || args.iter().any(|a| a.len() > 1024) {
        return Err("too many or too long sidecar arguments".into());
    }
    if args.iter().any(|a| a.chars().any(|c| c == '\0' || c == '\n' || c == '\r')) {
        return Err("sidecar arguments cannot contain line breaks or NUL".into());
    }
    if let Some(a) = args.iter().find(|a| {
        let flag = a.split('=').next().unwrap_or_default().to_ascii_lowercase();
        RESERVED_ARGS.contains(&flag.as_str())
    }) {
        return Err(format!("{a:?}: Sigil sets the sidecar's address, port and key itself"));
    }
    Ok(())
}

/// Validate an interpreter: empty (the default), a bare name looked up on PATH, or a full path.
pub fn validate_python(python: &str) -> Result<(), String> {
    if python.len() > 1024 || python.chars().any(|c| c.is_control()) {
        return Err("invalid Python interpreter path".into());
    }
    let has_dir = python.contains(['/', '\\']);
    if has_dir && !Path::new(python).is_absolute() {
        return Err("the Python interpreter must be a full path (or a name on PATH such as python)".into());
    }
    Ok(())
}

/// Validate a model folder: empty (the recipe's default) or a full path.
pub fn validate_model_dir(dir: &str) -> Result<(), String> {
    if dir.len() > 1024 || dir.chars().any(|c| c.is_control()) {
        return Err("invalid model folder".into());
    }
    if !dir.is_empty() && !Path::new(dir).is_absolute() {
        return Err("the model folder must be a full path".into());
    }
    Ok(())
}

/// The interpreter a recipe starts.
pub fn python_program(options: &SidecarV2Options) -> PathBuf {
    let p = options.python.trim();
    PathBuf::from(if p.is_empty() { DEFAULT_PYTHON } else { p })
}

/// Device flag for the script from the voice changer's EP rung.
fn device_arg(ep: &str) -> &'static str {
    match ep {
        "cuda" => "cuda",
        "dml" => "dml",
        _ => "cpu",
    }
}

/// Helper name: one helper per recipe, interpreter, script, model folder, arguments and device.
pub fn helper_name(recipe: &Recipe, options: &SidecarV2Options, script: &Path, ep: &str) -> String {
    let mut h = Sha256::new();
    for part in [
        python_program(options).display().to_string(),
        script.display().to_string(),
        options.model_dir.trim().to_string(),
        options.args.join("\u{1f}"),
        device_arg(ep).to_string(),
    ] {
        h.update(part.as_bytes());
        h.update([0u8]);
    }
    let digest = h.finalize();
    format!("vc-sidecar-{}-{}", recipe.id, digest[..4].iter().map(|b| format!("{b:02x}")).collect::<String>())
}

/// The helper spec of a recipe: `python <script> --host 127.0.0.1 --port {port} --device <ep>
/// [--model-dir <dir>] [args...]`, the key in [`KEY_ENV`], health [`HEALTH_PATH`].
pub fn helper_spec(recipe: &Recipe, options: &SidecarV2Options, script: &Path, ep: &str) -> HelperSpec {
    let mut args = vec![
        script.display().to_string(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        process_host::spawn::PORT_PLACEHOLDER.into(),
        "--device".into(),
        device_arg(ep).into(),
    ];
    let model_dir = options.model_dir.trim();
    if !model_dir.is_empty() {
        args.push("--model-dir".into());
        args.push(model_dir.into());
    }
    if recipe.cuda_dll_dirs && ep == "cuda" {
        for dir in nvidia_dll_dirs() {
            args.push("--dll-dir".into());
            args.push(dir.display().to_string());
        }
    }
    args.extend(options.args.iter().cloned());
    HelperSpec {
        name: helper_name(recipe, options, script, ep),
        program: python_program(options),
        args,
        env: vec![
            (KEY_ENV.into(), process_host::spawn::API_KEY_PLACEHOLDER.into()),
            ("PYTHONUNBUFFERED".into(), "1".into()),
            ("PYTHONIOENCODING".into(), "utf-8".into()),
        ],
        env_remove: vec![KEY_ENV.into()],
        cwd: script.parent().map(Path::to_path_buf),
        port: PortRequest::Auto,
        api_key: true,
        health: Some(HealthCheck {
            path: HEALTH_PATH.into(),
            timeout_ms: recipe.load_timeout_ms,
            interval_ms: 500,
            send_api_key: false,
        }),
        memory_limit_mb: None,
        restart: RestartPolicy::OnFailure {
            max_restarts: RESTART_MAX,
            backoff_ms: RESTART_BACKOFF_MS,
        },
        priority: HelperPriority::Normal,
        high_qos: true,
        stop_grace_ms: 0,
        stdio: StdioMode::Log,
    }
}

/// Folders of the installed NVIDIA runtime pack's DLLs (CUDA 13 / cuDNN 9), for ONNX Runtime
/// sidecars on the CUDA rung; empty when the pack is not installed.
fn nvidia_dll_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = crate::services::native_runtime::nvidia::store_dll_index()
        .map(|index| index.values().filter_map(|p| p.parent().map(Path::to_path_buf)).collect())
        .unwrap_or_default();
    dirs.sort();
    dirs.dedup();
    dirs
}

/// The WebSocket URL of a running helper.
pub fn helper_url(port: u16) -> String {
    format!("ws://127.0.0.1:{port}{WS_PATH}")
}

// ---------------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------------

/// Per helper name: the live lease and a generation counter (a stop scheduled by an old lease never
/// stops a newer helper of the same name).
#[derive(Default)]
struct Gate {
    generation: u64,
    live: Weak<HelperLease>,
}

impl Gate {
    /// Whether the lease of `generation`, just dropped, should stop the helper: it was the last
    /// one, and no newer lease adopted the helper meanwhile.
    fn should_stop(&self, generation: u64) -> bool {
        self.generation == generation && self.live.upgrade().is_none()
    }
}

/// Plain mutexes, never held across an `.await`: [`lease`] blocks (a loader thread) and the stop
/// after the last lease runs on its own short-lived thread, so neither ever runs inside an async
/// task (where a blocking lock would stall, or panic for tokio's mutex).
type GateMap = Mutex<HashMap<String, Arc<Mutex<Gate>>>>;

fn gates() -> &'static GateMap {
    static GATES: OnceLock<GateMap> = OnceLock::new();
    GATES.get_or_init(Default::default)
}

/// A voice's hold on a running helper. One per helper, shared by all of its voices (an `Arc`),
/// also across a restart of its process.
pub struct HelperLease {
    name: String,
    label: &'static str,
    generation: u64,
    gate: Arc<Mutex<Gate>>,
    vram: Mutex<Option<VramClaim>>,
    /// Stops the helper (process_host; tests record it).
    stop: fn(&str),
}

impl HelperLease {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Hold `mb` of GPU memory for the helper (resized when known better; one claim per helper).
    pub fn claim_vram(&self, mb: u64) {
        let mut v = self.vram.lock().unwrap_or_else(PoisonError::into_inner);
        match v.as_ref() {
            Some(c) => c.resize(mb),
            None => *v = Some(resources::claim_external_vram(Subsystem::Vc, &format!("voice changer sidecar ({})", self.label), mb)),
        }
    }
}

impl Drop for HelperLease {
    /// The last voice of a helper stops it (unless a newer lease of the same name adopted it in the
    /// meantime), from a thread of its own: the dropping thread (the voice changer's worker or
    /// loader) never waits for the process to exit.
    fn drop(&mut self) {
        let (name, generation, gate, stop) = (self.name.clone(), self.generation, self.gate.clone(), self.stop);
        let stopper = std::thread::Builder::new().name("sigil-vc-sidecar-stop".into()).spawn(move || {
            let g = gate.lock().unwrap_or_else(PoisonError::into_inner);
            if g.should_stop(generation) {
                stop(&name);
            }
        });
        if let Err(e) = stopper {
            tracing::warn!("voice changer: cannot stop sidecar helper {}: {e}", self.name);
        }
    }
}

/// How a bounded, cancellable wait ended.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Waited<T> {
    Done(T),
    TimedOut,
    Cancelled,
}

/// Wait for `rx` at most `wait`, in [`CANCEL_POLL`] slices, giving up when `cancel` is set.
pub(crate) fn wait_cancellable<T>(rx: &Receiver<T>, wait: Duration, cancel: Option<&AtomicBool>) -> Waited<T> {
    let deadline = Instant::now() + wait;
    loop {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Waited::Cancelled;
        }
        let now = Instant::now();
        if now >= deadline {
            return Waited::TimedOut;
        }
        match rx.recv_timeout((deadline - now).min(CANCEL_POLL)) {
            Ok(v) => return Waited::Done(v),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return Waited::TimedOut,
        }
    }
}

/// What [`lease`] needs from the process host (tests replace it).
trait Host {
    /// The helper `name` runs (healthy, with its port).
    fn running(&self, name: &str) -> bool;
    /// process_host is (re)starting `name` by itself (its restart policy after a crash).
    fn restarting(&self, name: &str) -> bool;
    /// Wait until `name` runs (`Done(true)`) or its restart ends otherwise (`Done(false)`).
    fn wait_running(&self, name: &str, wait: Duration, cancel: Option<&AtomicBool>) -> Waited<bool>;
    /// Start (or replace) the helper and wait for its health check.
    fn start(&self, spec: HelperSpec, wait: Duration, cancel: Option<&AtomicBool>) -> Waited<Result<(), String>>;
    /// The stop a lease runs (also when a start is abandoned).
    fn stopper(&self) -> fn(&str);
}

/// The real process host.
struct ProcessHost;

fn stop_helper(name: &str) {
    if let Err(e) = tauri::async_runtime::block_on(process_host::stop(name)) {
        tracing::warn!("voice changer: stopping sidecar helper {name}: {e}");
    }
}

impl Host for ProcessHost {
    fn running(&self, name: &str) -> bool {
        process_host::handle(name).is_some_and(|h| h.port.is_some())
    }

    fn restarting(&self, name: &str) -> bool {
        process_host::get(name).is_some_and(|i| matches!(i.state, HelperState::Starting | HelperState::Restarting))
    }

    fn wait_running(&self, name: &str, wait: Duration, cancel: Option<&AtomicBool>) -> Waited<bool> {
        let deadline = Instant::now() + wait;
        loop {
            if self.running(name) {
                return Waited::Done(true);
            }
            if !self.restarting(name) {
                return Waited::Done(false);
            }
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return Waited::Cancelled;
            }
            if Instant::now() >= deadline {
                return Waited::TimedOut;
            }
            std::thread::sleep(CANCEL_POLL);
        }
    }

    fn start(&self, spec: HelperSpec, wait: Duration, cancel: Option<&AtomicBool>) -> Waited<Result<(), String>> {
        let (tx, rx) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new().name("sigil-vc-sidecar-start".into()).spawn(move || {
            let started = tauri::async_runtime::block_on(process_host::spawn(spec));
            let _ = tx.send(started.and_then(|h| h.port.map(|_| ()).ok_or_else(|| "it has no port".to_string())));
        });
        if let Err(e) = spawned {
            return Waited::Done(Err(e.to_string()));
        }
        wait_cancellable(&rx, wait, cancel)
    }

    fn stopper(&self) -> fn(&str) {
        stop_helper
    }
}

/// Start `spec` (waiting for its health check, at most the recipe's load timeout) or share the
/// helper already running under its name. Blocking: call it on a loader thread. Returns
/// [`LOAD_CANCELLED`] soon after `cancel` is set (the start is called off unless another voice
/// holds the helper).
pub fn lease(spec: HelperSpec, recipe: &Recipe, cancel: Option<&AtomicBool>) -> Result<Arc<HelperLease>, String> {
    lease_on(&ProcessHost, spec, recipe, cancel)
}

fn lease_on(host: &dyn Host, spec: HelperSpec, recipe: &Recipe, cancel: Option<&AtomicBool>) -> Result<Arc<HelperLease>, String> {
    let name = spec.name.clone();
    let gate = gates()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry(name.clone())
        .or_default()
        .clone();
    let mut g = gate.lock().unwrap_or_else(PoisonError::into_inner);
    // The lease the helper's voices share, if any is alive: the helper keeps it across a restart
    // (a second lease would stop the helper under the first one's voices when it is dropped).
    let live = g.live.upgrade();
    let label = recipe.label;
    let new_lease = |g: &mut Gate| {
        g.generation += 1;
        let lease = Arc::new(HelperLease {
            name: name.clone(),
            label,
            generation: g.generation,
            gate: gate.clone(),
            vram: Mutex::new(None),
            stop: host.stopper(),
        });
        g.live = Arc::downgrade(&lease);
        lease
    };
    if host.running(&name) {
        // Share it, or adopt it (its last voice just let go: the stop that voice scheduled sees
        // the newer generation and leaves it running).
        return Ok(live.unwrap_or_else(|| new_lease(&mut g)));
    }
    let wait = Duration::from_millis(recipe.load_timeout_ms) + START_SLACK;
    if let Some(live) = &live {
        if host.restarting(&name) {
            // A voice holds the helper and process_host is restarting it after a crash: wait for
            // that rather than replacing it.
            tracing::info!("voice changer: waiting for the {} sidecar to restart ({name})", recipe.label);
            match host.wait_running(&name, wait + process_host_backoff(), cancel) {
                Waited::Done(true) => return Ok(live.clone()),
                Waited::Cancelled => return Err(LOAD_CANCELLED.to_string()),
                Waited::Done(false) | Waited::TimedOut => {}
            }
        }
    }
    tracing::info!("voice changer: starting the {} sidecar ({name})", recipe.label);
    // A start nobody needs any more is called off (process_host: the last request wins), unless a
    // voice holds the helper (it reconnects once the start finishes).
    let call_off = |why: &str| -> Result<Arc<HelperLease>, String> {
        if live.is_none() {
            (host.stopper())(&name);
        }
        Err(why.to_string())
    };
    match host.start(spec, wait, cancel) {
        Waited::Done(Ok(())) => Ok(live.clone().unwrap_or_else(|| new_lease(&mut g))),
        Waited::Done(Err(e)) => Err(format!("the {} sidecar did not start: {e}", recipe.label)),
        Waited::TimedOut => call_off(&format!("the {} sidecar did not start in time", recipe.label)),
        Waited::Cancelled => call_off(LOAD_CANCELLED),
    }
}

/// The longest backoff before a crashed recipe helper's restart (then it loads again).
fn process_host_backoff() -> Duration {
    Duration::from_millis(RESTART_BACKOFF_MS << (RESTART_MAX.max(1) - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(recipe: &str) -> SidecarV2Options {
        SidecarV2Options {
            recipe: recipe.into(),
            ..Default::default()
        }
    }

    #[test]
    fn every_recipe_is_a_known_option_with_a_script() {
        for r in TABLE {
            assert!(super::super::RECIPES.contains(&r.id), "{}", r.id);
            assert!(r.script.ends_with("_sidecar.py"));
            assert!(r.frame_ms >= 20.0 && r.frame_ms <= 500.0);
            assert_eq!(recipe(r.id), Some(&TABLE.iter().find(|x| x.id == r.id).copied().unwrap()));
        }
        assert!(recipe("custom").is_none() && recipe("nope").is_none());
    }

    /// The spec binds 127.0.0.1 on an auto port, keeps the key off the command line, and puts the
    /// user's arguments last (after Sigil's own flags).
    #[test]
    fn specs_bind_loopback_and_keep_the_key_in_the_environment() {
        let mut o = opts("xvc");
        o.model_dir = "D:/models/X-VC".into();
        o.args = vec!["--chunk-ms".into(), "2400".into()];
        let script = Path::new("C:/sigil/neural-sidecar/xvc_sidecar.py");
        let spec = helper_spec(recipe("xvc").unwrap(), &o, script, "cuda");
        assert!(spec.name.starts_with("vc-sidecar-xvc-") && spec.name.len() == "vc-sidecar-xvc-".len() + 8);
        process_host::spawn::validate_name(&spec.name).unwrap();
        assert_eq!(spec.program, Path::new("python"));
        let joined = spec.args.join(" ");
        assert!(
            joined.starts_with(&format!("{} --host 127.0.0.1 --port {{port}} --device cuda --model-dir D:/models/X-VC", script.display())),
            "{joined}"
        );
        assert!(joined.ends_with("--chunk-ms 2400"));
        assert!(!joined.contains("api_key") && !joined.contains("{api_key}"));
        assert_eq!(spec.env[0], (KEY_ENV.to_string(), "{api_key}".to_string()));
        assert!(spec.env_remove.contains(&KEY_ENV.to_string()));
        assert!(spec.api_key && spec.port == PortRequest::Auto && spec.high_qos);
        assert_eq!(spec.health.as_ref().unwrap().path, HEALTH_PATH);
        assert_eq!(spec.cwd.as_deref(), Some(Path::new("C:/sigil/neural-sidecar")));
        // The interpreter on PATH is structurally valid (no file check for a bare name).
        let err = process_host::spawn::validate(&spec).unwrap_err();
        assert!(err.contains("working directory"), "{err}");
    }

    #[test]
    fn helper_names_follow_what_the_helper_runs() {
        let script = Path::new("C:/s/seedvc_sidecar.py");
        let r = recipe("seedvc").unwrap();
        let a = helper_name(r, &opts("seedvc"), script, "cuda");
        assert_eq!(a, helper_name(r, &opts("seedvc"), script, "cuda"), "stable");
        assert_ne!(a, helper_name(r, &opts("seedvc"), script, "cpu"));
        let mut o = opts("seedvc");
        o.model_dir = "C:/m".into();
        assert_ne!(a, helper_name(r, &o, script, "cuda"));
        o.model_dir.clear();
        o.python = "C:/venv/python.exe".into();
        assert_ne!(a, helper_name(r, &o, script, "cuda"));
    }

    #[test]
    fn user_arguments_cannot_move_the_bind_or_the_key() {
        assert!(validate_args(&["--steps".into(), "10".into()]).is_ok());
        for bad in ["--host", "--HOST=0.0.0.0", "--port", "--api-key=x", "--allow-remote", "--key"] {
            assert!(validate_args(&[bad.into()]).is_err(), "{bad}");
        }
        assert!(validate_args(&["a\nb".into()]).is_err());
        assert!(validate_args(&vec!["x".to_string(); 65]).is_err());
        assert!(validate_python("").is_ok() && validate_python("python").is_ok() && validate_python("py").is_ok());
        assert!(validate_python("C:/venv/Scripts/python.exe").is_ok());
        assert!(validate_python("venv/python.exe").is_err());
        assert!(validate_model_dir("").is_ok() && validate_model_dir("D:/m").is_ok());
        assert!(validate_model_dir("models").is_err());
    }

    // --- Leases (a scripted process host) ------------------------------------------------------

    /// Helpers the fake host stopped.
    fn stopped() -> &'static Mutex<Vec<String>> {
        static STOPPED: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
        STOPPED.get_or_init(Default::default)
    }

    fn fake_stop(name: &str) {
        stopped().lock().unwrap().push(name.to_string());
    }

    fn was_stopped(name: &str) -> bool {
        stopped().lock().unwrap().iter().any(|n| n == name)
    }

    struct FakeHost {
        running: AtomicBool,
        restarting: AtomicBool,
        /// What a restart process_host is doing by itself ends with.
        restart_ok: AtomicBool,
        start: Mutex<Waited<Result<(), String>>>,
        starts: std::sync::atomic::AtomicUsize,
    }

    impl FakeHost {
        fn new() -> FakeHost {
            FakeHost {
                running: AtomicBool::new(false),
                restarting: AtomicBool::new(false),
                restart_ok: AtomicBool::new(true),
                start: Mutex::new(Waited::Done(Ok(()))),
                starts: Default::default(),
            }
        }

        fn starts(&self) -> usize {
            self.starts.load(Ordering::SeqCst)
        }
    }

    impl Host for FakeHost {
        fn running(&self, _name: &str) -> bool {
            self.running.load(Ordering::SeqCst)
        }
        fn restarting(&self, _name: &str) -> bool {
            self.restarting.load(Ordering::SeqCst)
        }
        fn wait_running(&self, _name: &str, _wait: Duration, _cancel: Option<&AtomicBool>) -> Waited<bool> {
            let ok = self.restart_ok.load(Ordering::SeqCst);
            self.running.store(ok, Ordering::SeqCst);
            self.restarting.store(false, Ordering::SeqCst);
            Waited::Done(ok)
        }
        fn start(&self, _spec: HelperSpec, _wait: Duration, _cancel: Option<&AtomicBool>) -> Waited<Result<(), String>> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            let outcome = self.start.lock().unwrap().clone();
            self.running.store(outcome == Waited::Done(Ok(())), Ordering::SeqCst);
            outcome
        }
        fn stopper(&self) -> fn(&str) {
            fake_stop
        }
    }

    fn named(tag: &str) -> HelperSpec {
        HelperSpec {
            name: format!("vc-sidecar-test-{tag}-{}", std::process::id()),
            ..Default::default()
        }
    }

    /// The voices of a helper share one lease, also after its process crashed and was started
    /// again (or restarted by process_host): letting go of one voice never stops the helper the
    /// others still use; the last one stops it.
    #[test]
    fn one_lease_is_shared_and_survives_a_restart_of_its_helper() {
        let host = FakeHost::new();
        let r = recipe("seedvc").unwrap();
        let spec = named("share");
        let name = spec.name.clone();
        let a = lease_on(&host, spec.clone(), r, None).unwrap();
        let b = lease_on(&host, spec.clone(), r, None).unwrap();
        assert!(Arc::ptr_eq(&a, &b), "voice B shares the running helper");
        assert_eq!(host.starts(), 1);
        // The helper crashed and process_host gave up: voice C starts it again under A's lease.
        host.running.store(false, Ordering::SeqCst);
        let c = lease_on(&host, spec.clone(), r, None).unwrap();
        assert!(Arc::ptr_eq(&a, &c), "no second lease for the same helper");
        assert_eq!((host.starts(), a.generation), (2, 1));
        // process_host is restarting it by itself: wait for that instead of replacing it.
        host.running.store(false, Ordering::SeqCst);
        host.restarting.store(true, Ordering::SeqCst);
        let d = lease_on(&host, spec.clone(), r, None).unwrap();
        assert!(Arc::ptr_eq(&a, &d));
        assert_eq!(host.starts(), 2, "the restart was waited for");
        drop((b, c, d));
        std::thread::sleep(Duration::from_millis(100));
        assert!(!was_stopped(&name), "voice A still uses the helper");
        drop(a);
        let end = Instant::now() + Duration::from_secs(5);
        while !was_stopped(&name) && Instant::now() < end {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(was_stopped(&name), "the last voice stops it");
    }

    /// A start nobody wants any more ends at once with the load's cancel message and is called
    /// off, unless a voice holds the helper (it reconnects when the start finishes).
    #[test]
    fn an_abandoned_start_is_called_off_unless_a_voice_holds_the_helper() {
        let r = recipe("xvc").unwrap();
        for (tag, outcome) in [("cancel", Waited::Cancelled), ("hang", Waited::TimedOut)] {
            let host = FakeHost::new();
            *host.start.lock().unwrap() = outcome.clone();
            let spec = named(&format!("alone-{tag}"));
            let err = lease_on(&host, spec.clone(), r, None).err().unwrap();
            assert!(if outcome == Waited::Cancelled { err == LOAD_CANCELLED } else { err.contains("in time") }, "{err}");
            assert!(was_stopped(&spec.name), "{tag}: nothing left starting without a voice");

            let host = FakeHost::new();
            let spec = named(&format!("held-{tag}"));
            let held = lease_on(&host, spec.clone(), r, None).unwrap();
            host.running.store(false, Ordering::SeqCst);
            *host.start.lock().unwrap() = outcome.clone();
            assert!(lease_on(&host, spec.clone(), r, None).is_err());
            assert!(!was_stopped(&spec.name), "{tag}: the held helper's start goes on");
            drop(held);
        }
        // A failed start reports why.
        let host = FakeHost::new();
        *host.start.lock().unwrap() = Waited::Done(Err("python: not found".into()));
        let err = lease_on(&host, named("fail"), r, None).err().unwrap();
        assert!(err.contains("did not start") && err.contains("python: not found"), "{err}");
    }

    /// A lease dropped while a newer one adopted its running helper leaves it running.
    #[test]
    fn a_stop_scheduled_by_an_old_lease_spares_an_adopted_helper() {
        let mut g = Gate { generation: 1, live: Weak::new() };
        assert!(g.should_stop(1), "the last lease stops its helper");
        g.generation = 2;
        assert!(!g.should_stop(1), "adopted by a newer lease");
        let host = FakeHost::new();
        host.running.store(true, Ordering::SeqCst);
        let spec = named("adopt");
        let first = lease_on(&host, spec.clone(), recipe("seedvc").unwrap(), None).unwrap();
        assert_eq!(host.starts(), 0, "a running helper is adopted, not restarted");
        let old = first.generation;
        let gate = first.gate.clone();
        drop(first);
        let second = lease_on(&host, spec.clone(), recipe("seedvc").unwrap(), None).unwrap();
        assert_eq!(host.starts(), 0);
        assert!(second.generation > old);
        assert!(!gate.lock().unwrap().should_stop(old), "the old lease's stop spares the adopted helper");
    }

    #[test]
    fn waits_end_on_the_value_the_deadline_or_a_cancel() {
        let (tx, rx) = std::sync::mpsc::channel::<u8>();
        let t0 = Instant::now();
        assert_eq!(wait_cancellable(&rx, Duration::from_millis(150), None), Waited::TimedOut);
        assert!(t0.elapsed() >= Duration::from_millis(150));
        let flag = Arc::new(AtomicBool::new(false));
        let setter = flag.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            setter.store(true, Ordering::Relaxed);
        });
        let t0 = Instant::now();
        assert_eq!(wait_cancellable(&rx, Duration::from_secs(60), Some(&flag)), Waited::Cancelled);
        assert!(t0.elapsed() < Duration::from_secs(2), "{:?}", t0.elapsed());
        tx.send(7).unwrap();
        assert_eq!(wait_cancellable(&rx, Duration::from_secs(60), None), Waited::Done(7));
    }

    /// The scripts are found next to their shared module (here: the source tree of a debug build,
    /// or the folder `SIGIL_SIDECAR_SCRIPTS` names).
    #[test]
    fn scripts_are_found_with_their_shared_module() {
        let dir = std::env::temp_dir().join(format!("sigil-sidecar-scripts-{}", std::process::id()));
        let app_data = dir.join("app");
        let sidecars = app_data.join("voice-changer").join("sidecars");
        std::fs::create_dir_all(&sidecars).unwrap();
        std::fs::write(sidecars.join("xvc_sidecar.py"), "").unwrap();
        let r = recipe("xvc").unwrap();
        // Without the shared module the copy does not count.
        let found = find_script(r, Some(&app_data));
        assert!(found.as_ref().map_or(true, |p| !p.starts_with(&sidecars)), "{found:?}");
        std::fs::write(sidecars.join(SHARED_MODULE), "").unwrap();
        assert_eq!(find_script(r, Some(&app_data)).unwrap(), sidecars.join("xvc_sidecar.py"));
        let _ = std::fs::remove_dir_all(&dir);
        #[cfg(debug_assertions)]
        for r in TABLE {
            let p = find_script(&r, None).unwrap_or_else(|e| panic!("{e}"));
            assert!(p.ends_with(r.script));
        }
    }
}
