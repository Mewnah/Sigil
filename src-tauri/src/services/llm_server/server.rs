//! Lifecycle of the one shared llama-server: resolve a config to a model file, a runtime pack and a
//! governor-approved plan; start it through `process_host` (replacing a server with another launch,
//! reusing one with the same launch); watch it (crash, restart, port change); hold its VRAM claim;
//! stop it. Status changes are emitted as [`STATE_EVENT`].
//!
//! Concurrency: starts are serialized by an async gate (a second caller with the same config gets
//! the running server). A stop never waits for the gate: it bumps the stop epoch and stops the
//! helper, which cancels a start that is still loading (process_host's last-request-wins); a start
//! requested before a stop returns `cancelled`. Every start and stop bumps a generation, so a
//! watcher or a start that finishes late never touches a newer server's state. A stop that lands
//! after a start published its state but before process_host took the start's ticket cannot stop
//! the helper (it does not exist yet); the start then finds itself superseded and stops the helper
//! it launched, so no server outlives a stop.

use super::catalog::{self, Availability};
use super::devices::{self, Pin};
use super::plan::{self, Knobs, ModelFacts, Planned, ALL_LAYERS};
use super::{
    gguf, LlmModelInfo, LlmServerConfig, LlmServerEndpoint, LlmServerError, LlmServerErrorKind, LlmServerPlan, LlmServerStarted, LlmServerState,
    LlmServerStatus, HELPER_NAME, MODEL_AUTO, STATE_EVENT,
};
use crate::services::model_store::{self, Catalog};
use crate::services::process_host::{
    self,
    recipes::{self, LlamaServerOptions},
    HelperLogLine, HelperPriority, HelperSpec, HelperState,
};
use crate::services::resources::{self, policy, RecommendationTopic, ResourceSnapshot, Subsystem, VramClaim};
use crate::services::{diagnostics, sysinfo, sysinfo::SystemInfo};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

const TARGET: &str = "sigil::llm_server";
/// How often the watcher looks at the helper.
const WATCH_INTERVAL: Duration = Duration::from_millis(400);
/// Log lines returned by `llm_server_logs` without a limit.
const DEFAULT_LOG_LINES: usize = 200;

fn err(kind: LlmServerErrorKind, message: impl Into<String>) -> LlmServerError {
    LlmServerError::new(kind, message)
}

// ---------------------------------------------------------------------------------------------
// Resolution: config → model facts → plan
// ---------------------------------------------------------------------------------------------

/// A config with its values checked.
pub fn normalize(config: &LlmServerConfig) -> Result<(String, Knobs), LlmServerError> {
    let model = match config.model.trim() {
        "" => MODEL_AUTO.to_string(),
        m => m.to_string(),
    };
    let variant = match config.variant.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => "auto".to_string(),
        v @ ("vulkan" | "cuda13" | "cpu") => v.to_string(),
        other => {
            return Err(err(
                LlmServerErrorKind::InvalidInput,
                format!("unknown runtime variant {other:?} (auto, vulkan, cuda13 or cpu)"),
            ))
        }
    };
    let flash_attn = match config
        .flash_attn
        .as_deref()
        .map(|s| s.trim().to_ascii_lowercase())
    {
        None => "auto".to_string(),
        Some(s) if s.is_empty() => "auto".to_string(),
        Some(s) if matches!(s.as_str(), "auto" | "on" | "off") => s,
        Some(s) => {
            return Err(err(
                LlmServerErrorKind::InvalidInput,
                format!("flash attention must be auto, on or off (got {s:?})"),
            ))
        }
    };
    Ok((
        model,
        Knobs {
            variant,
            ctx_size: config.ctx_size.filter(|&c| c > 0),
            gpu_layers: config.gpu_layers.map(|n| n.min(ALL_LAYERS)),
            flash_attn,
            allow_over_budget: config.allow_over_budget,
        },
    ))
}

/// The catalog model `auto` means now: the governor's choice, or the best runnable alternative.
pub fn auto_model(catalog: &Catalog, sys: &SystemInfo, snap: &ResourceSnapshot) -> String {
    let rec = resources::recommend(RecommendationTopic::MtLlm, sys, snap);
    std::iter::once(rec.choice.as_str())
        .chain(rec.alternatives.iter().map(String::as_str))
        .chain(std::iter::once(policy::ids::MT_HY_MT2_1_8B))
        .find(|id| catalog::entry(catalog, id).is_some() && catalog::meta(id).is_none_or(|m| m.availability == Availability::Ok))
        .unwrap_or(policy::ids::MT_HY_MT2_1_8B)
        .to_string()
}

/// Facts of a user-supplied GGUF (validated: absolute, present, a GGUF, the first part of a split).
pub fn user_model_facts(path_text: &str) -> Result<(ModelFacts, PathBuf), LlmServerError> {
    let path = PathBuf::from(path_text.trim());
    if !path.is_absolute() {
        return Err(err(LlmServerErrorKind::InvalidInput, "a model file must be given as a full path"));
    }
    let meta = std::fs::metadata(&path).map_err(|_| err(LlmServerErrorKind::InvalidInput, format!("model file not found: {}", path.display())))?;
    if !meta.is_file() {
        return Err(err(LlmServerErrorKind::InvalidInput, format!("not a file: {}", path.display())));
    }
    let info = gguf::read(&path).map_err(|e| err(LlmServerErrorKind::InvalidInput, format!("{}: {e}", path.display())))?;
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let split = info.split_count.unwrap_or(1);
    if split > 1 && !name.contains("-00001-of-") {
        return Err(err(
            LlmServerErrorKind::InvalidInput,
            "this model is split into several files: choose the one ending in -00001-of-…",
        ));
    }
    Ok((
        ModelFacts {
            id: path.display().to_string(),
            label: name,
            catalog: false,
            availability: if info.has_encoder {
                Availability::Experimental
            } else {
                Availability::Ok
            },
            size_bytes: meta.len().saturating_mul(u64::from(split.max(1))),
            vram_default_ctx_mb: None,
            kv_bytes_per_token: info.kv_bytes_per_token(),
            layers: info.block_count,
            ctx_train: info.context_length,
            vocab_size: info.vocab_size,
            installed: true,
        },
        path,
    ))
}

/// Facts of a catalog model (refined from its GGUF header when installed).
pub fn catalog_model_facts(catalog: &Catalog, id: &str, install_dir: Option<&Path>) -> Result<(ModelFacts, Option<PathBuf>), LlmServerError> {
    let entry = catalog::entry(catalog, id).ok_or_else(|| {
        err(
            LlmServerErrorKind::InvalidInput,
            format!("unknown translation model {id:?} (a catalog id, \"auto\" or a full path to a .gguf file)"),
        )
    })?;
    let meta = catalog::meta(id).cloned().unwrap_or_default();
    let model_path = install_dir
        .and_then(|d| catalog::model_path(d, entry))
        .filter(|p| p.is_file());
    let header = model_path.as_deref().and_then(|p| {
        gguf::read(p)
            .map_err(|e| tracing::warn!(target: TARGET, "{id}: cannot read the GGUF header ({e}); using catalog estimates"))
            .ok()
    });
    let mut facts = ModelFacts {
        id: id.to_string(),
        label: entry.title.clone().unwrap_or_else(|| id.to_string()),
        catalog: true,
        availability: meta.availability,
        size_bytes: catalog::gguf_bytes(entry),
        vram_default_ctx_mb: entry.min_vram_mb,
        kv_bytes_per_token: meta.kv_bytes_per_token,
        installed: model_path.is_some(),
        ..Default::default()
    };
    if let Some(h) = header {
        facts.kv_bytes_per_token = facts.kv_bytes_per_token.or(h.kv_bytes_per_token());
        facts.layers = h.block_count;
        facts.ctx_train = h.context_length;
        facts.vocab_size = h.vocab_size;
    }
    Ok((facts, model_path))
}

/// Everything decided before launching.
#[derive(Debug)]
pub struct Resolved {
    pub facts: ModelFacts,
    pub planned: Planned,
    pub model_path: Option<PathBuf>,
    pub runtime_dir: Option<PathBuf>,
    pub sys: SystemInfo,
}

/// Resolve a config against this PC (blocking: system probe on first use, file reads).
pub fn resolve(config: &LlmServerConfig) -> Result<Resolved, LlmServerError> {
    let catalog = Catalog::load().map_err(|e| err(LlmServerErrorKind::Runtime, e))?;
    let sys = sysinfo::cached();
    let snap = resources::snapshot();
    resolve_with(config, &catalog, sys, &snap, &|id| model_store::installed_path(id))
}

/// [`resolve`] with explicit inputs (tests).
pub fn resolve_with(
    config: &LlmServerConfig,
    catalog: &Catalog,
    sys: SystemInfo,
    snap: &ResourceSnapshot,
    installed: &dyn Fn(&str) -> Option<PathBuf>,
) -> Result<Resolved, LlmServerError> {
    let (model, knobs) = normalize(config)?;
    let model = if model == MODEL_AUTO { auto_model(catalog, &sys, snap) } else { model };
    let (facts, model_path) = if catalog::looks_like_path(&model) {
        let (f, p) = user_model_facts(&model)?;
        (f, Some(p))
    } else {
        catalog_model_facts(catalog, &model, installed(&model).as_deref())?
    };
    let runtime_installed = |id: &str| installed(id).is_some();
    let planned = plan::plan(
        &knobs,
        &facts,
        &plan::Env {
            sys: &sys,
            snap,
            runtime_installed: &runtime_installed,
        },
    );
    let runtime_dir = catalog::runtime_id(&planned.plan.variant).and_then(installed);
    Ok(Resolved {
        facts,
        planned,
        model_path,
        runtime_dir,
        sys,
    })
}

// ---------------------------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------------------------

/// Everything the helper is started with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    pub helper_name: String,
    pub exe: PathBuf,
    pub model_path: PathBuf,
    /// Reported as the status `model`: the catalog id, or the file name of a user GGUF.
    pub model: String,
    pub label: String,
    pub alias: String,
    pub variant: String,
    pub gpu_layers: u32,
    pub ctx_size: u32,
    pub flash_attn: String,
    pub threads: u32,
    pub no_poll: bool,
    pub below_normal: bool,
    pub pin: Pin,
    /// VRAM claimed while it runs (MiB).
    pub vram_mb: u64,
    pub load_timeout_ms: u64,
}

impl Launch {
    /// Two launches with the same key run the same server.
    fn same_server(&self, other: &Launch) -> bool {
        (
            &self.helper_name,
            &self.exe,
            &self.model_path,
            &self.variant,
            self.gpu_layers,
            self.ctx_size,
            &self.flash_attn,
            &self.pin,
        ) == (
            &other.helper_name,
            &other.exe,
            &other.model_path,
            &other.variant,
            other.gpu_layers,
            other.ctx_size,
            &other.flash_attn,
            &other.pin,
        )
    }
}

/// Model loading budget: a minute plus 30 s per GB (slow disks, big models), at most 10 minutes.
pub fn load_timeout_ms(size_bytes: u64) -> u64 {
    let gb = size_bytes.div_ceil(1 << 30);
    60_000u64
        .saturating_add(gb.saturating_mul(30_000))
        .min(600_000)
}

/// Build the launch for a resolved, allowed plan (blocking: may list the runtime's GPU devices).
pub fn make_launch(r: &Resolved) -> Result<Launch, LlmServerError> {
    let p = &r.planned.plan;
    let model_path = r
        .model_path
        .clone()
        .ok_or_else(|| err(LlmServerErrorKind::NotInstalled, format!("{} is not installed", r.facts.label)))?;
    let runtime_dir = r.runtime_dir.clone().ok_or_else(|| {
        let mut e = err(
            LlmServerErrorKind::NotInstalled,
            format!("the llama.cpp {} runtime is not installed", p.variant),
        );
        e.missing = catalog::runtime_id(&p.variant)
            .map(|s| vec![s.to_string()])
            .unwrap_or_default();
        e
    })?;
    let exe = recipes::llama_server_exe(&runtime_dir);
    let on_gpu = p.gpu_layers > 0;
    let pin = if on_gpu && devices::hardware_gpu_count(&r.sys) > 1 && exe.is_file() {
        match devices::list_devices(&exe) {
            Ok(list) => devices::pick_device(
                &list,
                r.sys
                    .primary()
                    .map(|g| g.adapter.name.as_str())
                    .unwrap_or(""),
            ),
            Err(e) => {
                tracing::warn!(target: TARGET, "cannot list llama.cpp's GPU devices ({e}); keeping the model on one device");
                Pin::SingleDevice
            }
        }
    } else {
        Pin::None
    };
    Ok(Launch {
        helper_name: HELPER_NAME.to_string(),
        exe,
        model: if r.facts.catalog { r.facts.id.clone() } else { r.facts.label.clone() },
        label: r.facts.label.clone(),
        alias: catalog::alias_for(&r.facts.id),
        model_path,
        variant: p.variant.clone(),
        gpu_layers: p.gpu_layers,
        ctx_size: p.ctx_size,
        flash_attn: p.flash_attn.clone(),
        threads: r.planned.threads,
        no_poll: r.planned.no_poll,
        below_normal: r.planned.below_normal,
        pin,
        vram_mb: if on_gpu { p.vram_mb } else { 0 },
        load_timeout_ms: load_timeout_ms(r.facts.size_bytes),
    })
}

/// The helper spec of a launch (pure). Flags beyond the recipe (all checked against b11146's
/// `--help`): no web UI, no automatic fitting (the plan decides), no chain-of-thought (translation),
/// plain log lines, no busy-polling CPU threads on full GPU offload, and the device pin.
pub fn build_spec(l: &Launch) -> HelperSpec {
    let mut extra: Vec<String> = ["--no-webui", "-fit", "off", "--reasoning", "off", "--log-colors", "off"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    if l.no_poll {
        extra.extend(["--poll".to_string(), "0".to_string()]);
    }
    if l.gpu_layers == 0 {
        extra.extend(["--device".to_string(), "none".to_string()]);
    } else {
        match &l.pin {
            Pin::Device(d) => extra.extend(["--device".to_string(), d.clone()]),
            Pin::SingleDevice => extra.extend(["-sm".to_string(), "none".to_string()]),
            Pin::None => {}
        }
    }
    let opts = LlamaServerOptions {
        name: l.helper_name.clone(),
        ctx_size: Some(l.ctx_size),
        gpu_layers: Some(l.gpu_layers),
        flash_attn: Some(l.flash_attn.clone()),
        parallel: Some(1),
        threads: Some(l.threads),
        alias: Some(l.alias.clone()),
        extra_args: extra,
        load_timeout_ms: l.load_timeout_ms,
        priority: if l.below_normal {
            HelperPriority::BelowNormal
        } else {
            HelperPriority::Normal
        },
        memory_limit_mb: None,
    };
    let mut spec = recipes::llama_server(&l.exe, &l.model_path, &opts);
    if let Some(h) = spec.health.as_mut() {
        // The port's owner is verified before the first probe, so the key never reaches another
        // program; sending it keeps readiness working even if a build protects /health.
        h.send_api_key = true;
    }
    spec
}

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------

struct Current {
    gen: u64,
    launch: Launch,
    status: LlmServerStatus,
    claim: Option<VramClaim>,
    measured: bool,
    /// Watcher rounds spent looking for the buffer sizes in the log (bounded).
    measure_tries: u32,
}

/// Watcher rounds (about 12 s) to find the logged buffer sizes after the server is ready.
const MAX_MEASURE_TRIES: u32 = 30;

#[derive(Default)]
struct State {
    /// Bumped by every start and stop.
    gen: u64,
    /// Bumped by every stop.
    stops: u64,
    current: Option<Current>,
    /// Shown while nothing is current (the last failure, or stopped).
    last: LlmServerStatus,
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

fn emit(status: &LlmServerStatus) {
    diagnostics::emit(STATE_EVENT, status);
}

fn base_status(l: &Launch, state: LlmServerState, vram_mb: u64) -> LlmServerStatus {
    LlmServerStatus {
        state,
        model: Some(l.model.clone()),
        variant: Some(l.variant.clone()),
        gpu_layers: Some(l.gpu_layers),
        ctx_size: Some(l.ctx_size),
        flash_attn: Some(l.flash_attn.clone()),
        base_url: None,
        vram_mb,
        restarts: 0,
        error: None,
    }
}

/// The current status (never contains the key).
pub fn status() -> LlmServerStatus {
    let st = state();
    match &st.current {
        Some(c) => c.status.clone(),
        None => st.last.clone(),
    }
}

/// Where to reach the running server.
pub fn endpoint() -> Option<LlmServerEndpoint> {
    let (name, alias) = {
        let st = state();
        let c = st.current.as_ref()?;
        if c.status.state != LlmServerState::Ready {
            return None;
        }
        (c.launch.helper_name.clone(), c.launch.alias.clone())
    };
    let h = process_host::handle(&name)?;
    Some(LlmServerEndpoint {
        base_url: format!("{}/v1", h.base_url?),
        api_key: h.api_key?,
        model: alias,
    })
}

/// The last `limit` lines of the server's output (key redacted by process_host).
pub fn logs(limit: Option<u32>) -> Vec<HelperLogLine> {
    let name = state()
        .current
        .as_ref()
        .map(|c| c.launch.helper_name.clone())
        .unwrap_or_else(|| HELPER_NAME.to_string());
    let mut lines = process_host::logs(&name);
    let limit = limit
        .map(|n| n.clamp(1, 1000) as usize)
        .unwrap_or(DEFAULT_LOG_LINES);
    if lines.len() > limit {
        lines.drain(..lines.len() - limit);
    }
    lines
}

// ---------------------------------------------------------------------------------------------
// Start / stop
// ---------------------------------------------------------------------------------------------

fn join_error(e: impl std::fmt::Display) -> LlmServerError {
    err(LlmServerErrorKind::Runtime, format!("llm-server task failed: {e}"))
}

/// Dry run (`llm_server_plan`).
pub async fn plan(config: LlmServerConfig) -> Result<LlmServerPlan, LlmServerError> {
    tauri::async_runtime::spawn_blocking(move || resolve(&config).map(|r| r.planned.plan))
        .await
        .map_err(join_error)?
}

/// The stop counter: a start that read an older value is cancelled.
pub fn stop_epoch() -> u64 {
    state().stops
}

/// The launch for a resolved plan, or why it cannot start: a refusal (unless the user accepted an
/// overridable one with `allow_over_budget`; the refused plan carries the GPU placement it refused,
/// so the override runs that placement with its VRAM claim), then missing downloads (blocking).
pub fn prepare_launch(r: &Resolved, allow_over_budget: bool) -> Result<Launch, LlmServerError> {
    let p = &r.planned;
    if !p.plan.allowed && (p.hard_refusal || !allow_over_budget) {
        let mut e = err(LlmServerErrorKind::Refused, p.plan.reason.clone());
        e.missing = p.plan.missing.clone();
        return Err(e);
    }
    if !p.plan.missing.is_empty() {
        let mut e = err(
            LlmServerErrorKind::NotInstalled,
            format!("{} needs downloads first: {}", r.facts.label, p.plan.missing.join(", ")),
        );
        e.missing = p.plan.missing.clone();
        return Err(e);
    }
    if p.plan.allowed {
        tracing::info!(target: TARGET, "plan: {}", p.plan.reason);
    } else {
        tracing::warn!(target: TARGET, "plan refused, started anyway at the user's request: {}", p.plan.reason);
    }
    make_launch(r)
}

/// Start (or reuse) the server for `config` (`llm_server_start`).
pub async fn start(config: LlmServerConfig) -> Result<LlmServerStarted, LlmServerError> {
    let stops_at_call = stop_epoch();
    let _gate = gate().lock().await;
    if state().stops != stops_at_call {
        return Err(err(LlmServerErrorKind::Cancelled, "cancelled: the server was stopped"));
    }
    let cfg = config.clone();
    let launch = tauri::async_runtime::spawn_blocking(move || -> Result<Launch, LlmServerError> {
        let r = resolve(&cfg)?;
        prepare_launch(&r, cfg.allow_over_budget)
    })
    .await
    .map_err(join_error)??;
    start_launch(launch, stops_at_call).await
}

/// How a spawn ended, decided under the state lock (acted on after it is released).
enum SpawnOutcome {
    Ready(LlmServerStatus, LlmServerEndpoint),
    /// Ready, but without a port or key (never for llama-server): stopped again.
    NoEndpoint(LlmServerStatus),
    Failed(LlmServerStatus, String),
    /// A stop (or a replacement) came while it started; the helper may run anyway.
    Superseded { started: bool },
}

/// Start `launch` unless the running server already is it. Caller holds the start gate (tests call
/// it directly with their own launch).
pub(super) async fn start_launch(launch: Launch, stops_at_call: u64) -> Result<LlmServerStarted, LlmServerError> {
    // Reuse the running server.
    let reuse = {
        let st = state();
        st.current
            .as_ref()
            .filter(|c| c.launch.same_server(&launch) && c.status.state == LlmServerState::Ready)
            .map(|c| (c.status.clone(), c.launch.helper_name.clone(), c.launch.alias.clone()))
    };
    if let Some((status, name, alias)) = reuse {
        let running = process_host::get(&name).is_some_and(|i| i.state == HelperState::Running);
        if let (true, Some(h)) = (running, process_host::handle(&name)) {
            if let (Some(base), Some(key)) = (h.base_url, h.api_key) {
                return Ok(LlmServerStarted {
                    status,
                    endpoint: LlmServerEndpoint {
                        base_url: format!("{base}/v1"),
                        api_key: key,
                        model: alias,
                    },
                });
            }
        }
    }

    // Retire whatever runs now (its VRAM is freed before the new model loads).
    let old = {
        let mut st = state();
        st.gen += 1;
        st.current.take()
    };
    if let Some(old) = old {
        tracing::info!(target: TARGET, "replacing {} ({})", old.launch.label, old.launch.variant);
        if let Err(e) = process_host::stop(&old.launch.helper_name).await {
            tracing::warn!(target: TARGET, "stopping the previous server: {e}");
        }
        drop(old);
    }

    let (gen, starting) = {
        let mut st = state();
        if st.stops != stops_at_call {
            return Err(err(LlmServerErrorKind::Cancelled, "cancelled: the server was stopped"));
        }
        st.gen += 1;
        let claim =
            (launch.vram_mb > 0).then(|| resources::claim_external_vram(Subsystem::Llm, &format!("llama-server: {}", launch.label), launch.vram_mb));
        let status = base_status(&launch, LlmServerState::Starting, launch.vram_mb);
        st.current = Some(Current {
            gen: st.gen,
            launch: launch.clone(),
            status: status.clone(),
            claim,
            measured: false,
            measure_tries: 0,
        });
        (st.gen, status)
    };
    emit(&starting);

    let spec = build_spec(&launch);
    #[cfg(test)]
    let spec = test_hooks::apply(spec);
    #[cfg(test)]
    test_hooks::run_before_spawn().await;
    let result = process_host::spawn(spec).await;

    // Decide under the lock, act (emit, stop a helper) after releasing it.
    let (outcome, released) = {
        let mut st = state();
        let still_ours = st.gen == gen && st.current.as_ref().is_some_and(|c| c.gen == gen);
        match result {
            Ok(handle) if still_ours => match (handle.base_url.clone(), handle.api_key.clone()) {
                (Some(base), Some(key)) => {
                    let cur = st.current.as_mut().expect("checked above");
                    cur.status.state = LlmServerState::Ready;
                    cur.status.base_url = Some(format!("{base}/v1"));
                    let endpoint = LlmServerEndpoint {
                        base_url: format!("{base}/v1"),
                        api_key: key,
                        model: launch.alias.clone(),
                    };
                    (SpawnOutcome::Ready(cur.status.clone(), endpoint), None)
                }
                _ => {
                    let failed = st.current.take();
                    let mut status = base_status(&launch, LlmServerState::Failed, 0);
                    status.error = Some("llama-server started without a port or key".into());
                    st.last = status.clone();
                    (SpawnOutcome::NoEndpoint(status), failed)
                }
            },
            Err(e) if still_ours => {
                let failed = st.current.take();
                let mut status = base_status(&launch, LlmServerState::Failed, 0);
                status.error = Some(e.clone());
                st.last = status.clone();
                (SpawnOutcome::Failed(status, e), failed)
            }
            Ok(_) => (SpawnOutcome::Superseded { started: true }, None),
            Err(_) => (SpawnOutcome::Superseded { started: false }, None),
        }
    };

    match outcome {
        SpawnOutcome::Ready(status, endpoint) => {
            emit(&status);
            tracing::info!(target: TARGET, "ready: {} ({}, {} GPU layers, ctx {})", launch.label, launch.variant, launch.gpu_layers, launch.ctx_size);
            tauri::async_runtime::spawn(watch(gen));
            Ok(LlmServerStarted { status, endpoint })
        }
        SpawnOutcome::NoEndpoint(status) => {
            if let Err(e) = process_host::stop(&launch.helper_name).await {
                tracing::warn!(target: TARGET, "stopping a server without an endpoint: {e}");
            }
            drop(released); // the VRAM claim goes once the process is gone
            emit(&status);
            Err(err(LlmServerErrorKind::StartFailed, "llama-server started without a port or key"))
        }
        SpawnOutcome::Failed(status, e) => {
            drop(released);
            emit(&status);
            tracing::warn!(target: TARGET, "llama-server failed to start: {e}");
            Err(err(LlmServerErrorKind::StartFailed, e))
        }
        SpawnOutcome::Superseded { started } => {
            if started {
                // A stop landed between publishing this start and process_host taking its ticket:
                // that stop found nothing to stop, and the helper started anyway. Starts are
                // serialized by the gate, so only a stop can have moved the generation; stop the
                // helper this start launched (no watcher and no VRAM claim would follow it).
                tracing::info!(target: TARGET, "{} became ready after a stop; stopping it", launch.label);
                if let Err(e) = process_host::stop(&launch.helper_name).await {
                    tracing::warn!(target: TARGET, "stopping a server that was started after a stop: {e}");
                }
            }
            Err(err(
                LlmServerErrorKind::Cancelled,
                "cancelled: the server was stopped or replaced while it started",
            ))
        }
    }
}

/// Stop the server (`llm_server_stop`); also cancels a start that is still loading.
pub async fn stop() -> Result<(), LlmServerError> {
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
            st.last = LlmServerStatus::default();
        }
        match &st.current {
            Some(c) => c.status.clone(),
            None => st.last.clone(),
        }
    };
    emit(&status);
    result.map_err(|e| err(LlmServerErrorKind::Runtime, e))
}

fn map_state(s: Option<HelperState>) -> LlmServerState {
    match s {
        Some(HelperState::Starting) => LlmServerState::Starting,
        Some(HelperState::Running) => LlmServerState::Ready,
        Some(HelperState::Restarting) => LlmServerState::Restarting,
        Some(HelperState::Failed) => LlmServerState::Failed,
        Some(HelperState::Stopped) | None => LlmServerState::Stopped,
    }
}

/// Runtime overhead added to the logged buffers.
fn runtime_base_mb(variant: &str) -> u64 {
    match variant {
        "cuda13" => devices::CUDA_BASE_MB,
        "cpu" => 0,
        _ => devices::VULKAN_BASE_MB,
    }
}

/// Follow the helper of generation `gen` until it stops or is replaced.
async fn watch(gen: u64) {
    loop {
        tokio::time::sleep(WATCH_INTERVAL).await;
        let (name, needs_measure) = {
            let st = state();
            match &st.current {
                Some(c) if c.gen == gen => (
                    c.launch.helper_name.clone(),
                    !c.measured && c.launch.gpu_layers > 0 && c.measure_tries < MAX_MEASURE_TRIES && c.status.state == LlmServerState::Ready,
                ),
                _ => return,
            }
        };
        let info = process_host::get(&name);
        let handle = process_host::handle(&name);
        let measured = if needs_measure {
            devices::measured_buffers_mb(&process_host::logs(&name))
        } else {
            None
        };
        let (changed, terminal, status) = {
            let mut st = state();
            let Some(cur) = st.current.as_mut().filter(|c| c.gen == gen) else {
                return;
            };
            let before = cur.status.clone();
            let new_state = map_state(info.as_ref().map(|i| i.state));
            cur.status.state = new_state;
            let restarts = info.as_ref().map_or(0, |i| i.restarts);
            if restarts != cur.status.restarts {
                // A restarted instance loads again: measure its buffers afresh.
                cur.measured = false;
                cur.measure_tries = 0;
            }
            cur.status.restarts = restarts;
            cur.status.error = info.as_ref().and_then(|i| i.error.clone());
            cur.status.base_url = match (&handle, new_state) {
                (Some(h), LlmServerState::Ready) => h.base_url.as_ref().map(|b| format!("{b}/v1")),
                _ => None,
            };
            if needs_measure {
                cur.measure_tries += 1;
            }
            if let Some(mb) = measured.filter(|_| new_state == LlmServerState::Ready) {
                let total = mb.saturating_add(runtime_base_mb(&cur.launch.variant));
                if let Some(c) = &cur.claim {
                    c.resize(total);
                }
                tracing::info!(target: TARGET, "{}: measured {total} MiB of video memory (planned {} MiB)", cur.launch.label, cur.launch.vram_mb);
                cur.status.vram_mb = total;
                cur.measured = true;
            }
            let terminal = matches!(new_state, LlmServerState::Failed | LlmServerState::Stopped);
            if terminal {
                cur.status.vram_mb = 0;
                cur.status.base_url = None;
            }
            let status = cur.status.clone();
            if terminal {
                let finished = st.current.take();
                st.last = status.clone();
                drop(finished); // releases the claim
            }
            (status != before || terminal, terminal, status)
        };
        if changed {
            if status.state == LlmServerState::Failed {
                tracing::warn!(target: TARGET, "llama-server failed: {}", status.error.as_deref().unwrap_or("unknown error"));
            }
            emit(&status);
        }
        if terminal {
            return;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Model list
// ---------------------------------------------------------------------------------------------

/// `list_llm_models` (blocking).
pub fn list_models() -> Result<Vec<LlmModelInfo>, LlmServerError> {
    let catalog = Catalog::load().map_err(|e| err(LlmServerErrorKind::Runtime, e))?;
    let sys = sysinfo::cached();
    let snap = resources::snapshot();
    Ok(list_models_with(&catalog, &sys, &snap, &|id| model_store::installed_path(id).is_some()))
}

/// [`list_models`] with explicit inputs (tests).
pub fn list_models_with(catalog: &Catalog, sys: &SystemInfo, snap: &ResourceSnapshot, installed: &dyn Fn(&str) -> bool) -> Vec<LlmModelInfo> {
    let auto = auto_model(catalog, sys, snap);
    catalog::entries(catalog)
        .into_iter()
        .map(|e| {
            let meta = catalog::meta(&e.id).cloned().unwrap_or_default();
            let size_bytes = catalog::gguf_bytes(e);
            let vram_mb = e.min_vram_mb.unwrap_or_else(|| {
                plan::full_vram_mb(
                    &ModelFacts {
                        size_bytes,
                        ..Default::default()
                    },
                    catalog::DEFAULT_CTX,
                )
            });
            LlmModelInfo {
                id: e.id.clone(),
                title: e.title.clone().unwrap_or_else(|| e.id.clone()),
                prompt_format: if meta.prompt_format.is_empty() {
                    "generic".into()
                } else {
                    meta.prompt_format.clone()
                },
                size_bytes,
                vram_mb,
                installed: installed(&e.id),
                specialist: meta.specialist(),
                languages: meta.languages.clone(),
                recommended: e.id == auto,
                role: if meta.role.is_empty() { "alternative".into() } else { meta.role.clone() },
                availability: meta.availability.as_str().into(),
                directions: meta.directions.clone(),
                sampling: meta.sampling.clone(),
                description: e.description.clone(),
                license: e.license.clone(),
            }
        })
        .collect()
}

/// Test hook: rewrite the helper spec before it is spawned (the lifecycle tests run a fake
/// llama-server built into the test binary).
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

    pub type BoxFuture = std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>;
    type BeforeSpawn = Box<dyn FnOnce() -> BoxFuture + Send>;
    static BEFORE_SPAWN: Mutex<Option<BeforeSpawn>> = Mutex::new(None);

    /// Run `f` once, in the next start, after its state is published and before `process_host`
    /// is asked to spawn (the window a concurrent stop can land in).
    pub fn before_spawn_once(f: impl FnOnce() -> BoxFuture + Send + 'static) {
        *BEFORE_SPAWN.lock().unwrap() = Some(Box::new(f));
    }

    pub(super) async fn run_before_spawn() {
        let hook = BEFORE_SPAWN.lock().unwrap().take();
        if let Some(f) = hook {
            f().await;
        }
    }
}
