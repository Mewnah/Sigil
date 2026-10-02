//! Lifecycle of one engine's helper ([`Slot`]): start (reusing a running server with the same
//! launch, replacing one with another, waiting for the process host's restart of one that
//! crashed), readiness, the adapter, a timed warm-up, the VRAM claim, a watcher, and stop. Status
//! changes are emitted as `tts_helpers:state`.
//!
//! Concurrency (as `qwen::server` and `llm_server::server`): starts are serialized per engine by an
//! async gate (a second caller with the same launch gets the running server). A stop never waits
//! for the gate: it bumps the stop epoch and stops the helper, which cancels a start that is still
//! loading (process_host's last-request-wins); a start requested before a stop returns
//! "cancelled". The order is the order the commands run in, which IPC does not guarantee for two
//! commands sent back to back: the TS side therefore sends a start only after its own pending stop
//! of the same engine finished (`stopGgmlHelper` / `startGgmlHelper` in `ggmlTts.ts`). Every start
//! and stop bumps a generation, so a watcher or a start that finishes late never touches a newer
//! server's state, and a start that finds itself superseded after its helper became ready stops it
//! again.

use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use super::adapter::{self, Adapter, Upstream};
use super::launch::{build_spec, Launch, Readiness};
use crate::services::diagnostics;
use crate::services::process_host::{self, HelperState};
use crate::services::resources::{self, Subsystem, VramClaim};
use crate::services::tts_helpers::{TtsHelperEndpoint, TtsHelperEngine, TtsHelperState, TtsHelperStatus, STATE_EVENT};

/// Log target of every ggml helper (messages name the engine).
pub const TARGET: &str = "sigil::tts::ggml";
/// How often the watcher looks at the helper.
const WATCH_INTERVAL: Duration = Duration::from_millis(400);
/// Time a warm-up request may take before the start gives up on it (the server is ready anyway).
const WARMUP_TIMEOUT: Duration = Duration::from_secs(120);
/// Interval of the "any HTTP answer" readiness probe.
const ANY_HTTP_INTERVAL: Duration = Duration::from_millis(250);

struct Current {
    gen: u64,
    launch: Launch,
    status: TtsHelperStatus,
    /// Released when dropped.
    #[allow(dead_code)] // RAII: held for the life of the server.
    claim: Option<VramClaim>,
    adapter: Option<Adapter>,
    restarts: u32,
    /// The process host's restart count when the server last answered (an [`Readiness::AnyHttp`]
    /// server is `Running` at once after a restart, before its model is loaded again).
    ready_restarts: u32,
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

/// One engine's helper (a `static` per engine module).
pub struct Slot {
    engine: TtsHelperEngine,
    /// English name in messages ("Magpie TTS").
    title: &'static str,
    state: OnceLock<Mutex<State>>,
    gate: OnceLock<tokio::sync::Mutex<()>>,
}

impl Slot {
    pub const fn new(engine: TtsHelperEngine, title: &'static str) -> Slot {
        Slot {
            engine,
            title,
            state: OnceLock::new(),
            gate: OnceLock::new(),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn gate(&self) -> &tokio::sync::Mutex<()> {
        self.gate.get_or_init(Default::default)
    }

    fn status_of(&self, model: Option<&str>, state: TtsHelperState, error: Option<String>) -> TtsHelperStatus {
        TtsHelperStatus {
            engine: self.engine,
            state,
            model: model.map(str::to_string),
            error,
        }
    }

    fn stopped_status(&self) -> TtsHelperStatus {
        self.status_of(None, TtsHelperState::Stopped, None)
    }

    /// The current status (never contains a key).
    pub fn status(&self) -> TtsHelperStatus {
        let st = self.state();
        match &st.current {
            Some(c) => c.status.clone(),
            None => st.last.clone().unwrap_or_else(|| self.stopped_status()),
        }
    }

    /// The stop counter: a start that read an older value is cancelled.
    pub fn stop_epoch(&self) -> u64 {
        self.state().stops
    }

    fn cancelled(&self) -> String {
        format!("cancelled: {} was stopped or replaced while it started", self.title)
    }

    /// Start (or reuse) the helper planned by `resolve` (run on a blocking thread: it checks files).
    pub async fn start<F>(&'static self, resolve: F) -> Result<TtsHelperEndpoint, String>
    where
        F: FnOnce() -> Result<Launch, String> + Send + 'static,
    {
        let stops_at_call = self.stop_epoch();
        let _gate = self.gate().lock().await;
        if self.state().stops != stops_at_call {
            return Err(format!("cancelled: {} was stopped", self.title));
        }
        let launch = tauri::async_runtime::spawn_blocking(resolve)
            .await
            .map_err(|e| format!("{} task failed: {e}", self.title))??;
        tracing::info!(target: TARGET, "{}: {}", self.title, launch.summary);
        self.start_launch(launch, stops_at_call).await
    }

    /// The endpoint of `launch`'s running helper (its port and key can change after a restart).
    fn endpoint_of(&self, launch: &Launch, adapter: Option<&Adapter>) -> Option<TtsHelperEndpoint> {
        let (base_url, api_key) = match adapter {
            Some(a) => (a.base_url(), a.key().to_string()),
            None => {
                let h = process_host::handle(&launch.helper_name)?;
                let base = h.base_url?;
                let key = if launch.server_key { h.api_key? } else { String::new() };
                (format!("{base}/v1"), key)
            }
        };
        Some(TtsHelperEndpoint {
            engine: self.engine,
            base_url,
            api_key,
            model: launch.model.clone(),
            pcm_sample_rate: launch.pcm_rate,
        })
    }

    /// The endpoint of server generation `gen` once the process host runs it: at once when it runs,
    /// after the process host's restart when it is restarting it (bounded by the load timeout),
    /// `None` when it is gone, failed or replaced, or a stop came. A server without a health route
    /// ([`Readiness::AnyHttp`]) that was restarted since it last answered must answer again first
    /// (the process host calls it running before its model is loaded); one that never does within
    /// the load timeout is `None` (the caller replaces it).
    async fn wait_for_running(&self, gen: u64, launch: &Launch, stops_at_call: u64) -> Option<TtsHelperEndpoint> {
        let deadline = Instant::now() + Duration::from_millis(launch.load_timeout_ms);
        let mut logged = false;
        loop {
            // `Some(restarts)`: running, but restarted since it last answered.
            let unconfirmed = {
                let st = self.state();
                match st.current.as_ref() {
                    Some(c) if c.gen == gen && st.stops == stops_at_call => {
                        match process_host::get(&launch.helper_name).filter(|i| i.state == HelperState::Running) {
                            Some(info) if launch.readiness != Readiness::AnyHttp || info.restarts == c.ready_restarts => {
                                return self.endpoint_of(launch, c.adapter.as_ref());
                            }
                            Some(info) => Some(info.restarts),
                            None => None,
                        }
                    }
                    _ => return None,
                }
            };
            if let Some(restarts) = unconfirmed {
                if !logged {
                    tracing::info!(target: TARGET, "{}: {} restarted; waiting until it answers again", self.title, launch.model);
                    logged = true;
                }
                let base = process_host::handle(&launch.helper_name).and_then(|h| h.base_url);
                if let Some(base) = base {
                    if answers_any_http(&base).await {
                        let mut st = self.state();
                        if let Some(c) = st.current.as_mut().filter(|c| c.gen == gen) {
                            c.ready_restarts = c.ready_restarts.max(restarts);
                        }
                        continue;
                    }
                }
                if Instant::now() >= deadline {
                    tracing::warn!(
                        target: TARGET,
                        "{}: {} did not answer within {} s after its restart",
                        self.title,
                        launch.model,
                        launch.load_timeout_ms / 1000
                    );
                    return None;
                }
                tokio::time::sleep(ANY_HTTP_INTERVAL).await;
                continue;
            }
            match process_host::get(&launch.helper_name).map(|i| i.state) {
                Some(HelperState::Starting | HelperState::Restarting) if Instant::now() < deadline => {
                    if !logged {
                        tracing::info!(target: TARGET, "{}: {} is restarting; waiting for it", self.title, launch.model);
                        logged = true;
                    }
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
                Some(HelperState::Running) => continue,
                _ => return None,
            }
        }
    }

    /// Start `launch` unless the running server already is it. The caller holds the start gate
    /// (tests call it directly with their own launch).
    pub async fn start_launch(&'static self, launch: Launch, stops_at_call: u64) -> Result<TtsHelperEndpoint, String> {
        // Reuse the running server, or wait for the process host's restart of it (after a crash)
        // instead of starting another one.
        let reuse = {
            let st = self.state();
            st.current
                .as_ref()
                .filter(|c| c.launch.same_server(&launch) && matches!(c.status.state, TtsHelperState::Ready | TtsHelperState::Starting))
                .map(|c| (c.gen, c.launch.clone()))
        };
        if let Some((gen, running)) = reuse {
            if let Some(endpoint) = self.wait_for_running(gen, &running, stops_at_call).await {
                return Ok(endpoint);
            }
        }

        // Retire whatever runs now (its VRAM is freed before the new model loads).
        let old = {
            let mut st = self.state();
            st.gen += 1;
            st.current.take()
        };
        if let Some(old) = old {
            tracing::info!(target: TARGET, "{}: replacing {} ({})", self.title, old.launch.model, old.launch.variant.as_str());
            if let Err(e) = process_host::stop(&old.launch.helper_name).await {
                tracing::warn!(target: TARGET, "{}: stopping the previous server: {e}", self.title);
            }
            drop(old);
        }

        let (gen, starting) = {
            let mut st = self.state();
            if st.stops != stops_at_call {
                return Err(format!("cancelled: {} was stopped", self.title));
            }
            st.gen += 1;
            let claim = (launch.vram_mb > 0).then(|| {
                resources::claim_external_vram(Subsystem::Tts, &format!("{}: {}", self.engine.as_str(), launch.model), launch.vram_mb)
            });
            let status = self.status_of(Some(&launch.model), TtsHelperState::Starting, None);
            st.current = Some(Current {
                gen: st.gen,
                launch: launch.clone(),
                status: status.clone(),
                claim,
                adapter: None,
                restarts: 0,
                // The first readiness (health route, or `wait_any_http`) confirms restart 0.
                ready_restarts: 0,
            });
            (st.gen, status)
        };
        emit(&starting);

        let spec = build_spec(&launch);
        let started_at = Instant::now();
        let spawned = process_host::spawn(spec).await;

        // Readiness, the adapter and the warm-up run outside the state lock.
        let mut adapter_handle: Option<Adapter> = None;
        let checked: Result<TtsHelperEndpoint, String> = async {
            let handle = spawned.as_ref().map_err(Clone::clone)?;
            if launch.readiness == Readiness::AnyHttp {
                let base = handle
                    .base_url
                    .clone()
                    .ok_or_else(|| format!("the {} helper started without a port", self.title))?;
                self.wait_any_http(&base, &launch, gen, stops_at_call).await?;
            }
            if let Some(kind) = launch.adapter {
                // Connection errors are retried while a crashed helper restarts and loads its model.
                let reconnect_for = Duration::from_millis(launch.load_timeout_ms);
                adapter_handle = Some(adapter::start(kind, Upstream::Helper(launch.helper_name.clone()), reconnect_for).await?);
            }
            let endpoint = self
                .endpoint_of(&launch, adapter_handle.as_ref())
                .ok_or_else(|| format!("the {} helper started without a port or key", self.title))?;
            tracing::info!(
                target: TARGET,
                "{}: ready in {} ms: {} on {}{}",
                self.title,
                started_at.elapsed().as_millis(),
                launch.model,
                launch.variant.as_str(),
                adapter_handle
                    .as_ref()
                    .map(|a| format!(" (through the adapter on 127.0.0.1:{})", a.port()))
                    .unwrap_or_default()
            );
            if let Some(body) = &launch.warmup {
                warm_up(self.title, &endpoint, body, &launch).await;
            }
            Ok(endpoint)
        }
        .await;

        // Decide under the lock, act (emit, stop a helper) after releasing it.
        enum Outcome {
            Ready(TtsHelperEndpoint),
            Failed(TtsHelperStatus, String),
            Superseded,
        }
        let (outcome, released) = {
            let mut st = self.state();
            let still_ours = st.gen == gen && st.stops == stops_at_call && st.current.as_ref().is_some_and(|c| c.gen == gen);
            match (checked, still_ours) {
                (Ok(endpoint), true) => {
                    let cur = st.current.as_mut().expect("checked above");
                    cur.status.state = TtsHelperState::Ready;
                    cur.adapter = adapter_handle.take();
                    (Outcome::Ready(endpoint), None)
                }
                (Err(e), true) => {
                    let failed = st.current.take();
                    let status = self.status_of(Some(&launch.model), TtsHelperState::Failed, Some(e.clone()));
                    st.last = Some(status.clone());
                    (Outcome::Failed(status, e), failed)
                }
                (_, false) => (Outcome::Superseded, None),
            }
        };
        // An adapter that was not kept (failure, superseded) shuts down here.
        drop(adapter_handle);

        match outcome {
            Outcome::Ready(endpoint) => {
                emit(&self.status());
                tauri::async_runtime::spawn(self.watch(gen));
                Ok(endpoint)
            }
            Outcome::Failed(status, e) => {
                if spawned.is_ok() {
                    // Started, but unusable (never ready, no endpoint): stop it again.
                    if let Err(stop_err) = process_host::stop(&launch.helper_name).await {
                        tracing::warn!(target: TARGET, "{}: stopping an unusable server: {stop_err}", self.title);
                    }
                }
                drop(released); // the VRAM claim goes once the process is gone
                emit(&status);
                tracing::warn!(target: TARGET, "{} failed to start: {e}", self.title);
                Err(e)
            }
            Outcome::Superseded => {
                if spawned.is_ok() {
                    tracing::info!(target: TARGET, "{}: {} became ready after a stop; stopping it", self.title, launch.model);
                    // Only when no newer start of the same helper took over meanwhile.
                    let newer = self.state().current.as_ref().is_some_and(|c| c.launch.helper_name == launch.helper_name);
                    if !newer {
                        if let Err(e) = process_host::stop(&launch.helper_name).await {
                            tracing::warn!(target: TARGET, "{}: stopping a server that was started after a stop: {e}", self.title);
                        }
                    }
                }
                Err(self.cancelled())
            }
        }
    }

    /// Wait until the server answers any HTTP request (servers without a health route listen once
    /// their model is loaded), the helper fails, a stop comes, or the load timeout. A helper the
    /// process host restarts meanwhile (a crash while loading) is waited for on its new port; the
    /// restart that answered is recorded as confirmed (see [`Self::wait_for_running`]).
    async fn wait_any_http(&self, base: &str, launch: &Launch, gen: u64, stops_at_call: u64) -> Result<(), String> {
        let deadline = Instant::now() + Duration::from_millis(launch.load_timeout_ms);
        loop {
            {
                let st = self.state();
                if st.stops != stops_at_call || st.current.as_ref().is_none_or(|c| c.gen != gen) {
                    return Err(self.cancelled());
                }
            }
            let restarts = match process_host::get(&launch.helper_name).map(|i| (i.state, i.error, i.restarts)) {
                Some((HelperState::Running | HelperState::Starting | HelperState::Restarting, _, restarts)) => restarts,
                Some((HelperState::Failed, error, _)) => {
                    return Err(format!("the {} helper exited while loading: {}", self.title, error.unwrap_or_default()));
                }
                other => {
                    return Err(format!("the {} helper is not running ({:?})", self.title, other.map(|o| o.0)));
                }
            };
            let current_base = process_host::handle(&launch.helper_name).and_then(|h| h.base_url);
            if answers_any_http(current_base.as_deref().unwrap_or(base)).await {
                let mut st = self.state();
                if let Some(c) = st.current.as_mut().filter(|c| c.gen == gen) {
                    c.ready_restarts = c.ready_restarts.max(restarts);
                }
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "the {} helper did not answer within {} s: check the log, or choose a smaller model",
                    self.title,
                    launch.load_timeout_ms / 1000
                ));
            }
            tokio::time::sleep(ANY_HTTP_INTERVAL).await;
        }
    }

    /// Stop the helper (also cancels a start that is still loading). Ok when it was not running.
    pub async fn stop(&self, default_helper_name: &str) -> Result<(), String> {
        let (name, cur) = {
            let mut st = self.state();
            st.stops += 1;
            st.gen += 1;
            let cur = st.current.take();
            let name = cur
                .as_ref()
                .map(|c| c.launch.helper_name.clone())
                .unwrap_or_else(|| default_helper_name.to_string());
            (name, cur)
        };
        let result = process_host::stop(&name).await;
        drop(cur); // releases the VRAM claim and shuts the adapter down once the process is gone
        let status = {
            let mut st = self.state();
            if st.current.is_none() {
                st.last = None;
            }
            match &st.current {
                Some(c) => c.status.clone(),
                None => self.stopped_status(),
            }
        };
        emit(&status);
        result
    }

    /// Follow the helper of generation `gen` until it stops or is replaced.
    async fn watch(&'static self, gen: u64) {
        loop {
            tokio::time::sleep(WATCH_INTERVAL).await;
            let name = {
                let st = self.state();
                match st.current.as_ref() {
                    Some(c) if c.gen == gen => c.launch.helper_name.clone(),
                    _ => return,
                }
            };
            let info = process_host::get(&name);
            let (changed, terminal, status) = {
                let mut st = self.state();
                let Some(cur) = st.current.as_mut().filter(|c| c.gen == gen) else {
                    return;
                };
                let before = cur.status.clone();
                let new_state = map_state(info.as_ref().map(|i| i.state));
                cur.status.state = new_state;
                let restarts = info.as_ref().map_or(0, |i| i.restarts);
                if restarts != cur.restarts {
                    cur.restarts = restarts;
                    tracing::info!(target: TARGET, "{}: the helper restarted ({restarts} so far)", self.title);
                }
                cur.status.error = match new_state {
                    TtsHelperState::Failed => info.as_ref().and_then(|i| i.error.clone()),
                    _ => None,
                };
                let terminal = matches!(new_state, TtsHelperState::Failed | TtsHelperState::Stopped);
                let status = cur.status.clone();
                if terminal {
                    let finished = st.current.take();
                    st.last = Some(status.clone());
                    drop(finished); // releases the claim, shuts the adapter down
                }
                (status != before || terminal, terminal, status)
            };
            if changed {
                if status.state == TtsHelperState::Failed {
                    tracing::warn!(target: TARGET, "{} failed: {}", self.title, status.error.as_deref().unwrap_or("unknown error"));
                }
                emit(&status);
            }
            if terminal {
                return;
            }
        }
    }
}

fn emit(status: &TtsHelperStatus) {
    diagnostics::emit(STATE_EVENT, status);
}

fn map_state(s: Option<HelperState>) -> TtsHelperState {
    match s {
        Some(HelperState::Starting) | Some(HelperState::Restarting) => TtsHelperState::Starting,
        Some(HelperState::Running) => TtsHelperState::Ready,
        Some(HelperState::Failed) => TtsHelperState::Failed,
        Some(HelperState::Stopped) | None => TtsHelperState::Stopped,
    }
}

/// Whether the server at `base` answers an HTTP request (any status) within 2 s.
async fn answers_any_http(base: &str) -> bool {
    loopback_client()
        .get(format!("{}/", base.trim_end_matches('/')))
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .is_ok()
}

/// HTTP client for loopback helpers: no proxy, no redirects.
pub fn loopback_client() -> &'static reqwest::Client {
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

/// The `rate=` parameter of an `audio/pcm` content type.
pub fn rate_of_content_type(value: &str) -> Option<u32> {
    value.split(';').skip(1).find_map(|p| {
        let (k, v) = p.split_once('=')?;
        k.trim().eq_ignore_ascii_case("rate").then(|| v.trim().trim_matches('"').parse().ok()).flatten()
    })
}

/// Speak `body` through `endpoint` (OpenAI shape, raw PCM) and time it; the audio is discarded.
pub async fn timed_speech(endpoint: &TtsHelperEndpoint, body: &serde_json::Value, timeout: Duration) -> Result<Timing, String> {
    use futures::StreamExt;
    let started = Instant::now();
    let mut rb = loopback_client()
        .post(format!("{}/audio/speech", endpoint.base_url))
        .json(body)
        .timeout(timeout);
    if !endpoint.api_key.is_empty() {
        rb = rb.bearer_auth(&endpoint.api_key);
    }
    let resp = rb.send().await.map_err(|e| format!("speech request failed: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("speech request answered {status}: {}", text.chars().take(300).collect::<String>()));
    }
    let rate = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(rate_of_content_type)
        .unwrap_or(endpoint.pcm_sample_rate)
        .max(1);
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
        audio_ms: (bytes / 2) as f64 * 1000.0 / f64::from(rate),
    })
}

/// A short request that runs every kernel once (CUDA graphs, Vulkan pipelines), so the first
/// caption does not pay for it; logs its time to first audio and real-time factor.
async fn warm_up(title: &str, endpoint: &TtsHelperEndpoint, body: &serde_json::Value, launch: &Launch) {
    let timeout = WARMUP_TIMEOUT.min(Duration::from_millis(launch.load_timeout_ms.max(30_000)));
    match timed_speech(endpoint, body, timeout).await {
        Ok(t) => tracing::info!(
            target: TARGET,
            "{title}: warm-up ({} on {}): first audio {:.0} ms, {:.0} ms of audio in {:.0} ms (RTF {})",
            launch.model,
            launch.variant.as_str(),
            t.first_audio_ms,
            t.audio_ms,
            t.total_ms,
            t.rtf().map(|r| format!("{r:.3}")).unwrap_or_else(|| "-".into())
        ),
        Err(e) => tracing::warn!(target: TARGET, "{title}: warm-up failed (the server keeps running): {e}"),
    }
}
