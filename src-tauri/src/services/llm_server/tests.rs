//! Lifecycle tests with a real child process: the "llama-server" is this test binary re-run as a
//! fake (an HTTP server with llama.cpp's `/health` behaviour, bearer-key checks and its buffer-size
//! log lines), started through the real `server::start_launch` → `process_host` path. Plus
//! resolution, spec and model-list tests on the embedded catalog. Nothing touches the network.
//!
//! The real-model run is `real_model_end_to_end` (ignored; see its doc comment).

use super::catalog::{self, RUNTIME_CPU, RUNTIME_VULKAN};
use super::devices::Pin;
use super::server::{self, build_spec, make_launch, resolve_with, test_hooks, Launch};
use super::*;
use crate::services::diagnostics::gpu_mem::{VENDOR_AMD, VENDOR_NVIDIA};
use crate::services::model_store::Catalog;
use crate::services::process_host::{self, HelperPriority, HelperState, RestartPolicy};
use crate::services::resources::{self, policy::ids, ResourceSnapshot, VramBudget};
use crate::services::sysinfo::test_support::{gpu, system};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const ROLE_ENV: &str = "SIGIL_LLM_FAKE";
const ENTRY: &str = "services::llm_server::tests::fake_llama_server_entry";

/// The lifecycle tests share the one server state: run them one at a time (an async lock, held
/// across awaits).
async fn serial() -> tokio::sync::MutexGuard<'static, ()> {
    static SERIAL: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
    SERIAL.get_or_init(Default::default).lock().await
}

fn test_exe() -> PathBuf {
    std::env::current_exe().expect("test binary path")
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sigil-llm-test-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Route every spawn of this test binary's llm server to the fake (role + load time + restarts).
fn install_fake(role: &'static str, load_ms: u64, restart: Option<RestartPolicy>) {
    test_hooks::set(move |mut spec| {
        let llama_args = spec.args.join("\t");
        spec.program = test_exe();
        spec.args = ["--exact", ENTRY, "--ignored", "--nocapture", "--test-threads=1", "-q"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        spec.env.push((ROLE_ENV.into(), role.into()));
        spec.env
            .push(("SIGIL_LLM_FAKE_PORT".into(), "{port}".into()));
        spec.env
            .push(("SIGIL_LLM_FAKE_LOAD_MS".into(), load_ms.to_string()));
        spec.env.push(("SIGIL_LLM_FAKE_ARGS".into(), llama_args));
        if let Some(h) = spec.health.as_mut() {
            h.interval_ms = 50;
            h.timeout_ms = 20_000;
        }
        if let Some(r) = restart {
            spec.restart = r;
        }
        spec
    });
}

fn launch(name: &str, ctx: u32) -> Launch {
    let dir = temp_dir(&format!("model-{name}-{ctx}"));
    let model = dir.join("fake.gguf");
    std::fs::write(&model, super::gguf::build::hunyuan_like()).unwrap();
    Launch {
        helper_name: name.into(),
        exe: test_exe(),
        model_path: model,
        model: ids::MT_HY_MT2_1_8B.into(),
        label: format!("Fake {name}"),
        alias: "hy-mt2-1.8b-q4_k_m".into(),
        variant: "vulkan".into(),
        gpu_layers: plan::ALL_LAYERS,
        ctx_size: ctx,
        flash_attn: "auto".into(),
        threads: 2,
        no_poll: true,
        below_normal: false,
        pin: Pin::Device("Vulkan0".into()),
        vram_mb: 1600,
        load_timeout_ms: 20_000,
    }
}

fn claim_mb(label: &str) -> Option<u64> {
    resources::snapshot()
        .claims
        .into_iter()
        .find(|c| c.label == format!("llama-server: {label}"))
        .map(|c| c.mb)
}

async fn wait_until(timeout: Duration, mut f: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if f() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    f()
}

fn client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

async fn chat_status(base_v1: &str, key: Option<&str>) -> u16 {
    let mut rb = client()
        .post(format!("{base_v1}/chat/completions"))
        .body("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")
        .timeout(Duration::from_secs(5));
    if let Some(k) = key {
        rb = rb.bearer_auth(k);
    }
    rb.send().await.map(|r| r.status().as_u16()).unwrap_or(0)
}

/// The current stop epoch (a start that no later stop cancels).
fn stops() -> u64 {
    server::stop_epoch()
}

#[tokio::test(flavor = "multi_thread")]
async fn start_reuse_measure_and_stop() {
    let _s = serial().await;
    install_fake("ok", 300, None);
    let l = launch("t-llm-life", 4096);
    let started = server::start_launch(l.clone(), stops())
        .await
        .expect("fake server starts");
    assert_eq!(started.status.state, LlmServerState::Ready);
    assert_eq!(started.status.model.as_deref(), Some(ids::MT_HY_MT2_1_8B));
    let base = started.endpoint.base_url.clone();
    assert!(base.starts_with("http://127.0.0.1:") && base.ends_with("/v1"), "{base}");
    assert_eq!(started.endpoint.api_key.len(), 64);
    assert_eq!(started.endpoint.model, "hy-mt2-1.8b-q4_k_m");
    assert_eq!(server::status().base_url.as_deref(), Some(base.as_str()));
    assert_eq!(server::endpoint(), Some(started.endpoint.clone()));
    let status_json = serde_json::to_string(&server::status()).unwrap();
    assert!(!status_json.contains(&started.endpoint.api_key), "the status never carries the key");

    // The key is required.
    assert_eq!(chat_status(&base, None).await, 401);
    assert_eq!(chat_status(&base, Some("wrong")).await, 401);
    assert_eq!(chat_status(&base, Some(&started.endpoint.api_key)).await, 200);

    // The flags reached the process; the key did not reach its command line.
    let logs = server::logs(None);
    let args_line = logs
        .iter()
        .find(|l| l.text.contains("fake-llama args:"))
        .map(|l| l.text.clone())
        .unwrap_or_default();
    for flag in [
        "--host\t127.0.0.1",
        "--no-webui",
        "-fit\toff",
        "--reasoning\toff",
        "--poll\t0",
        "--device\tVulkan0",
        "-ngl\t999",
        "--alias\thy-mt2-1.8b-q4_k_m",
    ] {
        assert!(args_line.contains(flag), "{flag} missing: {args_line}");
    }
    assert!(logs
        .iter()
        .all(|l| !l.text.contains(&started.endpoint.api_key)));
    assert!(server::logs(Some(2)).len() <= 2);

    // The claim starts at the plan's estimate and is resized to the logged buffers + Vulkan base.
    assert!(claim_mb(&l.label).is_some());
    assert!(
        wait_until(Duration::from_secs(8), || claim_mb(&l.label) == Some(1250 + devices::VULKAN_BASE_MB)).await,
        "claim {:?}",
        claim_mb(&l.label)
    );
    assert_eq!(server::status().vram_mb, 1250 + devices::VULKAN_BASE_MB);

    // Same launch: the running server is reused.
    let pid = process_host::get(&l.helper_name).unwrap().pid;
    let again = server::start_launch(l.clone(), stops()).await.unwrap();
    assert_eq!(again.endpoint, started.endpoint);
    assert_eq!(process_host::get(&l.helper_name).unwrap().pid, pid);

    server::stop().await.unwrap();
    assert_eq!(server::status().state, LlmServerState::Stopped);
    assert_eq!(server::endpoint(), None);
    assert_eq!(claim_mb(&l.label), None, "the claim is released");
    assert_eq!(process_host::get(&l.helper_name).map(|i| i.state), Some(HelperState::Stopped));
    let pid = pid.unwrap();
    assert!(process_host::job::wait_pid_exit(pid, Duration::from_secs(5)), "pid {pid} still running");
}

#[tokio::test(flavor = "multi_thread")]
async fn another_launch_replaces_the_server_and_its_claim() {
    let _s = serial().await;
    install_fake("ok", 100, None);
    let a = launch("t-llm-replace", 4096);
    let first = server::start_launch(a.clone(), stops()).await.unwrap();
    let pid_a = process_host::get(&a.helper_name).unwrap().pid.unwrap();
    let mut b = launch("t-llm-replace", 8192);
    b.label = "Fake replacement".into();
    b.vram_mb = 2100;
    let second = server::start_launch(b.clone(), stops()).await.unwrap();
    assert_eq!(second.status.ctx_size, Some(8192));
    assert_ne!(second.endpoint.base_url, "");
    assert_eq!(first.endpoint.api_key.len(), second.endpoint.api_key.len());
    assert!(process_host::job::wait_pid_exit(pid_a, Duration::from_secs(5)), "the old server is gone");
    assert_eq!(claim_mb(&a.label), None, "the old claim is released");
    assert!(claim_mb(&b.label).is_some());
    server::stop().await.unwrap();
    assert_eq!(claim_mb(&b.label), None);
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_during_a_slow_load_cancels_the_start() {
    let _s = serial().await;
    install_fake("ok", 15_000, None);
    let l = launch("t-llm-cancel", 4096);
    let epoch = stops();
    let task = tokio::spawn(server::start_launch(l.clone(), epoch));
    assert!(
        wait_until(Duration::from_secs(10), || server::status().state == LlmServerState::Starting
            && process_host::get(&l.helper_name).is_some_and(|i| i.pid.is_some()))
        .await
    );
    assert_eq!(server::status().model.as_deref(), Some(ids::MT_HY_MT2_1_8B));
    let t0 = Instant::now();
    server::stop().await.unwrap();
    let res = task.await.unwrap();
    assert_eq!(res.unwrap_err().kind, LlmServerErrorKind::Cancelled);
    assert!(t0.elapsed() < Duration::from_secs(5), "the 15 s load was cut short");
    assert_eq!(server::status().state, LlmServerState::Stopped);
    assert_eq!(claim_mb(&l.label), None);
    // A start requested before a stop is cancelled even if it gets the gate afterwards.
    let res = server::start_launch(l.clone(), epoch).await;
    assert_eq!(res.unwrap_err().kind, LlmServerErrorKind::Cancelled);
}

/// The pid process_host noted for the latest instance of `name`.
fn noted_pid(name: &str) -> Option<u32> {
    process_host::logs(name)
        .iter()
        .rev()
        .find_map(|l| l.text.strip_prefix("started pid "))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|p| p.parse().ok())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stop_before_the_spawn_takes_its_ticket_leaves_no_orphan() {
    let _s = serial().await;
    install_fake("ok", 100, None);
    let l = launch("t-llm-orphan", 4096);
    // The stop lands after the start published its state and before process_host saw the spawn:
    // process_host has nothing to stop yet, and the spawn wins its ticket afterwards.
    test_hooks::before_spawn_once(|| {
        Box::pin(async {
            server::stop().await.unwrap();
        })
    });
    let res = server::start_launch(l.clone(), stops()).await;
    assert_eq!(res.unwrap_err().kind, LlmServerErrorKind::Cancelled);
    // The helper that started anyway was stopped again: no process, no claim, no endpoint.
    let pid = noted_pid(&l.helper_name).expect("the fake was started");
    assert!(process_host::job::wait_pid_exit(pid, Duration::from_secs(5)), "pid {pid} still running");
    assert_ne!(process_host::get(&l.helper_name).map(|i| i.state), Some(HelperState::Running));
    assert_eq!(server::status().state, LlmServerState::Stopped);
    assert_eq!(server::endpoint(), None);
    assert_eq!(claim_mb(&l.label), None);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_crashing_server_ends_failed_and_releases_its_vram() {
    let _s = serial().await;
    install_fake(
        "crash_after_ready",
        50,
        Some(RestartPolicy::OnFailure {
            max_restarts: 1,
            backoff_ms: 100,
        }),
    );
    let l = launch("t-llm-crash", 4096);
    server::start_launch(l.clone(), stops()).await.unwrap();
    assert!(
        wait_until(Duration::from_secs(15), || server::status().state == LlmServerState::Failed).await,
        "status {:?}",
        server::status()
    );
    let st = server::status();
    assert!(st.error.as_deref().unwrap_or("").contains("gave up"), "{st:?}");
    assert_eq!(st.restarts, 1);
    assert_eq!(st.vram_mb, 0);
    assert_eq!(claim_mb(&l.label), None);
    assert_eq!(server::endpoint(), None);
    server::stop().await.unwrap();
}

// ---------------------------------------------------------------------------------------------
// Resolution, spec and model list (no processes)
// ---------------------------------------------------------------------------------------------

fn embedded() -> Catalog {
    Catalog::embedded().unwrap()
}

fn snap(vc_gpu: bool) -> ResourceSnapshot {
    ResourceSnapshot {
        budget: VramBudget::default(),
        claims: vec![],
        vc_running: vc_gpu,
        vc_gpu: vc_gpu.then_some(true),
    }
}

/// A store with Hy-MT2 1.8B / 7B (synthetic headers) and the Vulkan + CPU packs "installed".
struct FakeStore {
    dir: PathBuf,
}

impl FakeStore {
    fn new(tag: &str) -> Self {
        let dir = temp_dir(&format!("store-{tag}"));
        let cat = embedded();
        for id in [ids::MT_HY_MT2_1_8B, ids::MT_HY_MT2_7B] {
            let e = catalog::entry(&cat, id).unwrap();
            let d = dir.join(id.replace('/', "_"));
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join(catalog::model_file(e).unwrap()), super::gguf::build::hunyuan_like()).unwrap();
        }
        for rt in [RUNTIME_VULKAN, RUNTIME_CPU] {
            let d = dir.join(rt.replace('/', "_"));
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("llama-server.exe"), b"MZ").unwrap();
        }
        Self { dir }
    }

    fn installed(&self, id: &str) -> Option<PathBuf> {
        let d = self.dir.join(id.replace('/', "_"));
        d.is_dir().then_some(d)
    }
}

fn cfg(model: &str) -> LlmServerConfig {
    LlmServerConfig {
        model: model.into(),
        ..Default::default()
    }
}

#[test]
fn resolves_auto_to_the_governors_choice_with_header_facts() {
    let store = FakeStore::new("auto");
    let sys = system(vec![gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce RTX 4090", 24_142, Some("8.9"), Some("ada"))]);
    let r = resolve_with(&cfg("auto"), &embedded(), sys.clone(), &snap(false), &|id| store.installed(id)).unwrap();
    // 24 GB, voice changer off: Hy-MT2 7B on the GPU; the CUDA pack is not installed, so Vulkan.
    assert_eq!(r.planned.plan.model, ids::MT_HY_MT2_7B);
    assert!(r.planned.plan.allowed, "{}", r.planned.plan.reason);
    assert_eq!((r.planned.plan.variant.as_str(), r.planned.plan.device.as_str()), ("vulkan", "gpu"));
    assert!(r.planned.plan.missing.is_empty());
    assert_eq!(r.facts.layers, Some(32), "read from the installed GGUF header");
    assert!(r.model_path.as_ref().unwrap().is_file());
    let l = make_launch(&r).unwrap();
    assert_eq!(l.pin, Pin::None, "one GPU: nothing to pin");
    assert_eq!(l.alias, "hy-mt2-7b-q4_k_m");
    assert_eq!(l.vram_mb, 5300);
    assert_eq!(
        l.exe,
        store
            .installed(RUNTIME_VULKAN)
            .unwrap()
            .join("llama-server.exe")
    );

    // 8 GB with the voice changer on the GPU: auto picks 1.8B; an explicit 7B is refused.
    let small = system(vec![gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce RTX 4060", 8188, Some("8.9"), Some("ada"))]);
    let r = resolve_with(&cfg("auto"), &embedded(), small.clone(), &snap(true), &|id| store.installed(id)).unwrap();
    assert_eq!(r.planned.plan.model, ids::MT_HY_MT2_1_8B);
    assert!(r.planned.plan.allowed);
    let r = resolve_with(&cfg(ids::MT_HY_MT2_7B), &embedded(), small, &snap(true), &|id| store.installed(id)).unwrap();
    assert!(!r.planned.plan.allowed && !r.planned.hard_refusal);
    assert_eq!(r.planned.plan.reason_code, "vc_running");

    // Not installed: the missing ids (model and pack) for the consent UI.
    let r = resolve_with(&cfg(ids::MT_MILMMT_4B), &embedded(), sys.clone(), &snap(false), &|_| None).unwrap();
    assert_eq!(r.planned.plan.missing, vec![ids::MT_MILMMT_4B.to_string(), RUNTIME_VULKAN.to_string()]);
    let e = make_launch(&r).unwrap_err();
    assert_eq!(e.kind, LlmServerErrorKind::NotInstalled);

    // The unrunnable 1.25-bit build is never the automatic choice (a low-memory CPU PC).
    let mut weak = system(vec![gpu(0, 0x1414, "Microsoft Basic Render Driver", 0, None, None)]);
    weak.total_ram_mb = 8000;
    let r = resolve_with(&cfg("auto"), &embedded(), weak, &snap(false), &|id| store.installed(id)).unwrap();
    assert_eq!(r.planned.plan.model, ids::MT_HY_MT2_1_8B);
    assert_eq!(r.planned.plan.variant, "cpu");
}

#[test]
fn an_accepted_refusal_launches_the_refused_gpu_placement() {
    let store = FakeStore::new("override");
    let small = system(vec![gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce RTX 4060", 8188, Some("8.9"), Some("ada"))]);
    // Hy-MT2 7B on 8 GB with the voice changer on the GPU: refused, overridable.
    let r = resolve_with(&cfg(ids::MT_HY_MT2_7B), &embedded(), small.clone(), &snap(true), &|id| store.installed(id)).unwrap();
    assert!(!r.planned.plan.allowed && !r.planned.hard_refusal);
    let e = server::prepare_launch(&r, false).unwrap_err();
    assert_eq!(e.kind, LlmServerErrorKind::Refused);
    // "Start it anyway": the GPU run it refused, with the VRAM claim, not a silent CPU run.
    let l = server::prepare_launch(&r, true).unwrap();
    assert_eq!((l.variant.as_str(), l.gpu_layers, l.vram_mb), ("vulkan", plan::ALL_LAYERS, 5300));
    assert!(l.no_poll && !l.below_normal && l.threads == 2, "{l:?}");
    let spec = build_spec(&l);
    let args = spec.args.join(" ");
    assert!(args.contains("-ngl 999") && !args.contains("--device none"), "{args}");
    assert_eq!(spec.priority, HelperPriority::Normal);

    // Hand-set layers that were refused keep their count.
    let mut c = cfg(ids::MT_HY_MT2_7B);
    c.variant = "vulkan".into();
    c.gpu_layers = Some(32);
    let r = resolve_with(&c, &embedded(), small.clone(), &snap(true), &|id| store.installed(id)).unwrap();
    assert!(!r.planned.plan.allowed);
    let l = server::prepare_launch(&r, true).unwrap();
    assert_eq!(l.gpu_layers, 32);
    assert!(l.vram_mb > 0);
    assert!(build_spec(&l).args.join(" ").contains("-ngl 32"));

    // A hard refusal is never launched and offers no download.
    let amd = system(vec![gpu(0, VENDOR_AMD, "AMD Radeon RX 7800 XT", 16_368, None, Some("rdna3"))]);
    let mut c = cfg(ids::MT_HY_MT2_1_8B);
    c.variant = "cuda13".into();
    let r = resolve_with(&c, &embedded(), amd, &snap(false), &|id| store.installed(id)).unwrap();
    assert!(r.planned.hard_refusal);
    let e = server::prepare_launch(&r, true).unwrap_err();
    assert_eq!(e.kind, LlmServerErrorKind::Refused);
    assert!(e.missing.is_empty(), "{:?}", e.missing);

    // An allowed plan is launched as planned.
    let r = resolve_with(&cfg(ids::MT_HY_MT2_1_8B), &embedded(), small, &snap(true), &|id| store.installed(id)).unwrap();
    assert!(r.planned.plan.allowed);
    assert_eq!(server::prepare_launch(&r, false).unwrap().gpu_layers, plan::ALL_LAYERS);
}

#[test]
fn resolves_user_gguf_files_and_rejects_bad_input() {
    let dir = temp_dir("byo");
    let good = dir.join("My Model.gguf");
    std::fs::write(&good, super::gguf::build::hunyuan_like()).unwrap();
    let sys = system(vec![gpu(0, VENDOR_AMD, "AMD Radeon RX 7900 XTX", 24_560, None, Some("rdna3"))]);
    let store = FakeStore::new("byo");
    let r = resolve_with(&cfg(&good.display().to_string()), &embedded(), sys.clone(), &snap(false), &|id| {
        store.installed(id)
    })
    .unwrap();
    assert!(!r.facts.catalog && r.facts.installed);
    assert_eq!(r.facts.kv_bytes_per_token, Some(65_536));
    assert!(r.planned.plan.allowed && r.planned.plan.missing.is_empty(), "{:?}", r.planned.plan);
    let l = make_launch(&r).unwrap();
    assert_eq!((l.model.as_str(), l.alias.as_str()), ("My Model.gguf", "my-model"));

    let bad = dir.join("not-a-model.gguf");
    std::fs::write(&bad, b"PK\x03\x04 this is a zip file, not a GGUF model at all").unwrap();
    let e = resolve_with(&cfg(&bad.display().to_string()), &embedded(), sys.clone(), &snap(false), &|_| None).unwrap_err();
    assert_eq!(e.kind, LlmServerErrorKind::InvalidInput);
    assert!(e.message.contains("not a GGUF"), "{}", e.message);
    let e = resolve_with(
        &cfg(&dir.join("missing.gguf").display().to_string()),
        &embedded(),
        sys.clone(),
        &snap(false),
        &|_| None,
    )
    .unwrap_err();
    assert!(e.message.contains("not found"), "{}", e.message);
    let e = resolve_with(&cfg("relative\\model.gguf"), &embedded(), sys.clone(), &snap(false), &|_| None).unwrap_err();
    assert!(e.message.contains("full path"), "{}", e.message);
    let e = resolve_with(&cfg("mt/not-a-model"), &embedded(), sys.clone(), &snap(false), &|_| None).unwrap_err();
    assert_eq!(e.kind, LlmServerErrorKind::InvalidInput);
    let mut c = cfg("auto");
    c.variant = "rocm".into();
    assert!(resolve_with(&c, &embedded(), sys.clone(), &snap(false), &|_| None)
        .unwrap_err()
        .message
        .contains("variant"));
    let mut c = cfg("auto");
    c.flash_attn = Some("maybe".into());
    assert!(resolve_with(&c, &embedded(), sys, &snap(false), &|_| None)
        .unwrap_err()
        .message
        .contains("flash"));
}

#[test]
fn normalize_fills_defaults() {
    let (model, k) = server::normalize(&LlmServerConfig {
        model: "  ".into(),
        variant: " CUDA13 ".into(),
        ctx_size: Some(0),
        gpu_layers: Some(5000),
        flash_attn: Some("ON".into()),
        allow_over_budget: true,
    })
    .unwrap();
    assert_eq!(model, MODEL_AUTO);
    assert_eq!(k.variant, "cuda13");
    assert_eq!(k.ctx_size, None);
    assert_eq!(k.gpu_layers, Some(plan::ALL_LAYERS));
    assert_eq!(k.flash_attn, "on");
    assert!(k.allow_over_budget);
}

#[test]
fn spec_flags_follow_the_launch() {
    let mut l = launch("t-llm-spec", 4096);
    l.exe = PathBuf::from("C:/rt/llama-server.exe");
    let spec = build_spec(&l);
    let args = spec.args.join(" ");
    assert!(args.starts_with("--host 127.0.0.1 --port {port} -m "), "{args}");
    for part in [
        "-c 4096",
        "-ngl 999",
        "-fa auto",
        "-np 1",
        "-t 2",
        "--alias hy-mt2-1.8b-q4_k_m",
        "--no-webui",
        "-fit off",
        "--reasoning off",
        "--log-colors off",
        "--poll 0",
        "--device Vulkan0",
    ] {
        assert!(args.contains(part), "{part}: {args}");
    }
    assert!(!args.contains("api"), "{args}");
    assert!(spec.health.as_ref().unwrap().send_api_key);
    assert_eq!(spec.priority, HelperPriority::Normal);
    assert_eq!(spec.env, [("LLAMA_API_KEY".to_string(), "{api_key}".to_string())]);

    l.pin = Pin::SingleDevice;
    l.no_poll = false;
    let args = build_spec(&l).args.join(" ");
    assert!(
        args.contains("-sm none") && !args.contains("--device") && !args.contains("--poll"),
        "{args}"
    );

    l.gpu_layers = 0;
    l.variant = "cpu".into();
    l.below_normal = true;
    let spec = build_spec(&l);
    let args = spec.args.join(" ");
    assert!(
        args.contains("-ngl 0") && args.contains("--device none") && !args.contains("-sm none"),
        "{args}"
    );
    assert_eq!(spec.priority, HelperPriority::BelowNormal);
    assert_eq!(server::load_timeout_ms(0), 60_000);
    assert_eq!(server::load_timeout_ms(5 << 30), 210_000);
    assert_eq!(server::load_timeout_ms(u64::MAX), 600_000);
}

#[test]
fn model_list_marks_the_choice_and_the_specialists() {
    let cat = embedded();
    let sys = system(vec![gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce RTX 4060", 8188, Some("8.9"), Some("ada"))]);
    let installed = |id: &str| id == ids::MT_HY_MT2_1_8B;
    let rows = server::list_models_with(&cat, &sys, &snap(true), &installed);
    assert_eq!(rows.len(), catalog::ordered_ids().len());
    assert_eq!(rows[0].id, ids::MT_HY_MT2_1_8B, "catalog order");
    let rec: Vec<&LlmModelInfo> = rows.iter().filter(|r| r.recommended).collect();
    assert_eq!(rec.len(), 1);
    assert_eq!(rec[0].id, ids::MT_HY_MT2_1_8B, "8 GB with the voice changer on: 1.8B");
    assert!(
        rows.iter()
            .find(|r| r.id == ids::MT_HY_MT2_1_8B)
            .unwrap()
            .installed
    );
    assert!(
        !rows
            .iter()
            .find(|r| r.id == ids::MT_HY_MT2_7B)
            .unwrap()
            .installed
    );
    let tiny = rows
        .iter()
        .find(|r| r.id == ids::MT_HY_MT2_1_8B_TINY)
        .unwrap();
    assert_eq!(tiny.availability, "needs_runtime_update");
    let vntl = rows
        .iter()
        .find(|r| r.id == "mt/vntl-llama3-8b-v2-q5_k_m")
        .unwrap();
    assert!(vntl.specialist && vntl.role == "specialist");
    assert_eq!(
        vntl.directions,
        vec![LlmDirection {
            from: "ja".into(),
            to: "en".into()
        }]
    );
    assert_eq!(vntl.prompt_format, "vntl");
    let hy = rows.iter().find(|r| r.id == ids::MT_HY_MT2_7B).unwrap();
    assert_eq!(hy.sampling.as_ref().and_then(|s| s.top_k), Some(20));
    assert_eq!(hy.vram_mb, 5300);
    assert!(hy.size_bytes > 4_000_000_000 && hy.license == "Apache-2.0");
    // Wire names are camelCase.
    let v = serde_json::to_value(hy).unwrap();
    assert!(v.get("promptFormat").is_some() && v.get("sizeBytes").is_some() && v["sampling"].get("repetitionPenalty").is_some());
}

// ---------------------------------------------------------------------------------------------
// Manual run with a real llama-server and model
// ---------------------------------------------------------------------------------------------

/// End to end with the real llama.cpp runtime and Hy-MT2 1.8B through `server::start_launch`
/// (device pin, spec, readiness, a translation, the measured claim, stop). Manual:
///
/// ```powershell
/// $env:SIGIL_TEST_LLM_RUNTIME = "<installed runtime/llama-server-vulkan folder>"
/// $env:SIGIL_TEST_LLM_GGUF    = "<path>\Hy-MT2-1.8B-Q4_K_M.gguf"
/// cargo test --manifest-path src-tauri/Cargo.toml real_model_end_to_end -- --ignored --nocapture
/// ```
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a llama.cpp runtime pack and a GGUF (SIGIL_TEST_LLM_RUNTIME, SIGIL_TEST_LLM_GGUF)"]
async fn real_model_end_to_end() {
    let (Ok(rt), Ok(gguf)) = (std::env::var("SIGIL_TEST_LLM_RUNTIME"), std::env::var("SIGIL_TEST_LLM_GGUF")) else {
        eprintln!("SIGIL_TEST_LLM_RUNTIME / SIGIL_TEST_LLM_GGUF not set; skipping");
        return;
    };
    let _s = serial().await;
    let exe = Path::new(&rt).join("llama-server.exe");
    let sys = crate::services::sysinfo::cached();
    let pin = if devices::hardware_gpu_count(&sys) > 1 {
        let list = devices::list_devices(&exe).expect("device list");
        eprintln!("devices: {list:?}");
        devices::pick_device(&list, sys.primary().map(|g| g.adapter.name.as_str()).unwrap_or(""))
    } else {
        Pin::None
    };
    let mut l = launch("t-llm-real", 4096);
    l.helper_name = HELPER_NAME.into();
    l.exe = exe;
    l.model_path = PathBuf::from(&gguf);
    l.pin = pin;
    l.load_timeout_ms = 180_000;
    let t0 = Instant::now();
    let started = server::start_launch(l.clone(), server::stop_epoch())
        .await
        .expect("llama-server starts");
    eprintln!("ready in {} ms: {:?}", t0.elapsed().as_millis(), started.status);
    let body = serde_json::json!({
        "model": started.endpoint.model,
        "messages": [{"role": "user", "content":
            "Translate the following text into Japanese. Note that you should only output the translated result without any additional explanation:\n\nGood morning, everyone! The stream starts in five minutes."}],
        "max_tokens": 128, "temperature": 0.7, "top_p": 0.6, "top_k": 20, "repeat_penalty": 1.05,
    });
    let t1 = Instant::now();
    let resp = client()
        .post(format!("{}/chat/completions", started.endpoint.base_url))
        .bearer_auth(&started.endpoint.api_key)
        .json(&body)
        .timeout(Duration::from_secs(120))
        .send()
        .await
        .expect("chat");
    assert!(resp.status().is_success(), "HTTP {}", resp.status());
    let v: serde_json::Value = resp.json().await.unwrap();
    let text = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    eprintln!("translation in {} ms: {text}\ntimings: {}", t1.elapsed().as_millis(), v["timings"]);
    assert!(!text.trim().is_empty());
    tokio::time::sleep(Duration::from_secs(2)).await;
    eprintln!("status after measuring: {:?}", server::status());
    let pid = process_host::get(HELPER_NAME).and_then(|i| i.pid).unwrap();
    server::stop().await.unwrap();
    assert!(process_host::job::wait_pid_exit(pid, Duration::from_secs(5)));
}

// ---------------------------------------------------------------------------------------------
// The fake llama-server (runs in a child process)
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "child-process entry point of the llm_server tests; does nothing without SIGIL_LLM_FAKE"]
fn fake_llama_server_entry() {
    let Ok(role) = std::env::var(ROLE_ENV) else {
        return;
    };
    let port: u16 = std::env::var("SIGIL_LLM_FAKE_PORT")
        .unwrap()
        .parse()
        .unwrap();
    let key = std::env::var("LLAMA_API_KEY").unwrap_or_default();
    let load_ms: u64 = std::env::var("SIGIL_LLM_FAKE_LOAD_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let args = std::env::var("SIGIL_LLM_FAKE_ARGS").unwrap_or_default();
    eprintln!("fake-llama args: {args}");
    let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind");
    let t0 = Instant::now();
    eprintln!("load_tensors:      Vulkan0 model buffer size =  1000.00 MiB");
    eprintln!("llama_kv_cache:    Vulkan0 KV buffer size =   200.00 MiB");
    eprintln!("llama_context:    Vulkan0 compute buffer size =    50.00 MiB");
    eprintln!("llama_context: Vulkan_Host compute buffer size =    16.00 MiB");
    let mut ready_at: Option<Instant> = None;
    listener.set_nonblocking(true).unwrap();
    loop {
        if role == "crash_after_ready" && ready_at.is_some_and(|t| t.elapsed() > Duration::from_millis(300)) {
            std::process::exit(3);
        }
        let (mut sock, _) = match listener.accept() {
            Ok(s) => s,
            Err(_) => {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
        };
        let _ = sock.set_nonblocking(false);
        let _ = sock.set_read_timeout(Some(Duration::from_secs(5)));
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
            match sock.read(&mut tmp) {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
            }
        }
        let head = String::from_utf8_lossy(&buf).to_string();
        let path = head.split_whitespace().nth(1).unwrap_or("/").to_string();
        let loaded = t0.elapsed() >= Duration::from_millis(load_ms);
        let authed = head.lines().any(|l| {
            l.to_ascii_lowercase().starts_with("authorization:") && l.split_once(':').map(|(_, v)| v.trim()) == Some(format!("Bearer {key}").as_str())
        });
        let (status, body) = match path.as_str() {
            "/health" if loaded => {
                ready_at.get_or_insert_with(Instant::now);
                ("200 OK", "{\"status\":\"ok\"}")
            }
            "/health" => ("503 Service Unavailable", "{\"error\":{\"message\":\"Loading model\"}}"),
            "/v1/chat/completions" if !authed => ("401 Unauthorized", "{\"error\":{\"message\":\"Invalid API Key\"}}"),
            "/v1/chat/completions" => ("200 OK", "{\"choices\":[{\"message\":{\"content\":\"hello\"}}]}"),
            _ => ("404 Not Found", "{}"),
        };
        let _ = sock.write_all(
            format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        );
        let _ = sock.flush();
    }
}
