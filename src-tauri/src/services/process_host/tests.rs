//! Integration tests with real child processes. The helper under test is this test binary itself,
//! re-invoked with a filter that runs only [`fake_helper_entry`] and an environment variable that
//! picks its role (HTTP server, crasher, sleeper, process tree, memory hog, fake Sigil parent).
//! Nothing here touches the network or listens beyond 127.0.0.1 (no firewall prompt), except the
//! `#[ignore]`d exposure test.

use super::*;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, TcpListener};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const ROLE_ENV: &str = "SIGIL_PH_FAKE";
const ENTRY: &str = "services::process_host::tests::fake_helper_entry";
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

fn test_exe() -> PathBuf {
    std::env::current_exe().expect("test binary path")
}

fn fake_args() -> Vec<String> {
    ["--exact", ENTRY, "--ignored", "--nocapture", "--test-threads=1", "-q"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn fake_spec(name: &str, role: &str, env: &[(&str, &str)]) -> HelperSpec {
    let mut all = vec![(ROLE_ENV.to_string(), role.to_string())];
    all.extend(env.iter().map(|(k, v)| (k.to_string(), v.to_string())));
    HelperSpec {
        name: name.into(),
        program: test_exe(),
        args: fake_args(),
        env: all,
        ..Default::default()
    }
}

/// An HTTP fake on an auto port with a health check.
fn http_spec(name: &str, env: &[(&str, &str)]) -> HelperSpec {
    let mut e = vec![("SIGIL_PH_PORT", "{port}")];
    e.extend_from_slice(env);
    HelperSpec {
        port: PortRequest::Auto,
        health: Some(HealthCheck {
            path: "/health".into(),
            timeout_ms: 30_000,
            interval_ms: 50,
            send_api_key: false,
        }),
        ..fake_spec(name, "http", &e)
    }
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sigil-ph-test-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn http_client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

async fn get_status(url: &str, bearer: Option<&str>) -> u16 {
    let mut req = http_client().get(url).timeout(Duration::from_secs(5));
    if let Some(k) = bearer {
        req = req.bearer_auth(k);
    }
    req.send().await.map(|r| r.status().as_u16()).unwrap_or(0)
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

fn read_pids(path: &Path, timeout: Duration) -> Vec<u32> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(text) = std::fs::read_to_string(path) {
            let pids: Vec<u32> = text
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if !pids.is_empty() {
                return pids;
            }
        }
        assert!(Instant::now() < deadline, "timed out waiting for {}", path.display());
        std::thread::sleep(Duration::from_millis(25));
    }
}

/// Kills a directly spawned child when a test fails half-way.
struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// A copy of a small Windows program whose first static import is renamed to a DLL that does not
/// exist (same length, so the image stays valid): the loader fails it with STATUS_DLL_NOT_FOUND
/// (0xC0000135) before any of its code runs.
fn exe_with_missing_dll(dir: &Path) -> PathBuf {
    let system = std::env::var_os("SystemRoot")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
    let mut img = std::fs::read(system.join("System32").join("hostname.exe")).expect("hostname.exe");
    let u16_at = |img: &[u8], o: usize| usize::from(u16::from_le_bytes([img[o], img[o + 1]]));
    let u32_at = |img: &[u8], o: usize| u32::from_le_bytes([img[o], img[o + 1], img[o + 2], img[o + 3]]) as usize;
    let pe = u32_at(&img, 0x3c);
    assert_eq!(&img[pe..pe + 4], b"PE\0\0");
    let coff = pe + 4;
    let sections = u16_at(&img, coff + 2);
    let opt = coff + 20;
    let section_table = opt + u16_at(&img, coff + 16);
    let data_dirs = opt + if u16_at(&img, opt) == 0x20b { 112 } else { 96 };
    let to_offset = |img: &[u8], rva: usize| -> usize {
        (0..sections)
            .map(|i| section_table + 40 * i)
            .find_map(|sec| {
                let (vsize, va, raw, ptr) = (u32_at(img, sec + 8), u32_at(img, sec + 12), u32_at(img, sec + 16), u32_at(img, sec + 20));
                (va <= rva && rva < va + vsize.max(raw)).then(|| rva - va + ptr)
            })
            .expect("RVA inside a section")
    };
    // Data directory 1 = import table; the first IMAGE_IMPORT_DESCRIPTOR's Name RVA is at +12.
    let first_import = to_offset(&img, u32_at(&img, data_dirs + 8));
    let name = to_offset(&img, u32_at(&img, first_import + 12));
    let mut i = name;
    while img[i] != b'.' && img[i] != 0 {
        img[i] = b'q';
        i += 1;
    }
    assert!(i > name, "empty import name");
    let out = dir.join("sigil-ph-missing-dll.exe");
    std::fs::write(&out, &img).unwrap();
    out
}

fn spawn_fake_directly(role: &str, env: &[(&str, &Path)]) -> ChildGuard {
    let mut cmd = Command::new(test_exe());
    cmd.args(fake_args())
        .env(ROLE_ENV, role)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW);
    for (k, v) in env {
        cmd.env(k, v);
    }
    ChildGuard(cmd.spawn().expect("spawn fake parent"))
}

// ---------------------------------------------------------------------------------------------
// Spec validation and substitution
// ---------------------------------------------------------------------------------------------

#[test]
fn validate_rejects_bad_specs() {
    let ok = http_spec("ok-spec", &[]);
    spawn::validate(&ok).unwrap();
    let cases: Vec<(HelperSpec, &str)> = vec![
        (
            HelperSpec {
                name: "".into(),
                ..ok.clone()
            },
            "invalid helper name",
        ),
        (
            HelperSpec {
                name: "Upper".into(),
                ..ok.clone()
            },
            "invalid helper name",
        ),
        (
            HelperSpec {
                name: "-x".into(),
                ..ok.clone()
            },
            "invalid helper name",
        ),
        (
            HelperSpec {
                program: PathBuf::new(),
                ..ok.clone()
            },
            "no program",
        ),
        (
            HelperSpec {
                port: PortRequest::Fixed(0),
                ..ok.clone()
            },
            "port 0",
        ),
        (
            HelperSpec {
                port: PortRequest::None,
                ..ok.clone()
            },
            "needs a port",
        ),
        (
            HelperSpec {
                health: Some(HealthCheck {
                    path: "health".into(),
                    ..Default::default()
                }),
                ..ok.clone()
            },
            "invalid health path",
        ),
        (
            HelperSpec {
                health: Some(HealthCheck {
                    path: "//evil.example/x".into(),
                    ..Default::default()
                }),
                ..ok.clone()
            },
            "invalid health path",
        ),
        (
            HelperSpec {
                memory_limit_mb: Some(4),
                ..ok.clone()
            },
            "below the minimum",
        ),
        (
            HelperSpec {
                args: vec!["--key".into(), "{api_key}".into()],
                api_key: false,
                ..ok.clone()
            },
            "api_key is false",
        ),
        (
            HelperSpec {
                args: vec!["{port}".into()],
                port: PortRequest::None,
                health: None,
                ..ok.clone()
            },
            "has no port",
        ),
        (
            HelperSpec {
                env: vec![("A=B".into(), "1".into())],
                ..ok.clone()
            },
            "invalid environment",
        ),
        (
            HelperSpec {
                program: PathBuf::from("C:/definitely/not/here/helper.exe"),
                ..ok.clone()
            },
            "program not found",
        ),
        (
            HelperSpec {
                cwd: Some(PathBuf::from("C:/definitely/not/here")),
                ..ok.clone()
            },
            "working directory not found",
        ),
    ];
    for (spec, needle) in cases {
        let err = spawn::validate(&spec).expect_err(needle);
        assert!(err.contains(needle), "{err:?} should mention {needle:?}");
    }
}

#[test]
fn substitution_and_env_patterns() {
    assert_eq!(spawn::substitute("--port={port}", Some(8080), None), "--port=8080");
    assert_eq!(spawn::substitute("{api_key}/{port}/{api_key}", Some(1), Some("k")), "k/1/k");
    assert_eq!(spawn::substitute("plain", None, None), "plain");
    assert!(spawn::env_pattern_matches("LLAMA_ARG_*", "llama_arg_host"));
    assert!(spawn::env_pattern_matches("LLAMA_API_KEY", "llama_api_key"));
    assert!(!spawn::env_pattern_matches("LLAMA_API_KEY", "LLAMA_API_KEY_FILE"));
    assert!(!spawn::env_pattern_matches("LLAMA_ARG_*", "LLAMA"));
    assert!(!spawn::env_pattern_matches("É*", "E"));
}

#[test]
fn random_keys_are_long_and_unique() {
    let a = spawn::random_key().unwrap();
    let b = spawn::random_key().unwrap();
    assert_eq!(a.len(), 64);
    assert!(a.bytes().all(|c| c.is_ascii_hexdigit()));
    assert_ne!(a, b);
}

#[test]
fn contract_types_serialize_as_the_ts_mirror_expects() {
    let info = HelperInfo {
        name: "llama-server".into(),
        state: HelperState::Restarting,
        pid: Some(7),
        port: None,
        restarts: 2,
        error: Some("boom".into()),
    };
    assert_eq!(
        serde_json::to_value(&info).unwrap(),
        serde_json::json!({"name":"llama-server","state":"restarting","pid":7,"port":null,"restarts":2,"error":"boom"})
    );
    let handle = HelperHandle {
        name: "x".into(),
        pid: 1,
        port: Some(2),
        base_url: Some("http://127.0.0.1:2".into()),
        api_key: Some("secret".into()),
    };
    assert!(!serde_json::to_string(&handle).unwrap().contains("secret"));
    // Specs from older JSON (before the P1-E fields) still parse.
    let old = r#"{"name":"a","program":"a.exe","args":[],"env":[],"cwd":null,"port":{"mode":"auto"},
        "api_key":false,"health":{"path":"/health","timeout_ms":1,"interval_ms":1},"memory_limit_mb":null,
        "restart":{"mode":"on_failure","max_restarts":1,"backoff_ms":5}}"#;
    let spec: HelperSpec = serde_json::from_str(old).unwrap();
    assert_eq!(spec.priority, HelperPriority::Normal);
    assert!(!spec.health.unwrap().send_api_key);
}

// ---------------------------------------------------------------------------------------------
// Lifecycle with real processes
// ---------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn healthy_helper_gets_a_loopback_port_and_key_then_stops() {
    let name = "t-healthy";
    let spec = HelperSpec {
        api_key: true,
        ..http_spec(name, &[("SIGIL_PH_READY_MS", "400"), ("SIGIL_PH_KEY", "{api_key}")])
    };
    let t0 = Instant::now();
    let handle = spawn(spec).await.expect("helper starts");
    assert!(t0.elapsed() >= Duration::from_millis(400), "health must wait for 2xx (503 while loading)");
    let port = handle.port.expect("port");
    assert_eq!(handle.base_url.as_deref(), Some(format!("http://127.0.0.1:{port}").as_str()));
    let key = handle.api_key.clone().expect("key");
    assert_eq!(key.len(), 64);

    let base = handle.base_url.clone().unwrap();
    assert_eq!(get_status(&format!("{base}/auth"), Some(&key)).await, 200);
    assert_eq!(get_status(&format!("{base}/auth"), None).await, 401);
    assert_eq!(get_status(&format!("{base}/auth"), Some("wrong")).await, 401);

    let info = get(name).expect("listed");
    assert_eq!(info.state, HelperState::Running);
    assert_eq!(info.pid, Some(handle.pid));
    assert_eq!(info.port, Some(port));
    assert!(list().iter().any(|i| i.name == name));
    assert_eq!(super::handle(name).map(|h| h.pid), Some(handle.pid));

    // Output reached the log book, with the key redacted.
    assert!(
        wait_until(Duration::from_secs(5), || logs(name)
            .iter()
            .any(|l| l.text.contains("fake http helper")))
        .await
    );
    let lines = logs(name);
    assert!(lines.iter().all(|l| !l.text.contains(&key)), "key leaked into logs");
    assert!(lines.iter().any(|l| l.text.contains(logs::REDACTED)));

    stop(name).await.unwrap();
    assert!(job::wait_pid_exit(handle.pid, Duration::from_secs(5)), "helper must be dead after stop");
    assert_eq!(get(name).unwrap().state, HelperState::Stopped);
    assert!(super::handle(name).is_none());
    stop(name).await.unwrap(); // idempotent
    stop("t-never-started").await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn health_timeout_kills_the_helper_and_fails() {
    let name = "t-health-timeout";
    let spec = HelperSpec {
        port: PortRequest::Auto,
        health: Some(HealthCheck {
            path: "/health".into(),
            timeout_ms: 1500,
            interval_ms: 50,
            send_api_key: false,
        }),
        ..fake_spec(name, "sleep", &[])
    };
    let t0 = Instant::now();
    let err = spawn(spec).await.expect_err("never healthy");
    assert!(err.contains("did not become healthy within 1.5 s"), "{err}");
    assert!(t0.elapsed() < Duration::from_secs(20));
    let info = get(name).unwrap();
    assert_eq!(info.state, HelperState::Failed);
    assert!(info.error.unwrap().contains("did not become healthy"));
    assert!(info.pid.is_none());
    let started = logs(name)
        .iter()
        .find_map(|l| {
            l.text
                .strip_prefix("started pid ")
                .map(|s| s.split_whitespace().next().unwrap().to_string())
        })
        .expect("start note");
    let pid: u32 = started.parse().unwrap();
    assert!(job::wait_pid_exit(pid, Duration::from_secs(5)), "timed-out helper must be killed");
}

#[tokio::test(flavor = "multi_thread")]
async fn startup_crash_reports_exit_code_and_last_output() {
    let name = "t-startup-crash";
    let spec = http_spec(name, &[]);
    let spec = HelperSpec {
        env: vec![(ROLE_ENV.into(), "exit".into()), ("SIGIL_PH_EXIT_CODE".into(), "7".into())],
        ..spec
    };
    let err = spawn(spec).await.expect_err("crashes");
    assert!(err.contains("exited while starting (exit code 7)"), "{err}");
    assert!(err.contains("model file not found"), "last output missing: {err}");
    assert_eq!(get(name).unwrap().state, HelperState::Failed);
}

#[tokio::test(flavor = "multi_thread")]
async fn port_collision_is_retried_on_another_port() {
    let name = "t-port-collision";
    // Another program holds the first port the host "picks".
    let squatter = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let taken = squatter.local_addr().unwrap().port();
    ports::forced::set(name, &[taken]);
    let handle = spawn(http_spec(name, &[]))
        .await
        .expect("retried on a free port");
    let port = handle.port.unwrap();
    assert_ne!(port, taken);
    assert!(logs(name)
        .iter()
        .any(|l| l.text.contains(&format!("port {taken} was taken"))));
    assert_eq!(get(name).unwrap().restarts, 0, "a collision is not a failure");
    stop(name).await.unwrap();
    drop(squatter);
}

/// Another local HTTP server took the auto port before the helper could bind it and answers
/// `/health` with 200. The listener table shows it is not the helper's: it is a collision (retried
/// on a new port), and it never receives an HTTP request, let alone the key.
#[tokio::test(flavor = "multi_thread")]
async fn a_foreign_server_on_the_port_is_a_collision_and_never_gets_the_key() {
    let name = "t-foreign-http";
    let squatter = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let taken = squatter.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    {
        let seen = seen.clone();
        std::thread::spawn(move || {
            for stream in squatter.incoming() {
                let Ok(mut s) = stream else { continue };
                let _ = s.set_read_timeout(Some(Duration::from_secs(1)));
                let mut buf = [0u8; 4096];
                let n = s.read(&mut buf).unwrap_or(0);
                seen.lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&buf[..n]).into_owned());
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            }
        });
    }
    ports::forced::set(name, &[taken]);
    let spec = HelperSpec {
        api_key: true,
        health: Some(HealthCheck {
            path: "/health".into(),
            timeout_ms: 30_000,
            interval_ms: 50,
            send_api_key: true,
        }),
        // The helper keeps running after its bind failed (the worst case: no exit to notice).
        ..http_spec(name, &[("SIGIL_PH_LINGER_ON_BIND_ERROR", "1"), ("SIGIL_PH_KEY", "{api_key}")])
    };
    let handle = spawn(spec).await.expect("retried on a free port");
    let key = handle.api_key.clone().unwrap();
    assert_ne!(handle.port, Some(taken), "the foreign server's 200 was taken for the helper");
    assert_eq!(get(name).unwrap().restarts, 0, "a collision is not a failure");
    let lines = logs(name);
    assert!(
        lines
            .iter()
            .any(|l| l.text.contains(&format!("port {taken} was taken"))),
        "{lines:?}"
    );
    let requests = seen.lock().unwrap().clone();
    assert!(
        requests.iter().all(|r| r.is_empty() && !r.contains(&key)),
        "the squatter received {requests:?}"
    );
    // The lingering first instance was killed; the retry is the real helper.
    let first_pid: u32 = lines
        .iter()
        .find_map(|l| l.text.strip_prefix("started pid "))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|p| p.parse().ok())
        .expect("start note");
    assert_ne!(first_pid, handle.pid);
    assert!(job::wait_pid_exit(first_pid, Duration::from_secs(5)));
    let base = handle.base_url.clone().unwrap();
    assert_eq!(get_status(&format!("{base}/auth"), Some(&key)).await, 200);
    stop(name).await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn fixed_port_in_use_is_rejected_without_starting() {
    let name = "t-fixed-busy";
    let squatter = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let taken = squatter.local_addr().unwrap().port();
    let spec = HelperSpec {
        port: PortRequest::Fixed(taken),
        ..http_spec(name, &[])
    };
    let err = spawn(spec).await.expect_err("port busy");
    assert!(err.contains(&format!("port {taken} on 127.0.0.1 is already in use")), "{err}");
    assert!(!logs(name).iter().any(|l| l.text.starts_with("started pid")));
}

#[tokio::test(flavor = "multi_thread")]
async fn crashed_helper_is_restarted_then_given_up() {
    let name = "t-restart";
    let spec = HelperSpec {
        restart: RestartPolicy::OnFailure {
            max_restarts: 2,
            backoff_ms: 50,
        },
        api_key: true,
        ..http_spec(name, &[("SIGIL_PH_DIE_MS", "1500"), ("SIGIL_PH_KEY", "{api_key}")])
    };
    let first = spawn(spec).await.expect("starts");
    // Each instance dies 1.5 s after binding its port: two restarts, then the budget is spent.
    assert!(
        wait_until(Duration::from_secs(30), || get(name).is_some_and(|i| i.state == HelperState::Failed)).await,
        "{:?}",
        get(name)
    );
    let info = get(name).unwrap();
    assert_eq!(info.restarts, 2);
    let err = info.error.unwrap();
    assert!(err.contains("exited unexpectedly (exit code 3)"), "{err}");
    assert!(err.contains("gave up after 2 restarts"), "{err}");
    assert!(err.contains("simulated crash"), "{err}");
    assert!(!job::pid_alive(first.pid));
    let restarted = logs(name)
        .iter()
        .filter(|l| l.text.starts_with("restarting in"))
        .count();
    assert_eq!(restarted, 2);
    // The key survived the restarts (same log book, redacted everywhere).
    let key = first.api_key.unwrap();
    assert!(logs(name).iter().all(|l| !l.text.contains(&key)));
}

#[tokio::test(flavor = "multi_thread")]
async fn restart_brings_a_crashed_helper_back() {
    let name = "t-restart-ok";
    // The first instance dies 1 s after binding; the marker file makes later instances stay up.
    let dir = temp_dir(name);
    let marker = dir.join("crashed-once");
    let marker_s = marker.display().to_string();
    let spec = HelperSpec {
        restart: RestartPolicy::OnFailure {
            max_restarts: 3,
            backoff_ms: 50,
        },
        ..http_spec(name, &[("SIGIL_PH_DIE_ONCE", marker_s.as_str()), ("SIGIL_PH_DIE_MS", "1000")])
    };
    let first = spawn(spec).await.expect("starts");
    assert!(
        wait_until(Duration::from_secs(30), || get(name)
            .is_some_and(|i| i.state == HelperState::Running && i.restarts == 1 && i.pid != Some(first.pid)))
        .await,
        "{:?}",
        get(name)
    );
    let second = super::handle(name).expect("running again");
    assert_ne!(second.pid, first.pid);
    let url = format!("{}/health", second.base_url.unwrap());
    assert_eq!(get_status(&url, None).await, 200);
    stop(name).await.unwrap();
    assert!(job::wait_pid_exit(second.pid, Duration::from_secs(5)));
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn never_policy_marks_a_crash_failed() {
    let name = "t-never";
    let h = spawn(http_spec(name, &[("SIGIL_PH_DIE_MS", "1500")]))
        .await
        .expect("starts");
    assert!(wait_until(Duration::from_secs(20), || get(name).is_some_and(|i| i.state == HelperState::Failed)).await);
    let info = get(name).unwrap();
    assert_eq!(info.restarts, 0);
    assert!(info.error.unwrap().contains("exited unexpectedly"));
    assert!(!job::pid_alive(h.pid));
}

#[tokio::test(flavor = "multi_thread")]
async fn spawning_the_same_name_replaces_the_helper() {
    let name = "t-replace";
    let a = spawn(http_spec(name, &[])).await.unwrap();
    let b = spawn(http_spec(name, &[])).await.unwrap();
    assert_ne!(a.pid, b.pid);
    assert!(job::wait_pid_exit(a.pid, Duration::from_secs(5)), "the replaced helper must die");
    assert!(job::pid_alive(b.pid));
    assert_eq!(lifecycle::current_pid(name), Some(b.pid));
    stop(name).await.unwrap();
    assert!(job::wait_pid_exit(b.pid, Duration::from_secs(5)));
}

/// A helper that loads slowly and crashes soon after its health check passed must use up its
/// restart budget: the budget resets only after the helper *ran ready* for the stable period,
/// never because loading took longer than that.
#[tokio::test(flavor = "multi_thread")]
async fn slow_loading_helper_that_crashes_after_ready_still_exhausts_restarts() {
    let name = "t-slow-load-crash";
    // Stable after 1 s of ready running. Each instance loads for 1.2 s (503) and dies 0.3 s after
    // it became ready: its lifetime exceeds the stable period, its ready time never does.
    lifecycle::test_hooks::set_stable_after(name, Duration::from_secs(1));
    let spec = HelperSpec {
        restart: RestartPolicy::OnFailure {
            max_restarts: 2,
            backoff_ms: 50,
        },
        ..http_spec(name, &[("SIGIL_PH_READY_MS", "1200"), ("SIGIL_PH_DIE_MS", "1500")])
    };
    spawn(spec).await.expect("starts");
    assert!(
        wait_until(Duration::from_secs(40), || get(name).is_some_and(|i| i.state == HelperState::Failed)).await,
        "the budget kept resetting: {:?}",
        get(name)
    );
    let info = get(name).unwrap();
    assert_eq!(info.restarts, 2);
    assert!(info.error.unwrap().contains("gave up after 2 restarts"));
}

/// The other side: a helper that crashes only after running ready for the stable period gets a
/// fresh budget each time (here a budget of one restart never runs out).
#[tokio::test(flavor = "multi_thread")]
async fn a_helper_that_ran_ready_long_enough_gets_a_fresh_restart_budget() {
    let name = "t-stable-reset";
    lifecycle::test_hooks::set_stable_after(name, Duration::from_millis(300));
    let spec = HelperSpec {
        restart: RestartPolicy::OnFailure {
            max_restarts: 1,
            backoff_ms: 50,
        },
        ..http_spec(name, &[("SIGIL_PH_DIE_MS", "900")])
    };
    spawn(spec).await.expect("starts");
    assert!(
        wait_until(Duration::from_secs(30), || get(name).is_some_and(|i| i.restarts >= 3)).await,
        "{:?}",
        get(name)
    );
    assert_ne!(get(name).unwrap().state, HelperState::Failed);
    stop(name).await.unwrap();
    assert_eq!(get(name).unwrap().state, HelperState::Stopped);
}

/// A spawn queued behind the replacement of the running helper, then a stop: the stop is the
/// last request, so the queued spawn never starts and nothing is left running.
#[tokio::test(flavor = "multi_thread")]
async fn stop_cancels_a_spawn_queued_behind_a_replace() {
    let name = "t-stop-queued";
    // The helper ignores stdin EOF, so retiring it takes its whole 1.5 s grace...
    let slow_stop = HelperSpec {
        stop_grace_ms: 1500,
        ..fake_spec(name, "sleep", &[])
    };
    let first = spawn(slow_stop.clone()).await.expect("starts");
    // ...while the replacing spawn holds the gate, with no registry entry of its own yet.
    let replacing = tokio::spawn(spawn(slow_stop));
    tokio::time::sleep(Duration::from_millis(300)).await;
    stop(name).await.unwrap();
    let err = replacing.await.unwrap().expect_err("the later stop wins");
    assert!(err.contains("cancelled"), "{err}");
    assert!(job::wait_pid_exit(first.pid, Duration::from_secs(5)));
    assert_eq!(get(name).unwrap().state, HelperState::Stopped);
    assert!(lifecycle::current_pid(name).is_none(), "the queued spawn started a helper after the stop");
    assert!(super::handle(name).is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_during_a_slow_start_cancels_it_promptly() {
    let name = "t-cancel-start";
    let spec = HelperSpec {
        port: PortRequest::Auto,
        health: Some(HealthCheck {
            path: "/health".into(),
            timeout_ms: 60_000,
            interval_ms: 50,
            send_api_key: false,
        }),
        ..fake_spec(name, "sleep", &[])
    };
    let starter = tokio::spawn(spawn(spec));
    assert!(wait_until(Duration::from_secs(20), || lifecycle::current_pid(name).is_some()).await);
    let pid = lifecycle::current_pid(name).unwrap();
    let t0 = Instant::now();
    stop(name).await.unwrap();
    assert!(t0.elapsed() < Duration::from_secs(10), "stop waited for the health timeout");
    let err = starter.await.unwrap().expect_err("cancelled");
    assert!(err.contains("cancelled"), "{err}");
    assert!(job::wait_pid_exit(pid, Duration::from_secs(5)));
    assert_eq!(get(name).unwrap().state, HelperState::Stopped);
}

#[tokio::test(flavor = "multi_thread")]
async fn abandoning_the_spawn_future_still_finishes_the_start() {
    let name = "t-abandoned";
    let spec = http_spec(name, &[("SIGIL_PH_READY_MS", "800")]);
    // The caller gives up long before the helper is healthy.
    assert!(tokio::time::timeout(Duration::from_millis(100), spawn(spec))
        .await
        .is_err());
    assert!(
        wait_until(Duration::from_secs(30), || get(name).is_some_and(|i| i.state == HelperState::Running)).await,
        "{:?}",
        get(name)
    );
    let h = super::handle(name).expect("running and supervised");
    stop(name).await.unwrap();
    assert!(job::wait_pid_exit(h.pid, Duration::from_secs(5)));
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_closes_stdin_first_for_a_graceful_exit() {
    let name = "t-graceful";
    let spec = HelperSpec {
        stop_grace_ms: 10_000,
        ..fake_spec(name, "sleep", &[("SIGIL_PH_EXIT_ON_EOF", "1")])
    };
    let h = spawn(spec).await.expect("starts (no port, no health)");
    assert!(h.port.is_none() && h.base_url.is_none() && h.api_key.is_none());
    let t0 = Instant::now();
    stop(name).await.unwrap();
    assert!(t0.elapsed() < Duration::from_secs(8), "graceful exit should be quick");
    assert!(job::wait_pid_exit(h.pid, Duration::from_secs(5)));
    let lines = logs(name);
    assert!(
        lines
            .iter()
            .any(|l| l.text.contains("stdin closed, exiting cleanly")),
        "{lines:?}"
    );
    assert!(lines.iter().any(|l| l.text.contains("ended: exit code 0")), "{lines:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn protocol_helper_talks_over_stdio_and_stops_gracefully() {
    use std::io::BufRead;
    let name = "t-stdio";
    let spec = HelperSpec {
        stdio: StdioMode::Protocol,
        stop_grace_ms: 10_000,
        ..fake_spec(name, "jsonl", &[])
    };
    let h = spawn(spec).await.expect("starts");
    let io = take_stdio(name).expect("protocol pipes");
    assert_eq!(io.pid, h.pid);
    assert!(take_stdio(name).is_none(), "the pipes are handed out once");
    let HelperStdio { mut stdin, stdout, .. } = io;
    let (reply, mut stdin) = tokio::task::spawn_blocking(move || {
        stdin
            .write_all(
                b"{\"text\":\"hello\"}
",
            )
            .unwrap();
        stdin.flush().unwrap();
        let mut reader = std::io::BufReader::new(stdout);
        let mut line = String::new();
        // Skip the test harness banner; a real protocol helper prints only protocol lines.
        loop {
            line.clear();
            if reader.read_line(&mut line).unwrap() == 0 || line.starts_with("{\"echo\"") {
                return (line, stdin);
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(reply.trim(), r#"{"echo":{"text":"hello"}}"#);
    // stderr reaches the log; stdout (the protocol) does not.
    assert!(
        wait_until(Duration::from_secs(5), || logs(name)
            .iter()
            .any(|l| l.text.contains("jsonl helper ready")))
        .await
    );
    assert!(logs(name).iter().all(|l| !l.text.contains("echo")));

    stop(name).await.unwrap();
    assert!(job::wait_pid_exit(h.pid, Duration::from_secs(5)));
    let lines = logs(name);
    assert!(
        lines
            .iter()
            .any(|l| l.text.contains("jsonl helper: stdin closed")),
        "{lines:?}"
    );
    assert!(lines.iter().any(|l| l.text.contains("ended: exit code 0")), "{lines:?}");
    let late = stdin
        .write_all(
            b"late
",
        )
        .expect_err("closed by the stop");
    assert_eq!(late.kind(), std::io::ErrorKind::BrokenPipe);
    assert!(take_stdio("t-stdio-missing").is_none());
}

/// A protocol client that writes in a loop holds the stdin lock almost all the time; a stop must
/// still deliver EOF right after the in-flight write (clean exit 0), not wait out the grace period.
#[tokio::test(flavor = "multi_thread")]
async fn stop_reaches_a_protocol_helper_whose_client_keeps_writing() {
    let name = "t-busy-stdin";
    let spec = HelperSpec {
        stdio: StdioMode::Protocol,
        stop_grace_ms: 10_000,
        ..fake_spec(name, "sink", &[])
    };
    let h = spawn(spec).await.expect("starts");
    let HelperStdio { mut stdin, stdout, .. } = take_stdio(name).expect("protocol pipes");
    // Keep draining stdout (the test harness banner) so the helper never blocks on it.
    std::thread::spawn(move || {
        let mut stdout = stdout;
        let mut sink = Vec::new();
        let _ = stdout.read_to_end(&mut sink);
    });
    let writer = std::thread::spawn(move || {
        let chunk = vec![b'x'; 64 * 1024];
        let mut writes = 0u64;
        while stdin.write_all(&chunk).is_ok() {
            writes += 1;
        }
        (writes, stdin)
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    let t0 = Instant::now();
    stop(name).await.unwrap();
    assert!(
        t0.elapsed() < Duration::from_secs(5),
        "stop took {:?}: EOF never reached the helper",
        t0.elapsed()
    );
    let (writes, mut stdin) = tokio::task::spawn_blocking(move || writer.join().unwrap())
        .await
        .unwrap();
    assert!(writes > 0);
    let late = stdin.write_all(b"late").expect_err("closed by the stop");
    assert_eq!(late.kind(), std::io::ErrorKind::BrokenPipe);
    assert!(job::wait_pid_exit(h.pid, Duration::from_secs(5)));
    let lines = logs(name);
    assert!(lines.iter().any(|l| l.text.contains("sink: stdin closed")), "{lines:?}");
    assert!(lines.iter().any(|l| l.text.contains("ended: exit code 0")), "{lines:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_kills_the_whole_process_tree() {
    let name = "t-tree";
    let dir = temp_dir(name);
    let tree_file = dir.join("grandchild.pid");
    let tree_s = tree_file.display().to_string();
    let spec = fake_spec(name, "tree", &[("SIGIL_PH_TREE_FILE", tree_s.as_str())]);
    let h = spawn(spec).await.expect("starts");
    let grandchild = read_pids(&tree_file, Duration::from_secs(20))[0];
    assert!(job::pid_alive(grandchild));
    stop(name).await.unwrap();
    assert!(job::wait_pid_exit(h.pid, Duration::from_secs(5)));
    assert!(job::wait_pid_exit(grandchild, Duration::from_secs(5)), "grandchild escaped the job");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn helper_memory_cap_is_reported() {
    let name = "t-memcap";
    let spec = HelperSpec {
        memory_limit_mb: Some(1024),
        ..fake_spec(name, "alloc", &[("SIGIL_PH_ALLOC_MB", "4096")])
    };
    // Without a health check the helper may count as started before the allocation fails; either
    // way it ends up failed with the cap named as the reason.
    let _ = spawn(spec).await;
    assert!(wait_until(Duration::from_secs(20), || get(name).is_some_and(|i| i.state == HelperState::Failed)).await);
    let err = get(name).unwrap().error.unwrap();
    assert!(err.contains("ran into its 1024 MB memory limit"), "{err}");
    assert!(err.contains("allocation of 4096 MB failed"), "{err}");
}

/// Helpers start with critical-error, fault and OpenFile boxes disabled even when Sigil runs with
/// the default error mode (dialogs on), and Sigil's own mode is left as it was. Runs in a fake
/// Sigil that sets its error mode to 0; it also runs a program whose DLL is missing, which must
/// end promptly with 0xC0000135 instead of blocking on the loader's modal box.
#[test]
fn helpers_start_with_error_dialogs_disabled() {
    let dir = temp_dir("error-mode");
    let exe = exe_with_missing_dll(&dir);
    let exe_s = exe.display().to_string();
    let out = run_worker(&worker_spec("errmode-parent", &[("SIGIL_PH_MISSING_DLL_EXE", exe_s.as_str())])).unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.success(), "{}: {stdout}", out.describe());
    assert!(stdout.contains("errmode ok"), "{stdout}");
    let _ = std::fs::remove_dir_all(dir);
}

/// A helper whose program cannot be loaded (missing DLL) fails at once with the loader's reason and
/// is not restarted: the failure is deterministic.
#[tokio::test(flavor = "multi_thread")]
async fn a_helper_with_a_missing_dll_fails_at_once_and_is_not_restarted() {
    let dir = temp_dir("missing-dll");
    let exe = exe_with_missing_dll(&dir);
    let name = "t-missing-dll";
    let spec = HelperSpec {
        name: name.into(),
        program: exe,
        port: PortRequest::Auto,
        health: Some(HealthCheck {
            path: "/health".into(),
            timeout_ms: 30_000,
            interval_ms: 50,
            send_api_key: false,
        }),
        restart: RestartPolicy::OnFailure {
            max_restarts: 3,
            backoff_ms: 1000,
        },
        ..Default::default()
    };
    let t0 = Instant::now();
    let err = spawn(spec).await.expect_err("cannot start");
    assert!(
        err.contains("could not start (exit code 0xC0000135: a required DLL was not found)"),
        "{err}"
    );
    assert!(t0.elapsed() < Duration::from_secs(10), "{:?}", t0.elapsed());
    let info = get(name).unwrap();
    assert_eq!(info.state, HelperState::Failed);
    assert_eq!(info.restarts, 0);
    let _ = std::fs::remove_dir_all(dir);
}

/// The acceptance test: kill "Sigil" (a fake parent process that started a helper through the
/// process host) with TerminateProcess, as Task Manager does. The helper and its own child die.
#[test]
fn helpers_die_when_sigil_is_killed() {
    let dir = temp_dir("parent-killed");
    let pid_file = dir.join("pids.txt");
    let mut parent = spawn_fake_directly("parent", &[("SIGIL_PH_PID_FILE", &pid_file)]);
    let pids = read_pids(&pid_file, Duration::from_secs(60));
    assert_eq!(pids.len(), 2, "{pids:?}");
    let (helper, grandchild) = (pids[0], pids[1]);
    assert!(job::pid_alive(helper) && job::pid_alive(grandchild));
    parent.0.kill().expect("TerminateProcess the fake Sigil");
    let _ = parent.0.wait();
    assert!(job::wait_pid_exit(helper, Duration::from_secs(10)), "helper orphaned");
    assert!(job::wait_pid_exit(grandchild, Duration::from_secs(10)), "grandchild orphaned");
    let _ = std::fs::remove_dir_all(dir);
}

/// `shutdown_all` (RunEvent::Exit) kills every helper before Sigil exits; runs in a child process
/// because it refuses later spawns process-wide.
#[test]
fn shutdown_all_kills_every_helper() {
    let dir = temp_dir("shutdown-all");
    let pid_file = dir.join("pids.txt");
    let mut parent = spawn_fake_directly("shutdown", &[("SIGIL_PH_PID_FILE", &pid_file)]);
    let deadline = Instant::now() + Duration::from_secs(60);
    let status = loop {
        if let Some(s) = parent.0.try_wait().unwrap() {
            break s;
        }
        assert!(Instant::now() < deadline, "fake Sigil did not finish");
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(status.code(), Some(0), "helpers survived shutdown_all (see fake role 'shutdown')");
    let _ = std::fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------------------------------------
// Isolated workers
// ---------------------------------------------------------------------------------------------

fn worker_spec(role: &str, env: &[(&str, &str)]) -> WorkerSpec {
    let mut e = vec![(ROLE_ENV.to_string(), role.to_string())];
    e.extend(env.iter().map(|(k, v)| (k.to_string(), v.to_string())));
    WorkerSpec {
        name: format!("w-{role}"),
        program: test_exe(),
        args: fake_args(),
        env: e,
        timeout_ms: 60_000,
        ..Default::default()
    }
}

#[test]
fn worker_captures_output_and_exit_code() {
    let spec = WorkerSpec {
        stdin: Some(b"ping from the host".to_vec()),
        ..worker_spec("echo", &[])
    };
    let out = run_worker(&spec).unwrap();
    assert!(out.success(), "{}", out.describe());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("echo: ping from the host"), "{stdout}");
    assert!(!out.timed_out && !out.memory_limit_hit);

    let out = run_worker(&worker_spec("exit", &[("SIGIL_PH_EXIT_CODE", "9")])).unwrap();
    assert_eq!(out.exit_code, Some(9));
    assert!(!out.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("model file not found"));
    assert_eq!(out.describe(), "exit code 9");
}

#[test]
fn worker_time_limit_kills_it() {
    let spec = WorkerSpec {
        timeout_ms: 800,
        ..worker_spec("sleep", &[])
    };
    let t0 = Instant::now();
    let out = run_worker(&spec).unwrap();
    assert!(out.timed_out);
    assert_eq!(out.exit_code, None);
    assert!(t0.elapsed() < Duration::from_secs(10));
    assert!(out.describe().starts_with("timed out after"));
}

#[test]
fn worker_memory_cap_holds() {
    let spec = WorkerSpec {
        memory_limit_mb: Some(1024),
        ..worker_spec("alloc", &[("SIGIL_PH_ALLOC_MB", "4096")])
    };
    let out = run_worker(&spec).unwrap();
    assert_eq!(out.exit_code, Some(5), "{}", out.describe());
    assert!(out.memory_limit_hit, "the job must report the cap");
    // (PeakJobMemoryUsed includes the refused 4 GB request, so it is not compared with the cap.)
    assert!(String::from_utf8_lossy(&out.stderr).contains("allocation of 4096 MB failed"));

    let spec = WorkerSpec {
        memory_limit_mb: None,
        ..worker_spec("alloc", &[("SIGIL_PH_ALLOC_MB", "64")])
    };
    let out = run_worker(&spec).unwrap();
    assert!(out.success(), "{}", out.describe());
    assert!(!out.memory_limit_hit);
    assert!(out.peak_memory_mb >= 64, "peak {} MB", out.peak_memory_mb);
}

#[test]
fn worker_output_is_bounded() {
    let spec = WorkerSpec {
        max_output_bytes: 1000,
        ..worker_spec("flood", &[])
    };
    let out = run_worker(&spec).unwrap();
    assert!(out.success(), "{}", out.describe());
    assert_eq!(out.stdout.len(), 1000);
    assert!(out.stdout_truncated);
}

#[test]
fn worker_rejects_bad_specs() {
    assert!(run_worker(&WorkerSpec {
        name: "Bad Name".into(),
        ..worker_spec("echo", &[])
    })
    .is_err());
    assert!(run_worker(&WorkerSpec {
        timeout_ms: 0,
        ..worker_spec("echo", &[])
    })
    .is_err());
    let err = run_worker(&WorkerSpec {
        program: PathBuf::from("C:/definitely/not/here/w.exe"),
        ..worker_spec("echo", &[])
    })
    .unwrap_err();
    assert!(err.contains("program not found"), "{err}");
}

// ---------------------------------------------------------------------------------------------
// Manual / opt-in
// ---------------------------------------------------------------------------------------------

/// Manual: a helper that listens on every interface is stopped by the exposure audit. Ignored, and
/// opt-in even under `--include-ignored`, because listening beyond loopback makes Windows Firewall
/// prompt for the test binary.
/// Run: `SIGIL_TEST_EXPOSURE=1 cargo test --manifest-path src-tauri/Cargo.toml non_loopback -- --ignored`
#[tokio::test(flavor = "multi_thread")]
#[ignore = "listens on 0.0.0.0 (Windows Firewall prompt); run manually with SIGIL_TEST_EXPOSURE=1"]
async fn non_loopback_listener_is_rejected() {
    if std::env::var("SIGIL_TEST_EXPOSURE").as_deref() != Ok("1") {
        eprintln!("SIGIL_TEST_EXPOSURE=1 not set; skipping (would trigger a Windows Firewall prompt)");
        return;
    }
    if ports::non_loopback_addresses().is_empty() {
        eprintln!("no non-loopback address on this PC; nothing to audit");
        return;
    }
    let name = "t-exposed";
    let err = spawn(http_spec(name, &[("SIGIL_PH_BIND", "any")]))
        .await
        .expect_err("exposed");
    assert!(err.contains("helpers must bind 127.0.0.1 only"), "{err}");
    assert_eq!(get(name).unwrap().state, HelperState::Failed);
}

/// Manual: host overhead of a start (spawn → job → resume → first 2xx `/health`) and a stop (kill →
/// reaped), measured with the fake HTTP helper. Numbers for ADR 0005.
/// Run: `cargo test --manifest-path src-tauri/Cargo.toml measure_start_stop -- --ignored --nocapture`
#[tokio::test(flavor = "multi_thread")]
#[ignore = "timing report for ADR 0005; run manually"]
async fn measure_start_stop_latency() {
    let name = "t-measure";
    let (mut launch, mut boot, mut detect, mut total, mut stops) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..10 {
        let t0_ms = crate::services::diagnostics::logging::unix_ms_now();
        let t0 = Instant::now();
        let h = spawn(http_spec(name, &[])).await.expect("starts");
        let ready_ms = crate::services::diagnostics::logging::unix_ms_now();
        total.push(t0.elapsed().as_secs_f64() * 1000.0);
        let lines = logs(name);
        let at = |needle: &str| {
            lines
                .iter()
                .rev()
                .find(|l| l.text.contains(needle))
                .map(|l| l.at_unix_ms)
                .unwrap()
        };
        let started = at("started pid");
        let listening = at("fake http helper starting");
        launch.push(started.saturating_sub(t0_ms) as f64);
        boot.push(listening.saturating_sub(started) as f64);
        detect.push(ready_ms.saturating_sub(listening) as f64);
        let t1 = Instant::now();
        stop(name).await.unwrap();
        assert!(job::wait_pid_exit(h.pid, Duration::from_secs(5)));
        stops.push(t1.elapsed().as_secs_f64() * 1000.0);
    }
    let stat = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        format!("p50 {:.1} ms, max {:.1} ms", v[v.len() / 2], v[v.len() - 1])
    };
    eprintln!("spawn -> process launched in its job: {}", stat(&mut launch));
    eprintln!(
        "launched -> fake helper printed its first line (the helper's own start-up): {}",
        stat(&mut boot)
    );
    eprintln!(
        "first line -> host saw /health 2xx and finished the exposure audit: {}",
        stat(&mut detect)
    );
    eprintln!("spawn -> ready (total): {}", stat(&mut total));
    eprintln!("stop -> reaped: {}", stat(&mut stops));
}

/// Manual smoke test of the real llama-server (P1-E / ADR 0005). Set `SIGIL_TEST_LLAMA_SERVER` to
/// an installed pack's `llama-server.exe` and `SIGIL_TEST_LLAMA_GGUF` to a GGUF (Hy-MT2 1.8B Q4),
/// then run `cargo test --manifest-path src-tauri/Cargo.toml llama_server_smoke -- --ignored --nocapture`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a llama.cpp runtime pack and a GGUF (SIGIL_TEST_LLAMA_SERVER, SIGIL_TEST_LLAMA_GGUF)"]
async fn llama_server_smoke() {
    let (Ok(exe), Ok(model)) = (std::env::var("SIGIL_TEST_LLAMA_SERVER"), std::env::var("SIGIL_TEST_LLAMA_GGUF")) else {
        eprintln!("SIGIL_TEST_LLAMA_SERVER / SIGIL_TEST_LLAMA_GGUF not set; skipping");
        return;
    };
    let gpu_layers = std::env::var("SIGIL_TEST_LLAMA_NGL")
        .ok()
        .and_then(|v| v.parse().ok());
    let spec = recipes::llama_server(
        Path::new(&exe),
        Path::new(&model),
        &recipes::LlamaServerOptions {
            name: "t-llama-smoke".into(),
            gpu_layers,
            ..Default::default()
        },
    );
    let t0 = Instant::now();
    let h = spawn(spec)
        .await
        .expect("llama-server starts and reports /health 200");
    let load_ms = t0.elapsed().as_millis();
    let base = h.base_url.clone().unwrap();
    let key = h.api_key.clone().unwrap();
    assert!(ports::exposure(h.port.unwrap()).await.is_empty(), "llama-server must bind 127.0.0.1 only");

    let body = serde_json::json!({
        "messages": [{"role": "user", "content":
            "Translate the following segment into Japanese, without additional explanation.\n\nGood morning, everyone! The stream starts in five minutes."}],
        "max_tokens": 128,
        "temperature": 0.2,
    });
    // /health and /v1/models are public in llama-server; inference routes need the key.
    let unauth = http_client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&body)
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .expect("unauthenticated request");
    assert_eq!(unauth.status().as_u16(), 401, "the API key must be required");

    let t1 = Instant::now();
    let resp = http_client()
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(&key)
        .json(&body)
        .timeout(Duration::from_secs(120))
        .send()
        .await
        .expect("chat request");
    assert!(resp.status().is_success(), "HTTP {}", resp.status());
    let v: serde_json::Value = resp.json().await.unwrap();
    let text = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    let first_ms = t1.elapsed().as_millis();
    eprintln!("llama-server smoke: load {load_ms} ms, first translation {first_ms} ms: {text}");
    eprintln!("timings: {}", v["timings"]);
    assert!(!text.trim().is_empty());
    assert!(logs("t-llama-smoke").iter().all(|l| !l.text.contains(&key)));

    stop("t-llama-smoke").await.unwrap();
    assert!(job::wait_pid_exit(h.pid, Duration::from_secs(5)));
}

// ---------------------------------------------------------------------------------------------
// The fake helper (runs in a child process)
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "child-process entry point of the process_host tests; does nothing without SIGIL_PH_FAKE"]
fn fake_helper_entry() {
    let Ok(role) = std::env::var(ROLE_ENV) else {
        return;
    };
    fake::run(&role);
}

mod fake {
    use super::*;
    use std::io::BufRead;

    fn env_str(k: &str) -> String {
        std::env::var(k).unwrap_or_default()
    }

    fn env_u64(k: &str, default: u64) -> u64 {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    fn write_atomic(path: &str, text: &str) {
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, text).unwrap();
        std::fs::rename(&tmp, path).unwrap();
    }

    pub fn run(role: &str) -> ! {
        // Safety net: a fake never outlives a broken test by more than two minutes.
        std::thread::spawn(|| {
            std::thread::sleep(Duration::from_secs(120));
            std::process::exit(99);
        });
        match role {
            "http" => http(),
            "exit" => {
                eprintln!("error: failed to load model: model file not found");
                std::process::exit(env_u64("SIGIL_PH_EXIT_CODE", 1) as i32)
            }
            "sleep" => sleep(),
            "echo" => {
                let mut input = String::new();
                let _ = std::io::stdin().read_to_string(&mut input);
                println!("echo: {input}");
                std::process::exit(0)
            }
            "sink" => {
                eprintln!("sink ready");
                let mut buf = vec![0u8; 64 * 1024];
                let mut stdin = std::io::stdin().lock();
                while matches!(stdin.read(&mut buf), Ok(n) if n > 0) {}
                eprintln!("sink: stdin closed");
                std::process::exit(0)
            }
            "errmode" => {
                println!("error mode {:x}", spawn::process_error_mode());
                std::process::exit(0)
            }
            "errmode-parent" => errmode_parent(),
            "jsonl" => {
                eprintln!("jsonl helper ready");
                let mut out = std::io::stdout();
                for line in std::io::stdin().lock().lines() {
                    let Ok(line) = line else { break };
                    let _ = writeln!(out, "{{\"echo\":{}}}", line.trim());
                    let _ = out.flush();
                }
                eprintln!("jsonl helper: stdin closed");
                std::process::exit(0)
            }
            "flood" => {
                let mut out = std::io::stdout().lock();
                for i in 0..20_000 {
                    let _ = writeln!(out, "flood line {i}");
                }
                std::process::exit(0)
            }
            "alloc" => {
                let mb = env_u64("SIGIL_PH_ALLOC_MB", 64);
                let mut v: Vec<u8> = Vec::new();
                if v.try_reserve_exact((mb as usize) << 20).is_err() {
                    eprintln!("allocation of {mb} MB failed");
                    std::process::exit(5);
                }
                v.resize((mb as usize) << 20, 1);
                println!("allocated {mb} MB");
                std::process::exit(0)
            }
            "tree" => {
                // Never waited on: the grandchild must die with the job, not be reaped by its parent.
                #[allow(clippy::zombie_processes)]
                let child = Command::new(test_exe())
                    .args(fake_args())
                    .env(ROLE_ENV, "sleep")
                    .env_remove("SIGIL_PH_EXIT_ON_EOF")
                    .stdin(Stdio::null())
                    .spawn()
                    .expect("grandchild");
                write_atomic(&env_str("SIGIL_PH_TREE_FILE"), &child.id().to_string());
                sleep()
            }
            "parent" | "shutdown" => parent(role == "shutdown"),
            _ => std::process::exit(98),
        }
    }

    fn sleep() -> ! {
        if env_str("SIGIL_PH_EXIT_ON_EOF") == "1" {
            let mut sink = Vec::new();
            let _ = std::io::stdin().read_to_end(&mut sink);
            println!("stdin closed, exiting cleanly");
            std::process::exit(0);
        }
        loop {
            std::thread::sleep(Duration::from_secs(3600));
        }
    }

    fn http() -> ! {
        let port: u16 = env_str("SIGIL_PH_PORT").parse().expect("SIGIL_PH_PORT");
        let ready_after = Duration::from_millis(env_u64("SIGIL_PH_READY_MS", 0));
        let key = std::env::var("SIGIL_PH_KEY").ok().filter(|k| !k.is_empty());
        println!("fake http helper starting on port {port}, key {}", key.as_deref().unwrap_or("-"));
        let ip = if env_str("SIGIL_PH_BIND") == "any" {
            Ipv4Addr::UNSPECIFIED
        } else {
            Ipv4Addr::LOCALHOST
        };
        let listener = match TcpListener::bind((ip, port)) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("error: couldn't bind HTTP server socket: {e}");
                if env_str("SIGIL_PH_LINGER_ON_BIND_ERROR") == "1" {
                    loop {
                        std::thread::sleep(Duration::from_secs(3600));
                    }
                }
                std::process::exit(4);
            }
        };
        let die_ms = env_u64("SIGIL_PH_DIE_MS", 0);
        let die_once = env_str("SIGIL_PH_DIE_ONCE");
        let should_die = die_ms > 0 && (die_once.is_empty() || !Path::new(&die_once).exists());
        if should_die {
            if !die_once.is_empty() {
                let _ = std::fs::write(&die_once, b"1");
            }
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(die_ms));
                eprintln!("simulated crash");
                std::process::exit(3);
            });
        }
        let t0 = Instant::now();
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
            let mut head = Vec::new();
            let mut buf = [0u8; 1024];
            while !head.windows(4).any(|w| w == b"\r\n\r\n") && head.len() < 16 * 1024 {
                match s.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => head.extend_from_slice(&buf[..n]),
                }
            }
            let req = String::from_utf8_lossy(&head).to_string();
            let path = req.split_whitespace().nth(1).unwrap_or("/").to_string();
            let authed = key
                .as_ref()
                .is_some_and(|k| req.contains(&format!("Bearer {k}\r\n")));
            let (status, body) = match path.as_str() {
                "/health" if t0.elapsed() < ready_after => (503, r#"{"error":"Loading model"}"#.to_string()),
                "/health" => (200, r#"{"status":"ok"}"#.to_string()),
                "/pid" => (200, std::process::id().to_string()),
                "/auth" if authed => (200, "ok".to_string()),
                "/auth" => (401, "unauthorized".to_string()),
                _ => (404, "not found".to_string()),
            };
            let _ = write!(
                s,
                "HTTP/1.1 {status} X\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
        std::process::exit(0)
    }

    /// A fake Sigil with the default error mode (critical-error boxes on): its workers must still
    /// start with [`spawn::HELPER_ERROR_MODE`], its own mode must stay 0, and a program with a
    /// missing DLL must end with 0xC0000135 (not block on a modal box).
    fn errmode_parent() -> ! {
        spawn::set_process_error_mode(0);
        let out = run_worker(&worker_spec("errmode", &[])).expect("errmode worker");
        let child = String::from_utf8_lossy(&out.stdout)
            .lines()
            .find_map(|l| l.strip_prefix("error mode "))
            .and_then(|v| u32::from_str_radix(v.trim(), 16).ok());
        let mine = spawn::process_error_mode();
        println!("parent mode after launch {mine:#x}, child mode {child:?}");
        let inherited = child.is_some_and(|m| m & spawn::HELPER_ERROR_MODE == spawn::HELPER_ERROR_MODE);
        let restored = mine == 0;
        let mut loader_ok = true;
        let exe = env_str("SIGIL_PH_MISSING_DLL_EXE");
        if inherited && !exe.is_empty() {
            let out = run_worker(&WorkerSpec {
                name: "w-missing-dll".into(),
                program: PathBuf::from(&exe),
                timeout_ms: 20_000,
                ..Default::default()
            })
            .expect("missing-dll worker");
            println!("missing-dll program: {} after {} ms", out.describe(), out.elapsed_ms);
            loader_ok = !out.timed_out && out.exit_code == Some(0xC000_0135u32 as i32) && out.elapsed_ms < 10_000;
        }
        if inherited && restored && loader_ok {
            println!("errmode ok");
            std::process::exit(0);
        }
        std::process::exit(2)
    }

    /// A fake Sigil: starts a helper (which starts a grandchild) through the real process host.
    /// `shutdown`: then calls `shutdown_all` and exits 0 only if both died.
    fn parent(shutdown: bool) -> ! {
        let pid_file = env_str("SIGIL_PH_PID_FILE");
        let tree_file = format!("{pid_file}.tree");
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let spec = fake_spec("fake-sigil-child", "tree", &[("SIGIL_PH_TREE_FILE", tree_file.as_str())]);
        let h = rt
            .block_on(super::super::spawn(spec))
            .expect("helper starts");
        let grandchild = read_pids(Path::new(&tree_file), Duration::from_secs(30))[0];
        if shutdown {
            super::super::shutdown_all();
            let dead = job::wait_pid_exit(h.pid, Duration::from_secs(2)) && job::wait_pid_exit(grandchild, Duration::from_secs(2));
            let refused = rt
                .block_on(super::super::spawn(fake_spec("after-shutdown", "sleep", &[])))
                .is_err();
            std::process::exit(if dead && refused { 0 } else { 2 });
        }
        write_atomic(&pid_file, &format!("{} {grandchild}", h.pid));
        loop {
            std::thread::sleep(Duration::from_secs(3600));
        }
    }
}
