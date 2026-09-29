//! Readiness: poll `http://127.0.0.1:{port}{path}` until it answers 2xx, the helper exits, the start
//! is cancelled, or the timeout passes.
//!
//! Until the port accepts connections, each round is a bare TCP connect bounded by the poll
//! interval: on Windows a connect to a closed port is not refused at once (the stack re-sends the
//! SYN after every RST for about two seconds), so HTTP probes would notice the listener hundreds of
//! milliseconds late. Once it listens, HTTP probes follow immediately.
//!
//! Before the first HTTP probe (and so before any bearer key is sent), the OS listener table must
//! show that the socket answering on `127.0.0.1:port` belongs to a process in the helper's job. A
//! foreign owner (another program took the port in the pick-to-bind window) ends the wait with
//! [`Readiness::Foreign`], which the host treats as a port collision.
//!
//! The HTTP client never uses a proxy (a system or `HTTP_PROXY` proxy would otherwise see loopback
//! traffic and the key), never follows redirects and keeps no idle connections (it is shared by
//! callers on different tokio runtimes).

use super::{ports, HealthCheck};
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokio::sync::watch;

/// The exit of a helper process as seen by the waiter thread.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExitInfo {
    /// Process exit code (`None` if it could not be read).
    pub code: Option<i32>,
    /// Sigil stopped it (stop, replace, shutdown, failed health check).
    pub killed_by_host: bool,
    /// The job reported a memory-limit hit.
    pub memory_limit_hit: bool,
    /// The job's memory cap, when one was set.
    pub memory_limit_mb: Option<u64>,
    /// Peak committed memory of the job.
    pub peak_memory_mb: u64,
}

impl std::fmt::Display for ExitInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.killed_by_host {
            f.write_str("stopped by Sigil")?;
        } else {
            match self.code {
                Some(c) => f.write_str(&super::job::describe_exit_code(c))?,
                None => f.write_str("exit code unknown")?,
            }
        }
        if self.memory_limit_hit {
            match self.memory_limit_mb {
                Some(mb) => write!(f, "; it ran into its {mb} MB memory limit")?,
                None => f.write_str("; it ran into its memory limit")?,
            }
        }
        Ok(())
    }
}

/// How waiting for readiness ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Readiness {
    Ready,
    /// No 2xx within the timeout; `last` describes the last attempt.
    TimedOut {
        last: String,
    },
    Exited(ExitInfo),
    Cancelled,
    /// A process outside the helper's job owns the listener on the port (`pid`). Nothing but TCP
    /// connects was sent to it.
    Foreign {
        pid: u32,
    },
}

/// Who owns the listener that receives connections to `127.0.0.1:port`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Owner {
    Helper,
    Foreign(u32),
    /// The table shows no listener (it closed again, or the table lags a connect).
    Nobody,
    /// The table could not be read: proceed unverified.
    Unknown,
}

fn port_owner(port: u16, helper_pids: &[u32]) -> Owner {
    match ports::loopback_listener_pid(port) {
        Ok(Some(pid)) if helper_pids.contains(&pid) => Owner::Helper,
        Ok(Some(pid)) => Owner::Foreign(pid),
        Ok(None) => Owner::Nobody,
        Err(e) => {
            tracing::debug!(target: super::logs::LOG_TARGET, "listener table unavailable ({e}); port {port} owner not verified");
            Owner::Unknown
        }
    }
}

/// Consecutive "accepts connections, but the listener table shows nobody" rounds after which the
/// owner check gives up and proceeds unverified (a table that never shows the socket must not
/// make a working helper time out).
const MAX_OWNERLESS_ROUNDS: u32 = 3;

/// Connect deadline of an HTTP probe (only made once the port has accepted a TCP connection).
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(1);
/// Bounds of one TCP "is it listening yet" probe (the poll interval, clamped to these).
const TCP_PROBE_MIN: Duration = Duration::from_millis(20);
const TCP_PROBE_MAX: Duration = Duration::from_millis(250);

fn client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .pool_max_idle_per_host(0)
            .connect_timeout(CONNECT_TIMEOUT)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    })
}

/// Per-attempt deadline: long enough for a busy helper, short enough to notice exits and keep the
/// cadence (a connect to a closed loopback port can take ~2 s on Windows because of SYN retries).
pub fn attempt_timeout(check: &HealthCheck) -> Duration {
    Duration::from_millis(check.interval_ms.saturating_mul(8)).clamp(Duration::from_secs(1), Duration::from_secs(5))
}

/// Whether `127.0.0.1:port` accepts a TCP connection within `timeout` (the kernel completes a
/// loopback handshake in microseconds, however busy the helper is).
async fn tcp_listening(port: u16, timeout: Duration) -> bool {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    matches!(tokio::time::timeout(timeout, tokio::net::TcpStream::connect(addr)).await, Ok(Ok(_)))
}

/// One health request: `Ok(status)` when the helper answered at all.
async fn probe(url: &str, bearer: Option<&str>, timeout: Duration) -> Result<u16, String> {
    let mut req = client().get(url).timeout(timeout);
    if let Some(key) = bearer {
        req = req.bearer_auth(key);
    }
    match req.send().await {
        Ok(resp) => Ok(resp.status().as_u16()),
        Err(e) if e.is_timeout() => Err("no answer (timed out)".into()),
        Err(e) if e.is_connect() => Err("connection refused (not listening yet)".into()),
        Err(e) => Err(format!("request failed: {e}")),
    }
}

/// Resolves when `cancel` becomes true (never if its sender is gone).
pub async fn cancelled(cancel: &mut watch::Receiver<bool>) {
    if cancel.wait_for(|c| *c).await.is_err() {
        std::future::pending::<()>().await;
    }
}

/// Resolves with the exit of the process behind `exit`.
pub async fn exited(exit: &mut watch::Receiver<Option<ExitInfo>>) -> ExitInfo {
    match exit.wait_for(Option::is_some).await {
        Ok(v) => (*v).clone().unwrap_or_default(),
        Err(_) => ExitInfo::default(),
    }
}

/// Poll until ready. `helper_pids` lists the processes of the helper's job (the listener on the
/// port must belong to one of them). Returns the outcome and whether the helper ever answered HTTP
/// at all (any status), which rules out a port collision when it later exits.
pub async fn wait_ready(
    port: u16,
    check: &HealthCheck,
    bearer: Option<&str>,
    helper_pids: &(dyn Fn() -> Vec<u32> + Sync),
    exit: &mut watch::Receiver<Option<ExitInfo>>,
    cancel: &mut watch::Receiver<bool>,
) -> (Readiness, bool) {
    let url = format!("http://127.0.0.1:{port}{}", check.path);
    let deadline = Instant::now() + Duration::from_millis(check.timeout_ms);
    let interval = Duration::from_millis(check.interval_ms.clamp(20, 10_000));
    let per_attempt = attempt_timeout(check);
    let mut answered = false;
    let mut listening = false;
    let mut ownerless_rounds = 0u32;
    loop {
        let round = Instant::now();
        let left = deadline.saturating_duration_since(round);
        let result = if listening {
            let attempt = probe(&url, bearer, per_attempt.min(left.max(Duration::from_millis(50))));
            tokio::select! {
                biased;
                _ = cancelled(cancel) => return (Readiness::Cancelled, answered),
                info = exited(exit) => return (Readiness::Exited(info), answered),
                r = attempt => r,
            }
        } else {
            let wait = interval
                .clamp(TCP_PROBE_MIN, TCP_PROBE_MAX)
                .min(left.max(Duration::from_millis(10)));
            let accepted = tokio::select! {
                biased;
                _ = cancelled(cancel) => return (Readiness::Cancelled, answered),
                info = exited(exit) => return (Readiness::Exited(info), answered),
                ok = tcp_listening(port, wait) => ok,
            };
            if accepted {
                match port_owner(port, &helper_pids()) {
                    Owner::Foreign(pid) => return (Readiness::Foreign { pid }, answered),
                    Owner::Nobody if ownerless_rounds < MAX_OWNERLESS_ROUNDS => {
                        ownerless_rounds += 1;
                        Err("not listening yet (no listener in the TCP table)".to_string())
                    }
                    Owner::Helper | Owner::Nobody | Owner::Unknown => {
                        listening = true;
                        continue; // HTTP probe right away
                    }
                }
            } else {
                Err("not listening yet".to_string())
            }
        };
        let last = match result {
            Ok(status) if (200..300).contains(&status) => return (Readiness::Ready, true),
            Ok(status) => {
                answered = true;
                format!("HTTP {status}")
            }
            Err(e) => e,
        };
        if Instant::now() >= deadline {
            return (Readiness::TimedOut { last }, answered);
        }
        let pause = interval
            .saturating_sub(round.elapsed())
            .min(deadline.saturating_duration_since(Instant::now()));
        tokio::select! {
            biased;
            _ = cancelled(cancel) => return (Readiness::Cancelled, answered),
            info = exited(exit) => return (Readiness::Exited(info), answered),
            _ = tokio::time::sleep(pause) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The test process plays the helper (its own listeners count as the helper's).
    fn own_pid() -> Vec<u32> {
        vec![std::process::id()]
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_foreign_listener_is_reported_and_never_sent_a_request() {
        // Another program (here: this test process, which is not in the "helper's" pid list)
        // answers on the port the helper was given.
        let listener = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        // A real squatter keeps listening after the first connection.
        let keep_listening = listener.try_clone().unwrap();
        let squatter = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            s.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            let mut got = Vec::new();
            let _ = std::io::Read::read_to_end(&mut s, &mut got);
            got
        });
        let check = HealthCheck {
            timeout_ms: 10_000,
            interval_ms: 50,
            send_api_key: true,
            ..Default::default()
        };
        let (_exit_tx, mut exit_rx) = watch::channel(None);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        let not_us = || vec![std::process::id().wrapping_add(4)];
        let (r, answered) = wait_ready(port, &check, Some("secret-key"), &not_us, &mut exit_rx, &mut cancel_rx).await;
        assert_eq!(r, Readiness::Foreign { pid: std::process::id() });
        assert!(!answered);
        let received = squatter.join().unwrap();
        assert!(received.is_empty(), "the squatter got {:?}", String::from_utf8_lossy(&received));
        drop(keep_listening);
    }

    #[test]
    fn exit_info_messages() {
        let mut e = ExitInfo {
            code: Some(1),
            ..Default::default()
        };
        assert_eq!(e.to_string(), "exit code 1");
        e.memory_limit_hit = true;
        e.memory_limit_mb = Some(512);
        assert_eq!(e.to_string(), "exit code 1; it ran into its 512 MB memory limit");
        e.killed_by_host = true;
        e.memory_limit_hit = false;
        assert_eq!(e.to_string(), "stopped by Sigil");
    }

    #[test]
    fn attempt_timeout_is_clamped() {
        let mut c = HealthCheck::default();
        assert_eq!(attempt_timeout(&c), Duration::from_secs(2));
        c.interval_ms = 10;
        assert_eq!(attempt_timeout(&c), Duration::from_secs(1));
        c.interval_ms = 60_000;
        assert_eq!(attempt_timeout(&c), Duration::from_secs(5));
    }

    #[tokio::test]
    async fn times_out_against_a_closed_port_and_honours_cancel() {
        let port = super::super::ports::pick_free_port().unwrap();
        let check = HealthCheck {
            path: "/health".into(),
            timeout_ms: 600,
            interval_ms: 50,
            send_api_key: false,
        };
        let (_exit_tx, mut exit_rx) = watch::channel(None);
        let (cancel_tx, mut cancel_rx) = watch::channel(false);
        let t0 = Instant::now();
        let (r, answered) = wait_ready(port, &check, None, &own_pid, &mut exit_rx, &mut cancel_rx).await;
        assert_eq!(
            r,
            Readiness::TimedOut {
                last: "not listening yet".into()
            }
        );
        assert!(!answered);
        assert!(t0.elapsed() < Duration::from_secs(3), "{:?}", t0.elapsed());

        cancel_tx.send_replace(true);
        let (r, _) = wait_ready(port, &check, None, &own_pid, &mut exit_rx, &mut cancel_rx).await;
        assert_eq!(r, Readiness::Cancelled);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn notices_a_late_listener_within_an_interval() {
        let port = super::super::ports::pick_free_port().unwrap();
        let server = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            let listener = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, port)).unwrap();
            let bound = Instant::now();
            let (probe, _) = listener.accept().unwrap(); // the TCP probe
            drop(probe);
            let (mut s, _) = listener.accept().unwrap(); // the HTTP probe
            let mut buf = [0u8; 1024];
            let _ = std::io::Read::read(&mut s, &mut buf);
            let _ = std::io::Write::write_all(&mut s, b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            bound
        });
        let check = HealthCheck {
            timeout_ms: 10_000,
            interval_ms: 50,
            ..Default::default()
        };
        let (_exit_tx, mut exit_rx) = watch::channel(None);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        let (r, answered) = wait_ready(port, &check, None, &own_pid, &mut exit_rx, &mut cancel_rx).await;
        let ready = Instant::now();
        let bound = server.join().unwrap();
        assert_eq!(r, Readiness::Ready);
        assert!(answered);
        let lag = ready.saturating_duration_since(bound);
        assert!(lag < Duration::from_millis(200), "readiness noticed {lag:?} after the bind");
    }

    #[tokio::test]
    async fn reports_exit_while_waiting() {
        let port = super::super::ports::pick_free_port().unwrap();
        let check = HealthCheck {
            timeout_ms: 10_000,
            ..Default::default()
        };
        let (exit_tx, mut exit_rx) = watch::channel(None);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        exit_tx.send_replace(Some(ExitInfo {
            code: Some(7),
            ..Default::default()
        }));
        let (r, _) = wait_ready(port, &check, None, &own_pid, &mut exit_rx, &mut cancel_rx).await;
        assert_eq!(
            r,
            Readiness::Exited(ExitInfo {
                code: Some(7),
                ..Default::default()
            })
        );
    }
}
