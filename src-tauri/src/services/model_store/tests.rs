//! Integration tests of the model store against a local warp fixture server (loopback only; the
//! network is never touched). Each test gets its own server, app data folder and store.
//!
//! Covered: install + resume after an aborted transfer and after a stall; cancel keeps the partial
//! and the next install resumes from it; failover on 404 / 429 / HTML / wrong size / hash
//! mismatch; a corrupt or partial file is never installed; one download at a time (and one per
//! host with two slots); redirects keep the Range; archives extract, and zip-slip / symlinks / the
//! ratio cap are refused end to end; legacy files are adopted without any request; requires;
//! verify, remove, import, local assets, unpublished entries, trust-on-first-use rules; the Vosk
//! unpinned fetch; disk usage and housekeeping.

use super::catalog::{self, ValidateOpts};
use super::download::meta_path;
use super::events::RecordingSink;
use super::extract::ExtractLimits;
use super::install;
use super::layout::{read_record, Origin};
use super::legacy;
use super::queue::CANCELLED;
use super::store::{partial_key, Store, StoreSettings};
use super::verify::{git_blob_id, sha256_hex};
use super::{Catalog, DownloadError, EntryState, MirrorPreference, StoreConfig};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use warp::http::Response;
use warp::hyper::body::Bytes;
use warp::hyper::Body;
use warp::Filter;

// ---------------------------------------------------------------------------------------------
// Fixture server
// ---------------------------------------------------------------------------------------------

#[derive(Clone)]
enum Behavior {
    /// Honours Range.
    Serve(Arc<Vec<u8>>),
    /// The first `n` responses send `after` bytes of the requested range, then drop the connection.
    Abort { data: Arc<Vec<u8>>, after: usize, n: Arc<AtomicUsize> },
    /// The first `n` responses send `after` bytes, then go silent.
    Stall { data: Arc<Vec<u8>>, after: usize, n: Arc<AtomicUsize> },
    /// Chunks of `chunk` bytes every `delay`.
    Slow { data: Arc<Vec<u8>>, chunk: usize, delay: Duration },
    /// Like `Abort`, but every response is a plain 200 whatever the Range (with `n` = MAX: every
    /// attempt restarts from byte 0 and drops at `after`, so no attempt after the first progresses).
    AbortThenIgnoreRange { data: Arc<Vec<u8>>, after: usize, n: Arc<AtomicUsize> },
    Status(u16),
    /// 200 `text/html`.
    HtmlPage,
    /// 200 `application/octet-stream` whose body is an HTML page of `len` bytes.
    HtmlBody(usize),
    Redirect(String),
}

#[derive(Clone, Copy)]
enum Mode {
    Fast,
    AbortAfter(usize),
    StallAfter(usize),
    Slow(usize, Duration),
}

#[derive(Clone, Debug)]
struct Hit {
    path: String,
    range: Option<String>,
    if_range: Option<String>,
}

#[derive(Clone, Default)]
struct Counters {
    active: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
}

struct ActiveGuard(Arc<AtomicUsize>);

impl ActiveGuard {
    fn new(c: &Counters) -> Self {
        let now = c.active.fetch_add(1, Ordering::SeqCst) + 1;
        c.peak.fetch_max(now, Ordering::SeqCst);
        Self(c.active.clone())
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

struct Server {
    base: String,
    routes: Arc<Mutex<HashMap<String, Behavior>>>,
    hits: Arc<Mutex<Vec<Hit>>>,
    counters: Counters,
    _stop: tokio::sync::oneshot::Sender<()>,
}

struct BodyState {
    data: Arc<Vec<u8>>,
    pos: usize,
    sent: usize,
    mode: Mode,
    failed: bool,
    _guard: ActiveGuard,
}

fn body_stream(st: BodyState) -> Body {
    let stream = futures::stream::unfold(st, |mut st| async move {
        let end = st.data.len();
        if st.failed {
            return None;
        }
        let limit = match st.mode {
            Mode::AbortAfter(n) | Mode::StallAfter(n) => Some(n),
            _ => None,
        };
        if let Some(n) = limit {
            if st.sent >= n || st.pos >= end {
                if st.pos >= end {
                    return None;
                }
                if let Mode::StallAfter(_) = st.mode {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    return None;
                }
                // Let hyper flush what was sent (a real dropped connection keeps the bytes the
                // client already received), then fail the body so the connection is cut.
                tokio::time::sleep(Duration::from_millis(100)).await;
                st.failed = true;
                return Some((Err(io::Error::other("fixture: connection dropped")), st));
            }
        }
        if st.pos >= end {
            return None;
        }
        let chunk = match st.mode {
            Mode::Slow(chunk, delay) => {
                tokio::time::sleep(delay).await;
                chunk
            }
            _ => 64 * 1024,
        };
        let mut take = chunk.min(end - st.pos);
        if let Some(n) = limit {
            take = take.min(n - st.sent);
        }
        let bytes = Bytes::copy_from_slice(&st.data[st.pos..st.pos + take]);
        st.pos += take;
        st.sent += take;
        Some((Ok::<Bytes, io::Error>(bytes), st))
    });
    Body::wrap_stream(stream)
}

fn plain(status: u16, content_type: &str, body: impl Into<Body>) -> Response<Body> {
    Response::builder()
        .status(status)
        .header("content-type", content_type)
        .body(body.into())
        .unwrap()
}

fn ranged(data: Arc<Vec<u8>>, range: Option<&str>, mode: Mode, counters: &Counters, honour_range: bool) -> Response<Body> {
    let len = data.len();
    let start = range
        .filter(|_| honour_range)
        .and_then(|r| r.strip_prefix("bytes="))
        .and_then(|r| r.split('-').next())
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    if start > 0 && start >= len {
        return Response::builder()
            .status(416)
            .header("content-range", format!("bytes */{len}"))
            .body(Body::empty())
            .unwrap();
    }
    let mut b = Response::builder()
        .status(if start > 0 { 206 } else { 200 })
        .header("content-type", "application/octet-stream")
        .header("content-length", (len - start).to_string())
        .header("accept-ranges", "bytes")
        .header("etag", "\"fixture-etag\"");
    if start > 0 {
        b = b.header("content-range", format!("bytes {start}-{}/{len}", len - 1));
    }
    b.body(body_stream(BodyState {
        data,
        pos: start,
        sent: 0,
        mode,
        failed: false,
        _guard: ActiveGuard::new(counters),
    }))
    .unwrap()
}

fn take_one(n: &AtomicUsize) -> bool {
    n.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| v.checked_sub(1)).is_ok()
}

fn respond(b: Option<Behavior>, range: Option<String>, counters: &Counters) -> Response<Body> {
    let r = range.as_deref();
    match b {
        None => plain(404, "text/plain", "not found"),
        Some(Behavior::Serve(d)) => ranged(d, r, Mode::Fast, counters, true),
        Some(Behavior::Abort { data, after, n }) => {
            let mode = if take_one(&n) { Mode::AbortAfter(after) } else { Mode::Fast };
            ranged(data, r, mode, counters, true)
        }
        Some(Behavior::Stall { data, after, n }) => {
            let mode = if take_one(&n) { Mode::StallAfter(after) } else { Mode::Fast };
            ranged(data, r, mode, counters, true)
        }
        Some(Behavior::Slow { data, chunk, delay }) => ranged(data, r, Mode::Slow(chunk, delay), counters, true),
        Some(Behavior::AbortThenIgnoreRange { data, after, n }) => {
            let mode = if take_one(&n) { Mode::AbortAfter(after) } else { Mode::Fast };
            ranged(data, r, mode, counters, false)
        }
        Some(Behavior::Status(code)) => plain(code, "text/plain", format!("fixture status {code}")),
        Some(Behavior::HtmlPage) => plain(200, "text/html; charset=utf-8", "<!DOCTYPE html><html><body>Sign in</body></html>"),
        Some(Behavior::HtmlBody(len)) => {
            let mut page = b"<!DOCTYPE html><html><body>Access denied</body></html>".to_vec();
            page.resize(len, b' ');
            plain(200, "application/octet-stream", page)
        }
        Some(Behavior::Redirect(to)) => Response::builder()
            .status(302)
            .header("location", to)
            .body(Body::empty())
            .unwrap(),
    }
}

impl Server {
    async fn start() -> Server {
        Self::start_with(Counters::default()).await
    }

    /// A server sharing `counters` with others (cross-host concurrency).
    async fn start_with(counters: Counters) -> Server {
        let routes: Arc<Mutex<HashMap<String, Behavior>>> = Arc::default();
        let hits: Arc<Mutex<Vec<Hit>>> = Arc::default();
        let (r, h, c) = (routes.clone(), hits.clone(), counters.clone());
        let filter = warp::path::full()
            .and(warp::header::optional::<String>("range"))
            .and(warp::header::optional::<String>("if-range"))
            .map(move |path: warp::path::FullPath, range: Option<String>, if_range: Option<String>| {
                let p = path.as_str().to_string();
                h.lock().unwrap().push(Hit {
                    path: p.clone(),
                    range: range.clone(),
                    if_range,
                });
                let b = r.lock().unwrap().get(&p).cloned();
                respond(b, range, &c)
            });
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let (addr, fut) = warp::serve(filter).bind_with_graceful_shutdown(([127, 0, 0, 1], 0), async {
            let _ = rx.await;
        });
        tokio::spawn(fut);
        Server {
            base: format!("http://{addr}"),
            routes,
            hits,
            counters,
            _stop: tx,
        }
    }

    fn set(&self, path: &str, b: Behavior) {
        self.routes.lock().unwrap().insert(path.to_string(), b);
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn hits(&self) -> Vec<Hit> {
        self.hits.lock().unwrap().clone()
    }

    fn hits_to(&self, path: &str) -> Vec<Hit> {
        self.hits().into_iter().filter(|h| h.path == path).collect()
    }

    fn peak(&self) -> usize {
        self.counters.peak.load(Ordering::SeqCst)
    }
}

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

/// Deterministic, incompressible-looking bytes (never starts with '<').
fn blob(n: usize, seed: u32) -> Vec<u8> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
    (0..n)
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            if i == 0 {
                b'M'
            } else {
                (x >> 7) as u8
            }
        })
        .collect()
}

fn arc(d: &[u8]) -> Arc<Vec<u8>> {
    Arc::new(d.to_vec())
}

fn sources(urls: &[String]) -> Value {
    Value::Array(urls.iter().map(|u| json!({"kind": "direct", "url": u})).collect())
}

fn sha_file(path: &str, data: &[u8], urls: &[String]) -> Value {
    json!({"path": path, "size": data.len(), "sha256": sha256_hex(data), "sources": sources(urls)})
}

fn git_file(path: &str, data: &[u8], urls: &[String]) -> Value {
    json!({"path": path, "size": data.len(), "git_oid": git_blob_id(data), "sources": sources(urls)})
}

fn unpinned_file(path: &str, size: usize, urls: &[String]) -> Value {
    json!({"path": path, "size": size, "sources": sources(urls)})
}

fn entry(id: &str, family: &str, files: Vec<Value>) -> Value {
    json!({"id": id, "family": family, "kind": "model", "version": "1", "license": "MIT", "status": "published", "files": files})
}

fn catalog_of(entries: Vec<Value>) -> Catalog {
    let c: Catalog = serde_json::from_value(json!({ "entries": entries })).unwrap();
    catalog::validate(&c, ValidateOpts { allow_http_loopback: true }).unwrap();
    c
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let d = std::env::temp_dir().join(format!("sigil-ms-it-{tag}-{}-{}", std::process::id(), super::layout::nonce()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        TempDir(d)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn new_store(app: &Path, c: Catalog) -> (Arc<Store>, Arc<RecordingSink>) {
    new_store_with(app, c, StoreSettings::test_defaults())
}

fn new_store_with(app: &Path, c: Catalog, settings: StoreSettings) -> (Arc<Store>, Arc<RecordingSink>) {
    let store = Store::with_settings(app, c, settings);
    let sink = Arc::new(RecordingSink::default());
    store.set_sink(sink.clone());
    (store, sink)
}

fn ids(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn partial_of(store: &Store, id: &str) -> PathBuf {
    let e = store.catalog.get(id).unwrap();
    store.layout.downloads().join(format!("{}.partial", partial_key(e, &e.files[0])))
}

fn downloads_is_empty(store: &Store) -> bool {
    fs::read_dir(store.layout.downloads()).map(|rd| rd.count() == 0).unwrap_or(true)
}

fn range_start(h: &Hit) -> Option<u64> {
    h.range.as_deref()?.strip_prefix("bytes=")?.trim_end_matches('-').parse().ok()
}

// ---------------------------------------------------------------------------------------------
// Download, resume, failover
// ---------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn installs_and_resumes_after_an_aborted_transfer() {
    let srv = Server::start().await;
    let data = blob(300_000, 1);
    srv.set(
        "/m.bin",
        Behavior::Abort {
            data: arc(&data),
            after: 100_000,
            n: Arc::new(AtomicUsize::new(1)),
        },
    );
    let app = TempDir::new("resume");
    let (store, sink) = new_store(&app.0, catalog_of(vec![entry("t/m", "t", vec![sha_file("m.bin", &data, &[srv.url("/m.bin")])])]));
    let paths = store.ensure(&ids(&["t/m"])).await.unwrap();
    assert_eq!(fs::read(paths[0].join("m.bin")).unwrap(), data);
    let hits = srv.hits_to("/m.bin");
    assert_eq!(hits.len(), 2, "{hits:?}");
    assert_eq!(hits[0].range, None);
    let resumed_at = range_start(&hits[1]).expect("the retry resumes with Range");
    assert!(resumed_at > 0 && resumed_at <= 100_000, "{resumed_at}");
    assert_eq!(hits[1].if_range, None, "pinned files are proven by their hash, no If-Range needed");
    assert!(downloads_is_empty(&store), "partial and meta are gone after the install");
    let rec = read_record(&store.layout.record_path(store.catalog.get("t/m").unwrap()), "t/m").unwrap();
    assert_eq!(rec.origin, Origin::Download);
    assert!(rec.pinned);
    let states: Vec<EntryState> = sink.states.lock().unwrap().iter().map(|s| s.state).collect();
    assert_eq!(states.first(), Some(&EntryState::Queued));
    assert!(states.contains(&EntryState::Downloading));
    assert_eq!(states.last(), Some(&EntryState::Installed));
    {
        let progress = sink.progress.lock().unwrap();
        assert!(progress.iter().all(|p| p.id == "t/m" && p.bytes_total == Some(300_000)));
        assert!(progress.iter().any(|p| p.attempt == 2 && p.bytes_done == 300_000));
    }
    // Installed again: no network at all.
    let n = srv.hits().len();
    assert_eq!(store.ensure(&ids(&["t/m"])).await.unwrap(), paths);
    assert_eq!(srv.hits().len(), n);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stalled_transfer_is_resumed_with_range() {
    let srv = Server::start().await;
    let data = blob(200_000, 2);
    srv.set(
        "/s.bin",
        Behavior::Stall {
            data: arc(&data),
            after: 70_000,
            n: Arc::new(AtomicUsize::new(1)),
        },
    );
    let app = TempDir::new("stall");
    let (store, _) = new_store(&app.0, catalog_of(vec![entry("t/s", "t", vec![sha_file("s.bin", &data, &[srv.url("/s.bin")])])]));
    let started = std::time::Instant::now();
    let p = store.ensure(&ids(&["t/s"])).await.unwrap();
    assert_eq!(fs::read(p[0].join("s.bin")).unwrap(), data);
    assert!(started.elapsed() < Duration::from_secs(10));
    let hits = srv.hits_to("/s.bin");
    assert_eq!(hits.len(), 2);
    assert_eq!(range_start(&hits[1]), Some(70_000));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_server_ignoring_range_restarts_cleanly() {
    let srv = Server::start().await;
    let data = blob(150_000, 3);
    srv.set(
        "/a.bin",
        Behavior::AbortThenIgnoreRange {
            data: arc(&data),
            after: 50_000,
            n: Arc::new(AtomicUsize::new(1)),
        },
    );
    let app = TempDir::new("ignore-range");
    let (store, _) = new_store(&app.0, catalog_of(vec![entry("t/a", "t", vec![sha_file("a.bin", &data, &[srv.url("/a.bin")])])]));
    let p = store.ensure(&ids(&["t/a"])).await.unwrap();
    assert_eq!(fs::read(p[0].join("a.bin")).unwrap(), data);
    let hits = srv.hits_to("/a.bin");
    assert_eq!(hits.len(), 2);
    assert!(range_start(&hits[1]).is_some(), "the client asked to resume");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_keeps_the_partial_and_the_next_install_resumes() {
    let srv = Server::start().await;
    let data = blob(600_000, 4);
    srv.set(
        "/big.bin",
        Behavior::Slow {
            data: arc(&data),
            chunk: 16 * 1024,
            delay: Duration::from_millis(15),
        },
    );
    let app = TempDir::new("cancel");
    let (store, _) = new_store(&app.0, catalog_of(vec![entry("t/big", "t", vec![sha_file("big.bin", &data, &[srv.url("/big.bin")])])]));
    let queued = store.enqueue(&ids(&["t/big"])).unwrap();
    let job = queued[0].job.clone().unwrap();
    let partial = partial_of(&store, "t/big");
    for _ in 0..400 {
        if fs::metadata(&partial).map(|m| m.len() > 64 * 1024).unwrap_or(false) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(store.cancel("t/big"));
    assert_eq!(job.wait().await.unwrap_err(), CANCELLED);
    assert_eq!(store.status("t/big").unwrap().state, EntryState::Cancelled);
    let kept = fs::metadata(&partial).expect("cancel keeps the .partial").len();
    assert!(kept > 0 && kept < data.len() as u64);
    assert!(meta_path(&partial).exists(), "and its metadata");
    assert!(store.installed_path("t/big").is_none(), "a partial file is never installed");
    let entry_dir = store.layout.entry_dir(store.catalog.get("t/big").unwrap());
    assert!(!entry_dir.join("1").exists());

    srv.set("/big.bin", Behavior::Serve(arc(&data)));
    let p = store.ensure(&ids(&["t/big"])).await.unwrap();
    assert_eq!(fs::read(p[0].join("big.bin")).unwrap(), data);
    let last = srv.hits_to("/big.bin").pop().unwrap();
    assert_eq!(range_start(&last), Some(kept), "resumed exactly where the cancel left off");
    assert_eq!(store.status("t/big").unwrap().state, EntryState::Installed);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failover_on_404_429_html_wrong_size_and_hash_mismatch() {
    let srv = Server::start().await;
    let data = blob(120_000, 5);
    let mut corrupt = data.clone();
    corrupt[60_000] ^= 0x55;
    srv.set("/404", Behavior::Status(404));
    srv.set("/429", Behavior::Status(429));
    srv.set("/503", Behavior::Status(503));
    srv.set("/html", Behavior::HtmlPage);
    srv.set("/html-body", Behavior::HtmlBody(data.len()));
    srv.set("/short", Behavior::Serve(arc(&data[..1000])));
    srv.set("/corrupt", Behavior::Serve(arc(&corrupt)));
    srv.set("/good", Behavior::Serve(arc(&data)));
    let order = ["/404", "/429", "/503", "/html", "/html-body", "/short", "/corrupt", "/good"];
    let urls: Vec<String> = order.iter().map(|p| srv.url(p)).collect();
    let app = TempDir::new("failover");
    let (store, _) = new_store(&app.0, catalog_of(vec![entry("t/f", "t", vec![sha_file("f.bin", &data, &urls)])]));
    let p = store.ensure(&ids(&["t/f"])).await.unwrap();
    assert_eq!(fs::read(p[0].join("f.bin")).unwrap(), data);
    let seen: Vec<String> = srv.hits().into_iter().map(|h| h.path).collect();
    assert_eq!(seen, order, "each bad mirror is tried once, in order");
    assert!(downloads_is_empty(&store));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_corrupt_or_partial_file_is_never_installed() {
    let srv = Server::start().await;
    // Large enough that three truncated attempts (3 x 30 KB) can never complete it.
    let data = blob(200_000, 6);
    let mut corrupt = data.clone();
    corrupt[10] ^= 1;
    srv.set("/corrupt", Behavior::Serve(arc(&corrupt)));
    // Ignores Range and always drops at 30 KB: only the first attempt makes progress.
    srv.set(
        "/truncated",
        Behavior::AbortThenIgnoreRange {
            data: arc(&data),
            after: 30_000,
            n: Arc::new(AtomicUsize::new(usize::MAX)),
        },
    );
    let app = TempDir::new("never");
    let (store, _) = new_store(
        &app.0,
        catalog_of(vec![entry(
            "t/x",
            "t",
            vec![sha_file("x.bin", &data, &[srv.url("/corrupt"), srv.url("/truncated")])],
        )]),
    );
    let err = store.ensure(&ids(&["t/x"])).await.unwrap_err();
    assert!(err.contains("does not match the catalog"), "{err}");
    assert!(err.contains("connection") || err.contains("closed"), "{err}");
    let st = store.status("t/x").unwrap();
    assert_eq!(st.state, EntryState::Failed);
    assert!(st.error.is_some());
    assert!(store.installed_path("t/x").is_none());
    let e = store.catalog.get("t/x").unwrap();
    assert!(!store.layout.record_path(e).exists());
    assert!(!store.layout.entry_dir(e).join("1").exists());
    // The truncated bytes stay only as a resumable partial, never under the install folder.
    let partial = partial_of(&store, "t/x");
    assert!(fs::metadata(&partial).map(|m| m.len() < data.len() as u64).unwrap_or(true));
    // 1 on the corrupt mirror; on the truncating one, the first attempt made progress (it does not
    // count), then 3 attempts that added nothing.
    assert_eq!(srv.hits_to("/corrupt").len(), 1);
    assert_eq!(srv.hits_to("/truncated").len(), 4);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_transfer_that_keeps_progressing_is_resumed_past_the_attempt_limit() {
    let srv = Server::start().await;
    // Honours Range but drops every connection after 30 KB: a flaky link, not a broken mirror.
    let data = blob(200_000, 9);
    srv.set(
        "/flaky.bin",
        Behavior::Abort {
            data: arc(&data),
            after: 30_000,
            n: Arc::new(AtomicUsize::new(usize::MAX)),
        },
    );
    let app = TempDir::new("progress");
    let (store, _) = new_store(&app.0, catalog_of(vec![entry("t/flaky", "t", vec![sha_file("f.bin", &data, &[srv.url("/flaky.bin")])])]));
    let p = store.ensure(&ids(&["t/flaky"])).await.unwrap();
    assert_eq!(fs::read(p[0].join("f.bin")).unwrap(), data);
    let hits = srv.hits_to("/flaky.bin");
    assert!(hits.len() >= 7, "every drop was resumed: {} requests", hits.len());
    let starts: Vec<u64> = hits.iter().map(|h| range_start(h).unwrap_or(0)).collect();
    assert!(starts.windows(2).all(|w| w[1] >= w[0]), "always resumed forward: {starts:?}");

    // Resumes are capped per mirror (a server trickling bytes forever).
    let app2 = TempDir::new("progress-cap");
    let (capped, _) = new_store_with(
        &app2.0,
        catalog_of(vec![entry("t/flaky", "t", vec![sha_file("f.bin", &data, &[srv.url("/flaky.bin")])])]),
        StoreSettings {
            max_resumes_per_mirror: 2,
            ..StoreSettings::test_defaults()
        },
    );
    let before = srv.hits_to("/flaky.bin").len();
    let err = capped.ensure(&ids(&["t/flaky"])).await.unwrap_err();
    assert!(err.contains("resumes"), "{err}");
    let used = srv.hits_to("/flaky.bin").len() - before;
    assert!((3..7).contains(&used), "{used} requests");
    assert!(capped.installed_path("t/flaky").is_none());
    assert!(fs::metadata(partial_of(&capped, "t/flaky")).is_ok_and(|m| m.len() > 0), "the partial is kept for a later resume");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn redirects_keep_the_range_header() {
    let srv = Server::start().await;
    let data = blob(200_000, 7);
    srv.set("/resolve/m.bin", Behavior::Redirect("/cdn/m.bin".into()));
    srv.set(
        "/cdn/m.bin",
        Behavior::Abort {
            data: arc(&data),
            after: 80_000,
            n: Arc::new(AtomicUsize::new(1)),
        },
    );
    let app = TempDir::new("redirect");
    let (store, _) = new_store(&app.0, catalog_of(vec![entry("t/r", "t", vec![sha_file("m.bin", &data, &[srv.url("/resolve/m.bin")])])]));
    let p = store.ensure(&ids(&["t/r"])).await.unwrap();
    assert_eq!(fs::read(p[0].join("m.bin")).unwrap(), data);
    let cdn = srv.hits_to("/cdn/m.bin");
    assert_eq!(cdn.len(), 2);
    assert!(range_start(&cdn[1]).is_some_and(|s| s > 0 && s <= 80_000), "{cdn:?}");
    assert_eq!(srv.hits_to("/resolve/m.bin")[1].range, cdn[1].range);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn git_oid_pinned_files_verify() {
    let srv = Server::start().await;
    let readme = b"# Model card\nlicense: mit\n".to_vec();
    srv.set("/README.md", Behavior::Serve(arc(&readme)));
    srv.set("/bad/README.md", Behavior::Serve(arc(b"# Model card\nlicense: MIT\n")));
    let app = TempDir::new("git");
    let (store, _) = new_store(
        &app.0,
        catalog_of(vec![entry(
            "t/card",
            "t",
            vec![git_file("README.md", &readme, &[srv.url("/bad/README.md"), srv.url("/README.md")])],
        )]),
    );
    let p = store.ensure(&ids(&["t/card"])).await.unwrap();
    assert_eq!(fs::read(p[0].join("README.md")).unwrap(), readme);
    let rec = read_record(&store.layout.record_path(store.catalog.get("t/card").unwrap()), "t/card").unwrap();
    assert_eq!(rec.files[0].git_oid.as_deref(), Some(git_blob_id(&readme).as_str()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn complete_partials_are_verified_without_network() {
    let srv = Server::start().await;
    let data = blob(50_000, 8);
    srv.set("/p.bin", Behavior::Serve(arc(&data)));
    let app = TempDir::new("complete-partial");
    let (store, _) = new_store(&app.0, catalog_of(vec![entry("t/p", "t", vec![sha_file("p.bin", &data, &[srv.url("/p.bin")])])]));
    let partial = partial_of(&store, "t/p");
    fs::create_dir_all(partial.parent().unwrap()).unwrap();
    fs::write(&partial, &data).unwrap();
    store.ensure(&ids(&["t/p"])).await.unwrap();
    assert!(srv.hits().is_empty(), "a complete, correct partial needs no request");

    // A complete but corrupt partial is thrown away and downloaded again.
    let app2 = TempDir::new("complete-partial-bad");
    let (store2, _) = new_store(&app2.0, catalog_of(vec![entry("t/p", "t", vec![sha_file("p.bin", &data, &[srv.url("/p.bin")])])]));
    let partial = partial_of(&store2, "t/p");
    fs::create_dir_all(partial.parent().unwrap()).unwrap();
    let mut bad = data.clone();
    bad[0] ^= 1;
    fs::write(&partial, &bad).unwrap();
    let p = store2.ensure(&ids(&["t/p"])).await.unwrap();
    assert_eq!(fs::read(p[0].join("p.bin")).unwrap(), data);
    assert_eq!(srv.hits_to("/p.bin").len(), 1);
    assert_eq!(srv.hits_to("/p.bin")[0].range, None);
}

// ---------------------------------------------------------------------------------------------
// Queue
// ---------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_download_at_a_time_and_never_two_per_host() {
    let counters = Counters::default();
    let a = Server::start_with(counters.clone()).await;
    let b = Server::start_with(counters.clone()).await;
    let slow = |d: &[u8]| Behavior::Slow {
        data: arc(d),
        chunk: 8 * 1024,
        delay: Duration::from_millis(10),
    };
    let d: Vec<Vec<u8>> = (0..5).map(|i| blob(80_000, 20 + i)).collect();
    for (i, x) in d.iter().enumerate() {
        a.set(&format!("/{i}"), slow(x));
        b.set(&format!("/{i}"), slow(x));
    }
    let app = TempDir::new("queue");
    let c = catalog_of(vec![
        entry("t/a0", "t", vec![sha_file("f", &d[0], &[a.url("/0")])]),
        entry("t/a1", "t", vec![sha_file("f", &d[1], &[a.url("/1")])]),
        entry("t/a2", "t", vec![sha_file("f", &d[2], &[a.url("/2")])]),
        entry("t/a3", "t", vec![sha_file("f", &d[3], &[a.url("/3")])]),
        entry("t/b4", "t", vec![sha_file("f", &d[4], &[b.url("/4")])]),
    ]);
    let (store, _) = new_store(&app.0, c);
    // Default: one download at a time, even for three independent entries.
    store.ensure(&ids(&["t/a0", "t/a1", "t/a2"])).await.unwrap();
    assert_eq!(a.peak(), 1);
    let order: Vec<String> = a.hits().into_iter().map(|h| h.path).collect();
    assert_eq!(order, ["/0", "/1", "/2"], "first come, first served");

    // Two slots, but both entries on one host: still one transfer at a time.
    store
        .set_config(StoreConfig {
            mirror: MirrorPreference::Auto,
            custom_url: None,
            max_concurrent: 2,
        })
        .unwrap();
    counters.peak.store(0, Ordering::SeqCst);
    let s = store.clone();
    let t = tokio::spawn(async move { s.ensure(&ids(&["t/a3"])).await });
    store.ensure(&ids(&["t/b4"])).await.unwrap();
    t.await.unwrap().unwrap();
    assert_eq!(counters.peak.load(Ordering::SeqCst), 2, "two hosts, two slots: two transfers");

    let app2 = TempDir::new("queue-host");
    let c2 = catalog_of(vec![
        entry("t/x", "t", vec![sha_file("f", &d[0], &[a.url("/0")])]),
        entry("t/y", "t", vec![sha_file("f", &d[1], &[a.url("/1")])]),
    ]);
    let (store2, _) = new_store(&app2.0, c2);
    store2
        .set_config(StoreConfig {
            mirror: MirrorPreference::Auto,
            custom_url: None,
            max_concurrent: 2,
        })
        .unwrap();
    counters.peak.store(0, Ordering::SeqCst);
    store2.ensure(&ids(&["t/x", "t/y"])).await.unwrap();
    assert_eq!(counters.peak.load(Ordering::SeqCst), 1, "never two downloads from one host");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn requirements_install_first_and_paths_follow_input_order() {
    let srv = Server::start().await;
    let rt = blob(10_000, 30);
    let app_bytes = blob(10_000, 31);
    srv.set("/rt", Behavior::Serve(arc(&rt)));
    srv.set("/app", Behavior::Serve(arc(&app_bytes)));
    let mut app_entry = entry("t/app", "t", vec![sha_file("app.bin", &app_bytes, &[srv.url("/app")])]);
    app_entry["requires"] = json!(["t/rt"]);
    let mut rt_entry = entry("t/rt", "t", vec![sha_file("rt.dll", &rt, &[srv.url("/rt")])]);
    rt_entry["kind"] = json!("runtime");
    rt_entry["variant"] = json!("cpu");
    let app = TempDir::new("requires");
    let (store, _) = new_store(&app.0, catalog_of(vec![app_entry, rt_entry]));
    let paths = store.ensure(&ids(&["t/app", "t/rt"])).await.unwrap();
    let order: Vec<String> = srv.hits().into_iter().map(|h| h.path).collect();
    assert_eq!(order, ["/rt", "/app"]);
    assert!(paths[0].join("app.bin").is_file());
    assert!(paths[1].join("rt.dll").is_file());
    assert!(paths[1].starts_with(app.0.join("runtimes")), "runtime packs live under runtimes/");
    assert!(paths[1].ends_with(Path::new("rt").join("1").join("cpu")));
    let err = store.ensure(&ids(&["t/nope"])).await.unwrap_err();
    assert!(err.contains("unknown catalog id"), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_requirement_fails_its_dependents() {
    let srv = Server::start().await;
    srv.set("/rt", Behavior::Status(404));
    let d = blob(1000, 32);
    srv.set("/app", Behavior::Serve(arc(&d)));
    let mut app_entry = entry("t/app", "t", vec![sha_file("app.bin", &d, &[srv.url("/app")])]);
    app_entry["requires"] = json!(["t/rt"]);
    let rt_entry = entry("t/rt", "t", vec![sha_file("rt.bin", &blob(10, 33), &[srv.url("/rt")])]);
    let app = TempDir::new("requires-fail");
    let (store, _) = new_store(&app.0, catalog_of(vec![app_entry, rt_entry]));
    let err = store.ensure(&ids(&["t/app"])).await.unwrap_err();
    assert!(err.contains("t/rt (required) could not be installed"), "{err}");
    assert!(srv.hits_to("/app").is_empty());
}

// ---------------------------------------------------------------------------------------------
// Archives
// ---------------------------------------------------------------------------------------------

fn zip_of(members: &[(&str, &[u8])]) -> Vec<u8> {
    let mut w = zip::ZipWriter::new(io::Cursor::new(Vec::new()));
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for (name, data) in members {
        w.start_file(*name, opts).unwrap();
        w.write_all(data).unwrap();
    }
    w.finish().unwrap().into_inner()
}

fn archive_file(path: &str, data: &[u8], url: &str, strip: u32, include: &[&str]) -> Value {
    let mut f = sha_file(path, data, &[url.to_string()]);
    f["archive"] = json!({"format": "zip", "strip": strip, "include": include});
    f
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn archives_extract_into_the_entry_folder() {
    let srv = Server::start().await;
    let onnx = blob(40_000, 40);
    let z = zip_of(&[("pack-1.0/model.onnx", &onnx), ("pack-1.0/tokens.txt", b"a b c"), ("pack-1.0/docs/readme.md", b"doc")]);
    srv.set("/pack.zip", Behavior::Serve(arc(&z)));
    let app = TempDir::new("archive");
    let (store, sink) = new_store(
        &app.0,
        catalog_of(vec![entry(
            "t/pack",
            "t",
            vec![archive_file("pack.zip", &z, &srv.url("/pack.zip"), 1, &["*.onnx", "*.txt"])],
        )]),
    );
    let p = store.ensure(&ids(&["t/pack"])).await.unwrap();
    assert_eq!(fs::read(p[0].join("model.onnx")).unwrap(), onnx);
    assert_eq!(fs::read(p[0].join("tokens.txt")).unwrap(), b"a b c");
    assert!(!p[0].join("docs").exists());
    assert!(!p[0].join("pack.zip").exists(), "the archive itself is not installed");
    assert!(downloads_is_empty(&store), "and its download is removed once extracted");
    let rec = read_record(&store.layout.record_path(store.catalog.get("t/pack").unwrap()), "t/pack").unwrap();
    assert_eq!(rec.contents.len(), 2);
    assert!(rec.files[0].archive);
    assert!(sink.states.lock().unwrap().iter().any(|s| s.state == EntryState::Extracting));
    // Verification re-hashes the extracted files.
    let st = install::verify(&store, "t/pack").await.unwrap();
    assert_eq!(st.state, EntryState::Installed);
}

/// Member names real archives ship (sherpa's espeak-ng-data: `voices/!v/Mr serious`) that the
/// catalog path charset does not allow: installed, found again, verified and removed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn archive_members_with_spaces_punctuation_and_unicode_stay_installed() {
    let srv = Server::start().await;
    let onnx = blob(30_000, 41);
    let members: [(&str, &[u8]); 5] = [
        ("kokoro/model.onnx", &onnx),
        ("kokoro/espeak-ng-data/voices/!v/Mr serious", b"voice 1"),
        ("kokoro/espeak-ng-data/voices/!v/Alex", b"voice 2"),
        ("kokoro/espeak-ng-data/lang/(x)@y~z,w", b"lang"),
        ("kokoro/donn\u{e9}es/mod\u{e8}le.txt", b"unicode"),
    ];
    let z = zip_of(&members);
    srv.set("/kokoro.zip", Behavior::Serve(arc(&z)));
    let app = TempDir::new("archive-names");
    let c = catalog_of(vec![
        entry("t/kokoro", "t", vec![archive_file("kokoro.zip", &z, &srv.url("/kokoro.zip"), 1, &[])]),
        entry("vosk/names", "vosk", vec![archive_file("names.zip", &z, &srv.url("/kokoro.zip"), 1, &[])]),
    ]);
    let (store, _) = new_store(&app.0, c);
    for id in ["t/kokoro", "vosk/names"] {
        let root = store.ensure(&ids(&[id])).await.unwrap().remove(0);
        assert_eq!(fs::read(root.join("espeak-ng-data").join("voices").join("!v").join("Mr serious")).unwrap(), b"voice 1");
        assert_eq!(fs::read(root.join("donn\u{e9}es").join("mod\u{e8}le.txt")).unwrap(), b"unicode");
        assert_eq!(store.installed_path(id).as_deref(), Some(root.as_path()), "{id}: the record resolves every file");
        assert_eq!(store.status(id).unwrap().state, EntryState::Installed);
        let hits = srv.hits().len();
        assert_eq!(store.ensure(&ids(&[id])).await.unwrap(), std::slice::from_ref(&root), "{id}");
        assert_eq!(srv.hits().len(), hits, "{id}: installed means no second download");
        assert_eq!(install::verify(&store, id).await.unwrap().state, EntryState::Installed, "{id}");
        let st = install::remove(&store, id).await.unwrap();
        assert_eq!(st.state, EntryState::NotInstalled, "{id}");
        assert!(!root.join("espeak-ng-data").join("voices").join("!v").join("Mr serious").exists(), "{id}");
    }
    assert!(app.0.join("vosk-models").is_dir(), "the shared legacy folder itself stays");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unsafe_archives_are_refused_and_nothing_is_installed() {
    let srv = Server::start().await;
    let slip = zip_of(&[("ok.txt", b"fine"), ("../../evil.txt", b"pwned")]);
    let mut w = zip::ZipWriter::new(io::Cursor::new(Vec::new()));
    let opts = zip::write::SimpleFileOptions::default();
    w.start_file("model.onnx", opts).unwrap();
    w.write_all(b"onnx").unwrap();
    w.add_symlink("escape", "../../../Windows", opts).unwrap();
    let link = w.finish().unwrap().into_inner();
    let bomb = zip_of(&[("zeros.bin", &vec![0u8; 2 << 20])]);
    srv.set("/slip.zip", Behavior::Serve(arc(&slip)));
    srv.set("/link.zip", Behavior::Serve(arc(&link)));
    srv.set("/bomb.zip", Behavior::Serve(arc(&bomb)));
    let app = TempDir::new("unsafe-archive");
    let settings = StoreSettings {
        limits: ExtractLimits {
            ratio_floor_bytes: 0,
            max_ratio: 100,
            ..ExtractLimits::default()
        },
        ..StoreSettings::test_defaults()
    };
    let (store, _) = new_store_with(
        &app.0,
        catalog_of(vec![
            entry("t/slip", "t", vec![archive_file("slip.zip", &slip, &srv.url("/slip.zip"), 0, &[])]),
            entry("t/link", "t", vec![archive_file("link.zip", &link, &srv.url("/link.zip"), 0, &[])]),
            entry("t/bomb", "t", vec![archive_file("bomb.zip", &bomb, &srv.url("/bomb.zip"), 0, &[])]),
        ]),
        settings,
    );
    for (id, needle) in [("t/slip", ".."), ("t/link", "symbolic link"), ("t/bomb", "zip bomb")] {
        let err = store.ensure(&ids(&[id])).await.unwrap_err();
        assert!(err.contains(needle), "{id}: {err}");
        assert!(store.installed_path(id).is_none());
        let e = store.catalog.get(id).unwrap();
        assert!(!store.layout.entry_dir(e).join("1").exists());
        assert!(!partial_of(&store, id).exists(), "{id}: rejected archive bytes are not kept");
    }
    assert!(!app.0.join("evil.txt").exists() && !app.0.join("models").join("evil.txt").exists());
    let staging = fs::read_dir(store.layout.staging()).map(|rd| rd.count()).unwrap_or(0);
    assert_eq!(staging, 0, "staging folders are cleaned up");
}

// ---------------------------------------------------------------------------------------------
// Legacy folders
// ---------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn existing_legacy_files_are_adopted_without_downloading() {
    let srv = Server::start().await;
    let model = blob(70_000, 50);
    srv.set("/ggml-test.bin", Behavior::Serve(arc(&model)));
    let app = TempDir::new("adopt");
    let whisper = app.0.join("whisper");
    fs::create_dir_all(&whisper).unwrap();
    fs::write(whisper.join("ggml-test.bin"), &model).unwrap();
    let c = catalog_of(vec![
        entry("whisper/test", "whisper", vec![sha_file("ggml-test.bin", &model, &[srv.url("/ggml-test.bin")])]),
        entry("whisper/other", "whisper", vec![sha_file("ggml-other.bin", &model, &[srv.url("/ggml-test.bin")])]),
    ]);
    let (store, sink) = new_store(&app.0, c);
    // Startup scan: registered, no request.
    assert_eq!(legacy::adopt_all(&store), ["whisper/test"]);
    let st = store.status("whisper/test").unwrap();
    assert_eq!(st.state, EntryState::Installed);
    assert!(st.adopted);
    assert_eq!(store.installed_path("whisper/test").as_deref(), Some(whisper.as_path()));
    assert!(sink.states.lock().unwrap().iter().any(|s| s.id == "whisper/test" && s.adopted));
    assert!(srv.hits().is_empty());
    // ensure() on an adopted entry: still nothing downloaded.
    assert_eq!(store.ensure(&ids(&["whisper/test"])).await.unwrap(), std::slice::from_ref(&whisper));
    assert!(srv.hits().is_empty());

    // Adoption inside an install (no startup scan yet) works the same way.
    let app2 = TempDir::new("adopt-install");
    fs::create_dir_all(app2.0.join("whisper")).unwrap();
    fs::write(app2.0.join("whisper").join("ggml-test.bin"), &model).unwrap();
    let (store2, _) = new_store(
        &app2.0,
        catalog_of(vec![entry("whisper/test", "whisper", vec![sha_file("ggml-test.bin", &model, &[srv.url("/ggml-test.bin")])])]),
    );
    store2.ensure(&ids(&["whisper/test"])).await.unwrap();
    assert!(srv.hits().is_empty());
    assert!(store2.status("whisper/test").unwrap().adopted);

    // Removing an adopted entry deletes its file from the legacy folder, and only that file.
    fs::write(whisper.join("ggml-unrelated.bin"), b"user file").unwrap();
    let st = install::remove(&store, "whisper/test").await.unwrap();
    assert_eq!(st.state, EntryState::NotInstalled);
    assert!(!whisper.join("ggml-test.bin").exists());
    assert!(whisper.join("ggml-unrelated.bin").exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_mismatching_legacy_file_is_left_alone_until_installed() {
    let srv = Server::start().await;
    let model = blob(40_000, 51);
    let other = blob(40_000, 52);
    srv.set("/m.bin", Behavior::Serve(arc(&model)));
    let app = TempDir::new("adopt-mismatch");
    let whisper = app.0.join("whisper");
    fs::create_dir_all(&whisper).unwrap();
    fs::write(whisper.join("ggml-m.bin"), &other).unwrap();
    let (store, _) = new_store(
        &app.0,
        catalog_of(vec![entry("whisper/m", "whisper", vec![sha_file("ggml-m.bin", &model, &[srv.url("/m.bin")])])]),
    );
    assert!(legacy::adopt_all(&store).is_empty());
    assert_eq!(fs::read(whisper.join("ggml-m.bin")).unwrap(), other, "never modified by the scan");
    let cache = fs::read_to_string(store.layout.models.join(".adoption-cache.json")).unwrap();
    assert!(cache.contains("whisper/ggml-m.bin"), "{cache}");
    assert!(legacy::adopt_all(&store).is_empty());
    // An explicit install replaces it with the catalog's file.
    store.ensure(&ids(&["whisper/m"])).await.unwrap();
    assert_eq!(fs::read(whisper.join("ggml-m.bin")).unwrap(), model);
    assert!(!store.status("whisper/m").unwrap().adopted);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn vosk_entries_install_into_vosk_models() {
    let srv = Server::start().await;
    let z = zip_of(&[("vosk-model-x/am/final.mdl", &blob(30_000, 53)), ("vosk-model-x/README", b"x")]);
    srv.set("/vosk-model-x.zip", Behavior::Serve(arc(&z)));
    let app = TempDir::new("vosk");
    let (store, _) = new_store(
        &app.0,
        catalog_of(vec![entry("vosk/x", "vosk", vec![sha_file("vosk-model-x.zip", &z, &[srv.url("/vosk-model-x.zip")])])]),
    );
    let p = store.ensure(&ids(&["vosk/x"])).await.unwrap();
    assert_eq!(p[0], app.0.join("vosk-models"));
    assert_eq!(fs::read(p[0].join("vosk-model-x.zip")).unwrap(), z, "the zip itself (vosk-browser loads it)");
    assert_eq!(legacy::vosk_entry_id("vosk-model-x"), "vosk/x");
}

// ---------------------------------------------------------------------------------------------
// Verify, remove, import, local assets, policy
// ---------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn verify_unregisters_a_corrupted_install_and_remove_cleans_up() {
    let srv = Server::start().await;
    let data = blob(20_000, 60);
    srv.set("/v.bin", Behavior::Serve(arc(&data)));
    let app = TempDir::new("verify");
    let (store, _) = new_store(&app.0, catalog_of(vec![entry("t/v", "t", vec![sha_file("v.bin", &data, &[srv.url("/v.bin")])])]));
    let root = store.ensure(&ids(&["t/v"])).await.unwrap().remove(0);
    assert_eq!(install::verify(&store, "t/v").await.unwrap().state, EntryState::Installed);
    let mut bad = data.clone();
    bad[5] ^= 0xFF;
    fs::write(root.join("v.bin"), &bad).unwrap();
    let st = install::verify(&store, "t/v").await.unwrap();
    assert_eq!(st.state, EntryState::Failed);
    assert!(st.error.as_deref().unwrap_or("").contains("corrupt"), "{st:?}");
    assert!(store.installed_path("t/v").is_none(), "a corrupt install is no longer installed");
    // Reinstall, then remove: files, record and partials go.
    let root = store.ensure(&ids(&["t/v"])).await.unwrap().remove(0);
    assert_eq!(fs::read(root.join("v.bin")).unwrap(), data);
    let st = install::remove(&store, "t/v").await.unwrap();
    assert_eq!(st.state, EntryState::NotInstalled);
    assert!(!root.exists());
    assert!(!store.layout.record_path(store.catalog.get("t/v").unwrap()).exists());
    assert!(install::verify(&store, "t/v").await.is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remove_cancels_a_running_install_and_drops_its_partial() {
    let srv = Server::start().await;
    let data = blob(400_000, 61);
    srv.set(
        "/r.bin",
        Behavior::Slow {
            data: arc(&data),
            chunk: 8 * 1024,
            delay: Duration::from_millis(20),
        },
    );
    let app = TempDir::new("remove-running");
    let (store, _) = new_store(&app.0, catalog_of(vec![entry("t/r", "t", vec![sha_file("r.bin", &data, &[srv.url("/r.bin")])])]));
    store.enqueue(&ids(&["t/r"])).unwrap();
    let partial = partial_of(&store, "t/r");
    for _ in 0..300 {
        if fs::metadata(&partial).map(|m| m.len() > 0).unwrap_or(false) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let st = install::remove(&store, "t/r").await.unwrap();
    assert_eq!(st.state, EntryState::NotInstalled);
    assert!(!partial.exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn import_local_verifies_files_and_folders() {
    let a = blob(12_000, 70);
    let b = blob(8_000, 71);
    let app = TempDir::new("import");
    let src = TempDir::new("import-src");
    fs::write(src.0.join("one.bin"), &a).unwrap();
    fs::create_dir_all(src.0.join("pair")).unwrap();
    fs::write(src.0.join("pair").join("a.bin"), &a).unwrap();
    fs::write(src.0.join("pair").join("b.bin"), &b).unwrap();
    fs::write(src.0.join("wrong.bin"), &b).unwrap();
    let never = ["https://unreachable.invalid/x".to_string()];
    let c = catalog_of(vec![
        entry("t/one", "t", vec![sha_file("one.bin", &a, &never)]),
        entry("t/pair", "t", vec![sha_file("a.bin", &a, &never), sha_file("sub/b.bin", &b, &never)]),
    ]);
    let (store, _) = new_store(&app.0, c);
    let err = install::import_local(&store, "t/one", &src.0.join("wrong.bin").display().to_string())
        .await
        .unwrap_err();
    assert!(err.contains("catalog expects") || err.contains("does not match"), "{err}");
    assert!(store.installed_path("t/one").is_none());
    let st = install::import_local(&store, "t/one", &src.0.join("one.bin").display().to_string()).await.unwrap();
    assert_eq!(st.state, EntryState::Installed);
    let rec = read_record(&store.layout.record_path(store.catalog.get("t/one").unwrap()), "t/one").unwrap();
    assert_eq!(rec.origin, Origin::Import);
    assert!(src.0.join("one.bin").exists(), "import copies, never moves the user's file");
    let err = install::import_local(&store, "t/pair", &src.0.join("one.bin").display().to_string())
        .await
        .unwrap_err();
    assert!(err.contains("choose the folder"), "{err}");
    let st = install::import_local(&store, "t/pair", &src.0.join("pair").display().to_string()).await.unwrap();
    assert_eq!(st.state, EntryState::Installed);
    let root = store.installed_path("t/pair").unwrap();
    assert_eq!(fs::read(root.join("sub").join("b.bin")).unwrap(), b);
    assert!(downloads_is_empty(&store), "import copies leave nothing behind");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unpublished_entries_need_an_import_or_local_assets() {
    let pack = blob(9_000, 80);
    let mut e = entry("t/pack", "t", vec![unpinned_file("sigil_engine.dll", 0, &["https://github.com/Mewnah/Sigil/releases/download/assets-v1/sigil_engine.dll".to_string()])]);
    e["status"] = json!("unpublished");
    e["kind"] = json!("runtime");
    let app = TempDir::new("unpublished");
    let (store, _) = new_store(&app.0, catalog_of(vec![e.clone()]));
    let err = store.ensure(&ids(&["t/pack"])).await.unwrap_err();
    assert!(err.contains("not published") && err.contains("import"), "{err}");

    // A custom mirror serves unpublished entries only for hash-pinned files: an unpinned pack (DLLs)
    // installs only by import or local assets, in every build.
    let srv = Server::start().await;
    srv.set("/mirror/t/pack/sigil_engine.dll", Behavior::Serve(arc(&pack)));
    srv.set("/mirror/t/provider/provider.dll", Behavior::Serve(arc(&pack)));
    let mut pinned = entry(
        "t/provider",
        "t",
        vec![sha_file("provider.dll", &pack, &["https://github.com/Mewnah/Sigil/releases/download/assets-v1/provider.dll".to_string()])],
    );
    pinned["status"] = json!("unpublished");
    pinned["kind"] = json!("runtime");
    let app3 = TempDir::new("unpublished-mirror");
    let (store3, _) = new_store(&app3.0, catalog_of(vec![e.clone(), pinned]));
    store3
        .set_config(StoreConfig {
            mirror: MirrorPreference::Custom,
            custom_url: Some(srv.url("/mirror/")),
            max_concurrent: 1,
        })
        .unwrap();
    let err = store3.ensure(&ids(&["t/pack"])).await.unwrap_err();
    assert!(err.contains("not published") && err.contains("pins by hash"), "{err}");
    assert!(srv.hits().is_empty(), "nothing was fetched for the unpinned pack");
    let root = store3.ensure(&ids(&["t/provider"])).await.unwrap().remove(0);
    assert_eq!(fs::read(root.join("provider.dll")).unwrap(), pack);
    assert_eq!(srv.hits().len(), 1);

    let assets = TempDir::new("assets");
    fs::create_dir_all(assets.0.join("t").join("pack")).unwrap();
    fs::write(assets.0.join("t").join("pack").join("sigil_engine.dll"), &pack).unwrap();
    let app2 = TempDir::new("unpublished-assets");
    let settings = StoreSettings {
        local_assets: Some(assets.0.clone()),
        allow_tofu_network: false,
        ..StoreSettings::test_defaults()
    };
    let (store2, _) = new_store_with(&app2.0, catalog_of(vec![e]), settings);
    let root = store2.ensure(&ids(&["t/pack"])).await.unwrap().remove(0);
    assert_eq!(fs::read(root.join("sigil_engine.dll")).unwrap(), pack);
    let rec = read_record(&store2.layout.record_path(store2.catalog.get("t/pack").unwrap()), "t/pack").unwrap();
    assert_eq!(rec.origin, Origin::LocalAssets);
    assert!(!rec.pinned);
    assert_eq!(rec.files[0].sha256.as_deref(), Some(sha256_hex(&pack).as_str()), "trust on first use records the hash");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn trust_on_first_use_is_for_data_in_development_only() {
    let srv = Server::start().await;
    let data = blob(5_000, 90);
    srv.set("/data.bin", Behavior::Serve(arc(&data)));
    srv.set("/tool.dll", Behavior::Serve(arc(&data)));
    let c = || {
        catalog_of(vec![
            entry("t/data", "t", vec![unpinned_file("data.bin", data.len(), &[srv.url("/data.bin")])]),
            entry("t/tool", "t", vec![unpinned_file("tool.dll", data.len(), &[srv.url("/tool.dll")])]),
        ])
    };
    let app = TempDir::new("tofu-dev");
    let (dev, _) = new_store(&app.0, c());
    dev.ensure(&ids(&["t/data"])).await.unwrap();
    let err = dev.ensure(&ids(&["t/tool"])).await.unwrap_err();
    assert!(err.contains("executable"), "{err}");
    let app2 = TempDir::new("tofu-release");
    let (release, _) = new_store_with(
        &app2.0,
        c(),
        StoreSettings {
            allow_tofu_network: false,
            ..StoreSettings::test_defaults()
        },
    );
    let err = release.ensure(&ids(&["t/data"])).await.unwrap_err();
    assert!(err.contains("pinned"), "{err}");
    assert!(srv.hits_to("/tool.dll").is_empty());
    assert_eq!(srv.hits_to("/data.bin").len(), 1, "release never fetched it");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn custom_mirror_is_tried_first() {
    let srv = Server::start().await;
    let data = blob(6_000, 95);
    srv.set("/mirror/t/c/c.bin", Behavior::Serve(arc(&data)));
    srv.set("/origin/c.bin", Behavior::Serve(arc(&data)));
    let app = TempDir::new("custom");
    let (store, _) = new_store(&app.0, catalog_of(vec![entry("t/c", "t", vec![sha_file("c.bin", &data, &[srv.url("/origin/c.bin")])])]));
    store
        .set_config(StoreConfig {
            mirror: MirrorPreference::Custom,
            custom_url: Some(srv.url("/mirror/")),
            max_concurrent: 1,
        })
        .unwrap();
    store.ensure(&ids(&["t/c"])).await.unwrap();
    let paths: Vec<String> = srv.hits().into_iter().map(|h| h.path).collect();
    assert_eq!(paths, ["/mirror/t/c/c.bin"]);
}

// ---------------------------------------------------------------------------------------------
// Vosk unpinned fetch, disk usage, housekeeping, catalog cross-checks
// ---------------------------------------------------------------------------------------------

fn accept_zip(path: &Path, limits: &ExtractLimits) -> Result<(), String> {
    super::extract::validate_zip(path, limits, &|| false)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unpinned_fetch_resumes_validates_and_keeps_the_old_file_on_failure() {
    let srv = Server::start().await;
    let z = zip_of(&[("m/final.mdl", &blob(50_000, 100))]);
    srv.set(
        "/m.zip",
        Behavior::Abort {
            data: arc(&z),
            after: 20_000,
            n: Arc::new(AtomicUsize::new(1)),
        },
    );
    srv.set("/page.zip", Behavior::HtmlPage);
    srv.set("/broken.zip", Behavior::Serve(arc(&z[..z.len() - 100])));
    let app = TempDir::new("unpinned");
    let (store, sink) = new_store(&app.0, catalog_of(vec![]));
    let dest = app.0.join("vosk-models").join("vosk-model-m.zip");
    let cancel = super::CancelToken::default();
    let digest = store
        .fetch_unpinned("vosk/m", "vosk-model-m.zip", &srv.url("/m.zip"), &dest, accept_zip, &cancel)
        .await
        .unwrap();
    assert_eq!(fs::read(&dest).unwrap(), z);
    assert_eq!(digest.sha256(), Some(sha256_hex(&z).as_str()));
    let resume = &srv.hits_to("/m.zip")[1];
    assert_eq!(range_start(resume).map(|s| s > 0), Some(true));
    assert_eq!(
        resume.if_range.as_deref(),
        Some("\"fixture-etag\""),
        "an unpinned resume is conditional on the validator"
    );
    assert!(sink.progress.lock().unwrap().iter().all(|p| p.id == "vosk/m"));
    for bad in ["/page.zip", "/broken.zip"] {
        let r = store
            .fetch_unpinned("vosk/m", "vosk-model-m.zip", &srv.url(bad), &dest, accept_zip, &cancel)
            .await;
        assert!(matches!(r, Err(DownloadError::Failed(_))), "{bad}: {r:?}");
        assert_eq!(fs::read(&dest).unwrap(), z, "{bad}: the good zip is untouched");
    }
    assert!(downloads_is_empty(&store));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disk_usage_counts_installs_and_partials() {
    let srv = Server::start().await;
    let a = blob(10_000, 110);
    let b = blob(300_000, 111);
    srv.set("/a", Behavior::Serve(arc(&a)));
    srv.set(
        "/b",
        Behavior::AbortThenIgnoreRange {
            data: arc(&b),
            after: 50_000,
            n: Arc::new(AtomicUsize::new(usize::MAX)),
        },
    );
    let app = TempDir::new("disk");
    let (store, _) = new_store(
        &app.0,
        catalog_of(vec![
            entry("t/a", "t", vec![sha_file("a.bin", &a, &[srv.url("/a")])]),
            entry("u/b", "u", vec![sha_file("b.bin", &b, &[srv.url("/b")])]),
        ]),
    );
    store.ensure(&ids(&["t/a"])).await.unwrap();
    let _ = store.ensure(&ids(&["u/b"])).await;
    let du = store.disk_usage();
    assert_eq!(du.per_family.get("t"), Some(&10_000));
    let partial = fs::metadata(partial_of(&store, "u/b")).unwrap().len();
    assert!(partial > 0);
    assert_eq!(du.per_family.get("u"), Some(&partial));
    assert_eq!(du.total_bytes, 10_000 + partial);
    assert!(du.free_bytes.is_some() || cfg!(not(windows)));
    let listing = super::list_text(&store);
    assert!(listing.contains("t/a") && listing.contains("installed"), "{listing}");
}

#[test]
fn housekeeping_removes_leftovers_but_keeps_resumable_partials() {
    let app = TempDir::new("sweep");
    let data = blob(100, 120);
    let c = catalog_of(vec![entry("t/k", "t", vec![sha_file("k.bin", &data, &["https://x.invalid/k".to_string()])])]);
    let store = Store::for_test(&app.0, c);
    let old = std::time::SystemTime::now() - Duration::from_secs(40 * 24 * 3600);
    let set_old = |p: &Path| {
        let f = fs::OpenOptions::new().write(true).open(p).unwrap();
        f.set_modified(old).unwrap();
    };
    let staging_old = store.layout.staging().join("t--k.1.1");
    fs::create_dir_all(&staging_old).unwrap();
    let marker = staging_old.join("x");
    fs::write(&marker, b"x").unwrap();
    let trash = store.layout.trash().join("gone");
    fs::create_dir_all(&trash).unwrap();
    let known = partial_of(&store, "t/k");
    fs::create_dir_all(known.parent().unwrap()).unwrap();
    fs::write(&known, b"part").unwrap();
    let stranger = store.layout.downloads().join("url-deadbeef.partial");
    fs::write(&stranger, b"old").unwrap();
    set_old(&stranger);
    let orphan = store.layout.entry_dir(store.catalog.get("t/k").unwrap()).join("0.9");
    fs::create_dir_all(&orphan).unwrap();
    fs::write(orphan.join("k.bin"), b"old").unwrap();
    // Directory mtimes cannot be set portably; age the staging folder through its only file and
    // rely on the orphan rule's own age check being exercised with a fresh folder (kept).
    store.sweep();
    assert!(!trash.exists(), "the trash is emptied");
    assert!(known.exists(), "a resumable partial of a catalog file is kept");
    assert!(!stranger.exists(), "an old partial nobody references is removed");
    assert!(orphan.exists(), "a fresh unreferenced folder may still be committing: kept");
    assert!(staging_old.exists(), "fresh staging folders are kept");
}

fn runtime_pack(id: &str, version: &str, data: &[u8], url: &str) -> Value {
    let mut e = entry(id, "runtime", vec![sha_file("rt.dll", data, &[url.to_string()])]);
    e["kind"] = json!("runtime");
    e["variant"] = json!("cpu");
    e["version"] = json!(version);
    e
}

/// Updating a runtime pack while the old version's DLL is loaded: the old folder cannot move, is
/// listed in `models/.pending-trash.json`, counted in disk usage, and removed at the next start.
#[cfg(windows)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_old_pack_that_is_in_use_is_removed_at_the_next_start() {
    use std::os::windows::fs::OpenOptionsExt;
    let srv = Server::start().await;
    let v1 = blob(20_000, 130);
    let v2 = blob(30_000, 131);
    srv.set("/1/rt.dll", Behavior::Serve(arc(&v1)));
    srv.set("/2/rt.dll", Behavior::Serve(arc(&v2)));
    let app = TempDir::new("pending-trash");
    let (s1, _) = new_store(&app.0, catalog_of(vec![runtime_pack("runtime/rt", "1", &v1, &srv.url("/1/rt.dll"))]));
    let old = s1.ensure(&ids(&["runtime/rt"])).await.unwrap().remove(0);
    assert!(old.ends_with(Path::new("runtimes").join("rt").join("1").join("cpu")), "{}", old.display());
    // The loaded DLL: open without delete sharing, as LoadLibrary does.
    let loaded = fs::OpenOptions::new().read(true).share_mode(0x1).open(old.join("rt.dll")).unwrap();
    let catalog_v2 = || catalog_of(vec![runtime_pack("runtime/rt", "2", &v2, &srv.url("/2/rt.dll"))]);
    let (s2, _) = new_store(&app.0, catalog_v2());
    let new = s2.ensure(&ids(&["runtime/rt"])).await.unwrap().remove(0);
    assert_ne!(new, old);
    assert_eq!(fs::read(new.join("rt.dll")).unwrap(), v2);
    assert!(old.join("rt.dll").exists(), "in use: kept for now");
    let pending_file = app.0.join("models").join(".pending-trash.json");
    let pending = fs::read_to_string(&pending_file).unwrap();
    assert!(pending.contains("runtimes/rt/1/cpu"), "{pending}");
    let du = s2.disk_usage();
    assert_eq!(du.per_family.get("runtime"), Some(&((v1.len() + v2.len()) as u64)), "counted until it is gone");
    s2.sweep();
    assert!(old.exists(), "still loaded: kept");
    assert!(pending_file.exists());
    drop(loaded);
    // The next start.
    let (s3, _) = new_store(&app.0, catalog_v2());
    s3.sweep();
    assert!(!old.exists(), "removed at the next start");
    assert!(!pending_file.exists());
    assert_eq!(s3.installed_path("runtime/rt").as_deref(), Some(new.as_path()));
    assert_eq!(s3.disk_usage().per_family.get("runtime"), Some(&(v2.len() as u64)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runtime_leftovers_are_replaced_in_place_and_verify_removes_a_corrupt_pack() {
    let srv = Server::start().await;
    let data = blob(15_000, 132);
    srv.set("/rt.dll", Behavior::Serve(arc(&data)));
    let app = TempDir::new("runtime-leftover");
    let (store, _) = new_store(&app.0, catalog_of(vec![runtime_pack("runtime/lo", "1", &data, &srv.url("/rt.dll"))]));
    let e = store.catalog.get("runtime/lo").unwrap().clone();
    let preferred = store.layout.preferred_content_dir(&e);
    // A folder no record points at (a crash before the record write, or a verify failure).
    fs::create_dir_all(&preferred).unwrap();
    fs::write(preferred.join("junk.dll"), b"old").unwrap();
    let root = store.ensure(&ids(&["runtime/lo"])).await.unwrap().remove(0);
    assert_eq!(root, preferred, "the leftover is replaced, not skipped with a ~ sibling");
    assert!(!root.join("junk.dll").exists());
    // A corrupt pack is unregistered and its folder removed; the reinstall lands in the same place.
    fs::write(root.join("rt.dll"), blob(15_000, 133)).unwrap();
    assert_eq!(install::verify(&store, "runtime/lo").await.unwrap().state, EntryState::Failed);
    assert!(store.installed_path("runtime/lo").is_none());
    assert!(!root.exists(), "the corrupt runtime folder is removed");
    assert_eq!(store.ensure(&ids(&["runtime/lo"])).await.unwrap().remove(0), preferred);
    assert_eq!(fs::read(preferred.join("rt.dll")).unwrap(), data);
    let siblings: Vec<_> = fs::read_dir(preferred.parent().unwrap())
        .unwrap()
        .flatten()
        .map(|d| d.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(siblings, ["cpu"]);
}

/// Set a folder's modification time (Windows needs a backup-semantics handle for folders).
#[cfg(windows)]
fn age_dir(p: &Path, t: std::time::SystemTime) {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_WRITE_ATTRIBUTES: u32 = 0x100;
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    let f = fs::OpenOptions::new()
        .access_mode(FILE_WRITE_ATTRIBUTES)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(p)
        .unwrap();
    f.set_modified(t).unwrap();
}

/// A catalog that slipped past validation with one entry folder inside another's: the startup
/// sweep of the outer entry must not delete the inner entry's install.
#[cfg(windows)]
#[test]
fn the_sweep_never_deletes_another_entrys_folder() {
    let app = TempDir::new("sweep-nested");
    let never = ["https://x.invalid/a".to_string()];
    let c: Catalog = serde_json::from_value(json!({ "entries": [
        entry("t/sherpa", "t", vec![sha_file("a.bin", b"a", &never)]),
        entry("t/sherpa/zh", "t", vec![sha_file("b.bin", b"b", &never)]),
    ]}))
    .unwrap();
    assert!(catalog::validate(&c, ValidateOpts::default()).unwrap_err().contains("inside the folder of"));
    let store = Store::for_test(&app.0, c);
    let inner = store.layout.entry_dir(store.catalog.get("t/sherpa/zh").unwrap());
    fs::create_dir_all(inner.join("1")).unwrap();
    fs::write(inner.join("1").join("b.bin"), b"b").unwrap();
    let orphan = store.layout.entry_dir(store.catalog.get("t/sherpa").unwrap()).join("0.9");
    fs::create_dir_all(&orphan).unwrap();
    let old = std::time::SystemTime::now() - Duration::from_secs(40 * 24 * 3600);
    age_dir(&inner, old);
    age_dir(&orphan, old);
    store.sweep();
    assert!(inner.join("1").join("b.bin").exists(), "another entry's folder is never swept");
    assert!(!orphan.exists(), "an old unreferenced version folder still is");
}

/// A pending-trash entry that a record points at again is forgotten, never deleted; an entry that
/// does not resolve to a store folder is dropped.
#[test]
fn pending_removals_never_touch_a_recorded_install() {
    use super::layout::{self, InstalledRecord, PendingRoot, PendingTrash, RecordedContent, RECORD_SCHEMA};
    let app = TempDir::new("pending-recorded");
    let data = blob(100, 134);
    let c = catalog_of(vec![runtime_pack("runtime/keep", "1", &data, "https://x.invalid/rt.dll")]);
    let store = Store::for_test(&app.0, c);
    let e = store.catalog.get("runtime/keep").unwrap().clone();
    let root = store.layout.preferred_content_dir(&e);
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("rt.dll"), &data).unwrap();
    let record = InstalledRecord {
        schema: RECORD_SCHEMA,
        id: e.id.clone(),
        family: e.family.clone(),
        kind: e.kind,
        version: e.version.clone(),
        root: "runtimes/keep/1/cpu".into(),
        files: vec![],
        contents: vec![RecordedContent {
            path: "rt.dll".into(),
            size: data.len() as u64,
            sha256: Some(sha256_hex(&data)),
            git_oid: None,
        }],
        origin: Origin::Import,
        pinned: true,
        installed_at_unix_ms: 0,
    };
    layout::write_record(&store.layout.record_path(&e), &record).unwrap();
    let outside = app.0.join("outside");
    fs::create_dir_all(&outside).unwrap();
    let pending = PendingTrash {
        roots: vec![
            PendingRoot {
                root: "runtimes/keep/1/cpu".into(),
                family: "runtime".into(),
            },
            PendingRoot {
                root: "outside".into(),
                family: "runtime".into(),
            },
            PendingRoot {
                root: "../escape".into(),
                family: "runtime".into(),
            },
        ],
    };
    layout::write_pending_trash(&store.layout, &pending).unwrap();
    store.sweep();
    assert!(root.join("rt.dll").exists(), "recorded again: kept");
    assert!(outside.exists(), "not a store folder: never touched");
    assert_eq!(store.installed_path("runtime/keep").as_deref(), Some(root.as_path()));
    assert!(!store.layout.pending_trash_path().exists(), "every entry is dropped from the list");
}

#[test]
fn whisper_catalog_matches_the_whisper_plugin() {
    let c = Catalog::embedded().unwrap();
    let whisper: Vec<_> = c.entries.iter().filter(|e| e.family == "whisper").collect();
    assert_eq!(whisper.len(), 29, "every downloadable Whisper model of whisper.rs");
    for e in whisper {
        let id = e.id.strip_prefix("whisper/").unwrap();
        assert_eq!(legacy::whisper_entry_id(id), e.id);
        let info = crate::services::whisper::get_model_info(id).unwrap_or_else(|err| panic!("{}: {err}", e.id));
        assert!(!info.bring_your_own);
        assert_eq!(e.files.len(), 1);
        let f = &e.files[0];
        assert_eq!(f.path, info.filename, "{}", e.id);
        assert!(f.sha256.is_some() && f.size > 1_000_000, "{} is pinned", e.id);
        assert!(f.sources[0].url.ends_with(&format!("/{}", info.filename)));
        assert!(f.sources[0].url.starts_with("https://huggingface.co/ggerganov/whisper.cpp/resolve/"));
    }
}

#[test]
fn every_published_embedded_entry_is_pinned() {
    for e in &Catalog::embedded().unwrap().entries {
        if e.status == super::PublishStatus::Published {
            for f in &e.files {
                assert!(f.sha256.is_some() || f.git_oid.is_some(), "{} {}", e.id, f.path);
            }
        }
    }
}

#[test]
fn transient_states_are_reported_over_installed_ones() {
    let app = TempDir::new("transient");
    let c = catalog_of(vec![entry("t/z", "t", vec![sha_file("z", b"z", &["https://x.invalid/z".to_string()])])]);
    let store = Store::for_test(&app.0, c);
    store.set_state("t/z", Some((EntryState::Failed, Some("boom".into()))));
    let st = store.status("t/z").unwrap();
    assert_eq!((st.state, st.error.as_deref()), (EntryState::Failed, Some("boom")));
    assert_eq!(store.transient_state("t/z"), Some(EntryState::Failed));
    store.set_state("t/z", None);
    assert_eq!(store.status("t/z").unwrap().state, EntryState::NotInstalled);
}
