//! Download of one file: HTTP Range resume into a stable `<key>.partial`, a stall timeout instead
//! of a total timeout, a streaming hash, and failover across mirrors.
//!
//! Rules (plan "Model and runtime store"):
//! - Next mirror on connect errors, 403/404/429/5xx, an HTML page instead of the file, a size that
//!   disagrees with the catalog, or a hash mismatch.
//! - Same mirror again (with Range, after a short backoff) when a transfer stalls or drops
//!   mid-body: it was working, so resuming is cheaper than switching. Only attempts that added no
//!   bytes count toward `attempts_per_mirror`; an attempt that made progress resets that count, and
//!   those resumes have their own, much larger cap (`max_resumes_per_mirror`).
//! - Bytes from one mirror are continued on another only when the file is hash-pinned (the final
//!   hash proves the result). Unpinned files resume only from the same URL, with `If-Range`.
//! - A hash mismatch deletes the partial; a cancel keeps it (plus its `.partial.json`) for resume.
//! - The partial is opened without write sharing, so a second Sigil process (the app and a
//!   `--headless fetch`) cannot write the same file at once.
//! - Nothing here decides where the file ends up: the caller renames the verified partial.

use super::events::{human_bytes, ProgressMeter};
use super::layout::describe_io;
use super::mirrors::{host_key, ResolvedSource};
use super::queue::CancelToken;
use super::store::Store;
use super::verify::{hash_reader, Digest, Expected, FileHasher, HashError};
use super::{ProgressEvent, SourceKind};
use futures::StreamExt;
use reqwest::header::{ACCEPT_ENCODING, CONTENT_RANGE, CONTENT_TYPE, ETAG, IF_RANGE, LAST_MODIFIED, RANGE};
use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// What to download.
pub(crate) struct DownloadRequest<'a> {
    /// Entry id (progress events).
    pub entry_id: &'a str,
    /// Catalog file path (progress events and messages).
    pub file_label: &'a str,
    pub expected: &'a Expected,
    /// Expected byte size, when known.
    pub size: Option<u64>,
    /// In order of preference.
    pub sources: &'a [ResolvedSource],
    /// Stable partial file (`models/.downloads/<key>.partial`).
    pub partial: PathBuf,
    /// Continue bytes from one mirror on another (only safe when `expected` is pinned).
    pub cross_mirror_resume: bool,
    /// The file itself is HTML (skip the "web page instead of the file" check).
    pub allow_html: bool,
}

/// A complete, verified download (still at `path`, the partial).
#[derive(Debug)]
pub(crate) struct Downloaded {
    pub path: PathBuf,
    pub size: u64,
    pub digest: Digest,
    pub mirror: SourceKind,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DownloadError {
    Cancelled,
    Failed(String),
}

enum Step {
    Complete,
    Retry(String),
    NextMirror(String),
    Fatal(String),
    Cancelled,
}

/// `<key>.partial.json`: where the bytes came from, for `If-Range` and same-URL resume.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct PartialMeta {
    url: String,
    #[serde(default)]
    etag: Option<String>,
    #[serde(default)]
    last_modified: Option<String>,
}

/// The metadata file next to a partial.
pub(crate) fn meta_path(partial: &Path) -> PathBuf {
    let mut name = partial.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".json");
    partial.with_file_name(name)
}

/// Remove a partial and its metadata.
pub(crate) fn remove_partial(partial: &Path) {
    let _ = fs::remove_file(partial);
    let _ = fs::remove_file(meta_path(partial));
}

fn read_meta(partial: &Path) -> PartialMeta {
    fs::read(meta_path(partial))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

fn write_meta(partial: &Path, meta: &PartialMeta) {
    if let Ok(b) = serde_json::to_vec(meta) {
        let _ = super::layout::write_atomic(&meta_path(partial), &b);
    }
}

/// Open (or create) a partial for exclusive writing (others may only read).
pub(crate) fn open_partial(path: &Path) -> io::Result<File> {
    let mut o = OpenOptions::new();
    o.read(true).write(true).create(true).truncate(false);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_SHARE_READ: u32 = 0x1;
        o.share_mode(FILE_SHARE_READ);
    }
    o.open(path)
}

/// The partial file plus a hasher that has consumed exactly its `have` bytes (when `Some`).
struct PartialFile {
    file: File,
    have: u64,
    hasher: Option<FileHasher>,
}

impl PartialFile {
    fn truncate(&mut self) -> Result<(), String> {
        self.file.set_len(0).map_err(|e| describe_io("reset the partial download", &e))?;
        self.file.seek(SeekFrom::Start(0)).map_err(|e| describe_io("reset the partial download", &e))?;
        self.have = 0;
        self.hasher = None;
        Ok(())
    }

    fn append(&mut self, data: &[u8]) -> io::Result<()> {
        self.file.write_all(data)?;
        if let Some(h) = self.hasher.as_mut() {
            h.update(data);
        }
        self.have += data.len() as u64;
        Ok(())
    }
}

fn error_chain(err: &(dyn StdError + 'static)) -> String {
    let mut cur: &(dyn StdError + 'static) = err;
    while let Some(next) = cur.source() {
        cur = next;
    }
    cur.to_string()
}

fn kind_name(kind: SourceKind) -> String {
    serde_json::to_value(kind)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| format!("{kind:?}"))
}

/// `bytes <start>-<end>/<total|*>` or `bytes */<total>` → (start, total).
fn parse_content_range(v: &str) -> Option<(Option<u64>, Option<u64>)> {
    let rest = v.trim().strip_prefix("bytes")?.trim_start();
    let (range, total) = rest.split_once('/')?;
    let total = match total.trim() {
        "*" => None,
        t => Some(t.parse().ok()?),
    };
    let start = match range.trim() {
        "*" => None,
        r => Some(r.split_once('-')?.0.trim().parse().ok()?),
    };
    Some((start, total))
}

/// `Some(true)` for an HTML page, `Some(false)` when clearly not, `None` while undecided.
pub(crate) fn sniff_html(prefix: &[u8]) -> Option<bool> {
    let mut p = prefix;
    if p.starts_with(&[0xEF, 0xBB, 0xBF]) {
        p = &p[3..];
    }
    let start = p.iter().position(|b| !b.is_ascii_whitespace())?;
    let p = &p[start..];
    let mut undecided = false;
    for needle in [&b"<!doctype html"[..], b"<html", b"<head", b"<body"] {
        let n = needle.len().min(p.len());
        if p[..n].eq_ignore_ascii_case(&needle[..n]) {
            if n == needle.len() {
                return Some(true);
            }
            undecided = true;
        }
    }
    (!undecided).then_some(false)
}

/// Make `pf.hasher` cover the bytes already on disk (hashing the kept prefix on a worker thread).
async fn ensure_hasher(pf: &mut PartialFile, req: &DownloadRequest<'_>, cancel: &CancelToken) -> Result<(), Step> {
    if pf.hasher.is_some() {
        return Ok(());
    }
    let size_for_header = req.size.unwrap_or(pf.have);
    let mut hasher = FileHasher::new(req.expected, size_for_header);
    if pf.have > 0 {
        let mut file = pf.file.try_clone().map_err(|e| Step::Fatal(describe_io("read the partial download", &e)))?;
        let have = pf.have;
        let c = cancel.clone();
        let res = tokio::task::spawn_blocking(move || {
            file.seek(SeekFrom::Start(0))?;
            let n = match hash_reader(&mut file, Some(have), &mut hasher, &|| c.is_cancelled()) {
                Ok(n) => n,
                Err(HashError::Cancelled) => return Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled")),
                Err(HashError::Io(e)) => return Err(e),
            };
            if n != have {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "the partial download shrank"));
            }
            Ok(hasher)
        })
        .await
        .map_err(|e| Step::Fatal(format!("hash task failed: {e}")))?;
        match res {
            Ok(h) => hasher = h,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => return Err(Step::Cancelled),
            Err(_) => {
                pf.truncate().map_err(Step::Fatal)?;
                hasher = FileHasher::new(req.expected, req.size.unwrap_or(0));
            }
        }
    }
    pf.file
        .seek(SeekFrom::Start(pf.have))
        .map_err(|e| Step::Fatal(describe_io("seek in the partial download", &e)))?;
    pf.hasher = Some(hasher);
    Ok(())
}

/// Download `req`, trying its sources in order.
pub(crate) async fn download(store: &Store, req: &DownloadRequest<'_>, cancel: &CancelToken) -> Result<Downloaded, DownloadError> {
    if req.sources.is_empty() {
        return Err(DownloadError::Failed(format!("{} has no download source", req.file_label)));
    }
    if let Some(dir) = req.partial.parent() {
        fs::create_dir_all(dir).map_err(|e| DownloadError::Failed(describe_io("create the downloads folder", &e)))?;
    }
    let file = open_partial(&req.partial).map_err(|e| {
        DownloadError::Failed(match e.raw_os_error() {
            Some(32) | Some(33) => format!("{} is being downloaded by another Sigil process", req.file_label),
            _ => describe_io("open the partial download", &e),
        })
    })?;
    let have = file
        .metadata()
        .map_err(|e| DownloadError::Failed(describe_io("read the partial download", &e)))?
        .len();
    let mut pf = PartialFile { file, have, hasher: None };
    let result = run(store, req, &mut pf, cancel).await;
    let empty = pf.have == 0;
    drop(pf);
    if result.is_err() && empty {
        remove_partial(&req.partial);
    } else if result.is_ok() {
        let _ = fs::remove_file(meta_path(&req.partial));
    }
    result
}

async fn run(store: &Store, req: &DownloadRequest<'_>, pf: &mut PartialFile, cancel: &CancelToken) -> Result<Downloaded, DownloadError> {
    let settings = &store.settings;
    let mut meta = read_meta(&req.partial);
    if req.size.is_some_and(|size| pf.have > size) {
        pf.truncate().map_err(DownloadError::Failed)?;
    }
    let mut attempt: u32 = 0;
    let mut failures: Vec<String> = Vec::new();
    for src in req.sources {
        // Attempts on this mirror since the last one that made progress (counted against
        // `attempts_per_mirror`), and resumes after a drop or stall that did make progress.
        let mut tries: u32 = 0;
        let mut resumes: u32 = 0;
        loop {
            if cancel.is_cancelled() {
                return Err(DownloadError::Cancelled);
            }
            tries += 1;
            attempt += 1;
            if pf.have > 0 && !req.cross_mirror_resume && meta.url != src.url.as_str() {
                pf.truncate().map_err(DownloadError::Failed)?;
            }
            let resumed_from = pf.have;
            let complete_on_disk = req.size.is_some_and(|s| s > 0 && s == pf.have);
            let step = if complete_on_disk {
                Step::Complete
            } else {
                fetch_once(store, req, src, pf, &mut meta, cancel, attempt).await
            };
            let step = match step {
                Step::Complete => match finalize(req, pf, cancel).await {
                    Ok(digest) => {
                        return Ok(Downloaded {
                            path: req.partial.clone(),
                            size: pf.have,
                            digest,
                            mirror: src.kind,
                        })
                    }
                    Err(Step::NextMirror(msg)) => {
                        // Corrupt: never keep these bytes.
                        pf.truncate().map_err(DownloadError::Failed)?;
                        let _ = fs::remove_file(meta_path(&req.partial));
                        tracing::warn!(target: "sigil::models", "{}: {} ({})", req.entry_id, msg, src.url);
                        if resumed_from > 0 && tries < settings.attempts_per_mirror {
                            // The kept prefix may have been the bad part: once more from zero.
                            continue;
                        }
                        Step::NextMirror(msg)
                    }
                    Err(other) => other,
                },
                other => other,
            };
            match step {
                Step::Complete => unreachable!("handled above"),
                Step::Cancelled => return Err(DownloadError::Cancelled),
                Step::Fatal(msg) => return Err(DownloadError::Failed(msg)),
                Step::NextMirror(msg) => {
                    tracing::info!(target: "sigil::models", "{} {}: {} failed: {msg}", req.entry_id, req.file_label, kind_name(src.kind));
                    failures.push(format!("{}: {msg}", kind_name(src.kind)));
                    break;
                }
                Step::Retry(msg) => {
                    tracing::info!(target: "sigil::models", "{} {}: {} attempt {attempt}: {msg}", req.entry_id, req.file_label, kind_name(src.kind));
                    if pf.have > resumed_from {
                        // It was working (bytes arrived before the drop or stall): resume on the
                        // same mirror without spending an attempt; only resumes are capped.
                        resumes += 1;
                        tries = 0;
                        if resumes > settings.max_resumes_per_mirror {
                            failures.push(format!(
                                "{}: {msg} (gave up after {} resumes)",
                                kind_name(src.kind),
                                settings.max_resumes_per_mirror
                            ));
                            break;
                        }
                    } else if tries >= settings.attempts_per_mirror {
                        failures.push(format!("{}: {msg}", kind_name(src.kind)));
                        break;
                    }
                    let wait = settings
                        .retry_backoff
                        .get((tries as usize).saturating_sub(1))
                        .or(settings.retry_backoff.last())
                        .copied()
                        .unwrap_or_default();
                    tokio::select! {
                        _ = tokio::time::sleep(wait) => {}
                        _ = cancel.cancelled() => return Err(DownloadError::Cancelled),
                    }
                }
            }
        }
    }
    Err(DownloadError::Failed(format!(
        "could not download {} ({})",
        req.file_label,
        failures.join("; ")
    )))
}

/// Size and hash check of a complete body. `Err(Step::NextMirror)` means the bytes are wrong.
async fn finalize(req: &DownloadRequest<'_>, pf: &mut PartialFile, cancel: &CancelToken) -> Result<Digest, Step> {
    ensure_hasher(pf, req, cancel).await?;
    if let Some(size) = req.size {
        if pf.have != size {
            return Err(Step::NextMirror(format!("got {} bytes, the catalog expects {size}", pf.have)));
        }
    }
    pf.file
        .sync_all()
        .map_err(|e| Step::Fatal(describe_io("flush the download", &e)))?;
    let digest = pf.hasher.take().expect("ensure_hasher set it").finish();
    if !req.expected.matches(&digest) {
        return Err(Step::NextMirror(format!(
            "the file does not match the catalog ({} expected, got {})",
            req.expected.describe(),
            digest.describe()
        )));
    }
    Ok(digest)
}

#[allow(clippy::too_many_arguments)]
async fn fetch_once(
    store: &Store,
    req: &DownloadRequest<'_>,
    src: &ResolvedSource,
    pf: &mut PartialFile,
    meta: &mut PartialMeta,
    cancel: &CancelToken,
    attempt: u32,
) -> Step {
    let settings = &store.settings;
    // Hash the kept prefix before connecting (an idle connection would time out meanwhile).
    if let Err(step) = ensure_hasher(pf, req, cancel).await {
        return step;
    }
    let lock = store.queue.host_lock(&host_key(&src.url));
    let _host = tokio::select! {
        g = lock.lock() => g,
        _ = cancel.cancelled() => return Step::Cancelled,
    };
    let start = pf.have;
    let mut rb = store.client.get(src.url.clone()).header(ACCEPT_ENCODING, "identity");
    if start > 0 {
        rb = rb.header(RANGE, format!("bytes={start}-"));
        if !req.cross_mirror_resume {
            let validator = meta
                .etag
                .as_ref()
                .filter(|e| !e.starts_with("W/"))
                .or(meta.last_modified.as_ref());
            if let Some(v) = validator {
                rb = rb.header(IF_RANGE, v.as_str());
            }
        }
    }
    let sent = tokio::select! {
        r = tokio::time::timeout(settings.header_timeout, rb.send()) => r,
        _ = cancel.cancelled() => return Step::Cancelled,
    };
    let resp = match sent {
        Err(_) => {
            return Step::Retry(format!(
                "no response within {} s",
                settings.header_timeout.as_secs().max(1)
            ))
        }
        Ok(Err(e)) => {
            let cause = error_chain(&e);
            return if e.is_connect() || e.is_redirect() {
                Step::NextMirror(format!("cannot connect ({cause})"))
            } else if e.is_timeout() {
                Step::Retry(format!("timed out ({cause})"))
            } else {
                Step::Retry(cause)
            };
        }
        Ok(Ok(r)) => r,
    };
    let status = resp.status().as_u16();
    let header = |name: reqwest::header::HeaderName| {
        resp.headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
    };
    let content_range = header(CONTENT_RANGE);
    let content_type = header(CONTENT_TYPE).unwrap_or_default().to_ascii_lowercase();
    let etag = header(ETAG);
    let last_modified = header(LAST_MODIFIED);
    let total: Option<u64> = match status {
        206 => match content_range.as_deref().and_then(parse_content_range) {
            Some((Some(s), t)) if s == start => t,
            _ => {
                if let Err(e) = pf.truncate() {
                    return Step::Fatal(e);
                }
                return Step::Retry("the server resumed at the wrong position".into());
            }
        },
        200 => {
            if start > 0 {
                if let Err(e) = pf.truncate() {
                    return Step::Fatal(e);
                }
            }
            resp.content_length()
        }
        416 => {
            let t = content_range.as_deref().and_then(parse_content_range).and_then(|(_, t)| t);
            if start > 0 && (req.size == Some(start) || t == Some(start)) {
                return Step::Complete;
            }
            if let Err(e) = pf.truncate() {
                return Step::Fatal(e);
            }
            return Step::Retry("the server refused the resume position".into());
        }
        429 => return Step::NextMirror("HTTP 429 (rate limited)".into()),
        400..=599 => {
            let detail = crate::services::http_client::describe_failed_response(resp).await;
            return Step::NextMirror(detail);
        }
        other => return Step::NextMirror(format!("unexpected HTTP {other}")),
    };
    let start = pf.have;
    if let (Some(want), Some(t)) = (req.size, total) {
        if want != t {
            if let Err(e) = pf.truncate() {
                return Step::Fatal(e);
            }
            return Step::NextMirror(format!(
                "the server has {t} bytes but the catalog expects {want} (a different version?)"
            ));
        }
    }
    if !req.allow_html && content_type.starts_with("text/html") {
        return Step::NextMirror("served a web page instead of the file (blocked, rate-limited or moved?)".into());
    }
    *meta = PartialMeta {
        url: src.url.to_string(),
        etag,
        last_modified,
    };
    write_meta(&req.partial, meta);
    if start == 0 && pf.hasher.is_none() {
        pf.hasher = Some(FileHasher::new(req.expected, req.size.or(total).unwrap_or(0)));
    }

    let bytes_total = req.size.or(total);
    let emit = |done: u64, speed: f64| {
        store.emit_progress(ProgressEvent {
            id: req.entry_id.to_string(),
            file: req.file_label.to_string(),
            bytes_done: done,
            bytes_total,
            bytes_per_sec: speed,
            mirror: src.kind,
            attempt,
        })
    };
    let mut meter = ProgressMeter::with_interval(pf.have, settings.progress_interval);
    if let Some(speed) = meter.tick(pf.have, true) {
        emit(pf.have, speed);
    }
    let mut sniffing = start == 0 && !req.allow_html;
    let mut sniff: Vec<u8> = Vec::new();
    let mut stream = resp.bytes_stream();
    loop {
        let next = tokio::select! {
            n = tokio::time::timeout(settings.stall_timeout, stream.next()) => n,
            _ = cancel.cancelled() => {
                let _ = pf.file.flush();
                return Step::Cancelled;
            }
        };
        let chunk = match next {
            Err(_) => {
                return Step::Retry(format!(
                    "stalled: no data for {} s after {}",
                    settings.stall_timeout.as_secs().max(1),
                    human_bytes(pf.have)
                ))
            }
            Ok(None) => break,
            Ok(Some(Err(e))) => {
                return Step::Retry(format!("connection lost after {}: {}", human_bytes(pf.have), error_chain(&e)))
            }
            Ok(Some(Ok(c))) => c,
        };
        if sniffing {
            let take = chunk.len().min(512usize.saturating_sub(sniff.len()));
            sniff.extend_from_slice(&chunk[..take]);
            match sniff_html(&sniff) {
                Some(true) => {
                    if let Err(e) = pf.truncate() {
                        return Step::Fatal(e);
                    }
                    return Step::NextMirror("served a web page instead of the file (blocked, rate-limited or moved?)".into());
                }
                Some(false) => sniffing = false,
                None if sniff.len() >= 512 => sniffing = false,
                None => {}
            }
        }
        if let Some(want) = req.size {
            if pf.have + chunk.len() as u64 > want {
                if let Err(e) = pf.truncate() {
                    return Step::Fatal(e);
                }
                return Step::NextMirror(format!("the server sent more than the {want} bytes the catalog expects"));
            }
        }
        if let Err(e) = pf.append(&chunk) {
            // Bytes and hash may now disagree: start over next time.
            let _ = pf.truncate();
            return Step::Fatal(describe_io(&format!("write {}", req.file_label), &e));
        }
        if let Some(speed) = meter.tick(pf.have, false) {
            emit(pf.have, speed);
        }
    }
    if let Some(speed) = meter.tick(pf.have, true) {
        emit(pf.have, speed);
    }
    if let Some(want) = bytes_total {
        if pf.have < want {
            return Step::Retry(format!(
                "the connection closed after {} of {}",
                human_bytes(pf.have),
                human_bytes(want)
            ));
        }
    }
    Step::Complete
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_range_parses() {
        assert_eq!(parse_content_range("bytes 100-199/200"), Some((Some(100), Some(200))));
        assert_eq!(parse_content_range("bytes 0-0/*"), Some((Some(0), None)));
        assert_eq!(parse_content_range("bytes */5000"), Some((None, Some(5000))));
        assert_eq!(parse_content_range("items 1-2/3"), None);
        assert_eq!(parse_content_range("bytes x-1/2"), None);
    }

    #[test]
    fn html_is_sniffed_with_bom_whitespace_and_case() {
        assert_eq!(sniff_html(b"<!DOCTYPE html><html>"), Some(true));
        assert_eq!(sniff_html(b"\xEF\xBB\xBF\n  <HTML lang=en>"), Some(true));
        assert_eq!(sniff_html(b"<!doc"), None);
        assert_eq!(sniff_html(b"   "), None);
        assert_eq!(sniff_html(b"GGUF\x03\x00"), Some(false));
        assert_eq!(sniff_html(b"<?xml version"), Some(false));
        assert_eq!(sniff_html(b"PK\x03\x04"), Some(false));
    }

    #[test]
    fn meta_sits_next_to_the_partial() {
        let p = Path::new("C:/m/.downloads/abc.partial");
        assert_eq!(meta_path(p), Path::new("C:/m/.downloads/abc.partial.json"));
    }
}
