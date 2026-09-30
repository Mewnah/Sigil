//! Wire format of `sigil-bergamot.exe --jsonl` (protocol 1, specified in `docs/adr/0008-bergamot.md` and
//! implemented by `scripts/runtime-packs/bergamot/helper/src/main.cpp`): one JSON object per line in each direction, UTF-8.
//!
//! Requests (stdin) all carry a client-chosen `id` (u64) and an `op`:
//! - `hello {protocol}` → `{protocol, helper, engine, threads, arch}`; the first message of every connection.
//! - `load {key, model, lex, vocabs:[src, trg], gemm}` → `{key, ms}`: absolute paths; `key` names the model for later
//!   requests (`[a-z0-9_-]{1,40}`); loading an existing key replaces it.
//! - `unload {key}` → `{key}`.
//! - `translate {keys:[k] | [k1, k2], text, priority: "final"|"interim"}` → `{text, ms}`: two keys pivot through the
//!   first model's target language. Finals are scheduled before queued interims.
//! - `cancel {target}` → `{target, cancelled}`: drops the queued request `target` (it then fails with `cancelled`);
//!   one that already runs finishes normally.
//! - `stats` → `{loaded:[keys], queued, running, threads, translations}`.
//!
//! Replies (stdout): `{"id":N,"ok":true,"result":{...}}` or `{"id":N,"ok":false,"error":{"kind","message"}}` with
//! kinds `invalid_request`, `not_loaded`, `load_failed`, `translate_failed`, `cancelled`, `busy`, `internal`. A line
//! that is not valid JSON gets `"id":null`. Replies may come in any order. EOF on stdin = quit (exit 0 after the
//! running work). Diagnostics go to stderr only (the process host logs them).

use super::super::{MtError, MtErrorKind};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::BufRead;
use std::path::Path;

/// Protocol version this client speaks.
pub const PROTOCOL: u32 = 1;
/// Longest reply line accepted (a translation of the longest allowed input is far below this).
pub const MAX_LINE_BYTES: usize = 1 << 20;

/// Priority of a translate request in the helper's queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Priority {
    Final,
    Interim,
}

impl Priority {
    fn as_str(self) -> &'static str {
        match self {
            Priority::Final => "final",
            Priority::Interim => "interim",
        }
    }
}

fn path_str(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

pub fn hello() -> Value {
    json!({ "op": "hello", "protocol": PROTOCOL })
}

pub fn load(key: &str, files: &super::catalog::ModelFiles) -> Value {
    json!({
        "op": "load",
        "key": key,
        "model": path_str(&files.model),
        "lex": files.lex.as_deref().map(path_str),
        "vocabs": [path_str(&files.vocabs[0]), path_str(&files.vocabs[1])],
        "gemm": files.gemm,
    })
}

pub fn unload(key: &str) -> Value {
    json!({ "op": "unload", "key": key })
}

pub fn translate(keys: &[&str], text: &str, priority: Priority) -> Value {
    json!({ "op": "translate", "keys": keys, "text": text, "priority": priority.as_str() })
}

pub fn cancel(target: u64) -> Value {
    json!({ "op": "cancel", "target": target })
}

#[cfg(test)]
pub fn stats() -> Value {
    json!({ "op": "stats" })
}

/// Attach the request id (serialized as one line, without the newline).
pub fn encode(id: u64, mut body: Value) -> String {
    if let Value::Object(map) = &mut body {
        map.insert("id".into(), Value::from(id));
    }
    body.to_string()
}

/// Error object of a failed reply.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct HelperError {
    pub kind: String,
    pub message: String,
}

impl HelperError {
    pub fn is(&self, kind: &str) -> bool {
        self.kind == kind
    }

    /// The `MtError` a caller sees for this helper error.
    pub fn to_mt(&self) -> MtError {
        let message = if self.message.trim().is_empty() {
            format!("the Bergamot helper failed ({})", self.kind)
        } else {
            format!("Bergamot: {}", self.message.trim())
        };
        let kind = match self.kind.as_str() {
            "cancelled" => MtErrorKind::Cancelled,
            "busy" => MtErrorKind::Busy,
            "invalid_request" => MtErrorKind::InvalidInput,
            _ => MtErrorKind::Runtime,
        };
        MtError::new(kind, message)
    }
}

/// One reply line.
#[derive(Clone, Debug, PartialEq)]
pub struct Reply {
    /// `None` for replies to unparseable lines.
    pub id: Option<u64>,
    pub outcome: Result<Value, HelperError>,
}

#[derive(Deserialize)]
struct RawReply {
    id: Option<u64>,
    ok: bool,
    #[serde(default)]
    result: Value,
    #[serde(default)]
    error: Option<HelperError>,
}

/// Parse a reply line; `None` for anything that is not a protocol reply (a stray diagnostic line).
pub fn parse_reply(line: &str) -> Option<Reply> {
    let line = line.trim();
    if !line.starts_with('{') {
        return None;
    }
    let raw: RawReply = serde_json::from_str(line).ok()?;
    let outcome = if raw.ok {
        Ok(raw.result)
    } else {
        Err(raw.error.unwrap_or_else(|| HelperError {
            kind: "internal".into(),
            message: "failed without an error message".into(),
        }))
    };
    Some(Reply { id: raw.id, outcome })
}

/// Result of `hello`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct HelloInfo {
    pub protocol: u32,
    /// `sigil-bergamot <version>`.
    pub helper: String,
    /// Upstream engine and commit.
    pub engine: String,
    pub threads: u32,
    pub arch: String,
}

/// Result of `translate`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct Translated {
    pub text: String,
    pub ms: u64,
}

/// Result of `stats`.
#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct Stats {
    pub loaded: Vec<String>,
    pub queued: u64,
    pub running: u64,
    pub threads: u32,
    pub translations: u64,
    /// Number of `load` messages handled (fake helper and real helper alike).
    pub loads: u64,
}

/// Outcome of reading one line with a size cap.
#[derive(Debug, PartialEq, Eq)]
pub enum Line {
    Text(String),
    /// Longer than [`MAX_LINE_BYTES`]: the rest of the line was skipped.
    TooLong,
    Eof,
}

/// Read one `\n`-terminated line, never buffering more than `max` bytes of it.
pub fn read_line_capped(reader: &mut impl BufRead, max: usize) -> std::io::Result<Line> {
    let mut buf: Vec<u8> = Vec::new();
    let mut too_long = false;
    loop {
        let chunk = match reader.fill_buf() {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        if chunk.is_empty() {
            return Ok(if buf.is_empty() && !too_long {
                Line::Eof
            } else if too_long {
                Line::TooLong
            } else {
                Line::Text(String::from_utf8_lossy(&buf).into_owned())
            });
        }
        let (take, done) = match chunk.iter().position(|&b| b == b'\n') {
            Some(i) => (i + 1, true),
            None => (chunk.len(), false),
        };
        if !too_long {
            let room = max.saturating_sub(buf.len());
            if take > room + usize::from(done) {
                too_long = true;
                buf.clear();
            } else {
                buf.extend_from_slice(&chunk[..take]);
            }
        }
        reader.consume(take);
        if done {
            if too_long {
                return Ok(Line::TooLong);
            }
            while matches!(buf.last(), Some(b'\n' | b'\r')) {
                buf.pop();
            }
            return Ok(Line::Text(String::from_utf8_lossy(&buf).into_owned()));
        }
    }
}
