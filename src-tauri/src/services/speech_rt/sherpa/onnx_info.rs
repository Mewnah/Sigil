//! Reads the few ONNX graph facts the wrapper needs before handing a model to sherpa-onnx: the
//! graph's input names, how many outputs it has, and the model's metadata properties.
//!
//! - sherpa-onnx 1.13.8 picks its streaming transducer implementation by the decoder's output count
//!   (`OnlineRecognizerImpl::Create`): one output → the icefall / k2 transducer (greedy and modified
//!   beam search, hotwords); more → the NeMo transducer (Nemotron, cache-aware FastConformer), which
//!   decodes greedily only and calls `exit(-1)` on `modified_beam_search` or a hotwords stream. The
//!   wrapper counts the outputs the same way first, so such a config is an `Err` instead of a dead
//!   process.
//! - A NeMo streaming encoder with a `prompt_index` input is a prompt-conditioned multilingual model
//!   (Nemotron 3.5 ASR streaming); its `prompt_dictionary` metadata (a JSON object of language tag →
//!   prompt id) lists the languages the per-stream `language` option accepts. sherpa-onnx silently
//!   falls back to auto-detection for anything else, so the engine checks the language against it.
//!
//! This is a minimal protobuf walk (`ModelProto.graph` = field 7, `ModelProto.metadata_props` = 14,
//! `GraphProto.input` = 11, `GraphProto.output` = 12, `ValueInfoProto.name` = 1,
//! `StringStringEntryProto.key` / `value` = 1 / 2, as in `onnx.proto`): it seeks past every other
//! field, so large initializers are never read.

use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

/// `ModelProto.graph`.
const MODEL_GRAPH: u64 = 7;
/// `ModelProto.metadata_props`.
const MODEL_METADATA: u64 = 14;
/// `GraphProto.input`.
const GRAPH_INPUT: u64 = 11;
/// `GraphProto.output`.
const GRAPH_OUTPUT: u64 = 12;
/// `ValueInfoProto.name` and `StringStringEntryProto.key`.
const FIELD_1: u64 = 1;
/// `StringStringEntryProto.value`.
const FIELD_2: u64 = 2;
/// Longest string (name, metadata key or value) read into memory; longer ones are skipped.
const MAX_STRING: u64 = 1 << 20;

/// What [`summary`] reads from an ONNX model.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct OnnxSummary {
    /// Graph input names, in order (including inputs that are also initializers).
    pub inputs: Vec<String>,
    /// Number of graph outputs.
    pub outputs: usize,
    /// `metadata_props` key / value pairs, in file order (oversized values are left out).
    pub metadata: Vec<(String, String)>,
}

impl OnnxSummary {
    /// The first metadata value for `key`.
    pub fn meta(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub fn has_input(&self, name: &str) -> bool {
        self.inputs.iter().any(|i| i == name)
    }
}

/// Number of graph outputs of the ONNX model at `path`.
pub(crate) fn graph_output_count(path: &Path) -> Result<usize, String> {
    summary(path).map(|s| s.outputs)
}

/// Graph inputs, output count and metadata of the ONNX model at `path` (one pass, no initializer
/// data read).
pub(crate) fn summary(path: &Path) -> Result<OnnxSummary, String> {
    let file = File::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let len = file
        .metadata()
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?
        .len();
    let mut r = ProtoReader {
        inner: BufReader::new(file),
        pos: 0,
        len,
    };
    r.model_summary()
        .map_err(|e| format!("{} is not a readable ONNX model ({e})", path.display()))
}

struct ProtoReader<R> {
    inner: R,
    pos: u64,
    len: u64,
}

impl<R: Read + Seek> ProtoReader<R> {
    fn model_summary(&mut self) -> Result<OnnxSummary, String> {
        let mut out = OnnxSummary::default();
        let mut graph_seen = false;
        while self.pos < self.len {
            let (field, wire) = self.key()?;
            match (field, wire) {
                (MODEL_GRAPH, 2) => {
                    let end = self.bounded_end(self.len)?;
                    self.graph(end, &mut out)?;
                    graph_seen = true;
                }
                (MODEL_METADATA, 2) => {
                    let end = self.bounded_end(self.len)?;
                    if let Some(kv) = self.string_pair(end)? {
                        out.metadata.push(kv);
                    }
                }
                _ => self.skip_value(wire, self.len)?,
            }
        }
        if !graph_seen {
            return Err("no graph".into());
        }
        Ok(out)
    }

    fn graph(&mut self, end: u64, out: &mut OnnxSummary) -> Result<(), String> {
        while self.pos < end {
            let (f, w) = self.key()?;
            match (f, w) {
                (GRAPH_INPUT, 2) => {
                    let value_end = self.bounded_end(end)?;
                    if let Some(name) = self.first_string(value_end)? {
                        out.inputs.push(name);
                    }
                }
                (GRAPH_OUTPUT, 2) => {
                    out.outputs += 1;
                    self.skip_value(w, end)?;
                }
                _ => self.skip_value(w, end)?,
            }
        }
        if self.pos != end {
            return Err("graph overruns its length".into());
        }
        Ok(())
    }

    /// Field 1 (a string) of the message ending at `end`; the rest of the message is skipped.
    fn first_string(&mut self, end: u64) -> Result<Option<String>, String> {
        let mut name = None;
        while self.pos < end {
            let (f, w) = self.key()?;
            if f == FIELD_1 && w == 2 && name.is_none() {
                name = self.string(end)?;
            } else {
                self.skip_value(w, end)?;
            }
        }
        if self.pos != end {
            return Err("a value overruns its message".into());
        }
        Ok(name)
    }

    /// `(key, value)` of a `StringStringEntryProto` ending at `end` (`None` when either is missing
    /// or too long).
    fn string_pair(&mut self, end: u64) -> Result<Option<(String, String)>, String> {
        let (mut key, mut value) = (None, None);
        while self.pos < end {
            let (f, w) = self.key()?;
            match (f, w) {
                (FIELD_1, 2) => key = self.string(end)?,
                (FIELD_2, 2) => value = self.string(end)?,
                _ => self.skip_value(w, end)?,
            }
        }
        if self.pos != end {
            return Err("metadata overruns its message".into());
        }
        Ok(key.zip(value))
    }

    /// A length-delimited UTF-8 string ending by `limit`; `None` (skipped) when longer than
    /// [`MAX_STRING`].
    fn string(&mut self, limit: u64) -> Result<Option<String>, String> {
        let n = self.varint()?;
        if n > MAX_STRING {
            self.skip(n, limit)?;
            return Ok(None);
        }
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= limit)
            .ok_or_else(|| "a string overruns its message".to_string())?;
        let mut buf = vec![0u8; n as usize];
        self.inner
            .read_exact(&mut buf)
            .map_err(|e| format!("read failed: {e}"))?;
        self.pos = end;
        Ok(Some(String::from_utf8_lossy(&buf).into_owned()))
    }

    fn byte(&mut self) -> Result<u8, String> {
        if self.pos >= self.len {
            return Err("truncated".into());
        }
        let mut b = [0u8; 1];
        self.inner
            .read_exact(&mut b)
            .map_err(|e| format!("read failed: {e}"))?;
        self.pos += 1;
        Ok(b[0])
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        for i in 0..10 {
            let b = self.byte()?;
            v |= u64::from(b & 0x7f) << (7 * i);
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err("malformed varint".into())
    }

    /// `(field number, wire type)` of the next field.
    fn key(&mut self) -> Result<(u64, u8), String> {
        let k = self.varint()?;
        let field = k >> 3;
        if field == 0 {
            return Err("field number 0".into());
        }
        Ok((field, (k & 7) as u8))
    }

    /// Reads a length prefix and returns where that value ends (never past `limit`).
    fn bounded_end(&mut self, limit: u64) -> Result<u64, String> {
        let n = self.varint()?;
        self.pos
            .checked_add(n)
            .filter(|&end| end <= limit)
            .ok_or_else(|| "a field overruns its message".to_string())
    }

    fn skip(&mut self, n: u64, limit: u64) -> Result<(), String> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= limit)
            .ok_or_else(|| "a field overruns its message".to_string())?;
        let step = i64::try_from(n).map_err(|_| "a field is too long".to_string())?;
        self.inner
            .seek_relative(step)
            .map_err(|e| format!("seek failed: {e}"))?;
        self.pos = end;
        Ok(())
    }

    /// Skip the value of a field of wire type `wire` that must end by `limit`.
    fn skip_value(&mut self, wire: u8, limit: u64) -> Result<(), String> {
        match wire {
            0 => self.varint().map(drop),
            1 => self.skip(8, limit),
            2 => {
                let n = self.varint()?;
                self.skip(n, limit)
            }
            5 => self.skip(4, limit),
            other => Err(format!("unsupported wire type {other}")),
        }
    }
}
