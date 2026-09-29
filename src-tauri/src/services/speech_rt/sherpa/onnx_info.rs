//! Reads the one ONNX graph fact the wrapper needs before handing a model to sherpa-onnx: how
//! many outputs a graph has.
//!
//! sherpa-onnx 1.13.8 picks its streaming transducer implementation by the decoder's output count
//! (`OnlineRecognizerImpl::Create`): one output → the icefall / k2 transducer (greedy and modified
//! beam search, hotwords); more → the NeMo transducer (Nemotron, cache-aware FastConformer), which
//! decodes greedily only and calls `exit(-1)` on `modified_beam_search` or a hotwords stream. The
//! wrapper counts the outputs the same way first, so such a config is an `Err` instead of a dead
//! process.
//!
//! This is a minimal protobuf walk (`ModelProto.graph` = field 7, `GraphProto.output` = field 12,
//! as in `onnx.proto`): it seeks past every other field, so large initializers are never read.

use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

/// `ModelProto.graph`.
const MODEL_GRAPH: u64 = 7;
/// `GraphProto.output`.
const GRAPH_OUTPUT: u64 = 12;

/// Number of graph outputs of the ONNX model at `path`.
pub(crate) fn graph_output_count(path: &Path) -> Result<usize, String> {
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
    r.graph_output_count()
        .map_err(|e| format!("{} is not a readable ONNX model ({e})", path.display()))
}

struct ProtoReader<R> {
    inner: R,
    pos: u64,
    len: u64,
}

impl<R: Read + Seek> ProtoReader<R> {
    fn graph_output_count(&mut self) -> Result<usize, String> {
        while self.pos < self.len {
            let (field, wire) = self.key()?;
            if field == MODEL_GRAPH && wire == 2 {
                let end = self.bounded_end()?;
                let mut outputs = 0;
                while self.pos < end {
                    let (f, w) = self.key()?;
                    if f == GRAPH_OUTPUT && w == 2 {
                        outputs += 1;
                    }
                    self.skip_value(w, end)?;
                }
                if self.pos != end {
                    return Err("graph overruns its length".into());
                }
                return Ok(outputs);
            }
            self.skip_value(wire, self.len)?;
        }
        Err("no graph".into())
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

    /// Reads a length prefix and returns where that value ends (never past `self.len`).
    fn bounded_end(&mut self) -> Result<u64, String> {
        let n = self.varint()?;
        self.pos
            .checked_add(n)
            .filter(|&end| end <= self.len)
            .ok_or_else(|| "a field overruns the file".to_string())
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
