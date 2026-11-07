//! The speech-token embedding of the language model's decode steps, read from the external weights
//! of `token_embedding.onnx`.
//!
//! The exported embedding graph decides per position whether a token is text or speech: everything
//! up to the first `0` in a row is text when the row has one. That is right for the prompt, whose
//! text ends with `[STOP]` = 0, but a decode step's row is the one speech token just sampled, and
//! speech token 0 is a real audio token (S3 codes 0-6560): the graph would embed it as text. So the
//! decode steps compute `speech.weight[token] + speech_position.weight[step]` here, the exact sum
//! the reference implementation (and the graph, for every other token) uses, and the graph is only
//! run for the prompt. Reading the two tables also saves one ONNX Runtime call per token.
//!
//! The tables are found by name in the graph's initializers (`*speech.weight`,
//! `*speech_position.weight`, float, external data in a file next to the graph) with a minimal
//! protobuf walk (`ModelProto.graph` = 7, `GraphProto.initializer` = 5, `TensorProto.dims` = 1,
//! `data_type` = 2, `name` = 8, `external_data` = 13 (`key` = 1, `value` = 2), `data_location` = 14,
//! as in `onnx.proto`). The model checks the result against the graph once at load.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Largest graph file read whole (the embedding graph is about 6 KB; its weights are external).
const MAX_GRAPH_BYTES: u64 = 64 << 20;
/// `TensorProto.DataType.FLOAT`.
const FLOAT: i64 = 1;

/// One initializer's metadata.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Initializer {
    pub name: String,
    pub dims: Vec<i64>,
    pub data_type: i64,
    /// `data_location == EXTERNAL`.
    pub external: bool,
    /// `external_data` entries (`location`, `offset`, `length`, ...).
    pub external_data: Vec<(String, String)>,
}

impl Initializer {
    fn ext(&self, key: &str) -> Option<&str> {
        self.external_data
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Elements of the tensor (`None` on overflow or a negative dimension).
    pub fn elements(&self) -> Option<usize> {
        self.dims
            .iter()
            .try_fold(1usize, |acc, &d| usize::try_from(d).ok().and_then(|d| acc.checked_mul(d)))
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = *self.buf.get(self.pos).ok_or("truncated varint")?;
            self.pos += 1;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err("varint too long".into())
    }

    fn bytes(&mut self) -> Result<&'a [u8], String> {
        let len = usize::try_from(self.varint()?).map_err(|_| "length overflows")?;
        let end = self
            .pos
            .checked_add(len)
            .filter(|&e| e <= self.buf.len())
            .ok_or("truncated field")?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    /// Next field: `(number, wire type)`.
    fn key(&mut self) -> Result<(u64, u8), String> {
        let k = self.varint()?;
        Ok((k >> 3, (k & 7) as u8))
    }

    fn skip(&mut self, wire: u8) -> Result<(), String> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => {
                self.pos = self
                    .pos
                    .checked_add(8)
                    .filter(|&e| e <= self.buf.len())
                    .ok_or("truncated field")?
            }
            2 => {
                self.bytes()?;
            }
            5 => {
                self.pos = self
                    .pos
                    .checked_add(4)
                    .filter(|&e| e <= self.buf.len())
                    .ok_or("truncated field")?
            }
            w => return Err(format!("unsupported protobuf wire type {w}")),
        }
        Ok(())
    }
}

fn string(bytes: &[u8]) -> Result<String, String> {
    String::from_utf8(bytes.to_vec()).map_err(|_| "a name is not UTF-8".to_string())
}

fn parse_entry(buf: &[u8]) -> Result<(String, String), String> {
    let mut r = Reader::new(buf);
    let (mut k, mut v) = (String::new(), String::new());
    while !r.done() {
        match r.key()? {
            (1, 2) => k = string(r.bytes()?)?,
            (2, 2) => v = string(r.bytes()?)?,
            (_, w) => r.skip(w)?,
        }
    }
    Ok((k, v))
}

fn parse_tensor(buf: &[u8]) -> Result<Initializer, String> {
    let mut r = Reader::new(buf);
    let mut t = Initializer::default();
    while !r.done() {
        match r.key()? {
            (1, 0) => t.dims.push(r.varint()? as i64),
            (1, 2) => {
                let mut packed = Reader::new(r.bytes()?);
                while !packed.done() {
                    t.dims.push(packed.varint()? as i64);
                }
            }
            (2, 0) => t.data_type = r.varint()? as i64,
            (8, 2) => t.name = string(r.bytes()?)?,
            (13, 2) => t.external_data.push(parse_entry(r.bytes()?)?),
            (14, 0) => t.external = r.varint()? == 1,
            (_, w) => r.skip(w)?,
        }
    }
    Ok(t)
}

/// The initializers of an ONNX model's graph (metadata only; values are never decoded).
pub fn read_initializers(model: &[u8]) -> Result<Vec<Initializer>, String> {
    let mut r = Reader::new(model);
    let mut out = Vec::new();
    while !r.done() {
        match r.key()? {
            (7, 2) => {
                let mut g = Reader::new(r.bytes()?);
                while !g.done() {
                    match g.key()? {
                        (5, 2) => out.push(parse_tensor(g.bytes()?)?),
                        (_, w) => g.skip(w)?,
                    }
                }
            }
            (_, w) => r.skip(w)?,
        }
    }
    Ok(out)
}

/// A graph input or output declaration (`ValueInfoProto`): its name and tensor element type
/// (`TensorProto.DataType`, 0 when it is not a tensor).
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueInfo {
    pub name: String,
    pub elem_type: i64,
}

#[cfg(test)]
fn parse_value_info(buf: &[u8]) -> Result<ValueInfo, String> {
    let mut r = Reader::new(buf);
    let mut v = ValueInfo {
        name: String::new(),
        elem_type: 0,
    };
    while !r.done() {
        match r.key()? {
            (1, 2) => v.name = string(r.bytes()?)?,
            // TypeProto.tensor_type = 1 → TypeProto.Tensor.elem_type = 1.
            (2, 2) => {
                let mut t = Reader::new(r.bytes()?);
                while !t.done() {
                    match t.key()? {
                        (1, 2) => {
                            let mut tensor = Reader::new(t.bytes()?);
                            while !tensor.done() {
                                match tensor.key()? {
                                    (1, 0) => v.elem_type = tensor.varint()? as i64,
                                    (_, w) => tensor.skip(w)?,
                                }
                            }
                        }
                        (_, w) => t.skip(w)?,
                    }
                }
            }
            (_, w) => r.skip(w)?,
        }
    }
    Ok(v)
}

/// The inputs and outputs declared in a run of `GraphProto` fields (`input` = 11, `output` = 12),
/// such as the declarations cut from a graph file; other fields are skipped.
#[cfg(test)]
pub fn read_graph_io(fields: &[u8]) -> Result<(Vec<ValueInfo>, Vec<ValueInfo>), String> {
    let mut r = Reader::new(fields);
    let (mut inputs, mut outputs) = (Vec::new(), Vec::new());
    while !r.done() {
        match r.key()? {
            (11, 2) => inputs.push(parse_value_info(r.bytes()?)?),
            (12, 2) => outputs.push(parse_value_info(r.bytes()?)?),
            (_, w) => r.skip(w)?,
        }
    }
    Ok((inputs, outputs))
}

/// A float table read from a graph's external data.
#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    pub rows: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

impl Table {
    pub fn row(&self, i: usize) -> Option<&[f32]> {
        (i < self.rows).then(|| &self.data[i * self.width..(i + 1) * self.width])
    }
}

/// The external-data file of `init` next to `graph` (a plain file name or relative path without
/// `..`; never absolute).
fn external_file(graph: &Path, init: &Initializer) -> Result<PathBuf, String> {
    let location = init
        .ext("location")
        .ok_or_else(|| format!("{} has no external data location", init.name))?;
    let rel = Path::new(location);
    if rel.is_absolute()
        || rel
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err(format!("{} names an unsafe external data path {location:?}", init.name));
    }
    let dir = graph.parent().ok_or("the graph has no folder")?;
    Ok(dir.join(rel))
}

/// Read the first `max_rows` rows (all when `None`) of the 2-D float initializer `init` of `graph`.
fn read_table(graph: &Path, init: &Initializer, max_rows: Option<usize>) -> Result<Table, String> {
    if init.data_type != FLOAT || init.dims.len() != 2 {
        return Err(format!("{} is not a 2-D float table", init.name));
    }
    if !init.external {
        return Err(format!("{} is not stored as external data", init.name));
    }
    let rows = usize::try_from(init.dims[0]).map_err(|_| "bad table shape")?;
    let width = usize::try_from(init.dims[1]).map_err(|_| "bad table shape")?;
    let elements = init.elements().ok_or("bad table shape")?;
    let offset: u64 = init
        .ext("offset")
        .unwrap_or("0")
        .parse()
        .map_err(|_| format!("{} has a bad data offset", init.name))?;
    if let Some(len) = init.ext("length") {
        let len: usize = len
            .parse()
            .map_err(|_| format!("{} has a bad data length", init.name))?;
        if len != elements * 4 {
            return Err(format!("{} holds {len} bytes, not {} as its shape says", init.name, elements * 4));
        }
    }
    let take_rows = max_rows.map_or(rows, |m| m.min(rows));
    let bytes = take_rows
        .checked_mul(width)
        .and_then(|n| n.checked_mul(4))
        .ok_or("table too large")?;
    let path = external_file(graph, init)?;
    let mut f = File::open(&path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let file_len = f.metadata().map_err(|e| e.to_string())?.len();
    if offset + (elements as u64) * 4 > file_len {
        return Err(format!("{} is shorter than {} expects (reinstall the model)", path.display(), init.name));
    }
    f.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
    let mut raw = vec![0u8; bytes];
    f.read_exact(&mut raw)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let data = raw
        .as_chunks::<4>().0.iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(Table {
        rows: take_rows,
        width,
        data,
    })
}

/// `speech.weight` and `speech_position.weight` of the embedding graph (see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct SpeechEmbedding {
    tokens: Table,
    positions: Table,
}

impl SpeechEmbedding {
    /// Read the tables of the embedding graph at `graph`, keeping `max_positions` position rows.
    pub fn read(graph: &Path, max_positions: usize) -> Result<Self, String> {
        let len = std::fs::metadata(graph)
            .map_err(|e| format!("cannot read {}: {e}", graph.display()))?
            .len();
        if len > MAX_GRAPH_BYTES {
            return Err(format!("{} is too large to be the token embedding graph", graph.display()));
        }
        let bytes = std::fs::read(graph).map_err(|e| format!("cannot read {}: {e}", graph.display()))?;
        let inits = read_initializers(&bytes)?;
        let find = |suffix: &str| {
            inits
                .iter()
                .find(|i| i.name == suffix || i.name.ends_with(&format!("/{suffix}")) || i.name.ends_with(&format!(".{suffix}")))
                .ok_or_else(|| format!("the token embedding graph has no {suffix} table"))
        };
        let tokens = read_table(graph, find("speech.weight")?, None)?;
        let positions = read_table(graph, find("speech_position.weight")?, Some(max_positions))?;
        if tokens.width != positions.width || tokens.width == 0 {
            return Err("the speech embedding tables have different widths".into());
        }
        Ok(Self { tokens, positions })
    }

    pub fn width(&self) -> usize {
        self.tokens.width
    }

    pub fn max_position(&self) -> usize {
        self.positions.rows.saturating_sub(1)
    }

    /// Write the embedding of speech `token` at `position` into `out` (`width()` values).
    pub fn embed(&self, token: i64, position: usize, out: &mut [f32]) -> Result<(), String> {
        let t = usize::try_from(token)
            .ok()
            .and_then(|t| self.tokens.row(t))
            .ok_or_else(|| format!("speech token {token} is outside the vocabulary"))?;
        let p = self
            .positions
            .row(position)
            .ok_or_else(|| format!("speech position {position} is beyond the model's {} positions", self.positions.rows))?;
        if out.len() != t.len() {
            return Err("embedding width mismatch".into());
        }
        for ((o, a), b) in out.iter_mut().zip(t).zip(p) {
            *o = a + b;
        }
        Ok(())
    }
}
