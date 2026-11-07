//! Minimal ONNX protobuf reader: graph inputs / outputs (`ValueInfoProto`, skipping initializer
//! bytes) and the initializers' metadata with their `raw_data` file offsets (`TensorProto`), so a
//! model's I/O names, symbolic dimensions and weights can be read without loading it into ONNX
//! Runtime or reading the whole file into memory.
//!
//! # Frozen by P3B-S0 (signatures; P3B-E implements)
//! - [`read_graph_info`]`(path) -> Result<GraphInfo, String>` with [`GraphInfo`], [`ValueInfo`],
//!   [`Dim`] and [`InitializerInfo`]. RMVPE reads its input / output names, the mel time
//!   `dim_param` and its export kind from it at load, and the t32 pruning reads the 3×3 kernels of
//!   the installed `rmvpe.onnx` through [`InitializerInfo::raw_data`].
//!
//! # Added by P3B-E (additive)
//! - [`read_graph_info_bytes`]: the same walk over an in-memory model (the weightless graph
//!   templates compiled into the binary).
//! - [`InitializerInfo::external_data`] (the `external_data` key / value entries of an initializer
//!   whose values live in an external file) and [`InitializerInfo::external_range`].
//! - [`GraphInfo::initializer`], [`GraphInfo::metadata`] (`ModelProto.metadata_props`),
//!   [`InitializerInfo::element_count`] and [`element_size`].
//! - [`external_weights_image`] ([`WeightsImage`]): the external-data image of a weightless graph
//!   template, filled from the same-named initializers of an installed model (copied, cut to the
//!   centre kernel row, or rounded to fp16), for `gpu::ExternalWeights`.
//!
//! # How the walk works
//! The reader streams the protobuf with a buffered reader and seeks past everything it does not
//! need: nodes, and every initializer's `raw_data` (whose file offset and length are recorded). Only
//! the graph inputs and outputs are read into memory (bounded, a few KB). A 361 MB `rmvpe.onnx`
//! is walked in a few milliseconds with about 1 MB of reads.
//!
//! Every length is checked against its enclosing message and the file, and the reader caps names,
//! counts and nesting, so a malformed or hostile file returns an error, never a panic or a huge
//! allocation. Group wire types (deprecated, never written by ONNX exporters) are refused.

use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

/// One dimension of a tensor shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    /// A fixed size (`dim_value`).
    Fixed(i64),
    /// A symbolic size (`dim_param`).
    Param(String),
    /// Neither is set.
    Unknown,
}

/// A graph input or output (`ValueInfoProto`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueInfo {
    pub name: String,
    /// ONNX `TensorProto.DataType` (1 = float, 7 = int64, 10 = float16, ...); 0 when not a tensor.
    pub elem_type: i32,
    pub dims: Vec<Dim>,
}

/// An initializer's metadata (`TensorProto`, without its data).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InitializerInfo {
    pub name: String,
    pub data_type: i32,
    pub dims: Vec<i64>,
    /// `(byte offset in the file, byte length)` of `raw_data` when the tensor stores its values
    /// inline as raw bytes; `None` for typed fields (`float_data`, ...) or external data.
    pub raw_data: Option<(u64, u64)>,
    /// The tensor's values live in an external data file (`data_location = EXTERNAL`).
    pub external: bool,
    /// P3B-E: the `external_data` entries (`location`, `offset`, `length`, `checksum`), in file
    /// order; empty for inline tensors.
    pub external_data: Vec<(String, String)>,
}

/// What [`read_graph_info`] returns.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphInfo {
    /// Graph inputs that are not initializers.
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
    pub initializers: Vec<InitializerInfo>,
    /// `ModelProto.producer_name` (exporters tag their graphs; empty when unset).
    pub producer: String,
    /// P3B-E: `ModelProto.metadata_props` (key, value), in file order.
    pub metadata: Vec<(String, String)>,
}

/// ONNX `TensorProto.DataType` values this crate reads.
pub const DT_FLOAT: i32 = 1;
#[allow(dead_code)] // Part of the reader's API (tests read int64 initializers with it).
pub const DT_INT64: i32 = 7;
pub const DT_FLOAT16: i32 = 10;

/// Bytes per element of an ONNX data type, for the fixed-size types (`None` for strings and the
/// sub-byte types).
pub fn element_size(data_type: i32) -> Option<u64> {
    Some(match data_type {
        1 | 6 | 12 => 4,       // float, int32, uint32
        2 | 3 | 9 => 1,        // uint8, int8, bool
        4 | 5 | 10 | 16 => 2,  // uint16, int16, float16, bfloat16
        7 | 11 | 13 | 14 => 8, // int64, double, uint64, complex64
        15 => 16,              // complex128
        17..=20 => 1,          // float8 variants
        _ => return None,
    })
}

impl InitializerInfo {
    /// Elements of the tensor (product of its dims; 1 for a scalar), `None` on a negative dim or
    /// an overflow.
    pub fn element_count(&self) -> Option<u64> {
        self.dims
            .iter()
            .try_fold(1u64, |acc, &d| u64::try_from(d).ok().and_then(|d| acc.checked_mul(d)))
    }

    /// `(location, offset, length)` of an external tensor (`offset` defaults to 0; `length` is
    /// `None` when not recorded). `None` for inline tensors or malformed entries.
    pub fn external_range(&self) -> Option<(&str, u64, Option<u64>)> {
        if !self.external {
            return None;
        }
        let get = |k: &str| {
            self.external_data
                .iter()
                .find(|(key, _)| key == k)
                .map(|(_, v)| v.as_str())
        };
        let location = get("location").filter(|l| !l.is_empty())?;
        let offset = match get("offset") {
            Some(v) => v.trim().parse::<u64>().ok()?,
            None => 0,
        };
        let length = match get("length") {
            Some(v) => Some(v.trim().parse::<u64>().ok()?),
            None => None,
        };
        Some((location, offset, length))
    }
}

impl GraphInfo {
    /// The initializer named `name`.
    pub fn initializer(&self, name: &str) -> Option<&InitializerInfo> {
        self.initializers.iter().find(|i| i.name == name)
    }

    /// The value of metadata key `key`.
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// Read the graph metadata of the ONNX file at `path` (see the module docs).
pub fn read_graph_info(path: &Path) -> Result<GraphInfo, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let len = file
        .metadata()
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    walk(BufReader::with_capacity(64 * 1024, file), len).map_err(|e| format!("{}: not a readable ONNX model ({e})", path.display()))
}

/// [`read_graph_info`] over an in-memory model.
pub fn read_graph_info_bytes(bytes: &[u8]) -> Result<GraphInfo, String> {
    walk(BufReader::new(Cursor::new(bytes)), bytes.len() as u64).map_err(|e| format!("not a readable ONNX model ({e})"))
}

// ---------------------------------------------------------------------------------------------
// Limits (a malformed file errors out instead of allocating or looping without bound)
// ---------------------------------------------------------------------------------------------

/// Longest name / string read into memory.
const MAX_STRING: u64 = 64 * 1024;
/// Largest graph input / output (`ValueInfoProto`) read into memory.
const MAX_VALUE_INFO: u64 = 1 << 20;
/// Most graph inputs, outputs, metadata entries and external-data entries per tensor.
const MAX_ENTRIES: usize = 1 << 16;
/// Most initializers.
const MAX_INITIALIZERS: usize = 1 << 20;
/// Most dimensions of one tensor.
const MAX_DIMS: usize = 64;

// ---------------------------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------------------------

const WIRE_VARINT: u8 = 0;
const WIRE_I64: u8 = 1;
const WIRE_LEN: u8 = 2;
const WIRE_I32: u8 = 5;

/// A position-tracking protobuf reader over a seekable byte stream.
struct Reader<R: Read + Seek> {
    inner: BufReader<R>,
    pos: u64,
    len: u64,
}

impl<R: Read + Seek> Reader<R> {
    fn byte(&mut self) -> Result<u8, String> {
        if self.pos >= self.len {
            return Err("unexpected end of data".into());
        }
        let mut b = [0u8; 1];
        self.inner
            .read_exact(&mut b)
            .map_err(|e| format!("read at {}: {e}", self.pos))?;
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
        Err(format!("varint longer than 10 bytes at {}", self.pos))
    }

    /// The next field's `(number, wire type)`.
    fn key(&mut self) -> Result<(u64, u8), String> {
        let k = self.varint()?;
        let field = k >> 3;
        if field == 0 {
            return Err(format!("field number 0 at {}", self.pos));
        }
        Ok((field, (k & 7) as u8))
    }

    /// A length prefix, checked against the end of the enclosing message.
    fn length(&mut self, end: u64) -> Result<u64, String> {
        let n = self.varint()?;
        if n > end.saturating_sub(self.pos) {
            return Err(format!("length {n} at {} runs past its message (ends at {end})", self.pos));
        }
        Ok(n)
    }

    fn skip(&mut self, n: u64) -> Result<(), String> {
        if n > self.len.saturating_sub(self.pos) {
            return Err(format!("skip of {n} bytes at {} runs past the end", self.pos));
        }
        let delta = i64::try_from(n).map_err(|_| "skip too long".to_string())?;
        self.inner
            .seek_relative(delta)
            .map_err(|e| format!("seek at {}: {e}", self.pos))?;
        self.pos += n;
        Ok(())
    }

    /// Skip one field's value of wire type `wire` (inside a message ending at `end`).
    fn skip_value(&mut self, wire: u8, end: u64) -> Result<(), String> {
        match wire {
            WIRE_VARINT => self.varint().map(|_| ()),
            WIRE_I64 => self.skip(8),
            WIRE_LEN => {
                let n = self.length(end)?;
                self.skip(n)
            }
            WIRE_I32 => self.skip(4),
            other => Err(format!("unsupported wire type {other} at {}", self.pos)),
        }
    }

    fn bytes(&mut self, n: u64, cap: u64) -> Result<Vec<u8>, String> {
        if n > cap {
            return Err(format!("a {n}-byte field at {} exceeds the reader's {cap}-byte limit", self.pos));
        }
        let mut buf = vec![0u8; n as usize];
        self.inner
            .read_exact(&mut buf)
            .map_err(|e| format!("read at {}: {e}", self.pos))?;
        self.pos += n;
        Ok(buf)
    }

    fn string(&mut self, end: u64) -> Result<String, String> {
        let n = self.length(end)?;
        let b = self.bytes(n, MAX_STRING)?;
        Ok(String::from_utf8_lossy(&b).into_owned())
    }
}

fn walk<R: Read + Seek>(inner: BufReader<R>, len: u64) -> Result<GraphInfo, String> {
    let mut r = Reader { inner, pos: 0, len };
    let mut info = GraphInfo::default();
    let mut saw_graph = false;
    // ModelProto
    while r.pos < len {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (2, WIRE_LEN) => info.producer = r.string(len)?,
            (7, WIRE_LEN) => {
                let n = r.length(len)?;
                let end = r.pos + n;
                graph(&mut r, end, &mut info)?;
                saw_graph = true;
            }
            (14, WIRE_LEN) => {
                if info.metadata.len() >= MAX_ENTRIES {
                    return Err("too many metadata entries".into());
                }
                let n = r.length(len)?;
                let end = r.pos + n;
                info.metadata.push(string_pair(&mut r, end)?);
            }
            _ => r.skip_value(wire, len)?,
        }
    }
    if !saw_graph {
        return Err("no graph".into());
    }
    // IR < 4 lists initializers as graph inputs too: keep the real inputs only.
    if !info.initializers.is_empty() {
        let names: std::collections::HashSet<&str> = info.initializers.iter().map(|i| i.name.as_str()).collect();
        let inputs = std::mem::take(&mut info.inputs);
        info.inputs = inputs
            .into_iter()
            .filter(|v| !names.contains(v.name.as_str()))
            .collect();
    }
    Ok(info)
}

/// `GraphProto`: initializers (streamed), inputs and outputs (read); the rest skipped.
fn graph<R: Read + Seek>(r: &mut Reader<R>, end: u64, info: &mut GraphInfo) -> Result<(), String> {
    while r.pos < end {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (5, WIRE_LEN) => {
                if info.initializers.len() >= MAX_INITIALIZERS {
                    return Err("too many initializers".into());
                }
                let n = r.length(end)?;
                let tend = r.pos + n;
                info.initializers.push(tensor(r, tend)?);
            }
            (11 | 12, WIRE_LEN) => {
                let list = if field == 11 { &mut info.inputs } else { &mut info.outputs };
                if list.len() >= MAX_ENTRIES {
                    return Err("too many graph inputs or outputs".into());
                }
                let n = r.length(end)?;
                let bytes = r.bytes(n, MAX_VALUE_INFO)?;
                list.push(value_info(&bytes)?);
            }
            _ => r.skip_value(wire, end)?,
        }
    }
    if r.pos != end {
        return Err(format!("graph ends at {} instead of {end}", r.pos));
    }
    Ok(())
}

/// `TensorProto` metadata; `raw_data` is skipped with its offset recorded.
fn tensor<R: Read + Seek>(r: &mut Reader<R>, end: u64) -> Result<InitializerInfo, String> {
    let mut t = InitializerInfo {
        name: String::new(),
        data_type: 0,
        dims: Vec::new(),
        raw_data: None,
        external: false,
        external_data: Vec::new(),
    };
    while r.pos < end {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (1, WIRE_VARINT) => push_dim(&mut t.dims, r.varint()? as i64)?,
            (1, WIRE_LEN) => {
                let n = r.length(end)?;
                let pend = r.pos + n;
                while r.pos < pend {
                    push_dim(&mut t.dims, r.varint()? as i64)?;
                }
                if r.pos != pend {
                    return Err("packed dims overrun".into());
                }
            }
            (2, WIRE_VARINT) => t.data_type = r.varint()? as i32,
            (8, WIRE_LEN) => t.name = r.string(end)?,
            (9, WIRE_LEN) => {
                let n = r.length(end)?;
                t.raw_data = Some((r.pos, n));
                r.skip(n)?;
            }
            (13, WIRE_LEN) => {
                if t.external_data.len() >= MAX_ENTRIES {
                    return Err("too many external-data entries".into());
                }
                let n = r.length(end)?;
                let eend = r.pos + n;
                t.external_data.push(string_pair(r, eend)?);
            }
            (14, WIRE_VARINT) => t.external = r.varint()? == 1,
            _ => r.skip_value(wire, end)?,
        }
    }
    if r.pos != end {
        return Err(format!("tensor '{}' ends at {} instead of {end}", t.name, r.pos));
    }
    if t.external {
        t.raw_data = None;
    }
    Ok(t)
}

fn push_dim(dims: &mut Vec<i64>, d: i64) -> Result<(), String> {
    if dims.len() >= MAX_DIMS {
        return Err("too many dimensions".into());
    }
    dims.push(d);
    Ok(())
}

/// `StringStringEntryProto` (`1: key`, `2: value`).
fn string_pair<R: Read + Seek>(r: &mut Reader<R>, end: u64) -> Result<(String, String), String> {
    let (mut k, mut v) = (String::new(), String::new());
    while r.pos < end {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (1, WIRE_LEN) => k = r.string(end)?,
            (2, WIRE_LEN) => v = r.string(end)?,
            _ => r.skip_value(wire, end)?,
        }
    }
    if r.pos != end {
        return Err("string entry overrun".into());
    }
    Ok((k, v))
}

/// `ValueInfoProto` from its bytes.
fn value_info(bytes: &[u8]) -> Result<ValueInfo, String> {
    let len = bytes.len() as u64;
    let mut r = Reader {
        inner: BufReader::new(Cursor::new(bytes)),
        pos: 0,
        len,
    };
    let mut v = ValueInfo {
        name: String::new(),
        elem_type: 0,
        dims: Vec::new(),
    };
    while r.pos < len {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (1, WIRE_LEN) => v.name = r.string(len)?,
            (2, WIRE_LEN) => {
                // TypeProto: `1: tensor_type` (TypeProto.Tensor); other kinds stay elem_type 0.
                let n = r.length(len)?;
                let tend = r.pos + n;
                while r.pos < tend {
                    let (f, w) = r.key()?;
                    if (f, w) == (1, WIRE_LEN) {
                        let n = r.length(tend)?;
                        let ttend = r.pos + n;
                        tensor_type(&mut r, ttend, &mut v)?;
                    } else {
                        r.skip_value(w, tend)?;
                    }
                }
            }
            _ => r.skip_value(wire, len)?,
        }
    }
    Ok(v)
}

/// `TypeProto.Tensor`: `1: elem_type`, `2: shape` (`TensorShapeProto { 1: dim }`).
fn tensor_type<R: Read + Seek>(r: &mut Reader<R>, end: u64, v: &mut ValueInfo) -> Result<(), String> {
    while r.pos < end {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (1, WIRE_VARINT) => v.elem_type = r.varint()? as i32,
            (2, WIRE_LEN) => {
                let n = r.length(end)?;
                let send = r.pos + n;
                while r.pos < send {
                    let (f, w) = r.key()?;
                    if (f, w) == (1, WIRE_LEN) {
                        if v.dims.len() >= MAX_DIMS {
                            return Err("too many dimensions".into());
                        }
                        let n = r.length(send)?;
                        let dend = r.pos + n;
                        v.dims.push(dimension(r, dend)?);
                    } else {
                        r.skip_value(w, send)?;
                    }
                }
            }
            _ => r.skip_value(wire, end)?,
        }
    }
    Ok(())
}

/// `TensorShapeProto.Dimension`: `1: dim_value`, `2: dim_param`.
fn dimension<R: Read + Seek>(r: &mut Reader<R>, end: u64) -> Result<Dim, String> {
    let mut d = Dim::Unknown;
    while r.pos < end {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (1, WIRE_VARINT) => d = Dim::Fixed(r.varint()? as i64),
            (2, WIRE_LEN) => d = Dim::Param(r.string(end)?),
            _ => r.skip_value(wire, end)?,
        }
    }
    Ok(d)
}

// ---------------------------------------------------------------------------------------------
// Weightless templates: the external-data image from an installed model (P3B-E)
// ---------------------------------------------------------------------------------------------

/// The external-data image of a weightless template ([`external_weights_image`]).
#[derive(Debug)]
pub struct WeightsImage {
    /// The bytes of the template's external-data file (`gpu::ExternalWeights::image`).
    pub bytes: Vec<u8>,
    /// Initializers written.
    pub tensors: usize,
    /// ... of which were cut to their centre kernel row.
    pub centre_rows: usize,
    /// ... of which were rounded to fp16.
    pub to_f16: usize,
}

/// How one template initializer comes from its source tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Derivation {
    /// Keep only kernel row `Some(r)` of a `[O, I, kh, kw]` source (the template is `[O, I, 1, kw]`).
    row: Option<usize>,
    /// Round to fp16 (the template is `FLOAT16`, the source `FLOAT`).
    to_f16: bool,
}

/// What `template` (a template initializer) needs from `source`, or why it cannot come from it.
fn derivation(template: &InitializerInfo, source: &InitializerInfo) -> Result<Derivation, String> {
    let name = &template.name;
    if source.data_type != DT_FLOAT {
        return Err(format!("'{name}' is data type {} in the model, not float", source.data_type));
    }
    let to_f16 = match template.data_type {
        DT_FLOAT => false,
        DT_FLOAT16 => true,
        other => return Err(format!("'{name}' is data type {other} in the template (float or float16 expected)")),
    };
    if template.dims == source.dims {
        return Ok(Derivation { row: None, to_f16 });
    }
    // A k×k kernel cut to its centre row: `[O, I, kh, kw]` -> `[O, I, 1, kw]`, kh odd.
    match (template.dims.as_slice(), source.dims.as_slice()) {
        ([o, i, 1, kw], [so, si, kh, skw]) if o == so && i == si && kw == skw && *kh >= 3 && kh % 2 == 1 => Ok(Derivation {
            row: Some((*kh / 2) as usize),
            to_f16,
        }),
        _ => Err(format!(
            "'{name}' is {:?} in the template and {:?} in the model",
            template.dims, source.dims
        )),
    }
}

/// The external-data image of the weightless graph `template` (every initializer it stores in
/// `weights_file`, at the offsets it records), filled from the same-named initializers of the
/// model at `source_path` (`source` = [`read_graph_info`] of it). Each one must be an inline float
/// tensor (`raw_data`) of the template's shape, or a `[O, I, kh, kw]` kernel (odd `kh`) the template
/// stores as `[O, I, 1, kw]` (its centre row: the exact pruning of a convolution whose input is
/// one row high, so the other rows only ever multiply zero padding); a `FLOAT16` template tensor is
/// rounded to nearest-even from the source, and a value that overflows fp16 is an error. Template
/// initializers stored inline (biases, statistics) are not touched. The source is read once, in
/// file order, through one reused buffer.
pub fn external_weights_image(template: &GraphInfo, weights_file: &str, source: &GraphInfo, source_path: &Path) -> Result<WeightsImage, String> {
    struct Task<'a> {
        name: &'a str,
        dst: (usize, usize),
        src: (u64, u64),
        src_dims: &'a [i64],
        how: Derivation,
    }
    let mut tasks: Vec<Task<'_>> = Vec::new();
    let mut size = 0usize;
    for t in template.initializers.iter().filter(|t| t.external) {
        let (location, offset, length) = t
            .external_range()
            .ok_or_else(|| format!("template initializer '{}' has malformed external data", t.name))?;
        if location != weights_file {
            return Err(format!(
                "template initializer '{}' is stored in '{location}', not '{weights_file}'",
                t.name
            ));
        }
        let s = source
            .initializer(&t.name)
            .ok_or_else(|| format!("the model has no initializer '{}'", t.name))?;
        let how = derivation(t, s)?;
        let count = t
            .element_count()
            .ok_or_else(|| format!("template initializer '{}' has bad dims {:?}", t.name, t.dims))?;
        let bytes = count * element_size(t.data_type).unwrap_or(4);
        if length.is_some_and(|l| l != bytes) {
            return Err(format!("template initializer '{}' records {length:?} bytes for {bytes}", t.name));
        }
        let (src_off, src_len) = s
            .raw_data
            .ok_or_else(|| format!("the model's '{}' is not stored inline", t.name))?;
        let src_count = s
            .element_count()
            .ok_or_else(|| format!("the model's '{}' has bad dims {:?}", s.name, s.dims))?;
        if src_len != src_count * 4 {
            return Err(format!("the model's '{}' holds {src_len} bytes for {src_count} floats", s.name));
        }
        let start = usize::try_from(offset).map_err(|_| "template offset too large".to_string())?;
        let len = usize::try_from(bytes).map_err(|_| "template tensor too large".to_string())?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| "template offsets overflow".to_string())?;
        size = size.max(end);
        tasks.push(Task {
            name: &t.name,
            dst: (start, end),
            src: (src_off, src_len),
            src_dims: &s.dims,
            how,
        });
    }
    if tasks.is_empty() {
        return Err(format!("the template stores no initializer in '{weights_file}'"));
    }
    // No two tensors may share bytes of the image.
    let mut ranges: Vec<(usize, usize, &str)> = tasks.iter().map(|t| (t.dst.0, t.dst.1, t.name)).collect();
    ranges.sort_unstable();
    for w in ranges.windows(2) {
        if w[1].0 < w[0].1 {
            return Err(format!("template initializers '{}' and '{}' overlap in '{weights_file}'", w[0].2, w[1].2));
        }
    }
    tasks.sort_by_key(|t| t.src.0);
    let mut image = vec![0u8; size];
    let file = File::open(source_path).map_err(|e| format!("{}: {e}", source_path.display()))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    // Reused buffers: the source tensor's bytes, its kept values as f32, their fp16 rounding.
    let (mut scratch, mut kept, mut halves): (Vec<u8>, Vec<f32>, Vec<half::f16>) = (Vec::new(), Vec::new(), Vec::new());
    let (mut centre_rows, mut to_f16) = (0usize, 0usize);
    for t in &tasks {
        let src_len = usize::try_from(t.src.1).map_err(|_| "model tensor too large".to_string())?;
        scratch.resize(src_len, 0);
        reader
            .seek(SeekFrom::Start(t.src.0))
            .and_then(|_| reader.read_exact(&mut scratch))
            .map_err(|e| format!("{}: reading '{}': {e}", source_path.display(), t.name))?;
        // The kept bytes (little-endian f32): the whole tensor, or row `row` of every `kh × kw`
        // kernel of a `[O, I, kh, kw]` tensor.
        let kept_bytes: &[u8] = match t.how.row {
            None => &scratch,
            Some(row) => {
                centre_rows += 1;
                let (kh, kw) = (t.src_dims[2] as usize, t.src_dims[3] as usize);
                let mut w = 0usize;
                for kernel in 0..src_len / (kh * kw * 4) {
                    let from = (kernel * kh + row) * kw * 4;
                    scratch.copy_within(from..from + kw * 4, w);
                    w += kw * 4;
                }
                &scratch[..w]
            }
        };
        let dst = &mut image[t.dst.0..t.dst.1];
        if !t.how.to_f16 {
            if kept_bytes.len() != dst.len() {
                return Err(format!("'{}': {} bytes for a {}-byte tensor", t.name, kept_bytes.len(), dst.len()));
            }
            dst.copy_from_slice(kept_bytes);
            continue;
        }
        to_f16 += 1;
        kept.clear();
        kept.extend(
            kept_bytes
                .as_chunks::<4>().0.iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
        if kept.len() * 2 != dst.len() {
            return Err(format!("'{}': {} values for a {}-byte fp16 tensor", t.name, kept.len(), dst.len()));
        }
        halves.clear();
        halves.resize(kept.len(), half::f16::ZERO);
        {
            use half::slice::HalfFloatSliceExt;
            halves.convert_from_f32_slice(&kept);
        }
        for ((out, h), v) in dst.as_chunks_mut::<2>().0.iter_mut().zip(&halves).zip(&kept) {
            if v.is_finite() && !h.is_finite() {
                return Err(format!("'{}': {v} overflows fp16", t.name));
            }
            out.copy_from_slice(&h.to_le_bytes());
        }
    }
    Ok(WeightsImage {
        bytes: image,
        tensors: tasks.len(),
        centre_rows,
        to_f16,
    })
}

#[cfg(test)]
pub(crate) mod test_proto {
    //! A tiny protobuf writer for the tests of this module and of RMVPE's t32 loader: initializers
    //! with external data, metadata and both dim encodings (`test_onnx` covers inline graphs only).

    pub fn varint(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }

    pub fn key(out: &mut Vec<u8>, field: u32, wire: u8) {
        varint(out, (u64::from(field) << 3) | u64::from(wire));
    }

    pub fn field_varint(out: &mut Vec<u8>, field: u32, v: u64) {
        key(out, field, 0);
        varint(out, v);
    }

    pub fn field_bytes(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
        key(out, field, 2);
        varint(out, bytes.len() as u64);
        out.extend_from_slice(bytes);
    }

    pub fn string_pair(k: &str, v: &str) -> Vec<u8> {
        let mut e = Vec::new();
        field_bytes(&mut e, 1, k.as_bytes());
        field_bytes(&mut e, 2, v.as_bytes());
        e
    }

    /// A `TensorProto`: inline `raw` data, or (with `external`) external data at `location`.
    pub fn tensor(name: &str, data_type: u64, dims: &[u64], raw: Option<&[u8]>, external: Option<(&str, u64, u64)>) -> Vec<u8> {
        let mut t = Vec::new();
        let mut packed = Vec::new();
        for &d in dims {
            varint(&mut packed, d);
        }
        field_bytes(&mut t, 1, &packed);
        field_varint(&mut t, 2, data_type);
        field_bytes(&mut t, 8, name.as_bytes());
        if let Some(raw) = raw {
            field_bytes(&mut t, 9, raw);
        }
        if let Some((loc, off, len)) = external {
            field_bytes(&mut t, 13, &string_pair("location", loc));
            field_bytes(&mut t, 13, &string_pair("offset", &off.to_string()));
            field_bytes(&mut t, 13, &string_pair("length", &len.to_string()));
            field_varint(&mut t, 14, 1);
        }
        t
    }

    /// A `ValueInfoProto` with a tensor type; dims as `Ok(fixed)` / `Err(param)`.
    pub fn value_info(name: &str, elem_type: u64, dims: &[Result<u64, &str>]) -> Vec<u8> {
        let mut shape = Vec::new();
        for d in dims {
            let mut dim = Vec::new();
            match d {
                Ok(v) => field_varint(&mut dim, 1, *v),
                Err(p) => field_bytes(&mut dim, 2, p.as_bytes()),
            }
            field_bytes(&mut shape, 1, &dim);
        }
        let mut tt = Vec::new();
        field_varint(&mut tt, 1, elem_type);
        field_bytes(&mut tt, 2, &shape);
        let mut ty = Vec::new();
        field_bytes(&mut ty, 1, &tt);
        let mut v = Vec::new();
        field_bytes(&mut v, 1, name.as_bytes());
        field_bytes(&mut v, 2, &ty);
        v
    }

    /// A node `op(inputs) -> outputs` with raw attribute bytes (`AttributeProto`s).
    pub fn node(op: &str, name: &str, inputs: &[&str], outputs: &[&str], attrs: &[Vec<u8>]) -> Vec<u8> {
        let mut n = Vec::new();
        for i in inputs {
            field_bytes(&mut n, 1, i.as_bytes());
        }
        for o in outputs {
            field_bytes(&mut n, 2, o.as_bytes());
        }
        field_bytes(&mut n, 3, name.as_bytes());
        field_bytes(&mut n, 4, op.as_bytes());
        for a in attrs {
            field_bytes(&mut n, 5, a);
        }
        n
    }

    /// An `INTS` attribute.
    pub fn attr_ints(name: &str, values: &[i64]) -> Vec<u8> {
        let mut a = Vec::new();
        field_bytes(&mut a, 1, name.as_bytes());
        for v in values {
            field_varint(&mut a, 8, *v as u64);
        }
        field_varint(&mut a, 20, 7);
        a
    }

    /// A `ModelProto` (IR 7, default domain at `opset`) around the parts.
    pub fn model(
        nodes: &[Vec<u8>],
        initializers: &[Vec<u8>],
        inputs: &[Vec<u8>],
        outputs: &[Vec<u8>],
        metadata: &[(&str, &str)],
        opset: u64,
    ) -> Vec<u8> {
        let mut g = Vec::new();
        for n in nodes {
            field_bytes(&mut g, 1, n);
        }
        field_bytes(&mut g, 2, b"test");
        for i in initializers {
            field_bytes(&mut g, 5, i);
        }
        for i in inputs {
            field_bytes(&mut g, 11, i);
        }
        for o in outputs {
            field_bytes(&mut g, 12, o);
        }
        let mut op = Vec::new();
        field_varint(&mut op, 2, opset);
        let mut m = Vec::new();
        field_varint(&mut m, 1, 7);
        field_bytes(&mut m, 2, b"sigil-proto-test");
        field_bytes(&mut m, 7, &g);
        field_bytes(&mut m, 8, &op);
        for (k, v) in metadata {
            field_bytes(&mut m, 14, &string_pair(k, v));
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::test_proto as tp;
    use super::*;
    use crate::services::neural_audio::test_onnx::{Dim as TDim, Graph, FLOAT};

    fn temp_file(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let p = std::env::temp_dir().join(format!("sigil-onnx-proto-{tag}-{}-{nanos}.onnx", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    /// The graph builder's models: names, symbolic and fixed dims, element types, initializer
    /// metadata and raw-data offsets that point at the values in the file.
    #[test]
    fn reads_inputs_outputs_and_raw_data_offsets() {
        let w: Vec<f32> = (0..12).map(|i| i as f32 * 0.5 - 1.0).collect();
        let mut g = Graph::new();
        g.input("input", FLOAT, &[TDim::Fixed(1), TDim::Fixed(128), TDim::Named("time")])
            .init_f32("w", &[3, 4], &w)
            .init_i64("shape", &[2], &[4, 3])
            .node("Identity", &["input"], &["output"], &[])
            .output("output", FLOAT, &[TDim::Fixed(1), TDim::Named("time"), TDim::Fixed(360)]);
        let bytes = g.model_bytes(17);
        let path = temp_file("io", &bytes);
        let info = read_graph_info(&path).unwrap();
        let from_bytes = read_graph_info_bytes(&bytes).unwrap();
        assert_eq!(info, from_bytes);
        assert_eq!(info.producer, "sigil-test");
        assert_eq!(info.inputs.len(), 1);
        assert_eq!(info.inputs[0].name, "input");
        assert_eq!(info.inputs[0].elem_type, DT_FLOAT);
        assert_eq!(info.inputs[0].dims, vec![Dim::Fixed(1), Dim::Fixed(128), Dim::Param("time".into())]);
        assert_eq!(info.outputs[0].dims[1], Dim::Param("time".into()));
        assert_eq!(info.outputs[0].dims[2], Dim::Fixed(360));
        let wi = info.initializer("w").unwrap();
        assert_eq!((wi.data_type, wi.dims.clone(), wi.external), (DT_FLOAT, vec![3, 4], false));
        assert_eq!(wi.element_count(), Some(12));
        let (off, len) = wi.raw_data.unwrap();
        assert_eq!(len, 48);
        let raw = &bytes[off as usize..(off + len) as usize];
        let back: Vec<f32> = raw
            .as_chunks::<4>().0.iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(back, w);
        let si = info.initializer("shape").unwrap();
        assert_eq!((si.data_type, si.dims.clone()), (DT_INT64, vec![2]));
        let _ = std::fs::remove_file(&path);
    }

    /// External-data initializers (the weightless templates), unpacked dims, metadata, and IR < 4
    /// graphs that list their initializers as inputs.
    #[test]
    fn reads_external_data_metadata_and_drops_initializer_inputs() {
        let mut unpacked = Vec::new();
        for d in [2u64, 3] {
            tp::field_varint(&mut unpacked, 1, d);
        }
        tp::field_varint(&mut unpacked, 2, 1);
        tp::field_bytes(&mut unpacked, 8, b"u");
        tp::field_bytes(&mut unpacked, 9, &[0u8; 24]);
        let ext = tp::tensor("k", 10, &[512, 512, 1, 3], None, Some(("rmvpe.weights", 4096, 1_572_864)));
        let model = tp::model(
            &[tp::node("Identity", "n0", &["x"], &["y"], &[])],
            &[unpacked, ext],
            &[
                tp::value_info("x", 1, &[Ok(1), Err("n")]),
                tp::value_info("k", 10, &[Ok(512), Ok(512), Ok(1), Ok(3)]),
            ],
            &[tp::value_info("y", 1, &[Ok(1), Err("n")])],
            &[("sigil.format", "test"), ("sigil.source_sha256", "abc")],
            11,
        );
        let info = read_graph_info_bytes(&model).unwrap();
        assert_eq!(
            info.inputs
                .iter()
                .map(|v| v.name.as_str())
                .collect::<Vec<_>>(),
            vec!["x"],
            "the initializer input is dropped"
        );
        let u = info.initializer("u").unwrap();
        assert_eq!((u.dims.clone(), u.raw_data.map(|r| r.1)), (vec![2, 3], Some(24)));
        let k = info.initializer("k").unwrap();
        assert!(k.external && k.raw_data.is_none());
        assert_eq!(k.data_type, DT_FLOAT16);
        assert_eq!(k.external_range(), Some(("rmvpe.weights", 4096, Some(1_572_864))));
        assert_eq!(k.element_count(), Some(786_432));
        assert_eq!(info.metadata("sigil.source_sha256"), Some("abc"));
        assert_eq!(info.metadata("missing"), None);
        assert_eq!(element_size(DT_FLOAT16), Some(2));
        assert_eq!(element_size(DT_FLOAT), Some(4));
        assert_eq!(element_size(8), None);
    }

    /// Truncated, oversized and garbage inputs are errors (no panic, no huge allocation).
    #[test]
    fn malformed_models_are_errors() {
        let mut g = Graph::new();
        g.input("x", FLOAT, &[TDim::Fixed(1)])
            .init_f32("w", &[2], &[1.0, 2.0])
            .node("Identity", &["x"], &["y"], &[])
            .output("y", FLOAT, &[TDim::Fixed(1)]);
        let bytes = g.model_bytes(17);
        for cut in [1usize, 5, bytes.len() / 2, bytes.len() - 1] {
            assert!(read_graph_info_bytes(&bytes[..cut]).is_err(), "cut at {cut}");
        }
        assert!(read_graph_info_bytes(&[]).is_err(), "no graph");
        assert!(read_graph_info_bytes(&[0xff; 32]).is_err());
        // A string length far past the data.
        let mut huge = Vec::new();
        tp::key(&mut huge, 2, 2);
        tp::varint(&mut huge, u64::MAX >> 2);
        assert!(read_graph_info_bytes(&huge).is_err());
        // Group wire types are refused.
        let mut group = Vec::new();
        tp::key(&mut group, 3, 3);
        assert!(read_graph_info_bytes(&group).is_err());
        assert!(read_graph_info(Path::new("definitely-missing-model.onnx")).is_err());
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// A source model (inline float tensors) and a template over it (external tensors at the
    /// given offsets, `(name, data_type, dims, offset, length)`).
    /// A template initializer: `(name, data_type, dims, offset, length)`.
    type TemplateTensor<'a> = (&'a str, u64, &'a [u64], u64, u64);

    fn source_and_template(templates: &[TemplateTensor<'_>]) -> (std::path::PathBuf, GraphInfo, GraphInfo) {
        let a: Vec<f32> = (0..6).map(|i| i as f32 - 2.5).collect();
        let k: Vec<f32> = (0..2 * 2 * 3 * 3).map(|i| i as f32 * 0.25).collect();
        let h: Vec<f32> = vec![1.0, -0.5, 65_000.0, 1e-8];
        let source = tp::model(
            &[tp::node("Identity", "n0", &["x"], &["y"], &[])],
            &[
                tp::tensor("a", 1, &[2, 3], Some(&f32_bytes(&a)), None),
                tp::tensor("k", 1, &[2, 2, 3, 3], Some(&f32_bytes(&k)), None),
                tp::tensor("h", 1, &[4], Some(&f32_bytes(&h)), None),
                tp::tensor("big", 1, &[1], Some(&f32_bytes(&[1e6])), None),
            ],
            &[tp::value_info("x", 1, &[Ok(1)])],
            &[tp::value_info("y", 1, &[Ok(1)])],
            &[],
            11,
        );
        let path = temp_file("weights-src", &source);
        let inits: Vec<Vec<u8>> = templates
            .iter()
            .map(|(name, dt, dims, off, len)| tp::tensor(name, *dt, dims, None, Some(("w.bin", *off, *len))))
            .collect();
        let template = tp::model(
            &[tp::node("Identity", "n0", &["x"], &["y"], &[])],
            &inits,
            &[tp::value_info("x", 1, &[Ok(1)])],
            &[tp::value_info("y", 1, &[Ok(1)])],
            &[],
            11,
        );
        let s = read_graph_info(&path).unwrap();
        let t = read_graph_info_bytes(&template).unwrap();
        (path, s, t)
    }

    /// A template's image: tensors copied as they are, a 3×3 kernel cut to its centre row, and a
    /// tensor rounded to fp16, each at the template's offset.
    #[test]
    fn weights_image_copies_cuts_centre_rows_and_rounds_to_fp16() {
        let (path, source, template) = source_and_template(&[("a", 1, &[2, 3], 0, 24), ("k", 1, &[2, 2, 1, 3], 32, 48), ("h", 10, &[4], 96, 8)]);
        let image = external_weights_image(&template, "w.bin", &source, &path).unwrap();
        assert_eq!((image.tensors, image.centre_rows, image.to_f16), (3, 1, 1));
        assert_eq!(image.bytes.len(), 104);
        let f = |b: &[u8]| {
            b.as_chunks::<4>().0.iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
        };
        assert_eq!(f(&image.bytes[0..24]), (0..6).map(|i| i as f32 - 2.5).collect::<Vec<f32>>());
        // k[o][i][1][..] of the [2, 2, 3, 3] source (values 0.25 * flat index).
        let want: Vec<f32> = (0..4)
            .flat_map(|oi| (0..3).map(move |c| (oi * 9 + 3 + c) as f32 * 0.25))
            .collect();
        assert_eq!(f(&image.bytes[32..80]), want);
        assert_eq!(&image.bytes[24..32], &[0u8; 8], "gaps stay zero");
        let h: Vec<f32> = image.bytes[96..104]
            .as_chunks::<2>().0.iter()
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        assert_eq!(
            h,
            vec![1.0, -0.5, half::f16::from_f32(65_000.0).to_f32(), half::f16::from_f32(1e-8).to_f32()]
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Anything the image cannot be filled from exactly is an error: a missing tensor, other dims,
    /// a value that overflows fp16, overlapping tensors, a wrong length or another weights file.
    #[test]
    fn weights_image_rejects_what_does_not_fit() {
        let cases: [(&[TemplateTensor<'_>], &str); 6] = [
            (&[("missing", 1, &[2], 0, 8)], "no initializer 'missing'"),
            (&[("a", 1, &[3, 2], 0, 24)], "[3, 2] in the template"),
            (&[("big", 10, &[1], 0, 2)], "overflows fp16"),
            (&[("a", 1, &[2, 3], 0, 24), ("h", 1, &[4], 16, 16)], "overlap"),
            (&[("a", 1, &[2, 3], 0, 20)], "records Some(20) bytes"),
            (&[("k", 1, &[2, 2, 2, 3], 0, 96)], "in the template"),
        ];
        for (templates, want) in cases {
            let (path, source, template) = source_and_template(templates);
            let err = external_weights_image(&template, "w.bin", &source, &path).unwrap_err();
            assert!(err.contains(want), "{templates:?}: {err}");
            let _ = std::fs::remove_file(&path);
        }
        let (path, source, template) = source_and_template(&[("a", 1, &[2, 3], 0, 24)]);
        assert!(external_weights_image(&template, "other.bin", &source, &path)
            .unwrap_err()
            .contains("not 'other.bin'"));
        let _ = std::fs::remove_file(&path);
    }

    /// Dev rig: the real `rmvpe.onnx` (361 MB) is walked without reading its weights, and its
    /// metadata is what RMVPE's loader expects.
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache cargo test real_rmvpe_graph_info -- --ignored --nocapture`
    #[test]
    #[ignore = "dev rig: needs SIGIL_TEST_MODELS_DIR with the RMVPE model"]
    fn real_rmvpe_graph_info() {
        let Some(path) = crate::services::neural_audio::vc::f0_fusion::test_model("pitch-rmvpe") else {
            eprintln!("RMVPE model not found; skipped");
            return;
        };
        let t0 = std::time::Instant::now();
        let info = read_graph_info(&path).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "walked {} in {ms:.1} ms: {} initializers, inputs {:?}",
            path.display(),
            info.initializers.len(),
            info.inputs
        );
        assert_eq!(info.inputs.len(), 1);
        assert_eq!(info.inputs[0].dims, vec![Dim::Fixed(1), Dim::Fixed(128), Dim::Param("time".into())]);
        assert_eq!(info.initializers.len(), 282);
        assert!(info
            .initializers
            .iter()
            .all(|i| i.raw_data.is_some() && i.data_type == DT_FLOAT));
        let params: u64 = info
            .initializers
            .iter()
            .filter_map(|i| i.element_count())
            .sum();
        assert_eq!(params, 90_399_839);
    }
}
