//! Allow-list pickle VM for PyTorch checkpoints (`data.pkl` inside the torch zip format).
//!
//! A pickle is a program for a stack machine that can import and call *any* Python callable;
//! loading an untrusted `.pth` with `pickle` / `torch.load(weights_only=False)` runs arbitrary code.
//! This VM executes only the binary opcodes a `torch.save` checkpoint uses and resolves `GLOBAL`s
//! against a fixed allow-list; nothing is ever "called" except the few constructors below, which
//! are implemented here as pure data transforms:
//!
//! | global | effect |
//! | --- | --- |
//! | `collections.OrderedDict` | `REDUCE` with `()` → an (ordered) dict |
//! | `torch._utils._rebuild_tensor_v2` | → a [`TensorRef`] (storage view: offset, size, stride) |
//! | `torch._utils._rebuild_parameter` | → the wrapped tensor |
//! | `torch.{Float,Half,BFloat16,Long,Int}Storage` | storage dtype inside a persistent id |
//! | `_codecs.encode` | `(str, "latin1")` → bytes (protocol 2's encoding of `bytes`) |
//!
//! Anything else (`os.system`, `builtins.eval`, `NEWOBJ`, `BUILD`, `INST`, text-protocol opcodes,
//! extension registry, out-of-band buffers) is rejected with [`PickleError`].
//!
//! Memory: values live in an arena and containers refer to arena ids, so a memo reference is an id
//! (no clone of the referenced value) and mutation through the memo keeps pickle's object
//! identity. String and bytes payloads are shared (`Arc`), so using one memoised string as the key
//! of many dicts, or anywhere else, never copies it. Everything the VM materialises (arena values,
//! payloads, container slots, dict indexes, memo entries) is charged against
//! [`PickleLimits::max_memory`], and every other size is bounded too ([`PickleLimits`]);
//! malformed input returns an error, never panics. [`Pickled::to_json`] is bounded separately by a
//! byte-counting [`JsonBudget`] that a caller shares across every value it converts.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Arena index of a value.
pub type Id = u32;

/// Storage element type of a torch storage (`torch.<Type>Storage`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageDtype {
    F32,
    F16,
    Bf16,
    I64,
    I32,
}

impl StorageDtype {
    pub fn size(self) -> usize {
        match self {
            StorageDtype::F32 | StorageDtype::I32 => 4,
            StorageDtype::F16 | StorageDtype::Bf16 => 2,
            StorageDtype::I64 => 8,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            StorageDtype::F32 => "float32",
            StorageDtype::F16 => "float16",
            StorageDtype::Bf16 => "bfloat16",
            StorageDtype::I64 => "int64",
            StorageDtype::I32 => "int32",
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, StorageDtype::F32 | StorageDtype::F16 | StorageDtype::Bf16)
    }
}

/// Longest storage key / location accepted (torch writes `"0"`.. and `"cpu"` / `"cuda:0"`).
pub const MAX_STORAGE_FIELD: usize = 64;

/// A torch storage referenced by a persistent id: `("storage", <Type>Storage, key, location, numel)`.
/// `key` and `location` are at most [`MAX_STORAGE_FIELD`] bytes, so a clone is small and bounded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageRef {
    pub key: String,
    pub dtype: StorageDtype,
    pub location: String,
    /// Elements (of `dtype`) the storage holds.
    pub numel: u64,
}

/// A tensor view of a storage (`_rebuild_tensor_v2(storage, offset, size, stride, ...)`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRef {
    pub storage: StorageRef,
    /// Offset in elements.
    pub offset: u64,
    pub shape: Vec<u64>,
    pub stride: Vec<u64>,
}

/// The allow-listed callables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Global {
    OrderedDict,
    RebuildTensorV2,
    RebuildParameter,
    Storage(StorageDtype),
    CodecsEncode,
}

/// A value in the arena.
#[derive(Clone, Debug)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    /// Shared: a memo reference, a dict key or a `STACK_GLOBAL` operand never copies the text.
    Str(Arc<str>),
    Bytes(Arc<[u8]>),
    Tuple(Vec<Id>),
    List(Vec<Id>),
    /// `dict` and `OrderedDict` (insertion order kept; a repeated key overwrites in place).
    Dict(Vec<(Id, Id)>),
    Global(Global),
    Storage(Box<StorageRef>),
    Tensor(Box<TensorRef>),
}

impl Value {
    fn kind(&self) -> &'static str {
        match self {
            Value::None => "None",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "str",
            Value::Bytes(_) => "bytes",
            Value::Tuple(_) => "tuple",
            Value::List(_) => "list",
            Value::Dict(_) => "dict",
            Value::Global(_) => "global",
            Value::Storage(_) => "storage",
            Value::Tensor(_) => "tensor",
        }
    }

    /// Bytes charged against [`PickleLimits::max_memory`] when the value is created (the arena
    /// slot plus what it owns; shared payloads are charged once, where they are read).
    fn cost(&self) -> usize {
        let own = match self {
            Value::Str(s) => s.len(),
            Value::Bytes(b) => b.len(),
            Value::Tuple(v) | Value::List(v) => v.len() * std::mem::size_of::<Id>(),
            Value::Dict(v) => v.len() * std::mem::size_of::<(Id, Id)>(),
            Value::Storage(s) => std::mem::size_of::<StorageRef>() + s.key.len() + s.location.len(),
            Value::Tensor(t) => {
                std::mem::size_of::<TensorRef>()
                    + t.storage.key.len()
                    + t.storage.location.len()
                    + (t.shape.len() + t.stride.len()) * std::mem::size_of::<u64>()
            }
            Value::None | Value::Bool(_) | Value::Int(_) | Value::Float(_) | Value::Global(_) => 0,
        };
        std::mem::size_of::<Value>() + own
    }
}

/// Why a pickle was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PickleError {
    /// A `GLOBAL` outside the allow-list (the classic code-execution vector).
    ForbiddenGlobal(String),
    /// An opcode this VM does not execute (object construction, text protocol, extensions).
    ForbiddenOpcode(u8, usize),
    /// Structurally invalid pickle (truncated, stack underflow, bad argument types...).
    Malformed(String),
    /// A resource limit was hit.
    Limit(String),
}

impl fmt::Display for PickleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PickleError::ForbiddenGlobal(g) => write!(f, "the checkpoint references `{g}`, which is not allowed in a weights file"),
            PickleError::ForbiddenOpcode(op, at) => write!(f, "pickle opcode 0x{op:02x} at byte {at} is not allowed in a weights file"),
            PickleError::Malformed(m) => write!(f, "malformed pickle: {m}"),
            PickleError::Limit(m) => write!(f, "pickle exceeds a safety limit: {m}"),
        }
    }
}

/// Resource bounds for one pickle.
#[derive(Clone, Copy, Debug)]
pub struct PickleLimits {
    pub max_bytes: usize,
    pub max_values: usize,
    pub max_stack: usize,
    pub max_marks: usize,
    pub max_string: usize,
    pub max_container: usize,
    pub max_dims: usize,
    /// Bytes the VM may materialise in total (approximate: arena slots, string / bytes payloads,
    /// container slots, dict indexes and memo entries).
    pub max_memory: usize,
}

impl Default for PickleLimits {
    fn default() -> Self {
        Self {
            // RVC checkpoints: ~60 KB of pickle, ~10k values, ~1 MB materialised.
            max_bytes: 64 << 20,
            max_values: 4_000_000,
            max_stack: 1_000_000,
            max_marks: 100_000,
            max_string: 16 << 20,
            max_container: 4_000_000,
            max_dims: 16,
            max_memory: 256 << 20,
        }
    }
}

/// Most arguments any allow-listed call takes (`_rebuild_tensor_v2`: 7).
const MAX_CALL_ARGS: usize = 8;
/// Approximate cost of a new dict-index entry / a dict's index / a memo entry.
const DICT_INDEX_ENTRY_COST: usize = 48;
const DICT_INDEX_COST: usize = 64;
const MEMO_ENTRY_COST: usize = 16;

/// Budget for [`Pickled::to_json`]. One budget is meant to be shared by every conversion of one
/// checkpoint, so the total JSON produced is bounded however many values are converted.
#[derive(Clone, Copy, Debug)]
pub struct JsonBudget {
    /// Deepest nesting converted (deeper values make the conversion fail).
    pub max_depth: usize,
    /// Nodes (values and dict keys) still allowed.
    pub nodes: usize,
    /// Bytes of JSON still allowed: string / bytes / key payloads plus [`JSON_NODE_COST`] per node.
    pub bytes: usize,
    /// Longer strings (and bytes) are cut to this many bytes, at a character boundary, before
    /// they are copied.
    pub max_string: usize,
}

/// Bytes charged per JSON node on top of its payload.
pub const JSON_NODE_COST: usize = 32;

impl JsonBudget {
    /// Checkpoint metadata: RVC's config list and a few short strings fit many times over.
    pub const METADATA: JsonBudget = JsonBudget { max_depth: 8, nodes: 10_000, bytes: 1 << 20, max_string: 4096 };

    fn take(&mut self, payload: usize) -> bool {
        let cost = JSON_NODE_COST.saturating_add(payload);
        if self.nodes == 0 || self.bytes < cost {
            return false;
        }
        self.nodes -= 1;
        self.bytes -= cost;
        true
    }
}

/// `s` cut to at most `max` bytes at a character boundary.
fn cut(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// The unpickled object graph.
#[derive(Debug)]
pub struct Pickled {
    values: Vec<Value>,
    pub root: Id,
    /// Bytes charged against [`PickleLimits::max_memory`] while unpickling (tests check it).
    #[cfg(test)]
    pub memory: usize,
}

impl Pickled {
    pub fn get(&self, id: Id) -> &Value {
        &self.values[id as usize]
    }

    /// `dict` entries with string keys, in order (non-string keys are skipped).
    pub fn str_items(&self, id: Id) -> Option<Vec<(&str, Id)>> {
        match self.get(id) {
            Value::Dict(items) => Some(
                items
                    .iter()
                    .filter_map(|(k, v)| match self.get(*k) {
                        Value::Str(s) => Some((s.as_ref(), *v)),
                        _ => None,
                    })
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Plain data (no tensors / storages / globals) as JSON, bounded by `budget` (depth, nodes and
    /// bytes; long strings are cut). `None` when the value holds something that is not plain data
    /// or does not fit what is left of the budget (the budget stays charged for what was built).
    pub fn to_json(&self, id: Id, budget: &mut JsonBudget) -> Option<serde_json::Value> {
        let depth = budget.max_depth;
        self.to_json_inner(id, depth, budget)
    }

    fn to_json_inner(&self, id: Id, depth: usize, budget: &mut JsonBudget) -> Option<serde_json::Value> {
        use serde_json::Value as J;
        if depth == 0 {
            return None;
        }
        let max = budget.max_string;
        match self.get(id) {
            Value::Global(_) | Value::Storage(_) | Value::Tensor(_) => return None,
            // Charged with their payload below.
            Value::Str(_) | Value::Bytes(_) => {}
            _ => {
                if !budget.take(0) {
                    return None;
                }
            }
        }
        Some(match self.get(id) {
            Value::None => J::Null,
            Value::Bool(b) => J::Bool(*b),
            Value::Int(i) => J::from(*i),
            Value::Float(x) => serde_json::Number::from_f64(*x).map(J::Number).unwrap_or(J::Null),
            Value::Str(s) => {
                let s = cut(s, max);
                if !budget.take(s.len()) {
                    return None;
                }
                J::String(s.to_string())
            }
            Value::Bytes(b) => {
                let b = &b[..b.len().min(max)];
                // Latin-1: up to two UTF-8 bytes per byte.
                if !budget.take(b.len() * 2) {
                    return None;
                }
                J::String(b.iter().map(|&c| c as char).collect())
            }
            Value::Tuple(items) | Value::List(items) => {
                let mut out = Vec::new();
                for &i in items {
                    out.push(self.to_json_inner(i, depth - 1, budget)?);
                }
                J::Array(out)
            }
            Value::Dict(items) => {
                let mut map = serde_json::Map::new();
                for (k, v) in items {
                    let key = match self.get(*k) {
                        Value::Str(s) => {
                            let s = cut(s, max);
                            if !budget.take(s.len()) {
                                return None;
                            }
                            s.to_string()
                        }
                        Value::Int(i) => {
                            if !budget.take(20) {
                                return None;
                            }
                            i.to_string()
                        }
                        _ => return None,
                    };
                    map.insert(key, self.to_json_inner(*v, depth - 1, budget)?);
                }
                J::Object(map)
            }
            Value::Global(_) | Value::Storage(_) | Value::Tensor(_) => return None,
        })
    }
}

/// Run the VM over `data` (a whole pickle, `PROTO` to `STOP`).
pub fn unpickle(data: &[u8], limits: &PickleLimits) -> Result<Pickled, PickleError> {
    if data.len() > limits.max_bytes {
        return Err(PickleError::Limit(format!("{} bytes of pickle (max {})", data.len(), limits.max_bytes)));
    }
    Vm {
        data,
        pos: 0,
        limits,
        values: Vec::new(),
        stack: Vec::new(),
        marks: Vec::new(),
        memo: HashMap::new(),
        dict_index: HashMap::new(),
        memory: 0,
    }
    .run()
}

struct Vm<'a> {
    data: &'a [u8],
    pos: usize,
    limits: &'a PickleLimits,
    values: Vec<Value>,
    stack: Vec<Id>,
    marks: Vec<usize>,
    memo: HashMap<u32, Id>,
    /// Key -> entry position of every dict built so far (O(1) SETITEM on large dicts).
    dict_index: HashMap<Id, HashMap<DictKey, usize>>,
    /// Bytes charged so far (see [`PickleLimits::max_memory`]).
    memory: usize,
}

fn malformed<T>(msg: impl Into<String>) -> Result<T, PickleError> {
    Err(PickleError::Malformed(msg.into()))
}

impl<'a> Vm<'a> {
    fn run(mut self) -> Result<Pickled, PickleError> {
        loop {
            let at = self.pos;
            let op = self.u8()?;
            match op {
                0x80 => {
                    // PROTO
                    let v = self.u8()?;
                    if v > 5 {
                        return malformed(format!("unsupported pickle protocol {v}"));
                    }
                }
                0x95 => {
                    // FRAME (protocol 4): only a read-ahead hint
                    self.take(8)?;
                }
                b'.' => {
                    // STOP
                    let root = self.pop()?;
                    return Ok(Pickled {
                        values: self.values,
                        root,
                        #[cfg(test)]
                        memory: self.memory,
                    });
                }
                b'(' => {
                    if self.marks.len() >= self.limits.max_marks {
                        return Err(PickleError::Limit("too many nested marks".into()));
                    }
                    self.marks.push(self.stack.len());
                }
                b'0' => {
                    self.pop()?;
                }
                b'1' => {
                    // POP_MARK
                    let m = self.pop_mark()?;
                    self.stack.truncate(m);
                }
                b'2' => {
                    // DUP
                    let top = self.top()?;
                    self.push_id(top)?;
                }
                b'N' => {
                    self.push(Value::None)?;
                }
                0x88 => {
                    self.push(Value::Bool(true))?;
                }
                0x89 => {
                    self.push(Value::Bool(false))?;
                }
                b'J' => {
                    let v = i32::from_le_bytes(self.array()?);
                    self.push(Value::Int(v as i64))?;
                }
                b'K' => {
                    let v = self.u8()?;
                    self.push(Value::Int(v as i64))?;
                }
                b'M' => {
                    let v = u16::from_le_bytes(self.array()?);
                    self.push(Value::Int(v as i64))?;
                }
                0x8a | 0x8b => {
                    // LONG1 / LONG4: little-endian two's complement
                    let n = if op == 0x8a { self.u8()? as usize } else { self.len_u32()? };
                    let bytes = self.take(n)?;
                    if n > 8 {
                        return Err(PickleError::Limit(format!("{n}-byte integer")));
                    }
                    let mut v: i64 = 0;
                    for (i, b) in bytes.iter().enumerate() {
                        v |= (*b as i64) << (8 * i);
                    }
                    if n > 0 && n < 8 && bytes[n - 1] & 0x80 != 0 {
                        v -= 1i64 << (8 * n);
                    }
                    self.push(Value::Int(v))?;
                }
                b'G' => {
                    let v = f64::from_be_bytes(self.array()?);
                    self.push(Value::Float(v))?;
                }
                b'X' | 0x8c | 0x8d => {
                    // BINUNICODE / SHORT_BINUNICODE / BINUNICODE8
                    let n = match op {
                        b'X' => self.len_u32()?,
                        0x8c => self.u8()? as usize,
                        _ => self.len_u64()?,
                    };
                    let raw = self.take_string(n)?;
                    let s = std::str::from_utf8(raw).map_err(|_| PickleError::Malformed(format!("invalid UTF-8 string at byte {at}")))?;
                    self.push(Value::Str(Arc::from(s)))?;
                }
                b'T' | b'U' => {
                    // BINSTRING / SHORT_BINSTRING (Python 2 str): kept as latin-1 text
                    let n = if op == b'T' { self.len_u32()? } else { self.u8()? as usize };
                    let raw = self.take_string(n)?;
                    let text: String = raw.iter().map(|&c| c as char).collect();
                    self.push(Value::Str(Arc::from(text)))?;
                }
                b'B' | b'C' | 0x8e | 0x96 => {
                    // BINBYTES / SHORT_BINBYTES / BINBYTES8 / BYTEARRAY8
                    let n = match op {
                        b'B' => self.len_u32()?,
                        b'C' => self.u8()? as usize,
                        _ => self.len_u64()?,
                    };
                    let raw = self.take_string(n)?;
                    self.push(Value::Bytes(Arc::from(raw)))?;
                }
                b')' => {
                    self.push(Value::Tuple(Vec::new()))?;
                }
                b't' => {
                    let items = self.pop_to_mark()?;
                    self.push(Value::Tuple(items))?;
                }
                0x85..=0x87 => {
                    let n = (op - 0x84) as usize;
                    let items = self.pop_n(n)?;
                    self.push(Value::Tuple(items))?;
                }
                b']' => {
                    self.push(Value::List(Vec::new()))?;
                }
                b'l' => {
                    let items = self.pop_to_mark()?;
                    self.push(Value::List(items))?;
                }
                b'a' => {
                    let item = self.pop()?;
                    let list = self.top()?;
                    self.list_extend(list, vec![item])?;
                }
                b'e' => {
                    let items = self.pop_to_mark()?;
                    let list = self.top()?;
                    self.list_extend(list, items)?;
                }
                b'}' => {
                    self.push(Value::Dict(Vec::new()))?;
                }
                b'd' => {
                    let items = self.pop_to_mark()?;
                    let id = self.push(Value::Dict(Vec::new()))?;
                    self.dict_set(id, items)?;
                }
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    let dict = self.top()?;
                    self.dict_set(dict, vec![k, v])?;
                }
                b'u' => {
                    let items = self.pop_to_mark()?;
                    let dict = self.top()?;
                    self.dict_set(dict, items)?;
                }
                b'c' => {
                    // GLOBAL "module\nname\n"
                    let module = self.line()?;
                    let name = self.line()?;
                    let g = resolve_global(&module, &name)?;
                    self.push(Value::Global(g))?;
                }
                0x93 => {
                    // STACK_GLOBAL
                    let name = self.pop()?;
                    let module = self.pop()?;
                    let g = match (self.value(module), self.value(name)) {
                        (Value::Str(module), Value::Str(name)) => resolve_global(module, name)?,
                        _ => return malformed("STACK_GLOBAL needs two strings"),
                    };
                    self.push(Value::Global(g))?;
                }
                b'R' => {
                    let args = self.pop()?;
                    let func = self.pop()?;
                    let v = self.reduce(func, args)?;
                    self.push(v)?;
                }
                b'Q' => {
                    // BINPERSID
                    let pid = self.pop()?;
                    let v = self.persistent_load(pid)?;
                    self.push(v)?;
                }
                b'q' => {
                    let i = self.u8()? as u32;
                    self.memo_put(i)?;
                }
                b'r' => {
                    let i = u32::from_le_bytes(self.array()?);
                    self.memo_put(i)?;
                }
                0x94 => {
                    // MEMOIZE
                    let i = u32::try_from(self.memo.len()).map_err(|_| PickleError::Limit("memo".into()))?;
                    self.memo_put(i)?;
                }
                b'h' => {
                    let i = self.u8()? as u32;
                    self.memo_get(i)?;
                }
                b'j' => {
                    let i = u32::from_le_bytes(self.array()?);
                    self.memo_get(i)?;
                }
                _ => return Err(PickleError::ForbiddenOpcode(op, at)),
            }
        }
    }

    // ---- input -----------------------------------------------------------------------------

    fn take(&mut self, n: usize) -> Result<&'a [u8], PickleError> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.data.len());
        match end {
            Some(end) => {
                let s = &self.data[self.pos..end];
                self.pos = end;
                Ok(s)
            }
            None => malformed(format!("truncated at byte {}", self.pos)),
        }
    }

    fn take_string(&mut self, n: usize) -> Result<&'a [u8], PickleError> {
        if n > self.limits.max_string {
            return Err(PickleError::Limit(format!("{n}-byte string")));
        }
        self.take(n)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], PickleError> {
        let s = self.take(N)?;
        let mut a = [0u8; N];
        a.copy_from_slice(s);
        Ok(a)
    }

    fn u8(&mut self) -> Result<u8, PickleError> {
        Ok(self.take(1)?[0])
    }

    fn len_u32(&mut self) -> Result<usize, PickleError> {
        Ok(u32::from_le_bytes(self.array()?) as usize)
    }

    fn len_u64(&mut self) -> Result<usize, PickleError> {
        let n = u64::from_le_bytes(self.array()?);
        usize::try_from(n).map_err(|_| PickleError::Limit(format!("{n}-byte length")))
    }

    /// A `\n`-terminated ASCII line (GLOBAL operands), at most 256 bytes.
    fn line(&mut self) -> Result<String, PickleError> {
        let rest = &self.data[self.pos..];
        let n = rest
            .iter()
            .take(257)
            .position(|&c| c == b'\n')
            .ok_or(PickleError::Malformed("unterminated GLOBAL operand".into()))?;
        let s = std::str::from_utf8(&rest[..n])
            .ok()
            .filter(|s| s.is_ascii())
            .ok_or(PickleError::Malformed("non-ASCII GLOBAL operand".into()))?
            .to_string();
        self.pos += n + 1;
        Ok(s)
    }

    // ---- arena and stack -------------------------------------------------------------------

    fn value(&self, id: Id) -> &Value {
        &self.values[id as usize]
    }

    /// Charge `bytes` against the memory limit.
    fn charge(&mut self, bytes: usize) -> Result<(), PickleError> {
        self.memory = self.memory.saturating_add(bytes);
        if self.memory > self.limits.max_memory {
            return Err(PickleError::Limit(format!("more than {} bytes of objects", self.limits.max_memory)));
        }
        Ok(())
    }

    fn alloc(&mut self, v: Value) -> Result<Id, PickleError> {
        if self.values.len() >= self.limits.max_values {
            return Err(PickleError::Limit(format!("more than {} values", self.limits.max_values)));
        }
        self.charge(v.cost())?;
        self.values.push(v);
        Ok((self.values.len() - 1) as Id)
    }

    fn push(&mut self, v: Value) -> Result<Id, PickleError> {
        let id = self.alloc(v)?;
        self.push_id(id)?;
        Ok(id)
    }

    fn push_id(&mut self, id: Id) -> Result<(), PickleError> {
        if self.stack.len() >= self.limits.max_stack {
            return Err(PickleError::Limit("stack depth".into()));
        }
        self.stack.push(id);
        Ok(())
    }

    fn pop(&mut self) -> Result<Id, PickleError> {
        let floor = self.marks.last().copied().unwrap_or(0);
        if self.stack.len() <= floor {
            return malformed(format!("stack underflow at byte {}", self.pos));
        }
        Ok(self.stack.pop().expect("checked above"))
    }

    fn pop_n(&mut self, n: usize) -> Result<Vec<Id>, PickleError> {
        let floor = self.marks.last().copied().unwrap_or(0);
        if self.stack.len() < floor + n {
            return malformed(format!("stack underflow at byte {}", self.pos));
        }
        Ok(self.stack.split_off(self.stack.len() - n))
    }

    fn top(&self) -> Result<Id, PickleError> {
        let floor = self.marks.last().copied().unwrap_or(0);
        if self.stack.len() <= floor {
            return malformed(format!("stack underflow at byte {}", self.pos));
        }
        Ok(*self.stack.last().expect("checked above"))
    }

    fn pop_mark(&mut self) -> Result<usize, PickleError> {
        self.marks.pop().ok_or(PickleError::Malformed(format!("no MARK at byte {}", self.pos)))
    }

    fn pop_to_mark(&mut self) -> Result<Vec<Id>, PickleError> {
        let m = self.pop_mark()?;
        if m > self.stack.len() {
            return malformed("mark above the stack top");
        }
        let items = self.stack.split_off(m);
        if items.len() > self.limits.max_container {
            return Err(PickleError::Limit("container too large".into()));
        }
        Ok(items)
    }

    fn memo_put(&mut self, i: u32) -> Result<(), PickleError> {
        let top = self.top()?;
        if !self.memo.contains_key(&i) {
            if self.memo.len() >= self.limits.max_values {
                return Err(PickleError::Limit("memo size".into()));
            }
            self.charge(MEMO_ENTRY_COST)?;
        }
        self.memo.insert(i, top);
        Ok(())
    }

    fn memo_get(&mut self, i: u32) -> Result<(), PickleError> {
        let id = *self.memo.get(&i).ok_or(PickleError::Malformed(format!("memo key {i} unset")))?;
        self.push_id(id)
    }

    fn list_extend(&mut self, list: Id, items: Vec<Id>) -> Result<(), PickleError> {
        let max = self.limits.max_container;
        let len = match self.value(list) {
            Value::List(v) => v.len(),
            other => return malformed(format!("APPEND to a {}", other.kind())),
        };
        if len + items.len() > max {
            return Err(PickleError::Limit("list too long".into()));
        }
        self.charge(items.len() * std::mem::size_of::<Id>())?;
        if let Value::List(v) = &mut self.values[list as usize] {
            v.extend(items);
        }
        Ok(())
    }

    fn dict_set(&mut self, dict: Id, items: Vec<Id>) -> Result<(), PickleError> {
        if !items.len().is_multiple_of(2) {
            return malformed("odd number of dict items");
        }
        if !matches!(self.value(dict), Value::Dict(_)) {
            return malformed(format!("SETITEM on a {}", self.value(dict).kind()));
        }
        let max = self.limits.max_container;
        for kv in items.chunks(2) {
            // Keys compare by value for str / int (enough for checkpoint dicts); others append.
            // A str key shares the string's allocation, so it costs a pointer, not a copy.
            let key = self.dict_key(kv[0]);
            let slot = key
                .as_ref()
                .and_then(|k| self.dict_index.get(&dict).and_then(|m| m.get(k)).copied());
            if let Some(i) = slot {
                if let Value::Dict(entries) = &mut self.values[dict as usize] {
                    entries[i].1 = kv[1];
                }
                continue;
            }
            let len = match self.value(dict) {
                Value::Dict(entries) => entries.len(),
                _ => unreachable!("checked above"),
            };
            if len >= max {
                return Err(PickleError::Limit("dict too large".into()));
            }
            let mut cost = std::mem::size_of::<(Id, Id)>();
            if key.is_some() {
                cost += DICT_INDEX_ENTRY_COST;
                if !self.dict_index.contains_key(&dict) {
                    cost += DICT_INDEX_COST;
                }
            }
            self.charge(cost)?;
            if let Value::Dict(entries) = &mut self.values[dict as usize] {
                entries.push((kv[0], kv[1]));
            }
            if let Some(k) = key {
                self.dict_index.entry(dict).or_default().insert(k, len);
            }
        }
        Ok(())
    }

    fn dict_key(&self, id: Id) -> Option<DictKey> {
        match self.value(id) {
            Value::Str(s) => Some(DictKey::Str(Arc::clone(s))),
            Value::Int(i) => Some(DictKey::Int(*i)),
            _ => None,
        }
    }

    // ---- the allow-listed calls ------------------------------------------------------------

    /// Items of a call's argument tuple (at most [`MAX_CALL_ARGS`], so the copy is tiny).
    fn tuple_items(&self, id: Id, what: &str) -> Result<Vec<Id>, PickleError> {
        match self.value(id) {
            Value::Tuple(items) if items.len() <= MAX_CALL_ARGS => Ok(items.clone()),
            Value::Tuple(items) => malformed(format!("{what} with {} arguments", items.len())),
            other => malformed(format!("{what} arguments must be a tuple, not a {}", other.kind())),
        }
    }

    fn int_of(&self, id: Id, what: &str) -> Result<i64, PickleError> {
        match self.value(id) {
            Value::Int(i) => Ok(*i),
            Value::Bool(b) => Ok(*b as i64),
            other => malformed(format!("{what} must be an int, not a {}", other.kind())),
        }
    }

    fn dims_of(&self, id: Id, what: &str) -> Result<Vec<u64>, PickleError> {
        let items = match self.value(id) {
            Value::Tuple(v) | Value::List(v) => v,
            other => return malformed(format!("{what} must be a tuple, not a {}", other.kind())),
        };
        if items.len() > self.limits.max_dims {
            return Err(PickleError::Limit(format!("{} tensor dimensions", items.len())));
        }
        items
            .iter()
            .map(|&i| {
                let v = self.int_of(i, what)?;
                u64::try_from(v).map_err(|_| PickleError::Malformed(format!("negative {what}")))
            })
            .collect()
    }

    /// A storage key or location: short, and only characters torch writes there.
    fn storage_field(&self, id: Id, what: &str, extra: &[u8]) -> Result<String, PickleError> {
        let Value::Str(s) = self.value(id) else {
            return malformed(format!("storage {what} must be a string"));
        };
        if s.is_empty()
            || s.len() > MAX_STORAGE_FIELD
            || !s.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_' || extra.contains(&c))
        {
            return malformed(format!("unsafe storage {what} {:?}", cut(s, MAX_STORAGE_FIELD)));
        }
        Ok(s.to_string())
    }

    fn reduce(&mut self, func: Id, args: Id) -> Result<Value, PickleError> {
        let g = match self.value(func) {
            Value::Global(g) => *g,
            other => return malformed(format!("REDUCE on a {}", other.kind())),
        };
        let a = self.tuple_items(args, "REDUCE")?;
        match g {
            Global::OrderedDict => {
                if !a.is_empty() {
                    return malformed("OrderedDict(...) with arguments");
                }
                Ok(Value::Dict(Vec::new()))
            }
            Global::RebuildTensorV2 => {
                // (storage, storage_offset, size, stride, requires_grad, backward_hooks[, metadata])
                if a.len() != 6 && a.len() != 7 {
                    return malformed(format!("_rebuild_tensor_v2 with {} arguments", a.len()));
                }
                let storage = match self.value(a[0]) {
                    Value::Storage(s) => StorageRef::clone(s),
                    _ => return malformed("_rebuild_tensor_v2 without a storage"),
                };
                let offset = u64::try_from(self.int_of(a[1], "storage offset")?)
                    .map_err(|_| PickleError::Malformed("negative storage offset".into()))?;
                let shape = self.dims_of(a[2], "tensor size")?;
                let stride = self.dims_of(a[3], "tensor stride")?;
                if shape.len() != stride.len() {
                    return malformed("tensor size and stride ranks differ");
                }
                if !matches!(self.value(a[4]), Value::Bool(_) | Value::Int(_)) {
                    return malformed("requires_grad must be a bool");
                }
                if !matches!(self.value(a[5]), Value::Dict(d) if d.is_empty()) {
                    return malformed("tensor backward hooks are not allowed");
                }
                Ok(Value::Tensor(Box::new(TensorRef { storage, offset, shape, stride })))
            }
            Global::RebuildParameter => {
                // (data, requires_grad, backward_hooks)
                if a.len() != 3 {
                    return malformed(format!("_rebuild_parameter with {} arguments", a.len()));
                }
                let tensor = match self.value(a[0]) {
                    Value::Tensor(t) => t.clone(),
                    _ => return malformed("_rebuild_parameter without a tensor"),
                };
                if !matches!(self.value(a[2]), Value::Dict(d) if d.is_empty()) {
                    return malformed("parameter backward hooks are not allowed");
                }
                Ok(Value::Tensor(tensor))
            }
            Global::CodecsEncode => {
                if a.len() != 2 {
                    return malformed("_codecs.encode needs (str, encoding)");
                }
                let (Value::Str(text), Value::Str(enc)) = (self.value(a[0]), self.value(a[1])) else {
                    return malformed("_codecs.encode needs (str, encoding)");
                };
                if !matches!(enc.to_ascii_lowercase().as_str(), "latin1" | "latin-1" | "iso-8859-1") {
                    return malformed(format!("_codecs.encode to {}", cut(enc, 32)));
                }
                let bytes = text
                    .chars()
                    .map(|c| u8::try_from(c as u32).map_err(|_| PickleError::Malformed("non-latin-1 bytes literal".into())))
                    .collect::<Result<Vec<u8>, _>>()?;
                Ok(Value::Bytes(Arc::from(bytes)))
            }
            Global::Storage(_) => malformed("a storage type cannot be called"),
        }
    }

    fn persistent_load(&mut self, pid: Id) -> Result<Value, PickleError> {
        let a = self.tuple_items(pid, "persistent id")?;
        // ("storage", storage_type, key, location, numel)
        if a.len() != 5 {
            return malformed(format!("persistent id with {} fields", a.len()));
        }
        match self.value(a[0]) {
            Value::Str(s) if s.as_ref() == "storage" => {}
            _ => return malformed("persistent id is not a storage"),
        }
        let dtype = match self.value(a[1]) {
            Value::Global(Global::Storage(dtype)) => *dtype,
            _ => return malformed("persistent id without a storage type"),
        };
        // The key names a zip entry (`data/<key>`): no path characters.
        let key = self.storage_field(a[2], "key", &[])?;
        let location = self.storage_field(a[3], "location", b":.-")?;
        let numel = u64::try_from(self.int_of(a[4], "storage numel")?)
            .map_err(|_| PickleError::Malformed("negative storage size".into()))?;
        Ok(Value::Storage(Box::new(StorageRef { key, dtype, location, numel })))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum DictKey {
    Str(Arc<str>),
    Int(i64),
}

/// The allow-list (see the module docs).
fn resolve_global(module: &str, name: &str) -> Result<Global, PickleError> {
    Ok(match (module, name) {
        ("collections", "OrderedDict") => Global::OrderedDict,
        ("torch._utils", "_rebuild_tensor_v2") => Global::RebuildTensorV2,
        ("torch._utils", "_rebuild_parameter") => Global::RebuildParameter,
        ("torch", "FloatStorage") => Global::Storage(StorageDtype::F32),
        ("torch", "HalfStorage") => Global::Storage(StorageDtype::F16),
        ("torch", "BFloat16Storage") => Global::Storage(StorageDtype::Bf16),
        ("torch", "LongStorage") => Global::Storage(StorageDtype::I64),
        ("torch", "IntStorage") => Global::Storage(StorageDtype::I32),
        ("_codecs", "encode") => Global::CodecsEncode,
        _ => {
            return Err(PickleError::ForbiddenGlobal(format!(
                "{}.{}",
                cut(module, 256),
                cut(name, 256)
            )))
        }
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Tiny pickle assembler for tests (protocol 2 opcodes).
    #[derive(Default)]
    pub(crate) struct P(pub Vec<u8>);

    impl P {
        pub fn new() -> Self {
            P(vec![0x80, 2])
        }
        pub fn op(mut self, op: u8) -> Self {
            self.0.push(op);
            self
        }
        pub fn global(mut self, module: &str, name: &str) -> Self {
            self.0.push(b'c');
            self.0.extend_from_slice(module.as_bytes());
            self.0.push(b'\n');
            self.0.extend_from_slice(name.as_bytes());
            self.0.push(b'\n');
            self
        }
        pub fn str(mut self, s: &str) -> Self {
            self.0.push(b'X');
            self.0.extend_from_slice(&(s.len() as u32).to_le_bytes());
            self.0.extend_from_slice(s.as_bytes());
            self
        }
        pub fn int(mut self, v: i32) -> Self {
            self.0.push(b'J');
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        pub fn put(mut self, i: u8) -> Self {
            self.0.extend_from_slice(&[b'q', i]);
            self
        }
        pub fn get(mut self, i: u8) -> Self {
            self.0.extend_from_slice(&[b'h', i]);
            self
        }
        pub fn stop(mut self) -> Vec<u8> {
            self.0.push(b'.');
            self.0
        }
        /// `_rebuild_tensor_v2(("storage", torch.<ty>, key, "cpu", numel), offset, size, stride,
        /// False, OrderedDict())`.
        pub fn tensor(self, ty: &str, key: &str, numel: i32, offset: i32, size: &[i32], stride: &[i32]) -> Self {
            let mut p = self.global("torch._utils", "_rebuild_tensor_v2").op(b'(');
            p = p.op(b'(').str("storage").global("torch", ty).str(key).str("cpu").int(numel).op(b't').op(b'Q');
            p = p.int(offset).op(b'(');
            for &s in size {
                p = p.int(s);
            }
            p = p.op(b't').op(b'(');
            for &s in stride {
                p = p.int(s);
            }
            p = p.op(b't').op(0x89);
            p = p.global("collections", "OrderedDict").op(b')').op(b'R');
            p.op(b't').op(b'R')
        }
    }

    fn run(data: &[u8]) -> Result<Pickled, PickleError> {
        unpickle(data, &PickleLimits::default())
    }

    fn json(depth: usize) -> JsonBudget {
        JsonBudget { max_depth: depth, ..JsonBudget::METADATA }
    }

    #[test]
    fn plain_values_roundtrip() {
        let data = P::new()
            .op(b'}')
            .put(0)
            .op(b'(')
            .str("sr")
            .str("40k")
            .str("f0")
            .op(b'K')
            .op(1)
            .str("config")
            .op(b']')
            .op(b'(')
            .int(1025)
            .int(-3)
            .str("1")
            .op(0x88)
            .op(b'N')
            .op(b'e')
            .op(b'u')
            .stop();
        let p = run(&data).unwrap();
        let got = p.to_json(p.root, &mut json(8)).unwrap();
        assert_eq!(got, serde_json::json!({"sr": "40k", "f0": 1, "config": [1025, -3, "1", true, null]}));
    }

    #[test]
    fn tensor_and_ordered_dict() {
        let data = P::new()
            .global("collections", "OrderedDict")
            .op(b')')
            .op(b'R')
            .str("w")
            .tensor("HalfStorage", "0", 6, 0, &[2, 3], &[3, 1])
            .op(b's')
            .stop();
        let p = run(&data).unwrap();
        let items = p.str_items(p.root).unwrap();
        assert_eq!(items.len(), 1);
        let Value::Tensor(t) = p.get(items[0].1) else { panic!("not a tensor") };
        assert_eq!(t.storage.dtype, StorageDtype::F16);
        assert_eq!(t.storage.key, "0");
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.stride, vec![3, 1]);
    }

    #[test]
    fn forbidden_globals_are_rejected() {
        for (m, n) in [
            ("os", "system"),
            ("posix", "system"),
            ("nt", "system"),
            ("builtins", "eval"),
            ("builtins", "exec"),
            ("__builtin__", "eval"),
            ("subprocess", "Popen"),
            ("torch", "load"),
            ("torch._utils", "_rebuild_tensor"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("collections", "OrderedDictX"),
        ] {
            let data = P::new().global(m, n).str("echo pwned").op(0x85).op(b'R').stop();
            match run(&data) {
                Err(PickleError::ForbiddenGlobal(g)) => assert_eq!(g, format!("{m}.{n}")),
                other => panic!("{m}.{n}: {other:?}"),
            }
        }
        // STACK_GLOBAL goes through the same allow-list.
        let data = P::new().str("os").str("system").op(0x93).stop();
        assert!(matches!(run(&data), Err(PickleError::ForbiddenGlobal(_))));
    }

    #[test]
    fn object_construction_opcodes_are_rejected() {
        // NEWOBJ, NEWOBJ_EX, OBJ, INST, BUILD, EXT1, text INT / PUT / GET / PERSID, protocol 5 buffers.
        for op in [0x81u8, 0x92, b'o', b'i', b'b', 0x82, b'I', b'p', b'g', b'P', 0x97, 0x98, b'S', b'V', b'F', b'L'] {
            let data = P::new().op(b'N').op(op).stop();
            assert!(matches!(run(&data), Err(PickleError::ForbiddenOpcode(o, _)) if o == op), "opcode 0x{op:02x}");
        }
    }

    #[test]
    fn calls_are_type_checked() {
        // REDUCE on something that is not a global, storage types called directly, OrderedDict args.
        assert!(matches!(run(&P::new().str("x").op(b')').op(b'R').stop()), Err(PickleError::Malformed(_))));
        assert!(matches!(run(&P::new().global("torch", "FloatStorage").op(b')').op(b'R').stop()), Err(PickleError::Malformed(_))));
        assert!(matches!(
            run(&P::new().global("collections", "OrderedDict").op(b']').op(0x85).op(b'R').stop()),
            Err(PickleError::Malformed(_))
        ));
        // A persistent id that is not a storage.
        assert!(matches!(run(&P::new().str("module").op(0x85).op(b'Q').stop()), Err(PickleError::Malformed(_))));
        // Path-like storage keys are refused (they name zip entries).
        let bad_key = P::new().op(b'(').str("storage").global("torch", "FloatStorage").str("../x").str("cpu").int(1).op(b't').op(b'Q').stop();
        assert!(matches!(run(&bad_key), Err(PickleError::Malformed(_))));
        // Tensors with backward hooks.
        let hooks = P::new()
            .global("torch._utils", "_rebuild_tensor_v2")
            .op(b'(')
            .op(b'(')
            .str("storage")
            .global("torch", "FloatStorage")
            .str("0")
            .str("cpu")
            .int(1)
            .op(b't')
            .op(b'Q')
            .int(0)
            .op(b')')
            .op(b')')
            .op(0x89)
            .op(b'}')
            .str("k")
            .int(1)
            .op(b's')
            .op(b't')
            .op(b'R')
            .stop();
        assert!(matches!(run(&hooks), Err(PickleError::Malformed(_))));
    }

    #[test]
    fn memo_references_do_not_copy() {
        // l = [0]*64; then 40 rounds of l = [l, l] through the memo: 2^40 leaves if copied.
        let mut p = P::new().op(b']').put(0);
        for _ in 0..64 {
            p = p.int(7).op(b'a');
        }
        for _ in 0..40 {
            p = p.op(b'0').op(b']').get(0).op(b'a').get(0).op(b'a').put(0);
        }
        let data = p.stop();
        let out = run(&data).unwrap();
        assert!(out.values.len() < 200, "arena grew to {}", out.values.len());
    }

    #[test]
    fn truncation_and_underflow_are_errors() {
        let good = P::new().op(b'}').str("a").int(1).op(b's').stop();
        for cut in 0..good.len() {
            let r = run(&good[..cut]);
            assert!(r.is_err(), "prefix of {cut} bytes parsed");
        }
        assert!(matches!(run(&P::new().op(b'a').stop()), Err(PickleError::Malformed(_))));
        assert!(matches!(run(&P::new().op(b'u').stop()), Err(PickleError::Malformed(_))));
        assert!(matches!(run(&P::new().get(3).stop()), Err(PickleError::Malformed(_))));
        // POP cannot reach below the innermost mark.
        assert!(matches!(run(&P::new().int(1).op(b'(').op(b'0').stop()), Err(PickleError::Malformed(_))));
    }

    #[test]
    fn limits_hold() {
        let mut data = P::new().op(b']');
        for _ in 0..1000 {
            data = data.op(b'N').op(b'a');
        }
        let data = data.stop();
        let limits = PickleLimits { max_values: 100, ..PickleLimits::default() };
        assert!(matches!(unpickle(&data, &limits), Err(PickleError::Limit(_))));
        let limits = PickleLimits { max_bytes: 10, ..PickleLimits::default() };
        assert!(matches!(unpickle(&data, &limits), Err(PickleError::Limit(_))));
        let limits = PickleLimits { max_memory: 1000, ..PickleLimits::default() };
        assert!(matches!(unpickle(&data, &limits), Err(PickleError::Limit(_))));
        assert!(unpickle(&data, &PickleLimits::default()).unwrap().memory < 64 << 10);
        // A 9-byte LONG1 integer.
        let mut big = P::new().op(0x8a).op(9);
        big.0.extend_from_slice(&[1; 9]);
        assert!(matches!(run(&big.stop()), Err(PickleError::Limit(_))));
        // A huge declared string length.
        let mut s = P::new().op(b'X');
        s.0.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(run(&s.stop()).is_err());
        // Deep nesting in to_json is cut off instead of recursing without bound.
        let mut deep = P::new();
        for _ in 0..500 {
            deep = deep.op(b']');
        }
        for _ in 0..499 {
            deep = deep.op(b'a');
        }
        let p = run(&deep.stop()).unwrap();
        assert!(p.to_json(p.root, &mut json(32)).is_none());
    }

    #[test]
    fn long_and_float_decoding() {
        let mut p = P::new().op(b']');
        for bytes in [&[0x00u8][..], &[0xff], &[0x00, 0x01], &[0xff, 0x7f], &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80]] {
            p = p.op(0x8a).op(bytes.len() as u8);
            p.0.extend_from_slice(bytes);
            p = p.op(b'a');
        }
        p = p.op(b'G');
        p.0.extend_from_slice(&1.5f64.to_be_bytes());
        p = p.op(b'a');
        let out = run(&p.stop()).unwrap();
        assert_eq!(out.to_json(out.root, &mut json(4)).unwrap(), serde_json::json!([0, -1, 256, 32767, i64::MIN + 1, 1.5]));
    }

    #[test]
    fn memoised_strings_are_shared_not_copied() {
        // One 1 MB string, then 10,000 dicts {S: S} built from memo references: 10 GB if every
        // SETITEM copied its key.
        let big = "x".repeat(1 << 20);
        let mut p = P::new().str(&big).put(0).op(b'0').op(b']');
        for _ in 0..10_000 {
            p = p.op(b'}').get(0).get(0).op(b's').op(b'a');
        }
        let out = run(&p.stop()).unwrap();
        assert!(out.memory < 8 << 20, "{} bytes charged", out.memory);
        let Value::List(dicts) = out.get(out.root) else { panic!("not a list") };
        assert_eq!(dicts.len(), 10_000);
        let Value::Dict(entries) = out.get(dicts[9_999]) else { panic!("not a dict") };
        let (Value::Str(k), Value::Str(v)) = (out.get(entries[0].0), out.get(entries[0].1)) else { panic!("not strings") };
        assert!(Arc::ptr_eq(k, v) && k.len() == 1 << 20);
    }

    #[test]
    fn storage_fields_are_short_and_plain() {
        let pid = |key: &str, location: &str| {
            P::new().op(b'(').str("storage").global("torch", "FloatStorage").str(key).str(location).int(1).op(b't').op(b'Q').stop()
        };
        assert!(run(&pid("0", "cpu")).is_ok());
        assert!(run(&pid("12", "cuda:0")).is_ok());
        for (key, location) in [("0".to_string(), "c".repeat(65)), ("0".into(), "cpu/../x".into()), ("0".into(), String::new()), ("k".repeat(65), "cpu".into())] {
            assert!(matches!(run(&pid(&key, &location)), Err(PickleError::Malformed(_))), "{key:?} {location:?}");
        }
        // A memoised 1 MB location reused by many persistent ids is refused at the first one.
        let big = "c".repeat(1 << 20);
        let mut p = P::new().str(&big).put(0).op(b'0');
        for _ in 0..1000 {
            p = p.op(b'(').str("storage").global("torch", "HalfStorage").str("0").get(0).int(1).op(b't').op(b'Q').op(b'0');
        }
        assert!(matches!(run(&p.op(b'N').stop()), Err(PickleError::Malformed(_))));
    }

    #[test]
    fn copies_are_charged_against_the_memory_limit() {
        // _codecs.encode makes a new bytes object per call: 64 calls on a memoised 1 MB text.
        let text = "y".repeat(1 << 20);
        let mut p = P::new().global("_codecs", "encode").put(0).op(b'0').str(&text).str("latin1").op(0x86).put(1).op(b'0').op(b']');
        for _ in 0..64 {
            p = p.get(0).get(1).op(b'R').op(b'a');
        }
        let limits = PickleLimits { max_memory: 16 << 20, ..PickleLimits::default() };
        assert!(matches!(unpickle(&p.stop(), &limits), Err(PickleError::Limit(_))));
        // Call arguments are never copied in bulk: a 100k-item argument tuple is refused.
        let mut p = P::new().global("collections", "OrderedDict").op(b'(');
        for _ in 0..100_000 {
            p = p.op(b'N');
        }
        assert!(matches!(run(&p.op(b't').op(b'R').stop()), Err(PickleError::Malformed(_))));
    }

    #[test]
    fn to_json_is_bounded_in_bytes_across_calls() {
        // {"info": [S] * 99_999, "sr": "40k"} with S a memoised 1 MB string: ~100 GB of JSON if
        // every reference were copied.
        let big = "z".repeat(1 << 20);
        let mut p = P::new().op(b'}').str("info").op(b']').str(&big).put(1).op(b'a');
        for _ in 0..99_998 {
            p = p.get(1).op(b'a');
        }
        p = p.op(b's').str("sr").str("40k").op(b's');
        let out = run(&p.stop()).unwrap();
        let items = out.str_items(out.root).unwrap();
        assert_eq!(items.iter().map(|i| i.0).collect::<Vec<_>>(), ["info", "sr"]);
        let mut budget = JsonBudget::METADATA;
        let started = std::time::Instant::now();
        assert!(out.to_json(items[0].1, &mut budget).is_none(), "the list does not fit");
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        assert!(budget.bytes < budget.max_string + JSON_NODE_COST, "{budget:?}");
        // The budget is shared: a second conversion only gets what is left.
        let left = budget.bytes;
        assert!(out.to_json(items[0].1, &mut budget).is_none());
        assert!(budget.bytes <= left);
        assert_eq!(out.to_json(items[1].1, &mut json(2)), Some(serde_json::json!("40k")));
        // Long strings are cut at a character boundary before they are copied.
        let long = format!("a{}", "\u{e9}".repeat(5000));
        let out = run(&P::new().str(&long).stop()).unwrap();
        let cut = out.to_json(out.root, &mut json(2)).unwrap();
        assert_eq!(cut.as_str().unwrap().len(), 4095);
        assert!(long.starts_with(cut.as_str().unwrap()));
    }


    /// Random mutations of a valid checkpoint-shaped pickle never panic or hang.
    #[test]
    fn mutated_pickles_never_panic() {
        let base = P::new()
            .global("collections", "OrderedDict")
            .op(b')')
            .op(b'R')
            .put(0)
            .op(b'(')
            .str("weight")
            .op(b'}')
            .op(b'(')
            .str("a.weight")
            .tensor("HalfStorage", "0", 4, 0, &[2, 2], &[2, 1])
            .str("b.bias")
            .tensor("FloatStorage", "1", 2, 0, &[2], &[1])
            .op(b'u')
            .str("sr")
            .str("40k")
            .op(b'u')
            .stop();
        assert!(run(&base).is_ok());
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..4000 {
            let mut d = base.clone();
            for _ in 0..(1 + next() % 4) {
                let i = (next() as usize) % d.len();
                match next() % 3 {
                    0 => d[i] = next() as u8,
                    1 => {
                        d.remove(i);
                    }
                    _ => d.insert(i, next() as u8),
                }
            }
            let _ = run(&d);
        }
    }
}
