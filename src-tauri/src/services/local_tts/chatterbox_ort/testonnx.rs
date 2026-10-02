//! Test-only ONNX writer for the Chatterbox stand-in graphs: hand-encoded protobuf (field numbers
//! as in `onnx.proto`) with what `neural_audio::test_onnx` lacks here: operator domains
//! (`com.microsoft` for `GroupQueryAttention`), several opset imports, int32 / bool types, rank-0
//! values, float attributes and initializers stored as external data.

use std::path::Path;

pub const FLOAT: u64 = 1;
pub const INT32: u64 = 6;
pub const INT64: u64 = 7;
pub const BOOL: u64 = 9;

fn varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn key(out: &mut Vec<u8>, field: u32, wire: u8) {
    varint(out, (u64::from(field) << 3) | u64::from(wire));
}

fn field_varint(out: &mut Vec<u8>, field: u32, v: u64) {
    key(out, field, 0);
    varint(out, v);
}

fn field_bytes(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    key(out, field, 2);
    varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn field_f32(out: &mut Vec<u8>, field: u32, v: f32) {
    key(out, field, 5);
    out.extend_from_slice(&v.to_le_bytes());
}

/// A dimension: fixed or symbolic.
#[derive(Clone, Copy, Debug)]
pub enum Dim {
    N(u64),
    S(&'static str),
}

/// A node attribute.
#[allow(dead_code)] // The full set the writer supports; the stand-ins use some of them.
pub enum Attr {
    Int(&'static str, i64),
    Float(&'static str, f32),
    Ints(&'static str, Vec<i64>),
    /// A float tensor (`ConstantOfShape.value`, OneHot values are inputs instead).
    FloatTensor(&'static str, Vec<u64>, Vec<f32>),
}

fn tensor_raw(name: &str, data_type: u64, dims: &[u64], raw: &[u8]) -> Vec<u8> {
    let mut t = Vec::new();
    let mut packed = Vec::new();
    for &d in dims {
        varint(&mut packed, d);
    }
    if !dims.is_empty() {
        field_bytes(&mut t, 1, &packed);
    }
    field_varint(&mut t, 2, data_type);
    if !name.is_empty() {
        field_bytes(&mut t, 8, name.as_bytes());
    }
    field_bytes(&mut t, 9, raw);
    t
}

fn entry(k: &str, v: &str) -> Vec<u8> {
    let mut e = Vec::new();
    field_bytes(&mut e, 1, k.as_bytes());
    field_bytes(&mut e, 2, v.as_bytes());
    e
}

fn value_info(name: &str, elem: u64, dims: &[Dim]) -> Vec<u8> {
    let mut shape = Vec::new();
    for d in dims {
        let mut dim = Vec::new();
        match d {
            Dim::N(v) => field_varint(&mut dim, 1, *v),
            Dim::S(p) => field_bytes(&mut dim, 2, p.as_bytes()),
        }
        field_bytes(&mut shape, 1, &dim);
    }
    let mut tensor = Vec::new();
    field_varint(&mut tensor, 1, elem);
    // An empty shape message is a scalar (rank 0).
    field_bytes(&mut tensor, 2, &shape);
    let mut ty = Vec::new();
    field_bytes(&mut ty, 1, &tensor);
    let mut v = Vec::new();
    field_bytes(&mut v, 1, name.as_bytes());
    field_bytes(&mut v, 2, &ty);
    v
}

fn attribute(a: &Attr) -> Vec<u8> {
    let mut out = Vec::new();
    match a {
        Attr::Int(name, v) => {
            field_bytes(&mut out, 1, name.as_bytes());
            field_varint(&mut out, 3, *v as u64);
            field_varint(&mut out, 20, 2);
        }
        Attr::Float(name, v) => {
            field_bytes(&mut out, 1, name.as_bytes());
            field_f32(&mut out, 2, *v);
            field_varint(&mut out, 20, 1);
        }
        Attr::Ints(name, vs) => {
            field_bytes(&mut out, 1, name.as_bytes());
            for v in vs {
                field_varint(&mut out, 8, *v as u64);
            }
            field_varint(&mut out, 20, 7);
        }
        Attr::FloatTensor(name, dims, data) => {
            field_bytes(&mut out, 1, name.as_bytes());
            let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            field_bytes(&mut out, 5, &tensor_raw("", FLOAT, dims, &raw));
            field_varint(&mut out, 20, 4);
        }
    }
    out
}

/// An ONNX graph under construction.
#[derive(Default)]
pub struct Graph {
    nodes: Vec<Vec<u8>>,
    initializers: Vec<Vec<u8>>,
    inputs: Vec<Vec<u8>>,
    outputs: Vec<Vec<u8>>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn input(&mut self, name: &str, elem: u64, dims: &[Dim]) -> &mut Self {
        self.inputs.push(value_info(name, elem, dims));
        self
    }

    pub fn output(&mut self, name: &str, elem: u64, dims: &[Dim]) -> &mut Self {
        self.outputs.push(value_info(name, elem, dims));
        self
    }

    pub fn f32s(&mut self, name: &str, dims: &[u64], data: &[f32]) -> &mut Self {
        assert_eq!(dims.iter().product::<u64>() as usize, data.len(), "{name}");
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.initializers.push(tensor_raw(name, FLOAT, dims, &raw));
        self
    }

    pub fn i64s(&mut self, name: &str, dims: &[u64], data: &[i64]) -> &mut Self {
        assert_eq!(dims.iter().product::<u64>() as usize, data.len(), "{name}");
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.initializers.push(tensor_raw(name, INT64, dims, &raw));
        self
    }

    /// A float initializer stored in `file` (next to the model) at `offset`; the caller writes the
    /// bytes there.
    pub fn external_f32(&mut self, name: &str, dims: &[u64], file: &str, offset: u64) -> &mut Self {
        let mut t = Vec::new();
        let mut packed = Vec::new();
        for &d in dims {
            varint(&mut packed, d);
        }
        field_bytes(&mut t, 1, &packed);
        field_varint(&mut t, 2, FLOAT);
        field_bytes(&mut t, 8, name.as_bytes());
        let len = dims.iter().product::<u64>() * 4;
        field_bytes(&mut t, 13, &entry("location", file));
        field_bytes(&mut t, 13, &entry("offset", &offset.to_string()));
        field_bytes(&mut t, 13, &entry("length", &len.to_string()));
        field_varint(&mut t, 14, 1);
        self.initializers.push(t);
        self
    }

    pub fn node(&mut self, op: &str, inputs: &[&str], outputs: &[&str], attrs: &[Attr]) -> &mut Self {
        self.node_in("", op, inputs, outputs, attrs)
    }

    pub fn node_in(&mut self, domain: &str, op: &str, inputs: &[&str], outputs: &[&str], attrs: &[Attr]) -> &mut Self {
        let mut n = Vec::new();
        for i in inputs {
            field_bytes(&mut n, 1, i.as_bytes());
        }
        for o in outputs {
            field_bytes(&mut n, 2, o.as_bytes());
        }
        field_bytes(&mut n, 3, format!("{op}_{}", self.nodes.len()).as_bytes());
        field_bytes(&mut n, 4, op.as_bytes());
        for a in attrs {
            field_bytes(&mut n, 5, &attribute(a));
        }
        if !domain.is_empty() {
            field_bytes(&mut n, 7, domain.as_bytes());
        }
        self.nodes.push(n);
        self
    }

    /// Serialized `ModelProto` (IR 8) with the default domain at `opset` plus `extra` domains.
    pub fn model_bytes(&self, opset: u64, extra: &[(&str, u64)]) -> Vec<u8> {
        let mut graph = Vec::new();
        for n in &self.nodes {
            field_bytes(&mut graph, 1, n);
        }
        field_bytes(&mut graph, 2, b"sigil_test_graph");
        for i in &self.initializers {
            field_bytes(&mut graph, 5, i);
        }
        for i in &self.inputs {
            field_bytes(&mut graph, 11, i);
        }
        for o in &self.outputs {
            field_bytes(&mut graph, 12, o);
        }
        let mut model = Vec::new();
        field_varint(&mut model, 1, 8);
        field_bytes(&mut model, 2, b"sigil-test");
        field_bytes(&mut model, 7, &graph);
        let mut op = Vec::new();
        field_varint(&mut op, 2, opset);
        field_bytes(&mut model, 8, &op);
        for (domain, version) in extra {
            let mut op = Vec::new();
            field_bytes(&mut op, 1, domain.as_bytes());
            field_varint(&mut op, 2, *version);
            field_bytes(&mut model, 8, &op);
        }
        model
    }

    pub fn save(&self, path: &Path, opset: u64, extra: &[(&str, u64)]) {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).unwrap();
        }
        std::fs::write(path, self.model_bytes(opset, extra)).unwrap();
    }
}

/// Little-endian f32 bytes.
pub fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}
