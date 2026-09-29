//! Test-only ONNX graph builder: hand-encoded protobuf for tiny stand-in models (a fake Kitten
//! speech package, a fake RVC content encoder and generator), so engine paths run on Sigil's real
//! ONNX Runtime in `cargo test` without downloading models or needing the `onnx` Python package.
//!
//! Only what the stand-ins need is supported: float / int64 tensors, fixed or named dimensions,
//! raw-data initializers and int / ints / tensor attributes. Field numbers follow `onnx.proto`.

/// `TensorProto.DataType.FLOAT`.
pub const FLOAT: u64 = 1;
/// `TensorProto.DataType.INT64`.
pub const INT64: u64 = 7;

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

/// A tensor dimension: fixed, or symbolic (`dim_param`).
#[derive(Clone, Copy, Debug)]
pub enum Dim {
    Fixed(u64),
    Named(&'static str),
}

/// A node attribute.
pub enum Attr {
    Int(&'static str, i64),
    Ints(&'static str, Vec<i64>),
    /// A float tensor attribute (e.g. `ConstantOfShape.value`).
    FloatTensor(&'static str, Vec<u64>, Vec<f32>),
}

fn tensor_proto(name: &str, data_type: u64, dims: &[u64], raw: &[u8]) -> Vec<u8> {
    let mut t = Vec::new();
    let mut packed = Vec::new();
    for &d in dims {
        varint(&mut packed, d);
    }
    field_bytes(&mut t, 1, &packed);
    field_varint(&mut t, 2, data_type);
    if !name.is_empty() {
        field_bytes(&mut t, 8, name.as_bytes());
    }
    field_bytes(&mut t, 9, raw);
    t
}

fn value_info(name: &str, elem_type: u64, dims: &[Dim]) -> Vec<u8> {
    let mut shape = Vec::new();
    for d in dims {
        let mut dim = Vec::new();
        match d {
            Dim::Fixed(v) => field_varint(&mut dim, 1, *v),
            Dim::Named(p) => field_bytes(&mut dim, 2, p.as_bytes()),
        }
        field_bytes(&mut shape, 1, &dim);
    }
    let mut tensor = Vec::new();
    field_varint(&mut tensor, 1, elem_type);
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
            field_varint(&mut out, 20, 2); // INT
        }
        Attr::Ints(name, vs) => {
            field_bytes(&mut out, 1, name.as_bytes());
            for v in vs {
                field_varint(&mut out, 8, *v as u64);
            }
            field_varint(&mut out, 20, 7); // INTS
        }
        Attr::FloatTensor(name, dims, data) => {
            field_bytes(&mut out, 1, name.as_bytes());
            let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            field_bytes(&mut out, 5, &tensor_proto("", FLOAT, dims, &raw));
            field_varint(&mut out, 20, 4); // TENSOR
        }
    }
    out
}

/// An ONNX graph under construction (default domain, one opset).
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

    pub fn input(&mut self, name: &str, elem_type: u64, dims: &[Dim]) -> &mut Self {
        self.inputs.push(value_info(name, elem_type, dims));
        self
    }

    pub fn output(&mut self, name: &str, elem_type: u64, dims: &[Dim]) -> &mut Self {
        self.outputs.push(value_info(name, elem_type, dims));
        self
    }

    pub fn init_f32(&mut self, name: &str, dims: &[u64], data: &[f32]) -> &mut Self {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.initializers.push(tensor_proto(name, FLOAT, dims, &raw));
        self
    }

    pub fn init_i64(&mut self, name: &str, dims: &[u64], data: &[i64]) -> &mut Self {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.initializers.push(tensor_proto(name, INT64, dims, &raw));
        self
    }

    pub fn node(&mut self, op: &str, inputs: &[&str], outputs: &[&str], attrs: &[Attr]) -> &mut Self {
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
        self.nodes.push(n);
        self
    }

    /// Serialized `ModelProto` (IR 7) importing the default domain at `opset`.
    pub fn model_bytes(&self, opset: u64) -> Vec<u8> {
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
        let mut op = Vec::new();
        field_varint(&mut op, 2, opset);
        let mut model = Vec::new();
        field_varint(&mut model, 1, 7);
        field_bytes(&mut model, 2, b"sigil-test");
        field_bytes(&mut model, 7, &graph);
        field_bytes(&mut model, 8, &op);
        model
    }
}
