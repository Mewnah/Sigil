//! `layout.json` of a template (written by `scripts/vc-templates/export_rvc.py`; schema in
//! `scripts/vc-templates/layout_schema.json`): where every weight of the template graphs lives in
//! `weights.f32.bin` and which checkpoint tensor(s) it comes from.
//!
//! Loading validates the whole table (unique names, aligned non-overlapping ranges inside the file,
//! byte counts that match the shapes, weight_norm dims) and recomputes the `weights_id` the graphs
//! carry, so a hand-edited or mismatched layout is refused before any weight is written.
//!
//! Every graph variant records its parity status per device class (`parity`: `cpu` / `gpu` ->
//! `pass` / `fail`, from the ADR 0004 measurements); `session::template_session` refuses a variant
//! marked `fail` for the target's class unless the caller asks for an experiment, and a variant
//! `recommended` for a class must not be marked `fail` there.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::meta::CONFIG_LEN;
use super::{ErrorCode, ImportError};

pub const LAYOUT_FORMAT: &str = "sigil-rvc-layout";
pub const LAYOUT_VERSION: u32 = 1;
pub const LAYOUT_FILE: &str = "layout.json";
const MAX_LAYOUT_BYTES: u64 = 16 << 20;
const MAX_WEIGHTS_BYTES: u64 = 4 << 30;

#[derive(Clone, Debug, Deserialize)]
pub struct Layout {
    pub format: String,
    pub format_version: u32,
    pub template: String,
    pub arch: Arch,
    pub streaming: Streaming,
    pub weights: Weights,
    pub speakers: Speakers,
    #[serde(default)]
    pub ignored_keys: Vec<String>,
    pub graphs: BTreeMap<String, Graph>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Arch {
    pub version: String,
    pub f0: bool,
    pub vocoder: String,
    pub sample_rate: u32,
    /// Output samples per 10 ms frame.
    pub upp: u32,
    pub frame_rate: u32,
    pub emb_channels: u32,
    pub inter_channels: u32,
    pub gin_channels: u32,
    pub config: Vec<serde_json::Value>,
    pub arch_fields: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Streaming {
    /// Frames at the start of a flowdec window that only feed the flow.
    pub flow_margin: u32,
    /// Frames after those that only feed the decoder (measured receptive field).
    pub dec_margin: u32,
    /// Multiplier of the prior's unit Gaussian noise (`z_noise = N(0, 1) * noise_scale`).
    pub noise_scale: f32,
    pub knn_k: u32,
    pub knn_coarse_k: u32,
    pub min_index_rows: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Weights {
    pub file_f32: String,
    pub file_f16: String,
    pub alignment: u64,
    pub bytes_f32: u64,
    pub weights_id: String,
    pub tensors: Vec<TensorEntry>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TensorEntry {
    /// Initializer name in the graph (the RVC parameter name).
    pub name: String,
    pub graph: String,
    pub shape: Vec<usize>,
    /// Byte range in `weights.f32.bin` (halve both for `weights.f16.bin`).
    pub offset: u64,
    pub bytes: u64,
    pub source: Source,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Source {
    /// Copied as is.
    #[serde(default)]
    pub key: Option<String>,
    /// `w = g * v / ||v||` (norm over every axis but `dim`; -1 = the whole tensor).
    #[serde(default)]
    pub weight_norm: Option<WeightNormSource>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WeightNormSource {
    pub g: String,
    pub v: String,
    pub dim: i64,
    /// The already-folded key some exporters write instead of g / v.
    pub folded: String,
    pub g_shape: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Speakers {
    pub key: String,
    pub dim: usize,
    pub file: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Graph {
    /// Variant to run per device class (`cpu` / `gpu` -> `fp32` / `fp16`).
    #[serde(default)]
    pub recommended: BTreeMap<String, String>,
    pub variants: BTreeMap<String, GraphVariant>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct GraphVariant {
    pub file: String,
    pub sha256: String,
    pub bytes: u64,
    pub weights_file: Option<String>,
    /// Parity against PyTorch per device class (`cpu` / `gpu`); a missing class is untested.
    #[serde(default)]
    pub parity: BTreeMap<String, Parity>,
}

/// Parity of a graph variant on a device class (ADR 0004 gate: fp32 <= 1e-4, fp16 <= 5e-3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Parity {
    Pass,
    /// Misses the gate: never run unless explicitly experimenting.
    Fail,
    Untested,
}

/// Device classes of the parity table.
pub const DEVICE_CLASSES: [&str; 2] = ["cpu", "gpu"];

/// Device class of an execution provider name (`cpu`; every other EP is a `gpu`).
pub fn device_class(ep: &str) -> &'static str {
    if ep.eq_ignore_ascii_case("cpu") {
        "cpu"
    } else {
        "gpu"
    }
}

impl GraphVariant {
    pub fn parity_on(&self, class: &str) -> Parity {
        self.parity.get(class).copied().unwrap_or(Parity::Untested)
    }
}

fn invalid(template: &str, msg: impl Into<String>) -> ImportError {
    ImportError::new(ErrorCode::LayoutInvalid, format!("template {template}: {}", msg.into()))
}

fn safe_file_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'))
        && !name.starts_with('.')
}

/// Canonical id of a tensor table (see `sigil_layout.tensor_table_id` in the exporter): sha256 of
/// `[{"bytes":..,"name":..,"offset":..,"shape":[..]}, ...]` sorted by offset, compact JSON.
pub fn tensor_table_id(tensors: &[TensorEntry]) -> String {
    #[derive(Serialize)]
    struct Canon<'a> {
        bytes: u64,
        name: &'a str,
        offset: u64,
        shape: &'a [usize],
    }
    let mut rows: Vec<Canon<'_>> = tensors
        .iter()
        .map(|t| Canon { bytes: t.bytes, name: &t.name, offset: t.offset, shape: &t.shape })
        .collect();
    rows.sort_by_key(|r| r.offset);
    let blob = serde_json::to_vec(&rows).expect("plain data serialises");
    hex::encode(Sha256::digest(&blob))
}

impl Layout {
    /// `<dir>/layout.json`, validated.
    pub fn load(dir: &Path) -> Result<Layout, ImportError> {
        let path = dir.join(LAYOUT_FILE);
        let meta = std::fs::metadata(&path).map_err(|e| ImportError::new(ErrorCode::NoTemplate, format!("{}: {e}", path.display())))?;
        if meta.len() > MAX_LAYOUT_BYTES {
            return Err(ImportError::new(ErrorCode::LayoutInvalid, format!("{} is too large", path.display())));
        }
        let bytes = std::fs::read(&path).map_err(ImportError::io)?;
        Self::parse(&bytes)
    }

    pub fn parse(bytes: &[u8]) -> Result<Layout, ImportError> {
        let layout: Layout = serde_json::from_slice(bytes)
            .map_err(|e| ImportError::new(ErrorCode::LayoutInvalid, format!("layout.json: {e}")))?;
        layout.validate()?;
        Ok(layout)
    }

    fn validate(&self) -> Result<(), ImportError> {
        let t = self.template.as_str();
        if self.format != LAYOUT_FORMAT || self.format_version != LAYOUT_VERSION {
            return Err(invalid(t, format!("unsupported layout {} v{}", self.format, self.format_version)));
        }
        if t.is_empty() || !t.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-') {
            return Err(invalid(t, "bad template id"));
        }
        if self.arch.config.len() != CONFIG_LEN || self.arch.arch_fields.iter().any(|&i| i >= CONFIG_LEN) {
            return Err(invalid(t, "bad arch config"));
        }
        let a = &self.arch;
        if a.upp == 0 || a.sample_rate == 0 || a.frame_rate.checked_mul(a.upp) != Some(a.sample_rate) {
            return Err(invalid(t, "sample rate, upp and frame rate disagree"));
        }
        if a.emb_channels == 0 || a.inter_channels == 0 || !a.inter_channels.is_multiple_of(2) || a.gin_channels as usize != self.speakers.dim {
            return Err(invalid(t, "bad channel counts"));
        }
        let s = &self.streaming;
        if s.knn_k == 0 || s.knn_k > s.knn_coarse_k || s.knn_coarse_k > s.min_index_rows || s.noise_scale.is_nan() || s.noise_scale < 0.0 {
            return Err(invalid(t, "bad streaming parameters"));
        }
        let w = &self.weights;
        if w.alignment == 0 || !w.alignment.is_multiple_of(4) || w.bytes_f32 > MAX_WEIGHTS_BYTES || !w.bytes_f32.is_multiple_of(4) {
            return Err(invalid(t, "bad weights file geometry"));
        }
        for f in [&w.file_f32, &w.file_f16, &self.speakers.file] {
            if !safe_file_name(f) {
                return Err(invalid(t, format!("unsafe file name {f:?}")));
            }
        }
        let mut names = HashSet::new();
        let mut ranges: Vec<(u64, u64, &str)> = Vec::with_capacity(w.tensors.len());
        for e in &w.tensors {
            if !names.insert(e.name.as_str()) {
                return Err(invalid(t, format!("tensor {} listed twice", e.name)));
            }
            let numel = e
                .shape
                .iter()
                .try_fold(1u64, |acc, &d| if d == 0 { None } else { acc.checked_mul(d as u64) })
                .ok_or_else(|| invalid(t, format!("{}: bad shape {:?}", e.name, e.shape)))?;
            if numel.checked_mul(4) != Some(e.bytes) {
                return Err(invalid(t, format!("{}: {} bytes for {numel} elements", e.name, e.bytes)));
            }
            if e.offset % w.alignment != 0 {
                return Err(invalid(t, format!("{}: offset {} is not {}-aligned", e.name, e.offset, w.alignment)));
            }
            let end = e.offset.checked_add(e.bytes).filter(|&x| x <= w.bytes_f32);
            let Some(end) = end else {
                return Err(invalid(t, format!("{}: range outside the weights file", e.name)));
            };
            ranges.push((e.offset, end, &e.name));
            match (&e.source.key, &e.source.weight_norm) {
                (Some(k), None) if !k.is_empty() => {}
                (None, Some(wn)) => {
                    let rank = e.shape.len() as i64;
                    if wn.dim < -1 || wn.dim >= rank || wn.g.is_empty() || wn.v.is_empty() {
                        return Err(invalid(t, format!("{}: bad weight_norm source", e.name)));
                    }
                    let want: Vec<usize> = (0..e.shape.len())
                        .map(|i| if wn.dim == -1 || i as i64 != wn.dim { 1 } else { e.shape[i] })
                        .collect();
                    if wn.g_shape != want {
                        return Err(invalid(t, format!("{}: g shape {:?}, expected {want:?}", e.name, wn.g_shape)));
                    }
                }
                _ => return Err(invalid(t, format!("{}: needs exactly one of key / weight_norm", e.name))),
            }
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            if pair[1].0 < pair[0].1 {
                return Err(invalid(t, format!("{} overlaps {}", pair[1].2, pair[0].2)));
            }
        }
        if tensor_table_id(&w.tensors) != w.weights_id {
            return Err(invalid(t, "weights_id does not match the tensor table (edited or corrupted layout)"));
        }
        for (graph, g) in &self.graphs {
            for (class, variant) in &g.recommended {
                let ok = DEVICE_CLASSES.contains(&class.as_str())
                    && g.variants.get(variant).is_some_and(|v| v.parity_on(class) != Parity::Fail);
                if !ok {
                    return Err(invalid(t, format!("{graph}: recommended {class} variant {variant} is missing or fails parity there")));
                }
            }
            for (variant, v) in &g.variants {
                if !safe_file_name(&v.file) || v.sha256.len() != 64 {
                    return Err(invalid(t, format!("{graph}/{variant}: bad graph entry")));
                }
                if let Some(class) = v.parity.keys().find(|c| !DEVICE_CLASSES.contains(&c.as_str())) {
                    return Err(invalid(t, format!("{graph}/{variant}: unknown device class {class} in parity")));
                }
                if let Some(wf) = &v.weights_file {
                    if wf != &w.file_f32 && wf != &w.file_f16 {
                        return Err(invalid(t, format!("{graph}/{variant}: unknown weights file {wf}")));
                    }
                }
            }
        }
        Ok(())
    }

    /// Prefix of the f32 weights image a graph needs (all of its tensors end before this).
    pub fn graph_extent_f32(&self, graph: &str) -> u64 {
        self.weights
            .tensors
            .iter()
            .filter(|e| e.graph == graph)
            .map(|e| e.offset + e.bytes)
            .max()
            .unwrap_or(0)
    }

    /// Parity status of `graph.variant` on the device class of execution provider `ep`.
    pub fn parity(&self, graph: &str, variant: &str, ep: &str) -> Parity {
        self.graphs
            .get(graph)
            .and_then(|g| g.variants.get(variant))
            .map_or(Parity::Untested, |v| v.parity_on(device_class(ep)))
    }

    pub fn graph_variant(&self, graph: &str, variant: &str) -> Result<&GraphVariant, ImportError> {
        self.graphs
            .get(graph)
            .and_then(|g| g.variants.get(variant))
            .ok_or_else(|| invalid(&self.template, format!("no {graph}.{variant} graph")))
    }

    /// Whether a checkpoint tensor is consumed by this layout (weights or speaker table) or ignored.
    pub fn uses_key(&self, key: &str) -> bool {
        key == self.speakers.key
            || self.ignored_keys.iter().any(|p| key.starts_with(p.as_str()))
            || self.weights.tensors.iter().any(|e| match (&e.source.key, &e.source.weight_norm) {
                (Some(k), _) => k == key,
                (_, Some(wn)) => wn.g == key || wn.v == key || wn.folded == key,
                _ => false,
            })
    }
}

/// A template folder: `<templates_dir>/<template id>/layout.json` + graphs.
#[derive(Clone, Debug)]
pub struct Template {
    pub dir: PathBuf,
    pub layout: Layout,
}

impl Template {
    pub fn load(templates_dir: &Path, id: &str) -> Result<Template, ImportError> {
        if id.is_empty() || !id.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-') {
            return Err(ImportError::new(ErrorCode::NoTemplate, format!("bad template id {id:?}")));
        }
        let dir = templates_dir.join(id);
        if !dir.join(LAYOUT_FILE).is_file() {
            return Err(ImportError::new(
                ErrorCode::NoTemplate,
                format!("Sigil has no {id} template yet (looked in {})", templates_dir.display()),
            ));
        }
        let layout = Layout::load(&dir)?;
        if layout.template != id {
            return Err(invalid(id, format!("folder holds template {}", layout.template)));
        }
        Ok(Template { dir, layout })
    }

    /// A graph file's bytes, checked against the layout's sha256.
    pub fn graph_bytes(&self, graph: &str, variant: &str) -> Result<Vec<u8>, ImportError> {
        let v = self.layout.graph_variant(graph, variant)?;
        let path = self.dir.join(&v.file);
        let bytes = std::fs::read(&path).map_err(|e| ImportError::new(ErrorCode::NoTemplate, format!("{}: {e}", path.display())))?;
        if bytes.len() as u64 != v.bytes || hex::encode(Sha256::digest(&bytes)) != v.sha256 {
            return Err(invalid(&self.layout.template, format!("{} does not match layout.json", v.file)));
        }
        Ok(bytes)
    }
}
