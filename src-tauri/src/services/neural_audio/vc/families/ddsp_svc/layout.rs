//! `layout.json` of a DDSP-SVC template (P5-A; written by
//! `scripts/vc-families/ddsp_svc/export_templates.py`): the architecture the graphs were exported
//! for, the vocoder's mel settings baked into the `ddsp` graph, and where every weight lives in a
//! voice's `weights.f32.bin` (one section per graph, handed to that graph as the in-memory external
//! data file named by the section's `location`) with the checkpoint tensor it comes from.
//!
//! The engine embeds the layout of the template it runs ([`DdspLayout::embedded`]): importing a
//! voice needs no download, and a template installed from the model store must carry the same
//! `weights_id` (a different export is refused by name, never loaded with mismatched weights).
//! [`DdspLayout::parse`] validates the whole table: known graphs, aligned, non-overlapping tensors
//! inside their section, byte counts that match the shapes, sources and transforms that make sense.

use std::collections::{BTreeMap, HashSet};

use serde::Deserialize;

pub const LAYOUT_FORMAT: &str = "sigil-ddsp-layout";
pub const LAYOUT_VERSION: u32 = 1;
pub const LAYOUT_FILE: &str = "layout.json";
/// The graphs a template holds (and their section names in `weights.sections`).
pub const GRAPH_DDSP: &str = "ddsp";
pub const GRAPH_REFLOW: &str = "reflow_step";
const MAX_LAYOUT_BYTES: usize = 4 << 20;
const MAX_WEIGHTS_BYTES: u64 = 2 << 30;
const MAX_TENSORS: usize = 4096;
/// The interface the engine binds (`model.rs`): a template whose graphs take or give anything else
/// is another export and is refused by name.
const DDSP_INPUTS: [&str; 8] = ["hubert", "mel2ph", "f0", "volume", "randn", "spk", "formant_shift", "start_frame"];
const DDSP_OUTPUTS: [&str; 2] = ["x", "cond"];
const REFLOW_INPUTS: [&str; 4] = ["x", "t", "dt", "cond"];
const REFLOW_OUTPUTS: [&str; 1] = ["x_next"];

/// The embedded layout of the default 6.3 template (`vc/ddsp-svc-6.3`).
const EMBEDDED: &str = include_str!("layout.json");

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Arch {
    pub sampling_rate: u32,
    pub block_size: u32,
    pub win_length: u32,
    pub encoder_out_channels: u32,
    pub n_aux_layers: u32,
    pub n_aux_chans: u32,
    pub n_layers: u32,
    pub n_chans: u32,
    pub use_attention: bool,
    /// Mel bins (the vocoder's `num_mels`).
    pub out_dims: u32,
}

/// The vocoder settings the `ddsp` graph's mel was exported with (and the vocoder must have).
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct VocoderConfig {
    pub sampling_rate: u32,
    pub hop_size: u32,
    pub num_mels: u32,
    pub n_fft: u32,
    pub win_size: u32,
    pub fmin: f64,
    pub fmax: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct GraphInfo {
    pub file: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    #[serde(default)]
    pub bytes: u64,
    #[serde(default)]
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Section {
    pub offset: u64,
    pub bytes: u64,
    /// The external-data file name the graph's initializers reference.
    pub location: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct WeightNorm {
    pub g: String,
    pub v: String,
    pub dim: i64,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Source {
    #[serde(default)]
    pub key: Option<String>,
    #[serde(default)]
    pub weight_norm: Option<WeightNorm>,
}

impl Source {
    /// The checkpoint entries this source reads.
    pub fn keys(&self) -> Vec<&str> {
        match (&self.key, &self.weight_norm) {
            (Some(k), None) => vec![k.as_str()],
            (None, Some(wn)) => vec![wn.g.as_str(), wn.v.as_str()],
            _ => Vec::new(),
        }
    }

    /// The name of a weight_norm pair's weight with the norm removed (`X.weight` for
    /// `X.weight_g` / `X.weight_v`): what a checkpoint saved after `remove_weight_norm` holds
    /// instead of the pair. `None` for a plain key.
    pub fn folded_key(&self) -> Option<&str> {
        let wn = self.weight_norm.as_ref()?;
        wn.g.strip_suffix("_g").filter(|k| !k.is_empty())
    }

    /// A name for messages: the key, or the folded weight of a weight_norm pair.
    pub fn label(&self) -> String {
        match (&self.key, &self.weight_norm) {
            (Some(k), _) => k.clone(),
            (_, Some(wn)) => self.folded_key().unwrap_or(&wn.g).to_string(),
            _ => "?".into(),
        }
    }
}

/// How the checkpoint tensor becomes the initializer.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Transform {
    Identity,
    /// A 2-D tensor transposed (a Linear weight constant-folded into a MatMul).
    Transpose,
    /// The same values in another shape (a PReLU slope `[C]` as `[C, 1]`).
    Reshape,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub graph: String,
    pub shape: Vec<usize>,
    /// Byte offset inside the graph's section.
    pub offset: u64,
    pub bytes: u64,
    pub source: Source,
    pub transform: Transform,
    pub source_shape: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Weights {
    pub file: String,
    pub alignment: u64,
    pub bytes: u64,
    pub weights_id: String,
    pub sections: BTreeMap<String, Section>,
    pub tensors: Vec<TensorEntry>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Speakers {
    pub key: String,
    pub dim: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DdspLayout {
    pub format: String,
    pub format_version: u32,
    pub template: String,
    #[serde(default)]
    pub source: String,
    pub arch: Arch,
    pub vocoder: VocoderConfig,
    pub spec_min: f32,
    pub spec_max: f32,
    pub graphs: BTreeMap<String, GraphInfo>,
    pub weights: Weights,
    /// Sources a checkpoint may lack: a zero tensor is written instead (the pitch-augmentation
    /// embedding of a voice trained without it).
    #[serde(default)]
    pub optional_zero: Vec<String>,
    pub speakers: Speakers,
    /// Checkpoint entries no graph reads (constant buffers, unused embeddings).
    #[serde(default)]
    pub ignored_keys: Vec<String>,
}

fn invalid(msg: impl std::fmt::Display) -> String {
    format!("DDSP-SVC layout.json: {msg}")
}

impl DdspLayout {
    /// The layout of the template this build runs.
    pub fn embedded() -> Result<DdspLayout, String> {
        DdspLayout::parse(EMBEDDED)
    }

    /// Parse and validate (see the module docs).
    pub fn parse(text: &str) -> Result<DdspLayout, String> {
        if text.len() > MAX_LAYOUT_BYTES {
            return Err(invalid("too large"));
        }
        let l: DdspLayout = serde_json::from_str(text).map_err(invalid)?;
        l.validate()?;
        Ok(l)
    }

    /// Read `dir/layout.json`.
    pub fn read(dir: &std::path::Path) -> Result<DdspLayout, String> {
        let path = dir.join(LAYOUT_FILE);
        let meta = std::fs::metadata(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if meta.len() > MAX_LAYOUT_BYTES as u64 {
            return Err(invalid(format!("{} is too large", path.display())));
        }
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        DdspLayout::parse(&text)
    }

    fn validate(&self) -> Result<(), String> {
        if self.format != LAYOUT_FORMAT || self.format_version != LAYOUT_VERSION {
            return Err(invalid(format!("format {} v{} (expected {LAYOUT_FORMAT} v{LAYOUT_VERSION})", self.format, self.format_version)));
        }
        let a = &self.arch;
        if a.block_size == 0 || a.win_length == 0 || !a.win_length.is_multiple_of(a.block_size) || a.out_dims == 0 || a.n_aux_chans == 0 {
            return Err(invalid("block_size, win_length, out_dims and n_aux_chans must be set (win_length a multiple of block_size)"));
        }
        if self.vocoder.hop_size != a.block_size || self.vocoder.num_mels != a.out_dims || self.vocoder.sampling_rate != a.sampling_rate {
            return Err(invalid("the vocoder's hop, mel bins and rate must be the model's"));
        }
        if !(self.spec_min.is_finite() && self.spec_max.is_finite() && self.spec_max > self.spec_min) {
            return Err(invalid("spec_min / spec_max"));
        }
        let same_names = |got: &[String], want: &[&str]| {
            let mut a: Vec<&str> = got.iter().map(String::as_str).collect();
            let mut b = want.to_vec();
            a.sort_unstable();
            b.sort_unstable();
            a == b
        };
        for (g, inputs, outputs) in [(GRAPH_DDSP, &DDSP_INPUTS[..], &DDSP_OUTPUTS[..]), (GRAPH_REFLOW, &REFLOW_INPUTS[..], &REFLOW_OUTPUTS[..])] {
            let info = self.graphs.get(g).ok_or_else(|| invalid(format!("graph {g} missing")))?;
            if info.file.is_empty() || info.file.contains(['/', '\\']) || info.file.contains("..") {
                return Err(invalid(format!("graph {g}: bad file name {:?}", info.file)));
            }
            if !same_names(&info.inputs, inputs) || !same_names(&info.outputs, outputs) {
                return Err(invalid(format!(
                    "graph {g} takes {:?} and gives {:?}; this build runs {inputs:?} -> {outputs:?}",
                    info.inputs, info.outputs
                )));
            }
            if !info.sha256.is_empty() && (info.sha256.len() != 64 || !info.sha256.bytes().all(|c| c.is_ascii_hexdigit())) {
                return Err(invalid(format!("graph {g}: sha256")));
            }
            if !self.weights.sections.contains_key(g) {
                return Err(invalid(format!("graph {g} has no weights section")));
            }
        }
        let w = &self.weights;
        if w.alignment == 0 || !w.alignment.is_power_of_two() || w.bytes > MAX_WEIGHTS_BYTES || w.tensors.len() > MAX_TENSORS {
            return Err(invalid("weights: alignment / size / tensor count"));
        }
        if w.file != super::import::WEIGHTS_FILE {
            return Err(invalid(format!("weights.file {:?} (voices keep their weights in {})", w.file, super::import::WEIGHTS_FILE)));
        }
        if w.weights_id.len() != 64 || !w.weights_id.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err(invalid("weights_id"));
        }
        let mut sections: Vec<(&String, &Section)> = w.sections.iter().collect();
        sections.sort_by_key(|(_, s)| s.offset);
        let mut end = 0u64;
        let mut locations = HashSet::new();
        for (name, s) in &sections {
            if s.offset < end || s.offset.checked_add(s.bytes).is_none_or(|e| e > w.bytes) || s.offset % w.alignment != 0 {
                return Err(invalid(format!("section {name} overlaps or leaves the image")));
            }
            if !locations.insert(s.location.as_str()) || s.location.is_empty() || s.location.contains(['/', '\\']) {
                return Err(invalid(format!("section {name}: bad location {:?}", s.location)));
            }
            end = s.offset + s.bytes;
        }
        let mut names = HashSet::new();
        let mut keys = HashSet::new();
        let mut spans: BTreeMap<&str, Vec<(u64, u64)>> = BTreeMap::new();
        for t in &w.tensors {
            let section = w.sections.get(&t.graph).ok_or_else(|| invalid(format!("{}: unknown graph {}", t.name, t.graph)))?;
            if !names.insert((t.graph.as_str(), t.name.as_str())) {
                return Err(invalid(format!("duplicate tensor {}", t.name)));
            }
            let numel: usize = t.shape.iter().product();
            if numel == 0 || t.bytes != numel as u64 * 4 || t.source_shape.iter().product::<usize>() != numel {
                return Err(invalid(format!("{}: {} bytes for shape {:?} / source {:?}", t.name, t.bytes, t.shape, t.source_shape)));
            }
            if t.offset % w.alignment != 0 || t.offset.checked_add(t.bytes).is_none_or(|e| e > section.bytes) {
                return Err(invalid(format!("{}: outside its section", t.name)));
            }
            match t.transform {
                Transform::Identity if t.shape != t.source_shape => return Err(invalid(format!("{}: identity changes the shape", t.name))),
                Transform::Transpose
                    if !(t.source_shape.len() == 2 && t.shape == [t.source_shape[1], t.source_shape[0]]) =>
                {
                    return Err(invalid(format!("{}: transpose of {:?} is not {:?}", t.name, t.source_shape, t.shape)))
                }
                _ => {}
            }
            match (&t.source.key, &t.source.weight_norm) {
                (Some(k), None) if !k.is_empty() => {}
                (None, Some(wn)) if wn.dim >= -1 && wn.dim < t.source_shape.len() as i64 => {}
                _ => return Err(invalid(format!("{}: a source is one key or one weight_norm pair", t.name))),
            }
            for k in t.source.keys() {
                if !keys.insert(k.to_string()) {
                    return Err(invalid(format!("checkpoint tensor {k} feeds two weights")));
                }
            }
            spans.entry(t.graph.as_str()).or_default().push((t.offset, t.offset + t.bytes));
        }
        for (g, mut s) in spans {
            s.sort_unstable();
            if s.windows(2).any(|p| p[1].0 < p[0].1) {
                return Err(invalid(format!("overlapping tensors in {g}")));
            }
        }
        Ok(())
    }

    /// Why `installed` (a template's own `layout.json`) is not the export this layout describes, if
    /// it is not: another weights table, or graphs with other bytes (an older or newer export of the
    /// same architecture, whose graphs the voice's weights do not fit or whose numbers differ).
    pub fn export_mismatch(&self, installed: &DdspLayout) -> Option<String> {
        if installed.weights.weights_id != self.weights.weights_id {
            return Some(format!(
                "its weights table is another export ({}) than this build's ({})",
                installed.source, self.source
            ));
        }
        for (g, want) in &self.graphs {
            let Some(got) = installed.graphs.get(g) else {
                return Some(format!("it has no {g} graph"));
            };
            let differs = (!want.sha256.is_empty() && got.sha256 != want.sha256) || (want.bytes > 0 && got.bytes != want.bytes);
            if differs || got.file != want.file {
                return Some(format!("its {g} graph is another export than this build's"));
            }
        }
        None
    }

    /// The template's graph files in `dir`, checked to exist with the sizes this layout records.
    pub fn graph_files(&self, dir: &std::path::Path) -> Result<BTreeMap<String, std::path::PathBuf>, String> {
        let mut out = BTreeMap::new();
        for (g, info) in &self.graphs {
            let path = dir.join(&info.file);
            let len = std::fs::metadata(&path).map(|m| m.len()).map_err(|_| format!("{} is missing", info.file))?;
            if info.bytes > 0 && len != info.bytes {
                return Err(format!("{} is {len} bytes, the template's is {}", info.file, info.bytes));
            }
            out.insert(g.clone(), path);
        }
        Ok(out)
    }

    pub fn section(&self, graph: &str) -> Option<&Section> {
        self.weights.sections.get(graph)
    }

    /// Every checkpoint key the layout knows (sources, the folded weight a weight_norm pair may be
    /// saved as instead, the speaker table, ignored entries).
    pub fn known_keys(&self) -> HashSet<&str> {
        let mut k: HashSet<&str> = self.weights.tensors.iter().flat_map(|t| t.source.keys()).collect();
        k.extend(self.weights.tensors.iter().filter_map(|t| t.source.folded_key()));
        k.insert(self.speakers.key.as_str());
        k.extend(self.ignored_keys.iter().map(String::as_str));
        k
    }

    /// FFT bins of the DDSP filter (`win_length / 2 + 1`).
    pub fn hop(&self) -> usize {
        self.arch.block_size as usize
    }

    pub fn mels(&self) -> usize {
        self.arch.out_dims as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_layout_is_the_default_63_architecture() {
        let l = DdspLayout::embedded().unwrap();
        assert_eq!(l.template, "ddsp-svc-6.3");
        assert_eq!((l.arch.sampling_rate, l.arch.block_size, l.arch.win_length, l.arch.out_dims), (44_100, 512, 2048, 128));
        assert_eq!((l.arch.n_aux_layers, l.arch.n_aux_chans, l.arch.n_layers, l.arch.n_chans), (6, 512, 6, 1024));
        assert_eq!((l.vocoder.n_fft, l.vocoder.win_size, l.vocoder.fmin, l.vocoder.fmax), (2048, 2048, 40.0, 16_000.0));
        assert_eq!(l.speakers.dim, 512);
        // Every DDSP-SVC weight_norm layer and every LYNXNet2 Linear is in the table.
        let wn = l.weights.tensors.iter().filter(|t| t.source.weight_norm.is_some()).count();
        assert_eq!(wn, 5, "stack.0/2, stack2.0/2, dense_out");
        assert!(l.weights.tensors.iter().any(|t| t.source.key.as_deref() == Some("ddsp_model.unit2ctrl.aug_shift_embed.weight")));
        assert!(l.optional_zero.iter().any(|k| k.ends_with("aug_shift_embed.weight")));
        let ddsp = l.section(GRAPH_DDSP).unwrap();
        let flow = l.section(GRAPH_REFLOW).unwrap();
        assert_eq!(ddsp.offset, 0);
        assert_eq!(flow.offset + flow.bytes, l.weights.bytes);
    }

    #[test]
    fn validation_refuses_bad_tables() {
        let good: serde_json::Value = serde_json::from_str(EMBEDDED).unwrap();
        let edit = |f: &dyn Fn(&mut serde_json::Value)| {
            let mut v = good.clone();
            f(&mut v);
            DdspLayout::parse(&v.to_string())
        };
        assert!(edit(&|_| {}).is_ok());
        assert!(edit(&|v| v["format_version"] = 2.into()).is_err());
        assert!(edit(&|v| v["weights"]["tensors"][0]["bytes"] = 8.into()).is_err());
        assert!(edit(&|v| v["weights"]["tensors"][1]["offset"] = v["weights"]["tensors"][0]["offset"].clone()).is_err());
        assert!(edit(&|v| v["weights"]["tensors"][0]["offset"] = 3.into()).is_err());
        assert!(edit(&|v| v["graphs"]["ddsp"]["file"] = "../x.onnx".into()).is_err());
        assert!(edit(&|v| v["weights"]["weights_id"] = "nope".into()).is_err());
        assert!(edit(&|v| v["weights"]["tensors"][0]["source"] = serde_json::json!({})).is_err());
        assert!(edit(&|v| {
            let first = v["weights"]["tensors"][0]["source"].clone();
            v["weights"]["tensors"][1]["source"] = first;
        })
        .is_err());
        assert!(edit(&|v| v["arch"]["block_size"] = 500.into()).is_err());
        // The engine's graph interface and the voice's weights file.
        let e = edit(&|v| v["graphs"]["ddsp"]["inputs"][1] = "mel2ph_v2".into()).unwrap_err();
        assert!(e.contains("graph ddsp takes"), "{e}");
        assert!(edit(&|v| v["graphs"]["reflow_step"]["outputs"] = serde_json::json!(["x_next", "v"])).is_err());
        assert!(edit(&|v| v["graphs"]["ddsp"]["sha256"] = "zz".into()).is_err());
        assert!(edit(&|v| v["weights"]["file"] = "weights.bin".into()).is_err());
        // Input order does not matter (the engine binds by name).
        assert!(edit(&|v| v["graphs"]["reflow_step"]["inputs"] = serde_json::json!(["cond", "dt", "t", "x"])).is_ok());
    }

    #[test]
    fn export_mismatch_names_what_differs() {
        let base = DdspLayout::embedded().unwrap();
        assert_eq!(base.export_mismatch(&base), None);
        let mut other = base.clone();
        other.graphs.get_mut(GRAPH_DDSP).unwrap().sha256 = "0".repeat(64);
        assert!(base.export_mismatch(&other).unwrap().contains("ddsp graph"));
        let mut other = base.clone();
        other.weights.weights_id = "f".repeat(64);
        assert!(base.export_mismatch(&other).unwrap().contains("weights table"));
        // graph_files: present with the recorded sizes.
        let dir = std::env::temp_dir().join(format!("sigil-ddsp-layout-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut tiny = base.clone();
        for info in tiny.graphs.values_mut() {
            info.bytes = 3;
            std::fs::write(dir.join(&info.file), b"abc").unwrap();
        }
        assert_eq!(tiny.graph_files(&dir).unwrap().len(), 2);
        std::fs::write(dir.join(&tiny.graphs[GRAPH_REFLOW].file), b"abcd").unwrap();
        assert!(tiny.graph_files(&dir).unwrap_err().contains("4 bytes"));
        std::fs::remove_file(dir.join(&tiny.graphs[GRAPH_DDSP].file)).unwrap();
        assert!(tiny.graph_files(&dir).unwrap_err().contains("missing"));
        let _ = std::fs::remove_dir_all(dir);
    }
}
