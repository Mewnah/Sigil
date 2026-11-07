//! Voice model files of the `sbv2` engine: AivisSpeech `.aivmx` (an ONNX model whose metadata
//! carries the AIVM manifest, the Style-Bert-VITS2 hyper-parameters and the style vectors) and
//! plain Style-Bert-VITS2 ONNX exports (`model.onnx` + `style_vectors.json` / `.npy`, optionally
//! `config.json`, as `sbv2-api`'s converter writes them).
//!
//! Nothing here runs or unpickles anything: the ONNX file is only scanned for its top-level
//! `metadata_props` (protobuf field 14 of `ModelProto`), skipping the graph without reading it, and
//! the style vectors are a `.npy` / JSON float matrix. ONNX Runtime parses the graph when the
//! voice loads, as for every other ONNX voice in Sigil. The AIVM format is specified in
//! Aivis Project's `aivmlib` (MIT); field names follow its manifest schema 1.0.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use base64::Engine as _;
use serde::Deserialize;

/// Largest metadata value read from a model (manifests carry icons and voice samples as data URLs).
const MAX_META_VALUE: u64 = 64 * 1024 * 1024;
/// Largest sum of metadata values.
const MAX_META_TOTAL: u64 = 160 * 1024 * 1024;
/// Most metadata entries read.
const MAX_META_ENTRIES: usize = 256;
/// Largest style matrix accepted (styles × width floats).
const MAX_STYLE_VALUES: usize = 1 << 20;
/// Most speakers / styles listed for one model.
const MAX_SPEAKERS: usize = 1024;

/// ONNX `ModelProto` field numbers.
const FIELD_GRAPH: u64 = 7;
const FIELD_METADATA_PROPS: u64 = 14;

// -------------------------------------------------------------------------------------------------
// ONNX metadata scan
// -------------------------------------------------------------------------------------------------

fn read_varint<R: Read>(r: &mut R) -> Result<Option<u64>, String> {
    let mut value = 0u64;
    for i in 0..10 {
        let mut b = [0u8; 1];
        match r.read(&mut b) {
            Ok(0) if i == 0 => return Ok(None),
            Ok(0) => return Err("the file ends inside a field".into()),
            Ok(_) => {}
            Err(e) => return Err(format!("cannot read the model: {e}")),
        }
        value |= u64::from(b[0] & 0x7f) << (7 * i);
        if b[0] & 0x80 == 0 {
            return Ok(Some(value));
        }
    }
    Err("the file is not an ONNX model (malformed field)".into())
}

fn varint_in(buf: &[u8], at: &mut usize) -> Result<u64, String> {
    let mut value = 0u64;
    for i in 0..10 {
        let b = *buf.get(*at).ok_or("truncated metadata entry")?;
        *at += 1;
        value |= u64::from(b & 0x7f) << (7 * i);
        if b & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err("malformed metadata entry".into())
}

/// `StringStringEntryProto` → (key, value).
fn parse_entry(buf: &[u8]) -> Result<(String, String), String> {
    let (mut key, mut value) = (String::new(), String::new());
    let mut at = 0usize;
    while at < buf.len() {
        let tag = varint_in(buf, &mut at)?;
        match (tag >> 3, tag & 7) {
            (field, 2) => {
                let len = varint_in(buf, &mut at)? as usize;
                let end = at.checked_add(len).filter(|e| *e <= buf.len()).ok_or("truncated metadata entry")?;
                let s = String::from_utf8_lossy(&buf[at..end]).into_owned();
                match field {
                    1 => key = s,
                    2 => value = s,
                    _ => {}
                }
                at = end;
            }
            (_, 0) => {
                varint_in(buf, &mut at)?;
            }
            _ => return Err("malformed metadata entry".into()),
        }
    }
    Ok((key, value))
}

/// The `metadata_props` of an ONNX model file (every other top-level field is skipped without
/// reading it). `Err` when the file is not a protobuf `ModelProto` with a graph.
pub fn read_onnx_metadata<R: Read + Seek>(mut r: R) -> Result<BTreeMap<String, String>, String> {
    let len = r.seek(SeekFrom::End(0)).map_err(|e| format!("cannot read the model: {e}"))?;
    r.seek(SeekFrom::Start(0)).map_err(|e| format!("cannot read the model: {e}"))?;
    let mut r = std::io::BufReader::with_capacity(64 * 1024, r);
    let mut out = BTreeMap::new();
    let mut total = 0u64;
    let mut graph = false;
    let not_onnx = || "the file is not an ONNX model".to_string();
    while let Some(tag) = read_varint(&mut r)? {
        let (field, wire) = (tag >> 3, tag & 7);
        if field == 0 || field > 64 {
            return Err(not_onnx());
        }
        match wire {
            0 => {
                read_varint(&mut r)?.ok_or_else(not_onnx)?;
            }
            1 => {
                r.seek_relative(8).map_err(|e| e.to_string())?;
            }
            5 => {
                r.seek_relative(4).map_err(|e| e.to_string())?;
            }
            2 => {
                let n = read_varint(&mut r)?.ok_or_else(not_onnx)?;
                let here = r.stream_position().map_err(|e| e.to_string())?;
                if here.checked_add(n).is_none_or(|end| end > len) {
                    return Err(not_onnx());
                }
                if field == FIELD_METADATA_PROPS {
                    if n > MAX_META_VALUE || out.len() >= MAX_META_ENTRIES {
                        return Err("the model's metadata is too large".into());
                    }
                    total += n;
                    if total > MAX_META_TOTAL {
                        return Err("the model's metadata is too large".into());
                    }
                    let mut buf = vec![0u8; n as usize];
                    r.read_exact(&mut buf).map_err(|e| format!("cannot read the model: {e}"))?;
                    let (k, v) = parse_entry(&buf)?;
                    out.insert(k, v);
                } else {
                    if field == FIELD_GRAPH {
                        graph = true;
                    }
                    r.seek_relative(n as i64).map_err(|e| e.to_string())?;
                }
            }
            _ => return Err(not_onnx()),
        }
    }
    if !graph {
        return Err(not_onnx());
    }
    Ok(out)
}

// -------------------------------------------------------------------------------------------------
// AIVM manifest and hyper-parameters
// -------------------------------------------------------------------------------------------------

/// AIVM manifest (`aivm_manifest`, version 1.0; unknown fields ignored).
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct AivmManifest {
    pub name: String,
    pub description: String,
    pub creators: Vec<String>,
    pub model_architecture: String,
    pub model_format: String,
    pub uuid: String,
    pub version: String,
    pub speakers: Vec<AivmSpeaker>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct AivmSpeaker {
    pub name: String,
    pub local_id: u32,
    pub supported_languages: Vec<String>,
    pub styles: Vec<AivmStyle>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct AivmStyle {
    pub name: String,
    pub local_id: u32,
}

/// The parts of Style-Bert-VITS2's `config.json` (`aivm_hyper_parameters`) the engine reads.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct HyperParameters {
    pub model_name: String,
    pub version: String,
    pub data: HyperData,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct HyperData {
    pub use_jp_extra: Option<bool>,
    pub sampling_rate: u32,
    pub spk2id: BTreeMap<String, u32>,
    pub style2id: BTreeMap<String, u32>,
}

impl Default for HyperData {
    fn default() -> Self {
        Self { use_jp_extra: None, sampling_rate: 44_100, spk2id: BTreeMap::new(), style2id: BTreeMap::new() }
    }
}

impl HyperParameters {
    /// JP-Extra (`version` ends in `JP-Extra`, or `data.use_jp_extra`).
    pub fn jp_extra(&self) -> Option<bool> {
        if self.version.to_ascii_lowercase().ends_with("jp-extra") {
            return Some(true);
        }
        self.data.use_jp_extra
    }
}

// -------------------------------------------------------------------------------------------------
// Style vectors
// -------------------------------------------------------------------------------------------------

/// Style vectors: row 0 is the mean ("Neutral"), one row per style id.
#[derive(Clone, Debug, PartialEq)]
pub struct StyleVectors {
    pub rows: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

impl StyleVectors {
    fn new(rows: usize, width: usize, data: Vec<f32>) -> Result<Self, String> {
        if rows == 0 || width == 0 || data.len() != rows * width || data.len() > MAX_STYLE_VALUES {
            return Err(format!("the style vectors have an unexpected shape ({rows} × {width})"));
        }
        if data.iter().any(|v| !v.is_finite()) {
            return Err("the style vectors hold non-finite values".into());
        }
        Ok(Self { rows, width, data })
    }

    /// Style `id` blended with the mean by `weight` (`mean + (style − mean) × weight`), as
    /// Style-Bert-VITS2's `get_style_vector`.
    pub fn vector(&self, id: usize, weight: f32) -> Result<Vec<f32>, String> {
        if id >= self.rows {
            return Err(format!("style {id} does not exist (this voice has {} styles)", self.rows));
        }
        let mean = &self.data[..self.width];
        let style = &self.data[id * self.width..(id + 1) * self.width];
        Ok(mean.iter().zip(style).map(|(m, s)| m + (s - m) * weight).collect())
    }
}

/// A `.npy` float matrix (versions 1–3, little-endian f4 / f8, C or Fortran order).
pub fn parse_npy(bytes: &[u8]) -> Result<StyleVectors, String> {
    let bad = |why: &str| format!("the style vectors are not a valid .npy matrix ({why})");
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(bad("no NumPy header"));
    }
    let (header_len, start): (usize, usize) = match bytes[6] {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        2 | 3 if bytes.len() >= 12 => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
        _ => return Err(bad("unsupported version")),
    };
    let end = start.checked_add(header_len).filter(|e| *e <= bytes.len()).ok_or_else(|| bad("truncated header"))?;
    let header = std::str::from_utf8(&bytes[start..end]).map_err(|_| bad("header is not text"))?;
    let field = |name: &str| -> Option<&str> {
        let at = header.find(&format!("'{name}'"))? + name.len() + 2;
        let rest = header[at..].trim_start().strip_prefix(':')?.trim_start();
        Some(rest)
    };
    let descr = field("descr").and_then(|r| r.strip_prefix('\'')).and_then(|r| r.split('\'').next()).ok_or_else(|| bad("no dtype"))?;
    let fortran = field("fortran_order").is_some_and(|r| r.starts_with("True"));
    let shape_text = field("shape")
        .and_then(|r| r.strip_prefix('('))
        .and_then(|r| r.split(')').next())
        .ok_or_else(|| bad("no shape"))?;
    let shape: Vec<usize> = shape_text
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_end_matches('L').parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|_| bad("bad shape"))?;
    let [rows, width] = shape[..] else {
        return Err(bad("not a 2-D matrix"));
    };
    let count = rows.checked_mul(width).filter(|c| *c <= MAX_STYLE_VALUES).ok_or_else(|| bad("too large"))?;
    let payload = &bytes[end..];
    let values: Vec<f32> = match descr {
        "<f4" | "|f4" => {
            if payload.len() < count * 4 {
                return Err(bad("truncated data"));
            }
            payload[..count * 4].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
        }
        "<f8" => {
            if payload.len() < count * 8 {
                return Err(bad("truncated data"));
            }
            payload[..count * 8]
                .as_chunks::<8>().0.iter()
                .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
                .collect()
        }
        other => return Err(bad(&format!("dtype {other}"))),
    };
    let data = if fortran {
        let mut c = vec![0f32; count];
        for r in 0..rows {
            for k in 0..width {
                c[r * width + k] = values[k * rows + r];
            }
        }
        c
    } else {
        values
    };
    StyleVectors::new(rows, width, data)
}

/// `sbv2-api`'s `style_vectors.json`: `{"shape": [rows, width], "data": [[...], ...]}`.
pub fn parse_style_json(bytes: &[u8]) -> Result<StyleVectors, String> {
    #[derive(Deserialize)]
    struct Json {
        shape: [usize; 2],
        data: Vec<Vec<f32>>,
    }
    let j: Json = serde_json::from_slice(bytes).map_err(|e| format!("the style vectors JSON is not valid: {e}"))?;
    let [rows, width] = j.shape;
    if j.data.len() != rows || j.data.iter().any(|r| r.len() != width) {
        return Err("the style vectors JSON does not match its shape".into());
    }
    StyleVectors::new(rows, width, j.data.into_iter().flatten().collect())
}

// -------------------------------------------------------------------------------------------------
// A voice model's description
// -------------------------------------------------------------------------------------------------

/// One style of a speaker.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StyleInfo {
    pub name: String,
    /// Row of the style vectors.
    pub id: u32,
}

/// One speaker of a model.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerInfo {
    pub name: String,
    /// The model's `sid`.
    pub id: u32,
    pub styles: Vec<StyleInfo>,
    /// BCP-47 languages (AIVM; `["ja"]` when unknown).
    pub languages: Vec<String>,
}

/// What a voice model file says about itself.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelDescription {
    pub name: String,
    pub description: String,
    pub creators: Vec<String>,
    pub version: String,
    /// AIVM model UUID ("" for plain exports).
    pub uuid: String,
    /// `Style-Bert-VITS2 (JP-Extra)` or `Style-Bert-VITS2` (from the manifest or the config).
    pub architecture: String,
    pub jp_extra: bool,
    pub sample_rate: u32,
    pub speakers: Vec<SpeakerInfo>,
    pub styles: StyleVectors,
}

const ARCH_JP_EXTRA: &str = "Style-Bert-VITS2 (JP-Extra)";
const ARCH_SBV2: &str = "Style-Bert-VITS2";

fn clean(s: &str, max: usize) -> String {
    s.chars().filter(|c| !c.is_control()).take(max).collect::<String>().trim().to_string()
}

/// Speakers and styles from SBV2 `spk2id` / `style2id` maps (sorted by id; styles shared by every
/// speaker, `Neutral` first as in AivisSpeech).
fn speakers_from_maps(spk2id: &BTreeMap<String, u32>, style2id: &BTreeMap<String, u32>, fallback: &str, rows: usize) -> Vec<SpeakerInfo> {
    let mut styles: Vec<StyleInfo> = style2id
        .iter()
        .filter(|(_, id)| (**id as usize) < rows)
        .map(|(n, id)| StyleInfo { name: clean(n, 40), id: *id })
        .collect();
    if styles.is_empty() {
        styles = (0..rows.min(64) as u32)
            .map(|i| StyleInfo { name: if i == 0 { "Neutral".into() } else { format!("Style {i}") }, id: i })
            .collect();
    }
    styles.sort_by_key(|s| s.id);
    let mut speakers: Vec<SpeakerInfo> = spk2id
        .iter()
        .take(MAX_SPEAKERS)
        .map(|(n, id)| SpeakerInfo { name: clean(n, 80), id: *id, styles: styles.clone(), languages: vec!["ja".into()] })
        .collect();
    if speakers.is_empty() {
        speakers.push(SpeakerInfo { name: clean(fallback, 80), id: 0, styles, languages: vec!["ja".into()] });
    }
    speakers.sort_by_key(|s| s.id);
    speakers
}

/// Describe an `.aivmx` from its ONNX metadata.
pub fn describe_aivmx(meta: &BTreeMap<String, String>, file_stem: &str) -> Result<ModelDescription, String> {
    let manifest_json = meta
        .get("aivm_manifest")
        .ok_or("this ONNX file has no AIVM manifest; it is not an .aivmx voice")?;
    let manifest: AivmManifest =
        serde_json::from_str(manifest_json).map_err(|e| format!("the AIVM manifest is not valid: {e}"))?;
    if !manifest.model_format.is_empty() && !manifest.model_format.eq_ignore_ascii_case("ONNX") {
        return Err(format!("the AIVM manifest says {:?}, not ONNX", manifest.model_format));
    }
    let arch = manifest.model_architecture.trim();
    if !arch.is_empty() && !arch.starts_with(ARCH_SBV2) {
        return Err(format!("{arch} voices are not supported (Style-Bert-VITS2 voices only)"));
    }
    let hyper: HyperParameters = match meta.get("aivm_hyper_parameters") {
        Some(j) => serde_json::from_str(j).map_err(|e| format!("the voice's hyper-parameters are not valid: {e}"))?,
        None => HyperParameters::default(),
    };
    let styles_b64 = meta.get("aivm_style_vectors").ok_or("the .aivmx file holds no style vectors")?;
    let npy = base64::engine::general_purpose::STANDARD
        .decode(styles_b64.trim())
        .map_err(|e| format!("the style vectors are not valid base64: {e}"))?;
    let styles = parse_npy(&npy)?;
    let jp_extra = if arch.is_empty() { hyper.jp_extra().unwrap_or(true) } else { arch == ARCH_JP_EXTRA };

    let mut speakers: Vec<SpeakerInfo> = manifest
        .speakers
        .iter()
        .take(MAX_SPEAKERS)
        .map(|s| {
            let mut st: Vec<StyleInfo> = s
                .styles
                .iter()
                .filter(|x| (x.local_id as usize) < styles.rows)
                .map(|x| StyleInfo { name: clean(&x.name, 40), id: x.local_id })
                .collect();
            st.sort_by_key(|x| x.id);
            st.dedup_by_key(|x| x.id);
            let languages: Vec<String> = s.supported_languages.iter().map(|l| clean(l, 35)).filter(|l| !l.is_empty()).collect();
            SpeakerInfo {
                name: clean(&s.name, 80),
                id: s.local_id,
                styles: st,
                languages: if languages.is_empty() { vec!["ja".into()] } else { languages },
            }
        })
        .filter(|s| !s.styles.is_empty())
        .collect();
    speakers.sort_by_key(|s| s.id);
    if speakers.is_empty() {
        speakers = speakers_from_maps(&hyper.data.spk2id, &hyper.data.style2id, &manifest.name, styles.rows);
    }
    let name = Some(clean(&manifest.name, 80)).filter(|n| !n.is_empty()).unwrap_or_else(|| clean(file_stem, 80));
    Ok(ModelDescription {
        name,
        description: clean(&manifest.description, 280),
        creators: manifest.creators.iter().take(16).map(|c| clean(c, 255)).collect(),
        version: clean(&manifest.version, 64),
        uuid: clean(&manifest.uuid, 64),
        architecture: if jp_extra { ARCH_JP_EXTRA.into() } else { ARCH_SBV2.into() },
        jp_extra,
        sample_rate: if hyper.data.sampling_rate > 0 { hyper.data.sampling_rate } else { 44_100 },
        speakers,
        styles,
    })
}

/// Describe a plain SBV2 ONNX export from its style vectors and optional `config.json`.
pub fn describe_export(styles: StyleVectors, config: Option<&[u8]>, name: &str) -> Result<ModelDescription, String> {
    let hyper: HyperParameters = match config {
        Some(b) => serde_json::from_slice(b).map_err(|e| format!("config.json is not a Style-Bert-VITS2 config: {e}"))?,
        None => HyperParameters::default(),
    };
    let jp_extra = hyper.jp_extra().unwrap_or(true);
    let name = Some(clean(&hyper.model_name, 80)).filter(|n| !n.is_empty() && n != "Dummy").unwrap_or_else(|| clean(name, 80));
    let speakers = speakers_from_maps(&hyper.data.spk2id, &hyper.data.style2id, &name, styles.rows);
    Ok(ModelDescription {
        name,
        description: String::new(),
        creators: Vec::new(),
        version: String::new(),
        uuid: String::new(),
        architecture: if jp_extra { ARCH_JP_EXTRA.into() } else { ARCH_SBV2.into() },
        jp_extra,
        sample_rate: if hyper.data.sampling_rate > 0 { hyper.data.sampling_rate } else { 44_100 },
        speakers,
        styles,
    })
}

/// Describe the `.aivmx` at `path` (metadata scan only).
pub fn describe_aivmx_file(path: &Path) -> Result<ModelDescription, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let meta = read_onnx_metadata(file)?;
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    describe_aivmx(&meta, &stem)
}

// -------------------------------------------------------------------------------------------------
// Test fixtures (shared with the engine tests)
// -------------------------------------------------------------------------------------------------

/// Append ONNX `metadata_props` entries to serialized `ModelProto` bytes (protobuf fields may come
/// in any order), for test `.aivmx` files.
#[cfg(test)]
pub(crate) fn with_metadata(mut model: Vec<u8>, entries: &[(&str, &str)]) -> Vec<u8> {
    fn varint(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }
    fn bytes_field(out: &mut Vec<u8>, field: u64, b: &[u8]) {
        varint(out, (field << 3) | 2);
        varint(out, b.len() as u64);
        out.extend_from_slice(b);
    }
    for (k, v) in entries {
        let mut entry = Vec::new();
        bytes_field(&mut entry, 1, k.as_bytes());
        bytes_field(&mut entry, 2, v.as_bytes());
        bytes_field(&mut model, FIELD_METADATA_PROPS, &entry);
    }
    model
}

/// `.npy` v1 bytes of a little-endian f32 C-order matrix.
#[cfg(test)]
pub(crate) fn npy_f32(rows: usize, width: usize, values: &[f32]) -> Vec<u8> {
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {width}), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// An AIVM manifest for tests: one speaker per name with the given styles.
#[cfg(test)]
pub(crate) fn test_manifest(name: &str, speakers: &[(&str, &[&str])], jp_extra: bool) -> String {
    let speakers: Vec<serde_json::Value> = speakers
        .iter()
        .enumerate()
        .map(|(i, (n, styles))| {
            serde_json::json!({
                "name": n,
                "icon": "data:image/png;base64,AAAA",
                "supported_languages": ["ja"],
                "uuid": format!("00000000-0000-0000-0000-00000000000{i}"),
                "local_id": i,
                "styles": styles.iter().enumerate().map(|(k, s)| serde_json::json!({"name": s, "local_id": k, "voice_samples": []})).collect::<Vec<_>>(),
            })
        })
        .collect();
    serde_json::json!({
        "manifest_version": "1.0",
        "name": name,
        "description": "test voice",
        "creators": ["Sigil tests"],
        "license": null,
        "model_architecture": if jp_extra { ARCH_JP_EXTRA } else { ARCH_SBV2 },
        "model_format": "ONNX",
        "uuid": "11111111-2222-3333-4444-555555555555",
        "version": "1.0.0",
        "speakers": speakers,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn tiny_model() -> Vec<u8> {
        use crate::services::neural_audio::test_onnx::{Dim, Graph, FLOAT};
        let mut g = Graph::new();
        g.input("x", FLOAT, &[Dim::Fixed(1)]).output("y", FLOAT, &[Dim::Fixed(1)]).node("Identity", &["x"], &["y"], &[]);
        g.model_bytes(13)
    }

    #[test]
    fn metadata_is_read_past_the_graph() {
        let model = with_metadata(tiny_model(), &[("aivm_manifest", "{\"name\":\"x\"}"), ("other", "値")]);
        let meta = read_onnx_metadata(Cursor::new(model)).unwrap();
        assert_eq!(meta.get("aivm_manifest").map(String::as_str), Some("{\"name\":\"x\"}"));
        assert_eq!(meta.get("other").map(String::as_str), Some("値"));
        // A model without metadata has none; anything else is refused.
        assert!(read_onnx_metadata(Cursor::new(tiny_model())).unwrap().is_empty());
        assert!(read_onnx_metadata(Cursor::new(b"PK\x03\x04 not onnx at all".to_vec())).is_err());
        assert!(read_onnx_metadata(Cursor::new(Vec::new())).is_err());
        // A length past the end of the file is refused, not read.
        let mut bad = with_metadata(tiny_model(), &[("k", "v")]);
        bad.truncate(bad.len() - 1);
        assert!(read_onnx_metadata(Cursor::new(bad)).is_err());
    }

    #[test]
    fn npy_matrices_parse_in_both_orders() {
        let v = npy_f32(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let s = parse_npy(&v).unwrap();
        assert_eq!((s.rows, s.width), (2, 3));
        assert_eq!(s.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Fortran order is transposed into rows.
        let mut f = v.clone();
        let at = f.windows(5).position(|w| w == b"False").unwrap();
        f.splice(at..at + 5, b"True ".iter().copied());
        let s = parse_npy(&f).unwrap();
        assert_eq!(s.data, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
        assert!(parse_npy(b"\x93NUMPY\x01\x00").is_err());
        assert!(parse_npy(&v[..v.len() - 4]).is_err(), "truncated data");
        assert!(parse_npy(&npy_f32(1, 1, &[f32::NAN])).is_err());
    }

    #[test]
    fn style_vectors_blend_with_the_mean() {
        let s = StyleVectors::new(2, 2, vec![1.0, 1.0, 3.0, -1.0]).unwrap();
        assert_eq!(s.vector(0, 5.0).unwrap(), [1.0, 1.0], "the mean ignores the weight");
        assert_eq!(s.vector(1, 1.0).unwrap(), [3.0, -1.0]);
        assert_eq!(s.vector(1, 0.5).unwrap(), [2.0, 0.0]);
        assert!(s.vector(2, 1.0).is_err());
        let j = parse_style_json(br#"{"shape":[2,2],"data":[[1,1],[3,-1]]}"#).unwrap();
        assert_eq!(j, s);
        assert!(parse_style_json(br#"{"shape":[2,2],"data":[[1,1]]}"#).is_err());
    }

    #[test]
    fn aivmx_descriptions_come_from_the_manifest() {
        let npy = npy_f32(3, 2, &[0.0; 6]);
        let manifest = test_manifest("まお", &[("まお", &["ノーマル", "あまあま", "ささやき"]), ("Second", &["ノーマル"])], true);
        let hyper = r#"{"model_name":"mao","version":"2.0-JP-Extra","data":{"sampling_rate":44100,"spk2id":{"mao":0},"style2id":{"Neutral":0}}}"#;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&npy);
        let meta: BTreeMap<String, String> = [
            ("aivm_manifest".to_string(), manifest),
            ("aivm_hyper_parameters".to_string(), hyper.to_string()),
            ("aivm_style_vectors".to_string(), b64.clone()),
        ]
        .into_iter()
        .collect();
        let d = describe_aivmx(&meta, "file").unwrap();
        assert_eq!(d.name, "まお");
        assert!(d.jp_extra);
        assert_eq!(d.sample_rate, 44_100);
        assert_eq!(d.speakers.len(), 2);
        assert_eq!(d.speakers[0].styles.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["ノーマル", "あまあま", "ささやき"]);
        assert_eq!(d.speakers[1].id, 1);
        assert_eq!(d.styles.rows, 3);

        // Older multilingual voices are recognised; other architectures and formats are refused.
        let mut m2 = meta.clone();
        m2.insert("aivm_manifest".into(), test_manifest("x", &[("x", &["N"])], false));
        assert!(!describe_aivmx(&m2, "f").unwrap().jp_extra);
        let mut m3 = meta.clone();
        m3.insert("aivm_manifest".into(), m3["aivm_manifest"].replace("\"ONNX\"", "\"Safetensors\""));
        assert!(describe_aivmx(&m3, "f").is_err());
        let mut m4 = meta.clone();
        m4.remove("aivm_style_vectors");
        assert!(describe_aivmx(&m4, "f").is_err());
        assert!(describe_aivmx(&BTreeMap::new(), "f").is_err(), "no manifest = not an AIVMX voice");
    }

    #[test]
    fn exports_describe_from_their_config() {
        let styles = StyleVectors::new(2, 2, vec![0.0; 4]).unwrap();
        let config = br#"{"model_name":"tsukuyomi","version":"2.0-JP-Extra","data":{"sampling_rate":44100,"spk2id":{"tsukuyomi":0},"style2id":{"Neutral":0,"Happy":1}}}"#;
        let d = describe_export(styles.clone(), Some(config), "folder").unwrap();
        assert_eq!(d.name, "tsukuyomi");
        assert_eq!(d.speakers[0].styles.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["Neutral", "Happy"]);
        // Without a config: one speaker named after the folder, styles by number.
        let d = describe_export(styles, None, "my voice").unwrap();
        assert_eq!(d.name, "my voice");
        assert_eq!(d.speakers[0].styles.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["Neutral", "Style 1"]);
        assert!(d.jp_extra);
    }
}
