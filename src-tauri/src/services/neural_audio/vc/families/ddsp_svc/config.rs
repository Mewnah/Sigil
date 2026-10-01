//! A DDSP-SVC voice's `config.yaml` (P5-A): the YAML subset upstream's configs use, parsed without a
//! YAML crate (the file is untrusted: it is read in the isolated import worker, bounded, and never
//! evaluated), and the typed fields the importer checks against the template.
//!
//! The subset: `key: value` mappings nested by indentation, block sequences (`- item`), flow
//! sequences (`[a, b]`), `#` comments, single- and double-quoted strings, and plain scalars
//! (integers, floats, `true` / `false` / `yes` / `no`, `null` / `~`, anything else a string).
//! Anchors, aliases, tags, multi-line strings and documents are refused by name.

use serde_json::{Map, Value};

/// Largest `config.yaml` accepted (upstream's are about 2 KB).
pub const MAX_CONFIG_BYTES: usize = 64 << 10;
/// Deepest nesting accepted.
const MAX_DEPTH: usize = 8;
/// Most lines accepted.
const MAX_LINES: usize = 4000;

/// One significant line: its indentation and its text without the comment.
struct Line<'a> {
    no: usize,
    indent: usize,
    text: &'a str,
}

fn err(no: usize, msg: impl std::fmt::Display) -> String {
    format!("config.yaml line {no}: {msg}")
}

/// Cut a `#` comment that is not inside quotes.
fn strip_comment(s: &str) -> &str {
    let (mut single, mut double) = (false, false);
    let mut prev_space = true;
    for (i, c) in s.char_indices() {
        match c {
            '\'' if !double => single = !single,
            '"' if !single => double = !double,
            '#' if !single && !double && prev_space => return &s[..i],
            _ => {}
        }
        prev_space = c.is_whitespace();
    }
    s
}

fn lines(text: &str) -> Result<Vec<Line<'_>>, String> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let no = i + 1;
        if no > MAX_LINES {
            return Err(format!("config.yaml has more than {MAX_LINES} lines"));
        }
        let raw = raw.strip_suffix('\r').unwrap_or(raw);
        if raw.trim_start().starts_with('\t') || raw.starts_with('\t') {
            return Err(err(no, "tabs are not allowed for indentation"));
        }
        let body = strip_comment(raw).trim_end();
        let trimmed = body.trim_start();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "---" || trimmed == "..." || trimmed.starts_with('%') {
            if out.is_empty() && trimmed == "---" {
                continue;
            }
            return Err(err(no, "multiple YAML documents and directives are not supported"));
        }
        out.push(Line { no, indent: body.len() - trimmed.len(), text: trimmed });
    }
    Ok(out)
}

/// A plain or quoted scalar.
fn scalar(raw: &str, no: usize) -> Result<Value, String> {
    let s = raw.trim();
    if s.is_empty() {
        return Ok(Value::Null);
    }
    if let Some(first) = s.chars().next() {
        if matches!(first, '&' | '*' | '!' | '|' | '>' | '{' | '@' | '`') {
            return Err(err(no, format!("unsupported YAML syntax {first:?} (anchors, tags, block scalars, flow maps)")));
        }
    }
    if let Some(inner) = s.strip_prefix('"') {
        let inner = inner.strip_suffix('"').ok_or_else(|| err(no, "unterminated double-quoted string"))?;
        let mut out = String::with_capacity(inner.len());
        let mut chars = inner.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('\\') => out.push('\\'),
                    Some('"') => out.push('"'),
                    Some('/') => out.push('/'),
                    Some(other) => return Err(err(no, format!("unsupported escape \\{other}"))),
                    None => return Err(err(no, "dangling escape")),
                }
            } else {
                out.push(c);
            }
        }
        return Ok(Value::String(out));
    }
    if let Some(inner) = s.strip_prefix('\'') {
        let inner = inner.strip_suffix('\'').ok_or_else(|| err(no, "unterminated single-quoted string"))?;
        return Ok(Value::String(inner.replace("''", "'")));
    }
    if let Some(inner) = s.strip_prefix('[') {
        let inner = inner.strip_suffix(']').ok_or_else(|| err(no, "unterminated flow sequence"))?;
        if inner.contains('[') || inner.contains('{') {
            return Err(err(no, "nested flow collections are not supported"));
        }
        let items: Result<Vec<Value>, String> = if inner.trim().is_empty() {
            Ok(Vec::new())
        } else {
            inner.split(',').map(|p| scalar(p, no)).collect()
        };
        return items.map(Value::Array);
    }
    match s {
        "true" | "True" | "TRUE" | "yes" | "Yes" | "on" | "On" => return Ok(Value::Bool(true)),
        "false" | "False" | "FALSE" | "no" | "No" | "off" | "Off" => return Ok(Value::Bool(false)),
        "null" | "Null" | "NULL" | "~" => return Ok(Value::Null),
        _ => {}
    }
    if let Ok(i) = s.parse::<i64>() {
        return Ok(Value::from(i));
    }
    if s.chars().any(|c| c.is_ascii_digit()) && !s.contains(' ') {
        if let Ok(f) = s.parse::<f64>() {
            if f.is_finite() {
                return Ok(Value::from(f));
            }
        }
    }
    Ok(Value::String(s.to_string()))
}

/// Split `key: rest` (the key may be quoted); `None` when the line is not a mapping entry.
fn split_key(text: &str) -> Option<(String, &str)> {
    let (key, rest) = if let Some(q) = text.strip_prefix('"').or_else(|| text.strip_prefix('\'')) {
        let quote = text.as_bytes()[0] as char;
        let end = q.find(quote)?;
        let after = q[end + 1..].trim_start();
        (q[..end].to_string(), after.strip_prefix(':')?)
    } else {
        let at = text.find(": ").or_else(|| text.strip_suffix(':').map(|t| t.len()))?;
        (text[..at].trim().to_string(), &text[at + 1..])
    };
    (!key.is_empty()).then_some((key, rest))
}

/// Parse the block at `lines[*pos..]` indented exactly `indent`.
fn block(lines: &[Line<'_>], pos: &mut usize, indent: usize, depth: usize) -> Result<Value, String> {
    if depth > MAX_DEPTH {
        return Err(err(lines[*pos].no, format!("nested deeper than {MAX_DEPTH} levels")));
    }
    let first = &lines[*pos];
    if first.text == "-" || first.text.starts_with("- ") {
        let mut items = Vec::new();
        while *pos < lines.len() && lines[*pos].indent == indent && (lines[*pos].text == "-" || lines[*pos].text.starts_with("- ")) {
            let line = &lines[*pos];
            let rest = line.text[1..].trim();
            *pos += 1;
            if rest.is_empty() {
                if *pos < lines.len() && lines[*pos].indent > indent {
                    let child = lines[*pos].indent;
                    items.push(block(lines, pos, child, depth + 1)?);
                } else {
                    items.push(Value::Null);
                }
            } else if split_key(rest).is_some() {
                return Err(err(line.no, "mappings inside sequence items are not supported"));
            } else {
                items.push(scalar(rest, line.no)?);
            }
        }
        return Ok(Value::Array(items));
    }
    let mut map = Map::new();
    while *pos < lines.len() && lines[*pos].indent == indent {
        let line = &lines[*pos];
        let (key, rest) = split_key(line.text).ok_or_else(|| err(line.no, "expected `key: value`"))?;
        *pos += 1;
        let value = if rest.trim().is_empty() {
            if *pos < lines.len() && lines[*pos].indent > indent {
                let child = lines[*pos].indent;
                block(lines, pos, child, depth + 1)?
            } else if *pos < lines.len()
                && lines[*pos].indent == indent
                && (lines[*pos].text == "-" || lines[*pos].text.starts_with("- "))
            {
                // A sequence at the key's own indentation (`key:\n- a`), as PyYAML allows.
                block(lines, pos, indent, depth + 1)?
            } else {
                Value::Null
            }
        } else {
            scalar(rest, line.no)?
        };
        if map.insert(key.clone(), value).is_some() {
            return Err(err(line.no, format!("duplicate key {key:?}")));
        }
    }
    if *pos < lines.len() && lines[*pos].indent > indent {
        return Err(err(lines[*pos].no, "unexpected indentation"));
    }
    Ok(Value::Object(map))
}

/// Parse a `config.yaml` (see the module docs) into JSON.
pub fn parse_yaml(text: &str) -> Result<Value, String> {
    if text.len() > MAX_CONFIG_BYTES {
        return Err(format!("config.yaml is {} bytes (at most {MAX_CONFIG_BYTES})", text.len()));
    }
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let lines = lines(text)?;
    if lines.is_empty() {
        return Ok(Value::Object(Map::new()));
    }
    let mut pos = 0;
    let indent = lines[0].indent;
    let v = block(&lines, &mut pos, indent, 0)?;
    if pos < lines.len() {
        return Err(err(lines[pos].no, "unexpected indentation"));
    }
    Ok(v)
}

/// What the importer reads from a voice's `config.yaml` (upstream `configs/reflow.yaml` fields;
/// unset fields take upstream's 6.3 defaults).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DdspConfig {
    pub model_type: String,
    pub sampling_rate: u32,
    pub block_size: u32,
    pub encoder: String,
    pub encoder_sample_rate: u32,
    pub encoder_hop_size: u32,
    pub encoder_out_channels: u32,
    pub f0_extractor: String,
    pub f0_min: f32,
    pub f0_max: f32,
    pub volume_smooth_size: u32,
    pub win_length: u32,
    pub n_aux_layers: u32,
    pub n_aux_chans: u32,
    pub n_layers: u32,
    pub n_chans: u32,
    pub use_norm: bool,
    pub use_attention: bool,
    pub use_pitch_aug: bool,
    pub n_spk: u32,
    /// `model.t_start`: where the voice's flow was trained from (the trainer draws `t` in
    /// `[t_start, 1]`, so the velocity field is untrained below it). Inference starts at the larger
    /// of the asked `t_start` and this (upstream `main_reflow.py`). Records written before the field
    /// existed read 0 (no floor).
    #[serde(default)]
    pub t_start: f32,
    pub vocoder_type: String,
    pub vocoder_ckpt: String,
    pub infer_step: u32,
    pub infer_method: String,
}

fn section<'a>(root: &'a Value, name: &str) -> Option<&'a Map<String, Value>> {
    root.get(name).and_then(Value::as_object)
}

fn get_u32(s: Option<&Map<String, Value>>, key: &str, default: u32) -> Result<u32, String> {
    match s.and_then(|m| m.get(key)) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_u64()
            .or_else(|| v.as_f64().filter(|f| f.fract() == 0.0 && *f >= 0.0).map(|f| f as u64))
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| format!("config.yaml: {key} must be a whole number, not {v}")),
    }
}

fn get_f32(s: Option<&Map<String, Value>>, key: &str, default: f32) -> Result<f32, String> {
    match s.and_then(|m| m.get(key)) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_f64()
            .filter(|f| f.is_finite())
            .map(|f| f as f32)
            .ok_or_else(|| format!("config.yaml: {key} must be a number, not {v}")),
    }
}

fn get_bool(s: Option<&Map<String, Value>>, key: &str, default: bool) -> Result<bool, String> {
    match s.and_then(|m| m.get(key)) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(v) => Err(format!("config.yaml: {key} must be true or false, not {v}")),
    }
}

fn get_str(s: Option<&Map<String, Value>>, key: &str, default: &str) -> String {
    match s.and_then(|m| m.get(key)) {
        Some(Value::String(v)) => v.trim().chars().take(256).collect(),
        Some(Value::Number(n)) => n.to_string(),
        _ => default.to_string(),
    }
}

impl DdspConfig {
    /// The typed fields of a parsed `config.yaml`.
    pub fn from_value(root: &Value) -> Result<DdspConfig, String> {
        if !root.is_object() {
            return Err("config.yaml is not a mapping".into());
        }
        let data = section(root, "data");
        let model = section(root, "model");
        let vocoder = section(root, "vocoder");
        let infer = section(root, "infer");
        if data.is_none() || model.is_none() {
            return Err("config.yaml has no `data` / `model` sections: not a DDSP-SVC config".into());
        }
        let encoder = get_str(data, "encoder", "contentvec768l12tta2x");
        let default_hop = if encoder.ends_with("tta2x") { 160 } else { 320 };
        Ok(DdspConfig {
            model_type: get_str(model, "type", "RectifiedFlow"),
            sampling_rate: get_u32(data, "sampling_rate", 44_100)?,
            block_size: get_u32(data, "block_size", 512)?,
            encoder,
            encoder_sample_rate: get_u32(data, "encoder_sample_rate", 16_000)?,
            encoder_hop_size: get_u32(data, "encoder_hop_size", default_hop)?,
            encoder_out_channels: get_u32(data, "encoder_out_channels", 768)?,
            f0_extractor: get_str(data, "f0_extractor", "rmvpe"),
            f0_min: get_f32(data, "f0_min", 65.0)?,
            f0_max: get_f32(data, "f0_max", 800.0)?,
            volume_smooth_size: get_u32(data, "volume_smooth_size", 2048)?,
            win_length: get_u32(model, "win_length", 2048)?,
            n_aux_layers: get_u32(model, "n_aux_layers", 3)?,
            n_aux_chans: get_u32(model, "n_aux_chans", 256)?,
            n_layers: get_u32(model, "n_layers", 6)?,
            n_chans: get_u32(model, "n_chans", 512)?,
            use_norm: get_bool(model, "use_norm", false)?,
            use_attention: get_bool(model, "use_attention", false)?,
            use_pitch_aug: get_bool(model, "use_pitch_aug", false)?,
            n_spk: get_u32(model, "n_spk", 1)?.max(1),
            // Upstream treats a negative t_start as 0 (`reflow.py`); 1 and above trains no flow.
            t_start: get_f32(model, "t_start", 0.0)?.clamp(0.0, 1.0),
            vocoder_type: get_str(vocoder, "type", "nsf-hifigan"),
            vocoder_ckpt: get_str(vocoder, "ckpt", ""),
            infer_step: get_u32(infer, "infer_step", 50)?,
            infer_method: get_str(infer, "method", "euler"),
        })
    }

    /// Parse `config.yaml` text.
    pub fn parse(text: &str) -> Result<DdspConfig, String> {
        DdspConfig::from_value(&parse_yaml(text)?)
    }

    /// Why this voice cannot run on the template `arch` (and the default vocoder), if it cannot:
    /// every field that changes the graphs, named with both values.
    pub fn mismatch(&self, arch: &super::layout::Arch, vocoder: &super::layout::VocoderConfig) -> Option<String> {
        let mut bad: Vec<String> = Vec::new();
        if self.model_type != "RectifiedFlow" {
            bad.push(format!("model.type {:?} (Sigil runs the 6.3 RectifiedFlow voices)", self.model_type));
        }
        let mut check = |name: &str, got: u64, want: u64| {
            if got != want {
                bad.push(format!("{name} {got} (Sigil's DDSP-SVC 6.3 template: {want})"));
            }
        };
        check("data.sampling_rate", self.sampling_rate.into(), arch.sampling_rate.into());
        check("data.block_size", self.block_size.into(), arch.block_size.into());
        check("data.encoder_out_channels", self.encoder_out_channels.into(), arch.encoder_out_channels.into());
        check("model.win_length", self.win_length.into(), arch.win_length.into());
        check("model.n_aux_layers", self.n_aux_layers.into(), arch.n_aux_layers.into());
        check("model.n_aux_chans", self.n_aux_chans.into(), arch.n_aux_chans.into());
        check("model.n_layers", self.n_layers.into(), arch.n_layers.into());
        check("model.n_chans", self.n_chans.into(), arch.n_chans.into());
        if self.use_attention != arch.use_attention {
            bad.push(format!(
                "model.use_attention {} (Sigil's template: {})",
                self.use_attention, arch.use_attention
            ));
        }
        if !matches!(self.encoder.as_str(), "contentvec768l12tta2x" | "contentvec768l12") {
            bad.push(format!(
                "data.encoder {:?} (Sigil runs contentvec768l12tta2x and contentvec768l12 voices)",
                self.encoder
            ));
        }
        if self.encoder_sample_rate != 16_000 {
            bad.push(format!("data.encoder_sample_rate {} (ContentVec runs at 16000)", self.encoder_sample_rate));
        }
        let want_hop = if self.encoder == "contentvec768l12tta2x" { 160 } else { 320 };
        if self.encoder_hop_size != want_hop {
            bad.push(format!("data.encoder_hop_size {} ({} needs {want_hop})", self.encoder_hop_size, self.encoder));
        }
        if self.vocoder_type != "nsf-hifigan" {
            bad.push(format!(
                "vocoder.type {:?} (Sigil runs voices trained with the nsf-hifigan vocoder)",
                self.vocoder_type
            ));
        }
        if self.sampling_rate != vocoder.sampling_rate || self.block_size != vocoder.hop_size {
            bad.push(format!(
                "the vocoder runs {} Hz with hop {}, the voice {} Hz with block {}",
                vocoder.sampling_rate, vocoder.hop_size, self.sampling_rate, self.block_size
            ));
        }
        (!bad.is_empty()).then(|| bad.join("; "))
    }

    /// Unit frames per second of the encoder (`tta2x` interleaves two passes: 100 per second).
    pub fn tta2x(&self) -> bool {
        self.encoder == "contentvec768l12tta2x"
    }

    /// Volume window (samples at the model rate): upstream's `Volume_Extractor(hop,
    /// volume_smooth_size)`.
    pub fn volume_window(&self) -> u32 {
        self.volume_smooth_size.clamp(2, 1 << 16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UPSTREAM: &str = include_str!("../../../../../../../scripts/vc-families/ddsp_svc/upstream/configs/reflow.yaml");

    #[test]
    fn parses_upstreams_reflow_yaml() {
        let v = parse_yaml(UPSTREAM).unwrap();
        assert_eq!(v["data"]["f0_extractor"], "rmvpe");
        assert_eq!(v["data"]["sampling_rate"], 44100);
        assert_eq!(v["data"]["extensions"], serde_json::json!(["wav"]));
        assert_eq!(v["model"]["use_norm"], true);
        assert_eq!(v["train"]["lr"], 0.0005);
        assert_eq!(v["train"]["amp_dtype"], "fp16");
        assert_eq!(v["vocoder"]["ckpt"], "pretrain/nsf_hifigan/model");
        let c = DdspConfig::parse(UPSTREAM).unwrap();
        assert_eq!((c.sampling_rate, c.block_size, c.win_length), (44_100, 512, 2048));
        assert_eq!((c.n_aux_layers, c.n_aux_chans, c.n_layers, c.n_chans), (6, 512, 6, 1024));
        assert!(c.use_pitch_aug && !c.use_attention && c.tta2x());
        assert_eq!((c.f0_min, c.n_spk, c.volume_smooth_size, c.encoder_hop_size), (65.0, 1, 1024, 160));
        assert_eq!(c.t_start, 0.0);
    }

    /// `model.t_start`: read, clamped to 0..=1 like upstream, 0 when unset or in an older record.
    #[test]
    fn trained_t_start_is_read_and_clamped() {
        let with = |v: &str| DdspConfig::parse(&UPSTREAM.replace("t_start: 0.0", &format!("t_start: {v}"))).unwrap().t_start;
        assert_eq!(with("0.8"), 0.8);
        assert_eq!(with("-0.5"), 0.0);
        assert_eq!(with("1.5"), 1.0);
        assert!(DdspConfig::parse(&UPSTREAM.replace("t_start: 0.0", "t_start: soon")).unwrap_err().contains("t_start"));
        assert_eq!(DdspConfig::parse(&UPSTREAM.replace("t_start: 0.0", "unrelated: 0.9")).unwrap().t_start, 0.0);
        // A record written before the field existed.
        let mut old = serde_json::to_value(DdspConfig::parse(UPSTREAM).unwrap()).unwrap();
        old.as_object_mut().unwrap().remove("t_start");
        assert_eq!(serde_json::from_value::<DdspConfig>(old).unwrap().t_start, 0.0);
    }

    #[test]
    fn yaml_subset_edge_cases() {
        let v = parse_yaml("a:\n  b: 'it''s' # c\n  c: \"x#y\"\n  d: [1, 2.5, three]\nlist:\n- x\n- 2\ne: ~\nf: -3\ng: 1e-4\n").unwrap();
        assert_eq!(v["a"]["b"], "it's");
        assert_eq!(v["a"]["c"], "x#y");
        assert_eq!(v["a"]["d"], serde_json::json!([1, 2.5, "three"]));
        assert_eq!(v["list"], serde_json::json!(["x", 2]));
        assert!(v["e"].is_null());
        assert_eq!(v["f"], -3);
        assert_eq!(v["g"], 1e-4);
        // Windows line endings and a BOM.
        assert_eq!(parse_yaml("\u{feff}a: 1\r\nb: 2\r\n").unwrap()["b"], 2);
        for bad in [
            "a: &x 1\nb: *x\n",
            "a: !!str 1\n",
            "a: |\n  text\n",
            "a: 1\na: 2\n",
            "a:\n    b: 1\n  c: 2\n",
            "\ta: 1\n",
            "a: \"open\n",
            "a: {b: 1}\n",
            "---\na: 1\n---\nb: 2\n",
        ] {
            assert!(parse_yaml(bad).is_err(), "{bad:?}");
        }
        let deep: String = (0..12).map(|i| format!("{}k{i}:\n", "  ".repeat(i))).collect();
        assert!(parse_yaml(&deep).unwrap_err().contains("deeper"));
        assert!(parse_yaml(&"a: 1\n".repeat(20_000)).is_err());
    }

    #[test]
    fn typed_fields_and_mismatches() {
        assert!(DdspConfig::parse("foo: 1\n").unwrap_err().contains("not a DDSP-SVC config"));
        assert!(DdspConfig::parse("data:\n  block_size: big\nmodel:\n  n_spk: 1\n").unwrap_err().contains("block_size"));
        assert!(DdspConfig::parse("data:\n  block_size: -1\nmodel:\n  n_spk: 1\n").is_err());
        let layout = super::super::layout::DdspLayout::embedded().unwrap();
        let ok = DdspConfig::parse(UPSTREAM).unwrap();
        assert_eq!(ok.mismatch(&layout.arch, &layout.vocoder), None);
        let other = UPSTREAM.replace("n_chans: 1024", "n_chans: 512").replace("use_attention: false", "use_attention: true");
        let why = DdspConfig::parse(&other).unwrap().mismatch(&layout.arch, &layout.vocoder).unwrap();
        assert!(why.contains("model.n_chans 512") && why.contains("use_attention"), "{why}");
        let hubert = UPSTREAM.replace("'contentvec768l12tta2x'", "'hubertsoft'");
        assert!(DdspConfig::parse(&hubert).unwrap().mismatch(&layout.arch, &layout.vocoder).unwrap().contains("hubertsoft"));
        let log10 = UPSTREAM.replace("'nsf-hifigan'", "'nsf-hifigan-log10'");
        assert!(DdspConfig::parse(&log10).unwrap().mismatch(&layout.arch, &layout.vocoder).unwrap().contains("log10"));
    }
}
