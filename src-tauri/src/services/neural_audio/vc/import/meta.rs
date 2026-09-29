//! Checkpoint metadata detection and template selection.
//!
//! The weights are the ground truth: the version comes from `enc_p.emb_phone.weight` (256 inputs =
//! v1, 768 = v2) and pitch guidance from the presence of `enc_p.emb_pitch.weight`; the declared
//! `version` / `f0` / `sr` fields only produce a warning when they disagree. The sample rate and the
//! architecture come from `config` (RVC builds the model from it), the vocoder and the content
//! embedder from Applio's `vocoder` / `embedder_model` (default HiFi-GAN and ContentVec, what RVC
//! WebUI trains).

use serde::Serialize;
use serde_json::Value as J;

use super::layout::Layout;
use super::{ErrorCode, ImportError};

/// RVC config list length (see `scripts/vc-templates/rvc_arch/configs.py`).
pub const CONFIG_LEN: usize = 18;

/// The top-level checkpoint entries [`detect`] reads; `pth` converts only these to JSON.
pub const USED_KEYS: &[&str] = &["config", "sr", "f0", "version", "info", "vocoder", "embedder_model", "model_name", "author"];

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct CheckpointMeta {
    /// `v1` or `v2`.
    pub version: String,
    pub f0: bool,
    pub sample_rate: u32,
    /// `hifigan`, `mrf-hifigan` or `refinegan`.
    pub vocoder: String,
    /// Content embedder the voice was trained on (`contentvec`, `spin`, `japanese-hubert-base`, ...).
    pub embedder: String,
    pub speakers: usize,
    pub speaker_dim: usize,
    /// Free-text training info (`300epoch`), author / name when Applio recorded them.
    pub info: Option<String>,
    pub model_name: Option<String>,
    pub author: Option<String>,
    /// The RVC config list.
    pub config: Vec<J>,
    pub warnings: Vec<String>,
}

impl CheckpointMeta {
    /// Template id, e.g. `v2-40k-f0-hifigan`.
    pub fn template_id(&self) -> String {
        format!(
            "{}-{}k-{}-{}",
            self.version,
            self.sample_rate / 1000,
            if self.f0 { "f0" } else { "nof0" },
            self.vocoder
        )
    }
}

fn unsupported(msg: impl Into<String>) -> ImportError {
    ImportError::new(ErrorCode::UnsupportedModel, msg)
}

/// `"40k"`, `"48000"`, `40000` -> Hz.
pub fn parse_sample_rate(v: &J) -> Option<u32> {
    match v {
        J::Number(n) => n.as_u64().and_then(|n| u32::try_from(n).ok()),
        J::String(s) => {
            let s = s.trim().to_ascii_lowercase();
            match s.strip_suffix('k') {
                Some(k) => k.parse::<u32>().ok().map(|k| k * 1000),
                None => s.parse::<u32>().ok(),
            }
        }
        _ => None,
    }
    .filter(|&hz| (8_000..=96_000).contains(&hz))
}

fn vocoder_slug(v: &str) -> Option<&'static str> {
    match v.trim().to_ascii_lowercase().replace(['_', ' '], "-").as_str() {
        "" | "hifi-gan" | "hifigan" | "nsf-hifigan" | "hifi-gan-nsf" => Some("hifigan"),
        "mrf-hifi-gan" | "mrf-hifigan" | "mrf" => Some("mrf-hifigan"),
        "refinegan" | "refine-gan" => Some("refinegan"),
        _ => None,
    }
}

fn truthy(v: &J) -> Option<bool> {
    match v {
        J::Bool(b) => Some(*b),
        J::Number(n) => n.as_i64().map(|i| i != 0),
        J::String(s) => match s.trim() {
            "1" | "true" | "True" => Some(true),
            "0" | "false" | "False" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn text(meta: &serde_json::Map<String, J>, key: &str) -> Option<String> {
    meta.get(key).and_then(J::as_str).map(|s| s.chars().take(200).collect())
}

/// Detect the metadata of a checkpoint from its top-level entries and tensor shapes.
pub fn detect(meta: &serde_json::Map<String, J>, shape: impl Fn(&str) -> Option<Vec<usize>>) -> Result<CheckpointMeta, ImportError> {
    let mut warnings = Vec::new();
    let emb = shape("enc_p.emb_phone.weight")
        .ok_or_else(|| ImportError::new(ErrorCode::NotACheckpoint, "no enc_p.emb_phone.weight: not an RVC voice"))?;
    let version = match emb.as_slice() {
        [_, 256] => "v1",
        [_, 768] => "v2",
        other => return Err(unsupported(format!("content input of {other:?} (RVC v1 has 256, v2 has 768)"))),
    };
    if let Some(declared) = meta.get("version").and_then(J::as_str) {
        if !declared.eq_ignore_ascii_case(version) {
            warnings.push(format!("declared version {declared} but the weights are {version}"));
        }
    }
    let f0 = shape("enc_p.emb_pitch.weight").is_some();
    if let Some(declared) = meta.get("f0").and_then(truthy) {
        if declared != f0 {
            warnings.push(format!("declared f0={declared} but the weights say {f0}"));
        }
    }
    let config = match meta.get("config") {
        Some(J::Array(c)) if c.len() == CONFIG_LEN => c.clone(),
        Some(J::Array(c)) => return Err(unsupported(format!("config has {} entries (RVC has {CONFIG_LEN})", c.len()))),
        _ => return Err(ImportError::new(ErrorCode::NotACheckpoint, "no model config in the checkpoint")),
    };
    let sample_rate = parse_sample_rate(&config[CONFIG_LEN - 1])
        .or_else(|| meta.get("sr").and_then(parse_sample_rate))
        .ok_or_else(|| unsupported("no usable sample rate"))?;
    if let Some(declared) = meta.get("sr").and_then(parse_sample_rate) {
        if declared != sample_rate {
            warnings.push(format!("declared sr {declared} but the config says {sample_rate}"));
        }
    }
    let vocoder_raw = text(meta, "vocoder").unwrap_or_default();
    let vocoder = vocoder_slug(&vocoder_raw)
        .ok_or_else(|| unsupported(format!("unknown vocoder {vocoder_raw:?}")))?
        .to_string();
    let embedder = text(meta, "embedder_model")
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "contentvec".into());
    let (speakers, speaker_dim) = match shape("emb_g.weight").as_deref() {
        Some([n, d]) if *n > 0 && *d > 0 => (*n, *d),
        _ => return Err(ImportError::new(ErrorCode::NotACheckpoint, "no speaker table (emb_g.weight)")),
    };
    Ok(CheckpointMeta {
        version: version.into(),
        f0,
        sample_rate,
        vocoder,
        embedder,
        speakers,
        speaker_dim,
        info: text(meta, "info"),
        model_name: text(meta, "model_name"),
        author: text(meta, "author"),
        config,
        warnings,
    })
}

/// Numbers compare by value (`0` == `0.0`), everything else structurally.
fn same(a: &J, b: &J) -> bool {
    match (a, b) {
        (J::Number(x), J::Number(y)) => x.as_f64() == y.as_f64(),
        (J::Array(x), J::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q)),
        _ => a == b,
    }
}

/// The checkpoint's architecture fields must equal the template's (sample rate compared in Hz).
pub fn check_template(meta: &CheckpointMeta, layout: &Layout) -> Result<(), ImportError> {
    let arch = &layout.arch;
    if arch.version != meta.version || arch.f0 != meta.f0 || arch.vocoder != meta.vocoder || arch.sample_rate != meta.sample_rate {
        return Err(ImportError::new(
            ErrorCode::NoTemplate,
            format!("template {} does not cover a {} voice", layout.template, meta.template_id()),
        ));
    }
    for &i in &arch.arch_fields {
        let (ours, theirs) = (&arch.config[i], &meta.config[i]);
        let equal = if i == CONFIG_LEN - 1 {
            parse_sample_rate(ours) == parse_sample_rate(theirs)
        } else {
            same(ours, theirs)
        };
        if !equal {
            return Err(unsupported(format!(
                "config field {i} is {theirs}, the {} template needs {ours} (a non-standard architecture)",
                layout.template
            )));
        }
    }
    if meta.speaker_dim != layout.speakers.dim {
        return Err(unsupported(format!("speaker embedding of {} (template: {})", meta.speaker_dim, layout.speakers.dim)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn shapes(v1: bool, f0: bool) -> impl Fn(&str) -> Option<Vec<usize>> {
        move |k: &str| match k {
            "enc_p.emb_phone.weight" => Some(vec![192, if v1 { 256 } else { 768 }]),
            "enc_p.emb_pitch.weight" if f0 => Some(vec![256, 192]),
            "emb_g.weight" => Some(vec![109, 256]),
            _ => None,
        }
    }

    fn rvc_meta() -> serde_json::Map<String, J> {
        json!({
            "config": [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000],
            "info": "300epoch", "sr": "40k", "f0": 1, "version": "v2"
        })
        .as_object()
        .unwrap()
        .clone()
    }

    #[test]
    fn detects_webui_v2() {
        let m = detect(&rvc_meta(), shapes(false, true)).unwrap();
        assert_eq!(m.template_id(), "v2-40k-f0-hifigan");
        assert_eq!(m.embedder, "contentvec");
        assert_eq!((m.speakers, m.speaker_dim), (109, 256));
        assert!(m.warnings.is_empty(), "{:?}", m.warnings);
    }

    #[test]
    fn applio_fields_and_conflicts() {
        let mut meta = rvc_meta();
        meta.insert("vocoder".into(), json!("RefineGAN"));
        meta.insert("embedder_model".into(), json!("japanese-hubert-base"));
        meta.insert("version".into(), json!("v1"));
        let m = detect(&meta, shapes(false, true)).unwrap();
        assert_eq!(m.template_id(), "v2-40k-f0-refinegan");
        assert_eq!(m.embedder, "japanese-hubert-base");
        assert_eq!(m.warnings.len(), 1, "declared v1 vs v2 weights");
        meta.insert("vocoder".into(), json!("WaveNet"));
        assert_eq!(detect(&meta, shapes(false, true)).unwrap_err().code, ErrorCode::UnsupportedModel);
    }

    #[test]
    fn sample_rates_and_nono() {
        assert_eq!(parse_sample_rate(&json!("48k")), Some(48_000));
        assert_eq!(parse_sample_rate(&json!(32000)), Some(32_000));
        assert_eq!(parse_sample_rate(&json!("7")), None);
        let m = detect(&rvc_meta(), shapes(true, false)).unwrap();
        assert_eq!(m.template_id(), "v1-40k-nof0-hifigan");
        assert!(!m.warnings.is_empty());
    }
}
