//! Checkpoint metadata detection and template matching.
//!
//! The weights are the ground truth:
//! - the version comes from `enc_p.emb_phone.weight` (256 inputs = v1, 768 = v2);
//! - pitch guidance from the presence of `enc_p.emb_pitch.weight`;
//! - the vocoder from the decoder's module names (`dec.resblocks.*` = HiFi-GAN NSF, `dec.mrfs.*` =
//!   Applio's MRF HiFi-GAN, `dec.downsample_blocks.*` / `dec.pre_conv.*` = Applio's RefineGAN),
//!   when they identify it.
//!
//! Declared `version` / `f0` / `sr` / `vocoder` fields that disagree with the weights produce a
//! warning. The sample rate and the architecture come from `config` (RVC builds the model from it);
//! a checkpoint without one (a training checkpoint `G_*.pth`, a bare state dict) gets its template
//! from the tensor shapes instead (`import::select_template`). The content embedder comes from
//! Applio's `embedder_model` (default ContentVec, what RVC WebUI trains) and is mapped to Sigil's
//! embedder names (`vc::assets::embedder_id`): `contentvec` (v2), `contentvec-256` (v1), `spin-v2`,
//! `spin` (Applio's first SPIN, `vc/embedder-spin`; P3-I0), `hubert-ja`, `hubert-ko`, `hubert-zh`.
//! Embedders Sigil has no encoder for (custom and unknown embedders) are refused, since the voice
//! would play back as the wrong content.

use serde::Serialize;
use serde_json::Value as J;

use super::layout::Layout;
use super::{ErrorCode, ImportError};

/// RVC config list length (see `scripts/vc-templates/rvc_arch/configs.py`).
pub const CONFIG_LEN: usize = 18;

/// The top-level checkpoint entries [`detect`] reads; `pth` converts only these to JSON.
pub const USED_KEYS: &[&str] = &["config", "sr", "f0", "version", "info", "vocoder", "embedder_model", "model_name", "author"];

/// Sigil's embedder names (the manifest `embedder`, `vc::assets::embedder_id`).
pub const EMBEDDER_CONTENTVEC: &str = "contentvec";
pub const EMBEDDER_CONTENTVEC_256: &str = "contentvec-256";
pub const EMBEDDER_SPIN_V2: &str = "spin-v2";
pub const EMBEDDER_SPIN: &str = "spin";
pub const EMBEDDER_HUBERT_JA: &str = "hubert-ja";
pub const EMBEDDER_HUBERT_KO: &str = "hubert-ko";
pub const EMBEDDER_HUBERT_ZH: &str = "hubert-zh";

/// Vocoder slugs (template ids and `layout.json` `arch.vocoder`, after [`vocoder_slug`]).
pub const VOCODER_HIFIGAN: &str = "hifigan";
pub const VOCODER_MRF: &str = "mrf-hifigan";
pub const VOCODER_REFINEGAN: &str = "refinegan";

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct CheckpointMeta {
    /// `v1` or `v2`.
    pub version: String,
    pub f0: bool,
    /// Output sample rate (Hz). 0 until the template is chosen for a checkpoint without a config
    /// (`import::select_template` fills it from the template).
    pub sample_rate: u32,
    /// [`VOCODER_HIFIGAN`], [`VOCODER_MRF`] or [`VOCODER_REFINEGAN`].
    pub vocoder: String,
    /// Sigil's name of the content embedder the voice was trained on ([`EMBEDDER_CONTENTVEC`], ...).
    pub embedder: String,
    /// The checkpoint's own `embedder_model` value, when it has one (Applio).
    pub embedder_declared: Option<String>,
    pub speakers: usize,
    pub speaker_dim: usize,
    /// Free-text training info (`300epoch`), author / name when Applio recorded them.
    pub info: Option<String>,
    pub model_name: Option<String>,
    pub author: Option<String>,
    /// The RVC config list (empty when the checkpoint has none).
    pub config: Vec<J>,
    /// `inference` (RVC / Applio export), `training` (`G_*.pth` with a `model` entry) or
    /// `state_dict` (a bare dict of tensors).
    pub checkpoint_kind: String,
    pub warnings: Vec<String>,
}

impl CheckpointMeta {
    /// Template id, e.g. `v2-40k-f0-hifigan` (`?k` while the sample rate is unknown).
    pub fn template_id(&self) -> String {
        let sr = if self.sample_rate == 0 { "?".to_string() } else { (self.sample_rate / 1000).to_string() };
        format!("{}-{sr}k-{}-{}", self.version, if self.f0 { "f0" } else { "nof0" }, self.vocoder)
    }

    /// Whether the checkpoint carries RVC's config list (else the template is matched by shapes).
    pub fn has_config(&self) -> bool {
        !self.config.is_empty()
    }

    /// Feature channels of the embedder (768 for v2, 256 for v1).
    pub fn emb_channels(&self) -> usize {
        if self.version == "v1" {
            256
        } else {
            768
        }
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
                Some(k) => k.parse::<u32>().ok().and_then(|k| k.checked_mul(1000)),
                None => s.parse::<u32>().ok(),
            }
        }
        _ => None,
    }
    .filter(|&hz| (8_000..=96_000).contains(&hz))
}

/// The vocoder slug of a declared (Applio `vocoder`) or template vocoder name.
pub fn vocoder_slug(v: &str) -> Option<&'static str> {
    match v.trim().to_ascii_lowercase().replace(['_', ' '], "-").as_str() {
        "" | "hifi-gan" | "hifigan" | "nsf-hifigan" | "hifi-gan-nsf" | "hifigan-nsf" | "nsf" => Some(VOCODER_HIFIGAN),
        "mrf-hifi-gan" | "mrf-hifigan" | "mrf" | "hifigan-mrf" => Some(VOCODER_MRF),
        "refinegan" | "refine-gan" => Some(VOCODER_REFINEGAN),
        _ => None,
    }
}

/// The manifest's name of a vocoder slug (`hifigan`, `mrf`, `refinegan`; `manifest.rs` docs).
pub fn manifest_vocoder(slug: &str) -> &'static str {
    match vocoder_slug(slug) {
        Some(VOCODER_MRF) => "mrf",
        Some(VOCODER_REFINEGAN) => "refinegan",
        _ => "hifigan",
    }
}

/// The vocoder the decoder's module names identify, if they do.
pub fn vocoder_from_weights<'a>(mut names: impl Iterator<Item = &'a str>) -> Option<&'static str> {
    let mut found = None;
    for name in names.by_ref() {
        let Some(rest) = name.strip_prefix("dec.") else {
            continue;
        };
        if rest.starts_with("mrfs.") {
            return Some(VOCODER_MRF);
        }
        if rest.starts_with("downsample_blocks.") || rest.starts_with("pre_conv.") || rest.starts_with("upsample_conv_blocks.") {
            return Some(VOCODER_REFINEGAN);
        }
        if rest.starts_with("resblocks.") {
            found = Some(VOCODER_HIFIGAN);
        }
    }
    found
}

/// Sigil's embedder for a declared `embedder_model` (Applio, w-okada or Sigil names) and version.
pub fn canonical_embedder(declared: Option<&str>, version: &str) -> Result<&'static str, ImportError> {
    let raw = declared.map(|s| s.trim().to_ascii_lowercase().replace(['_', ' '], "-")).unwrap_or_default();
    let v2 = match raw.as_str() {
        "" | "contentvec" | "content-vec" | "contentvec-768" | "hubert-base" | "hubert" | "vec-768-layer-12" => EMBEDDER_CONTENTVEC,
        "contentvec-256" | "vec-256-layer-9" => {
            return if version == "v1" {
                Ok(EMBEDDER_CONTENTVEC_256)
            } else {
                Err(unsupported("a v2 voice declares the 256-channel (v1) ContentVec embedder"))
            }
        }
        "spin-v2" | "spinv2" => EMBEDDER_SPIN_V2,
        "japanese-hubert-base" | "hubert-base-japanese" | "rinna-hubert-base" | "hubert-ja" => EMBEDDER_HUBERT_JA,
        "korean-hubert-base" | "hubert-base-korean" | "hubert-ko" => EMBEDDER_HUBERT_KO,
        "chinese-hubert-base" | "hubert-base-chinese" | "chinese-hubert" | "hubert-zh" => EMBEDDER_HUBERT_ZH,
        "spin" | "spin-v1" | "spinv1" => EMBEDDER_SPIN,
        "custom" => {
            return Err(unsupported(
                "this voice was trained with a custom embedder; Sigil cannot tell which content encoder it needs",
            ))
        }
        other => {
            let shown: String = other.chars().take(64).collect();
            return Err(unsupported(format!("unknown embedder {shown:?}")));
        }
    };
    if version == "v1" {
        // RVC v1 voices read ContentVec's layer 9 through final_proj (256 channels); no other
        // embedder was ever trained into a v1 voice.
        return if v2 == EMBEDDER_CONTENTVEC {
            Ok(EMBEDDER_CONTENTVEC_256)
        } else {
            Err(unsupported(format!("a v1 voice with the {v2} embedder (v1 voices use 256-channel ContentVec)")))
        };
    }
    Ok(v2)
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
    meta.get(key)
        .and_then(J::as_str)
        .map(|s| s.chars().filter(|c| !c.is_control()).take(200).collect::<String>())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Detect the metadata of a checkpoint from its top-level entries, tensor shapes and tensor names.
/// `kind` is the checkpoint kind (`inference`, `training`, `state_dict`).
pub fn detect<'a>(
    meta: &serde_json::Map<String, J>,
    shape: impl Fn(&str) -> Option<Vec<usize>>,
    names: impl Iterator<Item = &'a str>,
    kind: &str,
) -> Result<CheckpointMeta, ImportError> {
    let mut warnings = Vec::new();
    let emb = shape("enc_p.emb_phone.weight")
        .ok_or_else(|| ImportError::new(ErrorCode::NotACheckpoint, "no enc_p.emb_phone.weight: not an RVC voice"))?;
    let version = match emb.as_slice() {
        [_, 256] => "v1",
        [_, 768] => "v2",
        other => return Err(unsupported(format!("content input of {other:?} (RVC v1 has 256, v2 has 768)"))),
    };
    if let Some(declared) = meta.get("version").and_then(J::as_str) {
        if !declared.trim().eq_ignore_ascii_case(version) {
            warnings.push(format!("declared version {} but the weights are {version}", declared.trim()));
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
        Some(_) => return Err(unsupported("the checkpoint's config is not RVC's config list")),
        None if kind == "inference" => return Err(ImportError::new(ErrorCode::NotACheckpoint, "no model config in the checkpoint")),
        None => Vec::new(),
    };
    let declared_sr = meta.get("sr").and_then(parse_sample_rate);
    let sample_rate = match config.get(CONFIG_LEN - 1) {
        Some(v) => parse_sample_rate(v).or(declared_sr).ok_or_else(|| unsupported("no usable sample rate"))?,
        // Chosen with the template (by shapes; a declared `sr` narrows the search).
        None => declared_sr.unwrap_or(0),
    };
    if let Some(declared) = declared_sr {
        if declared != sample_rate {
            warnings.push(format!("declared sr {declared} but the config says {sample_rate}"));
        }
    }
    let declared_vocoder = text(meta, "vocoder");
    let from_meta = match declared_vocoder.as_deref() {
        None => None,
        Some(v) => match vocoder_slug(v) {
            Some(slug) => Some(slug),
            None => return Err(unsupported(format!("unknown vocoder {v:?}"))),
        },
    };
    let vocoder = match (from_meta, vocoder_from_weights(names)) {
        (Some(declared), Some(weights)) if declared != weights => {
            warnings.push(format!("declared vocoder {declared} but the weights are {weights}"));
            weights
        }
        (_, Some(weights)) => weights,
        (Some(declared), None) => declared,
        (None, None) => VOCODER_HIFIGAN,
    };
    let embedder_declared = text(meta, "embedder_model");
    let embedder = canonical_embedder(embedder_declared.as_deref(), version)?;
    let (speakers, speaker_dim) = match shape("emb_g.weight").as_deref() {
        Some([n, d]) if *n > 0 && *d > 0 => (*n, *d),
        _ => return Err(ImportError::new(ErrorCode::NotACheckpoint, "no speaker table (emb_g.weight)")),
    };
    match kind {
        "training" => warnings.push("a training checkpoint (G_*.pth): its generator weights were imported".into()),
        "state_dict" => warnings.push("a bare state dict without RVC's metadata: the architecture was matched by tensor shapes".into()),
        _ => {}
    }
    Ok(CheckpointMeta {
        version: version.into(),
        f0,
        sample_rate,
        vocoder: vocoder.to_string(),
        embedder: embedder.to_string(),
        embedder_declared,
        speakers,
        speaker_dim,
        info: text(meta, "info"),
        model_name: text(meta, "model_name"),
        author: text(meta, "author"),
        config,
        checkpoint_kind: kind.to_string(),
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

/// Whether the template's architecture is the checkpoint's kind of voice (version, pitch guidance,
/// vocoder and, when known, sample rate).
pub fn covers(meta: &CheckpointMeta, layout: &Layout) -> bool {
    let arch = &layout.arch;
    arch.version == meta.version
        && arch.f0 == meta.f0
        && vocoder_slug(&arch.vocoder) == vocoder_slug(&meta.vocoder)
        && (meta.sample_rate == 0 || arch.sample_rate == meta.sample_rate)
}

/// The checkpoint's architecture fields must equal the template's (sample rate compared in Hz). A
/// checkpoint without a config is checked by its tensor shapes instead (the weights writer).
pub fn check_template(meta: &CheckpointMeta, layout: &Layout) -> Result<(), ImportError> {
    if !covers(meta, layout) {
        return Err(ImportError::new(
            ErrorCode::NoTemplate,
            format!("template {} does not cover a {} voice", layout.template, meta.template_id()),
        ));
    }
    let arch = &layout.arch;
    if meta.has_config() {
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

    const HIFIGAN_NAMES: &[&str] = &["enc_p.emb_phone.weight", "dec.conv_pre.weight", "dec.resblocks.0.convs1.0.weight_v"];

    fn rvc_meta() -> serde_json::Map<String, J> {
        json!({
            "config": [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000],
            "info": "300epoch", "sr": "40k", "f0": 1, "version": "v2"
        })
        .as_object()
        .unwrap()
        .clone()
    }

    fn run(meta: &serde_json::Map<String, J>, v1: bool, f0: bool, names: &[&str]) -> Result<CheckpointMeta, ImportError> {
        detect(meta, shapes(v1, f0), names.iter().copied(), "inference")
    }

    #[test]
    fn detects_webui_v2() {
        let m = run(&rvc_meta(), false, true, HIFIGAN_NAMES).unwrap();
        assert_eq!(m.template_id(), "v2-40k-f0-hifigan");
        assert_eq!(m.embedder, "contentvec");
        assert_eq!(m.embedder_declared, None);
        assert_eq!((m.speakers, m.speaker_dim), (109, 256));
        assert!(m.has_config());
        assert!(m.warnings.is_empty(), "{:?}", m.warnings);
    }

    #[test]
    fn applio_fields_and_conflicts() {
        let mut meta = rvc_meta();
        meta.insert("vocoder".into(), json!("RefineGAN"));
        meta.insert("embedder_model".into(), json!("japanese-hubert-base"));
        meta.insert("version".into(), json!("v1"));
        // No decoder names: the declared vocoder stands.
        let m = run(&meta, false, true, &[]).unwrap();
        assert_eq!(m.template_id(), "v2-40k-f0-refinegan");
        assert_eq!(m.embedder, "hubert-ja");
        assert_eq!(m.embedder_declared.as_deref(), Some("japanese-hubert-base"));
        assert_eq!(m.warnings.len(), 1, "declared v1 vs v2 weights");
        // The weights say HiFi-GAN: they win, with a warning.
        let m = run(&meta, false, true, HIFIGAN_NAMES).unwrap();
        assert_eq!(m.vocoder, VOCODER_HIFIGAN);
        assert!(m.warnings.iter().any(|w| w.contains("vocoder")), "{:?}", m.warnings);
        meta.insert("vocoder".into(), json!("WaveNet"));
        assert_eq!(run(&meta, false, true, &[]).unwrap_err().code, ErrorCode::UnsupportedModel);
    }

    #[test]
    fn vocoders_are_recognised_from_decoder_names() {
        assert_eq!(vocoder_from_weights(["dec.mrfs.0.layers.0.convs1.weight_v", "dec.resblocks.0.x"].into_iter()), Some(VOCODER_MRF));
        assert_eq!(vocoder_from_weights(["dec.pre_conv.weight_v"].into_iter()), Some(VOCODER_REFINEGAN));
        assert_eq!(vocoder_from_weights(["dec.downsample_blocks.0.weight"].into_iter()), Some(VOCODER_REFINEGAN));
        assert_eq!(vocoder_from_weights(HIFIGAN_NAMES.iter().copied()), Some(VOCODER_HIFIGAN));
        assert_eq!(vocoder_from_weights(["enc_p.emb_phone.weight"].into_iter()), None);
        for (name, slug) in [("HiFi-GAN", VOCODER_HIFIGAN), ("MRF HiFi-GAN", VOCODER_MRF), ("RefineGAN", VOCODER_REFINEGAN), ("", VOCODER_HIFIGAN)] {
            assert_eq!(vocoder_slug(name), Some(slug), "{name}");
        }
        assert_eq!(manifest_vocoder(VOCODER_MRF), "mrf");
        assert_eq!(manifest_vocoder("MRF HiFi-GAN"), "mrf");
        assert_eq!(manifest_vocoder(VOCODER_REFINEGAN), "refinegan");
        assert_eq!(manifest_vocoder(VOCODER_HIFIGAN), "hifigan");
    }

    #[test]
    fn embedders_map_to_sigil_names() {
        let ok = |d: Option<&str>, v: &str| canonical_embedder(d, v).unwrap();
        assert_eq!(ok(None, "v2"), EMBEDDER_CONTENTVEC);
        assert_eq!(ok(Some("contentvec"), "v2"), EMBEDDER_CONTENTVEC);
        assert_eq!(ok(None, "v1"), EMBEDDER_CONTENTVEC_256);
        assert_eq!(ok(Some("contentvec"), "v1"), EMBEDDER_CONTENTVEC_256);
        assert_eq!(ok(Some("spin-v2"), "v2"), EMBEDDER_SPIN_V2);
        assert_eq!(ok(Some("japanese-hubert-base"), "v2"), EMBEDDER_HUBERT_JA);
        assert_eq!(ok(Some("korean_hubert_base"), "v2"), EMBEDDER_HUBERT_KO);
        assert_eq!(ok(Some("Chinese-HuBERT-Base"), "v2"), EMBEDDER_HUBERT_ZH);
        assert_eq!(ok(Some("spin"), "v2"), EMBEDDER_SPIN);
        assert_eq!(ok(Some("SPIN_v1"), "v2"), EMBEDDER_SPIN);
        assert_eq!(canonical_embedder(Some("spin"), "v1").unwrap_err().code, ErrorCode::UnsupportedModel);
        for bad in ["custom", "wavlm-large"] {
            assert_eq!(canonical_embedder(Some(bad), "v2").unwrap_err().code, ErrorCode::UnsupportedModel, "{bad}");
        }
        assert_eq!(canonical_embedder(Some("spin-v2"), "v1").unwrap_err().code, ErrorCode::UnsupportedModel);
        assert_eq!(canonical_embedder(Some("contentvec-256"), "v2").unwrap_err().code, ErrorCode::UnsupportedModel);
        // Every name maps to a model-store embedder id.
        for name in [EMBEDDER_CONTENTVEC, EMBEDDER_CONTENTVEC_256, EMBEDDER_SPIN_V2, EMBEDDER_SPIN, EMBEDDER_HUBERT_JA, EMBEDDER_HUBERT_KO, EMBEDDER_HUBERT_ZH] {
            assert!(crate::services::neural_audio::vc::assets::embedder_id(name).is_some(), "{name}");
        }
    }

    #[test]
    fn sample_rates_and_nono() {
        assert_eq!(parse_sample_rate(&json!("48k")), Some(48_000));
        assert_eq!(parse_sample_rate(&json!(32000)), Some(32_000));
        assert_eq!(parse_sample_rate(&json!("7")), None);
        assert_eq!(parse_sample_rate(&json!("99999999999k")), None);
        let m = run(&rvc_meta(), true, false, HIFIGAN_NAMES).unwrap();
        assert_eq!(m.template_id(), "v1-40k-nof0-hifigan");
        assert_eq!(m.embedder, EMBEDDER_CONTENTVEC_256);
        assert!(!m.warnings.is_empty());
    }

    #[test]
    fn checkpoints_without_a_config() {
        let mut meta = rvc_meta();
        meta.remove("config");
        // An inference export must carry its config.
        assert_eq!(run(&meta, false, true, HIFIGAN_NAMES).unwrap_err().code, ErrorCode::NotACheckpoint);
        // A training checkpoint or a bare state dict is matched by shapes; a declared sr narrows it.
        let m = detect(&meta, shapes(false, true), HIFIGAN_NAMES.iter().copied(), "training").unwrap();
        assert!(!m.has_config());
        assert_eq!(m.sample_rate, 40_000);
        assert!(m.warnings.iter().any(|w| w.contains("training")));
        meta.remove("sr");
        let m = detect(&meta, shapes(false, true), HIFIGAN_NAMES.iter().copied(), "state_dict").unwrap();
        assert_eq!(m.sample_rate, 0);
        assert_eq!(m.template_id(), "v2-?k-f0-hifigan");
    }
}
