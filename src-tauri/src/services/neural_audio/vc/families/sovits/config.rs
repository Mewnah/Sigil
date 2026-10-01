//! so-vits-svc voice configuration (P5-D): what a so-vits-svc ONNX export does not carry in its
//! graph (sample rate, hop, speech encoder, speaker names), read from the files that travel with
//! it, and Sigil's own record of it in the slot (`sovits.json`).
//!
//! Two sources, merged (the training config wins where both say something):
//! - so-vits-svc's training `config.json` (4.0 / 4.1): `data.sampling_rate`, `data.hop_length`,
//!   `model.speech_encoder` (`vec768l12`, `vec256l9`, `hubertsoft`, `whisper-ppg`, ...),
//!   `model.ssl_dim`, `model.n_speakers`, `model.vol_embedding`, `spk` (name → id);
//! - the MoeVS / MoeSS JSON so-vits-svc's `onnx_export.py` writes next to the export
//!   (`checkpoints/<name>.json`): `Rate`, `Hop`, `Hubert` (`vec-768-layer-12`, ...; MoeSS's
//!   so-vits-svc 4.0 JSONs say `hubert4.0`), `CharaMix`, `Volume`, `HiddenSize`, `Characters`,
//!   `Type` (`SoVits`), `SoVits3`, `SoVits4`.
//!
//! Which of the two a file is follows its keys (`data` + `model` is a training config), never its
//! values: so-vits-svc 4.0 training configs name no speech encoder at all.
//!
//! Neither file is executed or trusted beyond these fields (bounded size, typed reads).

use serde::{Deserialize, Serialize};

use crate::services::neural_audio::vc::engine::content::EmbedderKind;

/// Largest config file read (so-vits configs are a few KB; MoeVS JSONs less).
pub const MAX_CONFIG_BYTES: u64 = 1 << 20;
/// Most speakers kept from a config (so-vits voices have a handful).
const MAX_SPEAKERS: usize = 1024;

/// What a voice's config files say (every field optional; see the module docs).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SovitsConfig {
    pub sample_rate: Option<u32>,
    pub hop: Option<u32>,
    /// so-vits-svc's `speech_encoder` (or MoeVS's `Hubert`), as written.
    pub speech_encoder: Option<String>,
    /// Content feature width (`ssl_dim` / `HiddenSize`).
    pub ssl_dim: Option<u32>,
    /// Speaker names by id.
    pub speakers: Vec<String>,
    pub n_speakers: Option<u32>,
    pub vol_embedding: Option<bool>,
    /// The export takes speaker mix weights (MoeVS `CharaMix`).
    pub character_mix: Option<bool>,
    /// A so-vits-svc 3.0 export (MoeVS `SoVits3`).
    pub sovits3: bool,
    /// Read from a so-vits-svc training config (`data` + `model`), not a MoeVS JSON (merged: any
    /// of the merged files was one).
    pub training: bool,
    /// Which files said something (for the import record).
    pub sources: Vec<String>,
}

fn num(v: Option<&serde_json::Value>) -> Option<u32> {
    v.and_then(|x| x.as_f64())
        .filter(|x| x.is_finite() && *x > 0.0 && *x < f64::from(u32::MAX))
        .map(|x| x.round() as u32)
}

fn flag(v: Option<&serde_json::Value>) -> Option<bool> {
    v.and_then(|x| x.as_bool().or_else(|| x.as_i64().map(|n| n != 0)))
}

fn text(v: Option<&serde_json::Value>) -> Option<String> {
    v.and_then(|x| x.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty() && s.len() <= 128)
        .map(str::to_string)
}

/// A speaker name kept for display (trimmed, at most 80 characters, no control characters).
fn speaker_name(s: &str) -> String {
    s.trim().chars().filter(|c| !c.is_control()).take(80).collect()
}

impl SovitsConfig {
    /// Parse one config file (`label` names it in the record): a so-vits-svc training config or a
    /// MoeVS JSON, told apart by their keys. `Ok(None)` when it is neither (another JSON file).
    pub fn parse(bytes: &[u8], label: &str) -> Result<Option<SovitsConfig>, String> {
        let v: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| format!("{label} is not valid JSON: {e}"))?;
        let Some(obj) = v.as_object() else {
            return Ok(None);
        };
        if obj.contains_key("data") && obj.contains_key("model") {
            return Ok(Some(Self::from_training_config(&v, label)));
        }
        if obj.contains_key("Rate") || obj.contains_key("Hop") || obj.contains_key("Characters") || obj.contains_key("SoVits4") {
            return Ok(Some(Self::from_moevs(&v, label)));
        }
        Ok(None)
    }

    fn from_training_config(v: &serde_json::Value, label: &str) -> SovitsConfig {
        let data = &v["data"];
        let model = &v["model"];
        let mut speakers: Vec<(u32, String)> = v
            .get("spk")
            .and_then(|s| s.as_object())
            .map(|m| {
                m.iter()
                    .take(MAX_SPEAKERS)
                    .filter_map(|(name, id)| id.as_u64().and_then(|i| u32::try_from(i).ok()).map(|id| (id, speaker_name(name))))
                    .collect()
            })
            .unwrap_or_default();
        speakers.sort_by_key(|(id, _)| *id);
        // Names by id; a gap keeps the slot (so index = id) with an empty name.
        let mut names = Vec::new();
        for (id, name) in speakers {
            let id = id as usize;
            if id >= MAX_SPEAKERS {
                continue;
            }
            if names.len() <= id {
                names.resize(id + 1, String::new());
            }
            names[id] = name;
        }
        SovitsConfig {
            sample_rate: num(data.get("sampling_rate")),
            hop: num(data.get("hop_length")),
            speech_encoder: text(model.get("speech_encoder")),
            ssl_dim: num(model.get("ssl_dim")),
            speakers: names,
            n_speakers: num(model.get("n_speakers")),
            vol_embedding: flag(model.get("vol_embedding")),
            character_mix: None,
            sovits3: false,
            training: true,
            sources: vec![label.to_string()],
        }
    }

    fn from_moevs(v: &serde_json::Value, label: &str) -> SovitsConfig {
        let speakers: Vec<String> = v
            .get("Characters")
            .and_then(|c| c.as_array())
            .map(|a| a.iter().take(MAX_SPEAKERS).map(|x| speaker_name(x.as_str().unwrap_or_default())).collect())
            .unwrap_or_default();
        SovitsConfig {
            sample_rate: num(v.get("Rate")),
            hop: num(v.get("Hop")),
            speech_encoder: text(v.get("Hubert")),
            ssl_dim: num(v.get("HiddenSize")),
            n_speakers: (!speakers.is_empty()).then_some(speakers.len() as u32),
            speakers,
            vol_embedding: flag(v.get("Volume")),
            character_mix: flag(v.get("CharaMix")),
            sovits3: flag(v.get("SoVits3")).unwrap_or(false),
            training: false,
            sources: vec![label.to_string()],
        }
    }

    /// `self` with what `other` adds (`self` wins where both say something).
    pub fn merged(mut self, other: SovitsConfig) -> SovitsConfig {
        self.sample_rate = self.sample_rate.or(other.sample_rate);
        self.hop = self.hop.or(other.hop);
        // MoeVS always writes `vec-<dim>-layer-<n>`; the training config names the real encoder.
        self.speech_encoder = self.speech_encoder.or(other.speech_encoder);
        self.ssl_dim = self.ssl_dim.or(other.ssl_dim);
        if self.speakers.is_empty() {
            self.speakers = other.speakers;
        }
        self.n_speakers = self.n_speakers.or(other.n_speakers);
        self.vol_embedding = self.vol_embedding.or(other.vol_embedding);
        self.character_mix = self.character_mix.or(other.character_mix);
        self.sovits3 |= other.sovits3;
        self.training |= other.training;
        self.sources.extend(other.sources);
        self
    }

    /// Whether this came from a so-vits-svc training config (decided by the parser, not by what the
    /// config names: 4.0 training configs have no `speech_encoder`).
    pub fn is_training_config(&self) -> bool {
        self.training
    }
}

/// The shared content encoder for a voice trained with `speech_encoder` (so-vits-svc's or MoeVS's
/// name; `None` = not named) whose graph takes `width`-wide features. so-vits-svc 4.1's `vec768l12`
/// and 4.0's `vec256l9` (MoeSS: `hubert4.0`) are RVC's ContentVec layer 12 / layer 9 + final_proj
/// (the shared pool's `contentvec` / `contentvec-256`); every other speech encoder is refused by
/// name. Returns the kind and a note when the encoder was taken from the width: `config_read` says
/// whether any config was read (so-vits-svc 4.0 training configs name no encoder; 4.0 always ran
/// ContentVec).
pub fn encoder_for(speech_encoder: Option<&str>, width: usize, config_read: bool) -> Result<(EmbedderKind, Option<String>), String> {
    let kind_for_width = |w: usize| match w {
        768 => Some(EmbedderKind::ContentVec768),
        256 => Some(EmbedderKind::ContentVec256),
        _ => None,
    };
    let Some(raw) = speech_encoder.map(str::trim).filter(|s| !s.is_empty()) else {
        let kind = kind_for_width(width).ok_or_else(|| {
            format!("the voice takes {width}-wide content features, which no content encoder Sigil has produces (so-vits-svc's vec768l12 is 768, vec256l9 256)")
        })?;
        let assumed = if width == 768 { "vec768l12" } else { "vec256l9" };
        let note = if config_read {
            format!("the voice's config does not name its speech encoder (so-vits-svc 4.0 configs do not), so it is taken as ContentVec ({assumed}) from its {width}-wide features")
        } else {
            format!("the voice's config.json is missing, so its speech encoder is assumed to be ContentVec ({assumed}); add the config.json if the voice sounds wrong")
        };
        return Ok((kind, Some(note)));
    };
    let name = raw.to_ascii_lowercase().replace(['_', ' '], "-");
    let kind = match name.as_str() {
        "vec768l12" | "vec-768-layer-12" | "contentvec" | "contentvec-768" => EmbedderKind::ContentVec768,
        // MoeSS's name for so-vits-svc 4.0's encoder (its `hubert4.0.onnx`: ContentVec layer 9 +
        // final_proj, 256 wide).
        "vec256l9" | "vec-256-layer-9" | "contentvec-256" | "hubert4.0" | "hubert-4.0" => EmbedderKind::ContentVec256,
        "hubertsoft" | "hubert-soft" | "hubert" => {
            return Err("the voice was trained with hubert-soft, a content encoder Sigil does not have; retrain it with vec768l12 (so-vits-svc 4.1's default)".into())
        }
        "whisper-ppg" | "whisper-ppg-large" => {
            return Err(format!("the voice was trained with {raw} (Whisper PPG features), which Sigil does not have; retrain it with vec768l12"))
        }
        "cnhubertlarge" | "dphubert" | "wavlmbase+" | "wavlmbase" => {
            return Err(format!("the voice was trained with the {raw} speech encoder, which Sigil does not have; retrain it with vec768l12"))
        }
        // MoeVS writes `vec-<dim>-layer-<n>` whatever the encoder; other layers are not ContentVec's.
        other if other.starts_with("vec-") => {
            return Err(format!("the voice takes {other} content features, which no content encoder Sigil has produces"))
        }
        other => return Err(format!("the voice was trained with the speech encoder '{other}', which Sigil does not have")),
    };
    if kind.channels() != width {
        return Err(format!(
            "the voice's config names {raw} ({}-wide features) but its graph takes {width}-wide features",
            kind.channels()
        ));
    }
    Ok((kind, None))
}

/// The sample rate of a so-vits-svc voice whose config is missing, from its measured hop: the
/// released so-vits-svc configs pair each hop with one rate (4.0 / 4.1 44.1 kHz: 512; the 4.0 32k
/// branch: 320; 48 kHz community configs: 480).
pub fn rate_for_hop(hop: u32) -> Option<u32> {
    match hop {
        512 | 441 => Some(44_100),
        320 => Some(32_000),
        480 => Some(48_000),
        256 => Some(22_050),
        _ => None,
    }
}

/// Sigil's record of an imported so-vits-svc voice (`sovits.json` in the slot; the manifest's
/// `import_record`). The manifest holds the generator, rate, width, encoder and speaker count.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SovitsRecord {
    /// `sigil-sovits-import`.
    pub format: String,
    pub version: u32,
    /// Output samples per frame.
    pub hop: u32,
    /// The speech encoder as the training config named it (`None`: no training config, or a 4.0
    /// one that names none; the encoder then came from a MoeVS JSON or the width).
    pub speech_encoder: Option<String>,
    /// Speaker names by id (may be empty).
    pub speakers: Vec<String>,
    /// The export takes speaker mix weights `[T, speakers]`.
    pub character_mix: bool,
    /// The export takes a `vol` input.
    pub volume: bool,
    /// Imported file name and the config files that described it.
    pub source: String,
    pub configs: Vec<String>,
}

/// `SovitsRecord::format`.
pub const RECORD_FORMAT: &str = "sigil-sovits-import";
/// `SovitsRecord::version`.
pub const RECORD_VERSION: u32 = 1;
/// The record's file name in the slot.
pub const RECORD_FILE: &str = "sovits.json";

impl SovitsRecord {
    /// Read `<slot>/sovits.json` (bounded), checking its format and hop.
    pub fn load(slot_dir: &std::path::Path) -> Result<SovitsRecord, String> {
        let path = slot_dir.join(RECORD_FILE);
        let meta = std::fs::metadata(&path).map_err(|_| format!("{RECORD_FILE} is missing in {}; add the voice again", slot_dir.display()))?;
        if meta.len() > MAX_CONFIG_BYTES {
            return Err(format!("{} is too large", path.display()));
        }
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let r: SovitsRecord = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        if r.format != RECORD_FORMAT {
            return Err(format!("{} is not a so-vits-svc import record", path.display()));
        }
        if r.version > RECORD_VERSION {
            return Err(format!("{} was written by a newer Sigil (version {})", path.display(), r.version));
        }
        if !(32..=4_096).contains(&r.hop) {
            return Err(format!("{} has an invalid hop ({})", path.display(), r.hop));
        }
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRAINING: &str = r#"{
        "train": {"log_interval": 200},
        "data": {"sampling_rate": 44100, "hop_length": 512, "unit_interpolate_mode": "nearest"},
        "model": {"ssl_dim": 768, "n_speakers": 2, "speech_encoder": "vec768l12", "vol_embedding": true, "gin_channels": 768},
        "spk": {"alto": 1, "nyaru": 0}
    }"#;

    const MOEVS: &str = r#"{
        "Folder": "nyaru", "Name": "nyaru", "Type": "SoVits", "Rate": 44100, "Hop": 512,
        "Hubert": "vec-768-layer-12", "SoVits4": true, "SoVits3": false, "CharaMix": true,
        "Volume": false, "HiddenSize": 768, "Characters": ["nyaru", "alto"], "Cluster": ""
    }"#;

    #[test]
    fn training_configs_and_moevs_jsons_parse() {
        let t = SovitsConfig::parse(TRAINING.as_bytes(), "config.json").unwrap().unwrap();
        assert_eq!((t.sample_rate, t.hop, t.ssl_dim, t.n_speakers), (Some(44_100), Some(512), Some(768), Some(2)));
        assert_eq!(t.speech_encoder.as_deref(), Some("vec768l12"));
        assert_eq!(t.speakers, vec!["nyaru".to_string(), "alto".to_string()], "names by id");
        assert_eq!(t.vol_embedding, Some(true));
        assert!(t.is_training_config());

        let m = SovitsConfig::parse(MOEVS.as_bytes(), "nyaru.json").unwrap().unwrap();
        assert_eq!((m.sample_rate, m.hop, m.character_mix, m.vol_embedding), (Some(44_100), Some(512), Some(true), Some(false)));
        assert_eq!(m.speech_encoder.as_deref(), Some("vec-768-layer-12"));
        assert!(!m.is_training_config());

        // The training config wins; MoeVS adds the mix flag.
        let both = t.clone().merged(m.clone());
        assert_eq!(both.speech_encoder.as_deref(), Some("vec768l12"));
        assert_eq!(both.vol_embedding, Some(true));
        assert_eq!(both.character_mix, Some(true));
        assert_eq!(both.sources, vec!["config.json".to_string(), "nyaru.json".to_string()]);

        assert!(SovitsConfig::parse(br#"{"other": 1}"#, "x.json").unwrap().is_none());
        assert!(SovitsConfig::parse(b"[1, 2]", "x.json").unwrap().is_none());
        assert!(SovitsConfig::parse(b"not json", "x.json").is_err());
    }

    #[test]
    fn speech_encoders_map_or_are_refused_by_name() {
        assert_eq!(encoder_for(Some("vec768l12"), 768, true).unwrap(), (EmbedderKind::ContentVec768, None));
        assert_eq!(encoder_for(Some("vec256l9"), 256, true).unwrap(), (EmbedderKind::ContentVec256, None));
        assert_eq!(encoder_for(Some("vec-768-layer-12"), 768, true).unwrap().0, EmbedderKind::ContentVec768);
        assert_eq!(encoder_for(Some("vec-256-layer-9"), 256, true).unwrap().0, EmbedderKind::ContentVec256);
        // MoeSS's so-vits-svc 4.0 JSONs (P5-D review).
        assert_eq!(encoder_for(Some("hubert4.0"), 256, true).unwrap(), (EmbedderKind::ContentVec256, None));
        for (name, width) in [("hubertsoft", 256), ("whisper-ppg", 1024), ("whisper-ppg-large", 1280), ("cnhubertlarge", 1024), ("dphubert", 768), ("wavlmbase+", 768), ("vec-1024-layer-24", 1024), ("mystery", 768)] {
            let err = encoder_for(Some(name), width, true).unwrap_err();
            assert!(err.contains(name) || err.contains("hubert-soft"), "{name}: {err}");
        }
        // A config that disagrees with the graph.
        assert!(encoder_for(Some("vec768l12"), 256, true).unwrap_err().contains("256"));
        // No config: assumed from the width, with a warning; other widths are refused.
        let (k, w) = encoder_for(None, 768, false).unwrap();
        assert_eq!(k, EmbedderKind::ContentVec768);
        let w = w.unwrap();
        assert!(w.contains("vec768l12") && w.contains("missing"), "{w}");
        assert!(encoder_for(None, 1024, false).is_err());
        // A config that names no encoder (so-vits-svc 4.0) is not called missing.
        let (k, w) = encoder_for(None, 256, true).unwrap();
        assert_eq!(k, EmbedderKind::ContentVec256);
        let w = w.unwrap();
        assert!(w.contains("does not name") && w.contains("vec256l9") && !w.contains("missing"), "{w}");
    }

    /// so-vits-svc 4.0 training configs (no `speech_encoder`) are training configs; a MoeSS JSON
    /// naming `hubert4.0` is not (P5-D review: both used to be decided by the encoder's name).
    #[test]
    fn the_parser_decides_what_a_config_is() {
        let v40 = br#"{"train": {}, "data": {"sampling_rate": 44100, "hop_length": 512}, "model": {"ssl_dim": 256, "n_speakers": 200}, "spk": {"nyaru": 0}}"#;
        let t = SovitsConfig::parse(v40, "config.json").unwrap().unwrap();
        assert!(t.is_training_config());
        assert_eq!((t.speech_encoder.as_deref(), t.ssl_dim), (None, Some(256)));
        let moess = br#"{"Folder": "nyaru", "Name": "nyaru", "Type": "SoVits", "Rate": 44100, "Hop": 512, "Hubert": "hubert4.0", "SoVits4": true, "Characters": ["nyaru"]}"#;
        let m = SovitsConfig::parse(moess, "nyaru.json").unwrap().unwrap();
        assert!(!m.is_training_config());
        let both = t.merged(m);
        assert!(both.is_training_config());
        assert_eq!(both.speech_encoder.as_deref(), Some("hubert4.0"), "MoeSS fills what the 4.0 config does not say");
        assert_eq!(encoder_for(both.speech_encoder.as_deref(), 256, true).unwrap().0, EmbedderKind::ContentVec256);
    }

    #[test]
    fn rates_follow_the_released_hops() {
        assert_eq!(rate_for_hop(512), Some(44_100));
        assert_eq!(rate_for_hop(320), Some(32_000));
        assert_eq!(rate_for_hop(480), Some(48_000));
        assert_eq!(rate_for_hop(300), None);
    }

    #[test]
    fn the_record_round_trips_and_is_checked() {
        let dir = std::env::temp_dir().join(format!("sigil-sovits-record-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(SovitsRecord::load(&dir).unwrap_err().contains("missing"));
        let r = SovitsRecord {
            format: RECORD_FORMAT.into(),
            version: RECORD_VERSION,
            hop: 512,
            speakers: vec!["a".into()],
            ..SovitsRecord::default()
        };
        std::fs::write(dir.join(RECORD_FILE), serde_json::to_vec(&r).unwrap()).unwrap();
        assert_eq!(SovitsRecord::load(&dir).unwrap(), r);
        let bad = SovitsRecord { hop: 0, ..r.clone() };
        std::fs::write(dir.join(RECORD_FILE), serde_json::to_vec(&bad).unwrap()).unwrap();
        assert!(SovitsRecord::load(&dir).unwrap_err().contains("hop"));
        let other = SovitsRecord { format: "x".into(), ..r };
        std::fs::write(dir.join(RECORD_FILE), serde_json::to_vec(&other).unwrap()).unwrap();
        assert!(SovitsRecord::load(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
