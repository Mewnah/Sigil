//! Voice slot manifest (`manifest.json` under voice-changer/models/<slot>/).
//!
//! Two slot formats share this file (the factory, `voice_changer::factory`, dispatches on
//! [`RvcManifest::slot_format`]):
//! - `"rvc"`: a bring-your-own RVC generator `.onnx` (the legacy full-decode path, `RvcModel`).
//!   Optional: without `manifest.json` the manifest is inferred in memory
//!   (`bases::load_or_infer_manifest`), and [`load_manifest`] only ever returns this format.
//! - `"sigil-rvc-1"` ([`SIGIL_RVC_FORMAT`]): a voice imported from `.pth` / `.zip` into a Sigil
//!   template (ADR 0004): `weights.f32.bin` + `speakers.f32.bin` + `sigil-import.json`, run by the
//!   Sigil Voice Engine (`neural_audio::vc::engine`). Written by the slot store (P3-B,
//!   `vc::import::slot`); read with [`load_slot_manifest`].
//!
//! # Frozen by P3-S0 (manifest v2; change only through INTEGRATE)
//! - The v1 fields keep their names, defaults and meaning; every v2 field is optional and omitted
//!   from the JSON when unset, so a v1 manifest round-trips unchanged.
//! - v2 fields: `name`, `template`, `weights`, `speakers`, `speaker_count`, `embedder`, `vocoder`,
//!   `fp16_ok`, `source_sha256`, `import_record`, `created_unix_ms`.
//! - [`load_manifest`] keeps its P0 contract (legacy ONNX slots only; a template voice is refused
//!   with a clear message), so the legacy path never tries to run a template slot.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// `format` of a bring-your-own RVC ONNX slot (the only v1 format).
pub const LEGACY_RVC_FORMAT: &str = "rvc";
/// `format` of a Sigil template voice (imported `.pth` / `.zip`, ADR 0004).
pub const SIGIL_RVC_FORMAT: &str = "sigil-rvc-1";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RvcVersion {
    V1,
    #[default]
    V2,
    Nono,
}

/// How a slot is run (from `manifest.format`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotFormat {
    /// Bring-your-own RVC generator `.onnx`: `RvcModel`, full decode (the proven legacy path).
    LegacyOnnxRvc,
    /// Sigil template voice: the Sigil Voice Engine (split graphs, partial decode, P3-C1).
    SigilRvc,
    /// A format this build does not run (a later family such as so-vits or DDSP-SVC, P5).
    Unknown(String),
}

impl SlotFormat {
    pub fn from_format(format: &str) -> Self {
        match format.trim() {
            LEGACY_RVC_FORMAT => SlotFormat::LegacyOnnxRvc,
            SIGIL_RVC_FORMAT => SlotFormat::SigilRvc,
            other => SlotFormat::Unknown(other.to_string()),
        }
    }

    /// The `format` string.
#[allow(dead_code)] // P3-S0 frozen API; its P3 consumer lands with the owning workstream.
pub fn as_str(&self) -> &str {
        match self {
            SlotFormat::LegacyOnnxRvc => LEGACY_RVC_FORMAT,
            SlotFormat::SigilRvc => SIGIL_RVC_FORMAT,
            SlotFormat::Unknown(s) => s,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RvcManifest {
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default)]
    pub version: RvcVersion,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_true")]
    pub if_f0: bool,
    #[serde(default = "default_emb_v2")]
    pub emb_channels: u32,
    #[serde(default = "default_emb_layer_v2")]
    pub emb_layer: u32,
    #[serde(default)]
    pub use_final_proj: bool,
    /// Generator `.onnx` of a legacy slot (required for format `rvc`; empty for template voices).
    #[serde(default)]
    pub generator: String,
    #[serde(default)]
    pub index: Option<String>,
    #[serde(default)]
    pub speaker_id: i64,
    #[serde(default)]
    pub hubert: Option<String>,

    // ---- v2 (P3-S0): all optional, omitted when unset ----
    /// Display name (the folder name is the id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Template id of a `sigil-rvc-1` voice, e.g. `v2-40k-f0-hifigan` (ADR 0004).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    /// Weights image of a template voice (`weights.f32.bin`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<String>,
    /// Speaker table of a template voice (`speakers.f32.bin`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speakers: Option<String>,
    /// Rows of the speaker table.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_count: Option<u32>,
    /// Content encoder the voice was trained with: `contentvec` (768 / layer 12, v2),
    /// `contentvec-256` (v1: layer 9 + final_proj), `spin-v2`, `hubert-ja`, `hubert-ko`,
    /// `hubert-zh` (Applio / RVC metadata names are mapped by the importer).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedder: Option<String>,
    /// Vocoder of the generator: `hifigan` (HiFiGAN-NSF), `refinegan`, `mrf`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocoder: Option<String>,
    /// The voice's weights fit half precision and its fp16 smoke decode (when run) stayed finite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fp16_ok: Option<bool>,
    /// sha256 of the imported source file (the import cache key).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_sha256: Option<String>,
    /// The importer's record in the slot (`sigil-import.json`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub import_record: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_unix_ms: Option<u64>,
}

fn default_format() -> String {
    LEGACY_RVC_FORMAT.into()
}
fn default_sample_rate() -> u32 {
    40_000
}
fn default_true() -> bool {
    true
}
fn default_emb_v2() -> u32 {
    768
}
fn default_emb_layer_v2() -> u32 {
    12
}

impl Default for RvcManifest {
    /// The serde defaults (a v2 40 kHz f0 legacy slot without a generator).
    fn default() -> Self {
        Self {
            format: default_format(),
            version: RvcVersion::default(),
            sample_rate: default_sample_rate(),
            if_f0: default_true(),
            emb_channels: default_emb_v2(),
            emb_layer: default_emb_layer_v2(),
            use_final_proj: false,
            generator: String::new(),
            index: None,
            speaker_id: 0,
            hubert: None,
            name: None,
            template: None,
            weights: None,
            speakers: None,
            speaker_count: None,
            embedder: None,
            vocoder: None,
            fp16_ok: None,
            source_sha256: None,
            import_record: None,
            created_unix_ms: None,
        }
    }
}

impl RvcManifest {
    pub fn generator_path(&self, slot_dir: &Path) -> PathBuf {
        slot_dir.join(&self.generator)
    }

    pub fn index_path(&self, slot_dir: &Path) -> Option<PathBuf> {
        self.index.as_ref().map(|i| slot_dir.join(i))
    }

    pub fn wants_f0(&self) -> bool {
        self.if_f0 && self.version != RvcVersion::Nono
    }

    /// How the slot is run.
    pub fn slot_format(&self) -> SlotFormat {
        SlotFormat::from_format(&self.format)
    }
}

/// Read and parse `<slot>/manifest.json` with the v1 normalization (Nono is f0-less; a v1 voice
/// with the v2 embedding defaults gets the v1 ones).
fn read_manifest(slot_dir: &Path) -> Result<RvcManifest, String> {
    let path = slot_dir.join("manifest.json");
    let bytes = fs::read(&path).map_err(|e| format!("Cannot read {}: {}", path.display(), e))?;
    let mut m: RvcManifest = serde_json::from_slice(&bytes)
        .map_err(|e| format!("Invalid manifest.json in {}: {}", slot_dir.display(), e))?;
    // Nono is always f0-less.
    if m.version == RvcVersion::Nono {
        m.if_f0 = false;
    }
    // v1 defaults if emb looks unset for v1
    if m.version == RvcVersion::V1 && m.emb_channels == 768 {
        m.emb_channels = 256;
        m.emb_layer = 9;
        m.use_final_proj = true;
    }
    Ok(m)
}

/// Load a legacy (bring-your-own ONNX, format `rvc`) slot manifest. Any other format is refused;
/// a template voice says which engine runs it.
pub fn load_manifest(slot_dir: &Path) -> Result<RvcManifest, String> {
    let m = read_manifest(slot_dir)?;
    match m.slot_format() {
        SlotFormat::LegacyOnnxRvc => {}
        SlotFormat::SigilRvc => {
            return Err(format!(
                "{} holds an imported voice ({SIGIL_RVC_FORMAT}); it runs on the Sigil Voice Engine, not as an ONNX generator",
                slot_dir.display()
            ))
        }
        SlotFormat::Unknown(f) => return Err(format!("Unsupported voice-changer format '{f}'")),
    }
    if m.generator.trim().is_empty() {
        return Err(format!("Invalid manifest.json in {}: missing field `generator`", slot_dir.display()));
    }
    Ok(m)
}

/// Load a slot manifest of any format this build knows (`rvc` or `sigil-rvc-1`), for the engine
/// factory and the slot store. A `sigil-rvc-1` manifest must name its template and weights.
pub fn load_slot_manifest(slot_dir: &Path) -> Result<RvcManifest, String> {
    let m = read_manifest(slot_dir)?;
    match m.slot_format() {
        SlotFormat::LegacyOnnxRvc => load_manifest(slot_dir),
        SlotFormat::SigilRvc => {
            if m.template.as_deref().is_none_or(|t| t.trim().is_empty()) {
                return Err(format!("Invalid manifest.json in {}: a {SIGIL_RVC_FORMAT} voice needs `template`", slot_dir.display()));
            }
            if m.weights.as_deref().is_none_or(|w| w.trim().is_empty()) {
                return Err(format!("Invalid manifest.json in {}: a {SIGIL_RVC_FORMAT} voice needs `weights`", slot_dir.display()));
            }
            Ok(m)
        }
        SlotFormat::Unknown(f) => Err(format!("Unsupported voice-changer format '{f}'")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn slot(tag: &str, manifest: &str) -> TempDir {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("sigil-vc-manifest-{tag}-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("manifest.json"), manifest).unwrap();
        TempDir(dir)
    }

    /// A v1 manifest (P0) parses, normalizes and re-serializes without any v2 key.
    #[test]
    fn v1_manifest_round_trips_without_v2_keys() {
        let s = slot("v1", r#"{"generator":"gen.onnx","version":"v1","index":"added.index"}"#);
        let m = load_manifest(&s.0).unwrap();
        assert_eq!(m.slot_format(), SlotFormat::LegacyOnnxRvc);
        assert_eq!((m.emb_channels, m.emb_layer, m.use_final_proj), (256, 9, true));
        let json = serde_json::to_value(&m).unwrap();
        for key in ["name", "template", "weights", "embedder", "vocoder", "fp16_ok", "source_sha256"] {
            assert!(json.get(key).is_none(), "{key} must be omitted when unset");
        }
        assert_eq!(load_slot_manifest(&s.0).unwrap(), m);
    }

    /// The legacy loader keeps refusing what it cannot run; the slot loader accepts template voices.
    #[test]
    fn template_voices_load_only_through_the_slot_loader() {
        let s = slot(
            "tpl",
            r#"{"format":"sigil-rvc-1","template":"v2-40k-f0-hifigan","weights":"weights.f32.bin",
                "speakers":"speakers.f32.bin","speaker_count":109,"embedder":"contentvec","vocoder":"hifigan","fp16_ok":true}"#,
        );
        let err = load_manifest(&s.0).unwrap_err();
        assert!(err.contains("Sigil Voice Engine"), "{err}");
        let m = load_slot_manifest(&s.0).unwrap();
        assert_eq!(m.slot_format(), SlotFormat::SigilRvc);
        assert_eq!(m.template.as_deref(), Some("v2-40k-f0-hifigan"));
        assert_eq!(m.speaker_count, Some(109));

        let s = slot("tpl-bad", r#"{"format":"sigil-rvc-1","weights":"weights.f32.bin"}"#);
        assert!(load_slot_manifest(&s.0).unwrap_err().contains("template"));
        let s = slot("legacy-no-gen", r#"{"format":"rvc"}"#);
        assert!(load_manifest(&s.0).unwrap_err().contains("generator"));
        let s = slot("future", r#"{"format":"ddsp-svc-6.3"}"#);
        assert!(load_slot_manifest(&s.0).unwrap_err().contains("Unsupported"));
        assert!(load_manifest(&s.0).unwrap_err().contains("Unsupported"));
    }

    #[test]
    fn default_matches_the_serde_defaults() {
        let parsed: RvcManifest = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed, RvcManifest::default());
        assert_eq!(SlotFormat::from_format("sigil-rvc-1").as_str(), SIGIL_RVC_FORMAT);
    }
}
