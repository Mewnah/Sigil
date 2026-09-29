//! RVC model slot manifest (`manifest.json` under voice-changer/models/<slot>/).
//! Optional: without it the manifest is inferred in memory (see `bases::load_or_infer_manifest`).

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RvcVersion {
    V1,
    #[default]
    V2,
    Nono,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub generator: String,
    #[serde(default)]
    pub index: Option<String>,
    #[serde(default)]
    pub speaker_id: i64,
    #[serde(default)]
    pub hubert: Option<String>,
}

fn default_format() -> String {
    "rvc".into()
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
}

pub fn load_manifest(slot_dir: &Path) -> Result<RvcManifest, String> {
    let path = slot_dir.join("manifest.json");
    let bytes = fs::read(&path).map_err(|e| format!("Cannot read {}: {}", path.display(), e))?;
    let mut m: RvcManifest = serde_json::from_slice(&bytes)
        .map_err(|e| format!("Invalid manifest.json in {}: {}", slot_dir.display(), e))?;
    if m.format != "rvc" {
        return Err(format!("Unsupported voice-changer format '{}'", m.format));
    }
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
