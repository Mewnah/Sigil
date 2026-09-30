//! The GGUF translation models Sigil offers (`model-catalog/llm-gguf.json`, family `mt-llm`): the
//! store entries (files, sizes, `min_vram_mb`) plus the per-model data of the fragment's `routing`
//! table (prompt format, role, availability, languages, KV size, sampler), embedded with
//! `include_str!` so it is present even when a catalog override replaces the entries.

use super::{LlmDirection, LlmSamplingPreset, MODEL_FAMILY};
use crate::services::model_store::{Catalog, CatalogEntry};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The fragment, embedded for its `routing` table.
const FRAGMENT_JSON: &str = include_str!("../../../model-catalog/llm-gguf.json");

/// Default `-c` when neither the config nor the model says otherwise.
pub const DEFAULT_CTX: u32 = 4096;

/// Runtime pack ids per variant (`model-catalog/runtime-llama.json`).
pub const RUNTIME_VULKAN: &str = "runtime/llama-server-vulkan";
pub const RUNTIME_CUDA13: &str = "runtime/llama-server-cuda13";
pub const RUNTIME_CPU: &str = "runtime/llama-server-cpu";

/// Whether the pinned llama-server can run a model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Availability {
    #[default]
    Ok,
    /// Needs a llama.cpp feature the pinned build lacks (the 1.25-bit Hy-MT2 needs STQ1_0).
    NeedsRuntimeUpdate,
    /// May not work (MADLAD-400 is an encoder-decoder model).
    Experimental,
}

impl Availability {
    pub fn as_str(self) -> &'static str {
        match self {
            Availability::Ok => "ok",
            Availability::NeedsRuntimeUpdate => "needs_runtime_update",
            Availability::Experimental => "experimental",
        }
    }
}

/// Per-model data from `routing.models`.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ModelMeta {
    pub prompt_format: String,
    /// `default`, `challenger`, `alternative`, `high_gpu`, `ab`, `specialist`.
    pub role: String,
    pub availability: Availability,
    pub kv_bytes_per_token: Option<u64>,
    pub languages: Vec<String>,
    pub directions: Vec<LlmDirection>,
    pub sampling: Option<LlmSamplingPreset>,
}

impl ModelMeta {
    pub fn specialist(&self) -> bool {
        self.role == "specialist"
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct Routing {
    ctx_default: Option<u32>,
    models: BTreeMap<String, ModelMeta>,
}

#[derive(Debug, Default, Deserialize)]
struct Fragment {
    #[serde(default)]
    routing: Routing,
    #[serde(default)]
    entries: Vec<serde_json::Value>,
}

struct Embedded {
    ctx_default: u32,
    /// Catalog order of the fragment's entries.
    order: Vec<String>,
    meta: BTreeMap<String, ModelMeta>,
}

fn embedded() -> &'static Embedded {
    static E: OnceLock<Embedded> = OnceLock::new();
    E.get_or_init(|| {
        let frag: Fragment = serde_json::from_str(FRAGMENT_JSON).unwrap_or_else(|e| {
            tracing::warn!(target: "sigil::llm_server", "llm-gguf.json routing table unreadable: {e}");
            Fragment::default()
        });
        let order = frag
            .entries
            .iter()
            .filter_map(|e| e.get("id").and_then(|v| v.as_str()).map(str::to_string))
            .collect();
        Embedded {
            ctx_default: frag.routing.ctx_default.unwrap_or(DEFAULT_CTX),
            order,
            meta: frag.routing.models,
        }
    })
}

/// Default context of the catalog models.
pub fn ctx_default() -> u32 {
    embedded().ctx_default
}

/// Metadata of a catalog model (`None` for ids the fragment does not describe).
pub fn meta(id: &str) -> Option<&'static ModelMeta> {
    embedded().meta.get(id)
}

/// Ids in the fragment's order.
pub fn ordered_ids() -> &'static [String] {
    &embedded().order
}

/// Runtime pack id of a variant.
pub fn runtime_id(variant: &str) -> Option<&'static str> {
    match variant {
        "vulkan" => Some(RUNTIME_VULKAN),
        "cuda13" => Some(RUNTIME_CUDA13),
        "cpu" => Some(RUNTIME_CPU),
        _ => None,
    }
}

/// A translation model entry of the effective catalog (family [`MODEL_FAMILY`]).
pub fn entry<'a>(catalog: &'a Catalog, id: &str) -> Option<&'a CatalogEntry> {
    catalog.get(id).filter(|e| e.family == MODEL_FAMILY)
}

/// Every translation model of the effective catalog: the fragment's order first, then any other
/// `mt-llm` entries (a development override may add some).
pub fn entries(catalog: &Catalog) -> Vec<&CatalogEntry> {
    let mut out: Vec<&CatalogEntry> = ordered_ids()
        .iter()
        .filter_map(|id| entry(catalog, id))
        .collect();
    for e in &catalog.entries {
        if e.family == MODEL_FAMILY && !out.iter().any(|o| o.id == e.id) {
            out.push(e);
        }
    }
    out
}

/// The GGUF to pass to `-m`: the first `.gguf` file of the entry that is not a vision projector (for
/// a split model, its `-00001-of-` part).
pub fn model_file(entry: &CatalogEntry) -> Option<&str> {
    let ggufs: Vec<&str> = entry
        .files
        .iter()
        .map(|f| f.path.as_str())
        .filter(|p| p.to_ascii_lowercase().ends_with(".gguf") && !p.to_ascii_lowercase().contains("mmproj"))
        .collect();
    ggufs
        .iter()
        .find(|p| p.contains("-00001-of-"))
        .or_else(|| ggufs.first())
        .copied()
}

/// Download size of the entry's GGUF files (bytes).
pub fn gguf_bytes(entry: &CatalogEntry) -> u64 {
    entry
        .files
        .iter()
        .filter(|f| f.path.to_ascii_lowercase().ends_with(".gguf"))
        .map(|f| f.size)
        .fold(0u64, u64::saturating_add)
}

/// Path of the model file inside an install directory.
pub fn model_path(install_dir: &Path, entry: &CatalogEntry) -> Option<PathBuf> {
    model_file(entry).map(|f| install_dir.join(f))
}

/// The `--alias` the server reports for a catalog id (`mt/hy-mt2-1.8b-q4_k_m` →
/// `hy-mt2-1.8b-q4_k_m`) or a user file (its sanitized stem).
pub fn alias_for(model: &str) -> String {
    let base = if looks_like_path(model) {
        Path::new(model)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    } else {
        model.rsplit('/').next().unwrap_or(model).to_string()
    };
    let cleaned: String = base
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches('-');
    if trimmed.is_empty() {
        "local-model".into()
    } else {
        trimmed.chars().take(80).collect()
    }
}

/// `config.model` is a file path rather than a catalog id (a drive letter, a UNC prefix or a
/// separator other than the id's `/`, or a `.gguf` extension).
pub fn looks_like_path(model: &str) -> bool {
    let m = model.trim();
    m.contains('\\') || m.starts_with("//") || m.as_bytes().get(1) == Some(&b':') || m.to_ascii_lowercase().ends_with(".gguf")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::resources::policy::ids;

    fn catalog() -> Catalog {
        Catalog::embedded().expect("embedded catalog")
    }

    #[test]
    fn every_fragment_model_has_routing_data_and_a_runnable_file() {
        let cat = catalog();
        let order = ordered_ids();
        assert!(order.len() >= 15, "{order:?}");
        for id in order {
            let e = entry(&cat, id).unwrap_or_else(|| panic!("{id} is not an mt-llm catalog entry"));
            let m = meta(id).unwrap_or_else(|| panic!("{id} has no routing data"));
            assert!(
                [
                    "hy-mt2",
                    "translategemma",
                    "milmmt",
                    "seed-x",
                    "lmt",
                    "sakura",
                    "vntl",
                    "madlad",
                    "generic"
                ]
                .contains(&m.prompt_format.as_str()),
                "{id}: prompt format {}",
                m.prompt_format
            );
            assert!(
                ["default", "challenger", "alternative", "high_gpu", "ab", "specialist"].contains(&m.role.as_str()),
                "{id}: role {}",
                m.role
            );
            let file = model_file(e).unwrap_or_else(|| panic!("{id} has no gguf"));
            assert!(!file.contains("mmproj"));
            assert!(gguf_bytes(e) > 100_000_000, "{id}");
            assert!(e.min_vram_mb.unwrap_or(0) > 0, "{id} needs min_vram_mb");
            assert!(e.files.iter().all(|f| f.sha256.is_some()), "{id} must be hash-pinned");
            assert!(
                e.files
                    .iter()
                    .flat_map(|f| &f.sources)
                    .all(|s| s.revision.as_deref().is_some_and(|r| r.len() == 40)),
                "{id} must be pinned to a commit"
            );
            for d in &m.directions {
                assert!(m.languages.contains(&d.from) && m.languages.contains(&d.to), "{id}: {d:?}");
            }
        }
        // No routing row without an entry.
        for id in embedded().meta.keys() {
            assert!(order.contains(id), "routing row {id} has no entry");
        }
        assert_eq!(ctx_default(), 4096);
    }

    #[test]
    fn policy_ids_exist_with_the_policy_footprints() {
        // resources::policy recommends these ids with these footprints; the catalog must agree
        // (the TS drift test checks policy.rs against resources/format.ts).
        let cat = catalog();
        for (id, mb) in [
            (ids::MT_HY_MT2_1_8B, 1600),
            (ids::MT_HY_MT2_1_8B_TINY, 700),
            (ids::MT_HY_MT2_7B, 5300),
            (ids::MT_MILMMT_4B, 3000),
            (ids::MT_MILMMT_12B, 8100),
            (ids::MT_HY_MT2_30B_A3B, 19_600),
        ] {
            let e = entry(&cat, id).unwrap_or_else(|| panic!("{id} missing"));
            assert_eq!(e.min_vram_mb, Some(mb), "{id}");
        }
        assert_eq!(meta(ids::MT_HY_MT2_1_8B_TINY).unwrap().availability, Availability::NeedsRuntimeUpdate);
        assert_eq!(meta(ids::MT_HY_MT2_1_8B).unwrap().availability, Availability::Ok);
        assert_eq!(meta(ids::MT_HY_MT2_1_8B).unwrap().kv_bytes_per_token, Some(65_536));
        assert!(meta(ids::MT_HY_MT2_1_8B)
            .unwrap()
            .languages
            .iter()
            .any(|l| l == "zh-Hant"));
        assert_eq!(meta("mt/madlad400-3b-q4_k_m").unwrap().availability, Availability::Experimental);
        assert!(meta("mt/vntl-llama3-8b-v2-q5_k_m").unwrap().specialist());
    }

    #[test]
    fn aliases_and_paths() {
        assert_eq!(alias_for("mt/hy-mt2-1.8b-q4_k_m"), "hy-mt2-1.8b-q4_k_m");
        assert_eq!(alias_for(r"D:\Models\My Model (Q4).gguf"), "my-model--q4");
        assert_eq!(alias_for("C:/x/!!!.gguf"), "local-model");
        assert!(looks_like_path(r"C:\m\a.gguf"));
        assert!(looks_like_path("D:/m/a.GGUF"));
        assert!(looks_like_path(r"\\nas\share\a"));
        assert!(!looks_like_path("mt/hy-mt2-7b-q4_k_m"));
        assert!(!looks_like_path("auto"));
        assert_eq!(runtime_id("cuda13"), Some(RUNTIME_CUDA13));
        assert_eq!(runtime_id("auto"), None);
    }

    #[test]
    fn split_and_projector_files_are_handled() {
        let mut e = entry(&catalog(), ids::MT_HY_MT2_1_8B).unwrap().clone();
        let mut proj = e.files[0].clone();
        proj.path = "model.mmproj-f16.gguf".into();
        let mut p2 = e.files[0].clone();
        p2.path = "model-00002-of-00002.gguf".into();
        let mut p1 = e.files[0].clone();
        p1.path = "model-00001-of-00002.gguf".into();
        e.files = vec![proj, p2, p1];
        assert_eq!(model_file(&e), Some("model-00001-of-00002.gguf"));
        assert_eq!(
            model_path(Path::new("C:/i"), &e),
            Some(Path::new("C:/i").join("model-00001-of-00002.gguf"))
        );
    }
}
