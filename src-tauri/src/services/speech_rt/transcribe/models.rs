//! The GGUF speech models transcribe.cpp runs for the `speech` plugin: catalog resolution
//! (`model-catalog/stt-gguf.json`, located only through `model_store::installed_path`),
//! bring-your-own files, and the per-family rules the engine enforces whatever the router says.
//!
//! Families (transcribe.cpp 0.2.4 `general.architecture`):
//!
//! | Family | Arch | Rules |
//! |---|---|---|
//! | Qwen3-ASR (1.7B Q8_0 / BF16) | `qwen3_asr` | 30 languages + language ID; the hint is honoured (`language X<asr_text>` prefix); no context biasing in 0.2.4; single-shot decode (cancel is checked only before it starts, so stale results are dropped by the caller) |
//! | Cohere Transcribe (Q4_K) | `cohere_asr` | no language ID: it needs the language, and it is **never** used for ja / zh / ko / ru / yue (it silently drops Japanese sentences and has no Russian; research §1.1), whatever transcribe.cpp lists |
//! | Granite Speech 5.0 TurboCTC | `granite5_ctc` | English only (CTC: cannot loop); lower-case text without punctuation in 0.2.4 |
//! | Voxtral Mini 4B Realtime (Q4_K_M) | `voxtral_realtime` | native streaming, 13 languages; the streaming path auto-detects (language hints are ignored there) |
//! | Whisper | `whisper` | initial prompt (hotwords), temperature fallback off for bounded latency |
//!
//! | Confucius4-R2T2 | any arch containing `r2t2` | append-only streaming (only committed text is shown) |
//!
//! Confucius4-R2T2 (NetEase's streaming fine-tune of Qwen3-ASR-1.7B) is not an architecture of the
//! pinned transcribe.cpp 0.2.4; later transcribe.cpp releases run it (official GGUFs:
//! `netease-youdao/Confucius4-R2T2-GGUF`). It has no catalog entry until the runtime is re-pinned;
//! the family mapping above makes a bring-your-own R2T2 file stream append-only on such a runtime,
//! and the engine's opt-in `appendOnly` mode gives any streaming model (Voxtral Realtime)
//! flicker-free partials.

use std::path::{Path, PathBuf};

use super::engine::{catalog, ErrorKind, TranscribeError};
use super::types::estimated_model_vram_mb;

/// Device memory (MB) a GPU runtime context costs while any engine of that runtime is loaded:
/// ggml-cuda's CUDA context and cuBLAS state measured 378-380 MB on the RTX 4090 (ADR 0003b). The
/// Vulkan build is unmeasured; the same budget is assumed.
pub const GPU_CONTEXT_MB: u64 = 400;

/// A transcribe.cpp model family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    Qwen3Asr,
    Cohere,
    Granite,
    VoxtralRealtime,
    /// Confucius4-R2T2: append-only streaming (a later transcribe.cpp than the pinned 0.2.4).
    R2t2,
    Whisper,
    Other,
}

/// Catalog id prefix → family (before the model is loaded; the loaded arch wins afterwards).
const ID_FAMILIES: &[(&str, Family)] = &[
    ("stt/qwen3-asr-1.7b", Family::Qwen3Asr),
    ("stt/cohere-", Family::Cohere),
    ("stt/granite-speech-", Family::Granite),
    ("stt/voxtral-", Family::VoxtralRealtime),
    ("stt/confucius4-r2t2", Family::R2t2),
];

impl Family {
    /// Family of a transcribe.cpp architecture name (`ModelInfo::arch`).
    pub fn from_arch(arch: &str) -> Family {
        let a = arch.trim().to_ascii_lowercase();
        if a.contains("r2t2") {
            Family::R2t2
        } else if a == "qwen3_asr" {
            Family::Qwen3Asr
        } else if a.starts_with("cohere") {
            Family::Cohere
        } else if a.starts_with("granite") {
            Family::Granite
        } else if a == "voxtral_realtime" {
            Family::VoxtralRealtime
        } else if a == "whisper" {
            Family::Whisper
        } else {
            Family::Other
        }
    }

    /// Family of a catalog id (`Other` for unknown ids and file paths).
    pub fn from_id(id: &str) -> Family {
        ID_FAMILIES
            .iter()
            .find(|(prefix, _)| id.starts_with(prefix))
            .map_or(Family::Other, |(_, f)| *f)
    }

    /// Languages this family must never transcribe, whatever the model lists.
    pub fn never_languages(self) -> &'static [&'static str] {
        match self {
            Family::Cohere => &["ja", "zh", "ko", "ru", "yue"],
            _ => &[],
        }
    }

    /// The family cannot detect the language and must be told it.
    pub fn needs_language(self) -> bool {
        self == Family::Cohere
    }

    /// Streams show only committed text (never retract), whatever the engine options say.
    pub fn append_only(self) -> bool {
        self == Family::R2t2
    }

    /// Initial prompt / hotwords through the decoder prompt (Whisper only in transcribe.cpp 0.2.4).
    pub fn takes_prompt(self) -> bool {
        self == Family::Whisper
    }

    pub fn label(self) -> &'static str {
        match self {
            Family::Qwen3Asr => "Qwen3-ASR",
            Family::Cohere => "Cohere Transcribe",
            Family::Granite => "Granite Speech",
            Family::VoxtralRealtime => "Voxtral Realtime",
            Family::R2t2 => "Confucius4-R2T2",
            Family::Whisper => "Whisper",
            Family::Other => "transcribe.cpp model",
        }
    }
}

/// A model file resolved from an engine spec.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedModel {
    /// Catalog id, or the file name of a bring-your-own model (logs, `EngineInfo::model`).
    pub label: String,
    pub catalog_id: Option<String>,
    /// Human title (catalog), else the label.
    pub title: String,
    /// The `.gguf` (or legacy whisper.cpp `.bin`) file.
    pub path: PathBuf,
    pub file_bytes: u64,
    /// Device memory (MB) the model needs on a GPU besides the runtime context: the catalog's
    /// `min_vram_mb` minus [`GPU_CONTEXT_MB`], else 1.5 × the file.
    pub gpu_mb: u64,
    /// Family from the catalog id (`Other` for files; the loaded arch refines it).
    pub family: Family,
}

fn is_model_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf") || e.eq_ignore_ascii_case("bin"))
}

/// Device memory of a model file on a GPU, from the catalog's `min_vram_mb` (total incl. the
/// runtime context) or the file size (pure).
pub(crate) fn gpu_footprint_mb(min_vram_mb: Option<u64>, file_bytes: u64) -> u64 {
    match min_vram_mb {
        Some(total) if total > GPU_CONTEXT_MB => total - GPU_CONTEXT_MB,
        _ => estimated_model_vram_mb(file_bytes).max(1),
    }
}

/// Resolve `model` (a catalog id of a GGUF speech model, or an absolute `.gguf` / `.bin` path).
/// A catalog model that is not installed is [`ErrorKind::NotInstalled`] with a message that starts
/// with its id (the pipeline reads the id from there); engines never download.
pub fn resolve(model: &str) -> Result<ResolvedModel, TranscribeError> {
    let model = model.trim();
    if model.is_empty() {
        return Err(TranscribeError::new(ErrorKind::InvalidInput, "no transcribe.cpp model given"));
    }
    let path = Path::new(model);
    if path.is_absolute() {
        if !path.is_file() {
            return Err(TranscribeError::new(
                ErrorKind::InvalidInput,
                format!("the model file {} does not exist", path.display()),
            ));
        }
        if !is_model_file(path) {
            return Err(TranscribeError::new(
                ErrorKind::InvalidInput,
                format!("{} is not a GGUF speech model (.gguf, or a whisper.cpp .bin)", path.display()),
            ));
        }
        let file_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let label = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| model.to_string());
        return Ok(ResolvedModel {
            title: label.clone(),
            label,
            catalog_id: None,
            path: path.to_path_buf(),
            file_bytes,
            gpu_mb: gpu_footprint_mb(None, file_bytes),
            family: Family::Other,
        });
    }
    let catalog = catalog()?;
    let entry = catalog.get(model).ok_or_else(|| {
        TranscribeError::new(
            ErrorKind::InvalidInput,
            format!("{model} is not a transcribe.cpp model of the catalog (stt-gguf.json) and not a model file path"),
        )
    })?;
    let file = entry
        .files
        .iter()
        .find(|f| is_model_file(Path::new(&f.path)))
        .ok_or_else(|| TranscribeError::new(ErrorKind::InvalidInput, format!("{model} has no GGUF model file")))?;
    let dir = crate::services::model_store::installed_path(&entry.id)
        .ok_or_else(|| TranscribeError::new(ErrorKind::NotInstalled, format!("{}: the model is not installed", entry.id)))?;
    let path = dir.join(&file.path);
    if !path.is_file() {
        return Err(TranscribeError::new(
            ErrorKind::NotInstalled,
            format!("{}: the installed model is missing {}", entry.id, file.path),
        ));
    }
    let file_bytes = std::fs::metadata(&path)
        .map(|m| m.len())
        .unwrap_or(file.size);
    Ok(ResolvedModel {
        label: entry.id.clone(),
        catalog_id: Some(entry.id.clone()),
        title: entry.title.clone().unwrap_or_else(|| entry.id.clone()),
        path,
        file_bytes,
        gpu_mb: gpu_footprint_mb(entry.min_vram_mb, file_bytes),
        family: Family::from_id(&entry.id),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn families_come_from_arch_and_id() {
        assert_eq!(Family::from_arch("qwen3_asr"), Family::Qwen3Asr);
        assert_eq!(Family::from_arch("cohere_asr"), Family::Cohere);
        assert_eq!(Family::from_arch("granite5_ctc"), Family::Granite);
        assert_eq!(Family::from_arch("voxtral_realtime"), Family::VoxtralRealtime);
        assert_eq!(Family::from_arch("voxtral"), Family::Other, "offline Voxtral is another arch");
        assert_eq!(Family::from_arch(" Whisper "), Family::Whisper);
        assert_eq!(Family::from_arch("r2t2_asr"), Family::R2t2);
        assert_eq!(Family::from_arch("qwen3_asr_r2t2"), Family::R2t2, "a fine-tune of Qwen3-ASR");
        assert_eq!(Family::from_id("stt/confucius4-r2t2-q8_0"), Family::R2t2);
        assert!(Family::R2t2.append_only() && !Family::VoxtralRealtime.append_only());
        assert_eq!(Family::from_id("stt/qwen3-asr-1.7b-q8_0"), Family::Qwen3Asr);
        assert_eq!(Family::from_id("stt/qwen3-asr-0.6b-int8"), Family::Other, "the sherpa model is not ours");
        assert_eq!(Family::from_id("stt/cohere-transcribe-q4_k"), Family::Cohere);
        assert_eq!(Family::from_id("stt/voxtral-mini-4b-realtime-q4_k_m"), Family::VoxtralRealtime);
        // Cohere: never ja / zh / ko / ru / yue, and always a language.
        for l in ["ja", "zh", "ko", "ru", "yue"] {
            assert!(Family::Cohere.never_languages().contains(&l));
        }
        assert!(Family::Cohere.needs_language());
        assert!(!Family::Qwen3Asr.needs_language());
        assert!(Family::Qwen3Asr.never_languages().is_empty());
        assert!(Family::Whisper.takes_prompt() && !Family::Qwen3Asr.takes_prompt());
    }

    #[test]
    fn gpu_footprints_exclude_the_runtime_context() {
        // Catalog totals include the context; the model claim is the rest.
        assert_eq!(gpu_footprint_mb(Some(3500), 2_185_030_624), 3100);
        // No catalog number: 1.5 x the file.
        assert_eq!(gpu_footprint_mb(None, 2_185_030_624), estimated_model_vram_mb(2_185_030_624));
        assert_eq!(gpu_footprint_mb(Some(100), 0), 1, "a bogus total never makes a zero claim");
    }

    #[test]
    fn resolution_errors_name_the_problem() {
        assert_eq!(resolve("").unwrap_err().kind, ErrorKind::InvalidInput);
        let e = resolve("stt/does-not-exist").unwrap_err();
        assert_eq!(e.kind, ErrorKind::InvalidInput);
        assert!(e.message.contains("stt-gguf.json"), "{e}");
        let missing = std::env::temp_dir().join("sigil-no-such-model.gguf");
        assert_eq!(resolve(&missing.to_string_lossy()).unwrap_err().kind, ErrorKind::InvalidInput);
        // A real file with the wrong extension.
        let dir = std::env::temp_dir().join(format!("sigil-transcribe-models-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let txt = dir.join("model.txt");
        std::fs::write(&txt, b"x").unwrap();
        assert_eq!(resolve(&txt.to_string_lossy()).unwrap_err().kind, ErrorKind::InvalidInput);
        let gguf = dir.join("Model.GGUF");
        std::fs::write(&gguf, vec![0u8; 3 * 1024 * 1024]).unwrap();
        let r = resolve(&gguf.to_string_lossy()).unwrap();
        assert_eq!((r.label.as_str(), r.catalog_id.as_deref(), r.family), ("Model.GGUF", None, Family::Other));
        assert_eq!(r.file_bytes, 3 * 1024 * 1024);
        assert_eq!(r.gpu_mb, 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every GGUF entry of stt-gguf.json resolves to NotInstalled (naming its id) or to its file,
    /// and has a family the engine knows.
    #[test]
    fn catalog_models_resolve_or_are_not_installed() {
        let catalog = catalog().expect("catalog");
        let ours: Vec<_> = catalog
            .entries
            .iter()
            .filter(|e| ID_FAMILIES.iter().any(|(p, _)| e.id.starts_with(p)))
            .collect();
        assert!(ours.len() >= 5, "stt-gguf.json lists the GPU-tier models: {:?}", ours.iter().map(|e| &e.id).collect::<Vec<_>>());
        for e in ours {
            assert_ne!(Family::from_id(&e.id), Family::Other, "{}", e.id);
            match resolve(&e.id) {
                Ok(r) => assert!(r.path.is_file()),
                Err(err) => {
                    assert_eq!(err.kind, ErrorKind::NotInstalled, "{err}");
                    assert!(err.message.starts_with(&format!("{}: ", e.id)), "{err}");
                }
            }
        }
    }
}
