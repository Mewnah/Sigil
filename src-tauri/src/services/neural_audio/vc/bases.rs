//! RVC base-model paths under `app_data/voice-changer/base/` and slot manifest helpers.

use std::fs;
use std::path::{Path, PathBuf};

use super::manifest::{load_manifest, RvcManifest, RvcVersion};

/// Known filenames we auto-resolve when inspector paths are empty.
const HUBERT_CANDIDATES: &[&str] = &[
    "hubert_base.onnx",
    "contentvec.onnx",
    "vec-768-layer-12.onnx",
    "hubert.onnx",
];
const FCPE_CANDIDATES: &[&str] = &["fcpe.onnx", "FCPE.onnx"];
const RMVPE_CANDIDATES: &[&str] = &["rmvpe.onnx", "RMVPE.onnx", "rmvpe.pt.onnx"];

pub fn voice_changer_root(app_data: &Path) -> PathBuf {
    app_data.join("voice-changer")
}

pub fn base_models_dir(app_data: &Path) -> PathBuf {
    voice_changer_root(app_data).join("base")
}

pub fn models_dir(app_data: &Path) -> PathBuf {
    voice_changer_root(app_data).join("models")
}

fn first_existing(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    for name in names {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Prefer an explicit non-empty path; otherwise search `base/` (and optionally `slot_dir`).
pub fn resolve_model_file(
    explicit: &str,
    base_dir: &Path,
    slot_dir: Option<&Path>,
    candidates: &[&str],
) -> Option<PathBuf> {
    let t = explicit.trim();
    if !t.is_empty() {
        let p = PathBuf::from(t);
        if p.is_file() {
            return Some(p);
        }
        return None;
    }
    if let Some(found) = first_existing(base_dir, candidates) {
        return Some(found);
    }
    if let Some(slot) = slot_dir {
        if let Some(found) = first_existing(slot, candidates) {
            return Some(found);
        }
    }
    None
}

pub fn resolve_hubert(explicit: &str, app_data: &Path, slot_dir: &Path) -> Option<PathBuf> {
    resolve_model_file(
        explicit,
        &base_models_dir(app_data),
        Some(slot_dir),
        HUBERT_CANDIDATES,
    )
}

pub fn resolve_fcpe(explicit: &str, app_data: &Path, slot_dir: &Path) -> Option<PathBuf> {
    resolve_model_file(
        explicit,
        &base_models_dir(app_data),
        Some(slot_dir),
        FCPE_CANDIDATES,
    )
}

pub fn resolve_rmvpe(explicit: &str, app_data: &Path, slot_dir: &Path) -> Option<PathBuf> {
    resolve_model_file(
        explicit,
        &base_models_dir(app_data),
        Some(slot_dir),
        RMVPE_CANDIDATES,
    )
}

/// Ensure `app_data/voice-changer/{base,models}` exist.
pub fn ensure_voice_changer_dirs(app_data: &Path) -> Result<(), String> {
    fs::create_dir_all(base_models_dir(app_data)).map_err(|e| e.to_string())?;
    fs::create_dir_all(models_dir(app_data)).map_err(|e| e.to_string())?;
    Ok(())
}

fn find_generator_onnx(slot_dir: &Path) -> Option<String> {
    const NAMES: &[&str] = &["gen.onnx", "model.onnx", "rvc.onnx", "G.onnx"];
    for name in NAMES {
        if slot_dir.join(name).is_file() {
            return Some((*name).to_string());
        }
    }
    let skip = ["hubert", "contentvec", "fcpe", "rmvpe", "vec-"];
    if let Ok(rd) = fs::read_dir(slot_dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) != Some("onnx") {
                continue;
            }
            let Some(fname) = p.file_name() else {
                continue;
            };
            let name = fname.to_string_lossy().to_lowercase();
            if skip.iter().any(|s| name.contains(s)) {
                continue;
            }
            return Some(fname.to_string_lossy().into_owned());
        }
    }
    None
}

/// Pick the slot's retrieval index: well-known names first, then any `*.index`, preferring
/// RVC's `added_*` files over `trained_*` ones (a trained-only index holds no vectors).
fn find_index_file(slot_dir: &Path) -> Option<String> {
    for name in ["added_IVF256_Flat_nprobe_1.index", "added.index", "index.index"] {
        if slot_dir.join(name).is_file() {
            return Some(name.to_string());
        }
    }
    let mut found: Vec<String> = fs::read_dir(slot_dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|p| p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("index"))
        .filter_map(|p| p.file_name().map(|f| f.to_string_lossy().into_owned()))
        .collect();
    found.sort();
    let rank = |name: &str| {
        let lower = name.to_lowercase();
        if lower.starts_with("added") {
            0
        } else if lower.starts_with("trained") {
            2
        } else {
            1
        }
    };
    found.into_iter().min_by_key(|name| rank(name))
}

/// Load `manifest.json`, or infer a default v2 manifest in memory when only the generator
/// `.onnx` (and optionally an `.index`) is present.
///
/// Returns `(manifest, inferred)`. Nothing is ever written to the slot folder: an inferred
/// manifest is refined from the generator ONNX when the model loads (see `RvcModel::load`), and a
/// read-only or user-managed folder never blocks Start. An existing `manifest.json` always wins.
pub fn load_or_infer_manifest(slot_dir: &Path) -> Result<(RvcManifest, bool), String> {
    if slot_dir.join("manifest.json").is_file() {
        return load_manifest(slot_dir).map(|m| (m, false));
    }
    let generator = find_generator_onnx(slot_dir).ok_or_else(|| {
        format!(
            "No manifest.json and no generator .onnx found in {}",
            slot_dir.display()
        )
    })?;
    let index = find_index_file(slot_dir);
    let m = RvcManifest {
        format: "rvc".into(),
        version: RvcVersion::V2,
        sample_rate: DEFAULT_INFERRED_SAMPLE_RATE,
        if_f0: true,
        emb_channels: 768,
        emb_layer: 12,
        use_final_proj: false,
        generator,
        index,
        speaker_id: 0,
        hubert: None,
    };
    Ok((m, true))
}

/// Sample rate assumed for an inferred manifest until the generator ONNX says otherwise.
pub const DEFAULT_INFERRED_SAMPLE_RATE: u32 = 40_000;

/// Preflight RVC paths for Start (sync, before spawning the audio thread).
pub fn preflight_rvc(
    slot_dir: &str,
    hubert: &str,
    fcpe: &str,
    rmvpe: &str,
    pitch_extractor: &str,
    app_data: &Path,
) -> Result<(PathBuf, PathBuf, PathBuf, Option<PathBuf>), String> {
    ensure_voice_changer_dirs(app_data)?;
    let slot = PathBuf::from(slot_dir.trim());
    if slot_dir.trim().is_empty() {
        return Err("Choose an RVC model slot folder.".into());
    }
    if !slot.is_dir() {
        return Err(format!("RVC model slot folder not found: {}", slot.display()));
    }
    // Validates manifest.json (or that a generator exists to infer one from); never writes.
    let _manifest = load_or_infer_manifest(&slot)?;
    let hubert_path = resolve_hubert(hubert, app_data, &slot).ok_or_else(|| {
        format!(
            "HuBERT / ContentVec ONNX not found. Set the path or place hubert_base.onnx / contentvec.onnx under {}",
            base_models_dir(app_data).display()
        )
    })?;
    let pitch = pitch_extractor.trim().to_lowercase();
    let fcpe_path = if matches!(pitch.as_str(), "dio" | "rmvpe" | "crepe") {
        PathBuf::new()
    } else {
        resolve_fcpe(fcpe, app_data, &slot).ok_or_else(|| {
            format!(
                "FCPE ONNX not found. Set the path, switch F0 to dio, or place fcpe.onnx under {}",
                base_models_dir(app_data).display()
            )
        })?
    };
    let rmvpe_path = if pitch == "rmvpe" || pitch == "crepe" {
        Some(resolve_rmvpe(rmvpe, app_data, &slot).ok_or_else(|| {
            format!(
                "RMVPE ONNX not found. Set the path or place rmvpe.onnx under {}",
                base_models_dir(app_data).display()
            )
        })?)
    } else {
        resolve_rmvpe(rmvpe, app_data, &slot)
    };
    Ok((slot, hubert_path, fcpe_path, rmvpe_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempSlot(PathBuf);

    impl TempSlot {
        fn new(tag: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let dir = std::env::temp_dir().join(format!(
                "sigil-vc-{tag}-{}-{nanos}",
                std::process::id()
            ));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempSlot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn manifest_is_inferred_in_memory_without_writing() {
        let slot = TempSlot::new("infer");
        fs::write(slot.0.join("my_voice.onnx"), b"not really onnx").unwrap();
        fs::write(slot.0.join("trained_IVF256_Flat_nprobe_1_my_voice_v2.index"), b"x").unwrap();
        fs::write(slot.0.join("added_IVF256_Flat_nprobe_1_my_voice_v2.index"), b"x").unwrap();

        let (m, inferred) = load_or_infer_manifest(&slot.0).unwrap();
        assert!(inferred);
        assert_eq!(m.generator, "my_voice.onnx");
        assert_eq!(
            m.index.as_deref(),
            Some("added_IVF256_Flat_nprobe_1_my_voice_v2.index")
        );
        assert_eq!(m.sample_rate, DEFAULT_INFERRED_SAMPLE_RATE);
        assert!(
            !slot.0.join("manifest.json").exists(),
            "inference must never write manifest.json"
        );
        // Preflight's manifest step is the same read-only call.
        let (_, again) = load_or_infer_manifest(&slot.0).unwrap();
        assert!(again);
        assert!(!slot.0.join("manifest.json").exists());
    }

    #[test]
    fn existing_manifest_wins() {
        let slot = TempSlot::new("explicit");
        fs::write(slot.0.join("gen.onnx"), b"x").unwrap();
        fs::write(
            slot.0.join("manifest.json"),
            br#"{"generator":"gen.onnx","sample_rate":48000,"version":"v1"}"#,
        )
        .unwrap();
        let (m, inferred) = load_or_infer_manifest(&slot.0).unwrap();
        assert!(!inferred);
        assert_eq!(m.sample_rate, 48_000);
        assert_eq!(m.version, RvcVersion::V1);
        assert_eq!(m.emb_channels, 256);
    }

    #[test]
    fn missing_generator_is_an_error() {
        let slot = TempSlot::new("empty");
        fs::write(slot.0.join("hubert_base.onnx"), b"x").unwrap();
        assert!(load_or_infer_manifest(&slot.0).is_err());
        assert!(!slot.0.join("manifest.json").exists());
    }
}
