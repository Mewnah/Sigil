//! RVC base-model paths under `app_data/voice-changer/base/` and slot manifest helpers.
//!
//! Two kinds of voice slot are preflighted here ([`preflight_rvc`]):
//! - bring-your-own ONNX generators (format `rvc`, with or without `manifest.json`): the content
//!   encoder comes from the explicit path or the legacy `base/` candidates;
//! - imported template voices (format `sigil-rvc-1`, written by `import::slot`): their weights
//!   must be present and the content encoder their manifest names must be installed, found by
//!   the engine's own rule ([`resolve_embedder`] runs `engine::content::resolve_encoder_file`: the
//!   explicit path only for ContentVec, then the model store entry `vc/embedder-*`, then, for
//!   ContentVec 768 / 256, the legacy `voice-changer/base/` names), so a voice that passes the
//!   preflight is one the engine can load.

use std::fs;
use std::path::{Path, PathBuf};

use super::manifest::{load_manifest, load_slot_manifest, RvcManifest, RvcVersion, SlotFormat};

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

/// The content-encoder file of an imported voice in `slot_dir`, found exactly as the engine finds it
/// (`engine::content::resolve_encoder_file`, for either device class): the explicit path only for
/// a ContentVec voice (it must exist; the other encoders always come from their store entry),
/// then the model-store entry of the manifest's embedder (any installed precision variant), then,
/// for ContentVec 768 / 256, the legacy names in `voice-changer/base/` (the slot's
/// `../../base`) and the slot. `embedder` is the manifest's name; `None` means ContentVec of the
/// voice's width. The error is the engine's.
pub fn resolve_embedder(explicit: &str, embedder: Option<&str>, emb_channels: u32, slot_dir: &Path) -> Result<PathBuf, String> {
    use super::engine::content::{resolve_encoder_file, EmbedderKind};
    use super::engine::options::PrecisionMode;
    let kind = EmbedderKind::from_manifest(embedder, emb_channels)?;
    let t = explicit.trim();
    let explicit = (!t.is_empty()).then(|| Path::new(t));
    let mut first_err = None;
    // A GPU run prefers the fp16 variant, a CPU run fp32 / int8: either installed variant counts.
    for gpu in [true, false] {
        match resolve_encoder_file(kind, explicit, Some(slot_dir), gpu, PrecisionMode::Auto) {
            Ok(r) => return Ok(r.path),
            Err(e) => {
                first_err.get_or_insert(e);
            }
        }
    }
    Err(first_err.unwrap_or_else(|| format!("the {} content encoder is not installed", kind.as_str())))
}

/// Ensure `app_data/voice-changer/{base,models}` exist.
pub fn ensure_voice_changer_dirs(app_data: &Path) -> Result<(), String> {
    fs::create_dir_all(base_models_dir(app_data)).map_err(|e| e.to_string())?;
    fs::create_dir_all(models_dir(app_data)).map_err(|e| e.to_string())?;
    Ok(())
}

pub(crate) fn find_generator_onnx(slot_dir: &Path) -> Option<String> {
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
pub(crate) fn find_index_file(slot_dir: &Path) -> Option<String> {
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
        ..RvcManifest::default()
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
    let template_voice = if slot.join("manifest.json").is_file() {
        load_slot_manifest(&slot).ok().filter(|m| m.slot_format() == SlotFormat::SigilRvc)
    } else {
        None
    };
    let hubert_path = match &template_voice {
        Some(m) => preflight_template_voice(&slot, m, hubert)?,
        None => {
            // Validates manifest.json (or that a generator exists to infer one from); never writes.
            let _manifest = load_or_infer_manifest(&slot)?;
            resolve_hubert(hubert, app_data, &slot).ok_or_else(|| {
                format!(
                    "HuBERT / ContentVec ONNX not found. Set the path or place hubert_base.onnx / contentvec.onnx under {}",
                    base_models_dir(app_data).display()
                )
            })?
        }
    };
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

/// Preflight of an imported template voice (format `sigil-rvc-1`): the engine runs in this build,
/// its weights are present and its content encoder resolves the way the engine resolves it
/// ([`resolve_embedder`]). Returns the HuBERT / ContentVec path from the settings unchanged (the
/// engine uses it only for a ContentVec voice and says so otherwise), or an empty path when it is
/// unset and the engine's shared encoder pool resolves the slot's embedder itself (so it can pick
/// the precision variant for the EP).
fn preflight_template_voice(slot: &Path, m: &RvcManifest, hubert: &str) -> Result<PathBuf, String> {
    if !super::engine::available() {
        return Err(format!(
            "{} holds an imported voice; the Sigil Voice Engine that runs imported voices is not available in this build yet",
            slot.display()
        ));
    }
    for f in [m.weights.as_deref().unwrap_or("weights.f32.bin"), m.speakers.as_deref().unwrap_or("speakers.f32.bin")] {
        if f.contains(['/', '\\']) || f.contains("..") || !slot.join(f).is_file() {
            return Err(format!("{f} is missing in {}; import the voice again", slot.display()));
        }
    }
    resolve_embedder(hubert, m.embedder.as_deref(), m.emb_channels, slot).map_err(|e| {
        let mut msg = e;
        if let Some(first) = msg.get_mut(..1) {
            first.make_ascii_uppercase();
        }
        msg
    })?;
    Ok(PathBuf::from(hubert.trim()))
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

    /// The installed model-store file of an embedder, if any (tests adapt to the machine's store).
    fn store(name: &str) -> Option<PathBuf> {
        use crate::services::neural_audio::vc::assets;
        let id = assets::embedder_id(name)?;
        [Some("fp32"), None, Some("fp16"), Some("int8")].into_iter().find_map(|v| assets::model_file(id, v))
    }

    /// `<app_data>/voice-changer/models/<name>` with a template-voice manifest, its weights and
    /// speakers (`extra` is spliced into the manifest JSON).
    fn template_slot(app: &Path, name: &str, extra: &str) -> PathBuf {
        let slot = models_dir(app).join(name);
        fs::create_dir_all(&slot).unwrap();
        let manifest = format!(
            r#"{{"format":"sigil-rvc-1","template":"v2-40k-f0-hifigan","weights":"weights.f32.bin","speakers":"speakers.f32.bin"{extra}}}"#
        );
        fs::write(slot.join("manifest.json"), manifest).unwrap();
        fs::write(slot.join("weights.f32.bin"), b"w").unwrap();
        fs::write(slot.join("speakers.f32.bin"), b"s").unwrap();
        slot
    }

    #[test]
    fn template_voices_are_preflighted_by_their_manifest() {
        let slot = TempSlot::new("tpl");
        let app = TempSlot::new("tpl-app");
        fs::write(
            slot.0.join("manifest.json"),
            br#"{"format":"sigil-rvc-1","template":"v2-40k-f0-hifigan","weights":"weights.f32.bin","speakers":"speakers.f32.bin","embedder":"spin-v2"}"#,
        )
        .unwrap();
        let err = preflight_rvc(&slot.0.to_string_lossy(), "", "", "", "dio", &app.0).unwrap_err();
        if super::super::engine::available() {
            assert!(err.contains("weights.f32.bin"), "{err}");
            // A ContentVec voice: the settings path is its encoder and must exist.
            let cv = template_slot(&app.0, "cv", r#","embedder":"contentvec""#);
            let err = preflight_rvc(&cv.to_string_lossy(), "C:/no/such/encoder.onnx", "", "", "dio", &app.0).unwrap_err();
            assert!(err.contains("encoder") && err.contains("C:/no/such"), "{err}");
            let enc = cv.join("custom.onnx");
            fs::write(&enc, b"x").unwrap();
            let (_, hubert, _, _) = preflight_rvc(&cv.to_string_lossy(), &enc.to_string_lossy(), "", "", "dio", &app.0).unwrap();
            assert_eq!(hubert, enc);
        } else {
            assert!(err.contains("Sigil Voice Engine"), "{err}");
        }
        // Never refused as "not an ONNX generator" by the legacy loader, never written to.
        assert!(!err.contains("ONNX generator"));
        assert!(!slot.0.join("gen.onnx").exists());
    }

    /// The preflight agrees with the engine: a HuBERT path set for bring-your-own voices does not
    /// stand in for another embedder, and a v1 voice finds the legacy 256-channel names.
    #[test]
    fn template_preflight_resolves_encoders_like_the_engine() {
        if !super::super::engine::available() {
            return;
        }
        let app = TempSlot::new("tpl-enc");
        let global = app.0.join("hubert_base.onnx");
        fs::write(&global, b"x").unwrap();
        for embedder in ["hubert-ja", "spin-v2"] {
            let slot = template_slot(&app.0, embedder, &format!(r#","embedder":"{embedder}""#));
            let got = preflight_rvc(&slot.to_string_lossy(), &global.to_string_lossy(), "", "", "dio", &app.0);
            match store(embedder) {
                None => {
                    let err = got.unwrap_err();
                    assert!(err.contains(embedder) && err.contains("not installed"), "{embedder}: {err}");
                }
                // Installed: it passes, and the settings path is handed on unchanged (the engine says
                // it is not used for this voice).
                Some(_) => assert_eq!(got.unwrap().1, global, "{embedder}"),
            }
        }
        // v1 (256 channels): `voice-changer/base/vec-256-layer-9.onnx` is the engine's legacy name.
        let v1 = template_slot(&app.0, "v1", r#","version":"v1""#);
        if store("contentvec-256").is_none() {
            let err = preflight_rvc(&v1.to_string_lossy(), "", "", "", "dio", &app.0).unwrap_err();
            assert!(err.contains("contentvec-256"), "{err}");
        }
        fs::create_dir_all(base_models_dir(&app.0)).unwrap();
        fs::write(base_models_dir(&app.0).join("vec-256-layer-9.onnx"), b"x").unwrap();
        let (_, hubert, _, _) = preflight_rvc(&v1.to_string_lossy(), "", "", "", "dio", &app.0).unwrap();
        assert_eq!(hubert, PathBuf::new(), "the engine resolves it");
        assert!(resolve_embedder("", None, 256, &v1).is_ok());
        // A 768 voice does not take the 256 file.
        if store("contentvec").is_none() {
            assert!(resolve_embedder("", None, 768, &v1).is_err());
        }
    }

    #[test]
    fn embedders_resolve_explicit_then_store_then_legacy() {
        let app = TempSlot::new("emb-app");
        let slot = models_dir(&app.0).join("voice");
        fs::create_dir_all(&slot).unwrap();
        let custom = slot.join("custom.onnx");
        fs::write(&custom, b"x").unwrap();
        // The explicit path is a ContentVec voice's encoder (and must exist) ...
        assert_eq!(resolve_embedder(&custom.to_string_lossy(), None, 768, &slot), Ok(custom.clone()));
        assert!(resolve_embedder("C:/no/such/encoder.onnx", None, 768, &slot).is_err());
        // ... and never another embedder's.
        if store("spin-v2").is_none() {
            assert!(resolve_embedder(&custom.to_string_lossy(), Some("spin-v2"), 768, &slot).is_err());
        }
        // ContentVec 768 falls back to the legacy base folder when the store has none.
        let base = base_models_dir(&app.0);
        fs::create_dir_all(&base).unwrap();
        fs::write(base.join("hubert_base.onnx"), b"x").unwrap();
        let cv = resolve_embedder("", None, 768, &slot).unwrap();
        assert!(store("contentvec").is_some() || cv == base.join("hubert_base.onnx"), "{}", cv.display());
        // The other embedders only come from the model store.
        for (name, width) in [("hubert-ja", 768), ("spin-v2", 768)] {
            if store(name).is_none() {
                assert!(resolve_embedder("", Some(name), width, &slot).is_err(), "{name}");
            }
        }
        // Unknown embedders and widths that do not match are the engine's errors.
        assert!(resolve_embedder("", Some("wavlm-large"), 768, &slot).unwrap_err().contains("wavlm-large"));
        assert!(resolve_embedder("", Some("spin-v2"), 256, &slot).is_err());
    }

    #[test]
    fn missing_generator_is_an_error() {
        let slot = TempSlot::new("empty");
        fs::write(slot.0.join("hubert_base.onnx"), b"x").unwrap();
        assert!(load_or_infer_manifest(&slot.0).is_err());
        assert!(!slot.0.join("manifest.json").exists());
    }
}
