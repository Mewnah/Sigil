//! Tests of the `local-tts` plugin.
//!
//! Unit tests run offline: the catalog table against `model-catalog/tts-local.json`, voice and
//! language rules, Kokoro / Piper configuration, text chunking, and the engine's queue, ordering,
//! cancellation, caching and sink pushes with a fake loader and sink.
//!
//! The `real_*` tests are ignored: they need the sherpa-onnx runtime and the catalog models
//! installed in the model store of `SIGIL_APP_DATA_DIR` (`Sigil.exe --headless fetch
//! tts/kokoro-v1.0 tts/kokoro-v1.1-zh tts/piper-ru_ru-irina-medium tts/kitten-mini-en-v0.8`), and
//! for the intelligibility checks the SenseVoice int8 package under
//! `SIGIL_TEST_MODELS_DIR/sherpa-onnx/models/`. A missing model skips its test with a message. Run:
//!
//! ```text
//! SIGIL_APP_DATA_DIR=%LOCALAPPDATA%/sigil-dev-cache/appdata-p2h SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%/sigil-dev-cache
//! cargo test --manifest-path src-tauri/Cargo.toml local_tts -- --ignored --test-threads=1 --nocapture
//! ```

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::catalog::{self, Family, VoiceSet};
use super::commands::{LocalTtsModelInfo, LocalTtsRequest, LocalTtsVoice};
use super::engine::{
    sanitize_speed, warm_up_text, Engine, LoadKey, Loader, OpenPlan, PcmWriter, Resolved, SinkPort, Synth, VoiceChoice,
    CANCELLED, WORKER_STOPPED,
};
use super::kokoro::{self, KokoroSetup};
use super::kokoro_ort;
use super::models::{self, ModelRef};
use super::{piper, text, voices};
use crate::services::speech_rt::sherpa::TtsModel;
use crate::services::tts_sink::{SinkError, SpeakReport};

fn tmp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sigil-local-tts-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// -------------------------------------------------------------------------------------------------
// Catalog
// -------------------------------------------------------------------------------------------------

const FRAGMENT: &str = include_str!("../../../model-catalog/tts-local.json");

#[test]
fn catalog_table_matches_the_fragment() {
    let doc: serde_json::Value = serde_json::from_str(FRAGMENT).expect("tts-local.json parses");
    let entries = doc["entries"].as_array().expect("entries");
    let fragment_ids: Vec<&str> = entries.iter().map(|e| e["id"].as_str().unwrap()).collect();
    let table_ids: Vec<&str> = catalog::all().iter().map(|m| m.id.as_str()).collect();
    assert_eq!(fragment_ids, table_ids, "same ids in the same order");
    let unique: HashSet<&str> = table_ids.iter().copied().collect();
    assert_eq!(unique.len(), table_ids.len(), "ids are unique");
    for (e, m) in entries.iter().zip(catalog::all()) {
        assert_eq!(e["title"].as_str(), Some(m.title.as_str()), "{}", m.id);
        assert_eq!(e["family"], "tts", "{}", m.id);
        assert_eq!(e["kind"], "model", "{}", m.id);
        assert_eq!(e["status"], "published", "{}", m.id);
        assert_eq!(e["requires"], serde_json::json!(["runtime/sherpa-onnx"]), "{}", m.id);
        let files = e["files"].as_array().unwrap();
        assert_eq!(files.len(), 1, "{}", m.id);
        let f = &files[0];
        assert_eq!(f["sha256"].as_str().map(str::len), Some(64), "{} is pinned", m.id);
        assert!(f["size"].as_u64().unwrap() > 10_000_000, "{}", m.id);
        assert_eq!(f["archive"]["format"], "tar.bz2");
        assert_eq!(f["archive"]["strip"], 1);
        let include: Vec<&str> = f["archive"]["include"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(include.contains(&"espeak-ng-data/**"), "{}", m.id);
        assert!(include.contains(&"tokens.txt"), "{}", m.id);
        let asset = f["path"].as_str().unwrap();
        let url = f["sources"][0]["url"].as_str().unwrap();
        assert!(url.ends_with(asset), "{url} downloads {asset}");
        assert!(url.starts_with("https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"));
        match m.family {
            Family::Kokoro => {
                assert!(asset.starts_with("kokoro-multi-lang-v1_"), "{asset}");
                assert!(!asset.contains("int8"), "Kokoro ships fp32: {asset}");
                assert!(include.contains(&"lexicon-*.txt") && include.contains(&"*.fst"));
            }
            Family::Kitten => assert!(asset.starts_with("kitten-") && asset.contains("-en-v0_8")),
            Family::Piper => {
                let stem = m.piper_voice.expect("piper stem");
                assert_eq!(asset, format!("vits-piper-{stem}.tar.bz2"));
                assert!(include.contains(&"*.onnx.json"));
                assert_eq!(m.id, format!("tts/piper-{}", stem.to_lowercase()));
            }
        }
    }
    assert!(catalog::get(catalog::DEFAULT_MODEL).is_some());
    assert_eq!(catalog::get(" tts/kokoro-v1.0 ").map(|m| m.family), Some(Family::Kokoro));
    assert!(catalog::get("tts/nope").is_none());
}

#[test]
fn catalog_models_cover_the_planned_languages() {
    let langs: HashSet<String> = catalog::all()
        .iter()
        .flat_map(|m| m.languages.iter().map(|l| voices::base(l)))
        .collect();
    // Kokoro's eight, plus Piper for ru, ar and the long tail (research §2.1).
    for l in ["en", "ja", "zh", "es", "fr", "hi", "it", "pt", "ru", "ar", "de", "uk", "pl", "tr", "vi", "id", "ur"] {
        assert!(langs.contains(l), "no local voice for {l}");
    }
    let ru = catalog::get("tts/piper-ru_ru-irina-medium").unwrap();
    assert_eq!(ru.languages, ["ru"]);
    assert_eq!(ru.sample_rate, 22_050);
    assert_eq!(catalog::get("tts/piper-pt_pt-tugao-medium").unwrap().languages, ["pt-PT"]);
    assert_eq!(catalog::get("tts/piper-el_gr-rapunzelina-low").unwrap().sample_rate, 16_000);
    let k = catalog::get("tts/kokoro-v1.0").unwrap();
    assert_eq!(k.languages, ["en", "ja", "zh", "es", "fr", "hi", "it", "pt-BR"]);
    assert_eq!(k.static_voices().len(), 54);
    assert_eq!(catalog::get("tts/kokoro-v1.1-zh").unwrap().static_voices().len(), 103);
    assert_eq!(catalog::get("tts/kitten-mini-en-v0.8").unwrap().static_voices().len(), 8);
    let piper_voices = ru.static_voices();
    assert_eq!(piper_voices.len(), 1);
    assert_eq!((piper_voices[0].id.as_str(), piper_voices[0].name.as_str()), ("0", "Irina"));
}

#[test]
fn catalog_info_reads_installed_piper_speakers() {
    let dir = tmp_dir("piper-info");
    std::fs::write(dir.join("uk_UA-ukrainian_tts-medium.onnx"), b"x").unwrap();
    std::fs::write(
        dir.join("uk_UA-ukrainian_tts-medium.onnx.json"),
        r#"{"audio":{"sample_rate":22050},"num_speakers":3,"speaker_id_map":{"lada":0,"mykyta":1,"tetiana":2}}"#,
    )
    .unwrap();
    let m = catalog::get("tts/piper-uk_ua-ukrainian_tts-medium").unwrap();
    let info = models::catalog_info(m, Some(&dir));
    assert!(info.installed);
    let names: Vec<&str> = info.voices.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(names, ["Lada", "Mykyta", "Tetiana"]);
    assert_eq!(info.voices[2].id, "2");
    assert_eq!(info.family, "piper");
    let not_installed = models::catalog_info(m, None);
    assert!(!not_installed.installed);
    assert_eq!(not_installed.voices.len(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

// -------------------------------------------------------------------------------------------------
// Voices and languages
// -------------------------------------------------------------------------------------------------

#[test]
fn kokoro_voice_tables_and_names() {
    assert_eq!(voices::KOKORO_V1_0.len(), 54);
    assert_eq!(voices::KOKORO_V1_0[3], "af_heart");
    assert_eq!(voices::KOKORO_V1_0[53], "em_santa", "added last (speaker id 53)");
    assert_eq!(voices::KOKORO_V1_1.len(), 103);
    assert_eq!(voices::KOKORO_V1_1[3], "zf_001");
    assert_eq!(voices::kokoro_voice_language("bf_emma"), Some("en-GB"));
    assert_eq!(voices::kokoro_voice_language("pm_alex"), Some("pt-BR"));
    assert_eq!(voices::kokoro_voice_language("zm_yunxi"), Some("zh"));
    assert_eq!(voices::kokoro_voice_language("xx"), None);
    assert_eq!(voices::kokoro_voice_gender("jm_kumo"), Some("male"));
    assert_eq!(voices::kokoro_display_name("af_heart"), "Heart");
    assert_eq!(voices::kokoro_display_name("zf_001"), "Female 001");
    assert_eq!(voices::kokoro_display_name("zm_100"), "Male 100");
    // Every voice of both tables has a language.
    for v in voices::KOKORO_V1_0.iter().chain(voices::KOKORO_V1_1) {
        assert!(voices::kokoro_voice_language(v).is_some(), "{v}");
        assert!(voices::kokoro_voice_gender(v).is_some(), "{v}");
    }
    assert_eq!(voices::speaker_id(voices::KOKORO_V1_0, "AF_HEART"), Some(3));
    assert_eq!(voices::speaker_id(voices::KOKORO_V1_0, "53"), Some(53));
    assert_eq!(voices::speaker_id(voices::KOKORO_V1_0, "54"), None);
    assert_eq!(voices::speaker_id(voices::KOKORO_V1_0, "nobody"), None);
}

#[test]
fn kokoro_default_voices_follow_the_language() {
    let t = voices::KOKORO_V1_0;
    assert_eq!(voices::kokoro_default_voice(t, None), "af_heart");
    assert_eq!(voices::kokoro_default_voice(t, Some("en")), "af_heart");
    assert_eq!(voices::kokoro_default_voice(t, Some("en-GB")), "bf_emma");
    assert_eq!(voices::kokoro_default_voice(t, Some("ja")), "jf_alpha");
    assert_eq!(voices::kokoro_default_voice(t, Some("pt")), "pf_dora");
    assert_eq!(voices::kokoro_default_voice(t, Some("zh")), "zf_xiaoxiao");
    assert_eq!(voices::kokoro_default_voice(t, Some("ru")), "af_heart", "unsupported → the default");
    // v1.1-zh has neither af_heart nor zf_xiaoxiao.
    assert_eq!(voices::kokoro_default_voice(voices::KOKORO_V1_1, Some("zh")), "zf_001");
    assert_eq!(voices::kokoro_default_voice(voices::KOKORO_V1_1, None), "af_maple", "the raw table's first voice");
    // A model's default follows its own language when the request names none, or names one the
    // model has no voice for: Kokoro v1.1-zh speaks Mandarin, not with its first (English) voice.
    let v11 = catalog::get("tts/kokoro-v1.1-zh").unwrap();
    assert_eq!(v11.default_voice(None), "zf_001");
    assert_eq!(v11.default_voice(Some("zh")), "zf_001");
    assert_eq!(v11.default_voice(Some("ja")), "zf_001", "no Japanese voice: the model's own language");
    assert_eq!(v11.default_voice(Some("en")), "af_maple");
    let setup = KokoroSetup::for_voice(&v11.default_voice(None), None);
    assert_eq!((setup.espeak, setup.mandarin_rules), ("cmn", true), "Mandarin frontend and number rules");
    assert_eq!(models::catalog_info(v11, None).default_voice, "zf_001");
    let v10 = catalog::get("tts/kokoro-v1.0").unwrap();
    assert_eq!(v10.default_voice(None), "af_heart");
    assert_eq!(v10.default_voice(Some("ja")), "jf_alpha");
    assert_eq!(v10.default_voice(Some("ru")), "af_heart");
    assert_eq!(voices::kokoro_primary_language(voices::KOKORO_V1_0).as_deref(), Some("en"));
    assert_eq!(voices::kokoro_primary_language(voices::KOKORO_V1_1).as_deref(), Some("zh"));
    assert_eq!(voices::kokoro_primary_language(&["jf_alpha", "af_heart"]).as_deref(), Some("ja"), "first seen wins a tie");
    assert_eq!(voices::kokoro_primary_language(&["xx"]), None);
    assert!(voices::kokoro_has_language(voices::KOKORO_V1_1, "en"));
    assert!(voices::kokoro_has_language(voices::KOKORO_V1_0, "pt"));
    assert!(voices::kokoro_has_language(voices::KOKORO_V1_0, "EN-gb"));
    assert!(!voices::kokoro_has_language(voices::KOKORO_V1_1, "ja"));
    assert_eq!(voices::kokoro_default_voice(&[], None), "af_heart", "an empty table does not panic");
}

#[test]
fn kitten_voice_names_fall_back_to_the_default_of_other_models() {
    assert!(voices::is_kitten_voice_name("jasper"));
    assert!(voices::is_kitten_voice_name(" Kiki "));
    assert!(voices::is_kitten_voice_name("expr-voice-2-f"));
    assert!(!voices::is_kitten_voice_name("7"), "numbers are speaker ids");
    assert!(!voices::is_kitten_voice_name("af_heart"));
    let default = || "af_heart".to_string();
    assert_eq!(models::pick_voice("Jasper", "folder", |_| false, default), "af_heart");
    assert_eq!(models::pick_voice("Jasper", "folder", |_| true, default), "Jasper", "a model that has it keeps it");
    assert_eq!(models::pick_voice("  ", "folder", |_| false, default), "af_heart");
    assert_eq!(models::pick_voice("Nobody", "folder", |_| false, default), "Nobody", "reported as unknown");
}

/// An ONNX model (the fake Kitten graph) with `metadata_props` appended: enough for the folder
/// resolution, which reads the family from the metadata (a protobuf message may repeat fields in
/// any order, so appending is valid).
fn onnx_with_metadata(props: &[(&str, &str)]) -> Vec<u8> {
    fn varint(mut n: u64, out: &mut Vec<u8>) {
        loop {
            let b = (n & 0x7f) as u8;
            n >>= 7;
            if n == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }
    fn bytes_field(field: u64, data: &[u8], out: &mut Vec<u8>) {
        varint((field << 3) | 2, out);
        varint(data.len() as u64, out);
        out.extend_from_slice(data);
    }
    let mut out = crate::services::local_tts::test_model::model_bytes();
    for (k, v) in props {
        let mut entry = Vec::new();
        bytes_field(1, k.as_bytes(), &mut entry);
        bytes_field(2, v.as_bytes(), &mut entry);
        bytes_field(14, &entry, &mut out);
    }
    out
}

#[test]
fn official_folders_resolve_their_own_default_voice_and_ignore_kitten_names() {
    let loader = models::StoreLoader::default();
    // An official Kokoro v1.1-zh folder (speaker table abridged).
    let kokoro = tmp_dir("kokoro-v11-folder");
    std::fs::write(
        kokoro.join("model.onnx"),
        onnx_with_metadata(&[
            ("model_type", "kokoro"),
            ("speaker_names", "af_maple,af_sol,bf_vale,zf_001,zf_002,zf_003,zm_009,zm_010"),
            ("sample_rate", "24000"),
        ]),
    )
    .unwrap();
    let folder = kokoro.display().to_string();
    let r = loader.resolve(&folder, "", None, Some(2)).expect("resolves");
    assert_eq!((r.voice.name.as_str(), r.voice.sid), ("zf_001", 3), "a Mandarin voice by default");
    assert_eq!(r.info.default_voice, "zf_001");
    assert!(r.key.variant.contains("lang=cmn"), "{}", r.key.variant);
    // The legacy neuralOnnx default "Jasper" is not an error: the folder's default voice speaks.
    let r = loader.resolve(&folder, "Jasper", None, Some(2)).expect("a Kitten name falls back");
    assert_eq!(r.voice.name, "zf_001");
    assert_eq!(loader.resolve(&folder, "af_sol", None, None).unwrap().voice.sid, 1);
    assert_eq!(loader.resolve(&folder, "", Some("en"), None).unwrap().voice.name, "af_maple");
    let err = loader.resolve(&folder, "Nobody", None, None).unwrap_err();
    assert!(err.contains("Nobody"), "{err}");

    // An official Piper folder.
    let piper_dir = tmp_dir("piper-folder");
    std::fs::write(
        piper_dir.join("ru_RU-irina-medium.onnx"),
        onnx_with_metadata(&[("model_type", "vits"), ("sample_rate", "22050")]),
    )
    .unwrap();
    std::fs::write(
        piper_dir.join("ru_RU-irina-medium.onnx.json"),
        r#"{"audio":{"sample_rate":22050},"language":{"code":"ru_RU"},"num_speakers":1,"speaker_id_map":{}}"#,
    )
    .unwrap();
    let folder = piper_dir.display().to_string();
    let r = loader.resolve(&folder, "Jasper", None, None).expect("a Kitten name falls back");
    assert_eq!(r.voice.sid, 0);
    assert_eq!(r.info.family, "piper");
    assert!(loader.resolve(&folder, "Nobody", None, None).is_err());
    let _ = std::fs::remove_dir_all(&kokoro);
    let _ = std::fs::remove_dir_all(&piper_dir);
}

#[test]
fn kitten_voices_use_kittenml_names() {
    let v = voices::kitten_voices();
    assert_eq!(v.len(), 8);
    assert_eq!((v[0].id.as_str(), v[0].gender.as_deref()), ("Jasper", Some("male")));
    assert_eq!((v[1].id.as_str(), v[1].gender.as_deref()), ("Bella", Some("female")));
    assert_eq!(voices::kitten_speaker_id("jasper"), Some(0));
    assert_eq!(voices::kitten_speaker_id("Kiki"), Some(7));
    assert_eq!(voices::kitten_speaker_id("expr-voice-3-f"), Some(3), "Luna");
    assert_eq!(voices::kitten_speaker_id("7"), Some(7));
    assert_eq!(voices::kitten_speaker_id("8"), None);
    assert_eq!(voices::kitten_speaker_id("Garfield"), None);
}

#[test]
fn languages_are_normalized() {
    let n = |s: &str| voices::normalize_language(s);
    assert_eq!(n("pt_br").as_deref(), Some("pt-BR"));
    assert_eq!(n(" EN ").as_deref(), Some("en"));
    assert_eq!(n("en-gb").as_deref(), Some("en-GB"));
    assert_eq!(n("zh-Hans").as_deref(), Some("zh"));
    assert_eq!(n("cmn").as_deref(), Some("zh"));
    assert_eq!(n("jpn").as_deref(), Some("ja"));
    assert_eq!(n("auto"), None);
    assert_eq!(n(""), None);
    assert_eq!(n("english!"), None);
    assert_eq!(voices::base("pt-BR"), "pt");
    assert_eq!(piper::language_tag("ru_RU").as_deref(), Some("ru"));
    assert_eq!(piper::language_tag("pt_PT").as_deref(), Some("pt-PT"));
    assert_eq!(piper::language_tag("en_GB").as_deref(), Some("en-GB"));
}

// -------------------------------------------------------------------------------------------------
// Kokoro configuration
// -------------------------------------------------------------------------------------------------

#[test]
fn kokoro_espeak_voices_are_voice_file_names() {
    // Measured: "en-gb" / "fr-fr" are language codes sherpa cannot set as espeak voices.
    assert_eq!(kokoro::espeak_voice("en-GB"), Some("en"));
    assert_eq!(kokoro::espeak_voice("en-US"), Some("en-us"));
    assert_eq!(kokoro::espeak_voice("fr"), Some("fr"));
    assert_eq!(kokoro::espeak_voice("pt-BR"), Some("pt-br"));
    assert_eq!(kokoro::espeak_voice("zh"), Some("cmn"));
    assert_eq!(kokoro::espeak_voice("ja"), Some("ja"));
    assert_eq!(kokoro::espeak_voice("de"), None, "not a Kokoro language");
    for lang in kokoro::LANGUAGES {
        assert!(kokoro::espeak_voice(lang).is_some(), "{lang}");
    }
}

#[test]
fn kokoro_setup_follows_the_voice_and_the_language_hint() {
    let s = KokoroSetup::for_voice("bf_emma", None);
    assert_eq!((s.espeak, s.british, s.mandarin_rules), ("en", true, false));
    let s = KokoroSetup::for_voice("af_heart", None);
    assert_eq!((s.espeak, s.british, s.mandarin_rules), ("en-us", false, false));
    let s = KokoroSetup::for_voice("zf_xiaoxiao", None);
    assert_eq!((s.espeak, s.mandarin_rules), ("cmn", true));
    assert_eq!(KokoroSetup::for_voice("jf_alpha", None).espeak, "ja");
    // Cross-lingual hint: an English voice reading Spanish.
    assert_eq!(KokoroSetup::for_voice("af_heart", Some("es")).espeak, "es");
    // A bare "en" keeps a British voice British.
    assert_eq!(KokoroSetup::for_voice("bf_emma", Some("en")).espeak, "en");
    // An unsupported hint is ignored.
    assert_eq!(KokoroSetup::for_voice("jf_alpha", Some("de")).espeak, "ja");
}

#[test]
fn kokoro_config_picks_lexicons_and_chinese_rules_per_voice() {
    let dir = tmp_dir("kokoro-config");
    for f in [
        "model.onnx",
        "voices.bin",
        "tokens.txt",
        "lexicon-us-en.txt",
        "lexicon-gb-en.txt",
        "lexicon-zh.txt",
        "phone-zh.fst",
        "date-zh.fst",
        "number-zh.fst",
    ] {
        std::fs::write(dir.join(f), b"x").unwrap();
    }
    std::fs::create_dir_all(dir.join("espeak-ng-data")).unwrap();
    let us = kokoro::config(&dir, &KokoroSetup::for_voice("af_heart", None), 4).unwrap();
    let TtsModel::Kokoro { lexicons, lang, data_dir, .. } = &us.model else {
        panic!("a Kokoro config")
    };
    let names: Vec<String> = lexicons
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["lexicon-us-en.txt", "lexicon-zh.txt"], "English first");
    assert_eq!(lang.as_deref(), Some("en-us"));
    assert!(data_dir.is_some());
    assert!(us.rule_fsts.is_empty(), "no Chinese number rules for English (they turn digits into hanzi)");
    assert_eq!(us.num_threads, Some(4));
    assert_eq!(us.max_num_sentences, Some(1));
    us.validate().unwrap();

    let gb = kokoro::config(&dir, &KokoroSetup::for_voice("bm_george", None), 2).unwrap();
    let TtsModel::Kokoro { lexicons, lang, .. } = &gb.model else { unreachable!() };
    assert!(lexicons[0].ends_with("lexicon-gb-en.txt"));
    assert_eq!(lang.as_deref(), Some("en"));

    let zh = kokoro::config(&dir, &KokoroSetup::for_voice("zm_yunxi", None), 2).unwrap();
    let fsts: Vec<String> = zh
        .rule_fsts
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(fsts, ["phone-zh.fst", "date-zh.fst", "number-zh.fst"]);

    // int8-only packages still load; a package without a model is an error.
    std::fs::remove_file(dir.join("model.onnx")).unwrap();
    assert!(kokoro::config(&dir, &KokoroSetup::for_voice("af_heart", None), 2).is_err());
    std::fs::write(dir.join("model.int8.onnx"), b"x").unwrap();
    let int8 = kokoro::config(&dir, &KokoroSetup::for_voice("af_heart", None), 2).unwrap();
    let TtsModel::Kokoro { model, .. } = &int8.model else { unreachable!() };
    assert!(model.ends_with("model.int8.onnx"));
    let _ = std::fs::remove_dir_all(&dir);
}

// -------------------------------------------------------------------------------------------------
// Piper
// -------------------------------------------------------------------------------------------------

#[test]
fn piper_voice_json_is_parsed() {
    let single = piper::parse_voice_json(
        r#"{"audio":{"sample_rate":22050,"quality":"medium"},"espeak":{"voice":"ru"},"language":{"code":"ru_RU"},
            "inference":{"noise_scale":0.667,"length_scale":1.1,"noise_w":0.8},"num_speakers":1,"speaker_id_map":{},
            "phoneme_id_map":{"_":[0]}}"#,
    )
    .unwrap();
    assert_eq!(single.sample_rate, Some(22_050));
    assert_eq!(single.espeak_voice.as_deref(), Some("ru"));
    assert_eq!(single.language_code.as_deref(), Some("ru_RU"));
    assert_eq!(single.noise_scale, Some(0.667));
    assert_eq!(single.length_scale, Some(1.1));
    assert!(single.speakers.is_empty());
    assert!((piper::effective_speed(&single, 1.0) - 1.0 / 1.1).abs() < 1e-6, "length scale folded into speed");
    assert_eq!(piper::speaker_id(&single, ""), Some(0));
    assert_eq!(piper::speaker_id(&single, "0"), Some(0));
    assert_eq!(piper::speaker_id(&single, "1"), None);
    let v = piper::voices(&single, "Irina", Some("ru"));
    assert_eq!((v.len(), v[0].name.as_str(), v[0].language.as_deref()), (1, "Irina", Some("ru")));

    let multi = piper::parse_voice_json(r#"{"num_speakers":3,"speaker_id_map":{"b_voice":1,"a_voice":0,"c_voice":2}}"#).unwrap();
    assert_eq!(multi.speakers, ["a_voice", "b_voice", "c_voice"]);
    assert_eq!(piper::speaker_id(&multi, "B voice"), Some(1));
    assert_eq!(piper::speaker_id(&multi, "c_voice"), Some(2));
    assert_eq!(piper::speaker_id(&multi, "2"), Some(2));
    assert_eq!(piper::speaker_id(&multi, "3"), None);
    // A sparse map does not describe usable ids.
    let sparse = piper::parse_voice_json(r#"{"speaker_id_map":{"a":0,"b":5}}"#).unwrap();
    assert!(sparse.speakers.is_empty());
    // Bad values are dropped, not trusted.
    let bad = piper::parse_voice_json(r#"{"inference":{"noise_scale":-1,"length_scale":0}}"#).unwrap();
    assert_eq!((bad.noise_scale, bad.length_scale), (None, None));
    assert!(piper::parse_voice_json("not json").is_err());
}

#[test]
fn piper_package_files_are_found() {
    let dir = tmp_dir("piper-files");
    assert!(piper::model_file(&dir).is_err(), "empty folder");
    std::fs::write(dir.join("a.onnx"), b"x").unwrap();
    assert!(piper::model_file(&dir).unwrap().ends_with("a.onnx"), "a lone model");
    std::fs::write(dir.join("b.onnx"), b"x").unwrap();
    assert!(piper::model_file(&dir).is_err(), "ambiguous without a .onnx.json");
    std::fs::write(dir.join("b.onnx.json"), b"{}").unwrap();
    assert!(piper::model_file(&dir).unwrap().ends_with("b.onnx"), "the one with a voice config");
    std::fs::write(dir.join("tokens.txt"), b"x").unwrap();
    std::fs::create_dir_all(dir.join("espeak-ng-data")).unwrap();
    let info = piper::voice_info(&dir.join("b.onnx")).unwrap();
    let c = piper::config(&dir, &dir.join("b.onnx"), &info, 3);
    let TtsModel::Vits { data_dir, lexicon, .. } = &c.model else { panic!("VITS") };
    assert!(data_dir.is_some() && lexicon.is_none());
    assert_eq!(c.num_threads, Some(3));
    c.validate().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

// -------------------------------------------------------------------------------------------------
// Text chunking
// -------------------------------------------------------------------------------------------------

#[test]
fn sentences_split_on_terminators_and_keep_closers() {
    assert_eq!(text::sentences("Hello there. How are you?  Fine!"), ["Hello there.", "How are you?", "Fine!"]);
    assert_eq!(text::sentences("He said \"stop.\" Then left."), ["He said \"stop.\"", "Then left."]);
    assert_eq!(text::sentences("こんにちは。元気ですか？はい！"), ["こんにちは。", "元気ですか？", "はい！"]);
    assert_eq!(text::sentences("「本当？」と聞いた。"), ["「本当？」", "と聞いた。"]);
    assert_eq!(text::sentences("Pi is 3.14 and v1.0 ships."), ["Pi is 3.14 and v1.0 ships."]);
    assert_eq!(text::sentences("Mr. Smith met Dr. Jones. Then lunch."), ["Mr. Smith met Dr. Jones.", "Then lunch."]);
    assert_eq!(text::sentences("line one\nline two"), ["line one", "line two"]);
    assert_eq!(text::sentences("Wait... what?!"), ["Wait...", "what?!"]);
    // Fragments without letters join a neighbour; nothing speakable is dropped.
    assert_eq!(text::sentences("Hello. ... !!"), ["Hello....!!"]);
    assert_eq!(text::sentences("... Hello."), ["... Hello."]);
    assert!(text::sentences("  ...  ").is_empty());
    assert!(text::sentences("").is_empty());
    assert_eq!(text::sentences("tab\there\u{7}bell"), ["tab here bell"], "control characters are spaces");
}

#[test]
fn long_sentences_split_at_clauses_then_spaces_then_anywhere() {
    let clause = format!("{}, {}", "a".repeat(60), "b".repeat(60));
    assert_eq!(text::split_long(&clause, 100), [format!("{},", "a".repeat(60)), "b".repeat(60)]);
    let words = "word ".repeat(40);
    for piece in text::split_long(words.trim(), 50) {
        assert!(piece.chars().count() <= 50, "{piece}");
        assert!(!piece.starts_with(' ') && !piece.ends_with(' '));
    }
    let cjk = "漢".repeat(250);
    let pieces = text::split_long(&cjk, 100);
    assert_eq!(pieces.len(), 3);
    assert!(pieces.iter().all(|p| p.chars().count() <= 100));
    assert_eq!(pieces.concat(), cjk, "no character lost");
    assert_eq!(text::split_long("short", 100), ["short"]);
}

#[test]
fn first_clause_split_only_when_both_parts_are_long_enough() {
    let s = "When the evening finally arrived, the whole village gathered in the square.";
    assert_eq!(
        text::split_first_clause(s),
        Some(("When the evening finally arrived,".into(), "the whole village gathered in the square.".into()))
    );
    assert_eq!(text::split_first_clause("Yes, sure."), None, "too short");
    assert_eq!(text::split_first_clause("Well, the whole village gathered in the square."), None, "first clause too short... ");
    assert_eq!(
        text::split_first_clause("The whole village gathered in the square, yes."),
        None,
        "rest too short"
    );
    assert_eq!(
        text::split_first_clause("今日はとても良い天気ですね、公園に散歩に行きましょう。"),
        Some(("今日はとても良い天気ですね、".into(), "公園に散歩に行きましょう。".into()))
    );
    // Only the first chunk of `chunks` is split.
    let two = format!("{s} {s}");
    let c = text::chunks(&two, true);
    assert_eq!(c.len(), 3);
    assert_eq!(c[0], "When the evening finally arrived,");
    assert_eq!(c[2], s);
    assert_eq!(text::chunks(&two, false).len(), 2);
}

// -------------------------------------------------------------------------------------------------
// Model references
// -------------------------------------------------------------------------------------------------

#[test]
fn model_refs_are_parsed() {
    assert!(matches!(models::parse_model_ref("").unwrap(), ModelRef::Catalog(m) if m.id == catalog::DEFAULT_MODEL));
    assert!(matches!(models::parse_model_ref(" tts/piper-ru_ru-irina-medium ").unwrap(), ModelRef::Catalog(m) if m.family == Family::Piper));
    assert!(models::parse_model_ref("tts/unknown").unwrap_err().contains("unknown local voice model"));
    assert_eq!(
        models::parse_model_ref("hub:kitten-tts-mini-0.8").unwrap(),
        ModelRef::Hub("KittenML/kitten-tts-mini-0.8".into())
    );
    assert!(models::parse_model_ref("hub:../escape").is_err());
    assert!(models::parse_model_ref("relative/folder").is_err());
    let abs = std::env::temp_dir();
    assert_eq!(models::parse_model_ref(&abs.display().to_string()).unwrap(), ModelRef::Folder(abs));
}

#[test]
fn kittenml_hub_repos_map_onto_catalog_packages() {
    use models::{hub_catalog_model, hub_choice, HubChoice, HUB_CATALOG};
    // Every mapped id is a Kitten catalog model, and the repo ids are normalized.
    for (repo, id) in HUB_CATALOG {
        let m = catalog::get(id).unwrap_or_else(|| panic!("{id} is not in the catalog"));
        assert_eq!(m.family, Family::Kitten, "{repo}");
        assert_eq!(
            crate::services::local_tts::kitten::hub::normalize_repo_id(repo).unwrap(),
            *repo,
            "{repo} is normalized"
        );
    }
    // The legacy neuralOnnx default, any case, with or without the org.
    for repo in ["KittenML/kitten-tts-mini-0.8", "kittenml/KITTEN-TTS-MINI-0.8", "kitten-tts-mini-0.8"] {
        assert_eq!(hub_catalog_model(repo).map(|m| m.id.as_str()), Some("tts/kitten-mini-en-v0.8"), "{repo}");
    }
    assert_eq!(
        hub_catalog_model("KittenML/kitten-tts-nano-0.8-int8").map(|m| m.id.as_str()),
        Some("tts/kitten-nano-en-v0.8")
    );
    assert!(hub_catalog_model("KittenML/kitten-tts-nano-0.2").is_none(), "older repos keep the Hub download");
    assert!(hub_catalog_model("someone/else").is_none());
    assert!(hub_catalog_model("../x").is_none());

    // Installed catalog package first (sherpa-onnx), then an earlier download, else download first.
    assert_eq!(hub_choice(true, true, true), HubChoice::Catalog);
    assert_eq!(hub_choice(true, true, false), HubChoice::Catalog);
    assert_eq!(hub_choice(true, false, true), HubChoice::Vendored);
    assert_eq!(hub_choice(true, false, false), HubChoice::NotInstalled);
    // No catalog package: the legacy Hub path (a local copy, or a download through the store).
    assert_eq!(hub_choice(false, false, true), HubChoice::Vendored);
    assert_eq!(hub_choice(false, false, false), HubChoice::Vendored);
}

#[test]
fn threads_are_clamped() {
    assert_eq!(models::effective_threads(Some(1)), 1);
    assert_eq!(models::effective_threads(Some(3)), 3);
    assert_eq!(models::effective_threads(Some(64)), 4);
    let auto = models::effective_threads(None);
    assert!((2..=4).contains(&auto), "{auto}");
    assert_eq!(models::effective_threads(Some(0)), auto);
}

#[test]
fn speech_package_folders_resolve_to_the_vendored_kitten() {
    let dir = crate::services::local_tts::test_model::write_package("local-tts-folder");
    let loader = models::StoreLoader::default();
    let r = loader
        .resolve(&dir.display().to_string(), "Luna", None, Some(2))
        .expect("resolves");
    assert_eq!(r.plan, OpenPlan::KittenPackage(dir.clone()));
    assert_eq!(r.info.family, "kitten");
    assert_eq!(r.voice.name, "Luna");
    // It loads and speaks through the engine (fake sink).
    let sink = Arc::new(FakeSink::default());
    let engine = Engine::new(Arc::new(models::StoreLoader::default()), sink.clone(), Duration::from_secs(60));
    let info = engine
        .load(dir.display().to_string(), "Jazz".into(), None, Some(2))
        .blocking_recv()
        .unwrap()
        .expect("loads");
    let names: Vec<&str> = info.voices.iter().map(|v| v.id.as_str()).collect();
    assert_eq!(names, ["Jazz", "Luna"], "aliases first, then voices no alias names");
    assert_eq!(info.default_voice, "Jasper", "KittenML's default voice when the package has it");
    sink.begin("u1");
    let report = engine
        .speak(
            LocalTtsRequest {
                model: dir.display().to_string(),
                voice: "Luna".into(),
                text: "Hello from Sigil.".into(),
                ..Default::default()
            },
            "u1".into(),
        )
        .blocking_recv()
        .unwrap()
        .expect("speaks");
    assert_eq!(report.sample_rate, crate::services::local_tts::kitten::SAMPLE_RATE);
    assert_eq!(sink.rate("u1"), Some(crate::services::local_tts::kitten::SAMPLE_RATE));
    let pushed = sink.samples("u1");
    assert_eq!(pushed.len() as u64, report.frames);
    assert!(!pushed.is_empty());
    let unknown = engine
        .speak(
            LocalTtsRequest {
                model: dir.display().to_string(),
                voice: "Nobody".into(),
                text: "Hi.".into(),
                ..Default::default()
            },
            "u1".into(),
        )
        .blocking_recv()
        .unwrap();
    assert!(unknown.is_err(), "an unknown voice is an error");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn missing_folders_and_non_voice_folders_are_errors() {
    let loader = models::StoreLoader::default();
    let missing = std::env::temp_dir().join("sigil-local-tts-no-such-folder");
    assert!(loader
        .resolve(&missing.display().to_string(), "", None, None)
        .unwrap_err()
        .contains("does not exist"));
    let empty = tmp_dir("not-a-voice");
    assert!(loader.resolve(&empty.display().to_string(), "", None, None).is_err());
    let _ = std::fs::remove_dir_all(&empty);
}

// -------------------------------------------------------------------------------------------------
// Vendored Kitten helpers
// -------------------------------------------------------------------------------------------------

#[test]
fn voices_bin_is_split_per_speaker() {
    use crate::services::local_tts::kitten::hub::split_voices_bin;
    let values: Vec<f32> = (0..2 * 3 * 4).map(|i| i as f32).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let names = vec!["a".to_string(), "b".to_string()];
    let v = split_voices_bin(&bytes, &names, 4).unwrap();
    assert_eq!(v.len(), 2);
    assert_eq!((v[1].0.as_str(), v[1].1, v[1].2), ("b", 3, 4));
    assert_eq!(v[1].3[0], 12.0);
    assert!(split_voices_bin(&bytes, &names, 5).is_err(), "rows must be whole");
    assert!(split_voices_bin(&bytes[..7], &names, 4).is_err(), "whole f32 values");
    assert!(split_voices_bin(&bytes, &[], 4).is_err());
    let three = vec!["a".into(), "b".into(), "c".into()];
    assert!(split_voices_bin(&bytes, &three, 4).is_ok(), "3 voices x 2 rows");
    let five: Vec<String> = (0..5).map(|i| i.to_string()).collect();
    assert!(split_voices_bin(&bytes, &five, 4).is_err(), "not divisible");
}

#[test]
fn vendored_kitten_runs_voice_tables_from_memory() {
    use crate::services::local_tts::kitten::model::KittenTtsOnnx;
    use crate::services::local_tts::test_model::{write_package, LEVEL, STYLE_DIM};
    let dir = write_package("voice-table");
    let rows = 4;
    let voices = vec![
        ("expr-voice-2-m".to_string(), rows, STYLE_DIM, vec![0.1; rows * STYLE_DIM]),
        ("expr-voice-2-f".to_string(), rows, STYLE_DIM, vec![0.2; rows * STYLE_DIM]),
    ];
    let aliases: HashMap<String, String> = [("Jasper", "expr-voice-2-m"), ("Bella", "expr-voice-2-f"), ("Luna", "expr-voice-3-f")]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    let tts = KittenTtsOnnx::load_with_voices(&dir.join("model.onnx"), voices, HashMap::new(), aliases).unwrap();
    assert!(tts.resolves_voice("Jasper") && tts.resolves_voice("expr-voice-2-f"));
    assert!(!tts.resolves_voice("Luna"), "alias of a voice that is not loaded");
    assert_eq!(tts.voice_names(), ["Bella", "Jasper"]);
    let audio = tts.generate("Hello there.", "Bella", 1.0, true).unwrap();
    assert!(!audio.is_empty() && audio.iter().all(|s| (s - LEVEL).abs() < 1e-6));
    let bad = KittenTtsOnnx::load_with_voices(
        &dir.join("model.onnx"),
        vec![("x".into(), 2, 8, vec![0.0; 3])],
        HashMap::new(),
        HashMap::new(),
    );
    assert!(bad.is_err(), "a matrix of the wrong size is refused");
    let _ = std::fs::remove_dir_all(&dir);
}

// -------------------------------------------------------------------------------------------------
// Engine (fake loader and sink)
// -------------------------------------------------------------------------------------------------

/// Shared record of what the fake engine did.
#[derive(Default)]
struct Record {
    opens: AtomicUsize,
    drops: AtomicUsize,
    /// Texts generated, in order (including warm-ups).
    texts: Mutex<Vec<String>>,
    /// Voices generated with.
    voices: Mutex<Vec<String>>,
    /// Name of the thread each synthesis ran on.
    threads: Mutex<Vec<String>>,
}

struct FakeLoader {
    rec: Arc<Record>,
    /// Pause per generated batch for texts containing "slow".
    slow: Duration,
}

impl FakeLoader {
    fn new(rec: Arc<Record>) -> Self {
        Self {
            rec,
            slow: Duration::from_millis(150),
        }
    }
}

impl Loader for FakeLoader {
    fn resolve(&self, model: &str, voice: &str, language: Option<&str>, threads: Option<u32>) -> Result<Resolved, String> {
        if model.starts_with("missing") {
            return Err(format!("{model} is not downloaded yet"));
        }
        Ok(Resolved {
            key: LoadKey::new(model.to_string(), language.unwrap_or_default().to_string(), threads.unwrap_or(2)),
            info: LocalTtsModelInfo {
                id: model.to_string(),
                family: "kokoro".into(),
                title: model.to_string(),
                sample_rate: 16_000,
                installed: true,
                ..Default::default()
            },
            voice: VoiceChoice {
                sid: 0,
                name: voice.to_string(),
            },
            speed_factor: if model == "half-speed" { 0.5 } else { 1.0 },
            language: None,
            plan: OpenPlan::Test(model.to_string()),
        })
    }

    fn open(&self, resolved: &Resolved) -> Result<Box<dyn Synth>, String> {
        self.rec.opens.fetch_add(1, Ordering::SeqCst);
        if resolved.key.model.starts_with("fail-open") {
            return Err("the model file is damaged".into());
        }
        if resolved.key.model.starts_with("panic-open") {
            panic!("fake load panic");
        }
        Ok(Box::new(FakeSynth {
            rec: self.rec.clone(),
            slow: self.slow,
            voices: (resolved.key.model == "with-voices").then(|| {
                vec![LocalTtsVoice {
                    id: "v1".into(),
                    name: "Voice 1".into(),
                    language: None,
                    gender: None,
                }]
            }),
        }))
    }
}

/// Generates 10 samples per character, in two batches, value = speed. Panics on texts containing
/// "panic" (a model-code bug such as the vendored Kitten's old number overflow).
struct FakeSynth {
    rec: Arc<Record>,
    slow: Duration,
    voices: Option<Vec<LocalTtsVoice>>,
}

impl Drop for FakeSynth {
    fn drop(&mut self) {
        self.rec.drops.fetch_add(1, Ordering::SeqCst);
    }
}

impl Synth for FakeSynth {
    fn sample_rate(&self) -> u32 {
        16_000
    }

    fn voices(&self) -> Option<Vec<LocalTtsVoice>> {
        self.voices.clone()
    }

    fn generate(
        &mut self,
        text: &str,
        voice: &VoiceChoice,
        speed: f32,
        cancel: &AtomicBool,
        on_chunk: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<bool, String> {
        self.rec.texts.lock().unwrap().push(text.to_string());
        self.rec.voices.lock().unwrap().push(voice.name.clone());
        self.rec
            .threads
            .lock()
            .unwrap()
            .push(std::thread::current().name().unwrap_or("").to_string());
        if text.contains("explode") {
            return Err("synthesis failed".into());
        }
        if text.contains("panic") {
            panic!("fake synthesis panic");
        }
        let n = text.chars().count() * 10;
        for half in [n / 2, n - n / 2] {
            if cancel.load(Ordering::SeqCst) {
                return Ok(false);
            }
            if text.contains("slow") {
                std::thread::sleep(self.slow);
            }
            if !on_chunk(&vec![speed; half]) {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[derive(Default)]
struct FakeSinkState {
    begun: HashSet<String>,
    cancelled: HashSet<String>,
    pushed: HashMap<String, Vec<f32>>,
    /// Sample rate of each utterance's pushes (every push of one utterance must agree).
    rates: HashMap<String, u32>,
    order: Vec<String>,
}

#[derive(Default)]
struct FakeSink {
    state: Arc<Mutex<FakeSinkState>>,
    queued_ms: AtomicU32,
    /// `order()` panics (a failure outside the engine's guarded model code).
    panic_order: AtomicBool,
}

impl FakeSink {
    fn begin(&self, id: &str) {
        let mut s = self.state.lock().unwrap();
        s.begun.insert(id.to_string());
        s.order.push(id.to_string());
    }

    fn cancel(&self, id: &str) {
        self.state.lock().unwrap().cancelled.insert(id.to_string());
    }

    fn rate(&self, id: &str) -> Option<u32> {
        self.state.lock().unwrap().rates.get(id).copied()
    }

    fn samples(&self, id: &str) -> Vec<f32> {
        self.state
            .lock()
            .unwrap()
            .pushed
            .get(id)
            .cloned()
            .unwrap_or_default()
    }
}

struct FakeWriter {
    id: String,
    state: Arc<Mutex<FakeSinkState>>,
}

impl PcmWriter for FakeWriter {
    fn push(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), SinkError> {
        let mut s = self.state.lock().unwrap();
        if s.cancelled.contains(&self.id) {
            return Err(SinkError::Cancelled);
        }
        let rate = *s.rates.entry(self.id.clone()).or_insert(sample_rate);
        assert_eq!(rate, sample_rate, "one rate per utterance");
        s.pushed
            .entry(self.id.clone())
            .or_default()
            .extend_from_slice(samples);
        Ok(())
    }

    fn is_cancelled(&self) -> bool {
        self.state.lock().unwrap().cancelled.contains(&self.id)
    }
}

impl SinkPort for FakeSink {
    fn writer(&self, utterance_id: &str) -> Result<Box<dyn PcmWriter>, SinkError> {
        if !self.state.lock().unwrap().begun.contains(utterance_id) {
            return Err(SinkError::NotFound(utterance_id.to_string()));
        }
        Ok(Box::new(FakeWriter {
            id: utterance_id.to_string(),
            state: self.state.clone(),
        }))
    }

    fn order(&self) -> Vec<String> {
        if self.panic_order.load(Ordering::SeqCst) {
            panic!("fake sink order panic");
        }
        self.state.lock().unwrap().order.clone()
    }

    fn queued_ms(&self) -> u32 {
        self.queued_ms.load(Ordering::SeqCst)
    }
}

fn fake_engine(idle: Duration) -> (Engine, Arc<Record>, Arc<FakeSink>) {
    let rec = Arc::new(Record::default());
    let sink = Arc::new(FakeSink::default());
    // Plenty queued: no first-clause split unless a test asks for it.
    sink.queued_ms.store(60_000, Ordering::SeqCst);
    let engine = Engine::new(Arc::new(FakeLoader::new(rec.clone())), sink.clone(), idle);
    (engine, rec, sink)
}

fn req(model: &str, text: &str) -> LocalTtsRequest {
    LocalTtsRequest {
        model: model.into(),
        voice: "v".into(),
        text: text.into(),
        ..Default::default()
    }
}

fn texts(rec: &Record) -> Vec<String> {
    rec.texts.lock().unwrap().clone()
}

fn wait_until(what: &str, cond: impl Fn() -> bool) {
    let t0 = Instant::now();
    while !cond() {
        assert!(t0.elapsed() < Duration::from_secs(5), "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn engine_speaks_into_the_sink_and_reports() {
    let (engine, rec, sink) = fake_engine(Duration::from_secs(60));
    sink.begin("u1");
    let report = engine
        .speak(req("m", "Hello there."), "u1".into())
        .blocking_recv()
        .unwrap()
        .expect("speaks");
    assert_eq!(report.sample_rate, 16_000);
    assert_eq!(sink.rate("u1"), Some(16_000), "pushed at the model's rate");
    assert_eq!(report.frames, 120);
    assert!(report.first_audio_ms <= report.total_ms);
    assert_eq!(sink.samples("u1").len(), 120);
    // The model loaded once and was warmed up with a short phrase first.
    assert_eq!(rec.opens.load(Ordering::SeqCst), 1);
    assert_eq!(texts(&rec), [warm_up_text(None), "Hello there."]);
    // Speed reaches the synthesizer (sanitized), times the model's factor.
    sink.begin("u2");
    let mut fast = req("m", "Hi.");
    fast.speed = 9.0;
    engine.speak(fast, "u2".into()).blocking_recv().unwrap().unwrap();
    assert!(sink.samples("u2").iter().all(|s| *s == 2.0), "clamped to 2");
    sink.begin("u3");
    engine
        .speak(req("half-speed", "Hi."), "u3".into())
        .blocking_recv()
        .unwrap()
        .unwrap();
    assert!(sink.samples("u3").iter().all(|s| *s == 0.5));
    assert_eq!(sanitize_speed(f32::NAN), 1.0);
    assert_eq!(sanitize_speed(0.1), 0.5);
}

#[test]
fn engine_splits_the_first_clause_only_when_nothing_is_queued_ahead() {
    let (engine, rec, sink) = fake_engine(Duration::from_secs(60));
    let s = "When the evening finally arrived, the whole village gathered in the square.";
    sink.begin("a");
    engine.speak(req("m", s), "a".into()).blocking_recv().unwrap().unwrap();
    assert_eq!(texts(&rec).last().map(String::as_str), Some(s), "audio queued ahead: whole sentence");
    sink.queued_ms.store(0, Ordering::SeqCst);
    sink.begin("b");
    engine.speak(req("m", s), "b".into()).blocking_recv().unwrap().unwrap();
    let t = texts(&rec);
    assert_eq!(&t[t.len() - 2..], ["When the evening finally arrived,", "the whole village gathered in the square."]);
}

#[test]
fn engine_reuses_the_loaded_model_and_switches_when_needed() {
    let (engine, rec, sink) = fake_engine(Duration::from_secs(60));
    for (i, model) in ["a", "a", "b", "b", "a"].iter().enumerate() {
        let id = format!("u{i}");
        sink.begin(&id);
        engine.speak(req(model, "Hi."), id).blocking_recv().unwrap().unwrap();
    }
    assert_eq!(rec.opens.load(Ordering::SeqCst), 3, "a, b, a");
    assert_eq!(rec.drops.load(Ordering::SeqCst), 2, "the previous model is freed on a switch");
    // A different language (Kokoro variant) or thread count is another load.
    let mut ja = req("a", "Hi.");
    ja.language = Some("ja".into());
    sink.begin("ja");
    engine.speak(ja, "ja".into()).blocking_recv().unwrap().unwrap();
    assert_eq!(rec.opens.load(Ordering::SeqCst), 4);
    // Unload frees it; the next request loads again.
    engine.unload().blocking_recv().unwrap().unwrap();
    assert_eq!(rec.drops.load(Ordering::SeqCst), 4);
    sink.begin("again");
    engine.speak(req("a", "Hi."), "again".into()).blocking_recv().unwrap().unwrap();
    assert_eq!(rec.opens.load(Ordering::SeqCst), 5);
}

#[test]
fn engine_load_reports_info_and_errors() {
    let (engine, rec, _sink) = fake_engine(Duration::from_secs(60));
    let info = engine
        .load("with-voices".into(), String::new(), None, Some(3))
        .blocking_recv()
        .unwrap()
        .unwrap();
    assert_eq!(info.voices.len(), 1, "voices the loaded model reports");
    assert_eq!(info.sample_rate, 16_000);
    let err = engine
        .load("missing-model".into(), String::new(), None, None)
        .blocking_recv()
        .unwrap()
        .unwrap_err();
    assert!(err.contains("not downloaded"), "{err}");
    for _ in 0..2 {
        let err = engine
            .load("fail-open".into(), String::new(), None, None)
            .blocking_recv()
            .unwrap()
            .unwrap_err();
        assert!(err.contains("damaged"));
    }
    assert_eq!(rec.opens.load(Ordering::SeqCst), 3, "a failed open is retried by the next request");
}

#[test]
fn engine_reports_sink_and_synthesis_errors() {
    let (engine, _rec, sink) = fake_engine(Duration::from_secs(60));
    let err = engine
        .speak(req("m", "Hi."), "never-begun".into())
        .blocking_recv()
        .unwrap()
        .unwrap_err();
    assert!(err.contains("never-begun"), "{err}");
    sink.begin("boom");
    let err = engine
        .speak(req("m", "Please explode."), "boom".into())
        .blocking_recv()
        .unwrap()
        .unwrap_err();
    assert_eq!(err, "synthesis failed");
    // A cancelled sink utterance answers "cancelled" without synthesizing.
    sink.begin("gone");
    sink.cancel("gone");
    let err = engine
        .speak(req("m", "Hi."), "gone".into())
        .blocking_recv()
        .unwrap()
        .unwrap_err();
    assert_eq!(err, CANCELLED);
}

#[test]
fn engine_orders_waiting_speak_jobs_by_the_sink() {
    let (engine, rec, sink) = fake_engine(Duration::from_secs(60));
    for id in ["first", "second", "third"] {
        sink.begin(id);
    }
    let busy = engine.speak(req("m", "slow first sentence."), "first".into());
    wait_until("the first job to run", || texts(&rec).iter().any(|t| t.contains("slow")));
    // Arrive out of order while the worker is busy.
    let third = engine.speak(req("m", "third."), "third".into());
    let second = engine.speak(req("m", "second."), "second".into());
    busy.blocking_recv().unwrap().unwrap();
    second.blocking_recv().unwrap().unwrap();
    third.blocking_recv().unwrap().unwrap();
    let t = texts(&rec);
    let pos = |s: &str| t.iter().position(|x| x == s).unwrap();
    assert!(pos("second.") < pos("third."), "sink order wins: {t:?}");
}

#[test]
fn engine_keeps_fifo_across_non_speak_jobs() {
    let (engine, rec, sink) = fake_engine(Duration::from_secs(60));
    for id in ["x", "y"] {
        sink.begin(id);
    }
    let busy = engine.speak(req("m", "slow warm."), "x".into());
    wait_until("the first job", || texts(&rec).iter().any(|t| t.contains("slow")));
    let synth = engine.synthesize(req("m", "preview."));
    let later = engine.speak(req("m", "later."), "y".into());
    busy.blocking_recv().unwrap().unwrap();
    let audio = synth.blocking_recv().unwrap().unwrap();
    later.blocking_recv().unwrap().unwrap();
    assert_eq!(audio.samples.len(), 80);
    assert_eq!(audio.sample_rate, 16_000);
    let t = texts(&rec);
    let pos = |s: &str| t.iter().position(|x| x == s).unwrap();
    assert!(pos("preview.") < pos("later."), "{t:?}");
}

#[test]
fn cancel_answers_waiting_jobs_and_stops_the_running_one() {
    let (engine, rec, sink) = fake_engine(Duration::from_secs(60));
    for id in ["run", "wait1", "wait2"] {
        sink.begin(id);
    }
    let running = engine.speak(req("m", "slow and long sentence."), "run".into());
    wait_until("the running job to push", || !sink.samples("run").is_empty());
    let w1 = engine.speak(req("m", "one."), "wait1".into());
    let w2 = engine.speak(req("m", "two."), "wait2".into());
    // One utterance: only its job goes.
    engine.cancel(Some("wait1"));
    assert_eq!(w1.blocking_recv().unwrap().unwrap_err(), CANCELLED);
    engine.cancel(Some("run"));
    assert_eq!(running.blocking_recv().unwrap().unwrap_err(), CANCELLED);
    // The running job stopped after its first batch.
    let full = "slow and long sentence.".chars().count() * 10;
    assert!(sink.samples("run").len() < full, "stopped early");
    w2.blocking_recv().unwrap().unwrap();
    assert!(!texts(&rec).contains(&"one.".to_string()), "a cancelled waiting job never synthesizes");
}

#[test]
fn cancel_all_spares_unload_and_a_sink_cancel_stops_speaking() {
    let (engine, rec, sink) = fake_engine(Duration::from_secs(60));
    sink.begin("run");
    sink.begin("next");
    let running = engine.speak(req("m", "slow sentence here."), "run".into());
    wait_until("the running job to push", || !sink.samples("run").is_empty());
    let load = engine.load("other".into(), String::new(), None, None);
    let next = engine.speak(req("m", "next."), "next".into());
    let unload = engine.unload();
    engine.cancel(None);
    assert_eq!(running.blocking_recv().unwrap().unwrap_err(), CANCELLED);
    assert_eq!(load.blocking_recv().unwrap().unwrap_err(), CANCELLED);
    assert_eq!(next.blocking_recv().unwrap().unwrap_err(), CANCELLED);
    unload.blocking_recv().unwrap().expect("unload is never cancelled");
    assert_eq!(engine.waiting(), 0);
    assert_eq!(rec.drops.load(Ordering::SeqCst), 1);

    // The TS pipeline's stop / skip cancels the sink utterance: the engine stops at the next push.
    sink.begin("skip");
    let skipped = engine.speak(req("m", "slow sentence to skip."), "skip".into());
    wait_until("pushes", || !sink.samples("skip").is_empty());
    sink.cancel("skip");
    assert_eq!(skipped.blocking_recv().unwrap().unwrap_err(), CANCELLED);
}

#[test]
fn inference_runs_only_on_the_worker_thread() {
    // Acceptance "no inference on the main thread": loads, warm-ups and every sentence run on the
    // engine's own `sigil-local-tts` thread, whichever thread submits them.
    let (engine, rec, sink) = fake_engine(Duration::from_secs(60));
    sink.begin("a");
    engine.speak(req("m", "One. Two."), "a".into()).blocking_recv().unwrap().unwrap();
    engine.synthesize(req("m", "Three.")).blocking_recv().unwrap().unwrap();
    let submitter = std::thread::spawn({
        let engine = engine.clone();
        move || engine.load("other".into(), String::new(), None, None).blocking_recv().unwrap().unwrap()
    });
    submitter.join().unwrap();
    let threads = rec.threads.lock().unwrap().clone();
    assert!(threads.len() >= 4, "{threads:?}");
    assert!(threads.iter().all(|t| t == "sigil-local-tts"), "{threads:?}");
}

#[test]
fn an_unload_followed_by_a_load_keeps_the_model() {
    let (engine, rec, sink) = fake_engine(Duration::from_secs(60));
    sink.begin("run");
    let running = engine.speak(req("m", "slow sentence here."), "run".into());
    wait_until("the running job to push", || !sink.samples("run").is_empty());
    // The TS backend stops (unload) and a new one starts (load) while the engine is busy.
    let unload = engine.unload();
    let load = engine.load("m".into(), "v".into(), None, None);
    running.blocking_recv().unwrap().expect("the running sentence finishes");
    unload.blocking_recv().unwrap().expect("unload answers");
    load.blocking_recv().unwrap().expect("loads");
    assert_eq!(rec.opens.load(Ordering::SeqCst), 1, "kept, not loaded again");
    assert_eq!(rec.drops.load(Ordering::SeqCst), 0);
    // An unload with nothing waiting frees it.
    engine.unload().blocking_recv().unwrap().unwrap();
    assert_eq!(rec.drops.load(Ordering::SeqCst), 1);
}

#[test]
fn an_idle_model_is_unloaded() {
    let (engine, rec, sink) = fake_engine(Duration::from_millis(150));
    sink.begin("u");
    engine.speak(req("m", "Hi."), "u".into()).blocking_recv().unwrap().unwrap();
    assert_eq!(rec.drops.load(Ordering::SeqCst), 0);
    wait_until("the idle unload", || rec.drops.load(Ordering::SeqCst) == 1);
    // The next request loads again.
    sink.begin("v");
    engine.speak(req("m", "Hi."), "v".into()).blocking_recv().unwrap().unwrap();
    assert_eq!(rec.opens.load(Ordering::SeqCst), 2);
}

#[test]
fn a_panic_in_model_code_fails_only_its_job() {
    let (engine, rec, sink) = fake_engine(Duration::from_secs(60));
    sink.begin("ok");
    engine.speak(req("m", "Hi."), "ok".into()).blocking_recv().unwrap().unwrap();
    // A panic while synthesizing answers that request with an error (instead of wedging the worker)...
    sink.begin("boom");
    sink.begin("queued");
    let boom = engine.speak(req("m", "Please panic."), "boom".into());
    let queued = engine.speak(req("m", "Queued."), "queued".into());
    let err = boom.blocking_recv().unwrap().unwrap_err();
    assert!(err.contains("fake synthesis panic"), "{err}");
    // ... drops the model, which may be half updated ...
    assert_eq!(rec.drops.load(Ordering::SeqCst), 1);
    // ... and the worker keeps serving: the job waiting behind it runs (loading the model again).
    queued.blocking_recv().unwrap().expect("a job waiting behind a panic still runs");
    assert_eq!(rec.opens.load(Ordering::SeqCst), 2);
    // Loads and previews are guarded the same way.
    let err = engine
        .load("panic-open".into(), String::new(), None, None)
        .blocking_recv()
        .unwrap()
        .unwrap_err();
    assert!(err.contains("fake load panic"), "{err}");
    let err = engine
        .synthesize(req("m", "panic in a preview."))
        .blocking_recv()
        .unwrap()
        .unwrap_err();
    assert!(err.contains("fake synthesis panic"), "{err}");
    let audio = engine
        .synthesize(req("m", "Fine."))
        .blocking_recv()
        .unwrap()
        .expect("still serving");
    assert!(!audio.samples.is_empty());
    let threads = rec.threads.lock().unwrap().clone();
    assert!(threads.iter().all(|t| t == "sigil-local-tts"), "{threads:?}");
}

#[test]
fn a_worker_that_ends_answers_the_waiting_jobs_and_the_next_request_restarts_it() {
    let (engine, rec, sink) = fake_engine(Duration::from_secs(60));
    for id in ["run", "w1", "w2", "later"] {
        sink.begin(id);
    }
    let running = engine.speak(req("m", "slow sentence here."), "run".into());
    wait_until("the running job to push", || !sink.samples("run").is_empty());
    // Choosing between two waiting speak jobs asks the sink for its play order, which now panics:
    // the worker thread ends outside the guarded model code.
    sink.panic_order.store(true, Ordering::SeqCst);
    let w1 = engine.speak(req("m", "one."), "w1".into());
    let w2 = engine.speak(req("m", "two."), "w2".into());
    running.blocking_recv().unwrap().expect("the running job finishes");
    assert_eq!(w1.blocking_recv().unwrap().unwrap_err(), WORKER_STOPPED, "answered, not left waiting");
    assert_eq!(w2.blocking_recv().unwrap().unwrap_err(), WORKER_STOPPED);
    assert_eq!(engine.waiting(), 0);
    // The next request starts a new worker, which loads the model again.
    sink.panic_order.store(false, Ordering::SeqCst);
    engine
        .speak(req("m", "Hi."), "later".into())
        .blocking_recv()
        .unwrap()
        .expect("a new worker serves");
    assert_eq!(rec.opens.load(Ordering::SeqCst), 2);
    assert_eq!(rec.drops.load(Ordering::SeqCst), 1, "the ended worker's model was freed");
}

#[test]
fn warm_up_phrases_match_the_language() {
    assert_eq!(warm_up_text(Some("ja")), "こんにちは。");
    assert_eq!(warm_up_text(Some("zh")), "你好。");
    assert_eq!(warm_up_text(Some("pt-BR")), "Olá.");
    assert_eq!(warm_up_text(Some("ru")), "Привет.");
    assert_eq!(warm_up_text(Some("sw")), "Hello.");
    assert_eq!(warm_up_text(None), "Hello.");
}

#[test]
fn catalog_voice_sets_are_consistent() {
    for m in catalog::all() {
        match (m.family, m.voices) {
            (Family::Kokoro, VoiceSet::Kokoro(t)) => assert!(t.contains(&m.default_voice(None).as_str())),
            (Family::Kitten, VoiceSet::Kitten08) => assert_eq!(m.default_voice(None), "Jasper"),
            (Family::Piper, VoiceSet::Piper) => {
                assert_eq!(m.default_voice(None), "0");
                assert!(m.piper_voice.is_some() && m.piper_name.is_some());
            }
            other => panic!("{}: family and voice set disagree: {other:?}", m.id),
        }
        assert!(!m.languages.is_empty(), "{}", m.id);
        assert!([16_000, 22_050, 24_000].contains(&m.sample_rate), "{}", m.id);
    }
}

// -------------------------------------------------------------------------------------------------
// Real models (ignored)
// -------------------------------------------------------------------------------------------------

mod real {
    use super::*;
    use crate::services::speech_rt::sherpa::testkit::{dll_lock, installed, runtime_installed, squash};
    use crate::services::speech_rt::sherpa::{detect, DecodeOptions, DetectHints, OfflineRecognizer, SherpaModel, SherpaRuntime};

    const SENSE_VOICE: &str = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17";

    fn ready(ids: &[&str]) -> bool {
        runtime_installed() && ids.iter().all(|id| installed(id).is_some())
    }

    fn engine() -> (Engine, Arc<FakeSink>) {
        let sink = Arc::new(FakeSink::default());
        sink.queued_ms.store(60_000, Ordering::SeqCst);
        let engine = Engine::new(Arc::new(models::StoreLoader::default()), sink.clone(), Duration::from_secs(600));
        (engine, sink)
    }

    fn sense_voice() -> Option<OfflineRecognizer> {
        let root = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from)?;
        let dir = root.join("sherpa-onnx").join("models").join(SENSE_VOICE);
        if !dir.is_dir() {
            eprintln!("{} is missing; intelligibility checks skipped", dir.display());
            return None;
        }
        let SherpaModel::Offline(cfg) = detect(&dir, &DetectHints::default()).unwrap() else {
            panic!("SenseVoice is an offline model")
        };
        let rt = SherpaRuntime::installed().expect("runtime");
        Some(OfflineRecognizer::new(&rt, &cfg).expect("SenseVoice loads"))
    }

    fn synth(engine: &Engine, model: &str, voice: &str, text: &str) -> (Vec<f32>, u32, Duration) {
        let t0 = Instant::now();
        let a = engine
            .synthesize(LocalTtsRequest {
                model: model.into(),
                voice: voice.into(),
                text: text.into(),
                threads: Some(4),
                ..Default::default()
            })
            .blocking_recv()
            .unwrap()
            .unwrap_or_else(|e| panic!("{model} / {voice}: {e}"));
        (a.samples, a.sample_rate, t0.elapsed())
    }

    fn rms(s: &[f32]) -> f32 {
        (s.iter().map(|x| x * x).sum::<f32>() / s.len().max(1) as f32).sqrt()
    }

    fn transcript(asr: &OfflineRecognizer, samples: &[f32], rate: u32) -> String {
        asr.transcribe(rate, samples, &DecodeOptions::default())
            .expect("transcribes")
            .text
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx and tts/kokoro-v1.0 installed"]
    fn real_kokoro_rtf_on_four_threads_and_first_audio() {
        let _g = dll_lock();
        if !ready(&["tts/kokoro-v1.0"]) {
            return;
        }
        // Kokoro on Sigil's ONNX Runtime (kokoro_ort) and plain sherpa-onnx, interleaved sentence
        // by sentence so both see the same background load. The first round meets every sentence
        // length for the first time (what real speech does); the second repeats them.
        let engine_on = |sigil_ort: bool| {
            let sink = Arc::new(FakeSink::default());
            sink.queued_ms.store(60_000, Ordering::SeqCst);
            let e = Engine::new(Arc::new(models::StoreLoader::new(sigil_ort)), sink.clone(), Duration::from_secs(600));
            let t0 = Instant::now();
            e.load("tts/kokoro-v1.0".into(), "af_heart".into(), None, Some(4))
                .blocking_recv()
                .unwrap()
                .expect("Kokoro loads");
            eprintln!("Kokoro v1.0 fp32 load + warm-up ({}): {} ms", if sigil_ort { "Sigil ORT" } else { "sherpa-onnx" }, t0.elapsed().as_millis());
            (e, sink)
        };
        let (engine, sink) = engine_on(true);
        let (sherpa, sherpa_sink) = engine_on(false);
        let sentences = [
            "The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun.",
            "When the evening finally arrived, the whole village gathered in the square to watch the lanterns rise.",
            "Please remember to bring your umbrella tomorrow, because the forecast says it will rain all day.",
            "Sigil turns your voice into captions and speech for streams and games.",
        ];
        let cpu = crate::services::bench::metrics::process_cpu_time;
        let speak = |e: &Engine, sink: &FakeSink, id: String, text: &str| -> (f64, f64, SpeakReport) {
            sink.begin(&id);
            let c0 = cpu();
            let t = Instant::now();
            let report = e
                .speak(
                    LocalTtsRequest {
                        model: "tts/kokoro-v1.0".into(),
                        voice: "af_heart".into(),
                        text: text.to_string(),
                        threads: Some(4),
                        ..Default::default()
                    },
                    id,
                )
                .blocking_recv()
                .unwrap()
                .expect("speaks");
            let wall = t.elapsed().as_secs_f64();
            // Process CPU seconds per second of audio (both engines idle otherwise).
            let cpu_s = match (c0, cpu()) {
                (Some(a), Some(b)) => b.saturating_sub(a).as_secs_f64(),
                _ => f64::NAN,
            };
            let audio_s = report.frames as f64 / f64::from(report.sample_rate);
            (wall / audio_s, cpu_s / audio_s, report)
        };
        let (mut ours, mut theirs, mut cpu_ours, mut cpu_theirs) = (Vec::new(), Vec::new(), 0.0, 0.0);
        for round in 0..2 {
            for (i, s) in sentences.iter().enumerate() {
                let (rtf, c_ours, report) = speak(&engine, &sink, format!("rtf{round}-{i}"), s);
                let (rtf_sherpa, c_sherpa, _) = speak(&sherpa, &sherpa_sink, format!("sherpa{round}-{i}"), s);
                eprintln!(
                    "sentence {i}: {:.2} s of audio, RTF {rtf:.3} (sherpa-onnx {rtf_sherpa:.3}), CPU s per audio s {c_ours:.2} ({c_sherpa:.2}), first audio {} ms",
                    report.frames as f64 / f64::from(report.sample_rate),
                    report.first_audio_ms
                );
                ours.push(rtf);
                theirs.push(rtf_sherpa);
                cpu_ours += c_ours;
                cpu_theirs += c_sherpa;
            }
        }
        eprintln!("CPU per audio second, summed over the sentences: {cpu_ours:.2} on Sigil ORT, {cpu_theirs:.2} inside sherpa-onnx");
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (m_ours, m_theirs) = (median(&mut ours), median(&mut theirs));
        eprintln!("Kokoro v1.0 fp32, 4 threads: median RTF {m_ours:.3} on Sigil ORT, {m_theirs:.3} inside sherpa-onnx (target 0.3)");
        // Generous bounds: the dev PC is shared with builds; the measured values go in the report.
        let default_median = if kokoro_ort::SIGIL_ORT_BY_DEFAULT { m_ours } else { m_theirs };
        assert!(default_median < 0.5, "Kokoro is far slower than expected: RTF {default_median:.3}");
        assert!(cpu_ours < cpu_theirs, "Sigil's runtime (no spinning) should use less CPU than sherpa-onnx's");

        // The first sentence of a burst (default runtime): its first clause is synthesized alone.
        let (default_engine, default_sink) = if kokoro_ort::SIGIL_ORT_BY_DEFAULT { (&engine, &sink) } else { (&sherpa, &sherpa_sink) };
        default_sink.queued_ms.store(0, Ordering::SeqCst);
        let (_, _, report) = speak(default_engine, default_sink, "first".into(), sentences[1]);
        eprintln!(
            "first sentence with the clause split: first audio {} ms, done {} ms",
            report.first_audio_ms, report.total_ms
        );
        assert!(report.first_audio_ms < report.total_ms);
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx and tts/kokoro-v1.0, SIGIL_TEST_MODELS_DIR with SenseVoice"]
    fn real_kokoro_speaks_english_japanese_and_chinese_intelligibly() {
        let _g = dll_lock();
        if !ready(&["tts/kokoro-v1.0"]) {
            return;
        }
        let (engine, _sink) = engine();
        let asr = sense_voice();
        let cases: &[(&str, &str, &[&str])] = &[
            // Numbers must stay numbers (the Chinese FSTs used to turn them into hanzi).
            ("af_heart", "I have 3 apples and 25 oranges.", &["apples", "oranges"]),
            ("bf_emma", "Good morning, the weather is lovely today.", &["morning", "weather"]),
            // Kana: sherpa-onnx 1.13.8's multi-language frontend reads kanji with Mandarin readings
            // (lexicon-zh), so Japanese is checked with kana only (see local_tts::kokoro).
            ("jf_alpha", "きょうはいいてんきですね。", &["天気"]),
            ("zf_xiaoxiao", "今天天气很好，我们去公园散步吧。", &["天气", "公园"]),
        ];
        for (voice, text, expect) in cases {
            let (samples, rate, took) = synth(&engine, "tts/kokoro-v1.0", voice, text);
            let secs = samples.len() as f32 / rate as f32;
            eprintln!("{voice}: {secs:.2} s in {} ms (rms {:.3})", took.as_millis(), rms(&samples));
            assert_eq!(rate, 24_000);
            assert!(secs > 0.8 && rms(&samples) > 0.01, "{voice}: silent or empty");
            if let Some(asr) = &asr {
                let heard = transcript(asr, &samples, rate);
                eprintln!("  heard: {heard}");
                let h = squash(&heard);
                for word in *expect {
                    assert!(h.contains(&squash(word)), "{voice}: {word:?} not heard in {heard:?}");
                }
                if *voice == "af_heart" {
                    assert!(h.contains("three") || h.contains('3'), "the number 3 was not read: {heard:?}");
                    assert!(h.contains("25") || h.contains("twentyfive"), "the number 25 was not read: {heard:?}");
                }
            }
        }
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx and tts/kokoro-v1.1-zh, SIGIL_TEST_MODELS_DIR with SenseVoice"]
    fn real_kokoro_v1_1_zh_speaks_mandarin_and_mixed_lines() {
        let _g = dll_lock();
        if !ready(&["tts/kokoro-v1.1-zh"]) {
            return;
        }
        let (engine, _sink) = engine();
        let info = engine
            .load("tts/kokoro-v1.1-zh".into(), "zf_001".into(), None, Some(4))
            .blocking_recv()
            .unwrap()
            .expect("loads");
        assert_eq!(info.voices.len(), 103);
        let asr = sense_voice();
        for (voice, text, word) in [
            ("zf_001", "今天天气很好，我们去公园散步吧。", "天气"),
            ("zm_010", "我明天要去北京开会。", "北京"),
            ("zf_001", "我们用Sigil做直播字幕。", "直播"),
        ] {
            let (samples, rate, _) = synth(&engine, "tts/kokoro-v1.1-zh", voice, text);
            assert!(samples.len() as f32 / rate as f32 > 0.8 && rms(&samples) > 0.01, "{voice}");
            if let Some(asr) = &asr {
                let heard = transcript(asr, &samples, rate);
                eprintln!("{voice}: {heard}");
                assert!(squash(&heard).contains(word), "{voice}: {word} not heard in {heard:?}");
            }
        }
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx and tts/piper-ru_ru-irina-medium"]
    fn real_piper_russian_voice_speaks() {
        let _g = dll_lock();
        let id = "tts/piper-ru_ru-irina-medium";
        if !ready(&[id]) {
            return;
        }
        let (engine, _sink) = engine();
        let info = engine
            .load(id.into(), String::new(), None, Some(2))
            .blocking_recv()
            .unwrap()
            .expect("loads");
        assert_eq!(info.sample_rate, 22_050);
        assert_eq!(info.languages, ["ru"]);
        let text = "Привет! Это проверка русского голоса. Сегодня хорошая погода, и мы идём гулять в парк.";
        let (samples, rate, took) = synth(&engine, id, "", text);
        let secs = samples.len() as f64 / f64::from(rate);
        let rtf = took.as_secs_f64() / secs;
        eprintln!("Piper ru: {secs:.2} s, RTF {rtf:.3}");
        assert!(secs > 3.0 && rms(&samples) > 0.01);
        assert!(rtf < 0.3, "Piper medium should be far faster than real time: {rtf:.3}");
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx and tts/kitten-mini-en-v0.8, SIGIL_TEST_MODELS_DIR with SenseVoice"]
    fn real_kitten_speaks_through_sherpa_and_the_vendored_fallback() {
        let _g = dll_lock();
        let id = "tts/kitten-mini-en-v0.8";
        if !ready(&[id]) {
            return;
        }
        let asr = sense_voice();
        let text = "Hello, this is a quick test of the kitten voice.";
        let (engine, _sink) = engine();
        let (samples, rate, _) = synth(&engine, id, "Bella", text);
        assert_eq!(rate, 24_000);
        assert!(rms(&samples) > 0.01);
        // The vendored Kitten on the same package (what runs when sherpa cannot load).
        let dir = installed(id).unwrap();
        let model = ["model.onnx", "model.fp32.onnx"]
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.is_file())
            .expect("model file");
        let speakers: Vec<String> = voices::KITTEN_V0_8.iter().map(|(_, raw)| raw.to_string()).collect();
        let aliases = voices::KITTEN_V0_8
            .iter()
            .map(|(a, raw)| (a.to_string(), raw.to_string()))
            .collect();
        let fallback = crate::services::local_tts::kitten::hub::load_sherpa_kitten_package(&model, &dir.join("voices.bin"), &speakers, aliases)
            .expect("the vendored Kitten reads the sherpa package");
        let vendored = fallback.generate(text, "Bella", 1.0, true).expect("vendored synthesis");
        assert!(rms(&vendored) > 0.01);
        if let Some(asr) = &asr {
            for (label, audio) in [("sherpa", &samples), ("vendored", &vendored)] {
                let heard = transcript(asr, audio, 24_000);
                eprintln!("Kitten {label}: {heard}");
                let h = squash(&heard);
                assert!(h.contains("hello") && h.contains("test"), "Kitten {label}: {heard:?}");
            }
        }
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx and tts/kokoro-v1.0 installed"]
    fn real_cancel_stops_a_long_utterance_and_folders_resolve() {
        let _g = dll_lock();
        let id = "tts/kokoro-v1.0";
        if !ready(&[id]) {
            return;
        }
        let (engine, sink) = engine();
        engine
            .load(id.into(), "af_heart".into(), None, Some(4))
            .blocking_recv()
            .unwrap()
            .unwrap();
        sink.begin("long");
        let long = "This is the first sentence of a long text. ".repeat(12);
        let rx = engine.speak(
            LocalTtsRequest {
                model: id.into(),
                voice: "af_heart".into(),
                text: long,
                threads: Some(4),
                ..Default::default()
            },
            "long".into(),
        );
        wait_until("the first chunk", || !sink.samples("long").is_empty());
        let t = Instant::now();
        engine.cancel(Some("long"));
        assert_eq!(rx.blocking_recv().unwrap().unwrap_err(), CANCELLED);
        eprintln!("cancel took effect after {} ms", t.elapsed().as_millis());
        assert!(t.elapsed() < Duration::from_secs(3), "stops within one sentence");

        // The installed package also resolves as a plain folder (metadata tells the family).
        let dir = installed(id).unwrap();
        let r = models::StoreLoader::default()
            .resolve(&dir.display().to_string(), "jf_alpha", None, Some(2))
            .expect("folder resolves");
        assert_eq!(r.info.family, "kokoro");
        assert_eq!(r.info.voices.len(), 54);
        assert_eq!(r.voice.sid, 37);
        assert!(matches!(&r.plan, OpenPlan::Sherpa { validate_first: true, .. }));
        assert!(r.key.variant.contains("lang=ja"));
    }

    /// Correlation of the overlapping part of two signals.
    fn correlation(a: &[f32], b: &[f32]) -> f64 {
        let n = a.len().min(b.len());
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for i in 0..n {
            ab += f64::from(a[i]) * f64::from(b[i]);
            aa += f64::from(a[i]).powi(2);
            bb += f64::from(b[i]).powi(2);
        }
        ab / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE)
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx and tts/kokoro-v1.0 installed"]
    fn real_kokoro_on_sigil_ort_matches_sherpa() {
        use crate::services::local_tts::kokoro_ort::{decode_tap, scale_silence, write_tap, KokoroModel};
        use crate::services::speech_rt::sherpa::{GenerateOptions, OfflineTts};
        let _g = dll_lock();
        let Some(dir) = installed("tts/kokoro-v1.0") else { return };
        if !runtime_installed() {
            return;
        }
        let rt = SherpaRuntime::installed().expect("runtime");
        let meta = crate::services::speech_rt::sherpa::onnx_info::summary(&dir.join("model.onnx")).unwrap();
        let tap = write_tap(&kokoro_ort::tap_dir(), &meta.metadata, "audio").unwrap();
        let real_cfg = kokoro::config(&dir, &KokoroSetup::for_voice("af_heart", None), 4).unwrap();
        let mut tap_cfg = real_cfg.clone();
        if let TtsModel::Kokoro { model, .. } = &mut tap_cfg.model {
            *model = tap;
        }
        let frontend = OfflineTts::new(&rt, &tap_cfg).expect("sherpa-onnx loads the token tap");
        let sherpa = OfflineTts::new(&rt, &real_cfg).unwrap();
        let mut model = KokoroModel::load(&dir.join("model.onnx"), &dir.join("voices.bin"), 54, 4).unwrap();
        let plain = GenerateOptions {
            sid: 3,
            speed: 1.0,
            silence_scale: Some(1.0),
        };
        let default_pauses = GenerateOptions {
            silence_scale: None,
            ..plain
        };
        let ours = |model: &mut KokoroModel, text: &str| -> Vec<f32> {
            let mut tapped = Vec::new();
            frontend
                .generate(text, &plain, None, Some(&mut |s: &[f32]| {
                    tapped.extend_from_slice(s);
                    true
                }))
                .unwrap();
            decode_tap(&tapped)
                .unwrap()
                .iter()
                .flat_map(|t| model.run(t, 3, 1.0).unwrap())
                .collect()
        };
        let sentences = [
            "Hello there, this is a test of the local voice.",
            "Please remember to bring your umbrella tomorrow, because the forecast says it will rain all day.",
        ];
        let (mut t_ours, mut t_sherpa, mut audio_s) = (0f64, 0f64, 0f64);
        for text in sentences {
            // Same tokens, same style row: the same speech (Kokoro's generator adds random noise, so
            // the samples differ slightly).
            let a = ours(&mut model, text);
            let b = sherpa.generate(text, &plain, None, None).unwrap().samples;
            let c = correlation(&a, &b);
            eprintln!("{text:?}: {} vs {} samples, correlation {c:.4}", a.len(), b.len());
            assert_eq!(a.len(), b.len(), "same length as sherpa-onnx");
            assert!(c > 0.98, "not the same speech as sherpa-onnx: correlation {c:.4}");
            // Pauses shortened like sherpa-onnx's default silence scale.
            let scaled = scale_silence(&a, 24_000, crate::services::speech_rt::sherpa::tts::DEFAULT_SILENCE_SCALE);
            let reference = sherpa.generate(text, &default_pauses, None, None).unwrap().samples.len();
            let diff = (scaled.len() as f64 - reference as f64).abs() / reference as f64;
            eprintln!("  with pauses shortened: {} vs {reference} samples", scaled.len());
            assert!(diff < 0.01, "pause shortening differs from sherpa-onnx by {:.1}%", diff * 100.0);
            // Speed, best of 3 each.
            let best = |f: &mut dyn FnMut()| {
                (0..3)
                    .map(|_| {
                        let t = Instant::now();
                        f();
                        t.elapsed().as_secs_f64()
                    })
                    .fold(f64::MAX, f64::min)
            };
            t_ours += best(&mut || {
                ours(&mut model, text);
            });
            t_sherpa += best(&mut || {
                sherpa.generate(text, &plain, None, None).unwrap();
            });
            audio_s += b.len() as f64 / 24_000.0;
        }
        // Best of 3 repeats the same lengths (see kokoro_ort on new lengths); logged, not asserted.
        let (rtf_ours, rtf_sherpa) = (t_ours / audio_s, t_sherpa / audio_s);
        eprintln!("4 threads, repeated lengths: RTF {rtf_ours:.3} on Sigil's ONNX Runtime vs {rtf_sherpa:.3} inside sherpa-onnx");
    }
}
