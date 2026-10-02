//! Tests of the `sbv2` engine: wire formats, the request mapping, and the whole engine on
//! stand-in graphs. The end-to-end tests need the jpreprocess dictionary
//! (`$SIGIL_TEST_MODELS_DIR/naist-jdic`, from jpreprocess v0.15.0's `naist-jdic-jpreprocess.tar.gz`);
//! `real_model_sbv2_speaks` also needs the real BERT (`$SIGIL_TEST_MODELS_DIR/sbv2-bert-ja/` with
//! `model_fp16.onnx` and `tokenizer.json`) and a voice (`$SIGIL_TEST_MODELS_DIR/sbv2-voice.aivmx`).

use std::sync::{Arc, Mutex};

use base64::Engine as _;

use super::engine::{LoadSpec, PcmWriter, SinkPort, Worker};
use super::format::{npy_f32, test_manifest, with_metadata};
use super::synth::{fake, CharTokenizer};
use super::*;
use crate::services::tts_sink::SinkError;

// -------------------------------------------------------------------------------------------------
// Wire formats and settings
// -------------------------------------------------------------------------------------------------

#[test]
fn model_rows_keep_the_frozen_fields() {
    let stored = StoredModel {
        id: "mao-abc123".into(),
        name: "まお".into(),
        format: library::FORMAT_AIVMX.into(),
        description: String::new(),
        creators: vec![],
        version: "1.0.0".into(),
        uuid: String::new(),
        architecture: "Style-Bert-VITS2 (JP-Extra)".into(),
        jp_extra: true,
        sample_rate: 44_100,
        speakers: vec![
            SpeakerInfo {
                name: "まお".into(),
                id: 0,
                styles: vec![format::StyleInfo { name: "ノーマル".into(), id: 0 }, format::StyleInfo { name: "あまあま".into(), id: 1 }],
                languages: vec!["ja".into()],
            },
            SpeakerInfo { name: "B".into(), id: 1, styles: vec![format::StyleInfo { name: "ノーマル".into(), id: 0 }], languages: vec!["ja".into()] },
        ],
        model_file: library::AIVMX_FILE.into(),
        styles_file: None,
        source_name: "mao.aivmx".into(),
        size_bytes: 10,
        sha256: "00".into(),
        imported_unix_ms: 1,
    };
    let row = Sbv2Model::from(&stored);
    let wire = serde_json::to_value(&row).unwrap();
    assert_eq!(wire["id"], "mao-abc123");
    assert_eq!(wire["format"], "aivmx");
    assert_eq!(wire["speakers"], serde_json::json!(["まお", "B"]));
    assert_eq!(wire["styles"], serde_json::json!(["ノーマル", "あまあま"]), "every style once, first seen first");
    assert_eq!(wire["voices"][0]["styles"][1]["id"], 1);
    assert_eq!(wire["sampleRate"], 44_100);
    // The scaffold's shape still deserializes (new fields default).
    let old: Sbv2Model = serde_json::from_str(r#"{"id":"a","name":"A","format":"sbv2","speakers":[],"styles":[]}"#).unwrap();
    assert!(old.voices.is_empty());
}

#[test]
fn speak_requests_default_and_clamp() {
    let r: Sbv2SpeakRequest = serde_json::from_str(r#"{"text":"こんにちは","style":"Happy","styleWeight":50,"speed":9,"sdpRatio":-1}"#).unwrap();
    assert_eq!(r.noise, synth::DEFAULT_NOISE);
    let job = SpeakJob::from(r);
    assert_eq!(job.style, "Happy");
    assert_eq!(job.style_weight, 20.0);
    assert_eq!(job.speed, 2.0);
    assert_eq!(job.sdp_ratio, 0.0);
    let nan = SpeakJob::from(Sbv2SpeakRequest { speed: f32::NAN, noise: f32::INFINITY, ..Sbv2SpeakRequest::default() });
    assert_eq!((nan.speed, nan.noise), (1.0, synth::DEFAULT_NOISE));
}

#[test]
fn start_settings_need_a_valid_model() {
    assert!(parse_config(Value::Null).unwrap_err().contains("Choose"));
    assert!(parse_config(serde_json::json!({ "model": " " })).is_err());
    assert!(parse_config(serde_json::json!({ "model": "../x" })).is_err());
    let c = parse_config(serde_json::json!({ "model": " mao-1 ", "ep": "dml", "threads": 2 })).unwrap();
    assert_eq!((c.model.as_str(), c.ep.as_str(), c.threads), ("mao-1", "dml", Some(2)));
    assert!(parse_config(serde_json::json!({ "model": 5 })).is_err());
}

#[test]
fn long_sentences_are_spoken_in_clauses() {
    assert!(chunks("  ").is_empty());
    assert_eq!(chunks("短い文です。"), ["短い文です。"]);
    let long = "あ".repeat(100) + "、" + &"い".repeat(100) + "。";
    let parts = chunks(&long);
    assert_eq!(parts.len(), 2);
    assert!(parts[0].ends_with('、'));
    assert!(parts.iter().all(|p| p.chars().count() <= MAX_CHUNK_CHARS));
    // No punctuation at all: cut at the limit.
    let run = "う".repeat(400);
    let parts = chunks(&run);
    assert_eq!(parts.iter().map(|p| p.chars().count()).sum::<usize>(), 400);
    assert!(parts.iter().all(|p| p.chars().count() <= MAX_CHUNK_CHARS));
}

#[test]
fn sentences_are_peak_normalized_like_style_bert_vits2() {
    // Quiet raw output is raised, loud output is scaled down instead of clipped; the shape is kept.
    let mut quiet = vec![0.1, -0.4, 0.2, 0.0];
    normalize_peak(&mut quiet);
    assert!((quiet[1] + OUTPUT_PEAK).abs() < 1e-6);
    assert!((quiet[0] / quiet[2] - 0.5).abs() < 1e-6);
    let mut loud = vec![1.6, -0.8, 0.4];
    normalize_peak(&mut loud);
    assert!((loud[0] - OUTPUT_PEAK).abs() < 1e-6);
    assert!((loud[1] + OUTPUT_PEAK / 2.0).abs() < 1e-6);
    assert!(loud.iter().all(|s| s.abs() <= OUTPUT_PEAK + 1e-6));
    // Silence (and a stray non-finite sample) stays silent.
    let mut silent = vec![0.0; 8];
    normalize_peak(&mut silent);
    assert!(silent.iter().all(|s| *s == 0.0));
    let mut odd = vec![0.5, f32::NAN, -0.25];
    normalize_peak(&mut odd);
    assert_eq!(odd[1], 0.0);
    assert!((odd[0] - OUTPUT_PEAK).abs() < 1e-6);
    let mut empty: Vec<f32> = Vec::new();
    normalize_peak(&mut empty);
}

#[test]
fn engine_facts_name_the_downloads() {
    let i = info();
    assert_eq!(i.engine, TtsHelperEngine::Sbv2);
    assert_eq!(i.requires, [BERT_MODEL_ID, DICTIONARY_ID]);
    assert!(!i.cloning);
    // Every catalog id this engine needs is in the embedded catalog.
    let catalog: Value = serde_json::from_str(crate::services::model_store::EMBEDDED_CATALOG_JSON).unwrap();
    let ids: Vec<&str> = catalog["entries"].as_array().unwrap().iter().filter_map(|e| e["id"].as_str()).collect();
    for id in [BERT_MODEL_ID, DICTIONARY_ID] {
        assert!(ids.contains(&id), "{id} is not in the catalog");
    }
}

#[test]
fn bert_files_prefer_fp16_and_need_the_tokenizer() {
    let dir = std::env::temp_dir().join(format!("sigil-sbv2-bert-files-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    assert!(bert_files(&dir).is_err());
    std::fs::write(dir.join("model.onnx"), b"x").unwrap();
    assert!(bert_files(&dir).unwrap_err().contains("tokenizer.json"));
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    assert_eq!(bert_files(&dir).unwrap().0, dir.join("model.onnx"));
    std::fs::write(dir.join("model_fp16.onnx"), b"x").unwrap();
    assert_eq!(bert_files(&dir).unwrap().0, dir.join("model_fp16.onnx"));
    let _ = std::fs::remove_dir_all(&dir);
}

// -------------------------------------------------------------------------------------------------
// Tokenizer
// -------------------------------------------------------------------------------------------------

/// A character-level `tokenizer.json` like the real one (WordLevel over single characters, NFKC).
fn char_tokenizer_json(chars: &str) -> String {
    let mut vocab = serde_json::Map::new();
    for (i, t) in ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"].iter().enumerate() {
        vocab.insert(t.to_string(), serde_json::json!(i));
    }
    for c in chars.chars() {
        let n = vocab.len();
        vocab.entry(c.to_string()).or_insert(serde_json::json!(n));
    }
    serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": { "type": "NFKC" },
        "pre_tokenizer": { "type": "Sequence", "pretokenizers": [
            { "type": "WhitespaceSplit" },
            { "type": "Split", "pattern": { "Regex": "." }, "behavior": "Isolated", "invert": false }
        ]},
        "post_processor": null,
        "decoder": null,
        "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]" }
    })
    .to_string()
}

#[test]
fn the_tokenizer_gives_one_id_per_character() {
    let dir = std::env::temp_dir().join(format!("sigil-sbv2-tok-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tokenizer.json");
    std::fs::write(&path, char_tokenizer_json("こんにちは,A")).unwrap();
    let tk = CharTokenizer::load(&path).unwrap();
    let ids = tk.ids("こんにちは");
    assert_eq!(ids.len(), 7);
    assert_eq!((ids[0], ids[6]), (1, 2), "CLS and SEP");
    // Unknown characters, and ones NFKC turns into several, are [UNK]; full-width letters fold.
    let ids = tk.ids("は㍿Ａ");
    assert_eq!(ids.len(), 5);
    assert_eq!(ids[2], 3);
    assert_eq!(ids[3], tk.ids("A")[1]);
    let _ = std::fs::remove_dir_all(&dir);
}

// -------------------------------------------------------------------------------------------------
// The engine on stand-in graphs
// -------------------------------------------------------------------------------------------------

#[derive(Default)]
struct RecordingSink {
    pushes: Mutex<Vec<(usize, u32)>>,
}

struct RecWriter(Arc<RecordingSink>);

impl PcmWriter for RecWriter {
    fn push(&mut self, samples: &[f32], rate: u32) -> Result<(), SinkError> {
        self.0.pushes.lock().unwrap().push((samples.len(), rate));
        Ok(())
    }
    fn is_cancelled(&self) -> bool {
        false
    }
}

struct RecSink(Arc<RecordingSink>);

impl SinkPort for RecSink {
    fn writer(&self, _: &str) -> Result<Box<dyn PcmWriter>, SinkError> {
        Ok(Box::new(RecWriter(Arc::clone(&self.0))))
    }
    fn order(&self) -> Vec<String> {
        Vec::new()
    }
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("sigil-sbv2-e2e-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// An `.aivmx` around the stand-in voice graph (two styles, 4 wide).
fn fake_aivmx(multilingual: bool) -> Vec<u8> {
    let npy = npy_f32(2, 4, &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    with_metadata(
        fake::voice(4, multilingual, true),
        &[
            ("aivm_manifest", &test_manifest("Test", &[("Test", &["ノーマル", "Joy"])], !multilingual)),
            ("aivm_hyper_parameters", r#"{"version":"2.0-JP-Extra","data":{"sampling_rate":44100}}"#),
            ("aivm_style_vectors", &base64::engine::general_purpose::STANDARD.encode(npy)),
        ],
    )
}

fn block<T>(rx: tokio::sync::oneshot::Receiver<T>) -> T {
    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(rx).expect("answered")
}

#[test]
#[ignore = "needs SIGIL_TEST_MODELS_DIR with naist-jdic (jpreprocess v0.15.0)"]
fn the_engine_speaks_an_imported_voice_through_the_worker() {
    let dict = g2p::test_dictionary().expect("SIGIL_TEST_MODELS_DIR/naist-jdic");
    let s = Scratch::new("worker");
    let bert_dir = s.0.join("bert");
    std::fs::create_dir_all(&bert_dir).unwrap();
    std::fs::write(bert_dir.join("model_fp16.onnx"), fake::bert(1024, true)).unwrap();
    std::fs::write(bert_dir.join("tokenizer.json"), char_tokenizer_json("こんにちは今日はいい天気ですね")).unwrap();
    let lib = s.0.join("lib");

    for multilingual in [false, true] {
        let src = s.0.join(format!("voice-{multilingual}.aivmx"));
        std::fs::write(&src, fake_aivmx(multilingual)).unwrap();
        let m = library::import(&lib, &src, &check_import).expect("the stand-in voice passes the import check");

        let backend = RealBackend {
            assets: Some(Assets { bert_dir: bert_dir.clone(), dictionary_dir: dict.clone() }),
            library: Some(lib.clone()),
            ..RealBackend::default()
        };
        let sink = Arc::new(RecordingSink::default());
        let w = Worker::new(Box::new(backend), Arc::new(RecSink(Arc::clone(&sink))));
        let loaded = block(w.load(LoadSpec { model: m.id.clone(), ep: "cpu".into(), threads: 1 })).unwrap();
        assert_eq!(loaded.sample_rate, 44_100);

        let req = Sbv2SpeakRequest { text: "今日はいい天気ですね。".into(), style: "Joy".into(), ..Sbv2SpeakRequest::default() };
        let report = block(w.speak(SpeakJob::from(req), "u1".into())).unwrap();
        // The stand-in returns 512 samples per (blanked) phoneme: a real sentence's length.
        assert!(report.frames >= 512 * 20, "{} frames", report.frames);
        assert_eq!(report.frames % 512, 0);
        let pushes = sink.pushes.lock().unwrap().clone();
        assert!(pushes.iter().all(|(n, r)| *r == 44_100 && *n <= 22_050), "{pushes:?}");
        assert_eq!(pushes.iter().map(|p| p.0 as u64).sum::<u64>(), report.frames);

        // Nothing readable: no audio, no error.
        let r = block(w.speak(SpeakJob::from(Sbv2SpeakRequest { text: "🎉".into(), ..Sbv2SpeakRequest::default() }), "u2".into())).unwrap();
        assert_eq!(r.frames, 0);
        block(w.unload()).unwrap();
        // After an unload a speak reloads nothing on its own: it says the voice is not started.
        let e = block(w.speak(SpeakJob::from(Sbv2SpeakRequest { text: "はい".into(), ..Sbv2SpeakRequest::default() }), "u3".into())).unwrap_err();
        assert!(e.contains("not started"), "{e}");
    }
}

#[test]
#[ignore = "needs SIGIL_TEST_MODELS_DIR with naist-jdic and sbv2-bert-ja/tokenizer.json"]
fn real_tokenizer_knows_what_the_frontend_writes() {
    // The BERT reads `bert_text` one token per character, aligned with `word2ph`: every character the
    // frontend keeps (kana, kanji, Latin letters after NFKC, the SBV2 punctuation) must be in the real
    // vocabulary, or its features would be [UNK]'s.
    let root = PathBuf::from(std::env::var_os("SIGIL_TEST_MODELS_DIR").expect("SIGIL_TEST_MODELS_DIR"));
    let tk = CharTokenizer::load(&root.join("sbv2-bert-ja").join(TOKENIZER_FILE)).expect("the real tokenizer loads");
    let fe = g2p::Frontend::load(&g2p::test_dictionary().expect("naist-jdic")).unwrap();
    /// `[UNK]` in the real vocabulary.
    const UNK_ID: i64 = 3;
    for text in [
        "今日はいい天気ですね。",
        "ＡＢＣとabcで100円！",
        "「はい」と言った…",
        "ラーメン食べたっけ？",
        "ﾃｽﾄです、2024年1月1日。",
        "えーと～そうですね・・・",
        "々や〇も読めます",
    ] {
        let a = fe.analyze(text).unwrap();
        let ids = tk.ids(&a.bert_text);
        assert_eq!(ids.len(), a.word2ph.len(), "{text}: {}", a.bert_text);
        let unknown: Vec<char> = a.bert_text.chars().zip(&ids[1..]).filter(|(_, id)| **id == UNK_ID).map(|(c, _)| c).collect();
        assert!(unknown.is_empty(), "{text}: {} has unknown characters {unknown:?}", a.bert_text);
    }
}

#[test]
#[ignore = "needs SIGIL_TEST_MODELS_DIR with naist-jdic, sbv2-bert-ja/ (model_fp16.onnx, tokenizer.json) and sbv2-voice.aivmx"]
fn real_model_sbv2_speaks() {
    let root = PathBuf::from(std::env::var_os("SIGIL_TEST_MODELS_DIR").expect("SIGIL_TEST_MODELS_DIR"));
    let dict = g2p::test_dictionary().expect("naist-jdic");
    let s = Scratch::new("real");
    let lib = s.0.join("lib");
    let m = library::import(&lib, &root.join("sbv2-voice.aivmx"), &check_import).expect("import the real voice");
    let backend = RealBackend {
        assets: Some(Assets { bert_dir: root.join("sbv2-bert-ja"), dictionary_dir: dict }),
        library: Some(lib),
        ..RealBackend::default()
    };
    let sink = Arc::new(RecordingSink::default());
    let w = Worker::new(Box::new(backend), Arc::new(RecSink(Arc::clone(&sink))));
    let t0 = std::time::Instant::now();
    block(w.load(LoadSpec { model: m.id.clone(), ep: "cpu".into(), threads: 4 })).unwrap();
    eprintln!("load + warm-up: {} ms", t0.elapsed().as_millis());
    let text = "こんにちは、今日はいい天気ですね。散歩に行きませんか？";
    let report = block(w.speak(SpeakJob::from(Sbv2SpeakRequest { text: text.into(), ..Sbv2SpeakRequest::default() }), "u".into())).unwrap();
    let seconds = report.frames as f64 / f64::from(report.sample_rate);
    eprintln!(
        "{} s of audio at {} Hz, first audio after {} ms (RTF {:.2})",
        seconds,
        report.sample_rate,
        report.first_audio_ms,
        f64::from(report.first_audio_ms) / 1000.0 / seconds
    );
    assert!((2.0..12.0).contains(&seconds), "{seconds} s");
}
