//! Tests of the Chatterbox engine (P6-I).
//!
//! - The text frontend against the model's real tokenizer (`testdata/tokenizer.json`).
//! - The real prompt-embedding graph (`testdata/token_embedding.onnx`) on ONNX Runtime, with a
//!   stand-in for its weight file: how it frames the prompt, and why decode steps embed speech
//!   tokens themselves (the graph turns speech token 0 into text).
//! - The language-model driver on a stand-in graph with ONNX Runtime's real `GroupQueryAttention`
//!   (in-place cache, CFG batch, stop token, length limit, cancel).
//! - The whole model (speech encoder, embedding, language model, flow, vocoder) on stand-in graphs:
//!   a library voice is prepared, cached in the voice library, and a sentence is rendered.
//! - Both real flow steps' input and output declarations (`testdata/flow_step_*.io.pb`) against
//!   what the driver binds, and the load-time check that refuses another export.
//! - Where `auto` / `dml` run (EP and graphics adapter) on adapter lists of real PCs.
//! - Real-model tests (`#[ignore]`): `SIGIL_APP_DATA_DIR` with `tts/chatterbox-mtl-v3` installed
//!   and `SIGIL_TEST_MODELS_DIR/sherpa-onnx/clips/en.wav` as the reference voice:
//!   `cargo test --manifest-path src-tauri/Cargo.toml local_tts -- --ignored --test-threads=1`.
//!   **Release gate:** `real_model_chatterbox_clones_a_library_voice` with the default
//!   `SIGIL_TEST_CHATTERBOX_EP=cpu` is the only run of the real CPU flow step
//!   (`flow_step_slim.onnx`, which every CPU user and every GPU-to-CPU fallback runs); run it, and
//!   with `cuda` / `dml` on a graphics card, before a release that changes this engine.

use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use ort::value::Tensor;

use super::embed::{read_graph_io, read_initializers, SpeechEmbedding, ValueInfo};
use super::lm::StaticLm;
use super::model::{self, ChatterboxModel, GenOptions, ModelFiles, Placement};
use super::sampling::{next_token, Rng, SamplingParams};
use super::testonnx::{f32_bytes, Attr, Dim, Graph, BOOL, FLOAT, INT32, INT64};
use super::text::{self, Cangjie, Frontend, JaReadings, RuStress};
use super::voice::Conditionals;
use super::*;
use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions, EP_CPU, EP_CUDA, EP_DML};
use crate::services::neural_audio::vc::families::reference::ReferenceEmbedding;

const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/services/local_tts/chatterbox_ort/testdata");

fn testdata(name: &str) -> PathBuf {
    Path::new(TESTDATA).join(name)
}

/// A fresh temporary folder for one test.
fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "sigil-chatterbox-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn real_frontend() -> Frontend {
    Frontend::load(&testdata("tokenizer.json"), None, None, None).expect("the real tokenizer loads")
}

fn token_id(f: &str) -> i64 {
    let t = tokenizers::Tokenizer::from_file(testdata("tokenizer.json")).unwrap();
    i64::from(t.token_to_id(f).unwrap_or_else(|| panic!("no token {f}")))
}

// -------------------------------------------------------------------------------------------------
// Text frontend
// -------------------------------------------------------------------------------------------------

#[test]
fn languages_normalize_to_the_models_tags() {
    assert_eq!(text::normalize_language("pt-BR"), Some("pt"));
    assert_eq!(text::normalize_language("nb"), Some("no"));
    assert_eq!(text::normalize_language("zh-Hant"), Some("zh"));
    assert_eq!(text::normalize_language("yue"), Some("zh"));
    assert_eq!(text::normalize_language("iw"), Some("he"));
    assert_eq!(text::normalize_language("JA"), Some("ja"));
    assert_eq!(text::normalize_language("vi"), None, "not one of the 23 languages");
    assert_eq!(text::normalize_language(""), None);
    assert_eq!(text::LANGUAGES.len(), 23);
}

#[test]
fn the_script_tells_the_language_when_the_tag_cannot() {
    assert_eq!(text::script_language("こんにちは、元気ですか？"), Some("ja"));
    assert_eq!(text::script_language("東京に行きます"), Some("ja"), "kana with kanji");
    assert_eq!(text::script_language("你好，世界"), Some("zh"));
    assert_eq!(text::script_language("안녕하세요"), Some("ko"));
    assert_eq!(text::script_language("Привет, как дела?"), Some("ru"));
    assert_eq!(text::script_language("Γεια σου"), Some("el"));
    assert_eq!(text::script_language("שלום"), Some("he"));
    assert_eq!(text::script_language("Hello there"), None, "Latin needs a tag");
    assert_eq!(text::script_language("1234 !?"), None);
    assert_eq!(text::resolve_language(Some("es"), "Hola"), "es");
    assert_eq!(
        text::resolve_language(Some("vi"), "Привет"),
        "ru",
        "an unsupported tag falls back to the script"
    );
    assert_eq!(text::resolve_language(None, "Hello"), "en");
}

#[test]
fn punctuation_is_normalized_like_the_reference() {
    assert_eq!(text::punc_norm("  hello   world "), "hello world.");
    assert_eq!(text::punc_norm("wait... what: now; “quoted” — ok"), "wait,  what, now,  \"quoted\" - ok.");
    assert_eq!(text::punc_norm("Done!"), "Done!");
    assert_eq!(text::punc_norm("終わり。"), "終わり。");
    assert_eq!(text::punc_norm("trailing -"), "trailing -");
    assert_eq!(text::punc_norm("   "), "");
}

#[test]
fn katakana_readings_become_hiragana_and_kanji_are_read_longest_first() {
    assert_eq!(text::katakana_to_hiragana("トウキョウー"), "とうきょうー");
    let tsv = "東京\tトウキョウ\n東\tヒガシ\n京\tキョウ\n春\tハル\n";
    let ja = JaReadings::parse(tsv);
    assert_eq!(ja.len(), 4);
    assert_eq!(ja.convert("東京に行く"), "とうきょうに行く", "longest match wins; unknown kanji stay");
    assert_eq!(ja.convert("春が来た"), " はるが来た", "a reading starting with は gets a space");
    assert_eq!(ja.convert("ひらがなだけ"), "ひらがなだけ");
    // Readings are stored NFKD (が = か + U+3099).
    let voiced = JaReadings::parse(&format!("学校\t{}\n", text::nfkd("ガッコウ")));
    assert_eq!(voiced.convert("学校"), text::nfkd("がっこう"));
}

#[test]
fn russian_stress_goes_after_the_stressed_vowel() {
    let ru = RuStress::parse("молоко\t5\nмой\t1\nеще\t2\n");
    assert_eq!(ru.len(), 3);
    let marked = ru.apply(&text::nfkd("молоко и мой"));
    assert_eq!(marked, text::nfkd("молоко\u{301} и мо\u{301}й"));
    // ё is looked up as е (and keeps its diaeresis); й spelled и + breve is recomposed for lookup.
    assert_eq!(ru.apply(&text::nfkd("ещё")), format!("{}\u{301}", text::nfkd("ещё")));
    // An unknown word and an already stressed word stay as they are.
    assert_eq!(ru.apply("кот"), "кот");
    assert_eq!(ru.apply("мо\u{301}й"), "мо\u{301}й");
}

#[test]
fn chinese_characters_become_cangjie_tokens() {
    let cj = Cangjie::parse(r#"{"日": "a", "曰": "xa", "㬪": "aaam2", "ab": "zz"}"#).unwrap();
    assert_eq!(cj.len(), 3, "multi-character keys are ignored");
    assert_eq!(cj.encode("日曰 x"), "[cj_a][cj_.][cj_x][cj_a][cj_.] x");
    assert_eq!(cj.encode("㬪"), "[cj_a][cj_a][cj_a][cj_m][cj_2][cj_.]");
    assert!(Cangjie::parse("[1, 2]").is_err());
}

#[test]
fn sentences_are_framed_for_the_embedding_graph() {
    let f = real_frontend();
    let ids = f.encode("Hello world", "en").unwrap().unwrap();
    assert_eq!(&ids[..2], &[text::EXAGGERATION_TOKEN, text::START_TEXT]);
    assert_eq!(&ids[ids.len() - 3..], &[text::STOP_TEXT, text::START_SPEECH, text::START_SPEECH]);
    assert_eq!(ids[2], token_id("[en]"));
    let space = token_id("[SPACE]");
    assert_eq!(ids.iter().filter(|&&i| i == space).count(), 1, "one space between the words");
    let prepared = f.prepare("Hello world", "en").unwrap().unwrap();
    assert_eq!(prepared, "[en]hello[SPACE]world.", "lower case, full stop, spelled spaces");
    // Positions: text from 0 at [START], the emotion slot and both speech starts at 0.
    let pos = text::positions(&ids);
    assert_eq!(pos[0], 0);
    assert_eq!(pos[1], 0);
    assert_eq!(pos[2], 1);
    assert_eq!(pos[ids.len() - 3], (ids.len() - 4) as i64, "[STOP] closes the text");
    assert_eq!(&pos[ids.len() - 2..], &[0, 0]);
    // Nothing to say.
    assert!(f.encode("  ... ", "en").unwrap().is_none());
    assert!(f.encode("", "en").unwrap().is_none());
}

#[test]
fn chinese_needs_its_table_and_uses_cangjie_tokens() {
    let dir = temp_dir("cj");
    let without = real_frontend();
    assert!(without
        .encode("你好", "zh")
        .unwrap_err()
        .contains("Cangjie"));
    let table = dir.join("Cangjie5_TC.json");
    std::fs::write(&table, r#"{"你": "onf", "好": "vnd"}"#).unwrap();
    let f = Frontend::load(&testdata("tokenizer.json"), Some(table), None, None).unwrap();
    let prepared = f.prepare("你好", "zh").unwrap().unwrap();
    assert!(prepared.starts_with("[zh]"), "{prepared}");
    assert!(prepared.contains("[cj_o][cj_n][cj_f][cj_.]"), "{prepared}");
    let ids = f.encode("你好", "zh").unwrap().unwrap();
    assert!(ids.contains(&token_id("[cj_o]")) && ids.contains(&token_id("[cj_.]")));
    assert!(ids.contains(&token_id("[zh]")));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn japanese_and_russian_use_their_tables_and_survive_without_them() {
    let dir = temp_dir("jaru");
    let ja = dir.join("japanese_readings.tsv");
    std::fs::write(&ja, "東京\tトウキョウ\n").unwrap();
    let ru = dir.join("russian_stress.tsv");
    std::fs::write(&ru, "мой\t1\n").unwrap();
    let f = Frontend::load(&testdata("tokenizer.json"), None, Some(ja), Some(ru)).unwrap();
    let p = f.prepare("東京へ", "ja").unwrap().unwrap();
    assert_eq!(p, format!("[ja]{}", text::nfkd("とうきょうへ.")), "kanji read, NFKD");
    let p = f.prepare("Мой дом", "ru").unwrap().unwrap();
    assert!(p.contains('\u{301}'), "{p}");
    // Missing tables: the text passes through (as the reference does without pykakasi).
    let bare = real_frontend();
    assert!(bare.prepare("東京", "ja").unwrap().unwrap().contains('東'));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_length_limit_follows_the_text() {
    assert_eq!(text::max_speech_tokens(""), 75, "three seconds at 25 tokens per second");
    let short = text::max_speech_tokens("Hi");
    let long = text::max_speech_tokens(&"word ".repeat(40));
    assert!(short < long);
    assert!(
        text::max_speech_tokens("今日は") > text::max_speech_tokens("abc"),
        "CJK characters count more"
    );
    assert_eq!(text::max_speech_tokens(&"a".repeat(10_000)), 1000);
}

// -------------------------------------------------------------------------------------------------
// The real prompt-embedding graph
// -------------------------------------------------------------------------------------------------

/// Offsets of the real graph's tables in `token_embedding.onnx.data` (from its initializers).
struct RealTables {
    emotion: u64,
    text: u64,
    text_pos: u64,
    speech: u64,
    speech_pos: u64,
}

const HID: usize = 1024;

/// A value for row `row` of table `t`, column `c` (distinct per table, row and column).
fn val(t: u32, row: usize, c: usize) -> f32 {
    t as f32 * 1000.0 + row as f32 + c as f32 * 1e-3
}

fn write_row(f: &mut std::fs::File, base: u64, row: usize, t: u32) {
    let data: Vec<f32> = (0..HID).map(|c| val(t, row, c)).collect();
    f.seek(SeekFrom::Start(base + (row * HID * 4) as u64))
        .unwrap();
    f.write_all(&f32_bytes(&data)).unwrap();
}

/// Copy the real graph into `dir` with a stand-in weight file: zeros except the rows written.
fn real_embedding_graph(dir: &Path, text_rows: &[usize], speech_rows: &[usize], pos_rows: &[usize]) -> (PathBuf, RealTables) {
    let graph = dir.join("token_embedding.onnx");
    std::fs::copy(testdata("token_embedding.onnx"), &graph).unwrap();
    let inits = read_initializers(&std::fs::read(&graph).unwrap()).unwrap();
    let offset = |name: &str| -> u64 {
        let i = inits
            .iter()
            .find(|i| i.name == name)
            .unwrap_or_else(|| panic!("no {name}"));
        assert!(i.external);
        i.external_data
            .iter()
            .find(|(k, _)| k == "offset")
            .unwrap()
            .1
            .parse()
            .unwrap()
    };
    let t = RealTables {
        emotion: offset("embed/onnx::MatMul_81"),
        text: offset("embed/text.weight"),
        text_pos: offset("embed/text_position.weight"),
        speech: offset("embed/speech.weight"),
        speech_pos: offset("embed/speech_position.weight"),
    };
    let mut f = std::fs::File::create(dir.join("token_embedding.onnx.data")).unwrap();
    f.set_len(68_808_704).unwrap();
    write_row(&mut f, t.emotion, 0, 9);
    for &r in text_rows {
        write_row(&mut f, t.text, r, 1);
    }
    for &r in pos_rows {
        write_row(&mut f, t.text_pos, r, 2);
        write_row(&mut f, t.speech_pos, r, 4);
    }
    for &r in speech_rows {
        write_row(&mut f, t.speech, r, 3);
    }
    drop(f);
    (graph, t)
}

#[test]
fn the_real_graphs_tables_are_found_by_name() {
    let inits = read_initializers(&std::fs::read(testdata("token_embedding.onnx")).unwrap()).unwrap();
    let speech = inits
        .iter()
        .find(|i| i.name == "embed/speech.weight")
        .unwrap();
    assert_eq!(speech.dims, [8194, 1024]);
    assert_eq!(speech.data_type, 1);
    assert!(speech.external);
    let pos = inits
        .iter()
        .find(|i| i.name == "embed/speech_position.weight")
        .unwrap();
    assert_eq!(pos.dims, [4100, 1024]);
    assert_eq!(pos.elements(), Some(4100 * 1024));
}

#[test]
fn the_real_embedding_graph_frames_the_prompt_like_the_reference() {
    let dir = temp_dir("realemb");
    let f = real_frontend();
    let ids = f.encode("hi", "en").unwrap().unwrap();
    let text_ids: Vec<usize> = ids
        .iter()
        .filter(|&&i| i < text::START_SPEECH)
        .map(|&i| i as usize)
        .collect();
    let (graph, _) = real_embedding_graph(&dir, &text_ids, &[0, 1, 6561], &(0..8).collect::<Vec<_>>());
    let (mut session, _) = build_session(None, EP_CPU, &VcSessionOptions::default(), ModelSource::File(&graph)).unwrap();
    let len = ids.len();
    let pos = text::positions(&ids);
    let cond: Vec<f32> = (0..2 * 2 * HID).map(|i| -(i as f32)).collect();
    let tile = |v: &[i64]| -> Vec<i64> { v.iter().chain(v.iter()).copied().collect() };
    let mut out = session
        .run(ort::inputs![
            "input_ids" => Tensor::<i64>::from_array(([2usize, len], tile(&ids))).unwrap(),
            "token_position_ids" => Tensor::<i64>::from_array(([2usize, len], tile(&pos))).unwrap(),
            "exaggeration" => Tensor::<f32>::from_array(([2usize], vec![0.5f32, 0.5])).unwrap(),
            "text_conditioning" => Tensor::<f32>::from_array(([2usize], vec![1.0f32, 0.0])).unwrap(),
            "conditioning" => Tensor::<f32>::from_array(([2usize, 2, HID], cond.clone())).unwrap()
        ])
        .unwrap();
    let v = out.remove("inputs_embeds").unwrap();
    let (shape, e) = v.try_extract_tensor::<f32>().unwrap();
    assert_eq!(shape.to_vec(), vec![2, (2 + len) as i64, HID as i64]);
    let at = |row: usize, p: usize, c: usize| e[(row * (2 + len) + p) * HID + c];
    for c in [0usize, 7, 1023] {
        // The conditioning comes first, unchanged.
        assert_eq!(at(0, 0, c), cond[c]);
        assert_eq!(at(1, 1, c), cond[(2 + 1) * HID + c]);
        // The emotion slot: the projection times the exaggeration.
        assert!((at(0, 2, c) - 0.5 * val(9, 0, c)).abs() < 1e-3);
        // Text tokens (row 0): token row + position row.
        for k in 1..len - 2 {
            let p = 2 + k;
            let want = val(1, ids[k] as usize, c) + val(2, pos[k] as usize, c);
            assert!((at(0, p, c) - want).abs() < 1e-2, "row 0 position {p}");
            // Row 1 (classifier-free guidance): the position only.
            assert!((at(1, p, c) - val(2, pos[k] as usize, c)).abs() < 1e-2, "row 1 position {p}");
        }
        // Both speech starts: speech row 6561 + speech position 0.
        for p in [2 + len - 2, 2 + len - 1] {
            let want = val(3, 6561, c) + val(4, 0, c);
            assert!((at(0, p, c) - want).abs() < 1e-2);
            assert!((at(1, p, c) - want).abs() < 1e-2);
        }
    }
    drop(out);

    // Decode steps: the graph embeds speech token 1 correctly, but token 0 as text...
    let speech = SpeechEmbedding::read(&graph, 8).unwrap();
    let mut single = |token: i64, position: i64| -> Vec<f32> {
        let mut out = session
            .run(ort::inputs![
                "input_ids" => Tensor::<i64>::from_array(([1usize, 1], vec![token])).unwrap(),
                "token_position_ids" => Tensor::<i64>::from_array(([1usize, 1], vec![position])).unwrap(),
                "exaggeration" => Tensor::<f32>::from_array(([1usize], vec![0.5f32])).unwrap(),
                "text_conditioning" => Tensor::<f32>::from_array(([1usize], vec![1.0f32])).unwrap(),
                "conditioning" => Tensor::<f32>::from_array(([1usize, 0, HID], Vec::<f32>::new())).unwrap()
            ])
            .unwrap();
        out.remove("inputs_embeds")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap()
            .1
            .to_vec()
    };
    let mut mine = vec![0f32; HID];
    speech.embed(1, 1, &mut mine).unwrap();
    let graph_one = single(1, 1);
    assert!(
        graph_one
            .iter()
            .zip(&mine)
            .all(|(a, b)| (a - b).abs() < 1e-3),
        "speech token 1 agrees"
    );
    let graph_zero = single(0, 1);
    speech.embed(0, 1, &mut mine).unwrap();
    assert!(
        (mine[0] - (val(3, 0, 0) + val(4, 1, 0))).abs() < 1e-3,
        "speech token 0 from the speech table"
    );
    assert!(
        (graph_zero[0] - (val(1, 0, 0) + val(2, 1, 0))).abs() < 1e-3,
        "the graph reads a lone 0 as the text stop token, so decode steps must not use it"
    );
    drop(session);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn unsafe_external_data_paths_are_refused() {
    let dir = temp_dir("unsafe");
    let mut g = Graph::new();
    g.input("input_ids", INT64, &[Dim::N(1), Dim::N(1)])
        .external_f32("embed/speech.weight", &[2, 2], "../outside.bin", 0)
        .external_f32("embed/speech_position.weight", &[2, 2], "pos.bin", 0)
        .node("Gather", &["embed/speech.weight", "input_ids"], &["y"], &[])
        .output("y", FLOAT, &[Dim::N(1), Dim::N(1), Dim::N(2)]);
    let path = dir.join("token_embedding.onnx");
    g.save(&path, 17, &[]);
    let err = SpeechEmbedding::read(&path, 2).unwrap_err();
    assert!(err.contains("unsafe"), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}

// -------------------------------------------------------------------------------------------------
// Sampling
// -------------------------------------------------------------------------------------------------

#[test]
fn guidance_penalty_and_greedy_pick_like_the_reference() {
    let vocab = 4;
    // cond, uncond
    let logits = [1.0, 2.0, 0.0, -1.0, 1.0, 3.0, 0.0, -3.0];
    let mut scratch = Vec::new();
    let mut rng = Rng::new(1);
    let greedy = SamplingParams {
        cfg_weight: 0.5,
        temperature: 0.0,
        repetition_penalty: 1.0,
        min_p: 0.0,
        top_p: 1.0,
    };
    // CFG: 2 + 0.5 (2 - 3) = 1.5 for token 1, 1 + 0 = 1 for token 0 → token 1.
    assert_eq!(next_token(&logits, 2, vocab, &[], &greedy, &mut rng, &mut scratch).unwrap(), 1);
    // Without the unconditional row the same.
    assert_eq!(next_token(&logits, 1, vocab, &[], &greedy, &mut rng, &mut scratch).unwrap(), 1);
    // A strong repetition penalty on token 1 hands the pick to token 0 (1.5 / 3 < 1).
    let penalized = SamplingParams {
        repetition_penalty: 3.0,
        ..greedy
    };
    assert_eq!(next_token(&logits, 2, vocab, &[1, 1], &penalized, &mut rng, &mut scratch).unwrap(), 0);
    // Negative logits are multiplied: token 3 (-0.5 → -1.5) loses its lead to token 0 (-1).
    let all_negative = [-1.0f32, -2.0, -3.0, -0.5];
    assert_eq!(next_token(&all_negative, 1, 4, &[3], &penalized, &mut rng, &mut scratch).unwrap(), 0);
    assert!(next_token(&logits[..3], 1, vocab, &[], &greedy, &mut rng, &mut scratch).is_err());
}

#[test]
fn min_p_and_top_p_prune_and_sampling_follows_the_probabilities() {
    let mut scratch = Vec::new();
    let logits = [2.0f32, 2.0, 0.0, -10.0];
    let params = SamplingParams {
        cfg_weight: 0.0,
        temperature: 1.0,
        repetition_penalty: 1.0,
        min_p: 0.5,
        top_p: 1.0,
    };
    // Tokens 0 and 1 are equally likely; 2 and 3 fall under half the top probability.
    let mut rng = Rng::new(42);
    let mut counts = [0usize; 4];
    for _ in 0..4000 {
        counts[next_token(&logits, 1, 4, &[], &params, &mut rng, &mut scratch).unwrap() as usize] += 1;
    }
    assert_eq!(counts[2] + counts[3], 0, "{counts:?}");
    assert!((1700..2300).contains(&counts[0]), "{counts:?}");
    // Top-p 0.4 keeps only the single most likely token of a peaked distribution.
    let peaked = [3.0f32, 1.0, 0.0, -1.0];
    let top = SamplingParams {
        min_p: 0.0,
        top_p: 0.4,
        ..params
    };
    for _ in 0..200 {
        assert_eq!(next_token(&peaked, 1, 4, &[], &top, &mut rng, &mut scratch).unwrap(), 0);
    }
    // The same seed gives the same draws.
    let draw = |seed| {
        let mut r = Rng::new(seed);
        (0..20)
            .map(|_| next_token(&logits, 1, 4, &[], &SamplingParams { min_p: 0.0, ..params }, &mut r, &mut Vec::new()).unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(draw(7), draw(7));
}

// -------------------------------------------------------------------------------------------------
// Stand-in graphs (H = 8: ONNX Runtime's attention wants head sizes in multiples of 8)
// -------------------------------------------------------------------------------------------------

const H: u64 = 8;
const VOCAB: u64 = 8194;
/// Positions in the stand-in embedding table.
const POSITIONS: u64 = model::MAX_POSITIONS as u64 + 1;

fn w(n: usize, scale: f32, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 1000) as f32 / 1000.0 * scale - scale / 2.0)
        .collect()
}

/// A stand-in language model: `layers` GQA layers (1 head of 8) and logits that follow `script`
/// (`script[total - 1]` is the next token), plus a little of the attention output.
fn standin_lm(path: &Path, layers: usize, script: &[i64]) {
    let mut g = Graph::new();
    g.input("inputs_embeds", FLOAT, &[Dim::S("batch"), Dim::S("sequence"), Dim::N(H)])
        .input("seqlens_k", INT32, &[Dim::S("batch")])
        .input("total_sequence_length", INT32, &[]);
    for l in 0..layers {
        let names = [format!("past_key_values.{l}.key"), format!("past_key_values.{l}.value")];
        for n in &names {
            g.input(n, FLOAT, &[Dim::S("batch"), Dim::N(1), Dim::S("cache_capacity"), Dim::N(H)]);
        }
        let (wq, wk, wv) = (format!("wq{l}"), format!("wk{l}"), format!("wv{l}"));
        let hh = (H * H) as usize;
        g.f32s(&wq, &[H, H], &w(hh, 1.0, 1 + l as u32))
            .f32s(&wk, &[H, H], &w(hh, 1.0, 7 + l as u32))
            .f32s(&wv, &[H, H], &w(hh, 1.0, 13 + l as u32));
        let (q, k, v) = (format!("q{l}"), format!("k{l}"), format!("v{l}"));
        g.node("MatMul", &["inputs_embeds", &wq], &[&q], &[])
            .node("MatMul", &["inputs_embeds", &wk], &[&k], &[])
            .node("MatMul", &["inputs_embeds", &wv], &[&v], &[]);
        let (attn, pk, pv) = (format!("attn{l}"), format!("present.{l}.key"), format!("present.{l}.value"));
        g.node_in(
            "com.microsoft",
            "GroupQueryAttention",
            &[&q, &k, &v, &names[0], &names[1], "seqlens_k", "total_sequence_length"],
            &[&attn, &pk, &pv],
            &[Attr::Int("num_heads", 1), Attr::Int("kv_num_heads", 1)],
        );
        g.output(&pk, FLOAT, &[Dim::S("batch"), Dim::N(1), Dim::S("cache_capacity"), Dim::N(H)])
            .output(&pv, FLOAT, &[Dim::S("batch"), Dim::N(1), Dim::S("cache_capacity"), Dim::N(H)]);
    }
    g.i64s("last_index", &[], &[-1])
        .f32s("wo", &[H, VOCAB], &w((H * VOCAB) as usize, 0.01, 99))
        .i64s("script", &[script.len() as u64], script)
        .i64s("depth", &[], &[VOCAB as i64])
        .f32s("onehot_values", &[2], &[-50.0, 50.0])
        .node("Gather", &["attn0", "last_index"], &["last"], &[Attr::Int("axis", 1)])
        .node("MatMul", &["last", "wo"], &["noise"], &[])
        .node("Gather", &["script", "seqlens_k"], &["next"], &[Attr::Int("axis", 0)])
        .node("OneHot", &["next", "depth", "onehot_values"], &["onehot"], &[Attr::Int("axis", -1)])
        .node("Add", &["onehot", "noise"], &["logits"], &[])
        .output("logits", FLOAT, &[Dim::S("batch"), Dim::N(VOCAB)]);
    g.save(path, 17, &[("com.microsoft", 1)]);
}

/// A stand-in embedding graph with the real one's I/O: speech table rows + position rows, after
/// the conditioning (tables stored as external data, like the real graph).
fn standin_token_embedding(path: &Path) {
    let speech = w((VOCAB * H) as usize, 2.0, 5);
    let pos = w((POSITIONS * H) as usize, 0.5, 6);
    let mut g = Graph::new();
    g.input("input_ids", INT64, &[Dim::S("batch"), Dim::S("token_sequence")])
        .input("token_position_ids", INT64, &[Dim::S("batch"), Dim::S("token_sequence")])
        .input("exaggeration", FLOAT, &[Dim::S("batch")])
        .input("text_conditioning", FLOAT, &[Dim::S("batch")])
        .input("conditioning", FLOAT, &[Dim::S("batch"), Dim::S("conditioning_sequence"), Dim::N(H)])
        .external_f32("embed/speech.weight", &[VOCAB, H], "token_embedding.onnx.data", 0)
        .external_f32(
            "embed/speech_position.weight",
            &[POSITIONS, H],
            "token_embedding.onnx.data",
            VOCAB * H * 4,
        )
        .node("Gather", &["embed/speech.weight", "input_ids"], &["e1"], &[Attr::Int("axis", 0)])
        .node(
            "Gather",
            &["embed/speech_position.weight", "token_position_ids"],
            &["e2"],
            &[Attr::Int("axis", 0)],
        )
        .node("Add", &["e1", "e2"], &["tok"], &[])
        .node("Concat", &["conditioning", "tok"], &["inputs_embeds"], &[Attr::Int("axis", 1)])
        .output("inputs_embeds", FLOAT, &[Dim::S("batch"), Dim::S("sequence"), Dim::N(H)]);
    g.save(path, 17, &[]);
    let mut data = f32_bytes(&speech);
    data.extend(f32_bytes(&pos));
    std::fs::write(path.with_file_name("token_embedding.onnx.data"), data).unwrap();
}

/// Prompt tokens, speaker vector and prompt mel the stand-in speech encoder returns.
const PROMPT_TOKENS: [i64; 3] = [11, 12, 13];
const PROMPT_FRAMES: u64 = 4;

fn standin_speech_encoder(path: &Path) {
    let mut g = Graph::new();
    g.input("audio_values", FLOAT, &[Dim::N(1), Dim::S("samples")])
        .f32s("feat", &[1, 2, H], &w(2 * H as usize, 1.0, 3))
        .i64s("tok", &[1, 3], &PROMPT_TOKENS)
        .f32s("spk", &[1, 192], &w(192, 1.0, 4))
        .f32s("pf", &[1, PROMPT_FRAMES, 80], &w((PROMPT_FRAMES * 80) as usize, 1.0, 8))
        .node("Identity", &["feat"], &["audio_features"], &[])
        .node("Identity", &["tok"], &["audio_tokens"], &[])
        .node("Identity", &["spk"], &["speaker_embeddings"], &[])
        .node("Identity", &["pf"], &["speaker_features"], &[])
        .output("audio_features", FLOAT, &[Dim::N(1), Dim::S("s"), Dim::N(H)])
        .output("audio_tokens", INT64, &[Dim::N(1), Dim::S("tokens")])
        .output("speaker_embeddings", FLOAT, &[Dim::N(1), Dim::N(192)])
        .output("speaker_features", FLOAT, &[Dim::N(1), Dim::S("frames"), Dim::N(80)]);
    g.save(path, 17, &[]);
}

/// Flow stand-ins: `prepare` makes `x` zeros of 2 frames per token, `step` adds
/// `delta · (1 + guidance)`, the vocoder flattens the frames after the prompt.
fn standin_flow(dir: &Path) {
    let mut p = Graph::new();
    p.input("speech_tokens", INT64, &[Dim::N(1), Dim::S("tokens")])
        .input("speaker_embeddings", FLOAT, &[Dim::N(1), Dim::N(192)])
        .input("speaker_features", FLOAT, &[Dim::N(1), Dim::S("frames"), Dim::N(80)])
        .i64s("one", &[], &[1])
        .i64s("two", &[], &[2])
        .i64s("axes0", &[1], &[0])
        .i64s("c180", &[2], &[1, 80])
        .i64s("c11", &[2], &[1, 1])
        .i64s("s0", &[1], &[0])
        .i64s("s80", &[1], &[80])
        .i64s("a1", &[1], &[1])
        .node("Shape", &["speech_tokens"], &["tshape"], &[])
        .node("Gather", &["tshape", "one"], &["n"], &[Attr::Int("axis", 0)])
        .node("Mul", &["n", "two"], &["m"], &[])
        .node("Unsqueeze", &["m", "axes0"], &["m1"], &[])
        .node("Concat", &["c180", "m1"], &["xshape"], &[Attr::Int("axis", 0)])
        .node("Concat", &["c11", "m1"], &["mshape"], &[Attr::Int("axis", 0)])
        .node("ConstantOfShape", &["xshape"], &["x"], &[Attr::FloatTensor("value", vec![1], vec![0.0])])
        .node(
            "ConstantOfShape",
            &["mshape"],
            &["maskf"],
            &[Attr::FloatTensor("value", vec![1], vec![1.0])],
        )
        .node("Cast", &["maskf"], &["mask"], &[Attr::Int("to", BOOL as i64)])
        .node("Identity", &["x"], &["mu"], &[])
        .node("Identity", &["x"], &["cond"], &[])
        .node("Slice", &["speaker_embeddings", "s0", "s80", "a1"], &["speakers"], &[])
        .node("Shape", &["speaker_features"], &["fshape"], &[])
        .node("Gather", &["fshape", "one"], &["prefix"], &[Attr::Int("axis", 0)])
        .output("x", FLOAT, &[Dim::N(1), Dim::N(80), Dim::S("flow_frames")])
        .output("mask", BOOL, &[Dim::N(1), Dim::N(1), Dim::S("flow_frames")])
        .output("mu", FLOAT, &[Dim::N(1), Dim::N(80), Dim::S("flow_frames")])
        .output("speakers", FLOAT, &[Dim::N(1), Dim::N(80)])
        .output("cond", FLOAT, &[Dim::N(1), Dim::N(80), Dim::S("flow_frames")])
        .output("prefix", INT64, &[]);
    p.save(&dir.join("onnx/flow_prepare_slim.onnx"), 17, &[]);

    let mut s = Graph::new();
    s.input("x", FLOAT, &[Dim::S("batch"), Dim::N(80), Dim::S("flow_frames")])
        .input("mask", BOOL, &[Dim::S("batch"), Dim::N(1), Dim::S("flow_frames")])
        .input("mu", FLOAT, &[Dim::S("batch"), Dim::N(80), Dim::S("flow_frames")])
        .input("speakers", FLOAT, &[Dim::S("batch"), Dim::N(80)])
        .input("cond", FLOAT, &[Dim::S("batch"), Dim::N(80), Dim::S("flow_frames")])
        .input("time", FLOAT, &[Dim::N(1)])
        .input("delta", FLOAT, &[Dim::N(1)])
        .input("guidance", FLOAT, &[Dim::N(1)])
        .f32s("one_f", &[1], &[1.0])
        .node("Add", &["guidance", "one_f"], &["g1"], &[])
        .node("Mul", &["delta", "g1"], &["d"], &[])
        .node("Add", &["x", "d"], &["next_x"], &[])
        .output("next_x", FLOAT, &[Dim::S("batch"), Dim::N(80), Dim::S("flow_frames")]);
    s.save(&dir.join("onnx/flow_step_slim.onnx"), 17, &[]);

    let mut v = Graph::new();
    v.input("x", FLOAT, &[Dim::S("batch"), Dim::N(80), Dim::S("flow_frames")])
        .input("prefix", INT64, &[Dim::N(1)])
        .i64s("end", &[1], &[1 << 40])
        .i64s("ax2", &[1], &[2])
        .i64s("flat", &[2], &[1, -1])
        .node("Slice", &["x", "prefix", "end", "ax2"], &["kept"], &[])
        .node("Reshape", &["kept", "flat"], &["waveform"], &[])
        .output("waveform", FLOAT, &[Dim::S("batch"), Dim::S("samples")]);
    v.save(&dir.join("onnx/vocoder_prefix_sum.onnx"), 17, &[]);
}

/// A complete stand-in model folder (the real tokenizer) whose language model says `script`.
fn standin_model(dir: &Path, script: &[i64]) {
    standin_speech_encoder(&dir.join("onnx/speech_encoder_slim.onnx"));
    standin_token_embedding(&dir.join("onnx/token_embedding.onnx"));
    standin_lm(&dir.join("onnx/static_language_model_int8.onnx"), 2, script);
    standin_flow(dir);
    std::fs::copy(testdata("tokenizer.json"), dir.join("tokenizer.json")).unwrap();
}

/// A script whose tokens start at the prefill's last position (`prompt_len - 1`).
fn script_from(prompt_len: usize, tokens: &[i64], total: usize) -> Vec<i64> {
    let mut s = vec![5i64; total];
    for (k, &t) in tokens.iter().enumerate() {
        s[prompt_len - 1 + k] = t;
    }
    s
}

fn cpu_lm(path: &Path) -> StaticLm {
    let (session, outcome) = build_session(None, EP_CPU, &VcSessionOptions::default(), ModelSource::File(path)).unwrap();
    StaticLm::new(session, &outcome.active, 0).unwrap()
}

fn prefill(batch: usize, len: usize) -> ort::value::DynValue {
    Tensor::<f32>::from_array(([batch, len, H as usize], w(batch * len * H as usize, 1.0, 21)))
        .unwrap()
        .into_dyn()
}

fn greedy() -> SamplingParams {
    SamplingParams {
        cfg_weight: 0.5,
        temperature: 0.8,
        repetition_penalty: 2.0,
        min_p: 0.05,
        top_p: 1.0,
    }
}

// -------------------------------------------------------------------------------------------------
// Language model driver
// -------------------------------------------------------------------------------------------------

#[test]
fn the_language_model_follows_its_logits_and_fills_the_cache_in_place() {
    let dir = temp_dir("lm");
    let graph = dir.join("onnx/static_language_model_int8.onnx");
    let embed_graph = dir.join("onnx/token_embedding.onnx");
    standin_token_embedding(&embed_graph);
    let embed = SpeechEmbedding::read(&embed_graph, model::MAX_POSITIONS + 1).unwrap();
    let prompt_len = 6;
    let wanted = [100i64, 0, 2500, 6560];
    let mut script = script_from(prompt_len, &wanted, 64);
    script[prompt_len - 1 + wanted.len()] = text::STOP_SPEECH;
    standin_lm(&graph, 2, &script);
    let mut lm = cpu_lm(&graph);
    assert_eq!(lm.ep(), EP_CPU);
    assert_eq!(lm.hidden(), H as usize);
    let input = prefill(2, prompt_len);
    let never = AtomicBool::new(false);
    let mut rng = Rng::new(3);
    let out = lm
        .generate(&input, 2, prompt_len, 40, &embed, &greedy(), &mut rng, &never)
        .unwrap()
        .unwrap();
    assert_eq!(out.tokens, wanted, "speech token 0 included");
    assert!(out.stopped);

    // The cache holds the keys of every position in place: the prompt's (x · Wk) and the decode
    // steps' (the speech embedding of each token at positions 1, 2, ...).
    let (shape, keys) = lm.cache(0, true).unwrap();
    assert_eq!(shape[0], 2);
    assert_eq!(shape[1], 1);
    assert!(shape[2] >= prompt_len + 40, "capacity {shape:?}");
    let cap = shape[2];
    let h = H as usize;
    let wk = w(h * h, 1.0, 7);
    let project = |x: &[f32]| -> Vec<f32> {
        (0..h)
            .map(|j| (0..h).map(|i| x[i] * wk[i * h + j]).sum())
            .collect()
    };
    let (_, xs) = input.try_extract_tensor::<f32>().unwrap();
    for row in 0..2 {
        for p in 0..prompt_len {
            let want = project(&xs[(row * prompt_len + p) * h..(row * prompt_len + p + 1) * h]);
            let got = &keys[(row * cap + p) * h..(row * cap + p + 1) * h];
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() < 1e-4, "row {row} position {p}: {got:?} vs {want:?}");
            }
        }
        // Decode steps: every sampled token is fed back at positions 1, 2, ... (the stop token
        // that followed the last one is not).
        for (i, &t) in wanted.iter().enumerate() {
            let mut e = vec![0f32; h];
            embed.embed(t, i + 1, &mut e).unwrap();
            let want = project(&e);
            let p = prompt_len + i;
            let got = &keys[(row * cap + p) * h..(row * cap + p + 1) * h];
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() < 1e-4, "row {row} step {i}: {got:?} vs {want:?}");
            }
        }
    }

    // The next sentence reuses the cache (same batch, it fits) and starts from position 0 again.
    let ptr = lm.cache_ptr(1, false).unwrap();
    let out = lm
        .generate(&input, 2, prompt_len, 40, &embed, &greedy(), &mut rng, &never)
        .unwrap()
        .unwrap();
    assert_eq!(out.tokens, wanted);
    assert_eq!(lm.cache_ptr(1, false), Some(ptr), "the cache was reused");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_language_model_stops_at_its_limit_runs_without_guidance_and_cancels() {
    let dir = temp_dir("lm2");
    let graph = dir.join("onnx/static_language_model_int8.onnx");
    let embed_graph = dir.join("onnx/token_embedding.onnx");
    standin_token_embedding(&embed_graph);
    let embed = SpeechEmbedding::read(&embed_graph, model::MAX_POSITIONS + 1).unwrap();
    let prompt_len = 3;
    // Never a stop token: the length limit ends it.
    let script: Vec<i64> = (0..80).map(|i| 1000 + i).collect();
    standin_lm(&graph, 1, &script);
    let mut lm = cpu_lm(&graph);
    let never = AtomicBool::new(false);
    let mut rng = Rng::new(9);
    let one_row = SamplingParams { cfg_weight: 0.0, ..greedy() };
    let out = lm
        .generate(&prefill(1, prompt_len), 1, prompt_len, 7, &embed, &one_row, &mut rng, &never)
        .unwrap()
        .unwrap();
    assert_eq!(out.tokens.len(), 7);
    assert!(!out.stopped);
    assert_eq!(
        out.tokens,
        (0..7)
            .map(|k| script[prompt_len - 1 + k])
            .collect::<Vec<_>>()
    );
    assert_eq!(lm.cache(0, true).unwrap().0[0], 1, "one row without guidance");
    let cancelled = AtomicBool::new(true);
    assert!(lm
        .generate(&prefill(1, prompt_len), 1, prompt_len, 7, &embed, &one_row, &mut rng, &cancelled)
        .unwrap()
        .is_none());
    assert!(
        lm.generate(&prefill(3, prompt_len), 3, prompt_len, 7, &embed, &one_row, &mut rng, &never)
            .is_err(),
        "at most two rows"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_graph_without_a_static_cache_is_refused() {
    let dir = temp_dir("notlm");
    let mut g = Graph::new();
    g.input("inputs_embeds", FLOAT, &[Dim::N(1), Dim::S("s"), Dim::N(H)])
        .node("Identity", &["inputs_embeds"], &["logits"], &[])
        .output("logits", FLOAT, &[Dim::N(1), Dim::S("s"), Dim::N(H)]);
    let path = dir.join("lm.onnx");
    g.save(&path, 17, &[]);
    let (session, _) = build_session(None, EP_CPU, &VcSessionOptions::default(), ModelSource::File(&path)).unwrap();
    let err = StaticLm::new(session, EP_CPU, 0).err().unwrap();
    assert!(err.contains("seqlens_k"), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}

// -------------------------------------------------------------------------------------------------
// The whole model on stand-ins
// -------------------------------------------------------------------------------------------------

/// A library voice in `app_data` (a 2 s tone at 24 kHz).
fn library_voice(app_data: &Path, id: &str) {
    let dir = app_data.join("voices").join(id);
    std::fs::create_dir_all(&dir).unwrap();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let path = dir.join("ref.wav");
    let mut wr = hound::WavWriter::create(&path, spec).unwrap();
    for i in 0..48_000 {
        let s = (i as f32 * 0.05).sin() * 8000.0;
        wr.write_sample(s as i16).unwrap();
    }
    wr.finalize().unwrap();
    let sha = crate::services::neural_audio::vc::families::reference::sha256_file(&path).unwrap();
    let meta = serde_json::json!({
        "id": id, "name": "Test voice", "createdUnixMs": 1, "durationMs": 2000, "sampleRate": 24000,
        "lufs": -20.0, "peakDbfs": -6.0, "clipped": false, "source": "imported", "sha256": sha, "cachedBackends": []
    });
    std::fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&meta).unwrap()).unwrap();
}

fn cpu_options() -> GenOptions {
    GenOptions {
        sampling: greedy(),
        seed: Some(1),
        ..GenOptions::default()
    }
}

#[test]
fn the_model_prepares_a_library_voice_caches_it_and_renders_a_sentence() {
    let dir = temp_dir("model");
    let app = dir.join("app");
    let model_dir = dir.join("model");
    library_voice(&app, "test-voice");
    let f = real_frontend();
    let ids = f.encode("Hello there.", "en").unwrap().unwrap();
    // Prompt: 2 conditioning rows + the framed text.
    let prompt_len = 2 + ids.len();
    let wanted = [40i64, 41, 6561, 42, 43];
    let mut script = script_from(prompt_len, &wanted, 200);
    script[prompt_len - 1 + wanted.len()] = text::STOP_SPEECH;
    standin_model(&model_dir, &script);
    let files = ModelFiles::discover(&model_dir, None).unwrap();
    assert!(files.flow_step_fp16.is_none() && files.cangjie.is_none());
    let mut m = ChatterboxModel::load(files.clone(), Placement::CPU, 2).unwrap();
    assert!(!m.on_gpu());
    assert_eq!(m.lm_ep(), EP_CPU);

    let cond = m.conditionals(&app, "test-voice").unwrap();
    assert_eq!(cond.tokens, PROMPT_TOKENS);
    assert_eq!(cond.prompt_frames as u64, PROMPT_FRAMES);
    let cache = crate::services::voice_library::read_cache(&app, "test-voice", CACHE_BACKEND).expect("cached in the library");
    let e = ReferenceEmbedding::from_bytes(&cache).unwrap();
    assert_eq!(e.backend, CACHE_BACKEND);
    assert_eq!(e.voice_id, "test-voice");

    let never = AtomicBool::new(false);
    // Without a seed the flow starts from the graph's own noise (zeros in the stand-in).
    let unseeded = GenOptions {
        seed: None,
        ..cpu_options()
    };
    let audio = m
        .synthesize("Hello there.", "en", &cond, &unseeded, &never)
        .unwrap()
        .unwrap();
    // Tokens kept: 4 (6561 is not audio); flow frames 2 per token of prompt + speech, minus the
    // prompt mel; 80 values per frame from the stand-in vocoder, each 1 + guidance after a full
    // integration.
    let kept = 4;
    let frames = 2 * (PROMPT_TOKENS.len() + kept) - PROMPT_FRAMES as usize;
    assert_eq!(audio.len(), 80 * frames);
    assert!(
        audio
            .iter()
            .all(|&v| (v - (1.0 + model::FLOW_GUIDANCE)).abs() < 1e-4),
        "{:?}",
        &audio[..4]
    );
    // With a seed the starting noise is drawn in Rust (standard normal): the same seed gives the
    // same waveform, another seed another one.
    let seeded = m
        .synthesize("Hello there.", "en", &cond, &cpu_options(), &never)
        .unwrap()
        .unwrap();
    assert_eq!(seeded.len(), audio.len());
    let again = m
        .synthesize("Hello there.", "en", &cond, &cpu_options(), &never)
        .unwrap()
        .unwrap();
    assert_eq!(seeded, again, "a seed reproduces the waveform");
    let other = m
        .synthesize("Hello there.", "en", &cond, &GenOptions { seed: Some(2), ..cpu_options() }, &never)
        .unwrap()
        .unwrap();
    assert_ne!(seeded, other);
    let noise: Vec<f64> = seeded
        .iter()
        .map(|&v| f64::from(v - (1.0 + model::FLOW_GUIDANCE)))
        .collect();
    let mean = noise.iter().sum::<f64>() / noise.len() as f64;
    let sd = (noise.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / noise.len() as f64).sqrt();
    assert!(mean.abs() < 0.2 && (0.8..1.2).contains(&sd), "mean {mean}, sd {sd}");
    // Nothing to say: empty audio, no error.
    assert!(m
        .synthesize(" ... ", "en", &cond, &cpu_options(), &never)
        .unwrap()
        .unwrap()
        .is_empty());
    // Cancelled: None.
    let cancelled = AtomicBool::new(true);
    assert!(m
        .synthesize("Hello there.", "en", &cond, &cpu_options(), &cancelled)
        .unwrap()
        .is_none());
    // A voice is required.
    assert!(m
        .conditionals(&app, "")
        .unwrap_err()
        .contains("voice library"));
    assert!(m.conditionals(&app, "missing-voice").is_err());
    drop(m);

    // A new model instance reads the voice from the library cache: plant a cache with other
    // values (same speech encoder) and they are used.
    let mut planted = (*cond).clone();
    planted.tokens = vec![1, 2];
    let fingerprint: String = crate::services::neural_audio::vc::families::reference::sha256_file(&files.speech_encoder)
        .unwrap()
        .chars()
        .take(16)
        .collect();
    let mut e = planted.to_embedding(&fingerprint);
    e.voice_id = "test-voice".into();
    e.backend = CACHE_BACKEND.into();
    e.clip_sha256 = crate::services::voice_library::reference_clip(&app, "test-voice")
        .unwrap()
        .sha256;
    crate::services::voice_library::write_cache(&app, "test-voice", CACHE_BACKEND, &e.to_bytes().unwrap()).unwrap();
    let mut m2 = ChatterboxModel::load(files.clone(), Placement::CPU, 2).unwrap();
    assert_eq!(
        m2.conditionals(&app, "test-voice").unwrap().tokens,
        vec![1, 2],
        "the library cache is used"
    );
    drop(m2);
    // A cache from another speech encoder is recomputed.
    let mut other = planted.to_embedding("0000000000000000");
    other.voice_id = "test-voice".into();
    other.backend = CACHE_BACKEND.into();
    other.clip_sha256 = e.clip_sha256.clone();
    crate::services::voice_library::write_cache(&app, "test-voice", CACHE_BACKEND, &other.to_bytes().unwrap()).unwrap();
    let mut m3 = ChatterboxModel::load(files, Placement::CPU, 2).unwrap();
    assert_eq!(m3.conditionals(&app, "test-voice").unwrap().tokens, PROMPT_TOKENS);
    drop(m3);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn an_over_long_sentence_is_split_into_pieces_the_model_takes() {
    let dir = temp_dir("split");
    let app = dir.join("app");
    let model_dir = dir.join("model");
    library_voice(&app, "split-voice");
    // The stand-in stops every 50 positions, so each piece ends soon after its prompt.
    let script: Vec<i64> = (0..6000)
        .map(|i| if i % 50 == 49 { text::STOP_SPEECH } else { 100 + (i % 7) })
        .collect();
    standin_model(&model_dir, &script);
    let mut m = ChatterboxModel::load(ModelFiles::discover(&model_dir, None).unwrap(), Placement::CPU, 2).unwrap();
    let cond = m.conditionals(&app, "split-voice").unwrap();
    let long = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, ".repeat(40);
    let err = real_frontend().encode(&long, "en").unwrap_err();
    assert!(err.starts_with(text::TOO_LONG), "{err}");
    let never = AtomicBool::new(false);
    let mut pieces = 0;
    let mut sink = |audio: &[f32], rate: u32| {
        assert_eq!(rate, model::SAMPLE_RATE);
        assert!(!audio.is_empty());
        pieces += 1;
        true
    };
    assert!(speak_text(&mut m, &long, "en", &cond, &cpu_options(), &never, &mut sink, 0).unwrap());
    assert!(pieces >= 2, "{pieces} pieces");
    // A stopped consumer stops the rest.
    let mut first_only = |_: &[f32], _: u32| false;
    assert!(!speak_text(&mut m, &long, "en", &cond, &cpu_options(), &never, &mut first_only, 0).unwrap());
    drop(m);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn conditionals_round_trip_through_the_library_cache_format() {
    let c = Conditionals {
        features: w(8, 1.0, 1),
        feature_len: 2,
        hidden: 4,
        tokens: vec![0, 6560, 17],
        speaker: w(192, 1.0, 2),
        prompt: w(160, 1.0, 3),
        prompt_frames: 2,
        mel: 80,
    };
    let e = c.to_embedding("abc");
    let bytes = e.to_bytes().unwrap();
    let back = ReferenceEmbedding::from_bytes(&bytes).unwrap();
    assert_eq!(Conditionals::from_embedding(&back, "abc"), Some(c.clone()));
    assert_eq!(Conditionals::from_embedding(&back, "other"), None, "another speech encoder");
    let mut bad = c;
    bad.tokens.push(7000);
    assert!(bad.validate().is_err(), "tokens beyond the speech vocabulary");
}

/// `TensorProto.DataType` of an element type the flow step uses.
fn onnx_elem_type(t: ort::value::TensorElementType) -> i64 {
    use ort::value::TensorElementType as T;
    match t {
        T::Float32 => FLOAT as i64,
        T::Bool => BOOL as i64,
        T::Int64 => INT64 as i64,
        other => panic!("no ONNX type for {other:?} here"),
    }
}

/// The flow step's declarations as [`model::FLOW_STEP_INPUTS`] / [`model::FLOW_STEP_OUTPUT`] say.
fn driver_flow_step_io() -> (Vec<ValueInfo>, Vec<ValueInfo>) {
    let info = |(name, ty): (&str, ort::value::TensorElementType)| ValueInfo {
        name: name.to_string(),
        elem_type: onnx_elem_type(ty),
    };
    (
        model::FLOW_STEP_INPUTS.iter().copied().map(info).collect(),
        vec![info(model::FLOW_STEP_OUTPUT)],
    )
}

#[test]
fn both_flow_step_exports_take_what_the_driver_binds() {
    // The input and output declarations of the real graphs at commit 3f44461 (testdata/README.md):
    // `flow_step_slim.onnx` (the CPU step, 286 MB with its weights inline, never loaded by any
    // other test) cut from a ranged read of the file's end, and `flow_step_fp16.onnx` from the
    // graph file.
    for file in ["flow_step_slim.io.pb", "flow_step_fp16.io.pb"] {
        let bytes = std::fs::read(testdata(file)).unwrap();
        assert_eq!(read_graph_io(&bytes).unwrap(), driver_flow_step_io(), "{file}");
    }
}

#[test]
fn a_flow_step_of_another_export_fails_at_load() {
    let dir = temp_dir("stepcheck");
    standin_model(&dir, &[5; 64]);
    let files = ModelFiles::discover(&dir, None).unwrap();
    // The stand-in step declares what the driver binds.
    let (step, _) = build_session(None, EP_CPU, &VcSessionOptions::default(), ModelSource::File(&files.flow_step)).unwrap();
    assert_eq!(model::flow_step_problem(&step), None);
    drop(step);
    assert!(model::Flow::load(&files, EP_CPU, 0, 2, false).is_ok());
    // A step whose `time` input has another name: refused at load with the reason, not on every
    // sentence.
    let mut s = Graph::new();
    s.input("x", FLOAT, &[Dim::S("batch"), Dim::N(80), Dim::S("flow_frames")])
        .input("mask", BOOL, &[Dim::S("batch"), Dim::N(1), Dim::S("flow_frames")])
        .input("mu", FLOAT, &[Dim::S("batch"), Dim::N(80), Dim::S("flow_frames")])
        .input("speakers", FLOAT, &[Dim::S("batch"), Dim::N(80)])
        .input("cond", FLOAT, &[Dim::S("batch"), Dim::N(80), Dim::S("flow_frames")])
        .input("t", FLOAT, &[Dim::N(1)])
        .input("delta", FLOAT, &[Dim::N(1)])
        .input("guidance", FLOAT, &[Dim::N(1)])
        .node("Add", &["x", "delta"], &["next_x"], &[])
        .output("next_x", FLOAT, &[Dim::S("batch"), Dim::N(80), Dim::S("flow_frames")]);
    s.save(&files.flow_step, 17, &[]);
    let (step, _) = build_session(None, EP_CPU, &VcSessionOptions::default(), ModelSource::File(&files.flow_step)).unwrap();
    assert_eq!(model::flow_step_problem(&step).as_deref(), Some("it has no time input"));
    drop(step);
    let err = model::Flow::load(&files, EP_CPU, 0, 2, false).err().expect("refused");
    assert!(err.contains("not the expected export") && err.contains("time"), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn seeded_flow_noise_is_standard_normal_and_reproducible() {
    let like = Tensor::<f32>::from_array(([1usize, 80, 50], vec![0f32; 4000]))
        .unwrap()
        .into_dyn();
    let draw = |seed| {
        let v = model::seeded_noise(&like, seed).unwrap();
        let (shape, data) = v.try_extract_tensor::<f32>().unwrap();
        assert_eq!(shape.to_vec(), vec![1, 80, 50]);
        data.to_vec()
    };
    let a = draw(7);
    assert_eq!(a, draw(7));
    assert_ne!(a, draw(8));
    let n = a.len() as f64;
    let mean = a.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let sd = (a.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / n).sqrt();
    assert!(mean.abs() < 0.05 && (sd - 1.0).abs() < 0.05, "mean {mean}, sd {sd}");
    assert!(a.iter().all(|v| v.is_finite()));
}

#[test]
fn the_cosine_schedule_spans_zero_to_one() {
    let s = model::cosine_schedule(6);
    assert_eq!(s.len(), 7);
    assert_eq!(s[0], 0.0);
    assert!((s[6] - 1.0).abs() < 1e-6);
    assert!(s.windows(2).all(|p| p[1] > p[0]));
    // Small steps first (cosine): the first step is smaller than the last.
    assert!(s[1] - s[0] < s[6] - s[5]);
    assert_eq!(model::cosine_schedule(0).len(), 2, "at least one step");
}

/// A DXGI adapter for placement tests.
fn adapter(index: u32, vendor_id: u32, device_id: u32, vram_mb: u64) -> crate::services::diagnostics::gpu_mem::GpuAdapter {
    crate::services::diagnostics::gpu_mem::GpuAdapter {
        index,
        name: "test".into(),
        vendor: "test".into(),
        vendor_id,
        device_id,
        luid: "0:0".into(),
        dedicated_vram_mb: vram_mb,
        shared_memory_mb: 16_384,
        software: false,
        driver_version: None,
        nvidia_driver: None,
        vram: None,
    }
}

#[test]
fn auto_picks_cuda_on_nvidia_directml_on_the_largest_discrete_card_else_the_cpu() {
    use crate::services::diagnostics::gpu_mem::{VENDOR_AMD, VENDOR_INTEL, VENDOR_MICROSOFT, VENDOR_NVIDIA};
    use model::{auto_placement, placement_for};
    let dml = |device| Placement { ep: EP_DML, device };
    let cuda0 = Placement { ep: EP_CUDA, device: 0 };
    let uhd770 = adapter(0, VENDOR_INTEL, 0xA780, 128);
    let rx6800 = adapter(1, VENDOR_AMD, 0x73BF, 16_384);
    let basic = crate::services::diagnostics::gpu_mem::GpuAdapter {
        software: true,
        ..adapter(2, VENDOR_MICROSOFT, 0x8C, 0)
    };
    assert_eq!(auto_placement(&[adapter(0, VENDOR_NVIDIA, 0x2684, 24_576)]), cuda0);
    assert_eq!(auto_placement(&[adapter(0, VENDOR_AMD, 0x744C, 8192)]), dml(0));
    // The integrated GPU that drives the panel is adapter 0, the discrete card adapter 1: every
    // session goes to adapter 1.
    assert_eq!(auto_placement(&[uhd770.clone(), rx6800.clone(), basic.clone()]), dml(1));
    // An APU's 4 GB UMA carve-out reports as dedicated memory: still integrated.
    let phoenix = adapter(0, VENDOR_AMD, 0x15BF, 4096);
    let arc_a770 = adapter(1, VENDOR_INTEL, 0x56A0, 16_000);
    assert_eq!(auto_placement(&[phoenix.clone(), arc_a770]), dml(1));
    assert_eq!(auto_placement(std::slice::from_ref(&phoenix)), Placement::CPU, "an APU alone runs on the CPU");
    assert_eq!(
        auto_placement(&[adapter(0, VENDOR_INTEL, 0x7D55, 2048)]),
        Placement::CPU,
        "Meteor Lake's integrated Arc"
    );
    assert_eq!(auto_placement(std::slice::from_ref(&uhd770)), Placement::CPU, "integrated graphics");
    // Two discrete cards: the one with more memory, wherever it is.
    assert_eq!(
        auto_placement(&[adapter(0, VENDOR_AMD, 0x73DF, 12_288), adapter(1, VENDOR_AMD, 0x744C, 24_576)]),
        dml(1)
    );
    assert_eq!(
        auto_placement(&[adapter(0, VENDOR_AMD, 0x744C, 24_576), adapter(1, VENDOR_AMD, 0x73DF, 12_288)]),
        dml(0)
    );
    assert_eq!(auto_placement(&[adapter(0, VENDOR_AMD, 0x6987, 1536)]), Placement::CPU, "under 2 GB");
    assert_eq!(auto_placement(std::slice::from_ref(&basic)), Placement::CPU, "software adapter");
    assert_eq!(auto_placement(&[]), Placement::CPU);
    // Explicit requests.
    let laptop = [uhd770.clone(), rx6800, basic.clone()];
    assert_eq!(placement_for(Some("dml"), &laptop), dml(1));
    assert_eq!(placement_for(Some(" DirectML "), &laptop), dml(1));
    assert_eq!(placement_for(Some("dml"), &[basic, uhd770]), dml(0), "asked for: integrated graphics");
    assert_eq!(placement_for(Some("dml"), &[]), dml(0));
    assert_eq!(placement_for(Some("cuda"), &laptop), cuda0);
    assert_eq!(placement_for(Some("cpu"), &laptop), Placement::CPU);
    assert_eq!(placement_for(Some(" Auto "), &laptop), dml(1));
    assert_eq!(placement_for(None, &laptop), dml(1));
    assert_eq!(placement_for(Some("auto"), &[phoenix]), Placement::CPU);
    assert_eq!(model::requested_placement(Some("cpu")), Placement::CPU);
    assert_eq!(model::requested_placement(Some("dml")).ep, EP_DML);
    // The device reaches every session: the ladder's DirectML device (and the language model's
    // cache allocator, which takes the same `device_for`).
    let opts = model::session_options(4, 1);
    assert_eq!(opts.cuda_device_index, 1);
    assert_eq!(crate::services::neural_audio::ort_session::device_for(EP_DML, EP_DML, &opts), 1);
    assert_eq!(Placement { ep: EP_DML, device: 1 }.label(), "dml@1");
    assert_eq!(Placement::CPU.label(), "cpu");
}

#[test]
fn request_options_follow_the_reference_defaults_and_ranges() {
    let req = |options: serde_json::Value| CloningTtsRequest {
        options,
        ..CloningTtsRequest::default()
    };
    let d = options_of(&req(serde_json::Value::Null));
    assert_eq!(d.exaggeration, 0.5);
    assert_eq!(d.sampling, SamplingParams::default());
    assert_eq!(d.flow_steps, None);
    assert_eq!(d.seed, None);
    let o = options_of(&req(serde_json::json!({
        "exaggeration": "1.5", "cfgWeight": 9, "temperature": 0.0, "flowSteps": 99, "seed": 12, "minP": -1
    })));
    assert_eq!(o.exaggeration, 1.5);
    assert_eq!(o.sampling.cfg_weight, 1.5);
    assert_eq!(o.sampling.temperature, 0.05);
    assert_eq!(o.sampling.min_p, 0.0);
    assert_eq!(o.flow_steps, Some(32));
    assert_eq!(o.seed, Some(12));
    assert_eq!(resolve_model("").unwrap(), MODEL_ID);
    assert!(resolve_model("tts/kokoro-v1.0").is_err());
    assert_eq!(resolve_threads(None), model::DEFAULT_THREADS);
    assert_eq!(resolve_threads(Some(99)), 8);
    let backend = ChatterboxBackend::app();
    let a = backend
        .key(&req(serde_json::json!({ "ep": "cpu", "exaggeration": 1.0 })))
        .unwrap();
    let b = backend
        .key(&req(serde_json::json!({ "ep": "cpu", "exaggeration": 2.0 })))
        .unwrap();
    assert_eq!(a, b, "sampling options never reload the model");
    assert_ne!(
        a,
        backend
            .key(&req(serde_json::json!({ "ep": "dml" })))
            .unwrap()
    );
    // The key carries the resolved placement (EP and device).
    assert_eq!(a, format!("{MODEL_ID}|cpu|t{}", model::DEFAULT_THREADS));
    let dml = backend
        .key(&req(serde_json::json!({ "ep": "dml" })))
        .unwrap();
    assert_eq!(dml, format!("{MODEL_ID}|{}|t{}", model::requested_placement(Some("dml")).label(), model::DEFAULT_THREADS));
}

#[test]
fn the_worker_speaks_a_library_voice_through_the_stand_in_model() {
    use super::super::pocket::worker::CloneWorker;
    use crate::services::local_tts::engine::{PcmWriter, SinkPort};
    use crate::services::tts_sink::SinkError;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct Sink {
        pushed: Mutex<Vec<(f32, u32)>>,
    }
    struct Writer(Arc<Sink>);
    impl PcmWriter for Writer {
        fn push(&mut self, samples: &[f32], rate: u32) -> Result<(), SinkError> {
            self.0
                .pushed
                .lock()
                .unwrap()
                .extend(samples.iter().map(|&s| (s, rate)));
            Ok(())
        }
        fn is_cancelled(&self) -> bool {
            false
        }
    }
    struct Port(Arc<Sink>);
    impl SinkPort for Port {
        fn writer(&self, _id: &str) -> Result<Box<dyn PcmWriter>, SinkError> {
            Ok(Box::new(Writer(self.0.clone())))
        }
        fn order(&self) -> Vec<String> {
            Vec::new()
        }
        fn queued_ms(&self) -> u32 {
            10_000
        }
    }

    let dir = temp_dir("worker");
    let app = dir.join("app");
    let model_dir = dir.join("model");
    library_voice(&app, "worker-voice");
    let ids = real_frontend()
        .encode("Hello there.", "en")
        .unwrap()
        .unwrap();
    let prompt_len = 2 + ids.len();
    let mut script = script_from(prompt_len, &[40, 41], 200);
    script[prompt_len + 1] = text::STOP_SPEECH;
    // The warm-up runs no synthesis on the CPU, so the script serves the sentence.
    standin_model(&model_dir, &script);
    let sink = Arc::new(Sink::default());
    let worker = CloneWorker::new(
        Arc::new(ChatterboxBackend::for_test(&app, &model_dir)),
        Arc::new(Port(sink.clone())),
        std::time::Duration::from_secs(60),
    );
    let base = CloningTtsRequest {
        reference_voice_id: "worker-voice".into(),
        options: serde_json::json!({ "ep": "cpu", "seed": 3, "temperature": 0.8 }),
        ..CloningTtsRequest::default()
    };
    // Start: loads and prepares the voice.
    let warm = worker
        .speak(base.clone(), String::new())
        .blocking_recv()
        .unwrap()
        .unwrap();
    assert_eq!(warm.frames, 0);
    let report = worker
        .speak(
            CloningTtsRequest {
                text: "Hello there.".into(),
                ..base.clone()
            },
            "utt-1".into(),
        )
        .blocking_recv()
        .unwrap()
        .unwrap();
    assert_eq!(report.sample_rate, model::SAMPLE_RATE);
    let pushed = sink.pushed.lock().unwrap();
    assert_eq!(report.frames as usize, pushed.len());
    assert_eq!(pushed.len(), 80 * (2 * (PROMPT_TOKENS.len() + 2) - PROMPT_FRAMES as usize));
    assert!(pushed.iter().all(|&(_, r)| r == model::SAMPLE_RATE));
    drop(pushed);
    // A missing voice fails the request, not the worker.
    let err = worker
        .speak(
            CloningTtsRequest {
                reference_voice_id: String::new(),
                text: "Hi.".into(),
                ..base.clone()
            },
            "utt-2".into(),
        )
        .blocking_recv()
        .unwrap()
        .unwrap_err();
    assert!(err.contains("voice library"), "{err}");
    worker.unload().blocking_recv().unwrap().unwrap();
    let _ = std::fs::remove_dir_all(dir);
}

// -------------------------------------------------------------------------------------------------
// Real model (ignored)
// -------------------------------------------------------------------------------------------------

/// The reference clip for real-model tests (`SIGIL_TEST_MODELS_DIR/sherpa-onnx/clips/en.wav`).
fn real_clip() -> Option<PathBuf> {
    let root = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from)?;
    let p = root.join("sherpa-onnx").join("clips").join("en.wav");
    p.is_file().then_some(p)
}

/// A temporary voice library holding `clip` as voice `id`.
fn library_from_clip(clip: &Path, id: &str) -> PathBuf {
    let app = temp_dir("reallib");
    let dir = app.join("voices").join(id);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(clip, dir.join("ref.wav")).unwrap();
    let sha = crate::services::neural_audio::vc::families::reference::sha256_file(&dir.join("ref.wav")).unwrap();
    let meta = serde_json::json!({ "id": id, "name": "Real voice", "source": "imported", "sha256": sha });
    std::fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&meta).unwrap()).unwrap();
    app
}

/// Copy the real graphs of `SIGIL_TEST_MODELS_DIR/chatterbox-v3-graphs/onnx` into `dir/onnx` with
/// zero-filled stand-ins for their external weight files (sized from the graphs' own initializer
/// tables, so ONNX Runtime accepts them); `None` when the graphs are not there.
fn real_graphs_with_zero_weights(dir: &Path) -> Option<()> {
    let src = PathBuf::from(std::env::var_os("SIGIL_TEST_MODELS_DIR")?)
        .join("chatterbox-v3-graphs")
        .join("onnx");
    if !src.is_dir() {
        return None;
    }
    std::fs::create_dir_all(dir.join("onnx")).unwrap();
    for e in std::fs::read_dir(&src).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".onnx") {
            continue;
        }
        let bytes = std::fs::read(e.path()).unwrap();
        std::fs::write(dir.join("onnx").join(&name), &bytes).unwrap();
        let mut sizes: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
        for init in read_initializers(&bytes).unwrap() {
            let get = |k: &str| {
                init.external_data
                    .iter()
                    .find(|(key, _)| key == k)
                    .map(|(_, v)| v.clone())
            };
            if let (Some(loc), Some(len)) = (get("location"), get("length")) {
                let off: u64 = get("offset").unwrap_or_else(|| "0".into()).parse().unwrap();
                let end = off + len.parse::<u64>().unwrap();
                let size = sizes.entry(loc).or_default();
                *size = (*size).max(end);
            }
        }
        for (loc, size) in sizes {
            let f = std::fs::File::create(dir.join("onnx").join(loc)).unwrap();
            f.set_len(size).unwrap();
        }
    }
    std::fs::copy(testdata("tokenizer.json"), dir.join("tokenizer.json")).unwrap();
    Some(())
}

#[test]
#[ignore = "needs SIGIL_TEST_MODELS_DIR/chatterbox-v3-graphs (the real graphs without weights, about 3 MB); writes 1.6 GB of zero-filled stand-in weights"]
fn real_graphs_run_with_stand_in_weights() {
    let dir = temp_dir("realgraphs");
    let model_dir = dir.join("model");
    if real_graphs_with_zero_weights(&model_dir).is_none() {
        eprintln!("SIGIL_TEST_MODELS_DIR/chatterbox-v3-graphs not found; skipped");
        return;
    }
    let app = dir.join("app");
    library_voice(&app, "graph-voice");
    // The real fp16 step declares what the driver binds (and what its testdata cut says).
    let fp16 = model_dir.join("onnx/flow_step_fp16.onnx");
    let (step, _) = build_session(None, EP_CPU, &VcSessionOptions::default(), ModelSource::File(&fp16)).unwrap();
    assert_eq!(model::flow_step_problem(&step), None, "flow_step_fp16.onnx");
    let declared: Vec<String> = step.inputs().iter().map(|i| i.name().to_string()).collect();
    let cut = read_graph_io(&std::fs::read(testdata("flow_step_fp16.io.pb")).unwrap()).unwrap();
    assert_eq!(declared, cut.0.iter().map(|v| v.name.clone()).collect::<Vec<_>>());
    drop(step);
    // The CPU step `flow_step_slim.onnx` (286 MB, weights inline) is not in the graph cache unless
    // someone put it there: then it runs here; else this run takes the fp16 step on the CPU, and
    // the slim step is covered only by its declarations (`both_flow_step_exports_take_what_the_
    // driver_binds`) and the real-model test.
    let slim = model_dir.join("onnx/flow_step_slim.onnx");
    let files = if slim.is_file() {
        eprintln!("running the real flow_step_slim.onnx");
        ModelFiles::discover(&model_dir, None).unwrap()
    } else {
        eprintln!(
            "flow_step_slim.onnx is not in SIGIL_TEST_MODELS_DIR/chatterbox-v3-graphs: its inputs and outputs \
             are checked from testdata/flow_step_slim.io.pb only; this run uses flow_step_fp16.onnx as the CPU step"
        );
        let onnx = model_dir.join("onnx");
        ModelFiles {
            dir: model_dir.clone(),
            speech_encoder: onnx.join("speech_encoder_slim.onnx"),
            token_embedding: onnx.join("token_embedding.onnx"),
            language_model: onnx.join("static_language_model_int8.onnx"),
            flow_prepare: onnx.join("flow_prepare_slim.onnx"),
            flow_step: fp16,
            flow_step_fp16: None,
            vocoder: onnx.join("vocoder_prefix_sum.onnx"),
            tokenizer: model_dir.join("tokenizer.json"),
            cangjie: None,
            japanese: None,
            russian: None,
        }
    };
    // The language model's layout, read from the real graph.
    let (session, _) = build_session(None, EP_CPU, &VcSessionOptions::default(), ModelSource::File(&files.language_model)).unwrap();
    let lm = StaticLm::new(session, EP_CPU, 0).unwrap();
    assert_eq!(lm.dims(), (30, 16, 64, 1024, 8194));
    drop(lm);
    // The whole model: load (with the embedding cross-check), a voice through the real speech
    // encoder, and a sentence through the real language model, flow and vocoder.
    let started = std::time::Instant::now();
    let mut m = ChatterboxModel::load(files, Placement::CPU, 4).unwrap();
    eprintln!("loaded in {} ms", started.elapsed().as_millis());
    let cond = m.conditionals(&app, "graph-voice").unwrap();
    eprintln!(
        "conditioning: {} x {} features, {} prompt tokens, {} speaker values, {} x {} prompt mel",
        cond.feature_len,
        cond.hidden,
        cond.tokens.len(),
        cond.speaker.len(),
        cond.prompt_frames,
        cond.mel
    );
    assert_eq!(cond.hidden, 1024);
    assert_eq!(cond.speaker.len(), 192);
    assert_eq!(cond.mel, 80);
    let never = AtomicBool::new(false);
    let opts = GenOptions {
        flow_steps: Some(2),
        seed: Some(1),
        ..GenOptions::default()
    };
    let t = std::time::Instant::now();
    let result = m.synthesize("Hi.", "en", &cond, &opts, &never);
    eprintln!(
        "synthesize: {:?} in {} ms",
        result.as_ref().map(|a| a.as_ref().map(Vec::len)),
        t.elapsed().as_millis()
    );
    let audio = result.unwrap().unwrap();
    // Zero weights make meaningless (even non-finite) samples, but the shapes are the real ones:
    // 480 samples (20 ms at 24 kHz) per mel frame, two frames per speech token.
    assert!(!audio.is_empty());
    assert_eq!(audio.len() % 960, 0, "{} samples", audio.len());
    eprintln!(
        "{} samples, {} non-finite (zero weights)",
        audio.len(),
        audio.iter().filter(|v| !v.is_finite()).count()
    );
    drop(m);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[ignore = "needs tts/chatterbox-mtl-v3 installed in SIGIL_APP_DATA_DIR and SIGIL_TEST_MODELS_DIR clips"]
fn real_model_chatterbox_clones_a_library_voice() {
    let Some(dir) = crate::services::model_store::installed_path(MODEL_ID) else {
        eprintln!("{MODEL_ID} is not installed; skipped");
        return;
    };
    let Some(clip) = real_clip() else {
        eprintln!("SIGIL_TEST_MODELS_DIR/sherpa-onnx/clips/en.wav not found; skipped");
        return;
    };
    let app = library_from_clip(&clip, "real-voice");
    let ep = std::env::var("SIGIL_TEST_CHATTERBOX_EP").unwrap_or_else(|_| "cpu".into());
    let gpu_flow = crate::services::model_store::installed_path(GPU_FLOW_ID);
    let files = ModelFiles::discover(&dir, gpu_flow.as_deref()).unwrap();
    let started = std::time::Instant::now();
    let mut m = ChatterboxModel::load(files, model::requested_placement(Some(&ep)), 4).unwrap();
    eprintln!("loaded on {} in {} ms", m.lm_ep(), started.elapsed().as_millis());
    let t = std::time::Instant::now();
    let cond = m.conditionals(&app, "real-voice").unwrap();
    eprintln!("voice prepared in {} ms ({} prompt tokens)", t.elapsed().as_millis(), cond.tokens.len());
    let never = AtomicBool::new(false);
    for (text_, lang) in [
        ("Hello! This is a cloned voice speaking.", "en"),
        ("Hola, ¿cómo estás?", "es"),
        ("こんにちは、元気ですか。", "ja"),
    ] {
        let t = std::time::Instant::now();
        let audio = m
            .synthesize(text_, lang, &cond, &cpu_options(), &never)
            .unwrap()
            .unwrap();
        let secs = audio.len() as f32 / model::SAMPLE_RATE as f32;
        let rms = (audio.iter().map(|v| v * v).sum::<f32>() / audio.len().max(1) as f32).sqrt();
        eprintln!(
            "{lang}: {secs:.2} s of audio in {} ms (RTF {:.2}), RMS {rms:.3}",
            t.elapsed().as_millis(),
            t.elapsed().as_secs_f32() / secs.max(1e-3)
        );
        assert!(secs > 0.5 && secs < 20.0, "{lang}: {secs} s");
        assert!(rms > 0.005, "{lang}: silent");
        if let Some(out) = std::env::var_os("SIGIL_TEST_OUT_DIR") {
            let path = PathBuf::from(out).join(format!("chatterbox-{lang}.wav"));
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: model::SAMPLE_RATE,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut wr = hound::WavWriter::create(&path, spec).unwrap();
            for s in &audio {
                wr.write_sample(*s).unwrap();
            }
            wr.finalize().unwrap();
        }
    }
    assert!(crate::services::voice_library::read_cache(&app, "real-voice", CACHE_BACKEND).is_some());
    let _ = std::fs::remove_dir_all(app);
}
