//! Tests of the NLLB engine: the decoding algorithms on a table-driven decoder, the ONNX Runtime plumbing on tiny
//! hand-encoded stand-in graphs with NLLB's exact input and output names (no model download), request validation,
//! sentence splitting, and (ignored) the real 600M model against the Python reference.

use super::*;
use crate::services::mt_local::tokenizer::test_tokenizer_json;

// ---------------------------------------------------------------------------------------------
// Decoding on a table-driven decoder
// ---------------------------------------------------------------------------------------------

/// `(prefix, [(next token, probability)])` rows of a [`TableDecoder`].
type Table = Vec<(Vec<i64>, Vec<(i64, f32)>)>;

/// Probabilities of the next token after each prefix; everything else is impossible.
struct TableDecoder {
    vocab: usize,
    table: Table,
    rows: Vec<Vec<i64>>,
    parents_seen: Vec<Vec<usize>>,
}

impl TableDecoder {
    fn new(vocab: usize, table: Table) -> Self {
        Self {
            vocab,
            table,
            rows: Vec::new(),
            parents_seen: Vec::new(),
        }
    }

    fn logits(&self) -> Logits {
        let mut data = vec![f32::NEG_INFINITY; self.rows.len() * self.vocab];
        for (r, prefix) in self.rows.iter().enumerate() {
            if let Some((_, probs)) = self.table.iter().find(|(p, _)| p == prefix) {
                for &(t, p) in probs {
                    data[r * self.vocab + t as usize] = p.ln();
                }
            }
        }
        Logits {
            rows: self.rows.len(),
            vocab: self.vocab,
            data,
        }
    }
}

impl StepDecoder for TableDecoder {
    fn first(&mut self, _prompt: &[i64]) -> Result<Logits, MtError> {
        self.rows = vec![Vec::new()];
        Ok(self.logits())
    }

    fn next(&mut self, tokens: &[i64], parents: &[usize]) -> Result<Logits, MtError> {
        self.parents_seen.push(parents.to_vec());
        self.rows = parents
            .iter()
            .zip(tokens)
            .map(|(&p, &t)| {
                let mut h = self.rows[p].clone();
                h.push(t);
                h
            })
            .collect();
        Ok(self.logits())
    }
}

const EOS: i64 = 2;

/// Greedy takes 4 (0.6) then 4 (0.35): 0.21. Beam finds 5 (0.4) then 5 (0.9): 0.36.
fn trap_table() -> Table {
    vec![
        (vec![], vec![(4, 0.6), (5, 0.4)]),
        (vec![4], vec![(4, 0.35), (5, 0.35), (EOS, 0.3)]),
        (vec![4, 4], vec![(EOS, 1.0)]),
        (vec![4, 5], vec![(EOS, 1.0)]),
        (vec![5], vec![(5, 0.9), (4, 0.05), (EOS, 0.05)]),
        (vec![5, 5], vec![(EOS, 1.0)]),
        (vec![5, 4], vec![(EOS, 1.0)]),
    ]
}

fn params(beam: usize) -> GenParams {
    GenParams {
        beam,
        max_new: 20,
        src_len: 2,
        length_penalty: 1.0,
    }
}

#[test]
fn greedy_and_beam_search_on_a_trap() {
    let never = || false;
    let mut dec = TableDecoder::new(8, trap_table());
    assert_eq!(generate(&mut dec, &[2, 7], EOS, &[], &params(1), &never).unwrap(), [4, 4]);
    let mut dec = TableDecoder::new(8, trap_table());
    assert_eq!(generate(&mut dec, &[2, 7], EOS, &[], &params(2), &never).unwrap(), [5, 5]);
    // Step 1 expands the single first row into two beams; step 2 keeps [5,5] (row 0) and [4,4] (row 1).
    assert_eq!(dec.parents_seen[0], [0, 0]);
    assert_eq!(dec.parents_seen[1], [1, 0], "beams are reordered: [5] came second in step 1");
}

#[test]
fn suppressed_tokens_are_never_generated() {
    let never = || false;
    let table = vec![(vec![], vec![(1, 0.9), (4, 0.1)]), (vec![4], vec![(EOS, 1.0)])];
    let mut dec = TableDecoder::new(8, table.clone());
    assert_eq!(generate(&mut dec, &[2, 7], EOS, &[1], &params(1), &never).unwrap(), [4]);
    let mut dec = TableDecoder::new(8, table);
    assert_eq!(generate(&mut dec, &[2, 7], EOS, &[1], &params(3), &never).unwrap(), [4]);
    // Nothing possible at all is an error, not an endless loop.
    let mut dec = TableDecoder::new(8, vec![(vec![], vec![(1, 1.0)])]);
    assert!(generate(&mut dec, &[2, 7], EOS, &[1], &params(1), &never).is_err());
}

#[test]
fn length_cap_and_cancellation() {
    // Always 4: stops at max_new (the runaway guard needs a longer output than the source + 6).
    let table: Vec<_> = (0..40).map(|n| (vec![4i64; n], vec![(4, 1.0)])).collect();
    let mut dec = TableDecoder::new(8, table.clone());
    let p = GenParams {
        max_new: 5,
        src_len: 10,
        ..params(1)
    };
    assert_eq!(generate(&mut dec, &[2, 7], EOS, &[], &p, &|| false).unwrap(), [4; 5]);
    let mut dec = TableDecoder::new(8, table.clone());
    let e = generate(&mut dec, &[2, 7], EOS, &[], &params(1), &|| true).unwrap_err();
    assert!(e.is_cancelled());
    // Cancelled mid-way (after the second step).
    let calls = std::cell::Cell::new(0);
    let later = || {
        calls.set(calls.get() + 1);
        calls.get() > 2
    };
    let mut dec = TableDecoder::new(8, table);
    assert!(generate(&mut dec, &[2, 7], EOS, &[], &params(2), &later).unwrap_err().is_cancelled());
}

#[test]
fn runaway_repetitions_are_cut() {
    // A loop of "4 5" well past the source length keeps one copy.
    let table: Vec<_> = (0..60)
        .map(|n| {
            let prefix: Vec<i64> = (0..n).map(|i| if i % 2 == 0 { 4 } else { 5 }).collect();
            let next = if n % 2 == 0 { 4 } else { 5 };
            (prefix, vec![(next, 1.0)])
        })
        .collect();
    let mut dec = TableDecoder::new(8, table.clone());
    let out = generate(&mut dec, &[2, 7], EOS, &[], &params(1), &|| false).unwrap();
    assert_eq!(out, [4, 5], "{out:?}");
    let mut dec = TableDecoder::new(8, table);
    let out = generate(&mut dec, &[2, 7], EOS, &[], &params(2), &|| false).unwrap();
    assert_eq!(out, [4, 5], "{out:?}");

    assert_eq!(runaway_repetition(&[4, 4, 4], 0), None, "not longer than the source + 6 yet");
    assert_eq!(runaway_repetition(&[1, 2, 3, 4, 5, 6, 7, 7, 7], 2), Some(7));
    assert_eq!(runaway_repetition(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 2), None);
    assert_eq!(runaway_repetition(&[9, 9, 9, 9, 1, 2, 3, 1, 2, 3, 1, 2, 3], 1), Some(7));
    // Legitimate repetition in a long source is left alone.
    assert_eq!(runaway_repetition(&[5, 5, 5, 5, 5, 5], 10), None);
}

#[test]
fn top_n_and_argmax_helpers() {
    let row = [0.5, f32::NEG_INFINITY, 2.0, 2.0, -1.0, f32::NAN];
    assert_eq!(argmax(&row), Some(2), "lowest index on ties");
    assert_eq!(top_n(&row, 3), [(2, 2.0), (3, 2.0), (0, 0.5)]);
    assert_eq!(top_n(&row, 10).len(), 4, "only finite values");
    assert_eq!(argmax(&[f32::NEG_INFINITY]), None);
    let lse = log_sum_exp(&[0.0, 0.0]).unwrap();
    assert!((lse - 2f32.ln()).abs() < 1e-6);
    assert_eq!(log_sum_exp(&[f32::NEG_INFINITY]), None);
    let mut l = Logits {
        rows: 2,
        vocab: 3,
        data: vec![1.0, f32::NAN, 3.0, 4.0, 5.0, 6.0],
    };
    l.suppress(&[2, 9]);
    assert_eq!(l.row(0)[..2], [1.0, f32::NEG_INFINITY]);
    assert_eq!(l.row(1), [4.0, 5.0, f32::NEG_INFINITY]);
}

#[test]
fn params_follow_the_source_length() {
    let p = GenParams::new(9, 10, 1024);
    assert_eq!((p.beam, p.max_new, p.src_len), (MAX_BEAM, 40, 10));
    assert_eq!(GenParams::new(0, 1000, 1024).max_new, MAX_NEW_TOKENS);
    assert_eq!(GenParams::new(1, 1000, 100).max_new, 96);
}

// ---------------------------------------------------------------------------------------------
// Sentence splitting and request validation
// ---------------------------------------------------------------------------------------------

#[test]
fn sentences_split_at_real_boundaries() {
    assert_eq!(split_sentences("Hello there. How are you?"), ["Hello there.", "How are you?"]);
    assert_eq!(split_sentences("Wait!! Really?! OK."), ["Wait!!", "Really?!", "OK."]);
    assert_eq!(split_sentences("今日は晴れ。明日は雨！どう？"), ["今日は晴れ。", "明日は雨！", "どう？"]);
    assert_eq!(split_sentences("「行くよ！」と言った。"), ["「行くよ！」", "と言った。"]);
    assert_eq!(split_sentences("First line\nsecond line"), ["First line", "second line"]);
    // No split: abbreviations, initials, decimals, a lower-case continuation, no space.
    assert_eq!(split_sentences("Mr. Smith met Dr. Jones."), ["Mr. Smith met Dr. Jones."]);
    assert_eq!(split_sentences("I met J. Smith today."), ["I met J. Smith today."]);
    assert_eq!(split_sentences("It costs 3.50 now."), ["It costs 3.50 now."]);
    assert_eq!(split_sentences("wait... what happened"), ["wait... what happened"]);
    assert_eq!(split_sentences("see e.g. this one"), ["see e.g. this one"]);
    assert_eq!(split_sentences("He said \"Go.\" Then left."), ["He said \"Go.\"", "Then left."]);
    assert_eq!(split_sentences("   "), Vec::<String>::new());
    assert_eq!(split_sentences("no punctuation"), ["no punctuation"]);
    assert_eq!(split_sentences("Trailing space. "), ["Trailing space."]);
}

#[test]
fn output_spacing_is_tidied() {
    assert_eq!(tidy_output("予想以上に  bossの喧嘩が 難しくなった", "jpn_Jpan"), "予想以上に bossの喧嘩が難しくなった");
    assert_eq!(tidy_output("皆様,今夜 流れに 参加してくれて ありがとう!", "jpn_Jpan"), "皆様,今夜流れに参加してくれてありがとう!");
    assert_eq!(tidy_output("我们 一起 玩 Minecraft 吧 。", "zho_Hans"), "我们一起玩 Minecraft 吧。");
    assert_eq!(tidy_output("  Hello   world ", "eng_Latn"), "Hello world");
    assert_eq!(tidy_output("안녕 하세요", "kor_Hang"), "안녕 하세요", "Korean keeps its spaces");
    assert_eq!(tidy_output("", "jpn_Jpan"), "");
}

fn request(text: &str, source: &str, target: &str) -> MtRequest {
    MtRequest {
        text: text.into(),
        source: source.into(),
        target: target.into(),
        ..Default::default()
    }
}

#[test]
fn requests_are_validated() {
    let p = prepare(&request(" Hello ", "en", "ja"), None).unwrap().unwrap();
    assert_eq!(
        p,
        Prepared {
            text: "Hello".into(),
            source: Some("eng_Latn"),
            target: "jpn_Jpan",
            model: NLLB_DEFAULT_MODEL.into(),
            threads: 2,
            beam: 1,
        }
    );
    // Unknown source: guessed later.
    assert_eq!(prepare(&request("Hello", "", "ja"), None).unwrap().unwrap().source, None);
    // The loaded model is the default; an explicit one wins.
    let loaded = "mt/nllb-200-distilled-1.3b-int8";
    assert_eq!(prepare(&request("Hi", "en", "ja"), Some(loaded)).unwrap().unwrap().model, loaded);
    let mut r = request("Hi", "en", "ja");
    r.model = Some(NLLB_DEFAULT_MODEL.into());
    assert_eq!(prepare(&r, Some(loaded)).unwrap().unwrap().model, NLLB_DEFAULT_MODEL);
    // Beam only for finals, capped.
    r.beam = Some(9);
    r.threads = Some(3);
    let p = prepare(&r, None).unwrap().unwrap();
    assert_eq!((p.beam, p.threads), (MAX_BEAM, 3));
    r.kind = MtKind::Interim;
    assert_eq!(prepare(&r, None).unwrap().unwrap().beam, 1);

    // Shortcuts: same language, no letters.
    assert_eq!(
        prepare(&request("Hello", "en-GB", "en"), None).unwrap(),
        Err(Shortcut::Unchanged("Hello".into()))
    );
    assert_eq!(prepare(&request(" 123 😀 ", "en", "ja"), None).unwrap(), Err(Shortcut::Unchanged("123 😀".into())));
    // Errors.
    assert_eq!(prepare(&request("  ", "en", "ja"), None).unwrap_err().kind, MtErrorKind::InvalidInput);
    assert_eq!(prepare(&request("Hi", "en", "tlh"), None).unwrap_err().kind, MtErrorKind::UnsupportedPair);
    assert_eq!(prepare(&request("Hi", "xx", "ja"), None).unwrap_err().kind, MtErrorKind::UnsupportedPair);
    let long = "a".repeat(MAX_TEXT_CHARS + 1);
    assert_eq!(prepare(&request(&long, "en", "ja"), None).unwrap_err().kind, MtErrorKind::InvalidInput);
    let mut r = request("Hi", "en", "ja");
    r.model = Some("stt/whisper-base".into());
    assert_eq!(prepare(&r, None).unwrap_err().kind, MtErrorKind::InvalidInput);
    r.model = Some("mt/nllb-../../x".into());
    assert!(prepare(&r, None).is_err());
}

#[test]
fn paths_are_redacted_from_messages() {
    let dir = Path::new(r"C:\Users\someone\AppData\Roaming\com.sigil.app\models\mt\nllb\261c31d");
    let msg = format!("cannot load {}\\decoder.onnx: bad", dir.display());
    assert_eq!(redact_paths(&msg, Some(dir)), "cannot load <model folder>\\decoder.onnx: bad");
    assert_eq!(redact_paths("plain", None), "plain");
}

#[test]
fn config_parsing() {
    let c = ModelConfig::parse(r#"{"d_model":1024,"decoder_attention_heads":16,"decoder_start_token_id":2,"max_position_embeddings":1024}"#).unwrap();
    assert_eq!((c.heads, c.head_dim, c.decoder_start, c.max_positions), (Some(16), Some(64), 2, 1024));
    let c = ModelConfig::parse("{}").unwrap();
    assert_eq!(c, ModelConfig::default());
    assert!(ModelConfig::parse("not json").is_err());
}

#[test]
fn model_files_are_found_in_the_folder_or_onnx() {
    let dir = std::env::temp_dir().join(format!("sigil-nllb-files-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("onnx")).unwrap();
    assert!(find_model_files(&dir).unwrap_err().contains("encoder"));
    std::fs::write(dir.join("onnx/encoder_model.onnx"), b"x").unwrap();
    std::fs::write(dir.join("onnx/encoder_model_quantized.onnx"), b"x").unwrap();
    std::fs::write(dir.join("onnx/decoder_model_merged.onnx"), b"x").unwrap();
    assert!(find_model_files(&dir).unwrap_err().contains("tokenizer"));
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    let f = find_model_files(&dir).unwrap();
    assert!(f.encoder.ends_with("onnx/encoder_model_quantized.onnx"), "{f:?}");
    assert!(f.decoder.ends_with("onnx/decoder_model_merged.onnx"));
    assert_eq!(f.config, None);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_missing_model_reports_not_installed() {
    // Point the model store at an empty scratch app-data folder only if nothing else did.
    let mut r = request("Hello", "en", "ja");
    r.model = Some("mt/nllb-test-not-in-the-catalog".into());
    let prepared = prepare(&r, None).unwrap().unwrap();
    let e = translate_prepared(prepared, None).await.unwrap_err();
    assert_eq!(e.kind, MtErrorKind::NotInstalled);
    assert_eq!(e.missing, ["mt/nllb-test-not-in-the-catalog"]);
}

// ---------------------------------------------------------------------------------------------
// ONNX Runtime plumbing on stand-in graphs
// ---------------------------------------------------------------------------------------------

/// Minimal ONNX protobuf writer (field numbers from onnx.proto).
mod onnx {
    pub const FLOAT: u64 = 1;
    pub const INT64: u64 = 7;
    pub const BOOL: u64 = 9;

    fn varint(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }

    fn field_varint(out: &mut Vec<u8>, field: u32, v: u64) {
        varint(out, u64::from(field) << 3);
        varint(out, v);
    }

    fn field_bytes(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
        varint(out, (u64::from(field) << 3) | 2);
        varint(out, bytes.len() as u64);
        out.extend_from_slice(bytes);
    }

    #[derive(Clone, Copy)]
    pub enum Dim {
        Fixed(u64),
        Named(&'static str),
    }

    fn value_info(name: &str, elem: u64, dims: &[Dim]) -> Vec<u8> {
        let mut shape = Vec::new();
        for d in dims {
            let mut dim = Vec::new();
            match d {
                Dim::Fixed(v) => field_varint(&mut dim, 1, *v),
                Dim::Named(n) => field_bytes(&mut dim, 2, n.as_bytes()),
            }
            field_bytes(&mut shape, 1, &dim);
        }
        let mut tensor = Vec::new();
        field_varint(&mut tensor, 1, elem);
        field_bytes(&mut tensor, 2, &shape);
        let mut ty = Vec::new();
        field_bytes(&mut ty, 1, &tensor);
        let mut v = Vec::new();
        field_bytes(&mut v, 1, name.as_bytes());
        field_bytes(&mut v, 2, &ty);
        v
    }

    #[derive(Default)]
    pub struct Graph {
        nodes: Vec<Vec<u8>>,
        inits: Vec<Vec<u8>>,
        inputs: Vec<Vec<u8>>,
        outputs: Vec<Vec<u8>>,
    }

    impl Graph {
        pub fn input(&mut self, name: &str, elem: u64, dims: &[Dim]) {
            self.inputs.push(value_info(name, elem, dims));
        }

        pub fn output(&mut self, name: &str, elem: u64, dims: &[Dim]) {
            self.outputs.push(value_info(name, elem, dims));
        }

        fn init(&mut self, name: &str, elem: u64, dims: &[u64], raw: Vec<u8>) {
            let mut t = Vec::new();
            if !dims.is_empty() {
                let mut packed = Vec::new();
                for &d in dims {
                    varint(&mut packed, d);
                }
                field_bytes(&mut t, 1, &packed);
            }
            field_varint(&mut t, 2, elem);
            field_bytes(&mut t, 8, name.as_bytes());
            field_bytes(&mut t, 9, &raw);
            self.inits.push(t);
        }

        pub fn i64s(&mut self, name: &str, dims: &[u64], values: &[i64]) {
            self.init(name, INT64, dims, values.iter().flat_map(|v| v.to_le_bytes()).collect());
        }

        pub fn f32s(&mut self, name: &str, dims: &[u64], values: &[f32]) {
            self.init(name, FLOAT, dims, values.iter().flat_map(|v| v.to_le_bytes()).collect());
        }

        pub fn node(&mut self, op: &str, inputs: &[&str], outputs: &[&str], ints: &[(&str, i64)]) {
            let mut n = Vec::new();
            for i in inputs {
                field_bytes(&mut n, 1, i.as_bytes());
            }
            for o in outputs {
                field_bytes(&mut n, 2, o.as_bytes());
            }
            field_bytes(&mut n, 3, format!("n{}_{op}", self.nodes.len()).as_bytes());
            field_bytes(&mut n, 4, op.as_bytes());
            for (name, v) in ints {
                let mut a = Vec::new();
                field_bytes(&mut a, 1, name.as_bytes());
                field_varint(&mut a, 3, *v as u64);
                field_varint(&mut a, 20, 2); // AttributeType.INT
                field_bytes(&mut n, 5, &a);
            }
            self.nodes.push(n);
        }

        pub fn model(&self) -> Vec<u8> {
            let mut g = Vec::new();
            for n in &self.nodes {
                field_bytes(&mut g, 1, n);
            }
            field_bytes(&mut g, 2, b"sigil_nllb_stand_in");
            for t in &self.inits {
                field_bytes(&mut g, 5, t);
            }
            for i in &self.inputs {
                field_bytes(&mut g, 11, i);
            }
            for o in &self.outputs {
                field_bytes(&mut g, 12, o);
            }
            let mut m = Vec::new();
            field_varint(&mut m, 1, 8);
            field_bytes(&mut m, 2, b"sigil-nllb-test");
            field_bytes(&mut m, 7, &g);
            let mut opset = Vec::new();
            field_varint(&mut opset, 2, 17);
            field_bytes(&mut m, 8, &opset);
            m
        }
    }
}

use onnx::Dim::{Fixed, Named};

/// Stand-in geometry: 2 layers, 2 heads of 2 values, 32 tokens; ends after `STOP_LEN` decoder positions.
const LAYERS: usize = 2;
const HEADS: u64 = 2;
const HEAD_DIM: u64 = 2;
const VOCAB: u64 = 32;
const MODULUS: i64 = 16;
const STOP_LEN: i64 = 8;

/// Encoder stand-in: `last_hidden_state[b, s, 0] = input_ids * attention_mask`.
fn stand_in_encoder() -> Vec<u8> {
    let mut g = onnx::Graph::default();
    g.input("input_ids", onnx::INT64, &[Named("batch_size"), Named("encoder_sequence_length")]);
    g.input("attention_mask", onnx::INT64, &[Named("batch_size"), Named("encoder_sequence_length")]);
    g.output(
        "last_hidden_state",
        onnx::FLOAT,
        &[Named("batch_size"), Named("encoder_sequence_length"), Fixed(1)],
    );
    g.i64s("axes2", &[1], &[2]);
    g.node("Mul", &["input_ids", "attention_mask"], &["masked"], &[]);
    g.node("Cast", &["masked"], &["masked_f"], &[("to", onnx::FLOAT as i64)]);
    g.node("Unsqueeze", &["masked_f", "axes2"], &["last_hidden_state"], &[]);
    g.model()
}

/// Merged-decoder stand-in with NLLB's names. Its cache holds the token history (every decoder key/value entry is
/// the input id at that position; the encoder cache is the encoder output), and the next token of each row is
/// `(4 · Σ history + Σ encoder cache) mod 16 + 4`, or `</s>` once the cache is [`STOP_LEN`] long. A wrong cache
/// (missing, not grown, rows mixed up by beam search) changes the tokens.
fn stand_in_decoder() -> Vec<u8> {
    let mut g = onnx::Graph::default();
    let b = Named("batch_size");
    g.input("encoder_attention_mask", onnx::INT64, &[b, Named("encoder_sequence_length")]);
    g.input("input_ids", onnx::INT64, &[b, Named("decoder_sequence_length")]);
    g.input("encoder_hidden_states", onnx::FLOAT, &[b, Named("encoder_sequence_length"), Fixed(1)]);
    for l in 0..LAYERS {
        for part in KV_PARTS {
            let len = if part.starts_with("decoder") { "past_decoder_sequence_length" } else { "encoder_sequence_length_out" };
            g.input(&format!("past_key_values.{l}.{part}"), onnx::FLOAT, &[b, Fixed(HEADS), Named(len), Fixed(HEAD_DIM)]);
        }
    }
    g.input("use_cache_branch", onnx::BOOL, &[Fixed(1)]);
    g.output("logits", onnx::FLOAT, &[b, Named("logits_len"), Fixed(VOCAB)]);
    for l in 0..LAYERS {
        for part in KV_PARTS {
            g.output(&format!("present.{l}.{part}"), onnx::FLOAT, &[b, Fixed(HEADS), Named("present_len"), Fixed(HEAD_DIM)]);
        }
    }
    g.i64s("axes13", &[2], &[1, 3]);
    g.i64s("axes1", &[1], &[1]);
    g.i64s("expand", &[4], &[1, HEADS as i64, 1, HEAD_DIM as i64]);
    g.i64s("sum_axes", &[3], &[1, 2, 3]);
    g.i64s("modulus", &[], &[MODULUS]);
    g.i64s("four", &[], &[4]);
    g.i64s("two", &[], &[2]);
    g.i64s("stop_len", &[], &[STOP_LEN]);
    g.i64s("eos", &[], &[EOS]);
    g.i64s("depth", &[], &[VOCAB as i64]);
    g.f32s("onehot_values", &[2], &[0.0, 5.0]);

    g.node("Cast", &["input_ids"], &["ids_f"], &[("to", onnx::FLOAT as i64)]);
    g.node("Unsqueeze", &["ids_f", "axes13"], &["ids_4d"], &[]);
    g.node("Expand", &["ids_4d", "expand"], &["new_kv"], &[]);
    g.node("Unsqueeze", &["encoder_hidden_states", "axes1"], &["enc_4d"], &[]);
    g.node("Expand", &["enc_4d", "expand"], &["enc_kv"], &[]);
    for l in 0..LAYERS {
        for part in ["decoder.key", "decoder.value"] {
            g.node(
                "Concat",
                &[&format!("past_key_values.{l}.{part}"), "new_kv"],
                &[&format!("present.{l}.{part}")],
                &[("axis", 2)],
            );
        }
        for part in ["encoder.key", "encoder.value"] {
            g.node("Identity", &["enc_kv"], &[&format!("present.{l}.{part}")], &[]);
        }
    }
    g.node("ReduceSum", &["present.0.decoder.key", "sum_axes"], &["hist"], &[("keepdims", 0)]);
    g.node("ReduceSum", &["past_key_values.0.encoder.key", "sum_axes"], &["encp"], &[("keepdims", 0)]);
    g.node("Add", &["hist", "encp"], &["total"], &[]);
    g.node("Cast", &["total"], &["total_i"], &[("to", onnx::INT64 as i64)]);
    g.node("Mod", &["total_i", "modulus"], &["m"], &[]);
    g.node("Add", &["m", "four"], &["cand"], &[]);
    g.node("Shape", &["present.0.decoder.key"], &["kv_shape"], &[]);
    g.node("Gather", &["kv_shape", "two"], &["cur_len"], &[("axis", 0)]);
    g.node("GreaterOrEqual", &["cur_len", "stop_len"], &["done"], &[]);
    g.node("Where", &["done", "eos", "cand"], &["next"], &[]);
    g.node("OneHot", &["next", "depth", "onehot_values"], &["oh"], &[("axis", -1)]);
    g.node("Unsqueeze", &["oh", "axes1"], &["logits"], &[]);
    g.model()
}

/// What the stand-in generates greedily for `src` (encoder ids) and `prompt`.
fn stand_in_expected(src: &[i64], prompt: &[i64]) -> Vec<i64> {
    let enc_sum: i64 = src.iter().sum::<i64>() * (HEADS * HEAD_DIM) as i64;
    let mut history: Vec<i64> = prompt.to_vec();
    let mut out = Vec::new();
    let mut first = true;
    loop {
        let hist = history.iter().sum::<i64>() * (HEADS * HEAD_DIM) as i64;
        let encp = if first { 0 } else { enc_sum };
        first = false;
        let next = if history.len() as i64 >= STOP_LEN { EOS } else { (hist + encp) % MODULUS + 4 };
        if next == EOS {
            return out;
        }
        out.push(next);
        history.push(next);
    }
}

fn stand_in_sessions() -> (Session, Session) {
    let opts = VcSessionOptions {
        intra_threads: Some(1),
        ..VcSessionOptions::default()
    };
    let (enc, _) = build_session(None, EP_CPU, &opts, ModelSource::Memory(&stand_in_encoder())).expect("encoder stand-in loads");
    let (dec, _) = build_session(None, EP_CPU, &opts, ModelSource::Memory(&stand_in_decoder())).expect("decoder stand-in loads");
    (enc, dec)
}

fn stand_in_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sigil-nllb-stand-in-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("encoder_model_quantized.onnx"), stand_in_encoder()).unwrap();
    std::fs::write(dir.join("decoder_model_merged_quantized.onnx"), stand_in_decoder()).unwrap();
    std::fs::write(dir.join("tokenizer.json"), test_tokenizer_json()).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{"d_model":4,"decoder_attention_heads":2,"decoder_start_token_id":2,"max_position_embeddings":64}"#,
    )
    .unwrap();
    dir
}

#[test]
fn layouts_of_the_stand_in_are_read() {
    let (enc, dec) = stand_in_sessions();
    let e = parse_encoder_layout(&enc).unwrap();
    assert_eq!((e.input_ids.as_str(), e.attention_mask.as_deref(), e.output.as_str()), ("input_ids", Some("attention_mask"), "last_hidden_state"));
    let d = parse_decoder_layout(&dec, &ModelConfig::default()).unwrap();
    assert_eq!(d.layers.len(), LAYERS);
    assert_eq!((d.heads, d.head_dim), (HEADS as usize, HEAD_DIM as usize));
    assert_eq!(d.layers[1].inputs[2], "past_key_values.1.encoder.key");
    assert_eq!(d.layers[1].outputs[3], "present.1.encoder.value");
    // The encoder is not a decoder.
    assert!(parse_decoder_layout(&enc, &ModelConfig::default()).is_err());
}

#[test]
fn greedy_decoding_runs_the_cache_like_the_reference_loop() {
    let (mut enc_session, mut dec_session) = stand_in_sessions();
    let layout = parse_decoder_layout(&dec_session, &ModelConfig::default()).unwrap();
    let enc_layout = parse_encoder_layout(&enc_session).unwrap();
    let src = [20i64, 4, 5, 2];
    let hidden = {
        let input = Tensor::<i64>::from_array(([1usize, src.len()], src.to_vec())).unwrap();
        let mask = ones_mask(1, src.len()).unwrap();
        let mut out = enc_session
            .run(ort::inputs![enc_layout.input_ids.as_str() => input, "attention_mask" => mask])
            .unwrap();
        out.remove("last_hidden_state").unwrap()
    };
    let prompt = [2i64, 21];
    let mut dec = OrtStepDecoder::new(&mut dec_session, &layout, hidden, src.len()).unwrap();
    let out = generate(&mut dec, &prompt, EOS, &[0, 1, 20, 21, 22, 23], &params(1), &|| false).unwrap();
    let expected = stand_in_expected(&src, &prompt);
    assert_eq!(out, expected);
    assert_eq!(out.len() as i64, STOP_LEN - 2, "ends when the cache reaches the stop length");
}

#[test]
fn beam_rows_are_expanded_and_reordered() {
    let (mut enc_session, mut dec_session) = stand_in_sessions();
    let layout = parse_decoder_layout(&dec_session, &ModelConfig::default()).unwrap();
    let src = [20i64, 4, 2];
    let hidden = {
        let input = Tensor::<i64>::from_array(([1usize, src.len()], src.to_vec())).unwrap();
        let mut out = enc_session
            .run(ort::inputs!["input_ids" => input, "attention_mask" => ones_mask(1, src.len()).unwrap()])
            .unwrap();
        out.remove("last_hidden_state").unwrap()
    };
    let mut dec = OrtStepDecoder::new(&mut dec_session, &layout, hidden, src.len()).unwrap();
    let enc_sum = src.iter().sum::<i64>() * 4;
    let next_of = |history: &[i64]| (history.iter().sum::<i64>() * 4 + enc_sum) % MODULUS + 4;
    let argmax_rows = |l: &Logits| (0..l.rows).map(|r| argmax(l.row(r)).unwrap() as i64).collect::<Vec<_>>();

    let first = dec.first(&[2, 21]).unwrap();
    assert_eq!(first.rows, 1);
    // Two rows from row 0 with different tokens.
    let l = dec.next(&[6, 9], &[0, 0]).unwrap();
    assert_eq!(argmax_rows(&l), [next_of(&[2, 21, 6]), next_of(&[2, 21, 9])]);
    // Swap the rows: row 0 continues row 1's history.
    let l = dec.next(&[7, 8], &[1, 0]).unwrap();
    assert_eq!(argmax_rows(&l), [next_of(&[2, 21, 9, 7]), next_of(&[2, 21, 6, 8])]);
    // Three rows, then back to one.
    let l = dec.next(&[4, 5, 6], &[0, 0, 1]).unwrap();
    assert_eq!(
        argmax_rows(&l),
        [next_of(&[2, 21, 9, 7, 4]), next_of(&[2, 21, 9, 7, 5]), next_of(&[2, 21, 6, 8, 6])]
    );
    // Back to one row (the third): cache length 6, then 7, then the stop length.
    let l = dec.next(&[10], &[2]).unwrap();
    assert_eq!(argmax_rows(&l), [next_of(&[2, 21, 6, 8, 6, 10])]);
    let l = dec.next(&[11], &[0]).unwrap();
    assert_eq!(argmax_rows(&l), [next_of(&[2, 21, 6, 8, 6, 10, 11])]);
    let l = dec.next(&[12], &[0]).unwrap();
    assert_eq!(argmax_rows(&l), [EOS], "cache length reached the stop length");
    assert!(dec.next(&[1, 2], &[0]).is_err());

    // A full beam search on the stand-in returns the greedy path (one-hot logits).
    let (mut enc_session, mut dec_session) = stand_in_sessions();
    let hidden = {
        let input = Tensor::<i64>::from_array(([1usize, src.len()], src.to_vec())).unwrap();
        let mut out = enc_session
            .run(ort::inputs!["input_ids" => input, "attention_mask" => ones_mask(1, src.len()).unwrap()])
            .unwrap();
        out.remove("last_hidden_state").unwrap()
    };
    let mut dec = OrtStepDecoder::new(&mut dec_session, &layout, hidden, src.len()).unwrap();
    let out = generate(&mut dec, &[2, 21], EOS, &[0, 1], &params(3), &|| false).unwrap();
    assert_eq!(out, stand_in_expected(&src, &[2, 21]));
}

#[test]
fn a_stand_in_model_folder_loads_and_translates() {
    let dir = stand_in_dir("model");
    let mut model = NllbModel::load(&dir, "mt/nllb-stand-in", 1).expect("the stand-in folder loads");
    assert_eq!(model.id(), "mt/nllb-stand-in");
    // Source "hello world" in English: [eng_Latn=20, 4, 5, </s>].
    let src = model.tokenizer().encode("hello world", "eng_Latn").unwrap();
    assert_eq!(src, [20, 4, 5, 2]);
    let ids = model.generate_ids(&src, 21, 1, &|| false).unwrap();
    assert_eq!(ids, stand_in_expected(&src, &[2, 21]));
    let beam = model.generate_ids(&src, 21, 2, &|| false).unwrap();
    assert_eq!(beam, ids);
    // Text in, text out (the stand-in's ids decode to words of the tiny vocabulary or nothing).
    let text = model.translate("hello world", Some("eng_Latn"), "fra_Latn", 1, &|| false).unwrap();
    let words: Vec<String> = ids
        .iter()
        .filter_map(|&id| {
            ["hello", "world", "good", "morning", "こんにちは", "世界", "bonjour", "le", "monde"]
                .get((id - 4) as usize)
                .map(|w| w.to_string())
        })
        .collect();
    assert_eq!(text, words.join(" "));
    // Two sentences are translated one by one and joined.
    assert!(model.translate("hello world. Good morning", Some("eng_Latn"), "fra_Latn", 1, &|| false).is_ok());
    // Unknown target / source languages.
    assert_eq!(
        model.translate("hello", Some("eng_Latn"), "deu_Latn", 1, &|| false).unwrap_err().kind,
        MtErrorKind::UnsupportedPair
    );
    assert_eq!(
        model.translate("hello", Some("kor_Hang"), "fra_Latn", 1, &|| false).unwrap_err().kind,
        MtErrorKind::UnsupportedPair
    );
    // Guessed source (Latin → English); no letters → unchanged.
    assert!(model.translate("hello", None, "fra_Latn", 1, &|| false).is_ok());
    assert_eq!(model.translate(" 42 ", None, "fra_Latn", 1, &|| false).unwrap(), "42");
    assert!(model.translate("hello", None, "fra_Latn", 1, &|| true).unwrap_err().is_cancelled());
    drop(model);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_broken_folder_is_rejected_with_a_reason() {
    let dir = stand_in_dir("broken");
    // The encoder stand-in in place of the decoder.
    std::fs::write(dir.join("decoder_model_merged_quantized.onnx"), stand_in_encoder()).unwrap();
    let e = NllbModel::load(&dir, "mt/nllb-broken", 1).unwrap_err();
    assert!(e.contains("merged decoder") || e.contains("no "), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------------------------
// The real model (ignored unless installed)
// ---------------------------------------------------------------------------------------------

fn real_model_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("SIGIL_NLLB_DIR") {
        return Some(dir.into());
    }
    crate::services::model_store::installed_path(NLLB_DEFAULT_MODEL)
}

#[test]
#[ignore = "needs SIGIL_APP_DATA_DIR with mt/nllb-200-distilled-600m-int8 installed (or SIGIL_NLLB_DIR)"]
fn real_model_matches_the_python_reference() {
    let dir = real_model_dir().expect("NLLB-600M is not installed");
    let started = Instant::now();
    let mut model = NllbModel::load(&dir, NLLB_DEFAULT_MODEL, 2).unwrap();
    eprintln!("loaded in {} ms", started.elapsed().as_millis());
    let fixture: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../scripts/mt-bench/fixtures/nllb_parity.json"))
            .unwrap(),
    )
    .unwrap();
    // Sigil links ONNX Runtime 1.28; the fixture may come from an older Python onnxruntime whose int8 kernels round
    // slightly differently. The ids must match, except that decoding may take the other branch of a near-tie (the
    // reference's top-1 / top-2 logit gap below NEAR_TIE at the first differing step).
    const NEAR_TIE: f64 = 0.05;
    let (mut total_ms, mut exact, mut ties, mut cases) = (0u128, 0usize, 0usize, 0usize);
    for case in fixture["cases"].as_array().unwrap() {
        let Some(want) = case.get("output_ids").and_then(|v| v.as_array()) else { continue };
        cases += 1;
        let want: Vec<i64> = want.iter().map(|v| v.as_i64().unwrap()).collect();
        let margins: Vec<f64> = case
            .get("output_margins")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|v| v.as_f64().unwrap()).collect())
            .unwrap_or_default();
        let src: Vec<i64> = case["ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        let tgt = case["target"].as_str().unwrap();
        let tgt_id = model.tokenizer().lang_id(tgt).unwrap();
        let t = Instant::now();
        let got = model.generate_ids(&src, tgt_id, 1, &|| false).unwrap();
        total_ms += t.elapsed().as_millis();
        if got == want {
            exact += 1;
            assert_eq!(model.tokenizer().decode(&got).unwrap(), case["output_text"].as_str().unwrap().trim());
            continue;
        }
        let k = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
        let gap = margins.get(k).copied().unwrap_or(f64::INFINITY);
        assert!(
            gap < NEAR_TIE,
            "{:?} -> {tgt}: differs at step {k} where the reference margin is {gap}
  rust   {got:?}
  python {want:?}",
            case["text"]
        );
        ties += 1;
        eprintln!(
            "near-tie at step {k} (reference gap {gap}) for {:?}: {:?}",
            case["text"],
            model.tokenizer().decode(&got).unwrap()
        );
    }
    assert!(exact * 10 >= cases * 8, "only {exact} of {cases} lines match the reference exactly");
    eprintln!("greedy parity: {exact}/{cases} exact, {ties} near-tie branches, {total_ms} ms decoding in total");
    // Beam search and the text API on a few lines.
    for (text, src, tgt) in [
        ("Hello, how are you today? I'll be right back.", Some("eng_Latn"), "jpn_Jpan"),
        ("今日はいい天気ですね。散歩に行きましょう。", None, "eng_Latn"),
        ("¿Dónde está la estación de tren?", Some("spa_Latn"), "deu_Latn"),
    ] {
        let t = Instant::now();
        let greedy = model.translate(text, src, tgt, 1, &|| false).unwrap();
        let g_ms = t.elapsed().as_millis();
        let t = Instant::now();
        let beam = model.translate(text, src, tgt, 4, &|| false).unwrap();
        eprintln!("{text} -> [{g_ms} ms] {greedy} | [beam 4, {} ms] {beam}", t.elapsed().as_millis());
        assert!(!greedy.is_empty() && !beam.is_empty());
    }
}

#[tokio::test]
#[ignore = "needs SIGIL_APP_DATA_DIR with mt/nllb-200-distilled-600m-int8 installed"]
async fn real_model_through_the_worker() {
    // The path of `nllb_translate`: validation, the model store, the worker thread (load on demand), cancellation.
    let mut r = request("Good morning everyone! Thanks for coming to the stream today.", "en", "ja");
    r.request_id = Some("t1".into());
    let prepared = prepare(&r, None).unwrap().unwrap();
    let started = Instant::now();
    let response = translate_prepared(prepared, r.request_id.clone()).await.unwrap();
    eprintln!("{} ms (with load): {}", started.elapsed().as_millis(), response.text);
    assert_eq!(response.engine, ENGINE_NLLB);
    assert_eq!(response.model, NLLB_DEFAULT_MODEL);
    assert!(!response.text.is_empty());
    assert!(!response.text.contains("  "));
    assert_eq!(engine().status().state, MtEngineState::Ready);
    assert_eq!(engine().loaded_id().as_deref(), Some(NLLB_DEFAULT_MODEL));

    // Unknown source (Korean script), loaded model reused.
    let r = request("오늘 날씨가 정말 좋네요.", "", "en");
    let prepared = prepare(&r, engine().loaded_id().as_deref()).unwrap().unwrap();
    let started = Instant::now();
    let response = translate_prepared(prepared, None).await.unwrap();
    eprintln!("{} ms: {}", started.elapsed().as_millis(), response.text);
    assert!(!response.text.chars().any(|c| ('\u{AC00}'..='\u{D7AF}').contains(&c)), "still Korean");

    // A cancelled request stops.
    let long = "This is a rather long sentence that keeps going so that decoding takes a while. ".repeat(6);
    let mut r = request(&long, "en", "de");
    r.request_id = Some("cancel-me".into());
    let prepared = prepare(&r, None).unwrap().unwrap();
    let task = tokio::spawn(translate_prepared(prepared, r.request_id.clone()));
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert!(engine().cancel(Some("cancel-me")) >= 1);
    let e = task.await.unwrap().unwrap_err();
    assert!(e.is_cancelled(), "{e:?}");

    // Unload returns the engine to idle.
    let (reply, rx) = tokio::sync::oneshot::channel();
    engine().send(Job::Unload { reply }).unwrap();
    rx.await.unwrap();
    assert_eq!(engine().status().state, MtEngineState::Idle);
    assert_eq!(engine().loaded_id(), None);
}
