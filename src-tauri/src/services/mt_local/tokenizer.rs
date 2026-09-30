//! NLLB tokenizer: the Hugging Face `tokenizer.json` of the model (SentencePiece BPE: the `nmt_nfkc` precompiled
//! normalizer, Metaspace pre-tokenizer, 256k BPE vocabulary plus the 202 FLORES-200 language tokens), loaded with the
//! `tokenizers` crate (default features off, `fancy-regex` backend; see Cargo.toml).
//!
//! Encoding follows `NllbTokenizer` with `legacy_behaviour=False`: `[<src_lang>] + pieces + [</s>]`. The template in
//! `tokenizer.json` hard-codes `eng_Latn` as the prefix, so pieces are encoded without special tokens and the
//! language token is added here. Before encoding, whitespace is cleaned the way sentencepiece's
//! `remove_extra_whitespaces` does ([`clean_source`]); without it `tokenizer.json` keeps a trailing `▁` that Python
//! sentencepiece drops. Decoding skips every special token (language tokens included).
//!
//! Parity: `scripts/mt-bench/nllb_reference.py` writes `scripts/mt-bench/fixtures/nllb_parity.json` with Python
//! sentencepiece's ids (the slow `NllbTokenizer` over `sentencepiece.bpe.model`) for 27 lines in 20 languages
//! (full-width forms, ideographic and no-break spaces, emoji, unknown symbols, zero-width characters). The ignored
//! test `matches_python_sentencepiece` checks this tokenizer against it; the fixture needs no model, only
//! `tokenizer.json`.
//!
//! Owned by P4-D.

use std::collections::HashMap;
use std::path::Path;

/// End of sequence (NLLB's decoder start token too).
pub const EOS_TOKEN: &str = "</s>";
const PAD_TOKEN: &str = "<pad>";
const BOS_TOKEN: &str = "<s>";

/// Wrapper around `tokenizers::Tokenizer` with NLLB's special tokens resolved once.
pub struct NllbTokenizer {
    inner: tokenizers::Tokenizer,
    eos: u32,
    /// FLORES code → token id, for the language tokens present in the vocabulary.
    languages: HashMap<String, u32>,
    /// Ids that must never be generated inside a translation (pad, bos, language tokens, `<mask>`).
    suppressed: Vec<u32>,
}

impl std::fmt::Debug for NllbTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NllbTokenizer")
            .field("eos", &self.eos)
            .field("languages", &self.languages.len())
            .finish()
    }
}

/// Collapse every whitespace run (spaces, tabs, newlines, U+3000, no-break spaces) into one space and trim the
/// ends, as sentencepiece's `remove_extra_whitespaces` does.
pub fn clean_source(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_flores_token(token: &str) -> bool {
    let b = token.as_bytes();
    b.len() == 8
        && b[3] == b'_'
        && b[..3].iter().all(u8::is_ascii_lowercase)
        && b[4].is_ascii_uppercase()
        && b[5..].iter().all(u8::is_ascii_lowercase)
}

impl NllbTokenizer {
    /// Load `tokenizer.json`.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("cannot load the tokenizer {}: {e}", file_name(path)))?;
        Self::from_tokenizer(inner)
    }

    /// Parse a `tokenizer.json` document.
    #[cfg(test)]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes).map_err(|e| format!("cannot parse the tokenizer: {e}"))?;
        Self::from_tokenizer(inner)
    }

    fn from_tokenizer(inner: tokenizers::Tokenizer) -> Result<Self, String> {
        let eos = inner
            .token_to_id(EOS_TOKEN)
            .ok_or_else(|| format!("the tokenizer has no {EOS_TOKEN} token (not an NLLB tokenizer)"))?;
        let mut languages = HashMap::new();
        let mut suppressed = Vec::new();
        for (id, token) in inner.get_added_vocabulary().get_added_tokens_decoder() {
            if is_flores_token(&token.content) {
                languages.insert(token.content.clone(), *id);
                suppressed.push(*id);
            } else if token.content == "<mask>" {
                suppressed.push(*id);
            }
        }
        if languages.is_empty() {
            return Err("the tokenizer has no FLORES-200 language tokens (not an NLLB tokenizer)".into());
        }
        suppressed.extend([PAD_TOKEN, BOS_TOKEN].iter().filter_map(|t| inner.token_to_id(t)).filter(|&id| id != eos));
        suppressed.sort_unstable();
        suppressed.dedup();
        Ok(Self {
            inner,
            eos,
            languages,
            suppressed,
        })
    }

    /// Id of a FLORES-200 language token (`jpn_Jpan`).
    pub fn lang_id(&self, flores: &str) -> Option<u32> {
        self.languages.get(flores).copied()
    }

    /// Number of language tokens.
    #[cfg(test)]
    pub fn language_count(&self) -> usize {
        self.languages.len()
    }

    pub fn eos_id(&self) -> u32 {
        self.eos
    }

    /// Ids a decoder must never produce inside a translation.
    pub fn suppressed_ids(&self) -> &[u32] {
        &self.suppressed
    }

    /// Source ids for the encoder: `[src_lang] + pieces(clean_source(text)) + [</s>]`. Errors when the language
    /// token is missing.
    pub fn encode(&self, text: &str, src_flores: &str) -> Result<Vec<i64>, String> {
        let lang = self
            .lang_id(src_flores)
            .ok_or_else(|| format!("the tokenizer has no language token {src_flores}"))?;
        let pieces = self.pieces(text)?;
        let mut ids = Vec::with_capacity(pieces.len() + 2);
        ids.push(i64::from(lang));
        ids.extend(pieces.iter().map(|&id| i64::from(id)));
        ids.push(i64::from(self.eos));
        Ok(ids)
    }

    /// Token ids of the cleaned text, without special tokens.
    pub fn pieces(&self, text: &str) -> Result<Vec<u32>, String> {
        let cleaned = clean_source(text);
        if cleaned.is_empty() {
            return Ok(Vec::new());
        }
        let encoding = self
            .inner
            .encode_fast(cleaned.as_str(), false)
            .map_err(|e| format!("tokenizer: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Text of generated ids, skipping special tokens (`</s>`, language tokens, padding).
    pub fn decode(&self, ids: &[i64]) -> Result<String, String> {
        let ids: Vec<u32> = ids.iter().filter_map(|&id| u32::try_from(id).ok()).collect();
        let text = self.inner.decode(&ids, true).map_err(|e| format!("tokenizer: {e}"))?;
        Ok(text.trim().to_string())
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// A tiny NLLB-shaped `tokenizer.json` (word-level model, Metaspace, the NLLB special tokens, three language
/// tokens) for tests that need a tokenizer but no model.
#[cfg(test)]
pub(crate) fn test_tokenizer_json() -> String {
    let words = ["hello", "world", "good", "morning", "こんにちは", "世界", "bonjour", "le", "monde"];
    let mut vocab = serde_json::Map::new();
    for (i, t) in ["<s>", "<pad>", "</s>", "<unk>"].iter().enumerate() {
        vocab.insert((*t).to_string(), i.into());
    }
    for (i, w) in words.iter().enumerate() {
        vocab.insert(format!("\u{2581}{w}"), (4 + i).into());
    }
    let mut added = Vec::new();
    for (id, t) in [(0, "<s>"), (1, "<pad>"), (2, "</s>"), (3, "<unk>")] {
        added.push(serde_json::json!({"id": id, "content": t, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}));
    }
    for (i, lang) in ["eng_Latn", "fra_Latn", "jpn_Jpan"].iter().enumerate() {
        let id = 20 + i;
        vocab.insert((*lang).to_string(), id.into());
        added.push(serde_json::json!({"id": id, "content": lang, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}));
    }
    vocab.insert("<mask>".into(), 23.into());
    added.push(serde_json::json!({"id": 23, "content": "<mask>", "single_word": false, "lstrip": true, "rstrip": false, "normalized": true, "special": true}));
    serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "Metaspace", "replacement": "\u{2581}", "add_prefix_space": true},
        "post_processor": null,
        "decoder": {"type": "Metaspace", "replacement": "\u{2581}", "add_prefix_space": true},
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> NllbTokenizer {
        NllbTokenizer::from_bytes(test_tokenizer_json().as_bytes()).expect("the test tokenizer parses")
    }

    #[test]
    fn encodes_with_the_source_language_and_eos() {
        let t = tiny();
        assert_eq!(t.eos_id(), 2);
        assert_eq!(t.lang_id("fra_Latn"), Some(21));
        assert_eq!(t.language_count(), 3);
        assert_eq!(t.encode("hello world", "eng_Latn").unwrap(), [20, 4, 5, 2]);
        assert_eq!(t.encode("bonjour le monde", "fra_Latn").unwrap(), [21, 10, 11, 12, 2]);
        // Whitespace is cleaned first; unknown words become <unk>.
        assert_eq!(t.encode("  hello\t\u{3000}world  \n", "eng_Latn").unwrap(), [20, 4, 5, 2]);
        assert_eq!(t.encode("hello mars", "eng_Latn").unwrap(), [20, 4, 3, 2]);
        assert_eq!(t.encode("   ", "eng_Latn").unwrap(), [20, 2]);
        assert!(t.encode("hello", "deu_Latn").is_err(), "no such language token");
    }

    #[test]
    fn decodes_without_special_tokens() {
        let t = tiny();
        assert_eq!(t.decode(&[21, 10, 11, 12, 2]).unwrap(), "bonjour le monde");
        assert_eq!(t.decode(&[2, 22, 8, 9]).unwrap(), "こんにちは 世界");
        assert_eq!(t.decode(&[]).unwrap(), "");
        assert_eq!(t.decode(&[-5, 4]).unwrap(), "hello", "negative ids are dropped");
    }

    #[test]
    fn language_tokens_and_padding_are_suppressed() {
        let t = tiny();
        assert_eq!(t.suppressed_ids(), [0, 1, 20, 21, 22, 23]);
        assert!(!t.suppressed_ids().contains(&t.eos_id()));
    }

    #[test]
    fn rejects_non_nllb_tokenizers() {
        let json = test_tokenizer_json().replace("eng_Latn", "english").replace("fra_Latn", "french").replace("jpn_Jpan", "japanese");
        assert!(NllbTokenizer::from_bytes(json.as_bytes()).unwrap_err().contains("language tokens"));
        assert!(NllbTokenizer::from_bytes(b"{").is_err());
    }

    #[test]
    fn whitespace_cleaning_matches_sentencepiece() {
        assert_eq!(clean_source("  a \t b\u{3000}c\u{a0}d\n\ne  "), "a b c d e");
        assert_eq!(clean_source(""), "");
        assert_eq!(clean_source("zero\u{200b}width"), "zero\u{200b}width", "not whitespace for sentencepiece");
        assert!(is_flores_token("jpn_Jpan"));
        assert!(!is_flores_token("<mask>"));
        assert!(!is_flores_token("JPN_Jpan"));
    }

    /// `tokenizer.json` of the installed NLLB-600M (model store of `SIGIL_APP_DATA_DIR`), or `SIGIL_NLLB_DIR`.
    pub(crate) fn real_model_dir() -> Option<std::path::PathBuf> {
        if let Some(dir) = std::env::var_os("SIGIL_NLLB_DIR") {
            return Some(dir.into());
        }
        crate::services::model_store::installed_path(super::super::NLLB_DEFAULT_MODEL)
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with mt/nllb-200-distilled-600m-int8 installed (or SIGIL_NLLB_DIR)"]
    fn matches_python_sentencepiece() {
        let dir = real_model_dir().expect("NLLB-600M is not installed");
        let started = std::time::Instant::now();
        let t = NllbTokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
        eprintln!("tokenizer.json loaded in {} ms", started.elapsed().as_millis());
        assert_eq!(t.language_count(), 202);
        assert_eq!(t.lang_id("eng_Latn"), Some(256047));
        for l in super::super::flores::LANGUAGES {
            assert!(t.lang_id(l.flores).is_some(), "{} missing from tokenizer.json", l.flores);
        }
        let fixture: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../scripts/mt-bench/fixtures/nllb_parity.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        assert!(cases.len() >= 20);
        for case in cases {
            let text = case["text"].as_str().unwrap();
            let src = case["source"].as_str().unwrap();
            let want: Vec<i64> = case["ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
            assert_eq!(t.encode(text, src).unwrap(), want, "{text:?}");
            if let Some(out) = case.get("output_ids").and_then(|v| v.as_array()) {
                let out: Vec<i64> = out.iter().map(|v| v.as_i64().unwrap()).collect();
                assert_eq!(t.decode(&out).unwrap(), case["output_text"].as_str().unwrap().trim(), "{text:?}");
            }
        }
    }
}
