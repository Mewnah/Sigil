//! Hotword (contextual biasing) text for the model families that take it, and the `bpe.vocab`
//! sherpa-onnx needs to encode hotwords for NeMo transducers.
//!
//! - **Transducers** decoded with modified beam search (Parakeet TDT / Ultra, k2 zipformers) take
//!   one phrase per line; sherpa-onnx encodes each phrase into model tokens and boosts those paths
//!   (Aho-Corasick context graph). For SentencePiece models (NeMo: `modeling_unit = "bpe"`) the
//!   encoder needs a `bpe.vocab` (`piece<TAB>score`), which the official Parakeet v3 package
//!   predates. [`ensure_bpe_vocab`] derives it from `tokens.txt`: SentencePiece writes pieces in
//!   descending score order (BPE: the merge order), and sherpa's BPE encoder only compares scores,
//!   so `-id` reproduces the model's own segmentation.
//! - **Qwen3-ASR** takes context text in its system prompt through the per-stream `hotwords`
//!   option (comma-separated terms); [`qwen3_context`] also appends the free-form prompt.
//!
//! Phrases are user text: control characters, the separators of each format and runs of
//! whitespace are removed, duplicates dropped, and count and length capped ([`MAX_PHRASES`],
//! [`MAX_PHRASE_CHARS`]) so a pasted essay cannot blow up the context graph or the prompt.

use std::io::Write as _;
use std::path::{Path, PathBuf};

/// Most phrases passed to a model.
pub const MAX_PHRASES: usize = 100;
/// Longest phrase (characters).
pub const MAX_PHRASE_CHARS: usize = 64;
/// Longest Qwen3-ASR context (characters; the prompt costs decode time on every utterance).
pub const MAX_QWEN3_CONTEXT_CHARS: usize = 1_000;

/// Cleaned, de-duplicated phrases: no control characters, none of `forbidden`, single spaces.
pub fn clean_phrases(phrases: &[String], forbidden: &[char]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for p in phrases {
        let cleaned: String = p
            .chars()
            .map(|c| if c.is_control() || forbidden.contains(&c) { ' ' } else { c })
            .collect();
        let words: Vec<&str> = cleaned.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }
        let phrase: String = words.join(" ").chars().take(MAX_PHRASE_CHARS).collect();
        let phrase = phrase.trim().to_string();
        if phrase.is_empty() || out.iter().any(|o| o.eq_ignore_ascii_case(&phrase)) {
            continue;
        }
        out.push(phrase);
        if out.len() == MAX_PHRASES {
            break;
        }
    }
    out
}

/// Hotwords for a transducer stream: one phrase per line. sherpa-onnx turns `/` into a line
/// break and reads a trailing ` :<score>` as a per-phrase boost, so both are removed from user
/// text. `None` when nothing is left.
pub fn transducer_hotwords(phrases: &[String]) -> Option<String> {
    let v = clean_phrases(phrases, &['/', ':']);
    (!v.is_empty()).then(|| v.join("\n"))
}

/// Qwen3-ASR context: the terms comma-separated, then the free-form prompt. `None` when empty.
pub fn qwen3_context(phrases: &[String], prompt: Option<&str>) -> Option<String> {
    let mut parts = clean_phrases(phrases, &[',', '，', '、']);
    if let Some(p) = prompt {
        let p: String = p
            .chars()
            .map(|c| if c.is_control() { ' ' } else { c })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        if !p.is_empty() {
            parts.push(p);
        }
    }
    let joined = parts.join(", ");
    let joined: String = joined.chars().take(MAX_QWEN3_CONTEXT_CHARS).collect();
    let joined = joined.trim().trim_end_matches(',').trim().to_string();
    (!joined.is_empty()).then_some(joined)
}

/// Pieces of a sherpa `tokens.txt` (`<piece> <id>` per line; the piece may contain spaces only in
/// theory, so the id is the last field). Returns `(piece, id)` in file order.
pub fn read_tokens(text: &str) -> Vec<(String, i64)> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim_end_matches(['\r', '\n']);
            let (piece, id) = line.rsplit_once(' ')?;
            let id: i64 = id.trim().parse().ok()?;
            (!piece.is_empty()).then(|| (piece.to_string(), id))
        })
        .collect()
}

/// `bpe.vocab` text for SentencePiece `tokens.txt` content: every piece but the transducer blank
/// (`<blk>`, appended by the exporters) with score `-id` (`<unk>` and other `<...>` control pieces
/// get 0, as in SentencePiece). `Err` when the file has no pieces or is not a SentencePiece vocab.
pub fn bpe_vocab_from_tokens(tokens_txt: &str) -> Result<String, String> {
    let tokens = read_tokens(tokens_txt);
    if tokens.is_empty() {
        return Err("tokens.txt holds no tokens".into());
    }
    if !tokens.iter().any(|(p, _)| p.contains('\u{2581}')) {
        return Err("tokens.txt is not a SentencePiece vocabulary (no ▁ word pieces)".into());
    }
    let mut out = String::with_capacity(tokens_txt.len() + tokens.len() * 4);
    for (piece, id) in tokens {
        if piece == "<blk>" || piece == "<blank>" {
            continue;
        }
        let control = piece.len() > 2 && piece.starts_with('<') && piece.ends_with('>');
        let score = if control { 0.0 } else { -(id as f64) };
        out.push_str(&piece);
        out.push('\t');
        out.push_str(&format!("{score}"));
        out.push('\n');
    }
    Ok(out)
}

/// A `bpe.vocab` for the model whose `tokens.txt` is `tokens`: the package's own `bpe.vocab` next
/// to it when present, else one derived with [`bpe_vocab_from_tokens`] and cached under `cache_dir`
/// (named after the tokens file's size and content hash, written atomically, reused afterwards).
pub fn ensure_bpe_vocab(tokens: &Path, cache_dir: &Path) -> Result<PathBuf, String> {
    if let Some(own) = tokens
        .parent()
        .map(|d| d.join("bpe.vocab"))
        .filter(|p| p.is_file())
    {
        return Ok(own);
    }
    let text = std::fs::read_to_string(tokens).map_err(|e| format!("cannot read {}: {e}", tokens.display()))?;
    let digest = {
        use sha2::{Digest, Sha256};
        let h = Sha256::digest(text.as_bytes());
        h.iter()
            .take(8)
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    let path = cache_dir.join(format!("bpe-{}-{digest}.vocab", text.len()));
    if path.is_file() {
        return Ok(path);
    }
    let vocab = bpe_vocab_from_tokens(&text).map_err(|e| format!("{}: {e}", tokens.display()))?;
    std::fs::create_dir_all(cache_dir).map_err(|e| format!("cannot create {}: {e}", cache_dir.display()))?;
    let tmp = cache_dir.join(format!(".bpe-{}-{digest}.{}.tmp", text.len(), std::process::id()));
    let write = || -> std::io::Result<()> {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(vocab.as_bytes())?;
        f.sync_all()?;
        Ok(())
    };
    if let Err(e) = write() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("cannot write {}: {e}", tmp.display()));
    }
    if let Err(e) = std::fs::rename(&tmp, &path) {
        let _ = std::fs::remove_file(&tmp);
        // Another process may have written the same file meanwhile.
        if !path.is_file() {
            return Err(format!("cannot write {}: {e}", path.display()));
        }
    }
    Ok(path)
}

/// Where derived `bpe.vocab` files live: `<app_data>/cache/sherpa`, else the temp folder.
pub fn default_cache_dir() -> PathBuf {
    crate::services::diagnostics::app_data_dir()
        .map(|d| d.join("cache").join("sherpa"))
        .unwrap_or_else(|| std::env::temp_dir().join("sigil-sherpa-cache"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn phrases_are_cleaned_and_capped() {
        let v = clean_phrases(&s(&["  Sigil ", "sigil", "VR\nChat", "", "  ", "a/b"]), &['/']);
        assert_eq!(v, s(&["Sigil", "VR Chat", "a b"]));
        let many: Vec<String> = (0..500).map(|i| format!("w{i}")).collect();
        assert_eq!(clean_phrases(&many, &[]).len(), MAX_PHRASES);
        let long = "x".repeat(500);
        assert_eq!(clean_phrases(&[long], &[])[0].chars().count(), MAX_PHRASE_CHARS);
    }

    #[test]
    fn transducer_hotwords_are_one_per_line_without_separators() {
        assert_eq!(
            transducer_hotwords(&s(&["Sigil", "Mewnah :5.0", "left/right"])).as_deref(),
            Some("Sigil\nMewnah 5.0\nleft right")
        );
        assert_eq!(transducer_hotwords(&s(&[" ", "\n"])), None);
        assert_eq!(transducer_hotwords(&[]), None);
    }

    #[test]
    fn qwen3_context_joins_terms_and_prompt() {
        assert_eq!(
            qwen3_context(&s(&["Sigil", "VRChat, OBS", "猫、犬"]), Some("  A stream\nabout avatars ")).as_deref(),
            Some("Sigil, VRChat OBS, 猫 犬, A stream about avatars")
        );
        assert_eq!(qwen3_context(&[], Some("  ")), None);
        assert_eq!(qwen3_context(&[], None), None);
        let long = "y".repeat(5000);
        assert!(qwen3_context(&[], Some(&long)).unwrap().chars().count() <= MAX_QWEN3_CONTEXT_CHARS);
    }

    #[test]
    fn bpe_vocab_orders_pieces_by_id_and_skips_the_blank() {
        let tokens = "<unk> 0\n▁t 1\nhe 2\n▁the 3\ne 4\n<blk> 5\n";
        let v = bpe_vocab_from_tokens(tokens).unwrap();
        assert_eq!(v, "<unk>\t0\n▁t\t-1\nhe\t-2\n▁the\t-3\ne\t-4\n");
        assert!(bpe_vocab_from_tokens("").is_err());
        assert!(bpe_vocab_from_tokens("a 0\nb 1\n")
            .unwrap_err()
            .contains("SentencePiece"));
        assert_eq!(read_tokens("x y 3\nbad\n▁ 4\r\n"), vec![("x y".to_string(), 3), ("▁".to_string(), 4)]);
    }

    #[test]
    fn bpe_vocab_is_derived_once_and_prefers_the_package_file() {
        let root = std::env::temp_dir().join(format!("sigil-bpe-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let pkg = root.join("pkg");
        std::fs::create_dir_all(&pkg).unwrap();
        std::fs::write(pkg.join("tokens.txt"), "<unk> 0\n▁a 1\n<blk> 2\n").unwrap();
        let cache = root.join("cache");
        let p1 = ensure_bpe_vocab(&pkg.join("tokens.txt"), &cache).unwrap();
        assert!(p1.starts_with(&cache));
        assert_eq!(std::fs::read_to_string(&p1).unwrap(), "<unk>\t0\n▁a\t-1\n");
        let p2 = ensure_bpe_vocab(&pkg.join("tokens.txt"), &cache).unwrap();
        assert_eq!(p1, p2);
        std::fs::write(pkg.join("bpe.vocab"), "own").unwrap();
        assert_eq!(ensure_bpe_vocab(&pkg.join("tokens.txt"), &cache).unwrap(), pkg.join("bpe.vocab"));
        assert!(ensure_bpe_vocab(&root.join("missing.txt"), &cache).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }
}
