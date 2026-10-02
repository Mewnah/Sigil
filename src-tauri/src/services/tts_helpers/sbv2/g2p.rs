//! Japanese grapheme-to-phoneme for Style-Bert-VITS2: text → phonemes, accent tones and the
//! character ↔ phoneme map (`word2ph`) the BERT features are spread over.
//!
//! Ported from `sbv2_core`'s `jtalk.rs` / `tts_util.rs` (MIT, Copyright (c) 2024 tuna2134, 2025-
//! neodyland, <https://github.com/neodyland/sbv2-api>), which rewrote Style-Bert-VITS2's
//! `style_bert_vits2/nlp/japanese/g2p.py` (litagin02; LGPL-3.0) on `jpreprocess`, the pure-Rust
//! OpenJTalk. Where the Rust port and the Python differ, this follows the Python:
//! - a long-vowel mark with no vowel before it becomes the `-` symbol (the port kept `ー`, which is
//!   not a symbol);
//! - an unreadable word becomes one `'` per character (the port used bytes);
//! - non-JP-Extra models read `ん` as `n` (the Python's `use_jp_extra=False`).
//!
//! It is also more forgiving than both: the Python raises (and the port panics) on an empty
//! reading, an unexpected accent pattern, a phoneme the accent labels do not have, or a word whose
//! phonemes have no character to sit on. Here an empty reading counts as a pause, accents are
//! folded to 0 / 1, a mismatched phoneme gets a low tone and re-synchronises, and orphan phonemes
//! join the previous character, so a sentence always speaks. Every repair is logged at debug.
//!
//! Numbers become kanji numerals before the analysis ([`text::normalize_text`]), as the Python's
//! `num2words` does (`sbv2_core` lost digits there).

use std::path::Path;

use jpreprocess::{DefaultTokenizer, JPreprocess, SystemDictionaryConfig};

use super::mora::{KATA_TO_PHONEMES, MAX_MORA_CHARS, VOWELS};
use super::text::{self, all_punctuation, is_punctuation, PAD};

const LOG: &str = "sigil::tts::sbv2";

/// Files of a jpreprocess (lindera) dictionary folder.
const DICTIONARY_MARKERS: [&str; 3] = ["dict.da", "matrix.mtx", "char_def.bin"];

/// The dictionary folder inside `root`: `root` itself or one folder below it (an archive that kept
/// its top folder).
pub fn dictionary_dir(root: &Path) -> Option<std::path::PathBuf> {
    let is_dict = |d: &Path| DICTIONARY_MARKERS.iter().all(|f| d.join(f).is_file());
    if is_dict(root) {
        return Some(root.to_path_buf());
    }
    let mut dirs: Vec<_> = std::fs::read_dir(root).ok()?.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    dirs.into_iter().find(|d| is_dict(d))
}

/// What the acoustic model needs from one sentence.
#[derive(Clone, Debug, PartialEq)]
pub struct Analysis {
    /// Phonemes, with [`PAD`] at both ends (before blanks are added).
    pub phones: Vec<String>,
    /// Accent of each phoneme: 0 low, 1 high.
    pub tones: Vec<i64>,
    /// Phonemes per character of [`Self::bert_text`], plus one for each end.
    pub word2ph: Vec<usize>,
    /// The text the BERT reads (one token per character).
    pub bert_text: String,
}

/// Model inputs of an [`Analysis`] (SBV2's `add_blank`: a blank between and around phonemes).
#[derive(Clone, Debug, PartialEq)]
pub struct Sequence {
    pub phones: Vec<i64>,
    pub tones: Vec<i64>,
    pub languages: Vec<i64>,
    /// `word2ph` for the blanked sequence (doubled, the first one plus one).
    pub word2ph: Vec<usize>,
}

impl Analysis {
    /// The blanked id sequences for a JP-Extra (`jp_extra`) or older multilingual model.
    pub fn sequence(&self, jp_extra: bool) -> Sequence {
        let ids: Vec<i64> = self
            .phones
            .iter()
            .map(|p| text::symbol_id(if !jp_extra && p == "N" { "n" } else { p.as_str() }))
            .collect();
        let tones: Vec<i64> = self.tones.iter().map(|t| t + text::JP_TONE_START).collect();
        let languages = vec![text::JP_LANGUAGE_ID; ids.len()];
        let mut word2ph: Vec<usize> = self.word2ph.iter().map(|n| n * 2).collect();
        if let Some(first) = word2ph.first_mut() {
            *first += 1;
        }
        Sequence {
            phones: text::intersperse(&ids, 0),
            tones: text::intersperse(&tones, 0),
            languages: text::intersperse(&languages, 0),
            word2ph,
        }
    }
}

/// One accent-label row (what the prosody marks need).
#[derive(Clone, Debug, PartialEq)]
struct Lab {
    /// Current phoneme (`sil`, `pau`, `a`, `k`, ...).
    p3: String,
    a1: i32,
    a2: i32,
    a3: i32,
    f1: i32,
    /// The previous accent phrase is a question (E3; read at the final `sil`).
    e3: bool,
}

/// The Japanese text frontend (OpenJTalk on a NAIST-JDIC dictionary).
pub struct Frontend {
    jp: JPreprocess<DefaultTokenizer>,
}

fn jp_err(e: impl std::fmt::Display) -> String {
    format!("Japanese text analysis failed: {e}")
}

impl Frontend {
    /// Load the jpreprocess dictionary in `dictionary` (the folder, or a folder holding it).
    pub fn load(dictionary: &Path) -> Result<Self, String> {
        let dir = dictionary_dir(dictionary)
            .ok_or_else(|| format!("{} does not hold a Japanese dictionary (dict.da, matrix.mtx, char_def.bin)", dictionary.display()))?;
        let system = SystemDictionaryConfig::File(dir.clone())
            .load()
            .map_err(|e| format!("cannot load the Japanese dictionary in {}: {e}", dir.display()))?;
        Ok(Self { jp: JPreprocess::with_dictionaries(system, None) })
    }

    /// `text` with every word that holds kanji or digits replaced by its katakana reading
    /// (OpenJTalk's, unvoiced marks removed); everything else is kept (NFKC-normalized). For TTS
    /// frontends without a Japanese dictionary (sherpa-onnx's Kokoro reads kanji with Mandarin
    /// readings): they then read the kana.
    pub fn kana_reading(&self, text: &str) -> Result<String, String> {
        let text = &text::read_signs(text);
        let mut njd = self.jp.text_to_njd(text).map_err(jp_err)?;
        njd.preprocess();
        let mut out = String::with_capacity(text.len() * 2);
        for node in &njd.nodes {
            let surface = node.get_string();
            let needs_reading = surface.chars().any(|c| is_kanji(c) || c.is_ascii_digit() || ('０'..='９').contains(&c));
            let pron = node.get_pron().to_string().replace('’', "");
            if needs_reading && !pron.is_empty() && pron != "、" && pron.chars().all(|c| ('\u{30A0}'..='\u{30FF}').contains(&c)) {
                out.push_str(&pron);
            } else {
                // jpreprocess widened ASCII for its dictionary; give the text back its own form.
                out.push_str(&text::nfkc(surface));
            }
        }
        Ok(out)
    }

    /// Analyse one sentence. `Ok` with no phonemes besides the two pads when nothing is readable.
    pub fn analyze(&self, raw: &str) -> Result<Analysis, String> {
        let normalized = text::normalize_text(raw);
        if normalized.is_empty() {
            return Ok(Analysis { phones: vec![PAD.into(), PAD.into()], tones: vec![0, 0], word2ph: vec![1, 1], bert_text: String::new() });
        }

        let mut njd = self.jp.text_to_njd(&normalized).map_err(jp_err)?;
        njd.preprocess();
        // Word string and pronunciation (unvoiced marks removed) of every node.
        let words: Vec<(String, String)> = njd
            .nodes
            .iter()
            .map(|n| (n.get_string().to_string(), n.get_pron().to_string().replace('’', "")))
            .collect();
        let labels = self.jp.make_label(Vec::<String>::from(njd));
        let labs: Vec<Lab> = labels
            .iter()
            .map(|l| Lab {
                p3: l.phoneme.c.clone().unwrap_or_default(),
                a1: l.mora.as_ref().map_or(-50, |m| i32::from(m.relative_accent_position)),
                a2: l.mora.as_ref().map_or(-50, |m| i32::from(m.position_forward)),
                a3: l.mora.as_ref().map_or(-50, |m| i32::from(m.position_backward)),
                f1: l.accent_phrase_curr.as_ref().map_or(-50, |a| i32::from(a.mora_count)),
                e3: l.accent_phrase_prev.as_ref().is_some_and(|a| a.is_interrogative),
            })
            .collect();

        let prosody = prosody_marks(&labs);
        let tones_wo_punct = phone_tones_without_punctuation(&prosody);
        let (seq_text, seq_kata) = sep_text_and_kata(&words);
        let sep_phonemes = handle_long(seq_kata.iter().map(|k| kata_to_phonemes(k)).collect());
        let phone_w_punct: Vec<String> = sep_phonemes.iter().flatten().cloned().collect();
        let phone_tones = align_tones(&phone_w_punct, &tones_wo_punct);

        // Phonemes per character, word by word (spread evenly over the word's characters).
        let mut word2ph: Vec<usize> = vec![1];
        let mut orphans = 0usize;
        for (word, phonemes) in seq_text.iter().zip(&sep_phonemes) {
            let chars = word.chars().count();
            let n = phonemes.len() + std::mem::take(&mut orphans);
            if chars == 0 {
                if n > 0 {
                    // Phonemes without a character (the Python fails here): they join the previous
                    // character, or the next one at the very start.
                    if word2ph.len() > 1 {
                        *word2ph.last_mut().expect("non-empty") += n;
                    } else {
                        orphans = n;
                    }
                    tracing::debug!(target: LOG, "sbv2 g2p: {n} phonemes of {word:?} have no character");
                }
                continue;
            }
            word2ph.extend(distribute_phone(n, chars));
        }
        if orphans > 0 {
            // Only orphans and no character at all: they sit on the start pad.
            word2ph[0] += orphans;
        }
        word2ph.push(1);

        let mut phones = Vec::with_capacity(phone_tones.len() + 2);
        let mut tones = Vec::with_capacity(phone_tones.len() + 2);
        phones.push(PAD.to_string());
        tones.push(0);
        for (p, t) in phone_tones {
            phones.push(p);
            tones.push(t);
        }
        phones.push(PAD.to_string());
        tones.push(0);

        let bert_text: String = seq_text.concat();
        let analysis = Analysis { phones, tones, word2ph, bert_text };
        check(&analysis)?;
        Ok(analysis)
    }
}

/// A CJK ideograph (kanji), including the iteration mark `々`.
fn is_kanji(c: char) -> bool {
    matches!(c, '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}' | '\u{F900}'..='\u{FAFF}' | '\u{20000}'..='\u{2FFFF}' | '々')
}

/// The invariants the model inputs rely on.
fn check(a: &Analysis) -> Result<(), String> {
    let chars = a.bert_text.chars().count();
    if a.word2ph.len() != chars + 2 {
        return Err(format!("internal text alignment error ({} entries for {} characters)", a.word2ph.len(), chars));
    }
    let total: usize = a.word2ph.iter().sum();
    if total != a.phones.len() || a.tones.len() != a.phones.len() {
        return Err(format!("internal phoneme alignment error ({total} mapped, {} phonemes)", a.phones.len()));
    }
    Ok(())
}

/// Word strings (punctuation folded) and their katakana readings (`、` readings resolved: a run of
/// punctuation keeps itself, an unreadable word becomes one `'` per character, `？` becomes `?`).
fn sep_text_and_kata(words: &[(String, String)]) -> (Vec<String>, Vec<String>) {
    let mut seq_text = Vec::with_capacity(words.len());
    let mut seq_kata = Vec::with_capacity(words.len());
    for (string, pron) in words {
        let word = text::replace_punctuation(string);
        let mut yomi = pron.clone();
        if !yomi.chars().any(|c| ('\u{30A0}'..='\u{30FF}').contains(&c)) {
            // No katakana reading: punctuation keeps itself (`？` read as `?`); any other word
            // OpenJTalk cannot read (`、`, `*`, an empty reading) becomes one pause per character.
            let folded = text::replace_punctuation(&yomi);
            yomi = if !word.is_empty() && all_punctuation(&word) {
                word.clone()
            } else if yomi != "、" && !folded.is_empty() && all_punctuation(&folded) {
                folded
            } else {
                if !word.is_empty() {
                    tracing::debug!(target: LOG, "sbv2 g2p: cannot read {word:?}; it becomes a pause");
                }
                "'".repeat(word.chars().count())
            };
        }
        seq_text.push(word);
        seq_kata.push(yomi);
    }
    (seq_text, seq_kata)
}

/// OpenJTalk accent labels → phonemes with prosody marks (`^` start, `$` / `?` end, `_` pause,
/// `#` accent-phrase border, `[` pitch rise, `]` pitch fall), as Style-Bert-VITS2's
/// `pyopenjtalk_g2p_prosody`.
fn prosody_marks(labs: &[Lab]) -> Vec<String> {
    let n = labs.len();
    let mut out = Vec::with_capacity(n * 2);
    for (i, lab) in labs.iter().enumerate() {
        let mut p3 = lab.p3.clone();
        if matches!(p3.as_str(), "A" | "E" | "I" | "O" | "U") {
            // Unvoiced vowels are read as normal vowels.
            p3 = p3.to_lowercase();
        }
        match p3.as_str() {
            "sil" => {
                if i == 0 {
                    out.push("^".to_string());
                } else if i == n - 1 {
                    out.push(if lab.e3 { "?" } else { "$" }.to_string());
                }
                continue;
            }
            "pau" => {
                out.push("_".to_string());
                continue;
            }
            "" => continue,
            _ => out.push(p3.clone()),
        }
        let a2_next = labs.get(i + 1).map_or(-50, |l| l.a2);
        if lab.a3 == 1 && a2_next == 1 && matches!(p3.as_str(), "a" | "e" | "i" | "o" | "u" | "N" | "cl") {
            out.push("#".to_string());
        } else if lab.a1 == 0 && a2_next == lab.a2 + 1 && lab.a2 != lab.f1 {
            out.push("]".to_string());
        } else if lab.a2 == 1 && a2_next == 2 {
            out.push("[".to_string());
        }
    }
    out
}

/// Accent of one phrase folded to 0 (low) / 1 (high): the lowest value becomes 0, every other 1
/// (the Python accepts only {0}, {0, 1} and {-1, 0}, which this maps the same way).
fn fix_phone_tone(phrase: &[(String, i64)]) -> Vec<(String, i64)> {
    let Some(min) = phrase.iter().map(|(_, t)| *t).min() else {
        return Vec::new();
    };
    let max = phrase.iter().map(|(_, t)| *t).max().unwrap_or(min);
    if max - min > 1 || (max == min && min != 0) {
        tracing::debug!(target: LOG, "sbv2 g2p: unexpected accent values {min}..{max}; folded to low / high");
    }
    phrase.iter().map(|(p, t)| (p.clone(), i64::from(*t > min))).collect()
}

/// Phonemes and accents without punctuation, phrase by phrase (`cl` → `q`).
fn phone_tones_without_punctuation(prosody: &[String]) -> Vec<(String, i64)> {
    let mut result = Vec::new();
    let mut phrase: Vec<(String, i64)> = Vec::new();
    let mut tone = 0i64;
    for letter in prosody {
        match letter.as_str() {
            "^" => {}
            "$" | "?" | "_" | "#" => {
                result.extend(fix_phone_tone(&phrase));
                phrase.clear();
                tone = 0;
            }
            "[" => tone += 1,
            "]" => tone -= 1,
            "cl" => phrase.push(("q".to_string(), tone)),
            other => phrase.push((other.to_string(), tone)),
        }
    }
    // A label list without its final `sil` still keeps its last phrase.
    result.extend(fix_phone_tone(&phrase));
    result
}

/// Katakana reading → phonemes (greedy longest mora match, then `ー` repeats the vowel before it).
/// A reading made only of punctuation is returned character by character.
fn kata_to_phonemes(kata: &str) -> Vec<String> {
    if all_punctuation(kata) {
        return kata.chars().map(|c| c.to_string()).collect();
    }
    let chars: Vec<char> = kata.chars().collect();
    let mut out: Vec<String> = Vec::with_capacity(chars.len() * 2);
    let mut i = 0;
    'scan: while i < chars.len() {
        for len in (1..=(*MAX_MORA_CHARS).min(chars.len() - i)).rev() {
            let key: String = chars[i..i + len].iter().collect();
            if let Some((consonant, vowel)) = KATA_TO_PHONEMES.get(&key) {
                if let Some(c) = consonant {
                    out.push(c.clone());
                }
                out.push(vowel.clone());
                i += len;
                continue 'scan;
            }
        }
        let c = chars[i];
        if c == 'ー' {
            // `ー` repeats the last letter before it (Python's `(\w)(ー*)` rule).
            match out.last().and_then(|p| p.chars().last()) {
                Some(prev) if prev.is_alphanumeric() => out.push(prev.to_string()),
                _ => out.push("ー".to_string()),
            }
        } else {
            tracing::debug!(target: LOG, "sbv2 g2p: no phoneme for {c:?}");
            out.push(c.to_string());
        }
        i += 1;
    }
    out
}

/// Long-vowel marks at the start of a word: the previous word's vowel (or `N`), else `-`; any
/// other `ー` repeats the letter before it.
fn handle_long(mut sep: Vec<Vec<String>>) -> Vec<Vec<String>> {
    for i in 0..sep.len() {
        if sep[i].is_empty() {
            continue;
        }
        if sep[i][0] == "ー" {
            let prev = sep[..i].iter().rev().find_map(|w| w.last()).cloned();
            sep[i][0] = match prev {
                Some(p) if VOWELS.contains(&p.as_str()) => p,
                // `。ーー` and the like: a dash mistaken for a long-vowel mark.
                _ => "-".to_string(),
            };
        }
        for j in 1..sep[i].len() {
            if sep[i][j] == "ー" {
                let prev = sep[i][j - 1].chars().last().map(|c| c.to_string()).unwrap_or_else(|| "-".to_string());
                sep[i][j] = prev;
            }
        }
    }
    sep
}

/// Phonemes with punctuation, each with the accent of the matching phoneme of the label-derived
/// list (punctuation and phonemes past its end get 0).
fn align_tones(with_punct: &[String], tones: &[(String, i64)]) -> Vec<(String, i64)> {
    /// How far ahead a mismatched phoneme may find its label again.
    const RESYNC: usize = 3;
    let mut out = Vec::with_capacity(with_punct.len());
    let mut k = 0usize;
    for phone in with_punct {
        if k < tones.len() && *phone == tones[k].0 {
            out.push((phone.clone(), tones[k].1));
            k += 1;
        } else if is_punctuation(phone) || k >= tones.len() {
            out.push((phone.clone(), 0));
        } else if let Some(skip) = (1..=RESYNC).find(|d| tones.get(k + d).is_some_and(|(p, _)| p == phone)) {
            tracing::debug!(target: LOG, "sbv2 g2p: accent labels skipped {skip} phonemes before {phone:?}");
            k += skip;
            out.push((phone.clone(), tones[k].1));
            k += 1;
        } else {
            tracing::debug!(target: LOG, "sbv2 g2p: {phone:?} has no accent label; low tone");
            out.push((phone.clone(), 0));
        }
    }
    out
}

/// Spread `n_phone` phonemes over `n_word` characters as evenly as possible (earlier characters
/// first), as Style-Bert-VITS2's `distribute_phone`.
fn distribute_phone(n_phone: usize, n_word: usize) -> Vec<usize> {
    if n_word == 0 {
        return Vec::new();
    }
    let base = n_phone / n_word;
    let extra = n_phone % n_word;
    (0..n_word).map(|i| base + usize::from(i < extra)).collect()
}

/// The jpreprocess NAIST-JDIC folder for the real-dictionary tests:
/// `$SIGIL_TEST_MODELS_DIR/naist-jdic` (from jpreprocess v0.15.0's `naist-jdic-jpreprocess.tar.gz`).
#[cfg(test)]
pub(crate) fn test_dictionary() -> Option<std::path::PathBuf> {
    let root = std::env::var_os("SIGIL_TEST_MODELS_DIR")?;
    dictionary_dir(&std::path::PathBuf::from(root).join("naist-jdic"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn katakana_becomes_phonemes() {
        assert_eq!(kata_to_phonemes("コンニチワ"), s(&["k", "o", "N", "n", "i", "ch", "i", "w", "a"]));
        assert_eq!(kata_to_phonemes("キャット"), s(&["ky", "a", "q", "t", "o"]));
        // Long-vowel marks repeat the vowel before them.
        assert_eq!(kata_to_phonemes("ラーメン"), s(&["r", "a", "a", "m", "e", "N"]));
        assert_eq!(kata_to_phonemes("ゴーーール"), s(&["g", "o", "o", "o", "o", "r", "u"]));
        // A leading mark stays for handle_long; punctuation passes through.
        assert_eq!(kata_to_phonemes("ーア"), s(&["ー", "a"]));
        assert_eq!(kata_to_phonemes("..."), s(&[".", ".", "."]));
        assert_eq!(kata_to_phonemes("''"), s(&["'", "'"]));
    }

    #[test]
    fn long_marks_at_word_starts_follow_the_python() {
        let out = handle_long(vec![s(&["k", "a"]), s(&["ー", "t", "o"]), s(&["."]), s(&["ー"])]);
        assert_eq!(out, vec![s(&["k", "a"]), s(&["a", "t", "o"]), s(&["."]), s(&["-"])]);
        // An empty word before is skipped when looking for the vowel.
        let out = handle_long(vec![s(&["s", "a", "N"]), vec![], s(&["ー"])]);
        assert_eq!(out[2], s(&["N"]));
        assert_eq!(handle_long(vec![s(&["ー", "ー"])]), vec![s(&["-", "-"])]);
    }

    #[test]
    fn phonemes_are_spread_over_characters_like_style_bert_vits2() {
        assert_eq!(distribute_phone(5, 2), [3, 2]);
        assert_eq!(distribute_phone(2, 3), [1, 1, 0]);
        assert_eq!(distribute_phone(0, 2), [0, 0]);
        assert_eq!(distribute_phone(4, 0), Vec::<usize>::new());
        // Same as the Python's "give the next phoneme to the first emptiest word" loop.
        for (np, nw) in [(7usize, 3usize), (1, 4), (9, 9), (10, 4)] {
            let mut per = vec![0usize; nw];
            for _ in 0..np {
                let min = *per.iter().min().unwrap();
                let at = per.iter().position(|&x| x == min).unwrap();
                per[at] += 1;
            }
            assert_eq!(distribute_phone(np, nw), per);
        }
    }

    #[test]
    fn accents_are_folded_to_low_and_high() {
        let p = |v: &[(&str, i64)]| v.iter().map(|(a, b)| (a.to_string(), *b)).collect::<Vec<_>>();
        assert_eq!(fix_phone_tone(&p(&[("a", 0), ("i", 1)])), p(&[("a", 0), ("i", 1)]));
        assert_eq!(fix_phone_tone(&p(&[("a", 0), ("i", -1)])), p(&[("a", 1), ("i", 0)]));
        assert_eq!(fix_phone_tone(&p(&[("a", 0)])), p(&[("a", 0)]));
        assert!(fix_phone_tone(&[]).is_empty());
        // Unexpected patterns do not fail the sentence.
        assert_eq!(fix_phone_tone(&p(&[("a", 1)])), p(&[("a", 0)]));
        assert_eq!(fix_phone_tone(&p(&[("a", -1), ("i", 0), ("u", 1)])), p(&[("a", 0), ("i", 1), ("u", 1)]));
    }

    #[test]
    fn prosody_marks_become_phrase_tones() {
        // ^ k o [ N n i ch i w a $ : low first mora, high after the rise.
        let prosody = s(&["^", "k", "o", "[", "N", "n", "i", "ch", "i", "w", "a", "$"]);
        let tones = phone_tones_without_punctuation(&prosody);
        let only: Vec<i64> = tones.iter().map(|(_, t)| *t).collect();
        assert_eq!(only, [0, 0, 1, 1, 1, 1, 1, 1, 1]);
        // `cl` is the small tsu; a fall then a new phrase.
        let prosody = s(&["^", "a", "]", "cl", "t", "a", "#", "k", "a", "?"]);
        let tones = phone_tones_without_punctuation(&prosody);
        assert_eq!(tones.iter().map(|(p, _)| p.as_str()).collect::<Vec<_>>(), ["a", "q", "t", "a", "k", "a"]);
        assert_eq!(tones.iter().map(|(_, t)| *t).collect::<Vec<_>>(), [1, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn tones_align_around_punctuation_and_resync() {
        let tones = vec![("k".to_string(), 0), ("a".to_string(), 1), ("t".to_string(), 1), ("a".to_string(), 0)];
        let out = align_tones(&s(&["k", "a", ",", "t", "a", "!"]), &tones);
        assert_eq!(out.iter().map(|(_, t)| *t).collect::<Vec<_>>(), [0, 1, 0, 1, 0, 0]);
        // A phoneme the labels lack gets a low tone, and the next ones still match.
        let out = align_tones(&s(&["k", "x", "a", "t", "a"]), &tones);
        assert_eq!(out.iter().map(|(_, t)| *t).collect::<Vec<_>>(), [0, 0, 1, 1, 0]);
        // Labels with an extra phoneme: re-synchronise ahead.
        let out = align_tones(&s(&["k", "t", "a"]), &tones);
        assert_eq!(out.iter().map(|(_, t)| *t).collect::<Vec<_>>(), [0, 1, 0]);
    }

    #[test]
    fn readings_of_punctuation_and_unreadable_words() {
        let words = vec![
            ("今日".to_string(), "キョー".to_string()),
            ("、".to_string(), "、".to_string()),
            ("ＡＢ".to_string(), "、".to_string()),
            ("？".to_string(), "？".to_string()),
            ("！".to_string(), "！".to_string()),
            ("★".to_string(), "".to_string()),
            ("ＸＹ".to_string(), "*".to_string()),
        ];
        let (text, kata) = sep_text_and_kata(&words);
        assert_eq!(text, s(&["今日", ",", "ＡＢ", "?", "!", "", "ＸＹ"]));
        assert_eq!(kata, s(&["キョー", ",", "''", "?", "!", "", "''"]));
    }

    #[test]
    fn sequences_add_blanks_and_offsets() {
        let a = Analysis {
            phones: s(&["_", "k", "a", "N", "_"]),
            tones: vec![0, 0, 1, 1, 0],
            word2ph: vec![1, 3, 1],
            bert_text: "缶".into(),
        };
        let seq = a.sequence(true);
        assert_eq!(seq.phones.len(), 11);
        assert_eq!(seq.phones[1], text::symbol_id("_"));
        assert_eq!(seq.phones[3], text::symbol_id("k"));
        assert_eq!(seq.phones[7], text::symbol_id("N"));
        assert_eq!(seq.tones, [0, 6, 0, 6, 0, 7, 0, 7, 0, 6, 0]);
        assert!(seq.languages.iter().enumerate().all(|(i, l)| *l == if i % 2 == 1 { 1 } else { 0 }));
        assert_eq!(seq.word2ph, [3, 6, 2]);
        assert_eq!(seq.word2ph.iter().sum::<usize>(), seq.phones.len());
        // Older multilingual models read ん as n.
        assert_eq!(a.sequence(false).phones[7], text::symbol_id("n"));
    }

    #[test]
    fn dictionary_folders_are_found_one_level_down() {
        let root = std::env::temp_dir().join(format!("sigil-sbv2-dict-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let inner = root.join("naist-jdic");
        std::fs::create_dir_all(&inner).unwrap();
        assert_eq!(dictionary_dir(&root), None);
        for f in DICTIONARY_MARKERS {
            std::fs::write(inner.join(f), b"x").unwrap();
        }
        assert_eq!(dictionary_dir(&root), Some(inner.clone()));
        assert_eq!(dictionary_dir(&inner), Some(inner.clone()));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR with naist-jdic (jpreprocess v0.15.0)"]
    fn real_dictionary_reads_japanese() {
        let dict = test_dictionary().expect("SIGIL_TEST_MODELS_DIR/naist-jdic");
        let fe = Frontend::load(&dict).unwrap();
        let a = fe.analyze("こんにちは、世界！").unwrap();
        assert_eq!(a.bert_text, "こんにちは,世界!");
        let phones: Vec<&str> = a.phones.iter().map(String::as_str).collect();
        assert_eq!(phones, ["_", "k", "o", "N", "n", "i", "ch", "i", "w", "a", ",", "s", "e", "k", "a", "i", "!", "_"]);
        // こんにちは: low first mora, then high (heiban).
        assert_eq!(&a.tones[1..10], &[0, 0, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(a.word2ph.len(), a.bert_text.chars().count() + 2);
        assert_eq!(a.word2ph.iter().sum::<usize>(), a.phones.len());

        // Numbers are read (OpenJTalk's digit rules), currency signs follow their amount.
        let n = fe.analyze("$100です").unwrap();
        assert!(n.bert_text.starts_with("百ドル"), "{}", n.bert_text);
        // A question ends with `?`; long vowels and small tsu.
        let q = fe.analyze("ラーメン食べたっけ？").unwrap();
        assert!(q.phones.contains(&"q".to_string()));
        assert_eq!(q.phones[q.phones.len() - 2], "?");
        // Unreadable text still analyses (emoji and Latin are dropped or become pauses).
        let e = fe.analyze("🎉 hello …").unwrap();
        assert_eq!(e.word2ph.iter().sum::<usize>(), e.phones.len());
        let empty = fe.analyze("🎉").unwrap();
        assert_eq!(empty.phones, ["_", "_"]);
    }

    #[test]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR with naist-jdic (jpreprocess v0.15.0)"]
    fn real_dictionary_gives_kana_readings() {
        let fe = Frontend::load(&test_dictionary().expect("SIGIL_TEST_MODELS_DIR/naist-jdic")).unwrap();
        // Kanji and numbers become katakana; kana, Latin text and punctuation keep their form.
        assert_eq!(fe.kana_reading("今日は良い天気ですね。").unwrap(), "キョーはヨイテンキですね。");
        let mixed = fe.kana_reading("Sigilで100円!").unwrap();
        assert!(fe.kana_reading("$5").unwrap().contains("ドル"));
        assert!(mixed.starts_with("Sigilで"), "{mixed}");
        assert!(mixed.contains("ヒャクエン"), "{mixed}");
        assert!(mixed.ends_with('!'), "{mixed}");
        assert_eq!(fe.kana_reading("hello world").unwrap(), "hello world");
    }
}
