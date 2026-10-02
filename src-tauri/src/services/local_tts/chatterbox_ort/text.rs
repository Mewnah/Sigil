//! Chatterbox Multilingual's text frontend: what `ChatterboxMultilingualTTS.generate` and its
//! `MTLTokenizer` do to a sentence before the language model sees it, ported from Resemble AI's
//! reference implementation (MIT) with the KitsuMate V3 export's language companions:
//!
//! 1. `punc_norm`: collapse whitespace, replace punctuation the training data rarely had
//!    (`...` → `, `, `:` → `,`, curly quotes → straight ones, ...) and end the sentence with a full
//!    stop when it has no closing punctuation.
//! 2. Lower case, then Unicode NFKD.
//! 3. Per language: Japanese kanji → hiragana readings (longest match in `japanese_readings.tsv`,
//!    UniDic readings, in place of pykakasi), Chinese → jieba word segmentation plus Cangjie codes
//!    (`[cj_x]` tokens from `Cangjie5_TC.json`), Russian → stress marks (U+0301 after the stressed
//!    vowel of words in `russian_stress.tsv`, unambiguous forms only). Korean is already decomposed
//!    into jamo by NFKD; Hebrew gets no diacritics (the reference implementation also skips them when
//!    its diacritizer is missing).
//! 4. Prefix the language tag (`[ja]`), spell spaces as `[SPACE]`, and tokenize with the model's
//!    grapheme BPE (`tokenizer.json`).
//! 5. Frame the ids as the exported embedding graph expects: `[EXAGGERATION] [START] text [STOP]
//!    [START_SPEECH] [START_SPEECH]` (the emotion slot, the text, and the two speech-start positions
//!    of the reference implementation's prompt).

use std::cell::OnceCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The languages Chatterbox Multilingual V3 speaks (its language tags).
pub const LANGUAGES: [&str; 23] = [
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
];

/// The emotion-exaggeration slot (its embedding is computed from the `exaggeration` input).
pub const EXAGGERATION_TOKEN: i64 = 6563;
/// Text start / stop tokens.
pub const START_TEXT: i64 = 255;
pub const STOP_TEXT: i64 = 0;
/// Speech start / stop tokens; speech tokens below [`SPEECH_VOCAB`] are audio.
pub const START_SPEECH: i64 = 6561;
pub const STOP_SPEECH: i64 = 6562;
pub const SPEECH_VOCAB: i64 = 6561;
/// Most text tokens of one language-model run (the text position table has 2050 rows). Longer
/// pieces are split (`chatterbox_ort::speak_text`), which also bounds the cache: a Chinese
/// character is about six tokens (its Cangjie code).
pub const MAX_TEXT_TOKENS: usize = 600;
/// Start of the error [`Frontend::encode`] returns for a piece over [`MAX_TEXT_TOKENS`].
pub const TOO_LONG: &str = "the sentence is too long";

const SPACE_TOKEN: &str = "[SPACE]";

/// The supported language of a BCP-47 tag (`pt-BR` → `pt`, `nb` → `no`, `zh-Hant` → `zh`).
pub fn normalize_language(tag: &str) -> Option<&'static str> {
    let lower = tag.trim().to_ascii_lowercase().replace('_', "-");
    let base = lower.split('-').next().unwrap_or("");
    let base = match base {
        "nb" | "nn" => "no",
        "iw" => "he",
        "cmn" | "yue" | "wuu" => "zh",
        "msa" | "zsm" => "ms",
        "swa" => "sw",
        other => other,
    };
    LANGUAGES.iter().copied().find(|l| *l == base)
}

fn is_han(c: char) -> bool {
    matches!(c as u32,
        0x3005 | 0x3007 | 0x303B
        | 0x3400..=0x4DBF
        | 0x4E00..=0x9FFF
        | 0xF900..=0xFAFF
        | 0x20000..=0x3134F)
}

fn is_kana(c: char) -> bool {
    matches!(c as u32, 0x3041..=0x309F | 0x30A0..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F)
}

fn is_hangul(c: char) -> bool {
    matches!(c as u32, 0x1100..=0x11FF | 0x3130..=0x318F | 0xAC00..=0xD7AF)
}

/// The language a sentence's script shows, when one script is most of its letters: kana (with or
/// without kanji) → `ja`, Han → `zh`, Hangul → `ko`, Cyrillic → `ru`, Arabic → `ar`, Hebrew →
/// `he`, Greek → `el`, Devanagari → `hi`. `None` for Latin text (the letters do not tell English
/// from Spanish) and text without letters. A CJK character counts like a word.
pub fn script_language(text: &str) -> Option<&'static str> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum S {
        Kana,
        Han,
        Hangul,
        Cyrillic,
        Arabic,
        Hebrew,
        Greek,
        Devanagari,
        Other,
    }
    let script = |c: char| -> S {
        if is_kana(c) {
            S::Kana
        } else if is_han(c) {
            S::Han
        } else if is_hangul(c) {
            S::Hangul
        } else {
            match c as u32 {
                0x0400..=0x052F => S::Cyrillic,
                0x0600..=0x06FF | 0x0750..=0x077F | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => S::Arabic,
                0x0590..=0x05FF => S::Hebrew,
                0x0370..=0x03FF | 0x1F00..=0x1FFF => S::Greek,
                0x0900..=0x097F => S::Devanagari,
                _ => S::Other,
            }
        }
    };
    let mut counts: HashMap<u8, usize> = HashMap::new();
    let key = |s: S| s as u8;
    let mut run: Option<S> = None;
    for c in text.chars() {
        if !c.is_alphabetic() {
            run = None;
            continue;
        }
        let s = script(c);
        // CJK: one character is about one word; other scripts: one word per run of letters.
        if matches!(s, S::Kana | S::Han | S::Hangul) || run != Some(s) {
            *counts.entry(key(s)).or_default() += 1;
        }
        run = Some(s);
    }
    let total: usize = counts.values().sum();
    if total == 0 {
        return None;
    }
    let n = |s: S| counts.get(&key(s)).copied().unwrap_or(0);
    if n(S::Kana) > 0 && (n(S::Kana) + n(S::Han)) * 2 > total {
        return Some("ja");
    }
    let pick = [
        (S::Han, "zh"),
        (S::Hangul, "ko"),
        (S::Cyrillic, "ru"),
        (S::Arabic, "ar"),
        (S::Hebrew, "he"),
        (S::Greek, "el"),
        (S::Devanagari, "hi"),
    ];
    pick.into_iter()
        .find(|(s, _)| n(*s) * 2 > total)
        .map(|(_, l)| l)
}

/// The language a sentence is spoken in: the requested tag when Chatterbox speaks it, else what
/// the script shows, else English.
pub fn resolve_language(requested: Option<&str>, text: &str) -> &'static str {
    requested
        .and_then(normalize_language)
        .or_else(|| script_language(text))
        .unwrap_or("en")
}

/// Resemble AI's `punc_norm` (see the module docs); `""` for a text without content.
pub fn punc_norm(text: &str) -> String {
    let mut t = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if t.is_empty() {
        return t;
    }
    const REPLACE: [(&str, &str); 12] = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ];
    for (from, to) in REPLACE {
        t = t.replace(from, to);
    }
    let mut t = t.trim_end_matches(' ').to_string();
    const ENDERS: [char; 10] = ['.', '!', '?', '-', ',', '、', '，', '。', '？', '！'];
    if !t.ends_with(ENDERS) {
        t.push('.');
    }
    t
}

/// Unicode NFKD (through `tokenizers`' normalizer, already a dependency).
pub fn nfkd(text: &str) -> String {
    use tokenizers::normalizers::unicode::NFKD;
    use tokenizers::{NormalizedString, Normalizer};
    let mut n = NormalizedString::from(text);
    match NFKD.normalize(&mut n) {
        Ok(()) => n.get().to_string(),
        Err(_) => text.to_string(),
    }
}

/// Katakana → hiragana (other characters, the long-vowel mark and combining marks unchanged).
pub fn katakana_to_hiragana(text: &str) -> String {
    text.chars()
        .map(|c| match c as u32 {
            0x30A1..=0x30F6 => char::from_u32(c as u32 - 0x60).unwrap_or(c),
            0x30FD | 0x30FE => char::from_u32(c as u32 - 0x60).unwrap_or(c),
            _ => c,
        })
        .collect()
}

/// Japanese kanji readings (`japanese_readings.tsv`: `surface<TAB>katakana reading`, NFKD; about
/// half a million rows). Kept compact: every surface and its hiragana reading in one string, plus
/// a table of offsets sorted by surface (binary search), instead of a map of small strings.
pub struct JaReadings {
    text: String,
    /// `(surface start, surface end = reading start, reading end)` byte offsets into `text`,
    /// sorted by surface, one row per surface.
    rows: Vec<(u32, u32, u32)>,
    /// Longest surface (characters).
    max_chars: usize,
}

impl JaReadings {
    pub fn parse(tsv: &str) -> Self {
        let mut text = String::with_capacity(tsv.len());
        let mut rows: Vec<(u32, u32, u32)> = Vec::new();
        let mut max_chars = 0;
        for line in tsv.lines() {
            let mut parts = line.splitn(2, '\t');
            let (Some(surface), Some(reading)) = (parts.next(), parts.next()) else {
                continue;
            };
            let (surface, reading) = (surface.trim(), reading.trim());
            if surface.is_empty() || reading.is_empty() || text.len() + surface.len() + reading.len() * 2 > u32::MAX as usize {
                continue;
            }
            max_chars = max_chars.max(surface.chars().count());
            let start = text.len() as u32;
            text.push_str(surface);
            let mid = text.len() as u32;
            text.push_str(&katakana_to_hiragana(reading));
            rows.push((start, mid, text.len() as u32));
        }
        // Sorted by surface; the first row of a repeated surface wins.
        rows.sort_by(|a, b| text[a.0 as usize..a.1 as usize].cmp(&text[b.0 as usize..b.1 as usize]));
        rows.dedup_by(|b, a| text[a.0 as usize..a.1 as usize] == text[b.0 as usize..b.1 as usize]);
        text.shrink_to_fit();
        rows.shrink_to_fit();
        Self { text, rows, max_chars }
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// The hiragana reading of `surface`.
    pub fn get(&self, surface: &str) -> Option<&str> {
        let t = &self.text;
        self.rows
            .binary_search_by(|r| t[r.0 as usize..r.1 as usize].cmp(surface))
            .ok()
            .map(|i| {
                let r = self.rows[i];
                &t[r.1 as usize..r.2 as usize]
            })
    }

    /// Replace kanji words by their hiragana readings (longest match first), as the reference
    /// implementation's pykakasi step does: a reading that starts with は or へ gets a space in
    /// front, and kanji without a reading stay as they are. `text` is NFKD.
    pub fn convert(&self, text: &str) -> String {
        if self.rows.is_empty() || !text.chars().any(is_han) {
            return text.to_string();
        }
        // Byte offset of every character, plus the end.
        let bounds: Vec<usize> = text
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(text.len()))
            .collect();
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut out = String::with_capacity(text.len() * 2);
        let mut i = 0;
        'outer: while i < n {
            let window = self.max_chars.min(n - i);
            if chars[i..i + window].iter().any(|&c| is_han(c)) {
                for len in (1..=window).rev() {
                    let key = &text[bounds[i]..bounds[i + len]];
                    if let Some(reading) = self.get(key) {
                        if reading.starts_with('は') || reading.starts_with('へ') {
                            out.push(' ');
                        }
                        out.push_str(reading);
                        i += len;
                        continue 'outer;
                    }
                }
            }
            out.push(chars[i]);
            i += 1;
        }
        out
    }
}

/// Russian word stress (`russian_stress.tsv`: `word<TAB>zero-based stressed character`, lower
/// case, ё written е).
pub struct RuStress {
    map: HashMap<String, usize>,
}

const COMBINING_ACUTE: char = '\u{0301}';
const COMBINING_BREVE: char = '\u{0306}';

fn is_combining(c: char) -> bool {
    matches!(c as u32, 0x0300..=0x036F)
}

/// Basic Cyrillic letters (upper and lower case, Ѐ-џ).
fn is_cyrillic_letter(c: char) -> bool {
    matches!(c as u32, 0x0400..=0x045F)
}

impl RuStress {
    pub fn parse(tsv: &str) -> Self {
        let mut map = HashMap::new();
        for line in tsv.lines() {
            let mut parts = line.splitn(2, '\t');
            let (Some(word), Some(pos)) = (parts.next(), parts.next()) else {
                continue;
            };
            if let Ok(p) = pos.trim().parse::<usize>() {
                let w = word.trim();
                if !w.is_empty() && p < w.chars().count() {
                    map.insert(w.to_string(), p);
                }
            }
        }
        Self { map }
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Mark the stressed vowel of every known word with U+0301 (after the vowel and its other
    /// combining marks). `text` is lower case NFKD (`й` = и + U+0306, `ё` = е + U+0308); words
    /// that already carry a stress mark are left alone.
    pub fn apply(&self, text: &str) -> String {
        if self.map.is_empty() {
            return text.to_string();
        }
        let mut out = String::with_capacity(text.len() + 16);
        let mut rest = text;
        while !rest.is_empty() {
            // Copy up to the next Cyrillic letter.
            let start = rest.find(is_cyrillic_letter).unwrap_or(rest.len());
            out.push_str(&rest[..start]);
            rest = &rest[start..];
            if rest.is_empty() {
                break;
            }
            // The word: Cyrillic letters, each with its combining marks.
            let mut end = 0;
            for (i, c) in rest.char_indices() {
                if is_cyrillic_letter(c) || (is_combining(c) && i > 0) {
                    end = i + c.len_utf8();
                } else {
                    break;
                }
            }
            let word = &rest[..end];
            out.push_str(&self.stress_word(word));
            rest = &rest[end..];
        }
        out
    }

    fn stress_word(&self, word: &str) -> String {
        if word.contains(COMBINING_ACUTE) {
            return word.to_string();
        }
        // Letters as the table spells them, with the byte range each covers (marks included).
        let mut key = String::with_capacity(word.len());
        let mut units: Vec<(usize, usize)> = Vec::new();
        let chars: Vec<(usize, char)> = word.char_indices().collect();
        let mut k = 0;
        while k < chars.len() {
            let (start, base) = chars[k];
            let mut end = start + base.len_utf8();
            let mut marks = Vec::new();
            let mut j = k + 1;
            while j < chars.len() && is_combining(chars[j].1) {
                marks.push(chars[j].1);
                end = chars[j].0 + chars[j].1.len_utf8();
                j += 1;
            }
            // NFKD spells й as и + U+0306 (the table keeps й) and ё as е + U+0308 (the table
            // writes е, so the diaeresis is simply not part of the key).
            let letter = match base {
                'и' if marks.contains(&COMBINING_BREVE) => 'й',
                'ё' => 'е',
                b => b,
            };
            key.push(letter);
            units.push((start, end));
            k = j;
        }
        match self.map.get(&key) {
            Some(&p) if p < units.len() => {
                let at = units[p].1;
                let mut s = String::with_capacity(word.len() + 2);
                s.push_str(&word[..at]);
                s.push(COMBINING_ACUTE);
                s.push_str(&word[at..]);
                s
            }
            _ => word.to_string(),
        }
    }
}

/// Chinese characters → Cangjie codes (`Cangjie5_TC.json`: `{glyph: code}`, the code already
/// carrying the disambiguating index of the reference implementation).
pub struct Cangjie {
    map: HashMap<char, String>,
}

impl Cangjie {
    pub fn parse(json: &str) -> Result<Self, String> {
        let raw: HashMap<String, String> = serde_json::from_str(json).map_err(|e| format!("the Cangjie table is not valid: {e}"))?;
        let mut map = HashMap::with_capacity(raw.len());
        for (glyph, code) in raw {
            let mut it = glyph.chars();
            if let (Some(c), None) = (it.next(), it.next()) {
                if !code.is_empty() {
                    map.insert(c, code);
                }
            }
        }
        Ok(Self { map })
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Each CJK letter with a code becomes `[cj_<c>]...[cj_.]`; everything else stays.
    pub fn encode(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len() * 4);
        for c in text.chars() {
            match self.map.get(&c) {
                Some(code) if c.is_alphabetic() && !c.is_ascii() => {
                    for k in code.chars() {
                        out.push_str("[cj_");
                        out.push(k);
                        out.push(']');
                    }
                    out.push_str("[cj_.]");
                }
                _ => out.push(c),
            }
        }
        out
    }
}

/// The frontend of a loaded model: the tokenizer plus the language tables, each loaded the first
/// time a sentence in its language arrives.
pub struct Frontend {
    tokenizer: tokenizers::Tokenizer,
    cangjie_path: Option<PathBuf>,
    ja_path: Option<PathBuf>,
    ru_path: Option<PathBuf>,
    cangjie: OnceCell<Result<Cangjie, String>>,
    ja: OnceCell<Option<JaReadings>>,
    ru: OnceCell<Option<RuStress>>,
    jieba: OnceCell<jieba_rs::Jieba>,
}

impl Frontend {
    pub fn load(tokenizer: &Path, cangjie: Option<PathBuf>, ja: Option<PathBuf>, ru: Option<PathBuf>) -> Result<Self, String> {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer).map_err(|e| format!("cannot load the Chatterbox tokenizer: {e}"))?;
        Self::with_tokenizer(tokenizer, cangjie, ja, ru)
    }

    pub fn with_tokenizer(
        tokenizer: tokenizers::Tokenizer,
        cangjie_path: Option<PathBuf>,
        ja_path: Option<PathBuf>,
        ru_path: Option<PathBuf>,
    ) -> Result<Self, String> {
        let check = |token: &str| {
            tokenizer
                .token_to_id(token)
                .ok_or_else(|| format!("the Chatterbox tokenizer has no {token} token (not a Chatterbox Multilingual tokenizer)"))
        };
        check(SPACE_TOKEN)?;
        check("[en]")?;
        Ok(Self {
            tokenizer,
            cangjie_path,
            ja_path,
            ru_path,
            cangjie: OnceCell::new(),
            ja: OnceCell::new(),
            ru: OnceCell::new(),
            jieba: OnceCell::new(),
        })
    }

    fn ja(&self) -> Option<&JaReadings> {
        self.ja
            .get_or_init(|| {
                let path = self.ja_path.as_ref()?;
                match std::fs::read_to_string(path) {
                    Ok(s) => Some(JaReadings::parse(&s)),
                    Err(e) => {
                        tracing::warn!(target: "sigil::tts", "Chatterbox: cannot read {}: {e}; kanji are not read out", path.display());
                        None
                    }
                }
            })
            .as_ref()
    }

    fn ru(&self) -> Option<&RuStress> {
        self.ru
            .get_or_init(|| {
                let path = self.ru_path.as_ref()?;
                match std::fs::read_to_string(path) {
                    Ok(s) => Some(RuStress::parse(&s)),
                    Err(e) => {
                        tracing::warn!(target: "sigil::tts", "Chatterbox: cannot read {}: {e}; no stress marks", path.display());
                        None
                    }
                }
            })
            .as_ref()
    }

    fn cangjie(&self) -> Result<&Cangjie, String> {
        self.cangjie
            .get_or_init(|| {
                let path = self
                    .cangjie_path
                    .as_ref()
                    .ok_or_else(|| "the Chinese character table (Cangjie5_TC.json) is missing; reinstall the model".to_string())?;
                let s = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
                Cangjie::parse(&s)
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// The language transform of step 3 (see the module docs) on lower-case NFKD text.
    fn transform(&self, text: &str, lang: &str) -> Result<String, String> {
        Ok(match lang {
            "ja" => match self.ja() {
                Some(readings) => nfkd(&readings.convert(text)),
                None => text.to_string(),
            },
            "zh" => {
                let jieba = self.jieba.get_or_init(jieba_rs::Jieba::new);
                let words: Vec<&str> = jieba
                    .cut(text, true)
                    .into_iter()
                    .map(|t| t.word)
                    .filter(|w| !w.trim().is_empty())
                    .collect();
                self.cangjie()?.encode(&words.join(" "))
            }
            "ru" => match self.ru() {
                Some(stress) => stress.apply(text),
                None => text.to_string(),
            },
            "ko" => text.trim().to_string(),
            _ => text.to_string(),
        })
    }

    /// The string handed to the tokenizer for `text` in `lang` (steps 1-4 of the module docs);
    /// `None` when nothing is left to say.
    pub fn prepare(&self, text: &str, lang: &str) -> Result<Option<String>, String> {
        let t = punc_norm(text);
        if t.is_empty() || !t.chars().any(char::is_alphanumeric) {
            return Ok(None);
        }
        let t = nfkd(&t.to_lowercase());
        let t = self.transform(&t, lang)?;
        let lang = normalize_language(lang).unwrap_or("en");
        Ok(Some(format!("[{lang}]{t}").replace(' ', SPACE_TOKEN)))
    }

    /// Token ids of `text` in `lang`, framed for the embedding graph (step 5); `None` when there
    /// is nothing to say.
    pub fn encode(&self, text: &str, lang: &str) -> Result<Option<Vec<i64>>, String> {
        let Some(prepared) = self.prepare(text, lang)? else {
            return Ok(None);
        };
        let enc = self
            .tokenizer
            .encode(prepared.as_str(), false)
            .map_err(|e| format!("cannot tokenize the sentence: {e}"))?;
        let ids = enc.get_ids();
        if ids.len() > MAX_TEXT_TOKENS {
            return Err(format!("{TOO_LONG} ({} tokens, at most {MAX_TEXT_TOKENS})", ids.len()));
        }
        let mut out = Vec::with_capacity(ids.len() + 5);
        out.extend([EXAGGERATION_TOKEN, START_TEXT]);
        out.extend(ids.iter().map(|&i| i64::from(i)));
        out.extend([STOP_TEXT, START_SPEECH, START_SPEECH]);
        Ok(Some(out))
    }
}

/// Positions of the framed ids (step 5): the text from 0 (`[START]` = 0), every speech-start token
/// at 0, the emotion slot at 0 (unused).
pub fn positions(ids: &[i64]) -> Vec<i64> {
    let mut text_pos = 0;
    ids.iter()
        .map(|&id| {
            if id >= START_SPEECH {
                0
            } else {
                let p = text_pos;
                text_pos += 1;
                p
            }
        })
        .collect()
}

/// Most speech tokens (25 per second) to generate for a sentence: three seconds plus a generous
/// speaking time per letter (0.12 s; a CJK character 0.35 s), at most 1000 (40 s). Stops a model
/// that never emits its stop token.
pub fn max_speech_tokens(text: &str) -> usize {
    let mut seconds = 3.0f32;
    for c in text.chars() {
        if is_han(c) || is_kana(c) || is_hangul(c) {
            seconds += 0.35;
        } else if c.is_alphanumeric() {
            seconds += 0.12;
        }
    }
    ((seconds * 25.0) as usize).clamp(50, 1000)
}
