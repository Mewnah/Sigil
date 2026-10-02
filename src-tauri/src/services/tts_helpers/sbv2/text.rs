//! Style-Bert-VITS2 text normalization and symbol table (Japanese).
//!
//! Ported from `sbv2_core` (`norm.rs`, `nlp.rs`, `utils.rs`; MIT, Copyright (c) 2024 tuna2134,
//! 2025- neodyland, <https://github.com/neodyland/sbv2-api>) and checked against Style-Bert-VITS2's
//! own Python (`style_bert_vits2/nlp/japanese/normalizer.py`, `symbols.py`; litagin02, AGPL-3.0 /
//! LGPL-3.0), which is the reference: where the Rust port differed from the Python, this file
//! follows the Python.
//!
//! - [`normalize_text`]: NFKC, digit separators, currency symbols and `%`, numbers as kanji
//!   numerals ([`numbers_to_kanji`], as the Python's `num2words(lang="ja")`: OpenJTalk then reads
//!   `2024年1月1日` with its counters, and the BERT sees the same text), `~` → `ー`, punctuation
//!   folded to the seven SBV2 punctuation marks, everything else that is not kana, kanji, Latin or
//!   Greek letters removed, combining (han)dakuten removed. (`sbv2_core` read digits through
//!   jpreprocess and re-joined the word strings, which lost digits: `1月` became `月`.)
//! - [`replace_punctuation`]: one left-to-right pass over an ordered table (the Rust port applied a
//!   hash map in random order, so `・・・` could become `,,,` instead of `...`).
//! - [`SYMBOLS`]: the 112-symbol table every SBV2 model embeds (`_`, the sorted union of the
//!   Chinese, Japanese and English phonemes, then the punctuation and `SP` / `UNK`).

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

/// SBV2's punctuation marks (kept in the text and spoken as pauses / intonation).
pub const PUNCTUATIONS: [&str; 7] = ["!", "?", "…", ",", ".", "'", "-"];

/// Whether `s` is one of [`PUNCTUATIONS`].
pub fn is_punctuation(s: &str) -> bool {
    PUNCTUATIONS.contains(&s)
}

/// Whether every character of `s` is a punctuation mark (true for `""`, as Python's
/// `set(s).issubset(PUNCTUATIONS)`).
pub fn all_punctuation(s: &str) -> bool {
    s.chars().all(|c| {
        let mut buf = [0u8; 4];
        is_punctuation(c.encode_utf8(&mut buf))
    })
}

/// Punctuation folding, in Style-Bert-VITS2's order (the first key that matches at a position wins).
const REPLACE_MAP: &[(&str, &str)] = &[
    ("：", ","),
    ("；", ","),
    ("，", ","),
    ("。", "."),
    ("！", "!"),
    ("？", "?"),
    ("\n", "."),
    ("．", "."),
    ("…", "..."),
    ("···", "..."),
    ("・・・", "..."),
    ("·", ","),
    ("・", ","),
    ("、", ","),
    ("$", "."),
    ("“", "'"),
    ("”", "'"),
    ("\"", "'"),
    ("‘", "'"),
    ("’", "'"),
    ("（", "'"),
    ("）", "'"),
    ("(", "'"),
    (")", "'"),
    ("《", "'"),
    ("》", "'"),
    ("【", "'"),
    ("】", "'"),
    ("[", "'"),
    ("]", "'"),
    // Hyphen and dash variants (after NFKC) become a plain hyphen-minus.
    ("\u{02d7}", "-"),
    ("\u{2010}", "-"),
    ("\u{2012}", "-"),
    ("\u{2013}", "-"),
    ("\u{2014}", "-"),
    ("\u{2015}", "-"),
    ("\u{2043}", "-"),
    ("\u{2212}", "-"),
    ("\u{23af}", "-"),
    ("\u{23e4}", "-"),
    ("\u{2500}", "-"),
    ("\u{2501}", "-"),
    ("\u{2e3a}", "-"),
    ("\u{2e3b}", "-"),
    ("「", "'"),
    ("」", "'"),
];

/// Everything that is not kana, kanji, an ASCII or full-width Latin letter, a Greek letter or one
/// of [`PUNCTUATIONS`].
static NOT_ALLOWED: LazyLock<Regex> = LazyLock::new(|| {
    let mut pattern = String::from(
        r"[^\u{3040}-\u{309F}\u{30A0}-\u{30FF}\u{4E00}-\u{9FFF}\u{3400}-\u{4DBF}\u{3005}\u{0041}-\u{005A}\u{0061}-\u{007A}\u{FF21}-\u{FF3A}\u{FF41}-\u{FF5A}\u{0370}-\u{03FF}\u{1F00}-\u{1FFF}",
    );
    for p in PUNCTUATIONS {
        pattern.push_str(&regex::escape(p));
    }
    pattern.push_str("]+");
    Regex::new(&pattern).expect("the SBV2 character class compiles")
});

/// Fold punctuation ([`REPLACE_MAP`], one pass) and drop every character SBV2 cannot read.
pub fn replace_punctuation(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    'outer: while !rest.is_empty() {
        for (from, to) in REPLACE_MAP {
            if let Some(after) = rest.strip_prefix(from) {
                out.push_str(to);
                rest = after;
                continue 'outer;
            }
        }
        let c = rest.chars().next().expect("non-empty");
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    NOT_ALLOWED.replace_all(&out, "").into_owned()
}

/// `1,000,000` → `1000000` (digit groups only).
static NUMBER_WITH_SEPARATOR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[0-9]{1,3}(,[0-9]{3})+").expect("regex"));
/// `$5`, `¥1.5` (a currency sign before a number).
static CURRENCY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"([$¥£€])([0-9.]*[0-9])").expect("regex"));

fn currency_word(sign: &str) -> &str {
    match sign {
        "$" => "ドル",
        "¥" => "円",
        "£" => "ポンド",
        "€" => "ユーロ",
        other => other,
    }
}

/// NFKC (Unicode compatibility composition), through `tokenizers`' normalizer (already a
/// dependency for the BERT tokenizer).
pub fn nfkc(text: &str) -> String {
    use tokenizers::normalizers::unicode::NFKC;
    use tokenizers::{NormalizedString, Normalizer};
    let mut n = NormalizedString::from(text);
    match NFKC.normalize(&mut n) {
        Ok(()) => n.get().to_string(),
        Err(_) => text.to_string(),
    }
}

/// `%` (after NFKC) read as a word.
fn percent_word(text: &str) -> String {
    text.replace('%', "パーセント")
}

/// A number written with digits (`2024`, `3.14`).
static NUMBER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[0-9]+(\.[0-9]+)?").expect("regex"));

const KANJI_DIGITS: [char; 10] = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
/// Units of each group of four digits.
const GROUP_UNITS: [&str; 6] = ["", "万", "億", "兆", "京", "垓"];

/// 0-9999 as kanji numerals (`1234` → `千二百三十四`; a leading one before 千, 百 and 十 is
/// silent, as `num2words(lang="ja")` writes it).
fn group_to_kanji(n: usize) -> String {
    let mut s = String::new();
    for (unit, div) in [('千', 1000), ('百', 100), ('十', 10)] {
        let d = n / div % 10;
        if d > 0 {
            if d > 1 {
                s.push(KANJI_DIGITS[d]);
            }
            s.push(unit);
        }
    }
    if !n.is_multiple_of(10) {
        s.push(KANJI_DIGITS[n % 10]);
    }
    s
}

/// A digit string as a kanji numeral (`2024` → `二千二十四`, `10000` → `一万`, `0` → `零`); one too
/// long for the units is read digit by digit.
pub fn integer_to_kanji(digits: &str) -> String {
    let t = digits.trim_start_matches('0');
    if t.is_empty() {
        return if digits.is_empty() { String::new() } else { "零".into() };
    }
    if t.len() > 4 * GROUP_UNITS.len() || !t.bytes().all(|b| b.is_ascii_digit()) {
        return t.bytes().filter(u8::is_ascii_digit).map(|b| KANJI_DIGITS[usize::from(b - b'0')]).collect();
    }
    let values: Vec<usize> = t.bytes().map(|b| usize::from(b - b'0')).collect();
    let mut groups = Vec::new();
    let mut end = values.len();
    while end > 0 {
        let start = end.saturating_sub(4);
        groups.push(values[start..end].iter().fold(0usize, |acc, d| acc * 10 + d));
        end = start;
    }
    let mut out = String::new();
    for (i, g) in groups.iter().enumerate().rev() {
        if *g > 0 {
            out.push_str(&group_to_kanji(*g));
            out.push_str(GROUP_UNITS[i]);
        }
    }
    out
}

/// Every number in `text` as kanji numerals; decimals as `点` and their digits one by one
/// (`3.05` → `三点〇五`).
pub fn numbers_to_kanji(text: &str) -> String {
    NUMBER
        .replace_all(text, |c: &regex::Captures| {
            let m = &c[0];
            match m.split_once('.') {
                Some((int, frac)) => {
                    let frac: String = frac.bytes().map(|b| if b == b'0' { '〇' } else { KANJI_DIGITS[usize::from(b - b'0')] }).collect();
                    format!("{}点{frac}", integer_to_kanji(int))
                }
                None => integer_to_kanji(m),
            }
        })
        .into_owned()
}

/// NFKC, digit separators removed, currency signs after their amount (`$5` → `5ドル`) and `%` as
/// a word: what a Japanese reader needs before it reads numbers.
pub fn read_signs(text: &str) -> String {
    let res = nfkc(text);
    let res = NUMBER_WITH_SEPARATOR.replace_all(&res, |c: &regex::Captures| c[0].replace(',', ""));
    let res = CURRENCY.replace_all(&res, |c: &regex::Captures| format!("{}{}", &c[2], currency_word(&c[1])));
    percent_word(&res)
}

/// Style-Bert-VITS2's Japanese text normalization (see the module docs).
pub fn normalize_text(text: &str) -> String {
    let res = numbers_to_kanji(&read_signs(text));
    let res = res.replace(['~', '～', '〜'], "ー");
    let res = replace_punctuation(&res);
    res.replace(['\u{3099}', '\u{309A}'], "")
}

// -------------------------------------------------------------------------------------------------
// Symbols
// -------------------------------------------------------------------------------------------------

const ZH_SYMBOLS: [&str; 65] = [
    "E", "En", "a", "ai", "an", "ang", "ao", "b", "c", "ch", "d", "e", "ei", "en", "eng", "er", "f", "g", "h", "i", "i0", "ia", "ian", "iang", "iao",
    "ie", "in", "ing", "iong", "ir", "iu", "j", "k", "l", "m", "n", "o", "ong", "ou", "p", "q", "r", "s", "sh", "t", "u", "ua", "uai", "uan", "uang",
    "ui", "un", "uo", "v", "van", "ve", "vn", "w", "x", "y", "z", "zh", "AA", "EE", "OO",
];
/// Japanese phonemes of SBV2.
pub const JP_SYMBOLS: [&str; 42] = [
    "N", "a", "a:", "b", "by", "ch", "d", "dy", "e", "e:", "f", "g", "gy", "h", "hy", "i", "i:", "j", "k", "ky", "m", "my", "n", "ny", "o", "o:", "p", "py",
    "q", "r", "ry", "s", "sh", "t", "ts", "ty", "u", "u:", "w", "y", "z", "zy",
];
const EN_SYMBOLS: [&str; 39] = [
    "aa", "ae", "ah", "ao", "aw", "ay", "b", "ch", "d", "dh", "eh", "er", "ey", "f", "g", "hh", "ih", "iy", "jh", "k", "l", "m", "n", "ng", "ow", "oy",
    "p", "r", "s", "sh", "t", "th", "uh", "uw", "V", "w", "y", "z", "zh",
];
/// Padding (and the sentence boundary of the g2p).
pub const PAD: &str = "_";
/// Unknown phoneme.
pub const UNK: &str = "UNK";
/// Number of symbols of every SBV2 model (the phoneme embedding's size).
pub const NUM_SYMBOLS: usize = 112;
/// Language id of Japanese (`language` input; Chinese 0, English 2).
pub const JP_LANGUAGE_ID: i64 = 1;
/// Tone offset of Japanese (after the six Chinese tones).
pub const JP_TONE_START: i64 = 6;

/// The symbol table, in model order.
pub static SYMBOLS: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    let mut normal: Vec<&'static str> = ZH_SYMBOLS.iter().chain(JP_SYMBOLS.iter()).chain(EN_SYMBOLS.iter()).copied().collect();
    normal.sort_unstable();
    normal.dedup();
    let mut symbols = vec![PAD];
    symbols.extend(normal);
    symbols.extend(PUNCTUATIONS);
    symbols.extend(["SP", UNK]);
    debug_assert_eq!(symbols.len(), NUM_SYMBOLS, "every SBV2 model has 112 symbols");
    symbols
});

static SYMBOL_IDS: LazyLock<HashMap<&'static str, i64>> =
    LazyLock::new(|| SYMBOLS.iter().enumerate().map(|(i, s)| (*s, i as i64)).collect());

/// Model id of a phoneme ([`UNK`]'s for anything unknown).
pub fn symbol_id(phone: &str) -> i64 {
    SYMBOL_IDS.get(phone).copied().unwrap_or_else(|| SYMBOL_IDS[UNK])
}

/// `[a, b]` → `[sep, a, sep, b, sep]` (SBV2's `add_blank`).
pub fn intersperse<T: Clone>(items: &[T], sep: T) -> Vec<T> {
    let mut out = Vec::with_capacity(items.len() * 2 + 1);
    out.push(sep.clone());
    for item in items {
        out.push(item.clone());
        out.push(sep.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_symbol_table_matches_style_bert_vits2() {
        assert_eq!(SYMBOLS.len(), NUM_SYMBOLS);
        // Indices computed with Style-Bert-VITS2's symbols.py.
        for (s, id) in [
            ("_", 0),
            ("N", 5),
            ("a", 8),
            ("ch", 22),
            ("k", 57),
            ("o", 65),
            ("q", 73),
            ("sh", 77),
            ("ts", 80),
            ("!", 103),
            ("?", 104),
            (",", 106),
            (".", 107),
            ("'", 108),
            ("-", 109),
            ("SP", 110),
            ("UNK", 111),
        ] {
            assert_eq!(symbol_id(s), id, "{s}");
        }
        assert_eq!(symbol_id("ー"), 111, "unknown phonemes map to UNK");
    }

    #[test]
    fn punctuation_is_folded_in_one_ordered_pass() {
        assert_eq!(replace_punctuation("こんにちは、世界。"), "こんにちは,世界.");
        // `・・・` is an ellipsis, not three commas (the order of the table matters).
        assert_eq!(replace_punctuation("えっ・・・"), "えっ...");
        assert_eq!(replace_punctuation("A・B"), "A,B");
        assert_eq!(replace_punctuation("「はい」（笑）"), "'はい''笑'");
        // Emoji, digits, spaces and symbols outside the table are dropped.
        assert_eq!(replace_punctuation("ok 123 🎉 #"), "ok");
        assert_eq!(replace_punctuation("ＡＢｃ"), "ＡＢｃ", "full-width letters are kept");
    }

    #[test]
    fn normalization_follows_style_bert_vits2() {
        // NFKC: full-width punctuation and half-width katakana.
        assert_eq!(normalize_text("ﾃｽﾄ！"), "テスト!");
        // Long-vowel tildes; the ellipsis character becomes three dots.
        assert_eq!(normalize_text("すご～い…"), "すごーい...");
        assert_eq!(normalize_text("すご〜い"), "すごーい");
        // Combining dakuten are removed.
        assert_eq!(normalize_text("あ\u{3099}"), "あ");
    }

    #[test]
    fn numbers_become_kanji_numerals_like_num2words() {
        assert_eq!(integer_to_kanji("0"), "零");
        assert_eq!(integer_to_kanji("7"), "七");
        assert_eq!(integer_to_kanji("10"), "十");
        assert_eq!(integer_to_kanji("15"), "十五");
        assert_eq!(integer_to_kanji("100"), "百");
        assert_eq!(integer_to_kanji("1000"), "千");
        assert_eq!(integer_to_kanji("2024"), "二千二十四");
        assert_eq!(integer_to_kanji("10000"), "一万");
        assert_eq!(integer_to_kanji("12345678"), "千二百三十四万五千六百七十八");
        assert_eq!(integer_to_kanji("100000000"), "一億");
        assert_eq!(integer_to_kanji("007"), "七");
        assert_eq!(numbers_to_kanji("3.05と0.5"), "三点〇五と零点五");
        assert_eq!(numbers_to_kanji("2024年1月1日"), "二千二十四年一月一日");
        // Too long for the units: digit by digit.
        assert_eq!(integer_to_kanji(&"1".repeat(30)), "一".repeat(30));
    }

    #[test]
    fn currency_separators_and_percent_are_read() {
        assert_eq!(normalize_text("1,234,567円"), "百二十三万四千五百六十七円");
        assert_eq!(normalize_text("$5と€2.5"), "五ドルと二点五ユーロ");
        // Full-width digits become ASCII (NFKC) so they are one number.
        assert_eq!(normalize_text("１２３"), "百二十三");
        assert_eq!(normalize_text("100%"), "百パーセント");
    }

    #[test]
    fn intersperse_adds_blanks() {
        assert_eq!(intersperse(&[1, 2], 0), [0, 1, 0, 2, 0]);
        assert_eq!(intersperse::<i32>(&[], 0), [0]);
    }

    #[test]
    fn punctuation_helpers() {
        assert!(all_punctuation("..."));
        assert!(all_punctuation(""));
        assert!(!all_punctuation("a."));
        assert!(is_punctuation("…"));
    }
}
