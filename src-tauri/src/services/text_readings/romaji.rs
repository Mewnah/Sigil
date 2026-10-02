//! Kana → Hepburn romaji for Japanese readings (P6-L).
//!
//! Modified Hepburn from a word's katakana reading and OpenJTalk pronunciation:
//! - long vowels where the pronunciation has `ー` get a macron (`トウキョウ` / `トーキョー` →
//!   `tōkyō`, `コーヒー` → `kōhī`), except `ei` and `ii`, which stay spelled (`gakusei`, `oishii`);
//!   a vowel that is only spelled long (`思う` `オモウ`) is not lengthened;
//! - particles follow the pronunciation (`は` → `wa`, `へ` → `e`, `を` → `o`);
//! - `ン` before a vowel or `y` is `n'` (`kin'en`), `ッ` doubles the next consonant (`tch` before
//!   `ch`).
//!
//! Single morae are converted with `wana_kana` (its tables cover the extended katakana such as
//! `ティ` / `ファ` / `ヴ`); everything around them (morae, sokuon, macrons) is done here.

use wana_kana::{ConvertJapanese, Options};

/// Small kana that join the preceding kana into one mora.
fn is_small_kana(c: char) -> bool {
    matches!(
        c,
        'ァ' | 'ィ' | 'ゥ' | 'ェ' | 'ォ' | 'ャ' | 'ュ' | 'ョ' | 'ヮ' | 'ぁ' | 'ぃ' | 'ぅ' | 'ぇ' | 'ぉ' | 'ゃ' | 'ゅ' | 'ょ' | 'ゎ'
    )
}

/// A katakana character (including `ー` and the small ones).
pub(crate) fn is_katakana(c: char) -> bool {
    matches!(c, '\u{30A1}'..='\u{30FA}' | 'ー' | 'ヽ' | 'ヾ')
}

/// A hiragana character.
pub(crate) fn is_hiragana(c: char) -> bool {
    matches!(c, '\u{3041}'..='\u{3096}' | 'ゝ' | 'ゞ')
}

/// Kana (either script) or the long-vowel mark.
pub(crate) fn is_kana(c: char) -> bool {
    is_katakana(c) || is_hiragana(c)
}

/// Katakana → hiragana, keeping `ー` (wana_kana with `keep_prolonged_sound_mark`). Only for kana:
/// wana_kana would also turn Latin letters into kana.
pub(crate) fn to_hiragana(kana: &str) -> String {
    kana.to_hiragana_with_opt(Options {
        keep_prolonged_sound_mark: true,
        pass_romaji: true,
        ..Options::default()
    })
}

/// Split kana into morae (`キャ`, `ッ`, `ン`, `ー` are one mora each).
pub(crate) fn morae(kana: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in kana.chars() {
        match out.last_mut() {
            Some(last) if is_small_kana(c) && !last.ends_with(['ッ', 'ン', 'ー', 'っ', 'ん']) => last.push(c),
            _ => out.push(c.to_string()),
        }
    }
    out
}

/// The vowel a romanized mora ends in.
fn last_vowel(romaji: &str) -> Option<char> {
    romaji.chars().last().filter(|c| matches!(c, 'a' | 'i' | 'u' | 'e' | 'o'))
}

fn macron(v: char) -> char {
    match v {
        'a' => 'ā',
        'i' => 'ī',
        'u' => 'ū',
        'e' => 'ē',
        'o' => 'ō',
        other => other,
    }
}

/// The vowel of a small vowel kana (`ァ` → `a`).
fn small_vowel(c: char) -> Option<char> {
    match c {
        'ァ' | 'ぁ' => Some('a'),
        'ィ' | 'ぃ' => Some('i'),
        'ゥ' | 'ぅ' => Some('u'),
        'ェ' | 'ぇ' => Some('e'),
        'ォ' | 'ぉ' => Some('o'),
        _ => None,
    }
}

/// Romaji of one mora that is not `ッ` / `ン` / `ー`.
fn mora_romaji(mora: &str) -> String {
    let hira = to_hiragana(mora);
    // を is pronounced (and written in Hepburn) `o`.
    if hira == "を" {
        return "o".into();
    }
    // Kana + small vowel (`ティ`, `ファ`, `ウィ`, `ヴァ`, `イェ`): the kana's consonant and the vowel.
    let chars: Vec<char> = hira.chars().collect();
    if let [first, second] = chars.as_slice() {
        if let Some(vowel) = small_vowel(*second) {
            let head = first.to_string().as_str().to_romaji();
            let consonant = head.trim_end_matches(['a', 'i', 'u', 'e', 'o']);
            let consonant = match (consonant, first) {
                ("", 'う') => "w",
                ("", 'い') => "y",
                (c, _) => c,
            };
            if consonant.chars().all(|c| c.is_ascii_lowercase()) {
                return format!("{consonant}{vowel}");
            }
        }
    }
    let r = hira.as_str().to_romaji();
    let clean = r.chars().all(|c| c.is_ascii_lowercase());
    if clean && !r.contains('x') {
        return r;
    }
    // A combination wana_kana spells with a small-kana `x` (e.g. `トゥ` → `toxu`): the first
    // kana's consonant plus the small vowel.
    let mut chars = hira.chars();
    if let (Some(first), Some(small)) = (chars.next(), chars.next()) {
        let head = first.to_string().as_str().to_romaji();
        let tail = small.to_string().as_str().to_romaji();
        let tail = tail.trim_start_matches('x');
        let consonant = head.trim_end_matches(['a', 'i', 'u', 'e', 'o']);
        if !consonant.is_empty() && consonant.chars().all(|c| c.is_ascii_lowercase()) && !tail.is_empty() {
            return format!("{consonant}{tail}");
        }
    }
    r.chars().filter(|c| c.is_ascii_alphabetic()).collect::<String>().to_ascii_lowercase()
}

/// A word-final `ッ` waiting for the next word's consonant (never left in finished text).
pub(crate) const SOKUON_MARK: char = '\u{E000}';
/// A word-initial `ー` (the auxiliary `う` of `遊ぼう`) lengthening the previous word's vowel.
pub(crate) const LONG_MARK: char = '\u{E001}';

/// The doubled consonant `ッ` adds before `next` (`t` before `ch`; nothing before a vowel).
fn geminate(next: &str) -> String {
    if next.starts_with("ch") {
        "t".into()
    } else {
        next.chars()
            .next()
            .filter(|c| c.is_ascii_lowercase() && !matches!(c, 'a' | 'i' | 'u' | 'e' | 'o' | 'n'))
            .map(String::from)
            .unwrap_or_default()
    }
}

/// Append `word` to `out`. When the two are written together (`joined`): a pending word-final
/// `ッ` doubles `word`'s first consonant (`言っ` + `た` → `itta`), a word-initial long mark
/// lengthens the vowel before it (`遊ぼ` + `う` → `asobō`), and `n` before a vowel or `y` gets an
/// apostrophe (`千` + `円` → `sen'en`). Otherwise the marks are dropped.
pub(crate) fn join_word(out: &mut String, word: &str, joined: bool) {
    let pending_sokuon = out.ends_with(SOKUON_MARK);
    if pending_sokuon {
        out.pop();
    }
    let (long, word) = match word.strip_prefix(LONG_MARK) {
        Some(rest) => (true, rest),
        None => (false, word),
    };
    if joined {
        if long {
            if let Some(v) = out.pop() {
                out.push(if matches!(v, 'a' | 'i' | 'u' | 'e' | 'o') { macron(v) } else { v });
            }
        }
        if pending_sokuon {
            out.push_str(&geminate(word));
        } else if out.ends_with('n') && word.starts_with(['a', 'i', 'u', 'e', 'o', 'y']) {
            out.push('\'');
        }
    }
    out.push_str(word);
}

/// Text with any unresolved marks removed.
pub(crate) fn finish(text: &str) -> String {
    text.replace([SOKUON_MARK, LONG_MARK], "")
}

/// One mora of a word: what to write for it.
#[derive(Clone, Debug, PartialEq)]
enum Unit {
    Mora(String),
    /// `ッ`: doubles the next consonant.
    Sokuon,
    /// `ン`.
    N,
    /// A long vowel: lengthen the previous vowel (macron).
    Long,
}

/// The morae of a word from its reading and pronunciation: the pronunciation decides what is
/// said (particles, long vowels); the reading keeps `ei` / `ii` spelled out.
fn word_units(read: &str, pron: &str) -> Vec<Unit> {
    let pron = pron.replace(['’', '\''], "");
    let read_m = morae(read);
    let pron_m = morae(&pron);
    let aligned = read_m.len() == pron_m.len();
    let mut units = Vec::with_capacity(pron_m.len());
    let mut prev_vowel: Option<char> = None;
    for (i, p) in pron_m.iter().enumerate() {
        let unit = match p.as_str() {
            "ッ" | "っ" => Unit::Sokuon,
            "ン" | "ん" => Unit::N,
            "ー" => {
                let spelled = aligned.then(|| read_m[i].as_str());
                match (spelled, prev_vowel) {
                    // 学生 ガクセイ / ガクセー, 美味しい オイシイ / オイシー: spelled out.
                    (Some("イ" | "い"), Some('e' | 'i')) => Unit::Mora("イ".into()),
                    _ => Unit::Long,
                }
            }
            other => Unit::Mora(other.to_string()),
        };
        if let Unit::Mora(m) = &unit {
            prev_vowel = last_vowel(&mora_romaji(m));
        } else if unit == Unit::N || unit == Unit::Sokuon {
            prev_vowel = None;
        }
        units.push(unit);
    }
    units
}

/// Hepburn romaji of one word (lower case).
pub(crate) fn word_romaji(read: &str, pron: &str) -> String {
    let units = word_units(read, pron);
    let romaji: Vec<Option<String>> = units
        .iter()
        .map(|u| match u {
            Unit::Mora(m) => Some(mora_romaji(m)),
            _ => None,
        })
        .collect();
    let next_mora = |i: usize| -> Option<&str> { romaji[i + 1..].iter().flatten().next().map(String::as_str) };
    let mut out = String::new();
    for (i, unit) in units.iter().enumerate() {
        match unit {
            Unit::Mora(_) => out.push_str(romaji[i].as_deref().unwrap_or_default()),
            Unit::Sokuon => match romaji.get(i + 1) {
                Some(Some(next)) => out.push_str(&geminate(next)),
                // Before the next word (`言っ` + `た` → `itta`): [`join_word`] resolves it.
                None => out.push(SOKUON_MARK),
                // Before ン / ー / another ッ: silent.
                Some(None) => {}
            },
            Unit::N => {
                out.push('n');
                if units.get(i + 1).is_some_and(|u| matches!(u, Unit::Mora(_)))
                    && next_mora(i).is_some_and(|n| n.starts_with(['a', 'i', 'u', 'e', 'o', 'y']))
                {
                    out.push('\'');
                }
            }
            Unit::Long if out.is_empty() => out.push(LONG_MARK),
            Unit::Long => {
                if let Some(v) = out.pop() {
                    if matches!(v, 'a' | 'i' | 'u' | 'e' | 'o') {
                        out.push(macron(v));
                    } else {
                        out.push(v);
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morae_join_small_kana() {
        assert_eq!(morae("キャッシュ"), ["キャ", "ッ", "シュ"]);
        assert_eq!(morae("トーキョー"), ["ト", "ー", "キョ", "ー"]);
        assert_eq!(morae("ティー"), ["ティ", "ー"]);
        assert_eq!(morae("ンャ"), ["ン", "ャ"]);
    }

    #[test]
    fn hepburn_basics() {
        for (read, pron, want) in [
            ("ワタシ", "ワタシ", "watashi"),
            ("ハ", "ワ", "wa"),
            ("ヘ", "エ", "e"),
            ("ヲ", "オ", "o"),
            ("チカテツ", "チカテツ", "chikatetsu"),
            ("フジサン", "フジサン", "fujisan"),
            ("ジュウ", "ジュー", "jū"),
            ("トウキョウ", "トーキョー", "tōkyō"),
            ("ガクセイ", "ガクセー", "gakusei"),
            ("オイシイ", "オイシー", "oishii"),
            ("オオキイ", "オーキー", "ōkii"),
            ("オカアサン", "オカーサン", "okāsan"),
            ("オネエサン", "オネーサン", "onēsan"),
            ("スウガク", "スーガク", "sūgaku"),
            ("オモウ", "オモウ", "omou"),
            ("コーヒー", "コーヒー", "kōhī"),
            ("ガッコウ", "ガッコー", "gakkō"),
            ("マッチャ", "マッチャ", "matcha"),
            ("キンエン", "キンエン", "kin'en"),
            ("ホンヤ", "ホンヤ", "hon'ya"),
            ("シンブン", "シンブン", "shinbun"),
            ("パーティー", "パーティー", "pātī"),
            ("ファイル", "ファイル", "fairu"),
            ("ヴァイオリン", "ヴァイオリン", "vaiorin"),
            ("デス", "デス’", "desu"),
            ("アッ", "アッ", "a"),
            ("イッ", "イッ", "i"),
        ] {
            assert_eq!(finish(&word_romaji(read, pron)), want, "{read} / {pron}");
        }
    }

    #[test]
    fn a_word_final_sokuon_doubles_the_next_word() {
        // 言っ + た → itta; 行っ + ちゃう → itchau; before a space or a vowel it is silent.
        for (next, joined, want) in [("ta", true, "itta"), ("chau", true, "itchau"), ("ta", false, "ita"), ("ou", true, "iou")] {
            let mut out = word_romaji("イッ", "イッ");
            join_word(&mut out, next, joined);
            assert_eq!(finish(&out), want, "{next} {joined}");
        }
    }

    #[test]
    fn a_word_initial_long_mark_and_n_join_across_words() {
        let mut out = word_romaji("アソボ", "アソボ");
        join_word(&mut out, &word_romaji("ウ", "ー"), true);
        assert_eq!(finish(&out), "asobō");
        let mut out = word_romaji("セン", "セン");
        join_word(&mut out, "en", true);
        assert_eq!(out, "sen'en");
        // Apart, nothing changes.
        let mut out = "sen".to_string();
        join_word(&mut out, "en", false);
        assert_eq!(out, "senen");
        assert_eq!(finish(&word_romaji("ヲ", "ヲ")), "o");
    }

    #[test]
    fn misaligned_readings_follow_the_pronunciation() {
        // Different mora counts: the pronunciation alone decides.
        assert_eq!(word_romaji("イウ", "ユー"), "yū");
    }

    #[test]
    fn kana_classes_and_hiragana() {
        assert!(is_katakana('ア') && is_katakana('ー') && !is_katakana('あ'));
        assert!(is_hiragana('あ') && !is_hiragana('ア') && !is_hiragana('漢'));
        assert_eq!(to_hiragana("ゲーム"), "げーむ");
        assert_eq!(to_hiragana("ガクセイ"), "がくせい");
    }
}
