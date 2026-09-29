//! Whisper output hardening: non-speech markers, repetition loops and whole-utterance
//! hallucinations ("thanks for watching" on noise).
//!
//! Rules (research §1.2):
//! - Per-language curated phrase lists (`phrases/<lang>.txt`, format documented in `en.txt`) are
//!   matched against WHOLE sentences only, never removed from inside real speech; an utterance is
//!   dropped when every sentence in it is a listed phrase. English is checked for every language
//!   (Whisper falls back to English stock phrases on noise).
//! - Soft phrases (`~`) are real things people say ("thank you", and the outros streamers really
//!   end on: "thanks for watching", "like and subscribe"), so they only count when Whisper itself
//!   reported a no-speech probability of at least [`SOFT_NO_SPEECH_PROB`]. Hard phrases are kept to
//!   credit lines nobody says ("subtitles by ...", amara.org, broadcaster tags).
//! - Repeats are collapsed only at 3 or more consecutive occurrences (single short words need 4,
//!   so "no no no" survives); Chinese/Japanese text is also checked with character n-grams, where
//!   one- and two-character units need 6 and 4 repeats ("はいはいはい" survives). Character units
//!   must contain a CJK character, so digit and Latin runs inside a CJK token ("1000000円",
//!   "面白いwwwwww") are never shortened.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Whisper no-speech probability from which soft phrases count as hallucinations. Measured with
/// whisper.rs' start-of-transcript probe: genuine short words at ordinary mic levels score up to
/// ~0.47 ("Bye." 0.27, "Hi" 0.47), hallucinations on silence and noise 0.88-0.95.
pub const SOFT_NO_SPEECH_PROB: f32 = 0.5;
/// Longest repeated unit considered, in words.
const MAX_WORD_UNIT: usize = 8;
/// Longest repeated unit considered, in characters (CJK).
const MAX_CHAR_UNIT: usize = 24;

const SOURCES: &[(&str, &str)] = &[
    ("en", include_str!("phrases/en.txt")),
    ("ja", include_str!("phrases/ja.txt")),
    ("zh", include_str!("phrases/zh.txt")),
    ("ko", include_str!("phrases/ko.txt")),
    ("es", include_str!("phrases/es.txt")),
    ("fr", include_str!("phrases/fr.txt")),
    ("de", include_str!("phrases/de.txt")),
    ("ru", include_str!("phrases/ru.txt")),
    ("pt", include_str!("phrases/pt.txt")),
];

#[derive(Debug)]
struct Phrase {
    norm: String,
    prefix: bool,
    soft: bool,
}

fn parse_phrases(src: &str) -> Vec<Phrase> {
    src.lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .filter_map(|l| {
            let (soft, l) = l.strip_prefix('~').map_or((false, l), |r| (true, r));
            let (prefix, l) = l.strip_suffix('*').map_or((false, l), |r| (true, r));
            let norm = normalize(l);
            (!norm.is_empty()).then_some(Phrase { norm, prefix, soft })
        })
        .collect()
}

fn lists() -> &'static HashMap<&'static str, Vec<Phrase>> {
    static LISTS: OnceLock<HashMap<&'static str, Vec<Phrase>>> = OnceLock::new();
    LISTS.get_or_init(|| SOURCES.iter().map(|(lang, src)| (*lang, parse_phrases(src))).collect())
}

/// Phrase-list key for a Whisper / BCP-47 language code (`zh-TW`, `yue`, `pt_br` ...).
fn list_key(language: &str) -> Option<&'static str> {
    let primary = language
        .trim()
        .split(['-', '_'])
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    let key = match primary.as_str() {
        "yue" | "cmn" | "zh" => "zh",
        other => other,
    };
    SOURCES.iter().map(|(k, _)| *k).find(|k| *k == key)
}

/// Full-width ASCII variants (U+FF01..U+FF5E) and the ideographic space to ASCII.
fn fold_width(c: char) -> char {
    match c {
        '\u{FF01}'..='\u{FF5E}' => char::from_u32(c as u32 - 0xFF01 + 0x21).unwrap_or(c),
        '\u{3000}' => ' ',
        _ => c,
    }
}

/// Comparison key: lower-case letters and digits only (any script), full-width folded.
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars().map(fold_width) {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        }
    }
    out
}

/// Han, Hiragana or Katakana (scripts written without spaces).
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF | 0x31F0..=0x31FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF
        | 0xFF66..=0xFF9F | 0x20000..=0x2FA1F)
}

fn is_hangul(c: char) -> bool {
    matches!(c as u32, 0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F)
}

/// Remove Whisper's non-speech annotations: `[MUSIC]`, `(laughs)`, `（拍手）`, `【音楽】`, `♪`.
pub fn strip_non_speech_markers(text: &str) -> String {
    const PAIRS: &[(char, char)] = &[('[', ']'), ('(', ')'), ('（', '）'), ('【', '】'), ('［', '］')];
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if let Some(&(_, close)) = PAIRS.iter().find(|(open, _)| *open == c) {
            if let Some(end) = chars[i + 1..].iter().position(|&d| d == close) {
                i += end + 2;
                out.push(' ');
                continue;
            }
        }
        if !matches!(c, '♪' | '♫' | '♬' | '♩') {
            out.push(c);
        }
        i += 1;
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Repeats a single-word unit needs before it is collapsed (dense scripts: a 3+ character word is
/// already a phrase).
fn word_unit_threshold(unit: &[String]) -> usize {
    if unit.len() >= 2 {
        return 3;
    }
    let dense = unit[0].chars().filter(|&c| is_cjk(c) || is_hangul(c)).count();
    if dense >= 3 {
        3
    } else {
        4
    }
}

fn char_unit_threshold(n: usize) -> usize {
    match n {
        1 => 6,
        2 => 4,
        _ => 3,
    }
}

/// Collapse runs of a repeated word unit (1..=8 words) to one occurrence.
fn collapse_word_repeats(text: &str) -> String {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let keys: Vec<String> = tokens.iter().map(|t| normalize(t)).collect();
    let mut out: Vec<&str> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    'outer: while i < tokens.len() {
        for n in 1..=MAX_WORD_UNIT.min((tokens.len() - i) / 3) {
            let unit = &keys[i..i + n];
            if unit.iter().all(String::is_empty) {
                continue;
            }
            let mut reps = 1;
            while i + (reps + 1) * n <= tokens.len() && keys[i + reps * n..i + (reps + 1) * n] == *unit {
                reps += 1;
            }
            if reps >= word_unit_threshold(unit) {
                out.extend_from_slice(&tokens[i..i + n]);
                i += reps * n;
                // A loop cut off mid-unit at the very end ("... I'm going I'm") loses the stub.
                let rest = &keys[i..];
                if !rest.is_empty() && rest.len() < n && unit.starts_with(rest) {
                    i = tokens.len();
                }
                continue 'outer;
            }
        }
        out.push(tokens[i]);
        i += 1;
    }
    out.join(" ")
}

/// Collapse runs of a repeated character unit (1..=24 chars) inside one CJK token. Only units that
/// contain a CJK character count: digits and Latin letters are left to the word-level rule, so a
/// number such as `1000000` is never shortened.
fn collapse_char_repeats(token: &str) -> String {
    let chars: Vec<char> = token.chars().collect();
    let mut out = String::with_capacity(token.len());
    let mut i = 0;
    'outer: while i < chars.len() {
        for n in 1..=MAX_CHAR_UNIT.min((chars.len() - i) / 3) {
            let unit = &chars[i..i + n];
            if !unit.iter().any(|&c| is_cjk(c)) {
                continue;
            }
            let mut reps = 1;
            while i + (reps + 1) * n <= chars.len() && chars[i + reps * n..i + (reps + 1) * n] == *unit {
                reps += 1;
            }
            if reps >= char_unit_threshold(n) {
                out.extend(unit);
                i += reps * n;
                let rest = &chars[i..];
                if !rest.is_empty() && rest.len() < n && unit.starts_with(rest) {
                    i = chars.len();
                }
                continue 'outer;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Collapse repetition loops: word units everywhere, character n-grams inside CJK tokens.
pub fn collapse_repeats(text: &str) -> String {
    let words = collapse_word_repeats(text);
    words
        .split(' ')
        .map(|tok| {
            let alnum = tok.chars().filter(|c| c.is_alphanumeric()).count();
            let cjk = tok.chars().filter(|&c| is_cjk(c)).count();
            if alnum > 0 && cjk * 2 >= alnum {
                collapse_char_repeats(tok)
            } else {
                tok.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Split into sentences at `!?…。！？` and newlines, and at `.` only when it ends a word (so
/// "amara.org" stays one sentence).
fn sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    for (i, &c) in chars.iter().enumerate() {
        let ends = match c {
            '!' | '?' | '…' | '。' | '！' | '？' | '\n' => true,
            '.' | '．' => chars.get(i + 1).is_none_or(|n| n.is_whitespace()),
            _ => false,
        };
        cur.push(c);
        if ends {
            out.push(std::mem::take(&mut cur));
        }
    }
    out.push(cur);
    out.into_iter().filter(|s| !normalize(s).is_empty()).collect()
}

fn matches_phrase(sentence_norm: &str, phrases: &[Phrase], no_speech_prob: f32) -> bool {
    phrases.iter().any(|p| {
        let hit = if p.prefix {
            sentence_norm.starts_with(&p.norm)
        } else {
            sentence_norm == p.norm
        };
        hit && (!p.soft || no_speech_prob >= SOFT_NO_SPEECH_PROB)
    })
}

/// Whether the whole utterance is a known hallucination for `language` (`None`/`"auto"`: English
/// list only). Text without any letters or digits counts as a hallucination.
pub fn is_hallucination(text: &str, language: Option<&str>, no_speech_prob: f32) -> bool {
    let all = lists();
    let en = all.get("en").map(Vec::as_slice).unwrap_or(&[]);
    let native = language
        .and_then(list_key)
        .filter(|k| *k != "en")
        .and_then(|k| all.get(k))
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let sentences = sentences(text);
    if sentences.is_empty() {
        return true;
    }
    sentences.iter().all(|s| {
        let norm = normalize(s);
        matches_phrase(&norm, en, no_speech_prob) || matches_phrase(&norm, native, no_speech_prob)
    })
}

/// Full post-filter for one Whisper utterance: strip markers, collapse loops, drop whole-utterance
/// hallucinations. `None` means "emit nothing".
pub fn clean_transcript(text: &str, language: Option<&str>, no_speech_prob: f32) -> Option<String> {
    let stripped = strip_non_speech_markers(text);
    let collapsed = collapse_repeats(&stripped);
    let collapsed = collapsed.trim();
    if collapsed.is_empty() || is_hallucination(collapsed, language, no_speech_prob) {
        None
    } else {
        Some(collapsed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_phrase_file_parses_and_is_non_trivial() {
        for (lang, src) in SOURCES {
            let phrases = parse_phrases(src);
            assert!(phrases.len() >= 8, "{lang}: only {} phrases", phrases.len());
            assert!(phrases.iter().all(|p| p.norm.chars().count() >= 2), "{lang}: degenerate phrase");
        }
        assert_eq!(list_key("zh-TW"), Some("zh"));
        assert_eq!(list_key("yue"), Some("zh"));
        assert_eq!(list_key("pt_br"), Some("pt"));
        assert_eq!(list_key("auto"), None);
    }

    /// No-speech probability of a hallucination on silence / noise (measured 0.88-0.95).
    const NOISE: f32 = 0.9;

    #[test]
    fn thanks_for_watching_is_dropped_only_as_a_whole_segment() {
        assert_eq!(clean_transcript("Thanks for watching!", Some("en"), NOISE), None);
        assert_eq!(clean_transcript(" Thank you for watching. ", Some("en"), NOISE), None);
        assert_eq!(clean_transcript("Thanks for watching! Please subscribe.", Some("auto"), NOISE), None);
        // Streamers really say it: confidently transcribed speech is kept.
        for said in [
            "Thanks for watching!",
            "Please subscribe!",
            "Like and subscribe!",
            "Don't forget to like and subscribe!",
            "See you in the next video!",
        ] {
            assert_eq!(clean_transcript(said, Some("en"), 0.0).as_deref(), Some(said), "{said}");
            assert_eq!(clean_transcript(said, Some("en"), 0.02).as_deref(), Some(said), "{said}");
            assert_eq!(clean_transcript(said, Some("en"), NOISE), None, "{said}");
        }
        // Inside real speech it is kept verbatim, whatever Whisper's doubt.
        assert_eq!(
            clean_transcript("Thanks for watching my back in that fight", Some("en"), NOISE).as_deref(),
            Some("Thanks for watching my back in that fight")
        );
        assert_eq!(
            clean_transcript("Great game. Thanks for watching!", Some("en"), NOISE).as_deref(),
            Some("Great game. Thanks for watching!")
        );
        // "let's go" is real speech (it is in the raw Bag-of-Hallucinations list).
        assert_eq!(clean_transcript("Let's go!", Some("en"), 0.9).as_deref(), Some("Let's go!"));
        assert_eq!(clean_transcript("let's go", None, 0.0).as_deref(), Some("let's go"));
        // Credits nobody says are hard: dropped even when Whisper is sure it heard speech. Dotted
        // domains stay one sentence and match the prefix rule.
        assert_eq!(clean_transcript("Subtitles by the Amara.org community", Some("en"), 0.0), None);
        assert_eq!(clean_transcript("Subtitles by Steve.", Some("en"), 0.0), None);
        // English stock phrases are also caught when another language is configured.
        assert_eq!(clean_transcript("Thank you for watching!", Some("ja"), NOISE), None);
        assert_eq!(
            clean_transcript("Thank you for watching!", Some("ja"), 0.0).as_deref(),
            Some("Thank you for watching!")
        );
    }

    #[test]
    fn native_outros_are_soft_and_credit_lines_hard() {
        for (said, lang) in [
            ("ご視聴ありがとうございました", "ja"),
            ("チャンネル登録よろしくお願いします！", "ja"),
            ("谢谢观看", "zh"),
            ("我们下期再见！", "zh"),
            ("시청해주셔서 감사합니다.", "ko"),
            ("¡Gracias por ver el video!", "es"),
            ("Merci d'avoir regardé !", "fr"),
            ("Danke fürs Zuschauen!", "de"),
            ("Спасибо за просмотр!", "ru"),
            ("Ставьте лайки, подписывайтесь!", "ru"),
            ("Obrigado por assistir!", "pt"),
        ] {
            assert_eq!(clean_transcript(said, Some(lang), 0.01).as_deref(), Some(said), "{lang}: {said}");
            assert_eq!(clean_transcript(said, Some(lang), NOISE), None, "{lang}: {said}");
        }
        for (credit, lang) in [
            ("请不吝点赞 订阅 转发 打赏支持明镜与点点栏目", "zh"),
            ("字幕由Amara.org社区提供", "zh"),
            ("Untertitel im Auftrag des ZDF, 2017", "de"),
            ("Copyright WDR 2021", "de"),
            ("Субтитры сделал DimaTorzok", "ru"),
            ("Sous-titres réalisés par la communauté d'Amara.org", "fr"),
            ("Subtítulos realizados por la comunidad de Amara.org", "es"),
            ("Legendas pela comunidade Amara.org", "pt"),
            ("字幕作成：田中", "ja"),
            ("MBC 뉴스 이덕영입니다.", "ko"),
        ] {
            assert_eq!(clean_transcript(credit, Some(lang), 0.0), None, "{lang}: {credit}");
        }
    }

    #[test]
    fn soft_phrases_need_whisper_to_doubt_the_speech() {
        assert_eq!(clean_transcript("Thank you.", Some("en"), 0.02).as_deref(), Some("Thank you."));
        assert_eq!(clean_transcript("Thank you.", Some("en"), 0.6), None);
        assert_eq!(clean_transcript("you", Some("en"), 0.55), None);
        assert_eq!(clean_transcript("ありがとうございました", Some("ja"), 0.05).as_deref(), Some("ありがとうございました"));
        assert_eq!(clean_transcript("ありがとうございました。", Some("ja"), SOFT_NO_SPEECH_PROB), None);
        // Genuine short words at ordinary mic levels score 0.2-0.47 on the probe: they survive.
        assert_eq!(clean_transcript("Bye.", Some("en"), 0.3).as_deref(), Some("Bye."));
        assert_eq!(clean_transcript("Bye.", Some("en"), 0.265).as_deref(), Some("Bye."));
        assert_eq!(clean_transcript("Thanks!", Some("en"), 0.47).as_deref(), Some("Thanks!"));
        assert_eq!(clean_transcript("감사합니다", Some("ko"), 0.3).as_deref(), Some("감사합니다"));
        assert_eq!(clean_transcript("Bye.", Some("en"), NOISE), None);
    }

    #[test]
    fn native_lists_catch_their_stock_phrases() {
        assert_eq!(clean_transcript("ご視聴ありがとうございました", Some("ja"), NOISE), None);
        assert_eq!(clean_transcript("请不吝点赞 订阅 转发 打赏支持明镜与点点栏目", Some("zh"), 0.0), None);
        assert_eq!(clean_transcript("字幕由Amara.org社区提供", Some("zh"), 0.0), None);
        assert_eq!(clean_transcript("시청해주셔서 감사합니다.", Some("ko"), NOISE), None);
        assert_eq!(clean_transcript("¡Gracias por ver el video!", Some("es"), NOISE), None);
        assert_eq!(clean_transcript("Sous-titres réalisés par la communauté d'Amara.org", Some("fr"), 0.0), None);
        assert_eq!(clean_transcript("Untertitel im Auftrag des ZDF, 2017", Some("de"), 0.0), None);
        assert_eq!(clean_transcript("Продолжение следует...", Some("ru"), NOISE), None);
        assert_eq!(clean_transcript("Obrigado por assistir!", Some("pt"), NOISE), None);
        // Full-width punctuation and spacing do not matter.
        assert_eq!(clean_transcript("ご視聴 ありがとう ございました！", Some("ja"), NOISE), None);
    }

    #[test]
    fn cjk_loops_are_collapsed_with_character_ngrams() {
        assert_eq!(collapse_repeats("ありがとうありがとうありがとうありがとう"), "ありがとう");
        assert_eq!(
            collapse_repeats("今日はありがとうありがとうありがとうございます"),
            "今日はありがとうございます"
        );
        // A loop cut off mid-unit at the end loses the stub.
        assert_eq!(collapse_repeats("好的好的好的好的好的好"), "好的");
        assert_eq!(collapse_repeats("我觉得我觉得我觉得我觉得我觉"), "我觉得");
        // Natural repetition survives: two-char units need 4, single chars need 6.
        assert_eq!(collapse_repeats("はいはいはい"), "はいはいはい");
        assert_eq!(collapse_repeats("ははは"), "ははは");
        assert_eq!(collapse_repeats("谢谢谢谢"), "谢谢谢谢");
        // Spaced CJK phrases loop too.
        assert_eq!(collapse_repeats("ありがとう ありがとう ありがとう"), "ありがとう");
        assert_eq!(
            clean_transcript("ご視聴ありがとうございましたご視聴ありがとうございましたご視聴ありがとうございました", Some("ja"), NOISE),
            None
        );
    }

    #[test]
    fn digit_and_latin_runs_inside_cjk_tokens_are_never_shortened() {
        assert_eq!(clean_transcript("賞金は1000000円でした", Some("ja"), 0.0).as_deref(), Some("賞金は1000000円でした"));
        assert_eq!(collapse_repeats("パスワードは111111です"), "パスワードは111111です");
        assert_eq!(collapse_repeats("我中了一等奖奖金是1000000元"), "我中了一等奖奖金是1000000元");
        assert_eq!(collapse_repeats("本当に面白いwwwwww"), "本当に面白いwwwwww");
        // CJK loops next to digits still collapse.
        assert_eq!(collapse_repeats("第1回好的好的好的好的"), "第1回好的");
    }

    #[test]
    fn word_loops_collapse_only_at_three_or_more() {
        assert_eq!(collapse_repeats("I'm going I'm going I'm going to the shop"), "I'm going to the shop");
        assert_eq!(collapse_repeats("Thank you. Thank you. Thank you. Thank you."), "Thank you.");
        assert_eq!(collapse_repeats("so so so so so so so so"), "so");
        assert_eq!(collapse_repeats("we did it we did it we did it we did"), "we did it");
        // Two repeats, and short single words up to three times, are natural speech.
        assert_eq!(collapse_repeats("let's go let's go"), "let's go let's go");
        assert_eq!(collapse_repeats("no no no"), "no no no");
        assert_eq!(collapse_repeats("go go go, we're late"), "go go go, we're late");
        assert_eq!(collapse_repeats(""), "");
    }

    #[test]
    fn non_speech_markers_are_stripped() {
        assert_eq!(strip_non_speech_markers("[MUSIC] hello (laughs) there ♪"), "hello there");
        assert_eq!(strip_non_speech_markers("（拍手）こんにちは【音楽】"), "こんにちは");
        assert_eq!(strip_non_speech_markers("unbalanced (bracket"), "unbalanced (bracket");
        assert_eq!(clean_transcript("[BLANK_AUDIO]", Some("en"), 0.0), None);
        assert_eq!(clean_transcript("(music) ...", Some("en"), 0.0), None);
    }
}
