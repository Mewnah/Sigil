//! Chinese readings (P6-L): Mandarin pinyin with tone sandhi and per-character ruby segments.
//!
//! # Pipeline
//! 1. `jieba-rs` (its bundled dictionary) splits the text into words with part-of-speech tags.
//!    Its HMM stays off: it glues unknown neighbours into "words" such as 人长 that hide the
//!    context below; text outside the dictionary stays one character per word.
//! 2. Syllables per word: the phrase table `zh_phrases.txt` first (neutral tones such as 东西
//!    `dōngxi`, polyphones such as 银行 `yínháng`, erhua such as 一会儿 `yíhuìr`), matched on one to
//!    three whole words or inside a longer word; then context rules: for one-character words
//!    (地 / 得 / 长 / 只 / 种 / 过 / 教 / 了, e.g. 跑**得**快 `de`, 我**得**走 `děi`) and inside words
//!    (一**只** `zhī`, 说**得**对 `de`, 听**不**懂 `bu`, the noun suffix 子 `zi`); then the `pinyin`
//!    crate's most common reading per character (with fixes for 长 `cháng`, 谁 `shéi`, 似 `sì`).
//! 3. Tone sandhi, in this order: 一 (`yí` before a fourth or neutral tone, `yì` before the others,
//!    `yī` as a number, at the end of a word and before digits, `yi` between a repeated verb:
//!    看一看); 不 (`bú` before a fourth tone, `bu` in 是不是); a word said twice keeps its sandhi
//!    (不要不要 `bú yào bú yào`, 一步一步 `yí bù yí bù`); then the third tone within a word
//!    (all but the last of a run of third tones become second: 可以 `kéyǐ`, 展览馆 `zhánlánguǎn`).
//! 4. Output: the pinyin line (syllables spaced, erhua `r` joined to its syllable, Chinese
//!    punctuation as ASCII, other text kept) and one ruby segment per character.

use std::collections::HashMap;

use jieba_rs::Jieba;
use pinyin::ToPinyin;

use super::{PinyinStyle, ReadingSegment};

/// The phrase table (see its header).
const PHRASES: &str = include_str!("zh_phrases.txt");
/// Most jieba words one phrase-table entry may span.
const MAX_PHRASE_WORDS: usize = 3;

/// A Han character (CJK unified ideographs, extensions, compatibility ideographs, 〇).
pub(crate) fn is_han(c: char) -> bool {
    matches!(c, '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}' | '\u{F900}'..='\u{FAFF}' | '\u{20000}'..='\u{3134F}' | '〇')
}

/// A digit, as written in Chinese numbers (not the magnitudes 十 / 百 / 千 / 万 / 亿).
fn is_digit_char(c: char) -> bool {
    matches!(c, '〇' | '零' | '一' | '二' | '三' | '四' | '五' | '六' | '七' | '八' | '九' | '两') || c.is_ascii_digit() || ('０'..='９').contains(&c)
}

/// Any numeral character, magnitudes included.
fn is_numeral(c: char) -> bool {
    is_digit_char(c) || matches!(c, '十' | '百' | '千' | '万' | '亿')
}

/// One pinyin syllable: base spelling (`ü` kept), tone 1–4 or 5 (neutral).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Syl {
    pub base: String,
    pub tone: u8,
    /// Kept out of tone sandhi (一 as a number).
    pub fixed: bool,
}

impl Syl {
    fn new(base: &str, tone: u8) -> Self {
        Self {
            base: base.to_string(),
            tone,
            fixed: false,
        }
    }

    /// `zhong1`, `ma5`, `nü3` / `nv3`, `r5`, `yi1!` (fixed).
    pub(crate) fn parse(s: &str) -> Option<Self> {
        let (s, fixed) = match s.strip_suffix('!') {
            Some(rest) => (rest, true),
            None => (s, false),
        };
        let tone = s.chars().last()?.to_digit(10)?;
        if !(1..=5).contains(&tone) {
            return None;
        }
        let base = s[..s.len() - 1].replace('v', "ü");
        if base.is_empty() || !base.chars().all(|c| c.is_ascii_lowercase() || c == 'ü' || c == 'ê') {
            return None;
        }
        Some(Self {
            base,
            tone: tone as u8,
            fixed,
        })
    }

    /// The `pinyin` crate's most common reading of `c`.
    fn of_char(c: char) -> Option<Self> {
        let p = c.to_pinyin()?;
        let numbered = p.with_tone_num_end();
        let tone = numbered.chars().last().and_then(|d| d.to_digit(10)).map_or(5, |d| d as u8);
        Some(Self::new(p.plain(), if (1..=4).contains(&tone) { tone } else { 5 }))
    }

    fn is_erhua(&self) -> bool {
        self.base == "r"
    }

    pub(crate) fn render(&self, style: PinyinStyle) -> String {
        if self.is_erhua() {
            return "r".into();
        }
        match style {
            PinyinStyle::None => self.base.clone(),
            PinyinStyle::Numbers => format!("{}{}", self.base, self.tone),
            PinyinStyle::Marks => with_tone_mark(&self.base, self.tone),
        }
    }
}

/// The toned form of a vowel (or of `n` / `m` in interjections).
fn toned(c: char, tone: u8) -> Option<String> {
    let set: [&str; 4] = match c {
        'a' => ["ā", "á", "ǎ", "à"],
        'e' => ["ē", "é", "ě", "è"],
        'i' => ["ī", "í", "ǐ", "ì"],
        'o' => ["ō", "ó", "ǒ", "ò"],
        'u' => ["ū", "ú", "ǔ", "ù"],
        'ü' => ["ǖ", "ǘ", "ǚ", "ǜ"],
        'ê' => ["ê\u{304}", "ế", "ê\u{30C}", "ề"],
        'n' => ["n\u{304}", "ń", "ň", "ǹ"],
        'm' => ["m\u{304}", "ḿ", "m\u{30C}", "m\u{300}"],
        _ => return None,
    };
    set.get(usize::from(tone).checked_sub(1)?).map(|s| s.to_string())
}

/// Standard tone-mark placement: `a`, else `e`, else the `o` of `ou`, else the last vowel; a
/// syllable without a vowel (`n`, `ng`, `m`, `hm`) marks its `n` / `m`.
fn with_tone_mark(base: &str, tone: u8) -> String {
    if !(1..=4).contains(&tone) {
        return base.to_string();
    }
    let chars: Vec<char> = base.chars().collect();
    let pos = chars
        .iter()
        .position(|&c| c == 'a')
        .or_else(|| chars.iter().position(|&c| c == 'e' || c == 'ê'))
        .or_else(|| chars.windows(2).position(|w| w == ['o', 'u']))
        .or_else(|| chars.iter().rposition(|&c| matches!(c, 'i' | 'o' | 'u' | 'ü')))
        .or_else(|| chars.iter().position(|&c| c == 'n' || c == 'm'));
    let Some(pos) = pos else { return base.to_string() };
    let Some(mark) = toned(chars[pos], tone) else { return base.to_string() };
    let mut out: String = chars[..pos].iter().collect();
    out.push_str(&mark);
    out.extend(&chars[pos + 1..]);
    out
}

/// What a jieba word (or a run of one) is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Han,
    Space,
    Punct,
    Other,
}

#[derive(Clone, Debug)]
struct Word<'a> {
    text: &'a str,
    tag: &'a str,
    kind: Kind,
}

fn kind_of(c: char) -> Kind {
    if is_han(c) {
        Kind::Han
    } else if c.is_whitespace() || c == '\u{3000}' {
        Kind::Space
    } else if c.is_alphanumeric() {
        Kind::Other
    } else {
        Kind::Punct
    }
}

/// Split a jieba word into runs of one kind (jieba keeps most mixed words apart already).
fn push_runs<'a>(out: &mut Vec<Word<'a>>, text: &'a str, tag: &'a str) {
    let mut start = 0;
    let mut kind: Option<Kind> = None;
    for (i, c) in text.char_indices() {
        let k = kind_of(c);
        // Punctuation is split per character so each mark gets its own spacing.
        if kind.is_some_and(|prev| prev != k || k == Kind::Punct) {
            out.push(Word {
                text: &text[start..i],
                tag,
                kind: kind.unwrap_or(Kind::Other),
            });
            start = i;
        }
        kind = Some(k);
    }
    if let Some(k) = kind {
        out.push(Word {
            text: &text[start..],
            tag,
            kind: k,
        });
    }
}

/// A Han syllable in text order.
#[derive(Clone, Debug)]
struct HanSyl {
    ch: char,
    syl: Syl,
    /// Sandhi word (third-tone sandhi stays inside one).
    group: usize,
    /// Position and length within the sandhi word.
    pos: usize,
    len: usize,
    /// Non-Han text (punctuation, a space, Latin) comes between this syllable and the previous one.
    break_before: bool,
}

/// Readings of one Chinese text.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChineseReadings {
    pub pinyin: String,
    pub segments: Vec<ReadingSegment>,
}

/// jieba plus the phrase table.
pub struct ChineseReader {
    jieba: Jieba,
    phrases: HashMap<String, Vec<Syl>>,
    /// Longest phrase (characters).
    max_phrase: usize,
}

impl Default for ChineseReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse the phrase table: `(word, syllables)` per entry, or the first problem.
pub(crate) fn parse_phrases(table: &str) -> Result<Vec<(String, Vec<Syl>)>, String> {
    let mut out = Vec::new();
    for (n, line) in table.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let word = parts.next().unwrap_or_default();
        let syls: Option<Vec<Syl>> = parts.map(Syl::parse).collect();
        let Some(syls) = syls else {
            return Err(format!("line {}: malformed syllable in '{line}'", n + 1));
        };
        if !word.chars().all(is_han) || word.chars().count() != syls.len() || syls.is_empty() {
            return Err(format!("line {}: '{word}' needs one syllable per character", n + 1));
        }
        out.push((word.to_string(), syls));
    }
    Ok(out)
}

impl ChineseReader {
    pub fn new() -> Self {
        let mut phrases = HashMap::new();
        // `the_phrase_table_is_well_formed` keeps the bundled table valid.
        for (word, syls) in parse_phrases(PHRASES).unwrap_or_else(|e| {
            tracing::warn!(target: super::LOG, "zh_phrases.txt: {e}");
            Vec::new()
        }) {
            phrases.insert(word, syls);
        }
        let max_phrase = phrases.keys().map(|w| w.chars().count()).max().unwrap_or(1);
        Self {
            jieba: Jieba::new(),
            phrases,
            max_phrase,
        }
    }

    /// Words of `text` with their jieba tags, split into runs of one kind.
    fn words<'a>(&'a self, text: &'a str) -> Vec<Word<'a>> {
        let mut words = Vec::new();
        // No HMM: it glues unknown neighbours into "words" (人长, 他长) that hide the context
        // rules; known words are what the phrase table and the rules need.
        for tag in self.jieba.tag(text, false) {
            push_runs(&mut words, tag.word, tag.tag);
        }
        words
    }

    /// jieba's words and tags of `text` (development output of the tests).
    #[cfg(test)]
    pub(crate) fn tokens(&self, text: &str) -> Vec<(String, String)> {
        self.words(text).iter().map(|w| (w.text.to_string(), w.tag.to_string())).collect()
    }

    /// Syllables of a Han word from the table, inside-word table matches and the character rules.
    fn word_syllables(&self, words: &[Word<'_>], k: usize) -> Vec<Syl> {
        let word = words[k].text;
        if let Some(s) = self.phrases.get(word) {
            return s.clone();
        }
        let chars: Vec<char> = word.chars().collect();
        let prev = k.checked_sub(1).map(|i| &words[i]);
        if chars.len() == 1 {
            let next = words.get(k + 1);
            return vec![single_char_syllable(chars[0], words[k].tag, prev, next)];
        }
        // Longest table matches inside the word, left to right; the rest per character.
        let mut out = Vec::with_capacity(chars.len());
        let mut i = 0;
        while i < chars.len() {
            let longest = (2..=self.max_phrase.min(chars.len() - i)).rev().find_map(|n| {
                let piece: String = chars[i..i + n].iter().collect();
                self.phrases.get(&piece).map(|s| (n, s))
            });
            if let Some((n, s)) = longest {
                out.extend(s.iter().cloned());
                i += n;
            } else {
                out.push(char_in_word(&chars, i));
                i += 1;
            }
        }
        // jieba glues the particle 地 to the verb after a reduplicated adverb (高高兴兴 / 地走了).
        if chars[0] == '地' && prev.is_some_and(|p| p.kind == Kind::Han && is_reduplicated(p.text)) {
            if let Some(first) = out.first_mut() {
                *first = Syl::new("de", 5);
            }
        }
        out
    }

    /// Readings of `text`.
    pub fn read(&self, text: &str, style: PinyinStyle) -> ChineseReadings {
        let words = self.words(text);
        // Syllables per word (None for non-Han words) and the sandhi group of each.
        let mut per_word: Vec<Option<Vec<Syl>>> = vec![None; words.len()];
        let mut group_of: Vec<usize> = vec![0; words.len()];
        let mut k = 0;
        let mut group = 0;
        while k < words.len() {
            if words[k].kind != Kind::Han {
                k += 1;
                continue;
            }
            // A table entry spanning several whole words (还/钱 → 还钱).
            let span = (2..=MAX_PHRASE_WORDS).rev().find_map(|m| {
                let ws = words.get(k..k + m)?;
                if !ws.iter().all(|w| w.kind == Kind::Han) {
                    return None;
                }
                let joined: String = ws.iter().map(|w| w.text).collect();
                let syls = self.phrases.get(&joined)?;
                Some((m, syls))
            });
            match span {
                Some((m, syls)) => {
                    let mut at = 0;
                    for w in k..k + m {
                        let n = words[w].text.chars().count();
                        per_word[w] = Some(syls[at..at + n].to_vec());
                        group_of[w] = group;
                        at += n;
                    }
                    k += m;
                }
                None => {
                    per_word[k] = Some(self.word_syllables(&words, k));
                    group_of[k] = group;
                    k += 1;
                }
            }
            group += 1;
        }

        // Flat syllables in text order, for sandhi.
        let mut flat: Vec<HanSyl> = Vec::new();
        let mut gap = true;
        for (w, word) in words.iter().enumerate() {
            match &per_word[w] {
                Some(syls) => {
                    for (c, s) in word.text.chars().zip(syls) {
                        flat.push(HanSyl {
                            ch: c,
                            syl: s.clone(),
                            group: group_of[w],
                            pos: 0,
                            len: 0,
                            break_before: std::mem::take(&mut gap),
                        });
                    }
                }
                None => gap = true,
            }
        }
        // Position of each syllable within its sandhi group.
        let mut start = 0;
        while start < flat.len() {
            let mut end = start;
            while end < flat.len() && flat[end].group == flat[start].group {
                end += 1;
            }
            for (i, s) in flat[start..end].iter_mut().enumerate() {
                s.pos = i;
                s.len = end - start;
            }
            start = end;
        }
        apply_sandhi(&mut flat);

        // Output.
        let mut tokens: Vec<(String, Glue)> = Vec::new();
        let mut segments: Vec<ReadingSegment> = Vec::with_capacity(text.chars().count());
        let mut next_syl = flat.iter();
        for (w, word) in words.iter().enumerate() {
            match word.kind {
                Kind::Han if per_word[w].is_some() => {
                    for c in word.text.chars() {
                        let s = next_syl.next().map(|h| &h.syl);
                        let rendered = s.map(|s| s.render(style)).unwrap_or_default();
                        let glue = if s.is_some_and(Syl::is_erhua) { Glue::Join } else { Glue::Space };
                        tokens.push((rendered.clone(), glue));
                        segments.push(ReadingSegment {
                            text: c.to_string(),
                            reading: Some(rendered).filter(|r| !r.is_empty()),
                        });
                    }
                }
                Kind::Space => segments.push(ReadingSegment {
                    text: word.text.to_string(),
                    reading: None,
                }),
                Kind::Punct => {
                    tokens.push(punct_ascii(word.text));
                    segments.push(ReadingSegment {
                        text: word.text.to_string(),
                        reading: None,
                    });
                }
                _ => {
                    tokens.push((word.text.to_string(), Glue::Space));
                    segments.push(ReadingSegment {
                        text: word.text.to_string(),
                        reading: None,
                    });
                }
            }
        }
        ChineseReadings {
            pinyin: join_tokens(&tokens),
            segments,
        }
    }
}

/// Demonstratives and quantity words a measure word follows.
const DEMONSTRATIVES: [&str; 11] = ["这", "那", "哪", "几", "每", "某", "一", "两", "半", "各", "多"];

fn tag_is(word: Option<&Word<'_>>, prefix: char) -> bool {
    word.is_some_and(|w| w.tag.starts_with(prefix))
}

fn follows_quantity(prev: Option<&Word<'_>>) -> bool {
    prev.is_some_and(|w| w.tag.starts_with('m') || w.tag.starts_with('q') || DEMONSTRATIVES.contains(&w.text) || w.text.chars().last().is_some_and(is_numeral))
}

/// AA or AABB (慢慢, 渐渐, 高高兴兴): a reduplicated adverb or adjective.
fn is_reduplicated(word: &str) -> bool {
    let c: Vec<char> = word.chars().collect();
    match c.as_slice() {
        [a, b] => a == b,
        [a, b, x, y] => a == b && x == y,
        _ => false,
    }
}

/// The word before a one-character 地 makes it the adverbial particle `de` (慢慢 / 认真 / 高兴 / 不停).
fn adverbial_before_de(prev: &Word<'_>) -> bool {
    prev.tag.starts_with(['a', 'd', 'z', 'b']) || is_reduplicated(prev.text)
}

/// The word before a one-character 地 makes it a noun (`dì`): a demonstrative or measure (这块地,
/// 这片地), a preposition (把地扫), 的 (他的地), a noun or conjunction (天和地), or a one-character
/// verb taking it as its object (买地, 扫地).
fn noun_before_di(prev: &Word<'_>) -> bool {
    prev.tag.starts_with(['r', 'p', 'q', 'u', 'n', 'c', 'f', 's', 't']) || (prev.tag.starts_with('v') && prev.text.chars().count() == 1)
}

/// A one-character word (`tag` is its jieba tag), read from its neighbours.
fn single_char_syllable(c: char, tag: &str, prev: Option<&Word<'_>>, next: Option<&Word<'_>>) -> Syl {
    let s = |base: &str, tone: u8| Syl::new(base, tone);
    match c {
        // 慢慢地走 / 开心地笑: the adverbial particle after an adverb or before a verb; the noun dì
        // (ground, land) at the start, after a demonstrative, preposition or 的, and always before
        // 是 / 有 (这块地是我的, 地有多大).
        '地' => {
            let Some(prev) = prev.filter(|p| p.kind == Kind::Han) else {
                return s("di", 4);
            };
            if next.is_some_and(|w| matches!(w.text, "是" | "有")) {
                s("di", 4)
            } else if adverbial_before_de(prev) || (tag_is(next, 'v') && !noun_before_di(prev)) {
                s("de", 5)
            } else {
                s("di", 4)
            }
        }
        // 跑得快 (complement) / 我得走 (must) / 得了 (obtain).
        '得' => {
            if tag_is(prev, 'v') || tag_is(prev, 'a') {
                s("de", 5)
            } else if tag_is(next, 'v') {
                s("dei", 3)
            } else {
                s("de", 2)
            }
        }
        // 长得 / 长了 / 长大 (grow) vs 很长 (long).
        '长' => {
            if next.is_some_and(|w| w.text.starts_with(['得', '了', '大', '高', '出', '满'])) {
                s("zhang", 3)
            } else {
                s("chang", 2)
            }
        }
        // 一只猫 (measure word) vs 只有 (only).
        '只' => {
            if follows_quantity(prev) {
                s("zhi", 1)
            } else {
                s("zhi", 3)
            }
        }
        // 这种 (kind) vs 种花 (plant).
        '种' => {
            if follows_quantity(prev) {
                s("zhong", 3)
            } else if tag_is(next, 'n') {
                s("zhong", 4)
            } else {
                s("zhong", 3)
            }
        }
        // 去过 (experienced aspect) vs 过 (pass).
        '过' => {
            if tag_is(prev, 'v') {
                s("guo", 5)
            } else {
                s("guo", 4)
            }
        }
        // On its own 教 is the verb "teach" (她教数学, 教我); the noun jiào (teaching, religion)
        // lives in words jieba keeps whole (教育, 宗教, 请教) or in the phrase table.
        '教' => {
            if tag.starts_with('n') || tag == "vn" {
                s("jiao", 4)
            } else {
                s("jiao", 1)
            }
        }
        // V不了 (unable to): liǎo.
        '了' => {
            if prev.is_some_and(|w| w.text == "不" || w.text == "不得") {
                s("liao", 3)
            } else {
                s("le", 5)
            }
        }
        '子' => s("zi", 3),
        _ => char_default(c),
    }
}

/// Readings of 不 in a three-character word that are not the V不C pattern.
const NOT_V_BU_C: [&str; 2] = ["要不是", "要不然"];

/// A character inside a longer word (not covered by the table), read from its neighbours there.
fn char_in_word(chars: &[char], i: usize) -> Syl {
    let len = chars.len();
    let prev = i.checked_sub(1).map(|p| chars[p]);
    let word = || chars.iter().collect::<String>();
    match chars[i] {
        // The noun suffix 子 is neutral (孩子, 桌子); a leading 子 is zǐ (子弹, 子女).
        '子' if i + 1 == len && len > 1 => Syl::new("zi", 5),
        '子' => Syl::new("zi", 3),
        // 一只 / 三只 / 这只: the measure word.
        '只' if prev.is_some_and(|p| is_numeral(p) || "这那哪几每某各半".contains(p)) => Syl::new("zhi", 1),
        // V得C (说得对, 来得及): the complement particle; 不得了 / 不得不 keep dé.
        '得' if len == 3 && i == 1 && chars[0] != '不' => Syl::new("de", 5),
        // V不C (听不懂, 看不见): unstressed.
        '不' if len == 3 && i == 1 && !NOT_V_BU_C.contains(&word().as_str()) => Syl::new("bu", 5),
        // V不了 / V得了 (大不了, 受得了, 不得了): able or unable to, liǎo.
        '了' if i + 1 == len && len >= 3 && matches!(prev, Some('不' | '得')) => Syl::new("liao", 3),
        // AA地 as one word (静静地, 偷偷地, 轻轻地): the adverbial particle.
        '地' if i + 1 == len && i >= 2 && chars[i - 1] == chars[i - 2] => Syl::new("de", 5),
        c => char_default(c),
    }
}

/// The most common reading of a character: a few corrections to the per-character table, then
/// the `pinyin` crate's first reading.
fn char_default(c: char) -> Syl {
    match c {
        '长' => Syl::new("chang", 2),
        '谁' => Syl::new("shei", 2),
        '似' => Syl::new("si", 4),
        _ => Syl::of_char(c).unwrap_or_else(|| Syl::new("", 5)),
    }
}

/// 不 / 一 changes, then third-tone sandhi within each word (see the module docs).
fn apply_sandhi(syls: &mut [HanSyl]) {
    let n = syls.len();
    for i in 0..n {
        if syls[i].syl.fixed {
            continue;
        }
        let prev = (i > 0 && !syls[i].break_before).then(|| syls[i - 1].clone());
        let next = (i + 1 < n && !syls[i + 1].break_before).then(|| syls[i + 1].clone());
        // The same 一 / 不 two syllables back: a word said twice (不要不要, 一步一步), not X一X.
        let said_twice = prev.is_some() && i >= 2 && !syls[i - 1].break_before && syls[i - 2].ch == syls[i].ch;
        // X一X / X不X (看一看, 是不是): the verb or adjective around 一 / 不 repeated.
        let repeated = !said_twice && matches!((&prev, &next), (Some(p), Some(nx)) if p.ch == nx.ch && p.ch != syls[i].ch);
        let cur = &mut syls[i];
        match (cur.ch, cur.syl.base.as_str(), cur.syl.tone) {
            ('一', "yi", 1) => {
                let as_number = prev.as_ref().is_some_and(|p| is_numeral(p.ch) || p.ch == '第')
                    || next.as_ref().is_some_and(|nx| is_digit_char(nx.ch))
                    || (cur.len > 1 && cur.pos + 1 == cur.len);
                if as_number {
                    continue;
                }
                if repeated {
                    cur.syl.tone = 5;
                } else if let Some(nx) = &next {
                    cur.syl.tone = if matches!(nx.syl.tone, 4 | 5) { 2 } else { 4 };
                }
            }
            ('不', "bu", 4) => {
                if repeated {
                    cur.syl.tone = 5;
                } else if next.as_ref().is_some_and(|nx| nx.syl.tone == 4 && nx.ch != '不') {
                    cur.syl.tone = 2;
                }
            }
            _ => {}
        }
    }
    // Third tone: runs of third tones inside one word.
    let mut i = 0;
    while i < n {
        if syls[i].syl.tone != 3 || syls[i].syl.fixed {
            i += 1;
            continue;
        }
        let mut end = i + 1;
        while end < n
            && syls[end].group == syls[i].group
            && !syls[end].break_before
            && syls[end].syl.tone == 3
            && !syls[end].syl.fixed
        {
            end += 1;
        }
        for s in &mut syls[i..end - 1] {
            s.syl.tone = 2;
        }
        i = end;
    }
}

/// How a pinyin token joins the one before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Glue {
    Space,
    Join,
    Open,
    Close,
}

/// Chinese punctuation as ASCII, and how it attaches.
fn punct_ascii(p: &str) -> (String, Glue) {
    let close = |s: &str| (s.to_string(), Glue::Close);
    let open = |s: &str| (s.to_string(), Glue::Open);
    match p {
        "，" | "," | "、" => close(","),
        "。" | "." | "．" => close("."),
        "！" | "!" => close("!"),
        "？" | "?" => close("?"),
        "：" | ":" => close(":"),
        "；" | ";" => close(";"),
        "…" | "……" => close("..."),
        "～" | "~" => close("~"),
        "“" | "「" | "『" | "《" | "〈" => open("\""),
        "”" | "」" | "』" | "》" | "〉" => close("\""),
        "（" | "(" => open("("),
        "）" | ")" => close(")"),
        "【" | "［" | "[" => open("["),
        "】" | "］" | "]" => close("]"),
        other => (other.to_string(), Glue::Space),
    }
}

fn join_tokens(tokens: &[(String, Glue)]) -> String {
    let mut out = String::new();
    let mut after_open = true;
    for (text, glue) in tokens {
        if text.is_empty() {
            continue;
        }
        let space = !out.is_empty() && !after_open && matches!(glue, Glue::Space | Glue::Open);
        if space {
            out.push(' ');
        }
        out.push_str(text);
        after_open = *glue == Glue::Open;
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syllables_parse_and_render() {
        let s = Syl::parse("zhong1").unwrap();
        assert_eq!((s.base.as_str(), s.tone, s.fixed), ("zhong", 1, false));
        assert_eq!(Syl::parse("nv3").unwrap().base, "nü");
        assert!(Syl::parse("yi1!").unwrap().fixed);
        for bad in ["zhong", "zhong6", "Zhong1", "1", "zh-ong1", ""] {
            assert!(Syl::parse(bad).is_none(), "{bad}");
        }
        let r = |s: &str, st| Syl::parse(s).unwrap().render(st);
        assert_eq!(r("hao3", PinyinStyle::Marks), "hǎo");
        assert_eq!(r("gui4", PinyinStyle::Marks), "guì");
        assert_eq!(r("liu2", PinyinStyle::Marks), "liú");
        assert_eq!(r("dou1", PinyinStyle::Marks), "dōu");
        assert_eq!(r("lüe4", PinyinStyle::Marks), "lüè");
        assert_eq!(r("nü3", PinyinStyle::Marks), "nǚ");
        assert_eq!(r("ma5", PinyinStyle::Marks), "ma");
        assert_eq!(r("ma5", PinyinStyle::Numbers), "ma5");
        assert_eq!(r("lü4", PinyinStyle::Numbers), "lü4");
        assert_eq!(r("lü4", PinyinStyle::None), "lü");
        assert_eq!(r("r5", PinyinStyle::Numbers), "r");
        assert_eq!(r("n2", PinyinStyle::Marks), "ń");
        assert_eq!(r("hng2", PinyinStyle::Marks), "hńg");
    }

    #[test]
    fn the_phrase_table_is_well_formed() {
        let entries = parse_phrases(PHRASES).expect("zh_phrases.txt parses");
        assert!(entries.len() > 400, "{} entries", entries.len());
        let mut seen = std::collections::HashSet::new();
        for (word, _) in &entries {
            assert!(seen.insert(word.clone()), "duplicate entry {word}");
        }
        assert!(parse_phrases("银行 yin2").is_err());
        assert!(parse_phrases("银行 yin2 hangx").is_err());
        assert!(parse_phrases("ab a1 b1").is_err());
    }

    #[test]
    fn crate_readings_keep_tones_and_neutral() {
        assert_eq!(Syl::of_char('中'), Some(Syl::new("zhong", 1)));
        assert_eq!(Syl::of_char('了'), Some(Syl::new("le", 5)));
        assert_eq!(Syl::of_char('绿'), Some(Syl::new("lü", 4)));
        assert_eq!(Syl::of_char('a'), None);
    }

    #[test]
    fn punctuation_and_tokens_join() {
        let t = |s: &str, g| (s.to_string(), g);
        assert_eq!(
            join_tokens(&[t("nǐ", Glue::Space), t("hǎo", Glue::Space), t(",", Glue::Close), t("\"", Glue::Open), t("hǎo", Glue::Space), t("\"", Glue::Close)]),
            "nǐ hǎo, \"hǎo\""
        );
        assert_eq!(join_tokens(&[t("yí", Glue::Space), t("huì", Glue::Space), t("r", Glue::Join)]), "yí huìr");
    }

    #[test]
    fn character_classes() {
        assert!(is_han('中') && is_han('〇') && !is_han('ア') && !is_han('a'));
        assert!(is_digit_char('三') && is_digit_char('7') && !is_digit_char('十'));
        assert!(is_numeral('十') && is_numeral('两'));
    }
}
