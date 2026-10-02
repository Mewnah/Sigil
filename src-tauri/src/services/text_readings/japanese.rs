//! Japanese readings (P6-L): hiragana, Hepburn romaji and ruby segments from jpreprocess.
//!
//! # Pipeline
//! 1. jpreprocess tokenizes the text (normalized for NAIST-JDIC: ASCII and half-width kana widened)
//!    and runs OpenJTalk's NJD rules (unknown kana words, numbers with counters such as
//!    `3本` → `サンボン`, accent phrases). The NJD features (`Vec<String>`, the same CSV jpreprocess
//!    prints) are parsed into [`Morph`]s, so everything after the analyzer is plain data that the
//!    unit tests feed directly.
//! 2. [`align`] maps the morphs back onto the original text. The NJD rules rewrite digits into
//!    kanji numerals (and insert or drop nodes) and drop spaces and unreadable Latin words, so a
//!    morph that does not match the text at the cursor starts a group that ends where a later morph
//!    matches again (the earliest such place): the skipped text belongs to the skipped morphs.
//! 3. Each aligned unit gets its reading: hiragana (`ー` kept), romaji per word ([`super::romaji`]),
//!    and a ruby segment when its text holds kanji or digits (okurigana and leading kana split off,
//!    so `食べ` is `食`(た) + `べ`). Latin words, symbols and spaces keep their own text (the romaji
//!    line narrows full-width letters). A dictionary word that mixes Latin with kana or kanji keeps
//!    its letters and reads the rest when the letters' names start or end its reading (Ｘ線
//!    `Ｘせん` / `X-sen` / `Ｘ` + `線`(せん), Ｔシャツ `T-shatsu`); otherwise it is read whole (Ｗ杯
//!    `わーるどかっぷ`).

use std::ops::Range;
use std::path::{Path, PathBuf};

use jpreprocess::{normalize_text_for_naist_jdic, DefaultTokenizer, JPreprocess, SystemDictionaryConfig};

use super::romaji::{self, is_hiragana, is_kana, is_katakana, to_hiragana};
use super::ReadingSegment;

/// Catalog id of the NAIST-JDIC dictionary built for jpreprocess 0.15 (shared with the
/// Style-Bert-VITS2 frontend; the entry lives in `model-catalog/runtime-sbv2.json`). The dictionary
/// must come from the same jpreprocess release as the crate.
pub const DICTIONARY_ID: &str = "text/jpreprocess-naist-jdic";

/// Files of a jpreprocess (lindera) system dictionary folder.
const DICTIONARY_FILES: [&str; 8] = [
    "char_def.bin",
    "dict.da",
    "dict.vals",
    "dict.words",
    "dict.wordsidx",
    "matrix.mtx",
    "metadata.json",
    "unk.bin",
];

/// The dictionary folder inside `root`: `root` itself or one folder below it (an archive that kept
/// its top folder).
pub fn dictionary_dir(root: &Path) -> Option<PathBuf> {
    let is_dict = |d: &Path| DICTIONARY_FILES.iter().all(|f| d.join(f).is_file());
    if is_dict(root) {
        return Some(root.to_path_buf());
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(root)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir() && is_dict(p))
        .collect();
    found.sort();
    found.into_iter().next()
}

/// The loaded analyzer (OpenJTalk frontend on NAIST-JDIC).
pub struct JapaneseAnalyzer {
    jp: JPreprocess<DefaultTokenizer>,
}

impl JapaneseAnalyzer {
    /// Load the jpreprocess dictionary in `dir` (the folder, or a folder holding it).
    pub fn load(dir: &Path) -> Result<Self, String> {
        let dict = dictionary_dir(dir).ok_or_else(|| format!("{} does not hold a jpreprocess dictionary", dir.display()))?;
        let system = SystemDictionaryConfig::File(dict.clone())
            .load()
            .map_err(|e| format!("cannot load the dictionary in {}: {e}", dict.display()))?;
        Ok(Self {
            jp: JPreprocess::with_dictionaries(system, None),
        })
    }

    /// The NJD morphs of `text` after OpenJTalk's rules.
    pub fn morphs(&self, text: &str) -> Result<Vec<Morph>, String> {
        Ok(Morph::parse_all(&self.features(text)?))
    }

    /// The NJD features of `text` (jpreprocess `run_frontend`).
    pub fn features(&self, text: &str) -> Result<Vec<String>, String> {
        self.jp
            .run_frontend(text)
            .map_err(|e| format!("Japanese text analysis failed: {e}"))
    }
}

/// One NJD node: `string,pos,pos1,pos2,pos3,ctype,cform,orig,read,pron,acc/mora,chain_rule,chain_flag`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Morph {
    pub surface: String,
    /// Part of speech and its three subgroups (`*` when unset).
    pub pos: [String; 4],
    /// Katakana reading (`None` when the dictionary has none).
    pub read: Option<String>,
    /// Katakana pronunciation with `ー` long vowels (unvoicing marks removed).
    pub pron: Option<String>,
}

/// Detail fields after the surface in an NJD feature line.
const DETAIL_FIELDS: usize = 12;

impl Morph {
    /// Parse one NJD feature line (`None` when malformed).
    pub fn parse(feature: &str) -> Option<Self> {
        let parts: Vec<&str> = feature.split(',').collect();
        if parts.len() < DETAIL_FIELDS + 1 {
            return None;
        }
        let (surface, details) = parts.split_at(parts.len() - DETAIL_FIELDS);
        let kana = |s: &str| -> Option<String> {
            let s = s.replace(['’', '\''], "");
            (!s.is_empty() && s.chars().all(|c| is_katakana(c) || is_hiragana(c))).then_some(s)
        };
        Some(Self {
            surface: surface.join(","),
            pos: [details[0], details[1], details[2], details[3]].map(str::to_string),
            read: kana(details[7]),
            pron: kana(details[8]),
        })
    }

    pub fn parse_all(features: &[String]) -> Vec<Self> {
        features.iter().filter_map(|f| Self::parse(f)).collect()
    }

    fn is(&self, pos: &str) -> bool {
        self.pos[0] == pos
    }

    fn is_sub(&self, pos: &str, sub: &str) -> bool {
        self.pos[0] == pos && self.pos[1] == sub
    }

    fn is_symbol(&self) -> bool {
        self.is("記号")
    }
}

// ---------------------------------------------------------------------------------------------
// Alignment
// ---------------------------------------------------------------------------------------------

/// Voiced / semi-voiced sound marks that combine with the kana before them when normalized.
fn is_sound_mark(c: char) -> bool {
    matches!(c, '\u{3099}' | '\u{309A}' | '\u{309B}' | '\u{309C}' | '\u{FF9E}' | '\u{FF9F}')
}

/// The text as jpreprocess normalizes it (one char per entry) and, per normalized char, the byte
/// range of the original text it came from.
pub(crate) fn normalize_with_map(text: &str) -> (Vec<char>, Vec<Range<usize>>) {
    let mut norm = Vec::with_capacity(text.len());
    let mut src = Vec::with_capacity(text.len());
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let end_of = |i: usize| chars.get(i + 1).map_or(text.len(), |(b, _)| *b);
    let mut i = 0;
    while i < chars.len() {
        let start = chars[i].0;
        let take = if chars.get(i + 1).is_some_and(|(_, n)| is_sound_mark(*n)) { 2 } else { 1 };
        let piece = &text[start..end_of(i + take - 1)];
        let normalized: Vec<char> = normalize_text_for_naist_jdic(piece).chars().collect();
        match normalized.as_slice() {
            [] => {}
            [one] => {
                norm.push(*one);
                src.push(start..end_of(i + take - 1));
            }
            _ => {
                // Not combined: one normalized char per original char.
                for (k, n) in normalized.iter().enumerate().take(take) {
                    norm.push(*n);
                    src.push(chars[i + k].0..end_of(i + k));
                }
            }
        }
        i += take;
    }
    (norm, src)
}

/// Morphs `morphs` read the normalized text `text` (char indices).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Unit {
    pub text: Range<usize>,
    pub morphs: Range<usize>,
}

/// How far ahead a mismatch looks for the morph that matches the text again.
const RESYNC_MORPHS: usize = 24;

fn find_at_or_after(text: &[char], from: usize, pat: &[char]) -> Option<usize> {
    if pat.is_empty() || pat.len() > text.len() {
        return None;
    }
    (from..=text.len() - pat.len()).find(|&p| text[p..].starts_with(pat))
}

/// Map the morphs onto the normalized text (see the module docs). Every char of `text` is in
/// exactly one unit; units are in text order.
pub(crate) fn align(text: &[char], morphs: &[Morph]) -> Vec<Unit> {
    let surfaces: Vec<Vec<char>> = morphs.iter().map(|m| m.surface.chars().collect()).collect();
    let mut units = Vec::with_capacity(morphs.len() + 2);
    let (mut c, mut i) = (0usize, 0usize);
    // Morphs without text of their own (inserted numerals) join the next unit.
    let mut carry: Option<usize> = None;
    while i < morphs.len() && c < text.len() {
        let s = &surfaces[i];
        if !s.is_empty() && text[c..].starts_with(s) {
            let first = carry.take().unwrap_or(i);
            units.push(Unit {
                text: c..c + s.len(),
                morphs: first..i + 1,
            });
            c += s.len();
            i += 1;
            continue;
        }
        // Resynchronize at the earliest place a later morph matches the text again.
        let best = (i..(i + RESYNC_MORPHS).min(morphs.len()))
            .filter_map(|j| find_at_or_after(text, c, &surfaces[j]).map(|p| (p, j)))
            .min();
        match best {
            Some((p, j)) if p > c => {
                let first = carry.take().unwrap_or(i);
                units.push(Unit { text: c..p, morphs: first..j });
                c = p;
                i = j;
            }
            Some((_, j)) => {
                // Morphs i..j have no text of their own (p == c, j > i).
                carry.get_or_insert(i);
                i = j;
            }
            None => {
                let first = carry.take().unwrap_or(i);
                units.push(Unit {
                    text: c..text.len(),
                    morphs: first..morphs.len(),
                });
                c = text.len();
                i = morphs.len();
            }
        }
    }
    if c < text.len() {
        let m = carry.take().map_or(morphs.len()..morphs.len(), |first| first..morphs.len());
        units.push(Unit { text: c..text.len(), morphs: m });
    }
    units
}

// ---------------------------------------------------------------------------------------------
// Readings
// ---------------------------------------------------------------------------------------------

/// Readings of one Japanese text.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JapaneseReadings {
    pub hiragana: Option<String>,
    pub romaji: Option<String>,
    pub segments: Vec<ReadingSegment>,
}

/// A CJK ideograph (kanji), including the iteration and abbreviation marks.
pub(crate) fn is_kanji(c: char) -> bool {
    matches!(c, '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}' | '\u{F900}'..='\u{FAFF}' | '\u{20000}'..='\u{3134F}' | '々' | '〆' | 'ヶ' | 'ヵ')
}

fn is_digit(c: char) -> bool {
    c.is_ascii_digit() || ('０'..='９').contains(&c)
}

fn is_latin(c: char) -> bool {
    c.is_ascii_alphabetic() || ('Ａ'..='Ｚ').contains(&c) || ('ａ'..='ｚ').contains(&c)
}

/// What one unit is, for the readings.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Piece<'a> {
    /// Words with a katakana reading (one morph, or a group such as a rewritten number).
    Word { reading: String, morphs: &'a [Morph] },
    /// Punctuation and other symbols.
    Symbol,
    /// Spaces, Latin words and anything without a reading: kept as written.
    Plain,
}

fn classify<'a>(original: &str, morphs: &'a [Morph]) -> Piece<'a> {
    if morphs.is_empty() {
        return Piece::Plain;
    }
    if original.chars().any(is_latin) {
        // Latin words (ＣＤ, VRChat) keep their spelling. A dictionary word that mixes Latin with
        // kana or kanji (Ｘ線, Ｔシャツ, Ｊリーグ) is read when its morphs spell exactly that text.
        let japanese = original.chars().any(|c| is_kana(c) || is_kanji(c));
        let surfaces: String = morphs.iter().map(|m| m.surface.as_str()).collect();
        if !japanese || morphs.iter().any(Morph::is_symbol) || surfaces != normalize_text_for_naist_jdic(original) {
            return Piece::Plain;
        }
    }
    if morphs.iter().all(Morph::is_symbol) {
        return Piece::Symbol;
    }
    let mut reading = String::new();
    for m in morphs {
        match (display_kana(m), m.is_symbol()) {
            (Some(r), false) => reading.push_str(&r),
            (_, true) => {}
            (None, false) if m.surface.chars().all(is_kana) => reading.push_str(&m.surface),
            (None, false) => return Piece::Plain,
        }
    }
    if reading.is_empty() {
        Piece::Plain
    } else {
        Piece::Word { reading, morphs }
    }
}

/// The kana a word's reading is written with: the dictionary reading, except that numbers and
/// counters take the sound changes OpenJTalk's number rules put in the pronunciation (三本
/// `サンボン`, 二十歳 `ニジュッサイ`), keeping the spelled vowel where the pronunciation has `ー`.
/// Other words keep their reading's spelling (particles `は`, long vowels `とうきょう`).
fn display_kana(m: &Morph) -> Option<String> {
    let read = m.read.as_deref()?;
    let numeric = m.is_sub("名詞", "数") || (m.is_sub("名詞", "接尾") && m.pos[2] == "助数詞");
    let Some(pron) = m.pron.as_deref().filter(|_| numeric) else {
        return Some(read.to_string());
    };
    let (rm, pm) = (romaji::morae(read), romaji::morae(pron));
    if rm.len() != pm.len() {
        // A shortened number (四 ヨン → ヨ in 四年): the pronunciation, unless it only differs by
        // long vowels the spelling keeps.
        return Some(if pron.contains('ー') { read } else { pron }.to_string());
    }
    Some(rm.iter().zip(&pm).map(|(r, p)| if p == "ー" { r.as_str() } else { p.as_str() }).collect())
}

/// Full-width Latin letters and digits as ASCII (for the romaji line).
fn narrow_latin(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            'Ａ'..='Ｚ' | 'ａ'..='ｚ' | '０'..='９' => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
            _ => c,
        })
        .collect()
}

/// How the dictionary spells a Latin letter inside a word, in katakana (Ｔシャツ `ティーシャツ`,
/// Ｊリーグ `ジェイリーグ`, Ｘ線 `エックスセン`).
fn letter_names(c: char) -> &'static [&'static str] {
    match narrow_latin(&c.to_string()).chars().next().unwrap_or(c).to_ascii_uppercase() {
        'A' => &["エー", "エイ"],
        'B' => &["ビー"],
        'C' => &["シー"],
        'D' => &["ディー", "デー"],
        'E' => &["イー"],
        'F' => &["エフ"],
        'G' => &["ジー"],
        'H' => &["エイチ", "エッチ"],
        'I' => &["アイ"],
        'J' => &["ジェー", "ジェイ"],
        'K' => &["ケー", "ケイ"],
        'L' => &["エル"],
        'M' => &["エム"],
        'N' => &["エヌ"],
        'O' => &["オー"],
        'P' => &["ピー"],
        'Q' => &["キュー"],
        'R' => &["アール"],
        'S' => &["エス"],
        'T' => &["ティー", "ティ"],
        'U' => &["ユー"],
        'V' => &["ブイ", "ヴィー"],
        'W' => &["ダブリュー", "ダブリュ"],
        'X' => &["エックス"],
        'Y' => &["ワイ"],
        'Z' => &["ゼット", "ゼッド", "ズィー"],
        _ => &[],
    }
}

/// Most Latin letters a mixed word may spell out before or after its kana / kanji.
const MAX_SPELLED_LETTERS: usize = 6;

/// `reading` without the spoken names of `letters` at its start.
fn strip_letter_names<'r>(letters: &[char], reading: &'r str) -> Option<&'r str> {
    let Some((first, rest)) = letters.split_first() else { return Some(reading) };
    letter_names(*first)
        .iter()
        .find_map(|name| strip_letter_names(rest, reading.strip_prefix(name)?))
}

/// `reading` without the spoken names of `letters` at its end.
fn strip_letter_names_end<'r>(letters: &[char], reading: &'r str) -> Option<&'r str> {
    let Some((last, rest)) = letters.split_last() else { return Some(reading) };
    letter_names(*last)
        .iter()
        .find_map(|name| strip_letter_names_end(rest, reading.strip_suffix(name)?))
}

/// A word that mixes Latin letters with kana or kanji: the letters before and after (kept as
/// written) and the rest with its own reading and pronunciation (Ｘ線 → `Ｘ` + 線 `セン`).
#[derive(Clone, Debug, PartialEq, Eq)]
struct LatinSplit {
    prefix: String,
    core: String,
    read: String,
    pron: String,
    suffix: String,
}

/// Split a mixed word (see [`LatinSplit`]) when the letters' names account for the start and
/// end of its reading; `None` otherwise (Ｗ杯 `ワールドカップ`), and the word is read whole.
fn split_latin(text: &str, read: &str, pron: &str) -> Option<LatinSplit> {
    let chars: Vec<char> = text.chars().collect();
    let lead = chars.iter().take_while(|&&c| is_latin(c)).count();
    let trail = chars[lead..].iter().rev().take_while(|&&c| is_latin(c)).count();
    let (prefix, rest) = chars.split_at(lead);
    let (core, suffix) = rest.split_at(rest.len() - trail);
    if core.is_empty() || lead + trail > MAX_SPELLED_LETTERS || !core.iter().all(|&c| is_kana(c) || is_kanji(c)) {
        return None;
    }
    let cut = |r: &str| -> Option<String> {
        let r = strip_letter_names(prefix, r)?;
        let r = strip_letter_names_end(suffix, r)?;
        (!r.is_empty()).then(|| r.to_string())
    };
    let core_read = cut(read)?;
    let core: String = core.iter().collect();
    // Kana read as they are spelled: anything else means the letters were not read by name.
    if core.chars().all(is_kana) && to_hiragana(&core) != to_hiragana(&core_read) {
        return None;
    }
    Some(LatinSplit {
        prefix: prefix.iter().collect(),
        pron: cut(pron).unwrap_or_else(|| core_read.clone()),
        read: core_read,
        core,
        suffix: suffix.iter().collect(),
    })
}

/// Ruby for `text` read `reading_hira`, with okurigana and leading kana split off: hiragana, or
/// romaji when only romaji is requested (`whole_romaji` when nothing was split off).
fn push_ruby(segments: &mut Vec<ReadingSegment>, text: &str, reading_hira: &str, hiragana: bool, whole_romaji: &str) {
    let (prefix, core, core_reading, suffix) = split_okurigana(text, reading_hira);
    let ruby = if hiragana {
        core_reading
    } else if prefix.is_empty() && suffix.is_empty() {
        // The whole unit: its own romaji (macrons, particles).
        romaji::finish(whole_romaji)
    } else {
        romaji::finish(&romaji::word_romaji(&core_reading, &core_reading))
    };
    segments.push(ReadingSegment { text: prefix, reading: None });
    segments.push(ReadingSegment { text: core, reading: Some(ruby) });
    segments.push(ReadingSegment { text: suffix, reading: None });
}

/// Romaji of a symbol (Japanese punctuation as ASCII) and how it attaches.
fn symbol_romaji(text: &str) -> (String, Glue) {
    let t = text.trim();
    let mapped = match t {
        "、" | "，" | "," => ",",
        "。" | "．" | "." => ".",
        "！" | "!" => "!",
        "？" | "?" => "?",
        "：" | ":" => ":",
        "；" | ";" => ";",
        "・" => "·",
        "…" | "……" | "‥" => "...",
        "〜" | "～" | "~" => "~",
        "「" | "『" | "“" => return ("\"".into(), Glue::Open),
        "」" | "』" | "”" => return ("\"".into(), Glue::Close),
        "（" | "(" => return ("(".into(), Glue::Open),
        "）" | ")" => return (")".into(), Glue::Close),
        "［" | "[" | "【" => return ("[".into(), Glue::Open),
        "］" | "]" | "】" => return ("]".into(), Glue::Close),
        _ => return (t.to_string(), Glue::Space),
    };
    (mapped.to_string(), Glue::Close)
}

/// How a romaji token joins the one before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Glue {
    /// Separated by a space.
    Space,
    /// Written together (inflections, suffixes, closing punctuation).
    Join,
    /// Joined with a hyphen (name suffixes: `tanaka-san`).
    Hyphen,
    /// Opening bracket or quote: space before, nothing after.
    Open,
    /// Closing punctuation: nothing before.
    Close,
}

/// Endings that join a verb, adjective or auxiliary before them (`tabete`, `ikeba`).
const JOINING_PARTICLES: [&str; 5] = ["て", "で", "ば", "ちゃ", "じゃ"];

/// How a word joins the word before it.
fn word_glue(prev: Option<&Morph>, cur: &Morph) -> Glue {
    let Some(prev) = prev else { return Glue::Space };
    let inflecting = |m: &Morph| m.is("動詞") || m.is("形容詞") || m.is("助動詞");
    if prev.is("接頭詞") {
        return Glue::Join;
    }
    if cur.is_sub("名詞", "接尾") {
        return if cur.pos[2] == "人名" { Glue::Hyphen } else { Glue::Join };
    }
    if cur.is_sub("名詞", "数") && prev.is_sub("名詞", "数") {
        return Glue::Join;
    }
    if cur.is_sub("動詞", "接尾") || cur.is_sub("形容詞", "接尾") {
        return Glue::Join;
    }
    if cur.is("助動詞") && inflecting(prev) {
        // The copula after an adjective stays apart (`oishii desu`).
        let copula = matches!(cur.surface.as_str(), "です" | "でし" | "でしょ" | "だ" | "だっ" | "だろ");
        return if copula && prev.is("形容詞") { Glue::Space } else { Glue::Join };
    }
    if cur.is_sub("助詞", "接続助詞") && JOINING_PARTICLES.contains(&cur.surface.as_str()) && inflecting(prev) {
        return Glue::Join;
    }
    // The nominalizer ん joins its verb (`ikun da`).
    if cur.is_sub("名詞", "非自立") && cur.surface == "ん" {
        return Glue::Join;
    }
    Glue::Space
}

/// Join romaji tokens with their glue.
fn join_romaji(tokens: &[(String, Glue)]) -> String {
    let mut out = String::new();
    let mut after_open = true;
    for (text, glue) in tokens {
        if text.is_empty() {
            continue;
        }
        let space = !out.is_empty()
            && !after_open
            && match glue {
                Glue::Space | Glue::Open => true,
                Glue::Join | Glue::Hyphen | Glue::Close => false,
            };
        if space {
            out.push(' ');
        }
        if *glue == Glue::Hyphen && !out.is_empty() {
            out.push('-');
        }
        romaji::join_word(&mut out, text, *glue == Glue::Join);
        after_open = *glue == Glue::Open;
    }
    romaji::finish(out.trim())
}

/// Split leading and trailing kana of `text` off a reading that spells them the same way
/// (`食べ` / `たべ` → `食` / `た` + `べ`). Returns `(prefix, core, core reading, suffix)`; no
/// split when the core or its reading would be empty.
fn split_okurigana(text: &str, reading: &str) -> (String, String, String, String) {
    let t: Vec<char> = text.chars().collect();
    let r: Vec<char> = reading.chars().collect();
    let hira = |c: char| -> char { to_hiragana(&c.to_string()).chars().next().unwrap_or(c) };
    let same = |a: char, b: char| is_kana(a) && hira(a) == hira(b);
    let mut lead = 0;
    while lead < t.len() && lead < r.len() && same(t[lead], r[lead]) {
        lead += 1;
    }
    let mut trail = 0;
    while trail < t.len() - lead && trail < r.len() - lead && same(t[t.len() - 1 - trail], r[r.len() - 1 - trail]) {
        trail += 1;
    }
    let core: String = t[lead..t.len() - trail].iter().collect();
    let core_reading: String = r[lead..r.len() - trail].iter().collect();
    if core.is_empty() || core_reading.is_empty() {
        return (String::new(), text.to_string(), reading.to_string(), String::new());
    }
    (
        t[..lead].iter().collect(),
        core,
        core_reading,
        t[t.len() - trail..].iter().collect(),
    )
}

/// Readings of `original` from its morphs (see the module docs). `hiragana` / `romaji` pick the
/// forms; the segments carry hiragana when it is requested, otherwise romaji.
pub fn read(original: &str, morphs: &[Morph], hiragana: bool, romaji: bool) -> JapaneseReadings {
    let (norm, src) = normalize_with_map(original);
    let units = align(&norm, morphs);
    let mut hira_out = String::new();
    let mut romaji_tokens: Vec<(String, Glue)> = Vec::new();
    let mut segments: Vec<ReadingSegment> = Vec::new();
    let mut prev_word: Option<&Morph> = None;
    // Original bytes covered so far (chars the normalization dropped stay plain text).
    let mut covered = 0usize;

    for unit in &units {
        let start = src.get(unit.text.start).map_or(original.len(), |r| r.start);
        let end = unit.text.end.checked_sub(1).and_then(|e| src.get(e)).map_or(start, |r| r.end);
        if start > covered {
            push_plain(&original[covered..start], &mut hira_out, &mut romaji_tokens, &mut segments);
            prev_word = None;
        }
        let text = &original[start.max(covered)..end.max(covered)];
        covered = end.max(covered);
        if text.is_empty() {
            continue;
        }
        let unit_morphs = &morphs[unit.morphs.clone()];
        match classify(text, unit_morphs) {
            Piece::Word { reading, morphs: ms } => {
                let reading_hira = to_hiragana(&reading);
                let pron: String = ms.iter().map(|m| m.pron.clone().or_else(|| m.read.clone()).unwrap_or_default()).collect();
                // A kana word without a dictionary entry: its spelling is its pronunciation.
                let pron = if pron.is_empty() { reading.clone() } else { pron };
                let latin = text.chars().any(is_latin);
                if let Some(split) = latin.then(|| split_latin(text, &reading, &pron)).flatten() {
                    // Ｘ線 / Ｔシャツ: the letters as written, the rest read (`Ｘせん`, `X-sen`).
                    let core_hira = to_hiragana(&split.read);
                    hira_out.push_str(&split.prefix);
                    hira_out.push_str(&core_hira);
                    hira_out.push_str(&split.suffix);
                    let core_romaji = romaji::word_romaji(&split.read, &split.pron);
                    let token = [narrow_latin(&split.prefix), core_romaji.clone(), narrow_latin(&split.suffix)]
                        .into_iter()
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>()
                        .join("-");
                    romaji_tokens.push((token, word_glue(prev_word, &ms[0])));
                    segments.push(ReadingSegment { text: split.prefix, reading: None });
                    if split.core.chars().any(is_kanji) {
                        push_ruby(&mut segments, &split.core, &core_hira, hiragana, &core_romaji);
                    } else {
                        segments.push(ReadingSegment { text: split.core, reading: None });
                    }
                    segments.push(ReadingSegment { text: split.suffix, reading: None });
                    prev_word = ms.iter().rev().find(|m| !m.is_symbol());
                    continue;
                }
                hira_out.push_str(&reading_hira);
                // Romaji word by word (a group, such as a rewritten number, is one word).
                let word_romaji: Vec<(String, Glue)> = if ms.len() == 1 || ms.iter().all(|m| m.is_sub("名詞", "数")) {
                    vec![(romaji::word_romaji(&reading, &pron), word_glue(prev_word, &ms[0]))]
                } else {
                    let mut prev = prev_word;
                    ms.iter()
                        .filter(|m| !m.is_symbol())
                        .map(|m| {
                            let read = m.read.clone().unwrap_or_else(|| m.surface.clone());
                            let pron = m.pron.clone().unwrap_or_else(|| read.clone());
                            let g = word_glue(prev, m);
                            prev = Some(m);
                            (romaji::word_romaji(&read, &pron), g)
                        })
                        .collect()
                };
                // Kanji and digits get ruby; so do the letters of a mixed word read whole (Ｗ杯).
                let needs_ruby = latin || text.chars().any(|c| is_kanji(c) || is_digit(c));
                if needs_ruby {
                    let whole: String = word_romaji.iter().map(|(t, _)| t.as_str()).collect();
                    push_ruby(&mut segments, text, &reading_hira, hiragana, &whole);
                } else {
                    segments.push(ReadingSegment {
                        text: text.to_string(),
                        reading: None,
                    });
                }
                romaji_tokens.extend(word_romaji);
                prev_word = ms.iter().rev().find(|m| !m.is_symbol());
            }
            Piece::Symbol => {
                hira_out.push_str(text);
                romaji_tokens.push(symbol_romaji(text));
                segments.push(ReadingSegment {
                    text: text.to_string(),
                    reading: None,
                });
                prev_word = None;
            }
            Piece::Plain => {
                push_plain(text, &mut hira_out, &mut romaji_tokens, &mut segments);
                prev_word = None;
            }
        }
    }
    if covered < original.len() {
        push_plain(&original[covered..], &mut hira_out, &mut romaji_tokens, &mut segments);
    }

    JapaneseReadings {
        hiragana: hiragana.then_some(hira_out),
        romaji: romaji.then(|| join_romaji(&romaji_tokens)),
        segments: if hiragana || romaji { segments } else { Vec::new() },
    }
}

/// Text kept as written (spaces, Latin words, unreadable pieces; full-width letters and digits
/// are narrowed in the romaji line).
fn push_plain(text: &str, hira: &mut String, romaji_tokens: &mut Vec<(String, Glue)>, segments: &mut Vec<ReadingSegment>) {
    hira.push_str(text);
    let words: Vec<&str> = text.split(|c: char| c.is_whitespace() || c == '\u{3000}').filter(|w| !w.is_empty()).collect();
    for w in words {
        romaji_tokens.push((narrow_latin(w), Glue::Space));
    }
    segments.push(ReadingSegment {
        text: text.to_string(),
        reading: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_map_tracks_combined_marks() {
        let (norm, src) = normalize_with_map("ｶﾞa b");
        assert_eq!(norm.iter().collect::<String>(), "ガａ\u{3000}ｂ");
        assert_eq!(src, vec![0..6, 6..7, 7..8, 8..9]);
        let (norm, src) = normalize_with_map("漢字");
        assert_eq!(norm, vec!['漢', '字']);
        assert_eq!(src, vec![0..3, 3..6]);
    }

    #[test]
    fn okurigana_is_split_off() {
        assert_eq!(split_okurigana("食べ", "たべ"), ("".into(), "食".into(), "た".into(), "べ".into()));
        assert_eq!(split_okurigana("お茶", "おちゃ"), ("お".into(), "茶".into(), "ちゃ".into(), "".into()));
        assert_eq!(split_okurigana("取り扱い", "とりあつかい"), ("".into(), "取り扱".into(), "とりあつか".into(), "い".into()));
        assert_eq!(split_okurigana("東京", "とうきょう"), ("".into(), "東京".into(), "とうきょう".into(), "".into()));
        // Katakana okurigana compares as hiragana; nothing left to read means no split.
        assert_eq!(split_okurigana("ダメ", "だめ"), ("".into(), "ダメ".into(), "だめ".into(), "".into()));
    }

    #[test]
    fn mixed_latin_words_split_on_the_letter_names() {
        let split = |t: &str, r: &str| split_latin(t, r, r).map(|s| (s.prefix, s.core, s.read, s.suffix));
        let s = |a: &str, b: &str, c: &str, d: &str| Some((a.to_string(), b.to_string(), c.to_string(), d.to_string()));
        assert_eq!(split("Ｘ線", "エックスセン"), s("Ｘ", "線", "セン", ""));
        assert_eq!(split("Tシャツ", "ティーシャツ"), s("T", "シャツ", "シャツ", ""));
        assert_eq!(split("ｊリーグ", "ジェイリーグ"), s("ｊ", "リーグ", "リーグ", ""));
        // A letter after the kana, and several letters (each with spelling variants, backtracking).
        assert_eq!(split("ビタミンＣ", "ビタミンシー"), s("", "ビタミン", "ビタミン", "Ｃ"));
        assert_eq!(split("ＡＴ車", "エーティーシャ"), s("ＡＴ", "車", "シャ", ""));
        assert_eq!(split("ＡＴ車", "エイティシャ"), s("ＡＴ", "車", "シャ", ""));
        // Not read by the letters' names, kana that disagree, only letters, letters in the middle.
        assert_eq!(split("Ｗ杯", "ワールドカップ"), None);
        assert_eq!(split("Ｔシャツ", "ティーシャッツ"), None);
        assert_eq!(split("ＣＤ", "シーディー"), None);
        assert_eq!(split("大Ｘ線", "ダイエックスセン"), None);
        assert_eq!(split("Ｘ", "エックス"), None);
        // The pronunciation is split the same way, or falls back to the reading.
        let sp = split_latin("Ｂ級", "ビーキュウ", "ビーキュー").unwrap();
        assert_eq!((sp.read.as_str(), sp.pron.as_str()), ("キュウ", "キュー"));
        assert_eq!(narrow_latin("ＣＤ３ab"), "CD3ab");
    }

    #[test]
    fn romaji_tokens_join_with_their_glue() {
        let t = |s: &str, g| (s.to_string(), g);
        assert_eq!(
            join_romaji(&[t("watashi", Glue::Space), t("wa", Glue::Space), t("tabe", Glue::Space), t("mashi", Glue::Join), t("ta", Glue::Join), t(".", Glue::Close)]),
            "watashi wa tabemashita."
        );
        assert_eq!(join_romaji(&[t("\"", Glue::Open), t("hai", Glue::Space), t("\"", Glue::Close), t("to", Glue::Space)]), "\"hai\" to");
        assert_eq!(join_romaji(&[t("tanaka", Glue::Space), t("san", Glue::Hyphen)]), "tanaka-san");
    }
}
