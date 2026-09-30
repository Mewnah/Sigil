//! Text chunking for local synthesis.
//!
//! sherpa-onnx synthesizes a chunk sentence by sentence and cannot stop inside a sentence, so the
//! engine hands it one chunk at a time:
//! - [`chunks`] splits a text into sentences (`.` `!` `?` `…` `。` `！` `？` and line breaks, keeping
//!   closing quotes and brackets with their sentence), merges fragments without letters into a
//!   neighbour, and splits sentences longer than [`MAX_CHUNK_CHARS`] at clause punctuation, then
//!   at spaces, then anywhere (CJK text without punctuation).
//! - With `first_clause`, a long first sentence is split once more at its first clause boundary
//!   (`,` `;` `:` `、` `，` ...), so the first audio of a burst starts after one clause instead of a
//!   whole sentence (Kokoro on 4 CPU threads needs about 0.3 s per second of speech). Only the
//!   first chunk is split this way, so later sentences keep whole-sentence prosody.
//!
//! The TS sentence pipeline already sends one sentence per request; this also covers previews,
//! legacy whole-line callers and sentences the pipeline merged.

/// Longest chunk handed to the engine (Kokoro takes at most ~510 phoneme tokens per call).
pub const MAX_CHUNK_CHARS: usize = 300;
/// A first clause is split off only when it and the rest have at least this many characters
/// (letters and digits; CJK characters count double).
pub const MIN_CLAUSE_WEIGHT: usize = 16;

fn is_terminator(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…' | '。' | '！' | '？' | '‼' | '⁇' | '⁈' | '⁉' | '؟' | '।' | '॥')
}

/// Characters that belong to the sentence they follow (closing quotes / brackets).
fn is_closer(c: char) -> bool {
    matches!(
        c,
        '"' | '\'' | ')' | ']' | '}' | '」' | '』' | '）' | '】' | '》' | '〉' | '”' | '’' | '»' | '›'
    )
}

fn is_clause_break(c: char) -> bool {
    matches!(c, ',' | ';' | ':' | '、' | '，' | '；' | '：' | '—' | '–' | '،' | '؛')
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF   // kana
        | 0x3400..=0x4DBF // CJK ext A
        | 0x4E00..=0x9FFF // CJK unified
        | 0xAC00..=0xD7AF // Hangul
        | 0xF900..=0xFAFF)
}

/// Letters and digits count 1, CJK characters 2 (they carry a syllable each).
fn weight(s: &str) -> usize {
    s.chars()
        .map(|c| {
            if is_cjk(c) {
                2
            } else if c.is_alphanumeric() {
                1
            } else {
                0
            }
        })
        .sum()
}

fn has_speech(s: &str) -> bool {
    s.chars().any(char::is_alphanumeric)
}

/// Split `text` into sentences (see the module docs). Whitespace is collapsed; the result never
/// contains empty strings.
pub fn sentences(text: &str) -> Vec<String> {
    let text: String = text
        .chars()
        .map(|c| if c.is_control() && c != '\n' { ' ' } else { c })
        .collect();
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\n' {
            push_sentence(&mut out, &mut cur);
            i += 1;
            continue;
        }
        cur.push(c);
        if is_terminator(c) {
            // Runs of terminators ("?!", "...") and closing quotes stay together.
            let mut j = i + 1;
            while j < chars.len() && (is_terminator(chars[j]) || is_closer(chars[j])) {
                cur.push(chars[j]);
                j += 1;
            }
            // An ASCII '.' ends a sentence only before whitespace or the end ("3.14", "v1.0" stay
            // whole) and not after a common abbreviation ("Mr. Smith"); CJK / other terminators
            // always end it.
            let ascii_dot = c == '.';
            let at_boundary = j >= chars.len() || chars[j].is_whitespace() || !ascii_dot;
            let abbreviation = ascii_dot && j == i + 1 && ends_with_abbreviation(&cur);
            if at_boundary && !abbreviation {
                push_sentence(&mut out, &mut cur);
            }
            i = j;
            continue;
        }
        i += 1;
    }
    push_sentence(&mut out, &mut cur);
    merge_fragments(out)
}

/// Abbreviations whose period does not end a sentence (English titles and Latin short forms).
const ABBREVIATIONS: &[&str] = &["mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "e.g", "i.e", "approx"];

/// `cur` ends with "<abbreviation>." as a whole word.
fn ends_with_abbreviation(cur: &str) -> bool {
    let body = cur.strip_suffix('.').unwrap_or(cur);
    let word = body
        .rsplit(|c: char| c.is_whitespace() || c == '(' || c == '"')
        .next()
        .unwrap_or("");
    !word.is_empty() && ABBREVIATIONS.iter().any(|a| a.eq_ignore_ascii_case(word))
}

fn push_sentence(out: &mut Vec<String>, cur: &mut String) {
    let s = cur.split_whitespace().collect::<Vec<_>>().join(" ");
    cur.clear();
    if !s.is_empty() {
        out.push(s);
    }
}

/// Fragments without letters or digits ("...", "!!", a lone quote) join the previous sentence
/// (or the next one when they come first).
fn merge_fragments(parts: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut pending = String::new();
    for p in parts {
        if !has_speech(&p) {
            match out.last_mut() {
                Some(last) => {
                    last.push_str(&p);
                }
                None => {
                    pending.push_str(&p);
                    pending.push(' ');
                }
            }
            continue;
        }
        if pending.is_empty() {
            out.push(p);
        } else {
            out.push(format!("{}{p}", std::mem::take(&mut pending)));
        }
    }
    out.into_iter().filter(|s| has_speech(s)).collect()
}

/// Split one sentence into pieces of at most `max` characters: at clause punctuation first, then
/// at spaces, then at any character boundary.
pub fn split_long(sentence: &str, max: usize) -> Vec<String> {
    let max = max.max(8);
    if sentence.chars().count() <= max {
        return vec![sentence.to_string()];
    }
    let mut out = Vec::new();
    let mut rest: &str = sentence;
    while rest.chars().count() > max {
        let cut = best_cut(rest, max);
        let (head, tail) = rest.split_at(cut);
        let head = head.trim();
        if !head.is_empty() {
            out.push(head.to_string());
        }
        rest = tail.trim_start();
    }
    if !rest.trim().is_empty() {
        out.push(rest.trim().to_string());
    }
    out
}

/// Byte index to cut `s` at, within its first `max` characters: after the last clause break,
/// else at the last space, else at `max` characters.
fn best_cut(s: &str, max: usize) -> usize {
    let limit = s
        .char_indices()
        .nth(max)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    let window = &s[..limit];
    // Keep at least a quarter of the window so a comma near the start does not make tiny pieces.
    let min = window.len() / 4;
    if let Some((i, c)) = window
        .char_indices()
        .rev()
        .find(|(i, c)| is_clause_break(*c) && *i >= min)
    {
        return i + c.len_utf8();
    }
    if let Some((i, _)) = window
        .char_indices()
        .rev()
        .find(|(i, c)| c.is_whitespace() && *i >= min)
    {
        return i;
    }
    limit.max(s.chars().next().map(char::len_utf8).unwrap_or(1))
}

/// Split off the first clause of `sentence` when both parts are long enough to be worth
/// synthesizing separately (see [`MIN_CLAUSE_WEIGHT`]); `None` keeps the sentence whole.
pub fn split_first_clause(sentence: &str) -> Option<(String, String)> {
    for (i, c) in sentence.char_indices() {
        if !is_clause_break(c) {
            continue;
        }
        let cut = i + c.len_utf8();
        let (head, tail) = sentence.split_at(cut);
        if weight(head) < MIN_CLAUSE_WEIGHT {
            continue;
        }
        if weight(tail) < MIN_CLAUSE_WEIGHT {
            return None;
        }
        return Some((head.trim().to_string(), tail.trim().to_string()));
    }
    None
}

/// The chunks to synthesize for `text`, in order (see the module docs).
pub fn chunks(text: &str, first_clause: bool) -> Vec<String> {
    let mut out: Vec<String> = sentences(text)
        .iter()
        .flat_map(|s| split_long(s, MAX_CHUNK_CHARS))
        .collect();
    if first_clause {
        if let Some(first) = out.first().cloned() {
            if let Some((head, tail)) = split_first_clause(&first) {
                out.splice(0..1, [head, tail]);
            }
        }
    }
    out
}
