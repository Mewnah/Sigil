//! Katakana mora ↔ phoneme tables of Style-Bert-VITS2 (`mora_list.json` and the loader are from
//! `sbv2_core`, MIT, Copyright (c) 2024 tuna2134, 2025- neodyland; the table itself is
//! Style-Bert-VITS2's `mora_list.py`).

use std::collections::HashMap;
use std::sync::LazyLock;

use serde::Deserialize;

#[derive(Deserialize)]
struct Mora {
    mora: String,
    consonant: Option<String>,
    vowel: String,
}

#[derive(Deserialize)]
struct MoraFile {
    minimum: Vec<Mora>,
    additional: Vec<Mora>,
}

static MORA_FILE: LazyLock<MoraFile> =
    LazyLock::new(|| serde_json::from_str(include_str!("mora_list.json")).expect("the bundled mora list parses"));

/// Katakana mora → (consonant, vowel), every mora including the additional spellings (`ヴォ`, ...).
pub static KATA_TO_PHONEMES: LazyLock<HashMap<String, (Option<String>, String)>> = LazyLock::new(|| {
    MORA_FILE
        .minimum
        .iter()
        .chain(MORA_FILE.additional.iter())
        .map(|m| (m.mora.clone(), (m.consonant.clone(), m.vowel.clone())))
        .collect()
});

/// Longest katakana mora key, in characters (greedy matching tries this many first).
pub static MAX_MORA_CHARS: LazyLock<usize> =
    LazyLock::new(|| KATA_TO_PHONEMES.keys().map(|k| k.chars().count()).max().unwrap_or(1));

/// Vowels and `N` (a long-vowel mark after one of these repeats it).
pub const VOWELS: [&str; 6] = ["a", "i", "u", "e", "o", "N"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_loads() {
        assert_eq!(KATA_TO_PHONEMES.get("キャ"), Some(&(Some("ky".to_string()), "a".to_string())));
        assert_eq!(KATA_TO_PHONEMES.get("ン"), Some(&(None, "N".to_string())));
        assert_eq!(KATA_TO_PHONEMES.get("ッ"), Some(&(None, "q".to_string())));
        assert_eq!(*MAX_MORA_CHARS, 2);
    }
}
