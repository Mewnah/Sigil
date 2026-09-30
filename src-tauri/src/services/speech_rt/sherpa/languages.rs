//! Language names and tags the sherpa-onnx models expect, and the language sets of Sigil's catalog
//! models.
//!
//! Sigil speaks normalized lower-case base codes (`en`, `ja`, `zh`, `yue`, `fil`, see
//! `speech::lang::normalize`); each model family wants its own spelling:
//! - Qwen3-ASR prompts with the English language name (`language Japanese<asr_text>`), so the
//!   per-stream `language` option must be the name, never the code (a code puts the prompt off its
//!   training distribution).
//! - SenseVoice's config takes `auto`, `zh`, `en`, `ja`, `ko` or `yue`.
//! - Nemotron 3.5 streaming takes a locale tag or base code from its encoder's
//!   `prompt_dictionary` metadata; sherpa-onnx silently falls back to auto-detection for anything
//!   else, so [`PromptDictionary`] tells the engine up front.

use std::collections::BTreeMap;

/// Qwen3-ASR's 30 languages: `(code, name)` (Qwen3-ASR model card; dialects ride on `zh`).
pub const QWEN3_LANGUAGES: &[(&str, &str)] = &[
    ("zh", "Chinese"),
    ("en", "English"),
    ("yue", "Cantonese"),
    ("ar", "Arabic"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ko", "Korean"),
    ("ru", "Russian"),
    ("th", "Thai"),
    ("vi", "Vietnamese"),
    ("ja", "Japanese"),
    ("tr", "Turkish"),
    ("hi", "Hindi"),
    ("ms", "Malay"),
    ("nl", "Dutch"),
    ("sv", "Swedish"),
    ("da", "Danish"),
    ("fi", "Finnish"),
    ("pl", "Polish"),
    ("cs", "Czech"),
    ("fil", "Filipino"),
    ("fa", "Persian"),
    ("el", "Greek"),
    ("hu", "Hungarian"),
    ("mk", "Macedonian"),
    ("ro", "Romanian"),
];

/// Qwen3-ASR language name for a normalized code (`ja` → `Japanese`); `tl` (Tagalog) maps to
/// Filipino.
pub fn qwen3_language_name(code: &str) -> Option<&'static str> {
    let code = match code {
        "tl" => "fil",
        c => c,
    };
    QWEN3_LANGUAGES
        .iter()
        .find(|(c, _)| *c == code)
        .map(|(_, n)| *n)
}

/// Normalized code for a Qwen3-ASR language name or code (`Japanese` / `japanese` / `ja` → `ja`).
pub fn qwen3_language_code(name: &str) -> Option<&'static str> {
    let n = name.trim();
    QWEN3_LANGUAGES
        .iter()
        .find(|(c, l)| l.eq_ignore_ascii_case(n) || c.eq_ignore_ascii_case(n))
        .map(|(c, _)| *c)
}

/// SenseVoice's languages (its config also takes `auto`).
pub const SENSE_VOICE_LANGUAGES: &[&str] = &["zh", "en", "ja", "ko", "yue"];

/// Nemotron 3.5 ASR streaming: the locales measured on FLEURS / LibriSpeech (transcribe.cpp's
/// validation table, 2026-06). Its tokenizer lists 40 locales, 8 of them "adaptation-ready" (need
/// fine-tuning); only languages with published accuracy are offered. `zh` is weak (CER ~19–20);
/// the router sends Chinese finals elsewhere.
pub const NEMOTRON_LANGUAGES: &[&str] = &[
    "ar", "bg", "cs", "da", "de", "en", "es", "et", "fi", "fr", "hi", "hr", "hu", "it", "ja", "ko", "nb", "nl", "pl", "pt", "ro", "ru", "sk", "sv",
    "tr", "uk", "vi", "zh",
];

/// Nemotron 3.5's "transcription-ready" locales (model card: 19 locales, these 15 base codes): its
/// stream finals are caption quality here; the other 13 "broad-coverage" languages of
/// [`NEMOTRON_LANGUAGES`] work, but the router prefers an utterance model's final for them.
pub const NEMOTRON_TRANSCRIPTION_READY: &[&str] = &["ar", "de", "en", "es", "fr", "hi", "it", "ja", "ko", "nl", "pt", "ru", "tr", "uk", "vi"];

/// Parakeet TDT 0.6B v3 / Parakeet Ultra: 25 European languages (detected by the model itself).
pub const PARAKEET_LANGUAGES: &[&str] = &[
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv",
    "uk",
];

/// English-only models.
pub const ENGLISH: &[&str] = &["en"];

/// The `prompt_dictionary` of a prompt-conditioned NeMo streaming encoder (Nemotron 3.5): language
/// tag → prompt id, normalized and extended with base-code aliases exactly as sherpa-onnx 1.13.8
/// does (`online-transducer-nemo-model.cc`), so [`PromptDictionary::prompt_id`] answers what
/// sherpa will use.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PromptDictionary {
    ids: BTreeMap<String, i64>,
    /// Prompt id of `auto` (sherpa's default 101 unless the metadata names another).
    auto_id: i64,
}

/// sherpa-onnx's `kDefaultAutoPromptId`.
const DEFAULT_AUTO_PROMPT_ID: i64 = 101;

impl PromptDictionary {
    /// Parse the metadata values (`prompt_dictionary` JSON object, optional `auto_prompt_id`).
    /// `None` when the dictionary is missing or has no usable language (sherpa then runs every
    /// language on auto).
    pub fn parse(dictionary: &str, auto_prompt_id: Option<&str>) -> Option<Self> {
        let map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(dictionary).ok()?;
        // nlohmann::json (sherpa-onnx) iterates object keys in sorted order; so does this, whatever
        // serde_json's map order is.
        let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        let mut ids = BTreeMap::new();
        let mut ordered = Vec::new();
        for (tag, id) in entries {
            let Some(id) = id.as_i64().filter(|v| *v >= 0) else { continue };
            let norm = normalize_tag(tag);
            if norm.is_empty() {
                continue;
            }
            if !ids.contains_key(&norm) {
                ids.insert(norm.clone(), id);
                ordered.push((norm, id));
            }
        }
        let mut auto_id = auto_prompt_id
            .and_then(|v| v.trim().parse::<i64>().ok())
            .filter(|v| *v >= 0)
            .unwrap_or(DEFAULT_AUTO_PROMPT_ID);
        if let Some(id) = ids.get("auto") {
            auto_id = *id;
        }
        // Base-code aliases (`ja-jp` → `ja`), first tag wins, never overriding an explicit entry.
        for (tag, id) in &ordered {
            if let Some((base, _)) = tag.split_once('-').filter(|(b, _)| !b.is_empty()) {
                ids.entry(base.to_string()).or_insert(*id);
            }
        }
        ids.values()
            .any(|id| *id != auto_id)
            .then_some(Self { ids, auto_id })
    }

    /// The prompt id sherpa-onnx uses for `language` (`None`: it would fall back to auto).
    pub fn prompt_id(&self, language: &str) -> Option<i64> {
        let norm = normalize_tag(language);
        if norm.is_empty() || norm == "auto" {
            return None;
        }
        self.ids
            .get(&norm)
            .copied()
            .filter(|id| *id != self.auto_id)
    }

    /// Whether `language` selects a language prompt (not auto).
    pub fn supports(&self, language: &str) -> bool {
        self.prompt_id(language).is_some()
    }

    /// Every base code with a language prompt, sorted.
    pub fn base_codes(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .ids
            .iter()
            .filter(|(tag, id)| !tag.contains('-') && **id != self.auto_id)
            .map(|(tag, _)| tag.clone())
            .collect();
        v.sort();
        v.dedup();
        v
    }
}

/// sherpa-onnx's `NormalizeLanguage`: trim, strip quotes and `<...>`, `_` → `-`, lower case.
fn normalize_tag(s: &str) -> String {
    let mut t = s.trim();
    for q in ['"', '\''] {
        if t.len() >= 2 && t.starts_with(q) && t.ends_with(q) {
            t = t[1..t.len() - 1].trim();
        }
    }
    if t.len() >= 2 && t.starts_with('<') && t.ends_with('>') {
        t = &t[1..t.len() - 1];
    }
    t.replace('_', "-").to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_names_round_trip() {
        assert_eq!(QWEN3_LANGUAGES.len(), 30);
        for (code, name) in QWEN3_LANGUAGES {
            assert_eq!(qwen3_language_name(code), Some(*name));
            assert_eq!(qwen3_language_code(name), Some(*code));
            assert_eq!(qwen3_language_code(&name.to_uppercase()), Some(*code));
        }
        assert_eq!(qwen3_language_name("tl"), Some("Filipino"));
        assert_eq!(qwen3_language_name("xx"), None);
        assert_eq!(qwen3_language_code(" Japanese "), Some("ja"));
        assert_eq!(qwen3_language_code("Klingon"), None);
    }

    #[test]
    fn prompt_dictionary_matches_sherpa_resolution() {
        let json = r#"{"auto": 101, "en-US": 0, "en-GB": 1, "ja-JP": 12, "de": 9, "de-DE": 9, "pt-BR": 5, "pt-PT": 6, "bad": -1, "x": "y"}"#;
        let d = PromptDictionary::parse(json, Some("101")).unwrap();
        // The first `en-*` tag in sorted key order (`en-GB`) becomes the `en` alias, as in sherpa.
        assert_eq!(d.prompt_id("en"), Some(1));
        assert_eq!(d.prompt_id("en-US"), Some(0));
        assert_eq!(d.prompt_id("en-GB"), Some(1));
        assert_eq!(d.prompt_id("EN_gb"), Some(1));
        assert_eq!(d.prompt_id("<ja-JP>"), Some(12));
        assert_eq!(d.prompt_id("ja"), Some(12));
        assert_eq!(d.prompt_id("pt"), Some(5));
        assert_eq!(d.prompt_id("auto"), None);
        assert_eq!(d.prompt_id(""), None);
        assert_eq!(d.prompt_id("ko"), None, "unknown languages fall back to auto");
        assert_eq!(d.prompt_id("bad"), None);
        assert!(d.supports("de") && !d.supports("xx"));
        assert_eq!(d.base_codes(), vec!["de", "en", "ja", "pt"]);
    }

    #[test]
    fn prompt_dictionary_without_languages_is_none() {
        assert!(PromptDictionary::parse("{}", None).is_none());
        assert!(PromptDictionary::parse(r#"{"auto": 101}"#, None).is_none());
        assert!(PromptDictionary::parse("not json", None).is_none());
        // `auto_prompt_id` metadata, and an explicit `auto` entry winning over it.
        let d = PromptDictionary::parse(r#"{"en": 3}"#, Some("7")).unwrap();
        assert_eq!(d.auto_id, 7);
        let d = PromptDictionary::parse(r#"{"en": 3, "auto": 9}"#, Some("7")).unwrap();
        assert_eq!(d.auto_id, 9);
        assert_eq!(d.prompt_id("en"), Some(3));
    }

    #[test]
    fn catalog_language_sets_are_normalized_codes() {
        for set in [
            NEMOTRON_LANGUAGES,
            NEMOTRON_TRANSCRIPTION_READY,
            PARAKEET_LANGUAGES,
            SENSE_VOICE_LANGUAGES,
            ENGLISH,
        ] {
            for code in set {
                assert_eq!(crate::services::speech::lang::normalize(code).as_deref(), Some(*code));
            }
        }
        for (code, _) in QWEN3_LANGUAGES {
            assert_eq!(crate::services::speech::lang::normalize(code).as_deref(), Some(*code));
        }
        assert!(NEMOTRON_TRANSCRIPTION_READY
            .iter()
            .all(|c| NEMOTRON_LANGUAGES.contains(c)));
        assert_eq!(
            NEMOTRON_LANGUAGES.len(),
            28,
            "19 transcription-ready + 13 broad-coverage locales, as base codes"
        );
    }
}
