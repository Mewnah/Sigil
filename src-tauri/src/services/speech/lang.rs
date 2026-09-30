//! Language codes for the speech stack (owned by P2-C; initial version by P2-S0).
//!
//! Sigil speaks lower-case base subtags everywhere (engines, the router, events, `TextEvent.lang`):
//! `en`, `ja`, `zh`, `yue`, `ko`, `ru`, `fil`, ... Engines that want another spelling (Qwen3-ASR's
//! English names, Nemotron's locale tags) convert at their edge.
//!
//! # Frozen signatures (P2-S0)
//! - [`AUTO`]: the "detect the language" value.
//! - [`normalize`]`(&str) -> Option<String>`: BCP-47 / locale tag → lower-case base subtag used by
//!   engines, the router and events (`en-US` → `en`, `zh-Hant-TW` → `zh`, `pt_BR` → `pt`,
//!   `yue` stays `yue`); `None` for `auto`, empty or malformed input.
//! - [`is_auto`]`(&str) -> bool`.
//!
//! Refinements (P2-C): Cantonese tags (`yue`, `zh-yue`, `zh-HK-yue`... any `zh-*` tag with a `yue`
//! subtag) become `yue`; deprecated ISO 639 codes become their current ones (`iw` → `he`, `in` →
//! `id`, `ji` → `yi`, `jw` → `jv`, `mo` → `ro`); three-letter codes of languages with a two-letter
//! code that Sigil's engines report become that code (`eng`, `jpn`, `zho`/`chi`/`cmn`, `kor`,
//! `rus`, `deu`/`ger`, `fra`/`fre`, `spa`, `por`, `ita`, `nld`/`dut`, `pol`, `ukr`, `tur`, `ara`,
//! `hin`, `vie`, `tha`, `ind`, `msa`/`may`, `tgl` → `fil`); `x-`/`i-` private-use and grandfathered
//! tags are rejected. Any other well-formed 2–3 letter base subtag passes through unchanged.

/// "Detect the language".
pub const AUTO: &str = "auto";

/// True for `auto` and empty input.
pub fn is_auto(code: &str) -> bool {
    let c = code.trim();
    c.is_empty() || c.eq_ignore_ascii_case(AUTO)
}

/// Deprecated or three-letter codes → the code Sigil uses.
const ALIASES: &[(&str, &str)] = &[
    ("iw", "he"),
    ("in", "id"),
    ("ji", "yi"),
    ("jw", "jv"),
    ("mo", "ro"),
    ("eng", "en"),
    ("jpn", "ja"),
    ("zho", "zh"),
    ("chi", "zh"),
    ("cmn", "zh"),
    ("kor", "ko"),
    ("rus", "ru"),
    ("deu", "de"),
    ("ger", "de"),
    ("fra", "fr"),
    ("fre", "fr"),
    ("spa", "es"),
    ("por", "pt"),
    ("ita", "it"),
    ("nld", "nl"),
    ("dut", "nl"),
    ("pol", "pl"),
    ("ukr", "uk"),
    ("tur", "tr"),
    ("ara", "ar"),
    ("hin", "hi"),
    ("vie", "vi"),
    ("tha", "th"),
    ("ind", "id"),
    ("msa", "ms"),
    ("may", "ms"),
    ("tgl", "fil"),
];

/// Normalize a language tag to its lower-case base subtag.
pub fn normalize(code: &str) -> Option<String> {
    if is_auto(code) {
        return None;
    }
    let tag = code.trim().replace('_', "-").to_ascii_lowercase();
    let mut parts = tag.split('-');
    let base = parts.next().unwrap_or("");
    if !(2..=3).contains(&base.len()) || !base.chars().all(|c| c.is_ascii_lowercase()) {
        return None;
    }
    // Cantonese is `yue`, or a `zh-*` tag with a `yue` subtag (`zh-yue`, `zh-Hant-yue`); every
    // other `zh-*` tag (including `zh-HK`) stays `zh`.
    if matches!(base, "zh" | "zho" | "chi") && parts.any(|p| p == "yue") {
        return Some("yue".into());
    }
    let base = ALIASES
        .iter()
        .find(|(from, _)| *from == base)
        .map_or(base, |(_, to)| to);
    Some(base.to_string())
}

/// Whether two tags name the same language ([`normalize`]d; `auto` matches nothing).
#[allow(dead_code)] // For drivers and engines comparing a detected language with the requested one.
pub fn same(a: &str, b: &str) -> bool {
    matches!((normalize(a), normalize(b)), (Some(x), Some(y)) if x == y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_common_tags() {
        assert_eq!(normalize("en-US").as_deref(), Some("en"));
        assert_eq!(normalize("zh-Hant-TW").as_deref(), Some("zh"));
        assert_eq!(normalize("pt_BR").as_deref(), Some("pt"));
        assert_eq!(normalize("yue").as_deref(), Some("yue"));
        assert_eq!(normalize("zh-yue").as_deref(), Some("yue"));
        assert_eq!(normalize(" JA ").as_deref(), Some("ja"));
        assert_eq!(normalize("auto"), None);
        assert_eq!(normalize(""), None);
        assert_eq!(normalize("e"), None);
        assert_eq!(normalize("12-x"), None);
        assert!(is_auto("Auto") && is_auto(" ") && !is_auto("en"));
    }

    #[test]
    fn maps_cantonese_aliases_and_rejects_private_use() {
        assert_eq!(normalize("zh-Hant-yue").as_deref(), Some("yue"));
        assert_eq!(normalize("zh-HK").as_deref(), Some("zh"));
        assert_eq!(normalize("yue-Hant-HK").as_deref(), Some("yue"));
        assert_eq!(normalize("iw").as_deref(), Some("he"));
        assert_eq!(normalize("in-ID").as_deref(), Some("id"));
        assert_eq!(normalize("jpn").as_deref(), Some("ja"));
        assert_eq!(normalize("cmn-Hans-CN").as_deref(), Some("zh"));
        assert_eq!(normalize("tgl").as_deref(), Some("fil"));
        assert_eq!(normalize("fil").as_deref(), Some("fil"));
        assert_eq!(normalize("nb-NO").as_deref(), Some("nb"));
        assert_eq!(normalize("x-klingon"), None);
        assert_eq!(normalize("i-klingon"), None);
        assert_eq!(normalize("english"), None);
        assert!(same("en-GB", "EN") && same("zh-yue", "yue") && !same("zh", "yue") && !same("auto", "auto"));
    }
}
