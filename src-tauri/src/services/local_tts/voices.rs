//! Voice tables of the catalog models (the order is the speaker id sherpa-onnx expects) and the
//! language / display-name rules shared by every local TTS family.
//!
//! - Kokoro v1.0 (54 voices) and v1.1-zh (103 voices): the `speaker_names` metadata of the official
//!   sherpa-onnx packages, in speaker-id order. The first letter of a Kokoro voice is its language
//!   (`a` US English, `b` British English, `e` Spanish, `f` French, `h` Hindi, `i` Italian,
//!   `j` Japanese, `p` Brazilian Portuguese, `z` Mandarin), the second its gender.
//! - Kitten 0.8 (8 voices): sherpa's speaker order (`expr-voice-2-m` = 0 ...) with the friendly
//!   names of KittenML's `config.json` `voice_aliases` (Jasper = `expr-voice-2-m`, ...), which
//!   the vendored Kitten and the legacy `neuralOnnx` settings use.
//!
//! A test with the real packages (`tests.rs`, ignored) checks these tables against the models'
//! own metadata.

use super::commands::LocalTtsVoice;

/// Kokoro v1.0 voices in speaker-id order (`speaker_names` of `kokoro-multi-lang-v1_0`).
pub const KOKORO_V1_0: &[&str] = &[
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah",
    "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "ef_dora", "em_alex",
    "ff_siwis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune",
    "jf_nezumi", "jf_tebukuro", "jm_kumo", "pf_dora", "pm_alex", "pm_santa", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao",
    "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang", "em_santa",
];

/// Kokoro v1.1-zh voices in speaker-id order (`speaker_names` of `kokoro-multi-lang-v1_1`).
pub const KOKORO_V1_1: &[&str] = &[
    "af_maple", "af_sol", "bf_vale", "zf_001", "zf_002", "zf_003", "zf_004", "zf_005", "zf_006", "zf_007", "zf_008",
    "zf_017", "zf_018", "zf_019", "zf_021", "zf_022", "zf_023", "zf_024", "zf_026", "zf_027", "zf_028", "zf_032",
    "zf_036", "zf_038", "zf_039", "zf_040", "zf_042", "zf_043", "zf_044", "zf_046", "zf_047", "zf_048", "zf_049",
    "zf_051", "zf_059", "zf_060", "zf_067", "zf_070", "zf_071", "zf_072", "zf_073", "zf_074", "zf_075", "zf_076",
    "zf_077", "zf_078", "zf_079", "zf_083", "zf_084", "zf_085", "zf_086", "zf_087", "zf_088", "zf_090", "zf_092",
    "zf_093", "zf_094", "zf_099", "zm_009", "zm_010", "zm_011", "zm_012", "zm_013", "zm_014", "zm_015", "zm_016",
    "zm_020", "zm_025", "zm_029", "zm_030", "zm_031", "zm_033", "zm_034", "zm_035", "zm_037", "zm_041", "zm_045",
    "zm_050", "zm_052", "zm_053", "zm_054", "zm_055", "zm_056", "zm_057", "zm_058", "zm_061", "zm_062", "zm_063",
    "zm_064", "zm_065", "zm_066", "zm_068", "zm_069", "zm_080", "zm_081", "zm_082", "zm_089", "zm_091", "zm_095",
    "zm_096", "zm_097", "zm_098", "zm_100",
];

/// Kitten 0.8 voices in sherpa speaker-id order: `(friendly name, sherpa speaker name)`.
pub const KITTEN_V0_8: &[(&str, &str)] = &[
    ("Jasper", "expr-voice-2-m"),
    ("Bella", "expr-voice-2-f"),
    ("Bruno", "expr-voice-3-m"),
    ("Luna", "expr-voice-3-f"),
    ("Hugo", "expr-voice-4-m"),
    ("Rosie", "expr-voice-4-f"),
    ("Leo", "expr-voice-5-m"),
    ("Kiki", "expr-voice-5-f"),
];

/// Default Kokoro voice per language (Kokoro's best-rated voice of each language; `af_heart` is
/// the upstream default).
const KOKORO_DEFAULTS: &[(&str, &str)] = &[
    ("en-US", "af_heart"),
    ("en-GB", "bf_emma"),
    ("es", "ef_dora"),
    ("fr", "ff_siwis"),
    ("hi", "hf_alpha"),
    ("it", "if_sara"),
    ("ja", "jf_alpha"),
    ("pt-BR", "pf_dora"),
    ("zh", "zf_xiaoxiao"),
];

/// The language of a Kokoro voice (from its first letter): a normalized BCP-47 tag.
pub fn kokoro_voice_language(voice: &str) -> Option<&'static str> {
    Some(match voice.as_bytes().first()? {
        b'a' => "en-US",
        b'b' => "en-GB",
        b'e' => "es",
        b'f' => "fr",
        b'h' => "hi",
        b'i' => "it",
        b'j' => "ja",
        b'p' => "pt-BR",
        b'z' => "zh",
        _ => return None,
    })
}

/// `female` / `male` from a Kokoro voice's second letter.
pub fn kokoro_voice_gender(voice: &str) -> Option<&'static str> {
    match voice.as_bytes().get(1)? {
        b'f' => Some("female"),
        b'm' => Some("male"),
        _ => None,
    }
}

/// Display name of a Kokoro voice: `af_heart` → `Heart`, `zf_001` → `Female 001`.
pub fn kokoro_display_name(voice: &str) -> String {
    let rest = voice.split_once('_').map(|(_, r)| r).unwrap_or(voice);
    if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
        let g = match kokoro_voice_gender(voice) {
            Some("male") => "Male",
            _ => "Female",
        };
        return format!("{g} {rest}");
    }
    capitalize(rest)
}

/// The default voice of a Kokoro table for `language` (a normalized tag; a base code such as
/// `en` or `pt` matches its first regional default), else `af_heart`, else the table's first
/// voice. Callers pass the model's own language when the request names none (see
/// [`kokoro_primary_language`]), so Kokoro v1.1-zh defaults to a Mandarin voice.
pub fn kokoro_default_voice<'a>(table: &[&'a str], language: Option<&str>) -> &'a str {
    if let Some(lang) = language {
        let lang = lang.to_ascii_lowercase();
        for (l, v) in KOKORO_DEFAULTS {
            let l_low = l.to_ascii_lowercase();
            let matches = l_low == lang || l_low.split('-').next() == Some(lang.as_str());
            if matches && table.contains(v) {
                return v;
            }
        }
        // Any voice of the language (Kokoro v1.1-zh has no zf_xiaoxiao).
        if let Some(v) = table.iter().find(|v| kokoro_voice_speaks(v, &lang)) {
            return v;
        }
    }
    if table.contains(&"af_heart") {
        "af_heart"
    } else {
        table.first().copied().unwrap_or("af_heart")
    }
}

/// Whether Kokoro voice `voice` speaks `language` (lower case; a base code matches every region).
fn kokoro_voice_speaks(voice: &str, language: &str) -> bool {
    kokoro_voice_language(voice).is_some_and(|l| l.eq_ignore_ascii_case(language) || base(l) == language)
}

/// Whether a Kokoro table has a voice of `language` (a normalized tag; a base code matches every
/// region).
pub fn kokoro_has_language(table: &[&str], language: &str) -> bool {
    let lang = language.to_ascii_lowercase();
    table.iter().any(|v| kokoro_voice_speaks(v, &lang))
}

/// The language most voices of a Kokoro table speak (a base code; the first seen wins a tie):
/// `en` for Kokoro v1.0, `zh` for v1.1-zh. `None` when no voice has a known language.
pub fn kokoro_primary_language(table: &[&str]) -> Option<String> {
    let mut counts: Vec<(String, usize)> = Vec::new();
    for l in table.iter().filter_map(|v| kokoro_voice_language(v)).map(base) {
        match counts.iter_mut().find(|(c, _)| *c == l) {
            Some((_, n)) => *n += 1,
            None => counts.push((l, 1)),
        }
    }
    // `max_by_key` keeps the last of equal maxima; reverse so the first seen wins.
    counts.into_iter().rev().max_by_key(|(_, n)| *n).map(|(l, _)| l)
}

/// Whether `voice` is a Kitten 0.8 voice name (KittenML's friendly name or the sherpa speaker
/// name, any case; not a plain number).
pub fn is_kitten_voice_name(voice: &str) -> bool {
    let v = voice.trim();
    KITTEN_V0_8
        .iter()
        .any(|(name, raw)| name.eq_ignore_ascii_case(v) || raw.eq_ignore_ascii_case(v))
}

/// The `LocalTtsVoice` rows of a Kokoro table.
pub fn kokoro_voices(table: &[&'static str]) -> Vec<LocalTtsVoice> {
    table
        .iter()
        .map(|v| LocalTtsVoice {
            id: (*v).to_string(),
            name: kokoro_display_name(v),
            language: kokoro_voice_language(v).map(str::to_string),
            gender: kokoro_voice_gender(v).map(str::to_string),
        })
        .collect()
}

/// The `LocalTtsVoice` rows of Kitten 0.8.
pub fn kitten_voices() -> Vec<LocalTtsVoice> {
    KITTEN_V0_8
        .iter()
        .map(|(name, raw)| LocalTtsVoice {
            id: (*name).to_string(),
            name: (*name).to_string(),
            language: Some("en-US".into()),
            gender: Some(if raw.ends_with("-f") { "female" } else { "male" }.into()),
        })
        .collect()
}

/// Speaker id of `voice` in a Kitten 0.8 package: the friendly name (any case), the sherpa speaker
/// name, or a plain number.
pub fn kitten_speaker_id(voice: &str) -> Option<u32> {
    let v = voice.trim();
    KITTEN_V0_8
        .iter()
        .position(|(name, raw)| name.eq_ignore_ascii_case(v) || raw.eq_ignore_ascii_case(v))
        .map(|i| i as u32)
        .or_else(|| v.parse::<u32>().ok().filter(|&n| (n as usize) < KITTEN_V0_8.len()))
}

/// Speaker id of `voice` in a named voice table: the name (any case) or a plain number.
pub fn speaker_id(table: &[&str], voice: &str) -> Option<u32> {
    let v = voice.trim();
    table
        .iter()
        .position(|n| n.eq_ignore_ascii_case(v))
        .map(|i| i as u32)
        .or_else(|| v.parse::<u32>().ok().filter(|&n| (n as usize) < table.len()))
}

/// `pt-BR` → `pt`.
pub fn base(tag: &str) -> String {
    tag.split(['-', '_'])
        .next()
        .unwrap_or(tag)
        .to_ascii_lowercase()
}

/// Normalize a language tag the way the TS side sends them: `pt_br` / `PT-br` → `pt-BR`, `EN` →
/// `en`, `zh-Hans` → `zh`, `cmn` → `zh`. `None` for blank or malformed input.
pub fn normalize_language(tag: &str) -> Option<String> {
    let t = tag.trim().replace('_', "-");
    if t.is_empty() || t.eq_ignore_ascii_case("auto") {
        return None;
    }
    let mut parts = t.split('-');
    let lang = parts.next()?.to_ascii_lowercase();
    if !(2..=3).contains(&lang.len()) || !lang.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    let lang = match lang.as_str() {
        "cmn" | "zho" | "chi" => "zh".to_string(),
        "jpn" => "ja".to_string(),
        "eng" => "en".to_string(),
        _ => lang,
    };
    if lang == "zh" {
        return Some(lang);
    }
    match parts.next() {
        Some(region) if region.len() == 2 && region.chars().all(|c| c.is_ascii_alphabetic()) => {
            Some(format!("{lang}-{}", region.to_ascii_uppercase()))
        }
        _ => Some(lang),
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
}
