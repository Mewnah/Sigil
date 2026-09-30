//! Canonical BCP-47 ↔ FLORES-200 codes for NLLB's language tokens (`en` → `eng_Latn`, `ja` → `jpn_Jpan`,
//! `zh` / `zh-Hans` → `zho_Hans`, `zh-Hant` / `zh-TW` → `zho_Hant`, ...), the English language names (reports and
//! prompts), and a script-based guess of the source language when the caller does not know it.
//!
//! [`LANGUAGES`] lists all 202 language tokens of NLLB-200 (the `tokenizer.json` added tokens `ace_Arab` ..
//! `zul_Latn`) with the canonical code Sigil uses for each: the ISO 639-1 code when there is one (`arb_Arab` → `ar`,
//! `pes_Arab` → `fa`, `zsm_Latn` → `ms`: NLLB's individual-language codes stand for the macrolanguage), else the
//! ISO 639-3 code (`ast`, `yue`), plus a script subtag for the second script of a language (`ks-Deva`, `ace-Arab`,
//! `taq-Tfng`). Must agree with the TS `translation/languages.ts` table (P4-C): the test
//! `agrees_with_the_ts_language_table` reads its mirror `translation/__tests__/languages.fixture.json`.
//!
//! Owned by P4-D.

/// One NLLB language: canonical code, FLORES-200 code, English name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloresLanguage {
    pub code: &'static str,
    pub flores: &'static str,
    pub name: &'static str,
}

const fn l(code: &'static str, flores: &'static str, name: &'static str) -> FloresLanguage {
    FloresLanguage { code, flores, name }
}

/// Every NLLB-200 language token (the order of `tokenizer.json`'s added tokens, roughly).
pub const LANGUAGES: &[FloresLanguage] = &[
    l("ace", "ace_Latn", "Acehnese"),
    l("ace-Arab", "ace_Arab", "Acehnese (Arabic script)"),
    l("acm", "acm_Arab", "Mesopotamian Arabic"),
    l("acq", "acq_Arab", "Ta'izzi-Adeni Arabic"),
    l("aeb", "aeb_Arab", "Tunisian Arabic"),
    l("af", "afr_Latn", "Afrikaans"),
    l("ajp", "ajp_Arab", "South Levantine Arabic"),
    l("ak", "aka_Latn", "Akan"),
    l("am", "amh_Ethi", "Amharic"),
    l("apc", "apc_Arab", "North Levantine Arabic"),
    l("ar", "arb_Arab", "Arabic"),
    l("ars", "ars_Arab", "Najdi Arabic"),
    l("ary", "ary_Arab", "Moroccan Arabic"),
    l("arz", "arz_Arab", "Egyptian Arabic"),
    l("as", "asm_Beng", "Assamese"),
    l("ast", "ast_Latn", "Asturian"),
    l("awa", "awa_Deva", "Awadhi"),
    l("ay", "ayr_Latn", "Aymara"),
    l("azb", "azb_Arab", "South Azerbaijani"),
    l("az", "azj_Latn", "Azerbaijani"),
    l("ba", "bak_Cyrl", "Bashkir"),
    l("bm", "bam_Latn", "Bambara"),
    l("ban", "ban_Latn", "Balinese"),
    l("be", "bel_Cyrl", "Belarusian"),
    l("bem", "bem_Latn", "Bemba"),
    l("bn", "ben_Beng", "Bengali"),
    l("bho", "bho_Deva", "Bhojpuri"),
    l("bjn-Arab", "bjn_Arab", "Banjar (Arabic script)"),
    l("bjn", "bjn_Latn", "Banjar"),
    l("bo", "bod_Tibt", "Tibetan"),
    l("bs", "bos_Latn", "Bosnian"),
    l("bug", "bug_Latn", "Buginese"),
    l("bg", "bul_Cyrl", "Bulgarian"),
    l("ca", "cat_Latn", "Catalan"),
    l("ceb", "ceb_Latn", "Cebuano"),
    l("cs", "ces_Latn", "Czech"),
    l("cjk", "cjk_Latn", "Chokwe"),
    l("ckb", "ckb_Arab", "Central Kurdish"),
    l("crh", "crh_Latn", "Crimean Tatar"),
    l("cy", "cym_Latn", "Welsh"),
    l("da", "dan_Latn", "Danish"),
    l("de", "deu_Latn", "German"),
    l("dik", "dik_Latn", "Southwestern Dinka"),
    l("dyu", "dyu_Latn", "Dyula"),
    l("dz", "dzo_Tibt", "Dzongkha"),
    l("el", "ell_Grek", "Greek"),
    l("en", "eng_Latn", "English"),
    l("eo", "epo_Latn", "Esperanto"),
    l("et", "est_Latn", "Estonian"),
    l("eu", "eus_Latn", "Basque"),
    l("ee", "ewe_Latn", "Ewe"),
    l("fo", "fao_Latn", "Faroese"),
    l("fa", "pes_Arab", "Persian"),
    l("fj", "fij_Latn", "Fijian"),
    l("fi", "fin_Latn", "Finnish"),
    l("fon", "fon_Latn", "Fon"),
    l("fr", "fra_Latn", "French"),
    l("fur", "fur_Latn", "Friulian"),
    l("ff", "fuv_Latn", "Nigerian Fulfulde"),
    l("gd", "gla_Latn", "Scottish Gaelic"),
    l("ga", "gle_Latn", "Irish"),
    l("gl", "glg_Latn", "Galician"),
    l("gn", "grn_Latn", "Guarani"),
    l("gu", "guj_Gujr", "Gujarati"),
    l("ht", "hat_Latn", "Haitian Creole"),
    l("ha", "hau_Latn", "Hausa"),
    l("he", "heb_Hebr", "Hebrew"),
    l("hi", "hin_Deva", "Hindi"),
    l("hne", "hne_Deva", "Chhattisgarhi"),
    l("hr", "hrv_Latn", "Croatian"),
    l("hu", "hun_Latn", "Hungarian"),
    l("hy", "hye_Armn", "Armenian"),
    l("ig", "ibo_Latn", "Igbo"),
    l("ilo", "ilo_Latn", "Ilocano"),
    l("id", "ind_Latn", "Indonesian"),
    l("is", "isl_Latn", "Icelandic"),
    l("it", "ita_Latn", "Italian"),
    l("jv", "jav_Latn", "Javanese"),
    l("ja", "jpn_Jpan", "Japanese"),
    l("kab", "kab_Latn", "Kabyle"),
    l("kac", "kac_Latn", "Jingpho"),
    l("kam", "kam_Latn", "Kamba"),
    l("kn", "kan_Knda", "Kannada"),
    l("ks", "kas_Arab", "Kashmiri"),
    l("ks-Deva", "kas_Deva", "Kashmiri (Devanagari)"),
    l("ka", "kat_Geor", "Georgian"),
    l("kr-Arab", "knc_Arab", "Central Kanuri (Arabic script)"),
    l("kr", "knc_Latn", "Central Kanuri"),
    l("kk", "kaz_Cyrl", "Kazakh"),
    l("kbp", "kbp_Latn", "Kabiyè"),
    l("kea", "kea_Latn", "Kabuverdianu"),
    l("km", "khm_Khmr", "Khmer"),
    l("ki", "kik_Latn", "Kikuyu"),
    l("rw", "kin_Latn", "Kinyarwanda"),
    l("ky", "kir_Cyrl", "Kyrgyz"),
    l("kmb", "kmb_Latn", "Kimbundu"),
    l("kg", "kon_Latn", "Kikongo"),
    l("ko", "kor_Hang", "Korean"),
    l("ku", "kmr_Latn", "Northern Kurdish"),
    l("lo", "lao_Laoo", "Lao"),
    l("lv", "lvs_Latn", "Latvian"),
    l("lij", "lij_Latn", "Ligurian"),
    l("li", "lim_Latn", "Limburgish"),
    l("ln", "lin_Latn", "Lingala"),
    l("lt", "lit_Latn", "Lithuanian"),
    l("lmo", "lmo_Latn", "Lombard"),
    l("ltg", "ltg_Latn", "Latgalian"),
    l("lb", "ltz_Latn", "Luxembourgish"),
    l("lua", "lua_Latn", "Luba-Kasai"),
    l("lg", "lug_Latn", "Ganda"),
    l("luo", "luo_Latn", "Luo"),
    l("lus", "lus_Latn", "Mizo"),
    l("mag", "mag_Deva", "Magahi"),
    l("mai", "mai_Deva", "Maithili"),
    l("ml", "mal_Mlym", "Malayalam"),
    l("mr", "mar_Deva", "Marathi"),
    l("min", "min_Latn", "Minangkabau"),
    l("mk", "mkd_Cyrl", "Macedonian"),
    l("mg", "plt_Latn", "Malagasy"),
    l("mt", "mlt_Latn", "Maltese"),
    l("mni", "mni_Beng", "Meitei (Bengali script)"),
    l("mn", "khk_Cyrl", "Mongolian"),
    l("mos", "mos_Latn", "Mossi"),
    l("mi", "mri_Latn", "Maori"),
    l("ms", "zsm_Latn", "Malay"),
    l("my", "mya_Mymr", "Burmese"),
    l("nl", "nld_Latn", "Dutch"),
    l("nn", "nno_Latn", "Norwegian Nynorsk"),
    l("nb", "nob_Latn", "Norwegian Bokmål"),
    l("ne", "npi_Deva", "Nepali"),
    l("nso", "nso_Latn", "Northern Sotho"),
    l("nus", "nus_Latn", "Nuer"),
    l("ny", "nya_Latn", "Nyanja"),
    l("oc", "oci_Latn", "Occitan"),
    l("om", "gaz_Latn", "Oromo"),
    l("or", "ory_Orya", "Odia"),
    l("pag", "pag_Latn", "Pangasinan"),
    l("pa", "pan_Guru", "Punjabi"),
    l("pap", "pap_Latn", "Papiamento"),
    l("pl", "pol_Latn", "Polish"),
    l("pt", "por_Latn", "Portuguese"),
    l("prs", "prs_Arab", "Dari"),
    l("ps", "pbt_Arab", "Pashto"),
    l("qu", "quy_Latn", "Quechua"),
    l("ro", "ron_Latn", "Romanian"),
    l("rn", "run_Latn", "Rundi"),
    l("ru", "rus_Cyrl", "Russian"),
    l("sg", "sag_Latn", "Sango"),
    l("sa", "san_Deva", "Sanskrit"),
    l("sat", "sat_Beng", "Santali"),
    l("scn", "scn_Latn", "Sicilian"),
    l("shn", "shn_Mymr", "Shan"),
    l("si", "sin_Sinh", "Sinhala"),
    l("sk", "slk_Latn", "Slovak"),
    l("sl", "slv_Latn", "Slovenian"),
    l("sm", "smo_Latn", "Samoan"),
    l("sn", "sna_Latn", "Shona"),
    l("sd", "snd_Arab", "Sindhi"),
    l("so", "som_Latn", "Somali"),
    l("st", "sot_Latn", "Southern Sotho"),
    l("es", "spa_Latn", "Spanish"),
    l("sq", "als_Latn", "Albanian"),
    l("sc", "srd_Latn", "Sardinian"),
    l("sr", "srp_Cyrl", "Serbian"),
    l("ss", "ssw_Latn", "Swati"),
    l("su", "sun_Latn", "Sundanese"),
    l("sv", "swe_Latn", "Swedish"),
    l("sw", "swh_Latn", "Swahili"),
    l("szl", "szl_Latn", "Silesian"),
    l("ta", "tam_Taml", "Tamil"),
    l("tt", "tat_Cyrl", "Tatar"),
    l("te", "tel_Telu", "Telugu"),
    l("tg", "tgk_Cyrl", "Tajik"),
    l("tl", "tgl_Latn", "Tagalog"),
    l("th", "tha_Thai", "Thai"),
    l("ti", "tir_Ethi", "Tigrinya"),
    l("taq", "taq_Latn", "Tamasheq"),
    l("taq-Tfng", "taq_Tfng", "Tamasheq (Tifinagh)"),
    l("tpi", "tpi_Latn", "Tok Pisin"),
    l("tn", "tsn_Latn", "Tswana"),
    l("ts", "tso_Latn", "Tsonga"),
    l("tk", "tuk_Latn", "Turkmen"),
    l("tum", "tum_Latn", "Tumbuka"),
    l("tr", "tur_Latn", "Turkish"),
    l("tw", "twi_Latn", "Twi"),
    l("tzm", "tzm_Tfng", "Central Atlas Tamazight"),
    l("ug", "uig_Arab", "Uyghur"),
    l("uk", "ukr_Cyrl", "Ukrainian"),
    l("umb", "umb_Latn", "Umbundu"),
    l("ur", "urd_Arab", "Urdu"),
    l("uz", "uzn_Latn", "Uzbek"),
    l("vec", "vec_Latn", "Venetian"),
    l("vi", "vie_Latn", "Vietnamese"),
    l("war", "war_Latn", "Waray"),
    l("wo", "wol_Latn", "Wolof"),
    l("xh", "xho_Latn", "Xhosa"),
    l("yi", "ydd_Hebr", "Yiddish"),
    l("yo", "yor_Latn", "Yoruba"),
    l("yue", "yue_Hant", "Cantonese"),
    l("zh", "zho_Hans", "Chinese (Simplified)"),
    l("zh-Hant", "zho_Hant", "Chinese (Traditional)"),
    l("zu", "zul_Latn", "Zulu"),
];

/// Codes that are neither a canonical code of [`LANGUAGES`] nor its base subtag: deprecated ISO 639-1 codes,
/// macrolanguage and individual-language spellings, and Chinese regions.
const ALIASES: &[(&str, &str)] = &[
    ("zh-hans", "zho_Hans"),
    ("zh-cn", "zho_Hans"),
    ("zh-sg", "zho_Hans"),
    ("zh-my", "zho_Hans"),
    ("zh-hant", "zho_Hant"),
    ("zh-tw", "zho_Hant"),
    ("zh-hk", "zho_Hant"),
    ("zh-mo", "zho_Hant"),
    ("cmn", "zho_Hans"),
    ("zho", "zho_Hans"),
    ("iw", "heb_Hebr"),
    ("ji", "ydd_Hebr"),
    ("jw", "jav_Latn"),
    ("in", "ind_Latn"),
    ("no", "nob_Latn"),
    ("fil", "tgl_Latn"),
    ("fa-af", "prs_Arab"),
    ("mo", "ron_Latn"),
    ("sh", "hrv_Latn"),
    ("ckb-arab", "ckb_Arab"),
    ("ku-arab", "ckb_Arab"),
    ("pa-guru", "pan_Guru"),
    ("sr-cyrl", "srp_Cyrl"),
    ("mni-beng", "mni_Beng"),
    ("sat-beng", "sat_Beng"),
];

fn find_code(code: &str) -> Option<&'static FloresLanguage> {
    LANGUAGES.iter().find(|l| l.code.eq_ignore_ascii_case(code))
}

/// FLORES-200 code for a canonical BCP-47 language, or `None` when NLLB has no such language (or the code is empty,
/// `auto` or not a language). Accepts any BCP-47-ish spelling (`EN_us`, `zh-TW`, `pt-BR`), ISO 639-3 codes
/// (`eng`, `jpn`) and FLORES codes themselves (`jpn_Jpan`).
pub fn to_flores(code: &str) -> Option<&'static str> {
    let raw = code.trim();
    if raw.is_empty() {
        return None;
    }
    // A FLORES code as-is (`jpn_Jpan`), matched case-insensitively.
    if raw.len() == 8 && raw.as_bytes()[3] == b'_' {
        return LANGUAGES.iter().find(|l| l.flores.eq_ignore_ascii_case(raw)).map(|l| l.flores);
    }
    let norm = raw.replace('_', "-").to_ascii_lowercase();
    if norm == "auto" || norm == "und" {
        return None;
    }
    if let Some(l) = find_code(&norm) {
        return Some(l.flores);
    }
    // Chinese: any Traditional script or region wins over the base, whatever else is tagged (`zh-Hant-TW`).
    let subtags: Vec<&str> = norm.split('-').filter(|s| !s.is_empty()).collect();
    let base = *subtags.first()?;
    if base == "zh" || base == "cmn" || base == "zho" {
        let traditional = subtags.iter().skip(1).any(|s| matches!(*s, "hant" | "tw" | "hk" | "mo"));
        return Some(if traditional { "zho_Hant" } else { "zho_Hans" });
    }
    if let Some((_, flores)) = ALIASES.iter().find(|(alias, _)| *alias == norm) {
        return Some(flores);
    }
    // Base plus script (`ks-Deva-IN` → `ks-Deva`).
    if subtags.len() >= 2 && subtags[1].len() == 4 {
        if let Some(l) = find_code(&format!("{base}-{}", subtags[1])) {
            return Some(l.flores);
        }
    }
    if let Some((_, flores)) = ALIASES.iter().find(|(alias, _)| *alias == base) {
        return Some(flores);
    }
    if let Some(l) = find_code(base) {
        return Some(l.flores);
    }
    // ISO 639-3 (`eng`, `jpn`, `swh`): the first three letters of the FLORES code.
    if base.len() == 3 {
        return LANGUAGES
            .iter()
            .find(|l| l.flores[..3].eq_ignore_ascii_case(base))
            .map(|l| l.flores);
    }
    None
}

/// The [`FloresLanguage`] row of a FLORES code.
pub fn by_flores(flores: &str) -> Option<&'static FloresLanguage> {
    LANGUAGES.iter().find(|l| l.flores == flores)
}

/// English name of a canonical code or FLORES code (`ja` → `Japanese`); `None` for unknown codes.
pub fn language_name(code: &str) -> Option<&'static str> {
    to_flores(code).and_then(by_flores).map(|l| l.name)
}

/// Whether text in this language is written without spaces between words (translations of several sentences are
/// joined without a space).
pub fn joins_without_spaces(flores: &str) -> bool {
    matches!(
        flores,
        "jpn_Jpan" | "zho_Hans" | "zho_Hant" | "yue_Hant" | "tha_Thai" | "lao_Laoo" | "khm_Khmr" | "mya_Mymr"
            | "shn_Mymr" | "bod_Tibt" | "dzo_Tibt"
    )
}

/// Writing systems the source guess can tell apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Script {
    Kana,
    Hangul,
    Han,
    Thai,
    Lao,
    Khmer,
    Myanmar,
    Georgian,
    Armenian,
    Hebrew,
    Greek,
    Arabic,
    Cyrillic,
    Devanagari,
    Bengali,
    Gurmukhi,
    Gujarati,
    Oriya,
    Tamil,
    Telugu,
    Kannada,
    Malayalam,
    Sinhala,
    Tibetan,
    Ethiopic,
    Latin,
}

fn script_of(c: char) -> Option<Script> {
    let u = c as u32;
    Some(match u {
        0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F => Script::Kana,
        0x1100..=0x11FF | 0x3130..=0x318F | 0xAC00..=0xD7AF => Script::Hangul,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2FFFF => Script::Han,
        0x0E00..=0x0E7F => Script::Thai,
        0x0E80..=0x0EFF => Script::Lao,
        0x1780..=0x17FF => Script::Khmer,
        0x1000..=0x109F => Script::Myanmar,
        0x10A0..=0x10FF => Script::Georgian,
        0x0530..=0x058F => Script::Armenian,
        0x0590..=0x05FF => Script::Hebrew,
        0x0370..=0x03FF | 0x1F00..=0x1FFF => Script::Greek,
        0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => Script::Arabic,
        0x0400..=0x052F => Script::Cyrillic,
        0x0900..=0x097F => Script::Devanagari,
        0x0980..=0x09FF => Script::Bengali,
        0x0A00..=0x0A7F => Script::Gurmukhi,
        0x0A80..=0x0AFF => Script::Gujarati,
        0x0B00..=0x0B7F => Script::Oriya,
        0x0B80..=0x0BFF => Script::Tamil,
        0x0C00..=0x0C7F => Script::Telugu,
        0x0C80..=0x0CFF => Script::Kannada,
        0x0D00..=0x0D7F => Script::Malayalam,
        0x0D80..=0x0DFF => Script::Sinhala,
        0x0F00..=0x0FFF => Script::Tibetan,
        0x1200..=0x139F => Script::Ethiopic,
        _ if c.is_alphabetic() && (u < 0x0250 || (0x1E00..=0x1EFF).contains(&u) || (0xFF21..=0xFF5A).contains(&u)) => {
            Script::Latin
        }
        _ => return None,
    })
}

/// Best-effort source language for text whose language the caller does not know: the FLORES code of the most common
/// writing system's main language (any kana → Japanese, Hangul → Korean, Han only → Simplified Chinese, Cyrillic →
/// Russian or Ukrainian, Arabic script → Arabic, Persian or Urdu, Latin → English, ...). NLLB's source token mostly
/// steers the encoder's language context, so
/// a same-script guess (Spanish tagged as English) still translates well. `None` when the text has no letters.
pub fn guess_flores(text: &str) -> Option<&'static str> {
    let mut counts: Vec<(Script, usize)> = Vec::new();
    for c in text.chars() {
        let Some(s) = script_of(c) else { continue };
        match counts.iter_mut().find(|(k, _)| *k == s) {
            Some((_, n)) => *n += 1,
            None => counts.push((s, 1)),
        }
    }
    if counts.is_empty() {
        return None;
    }
    // Japanese mixes kana with Han; any kana decides it.
    if counts.iter().any(|(s, _)| *s == Script::Kana) {
        return Some("jpn_Jpan");
    }
    if counts.iter().any(|(s, _)| *s == Script::Hangul) {
        return Some("kor_Hang");
    }
    let (script, _) = counts.iter().copied().max_by_key(|(_, n)| *n)?;
    Some(match script {
        Script::Kana => "jpn_Jpan",
        Script::Hangul => "kor_Hang",
        Script::Han => "zho_Hans",
        Script::Thai => "tha_Thai",
        Script::Lao => "lao_Laoo",
        Script::Khmer => "khm_Khmr",
        Script::Myanmar => "mya_Mymr",
        Script::Georgian => "kat_Geor",
        Script::Armenian => "hye_Armn",
        Script::Hebrew => "heb_Hebr",
        Script::Greek => "ell_Grek",
        // Letters only Urdu / Persian / Ukrainian use (the TS `guessLanguage` applies the same rules).
        Script::Arabic if text.chars().any(|c| "ےٹڈڑں".contains(c)) => "urd_Arab",
        Script::Arabic if text.chars().any(|c| "پچژگی".contains(c)) && !text.chars().any(|c| "يى".contains(c)) => "pes_Arab",
        Script::Arabic => "arb_Arab",
        Script::Cyrillic if text.chars().any(|c| "іїєґІЇЄҐ".contains(c)) => "ukr_Cyrl",
        Script::Cyrillic => "rus_Cyrl",
        Script::Devanagari => "hin_Deva",
        Script::Bengali => "ben_Beng",
        Script::Gurmukhi => "pan_Guru",
        Script::Gujarati => "guj_Gujr",
        Script::Oriya => "ory_Orya",
        Script::Tamil => "tam_Taml",
        Script::Telugu => "tel_Telu",
        Script::Kannada => "kan_Knda",
        Script::Malayalam => "mal_Mlym",
        Script::Sinhala => "sin_Sinh",
        Script::Tibetan => "bod_Tibt",
        Script::Ethiopic => "amh_Ethi",
        Script::Latin => "eng_Latn",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn the_table_has_every_nllb_language_once() {
        assert_eq!(LANGUAGES.len(), 202, "NLLB-200 has 202 language tokens");
        let flores: HashSet<&str> = LANGUAGES.iter().map(|l| l.flores).collect();
        assert_eq!(flores.len(), LANGUAGES.len(), "duplicate FLORES code");
        let codes: HashSet<String> = LANGUAGES.iter().map(|l| l.code.to_ascii_lowercase()).collect();
        assert_eq!(codes.len(), LANGUAGES.len(), "duplicate canonical code");
        for l in LANGUAGES {
            assert!(
                l.flores.len() == 8 && l.flores.as_bytes()[3] == b'_',
                "malformed FLORES code {}",
                l.flores
            );
            assert!(!l.name.is_empty());
            // Canonical codes round-trip.
            assert_eq!(to_flores(l.code), Some(l.flores), "{}", l.code);
            assert_eq!(to_flores(l.flores), Some(l.flores), "{}", l.flores);
        }
    }

    #[test]
    fn common_codes_map_like_the_ts_table() {
        for (code, flores) in [
            ("en", "eng_Latn"),
            ("EN_us", "eng_Latn"),
            ("en-GB", "eng_Latn"),
            ("ja", "jpn_Jpan"),
            ("ja-JP", "jpn_Jpan"),
            ("ko", "kor_Hang"),
            ("zh", "zho_Hans"),
            ("zh-Hans", "zho_Hans"),
            ("zh-CN", "zho_Hans"),
            ("zh-Hant", "zho_Hant"),
            ("zh-TW", "zho_Hant"),
            ("zh-HK", "zho_Hant"),
            ("zh-Hant-TW", "zho_Hant"),
            ("yue", "yue_Hant"),
            ("pt", "por_Latn"),
            ("pt-BR", "por_Latn"),
            ("pt-PT", "por_Latn"),
            ("ar", "arb_Arab"),
            ("fa", "pes_Arab"),
            ("fa-AF", "prs_Arab"),
            ("ms", "zsm_Latn"),
            ("no", "nob_Latn"),
            ("nb", "nob_Latn"),
            ("fil", "tgl_Latn"),
            ("tl", "tgl_Latn"),
            ("iw", "heb_Hebr"),
            ("he", "heb_Hebr"),
            ("sw", "swh_Latn"),
            ("ks-Deva", "kas_Deva"),
            ("ks-Deva-IN", "kas_Deva"),
            ("ks", "kas_Arab"),
            ("sr-Latn", "srp_Cyrl"),
            ("eng", "eng_Latn"),
            ("jpn", "jpn_Jpan"),
            ("swh", "swh_Latn"),
            ("JPN_JPAN", "jpn_Jpan"),
        ] {
            assert_eq!(to_flores(code), Some(flores), "{code}");
        }
        for code in ["", "  ", "auto", "und", "xx", "klingon", "tlh", "abc_Defg"] {
            assert_eq!(to_flores(code), None, "{code:?}");
        }
    }

    #[test]
    fn names_and_joining() {
        assert_eq!(language_name("ja"), Some("Japanese"));
        assert_eq!(language_name("zh-TW"), Some("Chinese (Traditional)"));
        assert_eq!(language_name("xx"), None);
        assert!(joins_without_spaces("jpn_Jpan"));
        assert!(joins_without_spaces("zho_Hant"));
        assert!(!joins_without_spaces("eng_Latn"));
        assert!(!joins_without_spaces("kor_Hang"), "Korean uses spaces");
    }

    #[test]
    fn source_guess_by_script() {
        for (text, flores) in [
            ("Hello there", "eng_Latn"),
            ("¿Dónde está?", "eng_Latn"),
            ("今日はいい天気ですね", "jpn_Jpan"),
            ("カタカナだけ", "jpn_Jpan"),
            ("我们一起玩吧", "zho_Hans"),
            ("안녕하세요", "kor_Hang"),
            ("Привет", "rus_Cyrl"),
            ("Привіт, як справи?", "ukr_Cyrl"),
            ("مرحبا", "arb_Arab"),
            ("سلام، حال شما چطور است؟", "pes_Arab"),
            ("آپ کیسے ہیں؟", "urd_Arab"),
            ("שלום", "heb_Hebr"),
            ("สวัสดี", "tha_Thai"),
            ("नमस्ते", "hin_Deva"),
            ("Γεια σου", "ell_Grek"),
            ("OK 今日はok", "jpn_Jpan"),
            ("ＦＵＬＬ", "eng_Latn"),
        ] {
            assert_eq!(guess_flores(text), Some(flores), "{text}");
        }
        assert_eq!(guess_flores("123 !!! 😀"), None);
        assert_eq!(guess_flores(""), None);
    }

    /// Known differences from the TS table (none since P4-I0 aligned Santali: FLORES-200 writes it `sat_Olck`,
    /// but NLLB-200's tokenizer has `sat_Beng`, and both tables now use the token that exists).
    const TS_KNOWN_DIFFERENCES: &[(&str, &str)] = &[];

    #[test]
    fn agrees_with_the_ts_language_table() {
        // P4-C's fixture mirrors `src/core/services/translation/languages.ts` (checked by its vitest).
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../src/core/services/translation/__tests__/languages.fixture.json");
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("skipped: {} not found", path.display());
            return;
        };
        let fixture: serde_json::Value = serde_json::from_str(&text).expect("languages.fixture.json parses");
        let rows = fixture["languages"].as_array().expect("languages array");
        assert!(rows.len() >= 20, "languages.fixture.json has too few rows");
        for row in rows {
            let code = row["code"].as_str().expect("code");
            match row["flores"].as_str() {
                // The TS side routes this language to NLLB: Rust must know it, as the same token.
                Some(f) => {
                    let rust = to_flores(code);
                    assert!(rust.is_some(), "TS {code} -> {f}, but Rust cannot map {code}");
                    if rust != Some(f) {
                        assert!(
                            TS_KNOWN_DIFFERENCES.contains(&(code, f)),
                            "TS {code} -> {f}, Rust -> {rust:?}"
                        );
                    }
                }
                // A language TS believes NLLB lacks only costs a missed route; report it.
                None => {
                    if let Some(f) = to_flores(code) {
                        eprintln!("note: languages.ts has flores: null for {code}, NLLB has {f}");
                    }
                }
            }
        }
    }
}
