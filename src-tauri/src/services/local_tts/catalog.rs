//! Sigil's local TTS catalog models (`model-catalog/tts-local.json`): catalog id → family, title,
//! languages, voices and output rate. The fragment says how to download an id; this table says
//! how to run it. `tests.rs` keeps the two in step (same ids, titles and the sherpa runtime
//! requirement).

use once_cell::sync::Lazy;

use super::commands::LocalTtsVoice;
use super::voices;

/// Model family (wire: `kokoro`, `kitten`, `piper`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    Kokoro,
    Kitten,
    Piper,
}

impl Family {
    pub fn as_str(self) -> &'static str {
        match self {
            Family::Kokoro => "kokoro",
            Family::Kitten => "kitten",
            Family::Piper => "piper",
        }
    }
}

/// The voices of a catalog model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoiceSet {
    /// A Kokoro speaker table ([`voices::KOKORO_V1_0`] / [`voices::KOKORO_V1_1`]).
    Kokoro(&'static [&'static str]),
    /// Kitten 0.8 ([`voices::KITTEN_V0_8`]).
    Kitten08,
    /// A Piper voice: speaker names come from the installed `.onnx.json`; until then one voice
    /// named after the catalog title.
    Piper,
}

/// One catalog model.
#[derive(Clone, Debug, PartialEq)]
pub struct CatalogModel {
    pub id: String,
    pub family: Family,
    /// Same as the fragment's `title`.
    pub title: String,
    /// Normalized language tags (`en`, `pt-BR`, `ru`, ...).
    pub languages: Vec<String>,
    pub voices: VoiceSet,
    pub sample_rate: u32,
    /// Piper: the model file stem (`ru_RU-irina-medium`).
    pub piper_voice: Option<&'static str>,
    /// Piper: the voice's name (`Irina`), used as its single voice before install.
    pub piper_name: Option<&'static str>,
}

impl CatalogModel {
    /// Voices before the model is installed (Piper speakers are known only after install).
    pub fn static_voices(&self) -> Vec<LocalTtsVoice> {
        match self.voices {
            VoiceSet::Kokoro(table) => voices::kokoro_voices(table),
            VoiceSet::Kitten08 => voices::kitten_voices(),
            VoiceSet::Piper => vec![LocalTtsVoice {
                id: "0".into(),
                name: self.piper_name.unwrap_or("Default").to_string(),
                language: self.languages.first().cloned(),
                gender: None,
            }],
        }
    }

    /// The voice used when a request names none: Kokoro's default for `language`, or for the
    /// model's own (first) language when `language` is unset or has no voice in the model (Kokoro
    /// v1.1-zh: a Mandarin voice, not its first, English, one).
    pub fn default_voice(&self, language: Option<&str>) -> String {
        match self.voices {
            VoiceSet::Kokoro(table) => {
                let own = self.languages.first().map(String::as_str);
                let language = language.filter(|l| voices::kokoro_has_language(table, l)).or(own);
                voices::kokoro_default_voice(table, language).to_string()
            }
            VoiceSet::Kitten08 => "Jasper".into(),
            VoiceSet::Piper => "0".into(),
        }
    }
}

/// The model used when a request names none: Kokoro v1.0 (the zero-setup voice).
pub const DEFAULT_MODEL: &str = "tts/kokoro-v1.0";

/// Piper voices: `(voice file stem, language, voice name)`. The catalog id is
/// `tts/piper-<stem, lower case>` and the title `<language>: <name> (Piper)`.
const PIPER: &[(&str, &str, &str)] = &[
    ("ru_RU-irina-medium", "Russian", "Irina"),
    ("ru_RU-dmitri-medium", "Russian", "Dmitri"),
    ("ar_JO-kareem-medium", "Arabic", "Kareem"),
    ("de_DE-thorsten-high", "German", "Thorsten"),
    ("uk_UA-ukrainian_tts-medium", "Ukrainian", "Lada, Mykyta, Tetiana"),
    ("pl_PL-gosia-medium", "Polish", "Gosia"),
    ("nl_NL-pim-medium", "Dutch", "Pim"),
    ("tr_TR-dfki-medium", "Turkish", "DFKI"),
    ("vi_VN-vais1000-medium", "Vietnamese", "VAIS1000"),
    ("id_ID-news_tts-medium", "Indonesian", "News"),
    ("ur_PK-fasih-medium", "Urdu", "Fasih"),
    ("pt_PT-tugao-medium", "European Portuguese", "Tugão"),
    ("sv_SE-nst-medium", "Swedish", "NST"),
    ("cs_CZ-jirka-medium", "Czech", "Jirka"),
    ("fi_FI-harri-medium", "Finnish", "Harri"),
    ("el_GR-rapunzelina-low", "Greek", "Rapunzelina"),
    ("hu_HU-anna-medium", "Hungarian", "Anna"),
    ("ro_RO-mihai-medium", "Romanian", "Mihai"),
    ("fa_IR-amir-medium", "Persian", "Amir"),
    ("ka_GE-natia-medium", "Georgian", "Natia"),
    ("kk_KZ-issai-high", "Kazakh", "ISSAI"),
    ("ne_NP-google-medium", "Nepali", "Google"),
    ("sr_RS-serbski_institut-medium", "Serbian", "Serbski Institut"),
    ("sw_CD-lanfrica-medium", "Swahili", "Lanfrica"),
    ("no_NO-talesyntese-medium", "Norwegian", "Talesyntese"),
    ("da_DK-talesyntese-medium", "Danish", "Talesyntese"),
    ("ca_ES-upc_ona-medium", "Catalan", "Ona"),
    ("is_IS-ugla-medium", "Icelandic", "Ugla"),
    ("sk_SK-lili-medium", "Slovak", "Lili"),
    ("sl_SI-artur-medium", "Slovenian", "Artur"),
    ("lv_LV-aivars-medium", "Latvian", "Aivars"),
    ("cy_GB-gwryw_gogleddol-medium", "Welsh", "Gwryw Gogleddol"),
    ("lb_LU-marylux-medium", "Luxembourgish", "Marylux"),
    ("ml_IN-meera-medium", "Malayalam", "Meera"),
    ("sq_AL-edon-medium", "Albanian", "Edon"),
    ("eu_ES-maider-medium", "Basque", "Maider"),
];

/// Piper's output rate by quality: `x_low` / `low` voices are 16 kHz, `medium` / `high` 22.05 kHz.
fn piper_rate(stem: &str) -> u32 {
    if stem.ends_with("-low") || stem.ends_with("-x_low") {
        16_000
    } else {
        22_050
    }
}

static MODELS: Lazy<Vec<CatalogModel>> = Lazy::new(|| {
    let kokoro_langs = || super::kokoro::LANGUAGES.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let mut v = vec![
        CatalogModel {
            id: "tts/kokoro-v1.0".into(),
            family: Family::Kokoro,
            title: "Kokoro v1.0 (multilingual)".into(),
            languages: kokoro_langs(),
            voices: VoiceSet::Kokoro(voices::KOKORO_V1_0),
            sample_rate: 24_000,
            piper_voice: None,
            piper_name: None,
        },
        CatalogModel {
            id: "tts/kokoro-v1.1-zh".into(),
            family: Family::Kokoro,
            title: "Kokoro v1.1-zh (Mandarin)".into(),
            languages: vec!["zh".into(), "en".into()],
            voices: VoiceSet::Kokoro(voices::KOKORO_V1_1),
            sample_rate: 24_000,
            piper_voice: None,
            piper_name: None,
        },
    ];
    for (size, title) in [
        ("mini", "Kitten mini 0.8 (English)"),
        ("micro", "Kitten micro 0.8 (English)"),
        ("nano", "Kitten nano 0.8 (English)"),
    ] {
        v.push(CatalogModel {
            id: format!("tts/kitten-{size}-en-v0.8"),
            family: Family::Kitten,
            title: title.into(),
            languages: vec!["en".into()],
            voices: VoiceSet::Kitten08,
            sample_rate: 24_000,
            piper_voice: None,
            piper_name: None,
        });
    }
    for (stem, language, name) in PIPER {
        let code = stem.split('-').next().unwrap_or(stem);
        v.push(CatalogModel {
            id: format!("tts/piper-{}", stem.to_lowercase()),
            family: Family::Piper,
            title: format!("{language}: {name} (Piper)"),
            languages: super::piper::language_tag(code).into_iter().collect(),
            voices: VoiceSet::Piper,
            sample_rate: piper_rate(stem),
            piper_voice: Some(stem),
            piper_name: Some(name),
        });
    }
    v
});

/// Every catalog model, in catalog order.
pub fn all() -> &'static [CatalogModel] {
    &MODELS
}

/// A catalog model by id.
pub fn get(id: &str) -> Option<&'static CatalogModel> {
    let id = id.trim();
    MODELS.iter().find(|m| m.id == id)
}
