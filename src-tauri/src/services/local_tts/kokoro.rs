//! Kokoro through sherpa-onnx: which espeak-ng voice, lexicons and text-normalization rules a
//! Kokoro voice needs.
//!
//! sherpa-onnx 1.13.8's multi-language Kokoro frontend looks each word up in the lexicons first
//! (English: misaki's US or British dictionary; Chinese: `lexicon-zh.txt`) and phonemizes the rest
//! with espeak-ng in the configured `lang`. Measured with the official v1.0 package (P2-H):
//! - `lang` must name an espeak-ng voice file (`espeak-ng-data/lang/**`, any case): `en-us`, `en`
//!   (British), `es`, `fr`, `hi`, `it`, `ja`, `pt-br`, `cmn`. `en-gb` and `fr-fr` are language
//!   codes, not voice names: sherpa then fails every sentence ("Failed to set eSpeak-ng voice").
//! - The Chinese rule FSTs (`phone-zh`, `date-zh`, `number-zh`) turn digits into Chinese
//!   characters before phonemization, so an English voice read "I have 3 apples" as "I have sand
//!   apples". They are applied only to Mandarin voices.
//! - `lang` and the lexicons are fixed when the model loads; a voice of another language reloads
//!   the model (about 1 s for the fp32 v1.0 model).
//! - Japanese: the frontend has no Japanese reading dictionary. Kana are read correctly (particles
//!   by their spelling, so は is "ha"), but kanji go through the Mandarin lexicon and come out with
//!   Chinese readings ("今日はいい天気ですね" was heard as "人流 拝見地位 です よね"; without
//!   `lexicon-zh.txt` espeak-ng drops them entirely, which is worse). Natural Japanese needs a
//!   kanji-to-kana reading pass (a morphological analyzer with a dictionary, e.g. vibrato + IPADIC
//!   through the model store) before sherpa; until then the panel points Japanese users to
//!   VOICEVOX / AivisSpeech.

use std::path::Path;

use super::voices;
use crate::services::speech_rt::sherpa::{TtsConfig, TtsModel};

/// espeak-ng voice for a Kokoro language (a normalized tag from [`voices::normalize_language`]).
pub fn espeak_voice(language: &str) -> Option<&'static str> {
    let lang = language.to_ascii_lowercase();
    Some(match lang.as_str() {
        "en-gb" => "en",
        "en" | "en-us" => "en-us",
        "es" | "es-es" | "es-mx" => "es",
        "fr" | "fr-fr" => "fr",
        "hi" | "hi-in" => "hi",
        "it" | "it-it" => "it",
        "ja" | "ja-jp" => "ja",
        "pt" | "pt-br" => "pt-br",
        "zh" | "zh-cn" | "cmn" => "cmn",
        _ => return None,
    })
}

/// Languages Kokoro speaks natively (its voices' languages).
pub const LANGUAGES: &[&str] = &["en", "ja", "zh", "es", "fr", "hi", "it", "pt-BR"];

/// How a Kokoro model is configured for one voice: the model-level part of the choice (a
/// different value needs a reload).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KokoroSetup {
    /// espeak-ng voice (`lang` of the sherpa config).
    pub espeak: &'static str,
    /// British English dictionary instead of the US one.
    pub british: bool,
    /// Mandarin text normalization (numbers, dates, phone numbers read in Chinese).
    pub mandarin_rules: bool,
}

impl KokoroSetup {
    /// The setup for `voice` (its own language), unless `language` names another Kokoro language
    /// (a cross-lingual read, e.g. an English voice reading Spanish).
    pub fn for_voice(voice: &str, language: Option<&str>) -> KokoroSetup {
        let voice_lang = voices::kokoro_voice_language(voice).unwrap_or("en-US");
        let lang = language
            .filter(|l| espeak_voice(l).is_some())
            // A bare "en" keeps the voice's own accent.
            .filter(|l| !(l.eq_ignore_ascii_case("en") && voices::base(voice_lang) == "en"))
            .unwrap_or(voice_lang);
        let espeak = espeak_voice(lang).unwrap_or("en-us");
        KokoroSetup {
            espeak,
            british: espeak == "en",
            mandarin_rules: espeak == "cmn",
        }
    }
}

/// The sherpa config of an installed Kokoro package in `dir` for `setup`.
pub fn config(dir: &Path, setup: &KokoroSetup, threads: u32) -> Result<TtsConfig, String> {
    let model = ["model.onnx", "model.int8.onnx"]
        .iter()
        .map(|f| dir.join(f))
        .find(|p| p.is_file())
        .ok_or_else(|| format!("{}: the Kokoro model file is missing", dir.display()))?;
    let english = if setup.british && dir.join("lexicon-gb-en.txt").is_file() {
        "lexicon-gb-en.txt"
    } else {
        "lexicon-us-en.txt"
    };
    // English first: sherpa consults the lexicons in order.
    let lexicons = [english, "lexicon-zh.txt"]
        .iter()
        .map(|f| dir.join(f))
        .filter(|p| p.is_file())
        .collect();
    let data_dir = dir.join("espeak-ng-data");
    let mut c = TtsConfig::new(TtsModel::Kokoro {
        model,
        voices: dir.join("voices.bin"),
        tokens: dir.join("tokens.txt"),
        data_dir: data_dir.is_dir().then_some(data_dir),
        lexicons,
        lang: Some(setup.espeak.to_string()),
    });
    if setup.mandarin_rules {
        // sherpa's documented order: phone numbers, then dates, then numbers.
        c.rule_fsts = ["phone-zh.fst", "date-zh.fst", "number-zh.fst"]
            .iter()
            .map(|f| dir.join(f))
            .filter(|p| p.is_file())
            .collect();
    }
    c.num_threads = Some(threads);
    c.max_num_sentences = Some(1);
    Ok(c)
}
