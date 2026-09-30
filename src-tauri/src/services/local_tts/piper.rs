//! Piper voices (VITS, `piper1-gpl` exports) through sherpa-onnx.
//!
//! An official sherpa-onnx Piper package (`vits-piper-<voice>`) holds `<voice>.onnx`, Piper's
//! `<voice>.onnx.json`, `tokens.txt` and `espeak-ng-data/`. sherpa reads the espeak voice, sample
//! rate and speaker count from the ONNX metadata; Sigil reads the `.onnx.json` for what sherpa's C
//! API does not take from the model: the voice's own noise / length scales and the speaker names
//! of multi-speaker voices.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::commands::LocalTtsVoice;
use crate::services::speech_rt::sherpa::{TtsConfig, TtsModel};

/// What Sigil uses from a Piper `.onnx.json`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PiperVoiceInfo {
    pub sample_rate: Option<u32>,
    /// espeak-ng voice (`espeak.voice`).
    pub espeak_voice: Option<String>,
    /// `language.code` (`ru_RU`).
    pub language_code: Option<String>,
    pub noise_scale: Option<f32>,
    pub noise_w: Option<f32>,
    /// Piper's default length scale (> 1 = slower); sherpa's config fixes it at 1, so it is folded
    /// into the speed.
    pub length_scale: Option<f32>,
    /// Speaker names by speaker id (multi-speaker voices; empty for single-speaker ones).
    pub speakers: Vec<String>,
}

#[derive(Deserialize, Default)]
struct RawJson {
    #[serde(default)]
    audio: RawAudio,
    #[serde(default)]
    espeak: RawEspeak,
    #[serde(default)]
    language: RawLanguage,
    #[serde(default)]
    inference: RawInference,
    #[serde(default)]
    num_speakers: Option<u32>,
    #[serde(default)]
    speaker_id_map: BTreeMap<String, u32>,
}

#[derive(Deserialize, Default)]
struct RawAudio {
    sample_rate: Option<u32>,
}

#[derive(Deserialize, Default)]
struct RawEspeak {
    voice: Option<String>,
}

#[derive(Deserialize, Default)]
struct RawLanguage {
    code: Option<String>,
}

#[derive(Deserialize, Default)]
struct RawInference {
    noise_scale: Option<f32>,
    length_scale: Option<f32>,
    noise_w: Option<f32>,
}

/// Parse a Piper `.onnx.json` (unknown fields ignored; missing ones `None`).
pub fn parse_voice_json(text: &str) -> Result<PiperVoiceInfo, String> {
    let raw: RawJson = serde_json::from_str(text).map_err(|e| format!("invalid Piper voice config: {e}"))?;
    let finite = |v: Option<f32>| v.filter(|x| x.is_finite() && *x >= 0.0);
    let mut speakers: Vec<(u32, String)> = raw
        .speaker_id_map
        .into_iter()
        .map(|(name, id)| (id, name))
        .collect();
    speakers.sort();
    // Only a dense 0..n map describes speaker ids sherpa accepts.
    let dense = speakers
        .iter()
        .enumerate()
        .all(|(i, (id, _))| *id as usize == i);
    // `num_speakers` (when given) must agree with the map.
    let count_ok = raw.num_speakers.is_none_or(|n| n as usize == speakers.len());
    let speakers = if dense && count_ok && speakers.len() > 1 {
        speakers.into_iter().map(|(_, name)| name).collect()
    } else {
        Vec::new()
    };
    Ok(PiperVoiceInfo {
        sample_rate: raw.audio.sample_rate.filter(|r| *r > 0),
        espeak_voice: raw.espeak.voice.filter(|v| !v.trim().is_empty()),
        language_code: raw.language.code.filter(|v| !v.trim().is_empty()),
        noise_scale: finite(raw.inference.noise_scale),
        noise_w: finite(raw.inference.noise_w),
        length_scale: finite(raw.inference.length_scale).filter(|x| *x > 0.0),
        speakers,
    })
}

/// The `.onnx` model of a Piper package folder (the file with a `.onnx.json` next to it first,
/// else the only `.onnx`).
pub fn model_file(dir: &Path) -> Result<PathBuf, String> {
    let mut models: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x.eq_ignore_ascii_case("onnx")))
        .collect();
    models.sort();
    if let Some(m) = models.iter().find(|m| json_path(m).is_file()) {
        return Ok(m.clone());
    }
    match models.len() {
        1 => Ok(models.remove(0)),
        0 => Err(format!("{}: no Piper model (.onnx) found", dir.display())),
        _ => Err(format!("{}: more than one .onnx model and no Piper .onnx.json", dir.display())),
    }
}

/// `<model>.onnx.json`.
pub fn json_path(model: &Path) -> PathBuf {
    let mut s = model.as_os_str().to_owned();
    s.push(".json");
    PathBuf::from(s)
}

/// Read the voice info of a package (`Default` when the `.onnx.json` is missing).
pub fn voice_info(model: &Path) -> Result<PiperVoiceInfo, String> {
    let path = json_path(model);
    if !path.is_file() {
        return Ok(PiperVoiceInfo::default());
    }
    let text = std::fs::read_to_string(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    parse_voice_json(&text)
}

/// The sherpa config of a Piper package.
pub fn config(dir: &Path, model: &Path, info: &PiperVoiceInfo, threads: u32) -> TtsConfig {
    let data_dir = dir.join("espeak-ng-data");
    let lexicon = dir.join("lexicon.txt");
    let mut c = TtsConfig::new(TtsModel::Vits {
        model: model.to_path_buf(),
        tokens: dir.join("tokens.txt"),
        lexicon: lexicon.is_file().then_some(lexicon),
        data_dir: data_dir.is_dir().then_some(data_dir),
        noise_scale: info.noise_scale,
        noise_scale_w: info.noise_w,
    });
    c.num_threads = Some(threads);
    c.max_num_sentences = Some(1);
    c
}

/// The speed sherpa must be given for a requested `speed` (Piper's own length scale folded in).
pub fn effective_speed(info: &PiperVoiceInfo, speed: f32) -> f32 {
    match info.length_scale {
        Some(l) if (l - 1.0).abs() > 1e-3 => speed / l,
        _ => speed,
    }
}

/// The voices of a Piper package: its named speakers, else one voice named `default_name`.
pub fn voices(info: &PiperVoiceInfo, default_name: &str, language: Option<&str>) -> Vec<LocalTtsVoice> {
    if info.speakers.is_empty() {
        return vec![LocalTtsVoice {
            id: "0".into(),
            name: default_name.to_string(),
            language: language.map(str::to_string),
            gender: None,
        }];
    }
    info.speakers
        .iter()
        .enumerate()
        .map(|(i, name)| LocalTtsVoice {
            id: i.to_string(),
            name: pretty_speaker(name),
            language: language.map(str::to_string),
            gender: None,
        })
        .collect()
}

/// Speaker id of `voice`: a number, or a speaker name of the voice (any case).
pub fn speaker_id(info: &PiperVoiceInfo, voice: &str) -> Option<u32> {
    let v = voice.trim();
    if v.is_empty() {
        return Some(0);
    }
    if let Ok(n) = v.parse::<u32>() {
        let count = info.speakers.len().max(1) as u32;
        return (n < count).then_some(n);
    }
    info.speakers
        .iter()
        .position(|s| s.eq_ignore_ascii_case(v) || pretty_speaker(s).eq_ignore_ascii_case(v))
        .map(|i| i as u32)
}

/// `ru_RU` / `pt-PT` → a normalized tag (`ru`, `pt-PT`): the region is kept only where it names a
/// variety Sigil distinguishes.
pub fn language_tag(code: &str) -> Option<String> {
    let norm = super::voices::normalize_language(code)?;
    let base = super::voices::base(&norm);
    Some(match base.as_str() {
        "pt" | "en" => norm,
        _ => base,
    })
}

fn pretty_speaker(name: &str) -> String {
    let cleaned = name.replace('_', " ");
    let mut c = cleaned.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
}
