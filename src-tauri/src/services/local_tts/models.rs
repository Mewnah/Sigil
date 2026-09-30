//! What a request's `model` means and how it is loaded (the production [`Loader`]).
//!
//! `model` is one of:
//! - a catalog id (`tts/...`, [`catalog`]): located with `model_store::installed_path` (never
//!   downloaded here: installing is the user's decision in the panel or the Models tab);
//! - an absolute folder: a legacy `speech_package_v1` folder (`config.json` + ONNX + voices NPZ;
//!   the vendored Kitten), or an official sherpa-onnx Kokoro / Kitten / Piper package, recognised
//!   by its ONNX metadata (`model_type`), loaded once in a child process first
//!   (`probe::validate_in_child`: sherpa-onnx aborts on some malformed models);
//! - `hub:<org>/<name>`: a KittenML Hugging Face repo (the legacy `neuralOnnx` Hub setting). A repo
//!   whose model is also a catalog package ([`HUB_CATALOG`]) runs as that catalog model once it is
//!   installed (sherpa-onnx, hash-verified download through `ensure()` from the panel); until then a
//!   copy an earlier Sigil already downloaded runs on the vendored Kitten, and without one the
//!   request fails naming the catalog id. Any other repo runs on the vendored Kitten, its files
//!   fetched through the model store (`kitten::hub`), as the legacy setting always did;
//! - `""`: [`catalog::DEFAULT_MODEL`].
//!
//! Resolving is cheap (a few `stat`s and small file reads) and says which [`LoadKey`] the request
//! needs; opening loads the model (about 0.5-1.4 s for Kokoro fp32 / Piper / Kitten). Kokoro runs in
//! sherpa-onnx, or (opt-in until `ort_session` enables the CPU memory arena, see [`kokoro_ort`]) as
//! sherpa-onnx's text frontend with the graph on Sigil's ONNX Runtime, falling back to plain
//! sherpa-onnx if that cannot load.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use super::catalog::{self, CatalogModel, VoiceSet};
use super::commands::{LocalTtsModelInfo, LocalTtsVoice};
use super::engine::{LoadKey, Loader, OpenPlan, Resolved, Synth, VoiceChoice};
use super::kitten::{self, hub};
use super::{kokoro, kokoro_ort, piper, voices};
use crate::services::model_store;
use crate::services::speech_rt::sherpa::lib::{DEFAULT_THREADS, MAX_THREADS};
use crate::services::speech_rt::sherpa::{
    probe, GenerateOptions, OfflineTts, SherpaModel, SherpaRuntime, TtsConfig, TtsModel,
};

/// Prefix of a Hugging Face Kitten repo in `model`.
pub const HUB_PREFIX: &str = "hub:";
/// KittenML Hub repos (normalized ids, compared without case) whose model is also a catalog package:
/// the official sherpa-onnx `kitten-*-en-v0_8` packages are exports of these repos (same weights and
/// voices). The TS side keeps the same table (`HUB_CATALOG` in
/// `src/core/services/tts/services/localNeural.ts`; a vitest drift test compares the two).
pub const HUB_CATALOG: &[(&str, &str)] = &[
    ("KittenML/kitten-tts-mini-0.8", "tts/kitten-mini-en-v0.8"),
    ("KittenML/kitten-tts-micro-0.8", "tts/kitten-micro-en-v0.8"),
    ("KittenML/kitten-tts-nano-0.8", "tts/kitten-nano-en-v0.8"),
    ("KittenML/kitten-tts-nano-0.8-fp32", "tts/kitten-nano-en-v0.8"),
    ("KittenML/kitten-tts-nano-0.8-int8", "tts/kitten-nano-en-v0.8"),
];
/// Longest wait for the isolated first load of a user's model folder.
const VALIDATE_TIMEOUT: Duration = Duration::from_secs(90);
/// Kitten's useful speed range.
const KITTEN_SPEED: (f32, f32) = (0.5, 2.0);

/// Where a request's model lives.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelRef {
    Catalog(&'static CatalogModel),
    Folder(PathBuf),
    Hub(String),
}

/// Parse `model` (see the module docs).
pub fn parse_model_ref(model: &str) -> Result<ModelRef, String> {
    let m = model.trim();
    if m.is_empty() {
        return Ok(ModelRef::Catalog(catalog::get(catalog::DEFAULT_MODEL).expect("the default model is in the catalog")));
    }
    if let Some(repo) = m.strip_prefix(HUB_PREFIX) {
        let repo = hub::normalize_repo_id(repo).map_err(|e| format!("{e:#}"))?;
        return Ok(ModelRef::Hub(repo));
    }
    if m.starts_with("tts/") {
        return catalog::get(m)
            .map(ModelRef::Catalog)
            .ok_or_else(|| format!("unknown local voice model {m:?}"));
    }
    let path = Path::new(m);
    if path.is_absolute() {
        return Ok(ModelRef::Folder(path.to_path_buf()));
    }
    Err(format!("{m:?} is neither a local voice model id (tts/...) nor an absolute folder path"))
}

/// Inference threads: the request's (clamped to 1-4), else half the physical cores within 2-4
/// (VRChat is often CPU-bound; Kokoro gains little beyond 4 threads).
pub fn effective_threads(requested: Option<u32>) -> u32 {
    match requested {
        Some(n) if n > 0 => n.min(MAX_THREADS),
        _ => auto_threads(),
    }
}

fn auto_threads() -> u32 {
    static AUTO: once_cell::sync::Lazy<u32> = once_cell::sync::Lazy::new(|| {
        let physical = crate::services::sysinfo::cpu::probe().physical_cores;
        (physical / 2).clamp(DEFAULT_THREADS, MAX_THREADS)
    });
    *AUTO
}

/// The error for a catalog model that is not installed; names the catalog id (the TS panel
/// offers the download).
pub fn not_installed(m: &CatalogModel) -> String {
    format!("{} is not downloaded yet ({}); download it in the voice panel or Models & Runtimes", m.title, m.id)
}

/// Rows of `list_local_tts_models`: every catalog model with its install state (and, for an
/// installed Piper voice, its speakers).
pub fn list_models() -> Vec<LocalTtsModelInfo> {
    catalog::all()
        .iter()
        .map(|m| {
            let dir = model_store::installed_path(&m.id);
            catalog_info(m, dir.as_deref())
        })
        .collect()
}

/// A catalog model's info; `dir` is its install folder when installed.
pub fn catalog_info(m: &CatalogModel, dir: Option<&Path>) -> LocalTtsModelInfo {
    let mut voices = m.static_voices();
    let mut sample_rate = m.sample_rate;
    if let (VoiceSet::Piper, Some(dir)) = (m.voices, dir) {
        if let Some(info) = piper::model_file(dir).ok().and_then(|f| piper::voice_info(&f).ok()) {
            voices = piper::voices(&info, m.piper_name.unwrap_or("Default"), m.languages.first().map(String::as_str));
            if let Some(rate) = info.sample_rate {
                sample_rate = rate;
            }
        }
    }
    LocalTtsModelInfo {
        id: m.id.clone(),
        family: m.family.as_str().into(),
        title: m.title.clone(),
        languages: m.languages.clone(),
        voices,
        sample_rate,
        installed: dir.is_some(),
        default_voice: m.default_voice(None),
    }
}

// -------------------------------------------------------------------------------------------------
// Resolution
// -------------------------------------------------------------------------------------------------

fn resolve_catalog(m: &'static CatalogModel, voice: &str, language: Option<&str>, threads: u32) -> Result<Resolved, String> {
    let dir = model_store::installed_path(&m.id).ok_or_else(|| not_installed(m))?;
    let info = catalog_info(m, Some(&dir));
    let key_model = m.id.clone();
    match m.voices {
        VoiceSet::Kokoro(table) => {
            let name = pick(voice, || m.default_voice(language));
            let sid = voices::speaker_id(table, &name).ok_or_else(|| unknown_voice(&name, &m.title))?;
            let name = table[sid as usize].to_string();
            let setup = kokoro::KokoroSetup::for_voice(&name, language);
            let config = kokoro::config(&dir, &setup, threads)?;
            let voice_language = voices::kokoro_voice_language(&name).map(str::to_string);
            Ok(Resolved {
                key: LoadKey::new(key_model, kokoro_variant(&setup), threads),
                info,
                voice: VoiceChoice { sid, name },
                speed_factor: 1.0,
                language: voice_language,
                plan: OpenPlan::Sherpa {
                    config: Box::new(config),
                    validate_first: false,
                    kitten_fallback: None,
                },
            })
        }
        VoiceSet::Kitten08 => {
            let name = pick(voice, || m.default_voice(language));
            let sid = voices::kitten_speaker_id(&name).ok_or_else(|| unknown_voice(&name, &m.title))?;
            let model = onnx_in(&dir, &["model.onnx", "model.fp32.onnx", "model.int8.onnx", "model.fp16.onnx"])?;
            let config = kitten_config(&dir, &model, threads);
            let speakers: Vec<String> = voices::KITTEN_V0_8.iter().map(|(_, raw)| raw.to_string()).collect();
            Ok(Resolved {
                key: LoadKey::new(key_model, String::new(), threads),
                info,
                voice: VoiceChoice {
                    sid,
                    name: voices::KITTEN_V0_8[sid as usize].0.to_string(),
                },
                speed_factor: 1.0,
                language: Some("en".into()),
                plan: OpenPlan::Sherpa {
                    config: Box::new(config),
                    validate_first: false,
                    kitten_fallback: Some(KittenFallback { model, dir, speakers }),
                },
            })
        }
        VoiceSet::Piper => {
            let model = piper::model_file(&dir)?;
            let pinfo = piper::voice_info(&model)?;
            let sid = piper::speaker_id(&pinfo, voice).ok_or_else(|| unknown_voice(voice, &m.title))?;
            let name = info
                .voices
                .get(sid as usize)
                .map(|v| v.name.clone())
                .unwrap_or_else(|| sid.to_string());
            let config = piper::config(&dir, &model, &pinfo, threads);
            Ok(Resolved {
                key: LoadKey::new(key_model, String::new(), threads),
                speed_factor: piper::effective_speed(&pinfo, 1.0),
                language: m.languages.first().cloned(),
                info,
                voice: VoiceChoice { sid, name },
                plan: OpenPlan::Sherpa {
                    config: Box::new(config),
                    validate_first: false,
                    kitten_fallback: None,
                },
            })
        }
    }
}

fn resolve_folder(dir: &Path, voice: &str, language: Option<&str>, threads: u32) -> Result<Resolved, String> {
    if !dir.is_dir() {
        return Err(format!("the voice folder {} does not exist", dir.display()));
    }
    let title = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| dir.display().to_string());
    let key_model = dir.display().to_string();
    let base_info = |family: &str, languages: Vec<String>, voices: Vec<LocalTtsVoice>, rate: u32, default_voice: String| LocalTtsModelInfo {
        id: key_model.clone(),
        family: family.into(),
        title: title.clone(),
        languages,
        voices,
        sample_rate: rate,
        installed: true,
        default_voice,
    };
    // A legacy speech package (the vendored Kitten's `config.json`).
    if let Ok(bytes) = std::fs::read(dir.join("config.json")) {
        if hub::ModelConfig::parse(&bytes).is_ok() {
            return Ok(Resolved {
                key: LoadKey::new(key_model.clone(), "speech_package_v1".into(), threads),
                info: base_info("kitten", vec!["en".into()], Vec::new(), kitten::SAMPLE_RATE, String::new()),
                voice: VoiceChoice {
                    sid: 0,
                    name: voice.trim().to_string(),
                },
                speed_factor: 1.0,
                language: Some("en".into()),
                plan: OpenPlan::KittenPackage(dir.to_path_buf()),
            });
        }
    }
    // An official sherpa-onnx package: the ONNX metadata tells the family.
    let model = onnx_in(dir, &["model.onnx", "model.fp32.onnx", "model.int8.onnx", "model.fp16.onnx"])
        .or_else(|_| piper::model_file(dir))?;
    let meta = crate::services::speech_rt::sherpa::onnx_info::summary(&model)?;
    let model_type = meta.meta("model_type").unwrap_or_default().to_ascii_lowercase();
    let speaker_names: Vec<String> = meta
        .meta("speaker_names")
        .map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
        .unwrap_or_default();
    let rate = meta
        .meta("sample_rate")
        .and_then(|r| r.trim().parse::<u32>().ok())
        .unwrap_or(24_000);
    let unrecognized = || format!("{} is not a Kokoro, Kitten or Piper voice folder (model type {model_type:?})", dir.display());
    match model_type.as_str() {
        "kokoro" => {
            if speaker_names.is_empty() {
                return Err(format!("{}: the Kokoro model lists no voices", dir.display()));
            }
            let table: Vec<&str> = speaker_names.iter().map(String::as_str).collect();
            // Defaults as for the catalog models, with the language most of the folder's voices
            // speak as its own (an official v1.1-zh folder: Mandarin).
            let own = voices::kokoro_primary_language(&table);
            let default_voice = voices::kokoro_default_voice(&table, own.as_deref()).to_string();
            let name = pick_voice(
                voice,
                &title,
                |v| voices::speaker_id(&table, v).is_some(),
                || {
                    let lang = language.filter(|l| voices::kokoro_has_language(&table, l)).or(own.as_deref());
                    voices::kokoro_default_voice(&table, lang).to_string()
                },
            );
            let sid = voices::speaker_id(&table, &name).ok_or_else(|| unknown_voice(&name, &title))?;
            let name = table[sid as usize].to_string();
            let setup = kokoro::KokoroSetup::for_voice(&name, language);
            let config = kokoro::config(dir, &setup, threads)?;
            let voice_rows = speaker_names
                .iter()
                .map(|v| LocalTtsVoice {
                    id: v.clone(),
                    name: voices::kokoro_display_name(v),
                    language: voices::kokoro_voice_language(v).map(str::to_string),
                    gender: voices::kokoro_voice_gender(v).map(str::to_string),
                })
                .collect();
            let mut langs: Vec<String> = Vec::new();
            for v in &speaker_names {
                if let Some(l) = voices::kokoro_voice_language(v) {
                    let l = if voices::base(l) == "en" { "en".to_string() } else { l.to_string() };
                    if !langs.contains(&l) {
                        langs.push(l);
                    }
                }
            }
            Ok(Resolved {
                key: LoadKey::new(key_model.clone(), kokoro_variant(&setup), threads),
                info: base_info("kokoro", langs, voice_rows, rate, default_voice),
                language: voices::kokoro_voice_language(&name).map(str::to_string),
                voice: VoiceChoice { sid, name },
                speed_factor: 1.0,
                plan: OpenPlan::Sherpa {
                    config: Box::new(config),
                    validate_first: true,
                    kitten_fallback: None,
                },
            })
        }
        t if t.starts_with("kitten") => {
            let known = speaker_names.len() == voices::KITTEN_V0_8.len()
                && voices::KITTEN_V0_8
                    .iter()
                    .zip(&speaker_names)
                    .all(|((_, raw), s)| raw == s);
            let (voice_rows, sid, name) = if known {
                let name = pick(voice, || "Jasper".to_string());
                let sid = voices::kitten_speaker_id(&name).ok_or_else(|| unknown_voice(&name, &title))?;
                (voices::kitten_voices(), sid, voices::KITTEN_V0_8[sid as usize].0.to_string())
            } else {
                let table: Vec<&str> = speaker_names.iter().map(String::as_str).collect();
                let name = pick_voice(
                    voice,
                    &title,
                    |v| table.is_empty() || voices::speaker_id(&table, v).is_some(),
                    || table.first().map(|s| s.to_string()).unwrap_or_default(),
                );
                let sid = if table.is_empty() {
                    0
                } else {
                    voices::speaker_id(&table, &name).ok_or_else(|| unknown_voice(&name, &title))?
                };
                let rows = speaker_names
                    .iter()
                    .map(|s| LocalTtsVoice {
                        id: s.clone(),
                        name: s.clone(),
                        language: Some("en-US".into()),
                        gender: None,
                    })
                    .collect();
                (rows, sid, name)
            };
            let default_voice = voice_rows.first().map(|v: &LocalTtsVoice| v.id.clone()).unwrap_or_default();
            Ok(Resolved {
                key: LoadKey::new(key_model.clone(), String::new(), threads),
                info: base_info("kitten", vec!["en".into()], voice_rows, rate, default_voice),
                voice: VoiceChoice { sid, name },
                speed_factor: 1.0,
                language: Some("en".into()),
                plan: OpenPlan::Sherpa {
                    config: Box::new(kitten_config(dir, &model, threads)),
                    validate_first: true,
                    kitten_fallback: None,
                },
            })
        }
        "vits" => {
            let pinfo = piper::voice_info(&model)?;
            let language_tag = pinfo
                .language_code
                .as_deref()
                .and_then(piper::language_tag)
                .or_else(|| meta.meta("voice").and_then(piper::language_tag));
            let voice_rows = piper::voices(&pinfo, &title, language_tag.as_deref());
            let voice = pick_voice(voice, &title, |v| piper::speaker_id(&pinfo, v).is_some(), String::new);
            let sid = piper::speaker_id(&pinfo, &voice).ok_or_else(|| unknown_voice(&voice, &title))?;
            let name = voice_rows
                .get(sid as usize)
                .map(|v| v.name.clone())
                .unwrap_or_else(|| sid.to_string());
            Ok(Resolved {
                key: LoadKey::new(key_model.clone(), String::new(), threads),
                info: base_info(
                    "piper",
                    language_tag.clone().into_iter().collect(),
                    voice_rows,
                    pinfo.sample_rate.unwrap_or(rate),
                    "0".into(),
                ),
                voice: VoiceChoice { sid, name },
                speed_factor: piper::effective_speed(&pinfo, 1.0),
                language: language_tag,
                plan: OpenPlan::Sherpa {
                    config: Box::new(piper::config(dir, &model, &pinfo, threads)),
                    validate_first: true,
                    kitten_fallback: None,
                },
            })
        }
        _ => Err(unrecognized()),
    }
}

/// The catalog model a KittenML Hub repo is also packaged as ([`HUB_CATALOG`]), if any.
pub fn hub_catalog_model(repo: &str) -> Option<&'static CatalogModel> {
    let repo = hub::normalize_repo_id(repo).ok()?;
    HUB_CATALOG
        .iter()
        .find(|(r, _)| r.eq_ignore_ascii_case(&repo))
        .and_then(|(_, id)| catalog::get(id))
}

/// How a `hub:` request runs (see [`hub_choice`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HubChoice {
    /// The installed catalog package (sherpa-onnx).
    Catalog,
    /// The vendored Kitten on the repo's files (a local copy, or downloaded through the store).
    Vendored,
    /// The catalog package must be downloaded first (the error names its id).
    NotInstalled,
}

/// Decide how a Hub repo runs: its catalog package when installed; else a local copy an earlier
/// Sigil downloaded (no new download); else the catalog package must be installed first; a repo
/// without a catalog package keeps the legacy Hub download.
pub fn hub_choice(has_catalog_model: bool, catalog_installed: bool, local_copy: bool) -> HubChoice {
    match (has_catalog_model, catalog_installed, local_copy) {
        (true, true, _) => HubChoice::Catalog,
        (_, _, true) | (false, _, false) => HubChoice::Vendored,
        (true, false, false) => HubChoice::NotInstalled,
    }
}

/// Whether the repo's files are already on this PC (the legacy hf-hub cache, or the app-data copy).
fn hub_local_copy(repo: &str) -> bool {
    if hub::legacy_hf_snapshot(repo).is_some_and(|d| hub::is_complete(&d)) {
        return true;
    }
    crate::services::diagnostics::app_data_dir().is_some_and(|app| hub::is_complete(&hub::repo_dir(&app, repo)))
}

fn resolve_hub(repo: &str, voice: &str, language: Option<&str>, threads: u32) -> Result<Resolved, String> {
    let Some(m) = hub_catalog_model(repo) else {
        return Ok(resolve_hub_vendored(repo, voice, threads));
    };
    let installed = model_store::installed_path(&m.id).is_some();
    // Only look for a local copy when the catalog package is not installed (a few stats).
    let local_copy = !installed && hub_local_copy(repo);
    match hub_choice(true, installed, local_copy) {
        HubChoice::Catalog => resolve_catalog(m, voice, language, threads),
        HubChoice::Vendored => Ok(resolve_hub_vendored(repo, voice, threads)),
        HubChoice::NotInstalled => Err(not_installed(m)),
    }
}

fn resolve_hub_vendored(repo: &str, voice: &str, threads: u32) -> Resolved {
    Resolved {
        key: LoadKey::new(format!("{HUB_PREFIX}{repo}"), "hub".into(), threads),
        info: LocalTtsModelInfo {
            id: format!("{HUB_PREFIX}{repo}"),
            family: "kitten".into(),
            title: repo.to_string(),
            languages: vec!["en".into()],
            voices: Vec::new(),
            sample_rate: kitten::SAMPLE_RATE,
            installed: true,
            default_voice: String::new(),
        },
        voice: VoiceChoice {
            sid: 0,
            name: voice.trim().to_string(),
        },
        speed_factor: 1.0,
        language: Some("en".into()),
        plan: OpenPlan::KittenHub(repo.to_string()),
    }
}

fn kokoro_variant(setup: &kokoro::KokoroSetup) -> String {
    format!(
        "lang={};lexicon={};zh-rules={}",
        setup.espeak,
        if setup.british { "gb" } else { "us" },
        setup.mandarin_rules
    )
}

fn kitten_config(dir: &Path, model: &Path, threads: u32) -> TtsConfig {
    let data_dir = dir.join("espeak-ng-data");
    let mut c = TtsConfig::new(TtsModel::Kitten {
        model: model.to_path_buf(),
        voices: dir.join("voices.bin"),
        tokens: dir.join("tokens.txt"),
        data_dir: data_dir.is_dir().then_some(data_dir),
    });
    c.num_threads = Some(threads);
    c.max_num_sentences = Some(1);
    c
}

/// The first of `preferred` present in `dir`, else the only `.onnx` file.
fn onnx_in(dir: &Path, preferred: &[&str]) -> Result<PathBuf, String> {
    if let Some(p) = preferred.iter().map(|f| dir.join(f)).find(|p| p.is_file()) {
        return Ok(p);
    }
    let found: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x.eq_ignore_ascii_case("onnx")))
        .collect();
    match found.as_slice() {
        [one] => Ok(one.clone()),
        [] => Err(format!("{}: no ONNX model found", dir.display())),
        _ => Err(format!("{}: more than one ONNX model; keep only one", dir.display())),
    }
}

fn pick(voice: &str, default: impl FnOnce() -> String) -> String {
    let v = voice.trim();
    if v.is_empty() {
        default()
    } else {
        v.to_string()
    }
}

/// The voice a folder request speaks with: `voice`, or `default` when it is blank, or when it is a
/// Kitten voice name (Jasper, ..., `expr-voice-*`) the model does not have. The legacy `neuralOnnx`
/// settings default to "Jasper", which a Kokoro or Piper folder does not know, and a folder lists
/// its voices only once it has loaded. Any other unknown name is kept (and reported as unknown).
pub fn pick_voice(voice: &str, title: &str, known: impl Fn(&str) -> bool, default: impl FnOnce() -> String) -> String {
    let v = voice.trim();
    if v.is_empty() {
        return default();
    }
    if !known(v) && voices::is_kitten_voice_name(v) {
        let fallback = default();
        tracing::info!(
            target: "sigil::tts",
            "local TTS: {title} has no voice {v:?} (a Kitten voice name); using its default voice {fallback:?}"
        );
        return fallback;
    }
    v.to_string()
}

fn unknown_voice(voice: &str, title: &str) -> String {
    format!("{title} has no voice {:?}", voice.trim())
}

// -------------------------------------------------------------------------------------------------
// Opening
// -------------------------------------------------------------------------------------------------

/// Where the vendored Kitten finds a sherpa Kitten package when the sherpa runtime cannot load.
#[derive(Clone, Debug, PartialEq)]
pub struct KittenFallback {
    pub model: PathBuf,
    pub dir: PathBuf,
    /// sherpa speaker names in speaker-id order (rows of `voices.bin`).
    pub speakers: Vec<String>,
}

/// The production loader: catalog / folder / Hub models, sherpa-onnx or the vendored Kitten.
pub struct StoreLoader {
    /// Serialized configs of user folders that loaded cleanly in a child process.
    validated: Mutex<HashSet<String>>,
    /// Run Kokoro's graph on Sigil's ONNX Runtime behind sherpa-onnx's frontend ([`kokoro_ort`])
    /// instead of entirely inside sherpa-onnx.
    kokoro_on_sigil_ort: bool,
}

impl Default for StoreLoader {
    fn default() -> Self {
        Self::new(kokoro_ort::use_sigil_runtime())
    }
}

impl StoreLoader {
    /// A loader; `kokoro_on_sigil_ort` picks the Kokoro runtime ([`kokoro_ort`]).
    pub fn new(kokoro_on_sigil_ort: bool) -> Self {
        Self {
            validated: Mutex::new(HashSet::new()),
            kokoro_on_sigil_ort,
        }
    }
}

impl Loader for StoreLoader {
    fn resolve(&self, model: &str, voice: &str, language: Option<&str>, threads: Option<u32>) -> Result<Resolved, String> {
        let threads = effective_threads(threads);
        let language = language.and_then(voices::normalize_language);
        let language = language.as_deref();
        match parse_model_ref(model)? {
            ModelRef::Catalog(m) => resolve_catalog(m, voice, language, threads),
            ModelRef::Folder(dir) => resolve_folder(&dir, voice, language, threads),
            ModelRef::Hub(repo) => resolve_hub(&repo, voice, language, threads),
        }
    }

    fn open(&self, resolved: &Resolved) -> Result<Box<dyn Synth>, String> {
        match &resolved.plan {
            OpenPlan::Sherpa {
                config,
                validate_first,
                kitten_fallback,
            } => match SherpaRuntime::installed() {
                Ok(rt) => {
                    if *validate_first {
                        self.validate(config)?;
                    }
                    if matches!(config.model, TtsModel::Kokoro { .. }) && self.kokoro_on_sigil_ort {
                        match KokoroHybridSynth::open(&rt, config) {
                            Ok(synth) => return Ok(Box::new(synth)),
                            Err(e) => tracing::warn!(
                                target: "sigil::tts",
                                "local TTS: Kokoro on Sigil's ONNX Runtime is unavailable ({e}); using sherpa-onnx's own runtime"
                            ),
                        }
                    }
                    let tts = OfflineTts::new(&rt, config)?;
                    Ok(Box::new(SherpaSynth { tts }))
                }
                Err(e) => match kitten_fallback {
                    Some(fb) => {
                        tracing::warn!(
                            target: "sigil::tts",
                            "local TTS: sherpa-onnx is unavailable ({e}); running Kitten on Sigil's own ONNX Runtime"
                        );
                        open_kitten_fallback(fb)
                    }
                    None => Err(e),
                },
            },
            OpenPlan::KittenPackage(dir) => {
                let tts = hub::load_from_dir(dir).map_err(|e| format!("cannot load the speech package {}: {e:#}", dir.display()))?;
                Ok(Box::new(KittenSynth::new(tts)))
            }
            OpenPlan::KittenHub(repo) => {
                let tts = hub::load_from_hub(repo).map_err(|e| format!("cannot load {repo}: {e:#}"))?;
                Ok(Box::new(KittenSynth::new(tts)))
            }
            #[cfg(test)]
            OpenPlan::Test(_) => Err("test plans are opened by the test loader".into()),
        }
    }
}

impl StoreLoader {
    /// Load a user folder's final config once in a child process (cached per config).
    fn validate(&self, config: &TtsConfig) -> Result<(), String> {
        let model = SherpaModel::Tts(config.clone());
        let key = serde_json::to_string(&model).map_err(|e| e.to_string())?;
        if self
            .validated
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&key)
        {
            return Ok(());
        }
        probe::validate_in_child(&model, VALIDATE_TIMEOUT).map_err(|e| format!("this voice folder cannot be used: {e}"))?;
        self.validated
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key);
        Ok(())
    }
}

fn open_kitten_fallback(fb: &KittenFallback) -> Result<Box<dyn Synth>, String> {
    let aliases = voices::KITTEN_V0_8
        .iter()
        .map(|(name, raw)| (name.to_string(), raw.to_string()))
        .collect();
    let tts = hub::load_sherpa_kitten_package(&fb.model, &fb.dir.join("voices.bin"), &fb.speakers, aliases)
        .map_err(|e| format!("cannot load Kitten on Sigil's ONNX Runtime: {e:#}"))?;
    crate::services::diagnostics::record_ep_outcome("tts.kitten", crate::services::diagnostics::EpOutcome::granted("cpu"));
    Ok(Box::new(KittenSynth::new(tts)))
}

/// A sherpa-onnx voice.
struct SherpaSynth {
    tts: OfflineTts,
}

impl Synth for SherpaSynth {
    fn sample_rate(&self) -> u32 {
        self.tts.sample_rate()
    }

    fn generate(
        &mut self,
        text: &str,
        voice: &VoiceChoice,
        speed: f32,
        cancel: &AtomicBool,
        on_chunk: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<bool, String> {
        let opts = GenerateOptions {
            sid: voice.sid,
            speed,
            silence_scale: None,
        };
        let mut cb = |s: &[f32]| on_chunk(s);
        let audio = self.tts.generate(text, &opts, Some(cancel), Some(&mut cb))?;
        Ok(!audio.cancelled)
    }
}

/// Kokoro with sherpa-onnx's text frontend and the graph on Sigil's ONNX Runtime ([`kokoro_ort`]:
/// the same speech for 15-20% less CPU than inside sherpa-onnx).
struct KokoroHybridSynth {
    /// sherpa-onnx running the token tap instead of the real graph.
    frontend: OfflineTts,
    model: kokoro_ort::KokoroModel,
    silence_scale: f32,
}

impl KokoroHybridSynth {
    fn open(rt: &std::sync::Arc<SherpaRuntime>, config: &TtsConfig) -> Result<Self, String> {
        let TtsModel::Kokoro { model, voices, .. } = &config.model else {
            return Err("not a Kokoro model".into());
        };
        config.validate()?;
        let meta = crate::services::speech_rt::sherpa::onnx_info::summary(model)?;
        let speakers = meta
            .meta("n_speakers")
            .and_then(|n| n.trim().parse::<usize>().ok())
            .filter(|n| *n > 0)
            .ok_or("the Kokoro model does not say how many voices it has")?;
        let tap = kokoro_ort::write_tap(&kokoro_ort::tap_dir(), &meta.metadata, "audio")?;
        let mut tap_config = config.clone();
        if let TtsModel::Kokoro { model, .. } = &mut tap_config.model {
            *model = tap;
        }
        // The tap is trivial: one thread, so sherpa's ONNX Runtime has no pool threads spinning
        // next to the real model's.
        tap_config.num_threads = Some(1);
        let frontend = OfflineTts::new(rt, &tap_config)?;
        if frontend.num_speakers() as usize != speakers {
            return Err(format!("the text frontend reports {} voices, the model {speakers}", frontend.num_speakers()));
        }
        let threads = config.num_threads.unwrap_or(DEFAULT_THREADS).clamp(1, MAX_THREADS);
        let model = kokoro_ort::KokoroModel::load(model, voices, speakers, threads)?;
        Ok(Self {
            frontend,
            model,
            silence_scale: crate::services::speech_rt::sherpa::tts::silence_scale(config.silence_scale),
        })
    }
}

impl Synth for KokoroHybridSynth {
    fn sample_rate(&self) -> u32 {
        self.frontend.sample_rate()
    }

    fn generate(
        &mut self,
        text: &str,
        voice: &VoiceChoice,
        speed: f32,
        cancel: &AtomicBool,
        on_chunk: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<bool, String> {
        if cancel.load(Ordering::Acquire) {
            return Ok(false);
        }
        // The frontend is fast (lexicons and espeak-ng); the tap's output is the token ids.
        let mut tapped: Vec<f32> = Vec::new();
        let opts = GenerateOptions {
            sid: voice.sid,
            speed: 1.0,
            // No scaling of the tap's "audio" (it holds no silence anyway).
            silence_scale: Some(1.0),
        };
        let mut collect = |s: &[f32]| {
            tapped.extend_from_slice(s);
            true
        };
        let out = self.frontend.generate(text, &opts, Some(cancel), Some(&mut collect))?;
        if out.cancelled {
            return Ok(false);
        }
        let rate = self.frontend.sample_rate();
        for tokens in kokoro_ort::decode_tap(&tapped)? {
            if cancel.load(Ordering::Acquire) {
                return Ok(false);
            }
            let audio = self.model.run(&tokens, voice.sid, speed)?;
            let audio = kokoro_ort::scale_silence(&audio, rate, self.silence_scale);
            if !audio.is_empty() && !on_chunk(&audio) {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// The vendored Kitten (Sigil's ONNX Runtime, pure-Rust espeak-ng).
struct KittenSynth {
    tts: kitten::KittenTTS,
    default_voice: String,
}

impl KittenSynth {
    fn new(tts: kitten::KittenTTS) -> Self {
        let mut names = tts.available_voices.clone();
        names.sort();
        // KittenML's friendly default, else the first voice.
        let default_voice = if tts.resolves_voice("Jasper") {
            "Jasper".to_string()
        } else {
            names.first().cloned().unwrap_or_default()
        };
        Self { tts, default_voice }
    }
}

impl Synth for KittenSynth {
    fn sample_rate(&self) -> u32 {
        kitten::SAMPLE_RATE
    }

    fn voices(&self) -> Option<Vec<LocalTtsVoice>> {
        let mut names = self.tts.voice_names();
        names.sort();
        Some(
            names
                .into_iter()
                .map(|n| LocalTtsVoice {
                    id: n.clone(),
                    name: n,
                    language: Some("en-US".into()),
                    gender: None,
                })
                .collect(),
        )
    }

    fn default_voice(&self) -> Option<String> {
        Some(self.default_voice.clone())
    }

    fn generate(
        &mut self,
        text: &str,
        voice: &VoiceChoice,
        speed: f32,
        cancel: &AtomicBool,
        on_chunk: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<bool, String> {
        if cancel.load(Ordering::Acquire) {
            return Ok(false);
        }
        let name = if voice.name.trim().is_empty() {
            self.default_voice.as_str()
        } else {
            voice.name.trim()
        };
        let speed = if speed.is_finite() {
            speed.clamp(KITTEN_SPEED.0, KITTEN_SPEED.1)
        } else {
            1.0
        };
        let audio = self
            .tts
            .generate(text, name, speed, true)
            .map_err(|e| format!("{e:#}"))?;
        if cancel.load(Ordering::Acquire) {
            return Ok(false);
        }
        if audio.is_empty() {
            return Ok(true);
        }
        Ok(on_chunk(&audio))
    }
}
