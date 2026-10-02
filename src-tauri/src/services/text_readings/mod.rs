//! `text-readings` plugin: readings of Japanese and Chinese caption text (P6-L; research §5).
//!
//! - Japanese ([`japanese`]): jpreprocess (OpenJTalk's frontend in Rust) on the NAIST-JDIC
//!   dictionary that the model store installs as [`japanese::DICTIONARY_ID`] (never a build-time
//!   download) gives each word's katakana reading and pronunciation, including numbers with their
//!   counters; `wana_kana` turns them into hiragana and Hepburn romaji ([`romaji`]: macrons for
//!   long vowels, particles は / へ / を as wa / e / o, words spaced and inflections joined).
//! - Chinese ([`chinese`]): `jieba-rs` word segmentation, then pinyin per word from a phrase table
//!   for polyphones and neutral tones ([`chinese`]'s `zh_phrases.txt`) with the `pinyin` crate per
//!   character otherwise, and tone sandhi (不 / 一 / third tone within a word), as tone marks or
//!   numbers. Traditional characters read per character; Cantonese text is read as Mandarin.
//! - The TS readings service (`src/core/services/readings/**`) calls [`text_readings_annotate`] for
//!   finals (and optionally interims) of `text.stt` / `others.stt` in those languages and publishes
//!   the result on `readings.stt` (TS `ReadingsEvent`), keyed by the text's `utteranceId`.
//!
//! # Frozen contract (P6-S0; change only through INTEGRATE; structs may gain fields)
//! - IPC (`plugin-commands/text-readings.txt`; wire camelCase; TS mirror `RustReadings*` in
//!   `src/types.ts`):
//!   - `text_readings_annotate { request: ReadingsRequest } -> ReadingsResult`: CPU work, run off the
//!     async runtime; loads the dictionary on first use. Errors: the dictionary is not installed
//!     (message names the catalog id), unsupported language, text over [`MAX_TEXT_CHARS`].
//!   - `text_readings_status -> ReadingsStatus`, `text_readings_unload -> ()` (frees the dictionary).
//!
//! # Output limits
//! The TS `ReadingsEventSchema` accepts at most 4000 UTF-16 units per reading string and 512
//! segments; [`ReadingsResult`]s stay within both ([`MAX_READING_UNITS`], [`MAX_SEGMENTS`]): a longer
//! reading is cut with `…`, and segments past the limit are merged into one segment without a
//! reading (so the segments still spell the whole text).

pub mod chinese;
pub mod japanese;
pub mod romaji;
#[cfg(test)]
mod tests;

use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime,
};

use crate::services::model_store;

/// Plugin name (`plugin:text-readings|...`, capability `text-readings:default`).
pub const PLUGIN_NAME: &str = "text-readings";
/// Longest text one request may carry (characters).
pub const MAX_TEXT_CHARS: usize = 2_000;
/// Longest reading string in a result (UTF-16 units, the TS schema's limit).
pub const MAX_READING_UNITS: usize = 4_000;
/// Most ruby segments in a result (the TS schema's limit).
pub const MAX_SEGMENTS: usize = 512;

const LOG: &str = "sigil::text_readings";

/// How pinyin shows tones (TS `RustPinyinStyle`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PinyinStyle {
    /// `nǐ hǎo`
    #[default]
    Marks,
    /// `ni3 hao3` (neutral tone `5`: `ma5`)
    Numbers,
    /// `ni hao`
    None,
}

/// What to annotate (TS `RustReadingsRequest`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ReadingsRequest {
    pub text: String,
    /// Lower-case base language of `text`: `ja`, `zh` (also `yue` / `cmn` / `zh-*`, read as Mandarin).
    pub lang: String,
    /// Japanese: hiragana reading.
    pub hiragana: bool,
    /// Japanese: Hepburn romaji.
    pub romaji: bool,
    /// Chinese: pinyin.
    pub pinyin: bool,
    pub pinyin_style: PinyinStyle,
}

/// One piece of the text with its reading (ruby text); `reading` is `None` where it equals the
/// text (kana, Latin, punctuation).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ReadingSegment {
    pub text: String,
    pub reading: Option<String>,
}

/// Readings of one text (TS `RustReadingsResult`): only the requested forms are `Some`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ReadingsResult {
    /// Language the text was read as (`ja` or `zh`).
    pub lang: String,
    pub hiragana: Option<String>,
    pub romaji: Option<String>,
    pub pinyin: Option<String>,
    /// Ruby segments in the requested script (hiragana for Japanese, romaji when only romaji is
    /// requested; pinyin for Chinese). They spell the whole text; empty when no form is requested.
    pub segments: Vec<ReadingSegment>,
}

/// What can be read right now (TS `RustReadingsStatus`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadingsStatus {
    /// The Japanese dictionary is installed.
    pub japanese: bool,
    /// Chinese readings work (bundled data).
    pub chinese: bool,
    /// Catalog id of the Japanese dictionary (for `<ModelRequirement>`).
    pub japanese_model: Option<String>,
    pub error: Option<String>,
}

/// The language a request is read as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadingLang {
    Japanese,
    Chinese,
}

impl ReadingLang {
    /// `ja` / `jpn` (with any region) → Japanese; `zh` / `zh-*` / `cmn` / `yue` / `zho` / `chi` →
    /// Chinese (read as Mandarin).
    pub(crate) fn parse(lang: &str) -> Option<Self> {
        let lang = lang.trim().to_ascii_lowercase().replace('_', "-");
        let base = lang.split('-').next().unwrap_or_default();
        match base {
            "ja" | "jpn" => Some(Self::Japanese),
            "zh" | "cmn" | "yue" | "zho" | "chi" => Some(Self::Chinese),
            _ => None,
        }
    }

    fn code(self) -> &'static str {
        match self {
            Self::Japanese => "ja",
            Self::Chinese => "zh",
        }
    }
}

/// Cut `s` to at most [`MAX_READING_UNITS`] UTF-16 units, ending in `…` when cut.
pub(crate) fn clamp_reading(s: String) -> String {
    if s.encode_utf16().count() <= MAX_READING_UNITS {
        return s;
    }
    let mut out = String::with_capacity(s.len());
    let mut units = 0;
    for c in s.chars() {
        if units + c.len_utf16() > MAX_READING_UNITS - 1 {
            break;
        }
        units += c.len_utf16();
        out.push(c);
    }
    out.push('…');
    out
}

/// Merge neighbouring segments without a reading, then cap the count at [`MAX_SEGMENTS`] (the
/// tail is merged into one segment without a reading). Empty segments are dropped.
pub(crate) fn finish_segments(segments: Vec<ReadingSegment>) -> Vec<ReadingSegment> {
    let mut out: Vec<ReadingSegment> = Vec::with_capacity(segments.len());
    for seg in segments {
        if seg.text.is_empty() {
            continue;
        }
        let reading = seg.reading.filter(|r| !r.is_empty() && *r != seg.text);
        match (out.last_mut(), &reading) {
            (Some(last), None) if last.reading.is_none() => last.text.push_str(&seg.text),
            _ => out.push(ReadingSegment { text: seg.text, reading }),
        }
    }
    if out.len() > MAX_SEGMENTS {
        let tail: String = out.drain(MAX_SEGMENTS - 1..).map(|s| s.text).collect();
        out.push(ReadingSegment { text: tail, reading: None });
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Readers (lazily loaded, shared)
// ---------------------------------------------------------------------------------------------

/// The loaded Japanese analyzer and the folder it came from.
struct LoadedJapanese {
    dir: PathBuf,
    analyzer: Arc<japanese::JapaneseAnalyzer>,
}

/// A dictionary folder that failed to load, and when (not retried for [`LOAD_RETRY_AFTER`]).
struct FailedLoad {
    dir: PathBuf,
    at: Instant,
    error: String,
}

/// How long a dictionary that failed to load is not tried again (a broken install would otherwise
/// be re-read for every caption line).
const LOAD_RETRY_AFTER: std::time::Duration = std::time::Duration::from_secs(30);

#[derive(Default)]
struct Readers {
    japanese: Mutex<Option<LoadedJapanese>>,
    chinese: Mutex<Option<Arc<chinese::ChineseReader>>>,
    /// Last dictionary load error (cleared by a successful load).
    error: Mutex<Option<String>>,
    failed: Mutex<Option<FailedLoad>>,
}

fn readers() -> &'static Readers {
    static READERS: OnceLock<Readers> = OnceLock::new();
    READERS.get_or_init(Readers::default)
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// The installed dictionary folder, if any.
fn installed_dictionary() -> Option<PathBuf> {
    model_store::installed_path(japanese::DICTIONARY_ID).and_then(|p| japanese::dictionary_dir(&p))
}

/// The Japanese analyzer, loading (or reloading after a reinstall) the installed dictionary.
fn japanese_analyzer() -> Result<Arc<japanese::JapaneseAnalyzer>, String> {
    let r = readers();
    let dir = installed_dictionary().ok_or_else(|| {
        format!(
            "The Japanese dictionary ({}) is not installed; install it from the Models tab or the readings settings",
            japanese::DICTIONARY_ID
        )
    })?;
    let mut slot = lock(&r.japanese);
    if let Some(loaded) = slot.as_ref().filter(|l| l.dir == dir) {
        return Ok(Arc::clone(&loaded.analyzer));
    }
    if let Some(failed) = lock(&r.failed).as_ref().filter(|f| f.dir == dir && f.at.elapsed() < LOAD_RETRY_AFTER) {
        return Err(failed.error.clone());
    }
    let started = Instant::now();
    match japanese::JapaneseAnalyzer::load(&dir) {
        Ok(analyzer) => {
            let analyzer = Arc::new(analyzer);
            tracing::info!(target: LOG, "Japanese dictionary loaded from {} in {} ms", dir.display(), started.elapsed().as_millis());
            *slot = Some(LoadedJapanese {
                dir,
                analyzer: Arc::clone(&analyzer),
            });
            *lock(&r.error) = None;
            *lock(&r.failed) = None;
            Ok(analyzer)
        }
        Err(e) => {
            let e = format!("The Japanese dictionary ({}) failed to load: {e}", japanese::DICTIONARY_ID);
            tracing::warn!(target: LOG, "{e}");
            *lock(&r.error) = Some(e.clone());
            *lock(&r.failed) = Some(FailedLoad {
                dir,
                at: Instant::now(),
                error: e.clone(),
            });
            Err(e)
        }
    }
}

/// The Chinese reader (jieba's bundled dictionary, built once on first use).
fn chinese_reader() -> Arc<chinese::ChineseReader> {
    let mut slot = lock(&readers().chinese);
    if let Some(reader) = slot.as_ref() {
        return Arc::clone(reader);
    }
    let started = Instant::now();
    let reader = Arc::new(chinese::ChineseReader::new());
    tracing::info!(target: LOG, "Chinese segmenter loaded in {} ms", started.elapsed().as_millis());
    *slot = Some(Arc::clone(&reader));
    reader
}

/// Annotate one request (blocking; dictionaries load on first use).
pub(crate) fn annotate(request: &ReadingsRequest) -> Result<ReadingsResult, String> {
    let lang = ReadingLang::parse(&request.lang)
        .ok_or_else(|| format!("Readings are available for Japanese and Chinese, not '{}'", request.lang))?;
    let chars = request.text.chars().count();
    if chars > MAX_TEXT_CHARS {
        return Err(format!("The text is too long for readings ({chars} characters, at most {MAX_TEXT_CHARS})"));
    }
    let mut result = ReadingsResult {
        lang: lang.code().to_string(),
        ..ReadingsResult::default()
    };
    match lang {
        ReadingLang::Japanese if request.hiragana || request.romaji => {
            if request.text.trim().is_empty() {
                return Ok(empty_forms(result, request, lang));
            }
            let analyzer = japanese_analyzer()?;
            let morphs = analyzer.morphs(&request.text)?;
            let readings = japanese::read(&request.text, &morphs, request.hiragana, request.romaji);
            result.hiragana = readings.hiragana.map(clamp_reading);
            result.romaji = readings.romaji.map(clamp_reading);
            result.segments = finish_segments(readings.segments);
        }
        ReadingLang::Chinese if request.pinyin => {
            if request.text.trim().is_empty() {
                return Ok(empty_forms(result, request, lang));
            }
            let readings = chinese_reader().read(&request.text, request.pinyin_style);
            result.pinyin = Some(clamp_reading(readings.pinyin));
            result.segments = finish_segments(readings.segments);
        }
        _ => {}
    }
    Ok(result)
}

/// The requested forms of an empty text (all empty strings).
fn empty_forms(mut result: ReadingsResult, request: &ReadingsRequest, lang: ReadingLang) -> ReadingsResult {
    match lang {
        ReadingLang::Japanese => {
            result.hiragana = request.hiragana.then(String::new);
            result.romaji = request.romaji.then(String::new);
        }
        ReadingLang::Chinese => result.pinyin = request.pinyin.then(String::new),
    }
    if !request.text.is_empty() {
        result.segments = vec![ReadingSegment {
            text: request.text.clone(),
            reading: None,
        }];
    }
    result
}

fn status_now() -> ReadingsStatus {
    ReadingsStatus {
        japanese: installed_dictionary().is_some(),
        chinese: true,
        japanese_model: Some(japanese::DICTIONARY_ID.to_string()),
        error: lock(&readers().error).clone(),
    }
}

#[command]
async fn text_readings_annotate(request: ReadingsRequest) -> Result<ReadingsResult, String> {
    tauri::async_runtime::spawn_blocking(move || annotate(&request))
        .await
        .map_err(|e| format!("Readings failed: {e}"))?
}

#[command]
async fn text_readings_status() -> Result<ReadingsStatus, String> {
    tauri::async_runtime::spawn_blocking(status_now)
        .await
        .map_err(|e| format!("Readings status failed: {e}"))
}

#[command]
async fn text_readings_unload() -> Result<(), String> {
    let r = readers();
    let japanese = lock(&r.japanese).take();
    let chinese = lock(&r.chinese).take();
    *lock(&r.failed) = None;
    if japanese.is_some() || chinese.is_some() {
        tracing::info!(target: LOG, "readings dictionaries unloaded");
    }
    // Dropping a dictionary frees tens of MB: do it off the async runtime.
    tauri::async_runtime::spawn_blocking(move || drop((japanese, chinese)))
        .await
        .map_err(|e| format!("Readings unload failed: {e}"))
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new(PLUGIN_NAME)
        .invoke_handler(tauri::generate_handler![
            text_readings_annotate,
            text_readings_status,
            text_readings_unload,
        ])
        .build()
}
