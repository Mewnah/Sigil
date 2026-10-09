//! The inputs of a speech-bench run that live on disk: the audio corpus (`sigil-speech-corpus/1`,
//! built by `scripts/speech-bench/corpus/build_corpus.py`), the MT sets (`mt_line` JSON lines) and
//! the TTS prompts (`tts_prompt` JSON lines). Reading only: nothing here writes or downloads.
//!
//! Item selection ([`Manifest::select`]) follows the spec's `select` block: `ids` override
//! everything; otherwise `kinds`, `languages` and `limit_per_language` (the first k utterances of a
//! language by `rank`, the same k under every condition so comparisons stay paired) apply first,
//! then `subset` / `conditions`. The result keeps the manifest's order.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;

use super::spec::{hex_digest, ItemKind, Select};

/// A corpus item (the members the runner needs; the manifest carries more).
#[derive(Clone, Debug, Deserialize)]
pub struct CorpusItem {
    pub id: String,
    pub utterance: String,
    pub kind: ItemKind,
    pub language: String,
    pub condition: String,
    /// Relative to the manifest's folder.
    pub audio: String,
    pub seconds: f64,
    #[serde(default)]
    pub rank: u64,
}

#[derive(Deserialize)]
struct RawManifest {
    schema: String,
    name: String,
    content_sha256: String,
    #[serde(default)]
    recipe_sha256: Option<String>,
    #[serde(default)]
    sample_rate: Option<u32>,
    #[serde(default)]
    subsets: BTreeMap<String, Vec<String>>,
    items: Vec<CorpusItem>,
}

/// A loaded corpus manifest.
#[derive(Clone, Debug)]
pub struct Manifest {
    pub name: String,
    pub content_sha256: String,
    pub recipe_sha256: Option<String>,
    /// sha256 of the `corpus.json` file read.
    pub manifest_sha256: String,
    pub subsets: BTreeMap<String, Vec<String>>,
    pub items: Vec<CorpusItem>,
    /// The manifest's folder (audio paths are relative to it).
    pub dir: PathBuf,
}

impl Manifest {
    /// Read `corpus.json`.
    pub fn load(path: &Path) -> Result<Manifest, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read the corpus manifest {}: {e}", path.display()))?;
        let mut m = Manifest::from_bytes(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        m.dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
        Ok(m)
    }

    /// Parse manifest bytes (the folder stays empty until [`load`](Self::load) sets it).
    pub fn from_bytes(bytes: &[u8]) -> Result<Manifest, String> {
        let raw: RawManifest = serde_json::from_slice(bytes).map_err(|e| format!("not a corpus manifest: {e}"))?;
        if raw.schema != "sigil-speech-corpus/1" {
            return Err(format!("unsupported corpus schema {:?} (want sigil-speech-corpus/1)", raw.schema));
        }
        if raw.sample_rate.is_some_and(|r| r != 16_000) {
            return Err("the corpus sample rate must be 16000".into());
        }
        let mut seen = HashSet::new();
        for it in &raw.items {
            if !seen.insert(it.id.as_str()) {
                return Err(format!("the corpus lists item {} twice", it.id));
            }
        }
        Ok(Manifest {
            name: raw.name,
            content_sha256: raw.content_sha256,
            recipe_sha256: raw.recipe_sha256,
            manifest_sha256: hex_digest(bytes),
            subsets: raw.subsets,
            items: raw.items,
            dir: PathBuf::new(),
        })
    }

    /// The items `sel` picks.
    pub fn select(&self, sel: &Select) -> Result<Vec<&CorpusItem>, String> {
        if !sel.ids.is_empty() {
            let by_id: HashMap<&str, &CorpusItem> = self.items.iter().map(|i| (i.id.as_str(), i)).collect();
            return sel
                .ids
                .iter()
                .map(|id| by_id.get(id.as_str()).copied().ok_or_else(|| format!("select.ids: the corpus has no item {id:?}")))
                .collect();
        }
        let subset: Option<&Vec<String>> = match &sel.subset {
            Some(name) => Some(self.subsets.get(name).ok_or_else(|| {
                format!(
                    "select.subset {name:?} is not in the corpus (subsets: {})",
                    self.subsets.keys().cloned().collect::<Vec<_>>().join(", ")
                )
            })?),
            None => None,
        };
        let kind_ok = |i: &CorpusItem| sel.kinds.is_empty() || sel.kinds.contains(&i.kind);
        let lang_ok = |i: &CorpusItem| sel.languages.is_empty() || sel.languages.iter().any(|l| l.eq_ignore_ascii_case(&i.language));
        let base: Vec<&CorpusItem> = self.items.iter().filter(|i| kind_ok(i) && lang_ok(i)).collect();
        // limit_per_language: the first k utterances by rank in each language.
        let allowed: Option<HashSet<&str>> = sel.limit_per_language.map(|k| {
            let mut per_lang: BTreeMap<&str, BTreeMap<(u64, usize), &str>> = BTreeMap::new();
            for (n, i) in base.iter().enumerate() {
                per_lang.entry(i.language.as_str()).or_default().entry((i.rank, n)).or_insert(i.utterance.as_str());
            }
            let mut keep = HashSet::new();
            for utterances in per_lang.values() {
                let mut taken = BTreeSet::new();
                for u in utterances.values() {
                    if taken.len() >= k {
                        break;
                    }
                    taken.insert(*u);
                }
                keep.extend(taken);
            }
            keep
        });
        Ok(base
            .into_iter()
            .filter(|i| allowed.as_ref().is_none_or(|a| a.contains(i.utterance.as_str())))
            .filter(|i| subset.is_none_or(|s| s.iter().any(|c| c == &i.condition)))
            .filter(|i| sel.conditions.is_empty() || sel.conditions.iter().any(|c| c == &i.condition))
            .collect())
    }

    /// The audio of `item` as 16 kHz mono f32.
    pub fn load_audio(&self, item: &CorpusItem) -> Result<Vec<f32>, String> {
        let rel = Path::new(&item.audio);
        if rel.is_absolute() || rel.components().any(|c| matches!(c, Component::ParentDir | Component::Prefix(_) | Component::RootDir)) {
            return Err(format!("{}: the audio path {:?} leaves the corpus folder", item.id, item.audio));
        }
        read_audio_16k(&self.dir.join(rel))
    }
}

/// A WAV file as 16 kHz mono f32 (any format, channels averaged, resampled when needed).
pub fn read_audio_16k(path: &Path) -> Result<Vec<f32>, String> {
    let (mono, rate) = crate::services::speech_rt::sherpa::probe::read_wav(path)?;
    if rate == 16_000 {
        Ok(mono)
    } else {
        super::super::signals::resample(&mono, rate, 16_000)
    }
}

// ---------------------------------------------------------------------------------------------
// MT sets
// ---------------------------------------------------------------------------------------------

/// A line of an MT set (`mt_line`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MtLine {
    pub id: String,
    pub source: String,
    pub target: String,
    pub src: String,
    pub refs: Vec<String>,
    /// `<document>#<segment>`; the previous line of the same document is the context.
    pub doc: Option<String>,
}

impl MtLine {
    pub fn direction(&self) -> String {
        format!("{}-{}", self.source, self.target)
    }

    /// The document part of `doc` (without the `#segment` suffix).
    pub fn doc_key(&self) -> Option<&str> {
        self.doc.as_deref().map(|d| d.rsplit_once('#').map_or(d, |(k, _)| k))
    }
}

#[derive(Deserialize)]
struct RawMtLine {
    id: Option<String>,
    source: String,
    target: String,
    src: String,
    #[serde(rename = "ref")]
    reference: Option<String>,
    #[serde(default)]
    refs: Vec<String>,
    #[serde(default)]
    doc: Option<String>,
}

/// Parse a JSON-lines MT set (blank lines and `#` comments skipped). A line without `id` gets
/// `<source>-<target>:<n>`, as mt-bench does.
pub fn parse_mt_lines(text: &str) -> Result<Vec<MtLine>, String> {
    let mut out = Vec::new();
    let mut counters: HashMap<String, usize> = HashMap::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let raw: RawMtLine = serde_json::from_str(line).map_err(|e| format!("MT set line {}: {e}", n + 1))?;
        let mut refs = raw.refs;
        refs.extend(raw.reference);
        refs.retain(|r| !r.trim().is_empty());
        if raw.src.trim().is_empty() {
            return Err(format!("MT set line {}: empty \"src\"", n + 1));
        }
        let direction = format!("{}-{}", raw.source.trim(), raw.target.trim());
        let count = counters.entry(direction.clone()).or_insert(0);
        *count += 1;
        out.push(MtLine {
            id: raw.id.unwrap_or_else(|| format!("{direction}:{count}")),
            source: raw.source.trim().to_string(),
            target: raw.target.trim().to_string(),
            src: raw.src,
            refs,
            doc: raw.doc,
        });
    }
    let mut seen = HashSet::new();
    for l in &out {
        if !seen.insert(l.id.as_str()) {
            return Err(format!("MT set: duplicate id {}", l.id));
        }
    }
    Ok(out)
}

/// Lines of the wanted `directions` (all when empty), at most `limit` per direction, in file order;
/// directions in first-seen order.
pub fn select_mt(lines: &[MtLine], directions: &[String], limit: Option<usize>) -> Vec<(String, Vec<MtLine>)> {
    let mut groups: Vec<(String, Vec<MtLine>)> = Vec::new();
    for line in lines {
        let dir = line.direction();
        if !directions.is_empty() && !directions.contains(&dir) {
            continue;
        }
        let idx = match groups.iter().position(|(d, _)| *d == dir) {
            Some(i) => i,
            None => {
                groups.push((dir, Vec::new()));
                groups.len() - 1
            }
        };
        if limit.is_none_or(|l| groups[idx].1.len() < l) {
            groups[idx].1.push(line.clone());
        }
    }
    groups
}

/// The context line of `line` for `context: previous`: the nearest earlier line of the same direction
/// and document (`all` is the whole set in file order).
pub fn previous_in_doc<'a>(all: &'a [MtLine], line: &MtLine) -> Option<&'a MtLine> {
    let key = line.doc_key()?;
    let dir = line.direction();
    let pos = all.iter().position(|l| l.id == line.id)?;
    all[..pos].iter().rev().find(|l| l.doc_key() == Some(key) && l.direction() == dir)
}

// ---------------------------------------------------------------------------------------------
// TTS prompts
// ---------------------------------------------------------------------------------------------

/// A TTS prompt (`tts_prompt`).
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct TtsPrompt {
    /// `<lang>-<kind>-<n>`: also the rendering's file name.
    pub id: String,
    pub lang: String,
    pub text: String,
    #[serde(default)]
    pub kind: String,
    /// Preset voice override for every engine.
    #[serde(default)]
    pub voice: Option<String>,
    /// Voice-library id of a reference clip for cloning engines.
    #[serde(default)]
    pub reference_voice: Option<String>,
}

/// Parse a JSON-lines prompt file.
pub fn parse_prompts(text: &str) -> Result<Vec<TtsPrompt>, String> {
    let mut out: Vec<TtsPrompt> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let p: TtsPrompt = serde_json::from_str(line).map_err(|e| format!("prompt line {}: {e}", n + 1))?;
        if p.text.trim().is_empty() {
            return Err(format!("prompt line {}: empty text", n + 1));
        }
        if !p.id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')) || p.id.is_empty() {
            return Err(format!("prompt line {}: id {:?} must be a plain file name stem", n + 1, p.id));
        }
        out.push(p);
    }
    let mut seen = HashSet::new();
    for p in &out {
        if !seen.insert(p.id.as_str()) {
            return Err(format!("prompts: duplicate id {}", p.id));
        }
    }
    Ok(out)
}

/// At most `limit` prompts per language (file order) among the languages in `langs` (all when empty).
pub fn select_prompts(prompts: &[TtsPrompt], langs: &[String], limit: Option<usize>) -> Vec<TtsPrompt> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut out = Vec::new();
    for p in prompts {
        if !langs.is_empty() && !langs.iter().any(|l| l.eq_ignore_ascii_case(&p.lang)) {
            continue;
        }
        let n = counts.entry(p.lang.as_str()).or_insert(0);
        if limit.is_some_and(|l| *n >= l) {
            continue;
        }
        *n += 1;
        out.push(p.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Manifest {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../scripts/speech-bench/tests/fixtures/manifest.small.json");
        Manifest::load(&path).expect("fixture manifest")
    }

    fn ids(items: &[&CorpusItem]) -> Vec<String> {
        items.iter().map(|i| i.id.clone()).collect()
    }

    #[test]
    fn the_fixture_manifest_loads_and_selects() {
        let m = fixture();
        assert_eq!(m.name, "fixture");
        assert_eq!(m.content_sha256, "248779555a2115cf7aa25e3b910f49196f07aa9fdbbd5594818062fd17014b5b");
        assert_eq!(m.items.len(), 9);
        assert_eq!(m.manifest_sha256.len(), 64);
        let all = m.select(&Select::default()).unwrap();
        assert_eq!(all.len(), 9);
        let quick = m.select(&Select { subset: Some("quick".into()), ..Select::default() }).unwrap();
        assert_eq!(quick.len(), 8, "the real clip is not in the quick subset");
        let synthetic_ja = m
            .select(&Select { kinds: vec![ItemKind::Synthetic], languages: vec!["ja".into()], ..Select::default() })
            .unwrap();
        assert_eq!(ids(&synthetic_ja), ["ja/2001/clean", "ja/2001/music_snr10", "ja/2002/clean", "ja/2002/music_snr10"]);
        let real = m.select(&Select { kinds: vec![ItemKind::Real], ..Select::default() }).unwrap();
        assert_eq!(ids(&real), ["real/2026-10-05-alice/clip01"]);
        let cond = m.select(&Select { conditions: vec!["music_snr10".into()], ..Select::default() }).unwrap();
        assert_eq!(cond.len(), 4);
    }

    #[test]
    fn limit_per_language_takes_the_same_utterances_under_every_condition() {
        let m = fixture();
        let one = m
            .select(&Select { subset: Some("quick".into()), limit_per_language: Some(1), ..Select::default() })
            .unwrap();
        assert_eq!(ids(&one), ["en/1001/clean", "en/1001/music_snr10", "ja/2001/clean", "ja/2001/music_snr10"]);
        // Applied before the conditions: one utterance, one condition each.
        let clean = m
            .select(&Select { conditions: vec!["clean".into()], limit_per_language: Some(1), ..Select::default() })
            .unwrap();
        assert_eq!(ids(&clean), ["en/1001/clean", "ja/2001/clean"]);
    }

    #[test]
    fn explicit_ids_override_the_filters_and_unknown_names_are_errors() {
        let m = fixture();
        let by_id = m
            .select(&Select {
                ids: vec!["ja/2002/clean".into(), "en/1001/clean".into()],
                languages: vec!["fr".into()],
                ..Select::default()
            })
            .unwrap();
        assert_eq!(ids(&by_id), ["ja/2002/clean", "en/1001/clean"]);
        assert!(m.select(&Select { ids: vec!["nope".into()], ..Select::default() }).unwrap_err().contains("no item"));
        let err = m.select(&Select { subset: Some("nope".into()), ..Select::default() }).unwrap_err();
        assert!(err.contains("quick") && err.contains("real"), "{err}");
    }

    #[test]
    fn audio_paths_cannot_leave_the_corpus_folder() {
        let m = fixture();
        let mut item = m.items[0].clone();
        item.audio = "../secret.wav".into();
        assert!(m.load_audio(&item).unwrap_err().contains("leaves the corpus folder"));
        item.audio = "C:/x.wav".into();
        assert!(m.load_audio(&item).is_err());
        item.audio = "audio/en/clean/missing.wav".into();
        assert!(m.load_audio(&item).unwrap_err().contains("cannot read"));
    }

    #[test]
    fn manifests_are_checked() {
        assert!(Manifest::from_bytes(b"{").is_err());
        assert!(Manifest::from_bytes(br#"{"schema":"other/1","name":"x","content_sha256":"0","items":[]}"#)
            .unwrap_err()
            .contains("unsupported"));
        let dup = br#"{"schema":"sigil-speech-corpus/1","name":"x","content_sha256":"0","items":[
            {"id":"a","utterance":"u","kind":"synthetic","language":"en","condition":"clean","audio":"a.wav","seconds":1.0},
            {"id":"a","utterance":"u","kind":"synthetic","language":"en","condition":"clean","audio":"a.wav","seconds":1.0}]}"#;
        assert!(Manifest::from_bytes(dup).unwrap_err().contains("twice"));
    }

    #[test]
    fn mt_sets_parse_select_and_find_their_context() {
        let text = r#"
# comment
{"id":"a:en-ja:1","source":"en","target":"ja","src":"One","ref":"いち","doc":"d1#1"}
{"id":"a:en-ja:2","source":"en","target":"ja","src":"Two","refs":["に","ふた"],"doc":"d1#2"}
{"id":"a:en-ja:3","source":"en","target":"ja","src":"Three","ref":"さん","doc":"d2#1"}
{"source":"ja","target":"en","src":"はい","ref":"Yes"}
{"id":"a:en-de:1","source":"en","target":"de","src":"Four","ref":"Vier","doc":"d1#3"}
"#;
        let lines = parse_mt_lines(text).unwrap();
        assert_eq!(lines.len(), 5);
        assert_eq!(lines[1].refs, ["に", "ふた"]);
        assert_eq!(lines[3].id, "ja-en:1", "a line without id gets <direction>:<n>");
        let groups = select_mt(&lines, &[], Some(2));
        assert_eq!(groups.iter().map(|(d, l)| (d.as_str(), l.len())).collect::<Vec<_>>(), [("en-ja", 2), ("ja-en", 1), ("en-de", 1)]);
        let only = select_mt(&lines, &["en-de".into()], None);
        assert_eq!(only.len(), 1);
        assert_eq!(previous_in_doc(&lines, &lines[1]).map(|l| l.id.as_str()), Some("a:en-ja:1"));
        assert!(previous_in_doc(&lines, &lines[2]).is_none(), "another document");
        assert!(previous_in_doc(&lines, &lines[4]).is_none(), "another direction");
        assert!(previous_in_doc(&lines, &lines[0]).is_none());
        assert!(parse_mt_lines(r#"{"id":"x","source":"en","target":"ja","src":" ","ref":"y"}"#).is_err());
        let dup = "{\"id\":\"x\",\"source\":\"en\",\"target\":\"ja\",\"src\":\"a\",\"ref\":\"b\"}\n{\"id\":\"x\",\"source\":\"en\",\"target\":\"ja\",\"src\":\"a\",\"ref\":\"b\"}";
        assert!(parse_mt_lines(dup).unwrap_err().contains("duplicate"));
    }

    #[test]
    fn tts_prompts_parse_and_select_per_language() {
        let text = r#"
{"id":"en-casual-1","lang":"en","text":"Hello","kind":"casual"}
{"id":"en-casual-2","lang":"en","text":"Hi there","kind":"casual","reference_voice":"v1"}
{"id":"ja-casual-1","lang":"ja","text":"こんにちは","kind":"casual","voice":"jf_alpha"}
"#;
        let prompts = parse_prompts(text).unwrap();
        assert_eq!(prompts.len(), 3);
        assert_eq!(prompts[1].reference_voice.as_deref(), Some("v1"));
        let one = select_prompts(&prompts, &[], Some(1));
        assert_eq!(one.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(), ["en-casual-1", "ja-casual-1"]);
        let ja = select_prompts(&prompts, &["ja".into()], None);
        assert_eq!(ja.len(), 1);
        assert!(parse_prompts(r#"{"id":"../x","lang":"en","text":"a"}"#).is_err(), "ids become file names");
        assert!(parse_prompts(r#"{"id":"a","lang":"en","text":" "}"#).is_err());
        assert!(parse_prompts("{\"id\":\"a\",\"lang\":\"en\",\"text\":\"x\"}\n{\"id\":\"a\",\"lang\":\"en\",\"text\":\"y\"}").is_err());
    }
}
