//! `sigil-speech-bench-export/1`: the part of a report a friend sends back (frozen by P7-S0,
//! `scripts/speech-bench/schemas/speech-bench-export.schema.json`).
//!
//! [`make_export`] is the Rust twin of `scripts/speech-bench/score/make_export.py` (the reference
//! implementation; a contract test derives `examples/export.example.json` from
//! `examples/report.example.json` with both), and [`privacy_errors`] the twin of
//! `scripts/speech-bench/validate.py::privacy_errors`. The rules:
//! - no audio paths and no `files`;
//! - every absolute path inside a string becomes `<path>/<file name>`;
//! - a bring-your-own model (anything but a catalog id) is `byo`;
//! - host and user names never leave the environment block;
//! - item text (hypotheses, translations, back-transcriptions, segment text) is kept only when the
//!   spec's `output.export.include_text` is true, and never for real clips (ids `real/...`);
//! - `include_items = false` keeps aggregates and groups only.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{json, Map, Value};

use super::spec::is_catalog_id;

/// Schema id of the export.
pub const EXPORT_SCHEMA: &str = "sigil-speech-bench-export/1";

/// Absolute paths inside a string: a drive, a UNC prefix, or a Unix home / mount folder.
static ABS_PATH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?:[A-Za-z]:[\\/]|\\\\|/(?:Users|home|mnt|tmp)/)[^\s"'|;,]*"#).expect("path pattern"));
/// The privacy check's looser test: a string that starts like a path, or runs through a home folder.
static LOOKS_LIKE_PATH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"^(?:[A-Za-z]:[\\/]|\\\\|/(?:Users|home|mnt|tmp)/)|[\\/](?:Users|home)[\\/][^\\/]+[\\/]"#).expect("path pattern"));

/// Environment members that name the machine or the user.
const PRIVATE_ENV_KEYS: [&str; 6] = ["hostname", "host", "computer_name", "user", "username", "user_name"];

/// `text` with every absolute path replaced by `<path>/<file name>`.
pub fn scrub_str(text: &str) -> String {
    ABS_PATH
        .replace_all(text, |m: &regex::Captures<'_>| {
            let whole = m.get(0).map_or("", |m| m.as_str());
            let trimmed = whole.trim_end_matches(['\\', '/']);
            let name = trimmed.rsplit(['\\', '/']).next().unwrap_or("");
            format!("<path>/{name}")
        })
        .into_owned()
}

/// [`scrub_str`] over every string of `value`, recursively.
pub fn scrub(value: Value) -> Value {
    match value {
        Value::String(s) => Value::String(scrub_str(&s)),
        Value::Array(items) => Value::Array(items.into_iter().map(scrub).collect()),
        Value::Object(map) => Value::Object(map.into_iter().map(|(k, v)| (k, scrub(v))).collect()),
        other => other,
    }
}

/// A catalog id as is, `byo` for a path; empty stays empty.
pub fn model_label(model: &str) -> String {
    if model.is_empty() || is_catalog_id(model) {
        model.to_string()
    } else {
        "byo".to_string()
    }
}

fn str_of<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

/// The spec as the export summarizes it: engines (id, task, backend, model label, provider, variant,
/// threads) and runs (name, mode, load profile, external load text, feed kind); no paths.
pub fn spec_summary(spec: &Value) -> Value {
    let engines: Vec<Value> = spec
        .get("engines")
        .and_then(Value::as_array)
        .map(|a| a.as_slice())
        .unwrap_or(&[])
        .iter()
        .map(|e| {
            json!({
                "id": e.get("id").cloned().unwrap_or(Value::Null),
                "task": e.get("task").cloned().unwrap_or(Value::Null),
                "backend": e.get("backend").cloned().unwrap_or(Value::Null),
                "model": model_label(str_of(e, "model").unwrap_or("")),
                "execution_provider": e.get("execution_provider").cloned().unwrap_or_else(|| json!("auto")),
                "variant": e.get("variant").cloned().unwrap_or(Value::Null),
                "threads": e.get("threads").cloned().unwrap_or(Value::Null),
            })
        })
        .collect();
    let runs: Vec<Value> = spec
        .get("runs")
        .and_then(Value::as_array)
        .map(|a| a.as_slice())
        .unwrap_or(&[])
        .iter()
        .map(|r| {
            let mut m = Map::new();
            m.insert("name".into(), r.get("name").cloned().unwrap_or(Value::Null));
            m.insert("mode".into(), r.get("mode").cloned().unwrap_or(Value::Null));
            m.insert("load_profile".into(), r.get("load_profile").cloned().unwrap_or_else(|| json!("none")));
            if let Some(ext) = str_of(r, "external_load").filter(|s| !s.is_empty()) {
                m.insert("external_load".into(), json!(ext));
            }
            if let Some(kind) = r.get("feed").and_then(|f| f.get("kind")).filter(|k| !k.is_null()) {
                m.insert("feed".into(), kind.clone());
            }
            Value::Object(m)
        })
        .collect();
    json!({ "engines": engines, "runs": runs })
}

/// Derive the export of `report` (a report as a JSON value). `now_ms` stamps `export.created_unix_ms`.
pub fn make_export(report: &Value, include_text: bool, include_items: bool, note: Option<&str>, now_ms: u64) -> Value {
    let env: Map<String, Value> = report
        .get("environment")
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter(|(k, _)| !PRIVATE_ENV_KEYS.contains(&k.to_ascii_lowercase().as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
        .unwrap_or_default();
    let engines: Vec<Value> = report
        .get("engines")
        .and_then(Value::as_array)
        .map(|a| a.as_slice())
        .unwrap_or(&[])
        .iter()
        .cloned()
        .map(|mut e| {
            if let Some(obj) = e.as_object_mut() {
                let label = model_label(obj.get("model").and_then(Value::as_str).unwrap_or(""));
                obj.insert("model".into(), json!(label));
            }
            e
        })
        .collect();
    let runs: Vec<Value> = report
        .get("runs")
        .and_then(Value::as_array)
        .map(|a| a.as_slice())
        .unwrap_or(&[])
        .iter()
        .cloned()
        .map(|mut run| {
            if let Some(obj) = run.as_object_mut() {
                obj.remove("files");
                if let Some(results) = obj.get_mut("results").and_then(Value::as_array_mut) {
                    for res in results {
                        let Some(res) = res.as_object_mut() else { continue };
                        if !include_items {
                            res.insert("items".into(), json!([]));
                            continue;
                        }
                        if let Some(items) = res.get_mut("items").and_then(Value::as_array_mut) {
                            for it in items {
                                strip_item(it, include_text);
                            }
                        }
                    }
                }
            }
            run
        })
        .collect();
    let mut out = Map::new();
    out.insert("schema".into(), json!(EXPORT_SCHEMA));
    out.insert(
        "report_schema".into(),
        report.get("schema").cloned().unwrap_or_else(|| json!("sigil-speech-bench-report/1")),
    );
    out.insert("name".into(), report.get("name").cloned().unwrap_or(Value::Null));
    out.insert("description".into(), report.get("description").cloned().unwrap_or_else(|| json!("")));
    out.insert("spec_sha256".into(), report.get("spec_sha256").cloned().unwrap_or(Value::Null));
    out.insert("spec_summary".into(), spec_summary(report.get("spec").unwrap_or(&Value::Null)));
    out.insert("started_unix_ms".into(), report.get("started_unix_ms").cloned().unwrap_or(Value::Null));
    out.insert("duration_ms".into(), report.get("duration_ms").cloned().unwrap_or(Value::Null));
    out.insert("passed".into(), report.get("passed").cloned().unwrap_or(Value::Null));
    out.insert("failures".into(), report.get("failures").cloned().unwrap_or_else(|| json!([])));
    out.insert("environment".into(), Value::Object(env));
    out.insert("engines".into(), Value::Array(engines));
    out.insert("runs".into(), Value::Array(runs));
    let mut meta = Map::new();
    meta.insert("created_unix_ms".into(), json!(now_ms));
    meta.insert("include_text".into(), json!(include_text));
    meta.insert("include_items".into(), json!(include_items));
    if let Some(n) = note.filter(|n| !n.is_empty()) {
        meta.insert("note".into(), json!(n));
    }
    out.insert("export".into(), Value::Object(meta));
    if let Some(c) = report.get("corpus").filter(|c| !c.is_null() && c.as_object().is_none_or(|m| !m.is_empty())) {
        out.insert("corpus".into(), c.clone());
    }
    scrub(Value::Object(out))
}

/// Remove from one item what an export never carries: audio paths, and text unless it may stay.
fn strip_item(item: &mut Value, include_text: bool) {
    let Some(obj) = item.as_object_mut() else { return };
    let real = obj.get("item").and_then(Value::as_str).is_some_and(|id| id.starts_with("real/"));
    let keep = include_text && !real;
    if obj.contains_key("audio") {
        obj.insert("audio".into(), Value::Null);
    }
    for key in ["text", "asr_text"] {
        if obj.contains_key(key) && !keep {
            obj.insert(key.into(), Value::Null);
        }
    }
    if !keep {
        if let Some(segments) = obj.get_mut("segments").and_then(Value::as_array_mut) {
            for seg in segments {
                if let Some(s) = seg.as_object_mut() {
                    s.insert("text".into(), json!(""));
                }
            }
        }
    }
}

fn walk_strings(value: &Value, path: &str, out: &mut Vec<(String, String)>) {
    match value {
        Value::Object(map) => map.iter().for_each(|(k, v)| walk_strings(v, &format!("{path}/{k}"), out)),
        Value::Array(items) => items.iter().enumerate().for_each(|(i, v)| walk_strings(v, &format!("{path}/{i}"), out)),
        Value::String(s) => out.push((path.to_string(), s.clone())),
        _ => {}
    }
}

/// The export rules, checked on an export document; empty when it is clean.
pub fn privacy_errors(export: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let mut strings = Vec::new();
    walk_strings(export, "", &mut strings);
    for (path, s) in &strings {
        if LOOKS_LIKE_PATH.is_match(s) {
            out.push(format!("{path}: looks like a local path ({:?})", s.chars().take(60).collect::<String>()));
        }
    }
    if let Some(env) = export.get("environment").and_then(Value::as_object) {
        for k in env.keys() {
            if PRIVATE_ENV_KEYS.contains(&k.to_ascii_lowercase().as_str()) {
                out.push(format!("environment/{k}: host or user names never go into an export"));
            }
        }
    }
    let include_text = export.get("export").and_then(|e| e.get("include_text")).and_then(Value::as_bool).unwrap_or(false);
    let runs = export.get("runs").and_then(Value::as_array).map(|a| a.as_slice()).unwrap_or(&[]);
    for (ri, run) in runs.iter().enumerate() {
        if run.get("files").is_some_and(|f| f.as_array().is_none_or(|a| !a.is_empty())) {
            out.push(format!("runs/{ri}/files: an export lists no files"));
        }
        let results = run.get("results").and_then(Value::as_array).map(|a| a.as_slice()).unwrap_or(&[]);
        for (si, res) in results.iter().enumerate() {
            let items = res.get("items").and_then(Value::as_array).map(|a| a.as_slice()).unwrap_or(&[]);
            for (ii, it) in items.iter().enumerate() {
                let at = format!("runs/{ri}/results/{si}/items/{ii}");
                if it.get("audio").is_some_and(|a| !a.is_null()) {
                    out.push(format!("{at}/audio: an export carries no audio"));
                }
                let real = it.get("item").and_then(Value::as_str).is_some_and(|id| id.starts_with("real/"));
                for key in ["text", "asr_text"] {
                    if it.get(key).is_some_and(|t| !t.is_null()) {
                        if real {
                            out.push(format!("{at}/{key}: text of a real clip never goes into an export"));
                        } else if !include_text {
                            out.push(format!("{at}/{key}: text present although export.include_text is false"));
                        }
                    }
                }
                let segment_text = it
                    .get("segments")
                    .and_then(Value::as_array)
                    .is_some_and(|segs| segs.iter().any(|s| s.get("text").and_then(Value::as_str).is_some_and(|t| !t.is_empty())));
                if segment_text && !include_text {
                    out.push(format!("{at}/segments: text present although export.include_text is false"));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> Value {
        json!({
            "schema": "sigil-speech-bench-report/1",
            "name": "t",
            "description": "",
            "spec_sha256": "0".repeat(64),
            "spec": {
                "engines": [
                    {"id": "a", "task": "stt", "backend": "whisper", "model": "whisper/small-q5_1"},
                    {"id": "b", "task": "stt", "backend": "sherpa-offline", "model": "C:\\Users\\friend\\models\\x", "threads": 2}
                ],
                "runs": [{"name": "r", "mode": "accuracy", "feed": {"kind": "utterance"}, "external_load": ""}]
            },
            "started_unix_ms": 1, "duration_ms": 2.0, "passed": true, "failures": ["C:\\Users\\friend\\x.wav failed"],
            "environment": {"sigil_version": "1", "hostname": "MYPC", "User": "friend", "cpu": "x"},
            "engines": [{"id": "b", "model": "C:\\Users\\friend\\models\\x", "error": "cannot open D:\\data\\a.gguf: no such file"}],
            "runs": [{
                "name": "r", "mode": "accuracy", "files": ["hyps/r.b.jsonl"],
                "results": [{"engine": "b", "items": [
                    {"item": "en/1/clean", "text": "hello world", "language": "en", "segments": [{"text": "hello", "start_ms": 0, "end_ms": 1}], "error": null},
                    {"item": "real/s1/c1", "text": "private words", "error": null},
                    {"item": "tts-1", "audio": "audio/b/tts-1.wav", "asr_text": "said", "error": null}
                ]}]
            }]
        })
    }

    #[test]
    fn paths_become_a_file_name_and_models_become_byo() {
        assert_eq!(scrub_str(r"failed at C:\Users\me\clips\a.wav now"), "failed at <path>/a.wav now");
        assert_eq!(scrub_str("opened /home/me/x.gguf, then /tmp/y"), "opened <path>/x.gguf, then <path>/y");
        assert_eq!(scrub_str(r"\\server\share\f.bin"), "<path>/f.bin");
        assert_eq!(scrub_str("no path here: a/b"), "no path here: a/b");
        assert_eq!(model_label("stt/qwen3-asr-0.6b-int8"), "stt/qwen3-asr-0.6b-int8");
        assert_eq!(model_label(r"C:\models\x.gguf"), "byo");
        assert_eq!(model_label(""), "");
        assert_eq!(model_label("whisper/small-q5_1"), "whisper/small-q5_1");
    }

    #[test]
    fn export_removes_audio_files_names_text_and_paths() {
        let exp = make_export(&report(), false, true, None, 5);
        assert_eq!(exp["schema"], EXPORT_SCHEMA);
        assert!(exp["environment"].get("hostname").is_none() && exp["environment"].get("User").is_none());
        assert_eq!(exp["environment"]["cpu"], "x");
        assert!(exp["runs"][0].get("files").is_none());
        let items = &exp["runs"][0]["results"][0]["items"];
        assert!(items[0]["text"].is_null());
        assert_eq!(items[0]["segments"][0]["text"], "");
        assert!(items[1]["text"].is_null());
        assert!(items[2]["audio"].is_null() && items[2]["asr_text"].is_null());
        assert_eq!(exp["engines"][0]["model"], "byo");
        assert_eq!(exp["engines"][0]["error"], "cannot open <path>/a.gguf: no such file");
        assert_eq!(exp["failures"][0], "<path>/x.wav failed");
        assert_eq!(exp["spec_summary"]["engines"][1]["model"], "byo");
        assert_eq!(exp["spec_summary"]["engines"][0]["execution_provider"], "auto");
        assert_eq!(exp["spec_summary"]["runs"][0], json!({"name": "r", "mode": "accuracy", "load_profile": "none", "feed": "utterance"}));
        assert_eq!(exp["export"], json!({"created_unix_ms": 5, "include_text": false, "include_items": true}));
        assert_eq!(privacy_errors(&exp), Vec::<String>::new());
    }

    #[test]
    fn text_stays_only_for_synthetic_items_when_the_spec_allows_it() {
        let exp = make_export(&report(), true, true, Some("rig: friend-3"), 5);
        let items = &exp["runs"][0]["results"][0]["items"];
        assert_eq!(items[0]["text"], "hello world");
        assert_eq!(items[0]["segments"][0]["text"], "hello");
        assert!(items[1]["text"].is_null(), "real clips never carry text");
        assert_eq!(items[2]["asr_text"], "said");
        assert!(items[2]["audio"].is_null(), "audio never goes");
        assert_eq!(exp["export"]["note"], "rig: friend-3");
        assert_eq!(privacy_errors(&exp), Vec::<String>::new());
    }

    #[test]
    fn without_items_only_aggregates_remain() {
        let exp = make_export(&report(), true, false, None, 5);
        assert_eq!(exp["runs"][0]["results"][0]["items"], json!([]));
        assert_eq!(exp["export"]["include_items"], false);
    }

    #[test]
    fn the_privacy_check_catches_every_leak() {
        let mut leaky = make_export(&report(), false, true, None, 5);
        assert!(privacy_errors(&leaky).is_empty());
        leaky["environment"]["hostname"] = json!("MYPC");
        leaky["runs"][0]["files"] = json!(["hyps/x.jsonl"]);
        leaky["runs"][0]["results"][0]["items"][0]["text"] = json!("leaked");
        leaky["runs"][0]["results"][0]["items"][1]["text"] = json!("leaked real");
        leaky["runs"][0]["results"][0]["items"][2]["audio"] = json!("audio/x.wav");
        leaky["runs"][0]["results"][0]["items"][0]["segments"][0]["text"] = json!("seg");
        leaky["failures"] = json!([r"C:\Users\me\a.wav", "ran from /home/me/bin/x"]);
        let errors = privacy_errors(&leaky).join("\n");
        for needle in [
            "host or user names",
            "lists no files",
            "text present although export.include_text is false",
            "text of a real clip never",
            "carries no audio",
            "segments: text present",
            "looks like a local path",
        ] {
            assert!(errors.contains(needle), "{needle} missing in:\n{errors}");
        }
    }
}
