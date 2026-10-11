//! `sigil-speech-bench-export/1`: the part of a report a friend sends back (frozen by P7-S0,
//! `scripts/speech-bench/schemas/speech-bench-export.schema.json`).
//!
//! [`make_export`] is the Rust twin of `scripts/speech-bench/score/make_export.py` (the reference
//! implementation; a contract test derives `examples/export.example.json` from
//! `examples/report.example.json` with both), and [`privacy_errors`] the twin of
//! `scripts/speech-bench/validate.py::privacy_errors`. The rules:
//! - no audio paths and no `files`;
//! - every absolute path inside a string becomes `<path>/<file name>` (folders may hold spaces, as in
//!   `C:\Users\Jane Doe\...` or `C:\Program Files\...`; a bare profile folder leaves no name at all);
//! - a bring-your-own model (anything but a catalog id) is `byo`;
//! - host and user names never leave the environment block;
//! - no names of other programs: a run state's other GPU users (`other_top_process`,
//!   `other_top_processes` and the run warning that lists them) shrink to a count / the share, and a
//!   power plan that is not one of Windows' built-in ones reads `custom`;
//! - item text (hypotheses, translations, back-transcriptions, segment text) is kept only when the
//!   spec's `output.export.include_text` is true, and never for real clips (ids `real/...`);
//! - `include_items = false` keeps aggregates and groups only.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{json, Map, Value};

use super::spec::is_catalog_id;

/// Schema id of the export.
pub const EXPORT_SCHEMA: &str = "sigil-speech-bench-export/1";

/// A user's profile folder: the drive form (`C:\Users\<name>`) and the Unix forms (`/Users/<name>`,
/// `/home/<name>`); the name may hold spaces. Group 1 is what follows the name when the path goes on
/// (a separator and one more character). A profile folder that is the path's end is replaced whole
/// (the user name must not survive as the "file name"); one that goes on is left to [`ABS_PATH`].
static PROFILE_PATH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?:\b[A-Za-z]:[\\/](?i:Users)[\\/]|/Users/|/home/)[^\\/:*?"'<>|\r\n;,]+(?:([\\/][^\s"'|;,\\/])|[\\/])?"#).expect("path pattern")
});
/// Absolute paths inside a string: a drive, a UNC prefix, or a Unix home / mount folder, then the
/// folders up to the last separator (a folder may hold spaces: `Jane Doe`, `Program Files (x86)`), then
/// the file name up to the next space or delimiter. The drive letter must start a word, so
/// `http://host/x` is not a path.
static ABS_PATH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?:\b[A-Za-z]:[\\/]|\\\\|/(?:Users|home|mnt|tmp)/)(?:[^\\/:*?"'<>|\r\n;,]*[\\/])*[^\s"'|;,]*"#).expect("path pattern")
});
/// The privacy check's looser test: a string that starts like a path, runs through a home folder,
/// still holds a Windows `\Users\` or `\AppData\` folder, or is a scrubbed `<path>/` that goes on
/// with a Windows folder (the sign of a path that was only partly replaced).
static LOOKS_LIKE_PATH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"^(?:[A-Za-z]:[\\/]|\\\\|/(?:Users|home|mnt|tmp)/)|[\\/](?:Users|home)[\\/][^\\/]+[\\/]|(?i:\\(?:Users|AppData)\\)|<path>/[^/\\]*\\"#).expect("path pattern")
});
/// The run warning that names the other GPU users (`bench::gpu_state::state_warnings`).
static OTHER_GPU_WARNING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^another process used the GPU \(up to ([0-9.]+) % SM").expect("warning pattern"));

/// Environment members that name the machine or the user.
const PRIVATE_ENV_KEYS: [&str; 6] = ["hostname", "host", "computer_name", "user", "username", "user_name"];

/// Windows' built-in power plans (the names an English Windows gives them). Any other name, an OEM's
/// or the user's own, is not exported.
const BUILTIN_POWER_PLANS: [&str; 4] = ["balanced", "high performance", "power saver", "ultimate performance"];

/// `text` with every absolute path replaced by `<path>/<file name>`.
pub fn scrub_str(text: &str) -> String {
    let profiles = PROFILE_PATH.replace_all(text, |m: &regex::Captures<'_>| {
        if m.get(1).is_some() {
            m.get(0).map_or("", |m| m.as_str()).to_string()
        } else {
            "<path>/".to_string()
        }
    });
    ABS_PATH
        .replace_all(&profiles, |m: &regex::Captures<'_>| {
            let whole = m.get(0).map_or("", |m| m.as_str());
            let trimmed = whole.trim_end_matches(['\\', '/']);
            let name = trimmed.rsplit(['\\', '/']).next().unwrap_or("");
            format!("<path>/{name}")
        })
        .into_owned()
}

/// A power plan name as the export keeps it: Windows' built-in plans as they are, anything else `custom`.
pub fn power_plan_label(name: &str) -> String {
    if BUILTIN_POWER_PLANS.contains(&name.trim().to_ascii_lowercase().as_str()) {
        name.to_string()
    } else {
        "custom".to_string()
    }
}

/// A run warning without the names of the programs that used the GPU.
pub fn redact_warning(warning: &str) -> String {
    match OTHER_GPU_WARNING.captures(warning) {
        Some(c) => format!("another process used the GPU (up to {} % SM)", &c[1]),
        None => warning.to_string(),
    }
}

/// One state sample without the other program's name and with a shareable power plan.
fn redact_sample(sample: &mut Value) {
    let Some(obj) = sample.as_object_mut() else { return };
    obj.remove("other_top_process");
    if let Some(plan) = obj.get("power_plan").and_then(Value::as_str).map(power_plan_label) {
        obj.insert("power_plan".into(), json!(plan));
    }
}

fn redact_warnings(warnings: &mut Value) {
    if let Some(list) = warnings.as_array_mut() {
        for w in list {
            if let Some(redacted) = w.as_str().map(redact_warning) {
                *w = json!(redacted);
            }
        }
    }
}

/// A run's `state` (`bench::gpu_state::RunState`) for the export: the names of the programs that
/// used the GPU become a count, power plans follow [`power_plan_label`].
fn redact_state(state: &mut Value) {
    let Some(obj) = state.as_object_mut() else { return };
    for key in ["start", "end"] {
        if let Some(sample) = obj.get_mut(key) {
            redact_sample(sample);
        }
    }
    if let Some(series) = obj.get_mut("series").and_then(Value::as_array_mut) {
        series.iter_mut().for_each(redact_sample);
    }
    if let Some(summary) = obj.get_mut("summary").and_then(Value::as_object_mut) {
        if let Some(Value::Array(names)) = summary.remove("other_top_processes") {
            summary.insert("other_top_processes_count".into(), json!(names.len()));
        }
    }
    if let Some(w) = obj.get_mut("warnings") {
        redact_warnings(w);
    }
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
                .map(|(k, v)| match (k.as_str(), v.as_str()) {
                    ("power_plan", Some(plan)) => (k.clone(), json!(power_plan_label(plan))),
                    _ => (k.clone(), v.clone()),
                })
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
                if let Some(state) = obj.get_mut("state") {
                    redact_state(state);
                }
                if let Some(warnings) = obj.get_mut("warnings") {
                    redact_warnings(warnings);
                }
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
    fn folders_with_spaces_are_scrubbed_whole() {
        // The profile of a user with a space in the name: nothing of it, or of the folders below, survives.
        let msg = r"cannot read the corpus manifest C:\Users\Jane Doe\AppData\Local\sigil-dev-cache\p7\corpus\corpus.json: The system cannot find the file specified. (os error 2)";
        let got = scrub_str(msg);
        assert_eq!(got, "cannot read the corpus manifest <path>/corpus.json: The system cannot find the file specified. (os error 2)");
        assert!(!got.contains("Jane") && !got.contains("Doe") && !got.contains("AppData"));
        // Program Files, with parentheses and a file name.
        assert_eq!(
            scrub_str(r"LoadLibrary failed for C:\Program Files (x86)\NVIDIA Corporation\cudart64_13.dll (126)"),
            "LoadLibrary failed for <path>/cudart64_13.dll (126)"
        );
        // Quotes end a path with spaces in it; so do semicolons and commas between paths. (A file
        // name is kept as written, so its own spaces only matter to what follows it.)
        assert_eq!(scrub_str(r#"cannot open "C:\Users\Jane Doe\My Models\a.gguf" now"#), r#"cannot open "<path>/a.gguf" now"#);
        assert_eq!(scrub_str(r"C:\Users\Jane Doe\a.wav, D:\My Stuff\b.wav"), "<path>/a.wav, <path>/b.wav");
        // Unix homes and UNC shares with spaces.
        assert_eq!(scrub_str("/home/Jane Doe/models/x.gguf failed"), "<path>/x.gguf failed");
        assert_eq!(scrub_str(r"\\NAS 1\Shared Models\f.bin"), "<path>/f.bin");
        // A profile folder on its own leaves no name, with or without spaces and a trailing separator.
        assert_eq!(scrub_str(r"home is C:\Users\Jane Doe"), "home is <path>/");
        assert_eq!(scrub_str(r"home is C:\Users\jane"), "home is <path>/");
        assert_eq!(scrub_str(r"home is C:\Users\jane\"), "home is <path>/");
        assert_eq!(scrub_str("home is /Users/jane"), "home is <path>/");
        // Names and folders that are not paths stay.
        assert_eq!(scrub_str("connect to http://127.0.0.1:8080/v1 failed"), "connect to http://127.0.0.1:8080/v1 failed");
        assert_eq!(scrub_str("see a/b and 5/6"), "see a/b and 5/6");
    }

    #[test]
    fn the_privacy_check_flags_a_path_that_was_only_partly_replaced() {
        let leftovers = [
            // What the old scrubber left of a profile with a space in the name.
            r"cannot read the corpus manifest <path>/Jane Doe\AppData\Local\p7\corpus\corpus.json: missing",
            r"x <path>/Doe\AppData\Local\x.json",
            r"fragment of a profile \Users\Jane\x",
            r"fragment \appdata\local\x",
        ];
        for l in leftovers {
            let doc = json!({ "failures": [l] });
            assert!(privacy_errors(&doc).iter().any(|e| e.contains("looks like a local path")), "{l}");
        }
        // The scrubbed forms are clean, and so are plain sentences with a slash.
        for ok in ["<path>/corpus.json: missing", "<path>/", "1/2 done", "http://localhost:8080/pubsub"] {
            assert!(privacy_errors(&json!({ "failures": [ok] })).is_empty(), "{ok}");
        }
        // The old regexes' bug, end to end: the export of an error with a spaced profile folder is clean.
        let mut r = report();
        r["failures"] = json!([r"cannot read the corpus manifest C:\Users\Jane Doe\AppData\Local\p7\corpus\corpus.json: missing"]);
        let exp = make_export(&r, false, true, None, 5);
        assert_eq!(exp["failures"][0], "cannot read the corpus manifest <path>/corpus.json: missing");
        assert_eq!(privacy_errors(&exp), Vec::<String>::new());
    }

    #[test]
    fn other_programs_and_custom_power_plans_do_not_leave_the_machine() {
        use crate::services::bench::gpu_state::{state_warnings, StateSummary};
        // The real warning of the state probe, with two programs named in it.
        let summary = StateSummary {
            other_gpu_util_max_pct: Some(37.0),
            other_top_processes: vec!["GameName.exe".into(), "obs64.exe".into()],
            ..StateSummary::default()
        };
        let warnings = state_warnings(&summary, Some("cuda"));
        let who = warnings.iter().find(|w| w.starts_with("another process used the GPU")).expect("the warning").clone();
        assert!(who.contains("GameName.exe"), "{who}");
        let mut r = report();
        r["environment"]["power_plan"] = json!("My Gaming Plan");
        r["runs"][0]["warnings"] = json!([who.clone(), "the GPU throttled during the run (power)"]);
        r["runs"][0]["state"] = json!({
            "start": {"power_plan": "Balanced", "other_top_process": "GameName.exe (pid 7, 37 % SM)", "sm_clock_mhz": 2100},
            "end": {"power_plan": "Dell Optimized", "other_top_process": null},
            "series": [{"other_top_process": "obs64.exe (pid 9, 12 % SM)", "power_plan": "High performance"}],
            "summary": {"samples": 3, "other_gpu_util_max_pct": 37.0, "other_top_processes": ["GameName.exe", "obs64.exe"]},
            "warnings": [who]
        });
        let exp = make_export(&r, false, true, None, 5);
        let text = exp.to_string();
        for needle in ["GameName", "obs64", "My Gaming Plan", "Dell Optimized"] {
            assert!(!text.contains(needle), "{needle} leaked:\n{text}");
        }
        assert_eq!(exp["environment"]["power_plan"], "custom");
        let state = &exp["runs"][0]["state"];
        assert_eq!(state["start"]["power_plan"], "Balanced", "a built-in plan stays");
        assert_eq!(state["series"][0]["power_plan"], "High performance");
        assert_eq!(state["end"]["power_plan"], "custom");
        assert_eq!(state["start"]["sm_clock_mhz"], 2100, "the rest of the sample stays");
        assert!(state["start"].get("other_top_process").is_none() && state["series"][0].get("other_top_process").is_none());
        assert!(state["summary"].get("other_top_processes").is_none());
        assert_eq!(state["summary"]["other_top_processes_count"], 2);
        assert_eq!(state["summary"]["other_gpu_util_max_pct"], 37.0);
        assert_eq!(state["warnings"][0], "another process used the GPU (up to 37 % SM)");
        assert_eq!(exp["runs"][0]["warnings"][0], "another process used the GPU (up to 37 % SM)");
        assert_eq!(exp["runs"][0]["warnings"][1], "the GPU throttled during the run (power)");
        assert_eq!(privacy_errors(&exp), Vec::<String>::new());
        // Windows' own plans keep their name in any case; the report itself is not touched.
        for plan in ["Balanced", "high performance", "Power saver", "Ultimate Performance"] {
            assert_eq!(power_plan_label(plan), plan);
        }
        assert_eq!(r["environment"]["power_plan"], "My Gaming Plan");
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
