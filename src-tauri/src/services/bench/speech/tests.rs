//! Contract tests of the speech bench: the frozen P7-S0 examples (`scripts/speech-bench/examples/`)
//! deserialize with the Rust types and round-trip through the export rules, and whole runs over
//! scripted drivers produce reports that follow the schema's rules (typed items, nearest-rank
//! aggregates, thresholds, hypotheses files, a clean export).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use serde_json::{json, Value};

use super::corpus::{MtLine, TtsPrompt};
use super::report::{ReportItem, Report, ResultStatus, RunStatus};
use super::spec::{canonical_json, canonical_sha256, spec_sha256, EngineDef, Feed, SpeechSpec};
use super::stt::{DriverInfo, SttDriver, SttOutput, SttRequest};
use super::tts::{TtsDriver, TtsOutput};
use super::{export, mt, run_spec_with, runtime_pack_ids, utc_stamp, Drivers};

fn examples() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../scripts/speech-bench/examples")
}

fn example_text(name: &str) -> String {
    std::fs::read_to_string(examples().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sigil-speech-bench-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

// ---------------------------------------------------------------------------------------------
// The frozen examples
// ---------------------------------------------------------------------------------------------

/// Every `examples/spec.*.json` parses with the strict types and passes the cross-reference rules.
#[test]
fn every_example_spec_deserializes_and_validates() {
    let mut seen = 0;
    for entry in std::fs::read_dir(examples()).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if !(name.starts_with("spec.") && name.ends_with(".json")) {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let (spec, _raw) = SpeechSpec::parse(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(spec.schema, "sigil-speech-bench-spec/1", "{name}");
        assert!(!spec.engines.is_empty() && !spec.runs.is_empty(), "{name}");
        // Every run resolves to engines of a task its mode takes.
        for run in &spec.runs {
            let engines = spec.run_engines(run);
            assert!(!engines.is_empty(), "{name}/{}", run.name);
            assert!(engines.iter().all(|e| run.mode.tasks().contains(&e.task)), "{name}/{}", run.name);
        }
        seen += 1;
    }
    assert_eq!(seen, 5, "one example spec per mode");
}

#[test]
fn the_example_specs_carry_the_documented_defaults() {
    let (acc, _) = SpeechSpec::parse(&example_text("spec.accuracy.json")).unwrap();
    assert_eq!(acc.seed, 20_261_002);
    let core = &acc.runs[0];
    assert_eq!((core.warmup, core.repeat, core.timeout_s), (1, 1, 120.0));
    assert!(!core.lock_required(), "accuracy runs need no lock");
    assert_eq!(core.lock_tag(), "P7-A");
    assert_eq!(core.feed, Feed { kind: super::spec::FeedKind::Utterance, ..Feed::default() });
    assert_eq!(core.feed.tail_silence_ms, 800);
    assert_eq!(acc.runs[1].feed.kind, super::spec::FeedKind::Pipeline);
    assert_eq!(acc.runs[1].feed.lead_silence_ms, 500);
    let real = &acc.runs[2];
    assert_eq!(real.select.as_ref().unwrap().kinds, [super::spec::ItemKind::Real]);
    // The run's select replaces the corpus default (the real clips are not in `core`).
    assert_eq!(acc.corpus.as_ref().unwrap().select.subset.as_deref(), Some("core"));
    assert!(real.select.as_ref().unwrap().subset.is_none());

    let (gpu, _) = SpeechSpec::parse(&example_text("spec.gpu_under_load.json")).unwrap();
    let h1 = &gpu.runs[0];
    assert!(h1.lock_required());
    assert_eq!(h1.load_profile, super::spec::LoadProfile::H1VrchatObs);
    assert_eq!(h1.repeat, 3);
    let g = h1.load.as_ref().unwrap().gpu.as_ref().unwrap();
    assert!(g.out_of_process && g.execution_provider == "dml" && g.size == 256 && g.depth == 8);
    assert_eq!(h1.thresholds["rtf_p95"].max, Some(0.3));
    assert!(gpu.runs[2].co_load.is_some());

    let (cpu, _) = SpeechSpec::parse(&example_text("spec.cpu_speed.json")).unwrap();
    assert!(cpu.runs.iter().all(|r| r.lock_required()));
    let (mtq, _) = SpeechSpec::parse(&example_text("spec.mt_quality.json")).unwrap();
    assert!(mtq.output.export.include_text, "the MT example exports text");
    assert!(!mtq.runs[0].lock_required());
    assert_eq!(mtq.runs[2].mt.as_ref().unwrap().context, super::spec::MtContext::Previous);
    let (tts, _) = SpeechSpec::parse(&example_text("spec.tts_listening.json")).unwrap();
    assert!(tts.runs[0].tts.as_ref().unwrap().dnsmos);
    assert_eq!(tts.runs[0].tts.as_ref().unwrap().intelligibility.as_ref().unwrap().engine, "qwen3-asr-0.6b");
}

/// `report.example.json` deserializes with the strict report types and keeps its meaning through a
/// serialize / deserialize round trip.
#[test]
fn the_example_report_deserializes_with_the_typed_report() {
    let text = example_text("report.example.json");
    let report: Report = serde_json::from_str(&text).expect("report.example.json");
    assert_eq!(report.schema, "sigil-speech-bench-report/1");
    assert_eq!(report.runs.len(), 3);
    assert_eq!(report.engines.len(), 4);
    let core = &report.runs[0];
    assert_eq!(core.status, RunStatus::Failed);
    let first = &core.results[0];
    assert_eq!((first.status, first.aggregate.items, first.aggregate.errors), (ResultStatus::Failed, 8, 1));
    assert_eq!(first.aggregate.get("rtf_p95"), Some(0.0912));
    assert_eq!(first.groups.len(), 4);
    assert!(matches!(first.items[0], ReportItem::Stt(_)));
    assert!(matches!(report.runs[1].results[0].items[0], ReportItem::Mt(_)));
    assert!(matches!(report.runs[2].results[0].items[0], ReportItem::Tts(_)));
    assert_eq!(report.runs[2].results[0].thresholds[0].metric, "ttfa_ms_p95");
    // Round trip: the same typed report comes back.
    let again: Report = serde_json::from_str(&serde_json::to_string_pretty(&report).unwrap()).unwrap();
    assert_eq!(again, report);
    // The aggregate keeps the report's key order on the wire.
    let wire = serde_json::to_string(&report.runs[0].results[0].aggregate).unwrap();
    assert!(wire.starts_with(r#"{"items":8,"errors":1,"audio_s":36.0,"wall_ms":2077.0,"rtf_mean":0.066,"#), "{wire}");
}

#[test]
fn items_with_scores_or_unknown_members_are_refused() {
    let mut v: Value = serde_json::from_str(&example_text("report.example.json")).unwrap();
    v["runs"][0]["results"][0]["items"][0]["wer"] = json!(0.1);
    assert!(serde_json::from_value::<Report>(v).is_err(), "items never carry scores");
    let mut v: Value = serde_json::from_str(&example_text("report.example.json")).unwrap();
    v["bogus"] = json!(1);
    assert!(serde_json::from_value::<Report>(v).is_err());
}

/// The report's `spec_sha256` is the canonical hash of its spec, whatever the key order.
#[test]
fn the_example_report_hash_is_the_canonical_hash_of_its_spec() {
    let v: Value = serde_json::from_str(&example_text("report.example.json")).unwrap();
    assert_eq!(v["spec_sha256"].as_str().unwrap(), canonical_sha256(&v["spec"]));
    assert_eq!(v["spec_sha256"].as_str().unwrap(), spec_sha256(&v["spec"]));
    // Reversing every object's keys changes nothing.
    fn reversed(v: &Value) -> Value {
        match v {
            Value::Object(m) => Value::Object(m.iter().rev().map(|(k, v)| (k.clone(), reversed(v))).collect()),
            Value::Array(a) => Value::Array(a.iter().map(reversed).collect()),
            other => other.clone(),
        }
    }
    assert_eq!(canonical_sha256(&reversed(&v["spec"])), v["spec_sha256"].as_str().unwrap());
    // Sorted keys, no whitespace, non-ASCII kept.
    assert_eq!(canonical_json(&json!({"b": [1, 2.5, "ひ"], "a": {"z": null, "y": true}})), r#"{"a":{"y":true,"z":null},"b":[1,2.5,"ひ"]}"#);
}

/// The export of the example report is exactly `examples/export.example.json`, and clean.
#[test]
fn the_example_export_is_the_derivation_of_the_example_report() {
    let report: Value = serde_json::from_str(&example_text("report.example.json")).unwrap();
    let want: Value = serde_json::from_str(&example_text("export.example.json")).unwrap();
    let created = want["export"]["created_unix_ms"].as_u64().unwrap();
    let got = export::make_export(&report, false, true, Some("Format example (made-up numbers)."), created);
    assert_eq!(got, want);
    assert!(export::privacy_errors(&got).is_empty());
    assert!(export::privacy_errors(&want).is_empty());
    // With text allowed, synthetic text stays and the export is still clean.
    let with_text = export::make_export(&report, true, true, None, created);
    assert!(with_text["runs"][0]["results"][0]["items"][0]["text"].is_string());
    assert!(export::privacy_errors(&with_text).is_empty());
    let no_items = export::make_export(&report, false, false, None, created);
    assert!(no_items["runs"].as_array().unwrap().iter().all(|r| r["results"].as_array().unwrap().iter().all(|x| x["items"] == json!([]))));
}

#[test]
fn specs_with_unknown_members_or_broken_references_are_refused() {
    let base = json!({
        "schema": "sigil-speech-bench-spec/1", "name": "t",
        "corpus": {"manifest": "c.json"},
        "engines": [
            {"id": "a", "task": "stt", "backend": "whisper", "model": "whisper/base-q5_1"},
            {"id": "m", "task": "mt", "backend": "nllb"}
        ],
        "runs": [{"name": "r", "mode": "accuracy", "engines": ["a"]}]
    });
    let parse = |v: &Value| SpeechSpec::parse(&v.to_string());
    assert!(parse(&base).is_ok());
    let with = |f: &dyn Fn(&mut Value)| {
        let mut v = base.clone();
        f(&mut v);
        v
    };
    // $comment is fine anywhere, including inside the bench's LoadSpec; anything else unknown is not.
    assert!(parse(&with(&|v| {
        v["$comment"] = json!("x");
        v["runs"][0]["load"] = json!({"$comment": "c", "cpu": {"threads": 1, "$comment": "d"}});
        v["engines"][0]["$comment"] = json!(["a", "b"]);
    }))
    .is_ok());
    for (what, spec) in [
        ("a top-level typo", with(&|v| v["bogus"] = json!(1))),
        ("an engine typo", with(&|v| v["engines"][0]["modle"] = json!("x"))),
        ("a run typo", with(&|v| v["runs"][0]["warmups"] = json!(1))),
        ("a load typo", with(&|v| v["runs"][0]["load"] = json!({"cpu": {"threads": 1, "dutty": 1}}))),
        ("a wrong schema id", with(&|v| v["schema"] = json!("sigil-speech-bench-spec/2"))),
        ("a bad name", with(&|v| v["name"] = json!("../x"))),
        ("a duplicate engine id", with(&|v| v["engines"][1]["id"] = json!("a"))),
        ("a backend of another task", with(&|v| v["engines"][0]["backend"] = json!("nllb"))),
        ("an unknown engine", with(&|v| v["runs"][0]["engines"] = json!(["zz"]))),
        ("an engine of another task", with(&|v| v["runs"][0]["engines"] = json!(["m"]))),
        ("a duplicate run name", with(&|v| v["runs"] = json!([{"name": "r", "mode": "accuracy"}, {"name": "r", "mode": "accuracy"}]))),
        ("a secret-looking option", with(&|v| v["engines"][0]["options"] = json!({"apiKey": "k"}))),
        ("an STT run without a corpus", with(&|v| {
            v.as_object_mut().unwrap().remove("corpus");
        })),
        ("mt_quality without an mt block", with(&|v| v["runs"] = json!([{"name": "r", "mode": "mt_quality"}]))),
        ("a bad direction", with(&|v| v["engines"][1]["directions"] = json!(["english-japanese"]))),
        ("a threshold without a limit", with(&|v| v["runs"][0]["thresholds"] = json!({"rtf_p95": {}}))),
        ("a bad execution provider", with(&|v| v["engines"][0]["execution_provider"] = json!("tpu"))),
        ("a chunk outside its range", with(&|v| v["runs"][0]["feed"] = json!({"chunk_ms": 5}))),
    ] {
        assert!(parse(&spec).is_err(), "{what} must be refused");
    }
    // apiKeyEnv names a variable and is allowed.
    assert!(parse(&with(&|v| v["engines"][0]["options"] = json!({"apiKeyEnv": "GROQ_API_KEY"}))).is_ok());
}

#[test]
fn the_default_folder_stamp_is_utc() {
    assert_eq!(utc_stamp(0), "19700101-000000");
    assert_eq!(utc_stamp(1_790_000_000), "20260921-141320");
    assert_eq!(utc_stamp(951_782_400), "20000229-000000");
}

#[test]
fn runtime_packs_follow_the_backend_and_variant() {
    assert_eq!(runtime_pack_ids("sherpa-offline", Some("cpu"), None), ["runtime/sherpa-onnx"]);
    assert_eq!(runtime_pack_ids("transcribe", Some("cuda13"), None), ["runtime/sigil-transcribe-cuda13"]);
    assert_eq!(runtime_pack_ids("llama-server", Some("vulkan"), None), ["runtime/llama-server-vulkan"]);
    assert_eq!(runtime_pack_ids("tts-helper", Some("cuda13"), Some("qwen3")), ["runtime/qwentts-server-cuda13"]);
    assert_eq!(runtime_pack_ids("tts-helper", Some("vulkan"), Some("breeze")), ["runtime/breeze-server-vulkan"]);
    assert!(runtime_pack_ids("whisper", Some("cpu"), None).is_empty());
    assert!(runtime_pack_ids("hypotheses", None, None).is_empty());
}

// ---------------------------------------------------------------------------------------------
// Whole runs over scripted drivers
// ---------------------------------------------------------------------------------------------

/// A scripted STT driver: texts by item id, failures by item id, a fixed compute time.
struct FakeStt {
    info: DriverInfo,
    texts: HashMap<String, String>,
    fail: HashSet<String>,
    compute_ms: f64,
    calls: Mutex<Vec<String>>,
}

impl SttDriver for FakeStt {
    fn info(&self) -> &DriverInfo {
        &self.info
    }
    fn transcribe(&mut self, req: &SttRequest<'_>) -> Result<SttOutput, String> {
        self.calls.lock().unwrap().push(req.item.to_string());
        if self.fail.contains(req.item) {
            return Err("timeout after 120 s".into());
        }
        Ok(SttOutput {
            text: self.texts.get(req.item).cloned().unwrap_or_else(|| format!("text of {}", req.item)),
            language: req.language.clone(),
            compute_ms: Some(self.compute_ms),
            final_latency_ms: Some(self.compute_ms),
            ..SttOutput::default()
        })
    }
}

struct FakeMt {
    info: DriverInfo,
    /// Translate by returning the first reference of lines whose id is in `exact`, garbage otherwise.
    exact: HashSet<String>,
    refs: HashMap<String, String>,
    contexts: std::sync::Arc<Mutex<Vec<Option<String>>>>,
}

impl mt::MtDriver for FakeMt {
    fn info(&self) -> &DriverInfo {
        &self.info
    }
    fn translate(&mut self, line: &MtLine, context: Option<&MtLine>, _timeout: Duration) -> Result<mt::MtOutput, String> {
        self.contexts.lock().unwrap().push(context.map(|c| c.id.clone()));
        if line.id.ends_with(":boom") {
            return Err("server error".into());
        }
        let text = if self.exact.contains(&line.id) { self.refs[&line.id].clone() } else { "zzzz".to_string() };
        Ok(mt::MtOutput { text, out_tokens: Some(7) })
    }
}

struct FakeTts {
    info: DriverInfo,
}

impl TtsDriver for FakeTts {
    fn info(&self) -> &DriverInfo {
        &self.info
    }
    fn render(&mut self, prompt: &TtsPrompt, _timeout: Duration) -> Result<TtsOutput, String> {
        if prompt.id.ends_with("-bad") {
            return Err("the voice failed to load".into());
        }
        Ok(TtsOutput {
            samples: vec![0.25; 24_000],
            sample_rate: 24_000,
            ttfa_ms: Some(120.0),
            total_ms: Some(500.0),
            voice: Some("v1".into()),
        })
    }
}

type SttFactory = Box<dyn Fn(&EngineDef) -> Result<Box<dyn SttDriver>, String>>;

struct FakeDrivers {
    stt: SttFactory,
    mt_contexts: std::sync::Arc<Mutex<Vec<Option<String>>>>,
    mt_exact: HashSet<String>,
    mt_refs: HashMap<String, String>,
}

impl FakeDrivers {
    fn new(stt: SttFactory) -> FakeDrivers {
        FakeDrivers {
            stt,
            mt_contexts: Default::default(),
            mt_exact: HashSet::new(),
            mt_refs: HashMap::new(),
        }
    }
}

fn cpu_info() -> DriverInfo {
    DriverInfo {
        device: "cpu".into(),
        threads: Some(2),
        runtime: Some("fake 1.0".into()),
        ..DriverInfo::default()
    }
}

impl Drivers for FakeDrivers {
    fn model_version(&self, _id: &str) -> Option<String> {
        None
    }
    fn runtime_pack(&self, _id: &str) -> Option<super::report::RuntimePack> {
        None
    }
    fn meter_gpu(&self) -> bool {
        false
    }
    fn stt(&self, def: &EngineDef, _feed: &Feed) -> Result<Box<dyn SttDriver>, String> {
        (self.stt)(def)
    }
    fn mt(&self, def: &EngineDef) -> Result<Box<dyn mt::MtDriver>, String> {
        if def.model.contains("missing") {
            return Err(format!("not installed: {}", def.model));
        }
        Ok(Box::new(FakeMt {
            info: cpu_info(),
            exact: self.mt_exact.clone(),
            refs: self.mt_refs.clone(),
            contexts: self.mt_contexts.clone(),
        }))
    }
    fn tts(&self, def: &EngineDef, _first_language: &str) -> Result<Box<dyn TtsDriver>, String> {
        if def.model.contains("missing") {
            return Err(format!("not installed: {}", def.model));
        }
        Ok(Box::new(FakeTts { info: cpu_info() }))
    }
}

/// Write a 1 s 16 kHz WAV.
fn write_wav(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    super::tts::write_pcm16_wav(path, &vec![0.1; 16_000], 16_000).unwrap();
}

/// A corpus on disk: en 1001 / 1002 (clean, music_snr10), ja 2001 (clean), and one real clip.
fn corpus(dir: &Path) -> PathBuf {
    let rows = [
        ("en/1001/clean", "en/1001", "synthetic", "en", "clean", 0),
        ("en/1001/music_snr10", "en/1001", "synthetic", "en", "music_snr10", 0),
        ("en/1002/clean", "en/1002", "synthetic", "en", "clean", 1),
        ("en/1002/music_snr10", "en/1002", "synthetic", "en", "music_snr10", 1),
        ("ja/2001/clean", "ja/2001", "synthetic", "ja", "clean", 0),
        ("real/2026-10-05-alice/clip01", "real/2026-10-05-alice/clip01", "real", "ja", "real", 0),
    ];
    let mut items = Vec::new();
    for (id, utt, kind, lang, cond, rank) in rows {
        let audio = format!("audio/{}.wav", id.replace('/', "_"));
        write_wav(&dir.join(&audio));
        items.push(json!({
            "id": id, "utterance": utt, "kind": kind, "language": lang, "condition": cond,
            "audio": audio, "seconds": 1.0, "sha256": "0".repeat(64), "reference": "hello", "rank": rank
        }));
    }
    let manifest = json!({
        "schema": "sigil-speech-corpus/1", "name": "test-corpus", "sample_rate": 16000,
        "content_sha256": "a".repeat(64), "recipe_sha256": "b".repeat(64),
        "subsets": {"quick": ["clean", "music_snr10"], "real": ["real"]},
        "items": items
    });
    let path = dir.join("corpus.json");
    std::fs::write(&path, manifest.to_string()).unwrap();
    path
}

fn read_report(out: &Path) -> Report {
    serde_json::from_str(&std::fs::read_to_string(out.join("report.json")).unwrap()).expect("report.json follows the typed report")
}

fn stt_factory(compute_ms: f64, fail: &'static [&'static str]) -> SttFactory {
    Box::new(move |def: &EngineDef| {
        if def.model.contains("missing") {
            return Err(format!("not installed: {}", def.model));
        }
        Ok(Box::new(FakeStt {
            info: DriverInfo {
                variant: Some("cpu".into()),
                ..cpu_info()
            },
            texts: HashMap::new(),
            fail: fail.iter().map(|s| s.to_string()).collect(),
            compute_ms,
            calls: Mutex::new(Vec::new()),
        }) as Box<dyn SttDriver>)
    })
}

#[test]
fn an_accuracy_run_writes_hypotheses_a_typed_report_and_a_clean_export() {
    let dir = temp_dir("accuracy");
    let manifest = corpus(&dir);
    let spec_file = dir.join("spec.json");
    let spec_json = json!({
        "$comment": "test",
        "schema": "sigil-speech-bench-spec/1", "name": "acc",
        "corpus": {"manifest": manifest.to_string_lossy(), "select": {"subset": "quick", "kinds": ["synthetic"]}},
        "engines": [
            {"id": "good", "task": "stt", "backend": "whisper", "model": "whisper/base-q5_1", "execution_provider": "cpu", "threads": 2},
            {"id": "flaky", "task": "stt", "backend": "sherpa-offline", "model": "stt/flaky", "execution_provider": "cpu", "languages": ["en"]},
            {"id": "absent", "task": "stt", "backend": "transcribe", "model": "stt/missing-model", "execution_provider": "cuda"}
        ],
        "runs": [
            {"name": "core", "mode": "accuracy", "warmup": 0, "thresholds": {"errors": {"max": 0}}},
            {"name": "real", "mode": "accuracy", "engines": ["good"], "select": {"kinds": ["real"]}, "warmup": 0}
        ]
    });
    std::fs::write(&spec_file, spec_json.to_string()).unwrap();
    let text = std::fs::read_to_string(&spec_file).unwrap();
    let (spec, raw) = SpeechSpec::parse(&text).unwrap();
    let drivers = FakeDrivers::new(Box::new(|def: &EngineDef| {
        let fail: &'static [&'static str] = if def.id == "flaky" { &["en/1002/music_snr10"] } else { &[] };
        stt_factory(250.0, fail)(def)
    }));
    let out = dir.join("out");
    let err = run_spec_with(&spec, &raw, &spec_file, &out, &drivers).expect_err("an engine failed, so the run fails");
    assert!(err.contains("FAILED") && err.contains("absent") && err.contains("not installed: stt/missing-model"), "{err}");

    let report = read_report(&out);
    assert!(!report.passed);
    assert_eq!(report.spec_sha256, spec_sha256(&raw));
    assert_eq!(report.spec, raw, "the spec as written, comments included");
    assert!(report.failures.iter().any(|f| f == "core / absent: not installed: stt/missing-model"), "{:?}", report.failures);
    assert!(report.failures.iter().any(|f| f.starts_with("core / flaky: 1 item failed (timeout after 120 s)")), "{:?}", report.failures);
    let corpus_stamp = report.corpus.as_ref().unwrap();
    assert_eq!((corpus_stamp.name.as_str(), corpus_stamp.items_selected), ("test-corpus", 6));
    assert_eq!(corpus_stamp.content_sha256, "a".repeat(64));

    let core = &report.runs[0];
    assert_eq!(core.status, RunStatus::Failed);
    assert_eq!(core.feed.as_ref().unwrap()["kind"], "utterance");
    let by = |name: &str| core.results.iter().find(|r| r.engine == name).unwrap();
    // The healthy engine: five synthetic items of the quick subset, typed STT items, no scores.
    let good = by("good");
    assert_eq!((good.status, good.aggregate.items, good.aggregate.errors), (ResultStatus::Passed, 5, 0));
    assert_eq!(good.execution_provider_used.as_deref(), Some("cpu"));
    assert_eq!(good.aggregate.get("compute_ms_p50"), Some(250.0));
    assert_eq!(good.aggregate.get("audio_s"), Some(5.0));
    assert_eq!(good.aggregate.get("rtf_p95"), Some(0.25));
    assert_eq!(good.aggregate.get("final_latency_ms_p95"), Some(250.0));
    let keys: Vec<(String, String)> = good.groups.iter().map(|g| (g.language.clone().unwrap(), g.condition.clone().unwrap())).collect();
    assert_eq!(keys, [("en".into(), "clean".into()), ("en".into(), "music_snr10".into()), ("ja".into(), "clean".into())]);
    let ReportItem::Stt(first) = &good.items[0] else { panic!("an STT item") };
    assert_eq!((first.item.as_str(), first.text.as_deref(), first.language_detected.as_deref()), ("en/1001/clean", Some("text of en/1001/clean"), Some("en")));
    assert!(first.error.is_none());
    // The flaky engine: restricted to English, one item timed out, the run carries on.
    let flaky = by("flaky");
    assert_eq!((flaky.status, flaky.aggregate.items, flaky.aggregate.errors), (ResultStatus::Failed, 4, 1));
    let bad = flaky.items.iter().find(|i| i.id() == "en/1002/music_snr10").unwrap();
    assert_eq!(bad.error(), Some("timeout after 120 s"));
    let ReportItem::Stt(bad) = bad else { panic!() };
    assert!(bad.text.is_none() && bad.compute_ms.is_none());
    // The engine whose pack is missing fails its own rows only.
    let absent = by("absent");
    assert_eq!(absent.status, ResultStatus::Error);
    assert_eq!(absent.error.as_deref(), Some("not installed: stt/missing-model"));
    assert!(absent.items.is_empty());
    // A threshold on a count: errors max 0 fails the flaky engine.
    assert!(flaky.thresholds.iter().any(|t| t.metric == "errors" && !t.passed));
    assert!(good.thresholds.iter().all(|t| t.passed));
    // The run's second part: real clips, selected by kind.
    let real = &report.runs[1];
    assert_eq!(real.status, RunStatus::Passed);
    assert_eq!(real.results[0].items.len(), 1);
    assert_eq!(real.results[0].items[0].id(), "real/2026-10-05-alice/clip01");

    // Engine rows: the requested and the used device, with the load recorded.
    let row = |id: &str| report.engines.iter().find(|e| e.id == id).unwrap();
    assert_eq!(row("good").execution_provider_requested, "cpu");
    assert_eq!(row("good").execution_provider_used.as_deref(), Some("cpu"));
    assert_eq!(row("absent").execution_provider_used, None);
    assert_eq!(row("absent").error.as_deref(), Some("not installed: stt/missing-model"));
    assert!(row("good").load_ms.is_some());

    // Hypotheses files: one report item per line.
    let hyps = std::fs::read_to_string(out.join("hyps/core.good.jsonl")).unwrap();
    let lines: Vec<&str> = hyps.lines().collect();
    assert_eq!(lines.len(), 5);
    let first: Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(first["item"], "en/1001/clean");
    assert!(out.join("hyps/core.flaky.jsonl").is_file() && out.join("hyps/real.good.jsonl").is_file());
    assert!(!out.join("hyps/core.absent.jsonl").exists());
    assert!(core.files.contains(&"hyps/core.good.jsonl".to_string()));
    assert!(std::fs::read_to_string(out.join("report.md")).unwrap().contains("| good |"));

    // The export: no text (include_text is off), no real-clip text, privacy rules hold.
    let export: Value = serde_json::from_str(&std::fs::read_to_string(out.join("export.json")).unwrap()).unwrap();
    assert_eq!(export["schema"], "sigil-speech-bench-export/1");
    assert!(export::privacy_errors(&export).is_empty());
    assert!(export["runs"][0]["results"][0]["items"][0]["text"].is_null());
    assert_eq!(export["spec_summary"]["engines"][2]["model"], "stt/missing-model");
    assert_eq!(report.files, ["report.md", "export.json"]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn text_is_exported_only_for_synthetic_items_when_the_spec_allows_it() {
    let dir = temp_dir("export-text");
    let manifest = corpus(&dir);
    let spec_file = dir.join("spec.json");
    let spec_json = json!({
        "schema": "sigil-speech-bench-spec/1", "name": "exp",
        "corpus": {"manifest": manifest.to_string_lossy(), "select": {"conditions": ["clean", "real"], "languages": ["ja"]}},
        "engines": [{"id": "e", "task": "stt", "backend": "whisper", "model": "whisper/base-q5_1", "execution_provider": "cpu"}],
        "runs": [{"name": "r", "mode": "accuracy", "warmup": 0}],
        "output": {"export": {"include_text": true}}
    });
    let (spec, raw) = SpeechSpec::parse(&spec_json.to_string()).unwrap();
    let drivers = FakeDrivers::new(stt_factory(10.0, &[]));
    let out = dir.join("out");
    run_spec_with(&spec, &raw, &spec_file, &out, &drivers).expect("passes");
    let export: Value = serde_json::from_str(&std::fs::read_to_string(out.join("export.json")).unwrap()).unwrap();
    let items = export["runs"][0]["results"][0]["items"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    let text_of = |id: &str| items.iter().find(|i| i["item"] == id).unwrap()["text"].clone();
    assert_eq!(text_of("ja/2001/clean"), "text of ja/2001/clean");
    assert!(text_of("real/2026-10-05-alice/clip01").is_null(), "real clips never carry text");
    assert!(export::privacy_errors(&export).is_empty());
    // The report itself keeps every hypothesis (the scorer needs them).
    let report = read_report(&out);
    let ReportItem::Stt(real) = report.runs[0].results[0].items.iter().find(|i| i.id().starts_with("real/")).unwrap() else { panic!() };
    assert!(real.text.is_some());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn speed_runs_need_the_gpu_lock_and_refuse_without_it() {
    let dir = temp_dir("lock");
    let manifest = corpus(&dir);
    let spec_file = dir.join("spec.json");
    let lock = dir.join("gpu.lock");
    let mk = |require: Value| {
        json!({
            "schema": "sigil-speech-bench-spec/1", "name": "speed",
            "corpus": {"manifest": manifest.to_string_lossy(), "select": {"subset": "quick", "limit_per_language": 1, "languages": ["en"], "conditions": ["clean"]}},
            "engines": [{"id": "e", "task": "stt", "backend": "whisper", "model": "whisper/base-q5_1", "execution_provider": "cpu"}],
            "runs": [{"name": "r", "mode": "cpu_speed", "warmup": 0, "repeat": 2, "state_period_ms": 0,
                      "gpu_lock": {"require": require, "tag": "P7-A", "path": lock.to_string_lossy()},
                      "thresholds": {"compute_ms_p95": {"max": 100.0}}}]
        })
    };
    let drivers = FakeDrivers::new(stt_factory(50.0, &[]));
    // Required, no lock file: refused with the reason, nothing measured.
    let (spec, raw) = SpeechSpec::parse(&mk(json!(true)).to_string()).unwrap();
    let out = dir.join("refused");
    let err = run_spec_with(&spec, &raw, &spec_file, &out, &drivers).unwrap_err();
    assert!(err.contains("gpu_lock.require"), "{err}");
    let report = read_report(&out);
    assert_eq!(report.runs[0].status, RunStatus::Error);
    assert!(report.runs[0].results.is_empty());
    assert_eq!(report.runs[0].gpu_lock.as_ref().map(|l| (l.required, l.verified)), Some((true, false)));
    // Another owner: refused, naming both.
    std::fs::write(&lock, r#"{"tag":"P3B-A","pid":1}"#).unwrap();
    let err = run_spec_with(&spec, &raw, &spec_file, &dir.join("refused2"), &drivers).unwrap_err();
    assert!(err.contains("P3B-A") && err.contains("P7-A"), "{err}");
    // Held by the tag: the run is measured (fake compute 50 ms, under the 100 ms threshold).
    std::fs::write(&lock, r#"{"tag":"P7-A","pid":1}"#).unwrap();
    let out = dir.join("held");
    run_spec_with(&spec, &raw, &spec_file, &out, &drivers).expect("passes under the lock");
    let report = read_report(&out);
    let lock_block = report.runs[0].gpu_lock.as_ref().unwrap();
    assert!(lock_block.verified && lock_block.held_by.as_deref() == Some("P7-A"));
    let res = &report.runs[0].results[0];
    assert_eq!((res.aggregate.items, res.aggregate.get("compute_ms_p95")), (1, Some(50.0)));
    assert!(res.thresholds[0].passed);
    assert_eq!(report.runs[0].load.as_ref().unwrap()["profile"], "none");
    // The same spec with a threshold the engine misses fails, and says which.
    let mut tight = mk(json!(true));
    tight["runs"][0]["thresholds"] = json!({"compute_ms_p95": {"max": 10.0}, "rtf_p95": {"max": 1.0}});
    let (spec2, raw2) = SpeechSpec::parse(&tight.to_string()).unwrap();
    let err = run_spec_with(&spec2, &raw2, &spec_file, &dir.join("tight"), &drivers).unwrap_err();
    assert!(err.contains("compute_ms_p95") && err.contains("max 10"), "{err}");
    // require=false: no lock needed, the owner (if any) is recorded unverified.
    std::fs::remove_file(&lock).unwrap();
    let (spec3, raw3) = SpeechSpec::parse(&mk(json!(false)).to_string()).unwrap();
    run_spec_with(&spec3, &raw3, &spec_file, &dir.join("relaxed"), &drivers).expect("no lock required");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mt_quality_translates_every_direction_with_chrf_and_context() {
    let dir = temp_dir("mt");
    let set = dir.join("set.jsonl");
    let lines = [
        r#"{"id":"s:en-ja:1","source":"en","target":"ja","src":"Hello there","ref":"こんにちは","doc":"d1#1"}"#,
        r#"{"id":"s:en-ja:2","source":"en","target":"ja","src":"How are you","ref":"お元気ですか","doc":"d1#2"}"#,
        r#"{"id":"s:en-ja:boom","source":"en","target":"ja","src":"Fails","ref":"失敗","doc":"d1#3"}"#,
        r#"{"id":"s:ja-en:1","source":"ja","target":"en","src":"はい","ref":"Yes","doc":"d2#1"}"#,
    ];
    std::fs::write(&set, lines.join("\n")).unwrap();
    let spec_file = dir.join("spec.json");
    let spec_json = json!({
        "schema": "sigil-speech-bench-spec/1", "name": "mtq",
        "engines": [
            {"id": "llm", "task": "mt", "backend": "openai", "execution_provider": "cuda", "options": {"baseUrl": "http://127.0.0.1:1/v1", "model": "m"}},
            {"id": "gone", "task": "mt", "backend": "nllb", "model": "mt/missing-nllb"},
            {"id": "ja-only", "task": "mt", "backend": "nllb", "directions": ["ja-en"]}
        ],
        "runs": [{"name": "q", "mode": "mt_quality", "warmup": 0, "mt": {"set": set.to_string_lossy(), "context": "previous"}}],
        "output": {"export": {"include_text": true}}
    });
    let (spec, raw) = SpeechSpec::parse(&spec_json.to_string()).unwrap();
    let mut drivers = FakeDrivers::new(stt_factory(1.0, &[]));
    drivers.mt_exact = ["s:en-ja:1", "s:ja-en:1"].iter().map(|s| s.to_string()).collect();
    drivers.mt_refs = HashMap::from([("s:en-ja:1".to_string(), "こんにちは".to_string()), ("s:ja-en:1".to_string(), "Yes".to_string())]);
    let out = dir.join("out");
    let err = run_spec_with(&spec, &raw, &spec_file, &out, &drivers).unwrap_err();
    assert!(err.contains("gone") && err.contains("not installed: mt/missing-nllb"), "{err}");
    let report = read_report(&out);
    let run = &report.runs[0];
    let by = |id: &str| run.results.iter().find(|r| r.engine == id).unwrap();
    let llm = by("llm");
    assert_eq!((llm.aggregate.items, llm.aggregate.errors), (4, 1));
    let ReportItem::Mt(first) = &llm.items[0] else { panic!("an MT item") };
    assert_eq!((first.item.as_str(), first.direction.as_str(), first.text.as_deref(), first.out_tokens), ("s:en-ja:1", "en-ja", Some("こんにちは"), Some(7)));
    assert_eq!(first.src_chars, Some(11));
    assert!(first.latency_ms.is_some());
    let boom = llm.items.iter().find(|i| i.id() == "s:en-ja:boom").unwrap();
    assert_eq!(boom.error(), Some("server error"));
    // chrF: the exact line scores 100, the failed and garbage lines pull the direction down.
    let en_ja = llm.groups.iter().find(|g| g.direction.as_deref() == Some("en-ja")).unwrap();
    let chrf = en_ja.aggregate.get("chrf").unwrap();
    assert!(chrf > 0.0 && chrf < 100.0, "{chrf}");
    let ja_en = llm.groups.iter().find(|g| g.direction.as_deref() == Some("ja-en")).unwrap();
    assert_eq!(ja_en.aggregate.get("chrf"), Some(100.0));
    assert_eq!(en_ja.aggregate.items, 3);
    assert!(llm.aggregate.get("latency_ms_p95").is_some() && llm.aggregate.get("chars_per_s").is_some());
    // An engine restricted to one direction sees only that one.
    assert_eq!(by("ja-only").aggregate.items, 1);
    assert_eq!(by("gone").status, ResultStatus::Error);
    // The context passed to an LLM backend is the previous line of the same document, none for others.
    let seen = drivers.mt_contexts.lock().unwrap().clone();
    assert!(seen.contains(&Some("s:en-ja:1".to_string())) && seen.contains(&Some("s:en-ja:2".to_string())));
    assert_eq!(seen.iter().filter(|c| c.is_some()).count(), 2, "only the openai engine takes context");
    // The export keeps translations (include_text) and stays clean.
    let export: Value = serde_json::from_str(&std::fs::read_to_string(out.join("export.json")).unwrap()).unwrap();
    assert_eq!(export["runs"][0]["results"][0]["items"][0]["text"], "こんにちは");
    assert!(export::privacy_errors(&export).is_empty());
    assert!(out.join("hyps/q.llm.jsonl").is_file());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn tts_listening_renders_wavs_and_reports_timings() {
    let dir = temp_dir("tts");
    let prompts = dir.join("prompts.jsonl");
    std::fs::write(
        &prompts,
        [
            r#"{"id":"en-casual-1","lang":"en","text":"Hello there.","kind":"casual"}"#,
            r#"{"id":"en-casual-bad","lang":"en","text":"This one fails.","kind":"casual"}"#,
            r#"{"id":"ja-casual-1","lang":"ja","text":"こんにちは","kind":"casual","voice":"jf"}"#,
        ]
        .join("\n"),
    )
    .unwrap();
    let spec_file = dir.join("spec.json");
    let spec_json = json!({
        "schema": "sigil-speech-bench-spec/1", "name": "ttsl",
        "engines": [
            {"id": "voice", "task": "tts", "backend": "local-tts", "model": "tts/kokoro-v1.0", "execution_provider": "cpu", "languages": ["en", "ja"]},
            {"id": "en-only", "task": "tts", "backend": "local-tts", "model": "tts/other", "languages": ["en"]},
            {"id": "gone", "task": "tts", "backend": "local-tts", "model": "tts/missing"}
        ],
        "runs": [{"name": "render", "mode": "tts_listening", "warmup": 0, "tts": {"prompts": prompts.to_string_lossy(), "limit_per_language": 2},
                  "thresholds": {"ttfa_ms_p95": {"max": 800}}}]
    });
    let (spec, raw) = SpeechSpec::parse(&spec_json.to_string()).unwrap();
    let drivers = FakeDrivers::new(stt_factory(1.0, &[]));
    let out = dir.join("out");
    let err = run_spec_with(&spec, &raw, &spec_file, &out, &drivers).unwrap_err();
    assert!(err.contains("gone"), "{err}");
    let report = read_report(&out);
    let run = &report.runs[0];
    let voice = run.results.iter().find(|r| r.engine == "voice").unwrap();
    assert_eq!((voice.aggregate.items, voice.aggregate.errors), (3, 1));
    let ReportItem::Tts(ok) = &voice.items[0] else { panic!("a TTS item") };
    assert_eq!(ok.audio.as_deref(), Some("audio/voice/en-casual-1.wav"));
    assert_eq!((ok.audio_s, ok.sample_rate, ok.ttfa_ms, ok.total_ms), (Some(1.0), Some(24_000), Some(120.0), Some(500.0)));
    assert_eq!(ok.rtf, Some(0.5));
    assert_eq!(ok.voice.as_deref(), Some("v1"));
    let wav = out.join("audio/voice/en-casual-1.wav");
    let mut reader = hound::WavReader::open(&wav).expect("a real WAV");
    let spec_wav = reader.spec();
    assert_eq!((spec_wav.channels, spec_wav.sample_rate, spec_wav.bits_per_sample), (1, 24_000, 16));
    assert_eq!(reader.samples::<i16>().count(), 24_000);
    let bad = voice.items.iter().find(|i| i.id() == "en-casual-bad").unwrap();
    assert_eq!(bad.error(), Some("the voice failed to load"));
    let ReportItem::Tts(bad) = bad else { panic!() };
    assert!(bad.audio.is_none());
    assert!(!out.join("audio/voice/en-casual-bad.wav").exists());
    assert_eq!(voice.aggregate.get("ttfa_ms_p95"), Some(120.0));
    assert!(voice.thresholds[0].passed);
    assert!(run.files.contains(&"audio/voice/en-casual-1.wav".to_string()));
    // The second engine renders English only; its audio goes to its own folder.
    let en_only = run.results.iter().find(|r| r.engine == "en-only").unwrap();
    assert_eq!(en_only.aggregate.items, 2);
    assert!(out.join("audio/en-only/en-casual-1.wav").is_file());
    // The export never carries audio paths.
    let export: Value = serde_json::from_str(&std::fs::read_to_string(out.join("export.json")).unwrap()).unwrap();
    assert!(export["runs"][0]["results"][0]["items"][0]["audio"].is_null());
    assert!(export::privacy_errors(&export).is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_run_with_nothing_to_select_is_an_error_and_the_other_runs_continue() {
    let dir = temp_dir("empty");
    let manifest = corpus(&dir);
    let spec_file = dir.join("spec.json");
    let spec_json = json!({
        "schema": "sigil-speech-bench-spec/1", "name": "empty",
        "corpus": {"manifest": manifest.to_string_lossy()},
        "engines": [{"id": "e", "task": "stt", "backend": "whisper", "model": "whisper/base-q5_1", "execution_provider": "cpu"}],
        "runs": [
            {"name": "none", "mode": "accuracy", "select": {"languages": ["fr"]}},
            {"name": "bad-subset", "mode": "accuracy", "select": {"subset": "nope"}},
            {"name": "ok", "mode": "accuracy", "select": {"conditions": ["clean"], "languages": ["en"]}, "warmup": 0}
        ]
    });
    let (spec, raw) = SpeechSpec::parse(&spec_json.to_string()).unwrap();
    let drivers = FakeDrivers::new(stt_factory(1.0, &[]));
    let out = dir.join("out");
    let err = run_spec_with(&spec, &raw, &spec_file, &out, &drivers).unwrap_err();
    assert!(err.contains("matches no corpus items") && err.contains("not in the corpus"), "{err}");
    let report = read_report(&out);
    assert_eq!(report.runs[0].status, RunStatus::Error);
    assert_eq!(report.runs[1].status, RunStatus::Error);
    assert!(report.runs[1].error.as_deref().unwrap().contains("quick") || report.runs[1].error.as_deref().unwrap().contains("subsets"));
    assert_eq!(report.runs[2].status, RunStatus::Passed);
    assert_eq!(report.runs[2].results[0].aggregate.items, 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_corpus_manifest_is_a_run_error_with_the_file_name_only_in_the_export() {
    let dir = temp_dir("nomanifest");
    let spec_file = dir.join("spec.json");
    let spec_json = json!({
        "schema": "sigil-speech-bench-spec/1", "name": "nm",
        "corpus": {"manifest": dir.join("gone").join("corpus.json").to_string_lossy()},
        "engines": [{"id": "e", "task": "stt", "backend": "whisper", "model": "whisper/base-q5_1"}],
        "runs": [{"name": "r", "mode": "accuracy"}]
    });
    let (spec, raw) = SpeechSpec::parse(&spec_json.to_string()).unwrap();
    let drivers = FakeDrivers::new(stt_factory(1.0, &[]));
    let out = dir.join("out");
    let err = run_spec_with(&spec, &raw, &spec_file, &out, &drivers).unwrap_err();
    assert!(err.contains("cannot read the corpus manifest"), "{err}");
    // The report keeps the message as is; the export has no absolute path in it.
    let export: Value = serde_json::from_str(&std::fs::read_to_string(out.join("export.json")).unwrap()).unwrap();
    assert!(export::privacy_errors(&export).is_empty(), "{:?}", export::privacy_errors(&export));
    assert!(export["runs"][0]["error"].as_str().unwrap().contains("<path>/corpus.json"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_speed_run_applies_the_load_and_reports_what_it_achieved() {
    let dir = temp_dir("load");
    let manifest = corpus(&dir);
    let spec_file = dir.join("spec.json");
    let spec_json = json!({
        "schema": "sigil-speech-bench-spec/1", "name": "loaded",
        "corpus": {"manifest": manifest.to_string_lossy(), "select": {"conditions": ["clean"], "languages": ["en"]}},
        "engines": [{"id": "e", "task": "stt", "backend": "whisper", "model": "whisper/base-q5_1", "execution_provider": "cpu"}],
        "runs": [{
            "name": "game", "mode": "cpu_speed", "warmup": 1, "repeat": 2, "state_period_ms": 0,
            "load_profile": "custom", "load": {"cpu": {"threads": 1, "duty": 0.5, "period_ms": 5}},
            "external_load": "VRChat busy world",
            "gpu_lock": {"require": false}
        }]
    });
    let (spec, raw) = SpeechSpec::parse(&spec_json.to_string()).unwrap();
    let drivers = FakeDrivers::new(stt_factory(5.0, &[]));
    let out = dir.join("out");
    run_spec_with(&spec, &raw, &spec_file, &out, &drivers).expect("passes");
    let report = read_report(&out);
    let run = &report.runs[0];
    let load = run.load.as_ref().unwrap();
    assert_eq!(load["profile"], "custom");
    assert_eq!(load["external"], "VRChat busy world");
    assert_eq!(load["spec"]["cpu"]["threads"], 1);
    assert_eq!(load["achieved"]["cpu"]["threads"], 1, "the burner reports what it did: {load}");
    assert!(load["achieved"]["cpu"]["cpu_ms"].as_f64().is_some());
    let res = &run.results[0];
    assert_eq!(res.aggregate.items, 2);
    assert!(res.aggregate.get("cpu_cores_mean").is_some(), "process CPU is measured in speed runs");
    assert!(res.aggregate.get("ram_peak_mb").is_some());
    // The export summary carries the load text and profile, never the load's internals as paths.
    let export: Value = serde_json::from_str(&std::fs::read_to_string(out.join("export.json")).unwrap()).unwrap();
    assert_eq!(export["spec_summary"]["runs"][0]["external_load"], "VRChat busy world");
    assert_eq!(export["spec_summary"]["runs"][0]["load_profile"], "custom");
    // A repeated pass keeps the first pass's text and the median of the timings.
    let ReportItem::Stt(first) = &res.items[0] else { panic!() };
    assert_eq!(first.compute_ms, Some(5.0));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn empty_hypotheses_for_clips_with_speech_raise_a_warning() {
    let dir = temp_dir("empty-text");
    let manifest = corpus(&dir);
    let spec_file = dir.join("spec.json");
    let spec_json = json!({
        "schema": "sigil-speech-bench-spec/1", "name": "emptytext",
        "corpus": {"manifest": manifest.to_string_lossy(), "select": {"conditions": ["clean"], "kinds": ["synthetic"]}},
        "engines": [{"id": "e", "task": "stt", "backend": "sherpa-offline", "model": "stt/x", "execution_provider": "cpu"}],
        "runs": [{"name": "r", "mode": "accuracy", "warmup": 0}]
    });
    let (spec, raw) = SpeechSpec::parse(&spec_json.to_string()).unwrap();
    let drivers = FakeDrivers::new(Box::new(|_def: &EngineDef| {
        Ok(Box::new(FakeStt {
            info: cpu_info(),
            texts: HashMap::from([("ja/2001/clean".to_string(), String::new())]),
            fail: HashSet::new(),
            compute_ms: 1.0,
            calls: Mutex::new(Vec::new()),
        }) as Box<dyn SttDriver>)
    }));
    let out = dir.join("out");
    run_spec_with(&spec, &raw, &spec_file, &out, &drivers).expect("an empty hypothesis is not an item error");
    let report = read_report(&out);
    let run = &report.runs[0];
    assert!(run.warnings.iter().any(|w| w.starts_with("e: 1 of 3 item(s) returned empty text")), "{:?}", run.warnings);
    let ReportItem::Stt(item) = run.results[0].items.iter().find(|i| i.id() == "ja/2001/clean").unwrap() else { panic!() };
    assert_eq!(item.text.as_deref(), Some(""), "the empty hypothesis stays in the report for the scorer");
    assert!(item.error.is_none());
    let _ = std::fs::remove_dir_all(&dir);
}
