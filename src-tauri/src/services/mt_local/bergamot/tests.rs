//! Tests of the Bergamot client: catalog routing and language codes, the wire format, the pack checks, and the
//! engine end to end against a fake helper. The fake is this test binary re-invoked through `process_host` with a
//! filter that runs only [`fake_bergamot_entry`], which speaks protocol 1 over stdio (so the real launch, pipes,
//! restarts and stops are exercised; only the translation itself is fake). Nothing touches the network.

use super::catalog::{self, Directions, Route};
use super::client::{Engine, EngineConfig, Environment};
use super::pack::{self, CpuFeatures};
use super::protocol::{self, HelperError, Line, Priority};
use crate::services::model_store::{Catalog, CatalogEntry, CatalogFile, CatalogSource, EntryKind, PublishStatus, SourceKind};
use crate::services::mt_local::{MtEngineState, MtError, MtErrorKind, MtKind, MtRequest};
use crate::services::process_host::{self, HelperSpec, StdioMode};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::{BufRead, Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------

fn entry(id: &str, files: &[&str]) -> CatalogEntry {
    CatalogEntry {
        id: id.into(),
        family: catalog::FAMILY.into(),
        kind: EntryKind::Model,
        version: "2.0".into(),
        title: None,
        description: None,
        license: "MPL-2.0".into(),
        status: PublishStatus::Published,
        requires: vec!["runtime/bergamot".into()],
        min_vram_mb: None,
        variant: None,
        files: files
            .iter()
            .map(|p| CatalogFile {
                path: (*p).into(),
                size: 1,
                sha256: None,
                git_oid: None,
                sources: vec![CatalogSource {
                    kind: SourceKind::Direct,
                    url: format!("https://example.invalid/{p}"),
                    revision: None,
                }],
                archive: None,
            })
            .collect(),
    }
}

/// Mozilla-style file names for a direction (`en_ja` → `model.enja.intgemm.alphas.bin`, ...).
fn shared_vocab_files(pair: &str) -> Vec<String> {
    vec![
        format!("model.{pair}.intgemm.alphas.bin"),
        format!("lex.50.50.{pair}.s2t.bin"),
        format!("vocab.{pair}.spm"),
    ]
}

fn split_vocab_files(pair: &str) -> Vec<String> {
    vec![
        format!("model.{pair}.intgemm.alphas.bin"),
        format!("lex.50.50.{pair}.s2t.bin"),
        format!("srcvocab.{pair}.spm"),
        format!("trgvocab.{pair}.spm"),
    ]
}

/// en↔ja, en→zh (split vocab), zh-hant→en, de→en, en→de, plus `crash_en` (its load kills the fake) and `slow_en`
/// (its load takes 1.5 s).
fn test_directions() -> Directions {
    let specs: Vec<(&str, Vec<String>)> = vec![
        ("mt-bergamot/en_ja", split_vocab_files("enja")),
        ("mt-bergamot/ja_en", shared_vocab_files("jaen")),
        ("mt-bergamot/en_zh", split_vocab_files("enzh")),
        ("mt-bergamot/zh-hant_en", shared_vocab_files("zhen")),
        ("mt-bergamot/de_en", shared_vocab_files("deen")),
        ("mt-bergamot/en_de", shared_vocab_files("ende")),
        ("mt-bergamot/crash_en", shared_vocab_files("crashen")),
        ("mt-bergamot/slow_en", shared_vocab_files("slowen")),
    ];
    let entries: Vec<CatalogEntry> = specs
        .iter()
        .map(|(id, files)| entry(id, &files.iter().map(String::as_str).collect::<Vec<_>>()))
        .collect();
    Directions::from_entries(entries.iter())
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sigil-bergamot-test-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

// ---------------------------------------------------------------------------------------------
// Catalog, language codes, routes
// ---------------------------------------------------------------------------------------------

#[test]
fn ids_parse_into_language_codes() {
    assert_eq!(catalog::parse_id("mt-bergamot/en_ja"), Some(("en".into(), "ja".into())));
    assert_eq!(catalog::parse_id("mt-bergamot/zh-hant_en"), Some(("zh-hant".into(), "en".into())));
    assert_eq!(catalog::parse_id("mt-bergamot/en_en"), None);
    assert_eq!(catalog::parse_id("mt-bergamot/enja"), None);
    assert_eq!(catalog::parse_id("mt/bergamot-en_ja"), None);
    assert_eq!(catalog::parse_id("mt-bergamot/EN_ja"), None);
    assert_eq!(catalog::parse_id("mt-bergamot/_ja"), None);
}

#[test]
fn language_codes_normalize_and_keep_chinese_scripts_apart() {
    assert_eq!(catalog::normalize_code(" zh-Hans ").as_deref(), Some("zh"));
    assert_eq!(catalog::normalize_code("zh_TW").as_deref(), Some("zh-hant"));
    assert_eq!(catalog::normalize_code("no").as_deref(), Some("nb"));
    assert_eq!(catalog::normalize_code(""), None);
    assert_eq!(catalog::canonical_case("zh-hant"), "zh-Hant");
    assert_eq!(catalog::canonical_case("pt-pt"), "pt-PT");
    assert_eq!(catalog::canonical_case("ja"), "ja");

    let d = test_directions();
    assert_eq!(d.lang("en-US").as_deref(), Some("en"));
    assert_eq!(d.lang("zh-Hant").as_deref(), Some("zh-hant"));
    assert_eq!(d.lang("zh-TW").as_deref(), Some("zh-hant"));
    assert_eq!(d.lang("zh").as_deref(), Some("zh"));
    assert_eq!(d.lang("fr"), None);

    // Traditional Chinese never falls back to the Simplified models.
    let only_simplified = Directions::from_entries([entry("mt-bergamot/zh_en", &["model.zhen.intgemm.alphas.bin", "vocab.zhen.spm"])].iter());
    assert_eq!(only_simplified.lang("zh-Hant"), None);
    assert_eq!(only_simplified.lang("zh-Hans").as_deref(), Some("zh"));
}

#[test]
fn routes_are_direct_pivot_or_unsupported() {
    let d = test_directions();
    let Ok(Route::Models { legs, pivot: false }) = d.route("en", "ja") else {
        panic!("en-ja is direct")
    };
    assert_eq!(legs.iter().map(|l| l.key.as_str()).collect::<Vec<_>>(), ["en_ja"]);
    assert_eq!(catalog::describe(&legs), "en-ja");

    let Ok(Route::Models { legs, pivot: true }) = d.route("ja", "zh") else {
        panic!("ja-zh pivots")
    };
    assert_eq!(legs.iter().map(|l| l.key.as_str()).collect::<Vec<_>>(), ["ja_en", "en_zh"]);
    assert_eq!(catalog::describe(&legs), "ja-zh via en");

    let Ok(Route::Models { legs, pivot: true }) = d.route("zh-Hant", "de") else {
        panic!("zh-Hant-de pivots")
    };
    assert_eq!(catalog::describe(&legs), "zh-Hant-de via en");

    assert_eq!(d.route("ja", "ja-JP").unwrap(), Route::Same);
    let e = d.route("", "ja").unwrap_err();
    assert_eq!(e.kind, MtErrorKind::UnsupportedPair);
    assert!(e.message.contains("source language"), "{}", e.message);
    assert_eq!(d.route("en", "fr").unwrap_err().kind, MtErrorKind::UnsupportedPair);
    // zh → ja needs zh→en, which this catalog lacks (only zh-hant→en).
    assert_eq!(d.route("zh", "ja").unwrap_err().kind, MtErrorKind::UnsupportedPair);
}

#[test]
fn pairs_list_directions_and_english_pivots_with_install_state() {
    let d = test_directions();
    let installed: HashSet<&str> = ["mt-bergamot/ja_en", "mt-bergamot/en_zh", "mt-bergamot/en_ja"].into();
    let mut calls = 0;
    let pairs = d.pairs(|id| {
        calls += 1;
        installed.contains(id)
    });
    assert_eq!(calls, d.len(), "one install lookup per catalog id");
    let find = |s: &str, t: &str| pairs.iter().find(|p| p.source == s && p.target == t);
    let direct = find("en", "ja").expect("en-ja");
    assert!(direct.installed && !direct.pivot && direct.ids == ["mt-bergamot/en_ja"]);
    let pivot = find("ja", "zh").expect("ja-zh via en");
    assert!(pivot.pivot && pivot.installed);
    assert_eq!(pivot.ids, ["mt-bergamot/ja_en", "mt-bergamot/en_zh"]);
    let half = find("zh-Hant", "ja").expect("zh-Hant-ja via en");
    assert!(half.pivot && !half.installed, "zh-hant_en is not installed");
    assert!(find("ja", "ja").is_none(), "no same-language pivot");
    assert!(find("de", "de").is_none());
    // Every pivot goes through English, never a direct pair twice.
    let mut seen = HashSet::new();
    for p in &pairs {
        assert!(seen.insert((p.source.clone(), p.target.clone())), "duplicate {p:?}");
    }
    let sorted = pairs
        .windows(2)
        .all(|w| (&w[0].source, &w[0].target) <= (&w[1].source, &w[1].target));
    assert!(sorted);
}

#[test]
fn model_files_follow_mozilla_naming() {
    let d = test_directions();
    let dir = temp_dir("roles");
    let enja = d.get("en", "ja").unwrap().clone();
    let jaen = d.get("ja", "en").unwrap().clone();
    // Nothing on disk yet: not installed.
    assert_eq!(catalog::model_files(&enja, &dir).unwrap_err().kind, MtErrorKind::NotInstalled);
    for f in enja.files.iter().chain(&jaen.files) {
        std::fs::write(dir.join(f), b"x").unwrap();
    }
    let split = catalog::model_files(&enja, &dir).unwrap();
    assert_eq!(split.gemm, "int8shiftAlphaAll");
    assert_eq!(split.vocabs[0], dir.join("srcvocab.enja.spm"));
    assert_eq!(split.vocabs[1], dir.join("trgvocab.enja.spm"));
    assert_eq!(split.lex.as_deref(), Some(dir.join("lex.50.50.enja.s2t.bin").as_path()));
    let shared = catalog::model_files(&jaen, &dir).unwrap();
    assert_eq!(shared.vocabs[0], shared.vocabs[1]);
    assert_eq!(catalog::gemm_precision("model.ende.intgemm8.bin"), Some("int8shiftAll"));
    assert_eq!(catalog::gemm_precision("model.ende.npz"), None);
    let float = Directions::from_entries([entry("mt-bergamot/fr_en", &["model.fren.bin", "vocab.fren.spm"])].iter());
    assert!(catalog::roles(float.get("fr", "en").unwrap())
        .unwrap_err()
        .message
        .contains("intgemm"));
    let _ = std::fs::remove_dir_all(dir);
}

/// The generated catalog (`mt-bergamot.json`) is usable: ids parse, every entry needs the runtime pack and has a
/// model, a shortlist and vocabularies, the important pairs exist, and every language is a canonical TS code.
#[test]
fn the_embedded_catalog_has_usable_bergamot_directions() {
    let catalog = Catalog::embedded().expect("embedded catalog");
    let runtime = catalog
        .get("runtime/bergamot")
        .expect("runtime/bergamot entry");
    assert_eq!(runtime.kind, EntryKind::Runtime);
    assert_eq!(runtime.status, PublishStatus::Unpublished);
    let entries: Vec<&CatalogEntry> = catalog
        .entries
        .iter()
        .filter(|e| e.family == catalog::FAMILY)
        .collect();
    assert!(
        entries.len() >= 100,
        "expected the Mozilla registry's ~106 directions, got {}",
        entries.len()
    );
    let d = Directions::from_catalog(&catalog);
    assert_eq!(d.len(), entries.len(), "every mt-bergamot id parses");
    for e in &entries {
        assert_eq!(e.requires, ["runtime/bergamot"], "{}", e.id);
        assert_eq!(e.status, PublishStatus::Published, "{}", e.id);
        assert!(e.files.iter().all(|f| f.sha256.is_some() && f.size > 0), "{} is fully pinned", e.id);
        let (s, t) = catalog::parse_id(&e.id).unwrap();
        assert!(s == "en" || t == "en", "{}: every Mozilla direction touches English", e.id);
        let roles = catalog::roles(d.get(&s, &t).unwrap()).unwrap_or_else(|err| panic!("{}: {err}", e.id));
        assert!(roles.lex.is_some(), "{} has a shortlist", e.id);
    }
    for (s, t) in [
        ("en", "ja"),
        ("ja", "en"),
        ("en", "ko"),
        ("ko", "en"),
        ("en", "zh"),
        ("zh", "en"),
        ("en", "zh-Hant"),
        ("en", "de"),
    ] {
        assert!(matches!(d.route(s, t), Ok(Route::Models { pivot: false, .. })), "{s}-{t} is direct");
    }
    assert!(
        matches!(d.route("ja", "ko"), Ok(Route::Models { pivot: true, .. })),
        "ja-ko pivots through English"
    );

    // Every language is a canonical code of the TS table (translation/__tests__/languages.fixture.json).
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../src/core/services/translation/__tests__/languages.fixture.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("skipped the TS cross-check: {} not found", path.display());
        return;
    };
    let fixture: Value = serde_json::from_str(&text).expect("fixture parses");
    let codes: HashSet<String> = fixture["languages"]
        .as_array()
        .expect("languages")
        .iter()
        .filter_map(|r| r["code"].as_str().map(str::to_string))
        .collect();
    for p in d.pairs(|_| false).iter().filter(|p| !p.pivot) {
        for code in [&p.source, &p.target] {
            assert!(codes.contains(code.as_str()), "Bergamot language {code} is not a canonical TS code");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------------------------

#[test]
fn requests_carry_id_op_and_fields() {
    let line = protocol::encode(7, protocol::translate(&["ja_en", "en_zh"], "こんにちは", Priority::Final));
    let v: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(
        v,
        json!({ "id": 7, "op": "translate", "keys": ["ja_en", "en_zh"], "text": "こんにちは", "priority": "final" })
    );
    assert!(!line.contains('\n'), "one request per line, even for multi-line text");
    let multi = protocol::encode(8, protocol::translate(&["en_ja"], "a\nb", Priority::Interim));
    assert!(!multi.contains('\n'));

    let files = catalog::ModelFiles {
        model: PathBuf::from("C:/m/model.enja.intgemm.alphas.bin"),
        lex: None,
        vocabs: [PathBuf::from("C:/m/src.spm"), PathBuf::from("C:/m/trg.spm")],
        gemm: "int8shiftAlphaAll",
    };
    let v: Value = serde_json::from_str(&protocol::encode(1, protocol::load("en_ja", &files))).unwrap();
    assert_eq!(v["op"], "load");
    assert_eq!(v["lex"], Value::Null);
    assert_eq!(v["vocabs"][1], "C:/m/trg.spm");
    assert_eq!(v["gemm"], "int8shiftAlphaAll");
    let v: Value = serde_json::from_str(&protocol::encode(2, protocol::hello())).unwrap();
    assert_eq!(v, json!({ "id": 2, "op": "hello", "protocol": protocol::PROTOCOL }));
    let v: Value = serde_json::from_str(&protocol::encode(3, protocol::cancel(9))).unwrap();
    assert_eq!(v, json!({ "id": 3, "op": "cancel", "target": 9 }));
}

#[test]
fn replies_parse_and_errors_map_to_mt_errors() {
    let ok = protocol::parse_reply(r#"{"id":3,"ok":true,"result":{"text":"hi","ms":4}}"#).unwrap();
    assert_eq!(ok.id, Some(3));
    assert_eq!(ok.outcome.unwrap()["text"], "hi");
    let err = protocol::parse_reply(r#" {"id":4,"ok":false,"error":{"kind":"not_loaded","message":"no model en_ja"}} "#).unwrap();
    let e = err.outcome.unwrap_err();
    assert!(e.is("not_loaded"));
    assert_eq!(e.to_mt().kind, MtErrorKind::Runtime);
    assert!(
        protocol::parse_reply(r#"{"id":null,"ok":false,"error":{"kind":"invalid_request","message":"bad json"}}"#)
            .unwrap()
            .id
            .is_none()
    );
    assert!(protocol::parse_reply("running 1 test").is_none());
    assert!(protocol::parse_reply("{not json").is_none());
    assert!(protocol::parse_reply(r#"{"id":5,"ok":false}"#)
        .unwrap()
        .outcome
        .unwrap_err()
        .is("internal"));

    let kind = |k: &str| {
        HelperError {
            kind: k.into(),
            message: "m".into(),
        }
        .to_mt()
        .kind
    };
    assert_eq!(kind("cancelled"), MtErrorKind::Cancelled);
    assert_eq!(kind("busy"), MtErrorKind::Busy);
    assert_eq!(kind("invalid_request"), MtErrorKind::InvalidInput);
    assert_eq!(kind("load_failed"), MtErrorKind::Runtime);
    assert!(HelperError {
        kind: "x".into(),
        message: " ".into()
    }
    .to_mt()
    .message
    .contains("(x)"));
}

#[test]
fn lines_are_read_with_a_cap() {
    let mut r = Cursor::new(b"a\r\nbb\n\nccc".to_vec());
    assert_eq!(protocol::read_line_capped(&mut r, 8).unwrap(), Line::Text("a".into()));
    assert_eq!(protocol::read_line_capped(&mut r, 8).unwrap(), Line::Text("bb".into()));
    assert_eq!(protocol::read_line_capped(&mut r, 8).unwrap(), Line::Text(String::new()));
    assert_eq!(protocol::read_line_capped(&mut r, 8).unwrap(), Line::Text("ccc".into()));
    assert_eq!(protocol::read_line_capped(&mut r, 8).unwrap(), Line::Eof);

    // Exactly the cap fits; one more byte is skipped up to the newline, and reading resumes after it.
    let mut r = std::io::BufReader::with_capacity(3, Cursor::new(b"1234\n12345\nok\n".to_vec()));
    assert_eq!(protocol::read_line_capped(&mut r, 4).unwrap(), Line::Text("1234".into()));
    assert_eq!(protocol::read_line_capped(&mut r, 4).unwrap(), Line::TooLong);
    assert_eq!(protocol::read_line_capped(&mut r, 4).unwrap(), Line::Text("ok".into()));
    assert_eq!(protocol::read_line_capped(&mut r, 4).unwrap(), Line::Eof);
}

// ---------------------------------------------------------------------------------------------
// Runtime pack checks
// ---------------------------------------------------------------------------------------------

#[test]
fn loaded_directions_fit_the_helper_memory_budget() {
    use super::client::{direction_mb, loaded_limit, MODEL_BUDGET_MB};
    use crate::services::mt_local::MAX_THREADS;
    use crate::services::process_host::recipes::BergamotOptions;
    assert_eq!(loaded_limit(6, 1), 6, "one worker: 6 directions (~1.1 GB)");
    assert_eq!(loaded_limit(6, 2), 3);
    assert_eq!(loaded_limit(6, 4), 2, "never fewer than the two legs of a pivot");
    assert_eq!(loaded_limit(1, 1), 1, "a smaller configured cap wins");

    // The peak is the limit itself (directions are unloaded before a load, never after), and it must stay under
    // the job's memory cap for every worker count, with room for the helper's own memory.
    let job_cap_mb = BergamotOptions::default()
        .memory_limit_mb
        .expect("the recipe caps the job") as usize;
    let max_loaded = EngineConfig::production().max_loaded;
    for threads in 1..=MAX_THREADS {
        let peak = loaded_limit(max_loaded, threads) * direction_mb(threads);
        assert!(peak <= MODEL_BUDGET_MB, "{threads} worker(s): {peak} MB over the model budget");
        assert!(peak < job_cap_mb, "{threads} worker(s): {peak} MB reaches the job cap of {job_cap_mb} MB");
    }
}

#[test]
fn cpu_requirements_follow_the_pack_arch() {
    let none = CpuFeatures::default();
    let avx2 = CpuFeatures {
        avx2: true,
        fma: true,
        avx512: false,
    };
    assert!(pack::arch_supported("sse2", none).is_ok());
    assert!(pack::arch_supported("", none).is_ok());
    assert!(pack::arch_supported("avx2", avx2).is_ok());
    assert!(pack::arch_supported("AVX2", none)
        .unwrap_err()
        .contains("AVX2"));
    assert!(pack::arch_supported(
        "avx2",
        CpuFeatures {
            avx2: true,
            fma: false,
            avx512: false
        }
    )
    .is_err());
    assert!(pack::arch_supported("avx512", avx2).is_err());
    assert!(pack::arch_supported("neon", avx2).is_err());
}

#[test]
fn packs_are_checked_before_the_helper_starts() {
    use sha2::{Digest, Sha256};
    let dir = temp_dir("pack");
    let exe = dir.join("sigil-bergamot.exe");
    std::fs::write(&exe, b"MZ fake helper").unwrap();
    let cpu = CpuFeatures {
        avx2: true,
        fma: true,
        avx512: false,
    };
    // No manifest: accepted (hand-assembled pack), the hello handshake still checks the protocol.
    assert_eq!(pack::check_pack(&dir, &exe, cpu).unwrap(), None);

    let sha = hex::encode(Sha256::digest(b"MZ fake helper"));
    let manifest = |protocol: u32, arch: &str, sha: &str, size: u64| {
        json!({ "name": "sigil-bergamot", "version": "t", "protocol": protocol, "arch": arch,
                "files": [{ "path": "sigil-bergamot.exe", "size": size, "sha256": sha }] })
        .to_string()
    };
    let write = |text: String| std::fs::write(dir.join(pack::MANIFEST), text).unwrap();

    write(manifest(protocol::PROTOCOL + 1, "avx2", &sha, 14));
    assert!(pack::check_pack(&dir, &exe, cpu)
        .unwrap_err()
        .message
        .contains("protocol"));
    write(manifest(protocol::PROTOCOL, "avx512", &sha, 14));
    assert!(pack::check_pack(&dir, &exe, cpu)
        .unwrap_err()
        .message
        .contains("AVX-512"));
    write(manifest(protocol::PROTOCOL, "avx2", &"0".repeat(64), 14));
    assert!(pack::check_pack(&dir, &exe, cpu)
        .unwrap_err()
        .message
        .contains("does not match"));
    write(manifest(protocol::PROTOCOL, "avx2", &sha, 14));
    let m = pack::check_pack(&dir, &exe, cpu)
        .unwrap()
        .expect("manifest");
    assert_eq!(m.arch, "avx2");
    // Verified once per (size, modified time): a changed file is hashed again.
    std::thread::sleep(Duration::from_millis(20));
    std::fs::write(&exe, b"MZ swapped!!!!").unwrap();
    assert!(pack::check_pack(&dir, &exe, cpu)
        .unwrap_err()
        .message
        .contains("does not match"));
    write("{ not json".into());
    assert!(pack::check_pack(&dir, &exe, cpu).is_err());
    let _ = std::fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------------------------------------
// Fake helper (protocol 1 over stdio)
// ---------------------------------------------------------------------------------------------

const ROLE_ENV: &str = "SIGIL_BG_FAKE";
const THREADS_ENV: &str = "SIGIL_BG_THREADS";
const ENTRY: &str = "services::mt_local::bergamot::tests::fake_bergamot_entry";

/// Entry point of the fake helper (only runs when re-invoked by a test with `SIGIL_BG_FAKE` set). Roles: `ok`,
/// `old-protocol` (answers hello with protocol 99), `slow-hello` (answers hello after 3 s), `cap<N>` (dies like a
/// job hitting its memory cap when a load would hold more than N models), `linger` (exits 1 s after its stdin
/// closed).
#[test]
#[ignore = "fake Bergamot helper process, started by the engine tests"]
fn fake_bergamot_entry() {
    let Ok(role) = std::env::var(ROLE_ENV) else { return };
    let threads: u32 = std::env::var(THREADS_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let cap: Option<usize> = role.strip_prefix("cap").and_then(|n| n.parse().ok());
    eprintln!("fake bergamot helper ready ({role})");
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    let mut loaded: Vec<String> = Vec::new();
    let (mut loads, mut translations) = (0u64, 0u64);
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let reply = |id: &Value, result: Result<Value, (&str, String)>| -> String {
            match result {
                Ok(v) => json!({ "id": id, "ok": true, "result": v }).to_string(),
                Err((k, m)) => json!({ "id": id, "ok": false, "error": { "kind": k, "message": m } }).to_string(),
            }
        };
        let msg: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(out, "{}", reply(&Value::Null, Err(("invalid_request", e.to_string()))));
                let _ = out.flush();
                continue;
            }
        };
        let id = msg["id"].clone();
        let result: Result<Value, (&str, String)> = match msg["op"].as_str().unwrap_or_default() {
            "hello" => {
                if role == "slow-hello" {
                    std::thread::sleep(Duration::from_secs(3));
                }
                Ok(json!({
                    "protocol": if role == "old-protocol" { 99 } else { protocol::PROTOCOL },
                    "helper": "sigil-bergamot fake", "engine": "fake", "threads": threads, "arch": "test",
                }))
            }
            "load" => {
                let key = msg["key"].as_str().unwrap_or_default().to_string();
                let paths = std::iter::once(&msg["model"]).chain(msg["vocabs"].as_array().into_iter().flatten());
                if key.starts_with("crash") {
                    std::process::exit(3);
                }
                if key.starts_with("slow") {
                    std::thread::sleep(Duration::from_millis(1500));
                }
                if let Some(cap) = cap {
                    if !loaded.contains(&key) && loaded.len() + 1 > cap {
                        eprintln!("fake bergamot helper: loading {key} next to {loaded:?} goes over the memory cap");
                        std::process::exit(4);
                    }
                }
                match paths
                    .map(|p| p.as_str().unwrap_or_default())
                    .find(|p| !Path::new(p).is_file())
                {
                    Some(missing) => Err(("load_failed", format!("cannot open {missing}"))),
                    None => {
                        loads += 1;
                        loaded.retain(|k| *k != key);
                        loaded.push(key.clone());
                        Ok(json!({ "key": key, "ms": 1 }))
                    }
                }
            }
            "unload" => {
                let key = msg["key"].as_str().unwrap_or_default();
                loaded.retain(|k| k != key);
                Ok(json!({ "key": key }))
            }
            "translate" => {
                let keys: Vec<String> = msg["keys"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|k| k.as_str().map(str::to_string))
                    .collect();
                let text = msg["text"].as_str().unwrap_or_default();
                if let Some(k) = keys.iter().find(|k| !loaded.contains(k)) {
                    Err(("not_loaded", format!("model {k} is not loaded")))
                } else if text.contains("__crash__") {
                    std::process::exit(3);
                } else if text.contains("__fail__") {
                    Err(("translate_failed", "the model failed".to_string()))
                } else {
                    if text.contains("__hang__") {
                        std::thread::sleep(Duration::from_secs(30));
                    }
                    if text.contains("__slow__") {
                        std::thread::sleep(Duration::from_millis(1500));
                    }
                    translations += 1;
                    Ok(json!({ "text": format!("<{}>{}", keys.join("+"), text), "ms": 1 }))
                }
            }
            "cancel" => Ok(json!({ "target": msg["target"], "cancelled": false })),
            "stats" => Ok(json!({ "loaded": loaded, "queued": 0, "running": 0, "threads": threads,
                                  "translations": translations, "loads": loads })),
            other => Err(("invalid_request", format!("unknown op {other:?}"))),
        };
        let _ = writeln!(out, "{}", reply(&id, result));
        let _ = out.flush();
    }
    eprintln!("fake bergamot helper: stdin closed");
    if role == "linger" {
        // Like the real helper finishing the work it runs before it exits.
        std::thread::sleep(Duration::from_millis(1000));
    }
    std::process::exit(0);
}

/// The test environment: `test_directions()`, model folders in a temp dir, the fake helper as the program.
struct FakeEnv {
    dirs: Arc<Directions>,
    root: PathBuf,
    installed: Mutex<HashSet<String>>,
    role: String,
    spec_error: Mutex<Option<MtError>>,
    /// How long a stop waits for the fake to exit after closing its stdin before killing it.
    stop_grace_ms: Mutex<u64>,
}

impl FakeEnv {
    /// Install folder of a catalog id (whether or not it counts as installed).
    fn model_dir(&self, id: &str) -> PathBuf {
        self.root.join(id.replace('/', "_"))
    }

    fn new(tag: &str, role: &str, installed: &[&str]) -> Arc<FakeEnv> {
        let dirs = Arc::new(test_directions());
        let root = temp_dir(tag);
        for d in dirs.iter() {
            let dir = root.join(d.id.replace('/', "_"));
            std::fs::create_dir_all(&dir).unwrap();
            for f in &d.files {
                std::fs::write(dir.join(f), b"model bytes").unwrap();
            }
        }
        Arc::new(FakeEnv {
            dirs,
            root,
            installed: Mutex::new(
                installed
                    .iter()
                    .map(|s| format!("mt-bergamot/{s}"))
                    .collect(),
            ),
            role: role.into(),
            spec_error: Mutex::new(None),
            stop_grace_ms: Mutex::new(2000),
        })
    }
}

impl Drop for FakeEnv {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

impl Environment for FakeEnv {
    fn directions(&self) -> Arc<Directions> {
        self.dirs.clone()
    }

    fn installed_path(&self, id: &str) -> Option<PathBuf> {
        self.installed
            .lock()
            .unwrap()
            .contains(id)
            .then(|| self.model_dir(id))
    }

    fn helper_spec(&self, name: &str, threads: u32) -> Result<HelperSpec, MtError> {
        if let Some(e) = self.spec_error.lock().unwrap().clone() {
            return Err(e);
        }
        Ok(HelperSpec {
            name: name.into(),
            program: std::env::current_exe().expect("test binary"),
            args: ["--exact", ENTRY, "--ignored", "--nocapture", "--test-threads=1", "-q"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            env: vec![(ROLE_ENV.into(), self.role.clone()), (THREADS_ENV.into(), threads.to_string())],
            stdio: StdioMode::Protocol,
            restart: process_host::RestartPolicy::OnFailure {
                max_restarts: 3,
                backoff_ms: 50,
            },
            stop_grace_ms: *self.stop_grace_ms.lock().unwrap(),
            ..Default::default()
        })
    }
}

fn test_config(name: &str) -> EngineConfig {
    EngineConfig {
        helper_name: name.into(),
        hello_timeout: Duration::from_secs(60),
        load_timeout: Duration::from_secs(30),
        final_timeout: Duration::from_secs(30),
        interim_timeout: Duration::from_secs(30),
        failure_cooldown: Duration::from_secs(1),
        restart_wait: Duration::from_secs(20),
        ..EngineConfig::production()
    }
}

fn request(text: &str, source: &str, target: &str, kind: MtKind, id: Option<&str>) -> MtRequest {
    MtRequest {
        text: text.into(),
        source: source.into(),
        target: target.into(),
        kind,
        request_id: id.map(str::to_string),
        threads: Some(1),
        ..Default::default()
    }
}

fn final_line(text: &str, source: &str, target: &str) -> MtRequest {
    request(text, source, target, MtKind::Final, None)
}

/// Stops the engine's helper when a test ends (also when it fails half-way).
struct Cleanup(Arc<Engine>);

impl Drop for Cleanup {
    fn drop(&mut self) {
        let name = self.0.config().helper_name.clone();
        let _ = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let _ = rt.block_on(process_host::stop(&name));
        })
        .join();
    }
}

fn engine(env: Arc<dyn Environment>, cfg: EngineConfig) -> (Arc<Engine>, Cleanup) {
    let e = Engine::new(env, cfg);
    (e.clone(), Cleanup(e))
}

// ---------------------------------------------------------------------------------------------
// Engine end to end
// ---------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn translates_direct_and_pivot_lines_loading_each_model_once() {
    let env = FakeEnv::new("e2e", "ok", &["en_ja", "ja_en", "en_zh"]);
    let (e, _cleanup) = engine(env, test_config("t-bergamot-e2e"));
    let statuses = Arc::new(Mutex::new(Vec::new()));
    let sink = statuses.clone();
    e.set_emitter(Box::new(move |s| sink.lock().unwrap().push(s.clone())));

    let r = e
        .translate(final_line("hello", "en", "ja"))
        .await
        .expect("en-ja");
    assert_eq!(r.text, "<en_ja>hello");
    assert_eq!(r.engine, "bergamot");
    assert_eq!(r.model, "en-ja");
    assert!(!r.pivot);
    let r = e
        .translate(final_line("  again  ", "en-US", "ja"))
        .await
        .expect("en-ja again");
    assert_eq!(r.text, "<en_ja>again", "input is trimmed");

    let r = e
        .translate(final_line("おはよう", "ja", "zh-Hans"))
        .await
        .expect("ja-zh via en");
    assert_eq!(r.text, "<ja_en+en_zh>おはよう");
    assert!(r.pivot);
    assert_eq!(r.model, "ja-zh via en");

    let stats = e.helper_stats().await.expect("stats");
    assert_eq!(stats.loads, 3, "en_ja, ja_en and en_zh loaded once each");
    assert_eq!(stats.translations, 3);
    let status = e.status();
    assert_eq!(status.state, MtEngineState::Ready);
    assert_eq!(status.model.as_deref(), Some("en-ja, en-zh, ja-en"));
    assert_eq!(status.threads, Some(1));
    let seen = statuses.lock().unwrap().clone();
    assert!(seen.iter().any(|s| s.state == MtEngineState::Loading), "loading was reported");
    assert!(seen.iter().all(|s| s.engine == "bergamot"));

    // Same language: returned unchanged without the helper.
    let same = e.translate(final_line("hi", "ja", "ja-JP")).await.unwrap();
    assert_eq!((same.text.as_str(), same.model.as_str()), ("hi", ""));

    e.stop().await;
    assert_eq!(e.status().state, MtEngineState::Idle);
    assert_eq!(
        process_host::get("t-bergamot-e2e").map(|i| i.state),
        Some(process_host::HelperState::Stopped)
    );
    // The next line starts it again.
    let r = e
        .translate(final_line("back", "ja", "en"))
        .await
        .expect("restarts on demand");
    assert_eq!(r.text, "<ja_en>back");
}

#[tokio::test(flavor = "multi_thread")]
async fn missing_models_and_pairs_fail_before_starting_the_helper() {
    let env = FakeEnv::new("missing", "ok", &["en_ja"]);
    let name = "t-bergamot-missing";
    let (e, _cleanup) = engine(env.clone(), test_config(name));

    let err = e.translate(final_line("x", "ja", "zh")).await.unwrap_err();
    assert_eq!(err.kind, MtErrorKind::NotInstalled);
    assert_eq!(err.missing, ["mt-bergamot/ja_en", "mt-bergamot/en_zh"]);
    assert_eq!(
        e.translate(final_line("x", "en", "fr"))
            .await
            .unwrap_err()
            .kind,
        MtErrorKind::UnsupportedPair
    );
    assert_eq!(
        e.translate(final_line("x", "", "ja"))
            .await
            .unwrap_err()
            .kind,
        MtErrorKind::UnsupportedPair
    );
    assert_eq!(
        e.translate(final_line("   ", "en", "ja"))
            .await
            .unwrap_err()
            .kind,
        MtErrorKind::InvalidInput
    );
    let long = "a".repeat(super::client::MAX_TEXT_CHARS + 1);
    assert_eq!(
        e.translate(final_line(&long, "en", "ja"))
            .await
            .unwrap_err()
            .kind,
        MtErrorKind::InvalidInput
    );
    assert!(
        process_host::get(name).is_none(),
        "no helper was started for requests that cannot succeed"
    );

    // The runtime pack is missing: the error names it.
    *env.spec_error.lock().unwrap() = Some(MtError::not_installed(vec!["runtime/bergamot".into()]));
    let err = e.translate(final_line("x", "en", "ja")).await.unwrap_err();
    assert_eq!(
        (err.kind, err.missing.clone()),
        (MtErrorKind::NotInstalled, vec!["runtime/bergamot".to_string()])
    );
    assert_eq!(e.status().state, MtEngineState::Error);

    // Pairs report what is installed.
    let pairs = e.pairs();
    assert!(pairs
        .iter()
        .any(|p| p.source == "en" && p.target == "ja" && p.installed));
    assert!(pairs
        .iter()
        .any(|p| p.source == "ja" && p.target == "zh" && p.pivot && !p.installed));
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_ends_a_request_at_once_and_the_helper_keeps_working() {
    let env = FakeEnv::new("cancel", "ok", &["en_ja"]);
    let (e, _cleanup) = engine(env, test_config("t-bergamot-cancel"));
    e.start(Some(1)).await.expect("starts");

    let slow = {
        let e = e.clone();
        tokio::spawn(async move {
            e.translate(request("__slow__", "en", "ja", MtKind::Interim, Some("mic:1:2#3")))
                .await
        })
    };
    tokio::time::sleep(Duration::from_millis(300)).await;
    let t0 = Instant::now();
    e.cancel(Some("mic:1:2#3"));
    let err = slow.await.unwrap().unwrap_err();
    assert!(err.is_cancelled(), "{err:?}");
    assert!(t0.elapsed() < Duration::from_millis(500), "cancel took {:?}", t0.elapsed());
    // Unknown ids and "cancel everything" with nothing in flight are harmless.
    e.cancel(Some("nope"));
    e.cancel(None);
    let r = e
        .translate(final_line("after", "en", "ja"))
        .await
        .expect("still works");
    assert_eq!(r.text, "<en_ja>after");
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_cancels_requests_in_flight() {
    let env = FakeEnv::new("stop", "ok", &["en_ja"]);
    let (e, _cleanup) = engine(env, test_config("t-bergamot-stop"));
    e.start(None).await.expect("starts");
    let slow = {
        let e = e.clone();
        tokio::spawn(async move { e.translate(final_line("__slow__", "en", "ja")).await })
    };
    tokio::time::sleep(Duration::from_millis(300)).await;
    e.stop().await;
    let err = slow.await.unwrap().unwrap_err();
    assert!(err.is_cancelled(), "a stop cancels, it does not restart the helper: {err:?}");
    assert_eq!(e.status().state, MtEngineState::Idle);
    assert_eq!(
        process_host::get("t-bergamot-stop").map(|i| i.state),
        Some(process_host::HelperState::Stopped)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_crash_is_retried_once_for_finals_and_the_helper_comes_back() {
    let env = FakeEnv::new("crash", "ok", &["en_ja", "crash_en"]);
    let (e, _cleanup) = engine(env.clone(), test_config("t-bergamot-crash"));
    let r = e.translate(final_line("before", "en", "ja")).await.unwrap();
    assert_eq!(r.text, "<en_ja>before");

    // A line that kills the helper: the final is tried twice, then fails with a runtime error.
    let err = e
        .translate(final_line("__crash__", "en", "ja"))
        .await
        .unwrap_err();
    assert_eq!(err.kind, MtErrorKind::Runtime, "{err:?}");
    assert!(err.message.contains("stopped"), "{}", err.message);
    // The helper restarts (process host or on demand) and later lines work, loading the model again.
    let r = e
        .translate(final_line("after", "en", "ja"))
        .await
        .expect("recovers");
    assert_eq!(r.text, "<en_ja>after");

    // An interim that kills it is not retried.
    let err = e
        .translate(request("__crash__", "en", "ja", MtKind::Interim, None))
        .await
        .unwrap_err();
    assert_eq!(err.kind, MtErrorKind::Runtime);
    assert!(e.translate(final_line("again", "en", "ja")).await.is_ok());

    // A model whose load kills the helper twice is refused from then on, with advice.
    let err = e
        .translate(final_line("x", "crash", "en"))
        .await
        .unwrap_err();
    assert_eq!(err.kind, MtErrorKind::Runtime);
    let err = e
        .translate(final_line("x", "crash", "en"))
        .await
        .unwrap_err();
    assert!(err.message.contains("twice"), "{}", err.message);
    assert!(e.translate(final_line("still", "en", "ja")).await.is_ok(), "other models keep working");

    // Reinstalling the model (its file changes) lifts the refusal: it is tried (and crashes) again.
    let model = env
        .model_dir("mt-bergamot/crash_en")
        .join("model.crashen.intgemm.alphas.bin");
    std::fs::write(&model, b"reinstalled model bytes").unwrap();
    let err = e
        .translate(final_line("x", "crash", "en"))
        .await
        .unwrap_err();
    assert!(!err.message.contains("twice"), "tried again after the reinstall: {}", err.message);
    let err = e
        .translate(final_line("x", "crash", "en"))
        .await
        .unwrap_err();
    assert!(err.message.contains("twice"), "{}", err.message);
    // So does a stop.
    e.stop().await;
    let err = e
        .translate(final_line("x", "crash", "en"))
        .await
        .unwrap_err();
    assert!(!err.message.contains("twice"), "tried again after a stop: {}", err.message);
}

#[tokio::test(flavor = "multi_thread")]
async fn helper_errors_and_protocol_mismatches_are_reported() {
    let env = FakeEnv::new("errors", "ok", &["en_ja"]);
    let (e, _cleanup) = engine(env, test_config("t-bergamot-errors"));
    let err = e
        .translate(final_line("__fail__", "en", "ja"))
        .await
        .unwrap_err();
    assert_eq!(err.kind, MtErrorKind::Runtime);
    assert!(err.message.contains("the model failed"), "{}", err.message);

    let old = FakeEnv::new("oldproto", "old-protocol", &["en_ja"]);
    let name = "t-bergamot-oldproto";
    let (e, _cleanup) = engine(old, test_config(name));
    let err = e.start(None).await.unwrap_err();
    assert!(err.message.contains("protocol 99"), "{}", err.message);
    assert_eq!(e.status().state, MtEngineState::Error);
    assert_ne!(
        process_host::get(name).map(|i| i.state),
        Some(process_host::HelperState::Running),
        "a helper with the wrong protocol is stopped"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_helper_that_stops_answering_is_restarted() {
    let env = FakeEnv::new("hang", "ok", &["en_ja"]);
    let name = "t-bergamot-hang";
    let cfg = EngineConfig {
        final_timeout: Duration::from_millis(1500),
        ..test_config(name)
    };
    let (e, _cleanup) = engine(env, cfg);
    let first = e.start(None).await.expect("starts");
    assert_eq!(first.state, MtEngineState::Ready);
    let pid_before = process_host::get(name).and_then(|i| i.pid);
    let err = e
        .translate(final_line("__hang__", "en", "ja"))
        .await
        .unwrap_err();
    assert!(err.message.contains("restarted"), "{}", err.message);
    let r = e
        .translate(final_line("fresh", "en", "ja"))
        .await
        .expect("a fresh helper answers");
    assert_eq!(r.text, "<en_ja>fresh");
    assert_ne!(process_host::get(name).and_then(|i| i.pid), pid_before);
}

#[tokio::test(flavor = "multi_thread")]
async fn interims_beyond_the_limit_are_busy_and_finals_are_not() {
    let env = FakeEnv::new("busy", "ok", &["en_ja"]);
    let cfg = EngineConfig {
        max_interims_in_flight: 1,
        ..test_config("t-bergamot-busy")
    };
    let (e, _cleanup) = engine(env, cfg);
    e.start(None).await.expect("starts");
    let slow = {
        let e = e.clone();
        tokio::spawn(async move {
            e.translate(request("__slow__", "en", "ja", MtKind::Interim, Some("a")))
                .await
        })
    };
    tokio::time::sleep(Duration::from_millis(200)).await;
    let err = e
        .translate(request("second", "en", "ja", MtKind::Interim, Some("b")))
        .await
        .unwrap_err();
    assert_eq!(err.kind, MtErrorKind::Busy);
    let r = e
        .translate(final_line("final", "en", "ja"))
        .await
        .expect("finals are never refused for interims");
    assert_eq!(r.text, "<en_ja>final");
    assert_eq!(slow.await.unwrap().unwrap().text, "<en_ja>__slow__");
    // The slot was released.
    assert!(e
        .translate(request("third", "en", "ja", MtKind::Interim, None))
        .await
        .is_ok());
}

#[tokio::test(flavor = "multi_thread")]
async fn least_recently_used_models_are_unloaded_beyond_the_cap() {
    let env = FakeEnv::new("lru", "ok", &["en_ja", "ja_en", "en_de"]);
    let cfg = EngineConfig {
        max_loaded: 2,
        ..test_config("t-bergamot-lru")
    };
    let (e, _cleanup) = engine(env, cfg);
    e.translate(final_line("1", "en", "ja")).await.unwrap();
    e.translate(final_line("2", "ja", "en")).await.unwrap();
    e.translate(final_line("3", "en", "de")).await.unwrap();
    let stats = e.helper_stats().await.unwrap();
    let mut loaded = stats.loaded.clone();
    loaded.sort();
    assert_eq!(loaded, ["en_de", "ja_en"], "en_ja was the least recently used");
    // It loads again when needed.
    assert_eq!(e.translate(final_line("4", "en", "ja")).await.unwrap().text, "<en_ja>4");
    assert_eq!(e.helper_stats().await.unwrap().loads, 4);
}

/// The fake dies like a job at its memory cap when it would hold more models than the limit: models are unloaded
/// before a load, never after it.
#[tokio::test(flavor = "multi_thread")]
async fn loads_make_room_first_so_the_helper_never_holds_more_than_the_limit() {
    let env = FakeEnv::new("peak", "cap2", &["en_ja", "ja_en", "en_zh", "de_en", "en_de"]);
    let name = "t-bergamot-peak";
    let cfg = EngineConfig {
        max_loaded: 2,
        ..test_config(name)
    };
    let (e, _cleanup) = engine(env, cfg);
    e.translate(final_line("1", "ja", "en")).await.unwrap();
    e.translate(final_line("2", "de", "en")).await.unwrap();
    let pid = process_host::get(name).and_then(|i| i.pid);
    // A pivot sharing one loaded leg: the other loaded model makes room first.
    assert_eq!(e.translate(final_line("3", "ja", "zh")).await.unwrap().text, "<ja_en+en_zh>3");
    // A pivot with two new legs: both loaded models make room.
    assert_eq!(e.translate(final_line("4", "de", "ja")).await.unwrap().text, "<de_en+en_ja>4");
    // An interim gets no retry, so going over the cap would lose it.
    assert_eq!(
        e.translate(request("5", "en", "de", MtKind::Interim, None))
            .await
            .unwrap()
            .text,
        "<en_de>5"
    );
    assert_eq!(process_host::get(name).and_then(|i| i.pid), pid, "the helper never went over its cap");
    let stats = e.helper_stats().await.unwrap();
    assert_eq!(stats.loads, 6);
    assert_eq!(stats.loaded.len(), 2);
}

/// A model a line is still using is never unloaded to make room: the next load waits for that line instead.
#[tokio::test(flavor = "multi_thread")]
async fn a_load_waits_for_lines_using_the_other_models() {
    let env = FakeEnv::new("pinned", "cap1", &["ja_en", "de_en"]);
    let name = "t-bergamot-pinned";
    let cfg = EngineConfig {
        max_loaded: 1,
        ..test_config(name)
    };
    let (e, _cleanup) = engine(env, cfg);
    e.start(Some(1)).await.expect("starts");
    let pid = process_host::get(name).and_then(|i| i.pid);
    let slow = {
        let e = e.clone();
        tokio::spawn(async move { e.translate(final_line("__slow__", "ja", "en")).await })
    };
    tokio::time::sleep(Duration::from_millis(300)).await;
    let r = e
        .translate(final_line("next", "de", "en"))
        .await
        .expect("waits for room");
    assert_eq!(r.text, "<de_en>next");
    assert_eq!(slow.await.unwrap().unwrap().text, "<ja_en>__slow__", "the line in flight kept its model");
    assert_eq!(process_host::get(name).and_then(|i| i.pid), pid, "the helper never went over its cap");
    assert_eq!(e.helper_stats().await.unwrap().loaded, ["de_en"]);
}

/// Cancelling a line while its model loads returns at once and frees its slot; the load still completes.
#[tokio::test(flavor = "multi_thread")]
async fn cancel_during_a_model_load_returns_at_once() {
    let env = FakeEnv::new("cancelload", "ok", &["slow_en"]);
    let cfg = EngineConfig {
        max_interims_in_flight: 1,
        ..test_config("t-bergamot-cancelload")
    };
    let (e, _cleanup) = engine(env, cfg);
    e.start(Some(1)).await.expect("starts");
    let first = {
        let e = e.clone();
        tokio::spawn(async move {
            e.translate(request("a", "slow", "en", MtKind::Interim, Some("line-1")))
                .await
        })
    };
    tokio::time::sleep(Duration::from_millis(300)).await;
    let t0 = Instant::now();
    e.cancel(Some("line-1"));
    let err = first.await.unwrap().unwrap_err();
    assert!(err.is_cancelled(), "{err:?}");
    assert!(t0.elapsed() < Duration::from_millis(500), "cancel took {:?}", t0.elapsed());
    // The interim slot is free at once (not `busy`), and the load the first line started is kept.
    let r = e
        .translate(request("b", "slow", "en", MtKind::Interim, Some("line-2")))
        .await
        .expect("the slot was released");
    assert_eq!(r.text, "<slow_en>b");
    let stats = e.helper_stats().await.unwrap();
    assert_eq!(
        (stats.loads, stats.translations),
        (1, 1),
        "loaded once; the cancelled line was never sent"
    );
}

/// Stopping the engine while a model loads is not the model's fault: it is never refused for that.
#[tokio::test(flavor = "multi_thread")]
async fn stops_during_a_load_do_not_count_against_the_model() {
    let env = FakeEnv::new("stopload", "ok", &["slow_en"]);
    // The stop kills the fake while it is still loading.
    *env.stop_grace_ms.lock().unwrap() = 100;
    let (e, _cleanup) = engine(env, test_config("t-bergamot-stopload"));
    for round in 0..3 {
        e.start(Some(1)).await.expect("starts");
        let line = {
            let e = e.clone();
            tokio::spawn(async move { e.translate(final_line("x", "slow", "en")).await })
        };
        tokio::time::sleep(Duration::from_millis(300)).await;
        e.stop().await;
        let err = line.await.unwrap().unwrap_err();
        assert!(err.is_cancelled(), "round {round}: {err:?}");
    }
    let r = e
        .translate(final_line("x", "slow", "en"))
        .await
        .expect("the model is not blamed for the stops");
    assert_eq!(r.text, "<slow_en>x");
}

/// A pipeline restart stops the old engine instance and starts the new one right away: the stop must not end the
/// helper the new start (and its warm-up line) uses.
#[tokio::test(flavor = "multi_thread")]
async fn a_start_right_after_a_stop_keeps_its_helper() {
    // The old helper takes a moment to exit, as the real one finishing its running work does.
    let env = FakeEnv::new("restart", "linger", &["en_ja"]);
    let name = "t-bergamot-restart";
    let (e, _cleanup) = engine(env, test_config(name));
    e.start(Some(1)).await.expect("starts");
    e.translate(final_line("before", "en", "ja")).await.unwrap();
    for round in 0..3 {
        let stopping = {
            let e = e.clone();
            tokio::spawn(async move { e.stop().await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        e.start(Some(1))
            .await
            .unwrap_or_else(|err| panic!("round {round}: the start after the stop failed: {err:?}"));
        let r = e
            .translate(final_line("warm", "en", "ja"))
            .await
            .unwrap_or_else(|err| panic!("round {round}: the warm-up line failed: {err:?}"));
        assert_eq!(r.text, "<en_ja>warm");
        stopping.await.unwrap();
        assert_eq!(e.status().state, MtEngineState::Ready, "round {round}");
        assert_eq!(
            process_host::get(name).map(|i| i.state),
            Some(process_host::HelperState::Running),
            "round {round}"
        );
        let pid = process_host::get(name).and_then(|i| i.pid);
        assert_eq!(
            e.translate(final_line("again", "en", "ja"))
                .await
                .unwrap()
                .text,
            "<en_ja>again"
        );
        assert_eq!(process_host::get(name).and_then(|i| i.pid), pid, "round {round}: no cold start");
    }
}

/// A stop while the helper is still starting ends the start at once instead of waiting for it.
#[tokio::test(flavor = "multi_thread")]
async fn a_stop_does_not_wait_out_a_slow_start() {
    let env = FakeEnv::new("slowhello", "slow-hello", &["en_ja"]);
    *env.stop_grace_ms.lock().unwrap() = 200;
    let name = "t-bergamot-slowhello";
    let (e, _cleanup) = engine(env, test_config(name));
    let starting = {
        let e = e.clone();
        tokio::spawn(async move { e.start(Some(1)).await })
    };
    tokio::time::sleep(Duration::from_millis(500)).await;
    let t0 = Instant::now();
    e.stop().await;
    assert!(t0.elapsed() < Duration::from_millis(1500), "the stop took {:?}", t0.elapsed());
    let err = starting.await.unwrap().unwrap_err();
    assert!(err.is_cancelled(), "{err:?}");
    assert_eq!(e.status().state, MtEngineState::Idle);
    assert_ne!(process_host::get(name).map(|i| i.state), Some(process_host::HelperState::Running));
}

/// Cancelling the line that started the helper returns at once, and the start it began completes for the next
/// line (a start dropped half-way would leave a running helper nobody can attach to until `restart_wait` runs out).
#[tokio::test(flavor = "multi_thread")]
async fn cancel_during_a_helper_start_keeps_the_start_for_the_next_line() {
    let env = FakeEnv::new("cancelstart", "slow-hello", &["en_ja"]);
    let name = "t-bergamot-cancelstart";
    let (e, _cleanup) = engine(env, test_config(name));
    let first = {
        let e = e.clone();
        tokio::spawn(async move {
            e.translate(request("a", "en", "ja", MtKind::Interim, Some("line-1")))
                .await
        })
    };
    tokio::time::sleep(Duration::from_millis(500)).await;
    let t0 = Instant::now();
    e.cancel(Some("line-1"));
    let err = first.await.unwrap().unwrap_err();
    assert!(err.is_cancelled(), "{err:?}");
    assert!(t0.elapsed() < Duration::from_millis(500), "cancel took {:?}", t0.elapsed());
    let pid = process_host::get(name).and_then(|i| i.pid);
    let t0 = Instant::now();
    let r = e
        .translate(final_line("next", "en", "ja"))
        .await
        .expect("the next line uses the helper the first one started");
    assert_eq!(r.text, "<en_ja>next");
    assert!(
        t0.elapsed() < Duration::from_secs(10),
        "the next line waited {:?} (restart_wait is {:?})",
        t0.elapsed(),
        test_config(name).restart_wait
    );
    assert_eq!(process_host::get(name).and_then(|i| i.pid), pid, "no second helper was started");
}

#[tokio::test(flavor = "multi_thread")]
async fn start_with_another_thread_count_restarts_the_helper() {
    let env = FakeEnv::new("threads", "ok", &["en_ja"]);
    let name = "t-bergamot-threads";
    let (e, _cleanup) = engine(env, test_config(name));
    let s = e.start(Some(1)).await.unwrap();
    assert_eq!(s.threads, Some(1));
    let pid = process_host::get(name).and_then(|i| i.pid);
    // Same count: kept.
    e.start(Some(1)).await.unwrap();
    assert_eq!(process_host::get(name).and_then(|i| i.pid), pid);
    let s = e.start(Some(3)).await.unwrap();
    assert_eq!(s.threads, Some(3));
    assert_ne!(process_host::get(name).and_then(|i| i.pid), pid);
    assert_eq!(e.helper_stats().await.unwrap().threads, 3);
    // Out-of-range counts are clamped (1..=4).
    let s = e.start(Some(99)).await.unwrap();
    assert_eq!(s.threads, Some(crate::services::mt_local::MAX_THREADS));
}

// ---------------------------------------------------------------------------------------------
// An external sigil-bergamot.exe through the engine (manual)
// ---------------------------------------------------------------------------------------------

/// The process host spec of the production recipe (`recipes::bergamot`) for an external exe, with the embedded
/// catalog's directions installed under `<models>/mt-bergamot/<key>/[<version>/]` (the model store layout), or
/// dummy files for the C++ fake build.
struct ExternalEnv {
    exe: PathBuf,
    dirs: Arc<Directions>,
    models: PathBuf,
}

impl Environment for ExternalEnv {
    fn directions(&self) -> Arc<Directions> {
        self.dirs.clone()
    }

    fn installed_path(&self, id: &str) -> Option<PathBuf> {
        let key = id.strip_prefix("mt-bergamot/")?;
        let base = self.models.join(key);
        let direction = self.dirs.iter().find(|d| d.id == id)?;
        let first = direction.files.first()?;
        if base.join(first).is_file() {
            return Some(base);
        }
        std::fs::read_dir(&base)
            .ok()?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .find(|p| p.join(first).is_file())
    }

    fn helper_spec(&self, name: &str, threads: u32) -> Result<HelperSpec, MtError> {
        Ok(crate::services::process_host::recipes::bergamot(
            &self.exe,
            &crate::services::process_host::recipes::BergamotOptions {
                name: name.into(),
                threads: Some(threads),
                cache_size: Some(super::CACHE_SIZE),
                ..Default::default()
            },
        ))
    }
}

/// `SIGIL_BERGAMOT_EXE=<pack>\sigil-bergamot.exe cargo test external_helper -- --ignored --nocapture`
/// - the C++ fake build (`build.ps1 -Fake`): protocol conformance with dummy model files;
/// - the real build with `SIGIL_TEST_MODELS_DIR` holding `mt-bergamot/en_ja` and `mt-bergamot/ja_en` (for example
///   `<app data>\models`): translations and per-line latency (P4-E acceptance: p95 under 150 ms).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs SIGIL_BERGAMOT_EXE (and SIGIL_TEST_MODELS_DIR for the real engine)"]
async fn external_helper_translates_through_the_engine() {
    let Some(exe) = std::env::var_os("SIGIL_BERGAMOT_EXE").map(PathBuf::from) else {
        eprintln!("skipped: SIGIL_BERGAMOT_EXE is not set");
        return;
    };
    let real_models = std::env::var_os("SIGIL_TEST_MODELS_DIR")
        .map(|d| PathBuf::from(d).join("mt-bergamot"))
        .filter(|d| d.join("en_ja").is_dir());
    // Keeps the C++ fake's dummy model folder alive for the test.
    let mut _dummy_models: Option<Arc<FakeEnv>> = None;
    let (env, lines): (ExternalEnv, Vec<(&str, &str, &str)>) = match &real_models {
        Some(models) => (
            ExternalEnv {
                exe,
                dirs: Arc::new(Directions::from_catalog(&Catalog::embedded().unwrap())),
                models: models.clone(),
            },
            vec![
                ("en", "ja", "Hello, can you hear me?"),
                ("en", "ja", "Let's meet at the plaza in five minutes."),
                ("en", "ja", "I think the event starts at eight tonight."),
                ("en", "ja", "Thank you for coming to the stream!"),
                ("en", "ja", "Could you say that one more time, slowly?"),
                ("ja", "en", "こんにちは、聞こえますか？"),
                ("ja", "en", "五分後に広場で会いましょう。"),
                ("ja", "en", "配信に来てくれてありがとう！"),
                ("ja", "en", "次はどのワールドに行くの？"),
                ("ja", "en", "もう一度ゆっくり言ってもらえますか？"),
            ],
        ),
        None => {
            let fake = FakeEnv::new("external", "unused", &[]);
            let env = ExternalEnv {
                exe,
                dirs: fake.dirs.clone(),
                models: fake.root.join("models"),
            };
            for d in env.dirs.iter() {
                let dir = env.models.join(&d.key).join("2.0");
                std::fs::create_dir_all(&dir).unwrap();
                for f in &d.files {
                    std::fs::write(dir.join(f), b"x").unwrap();
                }
            }
            _dummy_models = Some(fake);
            (
                env,
                vec![("en", "ja", "hello"), ("ja", "zh", "こんにちは"), ("en", "ja", "日本 \"q\"\n😀")],
            )
        }
    };
    let (e, _cleanup) = engine(Arc::new(env), test_config("t-bergamot-external"));
    let status = e
        .start(Some(1))
        .await
        .expect("the helper starts and answers hello");
    eprintln!("helper: {:?}", status.detail);
    // Warm up every direction once (model load and first-call allocations are not part of the line latency).
    for (s, t, text) in &lines {
        e.translate(final_line(text, s, t)).await.expect("warm-up");
    }
    let mut times = Vec::new();
    for (s, t, text) in &lines {
        let t0 = Instant::now();
        let r = e
            .translate(final_line(text, s, t))
            .await
            .expect("translates");
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        times.push(ms);
        eprintln!("{ms:7.1} ms  {s}->{t}  {text:?} -> {:?}{}", r.text, if r.pivot { " (pivot)" } else { "" });
        assert!(!r.text.trim().is_empty());
        if real_models.is_none() {
            let keys = match (*s, *t) {
                ("ja", "zh") => "ja_en+en_zh".to_string(),
                _ => format!("{s}_{t}"),
            };
            assert_eq!(r.text, format!("<{keys}>{}", text.trim()), "C++ fake echo");
        }
    }
    times.sort_by(f64::total_cmp);
    let p95 = times[((times.len() - 1) as f64 * 0.95).ceil() as usize];
    eprintln!("p50 {:.1} ms, p95 {p95:.1} ms over {} lines", times[(times.len() - 1) / 2], times.len());
    if real_models.is_some() {
        assert!(p95 < 150.0, "p95 {p95:.1} ms is over the 150 ms target");
    }
    e.stop().await;
    assert_eq!(e.status().state, MtEngineState::Idle);
}
