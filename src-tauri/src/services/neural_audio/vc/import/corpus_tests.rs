//! The importer's malformed-input corpus (`test-fixtures/vc/fuzz`, written by its deterministic
//! `generate_corpus.py`; plan T8: 200 files, 0 crashes or hangs).
//!
//! Every file is imported against the tiny templates on its own thread, under a watchdog, with
//! panics caught. Each must end in the result `corpus.json` expects: an error code, `ok` for the
//! valid controls, or `any` for random mutations (any result, as long as it is a result). The
//! test prints the corpus report (result and time per file, totals per result) with
//! `--nocapture`.
//!
//! [`community_corpus_report`] (ignored) is the T8 harness for a real community corpus: the import
//! through the isolated worker, then the decoded audio against PyTorch's.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde::Deserialize;

use super::{import_file, load_index, ErrorCode};

/// Longest a single import may take (a debug build; release is ~10x faster).
const PER_FILE_LIMIT: Duration = Duration::from_secs(60);

#[derive(Deserialize)]
struct Manifest {
    templates: String,
    files: Vec<CorpusFile>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct CorpusFile {
    pub file: String,
    pub expect: String,
    pub note: String,
}

pub(crate) fn fuzz_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("test-fixtures").join("vc").join("fuzz")
}

/// The templates folder and every corpus file with its expectation.
pub(crate) fn corpus() -> (PathBuf, Vec<(PathBuf, CorpusFile)>) {
    let dir = fuzz_dir();
    let m: Manifest = serde_json::from_slice(&std::fs::read(dir.join("corpus.json")).expect("corpus.json")).expect("corpus.json parses");
    let templates = dir.join(&m.templates);
    let files = m.files.into_iter().map(|f| (dir.join("corpus").join(&f.file), f)).collect();
    (templates, files)
}

/// How one import ended.
#[derive(Debug, PartialEq)]
pub(crate) enum Outcome {
    Ok,
    Err(ErrorCode, String),
    Panic(String),
    Hang,
}

impl Outcome {
    pub(crate) fn label(&self) -> String {
        match self {
            Outcome::Ok => "ok".into(),
            Outcome::Err(code, _) => code.as_str().into(),
            Outcome::Panic(_) => "PANIC".into(),
            Outcome::Hang => "HANG".into(),
        }
    }

    /// Whether this satisfies an expectation of `corpus.json`.
    pub(crate) fn meets(&self, expect: &str) -> bool {
        match self {
            Outcome::Panic(_) | Outcome::Hang => false,
            _ => expect == "any" || self.label() == expect,
        }
    }
}

/// Import `path` on its own thread with a time limit, catching panics.
fn import_guarded(path: &Path, templates: &Path, limit: Duration) -> (Outcome, Duration) {
    let (tx, rx) = mpsc::channel();
    let (p, t) = (path.to_path_buf(), templates.to_path_buf());
    let started = Instant::now();
    std::thread::Builder::new()
        .name("corpus-import".into())
        .spawn(move || {
            let result = std::panic::catch_unwind(|| import_file(&p, &t).map(|v| v.source.index_entry.clone()));
            let _ = tx.send(result);
        })
        .expect("spawn");
    let outcome = match rx.recv_timeout(limit) {
        Ok(Ok(Ok(_))) => Outcome::Ok,
        Ok(Ok(Err(e))) => Outcome::Err(e.code, e.message),
        Ok(Err(panic)) => Outcome::Panic(
            panic.downcast_ref::<String>().cloned().or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default(),
        ),
        Err(_) => Outcome::Hang,
    };
    (outcome, started.elapsed())
}

#[test]
fn the_fuzz_corpus_never_crashes_or_hangs() {
    let (templates, files) = corpus();
    assert_eq!(files.len(), 200, "the corpus has 200 files");
    let mut failures = Vec::new();
    let mut by_result: std::collections::BTreeMap<String, usize> = Default::default();
    let mut slowest = (Duration::ZERO, String::new());
    let started = Instant::now();
    eprintln!("{:<44} {:>8} {:<18} {:<18} {:>8}", "file", "bytes", "expected", "result", "ms");
    for (path, f) in &files {
        let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let (outcome, took) = import_guarded(path, &templates, PER_FILE_LIMIT);
        eprintln!("{:<44} {:>8} {:<18} {:<18} {:>8}", f.file, bytes, f.expect, outcome.label(), took.as_millis());
        *by_result.entry(outcome.label()).or_default() += 1;
        if took > slowest.0 {
            slowest = (took, f.file.clone());
        }
        if !outcome.meets(&f.expect) {
            failures.push(format!("{} ({}): expected {}, got {:?}", f.file, f.note, f.expect, outcome));
        }
    }
    eprintln!(
        "corpus report: {} files in {} ms; results {by_result:?}; slowest {} ({} ms)",
        files.len(),
        started.elapsed().as_millis(),
        slowest.1,
        slowest.0.as_millis()
    );
    assert!(failures.is_empty(), "{} corpus files misbehaved:\n{}", failures.len(), failures.join("\n"));
}

/// The valid controls import completely, and the archive's index is chosen and adoptable.
#[test]
fn the_corpus_controls_import_with_their_index() {
    let (templates, files) = corpus();
    let dir = fuzz_dir().join("corpus");
    let controls: Vec<&CorpusFile> = files.iter().map(|(_, f)| f).filter(|f| f.expect == "ok").collect();
    assert_eq!(controls.len(), 4);
    // Both state dicts carry the `_metadata` BUILD torch.save writes.
    let training = import_file(&dir.join("valid-training-G_2333.pth"), &templates).unwrap();
    assert_eq!((training.meta.checkpoint_kind.as_str(), training.meta.sample_rate), ("training", 16_000));
    assert!(training.meta.warnings.iter().any(|w| w.contains("inferred")), "{:?}", training.meta.warnings);
    let state = import_file(&dir.join("valid-state-dict.pth"), &templates).unwrap();
    assert_eq!(state.meta.checkpoint_kind, "state_dict");
    // Same weights, however they were packaged.
    assert_eq!(training.weights.f32_image, state.weights.f32_image);
    let archive = import_file(&dir.join("valid-voice-archive.zip"), &templates).unwrap();
    assert_eq!(archive.weights.f32_image, state.weights.f32_image);
    assert_eq!(archive.source.pth_entry.as_deref(), Some("Tiny/Tiny.pth"));
    assert_eq!(archive.source.index_entry.as_deref(), Some("Tiny/added_IVF2_Flat_nprobe_1_Tiny_v2.index"));
    let (index, warnings) = load_index(&archive, None);
    let index = index.unwrap_or_else(|| panic!("index not adopted: {warnings:?}"));
    assert_eq!(index.dim, 768);
    assert_eq!(index.origin, "archive:Tiny/added_IVF2_Flat_nprobe_1_Tiny_v2.index");
    // One voice saved at three training stages: the latest save, with a note, and its index.
    let epochs = import_file(&dir.join("valid-epochs-archive.zip"), &templates).unwrap();
    assert_eq!(epochs.source.pth_entry.as_deref(), Some("Tiny/Tiny_300e_6000s.pth"));
    assert_eq!(epochs.source.index_entry.as_deref(), Some("Tiny/added_IVF2_Flat_nprobe_1_Tiny_v2.index"));
    assert!(epochs.meta.warnings.iter().any(|w| w.contains("3 saves") && w.contains("step 6000")), "{:?}", epochs.meta.warnings);
    assert_eq!(epochs.weights.f32_image, state.weights.f32_image);
}

/// The fp16 relative L2 bound of plan T8.
const T8_REL_L2: f64 = 1e-2;
/// The per-model import budget of plan T8.
const T8_IMPORT_MS: u64 = 10_000;

/// A PyTorch reference bundle for an imported voice (`parity_test.py --bundle-out`): stored next to
/// the voice as `<file>.parity/reference.{json,bin}`, else made with the Python of
/// `SIGIL_TEST_VC_CORPUS_PYTORCH` (`1` = `python`) into `work`. `Err` says why there is none.
fn reference_bundle(source: &Path, record: &serde_json::Value, templates: &Path, work: &Path) -> Result<super::tests::Bundle, String> {
    use super::tests::Bundle;
    let stored = PathBuf::from(format!("{}.parity", source.display())).join("reference");
    if let Some(b) = Bundle::try_load(&stored) {
        return Ok(b);
    }
    let made = work.join("reference");
    if let Some(b) = Bundle::try_load(&made) {
        return Ok(b);
    }
    let Some(python) = std::env::var_os("SIGIL_TEST_VC_CORPUS_PYTORCH") else {
        return Err(format!("no reference: put one at {} or set SIGIL_TEST_VC_CORPUS_PYTORCH", stored.display()));
    };
    let python = if python == "1" { "python".into() } else { python };
    std::fs::create_dir_all(work).map_err(|e| e.to_string())?;
    // A voice archive: the .pth the importer chose, extracted (through the zip guard) for PyTorch.
    let pth = match record["source"]["pth_entry"].as_str() {
        Some(entry) => {
            let file = std::fs::File::open(source).map_err(|e| e.to_string())?;
            let mut zip = super::zipguard::GuardedZip::open(std::io::BufReader::new(file), super::zipguard::ZipLimits::VOICE_ARCHIVE)
                .map_err(|e| e.to_string())?;
            let bytes = zip.read(entry, 1 << 30).map_err(|e| e.to_string())?;
            let out = work.join("source.pth");
            std::fs::write(&out, bytes).map_err(|e| e.to_string())?;
            out
        }
        None => source.to_path_buf(),
    };
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("scripts").join("vc-templates").join("parity_test.py");
    let template = record["template"].as_str().unwrap_or_default();
    let out = std::process::Command::new(&python)
        .arg(&script)
        .arg("--pth")
        .arg(&pth)
        .args(["--template", template])
        .arg("--templates")
        .arg(templates)
        .arg("--bundle-out")
        .arg(work)
        .output()
        .map_err(|e| format!("{}: {e}", Path::new(&python).display()))?;
    // parity_test.py exits 1 when its own ONNX checks fail; the bundle is written either way.
    Bundle::try_load(&made).ok_or_else(|| {
        let err = String::from_utf8_lossy(&out.stderr);
        let tail: Vec<&str> = err.lines().rev().take(3).collect();
        format!("parity_test.py wrote no reference ({})", tail.into_iter().rev().collect::<Vec<_>>().join(" | "))
    })
}

/// T8 on a real community corpus (the "Dev: corpus report"). Every `.pth` / `.zip` in
/// `SIGIL_TEST_VC_CORPUS_DIR` (and its subfolders, one level deep) is imported the way the app
/// imports it: in the isolated worker (the test binary re-run as `vc-import-worker`, in a Job
/// Object with the app's memory cap and time limit), with the CPU smoke decode and the archive's or
/// the sibling index, against the bundled templates (or `SIGIL_VC_TEMPLATES_DIR`). Each imported
/// voice is then decoded on the CPU from its slot's `weights.f32.bin` and compared with PyTorch's
/// offline output ([`reference_bundle`], [`super::tests::chain_rel_l2`]): the relative L2 of the
/// fp16 chain the engine runs on a GPU (enc fp32, flowdec fp16), the all-fp16 chain and fp32.
///
/// Prints one row per voice and the T8 summary; writes `corpus-report.json` / `.md` and the slots
/// to `SIGIL_TEST_VC_CORPUS_OUT` (default: a temp folder). `SIGIL_TEST_VC_CORPUS_STRICT=1` asserts
/// T8: >= 95 % of the files import and match PyTorch within 1e-2 in the precision the engine runs
/// them (the fp16 mix, or fp32 for a voice the importer marked fp32-only); every other file is
/// refused with a specific code (not `io` / `job`); every import takes <= 10 s in the worker (time
/// it with `cargo test --release`).
#[test]
#[ignore = "needs SIGIL_TEST_VC_CORPUS_DIR (a folder of community .pth / .zip voices)"]
fn community_corpus_report() {
    use std::sync::atomic::AtomicBool;

    use super::worker::{self, WorkerJob, WorkerLimits};
    let Some(dir) = std::env::var_os("SIGIL_TEST_VC_CORPUS_DIR").map(PathBuf::from) else {
        eprintln!("SIGIL_TEST_VC_CORPUS_DIR is not set; skipped");
        return;
    };
    let templates = std::env::var_os("SIGIL_VC_TEMPLATES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("resources").join("vc-templates"));
    let out = std::env::var_os("SIGIL_TEST_VC_CORPUS_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join(format!("sigil-vc-corpus-{}", std::process::id())));
    let models = out.join("models");
    std::fs::create_dir_all(&models).unwrap();
    let mut files = Vec::new();
    let collect = |d: &Path| {
        let mut v: Vec<PathBuf> = std::fs::read_dir(d).into_iter().flatten().flatten().map(|e| e.path()).collect();
        v.sort();
        v
    };
    for p in collect(&dir) {
        if p.is_dir() {
            files.extend(collect(&p).into_iter().filter(|q| q.is_file()));
        } else {
            files.push(p);
        }
    }
    files.retain(|p| p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("pth") || e.eq_ignore_ascii_case("zip")));
    let command = worker::tests::fake("real");
    let mut rows = Vec::new();
    for (i, path) in files.iter().enumerate() {
        let slot_dir = models.join(format!("voice-{i:03}"));
        let index = if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("pth")) { super::slot::sibling_index(path) } else { None };
        let job = WorkerJob { index, ..WorkerJob::new(path.clone(), slot_dir.clone(), Some(templates.clone())) };
        let started = Instant::now();
        let result = worker::run_isolated_with(&command, &job, WorkerLimits::default(), &AtomicBool::new(false), None);
        let ms = started.elapsed().as_millis() as u64;
        let file = path.strip_prefix(&dir).unwrap_or(path).to_string_lossy().replace('\\', "/");
        let row = match &result {
            Ok(r) => {
                let fp16_ok = r["fp16_ok"].as_bool().unwrap_or(false);
                let parity = reference_bundle(path, r, &templates, &out.join("parity").join(format!("voice-{i:03}"))).and_then(|bundle| {
                    let template = super::layout::Template::load(&templates, r["template"].as_str().unwrap_or_default()).map_err(|e| e.to_string())?;
                    let weights = std::fs::read(slot_dir.join(super::WEIGHTS_FILE)).map_err(|e| e.to_string())?;
                    let speakers: Vec<f32> = std::fs::read(slot_dir.join(super::SPEAKERS_FILE))
                        .map_err(|e| e.to_string())?
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c))
                        .collect();
                    let dim = r["speakers"]["dim"].as_u64().unwrap_or(0) as usize;
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| super::tests::chain_rel_l2(&template, &weights, &speakers, dim, &bundle)))
                        .map_err(|p| format!("decode failed: {}", p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()))
                });
                // The precision the engine runs: the fp16 mix, or fp32 for an fp32-only voice.
                let (engine_rel_l2, parity_note) = match &parity {
                    Ok(c) if fp16_ok => (c.mixed, None),
                    Ok(c) => (Some(c.fp32), None),
                    Err(e) => (None, Some(e.clone())),
                };
                let t8 = engine_rel_l2.is_some_and(|e| e <= T8_REL_L2) && ms <= T8_IMPORT_MS;
                serde_json::json!({
                    "file": file, "result": "ok", "ms": ms, "worker": r["worker"], "template": r["template"],
                    "kind": r["meta"]["checkpoint_kind"], "embedder": r["meta"]["embedder"], "vocoder": r["meta"]["vocoder"],
                    "dtype": r["checkpoint_dtype"], "fp16_ok": fp16_ok, "index": r["index"]["origin"],
                    "rel_l2": parity.as_ref().ok().map(|c| serde_json::json!({ "engine": engine_rel_l2, "mixed": c.mixed, "fp16": c.fp16, "fp32": c.fp32 })),
                    "parity_note": parity_note, "t8": t8, "warnings": r["warnings"],
                })
            }
            Err(e) => serde_json::json!({ "file": file, "result": e.code.as_str(), "ms": ms, "reason": e.message, "t8": false }),
        };
        let detail = match (row["rel_l2"]["engine"].as_f64(), row["parity_note"].as_str(), row["reason"].as_str()) {
            (Some(l2), _, _) => format!("{}  rel L2 {l2:.2e}", row["template"].as_str().unwrap_or("")),
            (None, Some(note), _) => format!("{}  {note}", row["template"].as_str().unwrap_or("")),
            (_, _, Some(reason)) => reason.to_string(),
            _ => String::new(),
        };
        eprintln!("{:<60} {:<18} {:>6} ms  {detail}", file, row["result"].as_str().unwrap_or("?"), ms);
        rows.push(row);
    }
    let n = rows.len();
    let imported = rows.iter().filter(|r| r["result"] == "ok").count();
    let passing = rows.iter().filter(|r| r["t8"] == true).count();
    let unmeasured = rows.iter().filter(|r| r["result"] == "ok" && r["rel_l2"].is_null()).count();
    let slow: Vec<&serde_json::Value> = rows.iter().filter(|r| r["result"] == "ok" && r["ms"].as_u64().unwrap_or(0) > T8_IMPORT_MS).collect();
    let vague: Vec<&serde_json::Value> = rows.iter().filter(|r| matches!(r["result"].as_str(), Some("io" | "job"))).collect();
    let peak = rows.iter().filter_map(|r| r["worker"]["peak_memory_mb"].as_f64()).fold(0.0, f64::max);
    let share = if n == 0 { 0.0 } else { passing as f64 / n as f64 };
    let summary = format!(
        "T8 corpus: {passing}/{n} pass ({:.1} %, target >= 95 %): imported {imported}, within rel L2 {T8_REL_L2:.0e} of PyTorch in the engine's precision and {} s; \
         {unmeasured} imported without a PyTorch reference; {} over the import budget; {} refused without a specific code; worker peak {peak:.0} MB; templates {}",
        share * 100.0,
        T8_IMPORT_MS / 1000,
        slow.len(),
        vague.len(),
        templates.display()
    );
    eprintln!("{summary}");
    let mut md = format!("# Voice import corpus report\n\n{summary}\n\n| file | result | ms | peak MB | rel L2 (engine / fp16 / fp32) | template / reason |\n| --- | --- | --- | --- | --- | --- |\n");
    for r in &rows {
        let detail = r.get("template").filter(|t| !t.is_null()).or(r.get("reason")).and_then(|v| v.as_str()).unwrap_or("").replace('|', "/");
        let l2 = |k: &str| r["rel_l2"][k].as_f64().map_or("-".to_string(), |v| format!("{v:.2e}"));
        let l2 = if r["rel_l2"].is_null() { r["parity_note"].as_str().unwrap_or("-").replace('|', "/") } else { format!("{} / {} / {}", l2("engine"), l2("fp16"), l2("fp32")) };
        let peak = r["worker"]["peak_memory_mb"].as_f64().map_or("-".to_string(), |v| format!("{v:.0}"));
        md.push_str(&format!("| {} | {} | {} | {peak} | {l2} | {detail} |\n", r["file"].as_str().unwrap_or("?"), r["result"].as_str().unwrap_or("?"), r["ms"]));
    }
    std::fs::write(out.join("corpus-report.json"), serde_json::to_vec_pretty(&serde_json::json!({ "summary": summary, "rows": rows })).unwrap()).unwrap();
    std::fs::write(out.join("corpus-report.md"), md).unwrap();
    eprintln!("report: {}", out.join("corpus-report.md").display());
    if std::env::var_os("SIGIL_TEST_VC_CORPUS_STRICT").is_some() {
        assert!(share >= 0.95, "{summary}");
        assert!(slow.is_empty(), "{summary}");
        assert!(vague.is_empty(), "{summary}");
    }
}
