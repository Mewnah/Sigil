//! Tests of the Pocket TTS engine and of the cloning worker both P6-I engines share (P6-I).
//!
//! The worker runs against a scripted backend and an in-memory sink (ordering, warm-up, reload,
//! cancel, restart, panics, idle unload). Pocket's own pieces (request options, the `extra` JSON,
//! package discovery, built-in voices) are tested without the model; the real-model test
//! (`#[ignore]`) needs `tts/pocket-tts-en-int8` and the sherpa-onnx runtime in the model store of
//! `SIGIL_APP_DATA_DIR`, plus `SIGIL_TEST_MODELS_DIR/sherpa-onnx/clips/en.wav` as the reference:
//! `cargo test --manifest-path src-tauri/Cargo.toml local_tts -- --ignored --test-threads=1`.
//! **Release gate:** nothing else has run the real sherpa Pocket package (its file layout, the
//! `test_wavs/` built-in voices, the `temperature` key in `extra` come from reading sherpa-onnx's
//! sources); the test prints the package's file list, which belongs in the release notes of the
//! first run.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::sherpa::{self, PocketFiles, PocketOptions};
use super::worker::{option_f32, option_i64, option_str, CloneBackend, CloneWorker};
use super::*;
use crate::services::local_tts::engine::{PcmWriter, SinkPort, CANCELLED};
use crate::services::tts_sink::SinkError;

// -------------------------------------------------------------------------------------------------
// In-memory sink
// -------------------------------------------------------------------------------------------------

#[derive(Default)]
struct SinkState {
    begun: HashSet<String>,
    cancelled: HashSet<String>,
    pushed: HashMap<String, Vec<f32>>,
    order: Vec<String>,
}

#[derive(Default)]
struct FakeSink {
    state: Arc<Mutex<SinkState>>,
    queued_ms: AtomicU32,
}

impl FakeSink {
    fn begin(&self, id: &str) {
        let mut s = self.state.lock().unwrap();
        s.begun.insert(id.to_string());
        s.order.push(id.to_string());
    }

    fn cancel(&self, id: &str) {
        self.state.lock().unwrap().cancelled.insert(id.to_string());
    }

    fn samples(&self, id: &str) -> usize {
        self.state
            .lock()
            .unwrap()
            .pushed
            .get(id)
            .map_or(0, Vec::len)
    }
}

struct FakeWriter {
    id: String,
    state: Arc<Mutex<SinkState>>,
}

impl PcmWriter for FakeWriter {
    fn push(&mut self, samples: &[f32], _rate: u32) -> Result<(), SinkError> {
        let mut s = self.state.lock().unwrap();
        if s.cancelled.contains(&self.id) {
            return Err(SinkError::Cancelled);
        }
        s.pushed
            .entry(self.id.clone())
            .or_default()
            .extend_from_slice(samples);
        Ok(())
    }

    fn is_cancelled(&self) -> bool {
        self.state.lock().unwrap().cancelled.contains(&self.id)
    }
}

impl SinkPort for FakeSink {
    fn writer(&self, utterance_id: &str) -> Result<Box<dyn PcmWriter>, SinkError> {
        if !self.state.lock().unwrap().begun.contains(utterance_id) {
            return Err(SinkError::NotFound(utterance_id.to_string()));
        }
        Ok(Box::new(FakeWriter {
            id: utterance_id.to_string(),
            state: self.state.clone(),
        }))
    }

    fn order(&self) -> Vec<String> {
        self.state.lock().unwrap().order.clone()
    }

    fn queued_ms(&self) -> u32 {
        self.queued_ms.load(Ordering::SeqCst)
    }
}

// -------------------------------------------------------------------------------------------------
// Scripted backend
// -------------------------------------------------------------------------------------------------

#[derive(Default)]
struct Script {
    loads: AtomicUsize,
    drops: Arc<AtomicUsize>,
    warm_ups: AtomicUsize,
    /// Chunks synthesized, in order.
    texts: Mutex<Vec<String>>,
    /// Set while a `block` chunk runs.
    blocking: AtomicBool,
}

struct FakeModel {
    drops: Arc<AtomicUsize>,
}

impl Drop for FakeModel {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Clone)]
struct FakeBackend(Arc<Script>);

impl CloneBackend for FakeBackend {
    type Model = FakeModel;

    fn name(&self) -> &'static str {
        "fake"
    }

    fn key(&self, request: &CloningTtsRequest) -> Result<String, String> {
        if request.model == "bad" {
            return Err("unknown model".into());
        }
        Ok(format!("{}|{:?}", request.model, request.threads))
    }

    fn load(&self, _request: &CloningTtsRequest, _cancel: &AtomicBool) -> Result<FakeModel, String> {
        self.0.loads.fetch_add(1, Ordering::SeqCst);
        Ok(FakeModel { drops: self.0.drops.clone() })
    }

    fn warm_up(&self, _model: &mut FakeModel, request: &CloningTtsRequest, _cancel: &AtomicBool) -> Result<(), String> {
        self.0.warm_ups.fetch_add(1, Ordering::SeqCst);
        if request.reference_voice_id == "missing" {
            return Err("voice not found".into());
        }
        Ok(())
    }

    fn synthesize(
        &self,
        _model: &mut FakeModel,
        _request: &CloningTtsRequest,
        text: &str,
        cancel: &AtomicBool,
        out: &mut dyn FnMut(&[f32], u32) -> bool,
    ) -> Result<bool, String> {
        self.0.texts.lock().unwrap().push(text.to_string());
        if text.contains("panic") {
            panic!("scripted panic");
        }
        if text.contains("block") {
            self.0.blocking.store(true, Ordering::SeqCst);
            let started = Instant::now();
            while !cancel.load(Ordering::Acquire) {
                assert!(started.elapsed() < Duration::from_secs(10), "never cancelled");
                std::thread::sleep(Duration::from_millis(2));
            }
            self.0.blocking.store(false, Ordering::SeqCst);
            return Ok(false);
        }
        // 1.5 s of audio at 16 kHz in one call: the worker splits it into pushes of at most 1 s.
        let audio = vec![0.25f32; 24_000];
        Ok(out(&audio, 16_000))
    }
}

fn setup(idle: Duration) -> (CloneWorker<FakeBackend>, Arc<Script>, Arc<FakeSink>) {
    let script = Arc::new(Script::default());
    let sink = Arc::new(FakeSink {
        queued_ms: AtomicU32::new(10_000),
        ..FakeSink::default()
    });
    let worker = CloneWorker::new(Arc::new(FakeBackend(script.clone())), sink.clone(), idle);
    (worker, script, sink)
}

fn req(text: &str) -> CloningTtsRequest {
    CloningTtsRequest {
        model: "m".into(),
        reference_voice_id: "alice".into(),
        text: text.into(),
        ..CloningTtsRequest::default()
    }
}

fn wait_until(what: &str, f: impl Fn() -> bool) {
    let started = Instant::now();
    while !f() {
        assert!(started.elapsed() < Duration::from_secs(10), "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

// -------------------------------------------------------------------------------------------------
// Worker
// -------------------------------------------------------------------------------------------------

#[test]
fn a_sentence_goes_into_its_utterance_and_is_reported() {
    let (worker, script, sink) = setup(Duration::from_secs(60));
    sink.begin("u1");
    let report = worker
        .speak(req("Hello there."), "u1".into())
        .blocking_recv()
        .unwrap()
        .unwrap();
    assert_eq!(report.sample_rate, 16_000);
    assert_eq!(report.frames, 24_000);
    assert_eq!(sink.samples("u1"), 24_000);
    assert!(report.first_audio_ms <= report.total_ms);
    assert_eq!(script.loads.load(Ordering::SeqCst), 1);
    assert_eq!(script.warm_ups.load(Ordering::SeqCst), 1, "a load prepares the voice");
    // The same model serves the next sentence without a reload.
    sink.begin("u2");
    worker
        .speak(req("Again."), "u2".into())
        .blocking_recv()
        .unwrap()
        .unwrap();
    assert_eq!(script.loads.load(Ordering::SeqCst), 1);
    // A sentence needs a begun utterance.
    let err = worker
        .speak(req("Nobody."), "nope".into())
        .blocking_recv()
        .unwrap()
        .unwrap_err();
    assert!(err.contains("nope"), "{err}");
    let err = worker
        .speak(req("Nobody."), String::new())
        .blocking_recv()
        .unwrap()
        .unwrap_err();
    assert!(err.contains("utteranceId"), "{err}");
}

#[test]
fn an_empty_text_loads_and_prepares_the_voice_without_an_utterance() {
    let (worker, script, _sink) = setup(Duration::from_secs(60));
    let report = worker
        .speak(req(""), String::new())
        .blocking_recv()
        .unwrap()
        .unwrap();
    assert_eq!(report.frames, 0);
    assert_eq!(script.loads.load(Ordering::SeqCst), 1);
    assert_eq!(script.warm_ups.load(Ordering::SeqCst), 1);
    // Already loaded: the next warm-up prepares the (maybe new) voice only.
    worker
        .speak(req("  "), String::new())
        .blocking_recv()
        .unwrap()
        .unwrap();
    assert_eq!(script.loads.load(Ordering::SeqCst), 1);
    assert_eq!(script.warm_ups.load(Ordering::SeqCst), 2);
    // Errors of the load step reach the caller (Start shows them).
    let mut missing = req("");
    missing.reference_voice_id = "missing".into();
    assert_eq!(
        worker
            .speak(missing, String::new())
            .blocking_recv()
            .unwrap()
            .unwrap_err(),
        "voice not found"
    );
    let mut bad = req("");
    bad.model = "bad".into();
    assert_eq!(
        worker
            .speak(bad, String::new())
            .blocking_recv()
            .unwrap()
            .unwrap_err(),
        "unknown model"
    );
}

#[test]
fn a_voice_that_cannot_be_prepared_keeps_the_loaded_model() {
    let (worker, script, _sink) = setup(Duration::from_secs(60));
    let mut missing = req("");
    missing.reference_voice_id = "missing".into();
    assert_eq!(
        worker
            .speak(missing, String::new())
            .blocking_recv()
            .unwrap()
            .unwrap_err(),
        "voice not found"
    );
    assert_eq!(script.loads.load(Ordering::SeqCst), 1);
    // The voice is fixed: the model is still there.
    worker
        .speak(req(""), String::new())
        .blocking_recv()
        .unwrap()
        .unwrap();
    assert_eq!(script.loads.load(Ordering::SeqCst), 1, "the model stayed loaded");
    assert_eq!(script.drops.load(Ordering::SeqCst), 0);
}

#[test]
fn another_model_key_reloads_and_frees_the_previous_model() {
    let (worker, script, sink) = setup(Duration::from_secs(60));
    sink.begin("a");
    sink.begin("b");
    worker
        .speak(req("One."), "a".into())
        .blocking_recv()
        .unwrap()
        .unwrap();
    let mut other = req("Two.");
    other.threads = Some(2);
    worker
        .speak(other, "b".into())
        .blocking_recv()
        .unwrap()
        .unwrap();
    assert_eq!(script.loads.load(Ordering::SeqCst), 2);
    assert_eq!(script.drops.load(Ordering::SeqCst), 1, "the first model was freed");
    worker.unload().blocking_recv().unwrap().unwrap();
    assert_eq!(script.drops.load(Ordering::SeqCst), 2);
}

#[test]
fn cancel_stops_the_running_sentence_and_answers_the_waiting_ones() {
    let (worker, script, sink) = setup(Duration::from_secs(60));
    for id in ["run", "wait1", "wait2"] {
        sink.begin(id);
    }
    let running = worker.speak(req("block here"), "run".into());
    wait_until("the blocking chunk", || script.blocking.load(Ordering::SeqCst));
    let w1 = worker.speak(req("Later."), "wait1".into());
    let w2 = worker.speak(req("Later."), "wait2".into());
    // One utterance first: only it is cancelled.
    worker.cancel(Some("wait1"));
    assert_eq!(w1.blocking_recv().unwrap().unwrap_err(), CANCELLED);
    assert_eq!(worker.waiting(), 1);
    // Then everything.
    worker.cancel(None);
    assert_eq!(running.blocking_recv().unwrap().unwrap_err(), CANCELLED);
    assert_eq!(w2.blocking_recv().unwrap().unwrap_err(), CANCELLED);
    // A sink cancel (stop / skip) stops a sentence too.
    sink.begin("gone");
    sink.cancel("gone");
    assert_eq!(
        worker
            .speak(req("Bye."), "gone".into())
            .blocking_recv()
            .unwrap()
            .unwrap_err(),
        CANCELLED
    );
    // The worker still serves.
    sink.begin("after");
    worker
        .speak(req("Still here."), "after".into())
        .blocking_recv()
        .unwrap()
        .unwrap();
}

#[test]
fn waiting_sentences_run_in_the_sinks_play_order() {
    let (worker, script, sink) = setup(Duration::from_secs(60));
    sink.begin("first-block");
    let blocker = worker.speak(req("block"), "first-block".into());
    wait_until("the blocking chunk", || script.blocking.load(Ordering::SeqCst));
    // Begun in the order b, a; the requests arrive a, b.
    sink.begin("b");
    sink.begin("a");
    let ra = worker.speak(req("A sentence."), "a".into());
    let rb = worker.speak(req("B sentence."), "b".into());
    worker.cancel(Some("first-block"));
    assert_eq!(blocker.blocking_recv().unwrap().unwrap_err(), CANCELLED);
    rb.blocking_recv().unwrap().unwrap();
    ra.blocking_recv().unwrap().unwrap();
    let texts = script.texts.lock().unwrap().clone();
    assert_eq!(texts, ["block", "B sentence.", "A sentence."]);
}

#[test]
fn an_unload_with_a_request_behind_it_keeps_the_model() {
    let (worker, script, sink) = setup(Duration::from_secs(60));
    sink.begin("x");
    worker
        .speak(req("Load."), "x".into())
        .blocking_recv()
        .unwrap()
        .unwrap();
    sink.begin("block");
    let blocker = worker.speak(req("block"), "block".into());
    wait_until("the blocking chunk", || script.blocking.load(Ordering::SeqCst));
    // Stop then start while busy: unload, then a warm-up queue behind the running sentence.
    let unload = worker.unload();
    let start = worker.speak(req(""), String::new());
    worker.cancel(Some("block"));
    assert_eq!(blocker.blocking_recv().unwrap().unwrap_err(), CANCELLED);
    unload.blocking_recv().unwrap().unwrap();
    start.blocking_recv().unwrap().unwrap();
    assert_eq!(script.loads.load(Ordering::SeqCst), 1, "the restart kept the model");
    assert_eq!(script.drops.load(Ordering::SeqCst), 0);
}

#[test]
fn a_panic_fails_its_sentence_and_the_worker_goes_on() {
    let (worker, script, sink) = setup(Duration::from_secs(60));
    sink.begin("p");
    let err = worker
        .speak(req("panic now"), "p".into())
        .blocking_recv()
        .unwrap()
        .unwrap_err();
    assert!(err.contains("scripted panic"), "{err}");
    assert_eq!(script.drops.load(Ordering::SeqCst), 1, "the model was dropped");
    sink.begin("ok");
    worker
        .speak(req("Fine."), "ok".into())
        .blocking_recv()
        .unwrap()
        .unwrap();
    assert_eq!(script.loads.load(Ordering::SeqCst), 2, "loaded again");
}

#[test]
fn the_first_clause_is_spoken_alone_when_nothing_is_queued() {
    let (worker, script, sink) = setup(Duration::from_secs(60));
    let long = "This is a fairly long opening clause, and then the rest of the sentence follows here.";
    sink.queued_ms.store(0, Ordering::SeqCst);
    sink.begin("first");
    worker
        .speak(req(long), "first".into())
        .blocking_recv()
        .unwrap()
        .unwrap();
    assert_eq!(script.texts.lock().unwrap().len(), 2, "split at the first clause");
    sink.queued_ms.store(10_000, Ordering::SeqCst);
    sink.begin("later");
    worker
        .speak(req(long), "later".into())
        .blocking_recv()
        .unwrap()
        .unwrap();
    assert_eq!(script.texts.lock().unwrap().len(), 3, "whole sentence when audio is queued");
}

#[test]
fn an_idle_model_is_freed() {
    let (worker, script, sink) = setup(Duration::from_millis(50));
    sink.begin("i");
    worker
        .speak(req("Idle soon."), "i".into())
        .blocking_recv()
        .unwrap()
        .unwrap();
    wait_until("the idle unload", || script.drops.load(Ordering::SeqCst) == 1);
}

#[test]
fn a_started_model_is_held_until_stop() {
    let (worker, script, sink) = setup(Duration::from_millis(30));
    // Start (an empty text) holds the model: long gaps between sentences never unload it.
    worker
        .speak(req(""), String::new())
        .blocking_recv()
        .unwrap()
        .unwrap();
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(script.drops.load(Ordering::SeqCst), 0, "held while started");
    sink.begin("s1");
    worker
        .speak(req("After a long gap."), "s1".into())
        .blocking_recv()
        .unwrap()
        .unwrap();
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(script.loads.load(Ordering::SeqCst), 1, "no reload after the gap");
    assert_eq!(script.drops.load(Ordering::SeqCst), 0);
    // Stop releases it (and frees it at once).
    worker.unload().blocking_recv().unwrap().unwrap();
    assert_eq!(script.drops.load(Ordering::SeqCst), 1);
    // A model loaded by a sentence alone (a preview) is not held: the idle timer frees it.
    sink.begin("preview");
    worker
        .speak(req("Preview."), "preview".into())
        .blocking_recv()
        .unwrap()
        .unwrap();
    wait_until("the idle unload of the preview model", || script.drops.load(Ordering::SeqCst) == 2);
    // A Start that fails (its voice is missing) does not hold the model it loaded.
    let mut missing = req("");
    missing.reference_voice_id = "missing".into();
    assert!(worker
        .speak(missing, String::new())
        .blocking_recv()
        .unwrap()
        .is_err());
    wait_until("the idle unload after a failed start", || script.drops.load(Ordering::SeqCst) == 3);
    // A restart (stop, then start at once, queued behind a running sentence) keeps the hold.
    worker
        .speak(req(""), String::new())
        .blocking_recv()
        .unwrap()
        .unwrap();
    sink.begin("block");
    let blocker = worker.speak(req("block"), "block".into());
    wait_until("the blocking chunk", || script.blocking.load(Ordering::SeqCst));
    let unload = worker.unload();
    let start = worker.speak(req(""), String::new());
    worker.cancel(Some("block"));
    assert_eq!(blocker.blocking_recv().unwrap().unwrap_err(), CANCELLED);
    unload.blocking_recv().unwrap().unwrap();
    start.blocking_recv().unwrap().unwrap();
    let loads = script.loads.load(Ordering::SeqCst);
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(script.drops.load(Ordering::SeqCst), 3, "held again after the restart");
    assert_eq!(script.loads.load(Ordering::SeqCst), loads);
    worker.unload().blocking_recv().unwrap().unwrap();
    assert_eq!(script.drops.load(Ordering::SeqCst), 4);
}

#[test]
fn request_options_are_read_leniently() {
    let o = serde_json::json!({ "a": 1.5, "b": " 2.5 ", "c": "x", "d": null, "e": 7, "f": "8", "g": 9.9, "s": "  hi " });
    assert_eq!(option_f32(&o, "a"), Some(1.5));
    assert_eq!(option_f32(&o, "b"), Some(2.5));
    assert_eq!(option_f32(&o, "c"), None);
    assert_eq!(option_f32(&o, "d"), None);
    assert_eq!(option_f32(&o, "missing"), None);
    assert_eq!(option_i64(&o, "e"), Some(7));
    assert_eq!(option_i64(&o, "f"), Some(8));
    assert_eq!(option_i64(&o, "g"), Some(9));
    assert_eq!(option_str(&o, "s"), Some("hi"));
    assert_eq!(option_str(&serde_json::Value::Null, "s"), None);
}

// -------------------------------------------------------------------------------------------------
// Pocket
// -------------------------------------------------------------------------------------------------

#[test]
fn pocket_requests_resolve_their_model_threads_and_options() {
    assert_eq!(resolve_model("").unwrap(), DEFAULT_MODEL);
    assert_eq!(resolve_model(" tts/pocket-tts-en ").unwrap(), "tts/pocket-tts-en");
    assert!(resolve_model("tts/kokoro-v1.0").is_err());
    assert_eq!(resolve_threads(None), 2);
    assert_eq!(resolve_threads(Some(0)), 1);
    assert_eq!(resolve_threads(Some(16)), 4);
    let mut r = CloningTtsRequest::default();
    let d = options_of(&r);
    assert_eq!(d, PocketOptions::default());
    r.options = serde_json::json!({ "temperature": 5, "seed": -3, "steps": 40 });
    let o = options_of(&r);
    assert_eq!(o.temperature, sherpa::TEMPERATURE_RANGE.1);
    assert_eq!(o.seed, None, "a negative seed means random");
    assert_eq!(o.steps, Some(sherpa::STEPS_RANGE.1));
    r.options = serde_json::json!({ "seed": 42, "steps": "3", "temperature": "0.5" });
    let o = options_of(&r);
    assert_eq!((o.seed, o.steps, o.temperature), (Some(42), Some(3), 0.5));
    let backend = PocketBackend::app();
    let mut a = CloningTtsRequest::default();
    let k1 = backend.key(&a).unwrap();
    a.options = serde_json::json!({ "temperature": 0.3 });
    a.reference_voice_id = "someone".into();
    assert_eq!(backend.key(&a).unwrap(), k1, "voices and options never reload the model");
    a.threads = Some(3);
    assert_ne!(backend.key(&a).unwrap(), k1);
    a.model = "nope".into();
    assert!(backend.key(&a).is_err());
}

#[test]
fn the_extra_json_carries_temperature_seed_and_reference_length() {
    let e: serde_json::Value = serde_json::from_str(&PocketOptions::default().extra_json()).unwrap();
    assert!((e["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-6);
    assert_eq!(e["seed"], -1);
    assert_eq!(e["max_reference_audio_len"].as_f64().unwrap(), f64::from(sherpa::MAX_REFERENCE_SECONDS));
    let e: serde_json::Value = serde_json::from_str(
        &PocketOptions {
            temperature: f32::NAN,
            seed: Some(i64::MAX),
            steps: None,
        }
        .extra_json(),
    )
    .unwrap();
    assert!((e["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-6, "non-finite → default");
    assert_eq!(e["seed"], i64::from(i32::MAX), "seeds fit sherpa's int");
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "sigil-pocket-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn touch(p: &Path) {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).unwrap();
    }
    std::fs::write(p, b"x").unwrap();
}

#[test]
fn the_package_files_are_found_in_either_precision() {
    let dir = temp_dir("files");
    for f in [
        "lm_flow.int8.onnx",
        "lm_flow.onnx",
        "lm_main.int8.onnx",
        "encoder.onnx",
        "decoder.int8.onnx",
        "text_conditioner.onnx",
        "vocab.json",
    ] {
        touch(&dir.join(f));
    }
    let err = PocketFiles::discover(&dir, true).unwrap_err();
    assert!(err.contains("token_scores.json"), "{err}");
    touch(&dir.join("token_scores.json"));
    let int8 = PocketFiles::discover(&dir, true).unwrap();
    assert_eq!(int8.lm_flow.file_name().unwrap(), "lm_flow.int8.onnx");
    assert_eq!(int8.decoder.file_name().unwrap(), "decoder.int8.onnx");
    let full = PocketFiles::discover(&dir, false).unwrap();
    assert_eq!(
        full.lm_flow.file_name().unwrap(),
        "lm_flow.onnx",
        "fp32 preferred for the full-precision model"
    );
    assert_eq!(full.lm_main.file_name().unwrap(), "lm_main.int8.onnx", "the only one there");
    std::fs::remove_file(dir.join("encoder.onnx")).unwrap();
    assert!(PocketFiles::discover(&dir, true)
        .unwrap_err()
        .contains("encoder"));
    // Built-in voices: the package's sample WAVs, sorted.
    assert!(sherpa::builtin_voices(&dir).is_empty());
    for f in ["test_wavs/zed.wav", "test_wavs/bria.WAV", "test_wavs/notes.txt"] {
        touch(&dir.join(f));
    }
    let voices: Vec<String> = sherpa::builtin_voices(&dir)
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(voices, ["bria.WAV", "zed.wav"]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_library_voice_must_exist_and_have_a_valid_id() {
    let dir = temp_dir("voices");
    let backend = PocketBackend::with_app_data(&dir);
    let err = backend.reference_clip(&dir, "Not Valid").unwrap_err();
    assert!(err.contains("invalid voice id"), "{err}");
    assert!(backend.reference_clip(&dir, "nobody").is_err());
    // No library voice and no built-in voice in the package.
    assert!(backend
        .reference_clip(&dir, "")
        .unwrap_err()
        .contains("built-in voice"));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_cache_backend_name_is_valid() {
    assert!(crate::services::neural_audio::vc::families::reference::valid_backend(CACHE_BACKEND));
}

/// A temporary voice library holding `clip` as voice `id`.
fn library_from_clip(clip: &Path, id: &str) -> PathBuf {
    let app = temp_dir("reallib");
    let dir = app.join("voices").join(id);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(clip, dir.join("ref.wav")).unwrap();
    let sha = crate::services::neural_audio::vc::families::reference::sha256_file(&dir.join("ref.wav")).unwrap();
    let meta = serde_json::json!({ "id": id, "name": "Real voice", "source": "imported", "sha256": sha });
    std::fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&meta).unwrap()).unwrap();
    app
}

#[test]
#[ignore = "needs tts/pocket-tts-en-int8 and runtime/sherpa-onnx in SIGIL_APP_DATA_DIR and SIGIL_TEST_MODELS_DIR clips"]
fn real_model_pocket_clones_a_library_voice() {
    let _dll = crate::services::speech_rt::sherpa::testkit::dll_lock();
    if !crate::services::speech_rt::sherpa::testkit::runtime_installed() {
        return;
    }
    let Some(package) = crate::services::speech_rt::sherpa::testkit::installed(DEFAULT_MODEL) else {
        return;
    };
    // Record the real package layout (the files and the built-in voices come from reading
    // sherpa-onnx's sources; this run is their check): print every file, and require the sample
    // voices the panel's built-in voice uses.
    let mut listing = Vec::new();
    let mut stack = vec![package.clone()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                let size = p.metadata().map(|m| m.len()).unwrap_or(0);
                listing.push(format!("{} ({size} bytes)", p.strip_prefix(&package).unwrap().display()));
            }
        }
    }
    listing.sort();
    eprintln!("{DEFAULT_MODEL} package files:
  {}", listing.join("
  "));
    PocketFiles::discover(&package, true).expect("the package has every model file");
    assert!(
        !sherpa::builtin_voices(&package).is_empty(),
        "the package has no test_wavs/*.wav: Start without a library voice would fail"
    );
    let Some(clip) = crate::services::speech_rt::sherpa::testkit::clip_path("en.wav") else {
        return;
    };
    let app = library_from_clip(&clip, "real-voice");
    let backend = PocketBackend::with_app_data(&app);
    let request = CloningTtsRequest {
        reference_voice_id: "real-voice".into(),
        options: serde_json::json!({ "seed": 7 }),
        ..CloningTtsRequest::default()
    };
    let never = AtomicBool::new(false);
    let started = Instant::now();
    let mut model = backend.load(&request, &never).unwrap();
    backend.warm_up(&mut model, &request, &never).unwrap();
    eprintln!("loaded and warmed up in {} ms", started.elapsed().as_millis());
    for voice in ["real-voice", ""] {
        let request = CloningTtsRequest {
            reference_voice_id: voice.into(),
            ..request.clone()
        };
        let mut audio = Vec::new();
        let mut first: Option<Duration> = None;
        let t = Instant::now();
        let done = backend
            .synthesize(
                &mut model,
                &request,
                "Hello! This is a cloned voice speaking on the processor.",
                &never,
                &mut |s, rate| {
                    assert_eq!(rate, sherpa::SAMPLE_RATE);
                    first.get_or_insert(t.elapsed());
                    audio.extend_from_slice(s);
                    true
                },
            )
            .unwrap();
        assert!(done);
        let secs = audio.len() as f32 / sherpa::SAMPLE_RATE as f32;
        let rms = (audio.iter().map(|v| v * v).sum::<f32>() / audio.len().max(1) as f32).sqrt();
        eprintln!(
            "voice {voice:?}: {secs:.2} s in {} ms (first audio {} ms, RTF {:.2}), RMS {rms:.3}",
            t.elapsed().as_millis(),
            first.unwrap_or_default().as_millis(),
            t.elapsed().as_secs_f32() / secs.max(1e-3)
        );
        assert!(secs > 1.0 && secs < 15.0, "{secs} s");
        assert!(rms > 0.005, "silent");
    }
    let _ = std::fs::remove_dir_all(app);
}
