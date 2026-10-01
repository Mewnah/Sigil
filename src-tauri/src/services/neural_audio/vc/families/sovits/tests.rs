//! so-vits-svc family tests (P5-D): stand-in exports built with `test_onnx` (the real ONNX Runtime
//! path: import, slot checks, the factory request, fixed-block streaming through the real harness),
//! in every input variant the family accepts, and the specific rejections.
//!
//! The stand-in generator returns `GAIN · f0` held for `hop` samples per frame (`[1, 1, T · hop]`),
//! so the output shows which F0 the family fed on which frame; every other input is declared (and
//! fed, so the shapes and types are checked by ONNX Runtime) but unused.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use super::config::{SovitsRecord, RECORD_FILE};
use super::model::SovitsModel;
use super::slot::GENERATOR_FILE;
use super::*;
use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT, INT64};
use crate::services::neural_audio::vc::engine::options::{EffectiveLevers, F0Range, Lever, PrecisionMode};
use crate::services::neural_audio::vc::families::{self as fam, FamilyKind, FamilyRequest, FamilySlotContext, FamilySlotRequest};
use crate::services::neural_audio::vc::manifest::{load_slot_manifest, SlotFormat};
use crate::services::neural_audio::vc::stream::VcStreamHarness;
use crate::services::neural_audio::vc::test_models;
use crate::services::neural_audio::vc::traits::ConversionCtx;
use crate::services::neural_audio::vc::world::DioPitchExtractor;
use crate::services::neural_audio::RvcPresetProfile;

const FLOAT16: u64 = 10;
const GAIN: f32 = 1e-3;
const HOP: usize = 512;
const SR: u32 = 44_100;

/// What the stand-in export declares.
#[derive(Clone, Copy)]
struct Fake {
    half: bool,
    vol: bool,
    /// `CharaMix` speaker weights `[T, n]`.
    mix: Option<u64>,
    /// The speaker-weights axis is dynamic (`[T, n]` with `n` undeclared; the count must come
    /// from a config).
    mix_dynamic: bool,
    /// MoeVS (`mel2ph`, time-major `c`, `sid`); else so-vits-svc 4.0 without mel2ph (channel-major
    /// `c` at the frame rate, `g [1, 1]`, `c_lengths`, no noise).
    mel2ph: bool,
    uv_int: bool,
    width: u64,
}

impl Default for Fake {
    fn default() -> Self {
        Fake {
            half: false,
            vol: false,
            mix: None,
            mix_dynamic: false,
            mel2ph: true,
            uv_int: false,
            width: test_models::EMB as u64,
        }
    }
}

fn fake_export(f: Fake) -> Vec<u8> {
    let ft = if f.half { FLOAT16 } else { FLOAT };
    let mut g = Graph::new();
    if f.mel2ph {
        g.input("c", ft, &[Dim::Fixed(1), Dim::Named("s"), Dim::Fixed(f.width)])
            .input("f0", ft, &[Dim::Fixed(1), Dim::Named("t")])
            .input("mel2ph", INT64, &[Dim::Fixed(1), Dim::Named("t")])
            .input("uv", if f.uv_int { INT64 } else { ft }, &[Dim::Fixed(1), Dim::Named("t")])
            .input("noise", ft, &[Dim::Fixed(1), Dim::Fixed(192), Dim::Named("t")]);
        match f.mix {
            Some(n) => g.input("sid", ft, &[Dim::Named("t"), if f.mix_dynamic { Dim::Named("n") } else { Dim::Fixed(n) }]),
            None => g.input("sid", INT64, &[Dim::Fixed(1)]),
        };
        if f.vol {
            g.input("vol", ft, &[Dim::Fixed(1), Dim::Named("t")]);
        }
    } else {
        g.input("c", ft, &[Dim::Fixed(1), Dim::Fixed(f.width), Dim::Named("t")])
            .input("f0", ft, &[Dim::Fixed(1), Dim::Named("t")])
            .input("uv", ft, &[Dim::Fixed(1), Dim::Named("t")])
            .input("g", INT64, &[Dim::Fixed(1), Dim::Fixed(1)])
            .input("c_lengths", INT64, &[Dim::Fixed(1)]);
    }
    g.output("audio", ft, &[Dim::Fixed(1), Dim::Fixed(1), Dim::Named("n")])
        .init_i64("axes2", &[1], &[2])
        .init_i64("repeats", &[3], &[1, 1, HOP as i64])
        .init_i64("flat", &[3], &[1, 1, -1])
        .init_f32("gain", &[1], &[GAIN])
        .node("Cast", &["f0"], &["f32"], &[Attr::Int("to", FLOAT as i64)])
        .node("Unsqueeze", &["f32", "axes2"], &["f3"], &[])
        .node("Tile", &["f3", "repeats"], &["tiled"], &[])
        .node("Reshape", &["tiled", "flat"], &["r"], &[])
        .node("Mul", &["r", "gain"], &["a32"], &[])
        .node("Cast", &["a32"], &["audio"], &[Attr::Int("to", ft as i64)]);
    g.model_bytes(13)
}

/// A temp folder removed on drop.
struct Temp(PathBuf);

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn temp(tag: &str) -> Temp {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("sigil-sovits-{tag}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    Temp(dir)
}

const TRAINING_CONFIG: &str = r#"{
    "data": {"sampling_rate": 44100, "hop_length": 512},
    "model": {"ssl_dim": 768, "n_speakers": 2, "speech_encoder": "vec768l12", "vol_embedding": false},
    "spk": {"nyaru": 0, "alto": 1}
}"#;

/// `checkpoints/nyaru/nyaru_SoVits.onnx` (+ `checkpoints/nyaru.json`, MoeVS) and optionally the
/// training `config.json` next to the export, as so-vits-svc's exporter leaves them.
fn export_tree(root: &Path, f: Fake, training: Option<&str>, moevs: Option<&str>) -> PathBuf {
    let dir = root.join("checkpoints").join("nyaru");
    std::fs::create_dir_all(&dir).unwrap();
    let onnx = dir.join("nyaru_SoVits.onnx");
    std::fs::write(&onnx, fake_export(f)).unwrap();
    if let Some(cfg) = training {
        std::fs::write(dir.join("config.json"), cfg).unwrap();
    }
    if let Some(m) = moevs {
        std::fs::write(root.join("checkpoints").join("nyaru.json"), m).unwrap();
    }
    onnx
}

fn moevs_json(mix: bool, vol: bool, speakers: &[&str]) -> String {
    serde_json::json!({
        "Folder": "nyaru", "Name": "nyaru", "Type": "SoVits", "Rate": 44100, "Hop": 512,
        "Hubert": "vec-768-layer-12", "SoVits4": true, "SoVits3": false, "CharaMix": mix,
        "Volume": vol, "HiddenSize": 768, "Characters": speakers, "Cluster": ""
    })
    .to_string()
}

/// Add the voice at `source` into a fresh slot under `models`; returns the slot folder.
fn add_voice(models: &Path, source: &Path) -> Result<(PathBuf, Vec<String>), String> {
    let slot = models.join("voice");
    let _ = std::fs::remove_dir_all(&slot);
    std::fs::create_dir_all(&slot).unwrap();
    let request = FamilySlotRequest {
        family: "sovits".into(),
        source: Some(source.to_string_lossy().into_owned()),
        ..Default::default()
    };
    let cancel = AtomicBool::new(false);
    let progress = |_: &str, _: f64, _: &str| {};
    let options = fam::normalize_options(FamilyKind::Sovits, None).unwrap();
    let ctx = FamilySlotContext {
        models_dir: models,
        slot_dir: slot.clone(),
        name: "Nyaru".into(),
        options,
        app_data: None,
        templates_dir: None,
        exe: None,
        cancel: &cancel,
        progress: &progress,
    };
    fam::create_slot(FamilyKind::Sovits, &request, &ctx).map(|c| (slot, c.warnings))
}

fn levers(on: bool) -> EffectiveLevers {
    let mut l = EffectiveLevers::default();
    for lever in Lever::ALL {
        l.levers.set(lever, on);
    }
    l
}

fn request(slot: &Path, encoder: &Path, levers: EffectiveLevers, lookahead_ms: f64) -> FamilyRequest {
    FamilyRequest {
        kind: FamilyKind::Sovits,
        slot_dir: slot.to_path_buf(),
        manifest: load_slot_manifest(slot).unwrap(),
        app_data: None,
        ep: "cpu".into(),
        session: VcSessionOptions::default(),
        levers,
        precision: PrecisionMode::Auto,
        preset: RvcPresetProfile::LowLatency,
        block_ms: 40.0,
        chunk_frames: 0,
        device_rate: 48_000,
        extra_context_ms: 500.0,
        lookahead_ms,
        f0_range: F0Range::Speech,
        silence_skip: false,
        noise_seed: 7,
        pitch: Some(Box::new(DioPitchExtractor::new())),
        pitch_method: "dio".into(),
        content_encoder: Some(encoder.to_path_buf()),
        reference: None,
        latency_budget_ms: 300.0,
        cancel: None,
    }
}

fn tone(n: usize, hz: f32) -> Vec<f32> {
    (0..n).map(|i| 0.3 * (std::f32::consts::TAU * hz * i as f32 / 16_000.0).sin()).collect()
}

fn ctx(audio: &[f32], want: usize, pos: u64, speaker: i64) -> ConversionCtx<'_> {
    ConversionCtx {
        mono_16k: audio,
        new_frames: want,
        pitch_shift_semitones: 0.0,
        index_rate: 0.0,
        protect: 0.33,
        speaker_id: speaker,
        stream_pos_16k: pos,
        reference: None,
        singing: Default::default(),
    }
}

/// The stand-in content encoder (768 channels) in `root`.
fn encoder_file(root: &Path) -> PathBuf {
    let p = root.join("hubert_base.onnx");
    std::fs::write(&p, test_models::content_encoder_bytes()).unwrap();
    p
}

/// Most of `out` follows a 150 Hz input (the stand-in emits `GAIN · f0`).
fn assert_follows_150hz(out: &[f32], what: &str) {
    assert!(out.iter().all(|x| x.is_finite()), "{what}: finite");
    let near = out.iter().filter(|x| (**x - 0.15).abs() < 0.03).count();
    assert!(near > out.len() * 3 / 4, "{what}: {near} of {} samples follow the tone's F0", out.len());
}

/// The whole path: a MoeVS export with its training config imports, lists without a problem,
/// loads through the registry, runs fixed blocks at the configured context and returns exactly the
/// harness's length, following F0.
#[test]
fn an_exported_voice_imports_loads_and_converts() {
    let root = temp("e2e");
    let onnx = export_tree(&root.0, Fake::default(), Some(TRAINING_CONFIG), Some(&moevs_json(false, false, &["nyaru", "alto"])));
    let models = root.0.join("models");
    let (slot, warnings) = add_voice(&models, &onnx).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let m = load_slot_manifest(&slot).unwrap();
    assert_eq!(m.slot_format(), SlotFormat::Family(FamilyKind::Sovits));
    assert_eq!((m.generator.as_str(), m.sample_rate, m.emb_channels), (GENERATOR_FILE, SR, 768));
    assert_eq!((m.embedder.as_deref(), m.speaker_count, m.name.as_deref()), (Some("contentvec"), Some(2), Some("Nyaru")));
    assert!(m.source_sha256.is_some());
    let record = SovitsRecord::load(&slot).unwrap();
    assert_eq!(record.hop, 512);
    assert_eq!(record.speakers, vec!["nyaru".to_string(), "alto".to_string()]);
    assert_eq!(record.speech_encoder.as_deref(), Some("vec768l12"));
    assert_eq!(record.source, "nyaru_SoVits.onnx");
    assert!(slot.join("config.json").is_file(), "the training config is kept");
    assert_eq!(fam::slot_problem(FamilyKind::Sovits, &slot, &m), None);

    let encoder = encoder_file(&root.0);
    let engine = fam::create(request(&slot, &encoder, levers(true), 0.0)).unwrap();
    assert_eq!(engine.model_delay_16k, 0);
    assert_eq!(engine.outcomes.iter().map(|(c, _)| c.as_str()).collect::<Vec<_>>(), vec!["vc", "vc.content"]);
    // What this family runs; the rest is noted.
    for lever in [Lever::AnchoredNoise, Lever::F0Cache, Lever::Governor] {
        assert!(engine.levers.levers.get(lever), "{lever:?}");
    }
    for lever in [Lever::PartialDecode, Lever::NsfPhase0, Lever::GpuKnn, Lever::IoBinding, Lever::CudaGraphs, Lever::GpuPriority] {
        assert!(!engine.levers.levers.get(lever), "{lever:?}");
        assert!(engine.levers.notes.iter().any(|n| n.starts_with(lever.as_str())), "{lever:?}: {:?}", engine.levers.notes);
    }
    let mut model = engine.model;
    assert_eq!((model.input_sample_rate(), model.output_sample_rate()), (16_000, SR));
    assert_eq!(model.fixed_block(), Some(640), "40 ms blocks");
    let want = 640 + 481;
    let context = model.configure_stream(640, want, 8_000).unwrap();
    assert!(context >= 8_000 + 640 && context % 320 == 0, "{context}");
    assert!(model.configure_stream(320, want, 8_000).is_err(), "another block is refused");
    let audio = tone(context, 150.0);
    let out = model.convert(&ctx(&audio, want, 3_200, 0)).unwrap();
    assert_eq!(out.len(), (want * SR as usize + 8_000) / 16_000);
    assert_follows_150hz(&out, "moevs");
    // A pitch shift of an octave doubles what the generator gets.
    let mut up = ctx(&audio, want, 3_200 + 640, 1);
    up.pitch_shift_semitones = 12.0;
    let out = model.convert(&up).unwrap();
    let near = out.iter().filter(|x| (**x - 0.30).abs() < 0.05).count();
    assert!(near > out.len() * 3 / 4, "{near} of {}", out.len());
    // A whole-utterance call (the benchmark's offline ceiling) longer than any stream window.
    let long = tone(16_000 * 5, 150.0);
    let out = model.convert(&ctx(&long, long.len(), 0, 0)).unwrap();
    assert_eq!(out.len(), 5 * SR as usize);
    assert!(out.iter().all(|x| x.is_finite()));
}

/// Every input variant the family accepts runs (fp16 I/O, vol, speaker mix, int64 uv, so-vits-svc
/// 4.0 graphs without mel2ph, 256-wide content).
#[test]
fn every_export_variant_runs() {
    let root = temp("variants");
    let encoder = encoder_file(&root.0);
    let cases = [
        ("half", Fake { half: true, ..Fake::default() }, moevs_json(false, false, &["a"])),
        ("vol", Fake { vol: true, ..Fake::default() }, moevs_json(false, true, &["a"])),
        ("mix", Fake { mix: Some(3), ..Fake::default() }, moevs_json(true, false, &["a", "b", "c"])),
        ("uv-int", Fake { uv_int: true, ..Fake::default() }, moevs_json(false, false, &["a"])),
        ("no-mel2ph", Fake { mel2ph: false, ..Fake::default() }, moevs_json(false, false, &["a"])),
    ];
    for (tag, fake, moevs) in cases.iter().cloned() {
        let src = root.0.join(tag);
        let onnx = export_tree(&src, fake, None, Some(&moevs));
        let (slot, _) = add_voice(&src.join("models"), &onnx).unwrap_or_else(|e| panic!("{tag}: {e}"));
        let mut model = SovitsModel::load(request(&slot, &encoder, levers(true), 0.0))
            .unwrap_or_else(|e| panic!("{tag}: {e}"))
            .model;
        let context = model.configure_stream(640, 1_121, 4_800).unwrap();
        let audio = tone(context, 150.0);
        // Speaker ids beyond the voice's speakers are clamped (the mix is one-hot of the last).
        let out = model.convert(&ctx(&audio, 1_121, 0, 9)).unwrap_or_else(|e| panic!("{tag}: {e}"));
        assert_eq!(out.len(), (1_121 * SR as usize + 8_000) / 16_000, "{tag}");
        assert_follows_150hz(&out, tag);
    }
    // A 256-wide (vec256l9) voice imports and asks for the 256-channel ContentVec.
    let src = root.0.join("v256");
    let onnx = export_tree(&src, Fake { width: 256, ..Fake::default() }, None, None);
    let (slot, warnings) = add_voice(&src.join("models"), &onnx).unwrap();
    let m = load_slot_manifest(&slot).unwrap();
    assert_eq!((m.emb_channels, m.embedder.as_deref()), (256, Some("contentvec-256")));
    assert!(warnings.iter().any(|w| w.contains("vec256l9")), "assumed encoder: {warnings:?}");
    assert!(warnings.iter().any(|w| w.contains("44100 Hz")), "rate from the hop: {warnings:?}");
    // The 768-channel stand-in encoder does not fit it: the pool says so (no silent garbage), and
    // uses the installed 256-channel ContentVec instead when this PC has one.
    match SovitsModel::load(request(&slot, &encoder, levers(true), 0.0)) {
        Err(err) => assert!(err.contains("256") || err.contains("contentvec-256"), "{err}"),
        Ok(engine) => assert!(engine.warnings.iter().any(|w| w.contains("does not fit")), "{:?}", engine.warnings),
    }
}

/// Through the real stream harness (fixed blocks from 10 ms device blocks at 48 kHz): every call
/// returns exactly the device frames owed, and the converted tone comes through.
#[test]
fn the_stream_harness_runs_the_voice_in_fixed_blocks() {
    let root = temp("harness");
    let onnx = export_tree(&root.0, Fake::default(), Some(TRAINING_CONFIG), None);
    let (slot, _) = add_voice(&root.0.join("models"), &onnx).unwrap();
    let encoder = encoder_file(&root.0);
    let model = fam::create(request(&slot, &encoder, levers(true), 20.0)).unwrap();
    assert_eq!(model.model_delay_16k, 320, "20 ms of lookahead");
    let mut h = VcStreamHarness::new(model.model, 48_000, 48_000, 1, 1, 8_000, 960, 480).unwrap();
    assert_eq!(h.fixed_shape().map(|(block, _)| block), Some(640));
    let input: Vec<f32> = (0..48_000 * 2).map(|i| 0.3 * (std::f32::consts::TAU * 150.0 * i as f32 / 48_000.0).sin()).collect();
    let mut total = 0usize;
    let mut tail = Vec::new();
    for chunk in input.chunks(480) {
        let out = h.push_and_convert(chunk).unwrap();
        assert_eq!(out.len(), chunk.len(), "one output frame per input frame");
        total += out.len();
        tail = out;
    }
    assert_eq!(total, input.len());
    assert!(tail.iter().all(|x| x.is_finite()));
    let loud = tail.iter().filter(|x| x.abs() > 0.05).count();
    assert!(loud > tail.len() / 2, "the converted tone reaches the output ({loud} of {})", tail.len());
}

/// The silence skip returns silence for quiet ticks after the hangover, without running the graph.
#[test]
fn quiet_input_is_skipped_after_the_hangover() {
    let root = temp("silence");
    let onnx = export_tree(&root.0, Fake::default(), Some(TRAINING_CONFIG), None);
    let (slot, _) = add_voice(&root.0.join("models"), &onnx).unwrap();
    let encoder = encoder_file(&root.0);
    let mut req = request(&slot, &encoder, levers(false), 0.0);
    req.silence_skip = true;
    let mut model = SovitsModel::load(req).unwrap().model;
    let context = model.configure_stream(640, 1_121, 4_800).unwrap();
    let quiet = vec![0.0f32; context];
    let mut outs = Vec::new();
    for k in 0..6u64 {
        outs.push(model.convert(&ctx(&quiet, 1_121, k * 640, 0)).unwrap());
    }
    assert!(outs.iter().all(|o| o.len() == (1_121 * SR as usize + 8_000) / 16_000));
    assert!(outs.last().unwrap().iter().all(|&x| x == 0.0));
}

/// Files that are not a so-vits-svc export are refused by what they are, with the importer's codes.
#[test]
fn other_files_are_refused_specifically() {
    let root = temp("refuse");
    let models = root.0.join("models");
    // A training checkpoint.
    let pth = root.0.join("G_8000.pth");
    std::fs::write(&pth, b"PK\x03\x04 torch zip").unwrap();
    let err = add_voice(&models, &pth).unwrap_err();
    assert!(err.starts_with("unsupported_model") && err.contains("onnx_export.py"), "{err}");
    // An RVC generator.
    let rvc = root.0.join("rvc.onnx");
    std::fs::write(&rvc, test_models::generator_bytes()).unwrap();
    let err = add_voice(&models, &rvc).unwrap_err();
    assert!(err.starts_with("unsupported_model") && err.contains("RVC generator"), "{err}");
    // A content encoder (as in the wok000 pack).
    let enc = root.0.join("contentvec-f.onnx");
    std::fs::write(&enc, test_models::content_encoder_bytes()).unwrap();
    let err = add_voice(&models, &enc).unwrap_err();
    assert!(err.contains("content encoder"), "{err}");
    // A config that belongs to another voice (hop 320 vs the export's 512).
    let other = root.0.join("other");
    let cfg = TRAINING_CONFIG.replace("\"hop_length\": 512", "\"hop_length\": 320");
    let onnx = export_tree(&other, Fake::default(), Some(&cfg), None);
    let err = add_voice(&models, &onnx).unwrap_err();
    assert!(err.starts_with("shape_mismatch"), "{err}");
    // A speech encoder Sigil does not have.
    let soft = root.0.join("soft");
    let cfg = TRAINING_CONFIG.replace("vec768l12", "hubertsoft");
    let onnx = export_tree(&soft, Fake::default(), Some(&cfg), None);
    let err = add_voice(&models, &onnx).unwrap_err();
    assert!(err.starts_with("unsupported_model") && err.contains("hubert-soft"), "{err}");
    // A speaker-mix export without any speaker count.
    let mix = root.0.join("mix");
    let mut g = Graph::new();
    g.input("c", FLOAT, &[Dim::Fixed(1), Dim::Named("s"), Dim::Fixed(768)])
        .input("f0", FLOAT, &[Dim::Fixed(1), Dim::Named("t")])
        .input("mel2ph", INT64, &[Dim::Fixed(1), Dim::Named("t")])
        .input("uv", FLOAT, &[Dim::Fixed(1), Dim::Named("t")])
        .input("noise", FLOAT, &[Dim::Fixed(1), Dim::Fixed(192), Dim::Named("t")])
        .input("sid", FLOAT, &[Dim::Named("t"), Dim::Named("n")])
        .output("audio", FLOAT, &[Dim::Fixed(1), Dim::Named("t")])
        .node("Identity", &["f0"], &["audio"], &[]);
    std::fs::create_dir_all(&mix).unwrap();
    let p = mix.join("mix.onnx");
    std::fs::write(&p, g.model_bytes(13)).unwrap();
    let err = add_voice(&models, &p).unwrap_err();
    assert!(err.contains("speaker weights"), "{err}");
    // Nothing to import.
    let empty = root.0.join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    assert!(add_voice(&models, &empty).unwrap_err().starts_with("not_a_checkpoint"));
}

/// A folder and a zip of the same files import like the bare export; skipped files are reported.
#[test]
fn folders_and_zips_import() {
    let root = temp("bundles");
    let onnx = export_tree(&root.0, Fake::default(), Some(TRAINING_CONFIG), None);
    let folder = onnx.parent().unwrap().to_path_buf();
    std::fs::write(folder.join("kmeans_10000.pt"), b"\x80\x04 pickle").unwrap();
    let (slot, warnings) = add_voice(&root.0.join("models-a"), &folder).unwrap();
    assert_eq!(load_slot_manifest(&slot).unwrap().sample_rate, SR);
    assert!(warnings.iter().any(|w| w.contains("kmeans")), "{warnings:?}");

    let zip_bytes = crate::services::neural_audio::vc::import::zipguard::tests::build_zip(&[
        ("voice/nyaru_SoVits.onnx", &fake_export(Fake::default()), zip::CompressionMethod::Deflated),
        ("voice/config.json", TRAINING_CONFIG.as_bytes(), zip::CompressionMethod::Stored),
    ]);
    let zip_path = root.0.join("nyaru.zip");
    std::fs::write(&zip_path, zip_bytes).unwrap();
    let (slot, _) = add_voice(&root.0.join("models-b"), &zip_path).unwrap();
    let m = load_slot_manifest(&slot).unwrap();
    assert_eq!((m.sample_rate, m.speaker_count), (SR, Some(2)));
    assert!(slot.join(GENERATOR_FILE).is_file() && slot.join(RECORD_FILE).is_file());
    // Two exports in one archive are ambiguous.
    let two = crate::services::neural_audio::vc::import::zipguard::tests::build_zip(&[
        ("a.onnx", &fake_export(Fake::default()), zip::CompressionMethod::Stored),
        ("b.onnx", &fake_export(Fake::default()), zip::CompressionMethod::Stored),
    ]);
    std::fs::write(&zip_path, two).unwrap();
    assert!(add_voice(&root.0.join("models-c"), &zip_path).unwrap_err().starts_with("ambiguous_archive"));
}

/// A zip made by macOS Finder (`__MACOSX/` AppleDouble twins of every file) imports like any other,
/// with no ambiguity and no bogus JSON warning (P5-D review).
#[test]
fn macos_zips_import() {
    let root = temp("macos-zip");
    let apple_double = b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        ".as_slice();
    let export = fake_export(Fake::default());
    let zip_bytes = crate::services::neural_audio::vc::import::zipguard::tests::build_zip(&[
        ("nyaru/nyaru_SoVits.onnx", &export, zip::CompressionMethod::Deflated),
        ("nyaru/config.json", TRAINING_CONFIG.as_bytes(), zip::CompressionMethod::Stored),
        ("__MACOSX/nyaru/._nyaru_SoVits.onnx", apple_double, zip::CompressionMethod::Stored),
        ("__MACOSX/nyaru/._config.json", apple_double, zip::CompressionMethod::Stored),
    ]);
    let zip_path = root.0.join("nyaru.zip");
    std::fs::write(&zip_path, zip_bytes).unwrap();
    let (slot, warnings) = add_voice(&root.0.join("models"), &zip_path).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(load_slot_manifest(&slot).unwrap().speaker_count, Some(2));
    // so-vits-svc 4.0's generic `model.onnx` with its twin.
    let zip_bytes = crate::services::neural_audio::vc::import::zipguard::tests::build_zip(&[
        ("model.onnx", &export, zip::CompressionMethod::Deflated),
        ("__MACOSX/._model.onnx", apple_double, zip::CompressionMethod::Stored),
    ]);
    std::fs::write(&zip_path, zip_bytes).unwrap();
    add_voice(&root.0.join("models-2"), &zip_path).unwrap();
}

/// A so-vits-svc 4.0 export with its training config (no `speech_encoder`) keeps the config, says
/// the config names no encoder (not that it is missing), and a MoeSS JSON naming `hubert4.0` next
/// to it never displaces it (P5-D review).
#[test]
fn a_4_0_export_keeps_its_training_config() {
    let root = temp("v40");
    let v40 = r#"{
        "data": {"sampling_rate": 44100, "hop_length": 512},
        "model": {"ssl_dim": 256, "n_speakers": 200},
        "spk": {"nyaru": 0}
    }"#;
    let moess = r#"{"Folder": "nyaru", "Name": "nyaru", "Type": "SoVits", "Rate": 44100, "Hop": 512, "Hubert": "hubert4.0", "SoVits4": true, "Characters": ["nyaru"]}"#;
    let onnx = export_tree(&root.0, Fake { width: 256, ..Fake::default() }, Some(v40), None);
    let (slot, warnings) = add_voice(&root.0.join("models"), &onnx).unwrap();
    assert!(slot.join("config.json").is_file(), "the 4.0 training config is kept");
    assert!(!warnings.iter().any(|w| w.contains("missing")), "{warnings:?}");
    assert!(warnings.iter().any(|w| w.contains("does not name its speech encoder")), "{warnings:?}");
    let m = load_slot_manifest(&slot).unwrap();
    assert_eq!((m.emb_channels, m.embedder.as_deref(), m.speaker_count), (256, Some("contentvec-256"), Some(1)));
    assert_eq!(SovitsRecord::load(&slot).unwrap().speech_encoder, None);
    // With MoeSS's JSON (Hubert "hubert4.0") next to it: still the 4.0 voice, no warning at all.
    std::fs::write(root.0.join("checkpoints").join("nyaru.json"), moess).unwrap();
    let (slot, warnings) = add_voice(&root.0.join("models-2"), &onnx).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(slot.join("config.json").is_file());
    assert_eq!(load_slot_manifest(&slot).unwrap().embedder.as_deref(), Some("contentvec-256"));
}

/// so-vits-svc 4.1's `onnx_export.py` writes `checkpoints/<name>/<name>.json` next to
/// `<name>_SoVits.onnx`: that MoeVS JSON alone gives a speaker-mix export (dynamic speaker axis)
/// its speakers, rate and hop. A training config's `n_speakers: 200` (the 4.0 template's speaker
/// table) never counts as 200 speakers when it names three.
#[test]
fn the_exporters_moevs_json_next_to_the_export_is_read() {
    let root = temp("moevs-same-folder");
    let dir = root.0.join("checkpoints").join("nyaru");
    std::fs::create_dir_all(&dir).unwrap();
    let onnx = dir.join("nyaru_SoVits.onnx");
    let fake = Fake {
        mix: Some(3),
        mix_dynamic: true,
        ..Fake::default()
    };
    std::fs::write(&onnx, fake_export(fake)).unwrap();
    std::fs::write(dir.join("nyaru.json"), moevs_json(true, false, &["a", "b", "c"])).unwrap();
    let (slot, warnings) = add_voice(&root.0.join("models"), &onnx).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let m = load_slot_manifest(&slot).unwrap();
    assert_eq!((m.sample_rate, m.speaker_count), (SR, Some(3)));
    let record = SovitsRecord::load(&slot).unwrap();
    assert!(record.character_mix);
    assert_eq!(record.speakers, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    assert_eq!(record.configs, vec!["nyaru.json".to_string()]);
    let encoder = encoder_file(&root.0);
    let mut model = SovitsModel::load(request(&slot, &encoder, levers(true), 0.0)).unwrap().model;
    let context = model.configure_stream(640, 1_121, 4_800).unwrap();
    let out = model.convert(&ctx(&tone(context, 150.0), 1_121, 0, 2)).unwrap();
    assert_follows_150hz(&out, "same-folder moevs");

    // A training config of three named speakers that keeps the template's n_speakers: 200.
    let cfg = TRAINING_CONFIG
        .replace("\"n_speakers\": 2", "\"n_speakers\": 200")
        .replace("\"alto\": 1", "\"alto\": 1, \"tenor\": 2");
    std::fs::write(dir.join("config.json"), cfg).unwrap();
    let (slot, _) = add_voice(&root.0.join("models-2"), &onnx).unwrap();
    assert_eq!(load_slot_manifest(&slot).unwrap().speaker_count, Some(3));
}

/// A slot whose files went missing reports it (the slot listing's `problem`, the start preflight).
#[test]
fn broken_slots_report_their_problem() {
    let root = temp("problem");
    let onnx = export_tree(&root.0, Fake::default(), Some(TRAINING_CONFIG), None);
    let (slot, _) = add_voice(&root.0.join("models"), &onnx).unwrap();
    let m = load_slot_manifest(&slot).unwrap();
    std::fs::remove_file(slot.join(RECORD_FILE)).unwrap();
    assert!(fam::slot_problem(FamilyKind::Sovits, &slot, &m).unwrap().contains(RECORD_FILE));
    std::fs::remove_file(slot.join(GENERATOR_FILE)).unwrap();
    assert!(fam::slot_problem(FamilyKind::Sovits, &slot, &m).unwrap().contains("missing"));
    let mut spin = m.clone();
    spin.embedder = Some("spin-v2".into());
    assert!(check_slot(&slot, &spin).is_err());
}

/// The family's options: `cluster_ratio` clamps, and a voice without an index says the ratio has
/// no effect.
#[test]
fn cluster_ratio_without_an_index_is_a_warning() {
    let v = normalize_options(&serde_json::json!({"cluster_ratio": 3.0})).unwrap();
    assert_eq!(v, serde_json::json!({"cluster_ratio": 1.0}));
    assert_eq!(normalize_options(&serde_json::json!({"cluster_ratio": -1})).unwrap()["cluster_ratio"], 0.0);
    let root = temp("cluster");
    let onnx = export_tree(&root.0, Fake::default(), Some(TRAINING_CONFIG), None);
    let (slot, _) = add_voice(&root.0.join("models"), &onnx).unwrap();
    let mut req = request(&slot, &encoder_file(&root.0), levers(true), 0.0);
    req.manifest.family = Some(serde_json::json!({"cluster_ratio": 0.5}));
    let engine = SovitsModel::load(req).unwrap();
    assert!(engine.warnings.iter().any(|w| w.contains("cluster_ratio")), "{:?}", engine.warnings);
}

/// The registry reports the family as available, importing `.onnx` / `.zip`, with the pitch bank
/// and the content encoder.
#[test]
fn the_family_is_available_with_its_facts() {
    assert!(FamilyKind::Sovits.available());
    let i = info();
    assert_eq!((i.family, i.format, i.creation), ("sovits", fam::SOVITS_FORMAT, fam::CREATION_IMPORT));
    assert!(i.needs_pitch && i.needs_content_encoder && !i.zero_shot);
    assert!(i.experimental, "until a real so-vits-svc export is checked against upstream inference");
    assert!(run_import_job(&crate::services::neural_audio::vc::import::ImportJob::for_family("sovits"))
        .unwrap_err()
        .starts_with("job: "));
}
