//! DDSP-SVC family tests (P5-A).
//!
//! The CI tests run the tiny fixture (`test-fixtures/vc/ddsp_svc`, written by
//! `export_templates.py --tiny`): a real DDSP-SVC 6.3 training checkpoint of a tiny architecture,
//! its `config.yaml`, the template graphs of that architecture and one reference tick computed by
//! ONNX Runtime in Python. They prove the importer (allow-list unpickler, weight-norm folding,
//! transposes, the external-weights image), the tick's host side and the Euler loop against the
//! export, with a stand-in vocoder and stand-in units / pitch.
//!
//! `real_model_*` tests (`#[ignore]`) run the default template, the real vocoder and ContentVec
//! from `SIGIL_TEST_MODELS_DIR` (see [`RealAssets`]) and report the T10 singing figures.

use std::f32::consts::PI;
use std::path::{Path, PathBuf};

use super::import::{self, DdspRecord};
use super::layout::DdspLayout;
use super::model::{DdspFiles, DdspSvcModel, ModelParts, Shape, TickInputs, Tuning, UnitsSource, VocoderRuntime, VoiceSetup};
use super::*;
use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT};
use crate::services::neural_audio::vc::engine::options::{EffectiveLevers, Lever};
use crate::services::neural_audio::vc::import::ErrorCode;
use crate::services::neural_audio::vc::manifest::{load_slot_manifest, SlotFormat};
use crate::services::neural_audio::vc::traits::{ConversionCtx, PitchExtractor, VoiceControls, VoiceConversionModel};
use crate::services::neural_audio::vc::VcStreamHarness;

pub(super) fn tiny_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("test-fixtures").join("vc").join("ddsp_svc")
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sigil-ddsp-{tag}-{}-{}", std::process::id(), crate::services::neural_audio::vc::import::unix_ms()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Units from the audio itself: frame `t` = the mean of samples `[320 t, 320 t + 400)` times the
/// channel index plus a ramp (deterministic, follows the input).
struct FakeUnits(usize);

impl UnitsSource for FakeUnits {
    fn channels(&self) -> usize {
        self.0
    }

    fn encode(&mut self, x: &[f32], out: &mut Vec<f32>) -> Result<usize, String> {
        let n = if x.len() >= 400 { (x.len() - 400) / 320 + 1 } else { 0 };
        out.clear();
        for t in 0..n {
            let m: f32 = x[320 * t..320 * t + 400].iter().map(|v| v.abs()).sum::<f32>() / 400.0;
            out.extend((0..self.0).map(|c| m * (c as f32 + 1.0) + 0.01 * c as f32));
        }
        Ok(n)
    }
}

/// A fixed F0 for every 10 ms frame.
struct ConstPitch(f32);

impl PitchExtractor for ConstPitch {
    fn f0(&mut self, x: &[f32]) -> Result<Vec<f32>, String> {
        Ok(vec![self.0; x.len() / 160 + 1])
    }
}

/// The tiny vocoder stand-in: an NSF-like sine at the frames' F0 from `phase0` (cycles, the
/// running sum of `f0 / sr` per sample, like the real vocoder's source) plus a little of the mel,
/// both held for `hop` samples per frame (the real vocoder runs in the `real_model_*` tests).
fn stand_in_vocoder(dir: &Path, hop: usize, mels: usize) -> (PathBuf, VocoderRuntime) {
    let mut g = Graph::new();
    g.input("mel", FLOAT, &[Dim::Fixed(1), Dim::Fixed(mels as u64), Dim::Named("T")])
        .input("f0", FLOAT, &[Dim::Fixed(1), Dim::Named("T")])
        .input("phase0", FLOAT, &[Dim::Fixed(1)])
        .output("audio", FLOAT, &[Dim::Fixed(1), Dim::Named("S")])
        .init_i64("ax1", &[1], &[1])
        .init_i64("ax2", &[1], &[2])
        .init_i64("axis", &[], &[1])
        .init_i64("reps", &[3], &[1, 1, hop as i64])
        .init_i64("flat", &[2], &[1, -1])
        .init_f32("inv_sr", &[1], &[1.0 / 44_100.0])
        .init_f32("two_pi", &[1], &[2.0 * PI])
        .init_f32("amp", &[1], &[0.3])
        .init_f32("g2", &[1], &[1e-4])
        .node("Unsqueeze", &["f0", "ax2"], &["f3"], &[])
        .node("Tile", &["f3", "reps"], &["ft"], &[])
        .node("Reshape", &["ft", "flat"], &["fs"], &[])
        .node("Mul", &["fs", "inv_sr"], &["inc"], &[])
        .node("CumSum", &["inc", "axis"], &["cyc"], &[])
        .node("Add", &["cyc", "phase0"], &["ph"], &[])
        .node("Mul", &["ph", "two_pi"], &["rad"], &[])
        .node("Sin", &["rad"], &["sine"], &[])
        .node("Mul", &["sine", "amp"], &["src"], &[])
        .node("ReduceSum", &["mel", "ax1"], &["msum"], &[Attr::Int("keepdims", 0)])
        .node("Mul", &["msum", "g2"], &["b"], &[])
        .node("Unsqueeze", &["b", "ax2"], &["b3"], &[])
        .node("Tile", &["b3", "reps"], &["bt"], &[])
        .node("Reshape", &["bt", "flat"], &["bs"], &[])
        .node("Add", &["src", "bs"], &["audio"], &[]);
    let path = dir.join("vocoder.onnx");
    std::fs::write(&path, g.model_bytes(13)).unwrap();
    (path, VocoderRuntime { sampling_rate: 44_100, hop, num_mels: mels, upp: hop, source_sr: 44_100.0 })
}

fn reference() -> (serde_json::Value, Vec<f32>) {
    let dir = tiny_dir();
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("reference.json")).unwrap()).unwrap();
    let bin = std::fs::read(dir.join("reference.bin")).unwrap();
    let floats = bin.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    (json, floats)
}

fn array<'a>(json: &serde_json::Value, floats: &'a [f32], name: &str) -> &'a [f32] {
    let a = &json["arrays"][name];
    let off = a["offset"].as_u64().unwrap() as usize / 4;
    let n: usize = a["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).product();
    &floats[off..off + n]
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let num: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn all_levers() -> EffectiveLevers {
    let mut l = EffectiveLevers::default();
    for lever in [Lever::AnchoredNoise, Lever::NsfPhase0, Lever::F0Cache, Lever::IoBinding] {
        l.levers.set(lever, true);
    }
    l
}

/// The tiny voice, imported into `slot` (weights image) and loaded on the CPU.
fn tiny_model(tag: &str, pitch_hz: f32, silence_skip: bool) -> (DdspSvcModel, PathBuf) {
    let dir = tiny_dir();
    let layout = DdspLayout::read(&dir).unwrap();
    let voice = import::import_voice(&dir.join("model_100.pt"), &layout).expect("the tiny checkpoint imports");
    let slot = temp_dir(tag);
    import::write_slot(&slot, &voice, &layout, "Tiny", serde_json::json!({}), std::time::Instant::now()).unwrap();
    let (vocoder, rt) = stand_in_vocoder(&slot, layout.hop(), layout.mels());
    let config = DdspRecord::read(&slot).unwrap().config().unwrap();
    let model = DdspSvcModel::new(ModelParts {
        files: DdspFiles { ddsp: dir.join("ddsp.onnx"), reflow: dir.join("reflow_step.onnx"), vocoder },
        vocoder: rt,
        weights: slot.join(import::WEIGHTS_FILE),
        setup: VoiceSetup { tta2x: config.tta2x(), f0_min: config.f0_min, volume_window: config.volume_window() as usize, speakers: None },
        tuning: Tuning { block_16k: 1600, extra_16k: 4800, lookahead_16k: 0, steps: 3, t_start: 0.7 },
        units: Box::new(FakeUnits(layout.arch.encoder_out_channels as usize)),
        pitch: Box::new(ConstPitch(pitch_hz)),
        ep: "cpu".into(),
        session: VcSessionOptions { intra_threads: Some(1), ..VcSessionOptions::default() },
        levers: all_levers(),
        noise_seed: 7,
        silence_skip,
        outcomes: Vec::new(),
        warnings: Vec::new(),
        layout,
    })
    .unwrap();
    (model, slot)
}

/// The import reproduces the export's weights, and the graphs + the Rust Euler loop reproduce the
/// reference tick ONNX Runtime computed in Python (same inputs and draws).
#[test]
fn tiny_fixture_matches_the_python_reference_tick() {
    let (json, floats) = reference();
    let (mut model, slot) = tiny_model("parity", 220.0, false);
    let frames = json["frames"].as_u64().unwrap() as usize;
    let start = json["start_frame"].as_u64().unwrap() as usize;
    let units = json["units"].as_u64().unwrap() as usize;
    let steps = json["steps"].as_u64().unwrap() as u32;
    let t_start = json["t_start"].as_f64().unwrap() as f32;
    let shape = Shape {
        block_16k: 1600,
        want_16k: 1600,
        context_16k: 16_000,
        frames,
        start,
        reflow: frames - start,
        units_a: units,
        units_b: 0,
        units,
        out: 0,
        lookahead: 0,
    };
    model.build_graphs(shape).unwrap();
    let mel2ph: Vec<i64> = array(&json, &floats, "mel2ph").iter().map(|&v| v as i64).collect();
    let mut audio = Vec::new();
    model
        .run_graphs(
            &TickInputs {
                units: array(&json, &floats, "hubert"),
                mel2ph: &mel2ph,
                f0: array(&json, &floats, "f0"),
                volume: array(&json, &floats, "volume"),
                randn: array(&json, &floats, "randn"),
                spk: array(&json, &floats, "spk"),
                formant: json["formant_shift"].as_f64().unwrap() as f32,
                z: array(&json, &floats, "z"),
                phase0: 0.0,
            },
            steps,
            t_start,
            &mut audio,
        )
        .unwrap();
    let (x, cond, flow) = model.last_graph_outputs().unwrap();
    let ex = rel_l2(x, array(&json, &floats, "x"));
    let ec = rel_l2(cond, array(&json, &floats, "cond"));
    let ef = rel_l2(flow, array(&json, &floats, "x_final"));
    eprintln!("tiny parity: x {ex:.2e}, cond {ec:.2e}, flow {ef:.2e}");
    assert!(ex < 1e-4 && ec < 1e-4 && ef < 1e-4, "x {ex}, cond {ec}, flow {ef}");
    assert_eq!(audio.len(), 32 * (frames - start));
    assert!(audio.iter().all(|v| v.is_finite()));
    let _ = std::fs::remove_dir_all(slot);
}

/// The slot the importer writes: weights of the layout's size, a record that reads back with the
/// voice's config, and a manifest the slot loader runs as a DDSP-SVC voice.
#[test]
fn import_writes_a_ddsp_slot() {
    let dir = tiny_dir();
    let layout = DdspLayout::read(&dir).unwrap();
    let voice = import::import_voice(&dir.join("model_100.pt"), &layout).unwrap();
    assert_eq!(voice.global_step, Some(100));
    assert_eq!(voice.checkpoint_dtype, "float32");
    assert!(voice.speakers.is_none());
    let slot = temp_dir("slot");
    let rec = import::write_slot(&slot, &voice, &layout, "Tiny", serde_json::json!({"preset": "sigil_singing_fast"}), std::time::Instant::now()).unwrap();
    assert_eq!(std::fs::metadata(slot.join(import::WEIGHTS_FILE)).unwrap().len(), layout.weights.bytes);
    let back = DdspRecord::read(&slot).unwrap();
    assert_eq!(back.weights_id, layout.weights.weights_id);
    assert_eq!(back.config().unwrap(), voice.config);
    assert_eq!(back.checkpoint_kind, "training");
    assert_eq!(rec.global_step, Some(100));
    let m = load_slot_manifest(&slot).unwrap();
    assert_eq!(m.slot_format(), SlotFormat::Family(FamilyKind::DdspSvc));
    assert_eq!(m.sample_rate, 44_100);
    assert_eq!(super::super::options_of(&m)["preset"], "sigil_singing_fast");
    // The same folder through the folder source and a zip source.
    let folder = temp_dir("folder");
    for f in ["model_100.pt", "config.yaml"] {
        std::fs::copy(dir.join(f), folder.join(f)).unwrap();
    }
    std::fs::copy(dir.join("model_100.pt"), folder.join("model_50.pt")).unwrap();
    let v2 = import::import_voice(&folder, &layout).unwrap();
    assert_eq!(v2.source.checkpoint, "model_100.pt", "the highest step");
    assert_eq!(v2.image, voice.image);
    let zip_path = folder.join("voice.zip");
    {
        use std::io::Write;
        let f = std::fs::File::create(&zip_path).unwrap();
        let mut z = zip::ZipWriter::new(f);
        let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        z.start_file("exp/model_100.pt", opts).unwrap();
        z.write_all(&std::fs::read(dir.join("model_100.pt")).unwrap()).unwrap();
        z.start_file("exp/config.yaml", opts).unwrap();
        z.write_all(&std::fs::read(dir.join("config.yaml")).unwrap()).unwrap();
        z.finish().unwrap();
    }
    let v3 = import::import_voice(&zip_path, &layout).unwrap();
    assert_eq!((v3.source.kind, v3.image == voice.image), ("zip", true));
    let _ = std::fs::remove_dir_all(slot);
    let _ = std::fs::remove_dir_all(folder);
}

/// What the importer refuses, by name.
#[test]
fn import_refuses_what_the_template_cannot_run() {
    let dir = tiny_dir();
    let layout = DdspLayout::read(&dir).unwrap();
    let folder = temp_dir("refuse");
    std::fs::copy(dir.join("model_100.pt"), folder.join("model_100.pt")).unwrap();
    // No config.yaml next to the checkpoint.
    let e = import::import_voice(&folder.join("model_100.pt"), &layout).err().unwrap();
    assert!(e.code == ErrorCode::NotACheckpoint && e.message.contains("config.yaml"), "{e}");
    // Another architecture in the config.
    let cfg = std::fs::read_to_string(dir.join("config.yaml")).unwrap().replace("n_chans: 8", "n_chans: 16");
    std::fs::write(folder.join("config.yaml"), cfg).unwrap();
    let e = import::import_voice(&folder.join("model_100.pt"), &layout).err().unwrap();
    assert!(e.code == ErrorCode::UnsupportedModel && e.message.contains("model.n_chans 16"), "{e}");
    // The default template refuses the tiny voice (the worker's body runs the embedded layout).
    let job = crate::services::neural_audio::vc::import::ImportJob {
        source: dir.join("model_100.pt"),
        out_dir: folder.join("out"),
        ..crate::services::neural_audio::vc::import::ImportJob::for_family("ddsp_svc")
    };
    let e = run_import_job(&job).unwrap_err();
    assert!(e.starts_with("unsupported_model:") && e.contains("block_size 32"), "{e}");
    // Not a checkpoint at all.
    std::fs::write(folder.join("x.pt"), b"hello").unwrap();
    assert_eq!(import::import_voice(&folder.join("x.pt"), &layout).err().unwrap().code, ErrorCode::NotACheckpoint);
    let _ = std::fs::remove_dir_all(folder);
}

/// A voice's shapes: whole 10 ms, the returned samples inside the decoded frames, the context
/// covering the window and its volume windows.
#[test]
fn stream_shapes_cover_the_window() {
    let layout = DdspLayout::embedded().unwrap();
    let setup = VoiceSetup { tta2x: true, f0_min: 65.0, volume_window: 1024, speakers: None };
    for preset in PRESETS {
        let p = preset_shape(preset);
        let tuning = tuning_for(&DdspSvcOptions { preset: preset.into(), ..Default::default() }, 0.0, 0.0, 0.0, &mut Vec::new());
        assert_eq!(tuning.steps, p.steps);
        let want = default_want(tuning.block_16k, 44_100);
        let s = Shape::plan(&layout, &setup, &tuning, tuning.block_16k, want, 1024).unwrap();
        assert_eq!(s.context_16k % 160, 0);
        assert!(s.reflow * 512 >= s.out + 512 + model::DECODE_MARGIN_FRAMES * 512, "{preset}: {s:?}");
        assert!(s.frames * 512 + 512 + 512 <= s.context_16k * 441 / 160, "{preset}: {s:?}");
        assert_eq!(s.units, s.units_a + s.units_b);
        eprintln!("{preset}: {s:?}");
    }
    // sigil_singing at the default crossfade: ~upstream's 100 ms / 1 s window.
    let t = tuning_for(&DdspSvcOptions::default(), 0.0, 0.0, 0.0, &mut Vec::new());
    let s = Shape::plan(&layout, &setup, &t, t.block_16k, default_want(1600, 44_100), 0).unwrap();
    assert_eq!((s.block_16k, s.want_16k, s.out), (1600, 2081, 5736));
    assert!((90..=100).contains(&s.frames) && (17..=21).contains(&s.reflow), "{s:?}");
    // An explicit block from the settings wins (at least 50 ms, whole 10 ms).
    let mut w = Vec::new();
    assert_eq!(tuning_for(&DdspSvcOptions::default(), 0.0, 40.0, 0.0, &mut w).block_16k, 800);
    assert_eq!(w.len(), 1);
    assert_eq!(tuning_for(&DdspSvcOptions::default(), 0.0, 123.0, 30.0, &mut Vec::new()).block_16k, 2080);
    assert_eq!(tuning_for(&DdspSvcOptions::default(), 0.0, 0.0, 30.0, &mut Vec::new()).lookahead_16k, 480);
}

fn tone(n: usize, start: usize, hz: f32, amp: f32) -> Vec<f32> {
    (0..n).map(|i| amp * (2.0 * PI * hz * (start + i) as f32 / 16_000.0).sin()).collect()
}

/// The voice streams through the real harness: exact lengths, finite audio, the gate silences a
/// quiet input without running the models.
#[test]
fn tiny_voice_streams_through_the_harness() {
    let (model, slot) = tiny_model("stream", 220.0, true);
    let block = model.fixed_block().unwrap();
    assert_eq!(block, 1600);
    let mut h = VcStreamHarness::new(Box::new(model), 16_000, 16_000, 1, 1, 1024, 882, 1600).unwrap();
    h.set_voice_controls(&VoiceControls { silence_db: -40.0, ..VoiceControls::default() });
    let mut total = 0usize;
    let mut loud = 0usize;
    for k in 0..12 {
        let x = tone(1600, k * 1600, 220.0, 0.3);
        let y = h.push_and_convert(&x).unwrap();
        assert_eq!(y.len(), 1600);
        assert!(y.iter().all(|v| v.is_finite()));
        total += y.len();
        loud += y.iter().filter(|v| v.abs() > 1e-4).count();
    }
    assert_eq!(total, 12 * 1600);
    assert!(loud > 4 * 1600, "the voice is heard ({loud} loud samples)");
    // Silence: the gate closes, every tick is skipped and the output goes quiet.
    let mut last = Vec::new();
    for _ in 0..8 {
        last = h.push_and_convert(&vec![0.0; 1600]).unwrap();
    }
    assert!(last.iter().all(|v| v.abs() < 1e-6), "gated");
    drop(h);
    let _ = std::fs::remove_dir_all(slot);
}

/// With frames, noise and the NSF phase anchored to absolute time, two consecutive ticks render the
/// frames they share alike (away from the window edges), where fresh draws per tick do not.
#[test]
fn anchored_ticks_agree_on_shared_frames() {
    let run = |anchored: bool| -> f64 {
        let (mut model, slot) = tiny_model(if anchored { "anch" } else { "fresh" }, 330.0, false);
        if !anchored {
            let mut l = all_levers();
            l.levers.set(Lever::AnchoredNoise, false);
            l.levers.set(Lever::NsfPhase0, false);
            model.set_levers_for_test(l);
        }
        let ctx_len = model.configure(1600, 4000, 4800).unwrap();
        let mut outs = Vec::new();
        let mut pos = 0u64;
        for _ in 0..2 {
            let x = tone(ctx_len, pos as usize, 200.0, 0.2);
            let ctx = ConversionCtx {
                mono_16k: &x,
                new_frames: 4000,
                pitch_shift_semitones: 0.0,
                index_rate: 0.0,
                protect: 0.33,
                speaker_id: 0,
                stream_pos_16k: pos,
                reference: None,
                singing: Default::default(),
            };
            outs.push(model.convert(&ctx).unwrap());
            pos += 1600;
        }
        let _ = std::fs::remove_dir_all(slot);
        // Output sample i of tick 1 is absolute (pos1 end - 4000 + i) at 16k; tick 2 is 1600 later.
        // The shared span: tick 1's last (4000 - 1600) * 44100/16000 samples = tick 2's first.
        let shift = 1600 * 441 / 160;
        let a = &outs[0][shift..];
        let b = &outs[1][..a.len()];
        // Middle of the shared span (away from both windows' edges).
        let (a, b) = (&a[500..a.len() - 2000], &b[500..b.len() - 2000]);
        rel_l2(b, a)
    };
    let anchored = run(true);
    let fresh = run(false);
    eprintln!("shared frames: anchored {anchored:.3e}, fresh {fresh:.3e}");
    assert!(anchored < fresh, "anchoring makes overlapping ticks agree ({anchored} vs {fresh})");
}

/// The governor's context step (level 2 and below) halves the flow's Euler steps (at least 2), the
/// costliest knob a DDSP-SVC voice has; retrieval off (level 1) changes nothing, so the governor
/// skips it; full quality restores the steps.
#[test]
fn the_governor_halves_the_flow_steps() {
    use crate::services::neural_audio::vc::engine::governor::QualityLevel;
    let (mut model, slot) = tiny_model("gov", 220.0, false);
    assert_eq!(model.steps(), 3);
    assert!(!model.apply_quality(&QualityLevel::at(1)), "retrieval off changes nothing");
    assert!(model.apply_quality(&QualityLevel::at(2)));
    assert_eq!(model.steps(), 2);
    assert!(!model.apply_quality(&QualityLevel::at(4)), "already at the reduced count");
    assert!(model.apply_quality(&QualityLevel::FULL));
    assert_eq!(model.steps(), 3);
    let _ = std::fs::remove_dir_all(slot);
}

#[test]
fn options_normalize_and_registry_facts() {
    let v = normalize_options(&serde_json::json!({"preset": "nope", "steps": 99, "t_start": 3})).unwrap();
    assert_eq!(v, serde_json::json!({"preset": "sigil_singing", "steps": 50, "t_start": 1.0}));
    assert!(normalize_options(&serde_json::json!({"steps": "many"})).is_err());
    let i = info();
    assert!(available() && i.singing && !i.zero_shot && i.needs_pitch && i.needs_content_encoder);
    assert_eq!(i.execution_providers, &["cuda", "dml"]);
    assert!(i.requires.contains(&TEMPLATE_ID) && i.requires.contains(&VOCODER_ID));
    assert_eq!(FamilyKind::DdspSvc.info().format, "ddsp-svc-6.3");
    // The pitch methods that miss the top of the singing range say so.
    assert!(high_note_caveat("rmvpe").unwrap().contains("octave low"));
    assert!(high_note_caveat("fcpe").unwrap().contains("flat"));
    for ok in ["crepe", "swiftf0", "fusion", "dio"] {
        assert_eq!(high_note_caveat(ok), None, "{ok}");
    }
}

/// The default template's layout matches the catalog entry this build installs it from.
#[test]
fn catalog_entries_match_the_embedded_layout() {
    let catalog: serde_json::Value =
        serde_json::from_slice(&std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("model-catalog").join("vc-ddsp.json")).unwrap()).unwrap();
    let entries = catalog["entries"].as_array().unwrap();
    let find = |id: &str| entries.iter().find(|e| e["id"] == id).unwrap_or_else(|| panic!("{id} in vc-ddsp.json"));
    let layout = DdspLayout::embedded().unwrap();
    let t = find(TEMPLATE_ID);
    assert_eq!(t["status"], "unpublished");
    for g in ["ddsp", "reflow_step"] {
        let info = &layout.graphs[g];
        let file = t["files"].as_array().unwrap().iter().find(|f| f["path"] == info.file.as_str()).unwrap();
        assert_eq!(file["sha256"], info.sha256.as_str(), "{g}");
        assert_eq!(file["size"], info.bytes, "{g}");
    }
    assert!(t["files"].as_array().unwrap().iter().any(|f| f["path"] == "layout.json"));
    let v = find(VOCODER_ID);
    let paths: Vec<&str> = v["files"].as_array().unwrap().iter().map(|f| f["path"].as_str().unwrap()).collect();
    assert_eq!(paths, ["model.onnx", "config.json"]);
}

/// The installed template and vocoder are checked against the layout this build runs: graphs of
/// another export, a truncated graph and a vocoder of another configuration are refused by name.
#[test]
fn installed_assets_are_checked_against_the_layout() {
    let dir = tiny_dir();
    let layout = DdspLayout::read(&dir).unwrap();
    let template = temp_dir("tmpl");
    for f in ["layout.json", "ddsp.onnx", "reflow_step.onnx"] {
        std::fs::copy(dir.join(f), template.join(f)).unwrap();
    }
    let vocoder = temp_dir("voc");
    std::fs::write(vocoder.join("model.onnx"), b"stand-in").unwrap();
    let config = |mels: u32| {
        serde_json::json!({
            "sampling_rate": 44100, "num_mels": mels, "hop_size": 32, "n_fft": 64, "win_size": 64,
            "fmin": 40, "fmax": 16000, "upsample_rates": [4, 4, 2], "mini_nsf": true
        })
        .to_string()
    };
    std::fs::write(vocoder.join("config.json"), config(8)).unwrap();
    let assets = resolve_assets(&layout, &template, &vocoder).unwrap();
    assert_eq!(assets.files.ddsp, template.join("ddsp.onnx"));
    assert_eq!(assets.files.reflow, template.join("reflow_step.onnx"));
    // mini_nsf: the source runs after the first two upsampling stages.
    assert_eq!((assets.vocoder.upp, assets.vocoder.source_sr), (16, 22_050.0));
    // A vocoder of another configuration.
    std::fs::write(vocoder.join("config.json"), config(16)).unwrap();
    let e = resolve_assets(&layout, &template, &vocoder).err().unwrap();
    assert!(e.contains("not the one"), "{e}");
    std::fs::write(vocoder.join("config.json"), config(8)).unwrap();
    // A truncated graph.
    let ddsp = std::fs::read(template.join("ddsp.onnx")).unwrap();
    std::fs::write(template.join("ddsp.onnx"), &ddsp[..ddsp.len() / 2]).unwrap();
    let e = resolve_assets(&layout, &template, &vocoder).err().unwrap();
    assert!(e.contains("ddsp.onnx is") && e.contains("reinstall"), "{e}");
    std::fs::write(template.join("ddsp.onnx"), &ddsp).unwrap();
    // Another export of the graphs (its layout records other bytes).
    let mut other: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("layout.json")).unwrap()).unwrap();
    other["graphs"]["reflow_step"]["sha256"] = "1".repeat(64).into();
    std::fs::write(template.join("layout.json"), other.to_string()).unwrap();
    let e = resolve_assets(&layout, &template, &vocoder).err().unwrap();
    assert!(e.contains("reflow_step graph is another export"), "{e}");
    let _ = std::fs::remove_dir_all(template);
    let _ = std::fs::remove_dir_all(vocoder);
}

/// Upstream's volume gate is part of the voice's output whatever the silence skip: with the skip
/// off the models run on silence and the gate still silences it (the stand-in vocoder's source
/// sings on at the constant F0 otherwise, as the real one's noise excitation would hiss); with the
/// skip on the same ticks return the same zeros without running the models.
#[test]
fn the_gate_applies_whatever_the_silence_skip() {
    let mut quiet_ticks = [0usize; 2];
    for (i, skip) in [false, true].into_iter().enumerate() {
        let (mut model, slot) = tiny_model(if skip { "gate-skip" } else { "gate-run" }, 220.0, skip);
        model.set_voice_controls(&VoiceControls { silence_db: -40.0, ..VoiceControls::default() });
        let ctx_len = model.configure(1600, 1600, 4800).unwrap();
        // A tone, then silence long enough for whole windows of it.
        let tone_len = ctx_len + 3200;
        let mut x = tone(tone_len, 0, 220.0, 0.3);
        x.resize(tone_len + ctx_len + 9600, 0.0);
        let mut pos = 0usize;
        let mut loud_ticks = 0;
        while pos + ctx_len <= x.len() {
            let ctx = ConversionCtx {
                mono_16k: &x[pos..pos + ctx_len],
                new_frames: 1600,
                pitch_shift_semitones: 0.0,
                index_rate: 0.0,
                protect: 0.33,
                speaker_id: 0,
                stream_pos_16k: pos as u64,
                reference: None,
                singing: Default::default(),
            };
            let y = model.convert(&ctx).unwrap();
            assert!(y.iter().all(|v| v.is_finite()));
            // The returned samples are the context's last 100 ms: silent once they start a volume
            // window (1024 samples at 44.1 kHz) and a frame after the tone's end.
            let returned_from = pos + ctx_len - 1600;
            if returned_from > tone_len + 600 {
                assert!(y.iter().all(|&v| v == 0.0), "skip {skip}: the gate silences the quiet tick at {pos}");
                assert_eq!(model.tick_stats().skipped, skip, "skip {skip}: models ran at {pos}");
                quiet_ticks[i] += 1;
            } else if pos + ctx_len <= tone_len {
                loud_ticks += usize::from(y.iter().any(|v| v.abs() > 1e-3));
            }
            pos += 1600;
        }
        assert!(loud_ticks > 0, "skip {skip}: the tone is heard");
        let _ = std::fs::remove_dir_all(slot);
    }
    assert!(quiet_ticks[0] >= 3 && quiet_ticks == [quiet_ticks[0]; 2], "{quiet_ticks:?}");
}

/// The flow never starts below the `t_start` the voice was trained from (upstream
/// `main_reflow.py`): the presets' 0.7 is raised quietly, an explicit option with a warning, and
/// a start above the floor stays.
#[test]
fn the_flow_starts_at_or_above_the_trained_t_start() {
    let t = |opt: Option<f32>, floor: f32| -> (f32, usize) {
        let mut w = Vec::new();
        let tuning = tuning_for(&DdspSvcOptions { t_start: opt, ..Default::default() }, floor, 0.0, 0.0, &mut w);
        (tuning.t_start, w.len())
    };
    assert_eq!(t(None, 0.0), (0.7, 0));
    assert_eq!(t(None, 0.8), (0.8, 0));
    assert_eq!(t(Some(0.5), 0.8), (0.8, 1));
    assert_eq!(t(Some(0.9), 0.8), (0.9, 0));
    assert_eq!(t(Some(0.2), 0.0), (0.2, 0));
    assert_eq!(t(Some(0.5), f32::NAN), (0.5, 0));
    let mut w = Vec::new();
    effective_t_start(&DdspSvcOptions { t_start: Some(0.3), ..Default::default() }, 0.75, &mut w);
    assert!(w[0].contains("trained with t_start 0.75"), "{w:?}");
    // The importer keeps the trained floor in the record and says when it is above the presets'.
    let dir = tiny_dir();
    let layout = DdspLayout::read(&dir).unwrap();
    let folder = temp_dir("tstart");
    std::fs::copy(dir.join("model_100.pt"), folder.join("model_100.pt")).unwrap();
    let cfg = std::fs::read_to_string(dir.join("config.yaml")).unwrap().replace("t_start: 0.0", "t_start: 0.8");
    std::fs::write(folder.join("config.yaml"), cfg).unwrap();
    let voice = import::import_voice(&folder.join("model_100.pt"), &layout).unwrap();
    assert_eq!(voice.config.t_start, 0.8);
    assert!(voice.warnings.iter().any(|w| w.contains("t_start 0.8")), "{:?}", voice.warnings);
    let slot = temp_dir("tstart-slot");
    import::write_slot(&slot, &voice, &layout, "T", serde_json::json!({}), std::time::Instant::now()).unwrap();
    assert_eq!(DdspRecord::read(&slot).unwrap().config().unwrap().t_start, 0.8);
    let _ = std::fs::remove_dir_all(folder);
    let _ = std::fs::remove_dir_all(slot);
}

/// The output buffer a voice needs follows its burst: a 100 ms preset on CUDA keeps the measured
/// 70 / 140 ms, DirectML and `upstream_gui` (and long blocks, extra steps) get more, never past the
/// voice changer's range (prefill 5-300 ms, queue at least prefill + 5 ms, at most 600 ms).
#[test]
fn output_buffer_follows_the_preset_and_ep() {
    let o = |preset: &str, steps: Option<u32>| DdspSvcOptions { preset: preset.into(), steps, ..Default::default() };
    assert_eq!(min_output_buffer_ms(&o("sigil_singing", None), 0.0, "cuda"), (70.0, 140.0));
    assert_eq!(min_output_buffer_ms(&o("sigil_singing_fast", None), 0.0, "cuda"), (70.0, 140.0));
    assert_eq!(min_output_buffer_ms(&o("sigil_singing", None), 0.0, "dml"), (80.0, 160.0));
    assert_eq!(min_output_buffer_ms(&o("upstream_gui", None), 0.0, "cuda"), (100.0, 200.0));
    // upstream_gui on DirectML: bursts of about 100 ms underran 30 times per 9 s at 70 ms.
    assert_eq!(min_output_buffer_ms(&o("upstream_gui", None), 0.0, "dml"), (150.0, 300.0));
    assert_eq!(min_output_buffer_ms(&o("upstream_gui", None), 0.0, "cpu"), (150.0, 300.0));
    // A long explicit block ticks like upstream_gui; extra flow steps add their cost.
    assert_eq!(min_output_buffer_ms(&o("sigil_singing", None), 400.0, "cuda"), (100.0, 200.0));
    assert_eq!(min_output_buffer_ms(&o("sigil_singing", Some(50)), 0.0, "cuda"), (100.0, 200.0));
    assert_eq!(min_output_buffer_ms(&o("sigil_singing", Some(3)), 0.0, "cuda"), (70.0, 140.0));
    for preset in PRESETS {
        for steps in [None, Some(1), Some(50)] {
            for block in [0.0, 50.0, 500.0] {
                for ep in ["cuda", "dml", "cpu"] {
                    let (p, q) = min_output_buffer_ms(&o(preset, steps), block, ep);
                    assert!((70.0..=300.0).contains(&p) && q >= p + 5.0 && q <= 600.0, "{preset} {steps:?} {block} {ep}: {p} / {q}");
                }
            }
        }
    }
    // A loaded voice: the CUDA figures only when all three graphs run on CUDA.
    use crate::services::diagnostics::EpOutcome;
    let outcome = |k: &str, active: &str| (k.to_string(), EpOutcome::new("cuda", active, None));
    let gui = o("upstream_gui", None);
    let all_cuda = [outcome("vc.content", "dml"), outcome("vc.ddsp", "cuda"), outcome("vc.reflow", "cuda"), outcome("vc.vocoder", "cuda")];
    assert_eq!(voice_output_buffer_ms(&gui, 0.0, &all_cuda), (100.0, 200.0));
    let fell = [outcome("x.vc.ddsp", "cuda"), outcome("x.vc.reflow", "dml"), outcome("x.vc.vocoder", "cuda")];
    assert_eq!(voice_output_buffer_ms(&gui, 0.0, &fell), (150.0, 300.0));
    assert_eq!(voice_output_buffer_ms(&gui, 0.0, &[]), (150.0, 300.0));
}

/// One tensor of a rewritten checkpoint: its key, view shape and stride, and its storage values.
struct Entry {
    key: String,
    shape: Vec<usize>,
    stride: Vec<usize>,
    storage: Vec<f32>,
}

impl Entry {
    fn contiguous(key: &str, shape: &[usize], values: Vec<f32>) -> Entry {
        let mut stride = vec![1usize; shape.len()];
        for d in (0..shape.len().saturating_sub(1)).rev() {
            stride[d] = stride[d + 1] * shape[d + 1];
        }
        Entry { key: key.into(), shape: shape.to_vec(), stride, storage: values }
    }
}

/// The tiny checkpoint's weights as entries (`fold`: each weight_norm pair saved as its folded
/// weight, as after `remove_weight_norm`).
fn tiny_entries(fold: bool) -> Vec<Entry> {
    use crate::services::neural_audio::vc::import::fold::weight_norm_fold;
    let dir = tiny_dir();
    let layout = DdspLayout::read(&dir).unwrap();
    let f = std::io::BufReader::new(std::fs::File::open(dir.join("model_100.pt")).unwrap());
    let mut c = super::checkpoint::DdspCheckpoint::open(f).unwrap();
    let mut out = Vec::new();
    for t in &layout.weights.tensors {
        match (&t.source.key, &t.source.weight_norm) {
            (Some(k), _) => {
                if c.has(k) {
                    out.push(Entry::contiguous(k, &c.shape(k).unwrap(), c.read_f32(k).unwrap()));
                }
            }
            (None, Some(wn)) if fold => {
                let w = weight_norm_fold(&c.read_f32(&wn.g).unwrap(), &c.read_f32(&wn.v).unwrap(), &t.source_shape, wn.dim).unwrap();
                out.push(Entry::contiguous(t.source.folded_key().unwrap(), &t.source_shape, w));
            }
            (None, Some(wn)) => {
                for k in [&wn.g, &wn.v] {
                    out.push(Entry::contiguous(k, &c.shape(k).unwrap(), c.read_f32(k).unwrap()));
                }
            }
            _ => unreachable!(),
        }
    }
    out
}

/// `{"global_step": 100, "model": {entries}}` as `dir/model_100.pt`, with the tiny config.yaml.
fn write_tiny_checkpoint(dir: &Path, entries: &[Entry]) -> PathBuf {
    use crate::services::neural_audio::vc::import::pickle::tests::P;
    use crate::services::neural_audio::vc::import::pth::tests::torch_zip;
    let i32s = |v: &[usize]| v.iter().map(|&d| i32::try_from(d).unwrap()).collect::<Vec<i32>>();
    let mut p = P::new().op(b'}').op(b'(').str("global_step").int(100).str("model").op(b'}').op(b'(');
    let mut storages = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        let numel = i32::try_from(e.storage.len()).unwrap();
        p = p.str(&e.key).tensor("FloatStorage", &i.to_string(), numel, 0, &i32s(&e.shape), &i32s(&e.stride));
        storages.push((i.to_string(), e.storage.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()));
    }
    let pickle = p.op(b'u').op(b'u').stop();
    let refs: Vec<(&str, Vec<u8>)> = storages.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    let path = dir.join("model_100.pt");
    std::fs::write(&path, torch_zip(&pickle, &refs)).unwrap();
    std::fs::copy(tiny_dir().join("config.yaml"), dir.join("config.yaml")).unwrap();
    path
}

/// A checkpoint saved with its weight norms removed (`X.weight` instead of `X.weight_g` /
/// `X.weight_v`) imports to the same weights as the original.
#[test]
fn import_takes_checkpoints_with_the_weight_norm_removed() {
    let dir = tiny_dir();
    let layout = DdspLayout::read(&dir).unwrap();
    let original = import::import_voice(&dir.join("model_100.pt"), &layout).unwrap();
    for fold in [false, true] {
        let folder = temp_dir(if fold { "folded" } else { "rewritten" });
        let entries = tiny_entries(fold);
        assert_eq!(entries.iter().any(|e| e.key.ends_with("_g")), !fold);
        let voice = import::import_voice(&write_tiny_checkpoint(&folder, &entries), &layout).unwrap_or_else(|e| panic!("fold {fold}: {e}"));
        assert!(voice.image == original.image, "fold {fold}: the same weights");
        let _ = std::fs::remove_dir_all(folder);
    }
}

/// Shapes are checked before anything is read: a speaker table larger than the loader takes, a
/// broadcast (stride-0) speaker table, a weight-norm magnitude of the wrong size. A real table
/// imports (with a note when config.yaml counts other speakers).
#[test]
fn import_checks_shapes_before_reading() {
    let dir = tiny_dir();
    let layout = DdspLayout::read(&dir).unwrap();
    let key = layout.speakers.key.clone();
    let dim = layout.speakers.dim;
    let run = |tag: &str, extra: Vec<Entry>, edit: &dyn Fn(&mut Vec<Entry>)| {
        let folder = temp_dir(tag);
        let mut entries = tiny_entries(false);
        entries.extend(extra);
        edit(&mut entries);
        let r = import::import_voice(&write_tiny_checkpoint(&folder, &entries), &layout);
        let _ = std::fs::remove_dir_all(folder);
        r
    };
    // 4 Mi rows of `dim` values over one stored row: 128 MiB, past the 64 MiB the loader takes.
    let huge = Entry { key: key.clone(), shape: vec![1 << 22, dim], stride: vec![0, 1], storage: vec![0.5; dim] };
    let e = run("spk-huge", vec![huge], &|_| {}).err().unwrap();
    assert!(e.code == ErrorCode::ShapeMismatch && e.message.contains("speakers"), "{e}");
    // A small broadcast table: refused as a view, not a weight.
    let broadcast = Entry { key: key.clone(), shape: vec![3, dim], stride: vec![0, 1], storage: vec![0.5; dim] };
    let e = run("spk-broadcast", vec![broadcast], &|_| {}).err().unwrap();
    assert!(e.code == ErrorCode::TensorInvalid && e.message.contains("broadcast"), "{e}");
    // A weight-norm magnitude with the wrong number of values (a stride-0 view of millions).
    let e = run("g-shape", Vec::new(), &|entries| {
        let g = entries.iter_mut().find(|e| e.key.ends_with("weight_g")).unwrap();
        g.shape = vec![1 << 20, 1, 1];
        g.stride = vec![0, 0, 0];
        g.storage.truncate(1);
    })
    .err()
    .unwrap();
    assert!(e.code == ErrorCode::ShapeMismatch && e.message.contains("weight_g"), "{e}");
    // A real two-speaker table.
    let rows: Vec<f32> = (0..2 * dim).map(|i| i as f32 * 0.01).collect();
    let voice = run("spk-ok", vec![Entry::contiguous(&key, &[2, dim], rows.clone())], &|_| {}).unwrap();
    assert_eq!(voice.speakers, Some((rows, 2)));
    assert!(voice.warnings.iter().any(|w| w.contains("n_spk = 1")), "{:?}", voice.warnings);
}

// ---------------------------------------------------------------------------------------------
// Real models (manual): the default template, the real vocoder and ContentVec
// ---------------------------------------------------------------------------------------------

/// The real assets: `SIGIL_TEST_MODELS_DIR` holding `vc/ddsp-svc-6.3/`,
/// `vc/vocoder-pc-nsf-hifigan-44k/`, `vc/embedder-contentvec/model.fp32.onnx` and, for the pitch
/// methods that need them, `vc/pitch-fcpe/model.onnx` and `vc/pitch-rmvpe/model.onnx` (the
/// `SIGIL_LOCAL_ASSETS` layout, e.g. `%LOCALAPPDATA%/sigil-dev-cache/vc-assets`; `SIGIL_TEST_RMVPE`
/// names an RMVPE file kept elsewhere), and a voice folder (`SIGIL_TEST_DDSP_VOICE`, default
/// `<root>/ddsp-voice`: `model_*.pt` + `config.yaml`; `export_templates.py --voice` writes a
/// synthetic one).
pub(super) struct RealAssets {
    pub template: PathBuf,
    pub vocoder: PathBuf,
    pub contentvec: PathBuf,
    pub fcpe: Option<PathBuf>,
    pub rmvpe: Option<PathBuf>,
    pub voice: PathBuf,
}

pub(super) fn real_assets() -> Option<RealAssets> {
    let root = PathBuf::from(std::env::var_os("SIGIL_TEST_MODELS_DIR")?);
    let file = |p: PathBuf| p.is_file().then_some(p);
    let a = RealAssets {
        template: root.join("vc").join("ddsp-svc-6.3"),
        vocoder: root.join("vc").join("vocoder-pc-nsf-hifigan-44k"),
        contentvec: root.join("vc").join("embedder-contentvec").join("model.fp32.onnx"),
        fcpe: file(root.join("vc").join("pitch-fcpe").join("model.onnx")),
        rmvpe: std::env::var_os("SIGIL_TEST_RMVPE")
            .map(PathBuf::from)
            .and_then(file)
            .or_else(|| file(root.join("vc").join("pitch-rmvpe").join("model.onnx"))),
        voice: std::env::var_os("SIGIL_TEST_DDSP_VOICE").map(PathBuf::from).unwrap_or_else(|| root.join("ddsp-voice")),
    };
    (a.template.is_dir() && a.vocoder.is_dir() && a.contentvec.is_file() && a.voice.is_dir()).then_some(a)
}

/// F0 (Hz per `hop` samples, 0 = unvoiced) of `x` at `sr` over `fmin..fmax`, pYIN-style: per
/// frame, YIN's cumulative mean normalised difference, and for each of 20 thresholds (0.025 to 0.5)
/// a vote for the first dip under it (YIN's rule) or for "unvoiced" when none is; then the most
/// likely path through those candidates (Viterbi; a pitch jump costs 4 per octave, capped at two, a
/// voicing change 4). Frames quieter than -50 dBFS RMS are unvoiced. The output tracker of the
/// singing figures: unlike the pitch bank's extractors (about 2 kHz at most) it follows the output
/// to D#7 and beyond. On converted sweeps it agrees with librosa's pYIN (0 gross errors where
/// classic YIN read 3.0-3.3 %), and still finds the octave-low notes RMVPE gives the D#7 end.
pub(super) fn pitch_track(x: &[f32], sr: f32, hop: usize, fmin: f32, fmax: f32) -> Vec<f32> {
    const THRESHOLDS: usize = 20;
    let win = (sr / fmin * 2.0) as usize;
    let (tmin, tmax) = (((sr / fmax).floor() as usize).max(2), (sr / fmin).ceil() as usize);
    let mut d = vec![0.0f64; tmax + 2];
    let mut cmnd = vec![1.0f64; tmax + 2];
    // Per frame: candidate states (Hz, probability); the last one is "unvoiced" (0 Hz).
    let mut frames: Vec<Vec<(f32, f64)>> = Vec::new();
    let mut s = 0;
    while s + win + tmax + 2 < x.len() {
        let frame = &x[s..s + win + tmax + 2];
        s += hop;
        let mut states: Vec<(f32, f64)> = Vec::new();
        let mut unvoiced = THRESHOLDS;
        let rms = (frame[..win].iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / win as f64).sqrt();
        if rms >= 0.003 {
            let mut cum = 0.0;
            for tau in 1..=tmax + 1 {
                d[tau] = (0..win).map(|i| f64::from(frame[i] - frame[i + tau]).powi(2)).sum();
                cum += d[tau];
                cmnd[tau] = d[tau] * tau as f64 / cum.max(1e-12);
            }
            let dips: Vec<usize> = (tmin..=tmax).filter(|&t| cmnd[t] < cmnd[t - 1] && cmnd[t] <= cmnd[t + 1]).collect();
            let mut votes: Vec<(usize, usize)> = Vec::new();
            for k in 1..=THRESHOLDS {
                let threshold = 0.025 * k as f64;
                if let Some(&t) = dips.iter().find(|&&t| cmnd[t] < threshold) {
                    unvoiced -= 1;
                    match votes.iter_mut().find(|(c, _)| *c == t) {
                        Some(v) => v.1 += 1,
                        None => votes.push((t, 1)),
                    }
                }
            }
            for (t, n) in votes {
                let (a, b, c) = (cmnd[t - 1], cmnd[t], cmnd[t + 1]);
                let den = a - 2.0 * b + c;
                let shift = if den.abs() > 1e-12 { 0.5 * (a - c) / den } else { 0.0 };
                states.push(((f64::from(sr) / (t as f64 + shift.clamp(-1.0, 1.0))) as f32, n as f64 / THRESHOLDS as f64));
            }
        }
        let p_unvoiced = if states.is_empty() { 1.0 } else { (unvoiced as f64 / THRESHOLDS as f64).max(0.01) };
        states.push((0.0, p_unvoiced));
        frames.push(states);
    }
    if frames.is_empty() {
        return Vec::new();
    }
    let emission = |p: f64| -(p + 1e-3).ln();
    let transition = |a: f32, b: f32| -> f64 {
        match (a > 0.0, b > 0.0) {
            (true, true) => 4.0 * (f64::from(b) / f64::from(a)).log2().abs().min(2.0),
            (false, false) => 0.0,
            _ => 4.0,
        }
    };
    let mut cost: Vec<f64> = frames[0].iter().map(|&(_, p)| emission(p)).collect();
    let mut back: Vec<Vec<usize>> = vec![Vec::new()];
    for i in 1..frames.len() {
        let mut next = Vec::with_capacity(frames[i].len());
        let mut from = Vec::with_capacity(frames[i].len());
        for &(f, p) in &frames[i] {
            let (j, c) = frames[i - 1]
                .iter()
                .enumerate()
                .map(|(j, &(pf, _))| (j, cost[j] + transition(pf, f)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap_or((0, 0.0));
            next.push(c + emission(p));
            from.push(j);
        }
        cost = next;
        back.push(from);
    }
    let mut k = cost.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).map(|(k, _)| k).unwrap_or(0);
    let mut out = vec![0.0f32; frames.len()];
    for i in (0..frames.len()).rev() {
        out[i] = frames[i][k].0;
        if i > 0 {
            k = back[i][k];
        }
    }
    out
}

#[test]
fn pitch_track_follows_tones_to_d_sharp_7() {
    let sr = 44_100.0f32;
    for hz in [82.4f32, 440.0, 1244.5, 2489.0] {
        let x: Vec<f32> = (0..22_050).map(|i| 0.5 * (2.0 * PI * hz * i as f32 / sr).sin() + 0.2 * (4.0 * PI * hz * i as f32 / sr).sin()).collect();
        let f = pitch_track(&x, sr, 512, 70.0, 2800.0);
        assert!(f[2..f.len() - 2].iter().all(|&v| (v / hz - 1.0).abs() < 0.01), "{hz} Hz -> {f:?}");
    }
    // A strong second harmonic does not pull the track an octave up, nor a weak fundamental an octave
    // down.
    let x: Vec<f32> = (0..22_050).map(|i| 0.2 * (2.0 * PI * 300.0 * i as f32 / sr).sin() + 0.5 * (4.0 * PI * 300.0 * i as f32 / sr).sin()).collect();
    let f = pitch_track(&x, sr, 512, 70.0, 2800.0);
    assert!(f[2..f.len() - 2].iter().all(|&v| (v / 300.0 - 1.0).abs() < 0.01), "{f:?}");
    // Silence and white noise are unvoiced.
    assert!(pitch_track(&[0.0; 22_050], sr, 512, 70.0, 2800.0).iter().all(|&f| f == 0.0));
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let noise: Vec<f32> = (0..22_050)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect();
    let f = pitch_track(&noise, sr, 512, 70.0, 2800.0);
    assert!(f.iter().filter(|&&v| v > 0.0).count() * 10 < f.len(), "noise is mostly unvoiced: {f:?}");
}

/// A sung test signal at `rate`: after `lead` s of silence, 0.5 s held at `f_lo`, a log sweep
/// `f_lo → f_hi` over `seconds`, 0.5 s held at `f_hi`, a 30 ms fade and 0.3 s of silence; ±25 cent
/// vibrato at 5.5 Hz, eight harmonics through a vowel-like envelope. Returns the audio and its F0
/// per 10 ms (0 = silence and the fade).
pub(super) fn singing_sweep(rate: u32, seconds: f32, lead: f32, f_lo: f32, f_hi: f32) -> (Vec<f32>, Vec<f32>) {
    let sr = f64::from(rate);
    let per_10ms = (rate / 100) as usize;
    let sung = 0.5 + f64::from(seconds) + 0.5;
    let n = ((f64::from(lead) + sung + 0.03 + 0.3) * sr) as usize;
    let mut x = vec![0.0f32; n];
    let mut f0 = vec![0.0f32; n / per_10ms + 1];
    let mut phase = 0.0f64;
    let (lo, hi) = (f64::from(f_lo), f64::from(f_hi));
    for (i, v) in x.iter_mut().enumerate() {
        let t = i as f64 / sr - f64::from(lead);
        if t < 0.0 || t >= sung + 0.03 {
            continue;
        }
        let base = if t < 0.5 { lo } else { lo * (hi / lo).powf(((t - 0.5) / f64::from(seconds)).min(1.0)) };
        let f = base * 2f64.powf(0.25 * (2.0 * std::f64::consts::PI * 5.5 * t).sin() / 12.0);
        phase += f / sr;
        let mut s = 0.0;
        for h in 1..=8 {
            let fh = f * h as f64;
            if fh > 7_000.0 {
                break;
            }
            let formant = (-((fh - 700.0) / 900.0).powi(2)).exp() + 0.5 * (-((fh - 2_500.0) / 1_200.0).powi(2)).exp() + 0.05;
            s += formant / h as f64 * (2.0 * std::f64::consts::PI * h as f64 * phase).sin();
        }
        let fade = (t / 0.02).min(1.0).min((sung + 0.03 - t) / 0.03);
        *v = (0.25 * fade * s) as f32;
        if i % per_10ms == 0 && t < sung {
            f0[i / per_10ms] = f as f32;
        }
    }
    (x, f0)
}

/// How a pitch track (10 ms frames, 0 = unvoiced) follows a reference track scaled by a transpose.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PitchScore {
    /// Gross pitch error: frames voiced in both whose pitch is off by more than 20 %, over the
    /// frames voiced in both (the usual definition).
    pub gpe: f64,
    /// Reference frames voiced (in the scored span) and those the track voices too.
    pub ref_voiced: usize,
    pub both_voiced: usize,
    /// RMSE (cents) of the frames voiced in both that are not gross errors.
    pub cents: f64,
    /// Track lag behind the reference (10 ms frames).
    pub lag: usize,
}

impl PitchScore {
    /// Share of the reference's voiced frames the track voices too.
    pub fn recall(&self) -> f64 {
        self.both_voiced as f64 / self.ref_voiced.max(1) as f64
    }
}

/// Score `track` against `reference · 2^(semitones/12)` over the reference frames `span` whose
/// expected pitch is inside `hz` (frames outside are not counted), at the lag (`lags`) with the
/// smallest median pitch error (cents, over at least half the best overlap): with a sweep and a
/// vibrato in the reference the lag is unambiguous, which a count of gross errors is not, and the
/// median ignores the gross errors that a mean would trade alignment for.
pub(super) fn pitch_score(
    reference: &[f32],
    track: &[f32],
    semitones: f32,
    span: std::ops::Range<usize>,
    hz: std::ops::Range<f32>,
    lags: std::ops::Range<usize>,
) -> PitchScore {
    let k = 2f64.powf(f64::from(semitones) / 12.0);
    let score_at = |lag: usize| -> (PitchScore, f64) {
        let (mut rv, mut both, mut gross, mut sq) = (0usize, 0usize, 0usize, 0.0f64);
        let mut errors: Vec<f64> = Vec::new();
        for i in span.clone() {
            let Some(&r) = reference.get(i) else { break };
            let want = f64::from(r) * k;
            if r <= 0.0 || !hz.contains(&(want as f32)) {
                continue;
            }
            rv += 1;
            let o = track.get(i + lag).copied().unwrap_or(0.0);
            if o <= 0.0 {
                continue;
            }
            both += 1;
            let cents = 1200.0 * (f64::from(o) / want).log2();
            errors.push(cents.abs());
            if (f64::from(o) / want - 1.0).abs() > 0.2 {
                gross += 1;
            } else {
                sq += cents * cents;
            }
        }
        let good = both - gross;
        let s = PitchScore { gpe: gross as f64 / both.max(1) as f64, ref_voiced: rv, both_voiced: both, cents: (sq / good.max(1) as f64).sqrt(), lag };
        errors.sort_by(f64::total_cmp);
        (s, errors.get(errors.len() / 2).copied().unwrap_or(f64::INFINITY))
    };
    let all: Vec<(PitchScore, f64)> = lags.map(score_at).collect();
    let most = all.iter().map(|(s, _)| s.both_voiced).max().unwrap_or(0);
    all.into_iter()
        .filter(|(s, _)| 2 * s.both_voiced >= most && s.both_voiced > 0)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(s, _)| s)
        .unwrap_or_default()
}

#[test]
fn pitch_score_finds_the_lag_and_counts_gross_errors() {
    // A vibrato reference, the track an octave up (+12), 7 frames late, with 3 octave errors and 10
    // unvoiced frames.
    let reference: Vec<f32> = (0..200).map(|i| if i < 20 { 0.0 } else { 200.0 * (1.0 + 0.03 * (i as f32 * 0.35).sin()) }).collect();
    let mut track = vec![0.0f32; 7];
    track.extend(reference.iter().map(|&f| 2.0 * f));
    for i in [50, 60, 70] {
        track[i + 7] *= 2.0;
    }
    for v in &mut track[107..117] {
        *v = 0.0;
    }
    let s = pitch_score(&reference, &track, 12.0, 0..200, 0.0..f32::INFINITY, 0..20);
    assert_eq!(s.lag, 7);
    assert_eq!((s.ref_voiced, s.both_voiced), (180, 170));
    assert!((s.gpe - 3.0 / 170.0).abs() < 1e-12, "{s:?}");
    assert!(s.cents < 1e-3);
    assert!((s.recall() - 170.0 / 180.0).abs() < 1e-12);
    // Only the frames whose expected pitch is in the band count.
    let hi = pitch_score(&reference, &track, 12.0, 0..200, 410.0..f32::INFINITY, 7..8);
    assert!(hi.ref_voiced > 0 && hi.ref_voiced < 180, "{hi:?}");
}

/// The offset (samples) in `lo..=hi` at which `b[i + offset]` best matches `a[i]` over
/// `a[from..from + len]` (normalised cross-correlation).
fn best_offset(a: &[f32], b: &[f32], from: usize, len: usize, lo: i64, hi: i64) -> i64 {
    let mut best = (f64::NEG_INFINITY, 0i64);
    for off in lo..=hi {
        let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
        for (i, &x) in a.iter().enumerate().take(from + len).skip(from) {
            let j = i as i64 + off;
            let Some(&y) = usize::try_from(j).ok().and_then(|j| b.get(j)) else {
                continue;
            };
            let (x, y) = (f64::from(x), f64::from(y));
            ab += x * y;
            aa += x * x;
            bb += y * y;
        }
        let r = ab / (aa * bb).sqrt().max(1e-12);
        if r > best.0 {
            best = (r, off);
        }
    }
    best.1
}

fn write_wav(path: &Path, audio: &[f32], rate: u32) {
    let spec = hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for &v in audio {
        w.write_sample(v).unwrap();
    }
    w.finalize().unwrap();
}

/// A real voice for one EP, pitch method and preset (the levers the voice changer's `auto` gives a
/// GPU rung: the algorithmic ones, IOBinding, and CUDA graphs on CUDA).
fn real_model(a: &RealAssets, layout: &DdspLayout, slot: &Path, ep: &str, pitch_method: &str, preset: &str) -> Result<DdspSvcModel, String> {
    use crate::services::neural_audio::gpu::GpuLevers;
    use crate::services::neural_audio::vc::engine::content::{ContentEncoderPool, EmbedderKind};
    use crate::services::neural_audio::vc::engine::options::{F0Range, PrecisionMode};
    use crate::services::neural_audio::vc::f0_fusion::{create_pitch_extractor_with, PitchBankRequest};
    let record = DdspRecord::read(slot).ok_or("no record")?;
    let config = record.config().ok_or("no config")?;
    let opts = VcSessionOptions { intra_threads: Some(2), ..VcSessionOptions::default() };
    let units = ContentEncoderPool::global().acquire(EmbedderKind::ContentVec768, Some(&a.contentvec), ep, &opts, PrecisionMode::Fp32)?;
    let mut levers = all_levers();
    levers.levers.set(Lever::PartialDecode, true);
    if ep == "cuda" {
        levers.levers.set(Lever::CudaGraphs, true);
    }
    // `SIGIL_TEST_DDSP_LEVERS_OFF=f0_cache,...`: run those levers' legacy paths.
    for name in std::env::var("SIGIL_TEST_DDSP_LEVERS_OFF").unwrap_or_default().split(',').map(str::trim) {
        if let Some(l) = Lever::ALL.iter().copied().find(|l| l.as_str() == name) {
            levers.levers.set(l, false);
        }
    }
    let bank = create_pitch_extractor_with(
        &PitchBankRequest {
            method: pitch_method,
            ep,
            opts: &opts,
            range: F0Range::Singing,
            fcpe_path: a.fcpe.as_deref(),
            rmvpe_path: a.rmvpe.as_deref(),
            app_data: None,
            small_models_on_cpu: true,
            precision: crate::services::neural_audio::vc::engine::PrecisionMode::Auto,
            expected_call_frames: None,
            max_context_16k: None,
        },
        GpuLevers::from_effective(&levers),
    )?;
    let tuning = tuning_for(&DdspSvcOptions { preset: preset.into(), ..Default::default() }, 0.0, 0.0, 0.0, &mut Vec::new());
    DdspSvcModel::new(ModelParts {
        layout: layout.clone(),
        files: DdspFiles { ddsp: a.template.join("ddsp.onnx"), reflow: a.template.join("reflow_step.onnx"), vocoder: a.vocoder.join("model.onnx") },
        vocoder: VocoderRuntime::from_config(&a.vocoder.join("config.json"), layout)?,
        weights: slot.join(import::WEIGHTS_FILE),
        setup: VoiceSetup { tta2x: config.tta2x(), f0_min: config.f0_min, volume_window: config.volume_window() as usize, speakers: None },
        tuning,
        units: Box::new(model::PooledUnits(units)),
        pitch: bank.extractor,
        ep: ep.into(),
        session: opts,
        levers,
        noise_seed: 11,
        silence_skip: true,
        outcomes: bank.outcomes,
        warnings: bank.notes,
    })
}

/// A voice inside the harness that logs the F0 the vocoder received for every decoded frame
/// (absolute frame → Hz; a later tick overwrites an earlier one).
struct F0Probe {
    inner: DdspSvcModel,
    log: std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<i64, f32>>>,
}

impl VoiceConversionModel for F0Probe {
    fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        let out = self.inner.convert(ctx);
        if let Some((first, f0)) = self.inner.last_decoded_f0() {
            let mut log = self.log.lock().unwrap();
            for (i, &v) in f0.iter().enumerate() {
                log.insert(first + i as i64, v);
            }
        }
        out
    }

    fn input_sample_rate(&self) -> u32 {
        self.inner.input_sample_rate()
    }

    fn output_sample_rate(&self) -> u32 {
        self.inner.output_sample_rate()
    }

    fn min_context_samples(&self) -> usize {
        self.inner.min_context_samples()
    }

    fn warmup_at(&mut self, ctx: &ConversionCtx<'_>) -> Result<(), String> {
        self.inner.warmup_at(ctx)
    }

    fn set_voice_controls(&mut self, controls: &VoiceControls) {
        self.inner.set_voice_controls(controls);
    }

    fn apply_quality(&mut self, level: &crate::services::neural_audio::vc::engine::governor::QualityLevel) -> bool {
        self.inner.apply_quality(level)
    }

    fn reset_stream(&mut self) {
        self.inner.reset_stream();
    }

    fn fixed_block(&self) -> Option<usize> {
        self.inner.fixed_block()
    }

    fn configure_stream(&mut self, block: usize, want: usize, extra: usize) -> Result<usize, String> {
        self.inner.configure_stream(block, want, extra)
    }

    fn stream_report(&self) -> Option<crate::services::neural_audio::vc::traits::StreamReport> {
        self.inner.stream_report()
    }
}

/// The logged pitch path on the 10 ms grid of the harness's 16 kHz stream (0 where no frame was
/// decoded).
fn pitch_path_10ms(log: &std::collections::BTreeMap<i64, f32>, frames_10ms: usize, hop: usize, rate: u32) -> Vec<f32> {
    (0..frames_10ms)
        .map(|j| {
            let k = ((j as f64 * 0.01 * f64::from(rate)) / hop as f64).floor() as i64;
            log.get(&k).copied().unwrap_or(0.0)
        })
        .collect()
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// T10 singing (P5-A). A real DDSP-SVC voice (the default template, the real vocoder, ContentVec,
/// the pitch bank in the singing range) sings an E2 → D#6 sweep with vibrato through the stream
/// harness (48 kHz in, 44.1 kHz out, the voice changer's 20 ms crossfade) at shift 0 (E2 → D#6
/// out) and +12 (E3 → D#7 out). Scored, against the intended pitch (the input's × the transpose):
/// - **pitch path**: the F0 the vocoder received for every decoded frame (extraction, upstream's
///   post-processing, the transpose and the source-Nyquist clamp; independent of the voice's
///   timbre);
/// - **output**: the converted audio's pitch, by [`pitch_track`] over the whole range (separately above
///   2 kHz, the D#7 end), and by the benchmark's offline RMVPE (`F0Analyzer`) below 1 kHz, where it
///   reads this output reliably;
/// - the **offline ceiling**: the same voice converting the whole sweep in one call, scored the
///   same way, and the stream against it (F0 agreement, MCD).
///
/// Cost: per-stage compute at the stream's shape (direct ticks) and the harness's per-block
/// compute. EPs from `SIGIL_TEST_DDSP_EPS` (default `cuda,dml`), pitch methods from
/// `SIGIL_TEST_DDSP_PITCH` (default `rmvpe`), presets from `SIGIL_TEST_DDSP_PRESETS` (default all), WAVs
/// and the pitch tracks (CSV) into the folder `SIGIL_TEST_DDSP_WAV` when set.
/// `SIGIL_TEST_DDSP_T10=1` asserts T10 (gross pitch error ≤ 2 % on the pitch path and on the
/// output of every stream).
///
/// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache\vc-assets
/// SIGIL_TEST_DDSP_VOICE=%LOCALAPPDATA%\sigil-dev-cache\ddsp-svc\synthetic-voice
/// SIGIL_TEST_RMVPE=...\rmvpe.onnx SIGIL_APP_DATA_DIR=<app data with the NVIDIA runtime pack>
/// cargo test --release real_model_singing -- --ignored --nocapture`. A release test executable in
/// `target/release/deps` has no `onnxruntime_providers_cuda.dll` beside it, so its CUDA requests
/// fall back to DirectML (the outcomes column says so): for CUDA figures, copy it next to
/// `Sigil.exe`'s provider DLLs and run it there.
#[test]
#[ignore = "needs SIGIL_TEST_MODELS_DIR (DDSP-SVC template, vocoder, ContentVec) and a voice"]
fn real_model_singing_range_and_cost() {
    use crate::services::bench::quality::{f0_errors, mcd_db, F0Analyzer};
    use crate::services::bench::signals::resample;
    use crate::services::neural_audio::vc::engine::options::F0Range;
    let Some(a) = real_assets() else {
        eprintln!("SIGIL_TEST_MODELS_DIR / the voice are not set; skipped");
        return;
    };
    crate::services::neural_audio::ort_session::preload_test_cuda_runtime_quiet();
    let layout = DdspLayout::embedded().unwrap();
    let slot = temp_dir("real");
    let t0 = std::time::Instant::now();
    let voice = import::import_voice(&a.voice, &layout).expect("the voice imports");
    import::write_slot(&slot, &voice, &layout, "Real", serde_json::json!({}), std::time::Instant::now()).unwrap();
    eprintln!("import: {} ms ({} tensors, {} MB)", t0.elapsed().as_millis(), voice.tensors, voice.image.len() >> 20);
    let wav_dir = std::env::var_os("SIGIL_TEST_DDSP_WAV").map(PathBuf::from);
    let strict = std::env::var_os("SIGIL_TEST_DDSP_T10").is_some();
    let eps = std::env::var("SIGIL_TEST_DDSP_EPS").unwrap_or_else(|_| "cuda,dml".into());
    let methods = std::env::var("SIGIL_TEST_DDSP_PITCH").unwrap_or_else(|_| "rmvpe".into());
    let presets: Vec<String> = std::env::var("SIGIL_TEST_DDSP_PRESETS")
        .map(|v| v.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect())
        .unwrap_or_default();
    let (lead, seconds) = (0.5f32, 8.0f32);
    let (x48, _) = singing_sweep(48_000, seconds, lead, 82.41, 1244.5);
    let (x16, in_f0) = singing_sweep(16_000, seconds, lead, 82.41, 1244.5);
    // Scored from 100 ms after the onset (a note's first frames are the extractors' warm-up) to the
    // end of the sweep.
    let span = ((lead + 0.1) * 100.0) as usize..in_f0.len();
    // The benchmark's offline RMVPE reads this output an octave low above about 1 kHz (where
    // librosa's pYIN and YIN agree with the intended pitch): its scores stop there.
    let rmvpe_top = 1_000.0f32;
    let hop = layout.hop();
    let rate = layout.arch.sampling_rate;
    let cases: [(&str, f32); 5] = [
        ("sigil_singing", 0.0),
        ("sigil_singing", 12.0),
        ("sigil_singing_fast", 0.0),
        ("sigil_singing_fast", 12.0),
        ("upstream_gui", 12.0),
    ];
    let mut table = Vec::new();
    for ep in eps.split(',').map(str::trim).filter(|e| !e.is_empty()) {
        let mut analyzer = F0Analyzer::new("rmvpe", a.rmvpe.as_deref(), ep, F0Range::Singing);
        eprintln!("output F0 analysis: {} {:?}", analyzer.method, analyzer.notes);
        for method in methods.split(',').map(str::trim).filter(|m| !m.is_empty()) {
            for (preset, semitones) in cases {
                if !presets.is_empty() && !presets.iter().any(|p| p == preset) {
                    continue;
                }
                let label = format!("{ep} {method} {preset} {semitones:+}");
                // 1. The stream through the harness, logging the pitch path.
                let t_load = std::time::Instant::now();
                let inner = real_model(&a, &layout, &slot, ep, method, preset).unwrap_or_else(|e| panic!("{label}: {e}"));
                let block16 = inner.fixed_block().unwrap();
                let log = std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeMap::new()));
                let probe = F0Probe { inner, log: log.clone() };
                let device_block = block16 * 3;
                let mut h = VcStreamHarness::new(Box::new(probe), 48_000, 44_100, 1, 1, 0, 882, device_block).unwrap();
                let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
                let report = h.model_report().unwrap();
                h.set_controls(semitones, 0.0, 0.33, 0);
                h.set_voice_controls(&VoiceControls { silence_db: -45.0, ..VoiceControls::default() });
                log.lock().unwrap().clear();
                let mut stream = Vec::with_capacity(x48.len());
                let mut times = Vec::new();
                for chunk in x48.chunks_exact(device_block) {
                    let t = std::time::Instant::now();
                    stream.extend(h.push_and_convert(chunk).unwrap());
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                }
                // Flush the tail with silence so the last notes come out.
                for _ in 0..(1.2 * 48_000.0 / device_block as f64).ceil() as usize {
                    stream.extend(h.push_and_convert(&vec![0.0; device_block]).unwrap());
                }
                drop(h);
                let mut sorted: Vec<f64> = times.iter().skip(3).copied().collect();
                sorted.sort_by(f64::total_cmp);
                let path = pitch_path_10ms(&log.lock().unwrap(), in_f0.len() + 400, hop, rate);

                // 2. Per-stage cost at the stream's shape, then the offline ceiling (a second
                //    instance converting the whole sweep in one call).
                let mut m = real_model(&a, &layout, &slot, ep, method, preset).unwrap();
                let want = default_want(block16, rate);
                let ctx_len = m.configure(block16, want, 0).unwrap();
                let shape = m.shape().unwrap();
                let steps = m.steps();
                let mut stages: Vec<model::TickStats> = Vec::new();
                let mut pos = ((lead + 1.0) * 16_000.0) as usize;
                while pos + ctx_len <= x16.len() && stages.len() < 24 {
                    let ctx = ConversionCtx {
                        mono_16k: &x16[pos..pos + ctx_len],
                        new_frames: want,
                        pitch_shift_semitones: semitones,
                        index_rate: 0.0,
                        protect: 0.33,
                        speaker_id: 0,
                        stream_pos_16k: pos as u64,
                        reference: None,
                        singing: Default::default(),
                    };
                    m.convert(&ctx).unwrap();
                    stages.push(m.tick_stats());
                    pos += block16;
                }
                let median = |f: fn(&model::TickStats) -> f64| -> f64 {
                    let mut v: Vec<f64> = stages.iter().skip(2).map(f).collect();
                    v.sort_by(f64::total_cmp);
                    v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
                };
                let cost = [
                    median(|s| s.units_ms),
                    median(|s| s.f0_ms),
                    median(|s| s.ddsp_ms),
                    median(|s| s.flow_ms),
                    median(|s| s.vocoder_ms),
                    median(|s| s.total_ms),
                ];
                m.reset_stream();
                let n = x16.len() / 160 * 160;
                let ctx_len = m.configure(n, n, 0).unwrap();
                let mut padded = vec![0.0f32; ctx_len - n];
                padded.extend_from_slice(&x16[..n]);
                let t_ceiling = std::time::Instant::now();
                let ceiling = m
                    .convert(&ConversionCtx {
                        mono_16k: &padded,
                        new_frames: n,
                        pitch_shift_semitones: semitones,
                        index_rate: 0.0,
                        protect: 0.33,
                        speaker_id: 0,
                        stream_pos_16k: 0,
                        reference: None,
                        singing: Default::default(),
                    })
                    .unwrap();
                let ceiling_ms = t_ceiling.elapsed().as_secs_f64() * 1e3;
                // The ceiling's pitch path: its decoded frames start where the input does.
                let (first, f0) = m.last_decoded_f0().map(|(f, v)| (f, v.to_vec())).unwrap();
                let c_path: Vec<f32> = (0..in_f0.len())
                    .map(|j| {
                        let k = ((ctx_len - n) as f64 / 16_000.0 + j as f64 * 0.01) * f64::from(rate) / hop as f64;
                        f0.get((k.floor() as i64 - first).max(0) as usize).copied().unwrap_or(0.0)
                    })
                    .collect();
                drop(m);

                // 3. Scores.
                let all_hz = 0.0f32..f32::INFINITY;
                let s_path = pitch_score(&in_f0, &path, semitones, span.clone(), all_hz.clone(), 0..400);
                let c_paths = pitch_score(&in_f0, &c_path, semitones, span.clone(), all_hz.clone(), 0..3);
                let per_10ms = (rate / 100) as usize;
                let s_track = pitch_track(&stream, rate as f32, per_10ms, 70.0, 2_800.0);
                let c_track = pitch_track(&ceiling, rate as f32, per_10ms, 70.0, 2_800.0);
                let s_out = pitch_score(&in_f0, &s_track, semitones, span.clone(), all_hz.clone(), 0..60);
                let c_out = pitch_score(&in_f0, &c_track, semitones, span.clone(), all_hz.clone(), 0..4);
                // The benchmark's analyzer (offline RMVPE) where it is reliable on this output.
                let s16 = resample(&stream, rate, 16_000).unwrap();
                let c16 = resample(&ceiling, rate, 16_000).unwrap();
                let s_rmvpe = analyzer.contour(&s16).unwrap();
                let c_rmvpe = analyzer.contour(&c16).unwrap();
                let s_low = pitch_score(&in_f0, &s_rmvpe, semitones, span.clone(), 0.0..rmvpe_top, 0..60);
                let c_low = pitch_score(&in_f0, &c_rmvpe, semitones, span.clone(), 0.0..rmvpe_top, 0..4);
                // The top of the range (D#7 at +12): the pYIN-style track alone.
                let top = 2_000.0f32..f32::INFINITY;
                let s_top = pitch_score(&in_f0, &s_track, semitones, span.clone(), top.clone(), s_out.lag..s_out.lag + 1);
                // Stream against ceiling: the pYIN-style tracks at the lag between them, then the audio at
                // the sample offset that correlates best around it.
                let shift = s_out.lag.saturating_sub(c_out.lag);
                let agree = f0_errors(&c_track, s_track.get(shift..).unwrap_or_default()).unwrap_or_default();
                let coarse = (shift * per_10ms) as i64;
                let at = ((lead + 2.0) * rate as f32) as usize;
                let off = best_offset(&ceiling, &stream, at, rate as usize / 2, coarse - per_10ms as i64, coarse + per_10ms as i64);
                let aligned = stream.get(off.max(0) as usize..).unwrap_or_default();
                let mcd = mcd_db(&ceiling, aligned, rate).unwrap_or(f64::NAN);
                let line = format!(
                    "{label}: PATH GPE {:.2} % ({:.1} cents, {} frames) / ceiling {:.2} % ({:.1} cents) | OUT (pYIN-style) GPE {:.2} % ({:.1} cents, voiced {:.1} % of {}) / ceiling {:.2} % ({:.1} cents, voiced {:.1} %) | ≥ 2 kHz GPE {:.2} % (voiced {:.1} % of {}) | OUT < {rmvpe_top:.0} Hz (RMVPE) GPE {:.2} % (voiced {:.1} % of {}) / ceiling {:.2} % | stream vs ceiling: F0 {:.1} cents, GPE {:.2} %, V/UV {:.2} %, MCD {:.2} dB | delay {} ms | block p50 {:.1} / p95 {:.1} / max {:.1} ms (RTF p95 {:.3}) | stages ms: units {:.2}, f0 {:.2}, ddsp {:.2}, flow {:.2} ({steps} steps), vocoder {:.2}, tick {:.2} | T {} R {} context {} ms | load {load_ms:.0} ms, ceiling {ceiling_ms:.0} ms | {}",
                    100.0 * s_path.gpe,
                    s_path.cents,
                    s_path.both_voiced,
                    100.0 * c_paths.gpe,
                    c_paths.cents,
                    100.0 * s_out.gpe,
                    s_out.cents,
                    100.0 * s_out.recall(),
                    s_out.ref_voiced,
                    100.0 * c_out.gpe,
                    c_out.cents,
                    100.0 * c_out.recall(),
                    100.0 * s_top.gpe,
                    100.0 * s_top.recall(),
                    s_top.ref_voiced,
                    100.0 * s_low.gpe,
                    100.0 * s_low.recall(),
                    s_low.ref_voiced,
                    100.0 * c_low.gpe,
                    agree.rmse_cents,
                    100.0 * agree.gross_pitch_error,
                    100.0 * agree.vuv_error,
                    mcd,
                    s_out.lag * 10,
                    pct(&sorted, 0.5),
                    pct(&sorted, 0.95),
                    sorted[sorted.len() - 1],
                    pct(&sorted, 0.95) / (block16 as f64 / 16.0),
                    cost[0],
                    cost[1],
                    cost[2],
                    cost[3],
                    cost[4],
                    cost[5],
                    shape.frames,
                    shape.reflow,
                    shape.context_16k / 16,
                    report.outcomes.iter().map(|(c, o)| format!("{c}={}", o.active)).collect::<Vec<_>>().join(" "),
                );
                eprintln!("{line}");
                table.push(line);
                assert!(s_path.both_voiced > 0 && s_path.gpe.is_finite() && s_out.gpe.is_finite());
                if strict {
                    assert!(s_path.gpe <= 0.02, "T10: {label}: pitch-path gross pitch error {:.2} %", 100.0 * s_path.gpe);
                    assert!(s_out.gpe <= 0.02, "T10: {label}: output gross pitch error {:.2} %", 100.0 * s_out.gpe);
                }
                if let Some(dir) = &wav_dir {
                    std::fs::create_dir_all(dir).unwrap();
                    let tag = format!("{ep}-{method}-{preset}-{semitones:+}");
                    write_wav(&dir.join(format!("{tag}-stream.wav")), &stream, rate);
                    write_wav(&dir.join(format!("{tag}-ceiling.wav")), &ceiling, rate);
                    // The tracks on the input's 10 ms grid (each at its scored lag), for analysis.
                    let k = 2f32.powf(semitones / 12.0);
                    let at = |v: &[f32], i: usize, lag: usize| v.get(i + lag).copied().unwrap_or(0.0);
                    let mut csv = String::from("frame,intended,path,ceiling_path,out_track,ceiling_track,out_rmvpe,ceiling_rmvpe\n");
                    for (i, &f) in in_f0.iter().enumerate() {
                        csv.push_str(&format!(
                            "{i},{},{},{},{},{},{},{}\n",
                            f * k,
                            at(&path, i, s_path.lag),
                            at(&c_path, i, c_paths.lag),
                            at(&s_track, i, s_out.lag),
                            at(&c_track, i, c_out.lag),
                            at(&s_rmvpe, i, s_low.lag),
                            at(&c_rmvpe, i, c_low.lag)
                        ));
                    }
                    std::fs::write(dir.join(format!("{tag}-tracks.csv")), csv).unwrap();
                }
            }
        }
    }
    if let Some(dir) = &wav_dir {
        write_wav(&dir.join("input.wav"), &x16, 16_000);
    }
    eprintln!("\n---- summary ----\n{}", table.join("\n"));
    let _ = std::fs::remove_dir_all(slot);
}
