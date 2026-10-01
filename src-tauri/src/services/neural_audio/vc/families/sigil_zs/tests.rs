//! Sigil-ZS tests (`cargo test vc::families::sigil_zs`).
//!
//! The pipeline runs on Sigil's real ONNX Runtime (CPU) over stand-in graphs with the export's I/O
//! names and shapes (`test_onnx`): the reference is a function of the clip, the stand-in DiT step is
//! the rectified flow toward the reference's mean style (so a voice's output level identifies its
//! reference), and the stand-in HiFT returns each frame's mean mel. That checks the whole tick
//! (content → length regulator → Euler steps → partial decode → returned span), the stream harness's
//! exact length accounting at one fixed shape, the plan pool, the one-off path and the silence skip.
//!
//! The `#[ignore]`d tests need the real graphs: `SIGIL_TEST_MODELS_DIR` = the dev cache
//! (`vc-families/seed_vc/*.onnx`, `models/transcribe/samples/{jfk,ja-short}.wav`). The parity test
//! also needs `SIGIL_ZS_VECTORS` = the folder `scripts/vc-families/seed_vc/make_vectors.py` wrote
//! (the reference sampler's inputs and outputs, see that test).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::engine::{ModelConfig, SigilZsModel};
use super::graphs::{GraphName, GraphSet, REQUIRED};
use super::plans::{build_with_fallback, PhaseMode, PlanConfig, PlanReport, TickInputs, TickPlans};
use super::preset::{self, Geometry, Settings, HOP, MEL_BINS, N_HARM, SR_OUT, STYLE_DIM};
use super::reference;
use super::*;
use crate::services::neural_audio::gpu::GpuLevers;
use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT, INT64};
use crate::services::neural_audio::vc::engine::options::{EffectiveLevers, Lever, PrecisionMode};
use crate::services::neural_audio::vc::families::reference::{ReferenceClip, ReferenceEmbedding};
use crate::services::neural_audio::vc::stream::{HarnessContext, VcStreamHarness};
use crate::services::neural_audio::vc::traits::{ConversionCtx, VoiceControls, VoiceConversionModel};

// ---------------------------------------------------------------------------------------------
// Stand-in graphs
// ---------------------------------------------------------------------------------------------

fn n(name: &'static str) -> Dim {
    Dim::Named(name)
}

fn f(v: u64) -> Dim {
    Dim::Fixed(v)
}

/// `out = in + 0 · Σ means(unused...)`: keeps every declared input consumed.
fn absorb(g: &mut Graph, base: &str, unused: &[(&str, Vec<i64>)], out: &str) {
    let mut acc = String::new();
    for (i, (name, axes)) in unused.iter().enumerate() {
        let m = format!("absorb_m{i}_{out}");
        g.node("ReduceMean", &[name], &[&m], &[Attr::Ints("axes", axes.clone()), Attr::Int("keepdims", 0)]);
        if acc.is_empty() {
            acc = m;
        } else {
            let s = format!("absorb_s{i}_{out}");
            g.node("Add", &[&acc, &m], &[&s], &[]);
            acc = s;
        }
    }
    let z = format!("absorb_z_{out}");
    g.node("Mul", &[&acc, "zero"], &[&z], &[]);
    g.node("Add", &[base, &z], &[out], &[]);
}

/// A stand-in with the I/O of the exported graph `name` (see the module docs).
fn stand_in(name: GraphName) -> Vec<u8> {
    let mut g = Graph::new();
    g.init_f32("zero", &[1], &[0.0]).init_f32("one", &[1], &[1.0]);
    match name {
        GraphName::Xlsr => {
            g.input("wav16k", FLOAT, &[f(1), n("N")])
                .init_i64("ax1", &[1], &[1])
                .init_f32("w", &[1, 1, 400], &[1.0 / 400.0; 400])
                .init_i64("shape", &[3], &[1, 1, 1024])
                .node("Unsqueeze", &["wav16k", "ax1"], &["u"], &[])
                .node("Conv", &["u", "w"], &["c"], &[Attr::Ints("strides", vec![320])])
                .node("Transpose", &["c"], &["tr"], &[Attr::Ints("perm", vec![0, 2, 1])])
                .node("Expand", &["tr", "shape"], &["feats"], &[])
                .output("feats", FLOAT, &[f(1), n("T"), f(1024)]);
        }
        GraphName::Lr => {
            g.input("feats", FLOAT, &[f(1), n("T"), f(1024)])
                .input("n_frames", INT64, &[f(1)])
                .init_i64("i1", &[1], &[1])
                .init_i64("c384", &[1], &[384])
                .node("ReduceMean", &["feats"], &["m"], &[Attr::Ints("axes", vec![1, 2]), Attr::Int("keepdims", 1)])
                .node("Concat", &["i1", "n_frames", "c384"], &["shp"], &[Attr::Int("axis", 0)])
                .node("Expand", &["m", "shp"], &["cond"], &[])
                .output("cond", FLOAT, &[f(1), n("n"), f(384)]);
        }
        GraphName::DitStep | GraphName::DitStep1 => {
            g.input("x", FLOAT, &[f(1), f(80), n("L")])
                .input("prompt_x", FLOAT, &[f(1), f(80), n("L")])
                .input("cond", FLOAT, &[f(1), n("L"), f(384)])
                .input("style", FLOAT, &[f(1), f(192)])
                .input("t", FLOAT, &[f(1)])
                .input("dt", FLOAT, &[f(1)]);
            if name == GraphName::DitStep {
                g.input("cfg_rate", FLOAT, &[f(1)]);
            }
            g.input("prompt_len", INT64, &[f(1)])
                .node("Cast", &["prompt_len"], &["plf"], &[Attr::Int("to", 1)])
                .node("ReduceMean", &["style"], &["a0"], &[Attr::Ints("axes", vec![1]), Attr::Int("keepdims", 1)]);
            let mut unused = vec![("prompt_x", vec![1, 2]), ("cond", vec![1, 2]), ("plf", vec![0])];
            if name == GraphName::DitStep {
                unused.push(("cfg_rate", vec![0]));
            }
            absorb(&mut g, "a0", &unused, "a");
            // x_next = x + dt (a - x) / (1 - t): the rectified flow toward the style's mean.
            g.node("Sub", &["one", "t"], &["omt"], &[])
                .node("Sub", &["a", "x"], &["d"], &[])
                .node("Div", &["d", "omt"], &["v"], &[])
                .node("Mul", &["v", "dt"], &["s"], &[])
                .node("Add", &["x", "s"], &["x_next"], &[])
                .output("x_next", FLOAT, &[f(1), f(80), n("L")]);
        }
        GraphName::HiftF0 => {
            g.input("mel", FLOAT, &[f(1), f(80), n("M")])
                .init_f32("c200", &[1], &[200.0])
                .node("ReduceMean", &["mel"], &["m"], &[Attr::Ints("axes", vec![1]), Attr::Int("keepdims", 0)])
                .node("Mul", &["m", "zero"], &["m0"], &[])
                .node("Add", &["m0", "c200"], &["f0"], &[])
                .output("f0", FLOAT, &[f(1), n("M")]);
        }
        GraphName::HiftNsf | GraphName::Hift => {
            let frames = if name == GraphName::Hift { "M" } else { "W" };
            g.input("mel", FLOAT, &[f(1), f(80), n(frames)]);
            if name == GraphName::HiftNsf {
                g.input("f0", FLOAT, &[f(1), n("W")]);
            }
            g.input("harm_phase", FLOAT, &[f(1), f(9), f(1)])
                .input("sine_noise", FLOAT, &[f(1), f(9), n("S")])
                .init_i64("ax3", &[1], &[3])
                .init_i64("up", &[4], &[1, 1, 1, 256])
                .init_i64("flat", &[2], &[1, -1])
                .node("ReduceMean", &["mel"], &["mm"], &[Attr::Ints("axes", vec![1]), Attr::Int("keepdims", 1)])
                .node("Unsqueeze", &["mm", "ax3"], &["mu"], &[])
                .node("Expand", &["mu", "up"], &["me"], &[])
                .node("Reshape", &["me", "flat"], &["frames"], &[]);
            let mut unused = vec![("sine_noise", vec![1, 2]), ("harm_phase", vec![1, 2])];
            if name == GraphName::HiftNsf {
                unused.push(("f0", vec![1]));
            }
            absorb(&mut g, "frames", &unused, "audio");
            g.output("audio", FLOAT, &[f(1), n("S")]);
            if name == GraphName::Hift {
                g.init_f32("c200", &[1], &[200.0])
                    .node("ReduceMean", &["mel"], &["fm"], &[Attr::Ints("axes", vec![1]), Attr::Int("keepdims", 0)])
                    .node("Mul", &["fm", "zero"], &["fm0"], &[])
                    .node("Add", &["fm0", "c200"], &["f0"], &[])
                    .output("f0", FLOAT, &[f(1), n("M")]);
            }
        }
        GraphName::Campplus => {
            g.input("fbank", FLOAT, &[f(1), n("F"), f(80)])
                .init_i64("s11", &[2], &[1, 1])
                .init_i64("s192", &[2], &[1, 192])
                .init_f32("k", &[1], &[0.1])
                .node("Abs", &["fbank"], &["ab"], &[])
                .node("ReduceMean", &["ab"], &["m"], &[Attr::Ints("axes", vec![1, 2]), Attr::Int("keepdims", 1)])
                .node("Reshape", &["m", "s11"], &["r"], &[])
                .node("Mul", &["r", "k"], &["rk"], &[])
                .node("Expand", &["rk", "s192"], &["style"], &[])
                .output("style", FLOAT, &[f(1), f(192)]);
        }
        GraphName::Mel => {
            g.input("wav22k", FLOAT, &[f(1), n("S")])
                .init_i64("ax1", &[1], &[1])
                .init_i64("pads", &[6], &[0, 0, 384, 0, 0, 384])
                .init_f32("w", &[1, 1, 1024], &[1.0 / 1024.0; 1024])
                .init_i64("s80", &[3], &[1, 80, 1])
                .node("Unsqueeze", &["wav22k", "ax1"], &["u"], &[])
                .node("Pad", &["u", "pads"], &["p"], &[])
                .node("Conv", &["p", "w"], &["c"], &[Attr::Ints("strides", vec![256])])
                .node("Expand", &["c", "s80"], &["mel"], &[])
                .output("mel", FLOAT, &[f(1), f(80), n("M")]);
        }
    }
    g.model_bytes(17)
}

/// A temporary folder of stand-in graphs (removed on drop).
struct StandIns {
    root: PathBuf,
}

impl StandIns {
    fn new(tag: &str) -> StandIns {
        let root = std::env::temp_dir().join(format!(
            "sigil-zs-{tag}-{}-{}",
            std::process::id(),
            crate::services::neural_audio::vc::import::unix_ms()
        ));
        std::fs::create_dir_all(&root).unwrap();
        for g in REQUIRED {
            std::fs::write(root.join(g.file(false)), stand_in(g)).unwrap();
        }
        StandIns { root }
    }

    fn graphs(&self) -> GraphSet {
        GraphSet::from_dir(&self.root)
    }

    /// A WAV clip in the folder.
    fn clip(&self, name: &str, audio: &[f32], rate: u32) -> ReferenceClip {
        let p = self.root.join(name);
        crate::services::bench::signals::write_wav(&p, audio, rate, 1).unwrap();
        ReferenceClip::from_wav(&p).unwrap()
    }
}

impl Drop for StandIns {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn tone(len: usize, rate: u32, hz: f32, amp: f32) -> Vec<f32> {
    (0..len).map(|i| amp * (std::f32::consts::TAU * hz * i as f32 / rate as f32).sin()).collect()
}

fn noise(len: usize, amp: f32) -> Vec<f32> {
    let mut s = 0x1234_5678u64;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            amp * ((s >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0)
        })
        .collect()
}

fn cpu_opts() -> VcSessionOptions {
    VcSessionOptions { intra_threads: Some(1), ..Default::default() }
}

/// The voice changer's resolved levers for a GPU rung (the algorithmic ones on; the GPU ones do
/// nothing on the CPU plans of these tests).
fn levers(partial: bool, anchored: bool, phase: bool) -> EffectiveLevers {
    let mut l = EffectiveLevers::default();
    l.levers.set(Lever::PartialDecode, partial);
    l.levers.set(Lever::AnchoredNoise, anchored);
    l.levers.set(Lever::NsfPhase0, phase);
    l.levers.set(Lever::F0Cache, true);
    l
}

fn model(dir: &StandIns, clip: &ReferenceClip, options: SigilZsOptions, levers: EffectiveLevers, silence_skip: bool) -> (SigilZsModel, Arc<ReferenceEmbedding>) {
    let settings = Settings::new(&options, PrecisionMode::Auto);
    let graphs = dir.graphs();
    let e = reference::embedding(&graphs, clip, settings.preset.prompt_s, None, &cpu_opts()).unwrap();
    let prepared = reference::prepare(&e).unwrap();
    let m = SigilZsModel::new(ModelConfig {
        graphs,
        ep: "cpu".into(),
        session: cpu_opts(),
        levers,
        settings,
        reference: e.clone(),
        prepared,
        noise_seed: 99,
        silence_skip,
        warnings: Vec::new(),
    });
    (m, e)
}

/// The level a stand-in voice converts to: the mean of its reference style.
fn level_of(e: &ReferenceEmbedding) -> f32 {
    let s = &e.tensor(reference::STYLE).unwrap().data;
    s.iter().sum::<f32>() / s.len() as f32
}

fn ctx<'a>(mono: &'a [f32], new_frames: usize, pos: u64) -> ConversionCtx<'a> {
    ConversionCtx {
        mono_16k: mono,
        new_frames,
        pitch_shift_semitones: 0.0,
        index_rate: 0.0,
        protect: 0.33,
        speaker_id: 0,
        stream_pos_16k: pos,
        reference: None,
        singing: Default::default(),
    }
}

// ---------------------------------------------------------------------------------------------
// Pipeline on stand-ins
// ---------------------------------------------------------------------------------------------

#[test]
fn the_reference_has_the_dit_shapes() {
    let dir = StandIns::new("ref");
    let clip = dir.clip("ref.wav", &tone(48_000, 16_000, 180.0, 0.3), 16_000);
    let graphs = dir.graphs();
    let wav = reference::prompt_audio(&clip, 1.5).unwrap();
    assert_eq!(wav.len(), 33_075, "cut to the prompt length at 22.05 kHz");
    let tensors = reference::compute(&graphs, &wav, &cpu_opts()).unwrap();
    let p = preset::mel_frames(wav.len());
    assert_eq!(p, 129);
    let shapes: Vec<(&str, Vec<usize>)> = tensors.iter().map(|t| (t.name.as_str(), t.shape.clone())).collect();
    assert_eq!(
        shapes,
        vec![(reference::STYLE, vec![1, STYLE_DIM]), (reference::PROMPT_MEL, vec![1, MEL_BINS, p]), (reference::PROMPT_COND, vec![1, p, 384])]
    );
    // Too short a clip is refused by name.
    let short = dir.clip("short.wav", &tone(4_000, 16_000, 180.0, 0.3), 16_000);
    assert!(reference::prompt_audio(&short, 3.0).unwrap_err().contains("at least"));
    // The embedding of an ad-hoc clip carries its source.
    let e = reference::embedding(&graphs, &clip, 3.0, None, &cpu_opts()).unwrap();
    assert_eq!((e.backend.as_str(), e.clip_sha256.as_str()), ("sigil-zs-2.p3000", clip.sha256.as_str()));
    assert_eq!(reference::prepare(&e).unwrap().frames, 258);
}

/// A library voice's reference is computed once and then read from the voice library's cache
/// (`cache/sigil-zs-2.p3000.bin`); a cache that parses but holds unusable tensors is recomputed and
/// replaced.
#[test]
fn library_voices_cache_their_reference() {
    use crate::services::voice_library as vl;
    let dir = StandIns::new("cache");
    let app = dir.root.join("appdata");
    let voice = vl::voices_dir(&app).join("alice");
    std::fs::create_dir_all(&voice).unwrap();
    crate::services::bench::signals::write_wav(&voice.join(vl::CLIP_FILE), &noise(64_000, 0.25), 16_000, 1).unwrap();
    std::fs::write(voice.join(vl::META_FILE), br#"{"name": "Alice", "source": "imported"}"#).unwrap();
    let clip = vl::reference_clip(&app, "alice").unwrap();
    let graphs = dir.graphs();
    let backend = reference::backend(3.0);
    let first = reference::embedding(&graphs, &clip, 3.0, Some(&app), &cpu_opts()).unwrap();
    assert_eq!((first.voice_id.as_str(), first.backend.as_str()), ("alice", backend.as_str()));
    assert!(vl::read_cache(&app, "alice", &backend).is_some(), "stored in the library");
    // Read back from the cache: no graph runs (XLS-R is gone).
    let xlsr = dir.root.join(GraphName::Xlsr.file(false));
    let bytes = std::fs::read(&xlsr).unwrap();
    std::fs::remove_file(&xlsr).unwrap();
    let cached = reference::embedding(&graphs, &clip, 3.0, Some(&app), &cpu_opts()).unwrap();
    assert_eq!(*cached, *first);
    // Another prompt length is another backend: computed (and fails without the graph).
    assert!(reference::embedding(&graphs, &clip, 1.5, Some(&app), &cpu_opts()).is_err());
    std::fs::write(&xlsr, bytes).unwrap();
    // A cache that parses but holds the wrong tensors is recomputed and replaced.
    let bad = ReferenceEmbedding { tensors: vec![first.tensors[0].clone()], ..(*first).clone() };
    vl::write_cache(&app, "alice", &backend, &bad.to_bytes().unwrap()).unwrap();
    let fixed = reference::embedding(&graphs, &clip, 3.0, Some(&app), &cpu_opts()).unwrap();
    assert_eq!(*fixed, *first);
    let stored = ReferenceEmbedding::from_bytes(&vl::read_cache(&app, "alice", &backend).unwrap()).unwrap();
    assert_eq!(stored, *first, "the cache holds the recomputed reference again");
}

#[test]
fn a_tick_runs_the_whole_pipeline() {
    let dir = StandIns::new("tick");
    let clip = dir.clip("ref.wav", &noise(48_000, 0.2), 16_000);
    let graphs = dir.graphs();
    let e = reference::embedding(&graphs, &clip, 3.0, None, &cpu_opts()).unwrap();
    let r = reference::prepare(&e).unwrap();
    let level = level_of(&e);
    assert!(level > 0.01, "{level}");
    for (split, cfg) in [(true, true), (false, false)] {
        let g = Geometry::stream(&preset::SIGIL_LOW, 1_600, 2_082, r.frames, split).unwrap();
        let opts = cpu_opts();
        let plans_cfg = PlanConfig {
            graphs: &graphs,
            ep: "cpu",
            opts: &opts,
            levers: GpuLevers::default(),
            geometry: g,
            cfg,
            split,
            fp16_xlsr: false,
            fp16_dit: false,
            fp16_vocoder: false,
            component: "vc.zs.test",
        };
        let mut plans = TickPlans::build(&plans_cfg).unwrap();
        assert_eq!(plans.report.outcomes.len(), if split { 5 } else { 4 });
        assert!(plans.report.outcomes.iter().all(|(c, o)| c.starts_with("vc.zs.test.") && o.active == "cpu"));
        assert!(plans.report.on_cpu.is_empty() && plans.report.cpu_refusal().is_none(), "a CPU request is no fallback");
        assert!(!plans.warmed());
        let window = tone(g.n16, 16_000, 220.0, 0.2);
        let z: Vec<f32> = (0..MEL_BINS * g.n_mel).map(|i| ((i * 37 % 101) as f32 / 50.0) - 1.0).collect();
        let sine = vec![0.5f32; N_HARM * g.dec_samples()];
        let t_span = super::sampler::time_grid(6);
        let mut out = vec![9.0f32; g.out_samples];
        let inputs = TickInputs { owner: 1, reference: &r, t_span: &t_span, cfg_rate: 0.7, z: &z, sine: &sine, phase: PhaseMode::Start([0.0; N_HARM]) };
        plans.tick(&window, &inputs, &mut out).unwrap();
        assert!(plans.warmed());
        // Every generated mel frame flowed to the reference's level, so does the returned audio.
        assert!(plans.mel().iter().all(|&v| (v - level).abs() < 1e-4), "mel");
        assert!(out.iter().all(|&v| (v - level).abs() < 1e-4), "split {split}: {} vs {level}", out[0]);
        assert!(plans.decoded_f0().iter().all(|&f| f == 200.0));
        assert_eq!(plans.decoded_f0().len(), g.dec_frames());
        // The anchor mode needs partial decode.
        let anchored = TickInputs { phase: PhaseMode::Anchor { n: 100, target: [0.25; N_HARM] }, ..inputs };
        let r2 = plans.tick(&window, &anchored, &mut out);
        assert_eq!(r2.is_ok(), split);
        // A wrong window is refused before any graph runs.
        assert!(plans.tick(&window[1..], &inputs, &mut out).is_err());
    }
}

/// Through the production harness at several device blocks: one fixed shape, exactly the owed
/// frames every call, the voice's level once primed.
#[test]
fn streaming_is_exact_and_fixed_shape() {
    let dir = StandIns::new("stream");
    let clip = dir.clip("ref.wav", &noise(64_000, 0.25), 16_000);
    for (process_frames, rate, partial) in [(4_800usize, 48_000u32, true), (1_920, 48_000, true), (1_764, 44_100, false)] {
        let (m, e) = model(&dir, &clip, SigilZsOptions::default(), levers(partial, true, true), false);
        let level = level_of(&e);
        let mut h = VcStreamHarness::with_context(
            Box::new(m),
            rate,
            rate,
            1,
            2,
            1_024,
            441,
            process_frames,
            HarnessContext { reference: Some(e.clone()), singing: Default::default() },
        )
        .unwrap();
        let (block, context) = h.fixed_shape().expect("fixed-block mode");
        assert_eq!(block, 1_600);
        assert_eq!(context % 320, 0);
        let input = tone(process_frames * 40, rate, 200.0, 0.3);
        let mut out = Vec::new();
        for chunk in input.chunks(process_frames) {
            let o = h.push_and_convert(chunk).unwrap();
            assert_eq!(o.len(), chunk.len() * 2, "exactly the owed frames, stereo");
            out.extend(o);
        }
        let tail = &out[out.len() / 2..];
        let ok = tail.iter().filter(|&&v| (v - level).abs() < 0.02 * level.max(0.05)).count();
        assert!(ok * 10 >= tail.len() * 9, "{process_frames}@{rate}: {ok}/{} near {level}", tail.len());
    }
}

/// Voices of one shape share one set of plans and keep their own reference: interleaved calls
/// convert each to its own level (the voice-specific inputs are rewritten on every owner change).
#[test]
fn voices_share_plans_and_keep_their_references() {
    let dir = StandIns::new("pool");
    let a = dir.clip("a.wav", &noise(64_000, 0.25), 16_000);
    let b = dir.clip("b.wav", &tone(64_000, 16_000, 300.0, 0.05), 16_000);
    let (mut ma, ea) = model(&dir, &a, SigilZsOptions::default(), levers(true, true, true), false);
    let (mut mb, eb) = model(&dir, &b, SigilZsOptions::default(), levers(true, true, true), false);
    let (la, lb) = (level_of(&ea), level_of(&eb));
    assert!((la - lb).abs() > 0.01, "{la} {lb}");
    let ctx_a = ma.configure_stream(1_600, 2_082, 0).unwrap();
    let ctx_b = mb.configure_stream(1_600, 2_082, 0).unwrap();
    assert_eq!(ctx_a, ctx_b);
    let g = ma.geometry().unwrap();
    let window = tone(g.n16, 16_000, 200.0, 0.2);
    for k in 0..3u64 {
        let ca = ConversionCtx { reference: Some(&ea), ..ctx(&window, g.want_16k, k * 1_600) };
        let cb = ConversionCtx { reference: Some(&eb), ..ctx(&window, g.want_16k, k * 1_600) };
        let oa = ma.convert(&ca).unwrap();
        let ob = mb.convert(&cb).unwrap();
        assert_eq!((oa.len(), ob.len()), (g.out_samples, g.out_samples));
        assert!(oa.iter().all(|&v| (v - la).abs() < 1e-4), "voice a tick {k}");
        assert!(ob.iter().all(|&v| (v - lb).abs() < 1e-4), "voice b tick {k}");
    }
    // A hot reference swap with the same prompt length takes effect on the next tick.
    let cb = ConversionCtx { reference: Some(&eb), ..ctx(&window, g.want_16k, 9 * 1_600) };
    let swapped = ma.convert(&cb).unwrap();
    assert!(swapped.iter().all(|&v| (v - lb).abs() < 1e-4));
}

/// The voice that built a set of shared plans claimed their VRAM through its load
/// (`LoadedVoice::load_cancellable` measures what the load added; a voice that reuses them measures about
/// 0 MB). When it goes while others still use them, the plans claim the footprint themselves, so
/// the resource governor never counts live sessions as free; the claim goes with the last voice.
#[test]
fn shared_sessions_stay_claimed_until_the_last_voice_goes() {
    use crate::services::resources::{self, Subsystem};
    let dir = StandIns::new("claim");
    let clip = dir.clip("ref.wav", &noise(64_000, 0.25), 16_000);
    let voice = || model(&dir, &clip, SigilZsOptions::default(), levers(true, true, true), false).0;
    let claimed = |id: u64| resources::snapshot().claims.iter().find(|c| c.id == id).map(|c| (c.subsystem, c.mb));
    let (mut a, mut b) = (voice(), voice());
    a.configure_stream(1_600, 2_082, 0).unwrap();
    b.configure_stream(1_600, 2_082, 0).unwrap();
    assert!(a.built_plans() && !b.built_plans(), "b shares the sessions a built");
    // What a measured on the GPU around the build and the warm-up (the CPU has no VRAM).
    a.note_plans_footprint(1_234);
    assert_eq!(b.plans_claim(), None, "a's load claims them while a lives");
    // The reference switch: the outgoing voice (a) is dropped while the incoming one (b) plays.
    drop(a);
    let (id, mb) = b.plans_claim().expect("the sessions claim their footprint once a is gone");
    assert_eq!(mb, 1_234);
    assert_eq!(claimed(id), Some((Subsystem::Vc, 1_234)));
    // A voice that joins later shares the sessions and their claim.
    let mut c = voice();
    c.configure_stream(1_600, 2_082, 0).unwrap();
    assert!(!c.built_plans());
    assert_eq!(c.plans_claim().map(|(i, _)| i), Some(id));
    drop(b);
    assert_eq!(claimed(id), Some((Subsystem::Vc, 1_234)), "c still uses them");
    drop(c);
    assert_eq!(claimed(id), None, "released with the sessions");
    // A voice whose own load claimed sessions nobody else uses releases them with nothing left over.
    let mut d = voice();
    d.configure_stream(1_600, 2_082, 0).unwrap();
    assert!(d.built_plans(), "the pool let go of the old sessions");
    d.note_plans_footprint(999);
    assert_eq!(d.plans_claim(), None);
    drop(d);
    assert!(!resources::snapshot().claims.iter().any(|c| c.mb == 999 && c.label.contains("Sigil-ZS")));
}

/// A voice that shares plans another voice already ticked on skips the warm-up ticks (they would
/// hold the plans' lock while that voice plays: a dropout before a reference switch); cold plans
/// are warmed, and a warm-up never waits on plans another voice is ticking on.
#[test]
fn shared_warm_plans_are_not_warmed_again() {
    let dir = StandIns::new("warm");
    let clip = dir.clip("ref.wav", &noise(64_000, 0.25), 16_000);
    let voice = || model(&dir, &clip, SigilZsOptions::default(), levers(true, true, true), false).0;
    let (mut a, mut b) = (voice(), voice());
    a.configure_stream(1_600, 2_082, 0).unwrap();
    b.configure_stream(1_600, 2_082, 0).unwrap();
    let g = a.geometry().unwrap();
    let window = tone(g.n16, 16_000, 200.0, 0.2);
    assert_eq!(a.plan_ticks(), 0);
    // Cold plans: the first warm-up runs its two ticks (whichever voice it is).
    b.warmup_at(&ctx(&window, g.want_16k, 0)).unwrap();
    assert_eq!(a.plan_ticks(), 2);
    a.warmup_at(&ctx(&window, g.want_16k, 0)).unwrap();
    assert_eq!(a.plan_ticks(), 2, "warm already: no tick");
    // The stream still starts clean and converts.
    let out = a.convert(&ctx(&window, g.want_16k, 1_600)).unwrap();
    assert_eq!(out.len(), g.out_samples);
    assert_eq!(a.plan_ticks(), 3);
    // Another voice holds the plans' lock (mid-tick): the joining voice's warm-up does not wait.
    let mut c = voice();
    c.configure_stream(1_600, 2_082, 0).unwrap();
    let plans = c.plans_handle();
    let held = plans.lock().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let want = g.want_16k;
    let warm_window = window.clone();
    let worker = std::thread::spawn(move || {
        let ok = c.warmup_at(&ctx(&warm_window, want, 0)).is_ok();
        let _ = tx.send(ok);
        c
    });
    let ok = rx.recv_timeout(std::time::Duration::from_secs(30));
    drop(held);
    assert_eq!(ok, Ok(true), "the warm-up waited on busy plans");
    let c = worker.join().unwrap();
    assert_eq!(c.plan_ticks(), 3);
    // A one-off window is not the live shape: it runs (on the voice's own CPU plans).
    let mut d = voice();
    d.configure_stream(1_600, 2_082, 0).unwrap();
    d.warmup_at(&ctx(&window[..g.n16 - 320], g.want_16k, 0)).unwrap();
    assert_eq!(d.plan_ticks(), 3, "the stream's plans are untouched");
}

/// A stand-in EP ladder: CUDA → DirectML → CPU for a CUDA request, DirectML → CPU for a DirectML
/// one. CUDA runs when `cuda`; DirectML runs when `dml`, except pinned shapes it refuses
/// (`dml_refuses_pinned`: the HiFT graphs, ADR 0006 finding 5); the CPU takes everything.
fn ladder(requested: &str, cuda: bool, dml: bool, dml_refuses_pinned: bool, pinned: bool) -> &'static str {
    let rungs: &[&'static str] = match requested {
        "cuda" => &["cuda", "dml", "cpu"],
        "dml" => &["dml", "cpu"],
        _ => &["cpu"],
    };
    rungs
        .iter()
        .copied()
        .find(|r| match *r {
            "cuda" => cuda,
            "dml" => dml && !(pinned && dml_refuses_pinned),
            _ => true,
        })
        .unwrap()
}

/// `plans::build_plan`'s fallbacks. The review case: a CUDA request without the NVIDIA runtime
/// falls to DirectML, which refuses HiFT's pinned shapes; the pinned build lands on the CPU and is
/// rebuilt with dynamic shapes, which DirectML takes (before: HiFT stayed on the CPU, 93-127 ms per
/// call).
#[test]
fn pinned_shapes_no_gpu_took_are_rebuilt_dynamic() {
    let dims = [("M", 37i64)];
    // (requested EP, CUDA runs, DirectML runs, DirectML refuses pinned shapes, pin_on_dml)
    //   -> (pinned or dynamic per build, the EP the plan ends on)
    let cases = [
        (("cuda", false, true, true, false), (vec![true, false], "dml")),
        (("cuda", true, true, true, false), (vec![true], "cuda")),
        (("cuda", false, true, false, true), (vec![true], "dml")),
        (("cuda", false, false, true, false), (vec![true, false], "cpu")),
        (("dml", false, true, true, false), (vec![false], "dml")),
        (("dml", false, true, false, true), (vec![true], "dml")),
        (("dml", false, true, true, true), (vec![true, false], "dml")),
        (("cpu", false, false, false, true), (vec![true], "cpu")),
    ];
    for ((ep, cuda, dml, refuses, pin_on_dml), (want_builds, want_ep)) in cases {
        let mut builds = Vec::new();
        let got = build_with_fallback(
            "vc.zs.test.hift",
            ep,
            &dims,
            pin_on_dml,
            |d| {
                builds.push(!d.is_empty());
                Ok(ladder(ep, cuda, dml, refuses, !d.is_empty()))
            },
            |active| *active == "cpu",
        )
        .unwrap();
        assert_eq!((builds, got), (want_builds, want_ep), "{ep}: cuda {cuda}, dml {dml}, refuses pinned {refuses}, pin_on_dml {pin_on_dml}");
    }
    // A failed build is passed on, not retried.
    let mut calls = 0;
    let err = build_with_fallback::<&str>(
        "vc.zs.test.xlsr",
        "cuda",
        &dims,
        true,
        |_| {
            calls += 1;
            Err("no session".into())
        },
        |_| true,
    );
    assert_eq!((err, calls), (Err("no session".to_string()), 1));
}

/// A GPU voice whose heavy stages fell to the CPU through the EP ladder is refused (as an explicit
/// CPU request is); the light stages keep up there.
#[test]
fn heavy_stages_on_the_cpu_refuse_a_gpu_voice() {
    let mut r = PlanReport::default();
    assert_eq!(r.cpu_refusal(), None);
    r.on_cpu = vec![(GraphName::Lr, None), (GraphName::HiftF0, Some("CUDA: out of memory".into()))];
    assert_eq!(r.cpu_refusal(), None, "lr and hift_f0 take 1-2 ms on the CPU");
    r.on_cpu.push((GraphName::HiftNsf, Some("DirectML refused the shapes".into())));
    r.on_cpu.push((GraphName::Xlsr, Some("  ".into())));
    let why = r.cpu_refusal().unwrap();
    assert!(why.starts_with("Sigil-ZS: hift_nsf (DirectML refused the shapes), xlsr fell back to the CPU"), "{why}");
    assert!(!why.contains("hift_f0") && why.contains("real time"), "{why}");
    for g in [GraphName::DitStep, GraphName::DitStep1, GraphName::Hift] {
        let r = PlanReport { on_cpu: vec![(g, None)], ..Default::default() };
        assert!(r.cpu_refusal().unwrap().contains(g.stem()), "{g:?}");
    }
}

#[test]
fn one_off_conversions_return_exactly_the_new_frames() {
    let dir = StandIns::new("offline");
    let clip = dir.clip("ref.wav", &noise(48_000, 0.25), 16_000);
    let (mut m, e) = model(&dir, &clip, SigilZsOptions { preset: "sigil_lean".into(), ..Default::default() }, levers(true, true, true), false);
    let level = level_of(&e);
    let audio = tone(3 * 16_000, 16_000, 150.0, 0.2);
    for (len, new) in [(48_000usize, 48_000usize), (48_000, 20_000), (12_345, 12_345), (300, 300)] {
        let out = m.convert(&ctx(&audio[..len], new, 0)).unwrap();
        assert_eq!(out.len(), preset::out_len(new), "{len}/{new}");
        if len > 1_000 {
            let mid = &out[out.len() / 4..];
            assert!(mid.iter().all(|&v| (v - level).abs() < 1e-3), "{len}");
        }
    }
}

#[test]
fn silence_stops_inference_after_the_hangover() {
    let dir = StandIns::new("silence");
    let clip = dir.clip("ref.wav", &noise(48_000, 0.25), 16_000);
    let (mut m, e) = model(&dir, &clip, SigilZsOptions::default(), levers(true, true, true), true);
    m.set_voice_controls(&VoiceControls { silence_db: -40.0, ..Default::default() });
    m.configure_stream(1_600, 2_082, 0).unwrap();
    let g = m.geometry().unwrap();
    let quiet = vec![0.0f32; g.n16];
    let hangover = (super::engine::SILENCE_HANGOVER_MS / 100.0).ceil() as u64;
    for k in 0..=hangover + 2 {
        let out = m.convert(&ctx(&quiet, g.want_16k, k * 1_600)).unwrap();
        assert_eq!(out.len(), g.out_samples);
        let converted = out.iter().any(|&v| v != 0.0);
        assert_eq!(converted, k < hangover, "tick {k}: converted through the hangover only");
    }
    let loud = tone(g.n16, 16_000, 200.0, 0.3);
    let out = m.convert(&ctx(&loud, g.want_16k, 40 * 1_600)).unwrap();
    assert!(out.iter().all(|&v| (v - level_of(&e)).abs() < 1e-4), "speech converts again at once");
}

/// The governor's reduced levels run fewer Euler steps on the same plans (the analytic flow still
/// lands on the voice's level); steps that change nothing report `false`.
#[test]
fn the_governor_sheds_flow_steps() {
    use crate::services::neural_audio::vc::engine::governor::QualityLevel;
    let dir = StandIns::new("governor");
    let clip = dir.clip("ref.wav", &noise(48_000, 0.25), 16_000);
    let (mut m, e) = model(&dir, &clip, SigilZsOptions::default(), levers(true, true, true), false);
    m.configure_stream(1_600, 2_082, 0).unwrap();
    let g = m.geometry().unwrap();
    let window = tone(g.n16, 16_000, 200.0, 0.2);
    assert!(!m.apply_quality(&QualityLevel::at(1)), "retrieval off: nothing to do here");
    assert!(m.apply_quality(&QualityLevel::at(2)), "reduced context: fewer steps");
    assert!(!m.apply_quality(&QualityLevel::at(4)), "already reduced");
    let out = m.convert(&ctx(&window, g.want_16k, 1_600)).unwrap();
    assert!(out.iter().all(|&v| (v - level_of(&e)).abs() < 1e-4));
    assert!(m.apply_quality(&QualityLevel::FULL));
    let out = m.convert(&ctx(&window, g.want_16k, 2 * 1_600)).unwrap();
    assert!(out.iter().all(|&v| (v - level_of(&e)).abs() < 1e-4));
    // A voice already at the floor has nothing to shed.
    let (mut lean, _) = model(&dir, &clip, SigilZsOptions { steps: Some(3), ..Default::default() }, levers(true, true, true), false);
    assert!(!lean.apply_quality(&QualityLevel::at(2)));
}

/// A TensorRT-RTX request (Studio-TRT) runs the CUDA plans, noted; the DirectML high-priority queue
/// is kept; partial decode off turns the phase join off with it.
#[test]
fn levers_the_family_does_not_run_are_noted() {
    use crate::services::neural_audio::vc::engine::options::Accelerator;
    let dir = StandIns::new("levers");
    let clip = dir.clip("ref.wav", &noise(48_000, 0.25), 16_000);
    let mut trt = levers(true, true, true);
    trt.accelerator = Accelerator::TensorRtRtx;
    let (m, _) = model(&dir, &clip, SigilZsOptions::default(), trt, false);
    let l = m.effective_levers();
    assert_eq!(l.accelerator, Accelerator::None);
    assert!(l.notes.iter().any(|n| n.starts_with("tensorrt_rtx:")), "{:?}", l.notes);
    let mut hp = levers(false, true, true);
    hp.accelerator = Accelerator::DmlHighPriority;
    let (m, _) = model(&dir, &clip, SigilZsOptions::default(), hp, false);
    let l = m.effective_levers();
    assert_eq!(l.accelerator, Accelerator::DmlHighPriority);
    assert!(!l.levers.get(Lever::NsfPhase0) && !l.levers.get(Lever::F0Cache) && !l.levers.get(Lever::GpuKnn));
    assert!(l.notes.iter().any(|n| n.starts_with("nsf_phase0:")), "{:?}", l.notes);
}

/// The phase-continuous join: the second of two consecutive ticks anchors its NSF phase so the
/// excitation where they meet continues the first tick's.
#[test]
fn consecutive_ticks_continue_the_nsf_phase() {
    let dir = StandIns::new("phase");
    let clip = dir.clip("ref.wav", &noise(48_000, 0.25), 16_000);
    let (mut m, _) = model(&dir, &clip, SigilZsOptions::default(), levers(true, true, true), false);
    m.configure_stream(1_600, 2_082, 0).unwrap();
    let g = m.geometry().unwrap();
    let window = tone(g.n16, 16_000, 200.0, 0.2);
    m.convert(&ctx(&window, g.want_16k, 10 * 1_600)).unwrap();
    let (t0, f0_0, ph0) = m.last_phase().unwrap();
    m.convert(&ctx(&window, g.want_16k, 11 * 1_600)).unwrap();
    let (t1, f0_1, ph1) = m.last_phase().unwrap();
    assert_eq!((t0, t1), (10, 11));
    let (n_this, n_prev) = m.join_anchor_for_tests();
    let a = super::dsp::nsf_cycles_at(&f0_0, &ph0, n_prev);
    let b = super::dsp::nsf_cycles_at(&f0_1, &ph1, n_this);
    for h in 0..N_HARM {
        let d = (a[h] - b[h]).rem_euclid(1.0);
        assert!(d.min(1.0 - d) < 1e-4, "harmonic {h}: {} vs {}", a[h], b[h]);
    }
    // A gap (not the next tick) starts fresh from the voice's start phases.
    m.convert(&ctx(&window, g.want_16k, 20 * 1_600)).unwrap();
    let (t2, f0_2, ph2) = m.last_phase().unwrap();
    assert_eq!(t2, 20);
    let start = super::dsp::carry_phase(&f0_2, &[0.0; N_HARM], 0);
    assert_eq!(start.len(), N_HARM);
    assert_ne!(ph2, ph1);
}

#[test]
fn slots_need_the_graphs_and_valid_options() {
    // In a test process the model store has no Sigil-ZS entry installed.
    let m = super::super::family_manifest(FamilyKind::SigilZs, "Z", serde_json::json!({"preset": "sigil_lean"}));
    let err = check_slot(Path::new("C:/nowhere"), &m).unwrap_err();
    assert!(err.contains(graphs::ZS_GRAPHS), "{err}");
    let bad = super::super::family_manifest(FamilyKind::SigilZs, "Z", serde_json::json!({"steps": "many"}));
    assert!(check_slot(Path::new("C:/nowhere"), &bad).unwrap_err().contains("options"));
    // Normalization clamps and fills.
    let v = normalize_options(&serde_json::json!({"preset": "bogus", "steps": 99, "cfg_rate": 3.0, "prompt_s": 0.1})).unwrap();
    assert_eq!(v["preset"], "sigil_low");
    assert_eq!(v["steps"], 30);
    assert_eq!(v["cfg_rate"], 1.0);
    assert_eq!(v["prompt_s"], 0.5);
    assert!(available());
    let i = info();
    assert!(i.zero_shot && !i.experimental && !i.needs_pitch && i.requires == [graphs::ZS_GRAPHS]);
    assert!(run_import_job(&ImportJob::for_family("sigil_zs")).unwrap_err().starts_with("job: "));
}

/// The output buffer a slot's voice needs follows its preset and EP (`tune::min_output_buffer_ms`):
/// the default `sigil_low` on CUDA needs more than the Stable template's 80 / 150 ms.
#[test]
fn a_slots_minimum_buffer_follows_its_preset_and_ep() {
    let slot = |options: serde_json::Value| super::super::family_manifest(FamilyKind::SigilZs, "Z", options);
    assert_eq!(min_output_buffer_ms("cuda", &slot(serde_json::json!({}))), (95.0, 165.0));
    assert_eq!(min_output_buffer_ms("DML", &slot(serde_json::json!({}))), (85.0, 155.0));
    assert_eq!(min_output_buffer_ms("cuda", &slot(serde_json::json!({"preset": "sigil_lean"}))), (80.0, 150.0));
    assert_eq!(min_output_buffer_ms("dml", &slot(serde_json::json!({"preset": "upstream_readme"}))), (125.0, 195.0));
    // Bad options fall back to the default preset's buffer.
    assert_eq!(min_output_buffer_ms("cuda", &slot(serde_json::json!({"steps": "many"}))), (95.0, 165.0));
}

// ---------------------------------------------------------------------------------------------
// Real graphs (ignored)
// ---------------------------------------------------------------------------------------------

fn models_dir() -> Option<PathBuf> {
    let root = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from)?;
    let g = root.join("vc-families").join("seed_vc");
    g.join("xlsr.onnx").is_file().then_some(root)
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// SNR (dB) of `got` against `want`.
fn snr_db(want: &[f32], got: &[f32]) -> f64 {
    assert_eq!(want.len(), got.len());
    let s: f64 = want.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let e: f64 = want.iter().zip(got).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
    10.0 * (s / e.max(1e-30)).log10()
}

/// The Rust port against the reference sampler (`sampler.py` on ORT 1.28's CPU EP, the same graphs
/// and draws; `make_vectors.py`): the reference stage from the sampler's own 22.05 kHz prompt, then
/// one tick with partial decode and one with the whole-tick vocoder. CPU plans, so differences come
/// from the host arithmetic (resample, fbank, the time grid, phase carry) and summation order only.
#[test]
#[ignore = "needs SIGIL_TEST_MODELS_DIR (vc-families/seed_vc) and SIGIL_ZS_VECTORS (make_vectors.py output)"]
fn real_model_matches_the_reference_sampler() {
    let (Some(root), Some(vec_dir)) = (models_dir(), std::env::var_os("SIGIL_ZS_VECTORS").map(PathBuf::from)) else {
        eprintln!("SIGIL_TEST_MODELS_DIR / SIGIL_ZS_VECTORS not set; skipped");
        return;
    };
    let graphs = GraphSet::from_dir(&root.join("vc-families").join("seed_vc"));
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(vec_dir.join("meta.json")).unwrap()).unwrap();
    // The resampler and the helpers first.
    let x = read_f32(&vec_dir.join("resample_in.f32"));
    let want = read_f32(&vec_dir.join("resample_out.f32"));
    let got = super::dsp::resample_torchaudio(&x, 22_050, 16_000);
    let r = snr_db(&want, &got);
    eprintln!("resample 22050 -> 16000 vs torchaudio: {r:.1} dB");
    assert!(r > 100.0);
    for n in [1u32, 4, 6, 10, 30] {
        let want: Vec<f32> = meta["linspace"][n.to_string()].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        assert_eq!(super::dsp::linspace_f32(n), want, "linspace {n}");
    }
    for name in ["sigil_low", "sigil_lean"] {
        let m = &meta[name];
        let p = preset::preset(name);
        let ld = |k: &str| read_f32(&vec_dir.join(format!("{name}.{k}.f32")));
        let ref22 = ld("ref22");
        let (fb, frames) = super::dsp::fbank_centered(&ld("ref16"));
        assert_eq!(frames as u64, m["fbank_frames"].as_u64().unwrap());
        eprintln!("{name}: fbank vs torchaudio {:.1} dB", snr_db(&ld("fbank"), &fb));
        let ref16 = super::dsp::resample_torchaudio(&ref22, 22_050, 16_000);
        eprintln!("{name}: ref16 vs torchaudio {:.1} dB", snr_db(&ld("ref16"), &ref16));
        let tensors = reference::compute(&graphs, &ref22, &cpu_opts()).unwrap();
        for (t, k, gate) in [(&tensors[0], "style", 60.0), (&tensors[1], "prompt_mel", 90.0), (&tensors[2], "prompt_cond", 60.0)] {
            let s = snr_db(&ld(k), &t.data);
            eprintln!("{name}: {k} {s:.1} dB");
            assert!(s > gate, "{name} {k}: {s:.1} dB");
        }
        // One tick on the sampler's own reference and draws.
        let style = ld("style");
        let prompt_mel = ld("prompt_mel");
        let prompt_cond = ld("prompt_cond");
        let pf = prompt_mel.len() / MEL_BINS;
        let r = super::plans::Prepared { style, prompt_mel, prompt_cond, frames: pf };
        let ret_units = m["units"]["return_length"].as_u64().unwrap() as usize;
        let window = ld("window");
        let z_full = ld("z");
        let big_l = m["L"].as_u64().unwrap() as usize;
        let hp = ld("harm_phase");
        let sine_full = ld("sine_noise");
        for (split, out_key) in [(true, "out_partial"), (false, "out_full")] {
            let g = Geometry::stream(&p, p.block_units * 320, ret_units * 320, pf, split).unwrap();
            assert_eq!(g.n16, window.len());
            assert_eq!(g.big_l, big_l);
            let opts = VcSessionOptions { intra_threads: Some(2), ..Default::default() };
            let cfg = PlanConfig {
                graphs: &graphs,
                ep: "cpu",
                opts: &opts,
                levers: GpuLevers::default(),
                geometry: g,
                cfg: p.cfg_rate > 0.0,
                split,
                fp16_xlsr: false,
                fp16_dit: false,
                fp16_vocoder: false,
                component: "vc.zs.parity",
            };
            let mut plans = TickPlans::build(&cfg).unwrap();
            let mut z = vec![0.0f32; MEL_BINS * g.n_mel];
            for c in 0..MEL_BINS {
                z[c * g.n_mel..(c + 1) * g.n_mel].copy_from_slice(&z_full[c * big_l + pf..(c + 1) * big_l]);
            }
            let row = g.n_mel * HOP;
            let w = g.dec_samples();
            let mut sine = vec![0.0f32; N_HARM * w];
            for h in 0..N_HARM {
                sine[h * w..(h + 1) * w].copy_from_slice(&sine_full[h * row + g.dec_start * HOP..(h + 1) * row]);
            }
            let phase: [f32; N_HARM] = std::array::from_fn(|h| hp[h]);
            let t_span = super::sampler::time_grid(p.steps);
            let inputs = TickInputs { owner: 7, reference: &r, t_span: &t_span, cfg_rate: p.cfg_rate, z: &z, sine: &sine, phase: PhaseMode::Start(phase) };
            let mut out = vec![0.0f32; g.out_samples];
            plans.tick(&window, &inputs, &mut out).unwrap();
            let mel = snr_db(&ld("mel_out"), plans.mel());
            let audio = snr_db(&ld(out_key), &out);
            eprintln!("{name} {}: tick mel {mel:.1} dB, audio {audio:.1} dB", if split { "partial" } else { "whole" });
            assert!(mel > 60.0, "{name}: mel {mel:.1} dB");
            assert!(audio > 35.0, "{name}: audio {audio:.1} dB (the ADR gate)");
        }
    }
}

/// A CAM++ embedding with the export's exact graph (Sigil-ZS `campplus`, explicit segment pooling;
/// ADR 0006 finding 1), L2-normalised.
fn speaker_embedding(graphs: &GraphSet, audio_16k: &[f32]) -> Vec<f32> {
    let (fb, frames) = super::dsp::fbank_centered(audio_16k);
    let (path, _) = graphs.path(GraphName::Campplus, false);
    use crate::services::neural_audio::gpu::TensorSpec;
    let io = super::plans::GraphIo {
        inputs: &[TensorSpec::f32("fbank", &[1, frames, 80])],
        outputs: &[TensorSpec::f32("style", &[1, 192])],
        dims: &[("F", frames as i64)],
    };
    let mut plan = super::plans::build_plan("vc.zs.test.campplus", "cpu", &cpu_opts(), &path, io, GpuLevers::default(), true).unwrap();
    plan.input_f32("fbank").unwrap().copy_from_slice(&fb);
    plan.run().unwrap();
    let e = plan.output("style").unwrap().to_vec();
    let norm = e.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
    e.iter().map(|v| v / norm).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum()
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// The real voice streamed through the production harness on a GPU (`SIGIL_ZS_EP`, default
/// `cuda`; `SIGIL_ZS_PRESET`, default `sigil_low`): per-tick wall time, the SOLA joins, and the
/// speaker similarity (exact CAM++) of the streamed output, of the same model offline and of the
/// unconverted source to the reference. `SIGIL_ZS_OUT` = a folder for the WAVs.
#[test]
#[ignore = "needs SIGIL_TEST_MODELS_DIR (vc-families/seed_vc, models/transcribe/samples) and a GPU"]
fn real_model_streams_on_the_gpu() {
    let Some(root) = models_dir() else {
        eprintln!("SIGIL_TEST_MODELS_DIR not set; skipped");
        return;
    };
    crate::services::neural_audio::ort_session::preload_test_cuda_runtime_quiet();
    let ep = std::env::var("SIGIL_ZS_EP").unwrap_or_else(|_| "cuda".into());
    let preset_name = std::env::var("SIGIL_ZS_PRESET").unwrap_or_else(|_| "sigil_low".into());
    let lever_env = |k: &str, d: bool| std::env::var(k).map(|v| v == "1").unwrap_or(d);
    let graphs = GraphSet::from_dir(&root.join("vc-families").join("seed_vc"));
    let samples = root.join("models").join("transcribe").join("samples");
    let clip = ReferenceClip::from_wav(&samples.join("ja-short.wav")).unwrap();
    let env_num = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    let options = SigilZsOptions {
        preset: preset_name.clone(),
        steps: env_num("SIGIL_ZS_STEPS").map(|v| v as u32),
        cfg_rate: env_num("SIGIL_ZS_CFG"),
        prompt_s: env_num("SIGIL_ZS_PROMPT_S"),
        fp16_vocoder: Some(lever_env("SIGIL_ZS_FP16_VOCODER", false)),
        ..Default::default()
    };
    let precision = if lever_env("SIGIL_ZS_TF32", false) { PrecisionMode::Fp16 } else { PrecisionMode::Auto };
    let settings = Settings::new(&options, precision);
    // What the voice adds on the GPU (the measure `LoadedVoice::load_cancellable` claims): sessions, CUDA graphs,
    // working memory after the warm-up.
    let vram = crate::services::neural_audio::gpu::arena::VramProbe::for_rung(&ep, 0);
    let t0 = std::time::Instant::now();
    let e = reference::embedding(&graphs, &clip, settings.preset.prompt_s, None, &cpu_opts()).unwrap();
    let ref_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut lv = levers(lever_env("SIGIL_ZS_PARTIAL", true), lever_env("SIGIL_ZS_ANCHORED", true), lever_env("SIGIL_ZS_PHASE", true));
    lv.levers.set(Lever::IoBinding, true);
    lv.levers.set(Lever::CudaGraphs, ep == "cuda");
    lv.precision = precision;
    let mut m = SigilZsModel::new(ModelConfig {
        graphs: graphs.clone(),
        ep: ep.clone(),
        session: VcSessionOptions { intra_threads: Some(2), ..Default::default() },
        levers: lv,
        settings,
        reference: e.clone(),
        prepared: reference::prepare(&e).unwrap(),
        noise_seed: 1,
        silence_skip: false,
        warnings: Vec::new(),
    });
    m.set_join_shift(match std::env::var("SIGIL_ZS_JOIN").as_deref() {
        Ok("content") => super::engine::JoinShift::Content,
        Ok("output") => super::engine::JoinShift::Output,
        _ => super::engine::JoinShift::Grid,
    });
    let crossfade_ms: f64 = std::env::var("SIGIL_ZS_CROSSFADE_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(40.0);
    let source = std::env::var("SIGIL_ZS_SOURCE").unwrap_or_else(|_| "jfk.wav".into());
    let (src, rate, ch) = crate::services::bench::signals::read_wav(&samples.join(&source)).unwrap();
    let src: Vec<f32> = if ch > 1 { src.chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect() } else { src };
    let src48 = crate::services::bench::signals::resample(&src, rate, 48_000).unwrap();
    let block_frames = settings.block_16k() * 3;
    let t1 = std::time::Instant::now();
    let mut h = VcStreamHarness::with_context(
        Box::new(m),
        48_000,
        48_000,
        1,
        1,
        0,
        (crossfade_ms / 1000.0 * 22_050.0).round() as usize,
        block_frames,
        HarnessContext { reference: Some(e.clone()), singing: Default::default() },
    )
    .unwrap();
    let load_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let vram_loaded = vram.and_then(|p| p.delta_mb());
    // Where each stream session runs. A GPU request keeps every session on a GPU, including a CUDA
    // request whose ladder fell to DirectML (HiFT rebuilt with dynamic shapes; `build_plan`).
    let outcomes = h.model_report().map(|r| r.outcomes).unwrap_or_default();
    let placement: Vec<String> = outcomes.iter().map(|(c, o)| format!("{c}={}", o.active)).collect();
    eprintln!("Sigil-ZS sessions ({ep} requested): {}", placement.join(", "));
    assert!(!outcomes.is_empty() && outcomes.iter().all(|(_, o)| o.active != "cpu"), "{placement:?}");
    let mut out = Vec::new();
    let mut ticks = Vec::new();
    let mut joins = Vec::new();
    for chunk in src48.chunks(block_frames) {
        if chunk.len() < block_frames {
            break;
        }
        let t = std::time::Instant::now();
        out.extend(h.push_and_convert(chunk).unwrap());
        ticks.push(t.elapsed().as_secs_f64() * 1000.0);
        if let Some(j) = h.last_join() {
            joins.push(j);
        }
    }
    let n = ticks.len();
    let vram_streamed = vram.and_then(|p| p.delta_mb());
    let vram_note = match (vram_loaded, vram_streamed) {
        (Some(a), Some(b)) => format!("VRAM +{a:.0} MB loaded, +{b:.0} MB streaming; "),
        _ => String::new(),
    };
    let steady = &mut ticks[2.min(n)..].to_vec();
    let (p50, p95) = (percentile(steady, 0.5), percentile(steady, 0.95));
    let mean_corr = joins.iter().map(|j| f64::from(j.1)).sum::<f64>() / joins.len().max(1) as f64;
    let mean_off = joins.iter().map(|j| j.0 as f64).sum::<f64>() / joins.len().max(1) as f64;
    let out16 = crate::services::bench::signals::resample(&out, 48_000, 16_000).unwrap();
    let src16 = crate::services::bench::signals::resample(&src, rate, 16_000).unwrap();
    let ref16 = clip.load_mono(16_000).unwrap();
    // The same model offline over the whole utterance (CPU plans): the preset's settings (the
    // benchmark's ceiling), and with `SIGIL_ZS_UPSTREAM_OFFLINE=1` also upstream `inference.py`'s
    // defaults (T10: "speaker similarity >= upstream offline for the same reference").
    let offline_with = |settings: Settings, e: &Arc<ReferenceEmbedding>| -> (Vec<f32>, f64) {
        let mut m = SigilZsModel::new(ModelConfig {
            graphs: graphs.clone(),
            ep: "cpu".into(),
            session: VcSessionOptions { intra_threads: Some(4), ..Default::default() },
            levers: levers(false, true, false),
            settings,
            reference: e.clone(),
            prepared: reference::prepare(e).unwrap(),
            noise_seed: 1,
            silence_skip: false,
            warnings: Vec::new(),
        });
        let t = std::time::Instant::now();
        let audio = m.convert(&ctx(&src16, src16.len(), 0)).unwrap();
        (audio, t.elapsed().as_secs_f64() * 1000.0)
    };
    let (offline, offline_ms) = offline_with(Settings::new(&options, PrecisionMode::Auto), &e);
    let offline16 = crate::services::bench::signals::resample(&offline, SR_OUT, 16_000).unwrap();
    let er = speaker_embedding(&graphs, &ref16);
    let skip = 16_000; // the stream's priming
    let sim_stream = cosine(&er, &speaker_embedding(&graphs, &out16[skip.min(out16.len() / 2)..]));
    let sim_offline = cosine(&er, &speaker_embedding(&graphs, &offline16));
    let sim_source = cosine(&er, &speaker_embedding(&graphs, &src16));
    let upstream = lever_env("SIGIL_ZS_UPSTREAM_OFFLINE", false).then(|| upstream_offline(&graphs, &clip, &offline_with));
    let upstream_note = match &upstream {
        Some((audio, ms)) => {
            let a16 = crate::services::bench::signals::resample(audio, SR_OUT, 16_000).unwrap();
            format!(", upstream offline {:.3} ({ms:.0} ms on the CPU)", cosine(&er, &speaker_embedding(&graphs, &a16)))
        }
        None => String::new(),
    };
    eprintln!(
        "Sigil-ZS {preset_name} [steps {}, cfg {}, prompt {} s] on {ep} ({source}): reference {ref_ms:.0} ms, harness load {load_ms:.0} ms; {vram_note}{n} device blocks of {} ms: tick p50 {p50:.2} ms, p95 {p95:.2} ms; joins {} (mean corr {mean_corr:.3}, mean offset {mean_off:.1}); speaker cosine to the reference: stream {sim_stream:.3}, offline {sim_offline:.3} ({offline_ms:.0} ms on the CPU){upstream_note}, source {sim_source:.3}",
        settings.preset.steps,
        settings.preset.cfg_rate,
        settings.preset.prompt_s,
        block_frames / 48,
        joins.len()
    );
    if let Some(dir) = std::env::var_os("SIGIL_ZS_OUT").map(PathBuf::from) {
        let _ = std::fs::create_dir_all(&dir);
        let tag = format!("{preset_name}-{ep}-{}", source.trim_end_matches(".wav"));
        crate::services::bench::signals::write_wav(&dir.join(format!("{tag}-stream.wav")), &out, 48_000, 1).unwrap();
        crate::services::bench::signals::write_wav(&dir.join(format!("{tag}-offline.wav")), &offline, SR_OUT, 1).unwrap();
        if let Some((audio, _)) = &upstream {
            crate::services::bench::signals::write_wav(&dir.join(format!("{tag}-upstream-offline.wav")), audio, SR_OUT, 1).unwrap();
        }
    }
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(sim_stream > sim_source + 0.1, "the stream sounds like the reference ({sim_stream:.3} vs source {sim_source:.3})");
}

/// A one-off conversion of the test's source with given settings and reference: `(audio, ms)`.
type OfflineConvert<'a> = dyn Fn(Settings, &Arc<ReferenceEmbedding>) -> (Vec<f32>, f64) + 'a;

/// Upstream's offline conversion (`inference.py` defaults: the reference up to 25 s as the prompt,
/// 30 Euler steps, CFG 0.7, the whole-tick HiFT, one chunk for a source up to 30 s) through the same
/// graphs, which match upstream PyTorch (ADR 0006 parity). Returns `(audio at 22.05 kHz, ms)`.
fn upstream_offline(graphs: &GraphSet, clip: &ReferenceClip, convert: &OfflineConvert<'_>) -> (Vec<f32>, f64) {
    let mut wav = clip.load_mono(SR_OUT).unwrap();
    wav.truncate(SR_OUT as usize * 25);
    let opts = VcSessionOptions { intra_threads: Some(4), ..Default::default() };
    let e = Arc::new(ReferenceEmbedding { tensors: reference::compute(graphs, &wav, &opts).unwrap(), ..Default::default() });
    let options = SigilZsOptions { preset: "upstream_readme".into(), steps: Some(30), cfg_rate: Some(0.7), ..Default::default() };
    convert(Settings::new(&options, PrecisionMode::Fp32), &e)
}
