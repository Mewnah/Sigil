//! MeanVC2 family tests.
//!
//! - A pure-Rust graph set ([`FakeGraphs`]) with simple, exactly known math and a call log drives
//!   the loop and the model: length accounting, chunk invariance, the mean-flow step schedule, the
//!   offset wraps, anchored noise, the silence skip, discontinuities, errors, hot reference changes
//!   and the governor.
//! - ONNX stand-ins of the same math ([`stand_in_models`], hand-encoded with `test_onnx`) run the
//!   real ONNX Runtime path ([`OrtGraphs`], the I/O contract checks, `families::create` with a voice
//!   folder) in `cargo test`, and must match the pure-Rust set.
//! - `real_model_*` tests (`#[ignore]`) run Sigil's export of the real graphs from
//!   `SIGIL_TEST_MODELS_DIR/meanvc/` or the dev cache (`sigil-dev-cache/vc-families/meanvc/`):
//!   real-time factor through the production stream harness, and parity with the Python reference
//!   loop when `SIGIL_MEANVC_REFERENCE_DIR` holds one.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::graphs::{
    DitInputs, GraphName, GraphShapes, GraphSource, MeanVcGraphs, OrtGraphs, SpeakerState, ASR_WINDOW, BN_DIM, DIT_FRAMES, FBANK_WINDOW, HOP,
    MEL_BINS, SPK_DIM,
};
use super::model::{block_is_loud, MeanVcModel, ModelConfig, FADE_16K, SILENCE_PRIME_16K, SKIP_AFTER_QUIET_16K, TICK_16K};
use super::stream::{MeanVcStream, OFFSET_WRAP, PHASE_LEAD_16K};
use super::*;
use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT, INT64};
use crate::services::neural_audio::vc::engine::governor::QualityLevel;
use crate::services::neural_audio::vc::families::reference::ReferenceClip;
use crate::services::neural_audio::vc::traits::{ConversionCtx, SingingFlags, VoiceControls, VoiceConversionModel};
use crate::services::neural_audio::vc::VcStreamHarness;

// ---------------------------------------------------------------------------------------------
// The pure-Rust graph set
// ---------------------------------------------------------------------------------------------

/// One `dit_step` call as the fake saw it.
#[derive(Clone, Debug)]
struct DitCall {
    t: f32,
    r: f32,
    dt: f32,
    offset: i64,
    n_valid: i64,
    cache0: f32,
    x: Vec<f32>,
}

#[derive(Default)]
struct Log {
    asr_offsets: Vec<i64>,
    dit: Vec<DitCall>,
    gtm: usize,
    vocos_frames: Vec<usize>,
}

/// The math of the ONNX stand-ins, in Rust:
/// - fbank: every bin of frame `f` = mean of `wav[160 f .. 160 f + 400]`;
/// - asr: BN frame `j` (0, 1) = fbank frame `3 + 4 j`, bin `c % 80` per value; `att += 1e-3 offset`,
///   `cnn += rcs`;
/// - gtm: `k_mem[i] = spks[i % 256]`, `v_mem = 2 k_mem`;
/// - dit_step: `target = cond[..80] + spks[0] + 0.01 (t − r)`, `x_next = x − dt (x − target)`,
///   `cache_out = cache + dt + 1e-3 offset + 1e-6 n_valid`;
/// - vocos: `wav[t]` = the channel mean of mel frame `(t + 80) / 160` (a centred hold).
struct FakeGraphs {
    shapes: GraphShapes,
    log: Arc<Mutex<Log>>,
    /// Fail the n-th dit_step call (0-based).
    fail_dit_at: Option<usize>,
    dit_calls: usize,
}

impl FakeGraphs {
    fn new() -> (FakeGraphs, Arc<Mutex<Log>>) {
        let log = Arc::new(Mutex::new(Log::default()));
        (
            FakeGraphs {
                shapes: GraphShapes::meanvc2_40ms(),
                log: log.clone(),
                fail_dit_at: None,
                dit_calls: 0,
            },
            log,
        )
    }
}

impl MeanVcGraphs for FakeGraphs {
    fn shapes(&self) -> &GraphShapes {
        &self.shapes
    }

    fn fbank(&mut self, wav: &[f32], out: &mut Vec<f32>) -> Result<usize, String> {
        if wav.len() < FBANK_WINDOW {
            return Ok(0);
        }
        let frames = (wav.len() - FBANK_WINDOW) / HOP + 1;
        for f in 0..frames {
            let m = wav[f * HOP..f * HOP + FBANK_WINDOW].iter().sum::<f32>() / FBANK_WINDOW as f32;
            out.extend(std::iter::repeat_n(m, MEL_BINS));
        }
        Ok(frames)
    }

    fn asr(&mut self, fbank: &[f32], offset: i64, att: &mut [f32], cnn: &mut [f32], bn: &mut [f32]) -> Result<(), String> {
        assert_eq!(fbank.len(), ASR_WINDOW * MEL_BINS);
        for j in 0..2 {
            for c in 0..BN_DIM {
                bn[j * BN_DIM + c] = fbank[(3 + 4 * j) * MEL_BINS + c % MEL_BINS];
            }
        }
        let o = offset as f32 * 1e-3;
        att.iter_mut().for_each(|v| *v += o);
        cnn.iter_mut().for_each(|v| *v += 4.0);
        self.log.lock().unwrap().asr_offsets.push(offset);
        Ok(())
    }

    fn gtm(&mut self, spks: &[f32], k_mem: &mut [f32], v_mem: &mut [f32]) -> Result<(), String> {
        for (i, (k, v)) in k_mem.iter_mut().zip(v_mem.iter_mut()).enumerate() {
            *k = spks[i % SPK_DIM];
            *v = *k * 2.0;
        }
        self.log.lock().unwrap().gtm += 1;
        Ok(())
    }

    fn dit_step(&mut self, inputs: &DitInputs<'_>, speaker: &SpeakerState, x_next: &mut [f32], cache_out: &mut [f32]) -> Result<(), String> {
        let call = self.dit_calls;
        self.dit_calls += 1;
        if self.fail_dit_at == Some(call) {
            return Err("injected dit_step failure".into());
        }
        let shift = 0.01 * (inputs.t - inputs.r);
        for f in 0..DIT_FRAMES {
            for c in 0..MEL_BINS {
                let target = inputs.cond[f * BN_DIM + c] + speaker.spks[0] + shift;
                let x = inputs.x[f * MEL_BINS + c];
                x_next[f * MEL_BINS + c] = x - inputs.dt * (x - target);
            }
        }
        let (o, n) = (inputs.offset as f32 * 1e-3, inputs.n_valid as f32 * 1e-6);
        for (out, &c) in cache_out.iter_mut().zip(inputs.cache) {
            *out = c + inputs.dt + o + n;
        }
        self.log.lock().unwrap().dit.push(DitCall {
            t: inputs.t,
            r: inputs.r,
            dt: inputs.dt,
            offset: inputs.offset,
            n_valid: inputs.n_valid,
            cache0: inputs.cache[0],
            x: inputs.x.to_vec(),
        });
        Ok(())
    }

    fn vocos(&mut self, mel: &[f32], frames: usize, out: &mut Vec<f32>) -> Result<(), String> {
        let means: Vec<f32> = (0..frames)
            .map(|m| (0..MEL_BINS).map(|c| mel[c * frames + m]).sum::<f32>() / MEL_BINS as f32)
            .collect();
        out.clear();
        out.extend((0..HOP * frames.saturating_sub(1)).map(|t| means[((t + HOP / 2) / HOP).min(frames - 1)]));
        self.log.lock().unwrap().vocos_frames.push(frames);
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// The ONNX stand-ins (the same math; `GraphName::ALL` order)
// ---------------------------------------------------------------------------------------------

const OPSET: u64 = 13;

fn fbank_model() -> Vec<u8> {
    let mut g = Graph::new();
    g.input("wav16k", FLOAT, &[Dim::Fixed(1), Dim::Named("N")])
        .output("fbank", FLOAT, &[Dim::Fixed(1), Dim::Named("F"), Dim::Fixed(80)])
        .init_i64("shape3", &[3], &[1, 1, -1])
        .init_f32("ones", &[1, 80], &[1.0; 80])
        .node("Reshape", &["wav16k", "shape3"], &["w3"], &[])
        .node(
            "AveragePool",
            &["w3"],
            &["p"],
            &[Attr::Ints("kernel_shape", vec![400]), Attr::Ints("strides", vec![160])],
        )
        .node("Transpose", &["p"], &["pt"], &[Attr::Ints("perm", vec![0, 2, 1])])
        .node("Mul", &["pt", "ones"], &["fbank"], &[]);
    g.model_bytes(OPSET)
}

fn asr_model() -> Vec<u8> {
    let chan: Vec<i64> = (0..BN_DIM as i64).map(|c| c % MEL_BINS as i64).collect();
    let mut g = Graph::new();
    g.input("fbank", FLOAT, &[Dim::Fixed(1), Dim::Fixed(11), Dim::Fixed(80)])
        .input("offset", INT64, &[])
        .input("rcs", INT64, &[])
        .input("att_cache", FLOAT, &[Dim::Fixed(6), Dim::Fixed(4), Dim::Fixed(4), Dim::Fixed(128)])
        .input("cnn_cache", FLOAT, &[Dim::Fixed(6), Dim::Fixed(1), Dim::Fixed(256), Dim::Fixed(8)])
        .output("bn", FLOAT, &[Dim::Fixed(1), Dim::Fixed(2), Dim::Fixed(256)])
        .output("att_cache_out", FLOAT, &[Dim::Fixed(6), Dim::Fixed(4), Dim::Named("A"), Dim::Fixed(128)])
        .output("cnn_cache_out", FLOAT, &[Dim::Fixed(6), Dim::Fixed(1), Dim::Fixed(256), Dim::Fixed(8)])
        .init_i64("frames", &[2], &[3, 7])
        .init_i64("chan", &[BN_DIM as u64], &chan)
        .init_f32("k1e3", &[], &[1e-3])
        .node("Gather", &["fbank", "frames"], &["g1"], &[Attr::Int("axis", 1)])
        .node("Gather", &["g1", "chan"], &["bn"], &[Attr::Int("axis", 2)])
        .node("Cast", &["offset"], &["off_f"], &[Attr::Int("to", 1)])
        .node("Mul", &["off_f", "k1e3"], &["off_s"], &[])
        .node("Add", &["att_cache", "off_s"], &["att_cache_out"], &[])
        .node("Cast", &["rcs"], &["rcs_f"], &[Attr::Int("to", 1)])
        .node("Add", &["cnn_cache", "rcs_f"], &["cnn_cache_out"], &[]);
    g.model_bytes(OPSET)
}

fn gtm_model() -> Vec<u8> {
    let mut g = Graph::new();
    g.input("spks", FLOAT, &[Dim::Fixed(1), Dim::Fixed(256)])
        .output("k_mem", FLOAT, &[Dim::Fixed(1), Dim::Fixed(32), Dim::Fixed(256)])
        .output("v_mem", FLOAT, &[Dim::Fixed(1), Dim::Fixed(32), Dim::Fixed(256)])
        .init_i64("shape", &[3], &[1, 32, 256])
        .init_f32("two", &[], &[2.0])
        .node("Expand", &["spks", "shape"], &["k_mem"], &[])
        .node("Mul", &["k_mem", "two"], &["v_mem"], &[]);
    g.model_bytes(OPSET)
}

fn dit_model() -> Vec<u8> {
    dit_model_with_cache(&[4, 2, 2, 12, 64])
}

/// The DiT stand-in with the KV cache shape `cache` (the input and `cache_out`).
fn dit_model_with_cache(cache: &[u64]) -> Vec<u8> {
    let f = |d: u64| Dim::Fixed(d);
    let cache: Vec<Dim> = cache.iter().map(|&d| f(d)).collect();
    let mut g = Graph::new();
    g.input("x", FLOAT, &[f(1), f(8), f(80)])
        .input("t", FLOAT, &[f(1)])
        .input("r", FLOAT, &[f(1)])
        .input("dt", FLOAT, &[f(1)])
        .input("cond", FLOAT, &[f(1), f(8), f(256)])
        .input("spks", FLOAT, &[f(1), f(256)])
        .input("k_mem", FLOAT, &[f(1), f(32), f(256)])
        .input("v_mem", FLOAT, &[f(1), f(32), f(256)])
        .input("offset", INT64, &[f(1)])
        .input("n_valid", INT64, &[f(1)])
        .input("cache", FLOAT, &cache)
        .output("x_next", FLOAT, &[f(1), f(8), f(80)])
        .output("cache_out", FLOAT, &cache)
        .init_i64("s0", &[1], &[0])
        .init_i64("e80", &[1], &[80])
        .init_i64("e1", &[1], &[1])
        .init_i64("ax1", &[1], &[1])
        .init_i64("ax2", &[1], &[2])
        .init_f32("k001", &[], &[0.01])
        .init_f32("k1e3", &[], &[1e-3])
        .init_f32("k1e6", &[], &[1e-6])
        .init_f32("zero", &[], &[0.0])
        .node("Slice", &["cond", "s0", "e80", "ax2"], &["c80"], &[])
        .node("Slice", &["spks", "s0", "e1", "ax1"], &["sp0"], &[])
        .node("Add", &["c80", "sp0"], &["a1"], &[])
        .node("Sub", &["t", "r"], &["tr"], &[])
        .node("Mul", &["tr", "k001"], &["trs"], &[])
        .node("Add", &["a1", "trs"], &["target"], &[])
        .node("Sub", &["x", "target"], &["u"], &[])
        .node("Mul", &["dt", "u"], &["du"], &[])
        .node("Sub", &["x", "du"], &["x_next"], &[])
        // The memories feed the graph (as in the real DiT) without changing the stand-in's math.
        .node("ReduceSum", &["k_mem"], &["ks"], &[Attr::Int("keepdims", 0)])
        .node("ReduceSum", &["v_mem"], &["vs"], &[Attr::Int("keepdims", 0)])
        .node("Add", &["ks", "vs"], &["kvs"], &[])
        .node("Mul", &["kvs", "zero"], &["kv0"], &[])
        .node("Add", &["cache", "dt"], &["c1"], &[])
        .node("Cast", &["offset"], &["of"], &[Attr::Int("to", 1)])
        .node("Mul", &["of", "k1e3"], &["ofs"], &[])
        .node("Add", &["c1", "ofs"], &["c2"], &[])
        .node("Cast", &["n_valid"], &["nv"], &[Attr::Int("to", 1)])
        .node("Mul", &["nv", "k1e6"], &["nvs"], &[])
        .node("Add", &["c2", "nvs"], &["c3"], &[])
        .node("Add", &["c3", "kv0"], &["cache_out"], &[]);
    g.model_bytes(OPSET)
}

fn vocos_model() -> Vec<u8> {
    let mut g = Graph::new();
    g.input("mel", FLOAT, &[Dim::Fixed(1), Dim::Fixed(80), Dim::Named("M")])
        .output("wav", FLOAT, &[Dim::Fixed(1), Dim::Named("S")])
        .init_f32("w", &[1, 1, 160], &[1.0; 160])
        .init_i64("shape2", &[2], &[1, -1])
        .node("ReduceMean", &["mel"], &["m1"], &[Attr::Ints("axes", vec![1]), Attr::Int("keepdims", 1)])
        .node(
            "ConvTranspose",
            &["m1", "w"],
            &["c"],
            &[Attr::Ints("strides", vec![160]), Attr::Ints("pads", vec![80, 80])],
        )
        .node("Reshape", &["c", "shape2"], &["wav"], &[]);
    g.model_bytes(OPSET)
}

/// The five stand-ins in [`GraphName::ALL`] order.
fn stand_in_models() -> [Vec<u8>; 5] {
    [fbank_model(), asr_model(), gtm_model(), dit_model(), vocos_model()]
}

/// Write the stand-ins (and optionally more files) into a fresh voice folder.
fn stand_in_slot(tag: &str) -> PathBuf {
    let dir = speaker::tests::test_dir(&format!("slot-{tag}"));
    for (name, bytes) in GraphName::ALL.iter().zip(stand_in_models()) {
        std::fs::write(dir.join(name.file()), bytes).unwrap();
    }
    dir
}

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

fn spks(seed: f32) -> Vec<f32> {
    (0..SPK_DIM)
        .map(|i| ((i as f32 + 1.0) * 0.37 + seed).sin() * 0.5)
        .collect()
}

fn fake_model(block: usize, steps: u32, silence_skip: bool) -> (MeanVcModel, Arc<Mutex<Log>>) {
    let (graphs, log) = FakeGraphs::new();
    fake_model_with(graphs, log, block, steps, silence_skip)
}

fn fake_model_with(graphs: FakeGraphs, log: Arc<Mutex<Log>>, block: usize, steps: u32, silence_skip: bool) -> (MeanVcModel, Arc<Mutex<Log>>) {
    let stream = MeanVcStream::new(Box::new(graphs), &spks(0.0), steps, 7).unwrap();
    let model = MeanVcModel::new(
        stream,
        &ModelConfig {
            block_16k: block,
            steps,
            silence_skip,
            align_16k: 0,
        },
    )
    .unwrap();
    (model, log)
}

/// A slowly varying test signal (the stand-ins' fbank is a frame mean, so it must not average out).
fn signal(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            0.4 * (std::f32::consts::TAU * 3.0 * t).sin() + 0.2 * (std::f32::consts::TAU * 0.7 * t).cos()
        })
        .collect()
}

/// Drive a model like the fixed-block harness does (context = `want`, `stream_pos` advancing by
/// one block); returns the newest block of every call (the output stream).
fn drive(model: &mut MeanVcModel, input: &[f32], want: usize, reference: Option<&ReferenceEmbedding>) -> (Vec<f32>, Vec<Result<(), String>>) {
    let block = model.fixed_block().unwrap();
    let ctx_len = model.configure_stream(block, want, 0).unwrap();
    let mut ring = vec![0.0f32; ctx_len];
    let mut out = Vec::new();
    let mut results = Vec::new();
    for (i, chunk) in input.chunks_exact(block).enumerate() {
        ring.drain(..block);
        ring.extend_from_slice(chunk);
        let ctx = ConversionCtx {
            mono_16k: &ring,
            new_frames: want,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: (i * block) as u64,
            reference,
            singing: SingingFlags::SPEECH,
        };
        match model.convert(&ctx) {
            Ok(chunk) => {
                assert_eq!(chunk.len(), want, "call {i}: exactly new_frames samples");
                out.extend_from_slice(&chunk[want - block..]);
                results.push(Ok(()));
            }
            Err(e) => {
                out.extend(std::iter::repeat_n(f32::NAN, block));
                results.push(Err(e));
            }
        }
    }
    (out, results)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

// ---------------------------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------------------------

#[test]
fn a_cold_stream_reaches_one_block_per_block_after_its_first_call() {
    let (graphs, log) = FakeGraphs::new();
    let mut stream = MeanVcStream::new(Box::new(graphs), &spks(0.0), 2, 1).unwrap();
    stream.reset_cold();
    let x = signal(TICK_16K * 6);
    let mut produced = Vec::new();
    for chunk in x.as_chunks::<TICK_16K>().0 {
        let mut out = Vec::new();
        produced.push(stream.process(chunk, &mut out).unwrap());
        assert_eq!(out.len(), *produced.last().unwrap());
    }
    // The vocoder's first call has no overlap (320); every call after gives its input length.
    assert_eq!(produced, vec![320, 1280, 1280, 1280, 1280, 1280]);
    let log = log.lock().unwrap();
    // The phase lead makes an ASR window end on every 80 ms: one call per tick from the first.
    assert_eq!(log.asr_offsets.len(), 6);
    assert_eq!(log.asr_offsets[..3], [4, 6, 8]);
    // 2 DiT calls per tick (1 on the first), 2 steps each; Vocos 4 frames first, then 2 + 4.
    assert_eq!(log.dit.len(), (1 + 2 * 5) * 2);
    assert_eq!(log.vocos_frames[..3], [4, 6, 6]);
}

#[test]
fn without_the_phase_lead_every_window_waits_a_tick() {
    // The lead is what lets a tick's audio be encoded in the same tick (module docs of stream.rs).
    let (graphs, log) = FakeGraphs::new();
    let mut stream = MeanVcStream::new(Box::new(graphs), &spks(0.0), 1, 1).unwrap();
    stream.reset_cold();
    let mut out = Vec::new();
    stream.process(&vec![0.0; TICK_16K], &mut out).unwrap();
    assert_eq!(
        log.lock().unwrap().asr_offsets.len(),
        1,
        "with the lead the first 80 ms complete a window"
    );
    assert_eq!(
        PHASE_LEAD_16K + TICK_16K,
        FBANK_WINDOW + (ASR_WINDOW - 1) * HOP,
        "the lead completes exactly one window"
    );
}

#[test]
fn ticks_of_80_and_160_ms_give_the_same_output_stream() {
    let x = signal(TICK_16K * 40);
    let (mut a, _) = fake_model(TICK_16K, 2, false);
    let (mut b, _) = fake_model(2 * TICK_16K, 2, false);
    let (sa, ra) = drive(&mut a, &x, TICK_16K + 200, None);
    let (sb, rb) = drive(&mut b, &x, 2 * TICK_16K + 200, None);
    assert!(ra.iter().chain(&rb).all(Result::is_ok));
    assert_eq!(sa.len(), sb.len());
    assert_eq!(sa, sb, "the loop is chunk-invariant in the steady state (bit for bit)");
    assert!(sa.iter().any(|&v| v != sa[0]), "the stand-in output follows the input");
}

#[test]
fn mean_flow_steps_follow_upstream_and_share_the_calls_cache() {
    for steps in [1u32, 2] {
        let (mut model, log) = fake_model(TICK_16K, steps, false);
        log.lock().unwrap().dit.clear();
        let pre = model.silence_state().clone();
        let _ = drive(&mut model, &signal(TICK_16K), TICK_16K + 100, None);
        let log = log.lock().unwrap();
        assert_eq!(log.dit.len(), 2 * steps as usize, "two DiT calls per 80 ms tick");
        let expected: &[(f32, f32)] = if steps == 1 { &[(1.0, 0.0)] } else { &[(1.0, 0.5), (0.5, 0.0)] };
        for (i, call) in log.dit.iter().enumerate() {
            let (t, r) = expected[i % steps as usize];
            assert_eq!((call.t, call.r, call.dt), (t, r, 1.0 / steps as f32), "step {i} of {steps}");
        }
        // Every step of one call reads the same cache; the kept cache is the last step's output.
        let first = &log.dit[..steps as usize];
        assert!(first.iter().all(|c| c.cache0 == first[0].cache0));
        let second = &log.dit[steps as usize];
        let call = &first[0];
        let kept = call.cache0 + 1.0 / steps as f32 + call.offset as f32 * 1e-3 + call.n_valid as f32 * 1e-6;
        assert!((second.cache0 - kept).abs() < 1e-4, "{steps} steps: {} vs {kept}", second.cache0);
        assert_eq!(second.offset, call.offset + 4);
        drop(pre);
    }
}

#[test]
fn offsets_and_the_kv_cache_wrap_like_upstream() {
    let (graphs, log) = FakeGraphs::new();
    let mut stream = MeanVcStream::new(Box::new(graphs), &spks(0.0), 1, 1).unwrap();
    stream.reset_cold();
    // 1100 DiT calls (offset 0, 4, ..., wrapping at 4000) and 550 ASR calls.
    let zeros = vec![0.0f32; TICK_16K];
    let mut out = Vec::new();
    for _ in 0..551 {
        stream.process(&zeros, &mut out).unwrap();
    }
    let log = log.lock().unwrap();
    let offsets: Vec<i64> = log.dit.iter().map(|c| c.offset).collect();
    let n_valid: Vec<i64> = log.dit.iter().map(|c| c.n_valid).collect();
    assert_eq!(offsets[..4], [0, 4, 8, 12]);
    assert_eq!(n_valid[..5], [0, 4, 8, 12, 12], "n_valid stops at the cache's 12 frames");
    let wrap = offsets.iter().position(|&o| o == OFFSET_WRAP - 4).unwrap() + 1;
    assert_eq!((offsets[wrap], n_valid[wrap]), (0, 0), "offset 4000 restarts at 0 with an empty cache");
    assert_eq!(log.dit[wrap].cache0, 0.0, "the KV cache is cleared at the wrap");
    // ASR: 4, 6, ... checked at each call; the stand-in loop wraps at 4000 to max(4, offset - 4000).
    assert_eq!(log.asr_offsets[0], 4);
    assert!(log
        .asr_offsets
        .windows(2)
        .all(|w| w[1] == w[0] + 2 || (w[0] >= OFFSET_WRAP - 2 && w[1] < 10)));
}

#[test]
fn every_mel_frames_noise_is_drawn_once_and_reused_when_it_becomes_current() {
    let run = |seed: u64| {
        let (graphs, log) = FakeGraphs::new();
        let mut stream = MeanVcStream::new(Box::new(graphs), &spks(0.0), 2, seed).unwrap();
        stream.reset_cold();
        let mut out = Vec::new();
        for chunk in signal(TICK_16K * 4).as_chunks::<TICK_16K>().0 {
            stream.process(chunk, &mut out).unwrap();
        }
        let log = log.lock().unwrap();
        // The first step of each call sees the noise (later steps see the updated x).
        log.dit
            .iter()
            .step_by(2)
            .map(|c| c.x.clone())
            .collect::<Vec<_>>()
    };
    let a = run(7);
    assert!(a.len() >= 3);
    for w in a.windows(2) {
        assert_eq!(w[1][..4 * MEL_BINS], w[0][4 * MEL_BINS..], "the future frames' noise is reused");
        assert_ne!(w[1][4 * MEL_BINS..], w[0][4 * MEL_BINS..]);
    }
    let unit = a[0].iter().map(|v| v * v).sum::<f32>() / a[0].len() as f32;
    assert!((0.6..1.5).contains(&unit), "unit Gaussian noise (mean square {unit})");
    assert_eq!(run(7), a, "the same seed draws the same noise");
    assert_ne!(run(8)[0], a[0]);
}

// ---------------------------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------------------------

#[test]
fn the_model_starts_from_steady_silence_and_returns_one_block_per_call() {
    let (mut model, log) = fake_model(TICK_16K, 2, false);
    assert_eq!(model.input_sample_rate(), 16_000);
    assert_eq!(model.output_sample_rate(), 16_000);
    assert_eq!(model.fixed_block(), Some(TICK_16K));
    assert_eq!(model.min_context_samples(), TICK_16K);
    // The silence prime ran from cold (5 ASR calls) and the stream stands at its end.
    assert_eq!(log.lock().unwrap().asr_offsets.len(), SILENCE_PRIME_16K / TICK_16K);
    assert_eq!(model.stream().state(), model.silence_state());
    let want = TICK_16K + 333;
    let (out, results) = drive(&mut model, &signal(TICK_16K * 10), want, None);
    assert!(results.iter().all(Result::is_ok));
    assert_eq!(out.len(), TICK_16K * 10);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn returned_chunks_overlap_sample_for_sample() {
    let (mut model, _) = fake_model(TICK_16K, 2, false);
    let want = TICK_16K + 400;
    model.configure_stream(TICK_16K, want, 0).unwrap();
    let x = signal(TICK_16K * 8);
    let mut ring = vec![0.0f32; want];
    let mut previous: Option<Vec<f32>> = None;
    for (i, chunk) in x.as_chunks::<TICK_16K>().0.iter().enumerate() {
        ring.drain(..TICK_16K);
        ring.extend_from_slice(chunk);
        let ctx = ConversionCtx {
            mono_16k: &ring,
            new_frames: want,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: (i * TICK_16K) as u64,
            reference: None,
            singing: SingingFlags::SPEECH,
        };
        let out = model.convert(&ctx).unwrap();
        if let Some(prev) = previous {
            // The overlap the harness's SOLA crossfades is identical audio.
            assert_eq!(out[..want - TICK_16K], prev[TICK_16K..]);
        }
        previous = Some(out);
    }
}

#[test]
fn the_production_harness_joins_meanvc_blocks_at_offset_zero_with_exact_lengths() {
    for block in [TICK_16K / 2, TICK_16K, 2 * TICK_16K] {
        let (model, _) = fake_model(block, 2, false);
        let mut harness = VcStreamHarness::new(Box::new(model), 48_000, 48_000, 1, 1, 0, 480, 480).unwrap();
        assert_eq!(harness.fixed_shape().map(|(b, _)| b), Some(block));
        let x = signal(48_000 * 3);
        let mut total = 0;
        let mut joins = Vec::new();
        for chunk in x.as_chunks::<480>().0 {
            let out = harness.push_and_convert(chunk).unwrap();
            assert_eq!(out.len(), harness.last_owed_frames());
            total += out.len();
            if let Some(j) = harness.last_join() {
                joins.push(j);
            }
        }
        assert_eq!(total, x.len() - x.len() % 480, "block {block}");
        assert!(joins.len() > 5, "block {block}");
        for (offset, ncc) in &joins {
            assert_eq!(
                *offset, 0,
                "block {block}: consecutive MeanVC2 chunks agree, so SOLA joins at the nominal point"
            );
            assert!(*ncc > 0.999, "block {block}: ncc {ncc}");
        }
    }
}

#[test]
fn a_40_ms_block_spreads_the_tick_and_plays_the_same_stream_one_block_later() {
    // 1 step: the stand-in's x_next is its target up to rounding, so the two voices (whose noise
    // indices differ after their primes) agree to float precision.
    let half = TICK_16K / 2;
    let (mut split, log) = fake_model(half, 1, false);
    let (mut whole, _) = fake_model(TICK_16K, 1, false);
    let x = signal(TICK_16K * 30);
    let (dit_before, asr_before) = {
        let l = log.lock().unwrap();
        (l.dit.len(), l.asr_offsets.len())
    };
    // Every 40 ms call runs exactly one DiT call; the ASR runs on every other call.
    let mut per_call = Vec::new();
    let want = half + 200;
    split.configure_stream(half, want, 0).unwrap();
    let mut ring = vec![0.0f32; want];
    let mut a = Vec::new();
    for (i, chunk) in x.chunks_exact(half).enumerate() {
        ring.drain(..half);
        ring.extend_from_slice(chunk);
        let before = log.lock().unwrap().dit.len();
        let out = split
            .convert(&ConversionCtx {
                mono_16k: &ring,
                new_frames: want,
                pitch_shift_semitones: 0.0,
                index_rate: 0.0,
                protect: 0.33,
                speaker_id: 0,
                stream_pos_16k: (i * half) as u64,
                reference: None,
                singing: SingingFlags::SPEECH,
            })
            .unwrap();
        per_call.push(log.lock().unwrap().dit.len() - before);
        a.extend_from_slice(&out[want - half..]);
    }
    assert!(per_call.iter().all(|&n| n == 1), "one DiT call per 40 ms block: {per_call:?}");
    let l = log.lock().unwrap();
    assert_eq!(l.dit.len() - dit_before, x.len() / half);
    assert_eq!(l.asr_offsets.len() - asr_before, x.len() / TICK_16K, "one ASR call per 80 ms");
    drop(l);
    let (b, rb) = drive(&mut whole, &x, TICK_16K + 200, None);
    assert!(rb.iter().all(Result::is_ok));
    // The split voice plays the whole-tick stream exactly one 40 ms block later.
    let from = 2 * TICK_16K;
    let diff = max_abs_diff(&a[from + half..], &b[from..b.len() - half]);
    assert!(diff < 1e-5, "split vs whole tick differ by {diff}");
    assert!(b[from..].iter().any(|&v| (v - b[from]).abs() > 1e-3), "the stand-in output varies");
}

#[test]
fn the_silence_skip_stops_the_model_and_resumes_without_losing_the_onset() {
    // loud 0.5 s, silence 1.2 s, loud 0.8 s; 1 step (the stand-in output is then noise-free).
    let loud = |n: usize, at: usize| signal(at + n)[at..].to_vec();
    let mut x = loud(8_000, 0);
    x.extend(vec![0.0; 19_200]);
    x.extend(loud(12_800, 30_000));
    x.truncate(x.len() / TICK_16K * TICK_16K);
    let want = TICK_16K + 300;
    let (mut on, log_on) = fake_model(TICK_16K, 1, true);
    let (mut off, log_off) = fake_model(TICK_16K, 1, false);
    let (so, ro) = drive(&mut on, &x, want, None);
    let (sf, rf) = drive(&mut off, &x, want, None);
    assert!(ro.iter().chain(&rf).all(Result::is_ok));
    let blocks = x.len() / TICK_16K;
    assert!(on.blocks_skipped >= 8, "silence was skipped ({} blocks)", on.blocks_skipped);
    assert_eq!(on.blocks_converted + on.blocks_skipped, blocks as u64);
    assert_eq!(off.blocks_converted, blocks as u64);
    assert!(log_on.lock().unwrap().dit.len() < log_off.lock().unwrap().dit.len());
    // Before the skip the streams are identical (except the fade into the skip).
    let quiet_from = 8_000usize.div_ceil(TICK_16K) * TICK_16K;
    let skip_start = (quiet_from + SKIP_AFTER_QUIET_16K).div_ceil(TICK_16K) * TICK_16K;
    assert_eq!(so[..skip_start - FADE_16K], sf[..skip_start - FADE_16K]);
    // While skipping the output is silent.
    assert!(so[skip_start..skip_start + TICK_16K]
        .iter()
        .all(|&v| v == 0.0));
    // After the resume (the first loud block, the replayed quiet block before it, a 5 ms fade in)
    // the stream equals the one that never skipped. The loud part starts at 27 200 (block 21).
    let resume = 27_200 / TICK_16K * TICK_16K;
    assert!(
        so[resume - TICK_16K..resume].iter().all(|&v| v == 0.0),
        "still skipping before the onset block"
    );
    let tail = resume + FADE_16K;
    assert!(max_abs_diff(&so[tail..], &sf[tail..]) < 1e-6, "after the resume the skip changed nothing");
    assert!(so[tail..].iter().any(|&v| v != 0.0));
}

#[test]
fn the_silence_skip_works_with_40_ms_blocks() {
    // The shipping block of the 80 ms tick. A resume can restore the steady silence halfway through
    // an 80 ms ASR window, so the resumed stream chunks its windows 4 fbank frames apart from one
    // that never skipped (the same latency and length; the fade covers the cut): this checks the
    // behaviour, the whole-tick test above checks equality.
    let half = TICK_16K / 2;
    let loud = |n: usize, at: usize| signal(at + n)[at..].to_vec();
    let mut x = loud(8_000, 0);
    x.extend(vec![0.0; 19_200]);
    x.extend(loud(12_800, 30_000));
    x.truncate(x.len() / half * half);
    let want = half + 300;
    let (mut on, log_on) = fake_model(half, 1, true);
    let (mut off, log_off) = fake_model(half, 1, false);
    let (so, ro) = drive(&mut on, &x, want, None);
    let (sf, rf) = drive(&mut off, &x, want, None);
    assert!(ro.iter().chain(&rf).all(Result::is_ok));
    assert_eq!(so.len(), x.len());
    assert!(so.iter().all(|v| v.is_finite()));
    assert!(on.blocks_skipped >= 16, "silence was skipped ({} blocks)", on.blocks_skipped);
    assert_eq!(on.blocks_converted + on.blocks_skipped, (x.len() / half) as u64);
    assert!(log_on.lock().unwrap().dit.len() < log_off.lock().unwrap().dit.len());
    let quiet_from = 8_000usize.div_ceil(half) * half;
    let skip_start = (quiet_from + SKIP_AFTER_QUIET_16K).div_ceil(half) * half;
    assert_eq!(so[..skip_start - FADE_16K], sf[..skip_start - FADE_16K], "identical up to the skip");
    assert!(so[skip_start..skip_start + half].iter().all(|&v| v == 0.0), "silent while skipping");
    // The onset block (the loud part starts at 27 200) converts at once and stays close to the
    // stream that never skipped (the stand-in's output differs only by the ASR window grid).
    let resume = 27_200 / half * half;
    assert!(
        so[resume - half..resume].iter().all(|&v| v == 0.0),
        "still skipping before the onset block"
    );
    let tail = resume + FADE_16K;
    assert!(so[tail..tail + half].iter().all(|&v| v.abs() > 0.1), "the onset block is converted");
    let settled = resume + 4 * TICK_16K;
    let rms_diff = rms(&so[settled..]
        .iter()
        .zip(&sf[settled..])
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>());
    assert!(
        rms_diff < 0.1 * rms(&sf[settled..]),
        "after the resume the stream follows the input ({rms_diff})"
    );
}

#[test]
fn a_skipped_harness_tick_restarts_from_silence_with_fades() {
    for block in [TICK_16K / 2, TICK_16K] {
        discontinuity_case(block);
    }
}

fn discontinuity_case(block: usize) {
    let (mut model, _) = fake_model(block, 1, false);
    let want = block + 200;
    model.configure_stream(block, want, 0).unwrap();
    let x = signal(block * 6);
    let mut ring = vec![0.0f32; want];
    for (i, chunk) in x.chunks_exact(block).enumerate() {
        // Call 3 arrives two blocks late (the harness lost blocks): a discontinuity.
        let pos = if i >= 3 { (i + 2) * block } else { i * block };
        ring.drain(..block);
        ring.extend_from_slice(chunk);
        let ctx = ConversionCtx {
            mono_16k: &ring,
            new_frames: want,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: pos as u64,
            reference: None,
            singing: SingingFlags::SPEECH,
        };
        let out = model.convert(&ctx).unwrap();
        if i == 3 {
            // The old tail fades out to the cut and the new block fades in from it (the stand-in's
            // output sits around 0.5-0.9, never near zero on its own).
            let cut = want - block;
            assert!(out[cut - FADE_16K].abs() > 0.3 && out[cut + FADE_16K].abs() > 0.3, "block {block}");
            assert!(out[cut - 1].abs() < 0.02, "block {block}: faded out: {}", out[cut - 1]);
            assert!(out[cut].abs() < 0.02, "block {block}: fades in: {}", out[cut]);
            // The block was converted from the steady silence, not from the old timeline.
            assert_eq!(model.blocks_converted, 4);
        }
    }
}

#[test]
fn a_failed_call_keeps_the_timeline_and_the_next_call_recovers() {
    let (mut graphs, log) = FakeGraphs::new();
    // The prime's whole ticks make 2 DiT calls each, the cold first one 1 (1 step); fail the third
    // live call's first one.
    let prime_dits = 2 * (SILENCE_PRIME_16K / TICK_16K) - 1;
    graphs.fail_dit_at = Some(prime_dits + 4);
    let (mut model, _) = fake_model_with(graphs, log, TICK_16K, 1, false);
    let want = TICK_16K + 250;
    let x = signal(TICK_16K * 6);
    let (out, results) = drive(&mut model, &x, want, None);
    assert!(results[0].is_ok() && results[1].is_ok());
    assert!(results[2].as_ref().unwrap_err().contains("injected"));
    assert!(results[3..].iter().all(Result::is_ok), "the model recovers on the next call");
    assert_eq!(out.len(), x.len());
    assert!(out[3 * TICK_16K..].iter().all(|v| v.is_finite()));
}

#[test]
fn a_new_reference_of_this_family_switches_the_speaker_and_others_are_ignored() {
    let (mut model, log) = fake_model(TICK_16K, 1, false);
    let base_gtm = log.lock().unwrap().gtm;
    let x = signal(TICK_16K * 6);
    let want = TICK_16K + 100;
    let (a, _) = drive(&mut model, &x, want, None);
    let other = speaker::embedding("meanvc2-ecapa-0123456789ab", "bob", "cd", {
        let mut s = spks(0.0);
        s[0] -= 0.5;
        s
    });
    let (b, _) = drive(&mut model, &x, want, Some(&other));
    assert_eq!(log.lock().unwrap().gtm, base_gtm + 1, "the GTM ran once for the new speaker");
    // The stand-in's target follows spks[0]: -0.5 on the mel, -0.25 after Vocos's (mel + 1) / 2
    // (downward, so the output stays under the post stage's limiter ceiling).
    let steady = 4 * TICK_16K;
    let shift = b[steady] - a[steady];
    assert!((shift + 0.25).abs() < 1e-4, "speaker shift {shift}");
    // The same reference again: nothing to do. Another family's: refused once, speaker kept.
    let _ = drive(&mut model, &x[..2 * TICK_16K], want, Some(&other));
    let mut foreign = other.clone();
    foreign.backend = "sigil-zs-1".into();
    foreign.clip_sha256 = "ef".into();
    let _ = drive(&mut model, &x[..2 * TICK_16K], want, Some(&foreign));
    assert_eq!(log.lock().unwrap().gtm, base_gtm + 1);
}

#[test]
fn the_governor_trades_the_second_step_for_time_under_load() {
    let (mut model, _) = fake_model(TICK_16K, 2, false);
    assert_eq!(model.steps(), 2);
    assert!(!model.apply_quality(&QualityLevel::at(1)), "retrieval off: nothing to do here");
    assert!(model.apply_quality(&QualityLevel::at(2)));
    assert_eq!(model.steps(), 1);
    assert!(!model.apply_quality(&QualityLevel::at(3)), "already at one step");
    assert!(model.apply_quality(&QualityLevel::FULL));
    assert_eq!(model.steps(), 2);
    let (mut one, _) = fake_model(TICK_16K, 1, false);
    assert!(!one.apply_quality(&QualityLevel::at(4)), "a one-step voice has nothing to give up");
}

#[test]
fn warm_up_leaves_the_stream_at_steady_silence() {
    let (mut model, _) = fake_model(TICK_16K, 2, false);
    let tone = signal(TICK_16K + 300);
    model
        .warmup_at(&ConversionCtx {
            mono_16k: &tone,
            new_frames: tone.len(),
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: 0,
            reference: None,
            singing: SingingFlags::SPEECH,
        })
        .unwrap();
    // Only the noise index moved on.
    assert_eq!(model.stream().state().without_frame(), model.silence_state().without_frame());
    assert!(model.stream().state().frame() > model.silence_state().frame());
    assert!(model.configure_stream(2 * TICK_16K, 3000, 0).is_err(), "the block is the voice's");
    assert_eq!(model.configure_stream(TICK_16K, 1500, 999).unwrap(), 1500);
}

#[test]
fn quiet_blocks_are_detected_per_10_ms_frame() {
    let mut x = vec![0.0f32; TICK_16K];
    assert!(!block_is_loud(&x, -42.0));
    // One 10 ms frame at -30 dBFS RMS is enough.
    let a = 10f32.powf(-30.0 / 20.0) * std::f32::consts::SQRT_2;
    for (i, v) in x[640..800].iter_mut().enumerate() {
        *v = a * (std::f32::consts::TAU * i as f32 / 32.0).sin();
    }
    assert!(block_is_loud(&x, -42.0));
    assert!(!block_is_loud(&x, -20.0));
    assert!(block_is_loud(&[0.0; 160], -150.0), "a threshold under -120 dB never gates");
    let (mut model, _) = fake_model(TICK_16K, 1, true);
    model.set_voice_controls(&VoiceControls {
        silence_db: f32::NAN,
        ..VoiceControls::default()
    });
    model.set_voice_controls(&VoiceControls {
        silence_db: -60.0,
        ..VoiceControls::default()
    });
}

fn fake_model_aligned(block: usize, steps: u32, silence_skip: bool, align_16k: usize) -> MeanVcModel {
    let (graphs, _) = FakeGraphs::new();
    let stream = MeanVcStream::new(Box::new(graphs), &spks(0.0), steps, 7).unwrap();
    MeanVcModel::new(
        stream,
        &ModelConfig {
            block_16k: block,
            steps,
            silence_skip,
            align_16k,
        },
    )
    .unwrap()
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

#[test]
fn rms_mix_follows_the_input_delayed_by_the_models_content_delay() {
    // The input drops by 26 dB at S; the stand-in's own output level barely moves (it sits on a
    // constant), so rms_mix 0 must carry the drop, at S + D (the content delay), not at S.
    let d = 3_200;
    let s = 15 * TICK_16K;
    let x: Vec<f32> = signal(TICK_16K * 30)
        .iter()
        .enumerate()
        .map(|(i, &v)| if i < s { v } else { v * 0.05 })
        .collect();
    let want = TICK_16K + 200;
    let mut own = fake_model_aligned(TICK_16K, 1, false, d);
    let mut mixed = fake_model_aligned(TICK_16K, 1, false, d);
    mixed.set_voice_controls(&VoiceControls {
        rms_mix: 0.0,
        ..VoiceControls::default()
    });
    let (oa, ra) = drive(&mut own, &x, want, None);
    let (ob, rb) = drive(&mut mixed, &x, want, None);
    assert!(ra.iter().chain(&rb).all(Result::is_ok));
    let loud = rms(&ob[s + d - 1_600..s + d - 800]);
    let still_loud = rms(&ob[s + 800..s + 1_600]);
    let quiet = rms(&ob[s + d + 800..s + d + 1_600]);
    assert!(loud / quiet > 8.0, "rms_mix 0 follows the input's drop ({loud} -> {quiet})");
    assert!(
        still_loud / quiet > 8.0,
        "the drop arrives with the content delay, not before ({still_loud} vs {quiet})"
    );
    let own_ratio = rms(&oa[s + d - 1_600..s + d - 800]) / rms(&oa[s + d + 800..s + d + 1_600]);
    assert!(own_ratio < 1.5, "rms_mix 1 keeps the model's own level ({own_ratio})");
    // Below the limiter's ceiling the default controls change nothing (bit for bit).
    let (mut plain, _) = fake_model(TICK_16K, 1, false);
    let (op, _) = drive(&mut plain, &x, want, None);
    assert_eq!(op, oa);
}

#[test]
fn the_limiter_and_the_formant_shift_keep_the_output_bounded_and_finite() {
    use crate::services::neural_audio::vc::engine::post::LIMITER_CEILING_DB;
    let ceiling = 10f32.powf(LIMITER_CEILING_DB / 20.0);
    // A loud input with rms_mix 0 asks for more than full scale; the formant shift is on.
    let x: Vec<f32> = (0..TICK_16K * 20)
        .map(|i| 0.99 * (std::f32::consts::TAU * 220.0 * i as f32 / 16_000.0).sin())
        .collect();
    let mut model = fake_model_aligned(TICK_16K, 1, false, MODEL_DELAY_16K);
    model.set_voice_controls(&VoiceControls {
        rms_mix: 0.0,
        formant_semitones: 3.0,
        ..VoiceControls::default()
    });
    let (out, results) = drive(&mut model, &x, TICK_16K + 160, None);
    assert!(results.iter().all(Result::is_ok));
    assert!(out.iter().all(|v| v.is_finite()));
    let peak = out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(peak <= ceiling + 1e-6, "limited to {ceiling}: {peak}");
    assert!(peak > 0.5 * ceiling, "the loud input came through ({peak})");
    // Silence through the skip with the formant shift and rms_mix on stays finite and ends silent.
    let mut model = fake_model_aligned(TICK_16K, 2, true, MODEL_DELAY_16K);
    model.set_voice_controls(&VoiceControls {
        rms_mix: 0.5,
        formant_semitones: -6.0,
        ..VoiceControls::default()
    });
    let (out, results) = drive(&mut model, &vec![0.0; TICK_16K * 25], TICK_16K + 160, None);
    assert!(results.iter().all(Result::is_ok));
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(out[out.len() - TICK_16K..].iter().all(|&v| v == 0.0), "skipping: silent");
}

// ---------------------------------------------------------------------------------------------
// ONNX Runtime
// ---------------------------------------------------------------------------------------------

fn ort_opts() -> VcSessionOptions {
    VcSessionOptions {
        streamer_profile: true,
        intra_threads: Some(1),
        ..VcSessionOptions::default()
    }
}

#[test]
fn the_onnx_contract_runs_on_onnx_runtime_and_matches_the_reference_loop() {
    let models = stand_in_models();
    let (graphs, outcomes) = OrtGraphs::load(&GraphSource::Memory(&models), &ort_opts()).unwrap();
    assert_eq!(graphs.shapes(), &GraphShapes::meanvc2_40ms());
    let components: Vec<&str> = outcomes.iter().map(|(c, _)| c.as_str()).collect();
    assert_eq!(components, ["vc.fbank", "vc.content", "vc.timbre", "vc", "vc.vocoder"]);
    assert!(outcomes.iter().all(|(_, o)| o.active == "cpu"));
    for steps in [1u32, 2] {
        let stream = MeanVcStream::new(
            Box::new(
                OrtGraphs::load(&GraphSource::Memory(&models), &ort_opts())
                    .unwrap()
                    .0,
            ),
            &spks(0.3),
            steps,
            11,
        )
        .unwrap();
        let mut ort = MeanVcModel::new(
            stream,
            &ModelConfig {
                block_16k: TICK_16K,
                steps,
                silence_skip: false,
                align_16k: 0,
            },
        )
        .unwrap();
        let (fake, _) = FakeGraphs::new();
        let stream = MeanVcStream::new(Box::new(fake), &spks(0.3), steps, 11).unwrap();
        let mut rust = MeanVcModel::new(
            stream,
            &ModelConfig {
                block_16k: TICK_16K,
                steps,
                silence_skip: false,
                align_16k: 0,
            },
        )
        .unwrap();
        let x = signal(TICK_16K * 12);
        let (a, ra) = drive(&mut ort, &x, TICK_16K + 64, None);
        let (b, rb) = drive(&mut rust, &x, TICK_16K + 64, None);
        assert!(ra.iter().chain(&rb).all(Result::is_ok));
        let diff = max_abs_diff(&a, &b);
        assert!(diff < 1e-4, "{steps} steps: ONNX Runtime and the reference loop differ by {diff}");
    }
}

#[test]
fn graphs_that_break_the_contract_are_refused_by_name() {
    let mut models = stand_in_models();
    // vocos where asr belongs.
    models[1] = vocos_model();
    let err = OrtGraphs::load(&GraphSource::Memory(&models), &ort_opts())
        .err()
        .unwrap();
    assert!(err.contains("asr"), "{err}");
    // A DiT with a dynamic cache shape.
    let mut g = Graph::new();
    g.input("x", FLOAT, &[Dim::Fixed(1), Dim::Fixed(8), Dim::Fixed(80)])
        .output("y", FLOAT, &[Dim::Fixed(1), Dim::Fixed(8), Dim::Fixed(80)])
        .node("Identity", &["x"], &["y"], &[]);
    let mut models = stand_in_models();
    models[3] = g.model_bytes(OPSET);
    let err = OrtGraphs::load(&GraphSource::Memory(&models), &ort_opts())
        .err()
        .unwrap();
    assert!(err.contains("dit_step"), "{err}");
}

/// A graph declaring an absurd or misshapen state is refused by name before anything is allocated
/// from its shape (a `vec![0.0; 4.8e13]` would abort the process, not just fail the load).
#[test]
fn state_shapes_are_bounded_before_anything_is_allocated() {
    for (cache, why) in [
        (vec![4u64, 2, 2, 12, 1_000_000_000_000], "48 trillion values"),
        (vec![4, 2, 2, 12, 1 << 40], "4.2e14 values"),
        (vec![2048, 2, 2, 12, 64], "6.3 M values, over the cap"),
        (vec![4, 2, 12, 64], "rank 4: the loop reads the frame count at index 3 of 5"),
    ] {
        let mut models = stand_in_models();
        models[3] = dit_model_with_cache(&cache);
        let err = OrtGraphs::load(&GraphSource::Memory(&models), &ort_opts())
            .err()
            .unwrap_or_else(|| panic!("{why}: loaded"));
        assert!(err.contains("dit_step") && err.contains("cache"), "{why}: {err}");
    }
    // The bound itself: the 40 ms export is far below it, a product that overflows is caught.
    assert!(GraphShapes::meanvc2_40ms().validate().is_ok());
    assert_eq!(GraphShapes::len(&[usize::MAX, 2]), usize::MAX);
    assert_eq!(GraphShapes::len(&[4, 2, 2, 12, 64]), 12_288);
    // The loop checks any graph set (a stand-in included) before it allocates its state.
    type Break = fn(&mut GraphShapes);
    let cases: [(Break, &str); 5] = [
        (|s| s.dit_cache = vec![4, 2, 2, 12, usize::MAX / 2], "dit_step cache"),
        (|s| s.att_cache = vec![6, 4, 4, 128, 1], "att_cache"),
        (|s| s.cnn_cache = vec![6, 1, 256, 0], "cnn_cache"),
        (|s| s.memory = vec![1, 32], "k_mem"),
        (|s| s.memory = vec![1, 1 << 12, 1 << 12], "k_mem"),
    ];
    for (break_it, name) in cases {
        let (mut fake, _) = FakeGraphs::new();
        break_it(&mut fake.shapes);
        let err = MeanVcStream::new(Box::new(fake), &spks(0.0), 1, 1)
            .err()
            .unwrap_or_else(|| panic!("{name}: accepted"));
        assert!(err.contains(name), "{name}: {err}");
    }
}

fn family_request(slot: &Path, reference: Option<ReferenceClip>, options: serde_json::Value) -> FamilyRequest {
    let manifest = crate::services::neural_audio::vc::families::family_manifest(FamilyKind::MeanVc, "Mean", options);
    crate::services::neural_audio::vc::families::write_family_manifest(slot, &manifest).unwrap();
    FamilyRequest {
        kind: FamilyKind::MeanVc,
        slot_dir: slot.to_path_buf(),
        manifest,
        app_data: None,
        ep: "cuda".into(),
        session: VcSessionOptions::default(),
        levers: {
            let mut l = EffectiveLevers::default();
            l.levers.set(Lever::IoBinding, true);
            l.levers.set(Lever::Governor, true);
            l
        },
        precision: PrecisionMode::Auto,
        preset: crate::services::neural_audio::RvcPresetProfile::Streamer,
        block_ms: 0.0,
        chunk_frames: 480,
        device_rate: 48_000,
        extra_context_ms: 0.0,
        lookahead_ms: 0.0,
        f0_range: F0Range::Speech,
        silence_skip: true,
        noise_seed: 3,
        pitch: None,
        pitch_method: String::new(),
        content_encoder: None,
        reference,
        latency_budget_ms: 300.0,
        cancel: None,
    }
}

fn reference_wav(dir: &Path, seconds: f32) -> ReferenceClip {
    let path = dir.join("ref.wav");
    let x: Vec<f32> = signal((22_050.0 * seconds) as usize)
        .iter()
        .map(|v| v * 0.5)
        .collect();
    crate::services::bench::signals::write_wav(&path, &x, 22_050, 1).unwrap();
    ReferenceClip::from_wav(&path).unwrap()
}

#[test]
fn a_voice_folder_with_its_own_graphs_and_speaker_npy_loads_through_the_family() {
    let slot = stand_in_slot("npy");
    let spk = spks(0.9);
    std::fs::write(slot.join(speaker::SPEAKER_NPY), super::npy::encode(&spk, "(256,)", "<f4", 1)).unwrap();
    let manifest =
        crate::services::neural_audio::vc::families::family_manifest(FamilyKind::MeanVc, "Mean", serde_json::json!({"steps": 1, "tick_ms": 160}));
    check_slot(&slot, &manifest).unwrap();
    let clip = reference_wav(&slot, 2.0);
    let engine = create(family_request(&slot, Some(clip), serde_json::json!({"steps": 1, "tick_ms": 160}))).unwrap();
    assert_eq!(engine.model.fixed_block(), Some(2 * TICK_16K), "tick_ms 160");
    assert_eq!(engine.model_delay_16k, MODEL_DELAY_16K);
    let reference = engine.reference.as_ref().unwrap();
    assert_eq!(reference.backend, speaker::NPY_BACKEND);
    assert_eq!(speaker::spks_of(reference).unwrap(), spk);
    assert!(engine.warnings.iter().any(|w| w.contains("speaker.npy")), "{:?}", engine.warnings);
    // The rung is CUDA, the voice runs on the CPU; the RVC levers it does not use are noted.
    assert!(engine.outcomes.iter().all(|(_, o)| o.active == "cpu"));
    assert!(engine.levers.levers.get(Lever::AnchoredNoise) && engine.levers.levers.get(Lever::Governor));
    assert!(!engine.levers.levers.get(Lever::IoBinding));
    assert!(engine.levers.notes.iter().any(|n| n.contains("io_binding")));
    assert!(engine
        .levers
        .notes
        .iter()
        .any(|n| n.contains("CPU on every execution provider")));
    // Through the production harness, with the family's reference as the context.
    let context = crate::services::neural_audio::vc::stream::HarnessContext {
        reference: engine.reference.clone(),
        singing: SingingFlags::SPEECH,
    };
    let mut harness = VcStreamHarness::with_context(engine.model, 48_000, 48_000, 2, 2, 0, 320, 960, context).unwrap();
    let x: Vec<f32> = signal(48_000 * 2).iter().flat_map(|&v| [v, v]).collect();
    for chunk in x.as_chunks::<{ 960 * 2 }>().0 {
        let out = harness.push_and_convert(chunk).unwrap();
        assert_eq!(out.len(), harness.last_owed_frames() * 2);
    }
    let _ = std::fs::remove_dir_all(&slot);
}

#[test]
fn a_reference_clip_runs_through_the_speaker_encoder_graph() {
    let slot = stand_in_slot("enc");
    std::fs::write(slot.join(speaker::SPEAKER_ONNX), speaker::tests::stand_in_encoder()).unwrap();
    let clip = reference_wav(&slot, 1.5);
    let engine = create(family_request(&slot, Some(clip.clone()), serde_json::json!({}))).unwrap();
    let reference = engine.reference.as_ref().unwrap();
    let backend = speaker::encoder_backend(&slot.join(speaker::SPEAKER_ONNX)).unwrap();
    assert!(backend.starts_with(speaker::ENCODER_BACKEND_PREFIX), "{backend}");
    assert_eq!(
        (reference.backend.as_str(), reference.clip_sha256.as_str()),
        (backend.as_str(), clip.sha256.as_str())
    );
    let s = speaker::spks_of(reference).unwrap();
    // emb[i] = mean(clip) (i + 1) + 0.01 (i + 1): the ratio of two entries is their index ratio.
    assert!((s[9] / s[0] - 10.0).abs() < 1e-3, "{} {}", s[0], s[9]);
    assert_eq!(engine.model.fixed_block(), Some(TICK_16K / 2), "default tick 80 ms, as two 40 ms blocks");
    assert_eq!(engine.model_delay_16k, MODEL_DELAY_16K + TICK_16K / 2, "the deferred half of the tick");
    // Without a reference (and no speaker.npy) the voice cannot load.
    let err = create(family_request(&slot, None, serde_json::json!({})))
        .err()
        .unwrap();
    assert!(err.contains("reference voice"), "{err}");
    // Without the encoder the reference cannot be analysed.
    std::fs::remove_file(slot.join(speaker::SPEAKER_ONNX)).unwrap();
    if crate::services::model_store::installed_path(SPEAKER_ID).is_none() {
        let err = create(family_request(&slot, Some(clip), serde_json::json!({})))
            .err()
            .unwrap();
        assert!(err.contains(SPEAKER_ID), "{err}");
    }
    let _ = std::fs::remove_dir_all(&slot);
}

/// A library voice `id` under `app_data` (its `ref.wav`, which the cache needs) as a clip.
fn library_clip(app_data: &Path, id: &str, seconds: f32) -> ReferenceClip {
    let dir = crate::services::voice_library::voice_dir(app_data, id).unwrap();
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(crate::services::voice_library::CLIP_FILE);
    let x: Vec<f32> = signal((22_050.0 * seconds) as usize)
        .iter()
        .map(|v| v * 0.5)
        .collect();
    crate::services::bench::signals::write_wav(&path, &x, 22_050, 1).unwrap();
    let mut clip = ReferenceClip::from_wav(&path).unwrap();
    clip.voice_id = id.into();
    clip.name = "Alice".into();
    clip
}

/// The library caches one embedding per encoder: a re-imported encoder (other weights, same size)
/// computes its own instead of loading the first one's, a voice folder's own encoder never shares
/// the store's, and with no encoder at all the newest cached embedding of the clip still loads.
#[test]
fn cached_embeddings_are_keyed_by_the_encoder_that_made_them() {
    let slot = stand_in_slot("cache-key");
    let app_data = speaker::tests::test_dir("cache-key-app");
    let clip = library_clip(&app_data, "alice", 1.5);
    let request = |clip: &ReferenceClip| {
        let mut r = family_request(&slot, Some(clip.clone()), serde_json::json!({}));
        r.app_data = Some(app_data.clone());
        r
    };
    let encoder = slot.join(speaker::SPEAKER_ONNX);
    let spks_now = |engine: &FamilyEngine| speaker::spks_of(engine.reference.as_ref().unwrap()).unwrap();
    // First encoder: computed and cached under its fingerprint.
    std::fs::write(&encoder, speaker::tests::stand_in_encoder_with_bias(0.01)).unwrap();
    let first = create(request(&clip)).unwrap();
    let backend_a = speaker::encoder_backend(&encoder).unwrap();
    assert_eq!(first.reference.as_ref().unwrap().backend, backend_a);
    let cached_a = crate::services::voice_library::read_cache(&app_data, "alice", &backend_a).expect("cached under the encoder's key");
    assert_eq!(ReferenceEmbedding::from_bytes(&cached_a).unwrap(), **first.reference.as_ref().unwrap());
    // The next load with the same encoder reads that cache entry (marked here so a recomputation
    // would show).
    let mut marked = (**first.reference.as_ref().unwrap()).clone();
    marked.tensors[0].data[0] += 1.0;
    crate::services::voice_library::write_cache(&app_data, "alice", &backend_a, &marked.to_bytes().unwrap()).unwrap();
    let again = create(request(&clip)).unwrap();
    assert_eq!(spks_now(&again), speaker::spks_of(&marked).unwrap(), "read from cache/{backend_a}.bin");
    let a = spks_now(&first);
    // A fixed re-export replaces the encoder: a new key, a new embedding, never the stale one.
    std::fs::write(&encoder, speaker::tests::stand_in_encoder_with_bias(0.05)).unwrap();
    let backend_b = speaker::encoder_backend(&encoder).unwrap();
    assert_ne!(backend_b, backend_a);
    let second = create(request(&clip)).unwrap();
    let b = spks_now(&second);
    assert_eq!(second.reference.as_ref().unwrap().backend, backend_b);
    assert!((b[0] - a[0] - 0.04).abs() < 1e-5, "the new encoder's bias: {} vs {}", b[0], a[0]);
    assert!(crate::services::voice_library::read_cache(&app_data, "alice", &backend_b).is_some());
    // No encoder left (none in the store either): the newest cached embedding of this clip loads,
    // with a warning naming the missing encoder.
    std::fs::remove_file(&encoder).unwrap();
    if crate::services::model_store::installed_path(SPEAKER_ID).is_none() {
        let fallback = create(request(&clip)).unwrap();
        assert_eq!(spks_now(&fallback), b, "the newest cache (the second encoder's)");
        assert!(fallback.warnings.iter().any(|w| w.contains(SPEAKER_ID) && w.contains("Alice")), "{:?}", fallback.warnings);
        // Another clip of the voice (re-recorded) has no cache: the load names the encoder.
        let rerecorded = library_clip(&app_data, "alice", 2.0);
        assert_ne!(rerecorded.sha256, clip.sha256);
        let err = create(request(&rerecorded)).err().unwrap();
        assert!(err.contains(SPEAKER_ID), "{err}");
        // An ad-hoc clip is never cached.
        assert!(speaker::latest_cached(&app_data, &reference_wav(&slot, 1.5)).is_none());
    }
    let _ = std::fs::remove_dir_all(&slot);
    let _ = std::fs::remove_dir_all(&app_data);
}

/// The 160 ms tick computes each tick in one burst and asks for the Stable output buffer; the
/// default split 80 ms tick keeps the user's (P5-C bench).
#[test]
fn the_whole_160_ms_tick_asks_for_the_stable_output_buffer() {
    let manifest = |options: serde_json::Value| crate::services::neural_audio::vc::families::family_manifest(FamilyKind::MeanVc, "Mean", options);
    assert_eq!(min_output_buffer_ms(&manifest(serde_json::json!({}))), None, "default: 80 ms as two 40 ms blocks");
    assert_eq!(min_output_buffer_ms(&manifest(serde_json::json!({"steps": 1, "tick_ms": 80}))), None);
    assert_eq!(min_output_buffer_ms(&manifest(serde_json::json!({"tick_ms": 160}))), Some((80.0, 150.0)));
    assert_eq!(min_output_buffer_ms(&manifest(serde_json::json!({"steps": 1, "tick_ms": 400}))), Some((80.0, 150.0)), "normalized to 160");
    assert_eq!(min_output_buffer_ms(&manifest(serde_json::json!({"steps": "two"}))), None, "refused at load instead");
    // Every block size the options can produce: one DiT call per call needs no more than the
    // default buffer; a burst of several needs the minimum, which is a valid voice-changer buffer
    // (prefill 35-300 ms, queue at least prefill + 5 ms; `families` checks its own the same way).
    for tick_ms in [80u32, 160] {
        for steps in [1u32, 2] {
            let o = MeanVcOptions { steps, tick_ms };
            match o.min_output_buffer_ms() {
                None => assert_eq!(o.block_16k(), super::stream::DIT_HOP_16K, "{o:?}"),
                Some((prefill, queue)) => {
                    assert!(o.block_16k() > super::stream::DIT_HOP_16K, "{o:?}");
                    assert!((35.0..=300.0).contains(&prefill) && queue >= prefill + 5.0 && queue <= 600.0, "{o:?}");
                }
            }
        }
    }
}

#[test]
fn slots_without_graphs_report_the_store_entry() {
    let dir = speaker::tests::test_dir("empty");
    let manifest = crate::services::neural_audio::vc::families::family_manifest(FamilyKind::MeanVc, "Mean", serde_json::json!({}));
    if crate::services::model_store::installed_path(CATALOG_ID).is_none() {
        assert!(check_slot(&dir, &manifest)
            .unwrap_err()
            .contains(CATALOG_ID));
    }
    let bad = crate::services::neural_audio::vc::families::family_manifest(FamilyKind::MeanVc, "Mean", serde_json::json!({"steps": "two"}));
    assert!(check_slot(&dir, &bad).unwrap_err().contains("options"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn options_normalize_to_what_the_engine_runs() {
    let n = |v: serde_json::Value| normalize_options(&v).unwrap();
    assert_eq!(n(serde_json::json!({})), serde_json::json!({"steps": 2, "tick_ms": 80}));
    assert_eq!(
        n(serde_json::json!({"steps": 0, "tick_ms": 100})),
        serde_json::json!({"steps": 1, "tick_ms": 80})
    );
    assert_eq!(
        n(serde_json::json!({"steps": 9, "tick_ms": 160})),
        serde_json::json!({"steps": 2, "tick_ms": 160})
    );
    assert_eq!(
        n(serde_json::json!({"tick_ms": 400, "extra": true})),
        serde_json::json!({"steps": 2, "tick_ms": 160})
    );
    assert!(normalize_options(&serde_json::json!({"steps": -1})).is_err());
    let info = info();
    assert!(available() && info.zero_shot && info.experimental && !info.needs_pitch && !info.needs_content_encoder);
    // The speaker encoder is optional (speaker.npy, a cached embedding): only the graphs are
    // required, so the requirement banner clears on voices that work without it.
    assert_eq!(info.requires, [CATALOG_ID]);
    assert!(info.execution_providers.contains(&"cpu"));
    assert_eq!(
        info.controls,
        ["formant", "rms_mix", "silence_db"],
        "what the model answers (post stage + silence skip)"
    );
}

// ---------------------------------------------------------------------------------------------
// Real graphs (Sigil's export; #[ignore])
// ---------------------------------------------------------------------------------------------

/// The exported graphs: `SIGIL_TEST_MODELS_DIR/meanvc/`, else the dev cache's.
fn real_graphs_dir() -> Option<PathBuf> {
    let candidates = [
        std::env::var_os("SIGIL_TEST_MODELS_DIR").map(|d| PathBuf::from(d).join("meanvc")),
        std::env::var_os("LOCALAPPDATA").map(|d| {
            PathBuf::from(d)
                .join("sigil-dev-cache")
                .join("vc-families")
                .join("meanvc")
        }),
    ];
    candidates
        .into_iter()
        .flatten()
        .find(|d| super::graphs::missing_files(d).is_empty())
}

/// A speech sample (16 kHz mono): `SIGIL_TEST_MODELS_DIR/samples/<name>`, else the dev cache's.
fn real_speech(name: &str) -> Option<Vec<f32>> {
    let candidates = [
        std::env::var_os("SIGIL_TEST_MODELS_DIR").map(|d| PathBuf::from(d).join("samples").join(name)),
        std::env::var_os("LOCALAPPDATA").map(|d| {
            PathBuf::from(d)
                .join("sigil-dev-cache")
                .join("models")
                .join("transcribe")
                .join("samples")
                .join(name)
        }),
    ];
    let path = candidates.into_iter().flatten().find(|p| p.is_file())?;
    let (data, rate, ch) = crate::services::bench::signals::read_wav(&path).ok()?;
    let mono: Vec<f32> = data
        .chunks_exact(ch.max(1))
        .map(|f| f.iter().sum::<f32>() / f.len() as f32)
        .collect();
    crate::services::bench::signals::resample(&mono, rate, 16_000).ok()
}

fn unit_speaker() -> Vec<f32> {
    let s = spks(0.0);
    let n = s.iter().map(|v| v * v).sum::<f32>().sqrt();
    s.iter().map(|v| v / n).collect()
}

/// Times every `convert` of the wrapped model (the model's own compute, without the harness).
struct Timed {
    inner: MeanVcModel,
    calls: Arc<Mutex<Vec<f64>>>,
}

impl VoiceConversionModel for Timed {
    fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        let t = std::time::Instant::now();
        let out = self.inner.convert(ctx);
        self.calls
            .lock()
            .unwrap()
            .push(t.elapsed().as_secs_f64() * 1000.0);
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
        self.inner.set_voice_controls(controls)
    }
    fn fixed_block(&self) -> Option<usize> {
        self.inner.fixed_block()
    }
    fn configure_stream(&mut self, block: usize, want: usize, extra: usize) -> Result<usize, String> {
        self.inner.configure_stream(block, want, extra)
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

#[test]
#[ignore = "needs Sigil's MeanVC2 export (SIGIL_TEST_MODELS_DIR/meanvc or the dev cache)"]
fn real_model_meanvc2_runs_in_real_time_on_one_cpu_thread() {
    let Some(dir) = real_graphs_dir() else {
        eprintln!("skipped: no MeanVC2 graphs");
        return;
    };
    let speech = real_speech("jfk.wav").unwrap_or_else(|| signal(16_000 * 11));
    let mut input = speech.clone();
    while input.len() < 16_000 * 20 {
        input.extend_from_slice(&speech);
    }
    let device = crate::services::bench::signals::resample(&input, 16_000, 48_000).unwrap();
    let half = TICK_16K / 2;
    // The shipping shapes (the 80 ms tick as 40 ms blocks, the 160 ms tick whole) and the whole 80 ms tick.
    for (steps, block, skip) in [
        (2u32, half, false),
        (1, half, false),
        (2, TICK_16K, false),
        (2, 2 * TICK_16K, false),
        (2, half, true),
    ] {
        let t_load = std::time::Instant::now();
        let (graphs, _) = OrtGraphs::load(&GraphSource::Dir(&dir), &ort_opts()).unwrap();
        let stream = MeanVcStream::new(Box::new(graphs), &unit_speaker(), steps, 1).unwrap();
        let model = MeanVcModel::new(
            stream,
            &ModelConfig {
                block_16k: block,
                steps,
                silence_skip: skip,
                align_16k: MODEL_DELAY_16K + TICK_16K.saturating_sub(block),
            },
        )
        .unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        // 48 kHz device in 10 ms blocks, the voice changer's default shape.
        let timed = Timed {
            inner: model,
            calls: calls.clone(),
        };
        let mut harness = VcStreamHarness::new(Box::new(timed), 48_000, 48_000, 1, 1, 0, 480, 480).unwrap();
        let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
        calls.lock().unwrap().clear();
        let mut out = Vec::new();
        let t_run = std::time::Instant::now();
        for chunk in device.as_chunks::<480>().0 {
            out.extend(harness.push_and_convert(chunk).unwrap());
        }
        let wall = t_run.elapsed().as_secs_f64() * 1000.0;
        let mut model_ms = calls.lock().unwrap().clone();
        let model_total: f64 = model_ms.iter().sum();
        model_ms.sort_by(f64::total_cmp);
        let seconds = input.len() as f64 / 16_000.0;
        let block_ms = block as f64 / 16.0;
        eprintln!(
            "MeanVC2 {steps} step(s), {block_ms} ms block{}: model per block p50 {:.2} p95 {:.2} p99 {:.2} max {:.2} ms (RTF p95 {:.3}); model RTF {:.3}, with the harness {:.3}; load {load_ms:.0} ms; {seconds:.1} s, {} blocks",
            if skip { ", silence skip" } else { "" },
            percentile(&model_ms, 0.5),
            percentile(&model_ms, 0.95),
            percentile(&model_ms, 0.99),
            percentile(&model_ms, 1.0),
            percentile(&model_ms, 0.95) / block_ms,
            model_total / (seconds * 1000.0),
            wall / (seconds * 1000.0),
            model_ms.len()
        );
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().map(|v| v * v).sum::<f32>() > 1.0, "audible output");
        assert!(percentile(&model_ms, 0.99) < block_ms, "MeanVC2 must run in real time on one CPU thread");
    }
    // Silence with the skip on: after the hangover nothing runs.
    let (graphs, _) = OrtGraphs::load(&GraphSource::Dir(&dir), &ort_opts()).unwrap();
    let stream = MeanVcStream::new(Box::new(graphs), &unit_speaker(), 2, 1).unwrap();
    let model = MeanVcModel::new(
        stream,
        &ModelConfig {
            block_16k: half,
            steps: 2,
            silence_skip: true,
            align_16k: MODEL_DELAY_16K + half,
        },
    )
    .unwrap();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut harness = VcStreamHarness::new(
        Box::new(Timed {
            inner: model,
            calls: calls.clone(),
        }),
        48_000,
        48_000,
        1,
        1,
        0,
        480,
        480,
    )
    .unwrap();
    calls.lock().unwrap().clear();
    for chunk in vec![0.0f32; 48_000 * 10].chunks_exact(480) {
        harness.push_and_convert(chunk).unwrap();
    }
    let silent: f64 = calls.lock().unwrap().iter().sum();
    eprintln!(
        "MeanVC2 silence skip: {silent:.1} ms of model time over 10 s of silence ({:.4} cores)",
        silent / 10_000.0
    );
    assert!(silent / 10_000.0 < 0.02, "T4: a silent user costs at most 0.02 cores");
}

/// Parity with the Python reference loop (`scripts/vc-families/meanvc/rust_reference.py --out <dir>`:
/// sampler.py's `SigilMeanVC` with the 4k + 1 generalisation, the phase lead and Sigil's anchored
/// noise, whole 80 ms ticks, from cold). `SIGIL_MEANVC_REFERENCE_DIR` holds its `input.f32`,
/// `spks.npy`, `reference.f32` (960 zeros, then the loop's output) and `meta.json` (`steps`,
/// `noise_seed`). P5-C: 85.3 dB on 4 s of `jfk.wav` (gate 35 dB).
#[test]
#[ignore = "needs Sigil's MeanVC2 export and SIGIL_MEANVC_REFERENCE_DIR (the Python reference loop's output)"]
fn real_model_matches_the_python_reference_loop() {
    let (Some(dir), Some(ref_dir)) = (real_graphs_dir(), std::env::var_os("SIGIL_MEANVC_REFERENCE_DIR").map(PathBuf::from)) else {
        eprintln!("skipped: no MeanVC2 graphs or no SIGIL_MEANVC_REFERENCE_DIR");
        return;
    };
    let read_f32 = |p: &Path| -> Vec<f32> {
        std::fs::read(p)
            .unwrap()
            .as_chunks::<4>().0.iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let input = read_f32(&ref_dir.join("input.f32"));
    let reference = read_f32(&ref_dir.join("reference.f32"));
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(ref_dir.join("meta.json")).unwrap()).unwrap();
    let steps = meta["steps"].as_u64().unwrap() as u32;
    let seed = meta["noise_seed"].as_u64().unwrap();
    let spk = super::npy::read_vector(&ref_dir.join("spks.npy"), SPK_DIM).unwrap();
    let (graphs, _) = OrtGraphs::load(&GraphSource::Dir(&dir), &ort_opts()).unwrap();
    let mut stream = MeanVcStream::new(Box::new(graphs), &spk, steps, seed).unwrap();
    stream.reset_cold();
    let mut out = vec![0.0f32; 960];
    for chunk in input.as_chunks::<TICK_16K>().0 {
        stream.process(chunk, &mut out).unwrap();
    }
    assert_eq!(out.len(), reference.len());
    let signal: f64 = reference.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let noise: f64 = out
        .iter()
        .zip(&reference)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    let snr = 10.0 * (signal / noise.max(1e-30)).log10();
    eprintln!("MeanVC2 Rust loop vs the Python reference: SNR {snr:.1} dB over {} samples", out.len());
    assert!(snr >= 35.0, "parity gate (ADR 0006): {snr:.1} dB");
}

#[test]
#[ignore = "needs Sigil's MeanVC2 export (SIGIL_TEST_MODELS_DIR/meanvc or the dev cache)"]
fn real_model_graph_costs() {
    let Some(dir) = real_graphs_dir() else {
        eprintln!("skipped: no MeanVC2 graphs");
        return;
    };
    let (mut g, _) = OrtGraphs::load(&GraphSource::Dir(&dir), &ort_opts()).unwrap();
    let shapes = g.shapes().clone();
    let time = |label: &str, f: &mut dyn FnMut()| {
        for _ in 0..10 {
            f();
        }
        let mut v: Vec<f64> = (0..100)
            .map(|_| {
                let t = std::time::Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        v.sort_by(f64::total_cmp);
        eprintln!("{label}: median {:.3} ms p90 {:.3} ms", percentile(&v, 0.5), percentile(&v, 0.9));
    };
    let wav = signal(1520);
    let mut fb = Vec::new();
    time("fbank 1520", &mut || {
        fb.clear();
        g.fbank(&wav, &mut fb).unwrap();
    });
    let window = vec![0.1f32; ASR_WINDOW * MEL_BINS];
    let mut att = vec![0.0; GraphShapes::len(&shapes.att_cache)];
    let mut cnn = vec![0.0; GraphShapes::len(&shapes.cnn_cache)];
    let mut bn = vec![0.0; 2 * BN_DIM];
    time("asr", &mut || g.asr(&window, 8, &mut att, &mut cnn, &mut bn).unwrap());
    let mem = GraphShapes::len(&shapes.memory);
    let speaker = SpeakerState {
        spks: unit_speaker(),
        k_mem: vec![0.01; mem],
        v_mem: vec![0.01; mem],
    };
    let x = vec![0.1f32; DIT_FRAMES * MEL_BINS];
    let cond = vec![0.1f32; DIT_FRAMES * BN_DIM];
    let cache = vec![0.0f32; GraphShapes::len(&shapes.dit_cache)];
    let mut xn = vec![0.0; x.len()];
    let mut co = vec![0.0; cache.len()];
    time("dit_step", &mut || {
        g.dit_step(
            &DitInputs {
                x: &x,
                t: 1.0,
                r: 0.5,
                dt: 0.5,
                cond: &cond,
                offset: 40,
                n_valid: 12,
                cache: &cache,
            },
            &speaker,
            &mut xn,
            &mut co,
        )
        .unwrap()
    });
    let mel = vec![0.5f32; MEL_BINS * 6];
    let mut wavout = Vec::new();
    time("vocos 6", &mut || g.vocos(&mel, 6, &mut wavout).unwrap());
}

#[test]
#[ignore = "needs Sigil's MeanVC2 export (SIGIL_TEST_MODELS_DIR/meanvc or the dev cache)"]
fn real_model_stream_costs() {
    let Some(dir) = real_graphs_dir() else {
        return;
    };
    let speech = real_speech("jfk.wav").unwrap();
    // Interleaved rounds (the rig is shared). Flushing denormals (MXCSR FTZ/DAZ on the worker
    // thread) was measured here in P5-C and gave no gain beyond the rig's noise, so the loop runs in
    // the default floating-point mode.
    let mut times: std::collections::BTreeMap<u32, Vec<f64>> = Default::default();
    for _round in 0..3 {
        for steps in [1u32, 2] {
            let (graphs, _) = OrtGraphs::load(&GraphSource::Dir(&dir), &ort_opts()).unwrap();
            let mut stream = MeanVcStream::new(Box::new(graphs), &unit_speaker(), steps, 1).unwrap();
            stream.reset_cold();
            let mut out = Vec::new();
            let v = times.entry(steps).or_default();
            for chunk in speech.as_chunks::<TICK_16K>().0 {
                let t = std::time::Instant::now();
                stream.process(chunk, &mut out).unwrap();
                v.push(t.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }
    for (steps, v) in times.iter_mut() {
        v.sort_by(f64::total_cmp);
        eprintln!(
            "stream.process (80 ms) {steps} steps: median {:.3} ms p90 {:.3} p99 {:.3} ({} calls)",
            percentile(v, 0.5),
            percentile(v, 0.9),
            percentile(v, 0.99),
            v.len()
        );
    }
    let mut nf = crate::services::neural_audio::vc::engine::noise::NoiseFrames::new(
        crate::services::neural_audio::vc::engine::noise::AnchoredNoise::new(1),
        crate::services::neural_audio::vc::engine::noise::NoiseStream::Flow,
        80,
        8,
    );
    let mut x = vec![0.0f32; 640];
    let t = std::time::Instant::now();
    for i in 0..1000u64 {
        nf.copy_window(i * 4, 8, 1.0, false, &mut x);
    }
    eprintln!("noise copy_window: {:.4} ms per DiT call", t.elapsed().as_secs_f64());
}
