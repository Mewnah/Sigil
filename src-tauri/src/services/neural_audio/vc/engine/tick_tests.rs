//! Engine tick tests and the dev-rig renders / benches (split out of `engine/mod.rs` by P3B-S0).
//! Owner: P3B-D.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::vc::manifest::load_slot_manifest;
use crate::services::neural_audio::vc::stream::VcStreamHarness;
use crate::services::neural_audio::vc::test_models;
use crate::services::neural_audio::vc::traits::{ConversionCtx, VoiceControls, VoiceConversionModel};
use crate::services::neural_audio::vc::world::DioPitchExtractor;

use super::content_stats::PlanSwitch;
use super::governor::QualityLevel;
use super::test_support::*;
use super::{content, presets, EffectiveLevers, F0Range, LoadOptions, Lever, PrecisionMode, SigilRvcEngine, SigilRvcRequest};

/// Stream `engine` through the fixed-block harness (48 kHz, 40 ms device blocks) over 2 s of the
/// voiced test signal with `index_rate` and protect 0.33, and return the output.
fn render_through_harness(engine: SigilRvcEngine, index_rate: f32) -> Vec<f32> {
    let mut h = VcStreamHarness::new(Box::new(engine), 48_000, 48_000, 1, 1, 1_600, 320, 1_920).unwrap();
    h.set_controls(2.0, index_rate, 0.33, 1);
    let input: Vec<f32> = voice_16k(16_000 * 2).iter().flat_map(|&x| [x, x, x]).collect();
    let mut out = Vec::new();
    for chunk in input.as_chunks::<1_920>().0 {
        out.extend(h.push_and_convert(chunk).unwrap());
    }
    out
}

/// The tiny fixture voice with a retrieval index (768-dim CPU IVF over clustered rows), so
/// retrieval's frame mapping is exercised.
fn tiny_index(s: &TinySlot) -> PathBuf {
    use crate::services::neural_audio::vc::faiss::tests::{clustered, ivf_of};
    let (centres, data) = clustered(768, 256, 8, 11);
    let path = s.slot.join("tiny.index");
    std::fs::write(&path, ivf_of(768, &centres, &data, 2)).unwrap();
    path
}

/// P3B-D: with a content window longer than the context (1 s of content context against a
/// 0.44 s `enc` window), the features of the context's frames reach `enc`, protect and retrieval
/// at the same frames as without it. The stand-in encoder is a 400/320 convolution, so a context
/// frame's features do not depend on the audio before it: the output must be identical (to the
/// convolution's last bits), which pins the three offset mappings (the 2x upsampling's origin,
/// protect's copy, retrieval's start).
#[test]
fn content_window_offsets_map_onto_the_enc_window() {
    let s = tiny_slot("content-window");
    let index = tiny_index(&s);
    let load = |content_context_16k: Option<usize>| {
        let mut req = request(&s, levers_all(true), 40.0, 100.0);
        req.index = Some(index.clone());
        req.content_context_16k = content_context_16k;
        SigilRvcEngine::load_with(req, LoadOptions { allow_any_frame_rate: true }).unwrap()
    };
    let plain = load(None);
    let c = plain.plan().context_16k;
    assert!(plain.retrieval.is_some(), "the fixture voice has its index");
    let reference = render_through_harness(plain, 0.6);
    let wide = load(Some(16_000));
    assert_eq!(wide.plan().context_16k, c, "the harness's context is unchanged");
    let cw = wide.plan().content_window_16k(wide.content_context_16k);
    assert!(cw >= 16_000 && cw > c);
    let encoder = wide.encoder.clone().unwrap();
    let out = render_through_harness(wide, 0.6);
    assert!(
        encoder.lock().unwrap().dynamic_lengths().contains(&cw),
        "the encoder saw the {cw}-sample content window: {:?}",
        encoder.lock().unwrap().dynamic_lengths()
    );
    assert_eq!(out.len(), reference.len());
    let loud = out.iter().filter(|x| x.abs() > 1e-4).count();
    assert!(loud > out.len() / 4);
    // Identical up to the convolution's last bits on a longer input (its GEMM blocks differently);
    // a mapping one 50 Hz frame off changes the output by the signal's own scale.
    let diff = out.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let peak = reference.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(diff <= 1e-5 * peak.max(1.0), "a content window with left context changed the context frames' mapping (max |diff| {diff}, peak {peak})");
}

/// P3B-D: the overlapped F0 (the helper thread, forced on for the CPU fixture) gives the serial
/// path's output bit for bit, with the F0 cache on and off, through silence skips and resets.
#[test]
fn f0_overlap_matches_serial() {
    let s = tiny_slot("f0-overlap");
    for on in [true, false] {
        let serial = tiny_engine(&s, levers_all(on));
        assert!(!serial.f0_overlapped(), "the CPU rung runs F0 serially");
        let reference = render_through_harness(serial, 0.0);
        let mut overlapped = tiny_engine(&s, levers_all(on));
        overlapped.f0_helper.configure(true, None);
        assert!(overlapped.f0_overlapped());
        let out = render_through_harness(overlapped, 0.0);
        assert_eq!(out, reference, "levers {on}: the overlapped F0 changed the output");
    }
}

/// P3B-D `content_norm`: requested without an installed stream variant, the lever goes off with a
/// note; with one (the stand-in), the voice streams on it, its statistics turn from the window's own
/// to running ones once 0.5 s of audio is in, a stream reset forgets them, and the plan path (forced
/// on, IOBinding on the CPU) runs the stream variant too.
#[test]
fn content_norm_runs_the_stream_variant_with_running_statistics() {
    use super::content::{read_conv0, tests::stream_standin_bytes, StreamVariant};
    let s = tiny_slot("content-norm");
    // Requested, nothing installed: off with a note.
    let mut levers = levers_all(true);
    levers.levers.set(Lever::ContentNorm, true);
    let e = SigilRvcEngine::load_with(request(&s, levers.clone(), 40.0, 100.0), LoadOptions { allow_any_frame_rate: true }).unwrap();
    assert!(!e.effective_levers().levers.get(Lever::ContentNorm));
    assert!(e.effective_levers().notes.iter().any(|n| n.starts_with("content_norm:")), "{:?}", e.effective_levers().notes);
    assert!(!e.content_norm_active());
    // With the stand-in stream variant.
    let conv0: Vec<f32> = (0..512 * 10).map(|i| ((i * 37 % 101) as f32 / 50.0 - 1.0) * 0.15).collect();
    let stream = s.root.join("model.stream.fp32.onnx");
    std::fs::write(&stream, stream_standin_bytes(&conv0)).unwrap();
    std::fs::write(s.root.join("conv0.f32.bin"), conv0.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let sv = || StreamVariant {
        path: stream.clone(),
        variant: "stream.fp32",
        cpu_fallback: None,
        conv0: read_conv0(&s.root.join("conv0.f32.bin")).unwrap(),
    };
    for plan in [false, true] {
        let mut e = tiny_engine(&s, levers_all(true));
        e.enable_content_norm(sv()).unwrap();
        if plan {
            e.content_stats.plan = PlanSwitch::On;
            e.levers.levers.set(Lever::IoBinding, true);
        }
        assert!(e.content_norm_active());
        let p = *e.plan();
        let audio = voice_16k(16_000 * 2);
        e.set_silence_skip(false);
        for tick in 0..30usize {
            let pos = tick * p.block_16k;
            let ctx = ConversionCtx {
                mono_16k: &audio[pos..pos + p.context_16k],
                new_frames: p.want_16k,
                pitch_shift_semitones: 0.0,
                index_rate: 0.0,
                protect: 0.33,
                speaker_id: 0,
                stream_pos_16k: pos as u64,
                reference: None,
                singing: Default::default(),
            };
            let out = e.convert(&ctx).unwrap();
            assert!(out.iter().all(|x| x.is_finite()));
            let running = e.content_stats.norm.as_ref().unwrap().running.last_running;
            // The first context holds 21 entries of 20 ms; 25 (0.5 s) are there two ticks later.
            assert_eq!(running, tick >= 2, "plan {plan} tick {tick}");
        }
        assert_eq!(e.content_runs_fixed_shape(), plan);
        e.reset_stream();
        assert_eq!(e.content_stats.norm.as_ref().unwrap().running.held(), 0, "a reset forgets the statistics");
        let stream_encoder = std::sync::Arc::downgrade(&e.content_stats.norm.as_ref().unwrap().encoder);
        // The harness streams it end to end.
        let out = render_through_harness(e, 0.0);
        let loud = out.iter().filter(|x| x.abs() > 1e-4).count();
        assert!(loud > out.len() / 4, "plan {plan}: {loud} of {} samples carry audio", out.len());
        // The voice is gone: the pool freed its stream-variant encoder (and with it the plan).
        assert!(stream_encoder.upgrade().is_none(), "plan {plan}: the stream encoder outlived its only voice");
    }
}

/// P3B-D: the content encoder's fixed-shape plan (forced on for the CPU fixture with IOBinding: the
/// plan code of the GPU rungs) gives the dynamic session's output, the voice reports it, and the
/// kill switch's path (plan off) stays the P3 one.
#[test]
fn content_plan_matches_the_dynamic_session() {
    let s = tiny_slot("content-plan");
    let reference = render_through_harness(tiny_engine(&s, levers_all(true)), 0.0);
    let mut e = tiny_engine(&s, levers_all(true));
    assert!(!e.content_runs_fixed_shape(), "the CPU rung keeps the dynamic session");
    assert_eq!(e.content_stats.plan, PlanSwitch::Off);
    e.content_stats.plan = PlanSwitch::On;
    e.levers.levers.set(Lever::IoBinding, true);
    // One tick builds and joins the plan.
    let p = *e.plan();
    let audio = voice_16k(p.context_16k);
    let ctx = ConversionCtx {
        mono_16k: &audio,
        new_frames: p.want_16k,
        pitch_shift_semitones: 0.0,
        index_rate: 0.0,
        protect: 0.33,
        speaker_id: 0,
        stream_pos_16k: 0,
        reference: None,
        singing: Default::default(),
    };
    e.convert(&ctx).unwrap();
    assert!(e.content_runs_fixed_shape());
    let outcome = e.ep_outcomes().iter().find(|(c, _)| c == "vc.content").unwrap().1.clone();
    assert_eq!(outcome.active, "cpu");
    e.reset_stream();
    let encoder = e.encoder.clone().unwrap();
    assert_eq!(encoder.lock().unwrap().plan_count(), 1);
    let out = render_through_harness(e, 0.0);
    // The voice is gone (the harness dropped it): its plan went with its lease.
    assert_eq!(encoder.lock().unwrap().plan_count(), 0, "a dropped voice's plan is freed");
    assert_eq!(out.len(), reference.len());
    let num: f64 = out.iter().zip(&reference).map(|(a, b)| f64::from(a - b).powi(2)).sum();
    let den: f64 = reference.iter().map(|b| f64::from(*b).powi(2)).sum();
    assert!((num / den).sqrt() < 1e-5, "plan vs dynamic output rel L2 {}", (num / den).sqrt());
}

/// The whole engine through the fixed-block stream harness: exact lengths, one shape per call,
/// finite non-silent output, with every lever on and with every lever off (legacy paths).
#[test]
fn engine_streams_through_the_fixed_block_harness() {
    let s = tiny_slot("stream");
    for on in [true, false] {
        let engine = tiny_engine(&s, levers_all(on));
        let (block, context) = (engine.plan().block_16k, engine.plan().context_16k);
        let mut h = VcStreamHarness::new(Box::new(engine), 48_000, 48_000, 1, 1, 1_600, 320, 1_920).unwrap();
        let (hb, hc) = h.fixed_shape().expect("fixed-block mode");
        assert_eq!(hb, block);
        assert!(hc >= context);
        assert_eq!(h.fixed_hold_back_frames(), 0, "device block = model block");
        let input: Vec<f32> = voice_16k(16_000 * 3)
            .iter()
            .flat_map(|&x| [x, x, x])
            .collect();
        let mut out = Vec::new();
        for chunk in input.as_chunks::<1_920>().0 {
            let o = h
                .push_and_convert(chunk)
                .unwrap_or_else(|e| panic!("levers {on}: {e}"));
            assert_eq!(o.len(), h.last_owed_frames());
            out.extend(o);
        }
        assert_eq!(out.len(), input.len());
        assert!(out.iter().all(|x| x.is_finite()));
        let loud = out.iter().filter(|x| x.abs() > 1e-4).count();
        assert!(loud > out.len() / 4, "levers {on}: {loud} of {} samples carry audio", out.len());
    }
}

/// FNV-1a over the output bits of the fixture voice streamed through the fixed-block harness
/// (levers on, levers off, 20 ms lookahead): printed so a refactor can be checked bit-for-bit
/// against the build before it (P3B-S0), and asserted reproducible within one build.
#[test]
fn fixture_render_is_bit_reproducible() {
    let s = tiny_slot("render-hash");
    let render = |on: bool, lookahead_ms: f64| -> u64 {
        let mut engine = tiny_engine(&s, levers_all(on));
        engine.set_lookahead_ms(lookahead_ms).unwrap();
        let mut h = VcStreamHarness::new(Box::new(engine), 48_000, 48_000, 1, 1, 1_600, 320, 1_920).unwrap();
        h.set_controls(2.0, 0.0, 0.33, 1);
        let input: Vec<f32> = voice_16k(16_000 * 2).iter().flat_map(|&x| [x, x, x]).collect();
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for chunk in input.as_chunks::<1_920>().0 {
            for x in h.push_and_convert(chunk).unwrap() {
                for b in x.to_bits().to_le_bytes() {
                    hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
        }
        hash
    };
    for (on, la) in [(true, 0.0), (false, 0.0), (true, 20.0)] {
        let a = render(on, la);
        eprintln!("fixture render hash (levers {on}, lookahead {la} ms): {a:016x}");
        assert_eq!(a, render(on, la), "levers {on}, lookahead {la} ms");
    }
}

/// The levers' legacy paths are what they say: F0 over the whole context and flowdec over the
/// whole context when off; the new block plus the margins, and the return window, when on.
#[test]
fn levers_select_incremental_or_whole_context_work() {
    let s = tiny_slot("levers");
    let audio = voice_16k(16_000 * 2);
    for on in [true, false] {
        let mut e = tiny_engine(&s, levers_all(on));
        // The test signal pauses; every tick must convert here.
        e.set_silence_skip(false);
        let p = *e.plan();
        assert_eq!(p.flow_frames == p.enc_frames, !on);
        let mut pos = 0usize;
        for tick in 0..20 {
            let ctx = ConversionCtx {
                mono_16k: &audio[pos..pos + p.context_16k],
                new_frames: p.want_16k,
                pitch_shift_semitones: 0.0,
                index_rate: 0.0,
                protect: 0.33,
                speaker_id: 0,
                stream_pos_16k: pos as u64,
                reference: None,
                singing: Default::default(),
            };
            let out = e.convert(&ctx).unwrap();
            assert_eq!(out.len(), p.out_samples());
            let st = e.tick_stats();
            assert!(!st.skipped);
            if on && tick > 0 {
                let cache = e.f0_cache.config();
                assert_eq!(
                    st.f0_frames,
                    p.block_16k / 160 + cache.margin_frames + e.f0_cache.lead_frames(),
                    "tick {tick}"
                );
            } else {
                assert_eq!(st.f0_frames, p.enc_frames, "tick {tick}");
            }
            pos += p.block_16k;
        }
    }
}

/// Silence costs no inference after a short hangover, and speech resumes cleanly; with the
/// skip off every tick converts.
#[test]
fn silence_skips_inference_after_the_hangover() {
    let s = tiny_slot("silence");
    let mut e = tiny_engine(&s, levers_all(true));
    let p = *e.plan();
    let quiet = vec![0.0f32; p.context_16k];
    let loud = voice_16k(p.context_16k);
    let mut skipped = Vec::new();
    for tick in 0..8u64 {
        let buf = if tick < 6 { &quiet } else { &loud };
        let ctx = ConversionCtx {
            mono_16k: buf,
            new_frames: p.want_16k,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: tick * p.block_16k as u64,
            reference: None,
            singing: Default::default(),
        };
        let out = e.convert(&ctx).unwrap();
        assert_eq!(out.len(), p.out_samples());
        skipped.push(e.tick_stats().skipped);
        if e.tick_stats().skipped {
            assert!(out.iter().all(|&x| x == 0.0));
        }
    }
    // 40 ms blocks, 80 ms hangover: two quiet ticks convert, the next four skip, speech converts.
    assert_eq!(skipped, [false, false, true, true, true, true, false, false]);
    e.set_silence_skip(false);
    let ctx = ConversionCtx {
        mono_16k: &quiet,
        new_frames: p.want_16k,
        pitch_shift_semitones: 0.0,
        index_rate: 0.0,
        protect: 0.33,
        speaker_id: 0,
        stream_pos_16k: 100 * p.block_16k as u64,
        reference: None,
        singing: Default::default(),
    };
    for _ in 0..5 {
        e.convert(&ctx).unwrap();
        assert!(!e.tick_stats().skipped);
    }
}

/// With the skip off, quiet speech (below `silence_db`) is converted *and* heard: the post
/// stage's silence gate follows the skip (review of P3-C1: it muted it after 150 ms). With the
/// skip on the same input goes silent after the hangover.
#[test]
fn silence_skip_off_keeps_quiet_speech_audible() {
    let s = tiny_slot("quiet");
    // A steady voiced tone at -48 dBFS RMS, below the default -42 dB threshold.
    let rms = 10f32.powf(-48.0 / 20.0);
    let tone: Vec<f32> = (0..16_000 * 2)
        .map(|i| rms * std::f32::consts::SQRT_2 * (std::f32::consts::TAU * 160.0 * i as f32 / 16_000.0).sin())
        .collect();
    for skip in [true, false] {
        let mut e = tiny_engine(&s, levers_all(true));
        e.set_silence_skip(skip);
        let p = *e.plan();
        let mut late_energy = 0.0f64;
        for tick in 0..16usize {
            let pos = tick * p.block_16k;
            let ctx = ConversionCtx {
                mono_16k: &tone[pos..pos + p.context_16k],
                new_frames: p.want_16k,
                pitch_shift_semitones: 0.0,
                index_rate: 0.0,
                protect: 0.33,
                speaker_id: 0,
                stream_pos_16k: pos as u64,
                reference: None,
                singing: Default::default(),
            };
            let out = e.convert(&ctx).unwrap();
            assert_eq!(out.len(), p.out_samples());
            // 40 ms blocks, 80 ms hangover: two quiet ticks convert, then the skip.
            assert_eq!(e.tick_stats().skipped, skip && tick >= 2, "skip {skip} tick {tick}");
            // 40 ms blocks: from tick 8 on the post gate's 150 ms hangover is long over.
            if tick >= 8 {
                late_energy += out.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>();
            }
        }
        if skip {
            assert_eq!(late_energy, 0.0);
        } else {
            assert!(late_energy > 1e-9, "quiet speech with the skip off was muted ({late_energy})");
        }
    }
    // Switching the skip back on gates again.
    let mut e = tiny_engine(&s, levers_all(true));
    e.set_silence_skip(false);
    e.set_silence_skip(true);
    assert!(e.silence_skip);
}

/// Speaker blends, the governor's retrieval step (no index here: nothing to apply) and stream
/// resets.
#[test]
fn speaker_blend_quality_steps_and_resets() {
    let s = tiny_slot("blend");
    let mut e = tiny_engine(&s, levers_all(true));
    assert!(e.set_speaker_blend(Some(&[(0, 0.0)])).is_err());
    e.set_speaker_blend(Some(&[(0, 1.0), (2, 1.0)])).unwrap();
    let p = *e.plan();
    let audio = voice_16k(p.context_16k);
    let ctx = ConversionCtx {
        mono_16k: &audio,
        new_frames: p.want_16k,
        pitch_shift_semitones: 0.0,
        index_rate: 0.5,
        protect: 0.2,
        speaker_id: 1,
        stream_pos_16k: 0,
        reference: None,
        singing: Default::default(),
    };
    let blended = e.convert(&ctx).unwrap();
    e.set_speaker_blend(None).unwrap();
    e.reset_stream();
    let single = e.convert(&ctx).unwrap();
    assert_eq!(blended.len(), single.len());
    assert_ne!(blended, single);
    let off = QualityLevel {
        level: 1,
        retrieval: false,
        ..QualityLevel::FULL
    };
    assert!(!e.apply_quality(&off), "no index: the step does nothing here");
    e.set_voice_controls(&VoiceControls {
        noise_scale: 0.0,
        ..VoiceControls::default()
    });
    e.reset_stream();
    let quiet_noise = e.convert(&ctx).unwrap();
    assert!(quiet_noise.iter().all(|x| x.is_finite()));
}

// ---- dev-rig measurements (ignored; need the MIT default voice) ---------------------------

/// Renders a speech clip through the default voice with the fixed-block harness, all levers on
/// and all four legacy paths, writes both WAVs for listening, and prints the spectral-flux
/// spike statistics (a click detector: the ratio of the 99.9th flux percentile to the median).
/// `SIGIL_TEST_MODELS_DIR`, `SIGIL_TEST_CONTENT_ENCODER` (a ContentVec 768 ONNX; required),
/// optional `SIGIL_SVE_CLIP` (16 kHz mono WAV; default the dev cache's
/// `sherpa-onnx/clips/en.wav`), `SIGIL_SVE_RENDER_DIR` (where the WAVs go), `SIGIL_BENCH_EPS`
/// (first EP used, default `dml`).
#[test]
#[ignore = "dev-rig listening render: needs the default voice and a ContentVec export"]
fn sve_render_default_voice() {
    let (Some(models), Some(encoder)) = (
        std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from),
        std::env::var_os("SIGIL_TEST_CONTENT_ENCODER").map(PathBuf::from),
    ) else {
        eprintln!("SIGIL_TEST_MODELS_DIR / SIGIL_TEST_CONTENT_ENCODER not set; skipped");
        return;
    };
    let clip = std::env::var_os("SIGIL_SVE_CLIP")
        .map(PathBuf::from)
        .unwrap_or_else(|| models.join("sherpa-onnx").join("clips").join("en.wav"));
    let mut reader = hound::WavReader::open(&clip).expect("clip");
    let spec = reader.spec();
    assert_eq!((spec.sample_rate, spec.channels), (16_000, 1), "a 16 kHz mono clip");
    let input: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|s| s.unwrap() as f32 / (1u64 << (spec.bits_per_sample - 1)) as f32)
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().map(Result::unwrap).collect(),
    };
    let templates = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("resources")
        .join("vc-templates");
    let voice =
        crate::services::neural_audio::vc::import::import_file(&models.join("rvc-default-voice").join("default.pth"), &templates).unwrap();
    let root = std::env::var_os("SIGIL_SVE_RENDER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join(format!("sigil-sve-render-{}", std::process::id())));
    let slot = root.join("slot");
    std::fs::create_dir_all(&slot).unwrap();
    std::fs::write(slot.join("weights.f32.bin"), &voice.weights.f32_image).unwrap();
    let speakers: Vec<u8> = voice
        .weights
        .speakers
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(slot.join("speakers.f32.bin"), speakers).unwrap();
    let record = serde_json::json!({"template": voice.template.layout.template, "weights_id": voice.template.layout.weights.weights_id});
    std::fs::write(slot.join("sigil-import.json"), serde_json::to_vec(&record).unwrap()).unwrap();
    let manifest = serde_json::json!({"format": "sigil-rvc-1", "template": voice.template.layout.template, "weights": "weights.f32.bin", "speakers": "speakers.f32.bin", "embedder": "contentvec"});
    std::fs::write(slot.join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
    let ep = std::env::var("SIGIL_BENCH_EPS")
        .ok()
        .and_then(|v| v.split(',').next().map(str::to_string))
        .unwrap_or_else(|| "dml".into());
    let all = [Lever::PartialDecode, Lever::AnchoredNoise, Lever::NsfPhase0, Lever::F0Cache];
    let variants: [(&str, &[Lever], f64); 9] = [
        ("levers-on", &all, 0.0),
        ("levers-legacy", &[], 0.0),
        ("no-anchored-noise", &[Lever::PartialDecode, Lever::NsfPhase0, Lever::F0Cache], 0.0),
        ("no-phase0", &[Lever::PartialDecode, Lever::AnchoredNoise, Lever::F0Cache], 0.0),
        ("no-f0-cache", &[Lever::PartialDecode, Lever::AnchoredNoise, Lever::NsfPhase0], 0.0),
        ("levers-on-lookahead10", &all, 10.0),
        ("levers-on-lookahead20", &all, 20.0),
        ("levers-on-lookahead40", &all, 40.0),
        ("levers-legacy-lookahead20", &[], 20.0),
    ];
    for (label, on, lookahead_ms) in variants {
        let mut levers = EffectiveLevers::default();
        for &lever in on {
            levers.levers.set(lever, true);
        }
        let req = SigilRvcRequest {
            slot_dir: slot.clone(),
            manifest: load_slot_manifest(&slot).unwrap(),
            templates_dir: templates.clone(),
            ep: ep.clone(),
            session: VcSessionOptions::default(),
            levers,
            pitch: Box::new(DioPitchExtractor::new()),
            content_encoder: Some(encoder.clone()),
            index: None,
            preset: presets::engine_preset(RvcPresetProfile::LowLatency, 40.0, 0, 16_000, 96.0, PrecisionMode::Auto),
            f0_range: F0Range::Speech,
            noise_seed: 3,
            lookahead_ms: 0.0,
            stream: None,
            content_context_16k: None,
        };
        let mut engine = SigilRvcEngine::load(req).unwrap();
        engine.set_silence_skip(false);
        engine.set_lookahead_ms(lookahead_ms).unwrap();
        let block = engine.plan().block_16k;
        let mut h = VcStreamHarness::new(Box::new(engine), 16_000, 48_000, 1, 1, 1_536, 960, block).unwrap();
        h.set_controls(4.0, 0.0, 0.33, 0);
        let mut out = Vec::new();
        let mut joins: Vec<(isize, f32)> = Vec::new();
        for chunk in input.chunks_exact(block) {
            out.extend(h.push_and_convert(chunk).unwrap());
            joins.extend(h.last_join());
        }
        let mut nccs: Vec<f64> = joins.iter().map(|&(_, c)| f64::from(c)).collect();
        let centred = joins.iter().filter(|&&(o, _)| o == 0).count() as f64 / joins.len().max(1) as f64;
        let mean_shift = joins
            .iter()
            .map(|&(o, _)| o.unsigned_abs() as f64)
            .sum::<f64>()
            / joins.len().max(1) as f64;
        let mean_ncc = nccs.iter().sum::<f64>() / nccs.len().max(1) as f64;
        let p10_ncc = percentile(&mut nccs, 0.1);
        let path = root.join(format!("{label}.wav"));
        let mut w = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 48_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        for &s in &out {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();
        // Seams: the harness joins model blocks every `block` input samples (1920 output
        // frames here). Flux over 2.5 ms hops folded onto the 16 phases of that period: speech
        // spreads evenly over the phases, clicks at the joins pile up on one.
        let period = block * 3;
        let hop = period / 16;
        let flux = spectral_flux(&out, 4 * hop, hop);
        let mut phases = vec![(0.0f64, 0usize); 16];
        for (i, &f) in flux.iter().enumerate() {
            let p = &mut phases[i % 16];
            p.0 += f;
            p.1 += 1;
        }
        let mut means: Vec<f64> = phases.iter().map(|&(s, n)| s / n.max(1) as f64).collect();
        let top = means.iter().cloned().fold(0.0, f64::max);
        let median = percentile(&mut means, 0.5);
        let rms = (out
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            / out.len() as f64)
            .sqrt();
        eprintln!(
            "{ep} {label}: {:.2} s, rms {rms:.4}, seam flux (max phase / median phase) {:.3}, joins {}: NCC mean {mean_ncc:.3} p10 {p10_ncc:.3}, centred {:.0} %, mean shift {mean_shift:.1} samples → {}",
            out.len() as f64 / 48_000.0,
            top / median.max(1e-12),
            joins.len(),
            centred * 100.0,
            path.display()
        );
    }
}

/// Per-block compute of the engine on the MIT default voice (v2 40k, bundled template) for
/// growing extra context, with each algorithmic lever on and off (plan T2: "growing the extra
/// context from 64 ms to 512 ms raises p95 by ≤ 30 %"). Prints a table; asserts nothing about
/// speed (the dev rig is shared).
///
/// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache` (holds `rvc-default-voice/default.pth`),
/// optional `SIGIL_TEST_CONTENT_ENCODER=<ContentVec 768 ONNX>` (else a 400/320 conv stand-in,
/// flagged in the output), `SIGIL_BENCH_EPS=cpu,dml,cuda` (CUDA also needs
/// `SIGIL_TEST_CUDA_BIN` / `SIGIL_TEST_CUDNN_BIN`), `SIGIL_BENCH_TICKS` (default 150):
/// `cargo test --manifest-path src-tauri/Cargo.toml sve_context_scaling_bench -- --ignored --nocapture`
#[test]
#[ignore = "dev-rig benchmark: needs SIGIL_TEST_MODELS_DIR with rvc-default-voice/default.pth"]
fn sve_context_scaling_bench() {
    let Some(models) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from) else {
        eprintln!("SIGIL_TEST_MODELS_DIR is not set; skipped");
        return;
    };
    let pth = models.join("rvc-default-voice").join("default.pth");
    let templates = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("resources")
        .join("vc-templates");
    let voice = crate::services::neural_audio::vc::import::import_file(&pth, &templates).expect("default voice imports");
    let root = std::env::temp_dir().join(format!("sigil-sve-bench-{}", std::process::id()));
    let slot = root.join("voice-changer").join("models").join("default");
    std::fs::create_dir_all(&slot).unwrap();
    std::fs::write(slot.join("weights.f32.bin"), &voice.weights.f32_image).unwrap();
    let speakers: Vec<u8> = voice
        .weights
        .speakers
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(slot.join("speakers.f32.bin"), speakers).unwrap();
    let record = serde_json::json!({"template": voice.template.layout.template, "weights_id": voice.template.layout.weights.weights_id});
    std::fs::write(slot.join("sigil-import.json"), serde_json::to_vec(&record).unwrap()).unwrap();
    let manifest = serde_json::json!({
        "format": "sigil-rvc-1", "template": voice.template.layout.template, "weights": "weights.f32.bin",
        "speakers": "speakers.f32.bin", "embedder": "contentvec", "fp16_ok": true
    });
    std::fs::write(slot.join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
    // A ContentVec export folder (`model.<variant>.onnx`, P3-A's layout): each EP gets the
    // variant the pool would pick; else one explicit file; else the conv stand-in.
    let encoder_dir = std::env::var_os("SIGIL_TEST_CONTENT_ENCODER_DIR").map(PathBuf::from);
    let (fixed_encoder, encoder_label) = match (&encoder_dir, std::env::var_os("SIGIL_TEST_CONTENT_ENCODER").map(PathBuf::from)) {
        (Some(_), _) => (None, "ContentVec (per-EP variant)"),
        (None, Some(p)) => (Some(p), "ContentVec"),
        (None, None) => {
            let p = root.join("standin-content.onnx");
            std::fs::write(&p, test_models::content_encoder_bytes()).unwrap();
            (Some(p), "stand-in conv (not ContentVec)")
        }
    };
    let encoder_for = |ep: &str| -> (PathBuf, &'static str) {
        if let Some(p) = &fixed_encoder {
            return (p.clone(), "file");
        }
        let dir = encoder_dir.as_ref().unwrap();
        for v in content::variant_order(
            ep != "cpu",
            PrecisionMode::Auto,
            crate::services::neural_audio::vc::int8_gate::int8_content_encoder_enabled(),
        ) {
            let p = dir.join(crate::services::neural_audio::vc::assets::file_name(v));
            if p.is_file() {
                return (p, v.unwrap_or("model"));
            }
        }
        panic!("no content encoder variant in {}", dir.display());
    };
    let cuda_ok = crate::services::neural_audio::ort_session::preload_test_cuda_runtime();
    let eps: Vec<String> = std::env::var("SIGIL_BENCH_EPS")
        .unwrap_or_else(|_| "cpu".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && (s != "cuda" || cuda_ok))
        .collect();
    let ticks: usize = std::env::var("SIGIL_BENCH_TICKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(150);
    let audio = voice_16k(16_000 * 30);
    eprintln!("SVE per-block compute, default voice v2 40k, 40 ms block, dio F0, content encoder: {encoder_label}, {ticks} ticks");
    eprintln!("| EP | levers | extra ms | ctx frames | flowdec F | precision | p50 ms | p95 ms | content | f0 | enc | flowdec | f0 frames |");
    eprintln!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    let extras: Vec<f64> = std::env::var("SIGIL_BENCH_EXTRA_MS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![64.0, 192.0, 512.0, 1_000.0, 2_000.0]);
    for ep in &eps {
        let (encoder, variant) = encoder_for(ep);
        eprintln!("  {ep}: content encoder {} ({variant})", encoder.display());
        for (label, partial, f0c, precision) in [
            ("all on", true, true, PrecisionMode::Auto),
            ("all on fp32", true, true, PrecisionMode::Fp32),
            ("partial off", false, true, PrecisionMode::Auto),
            ("f0 cache off", true, false, PrecisionMode::Auto),
        ] {
            if precision == PrecisionMode::Fp32 && ep == "cpu" {
                continue;
            }
            for &extra in &extras {
                let mut levers = EffectiveLevers {
                    precision,
                    ..EffectiveLevers::default()
                };
                for lever in [Lever::AnchoredNoise, Lever::NsfPhase0] {
                    levers.levers.set(lever, true);
                }
                levers.levers.set(Lever::PartialDecode, partial);
                levers.levers.set(Lever::F0Cache, f0c);
                let req = SigilRvcRequest {
                    slot_dir: slot.clone(),
                    manifest: load_slot_manifest(&slot).unwrap(),
                    templates_dir: templates.clone(),
                    ep: ep.clone(),
                    session: VcSessionOptions {
                        intra_threads: Some(2),
                        component_prefix: Some("bench"),
                        ..VcSessionOptions::default()
                    },
                    levers,
                    pitch: Box::new(DioPitchExtractor::new()),
                    content_encoder: Some(encoder.clone()),
                    index: None,
                    preset: presets::engine_preset(RvcPresetProfile::LowLatency, 40.0, 0, 48_000, extra, PrecisionMode::Auto),
                    f0_range: F0Range::Speech,
                    noise_seed: 7,
                    lookahead_ms: 0.0,
                    stream: None,
                    content_context_16k: None,
                };
                let mut e = SigilRvcEngine::load(req).expect("engine loads");
                e.set_silence_skip(false);
                let p = *e.plan();
                let (mut total, mut content, mut f0, mut enc, mut fd) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
                let mut f0_frames = 0;
                for tick in 0..ticks + 20 {
                    let pos = (tick * p.block_16k) % (audio.len() - p.context_16k);
                    let pos = pos - pos % p.block_16k;
                    let ctx = ConversionCtx {
                        mono_16k: &audio[pos..pos + p.context_16k],
                        new_frames: p.want_16k,
                        pitch_shift_semitones: 0.0,
                        index_rate: 0.0,
                        protect: 0.33,
                        speaker_id: 0,
                        stream_pos_16k: (tick * p.block_16k) as u64,
                        reference: None,
                        singing: Default::default(),
                    };
                    let started = Instant::now();
                    e.convert(&ctx).unwrap();
                    let ms = started.elapsed().as_secs_f64() * 1000.0;
                    if tick >= 20 {
                        let st = e.tick_stats();
                        total.push(ms);
                        content.push(st.content_ms);
                        f0.push(st.f0_ms);
                        enc.push(st.enc_ms);
                        fd.push(st.flowdec_ms);
                        f0_frames = st.f0_frames;
                    }
                }
                let outcome = e
                    .ep_outcomes()
                    .iter()
                    .find(|(c, _)| c == "vc.flowdec")
                    .map(|(_, o)| o.clone())
                    .expect("flowdec outcome");
                if let Some(why) = &outcome.reason {
                    eprintln!("  ({ep} fell back: {why})");
                }
                let active = outcome.active;
                eprintln!(
                    "| {ep}→{active} | {label} | {extra} | {} | {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {f0_frames} |",
                    p.enc_frames,
                    p.flow_frames,
                    e.flowdec_precision(),
                    percentile(&mut total.clone(), 0.5),
                    percentile(&mut total, 0.95),
                    percentile(&mut content, 0.5),
                    percentile(&mut f0, 0.5),
                    percentile(&mut enc, 0.5),
                    percentile(&mut fd, 0.5),
                );
            }
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}

// ---- P3B-D dev-rig measurements (ignored; hold the GPU timing lock, rule R2) ----------------

/// A 16 kHz mono WAV as f32.
fn read_wav_16k(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let spec = reader.spec();
    assert_eq!((spec.sample_rate, spec.channels), (16_000, 1), "{}: a 16 kHz mono clip", path.display());
    match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|s| s.unwrap() as f32 / (1u64 << (spec.bits_per_sample - 1)) as f32)
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().map(Result::unwrap).collect(),
    }
}

/// Speech clips of the dev cache (`SIGIL_TEST_SPEECH_DIR`, default
/// `%LOCALAPPDATA%/sigil-dev-cache/models/transcribe/samples`): every 16 kHz mono WAV there.
fn dev_speech() -> Vec<(String, Vec<f32>)> {
    let dir = std::env::var_os("SIGIL_TEST_SPEECH_DIR").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var("LOCALAPPDATA").unwrap_or_default())
            .join("sigil-dev-cache")
            .join("models")
            .join("transcribe")
            .join("samples")
    });
    let mut clips: Vec<(String, Vec<f32>)> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "wav"))
                .filter_map(|p| {
                    let r = hound::WavReader::open(&p).ok()?;
                    (r.spec().sample_rate == 16_000 && r.spec().channels == 1)
                        .then(|| (p.file_name().unwrap().to_string_lossy().into_owned(), read_wav_16k(&p)))
                })
                .collect()
        })
        .unwrap_or_default();
    clips.sort_by(|a, b| a.0.cmp(&b.0));
    clips
}

fn gpu_eps() -> Vec<String> {
    crate::services::neural_audio::ort_session::preload_test_cuda_runtime_quiet();
    std::env::var("SIGIL_BENCH_EPS")
        .unwrap_or_else(|_| "cuda,dml".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// P3B-D measurement (2) and the plan's parity / shape / VRAM checks, per GPU EP and content
/// encoder variant (`SIGIL_TEST_CONTENT_ENCODER_DIR` holding `model.fp16.onnx` / `model.fp32.onnx`,
/// e.g. the app data's `models/vc/embedder-contentvec/1`; `SIGIL_APP_DATA_DIR` for the NVIDIA
/// runtime pack; `SIGIL_BENCH_EPS`, default `cuda,dml`):
/// - the dynamic session's per-call cost at the 20 ms-crossfade window (7040 samples) and at the
///   10 ms-crossfade window (6720): first on a session that ran 7040 first, then on a fresh session
///   that runs 6720 from its first call, then alternating: whether DirectML's 34-49 ms calls come
///   from first-shape specialisation or from alternating shapes;
/// - the fixed-shape plan at both windows and at the 20 ms-lookahead window (7360): build time,
///   VRAM, per-call cost, the levers it got;
/// - parity: plan against a fresh dynamic session on 8 speech windows per length (relative L2; the
///   acceptance for DirectML is at most 1e-3), and a CUDA graph that passed its check;
/// - DirectML lifetime: the dynamic session released while the plan keeps running (ADR 0006
///   finding 5).
///
/// `cargo test --manifest-path src-tauri/Cargo.toml content_plan_dev_rig -- --ignored --nocapture`
#[test]
#[ignore = "dev-rig: needs SIGIL_TEST_CONTENT_ENCODER_DIR and a GPU (hold the GPU timing lock)"]
fn content_plan_dev_rig() {
    use super::content::{ContentEncoder, EmbedderKind};
    use crate::services::neural_audio::gpu::GpuLevers;
    let Some(dir) = std::env::var_os("SIGIL_TEST_CONTENT_ENCODER_DIR").map(PathBuf::from) else {
        eprintln!("SIGIL_TEST_CONTENT_ENCODER_DIR is not set; skipped");
        return;
    };
    let clips = dev_speech();
    assert!(!clips.is_empty(), "no 16 kHz speech clips (SIGIL_TEST_SPEECH_DIR)");
    let runs: usize = std::env::var("SIGIL_BENCH_TICKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    let opts = VcSessionOptions::default();
    // 8 speech windows of a length, spread over the clips.
    let windows = |len: usize| -> Vec<Vec<f32>> {
        let mut w = Vec::new();
        for (_, clip) in &clips {
            let n = clip.len().saturating_sub(len);
            for k in 0..4 {
                if w.len() < 8 && n > 0 {
                    let at = (n / 5) * (k + 1);
                    w.push(clip[at..at + len].to_vec());
                }
            }
        }
        w
    };
    let (l20, l10, l20la) = (7_040usize, 6_720usize, 7_360usize);
    // Extra plan lengths (e.g. 16000,32000: 1 s and 2 s content windows), SIGIL_TEST_CONTENT_LENGTHS.
    let extra_lengths: Vec<usize> = std::env::var("SIGIL_TEST_CONTENT_LENGTHS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();
    let selected: Vec<String> = std::env::var("SIGIL_TEST_CONTENT_VARIANTS")
        .unwrap_or_else(|_| "fp16,fp32".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    for ep in gpu_eps() {
        for variant in selected.iter().map(String::as_str) {
            let path = dir.join(format!("model.{variant}.onnx"));
            if !path.is_file() {
                continue;
            }
            let variant: &'static str = if variant == "fp16" { "fp16" } else { "fp32" };
            let time = |enc: &mut ContentEncoder, w: &[f32], n: usize| -> f64 {
                let mut out = Vec::new();
                let mut v: Vec<f64> = (0..n)
                    .map(|_| {
                        let t = Instant::now();
                        enc.encode(w, &mut out).unwrap();
                        ms_since(t)
                    })
                    .collect();
                percentile(&mut v, 0.5)
            };
            let w20 = windows(l20).swap_remove(0);
            let w10 = windows(l10).swap_remove(0);
            // (2) the dynamic session's shapes (and its VRAM at the 20 ms-crossfade window).
            let probe = crate::services::neural_audio::gpu::arena::VramProbe::for_rung(&ep, 0);
            let t = Instant::now();
            let mut a = ContentEncoder::load(EmbedderKind::ContentVec768, &path, variant, &ep, &opts).unwrap();
            let load_ms = ms_since(t);
            if a.outcome().active == "cpu" {
                eprintln!("{ep} {variant}: the session fell back to the CPU ({:?}); skipped", a.outcome().reason);
                continue;
            }
            let t = Instant::now();
            a.encode(&w20, &mut Vec::new()).unwrap();
            let first20 = ms_since(t);
            let a20 = time(&mut a, &w20, runs);
            let dynamic_vram = probe.and_then(|p| p.delta_mb());
            let t = Instant::now();
            a.encode(&w10, &mut Vec::new()).unwrap();
            let first10_after20 = ms_since(t);
            let a10 = time(&mut a, &w10, runs);
            drop(a);
            let mut b = ContentEncoder::load(EmbedderKind::ContentVec768, &path, variant, &ep, &opts).unwrap();
            b.encode(&w10, &mut Vec::new()).unwrap();
            let b10 = time(&mut b, &w10, runs);
            let mut alt: Vec<f64> = Vec::new();
            let mut out = Vec::new();
            for k in 0..runs {
                let w = if k % 2 == 0 { &w20 } else { &w10 };
                let t = Instant::now();
                b.encode(w, &mut out).unwrap();
                alt.push(ms_since(t));
            }
            let alt_p50 = percentile(&mut alt, 0.5);
            eprintln!(
                "{ep} {variant} dynamic: load {load_ms:.0} ms, VRAM at 7040 {:?} MB | first run 7040 {first20:.1} ms, p50 7040 {a20:.2} ms | then 6720: first {first10_after20:.1} ms, p50 {a10:.2} ms | fresh session at 6720: p50 {b10:.2} ms | alternating 7040/6720: p50 {alt_p50:.2} ms | lengths seen {:?}",
                dynamic_vram.map(f64::round),
                b.dynamic_lengths()
            );
            drop(b);
            // The plans (on CUDA with and without the graph: its speed against the VRAM it holds).
            let graph_cases: &[bool] = if ep == "cuda" { &[true, false] } else { &[false] };
            for (&graphs, len) in graph_cases
                .iter()
                .flat_map(|g| [l20, l10, l20la].into_iter().chain(extra_lengths.iter().copied()).map(move |len| (g, len)))
            {
                let levers = GpuLevers {
                    io_binding: true,
                    cuda_graphs: graphs,
                    ..GpuLevers::default()
                };
                let ws = windows(len);
                let mut enc = ContentEncoder::load(EmbedderKind::ContentVec768, &path, variant, &ep, &opts).unwrap();
                let mut reference = Vec::new();
                let mut lease = None;
                let mut got = Vec::new();
                let t = Instant::now();
                let (_, report) = enc.encode_fixed(&ws[0], levers, &mut lease, &mut got).unwrap();
                let build_ms = ms_since(t);
                let report = report.unwrap();
                assert!(report.fixed, "{ep} {variant} {len}: no plan: {:?}", report.notes);
                // The dynamic session is released (no dynamic caller): the plan keeps running.
                let released = !enc.has_dynamic_session();
                let mut v: Vec<f64> = (0..runs)
                    .map(|k| {
                        let t = Instant::now();
                        enc.encode_fixed(&ws[k % ws.len()], levers, &mut lease, &mut got).unwrap();
                        ms_since(t)
                    })
                    .collect();
                let p50 = percentile(&mut v.clone(), 0.5);
                let p95 = percentile(&mut v, 0.95);
                // Parity on 8 windows against a fresh dynamic session.
                let mut dynamic = ContentEncoder::load(EmbedderKind::ContentVec768, &path, variant, &ep, &opts).unwrap();
                let mut worst = 0.0f64;
                for w in &ws {
                    dynamic.encode(w, &mut reference).unwrap();
                    enc.encode_fixed(w, levers, &mut lease, &mut got).unwrap();
                    worst = worst.max(content::rel_l2(&got, &reference));
                }
                eprintln!(
                    "{ep} {variant} plan {len}: build {build_ms:.0} ms, levers {:?}, VRAM {:?} MB, p50 {p50:.2} ms, p95 {p95:.2} ms, parity max rel L2 {worst:.2e} over {} windows, dynamic session released {released} {}",
                    report.levers,
                    report.vram_mb.map(f64::round),
                    ws.len(),
                    if report.notes.is_empty() { String::new() } else { format!("(notes: {})", report.notes.join("; ")) }
                );
                // Both precisions: the shipping fp16 variant is bit-identical on DirectML (measured 0),
                // fp32 within 2.3e-6.
                if ep == "dml" {
                    assert!(worst <= 1e-3, "DirectML {variant} static vs dynamic rel L2 {worst:.2e} > 1e-3 at {len}");
                }
                if ep == "cuda" && graphs {
                    assert!(report.levers.cuda_graphs, "the CUDA graph did not pass its check: {:?}", report.notes);
                }
            }
        }
    }
}

/// P3B-D cold load of the content encoder on each GPU EP (fp16 store variant): the eager path the
/// engine's load takes today (`acquire_for_slot`: the dynamic session, then at the first tick a
/// reference run on it, the plan and the release of the dynamic session) against the deferred path
/// (`acquire_for_engine`: the plan straight from the export contract, no dynamic session). Wall
/// time to the first planned call and the VRAM peak while loading (the dynamic session and the plan
/// are both alive until the eager path releases the first). `SIGIL_TEST_CONTENT_ENCODER_DIR` (with
/// `embedder.json`), `SIGIL_BENCH_EPS`.
#[test]
#[ignore = "dev-rig: needs SIGIL_TEST_CONTENT_ENCODER_DIR and a GPU (hold the GPU timing lock)"]
fn content_cold_load_dev_rig() {
    use super::content::{ContentEncoder, EmbedderKind};
    use crate::services::neural_audio::gpu::arena::VramProbe;
    use crate::services::neural_audio::gpu::GpuLevers;
    let Some(dir) = std::env::var_os("SIGIL_TEST_CONTENT_ENCODER_DIR").map(PathBuf::from) else {
        eprintln!("SIGIL_TEST_CONTENT_ENCODER_DIR is not set; skipped");
        return;
    };
    let path = dir.join("model.fp16.onnx");
    let clips = dev_speech();
    let w = clips.first().expect("speech clips").1[16_000..16_000 + 7_040].to_vec();
    let opts = VcSessionOptions::default();
    for ep in gpu_eps() {
        let levers = GpuLevers {
            io_binding: true,
            cuda_graphs: ep == "cuda",
            ..GpuLevers::default()
        };
        // The EP's process-wide state (CUDA context, DirectML device) first, as in the engine,
        // where the generator loads before the content encoder.
        drop(ContentEncoder::deferred(EmbedderKind::ContentVec768, &path, "fp16", &ep, &opts, None).and_then(|mut e| {
            let mut lease = None;
            e.encode_fixed(&w, levers, &mut lease, &mut Vec::new()).map(|_| ())
        }));
        for deferred in [false, true, false, true] {
            let probe = VramProbe::for_rung(&ep, 0);
            // The load's VRAM peak, sampled every 2 ms on another thread.
            let done = std::sync::atomic::AtomicBool::new(false);
            let (session_ms, total_ms, report, has_dynamic, peak) = std::thread::scope(|sc| {
                let sampler = sc.spawn(|| {
                    let mut peak = 0.0f64;
                    while !done.load(std::sync::atomic::Ordering::Relaxed) {
                        if let Some(d) = probe.and_then(|p| p.delta_mb()) {
                            peak = peak.max(d);
                        }
                        std::thread::sleep(std::time::Duration::from_millis(2));
                    }
                    peak
                });
                let t = Instant::now();
                let mut enc = if deferred {
                    ContentEncoder::deferred(EmbedderKind::ContentVec768, &path, "fp16", &ep, &opts, None).unwrap()
                } else {
                    ContentEncoder::load(EmbedderKind::ContentVec768, &path, "fp16", &ep, &opts).unwrap()
                };
                let session_ms = ms_since(t);
                let mut lease = None;
                let mut feats = Vec::new();
                let (_, report) = enc.encode_fixed(&w, levers, &mut lease, &mut feats).unwrap();
                let total_ms = ms_since(t);
                let has_dynamic = enc.has_dynamic_session();
                // Hold the encoder while the sampler sees the steady state.
                std::thread::sleep(std::time::Duration::from_millis(50));
                done.store(true, std::sync::atomic::Ordering::Relaxed);
                let peak = sampler.join().unwrap();
                drop(enc);
                (session_ms, total_ms, report.unwrap(), has_dynamic, peak)
            });
            eprintln!(
                "{ep} fp16 {}: session {session_ms:.0} ms, to the first planned call {total_ms:.0} ms, plan {:?} MB, load VRAM peak {peak:.0} MB, fixed {}, dynamic session left {has_dynamic}",
                if deferred { "deferred" } else { "eager" },
                report.vram_mb.map(f64::round),
                report.fixed,
            );
        }
    }
}

/// P3B-D measurement (4): the content encoder alone, RMVPE alone and both at once from two threads,
/// at the live shapes (the 7040-sample content window; RMVPE's incremental call over 24 frames), on
/// each GPU EP: whether the F0 overlap can hide F0 behind content (concurrent close to the longer of
/// the two) or the queues serialise (concurrent close to the sum). Content runs as the fixed-shape
/// plan and as the dynamic session. `SIGIL_TEST_CONTENT_ENCODER_DIR` (as above), `SIGIL_TEST_RMVPE`
/// (the RMVPE ONNX), `SIGIL_BENCH_EPS`, `SIGIL_BENCH_TICKS` (default 100).
#[test]
#[ignore = "dev-rig: needs SIGIL_TEST_CONTENT_ENCODER_DIR, SIGIL_TEST_RMVPE and a GPU (hold the GPU timing lock)"]
fn content_and_f0_concurrency_bench() {
    use super::content::{ContentEncoder, ContentLease, EmbedderKind};
    use crate::services::neural_audio::gpu::GpuLevers;
    use crate::services::neural_audio::vc::rmvpe::RmvpePitchExtractor;
    use crate::services::neural_audio::vc::traits::PitchExtractor;
    let (Some(dir), Some(rmvpe)) = (
        std::env::var_os("SIGIL_TEST_CONTENT_ENCODER_DIR").map(PathBuf::from),
        std::env::var("SIGIL_TEST_RMVPE").ok(),
    ) else {
        eprintln!("SIGIL_TEST_CONTENT_ENCODER_DIR / SIGIL_TEST_RMVPE not set; skipped");
        return;
    };
    let clips = dev_speech();
    let speech = &clips.first().expect("speech clips").1;
    let runs: usize = std::env::var("SIGIL_BENCH_TICKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let opts = VcSessionOptions::default();
    let content_w = speech[16_000..16_000 + 7_040].to_vec();
    let f0_w = content_w[7_040 - 3_840..].to_vec();
    for ep in gpu_eps() {
        let levers = GpuLevers {
            io_binding: true,
            cuda_graphs: ep == "cuda",
            ..GpuLevers::default()
        };
        let path = dir.join("model.fp16.onnx");
        let mut pitch = RmvpePitchExtractor::load_with_levers(&rmvpe, &ep, &opts, levers).unwrap();
        for plan in [true, false] {
            let mut enc = ContentEncoder::load(EmbedderKind::ContentVec768, &path, "fp16", &ep, &opts).unwrap();
            let mut lease: Option<ContentLease> = None;
            let mut feats = Vec::new();
            let content_w = &content_w;
            let content = move |enc: &mut ContentEncoder, feats: &mut Vec<f32>, lease: &mut Option<ContentLease>| {
                if plan {
                    enc.encode_fixed(content_w, levers, lease, feats).unwrap();
                } else {
                    enc.encode(content_w, feats).unwrap();
                }
            };
            for _ in 0..5 {
                content(&mut enc, &mut feats, &mut lease);
                pitch.f0(&f0_w).unwrap();
            }
            let mut c_alone: Vec<f64> = (0..runs)
                .map(|_| {
                    let t = Instant::now();
                    content(&mut enc, &mut feats, &mut lease);
                    ms_since(t)
                })
                .collect();
            let mut f_alone: Vec<f64> = (0..runs)
                .map(|_| {
                    let t = Instant::now();
                    pitch.f0(&f0_w).unwrap();
                    ms_since(t)
                })
                .collect();
            let mut both: Vec<f64> = Vec::with_capacity(runs);
            let start = std::sync::Barrier::new(2);
            let done = std::sync::Barrier::new(2);
            std::thread::scope(|s| {
                let pitch = &mut pitch;
                let (start, done, f0_w) = (&start, &done, &f0_w);
                let f0_thread = s.spawn(move || {
                    for _ in 0..runs {
                        start.wait();
                        pitch.f0(f0_w).unwrap();
                        done.wait();
                    }
                });
                for _ in 0..runs {
                    start.wait();
                    let t = Instant::now();
                    content(&mut enc, &mut feats, &mut lease);
                    done.wait();
                    both.push(ms_since(t));
                }
                f0_thread.join().unwrap();
            });
            let (c50, f50, b50) = (percentile(&mut c_alone, 0.5), percentile(&mut f_alone, 0.5), percentile(&mut both, 0.5));
            let (c95, f95, b95) = (percentile(&mut c_alone, 0.95), percentile(&mut f_alone, 0.95), percentile(&mut both, 0.95));
            eprintln!(
                "{ep} content {}: content alone p50 {c50:.2} / p95 {c95:.2} ms | RMVPE alone p50 {f50:.2} / p95 {f95:.2} ms | both concurrently p50 {b50:.2} / p95 {b95:.2} ms (sum {:.2}, max {:.2}: the overlap hides {:.0} % of the shorter)",
                if plan { "plan" } else { "dynamic" },
                c50 + f50,
                c50.max(f50),
                100.0 * ((c50 + f50 - b50) / c50.min(f50)).clamp(-9.99, 9.99)
            );
        }
    }
}

/// P3B-D review fix: the deferred plan's CPU cross-check (`content::PLAN_CPU_CHECK_REL_L2`)
/// calibrated and timed on each GPU EP and store variant. Per window length (the live shapes and the
/// 1 s / 2 s content windows), on 8 speech windows: the worst relative L2 of the plan's features
/// against (a) a CPU session of the same file, which is the check, (b) the CPU session of the fp32
/// file (a reference with other numerics), and (c) a wrongly scaled copy of the plan's own output
/// (what a broken compile reads like, to show the margin below the limit). Also the CPU session's
/// build and run time, and the deferred plan's time to its first planned call with the
/// cross-check off and on (a cold shape: the verdict is not on record). Asserts the check's worst
/// difference stays under half the limit. `SIGIL_TEST_CONTENT_ENCODER_DIR` (model.fp16.onnx and
/// model.fp32.onnx with embedder.json), `SIGIL_BENCH_EPS`.
#[test]
#[ignore = "dev-rig: needs SIGIL_TEST_CONTENT_ENCODER_DIR and a GPU (hold the GPU timing lock)"]
fn content_plan_cpu_reference_dev_rig() {
    use super::content::{cpu_references_for_tests, ContentEncoder, EmbedderKind, CONTENT_CHECK_ENV, PLAN_CPU_CHECK_REL_L2};
    use crate::services::neural_audio::gpu::GpuLevers;
    let Some(dir) = std::env::var_os("SIGIL_TEST_CONTENT_ENCODER_DIR").map(PathBuf::from) else {
        eprintln!("SIGIL_TEST_CONTENT_ENCODER_DIR is not set; skipped");
        return;
    };
    let clips = dev_speech();
    assert!(!clips.is_empty(), "no 16 kHz speech clips (SIGIL_TEST_SPEECH_DIR)");
    let opts = VcSessionOptions::default();
    let windows = |len: usize| -> Vec<Vec<f32>> {
        let mut w = Vec::new();
        for (_, clip) in &clips {
            let n = clip.len().saturating_sub(len);
            for k in 0..4 {
                if w.len() < 8 && n > 0 {
                    let at = (n / 5) * (k + 1);
                    w.push(clip[at..at + len].to_vec());
                }
            }
        }
        w
    };
    let fp32_file = dir.join("model.fp32.onnx");
    for ep in gpu_eps() {
        let levers = GpuLevers {
            io_binding: true,
            cuda_graphs: ep == "cuda",
            ..GpuLevers::default()
        };
        for variant in ["fp16", "fp32"] {
            let path = dir.join(format!("model.{variant}.onnx"));
            if !path.is_file() {
                continue;
            }
            let variant: &'static str = if variant == "fp16" { "fp16" } else { "fp32" };
            for len in [7_040usize, 6_720, 7_360, 16_000, 32_000] {
                let ws = windows(len);
                // (cold) the first planned call of a deferred encoder, cross-check off, then on.
                let first_call = |check: bool| -> f64 {
                    if check {
                        std::env::remove_var(CONTENT_CHECK_ENV);
                    } else {
                        std::env::set_var(CONTENT_CHECK_ENV, "0");
                    }
                    let mut enc = ContentEncoder::deferred(EmbedderKind::ContentVec768, &path, variant, &ep, &opts, None).unwrap();
                    let t = Instant::now();
                    let (_, report) = enc.encode_fixed(&ws[0], levers, &mut None, &mut Vec::new()).unwrap();
                    let ms = ms_since(t);
                    std::env::remove_var(CONTENT_CHECK_ENV);
                    let report = report.unwrap();
                    assert!(report.fixed, "{ep} {variant} {len} check {check}: {:?}", report.notes);
                    ms
                };
                let _warm = first_call(false);
                let off_ms = first_call(false);
                let on_ms = first_call(true);
                // (a) / (b) / (c) on every window with the plan the check built.
                let mut enc = ContentEncoder::deferred(EmbedderKind::ContentVec768, &path, variant, &ep, &opts, None).unwrap();
                let mut lease = None;
                let mut got = Vec::new();
                let plan: Vec<Vec<f32>> = ws
                    .iter()
                    .map(|w| {
                        enc.encode_fixed(w, levers, &mut lease, &mut got).unwrap();
                        got.clone()
                    })
                    .collect();
                assert!(lease.as_ref().is_some_and(|l| l.fixed()), "{ep} {variant} {len}: no plan");
                let t = Instant::now();
                let (cpu, build_ms) = cpu_references_for_tests(EmbedderKind::ContentVec768, &path, &opts, &ws).unwrap();
                let cpu_ms = ms_since(t);
                let worst = |a: &[Vec<f32>], b: &[Vec<f32>]| a.iter().zip(b).map(|(x, y)| content::rel_l2(x, y)).fold(0.0f64, f64::max);
                let same_file = worst(&plan, &cpu);
                let fp32_cpu = if variant == "fp16" && fp32_file.is_file() {
                    cpu_references_for_tests(EmbedderKind::ContentVec768, &fp32_file, &opts, &ws)
                        .map(|(r, _)| worst(&plan, &r))
                        .ok()
                } else {
                    None
                };
                let scaled: Vec<Vec<f32>> = plan.iter().map(|x| x.iter().map(|v| v * 1.3).collect()).collect();
                let broken = worst(&scaled, &plan);
                eprintln!(
                    "{ep} {variant} {len}: first planned call check off {off_ms:.0} ms, on {on_ms:.0} ms (+{:.0}) | CPU session build {build_ms:.0} ms, {} windows in {:.0} ms total | worst rel L2 plan vs CPU same file {same_file:.2e}, vs CPU fp32 file {} | a 30 % scaled plan reads {broken:.2e} (limit {PLAN_CPU_CHECK_REL_L2:.0e})",
                    on_ms - off_ms,
                    ws.len(),
                    cpu_ms - build_ms,
                    fp32_cpu.map_or("-".to_string(), |v| format!("{v:.2e}"))
                );
                assert!(
                    same_file <= PLAN_CPU_CHECK_REL_L2 / 2.0,
                    "{ep} {variant} {len}: the plan differs from its CPU session by {same_file:.2e}, over half the check limit {PLAN_CPU_CHECK_REL_L2:.0e}"
                );
            }
        }
    }
}

/// P3B-D precision honesty: fp16 against fp32 ContentVec at the stream's window lengths (0.1 s to
/// 0.5 s, and the 1 s / 2 s content windows) on each GPU EP, on 8 speech windows per length:
/// relative L2 and mean frame cosine of the features against the export gate (fp16: rel L2 <= 1e-2,
/// cosine >= 0.9995). Both run as fixed-shape plans (the engine's GPU path). The export's own
/// DirectML gate failed fp16 on a 6 s input; this measures the lengths the engine runs.
/// `SIGIL_TEST_CONTENT_ENCODER_DIR` (model.fp16.onnx and model.fp32.onnx), `SIGIL_BENCH_EPS`.
#[test]
#[ignore = "dev-rig: needs SIGIL_TEST_CONTENT_ENCODER_DIR and a GPU"]
fn content_fp16_stream_parity() {
    use super::content::{ContentEncoder, EmbedderKind};
    use crate::services::neural_audio::gpu::GpuLevers;
    let Some(dir) = std::env::var_os("SIGIL_TEST_CONTENT_ENCODER_DIR").map(PathBuf::from) else {
        eprintln!("SIGIL_TEST_CONTENT_ENCODER_DIR is not set; skipped");
        return;
    };
    let clips = dev_speech();
    let opts = VcSessionOptions::default();
    let cos_mean = |a: &[f32], b: &[f32], c: usize| -> f64 {
        let frames = a.len() / c;
        (0..frames)
            .map(|t| {
                let (x, y) = (&a[t * c..(t + 1) * c], &b[t * c..(t + 1) * c]);
                let dot: f64 = x.iter().zip(y).map(|(p, q)| f64::from(*p) * f64::from(*q)).sum();
                let nx: f64 = x.iter().map(|p| f64::from(*p).powi(2)).sum::<f64>().sqrt();
                let ny: f64 = y.iter().map(|q| f64::from(*q).powi(2)).sum::<f64>().sqrt();
                dot / (nx * ny + 1e-12)
            })
            .sum::<f64>()
            / frames.max(1) as f64
    };
    for ep in gpu_eps() {
        let levers = GpuLevers {
            io_binding: true,
            cuda_graphs: ep == "cuda",
            ..GpuLevers::default()
        };
        for len in [1_600usize, 3_200, 4_800, 7_040, 8_000, 16_000, 32_000] {
            let mut h = ContentEncoder::load(EmbedderKind::ContentVec768, &dir.join("model.fp16.onnx"), "fp16", &ep, &opts).unwrap();
            let mut f = ContentEncoder::load(EmbedderKind::ContentVec768, &dir.join("model.fp32.onnx"), "fp32", &ep, &opts).unwrap();
            let (mut lh, mut lf) = (None, None);
            let (mut a, mut b) = (Vec::new(), Vec::new());
            let (mut worst_rel, mut worst_cos, mut n) = (0.0f64, 1.0f64, 0usize);
            for (_, clip) in &clips {
                for k in 0..2 {
                    if clip.len() < len * 3 {
                        continue;
                    }
                    let at = (clip.len() - len) / 3 * (k + 1);
                    let w = &clip[at..at + len];
                    h.encode_fixed(w, levers, &mut lh, &mut a).unwrap();
                    f.encode_fixed(w, levers, &mut lf, &mut b).unwrap();
                    worst_rel = worst_rel.max(content::rel_l2(&a, &b));
                    worst_cos = worst_cos.min(cos_mean(&a, &b, 768));
                    n += 1;
                }
            }
            let pass = worst_rel <= 1e-2 && worst_cos >= 0.9995;
            eprintln!(
                "{ep} fp16 vs fp32 at {len} samples ({:.2} s): worst rel L2 {worst_rel:.2e}, worst mean cosine {worst_cos:.5} over {n} windows: {}",
                len as f64 / 16_000.0,
                if pass { "PASS" } else { "FAIL (fp16 gate: rel L2 <= 1e-2, cosine >= 0.9995)" }
            );
        }
    }
}
