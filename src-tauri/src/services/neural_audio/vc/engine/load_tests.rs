//! Engine load and re-plan tests (split out of `engine/mod.rs` by P3B-S0, unchanged). Owner:
//! P3B-H.

use crate::services::neural_audio::vc::manifest::load_slot_manifest;
use crate::services::neural_audio::vc::stream::VcStreamHarness;
use crate::services::neural_audio::vc::traits::{ConversionCtx, VoiceConversionModel};

use super::load::{generator_shape, PitchBuilder};
use super::plan::{ShapePlan, HOP_16K};
use super::test_support::*;
use super::{EmbedderKind, Lever, LoadOptions, SigilRvcEngine};

/// The production loader refuses the toy fixture template (its frames are not 10 ms); the test
/// loader runs it. Everything the engine reports is filled in.
#[test]
fn tiny_voice_loads_with_its_levers_and_outcomes() {
    let s = tiny_slot("load");
    let err = SigilRvcEngine::load(request(&s, levers_all(true), 40.0, 100.0))
        .err()
        .unwrap();
    assert!(err.contains("frames per second"), "{err}");
    let e = tiny_engine(&s, levers_all(true));
    assert_eq!(e.output_sample_rate(), 800);
    assert_eq!(e.fixed_block(), Some(640));
    assert_eq!(e.embedder(), EmbedderKind::ContentVec768);
    assert_eq!(e.flowdec_precision(), "fp32");
    let comps: Vec<&str> = e.ep_outcomes().iter().map(|(c, _)| c.as_str()).collect();
    assert_eq!(comps, ["vc.enc", "vc.flowdec", "vc.content"]);
    assert!(e.ep_outcomes().iter().all(|(_, o)| o.active == "cpu"));
    for lever in [Lever::PartialDecode, Lever::AnchoredNoise, Lever::NsfPhase0, Lever::F0Cache] {
        assert!(e.effective_levers().levers.get(lever), "{}", lever.as_str());
    }
    assert!(e.warnings().is_empty(), "{:?}", e.warnings());
    let p = *e.plan();
    assert_eq!((p.block_16k, p.partial_decode), (640, true));
    assert!(p.enc_frames >= p.flow_frames);
    // A lever the build resolved on but the plans cannot run is reported off with a note.
    let mut l = levers_all(true);
    l.levers.set(Lever::CudaGraphs, true);
    let e = tiny_engine(&s, l);
    assert!(!e.effective_levers().levers.get(Lever::CudaGraphs));
    assert!(e
        .effective_levers()
        .notes
        .iter()
        .any(|n| n.starts_with("cuda_graphs:")));
}

/// A re-plan rebuilds the generator only for new tensor shapes, and afterwards the report the
/// factory reads (`stream_report`) carries the outcomes and levers of the sessions that run; a
/// shape the template cannot build leaves the old sessions running.
#[test]
fn replans_rebuild_only_new_shapes_and_refresh_the_report() {
    let s = tiny_slot("replan");
    let mut l = levers_all(true);
    l.levers.set(Lever::CudaGraphs, true);
    let mut e = tiny_engine(&s, l);
    let before = *e.plan();
    assert_eq!(e.generator_builds, 1);
    // Another SOLA tail within the same frames: same sessions.
    let want = before.return_frames * HOP_16K - 1;
    assert!(want > before.want_16k);
    e.configure_stream(640, want, 1_600).unwrap();
    assert_eq!((e.generator_builds, e.plan().want_16k), (1, want));
    assert_eq!(generator_shape(e.plan()), generator_shape(&before));
    // A longer tail: new shapes, one rebuild, a report of what runs now.
    e.configure_stream(640, 640 + 2_000, 1_600).unwrap();
    assert_eq!(e.generator_builds, 2);
    let r = e.stream_report().expect("the engine reports");
    let comps: Vec<&str> = r.outcomes.iter().map(|(c, _)| c.as_str()).collect();
    assert_eq!(comps, ["vc.enc", "vc.flowdec", "vc.content"]);
    assert_eq!(&r.levers, e.effective_levers());
    assert!(!r.levers.levers.get(Lever::CudaGraphs));
    let notes = r.levers.notes.iter().filter(|n| n.starts_with("cuda_graphs:")).count();
    assert_eq!(notes, 1, "{:?}", r.levers.notes);
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
    assert_eq!(e.convert(&ctx).unwrap().len(), p.out_samples());
    // A plan at another output rate cannot be built for this template: the error is returned
    // and the old sessions keep running at the old plan.
    let streaming = e.template.layout.streaming.clone();
    let bad = ShapePlan::for_stream(&streaming, e.output_sample_rate + 100, true, 640, p.want_16k, 1_600).unwrap();
    assert!(e.replan(bad).is_err());
    assert_eq!(*e.plan(), p);
    assert!(e.generator.is_some());
    assert_eq!(e.convert(&ctx).unwrap().len(), p.out_samples());
}

/// The harness's real SOLA tail and extra context resize the plan (and rebuild the sessions)
/// when they differ from the preset's defaults; a block other than the plan's is refused.
#[test]
fn configure_stream_replans_for_the_harness_shape() {
    let s = tiny_slot("configure");
    let mut e = tiny_engine(&s, levers_all(true));
    let before = *e.plan();
    let same = e.configure_stream(640, before.want_16k, 1_600).unwrap();
    assert_eq!(same, before.context_16k);
    assert_eq!(*e.plan(), before, "an unchanged shape does not rebuild");
    let longer = e.configure_stream(640, 640 + 2_000, 4_800).unwrap();
    let after = *e.plan();
    assert_eq!(longer, after.context_16k);
    assert!(after.return_frames > before.return_frames);
    assert!(after.context_16k >= 4_800 + 640);
    let audio = voice_16k(after.context_16k);
    let ctx = ConversionCtx {
        mono_16k: &audio,
        new_frames: after.want_16k,
        pitch_shift_semitones: 2.0,
        index_rate: 0.0,
        protect: 0.33,
        speaker_id: 2,
        stream_pos_16k: 0,
        reference: None,
        singing: Default::default(),
    };
    assert_eq!(e.convert(&ctx).unwrap().len(), after.out_samples());
    assert!(e.configure_stream(320, 1_000, 0).is_err());
    // A one-off conversion at another shape is cut / padded to the plan (fewer samples at most).
    let short = voice_16k(3_000);
    let ctx = ConversionCtx {
        mono_16k: &short,
        new_frames: 1_000,
        ..ctx
    };
    let out = e.convert(&ctx).unwrap();
    assert_eq!(out.len(), (1_000 * 800 + 8_000) / 16_000);
}

/// Lookahead grows the decoded span by its frames and returns the same number of samples,
/// through the harness too (the harness sizes its context from the replanned engine).
#[test]
fn lookahead_decodes_future_context_and_keeps_lengths() {
    let s = tiny_slot("lookahead");
    let mut e = tiny_engine(&s, levers_all(true));
    let base = *e.plan();
    assert_eq!(base.lookahead_frames, 0);
    assert!(e.set_lookahead_ms(90.0).is_err(), "at most 80 ms");
    e.set_lookahead_ms(20.0).unwrap();
    let p = *e.plan();
    assert_eq!(p.lookahead_frames, 2);
    assert_eq!(p.return_frames, base.return_frames + 2);
    assert_eq!(p.out_samples(), base.out_samples());
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
    assert_eq!(e.convert(&ctx).unwrap().len(), p.out_samples());
    let mut h = VcStreamHarness::new(Box::new(e), 48_000, 48_000, 1, 1, 1_600, 320, 1_920).unwrap();
    let input: Vec<f32> = voice_16k(16_000).iter().flat_map(|&x| [x, x, x]).collect();
    let mut total = 0;
    for chunk in input.as_chunks::<1_920>().0 {
        total += h.push_and_convert(chunk).unwrap().len();
    }
    assert_eq!(total, input.len());
}

/// Slots that do not match their template are refused with a reason, before any session runs.
#[test]
fn mismatched_slots_are_refused() {
    let s = tiny_slot("refuse");
    let load = |s: &TinySlot| SigilRvcEngine::load_with(request(s, levers_all(true), 40.0, 100.0), LoadOptions { allow_any_frame_rate: true });
    // An import record from another export of the template.
    let record = s.slot.join("sigil-import.json");
    let original = std::fs::read(&record).unwrap();
    let mut v: serde_json::Value = serde_json::from_slice(&original).unwrap();
    v["weights_id"] = serde_json::json!("0".repeat(64));
    std::fs::write(&record, serde_json::to_vec(&v).unwrap()).unwrap();
    let err = load(&s).err().unwrap();
    assert!(err.contains("import the voice again"), "{err}");
    std::fs::write(&record, &original).unwrap();
    // A truncated weights file.
    let weights = s.slot.join("weights.f32.bin");
    let full = std::fs::read(&weights).unwrap();
    std::fs::write(&weights, &full[..full.len() - 64]).unwrap();
    let err = load(&s).err().unwrap();
    assert!(err.contains("bytes"), "{err}");
    std::fs::write(&weights, &full).unwrap();
    // A file name that leaves the slot.
    let mut m = load_slot_manifest(&s.slot).unwrap();
    m.weights = Some("../weights.f32.bin".into());
    let mut req = request(&s, levers_all(true), 40.0, 100.0);
    req.manifest = m;
    let err = SigilRvcEngine::load_with(req, LoadOptions { allow_any_frame_rate: true })
        .err()
        .unwrap();
    assert!(err.contains("plain file name"), "{err}");
    // An embedder Sigil does not have.
    let mut req = request(&s, levers_all(true), 40.0, 100.0);
    req.manifest.embedder = Some("wavlm".into());
    let err = SigilRvcEngine::load_with(req, LoadOptions { allow_any_frame_rate: true })
        .err()
        .unwrap();
    assert!(err.contains("wavlm"), "{err}");
    // And the intact slot still loads.
    assert!(load(&s).is_ok());
}

// ---- Final shapes and the parallel cold start (P3B-H) ----------------------------------------

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use super::load::{LoadParams, CANCELLED};
use super::StreamGeometry;
use crate::services::neural_audio::vc::stream::{sola_want_16k, HarnessOptions, SolaSlack};
use crate::services::neural_audio::vc::world::DioPitchExtractor;

fn tiny_load(s: &TinySlot, req: super::SigilRvcRequest, params: LoadParams<'_>) -> Result<SigilRvcEngine, String> {
    let _ = s;
    SigilRvcEngine::load_full(req, LoadOptions { allow_any_frame_rate: true }, params)
}

/// The harness's geometry for a 10 ms crossfade on the fixture's 800 Hz output.
fn geometry(block_16k: usize, crossfade_ms: f64, extra_16k: usize) -> StreamGeometry {
    StreamGeometry {
        want_16k: sola_want_16k(block_16k, crossfade_ms, 800, HarnessOptions::default().with_env()),
        harness_extra_16k: extra_16k,
        crossfade_ms,
    }
}

/// A lookahead and a crossfade other than 20 ms are planned in the load: one generator build where
/// P5 built two, the harness's `configure_stream` finds the plan final, and the plan is the one the
/// old flow (load, then `set_lookahead_ms`, then `configure_stream`) ended on.
#[test]
fn the_load_plans_the_final_shapes_with_one_generator_build() {
    let s = tiny_slot("final-shapes");
    let g = geometry(640, 10.0, 1_600);
    // The old flow, step by step.
    let mut old = SigilRvcEngine::load_with(request(&s, levers_all(true), 40.0, 100.0), LoadOptions { allow_any_frame_rate: true }).unwrap();
    old.set_lookahead_ms(20.0).unwrap();
    old.configure_stream(640, g.want_16k, g.harness_extra_16k).unwrap();
    assert!(old.generator_builds >= 2, "P5's flow rebuilt the generator ({})", old.generator_builds);
    // The new flow: one build, then configure_stream changes nothing.
    let mut req = request(&s, levers_all(true), 40.0, 100.0);
    req.lookahead_ms = 20.0;
    req.stream = Some(g);
    let mut e = tiny_load(&s, req, LoadParams::default()).unwrap();
    assert_eq!(e.generator_builds, 1);
    assert_eq!(*e.plan(), *old.plan(), "the same final plan");
    assert_eq!(e.plan().lookahead_frames, 2);
    assert_eq!(e.plan().want_16k, g.want_16k);
    let context = e.configure_stream(640, g.want_16k, g.harness_extra_16k).unwrap();
    assert_eq!((e.generator_builds, context), (1, e.plan().context_16k), "configure_stream rebuilt nothing");
    // And the voice converts at that shape through the harness.
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
    assert_eq!(e.convert(&ctx).unwrap().len(), p.out_samples());
    // The kill switch (P5's flow): the request's lookahead and geometry are ignored by the load.
    let mut req = request(&s, levers_all(true), 40.0, 100.0);
    req.lookahead_ms = 20.0;
    req.stream = Some(g);
    let e = tiny_load(&s, req, LoadParams { final_shapes: Some(false), ..LoadParams::default() }).unwrap();
    assert_eq!((e.plan().lookahead_frames, e.generator_builds), (0, 1));
    assert_ne!(e.plan().want_16k, g.want_16k);
    // A stream geometry whose tail is on the frame grid plans one frame less than P3's.
    let aligned = HarnessOptions { sola_slack: Some(SolaSlack::Aligned), ..HarnessOptions::default() };
    let legacy = HarnessOptions { sola_slack: Some(SolaSlack::Legacy), ..HarnessOptions::default() };
    let (a, l) = (sola_want_16k(640, 20.0, 800, aligned), sola_want_16k(640, 20.0, 800, legacy));
    assert!(a < l && a.div_ceil(160) < l.div_ceil(160), "{a} vs {l}");
}

/// Parallel and sequential loads produce the same voice: the same plan, outcomes, levers and bit-
/// identical audio, with or without a pitch builder, and the pitch builder gets the sizes of the
/// final plan.
#[test]
fn parallel_and_sequential_loads_are_the_same_voice() {
    let s = tiny_slot("parallel");
    let render = |e: &mut SigilRvcEngine| -> Vec<f32> {
        let p = *e.plan();
        let audio = voice_16k(p.context_16k + 3 * p.block_16k);
        let mut out = Vec::new();
        for k in 0..3 {
            let start = k * p.block_16k;
            let ctx = ConversionCtx {
                mono_16k: &audio[start..start + p.context_16k],
                new_frames: p.want_16k,
                pitch_shift_semitones: 0.0,
                index_rate: 0.0,
                protect: 0.33,
                speaker_id: 0,
                stream_pos_16k: start as u64,
                reference: None,
                singing: Default::default(),
            };
            out.extend(e.convert(&ctx).unwrap());
        }
        out
    };
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut engines = Vec::new();
    for parallel in [false, true] {
        let mut req = request(&s, levers_all(true), 40.0, 100.0);
        req.lookahead_ms = 10.0;
        req.pitch = SigilRvcEngine::pending_pitch();
        let log = seen.clone();
        let builder: PitchBuilder<'_> = Box::new(move |f0_frames, window| {
            log.lock().unwrap().push((f0_frames, window));
            Ok(Box::new(DioPitchExtractor::new()))
        });
        let e = tiny_load(
            &s,
            req,
            LoadParams {
                pitch: Some(builder),
                parallel: Some(parallel),
                ..LoadParams::default()
            },
        )
        .unwrap();
        engines.push(e);
    }
    let (mut a, mut b) = {
        let b = engines.pop().unwrap();
        let a = engines.pop().unwrap();
        (a, b)
    };
    assert_eq!(*a.plan(), *b.plan());
    assert_eq!(a.generator_builds, b.generator_builds);
    let comps = |e: &SigilRvcEngine| e.ep_outcomes().iter().map(|(c, o)| (c.clone(), o.active.clone())).collect::<Vec<_>>();
    assert_eq!(comps(&a), comps(&b));
    assert_eq!(a.effective_levers(), b.effective_levers());
    assert_eq!(a.warnings(), b.warnings());
    // The builder saw the final plan's sizes: one block + the cache margin and lead, and the window.
    let p = *a.plan();
    let expect = (p.block_16k / HOP_16K + p.f0_margin_frames + p.f0_lead_frames, p.context_16k);
    assert_eq!(*seen.lock().unwrap(), vec![expect, expect]);
    let (ya, yb) = (render(&mut a), render(&mut b));
    assert_eq!(ya.len(), yb.len());
    assert!(ya.iter().zip(&yb).all(|(x, y)| x.to_bits() == y.to_bits()), "the audio differs between the load orders");
    assert!(ya.iter().any(|v| v.abs() > 1e-6), "the voice is not silent");
}

/// A load nobody wants any more stops at its next stage: before it starts, and at the join after
/// its branches (a flag set while the pitch bank builds), in either load order. Nothing leaks: the
/// next load of the same voice works.
#[test]
fn a_cancelled_load_stops_at_each_join() {
    let s = tiny_slot("cancel");
    for parallel in [false, true] {
        // Called off before it starts.
        let flag = AtomicBool::new(true);
        let err = tiny_load(
            &s,
            request(&s, levers_all(true), 40.0, 100.0),
            LoadParams { cancel: Some(&flag), parallel: Some(parallel), ..LoadParams::default() },
        )
        .err()
        .unwrap();
        assert_eq!(err, CANCELLED);
        // Called off while the pitch bank builds: the join notices.
        let flag = AtomicBool::new(false);
        let builds = AtomicUsize::new(0);
        let mut req = request(&s, levers_all(true), 40.0, 100.0);
        req.pitch = SigilRvcEngine::pending_pitch();
        let (flag_ref, builds_ref) = (&flag, &builds);
        let builder: PitchBuilder<'_> = Box::new(move |_, _| {
            builds_ref.fetch_add(1, Ordering::SeqCst);
            flag_ref.store(true, Ordering::SeqCst);
            Ok(Box::new(DioPitchExtractor::new()))
        });
        let err = tiny_load(
            &s,
            req,
            LoadParams { pitch: Some(builder), cancel: Some(&flag), parallel: Some(parallel), ..LoadParams::default() },
        )
        .err()
        .unwrap();
        assert_eq!(err, CANCELLED, "parallel {parallel}");
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        // A failing branch reports its own error (P5's order: the pitch bank first).
        let mut req = request(&s, levers_all(true), 40.0, 100.0);
        req.pitch = SigilRvcEngine::pending_pitch();
        let builder: PitchBuilder<'_> = Box::new(|_, _| Err("rmvpe is not installed".into()));
        let err = tiny_load(&s, req, LoadParams { pitch: Some(builder), parallel: Some(parallel), ..LoadParams::default() })
            .err()
            .unwrap();
        assert!(err.contains("rmvpe is not installed") && err.contains("voice "), "{err}");
    }
    // The pool and the slot are intact.
    assert!(tiny_load(&s, request(&s, levers_all(true), 40.0, 100.0), LoadParams::default()).is_ok());
}

/// The factory's load is followed by the harness's own warm-up ticks, so the load runs one
/// validation conversion; every other caller keeps two.
#[test]
fn the_load_validates_with_one_conversion_when_the_harness_warms_next() {
    let s = tiny_slot("warm-passes");
    let ticks = |e: &SigilRvcEngine| e.tick;
    let one = tiny_load(&s, request(&s, levers_all(true), 40.0, 100.0), LoadParams { harness_warmup_follows: true, ..LoadParams::default() }).unwrap();
    let two = tiny_load(&s, request(&s, levers_all(true), 40.0, 100.0), LoadParams::default()).unwrap();
    assert_eq!((ticks(&one), ticks(&two)), (1, 2));
}
