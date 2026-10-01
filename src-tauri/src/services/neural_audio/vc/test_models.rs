//! Test-only fake RVC slot: a content-encoder stand-in and an f0 generator stand-in built with
//! `neural_audio::test_onnx`, so the real [`RvcModel`] path (session factory and EP ladder, HuBERT
//! run, feature upsampling, pitch, generator feed by role, output trim) runs on every execution
//! provider in `cargo test` without real voice models.
//!
//! - Content encoder: `source` f32 `[1, n]` → `feats` f32 `[1, T, 768]` through a real `Conv`
//!   (kernel 400, stride 320, like HuBERT's first layer; cuDNN on CUDA), `T = (n - 400) / 320 + 1`.
//! - Generator: RVC v2 f0 inputs (`phone`, `phone_lengths`, `pitch`, `pitchf`, `ds`, `rnd`) →
//!   `audio` f32 `[1, 1, t * 400]` = `0.001 * pitchf` held for 400 samples per 10 ms frame (40 kHz),
//!   so a voiced input gives a non-silent staircase that follows F0.

use std::f32::consts::PI;
use std::path::PathBuf;

use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT, INT64};

use super::rvc::RvcModel;
use super::traits::{ConversionCtx, VoiceConversionModel};
use super::world::DioPitchExtractor;

/// Content feature width (RVC v2).
pub const EMB: usize = 768;
/// Output samples per 10 ms frame at 40 kHz.
pub const HOP: usize = 400;
/// Generator output gain on `pitchf` (Hz).
pub const GAIN: f32 = 1e-3;

/// The content-encoder stand-in.
pub fn content_encoder_bytes() -> Vec<u8> {
    let w: Vec<f32> = (0..EMB * 400).map(|i| ((i % 7) as f32 - 3.0) * 1e-3).collect();
    let mut g = Graph::new();
    g.input("source", FLOAT, &[Dim::Fixed(1), Dim::Named("n")])
        .output("feats", FLOAT, &[Dim::Fixed(1), Dim::Named("t"), Dim::Fixed(EMB as u64)])
        .init_i64("axes1", &[1], &[1])
        .init_f32("conv_w", &[EMB as u64, 1, 400], &w)
        .node("Unsqueeze", &["source", "axes1"], &["x3"], &[])
        .node(
            "Conv",
            &["x3", "conv_w"],
            &["c"],
            &[Attr::Ints("strides", vec![320]), Attr::Ints("kernel_shape", vec![400]), Attr::Int("group", 1)],
        )
        .node("Transpose", &["c"], &["feats"], &[Attr::Ints("perm", vec![0, 2, 1])]);
    g.model_bytes(13)
}

/// The generator stand-in (RVC v2 with f0).
pub fn generator_bytes() -> Vec<u8> {
    let mut g = Graph::new();
    g.input("phone", FLOAT, &[Dim::Fixed(1), Dim::Named("t"), Dim::Fixed(EMB as u64)])
        .input("phone_lengths", INT64, &[Dim::Fixed(1)])
        .input("pitch", INT64, &[Dim::Fixed(1), Dim::Named("t")])
        .input("pitchf", FLOAT, &[Dim::Fixed(1), Dim::Named("t")])
        .input("ds", INT64, &[Dim::Fixed(1)])
        .input("rnd", FLOAT, &[Dim::Fixed(1), Dim::Fixed(192), Dim::Named("t")])
        .output("audio", FLOAT, &[Dim::Fixed(1), Dim::Fixed(1), Dim::Named("s")])
        .init_i64("axes2", &[1], &[2])
        .init_i64("repeats", &[3], &[1, 1, HOP as i64])
        .init_i64("flat", &[3], &[1, 1, -1])
        .init_f32("gain", &[1], &[GAIN])
        .node("Unsqueeze", &["pitchf", "axes2"], &["p3"], &[])
        .node("Tile", &["p3", "repeats"], &["tiled"], &[])
        .node("Reshape", &["tiled", "flat"], &["r"], &[])
        .node("Mul", &["r", "gain"], &["audio"], &[]);
    g.model_bytes(13)
}

/// Write the fake slot (`<root>/models/fake/fake_v2_40k.onnx`, no manifest.json) and the content
/// encoder (`<root>/base/hubert_base.onnx`) into a fresh temp folder; returns `(root, slot, hubert)`.
pub fn write_slot(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
    let root = std::env::temp_dir().join(format!("sigil-fake-rvc-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let slot = root.join("models").join("fake");
    let base = root.join("base");
    std::fs::create_dir_all(&slot).expect("slot dir");
    std::fs::create_dir_all(&base).expect("base dir");
    std::fs::write(slot.join("fake_v2_40k.onnx"), generator_bytes()).expect("generator");
    let hubert = base.join("hubert_base.onnx");
    std::fs::write(&hubert, content_encoder_bytes()).expect("content encoder");
    (root, slot, hubert)
}

/// Load the fake slot on `ep` and convert 100 ms of a 150 Hz tone (1 s of context).
pub fn convert_tone_on(ep: &str, tag: &str) -> Vec<f32> {
    let (root, slot, hubert) = write_slot(tag);
    let mut model = RvcModel::load(
        &slot,
        hubert.to_str().unwrap(),
        Box::new(DioPitchExtractor::new()),
        ep,
        &VcSessionOptions::default(),
        false,
    )
    .expect("the fake RVC slot loads");
    assert_eq!(model.output_sample_rate(), 40_000);
    let audio: Vec<f32> = (0..16_000).map(|i| 0.3 * (2.0 * PI * 150.0 * i as f32 / 16_000.0).sin()).collect();
    let ctx = ConversionCtx {
        mono_16k: &audio,
        new_frames: 1_600,
        pitch_shift_semitones: 0.0,
        index_rate: 0.0,
        protect: 0.33,
        speaker_id: 0,
        stream_pos_16k: 0,
    };
    let out = model.convert(&ctx).expect("conversion");
    drop(model);
    let _ = std::fs::remove_dir_all(&root);
    out
}

fn assert_tone_output(out: &[f32]) {
    assert_eq!(out.len(), 4_000, "100 ms at 40 kHz");
    assert!(out.iter().all(|x| x.is_finite()));
    // Voiced 150 Hz: the generator emits 0.001 * F0 ≈ 0.15 (dio may leave edge frames unvoiced).
    let loud = out.iter().filter(|x| (**x - 0.15).abs() < 0.05).count();
    assert!(loud > out.len() / 2, "{loud} of {} samples follow the tone's F0", out.len());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::diagnostics::ep_outcome;

    #[test]
    fn fake_rvc_slot_converts_on_the_cpu() {
        assert_tone_output(&convert_tone_on("cpu", "cpu"));
    }

    /// DirectML where the PC has it (the dev rig and most Windows PCs), else the ladder's CPU
    /// fallback; either way the RVC path must convert.
    #[cfg(windows)]
    #[test]
    fn fake_rvc_slot_converts_on_directml_or_its_fallback() {
        let out = convert_tone_on("dml", "dml");
        eprintln!("fake RVC on dml → {:?}", ep_outcome("vc"));
        assert_tone_output(&out);
    }

    /// Manual CUDA 13 smoke of the RVC path (see `ort_session::cuda13_runtime_smoke` for the env
    /// vars): `cargo test -- --ignored fake_rvc_slot_converts_on_cuda13 --nocapture`
    #[test]
    #[ignore = "needs an NVIDIA GPU and CUDA 13 DLLs (SIGIL_TEST_CUDA_BIN)"]
    fn fake_rvc_slot_converts_on_cuda13() {
        if !crate::services::neural_audio::ort_session::preload_test_cuda_runtime() {
            return;
        }
        let out = convert_tone_on("cuda", "cuda");
        let outcome = ep_outcome("vc").expect("recorded");
        eprintln!("fake RVC on cuda → {outcome:?}; content → {:?}", ep_outcome("vc.content"));
        assert_eq!(outcome.active, "cuda", "{outcome:?}");
        assert_tone_output(&out);
    }
}
