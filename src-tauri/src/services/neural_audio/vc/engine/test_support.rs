//! Shared fixtures of the engine tests (`load_tests`, `tick_tests`; split out of `engine/mod.rs` by
//! P3B-S0). Frozen: the load (P3B-H) and tick (P3B-D) workstreams both use these helpers, so they
//! change only through INTEGRATE; add new helpers to the test file that needs them.

use std::path::PathBuf;

use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::vc::manifest::load_slot_manifest;
use crate::services::neural_audio::vc::test_models;
use crate::services::neural_audio::vc::world::DioPitchExtractor;
pub(super) use crate::services::neural_audio::RvcPresetProfile;

use super::generator::tests::tiny_voice;
pub(super) use super::generator::tests::tiny_dir;
use super::{presets, EffectiveLevers, F0Range, Lever, LoadOptions, PrecisionMode, SigilRvcEngine, SigilRvcRequest};

/// A slot folder holding the tiny fixture voice (imported, with a manifest) and the stand-in
/// content encoder; removed on drop.
pub(super) struct TinySlot {
    pub(super) root: PathBuf,
    pub(super) slot: PathBuf,
    pub(super) encoder: PathBuf,
}

impl Drop for TinySlot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

pub(super) fn tiny_slot(tag: &str) -> TinySlot {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let root = std::env::temp_dir().join(format!("sigil-sve-{tag}-{}-{nanos}", std::process::id()));
    let slot = root.join("voice-changer").join("models").join("tiny");
    std::fs::create_dir_all(&slot).unwrap();
    let voice = tiny_voice();
    // The slot files the importer writes (P3-B's slot store owns the real writer): the weights
    // image, the speaker table and the import record fields the engine checks.
    std::fs::write(slot.join("weights.f32.bin"), &voice.weights.f32_image).unwrap();
    let speakers: Vec<u8> = voice
        .weights
        .speakers
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(slot.join("speakers.f32.bin"), speakers).unwrap();
    let record = serde_json::json!({
        "format": "sigil-rvc-import",
        "template": voice.template.layout.template,
        "weights_id": voice.template.layout.weights.weights_id,
    });
    std::fs::write(slot.join("sigil-import.json"), serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    let manifest = serde_json::json!({
        "format": "sigil-rvc-1",
        "template": voice.template.layout.template,
        "weights": "weights.f32.bin",
        "speakers": "speakers.f32.bin",
        "speaker_count": voice.weights.n_speakers,
        "embedder": "contentvec",
        "vocoder": "hifigan",
        "fp16_ok": true,
        "import_record": "sigil-import.json"
    });
    std::fs::write(slot.join("manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    let encoder = root.join("content.onnx");
    std::fs::write(&encoder, test_models::content_encoder_bytes()).unwrap();
    TinySlot { root, slot, encoder }
}

pub(super) fn levers_all(on: bool) -> EffectiveLevers {
    let mut e = EffectiveLevers {
        precision: PrecisionMode::Fp32,
        ..EffectiveLevers::default()
    };
    for lever in [Lever::PartialDecode, Lever::AnchoredNoise, Lever::NsfPhase0, Lever::F0Cache] {
        e.levers.set(lever, on);
    }
    e
}

pub(super) fn request(s: &TinySlot, levers: EffectiveLevers, block_ms: f64, extra_ms: f64) -> SigilRvcRequest {
    SigilRvcRequest {
        slot_dir: s.slot.clone(),
        manifest: load_slot_manifest(&s.slot).unwrap(),
        templates_dir: tiny_dir(),
        ep: "cpu".into(),
        session: VcSessionOptions::default(),
        levers,
        pitch: Box::new(DioPitchExtractor::new()),
        content_encoder: Some(s.encoder.clone()),
        index: None,
        preset: presets::engine_preset(RvcPresetProfile::LowLatency, block_ms, 0, 48_000, extra_ms, PrecisionMode::Fp32),
        f0_range: F0Range::Speech,
        noise_seed: 1234,
        lookahead_ms: 0.0,
        stream: None,
        content_context_16k: None,
    }
}

pub(super) fn tiny_engine(s: &TinySlot, levers: EffectiveLevers) -> SigilRvcEngine {
    SigilRvcEngine::load_with(request(s, levers, 40.0, 100.0), LoadOptions { allow_any_frame_rate: true }).unwrap()
}

/// A voiced test signal with pauses (16 kHz).
pub(super) fn voice_16k(len: usize) -> Vec<f32> {
    let mut phase = 0.0f64;
    (0..len)
        .map(|i| {
            let t = i as f64 / 16_000.0;
            phase += (140.0 + 40.0 * (t * 1.3).sin()) / 16_000.0;
            let gate = if (t % 1.0) > 0.8 { 0.0 } else { 1.0 };
            (gate * 0.3 * (std::f64::consts::TAU * phase).sin()) as f32
        })
        .collect()
}

/// Spectral flux (L1 of the positive magnitude change between frames) of `x` with a `win`-sample
/// Hann window every `hop` samples: block-boundary clicks show up as spikes.
pub(super) fn spectral_flux(x: &[f32], win: usize, hop: usize) -> Vec<f64> {
    use realfft::RealFftPlanner;
    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(win);
    let (mut buf, mut spec) = (fft.make_input_vec(), fft.make_output_vec());
    let hann: Vec<f32> = (0..win)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / win as f32).cos())
        .collect();
    let mut prev: Vec<f32> = vec![0.0; spec.len()];
    let mut flux = Vec::new();
    let mut i = 0;
    while i + win <= x.len() {
        for (k, b) in buf.iter_mut().enumerate() {
            *b = x[i + k] * hann[k];
        }
        fft.process(&mut buf, &mut spec).unwrap();
        let mut f = 0.0f64;
        for (p, c) in prev.iter_mut().zip(&spec) {
            let m = c.norm();
            f += f64::from((m - *p).max(0.0));
            *p = m;
        }
        flux.push(f);
        i += hop;
    }
    flux
}

pub(super) fn percentile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if v.is_empty() {
        return f64::NAN;
    }
    let k = ((v.len() - 1) as f64 * p).round() as usize;
    v[k]
}
