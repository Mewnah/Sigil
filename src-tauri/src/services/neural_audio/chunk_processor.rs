//! Classic (DSP) voice chunk pipeline: phase vocoder pitch, device-rate resample, formant tilt.
//!
//! P3-D: on the RVC path the one-pole tilt gives way to the true cepstral formant shift of the
//! post stage (`vc::engine::post`, driven by `VoiceControls::formant_semitones`);
//! [`rvc_output_tilt`] says which tilt the worker still applies to converted RVC audio. The classic,
//! ONNX and sidecar modes keep the tilt unchanged.

use std::mem;

use pitch_shift::PitchShifter;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// Default input frames per processing block (per channel) when not overridden.
pub const DEFAULT_CHUNK_FRAMES: usize = 512;

/// Allowed chunk sizes (in frames per channel) for voice processing.
/// Smaller values react faster but run the pipeline more often; larger values add latency per step
/// but can ease load on slower CPUs/GPUs. Starts at 160: smaller sizes can yield empty resampler output.
/// Above 4096, steps are coarser (up to 48000 ≈ 1 s at 48 kHz) for very weak hardware.
pub const ALLOWED_CHUNK_FRAMES: &[u32] = &[
    160, 192, 224, 256, 320, 384, 448, 512, 576, 640, 704, 768, 896, 1024, 1280, 1536, 2048, 2304,
    2560, 2816, 3072, 3584, 4096, 5120, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768, 40960,
    48000,
];

/// Allowed chunk sizes (in frames per channel) for voice processing.
#[inline]
pub fn clamp_chunk_frames(v: u32) -> usize {
    if ALLOWED_CHUNK_FRAMES.contains(&v) {
        v as usize
    } else {
        DEFAULT_CHUNK_FRAMES
    }
}

/// The engine block (in 10 ms frames) a block of `block_ms` plans to: rounded up to whole 10 ms
/// frames, then to a whole number of 20 ms (`engine::plan::block_frames`), at least two.
fn plan_block_frames(block_ms: f64) -> usize {
    ((block_ms / 10.0 - 1e-9).ceil().max(1.0) as usize).div_ceil(2) * 2
}

/// The device block of a template voice under `device_block = model_divisor` (P3B-H): a divisor of
/// the engine block, so every `block / chunk`-th device block completes a model block exactly when its
/// audio is due and the harness holds back only `block - chunk` instead of the burst cushion an
/// unrelated size needs (512 frames against a 40 ms block at 48 kHz cost 7.4 ms more). It is 10 ms at
/// the device rate when that is a whole number of frames (480 at 48 kHz, 441 at 44.1 kHz; the engine
/// block is always a whole number of 20 ms, so 10 ms divides it), else the default 512. `None` when the
/// engine block is derived from the device block (`block_ms` 0: see [`model_divisor_chunk_auto`]) or the
/// rate has no whole-frame 10 ms. (The capture hub does not report the WASAPI period, so no
/// device-reported period is considered.)
pub fn model_divisor_chunk(block_ms: f64, rate: u32) -> Option<usize> {
    if !(block_ms.is_finite() && block_ms > 0.0) || rate == 0 {
        return None;
    }
    let ten_ms = rate.is_multiple_of(100).then_some((rate / 100) as usize)?;
    let block = (plan_block_frames(block_ms) * ten_ms).max(1);
    let divides = |p: usize| p >= 160 && p <= block && block.is_multiple_of(p);
    Some(if divides(ten_ms) { ten_ms } else { DEFAULT_CHUNK_FRAMES })
}

/// [`model_divisor_chunk`] with the engine block on automatic (`block_ms` 0). The engine then derives
/// its block from the device block (`presets::engine_preset`: `chunk_frames` at `rate`, capped at the
/// longest block, planned to whole 20 ms), so a 512-frame chunk at 48 kHz (10.7 ms) runs 20 ms blocks
/// that 512 does not divide, while 480 does. The divisor is taken only when the smaller device block
/// derives the SAME engine block (a 1024-frame chunk runs 40 ms blocks; 10 ms chunks would shrink them
/// to 20 ms, which changes the block, not the alignment): otherwise `None` and the chunk stays.
pub fn model_divisor_chunk_auto(chunk_frames: usize, rate: u32) -> Option<usize> {
    use super::vc::engine::presets::MAX_DERIVED_BLOCK_MS;
    if chunk_frames == 0 || rate == 0 {
        return None;
    }
    let derived = |frames: usize| plan_block_frames((frames as f64 * 1000.0 / f64::from(rate)).min(MAX_DERIVED_BLOCK_MS));
    let chunk = model_divisor_chunk(derived(chunk_frames) as f64 * 10.0, rate)?;
    (derived(chunk) == derived(chunk_frames)).then_some(chunk)
}

const PITCH_EPSILON: f32 = 0.01;

/// One-pole lowpass state per channel (simple formant / brightness tilt).
#[derive(Clone, Default)]
pub struct FormantLpState {
    lp: f32,
}

fn clamp_vocoder_window_ms(v: u32) -> usize {
    v.clamp(30, 60) as usize
}

fn clamp_vocoder_oversample(v: u32) -> usize {
    match v {
        1 | 2 | 4 | 8 | 16 | 32 => v as usize,
        _ => 8usize,
    }
}

fn device_resample_ratio(r_in: u32, r_out: u32) -> f64 {
    r_out as f64 / r_in as f64
}

fn sinc_params() -> SincInterpolationParameters {
    SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        oversampling_factor: 256,
        interpolation: SincInterpolationType::Cubic,
        window: WindowFunction::BlackmanHarris2,
    }
}

fn make_device_resampler(
    ratio: f64,
    channels: usize,
    chunk_frames: usize,
) -> Result<SincFixedIn<f32>, rubato::ResamplerConstructionError> {
    const MAX_REL_RATIO: f64 = 4.0;
    SincFixedIn::new(
        ratio,
        MAX_REL_RATIO,
        sinc_params(),
        chunk_frames,
        channels,
    )
}

/// Fill `planes` (each row `frames` samples) from interleaved input; reuses storage.
fn interleaved_to_planar_into(
    interleaved: &[f32],
    ch_in: usize,
    ch_work: usize,
    planes: &mut [Vec<f32>],
    frames: usize,
) {
    debug_assert_eq!(interleaved.len(), frames * ch_in);
    for row in planes.iter_mut().take(ch_work) {
        if row.len() != frames {
            row.resize(frames, 0.0);
        }
    }
    for f in 0..frames {
        for c in 0..ch_work {
            let v = if c < ch_in {
                interleaved[f * ch_in + c]
            } else {
                interleaved[f * ch_in + ch_in - 1]
            };
            planes[c][f] = v;
        }
    }
}

fn planar_to_interleaved_into(planes: &[Vec<f32>], ch_work: usize, ch_out: usize, out: &mut Vec<f32>) {
    let frames = planes.first().map(|p| p.len()).unwrap_or(0);
    let len = frames * ch_out;
    out.clear();
    out.resize(len, 0.0f32);
    let mut i = 0;
    for f in 0..frames {
        for c in 0..ch_out {
            let v = if c < ch_work {
                planes[c].get(f).copied().unwrap_or(0.0)
            } else {
                planes[ch_work - 1].get(f).copied().unwrap_or(0.0)
            };
            out[i] = v;
            i += 1;
        }
    }
}

fn apply_phase_vocoder_channels(
    shifters: &mut [PitchShifter],
    oversample: usize,
    pitch_semitones: f32,
    planes_in: &[Vec<f32>],
    planes_out: &mut [Vec<f32>],
) {
    let frames = planes_in[0].len();
    let shift = if pitch_semitones.abs() < PITCH_EPSILON {
        0.0f32
    } else {
        pitch_semitones
    };
    for c in 0..planes_in.len() {
        shifters[c].shift_pitch(
            oversample,
            shift,
            &planes_in[c][..frames],
            &mut planes_out[c][..frames],
        );
    }
}

/// Shifts smaller than this (semitones) leave the true formant shift off.
const FORMANT_SEMITONES_EPSILON: f32 = 0.01;

/// The one-pole tilt the worker applies to converted RVC audio (P3-D): none while the true formant
/// shift is on (`formant_semitones`, applied by the model's post stage), else the legacy
/// `formant_shift` tilt, so a voice set up before P3 sounds exactly as it did.
pub fn rvc_output_tilt(formant_shift: f32, formant_semitones: f32) -> f32 {
    if formant_semitones.is_finite() && formant_semitones.abs() >= FORMANT_SEMITONES_EPSILON {
        0.0
    } else {
        formant_shift
    }
}

/// Spectral tilt: negative blends toward lowpassed (darker); positive emphasizes highs.
pub fn apply_formant_interleaved(
    samples: &mut [f32],
    ch: usize,
    formant: f32,
    states: &mut [FormantLpState],
    sample_rate: f64,
) {
    if formant.abs() < 1e-4 || ch == 0 {
        return;
    }
    let fc = 2000.0;
    let alpha = (1.0 - (-2.0 * std::f64::consts::PI * fc / sample_rate).exp()) as f32;
    debug_assert!(states.len() >= ch);
    for frame in samples.chunks_exact_mut(ch) {
        for (sample, st) in frame.iter_mut().zip(states.iter_mut()) {
            let x = *sample;
            st.lp += alpha * (x - st.lp);
            *sample = if formant >= 0.0 {
                x + formant * (x - st.lp)
            } else {
                let w = -formant;
                (1.0 - w) * x + w * st.lp
            };
        }
    }
}

/// Pluggable per-chunk pipeline for mic (or TTS) → output device rate (classic DSP here; ONNX/sidecar conversion runs upstream in the voice-changer plugin, then feeds this processor).
pub trait VoiceChunkPipeline: Send {
    fn process_chunk(
        &mut self,
        interleaved: &[f32],
        pitch_semitones: f32,
        formant_shift: f32,
        vocoder_window_ms: u32,
        vocoder_oversample: u32,
    ) -> Result<Vec<f32>, String>;
}

/// Phase vocoder + rubato resample to output device rate + formant tilt.
pub struct ClassicPhaseVocoderProcessor {
    r_in: u32,
    r_out: u32,
    ch_in: usize,
    ch_work: usize,
    ch_out: usize,
    phase_shifters: Vec<PitchShifter>,
    resampler: SincFixedIn<f32>,
    /// Reused each chunk (deinterleaved input before pitch shift).
    input_planes: Vec<Vec<f32>>,
    pitched_planes: Vec<Vec<f32>>,
    formant_states: Vec<FormantLpState>,
    last_window_ms: usize,
    chunk_frames: usize,
    /// Reused for resampler → interleaved output before returning to caller.
    interleaved_out: Vec<f32>,
}

impl ClassicPhaseVocoderProcessor {
    pub fn new(
        r_in: u32,
        r_out: u32,
        ch_in: usize,
        ch_out: usize,
        vocoder_window_ms: u32,
        chunk_frames: usize,
    ) -> Result<Self, String> {
        if chunk_frames == 0 {
            return Err("chunk_frames must be non-zero".into());
        }
        let ch_work = ch_in.max(ch_out).max(1);
        let window = clamp_vocoder_window_ms(vocoder_window_ms);
        let ratio = device_resample_ratio(r_in, r_out);
        let resampler = make_device_resampler(ratio, ch_work, chunk_frames)
            .map_err(|e| format!("Voice changer resampler: {}", e))?;
        let phase_shifters: Vec<PitchShifter> = (0..ch_work)
            .map(|_| PitchShifter::new(window, r_in as usize))
            .collect();
        let input_planes: Vec<Vec<f32>> = (0..ch_work)
            .map(|_| vec![0.0f32; chunk_frames])
            .collect();
        let pitched_planes: Vec<Vec<f32>> = (0..ch_work)
            .map(|_| vec![0.0f32; chunk_frames])
            .collect();
        let formant_states = vec![FormantLpState::default(); ch_work];

        Ok(Self {
            r_in,
            r_out,
            ch_in,
            ch_work,
            ch_out,
            phase_shifters,
            resampler,
            input_planes,
            pitched_planes,
            formant_states,
            last_window_ms: window,
            chunk_frames,
            interleaved_out: Vec::new(),
        })
    }

    /// `interleaved` must be `chunk_frames * ch_in` samples.
    pub fn process(
        &mut self,
        interleaved: &[f32],
        pitch_semitones: f32,
        formant_shift: f32,
        vocoder_window_ms: u32,
        vocoder_oversample: u32,
    ) -> Result<Vec<f32>, String> {
        let need = self.chunk_frames * self.ch_in;
        if interleaved.len() < need {
            return Err(format!(
                "expected {} samples, got {}",
                need,
                interleaved.len()
            ));
        }
        let interleaved = &interleaved[..need];

        let window_ms = clamp_vocoder_window_ms(vocoder_window_ms);
        if window_ms != self.last_window_ms {
            self.last_window_ms = window_ms;
            self.phase_shifters = (0..self.ch_work)
                .map(|_| PitchShifter::new(window_ms, self.r_in as usize))
                .collect();
        }

        let oversample = clamp_vocoder_oversample(vocoder_oversample);
        interleaved_to_planar_into(
            interleaved,
            self.ch_in,
            self.ch_work,
            &mut self.input_planes,
            self.chunk_frames,
        );
        apply_phase_vocoder_channels(
            &mut self.phase_shifters,
            oversample,
            pitch_semitones,
            &self.input_planes,
            &mut self.pitched_planes,
        );

        match self.resampler.process(&self.pitched_planes, None) {
            Ok(out_planes) => {
                planar_to_interleaved_into(
                    &out_planes,
                    self.ch_work,
                    self.ch_out,
                    &mut self.interleaved_out,
                );
            }
            Err(e) => {
                tracing::warn!("voice changer: classic resample error: {e}");
                planar_to_interleaved_into(
                    &self.pitched_planes,
                    self.ch_work,
                    self.ch_out,
                    &mut self.interleaved_out,
                );
            }
        }

        apply_formant_interleaved(
            &mut self.interleaved_out,
            self.ch_out,
            formant_shift,
            &mut self.formant_states,
            self.r_out as f64,
        );

        let cap = self.interleaved_out.capacity();
        let out = mem::take(&mut self.interleaved_out);
        self.interleaved_out.reserve(cap);
        Ok(out)
    }
}

impl VoiceChunkPipeline for ClassicPhaseVocoderProcessor {
    fn process_chunk(
        &mut self,
        interleaved: &[f32],
        pitch_semitones: f32,
        formant_shift: f32,
        vocoder_window_ms: u32,
        vocoder_oversample: u32,
    ) -> Result<Vec<f32>, String> {
        self.process(
            interleaved,
            pitch_semitones,
            formant_shift,
            vocoder_window_ms,
            vocoder_oversample,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_chunk_pipeline_trait_smoke() {
        let cf = DEFAULT_CHUNK_FRAMES;
        let mut p = ClassicPhaseVocoderProcessor::new(48_000, 48_000, 1, 1, 45, cf).unwrap();
        let chunk = vec![0.01f32; cf];
        let out = VoiceChunkPipeline::process_chunk(&mut p, &chunk, 0.0, 0.0, 45, 8).unwrap();
        // Rubato may emit a slightly different frame count than input for a fixed chunk.
        assert!(!out.is_empty());
        assert!(out.len() <= cf * 2);
    }

    #[test]
    fn phase_vocoder_preserves_buffer_length() {
        let mut shifter = PitchShifter::new(45usize, 48000);
        let len = DEFAULT_CHUNK_FRAMES;
        let in_b: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        let mut out_b = vec![0.0f32; len];
        shifter.shift_pitch(8usize, 5.0f32, &in_b, &mut out_b);
        assert_eq!(in_b.len(), out_b.len());
    }

    #[test]
    fn classic_pipeline_oversample_1_and_2_smoke() {
        let cf = DEFAULT_CHUNK_FRAMES;
        let chunk = vec![0.01f32; cf];
        for os in [1u32, 2u32] {
            let mut p = ClassicPhaseVocoderProcessor::new(48_000, 48_000, 1, 1, 45, cf).unwrap();
            let out = p
                .process(&chunk, 2.0f32, 0.0f32, 45, os)
                .expect("oversample 1/2 should process");
            assert!(!out.is_empty());
            assert!(out.len() <= cf * 2);
        }
    }

    #[test]
    fn classic_pipeline_chunk_256_smoke() {
        let cf = 256usize;
        let mut p = ClassicPhaseVocoderProcessor::new(48_000, 48_000, 1, 1, 45, cf).unwrap();
        let chunk = vec![0.01f32; cf];
        let out = p.process(&chunk, 0.0f32, 0.0f32, 45, 8).unwrap();
        assert!(!out.is_empty());
    }

    #[test]
    fn classic_pipeline_chunk_min_max_allowlist_smoke() {
        for cf_u32 in [160u32, 4096u32, 48_000u32] {
            let cf = cf_u32 as usize;
            let mut p = ClassicPhaseVocoderProcessor::new(48_000, 48_000, 1, 1, 45, cf).unwrap();
            let chunk = vec![0.01f32; cf];
            let out = p.process(&chunk, 1.0f32, 0.0f32, 45, 8).unwrap();
            assert!(!out.is_empty());
        }
    }

    #[test]
    fn rvc_tilt_gives_way_to_the_true_formant_shift() {
        // Legacy voices keep their tilt ...
        assert_eq!(rvc_output_tilt(0.4, 0.0), 0.4);
        assert_eq!(rvc_output_tilt(-0.3, 0.005), -0.3);
        assert_eq!(rvc_output_tilt(0.2, f32::NAN), 0.2);
        // ... and the true shift replaces it (never both).
        assert_eq!(rvc_output_tilt(0.4, 3.0), 0.0);
        assert_eq!(rvc_output_tilt(0.4, -12.0), 0.0);
        // A zero tilt is the bypass of apply_formant_interleaved.
        let mut x = vec![0.1f32, -0.2, 0.3, -0.4];
        let before = x.clone();
        apply_formant_interleaved(&mut x, 1, rvc_output_tilt(0.5, 2.0), &mut [FormantLpState::default()], 48_000.0);
        assert_eq!(x, before);
    }

    #[test]
    fn the_model_divisor_block_divides_the_engine_block() {
        // 40 ms blocks: 10 ms at the device rate, whole frames at 48 / 44.1 / 32 / 16 kHz.
        assert_eq!(model_divisor_chunk(40.0, 48_000), Some(480));
        assert_eq!(model_divisor_chunk(40.0, 44_100), Some(441));
        assert_eq!(model_divisor_chunk(40.0, 32_000), Some(320));
        assert_eq!(model_divisor_chunk(40.0, 16_000), Some(160));
        // Odd 10 ms counts round up to a whole 20 ms like the engine's plan: 30 ms is 40 ms.
        assert_eq!(model_divisor_chunk(30.0, 48_000), Some(480));
        assert_eq!(model_divisor_chunk(20.0, 48_000), Some(480));
        assert_eq!(model_divisor_chunk(200.0, 48_000), Some(480));
        // A rate without a whole-frame 10 ms has no divisor.
        assert_eq!(model_divisor_chunk(40.0, 22_050), None);
        // A derived block (block_ms 0) is the automatic case's.
        assert_eq!(model_divisor_chunk(0.0, 48_000), None);
        assert_eq!(model_divisor_chunk(f64::NAN, 48_000), None);
        for rate in [16_000u32, 32_000, 44_100, 48_000, 96_000] {
            for ms in [20.0, 40.0, 60.0, 80.0, 100.0, 120.0] {
                let chunk = model_divisor_chunk(ms, rate).unwrap();
                let block = (ms / 20.0).ceil() as usize * 2 * (rate / 100) as usize;
                assert!(chunk >= 160 && block.is_multiple_of(chunk), "{ms} ms @ {rate}: chunk {chunk} of {block}");
            }
        }
    }

    /// The default setup (block on automatic, 512-frame chunk) used to ignore `model_divisor` silently:
    /// the engine derives its block from the chunk, so the divisor of "the" block was never looked up.
    #[test]
    fn the_automatic_engine_block_gets_a_divisor_too() {
        // 512 @ 48 kHz (10.7 ms) plans 20 ms blocks; 480 divides them, and 10 ms chunks derive the same
        // 20 ms block.
        assert_eq!(model_divisor_chunk_auto(512, 48_000), Some(480));
        assert_eq!(model_divisor_chunk_auto(512, 44_100), Some(441));
        assert_eq!(model_divisor_chunk_auto(480, 48_000), Some(480));
        assert_eq!(model_divisor_chunk_auto(960, 48_000), Some(480));
        assert_eq!(model_divisor_chunk_auto(160, 48_000), Some(480));
        assert_eq!(model_divisor_chunk_auto(320, 32_000), Some(320));
        // Chunks of 20 ms and more derive longer engine blocks; 10 ms chunks would change the block.
        assert_eq!(model_divisor_chunk_auto(1_024, 48_000), None);
        assert_eq!(model_divisor_chunk_auto(2_048, 48_000), None);
        assert_eq!(model_divisor_chunk_auto(12_288, 48_000), None, "past the 200 ms cap too");
        // No whole-frame 10 ms, or nothing to derive from.
        assert_eq!(model_divisor_chunk_auto(512, 22_050), None);
        assert_eq!(model_divisor_chunk_auto(0, 48_000), None);
        assert_eq!(model_divisor_chunk_auto(512, 0), None);
        // Whatever it returns derives the engine block of the chunk it replaces (the plan's rounding).
        use crate::services::neural_audio::vc::engine::{plan::block_frames, presets::MAX_DERIVED_BLOCK_MS};
        for rate in [16_000u32, 32_000, 44_100, 48_000, 96_000] {
            for chunk in ALLOWED_CHUNK_FRAMES.iter().map(|&c| c as usize) {
                let Some(divisor) = model_divisor_chunk_auto(chunk, rate) else { continue };
                let block = |frames: usize| block_frames((frames as f64 * 1000.0 / f64::from(rate)).min(MAX_DERIVED_BLOCK_MS)).unwrap();
                assert_eq!(block(divisor), block(chunk), "{chunk} @ {rate} -> {divisor}");
                assert!((block(chunk) * 10 * rate as usize / 1000).is_multiple_of(divisor), "{chunk} @ {rate} -> {divisor} divides the block");
            }
        }
    }

    #[test]
    fn clamp_chunk_frames_respects_allowlist() {
        assert_eq!(clamp_chunk_frames(160), 160);
        assert_eq!(clamp_chunk_frames(48_000), 48_000);
        assert_eq!(clamp_chunk_frames(999_999), DEFAULT_CHUNK_FRAMES);
    }
}
