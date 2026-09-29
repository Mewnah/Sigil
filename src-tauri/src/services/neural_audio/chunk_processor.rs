//! Classic (DSP) voice chunk pipeline: phase vocoder pitch, device-rate resample, formant tilt.

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
    fn clamp_chunk_frames_respects_allowlist() {
        assert_eq!(clamp_chunk_frames(160), 160);
        assert_eq!(clamp_chunk_frames(48_000), 48_000);
        assert_eq!(clamp_chunk_frames(999_999), DEFAULT_CHUNK_FRAMES);
    }
}
