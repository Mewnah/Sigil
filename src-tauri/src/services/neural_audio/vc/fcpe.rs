//! FCPE ONNX pitch extractor (default for Streamer / CPU).
//!
//! Supported contract: 16 kHz waveform `[1, N]` (or `[N]`), optional scalar `threshold` input,
//! F0 output in Hz `[1, T]` / `[T]` / `[1, T, 1]` on the hop-160 grid. Exports whose audio input
//! is a mel spectrogram are rejected at load (Sigil does not implement torchfcpe's mel front end),
//! so the graph is never fed a mismatched tensor.

use ort::session::Session;

use crate::services::neural_audio::ort_session::{session_builder_for_vc, VcSessionOptions};

use super::f0_post::smooth_f0;
use super::pitch_onnx::{flat_f0, waveform_shape, MainInput, PitchInputs};
use super::traits::PitchExtractor;

/// torchfcpe's default voicing threshold, fed to exports that expose a `threshold` input.
const FCPE_THRESHOLD: f32 = 0.006;

pub struct FcpePitchExtractor {
    session: Session,
    inputs: PitchInputs,
    wave_rank: usize,
}

impl FcpePitchExtractor {
    pub fn load(model_path: &str, execution_provider_hint: &str, opts: &VcSessionOptions) -> Result<Self, String> {
        let path = model_path.trim();
        if path.is_empty() {
            return Err("FCPE model path is empty".into());
        }
        let session = session_builder_for_vc(execution_provider_hint, opts)?
            .component("vc.pitch")
            .commit_from_file(path)
            .map_err(|e| format!("Failed to load FCPE ONNX {}: {}", path, e))?;
        let inputs = PitchInputs::inspect(&session, "FCPE", path, FCPE_THRESHOLD)?;
        let wave_rank = match inputs.main {
            MainInput::Waveform { rank } => rank,
            MainInput::LogMel { .. } => {
                return Err(format!(
                    "FCPE ONNX {} expects a mel-spectrogram input {:?}; Sigil feeds a 16 kHz \
                     waveform [1, N]. Use a waveform-input FCPE export, or switch F0 to rmvpe or dio.",
                    path,
                    inputs.main_dims()
                ))
            }
        };
        Ok(Self { session, inputs, wave_rank })
    }
}

impl PitchExtractor for FcpePitchExtractor {
    fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
        if mono_16k.is_empty() {
            return Ok(Vec::new());
        }
        let shape = waveform_shape(self.wave_rank, mono_16k.len());
        let outputs = self
            .inputs
            .run(&mut self.session, shape, mono_16k, "FCPE")?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("FCPE output: {}", e))?;
        let dims: Vec<usize> = shape.iter().map(|d| (*d).max(0) as usize).collect();
        let mut f0 = flat_f0(&dims, data).ok_or_else(|| format!("FCPE output shape {dims:?} unsupported; expected F0 [1, T] or [T]"))?;
        for v in f0.iter_mut() {
            if !v.is_finite() || *v < 0.0 {
                *v = 0.0;
            }
        }
        smooth_f0(&mut f0, 3);
        Ok(f0)
    }
}
