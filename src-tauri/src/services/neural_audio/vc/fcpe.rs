//! FCPE ONNX pitch extractor (default for Streamer / CPU).
//!
//! Supported contract: 16 kHz waveform `[1, N]` (or `[N]`), optional scalar `threshold` input,
//! F0 output in Hz `[1, T]` / `[T]` / `[1, T, 1]` on the hop-160 grid. Exports whose audio input
//! is a mel spectrogram are rejected at load (Sigil does not implement torchfcpe's mel front end),
//! so the graph is never fed a mismatched tensor.

use ort::session::Session;

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions};

use super::f0_post::smooth_f0;
use super::pitch_onnx::{flat_f0, waveform_shape, MainInput, PitchInputs, ShapePin};
use super::traits::PitchExtractor;

/// torchfcpe's default voicing threshold, fed to exports that expose a `threshold` input.
const FCPE_THRESHOLD: f32 = 0.006;

pub struct FcpePitchExtractor {
    session: Session,
    inputs: PitchInputs,
    wave_rank: usize,
    /// DirectML: every call runs the first call's length, front-padded with silence in whole
    /// 10 ms frames (`pitch_onnx::ShapePin`); other EPs run each call as it is.
    pin: ShapePin,
    padded: Vec<f32>,
    /// EP outcome of the load (`vc.pitch`).
    outcome: EpOutcome,
}

impl FcpePitchExtractor {
    pub fn load(model_path: &str, execution_provider_hint: &str, opts: &VcSessionOptions) -> Result<Self, String> {
        let path = model_path.trim();
        if path.is_empty() {
            return Err("FCPE model path is empty".into());
        }
        let (session, outcome) = build_session(
            Some("vc.pitch"),
            execution_provider_hint,
            opts,
            ModelSource::File(std::path::Path::new(path)),
        )
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
        Ok(Self {
            session,
            inputs,
            wave_rank,
            pin: ShapePin::for_ep(&outcome.active),
            padded: Vec::new(),
            outcome,
        })
    }

    /// EP outcome of the load (`vc.pitch`), for callers that report where F0 runs.
    pub fn outcome(&self) -> &EpOutcome {
        &self.outcome
    }
}

impl PitchExtractor for FcpePitchExtractor {
    fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
        if mono_16k.is_empty() {
            return Ok(Vec::new());
        }
        let n = mono_16k.len();
        let run_n = self.pin.size_aligned(n, 160);
        let lead_frames = (run_n - n) / 160;
        let audio: &[f32] = if run_n == n {
            mono_16k
        } else {
            self.padded.clear();
            self.padded.resize(run_n - n, 0.0);
            self.padded.extend_from_slice(mono_16k);
            &self.padded
        };
        let shape = waveform_shape(self.wave_rank, run_n);
        let outputs = self.inputs.run(&mut self.session, shape, audio, "FCPE")?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("FCPE output: {}", e))?;
        let dims: Vec<usize> = shape.iter().map(|d| (*d).max(0) as usize).collect();
        let mut f0 = flat_f0(&dims, data).ok_or_else(|| format!("FCPE output shape {dims:?} unsupported; expected F0 [1, T] or [T]"))?;
        if lead_frames > 0 {
            f0.drain(..lead_frames.min(f0.len()));
        }
        for v in f0.iter_mut() {
            if !v.is_finite() || *v < 0.0 {
                *v = 0.0;
            }
        }
        smooth_f0(&mut f0, 3);
        Ok(f0)
    }

    // No `receptive_frames` (the F0 cache's default lead-in, 12 frames). Measured at the P3 final
    // gate: a 20-frame lead-in improves the pitch track against whole-clip F0 (V/UV 3.31 % -> 2.85 %,
    // `engine::f0_cache::tests::lead_sweep_against_whole_clip_f0`), but the converted audio did not
    // follow on the benchmark (`--headless bench` fidelity, 5 clips, the default voice): on CUDA with
    // 40 ms blocks DNSMOS fell 0.06-0.18 on every clip and the speaker cosine by 0.009 on average;
    // on the CPU Streamer preset (80 ms blocks) it was neutral. Revisit with the P7 corpus.
}
