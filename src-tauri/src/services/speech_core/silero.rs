//! Silero VAD v6.2 on Sigil's ONNX Runtime (CPU, 1 thread, spinning off, stateful 512-sample frames).
//!
//! Model: Silero VAD v6.2, MIT License, Copyright (c) 2020-present Silero Team
//! (<https://github.com/snakers4/silero-vad>). The file `src-tauri/resources/models/silero_vad_v6_2.onnx`
//! is upstream `src/silero_vad/data/silero_vad.onnx` at tag `v6.2` (2,327,524 bytes, sha256
//! `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`). It is embedded with
//! `include_bytes!`, so Whisper and utterance capture gate speech offline with zero setup.
//!
//! I/O contract (checked at load and verified against the embedded model by the tests below):
//! - `input`  f32 `[1, 576]`: the last 64 samples of the previous frame (zeros at stream start)
//!   followed by the new 512-sample 16 kHz frame. Without the 64-sample context v5/v6 models
//!   under-detect speech.
//! - `state`  f32 `[2, 1, 128]`: recurrent state, zeros at stream start, replaced by `stateN`.
//! - `sr`     i64 scalar: 16000.
//! - `output` f32 `[1, 1]`: speech probability of the frame; `stateN` f32 `[2, 1, 128]`.
//!
//! Input and output names are resolved by role (dtype and rank) rather than hard-coded, so a
//! re-export that renames them keeps working; anything that does not fit the contract is rejected
//! with a message.

use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::Tensor;

use super::{SAMPLE_RATE, VAD_FRAME_SAMPLES};
use crate::services::diagnostics::{record_ep_outcome, EpOutcome};
use crate::services::neural_audio::ort_session::{session_builder_for_vc, VcSessionOptions};

/// The embedded Silero VAD v6.2 ONNX model.
pub(crate) const SILERO_MODEL_BYTES: &[u8] = include_bytes!("../../../resources/models/silero_vad_v6_2.onnx");

/// Samples of the previous frame prepended to every frame (16 kHz).
pub const CONTEXT_SAMPLES: usize = 64;
/// Elements of the recurrent state `[2, 1, 128]`.
const STATE_LEN: usize = 2 * 128;
/// `input` length: context + frame.
const INPUT_LEN: usize = CONTEXT_SAMPLES + VAD_FRAME_SAMPLES;

/// Resolved graph names (by role).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SileroIo {
    pub input: String,
    pub state: String,
    pub sr: String,
    pub prob: String,
    pub state_out: String,
}

impl SileroIo {
    /// Match the session's declared inputs/outputs to the Silero roles.
    fn inspect(session: &Session) -> Result<Self, String> {
        let mut input = None;
        let mut state = None;
        let mut sr = None;
        for i in session.inputs() {
            let name = i.name().to_string();
            let rank = i.dtype().tensor_shape().map(|s| s.len());
            match (i.dtype().tensor_type(), rank) {
                (Some(TensorElementType::Float32), Some(2)) if input.is_none() => input = Some(name),
                (Some(TensorElementType::Float32), Some(3)) if state.is_none() => state = Some(name),
                (Some(TensorElementType::Int64), Some(0 | 1)) if sr.is_none() => sr = Some(name),
                other => return Err(format!("Silero VAD: unexpected input '{name}' {other:?}")),
            }
        }
        let mut prob = None;
        let mut state_out = None;
        for o in session.outputs() {
            let name = o.name().to_string();
            let rank = o.dtype().tensor_shape().map(|s| s.len());
            match (o.dtype().tensor_type(), rank) {
                (Some(TensorElementType::Float32), Some(3)) if state_out.is_none() => state_out = Some(name),
                (Some(TensorElementType::Float32), Some(_)) if prob.is_none() => prob = Some(name),
                other => return Err(format!("Silero VAD: unexpected output '{name}' {other:?}")),
            }
        }
        let missing = |what: &str| format!("Silero VAD: the model has no {what}");
        Ok(Self {
            input: input.ok_or_else(|| missing("f32 [1, N] audio input"))?,
            state: state.ok_or_else(|| missing("f32 [2, 1, 128] state input"))?,
            sr: sr.ok_or_else(|| missing("int64 sample-rate input"))?,
            prob: prob.ok_or_else(|| missing("probability output"))?,
            state_out: state_out.ok_or_else(|| missing("state output"))?,
        })
    }
}

/// One stateful Silero VAD stream. Not `Sync`; one per segmenter.
pub struct SileroVad {
    session: Session,
    io: SileroIo,
    /// `[1, 576]`: `[context(64) | frame(512)]`; the context is carried over in place.
    input: Tensor<f32>,
    /// `[2, 1, 128]`.
    state: Tensor<f32>,
    /// Scalar 16000.
    sr: Tensor<i64>,
}

impl SileroVad {
    /// Load the embedded model on the CPU EP (1 intra + 1 inter thread, sequential, no spinning).
    pub fn new() -> Result<Self, String> {
        let opts = VcSessionOptions {
            streamer_profile: true,
            ..VcSessionOptions::default()
        };
        let session = session_builder_for_vc("cpu", &opts)?
            .commit_from_memory(SILERO_MODEL_BYTES)
            .map_err(|e| format!("Silero VAD: failed to load the model: {e}"))?;
        let io = SileroIo::inspect(&session)?;
        let tensor_err = |e: ort::Error| format!("Silero VAD: tensor: {e}");
        let input = Tensor::from_array(([1usize, INPUT_LEN], vec![0f32; INPUT_LEN])).map_err(tensor_err)?;
        let state = Tensor::from_array(([2usize, 1, 128], vec![0f32; STATE_LEN])).map_err(tensor_err)?;
        let sr = Tensor::from_array(((), vec![i64::from(SAMPLE_RATE)])).map_err(tensor_err)?;
        // Silero is CPU-only by design (sub-millisecond per frame); recorded so the diagnostics view
        // lists every ORT session Sigil runs.
        record_ep_outcome("stt.vad", EpOutcome::granted("cpu"));
        Ok(Self {
            session,
            io,
            input,
            state,
            sr,
        })
    }

    /// Names resolved at load.
    #[cfg(test)]
    pub(crate) fn io(&self) -> &SileroIo {
        &self.io
    }

    /// Speech probability (0..1) of one 512-sample 16 kHz frame; advances the recurrent state and
    /// the 64-sample context. Non-finite samples are treated as silence so one bad buffer can never
    /// poison the state for the rest of the stream.
    pub fn predict(&mut self, frame: &[f32]) -> Result<f32, String> {
        if frame.len() != VAD_FRAME_SAMPLES {
            return Err(format!("Silero VAD: frame has {} samples, expected {VAD_FRAME_SAMPLES}", frame.len()));
        }
        {
            let (_, buf) = self.input.extract_tensor_mut();
            // The previous frame's last 64 samples become this frame's context.
            buf.copy_within(VAD_FRAME_SAMPLES..INPUT_LEN, 0);
            for (dst, &s) in buf[CONTEXT_SAMPLES..].iter_mut().zip(frame) {
                *dst = if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 };
            }
        }
        let outputs = self
            .session
            .run(ort::inputs![
                self.io.input.as_str() => &self.input,
                self.io.state.as_str() => &self.state,
                self.io.sr.as_str() => &self.sr,
            ])
            .map_err(|e| format!("Silero VAD: inference failed: {e}"))?;
        let prob = {
            let (_, p) = outputs[self.io.prob.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Silero VAD: probability output: {e}"))?;
            p.first().copied().unwrap_or(0.0)
        };
        let (_, next) = outputs[self.io.state_out.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Silero VAD: state output: {e}"))?;
        if next.len() != STATE_LEN {
            return Err(format!("Silero VAD: state output has {} values, expected {STATE_LEN}", next.len()));
        }
        self.state.extract_tensor_mut().1.copy_from_slice(next);
        Ok(if prob.is_finite() { prob.clamp(0.0, 1.0) } else { 0.0 })
    }

    /// Zero the recurrent state and the context (start of a new, unrelated stream).
    pub fn reset(&mut self) {
        self.state.extract_tensor_mut().1.fill(0.0);
        self.input.extract_tensor_mut().1.fill(0.0);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The SAPI-generated fixture ("Hello there, the weather is lovely today.", 16 kHz mono PCM16).
    pub(crate) fn speech_fixture() -> Vec<f32> {
        let bytes: &[u8] = include_bytes!("testdata/speech_16k.wav");
        let reader = hound::WavReader::new(std::io::Cursor::new(bytes)).expect("fixture wav");
        let spec = reader.spec();
        assert_eq!((spec.sample_rate, spec.channels, spec.bits_per_sample), (16_000, 1, 16));
        reader.into_samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
    }

    fn frame_probs(vad: &mut SileroVad, audio: &[f32]) -> Vec<f32> {
        audio
            .chunks_exact(VAD_FRAME_SAMPLES)
            .map(|f| vad.predict(f).expect("predict"))
            .collect()
    }

    #[test]
    fn io_contract_matches_the_embedded_model() {
        let vad = SileroVad::new().expect("load silero");
        // Upstream names (utils_vad.py OnnxWrapper feeds input/state/sr and reads output, stateN).
        assert_eq!(
            vad.io(),
            &SileroIo {
                input: "input".into(),
                state: "state".into(),
                sr: "sr".into(),
                prob: "output".into(),
                state_out: "stateN".into(),
            }
        );
        // Declared shapes: state [2, B, 128], input [B, N].
        let state = vad.session.inputs().iter().find(|i| i.name() == "state").unwrap();
        let dims: Vec<i64> = state.dtype().tensor_shape().unwrap().to_vec();
        assert_eq!(dims.len(), 3);
        assert_eq!((dims[0], dims[2]), (2, 128));
        let input = vad.session.inputs().iter().find(|i| i.name() == "input").unwrap();
        assert_eq!(input.dtype().tensor_shape().unwrap().len(), 2);
        assert!(crate::services::diagnostics::ep_outcome("stt.vad").is_some_and(|o| o.active == "cpu"));
    }

    #[test]
    fn silence_and_low_noise_score_low_and_speech_scores_high() {
        let mut vad = SileroVad::new().unwrap();
        let silence = vec![0f32; 16_000 * 3];
        assert!(frame_probs(&mut vad, &silence).iter().all(|&p| p < 0.1));

        // -50 dBFS deterministic white-ish noise (LCG): not speech.
        vad.reset();
        let mut seed = 0x1234_5678u32;
        let noise: Vec<f32> = (0..16_000 * 3)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.0063
            })
            .collect();
        assert!(frame_probs(&mut vad, &noise).iter().all(|&p| p < 0.35));

        vad.reset();
        let speech = speech_fixture();
        let probs = frame_probs(&mut vad, &speech);
        let voiced = probs.iter().filter(|&&p| p >= 0.5).count();
        // "Hello there, the weather is lovely today." is ~2.5 s of voiced speech in ~3.3 s.
        assert!(voiced * VAD_FRAME_SAMPLES >= 16_000, "only {voiced} speech frames: {probs:?}");
        assert!(probs.iter().cloned().fold(0.0, f32::max) > 0.9);
    }

    #[test]
    fn state_and_context_are_carried_and_reset() {
        let speech = speech_fixture();
        let mut vad = SileroVad::new().unwrap();
        let a = frame_probs(&mut vad, &speech);
        vad.reset();
        let b = frame_probs(&mut vad, &speech);
        // Deterministic after reset (same stream, same state trajectory).
        assert_eq!(a, b);
        // Stateful: the same frame scores differently with and without the preceding audio.
        vad.reset();
        let mid = speech.len() / VAD_FRAME_SAMPLES / 2 * VAD_FRAME_SAMPLES;
        let cold = vad.predict(&speech[mid..mid + VAD_FRAME_SAMPLES]).unwrap();
        assert_ne!(cold, a[mid / VAD_FRAME_SAMPLES]);
        // Wrong frame sizes are rejected, NaN input is sanitized.
        assert!(vad.predict(&speech[..100]).is_err());
        let nan = vec![f32::NAN; VAD_FRAME_SAMPLES];
        assert!(vad.predict(&nan).unwrap().is_finite());
    }
}
