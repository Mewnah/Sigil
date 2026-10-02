//! A cloned voice's conditioning: what Chatterbox's speech encoder computes from a reference clip
//! (`speech_encoder_slim.onnx`, 24 kHz mono in; it resamples, trims to 10 s / 6 s and runs the S3
//! tokenizer, the voice encoder, CAM++ and the prompt mel itself), stored in the shared voice
//! library's cache (`voices/<id>/cache/chatterbox-ort.bin`, the `ReferenceEmbedding` codec):
//!
//! - `audio_features [1, S, H]`: the language model's conditioning (speaker projection and the
//!   perceiver-resampled prompt tokens), prepended to every prompt;
//! - `audio_tokens [1, P]`: the prompt's speech tokens (prepended to the generated ones for the flow);
//! - `speaker_embeddings [1, 192]`: the flow's speaker vector;
//! - `speaker_features [1, F, 80]`: the prompt mel the flow continues from.
//!
//! The cache also names the speech encoder it came from (an empty tensor `model:<fingerprint>`),
//! so a model update recomputes it.

use crate::services::neural_audio::vc::families::reference::{ReferenceEmbedding, ReferenceTensor};

const FEATURES: &str = "audio_features";
const TOKENS: &str = "audio_tokens";
const SPEAKER: &str = "speaker_embeddings";
const PROMPT: &str = "speaker_features";
const MODEL_TAG: &str = "model:";

/// The conditioning of one reference voice (see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct Conditionals {
    /// `[S, hidden]`, row-major.
    pub features: Vec<f32>,
    pub feature_len: usize,
    pub hidden: usize,
    pub tokens: Vec<i64>,
    pub speaker: Vec<f32>,
    /// `[frames, mel]`, row-major.
    pub prompt: Vec<f32>,
    pub prompt_frames: usize,
    pub mel: usize,
}

impl Conditionals {
    /// Check the shapes agree (a cache or encoder output that does not is refused).
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden == 0 || self.feature_len == 0 || self.features.len() != self.feature_len * self.hidden {
            return Err("the voice conditioning has an invalid feature block".into());
        }
        if self.mel == 0 || self.prompt.len() != self.prompt_frames * self.mel {
            return Err("the voice conditioning has an invalid prompt".into());
        }
        if self.speaker.is_empty() {
            return Err("the voice conditioning has no speaker vector".into());
        }
        if self
            .tokens
            .iter()
            .any(|&t| !(0..super::text::SPEECH_VOCAB).contains(&t))
        {
            return Err("the voice conditioning has invalid prompt tokens".into());
        }
        Ok(())
    }

    /// The library cache form, tagged with the speech encoder's `fingerprint`.
    pub fn to_embedding(&self, fingerprint: &str) -> ReferenceEmbedding {
        ReferenceEmbedding {
            voice_id: String::new(),
            backend: String::new(),
            clip_sha256: String::new(),
            tensors: vec![
                ReferenceTensor::new(FEATURES, vec![1, self.feature_len, self.hidden], self.features.clone()),
                // Speech tokens are below 6561, exact in f32.
                ReferenceTensor::new(TOKENS, vec![1, self.tokens.len()], self.tokens.iter().map(|&t| t as f32).collect()),
                ReferenceTensor::new(SPEAKER, vec![1, self.speaker.len()], self.speaker.clone()),
                ReferenceTensor::new(PROMPT, vec![1, self.prompt_frames, self.mel], self.prompt.clone()),
                ReferenceTensor::new(format!("{MODEL_TAG}{fingerprint}"), vec![0], Vec::new()),
            ],
        }
    }

    /// Back from the cache form; `None` when it was made by another speech encoder or is malformed.
    pub fn from_embedding(e: &ReferenceEmbedding, fingerprint: &str) -> Option<Self> {
        let tag = format!("{MODEL_TAG}{fingerprint}");
        e.tensor(&tag)?;
        let f = e.tensor(FEATURES)?;
        let t = e.tensor(TOKENS)?;
        let s = e.tensor(SPEAKER)?;
        let p = e.tensor(PROMPT)?;
        let (feature_len, hidden) = match f.shape.as_slice() {
            [1, n, h] => (*n, *h),
            _ => return None,
        };
        let (prompt_frames, mel) = match p.shape.as_slice() {
            [1, n, m] => (*n, *m),
            _ => return None,
        };
        let mut tokens = Vec::with_capacity(t.data.len());
        for &v in &t.data {
            if !(v.is_finite() && v.fract() == 0.0) {
                return None;
            }
            tokens.push(v as i64);
        }
        let c = Self {
            features: f.data.clone(),
            feature_len,
            hidden,
            tokens,
            speaker: s.data.clone(),
            prompt: p.data.clone(),
            prompt_frames,
            mel,
        };
        c.validate().ok()?;
        Some(c)
    }
}
