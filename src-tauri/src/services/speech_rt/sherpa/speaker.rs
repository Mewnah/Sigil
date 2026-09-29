//! Speaker embeddings (`SherpaOnnxSpeakerEmbeddingExtractor`: 3D-Speaker CAM++ / ERes2Net,
//! WeSpeaker, NeMo TitaNet ONNX exports).
//!
//! For the shared voice library (P5) and the benchmark's speaker-similarity score; compare two
//! embeddings with [`cosine_similarity`].

use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::bindings::{SherpaOnnxSpeakerEmbeddingExtractor, SherpaOnnxSpeakerEmbeddingExtractorConfig};
use super::lib::{floats, rate_i32, require_files, threads, CStrArena, CallLock, SherpaRuntime, CPU_PROVIDER, MAX_SAMPLES_PER_CALL};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpeakerEmbeddingConfig {
    pub model: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_threads: Option<u32>,
}

impl SpeakerEmbeddingConfig {
    pub fn validate(&self) -> Result<(), String> {
        require_files("sherpa speaker embedding model", [self.model.as_path()])
    }

    pub(crate) fn to_raw(&self, arena: &mut CStrArena) -> Result<SherpaOnnxSpeakerEmbeddingExtractorConfig, String> {
        Ok(SherpaOnnxSpeakerEmbeddingExtractorConfig {
            model: arena.path(&self.model)?,
            num_threads: threads(self.num_threads),
            debug: 0,
            provider: CPU_PROVIDER.as_ptr(),
        })
    }
}

/// A loaded speaker-embedding model. `Send + Sync`.
pub struct SpeakerEmbeddingExtractor {
    rt: Arc<SherpaRuntime>,
    raw: NonNull<SherpaOnnxSpeakerEmbeddingExtractor>,
    lock: CallLock,
    dim: usize,
}

// SAFETY: no thread affinity; every call holds `lock`.
unsafe impl Send for SpeakerEmbeddingExtractor {}
unsafe impl Sync for SpeakerEmbeddingExtractor {}

impl Drop for SpeakerEmbeddingExtractor {
    fn drop(&mut self) {
        let _g = self.lock.lock();
        // SAFETY: created by SherpaOnnxCreateSpeakerEmbeddingExtractor, destroyed once.
        unsafe { (self.rt.api().SherpaOnnxDestroySpeakerEmbeddingExtractor)(self.raw.as_ptr()) }
    }
}

impl SpeakerEmbeddingExtractor {
    /// Load the model (blocking).
    pub fn new(rt: &Arc<SherpaRuntime>, config: &SpeakerEmbeddingConfig) -> Result<Self, String> {
        config.validate()?;
        let mut arena = CStrArena::default();
        let raw_cfg = config.to_raw(&mut arena)?;
        // SAFETY: `raw_cfg` points into `arena`, alive for the call; the C API copies everything.
        let raw = unsafe { (rt.api().SherpaOnnxCreateSpeakerEmbeddingExtractor)(&raw_cfg) };
        let raw = NonNull::new(raw.cast_mut())
            .ok_or_else(|| format!("sherpa-onnx could not load the speaker embedding model {}", config.model.display()))?;
        // SAFETY: valid extractor.
        let dim = unsafe { (rt.api().SherpaOnnxSpeakerEmbeddingExtractorDim)(raw.as_ptr()) };
        let ex = Self {
            rt: rt.clone(),
            raw,
            lock: CallLock::default(),
            dim: dim.max(0) as usize,
        };
        if ex.dim == 0 {
            return Err(format!("{} reports an empty embedding", config.model.display()));
        }
        Ok(ex)
    }

    /// Embedding size.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The embedding of mono `samples` (blocking). Errors when the clip is too short for the model.
    pub fn embed(&self, sample_rate: u32, samples: &[f32]) -> Result<Vec<f32>, String> {
        let rate = rate_i32(sample_rate)?;
        let api = self.rt.api();
        let _g = self.lock.lock();
        // SAFETY: valid extractor; the stream is created, fed, used and destroyed under the lock;
        // the embedding (`dim` floats) is copied before it is destroyed.
        unsafe {
            let stream = (api.SherpaOnnxSpeakerEmbeddingExtractorCreateStream)(self.raw.as_ptr());
            if stream.is_null() {
                return Err("sherpa-onnx could not create a speaker embedding stream".into());
            }
            for chunk in samples.chunks(MAX_SAMPLES_PER_CALL) {
                (api.SherpaOnnxOnlineStreamAcceptWaveform)(stream, rate, chunk.as_ptr(), chunk.len() as i32);
            }
            (api.SherpaOnnxOnlineStreamInputFinished)(stream);
            let result = if (api.SherpaOnnxSpeakerEmbeddingExtractorIsReady)(self.raw.as_ptr(), stream) == 0 {
                Err("the clip is too short for a speaker embedding".to_string())
            } else {
                let v = (api.SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding)(self.raw.as_ptr(), stream);
                if v.is_null() {
                    Err("sherpa-onnx returned no speaker embedding".to_string())
                } else {
                    let out = floats(v, self.dim as i32);
                    (api.SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding)(v);
                    Ok(out)
                }
            };
            (api.SherpaOnnxDestroyOnlineStream)(stream);
            result
        }
    }
}

/// Cosine similarity of two embeddings (`None` when the sizes differ or a vector is all zeros).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return None;
    }
    Some((dot / (na.sqrt() * nb.sqrt())) as f32)
}
