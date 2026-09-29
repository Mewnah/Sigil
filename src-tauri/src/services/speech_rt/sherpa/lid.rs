//! Spoken language identification (`SherpaOnnxSpokenLanguageIdentification`, Whisper-based).
//!
//! Used by the P2 router when the user leaves the language on auto for engines that need one
//! (Nemotron is worse on auto-detect; Cohere must never see ja/zh/ko). A multilingual Whisper
//! tiny / base encoder + decoder pair is enough: only the language token is decoded.

use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::bindings::{SherpaOnnxSpokenLanguageIdentification, SherpaOnnxSpokenLanguageIdentificationConfig};
use super::lib::{owned_str, rate_i32, require_files, threads, CStrArena, CallLock, SherpaRuntime, CPU_PROVIDER};

/// Whisper encoder / decoder of a multilingual model (not `*.en`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LanguageIdConfig {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    /// Tail padding in frames; `None` = sherpa's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tail_paddings: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_threads: Option<u32>,
}

impl LanguageIdConfig {
    pub fn validate(&self) -> Result<(), String> {
        require_files("sherpa language identification model", [self.encoder.as_path(), self.decoder.as_path()])
    }

    pub(crate) fn to_raw(&self, arena: &mut CStrArena) -> Result<SherpaOnnxSpokenLanguageIdentificationConfig, String> {
        let mut c = SherpaOnnxSpokenLanguageIdentificationConfig::default();
        c.whisper.encoder = arena.path(&self.encoder)?;
        c.whisper.decoder = arena.path(&self.decoder)?;
        c.whisper.tail_paddings = self.tail_paddings.unwrap_or(-1);
        c.num_threads = threads(self.num_threads);
        c.debug = 0;
        c.provider = CPU_PROVIDER.as_ptr();
        Ok(c)
    }
}

/// A loaded language identifier. `Send + Sync`.
pub struct SpokenLanguageId {
    rt: Arc<SherpaRuntime>,
    raw: NonNull<SherpaOnnxSpokenLanguageIdentification>,
    lock: CallLock,
}

// SAFETY: no thread affinity; every call holds `lock`.
unsafe impl Send for SpokenLanguageId {}
unsafe impl Sync for SpokenLanguageId {}

impl Drop for SpokenLanguageId {
    fn drop(&mut self) {
        let _g = self.lock.lock();
        // SAFETY: created by SherpaOnnxCreateSpokenLanguageIdentification, destroyed once.
        unsafe { (self.rt.api().SherpaOnnxDestroySpokenLanguageIdentification)(self.raw.as_ptr()) }
    }
}

impl SpokenLanguageId {
    /// Load the model (blocking).
    pub fn new(rt: &Arc<SherpaRuntime>, config: &LanguageIdConfig) -> Result<Self, String> {
        config.validate()?;
        let mut arena = CStrArena::default();
        let raw_cfg = config.to_raw(&mut arena)?;
        // SAFETY: `raw_cfg` points into `arena`, alive for the call; the C API copies everything.
        let raw = unsafe { (rt.api().SherpaOnnxCreateSpokenLanguageIdentification)(&raw_cfg) };
        let raw = NonNull::new(raw.cast_mut()).ok_or_else(|| {
            format!(
                "sherpa-onnx could not load the language identification model {} (a multilingual Whisper \
                 encoder/decoder pair is required)",
                config.encoder.display()
            )
        })?;
        Ok(Self {
            rt: rt.clone(),
            raw,
            lock: CallLock::default(),
        })
    }

    /// The spoken language of mono `samples` (ISO 639-1 code such as `en`, `ja`, `zh`). Blocking;
    /// a few seconds of speech is enough.
    pub fn identify(&self, sample_rate: u32, samples: &[f32]) -> Result<String, String> {
        // Whisper needs at least one feature frame; a C++ exception on empty input would abort.
        if (samples.len() as u64) * 1000 < u64::from(sample_rate.max(1)) * super::offline::MIN_UTTERANCE_MS {
            return Err("the clip is too short to identify its language".into());
        }
        if samples.len() > i32::MAX as usize {
            return Err("the clip is too long for language identification".into());
        }
        let rate = rate_i32(sample_rate)?;
        let api = self.rt.api();
        let _g = self.lock.lock();
        // SAFETY: valid engine; the stream is created, fed, computed and destroyed under the lock;
        // the result string is copied before it is destroyed.
        unsafe {
            let stream = (api.SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream)(self.raw.as_ptr());
            if stream.is_null() {
                return Err("sherpa-onnx could not create a language identification stream".into());
            }
            (api.SherpaOnnxAcceptWaveformOffline)(stream, rate, samples.as_ptr(), samples.len() as i32);
            let r = (api.SherpaOnnxSpokenLanguageIdentificationCompute)(self.raw.as_ptr(), stream);
            let lang = if r.is_null() { String::new() } else { owned_str((*r).lang) };
            if !r.is_null() {
                (api.SherpaOnnxDestroySpokenLanguageIdentificationResult)(r);
            }
            (api.SherpaOnnxDestroyOfflineStream)(stream);
            let lang = lang.trim().to_lowercase();
            if lang.is_empty() {
                Err("sherpa-onnx returned no language".into())
            } else {
                Ok(lang)
            }
        }
    }
}
