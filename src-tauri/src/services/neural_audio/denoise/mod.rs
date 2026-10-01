//! Input denoisers on the voice-changer path only (P3-E; research §4.3#8): GTCRN at 16 kHz inside
//! the stream harness (lowest CPU), DeepFilterNet3 / DPDFNet at 48 kHz on the device-rate
//! conditioned copy before the aux (TTS) mix. STT keeps the raw capture feed; with the denoiser
//! off, end-to-end latency must be unchanged (±2 ms).
//!
//! - [`stft`]: the streaming STFT shared by the models (constant delay for any call length).
//! - [`gtcrn`]: GTCRN (16 kHz, about 0.5 MB, 16-32 ms of latency, no model lookahead).
//! - [`dfn`]: DPDFNet-2 48 kHz HR (about 50 ms of latency including its 40 ms lookahead);
//!   DeepFilterNet3 is not streamable through ONNX Runtime and runs DPDFNet ([`effective_kind`]).
//! - [`harness`]: [`harness::DenoisedModel`], the 16 kHz denoiser inside the stream harness (a
//!   `VoiceConversionModel` wrapper that denoises exactly the new samples of every call).
//!
//! Where they run (the worker, `voice_changer::worker`): `gtcrn` wraps the model (the harness's
//! 16 kHz input), `dpdfnet` runs on the capture's device-rate copy (resampled to 48 kHz when the
//! device runs at another rate) after the input gain and before the aux mix. Models come from the
//! model store (`vc::assets::DENOISE_*`); a missing model is a warning and the run goes on without
//! the denoiser.
//!
//! # Frozen by P3-S0 (signatures; P3-E implements)
//! - [`DenoiserKind`] (re-exported from `vc::engine::options`), [`available`]`(kind) -> bool`.
//! - [`Denoiser`] trait (`sample_rate`, `latency_samples`, `process` in place and allocation-free
//!   after construction, `reset`) and [`create`]`(kind, ep, opts) -> Result<Box<dyn Denoiser>,
//!   String>` (`Off` is an error: callers skip the stage).
//!
//! # Added by P3-E
//! [`create_with`], [`model_path`], [`effective_kind`], [`stft`], [`harness`].

pub mod dfn;
pub mod gtcrn;
pub mod harness;
pub mod stft;

use std::path::PathBuf;

use ort::session::Session;

use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::vc::assets;

pub use crate::services::neural_audio::vc::engine::options::DenoiserKind;

/// Whether `kind` runs in this build (`Off` always does).
pub fn available(kind: DenoiserKind) -> bool {
    match kind {
        DenoiserKind::Off => true,
        DenoiserKind::Gtcrn => gtcrn::available(),
        DenoiserKind::Dfn3 | DenoiserKind::Dpdfnet => dfn::available(kind),
    }
}

/// The denoiser that runs for a stored `kind`, and why it differs (a kind this build cannot run
/// runs its stand-in: `dfn3` → `dpdfnet`).
pub fn effective_kind(kind: DenoiserKind) -> (DenoiserKind, Option<String>) {
    if available(kind) {
        return (kind, None);
    }
    match kind {
        DenoiserKind::Dfn3 => (
            DenoiserKind::Dpdfnet,
            Some(format!("dfn3: {}; using dpdfnet (the same architecture, streaming)", dfn::DFN3_UNAVAILABLE)),
        ),
        other => (DenoiserKind::Off, Some(format!("{}: not available in this build", other.as_str()))),
    }
}

/// One denoiser instance (a single mono stream).
pub trait Denoiser: Send {
    /// Rate `process` expects (16 000 for GTCRN, 48 000 for DeepFilterNet3 / DPDFNet).
    fn sample_rate(&self) -> u32;
    /// Algorithmic latency in samples at [`sample_rate`](Self::sample_rate).
    fn latency_samples(&self) -> usize;
    /// Denoise `audio` in place (any length; the denoiser buffers internally). The output is the
    /// input delayed by [`latency_samples`](Self::latency_samples). On `Err` the audio was still
    /// processed: frames the model failed on passed through unprocessed, the timing is intact.
    fn process(&mut self, audio: &mut [f32]) -> Result<(), String>;
    /// Drop internal state (stream gap / restart).
    fn reset(&mut self);
    /// Call lengths that keep [`latency_samples`](Self::latency_samples) exact: every `process` call
    /// since the last [`reset`](Self::reset) should be a multiple of this (the alignment of the chunk
    /// promise the denoiser was built with; 1 = any length). A caller whose first call of a stream
    /// has another length pads its front with silence.
    fn granularity(&self) -> usize {
        1
    }
}

/// The installed model of `kind` (the model-store entry; see `vc::assets`).
pub fn model_path(kind: DenoiserKind) -> Result<PathBuf, String> {
    let id = match kind {
        DenoiserKind::Off => return Err("no denoiser selected".into()),
        DenoiserKind::Gtcrn => assets::DENOISE_GTCRN,
        DenoiserKind::Dfn3 => assets::DENOISE_DFN3,
        DenoiserKind::Dpdfnet => assets::DENOISE_DPDFNET,
    };
    assets::model_file(id, None).ok_or_else(|| format!("the {} denoiser is not installed (model {id})", kind.as_str()))
}

/// Load a denoiser of `kind` (model from the model store, `vc::assets::DENOISE_*`).
#[allow(dead_code)] // P3-S0 frozen entry point; the worker uses `create_with` (its call-length promise).
pub fn create(kind: DenoiserKind, ep: &str, opts: &VcSessionOptions) -> Result<Box<dyn Denoiser>, String> {
    create_with(kind, ep, opts, None)
}

/// [`create`] with a call-length promise: every `process` call is a multiple of `chunk` samples
/// (lowers the latency by up to one hop), `None` = any length.
pub fn create_with(kind: DenoiserKind, ep: &str, opts: &VcSessionOptions, chunk: Option<usize>) -> Result<Box<dyn Denoiser>, String> {
    match kind {
        DenoiserKind::Off => Err("no denoiser selected".into()),
        DenoiserKind::Gtcrn => {
            let path = model_path(kind)?;
            gtcrn::load_from(&path, ep, opts, chunk).map(|d| Box::new(d) as Box<dyn Denoiser>)
        }
        DenoiserKind::Dfn3 | DenoiserKind::Dpdfnet => {
            if !dfn::available(kind) {
                return Err(format!("{}: {}", kind.as_str(), dfn::DFN3_UNAVAILABLE));
            }
            let path = model_path(kind)?;
            dfn::load_from(&path, ep, opts, chunk).map(|d| Box::new(d) as Box<dyn Denoiser>)
        }
    }
}

/// Check a denoiser graph's declared inputs (name, float32, the expected shape) and outputs (names),
/// so a wrong file fails at load with a clear message instead of on the first frame.
pub(crate) fn check_io(session: &Session, label: &str, inputs: &[(&str, Vec<i64>)], outputs: &[&str]) -> Result<(), String> {
    for (name, shape) in inputs {
        let input = session
            .inputs()
            .iter()
            .find(|i| i.name() == *name)
            .ok_or_else(|| format!("{label}: the model has no input `{name}` (is this the right file?)"))?;
        if input.dtype().tensor_type() != Some(ort::value::TensorElementType::Float32) {
            return Err(format!("{label}: input `{name}` is not float32"));
        }
        let declared = input.dtype().tensor_shape().map(|s| s.to_vec()).unwrap_or_default();
        let compatible = declared.len() == shape.len() && declared.iter().zip(shape).all(|(d, s)| *d < 0 || d == s);
        if !compatible {
            return Err(format!("{label}: input `{name}` has shape {declared:?}, expected {shape:?}"));
        }
    }
    for name in outputs {
        if !session.outputs().iter().any(|o| o.name() == *name) {
            return Err(format!("{label}: the model has no output `{name}`"));
        }
    }
    Ok(())
}

/// The model file of `kind` for tests that run the real models: `SIGIL_TEST_MODELS_DIR`
/// (`vc-assets/vc/<entry>/<file>` or `vc/<entry>/<file>`), else the dev cache's staged assets.
#[cfg(test)]
pub(crate) fn test_model(kind: DenoiserKind) -> Option<PathBuf> {
    let entry = match kind {
        DenoiserKind::Gtcrn => "denoise-gtcrn",
        DenoiserKind::Dpdfnet => "denoise-dpdfnet",
        _ => return None,
    };
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(r) = std::env::var_os("SIGIL_TEST_MODELS_DIR") {
        roots.push(PathBuf::from(r));
    }
    if let Some(l) = std::env::var_os("LOCALAPPDATA") {
        roots.push(PathBuf::from(l).join("sigil-dev-cache"));
    }
    roots
        .into_iter()
        .flat_map(|r| [r.join("vc-assets").join("vc").join(entry).join("model.onnx"), r.join("vc").join(entry).join("model.onnx")])
        .find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_resolve_to_what_runs() {
        assert!(available(DenoiserKind::Off));
        assert!(available(DenoiserKind::Gtcrn));
        assert!(available(DenoiserKind::Dpdfnet));
        assert!(!available(DenoiserKind::Dfn3));
        assert_eq!(effective_kind(DenoiserKind::Gtcrn), (DenoiserKind::Gtcrn, None));
        assert_eq!(effective_kind(DenoiserKind::Off), (DenoiserKind::Off, None));
        let (k, note) = effective_kind(DenoiserKind::Dfn3);
        assert_eq!(k, DenoiserKind::Dpdfnet);
        assert!(note.unwrap().starts_with("dfn3:"));
        assert!(create(DenoiserKind::Off, "cpu", &VcSessionOptions::default()).is_err());
        assert!(model_path(DenoiserKind::Off).is_err());
    }
}
