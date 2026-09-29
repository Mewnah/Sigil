//! In-process local TTS engines on Sigil's own ONNX Runtime.
//!
//! - [`kitten`]: KittenTTS, vendored from the `kittentts` 0.3.0 crate (Apache-2.0; see
//!   `kitten/NOTICE`). The `neural-onnx-tts` plugin (`services/neural_onnx_tts.rs`) drives it.
//!
//! P2 adds Kokoro and Piper runners here, sharing the `espeak-ng` phonemizer dependency.

pub mod kitten;

/// Offline stand-in speech package for tests (hand-encoded ONNX + NPZ).
#[cfg(test)]
pub(crate) mod test_model;
