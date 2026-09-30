//! In-process local TTS: the `local-tts` plugin (P2-H) and the vendored Kitten.
//!
//! - [`commands`]: the `local-tts` plugin (IPC contract frozen by P2-S0, see its docs).
//! - [`engine`]: the `sigil-local-tts` worker (one loaded model, ordered jobs, cancel, pushes into
//!   the TTS sink).
//! - [`models`]: what a request's `model` means (catalog id, folder, Hub repo) and how it loads
//!   (sherpa-onnx `OfflineTts` with C espeak-ng, or the vendored Kitten).
//! - [`catalog`]: the catalog models (`model-catalog/tts-local.json`) — family, voices, languages.
//! - [`kokoro`], [`piper`]: per-family configuration; [`voices`]: voice tables and language rules;
//!   [`text`]: sentence / clause chunking.
//! - [`kokoro_ort`]: Kokoro's graph on Sigil's ONNX Runtime behind sherpa-onnx's text frontend (a
//!   token "tap" graph); opt-in until `ort_session` enables ONNX Runtime's CPU memory arena.
//! - [`kitten`]: KittenTTS, vendored from the `kittentts` 0.3.0 crate (Apache-2.0; see
//!   `kitten/NOTICE`): the fallback for catalog Kitten models when sherpa-onnx cannot load, and
//!   the engine for legacy `speech_package_v1` folders and Hub repos (`neuralOnnx` settings; the
//!   `neural-onnx-tts` plugin also drives it).
//!
//! Measured on the dev CPU (Ryzen 7 7800X3D; sherpa-onnx 1.13.8 with its ONNX Runtime 1.28.2), fp32
//! Kokoro v1.0 at 4 threads: RTF ≈ 0.22-0.25 idle for one sentence, 0.29-0.33 while other builds run
//! (2 threads: 0.40-0.47); load ≈ 1.1-1.4 s. The int8 build is not faster (RTF ≈ 1.0) and is not in
//! the catalog. Piper medium RTF ≈ 0.04 idle; Kitten mini ≈ 0.6. Sigil's ONNX Runtime runs the same
//! Kokoro graph at 0.23-0.24 once its CPU memory arena is on (see [`kokoro_ort`]).

pub mod catalog;
pub mod commands;
pub mod engine;
pub mod kitten;
pub mod kokoro;
pub mod kokoro_ort;
pub mod models;
pub mod piper;
pub mod text;
pub mod voices;

pub use commands::init;

/// Offline stand-in speech package for tests (hand-encoded ONNX + NPZ).
#[cfg(test)]
pub(crate) mod test_model;

#[cfg(test)]
mod tests;
